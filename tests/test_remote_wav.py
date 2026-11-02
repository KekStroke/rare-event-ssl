import hashlib
import io
import os
import subprocess
import sys
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

import requests

from marine_ssl.remote_wav import acquire_wav_prefix


def wav(payload, *, code=1, channels=1, rate=8, bits=16, extra=b""):
    align = channels * bits // 8
    fmt = struct.pack("<HHIIHH", code, channels, rate, rate * align, align, bits)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + extra + b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks


class Response:
    def __init__(self, body, headers, status=206):
        self.body, self.headers, self.status_code = body, headers, status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.body), 7):
            yield self.body[offset:offset + 7]


class Session:
    def __init__(self, source, *, bad=None):
        self.source, self.calls, self.bad = source, [], bad

    def get(self, url, **kwargs):
        self.calls.append(kwargs)
        start, end = map(int, kwargs["headers"]["Range"][6:].split("-"))
        body = self.source[start:end + 1]
        headers = {"Content-Range": f"bytes {start}-{end}/{len(self.source)}",
                   "Content-Length": str(len(body)), "ETag": '"version"'}
        if self.bad == "etag" or (self.bad == "etag_body" and len(self.calls) > 1):
            headers["ETag"] = '"changed"'
        if self.bad == "range":
            headers["Content-Range"] = f"bytes {start + 1}-{end}/{len(self.source)}"
        if self.bad == "length":
            headers["Content-Length"] = str(len(body) + 1)
        if self.bad == "truncated" and len(self.calls) > 1:
            body = body[:-1]
        return Response(body, headers, status=200 if self.bad == "ignored" else 206)


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "prefix.wav"

    def row(self, source):
        return {"url": "https://public.example/daily.wav", "source_id": "daily", "bytes": len(source), "source_etag": '"version"'}

    def test_exact_source_interval_normalized_lengths_and_materialized_hash(self):
        payload = bytes(range(64))
        source = wav(payload, extra=b"LIST" + struct.pack("<I", 4) + b"INFO")
        session = Session(source)
        with patch("marine_ssl.remote_wav.HEADER_LIMIT", 64):
            result = acquire_wav_prefix(self.row(source), self.path, 1, session=session)
        self.assertEqual("bytes=0-63", session.calls[0]["headers"]["Range"])
        selected = self.path.read_bytes()
        with wave.open(io.BytesIO(selected)) as reader:
            self.assertEqual(8, reader.getnframes())
            self.assertEqual(payload[:16], reader.readframes(100))
        self.assertEqual(len(selected) - 8, struct.unpack_from("<I", selected, 4)[0])
        self.assertEqual(16, result["byte_range"][1] - result["byte_range"][0] + 1)
        self.assertEqual(hashlib.sha256(selected).hexdigest(), result["materialized_sha256"])
        self.assertFalse(result["source_sha256_verified"])
        self.assertEqual(64, result["original_data_bytes"])
        self.assertEqual(2, len(session.calls))
        self.assertTrue(all(call["headers"]["If-Match"] == '"version"' for call in session.calls))
        self.assertEqual(result, acquire_wav_prefix(self.row(source), self.path, 1, session=session))
        self.assertEqual(3, len(session.calls))  # Reuse requires only the bounded version/header probe.

    def test_odd_pcm_data_gets_structural_pad_not_an_extra_sample(self):
        source = wav(bytes(range(16)), bits=8, rate=3)
        result = acquire_wav_prefix(self.row(source), self.path, 1, session=Session(source))
        with wave.open(str(self.path)) as reader:
            self.assertEqual(3, reader.getnframes())
            self.assertEqual(bytes(range(3)), reader.readframes(100))
        self.assertEqual(1, result["materialized_padding_bytes"])

    def test_pcm24_and_ieee_float_are_not_reencoded(self):
        for code, bits in ((1, 24), (3, 32)):
            with self.subTest(code=code):
                path = self.path.with_name(f"{code}.wav")
                source = wav(bytes(range(96)), code=code, bits=bits, rate=4)
                result = acquire_wav_prefix(self.row(source), path, 1, session=Session(source))
                offset = result["original_header_bytes"]
                self.assertEqual(source[offset:offset + 4 * bits // 8], path.read_bytes()[offset:offset + 4 * bits // 8])

    def test_mismatched_etag_range_or_length_never_materializes(self):
        source = wav(bytes(range(64)))
        for bad in ("etag", "etag_body", "range", "length"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                acquire_wav_prefix(self.row(source), self.path, 1, session=Session(source, bad=bad))
            self.assertFalse(self.path.exists())

    def test_body_short_read_retries_boundedly_and_never_finalizes(self):
        source = wav(bytes(range(64)))
        session = Session(source, bad="truncated")
        with patch("marine_ssl.remote_wav.time.sleep"), self.assertRaises(requests.ConnectionError):
            acquire_wav_prefix(self.row(source), self.path, 1, session=session)
        self.assertEqual(4, len(session.calls))
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_name(self.path.name + ".range.part").exists())

    def test_abandoned_lock_marker_and_terminated_owner_do_not_block_resume(self):
        source = wav(bytes(range(64)))
        lock = self.path.with_name(self.path.name + ".range.lock")
        lock.write_bytes(b"old marker")
        child_code = """
import os,sys,time
stream=open(sys.argv[1], 'a+b')
if os.name == 'nt':
    import msvcrt
    if os.fstat(stream.fileno()).st_size == 0:
        stream.write(b'0');stream.flush()
    stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
else:
    import fcntl
    fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
print('locked',flush=True)
time.sleep(60)
"""
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        child = subprocess.Popen([sys.executable, "-c", child_code, str(lock)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, creationflags=flags)
        try:
            self.assertEqual("locked", child.stdout.readline().strip())
            with self.assertRaises(OSError):
                acquire_wav_prefix(self.row(source), self.path, 1, session=Session(source))
            self.assertFalse(self.path.exists())
            child.terminate()
            child.wait(timeout=5)
            result = acquire_wav_prefix(self.row(source), self.path, 1, session=Session(source))
            self.assertEqual(8, result["selected_frames"])
            self.assertTrue(lock.exists())  # Never unlink a potentially locked inode.
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            child.stdout.close()
            child.stderr.close()

    def test_unsupported_layout_or_ignored_range_falls_back(self):
        for source in (b"RF64" + bytes(100), wav(bytes(64), code=6),
                       wav(bytes(64), extra=b"fact" + struct.pack("<II", 4, 32))):
            self.assertIsNone(acquire_wav_prefix(self.row(source), self.path, 1, session=Session(source)))
        source = wav(bytes(64))
        self.assertIsNone(acquire_wav_prefix(self.row(source), self.path, 1, session=Session(source, bad="ignored")))
        self.assertFalse(self.path.exists())

    def test_invalid_frame_geometry_and_unowned_existing_target_are_preserved(self):
        source = bytearray(wav(bytes(64)))
        struct.pack_into("<H", source, 32, 9)
        with self.assertRaises(ValueError):
            acquire_wav_prefix(self.row(source), self.path, 1, session=Session(source))
        source = wav(bytes(64))
        self.path.write_bytes(b"owned elsewhere")
        with self.assertRaises(ValueError):
            acquire_wav_prefix(self.row(source), self.path, 1, session=Session(source))
        self.assertEqual(b"owned elsewhere", self.path.read_bytes())


if __name__ == "__main__":
    unittest.main()
