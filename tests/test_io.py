import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from marine_ssl.io import download


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Response:
    def __init__(self, body=b"", *, status=200, headers=None, error=None):
        self.body, self.status_code = body, status
        self.headers, self.error = headers or {}, error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, size):
        yield self.body
        if self.error:
            raise self.error


class Session:
    def __init__(self, *responses):
        self.responses, self.calls = iter(responses), []

    def get(self, url, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "audio.wav"
        self.part = self.path.with_name(self.path.name + ".part")
        self.receipt = self.path.with_name(self.path.name + ".receipt.json")
        patcher = patch("marine_ssl.io.time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_expected_sha_overrides_old_receipt(self):
        self.path.write_bytes(b"old")
        self.receipt.write_text(json.dumps({"schema_version": 1, "url": "https://example/audio",
                                            "bytes": 3, "sha256": sha(b"old")}))
        session = Session(Response(b"new", headers={"Content-Length": "3"}))
        download("https://example/audio", self.path, size=3, sha256=sha(b"new"), session=session)
        self.assertEqual(b"new", self.path.read_bytes())
        self.assertEqual(1, len(session.calls))

    def test_no_checksum_restarts_unknown_partial(self):
        self.part.write_bytes(b"old")
        session = Session(Response(b"new", headers={"Content-Length": "3"}))
        download("https://example/audio", self.path, size=3, session=session)
        self.assertNotIn("Range", session.calls[0]["headers"])
        self.assertEqual(b"new", self.path.read_bytes())

    def test_interrupted_checksum_transfer_resumes_exact_offset(self):
        session = Session(Response(b"abc", error=requests.ConnectionError("interrupted")),
                          Response(b"def", status=206, headers={"Content-Range": "bytes 3-5/6"}))
        download("https://example/audio", self.path, size=6, sha256=sha(b"abcdef"), session=session)
        self.assertEqual("bytes=3-", session.calls[1]["headers"]["Range"])
        self.assertEqual("identity", session.calls[1]["headers"]["Accept-Encoding"])
        self.assertEqual(b"abcdef", self.path.read_bytes())

    def test_ignored_range_overwrites_instead_of_appending(self):
        self.part.write_bytes(b"abc")
        session = Session(Response(b"abcdef", headers={"Content-Length": "6"}))
        download("https://example/audio", self.path, size=6, sha256=sha(b"abcdef"), session=session)
        self.assertEqual(b"abcdef", self.path.read_bytes())

    def test_partial_response_without_known_size_is_not_finalized(self):
        session = Session(*(Response(b"abc", status=206, headers={"Content-Range": "bytes 0-2/6"}) for _ in range(5)))
        with self.assertRaises(requests.ConnectionError):
            download("https://example/audio", self.path, session=session)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.receipt.exists())

    def test_bad_range_and_body_lengths_never_get_receipts(self):
        for headers in ({"Content-Range": "bytes 1-3/6"},
                        {"Content-Range": "bytes 0-5/6", "Content-Length": "3"},
                        {"Content-Range": "bytes 0-2/3", "Content-Encoding": "gzip"}):
            with self.subTest(headers=headers):
                session = Session(*(Response(b"abc", status=206, headers=headers) for _ in range(5)))
                with self.assertRaises(ValueError):
                    download("https://example/audio", self.path, session=session)
                self.assertFalse(self.path.exists())
                self.assertFalse(self.receipt.exists())

    def test_truncated_content_length_is_not_finalized(self):
        session = Session(*(Response(b"abc", headers={"Content-Length": "6"}) for _ in range(5)))
        with self.assertRaises(requests.ConnectionError):
            download("https://example/audio", self.path, session=session)
        self.assertFalse(self.path.exists())

    def test_oversized_partial_is_reset(self):
        self.part.write_bytes(b"too long")
        session = Session(Response(b"new", headers={"Content-Length": "3"}))
        download("https://example/audio", self.path, size=3, sha256=sha(b"new"), session=session)
        self.assertNotIn("Range", session.calls[0]["headers"])
        self.assertEqual(b"new", self.path.read_bytes())

    def test_416_is_only_accepted_when_final_source_hash_matches(self):
        self.part.write_bytes(b"new")
        session = Session(Response(status=416))
        download("https://example/audio", self.path, size=3, sha256=sha(b"new"), session=session)
        self.assertEqual(b"new", self.path.read_bytes())

    def test_corrupt_receipt_is_replaced_after_fresh_download(self):
        self.path.write_bytes(b"old")
        self.receipt.write_text("{broken")
        session = Session(Response(b"new"))
        download("https://example/audio", self.path, session=session)
        self.assertEqual(b"new", self.path.read_bytes())
        self.assertEqual(sha(b"new"), json.loads(self.receipt.read_text())["sha256"])

    def test_existing_source_verified_file_gets_missing_receipt(self):
        self.path.write_bytes(b"data")
        download("https://example/audio", self.path, sha256=sha(b"data"), session=Session())
        receipt = json.loads(self.receipt.read_text())
        self.assertTrue(receipt["source_sha256_verified"])
        self.assertEqual(sha(b"data"), receipt["sha256"])

    def test_verified_receipt_reuses_local_file(self):
        first = Session(Response(b"data"))
        download("https://example/audio", self.path, session=first)
        download("https://example/audio", self.path, session=Session())


if __name__ == "__main__":
    unittest.main()
