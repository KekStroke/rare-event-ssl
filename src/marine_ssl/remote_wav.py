"""Materialize a frame-exact prefix of a version-checked uncompressed RIFF WAV."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import time

import requests

from .io import client, digest, write_json

HEADER_LIMIT = 1024 * 1024


def _etag(value):
    value = str(value or "").strip()
    if not value or value.startswith("W/") or any(c in value for c in "\r\n"):
        return None
    return '"' + value.strip('"') + '"'


def _response(session, url, start, end, size, etag, *, probe=False):
    response = session.get(url, stream=True, timeout=(30, 180), headers={
        "Range": f"bytes={start}-{end}", "If-Match": etag, "Accept-Encoding": "identity"})
    try:
        if probe and response.status_code == 200:
            response.close()
            return None
        response.raise_for_status()
        if response.status_code != 206:
            raise ValueError("Selected WAV interval requires HTTP 206")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match or tuple(map(int, match.groups())) != (start, end, size):
            raise ValueError("Selected WAV Content-Range differs from source inventory")
        if int(response.headers.get("Content-Length", -1)) != end - start + 1:
            raise ValueError("Selected WAV response length differs from requested range")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("Encoded WAV range cannot preserve source byte offsets")
        if _etag(response.headers.get("ETag")) != etag:
            raise ValueError("Selected WAV source ETag changed or is absent")
        return response
    except BaseException:
        response.close()
        raise


def _header(data, size, max_seconds):
    data = bytes(data)
    # Unknown containers/layouts use the ordinary full-object acquisition path.
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    riff_end = struct.unpack_from("<I", data, 4)[0] + 8
    if riff_end > size:
        raise ValueError("RIFF length exceeds the source object")
    position, fmt = 12, None
    while position + 8 <= len(data):
        tag, length = data[position:position + 4], struct.unpack_from("<I", data, position + 4)[0]
        body = position + 8
        if body + length > riff_end:
            raise ValueError("WAV chunk exceeds the source RIFF boundary")
        if tag == b"data":
            if fmt is None:
                return None
            code, channels, rate, align, bits = fmt
            if length % align:
                raise ValueError("WAV data length is not an exact frame count")
            frames = min(length // align, math.floor(max_seconds * rate))
            if frames < 1 or frames >= length // align:
                return None
            selected_bytes = frames * align
            header = bytearray(data[:body])
            padding = selected_bytes % 2
            struct.pack_into("<I", header, 4, len(header) + selected_bytes + padding - 8)
            struct.pack_into("<I", header, position + 4, selected_bytes)
            return bytes(header), {
                "format_code": code, "channels": channels, "sample_rate": rate,
                "bits_per_sample": bits, "block_align": align, "selected_frames": frames,
                "duration_seconds": frames / rate, "original_data_bytes": length,
                "original_data_frames": length // align, "original_header_bytes": body,
                "original_riff_bytes": riff_end, "source_header_sha256": hashlib.sha256(data[:body]).hexdigest(),
                "byte_range": [body, body + selected_bytes - 1],
                "materialized_padding_bytes": padding,
                "materialized_bytes": body + selected_bytes + padding}
        if body + length > len(data):
            return None  # A header larger than HEADER_LIMIT is deliberately unsupported.
        if tag == b"fmt ":
            if fmt is not None or length not in {16, 18, 40}:
                return None
            code, channels, rate, byte_rate, align, bits = struct.unpack_from("<HHIIHH", data, body)
            effective = code
            if code == 0xFFFE:
                if length != 40 or struct.unpack_from("<H", data, body + 16)[0] != 22:
                    return None
                valid_bits, mask = struct.unpack_from("<HI", data, body + 18)
                guid = data[body + 24:body + 40]
                if guid[4:] != bytes.fromhex("00001000800000aa00389b71"):
                    return None
                effective = struct.unpack_from("<I", guid)[0]
                if not 0 < valid_bits <= bits or (mask and mask.bit_count() != channels):
                    raise ValueError("Invalid extensible WAV channel/valid-bit metadata")
            elif length == 18 and struct.unpack_from("<H", data, body + 16)[0] != 0:
                return None
            elif length == 40:
                return None
            if effective not in {1, 3} or bits not in ({8, 16, 24, 32} if effective == 1 else {32, 64}):
                return None
            if not channels or not rate or align != channels * (bits // 8) or byte_rate != rate * align:
                raise ValueError("Invalid uncompressed WAV frame geometry")
            fmt = code, channels, rate, align, bits
        elif tag not in {b"LIST", b"JUNK", b"PAD ", b"bext"}:
            # fact/cue/smpl contain sample counts or positions that would need
            # separate normalization; do not copy inconsistent metadata.
            return None
        position = body + length + length % 2
    return None


def acquire_wav_prefix(row, path, max_seconds, session=None):
    """Return interval provenance, or None for a safe full-download fallback.

    Requires a known full-object size and strong source ETag. The downloaded PCM
    is unchanged. Only the owned RIFF/data lengths and structural pad are written;
    source SHA verification is never claimed for the materialized subset.
    """
    max_seconds = float(max_seconds)
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("WAV prefix duration must be positive and finite")
    size, etag = row.get("bytes", row.get("size")), _etag(row.get("source_etag") or row.get("etag"))
    if not isinstance(size, int) or isinstance(size, bool) or size < 12 or etag is None:
        return None
    session, path = session or client(), Path(path)
    probe_end = min(size, HEADER_LIMIT) - 1
    for attempt in range(3):
        try:
            response = _response(session, row["url"], 0, probe_end, size, etag, probe=True)
            if response is None:
                return None
            with response:
                data = bytearray()
                for block in response.iter_content(65536):
                    if len(data) + len(block) > probe_end + 1:
                        raise ValueError("WAV header probe exceeded its bounded range")
                    data.extend(block)
                if len(data) != probe_end + 1:
                    raise requests.ConnectionError("Incomplete WAV header range")
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    parsed = _header(data, size, max_seconds)
    if parsed is None:
        return None
    header, metadata = parsed
    metadata.update(schema_version=1, source_id=row.get("source_id"), source_etag=etag,
                    source_object_bytes=size, interval_start_frame=0,
                    source_etag_verified=True, source_sha256_verified=False,
                    materialized_kind="uncompressed_wav_source_prefix", path=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_name(path.name + ".range.receipt.json")
    temporary = path.with_name(path.name + ".range.part")
    lock = path.with_name(path.name + ".range.lock")
    for candidate in (path, receipt, temporary, lock):
        if candidate.is_symlink():
            raise ValueError("Symlink in selected WAV materialization")
    lock.open("x").close()
    try:
        if path.exists():
            if not receipt.exists():
                raise ValueError("Existing WAV materialization has no ownership receipt")
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            keys = ("schema_version", "source_id", "source_etag", "source_object_bytes", "source_header_sha256", "byte_range", "materialized_bytes")
            if (any(saved.get(key) != metadata.get(key) for key in keys)
                    or path.stat().st_size != metadata["materialized_bytes"]
                    or digest(path) != saved.get("materialized_sha256")):
                raise ValueError("Existing WAV materialization differs from its source/receipt")
            return saved
        start, end = metadata["byte_range"]
        for attempt in range(3):
            try:
                checksum, received = hashlib.sha256(header), 0
                with _response(session, row["url"], start, end, size, etag) as response:
                    with temporary.open("wb") as stream:
                        stream.write(header)
                        for block in response.iter_content(1024 * 1024):
                            if received + len(block) > end - start + 1:
                                raise ValueError("WAV body exceeded the selected interval")
                            stream.write(block)
                            checksum.update(block)
                            received += len(block)
                        if received != end - start + 1:
                            raise requests.ConnectionError("Incomplete selected WAV interval")
                        if metadata["materialized_padding_bytes"]:
                            stream.write(b"\0")
                            checksum.update(b"\0")
                if temporary.stat().st_size != metadata["materialized_bytes"]:
                    raise ValueError("Materialized WAV length is inconsistent")
                metadata["materialized_sha256"] = checksum.hexdigest()
                os.replace(temporary, path)
                write_json(receipt, metadata)
                return metadata
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    finally:
        temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
