"""Small, atomic HTTP and manifest helpers; no credentials are logged."""

import hashlib
import json
import os
import re
import time
import threading
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_clients = threading.local()


def client():
    """Reuse one bounded connection pool per calling thread and process."""
    pid = os.getpid()
    if getattr(_clients, "pid", None) != pid or not hasattr(_clients, "session"):
        session = requests.Session()
        session.mount("https://", HTTPAdapter(
            pool_connections=4, pool_maxsize=2, pool_block=True,
            max_retries=Retry(total=5, backoff_factor=1,
                              status_forcelist=[429, 500, 502, 503, 504],
                              allowed_methods=["GET", "HEAD"],
                              respect_retry_after_header=True)))
        session.headers["User-Agent"] = "rare-event-ssl-data/0.1 (research acquisition)"
        _clients.session, _clients.pid = session, pid
    return _clients.session


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(url, path, *, size=None, sha256=None, session=None):
    """Atomically transfer raw bytes; resume only against a published SHA-256.

    Without a source checksum, retries restart rather than combine potentially
    different object versions. Receipts detect later local corruption, but do
    not substitute for a published checksum.
    """
    session = session or client()
    if sha256:
        sha256 = str(sha256).lower().removeprefix("sha256:")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("invalid source SHA-256")
    if size is not None:
        size = int(size)
        if size < 0:
            raise ValueError("negative source size")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_name(path.name + ".receipt.json")
    if path.exists() and (size is None or path.stat().st_size == size):
        actual = digest(path)
        if sha256 and actual == sha256:
            write_json(receipt, {"schema_version": 1, "url": url, "bytes": path.stat().st_size,
                                 "sha256": actual, "source_sha256_verified": True})
            return path
        if not sha256 and receipt.exists():
            try:
                saved = json.loads(receipt.read_text(encoding="utf-8"))
                if (saved.get("schema_version") == 1 and saved.get("url") == url
                        and saved.get("bytes") == path.stat().st_size
                        and saved.get("sha256") == actual):
                    return path
            except (OSError, ValueError, AttributeError):
                pass
    temporary = path.with_name(path.name + ".part")
    for attempt in range(5):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset and (not sha256 or (size is not None and offset > size)):
            temporary.unlink()
            offset = 0
        try:
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            total = size
            with session.get(url, stream=True, timeout=(30, 180), headers=headers) as response:
                if response.status_code == 416 and offset and sha256:
                    # Only the final source hash can establish that a retained
                    # partial is already complete; HTTP 416 itself proves none.
                    pass
                else:
                    response.raise_for_status()
                    if response.status_code not in {200, 206}:
                        raise ValueError("unexpected HTTP transfer status")
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise ValueError("encoded response cannot verify raw byte offsets")
                    expected_body = None
                    if response.status_code == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                             response.headers.get("Content-Range", ""))
                        if not match:
                            raise ValueError("invalid Content-Range")
                        start, end, remote_total = map(int, match.groups())
                        if start != offset or end < start or end >= remote_total:
                            raise ValueError("server returned an incorrect byte range")
                        if total is not None and total != remote_total:
                            raise ValueError("range total differs from source metadata")
                        total = remote_total
                        expected_body = end - start + 1
                    else:
                        offset = 0  # The server may ignore Range; overwrite safely.
                    length = response.headers.get("Content-Length")
                    if length is not None:
                        length = int(length)
                        if length < 0 or (expected_body is not None and length != expected_body):
                            raise ValueError("response length differs from byte range")
                        expected_body = length
                    received = 0
                    with temporary.open("ab" if offset else "wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
                            received += len(block)
                    if expected_body is not None and received != expected_body:
                        raise requests.ConnectionError("incomplete HTTP response body")
            actual_size = temporary.stat().st_size
            if total is not None and actual_size < total:
                raise requests.ConnectionError("source transfer is not yet complete")
            if total is not None and actual_size != total:
                raise ValueError("download size differs from source metadata")
            actual_sha = digest(temporary)
            if sha256 and actual_sha != sha256:
                raise ValueError("source SHA-256 mismatch")
            os.replace(temporary, path)
            write_json(receipt, {"schema_version": 1, "url": url, "bytes": actual_size,
                                 "sha256": actual_sha, "source_sha256_verified": bool(sha256)})
            return path
        except (requests.RequestException, OSError, ValueError) as error:
            if isinstance(error, ValueError) and temporary.exists():
                temporary.unlink()
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError("unreachable")
