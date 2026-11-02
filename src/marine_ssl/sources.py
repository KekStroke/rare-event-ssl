"""Public metadata catalogs. Audio is selected before any bulk transfer."""

import csv
import io
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from .io import client, write_json

AUDIO_SUFFIXES = {".wav", ".flac", ".aif", ".aiff", ".mp3", ".ogg"}


def _relative_metadata_path(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or "\x00" in value or re.match(r"^[A-Za-z]:", value)):
        raise ValueError("unsafe source metadata path")
    return path


def hf_catalog(repo, metadata_dir, *, revision=None, session=None):
    session = session or client()
    response = session.get(f"https://huggingface.co/api/datasets/{repo}", timeout=120)
    response.raise_for_status()
    info = response.json()
    revision = revision or info["sha"]
    root = Path(metadata_dir) / repo.split("/")[-1]
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "repository.json", {**info, "pinned_revision": revision})
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000"
    entries = []
    while url:
        response = session.get(url, timeout=120)
        response.raise_for_status()
        entries.extend(response.json())
        url = response.links.get("next", {}).get("url")
    write_json(root / "tree.json", entries)
    tables = {}
    for item in entries:
        if item["type"] != "file" or Path(item["path"]).suffix.lower() not in {".csv", ".md"}:
            continue
        if item.get("size", 0) > 30_000_000:
            continue
        content = session.get(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(item['path'])}", timeout=120)
        content.raise_for_status()
        target = root / _relative_metadata_path(item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.content)
        if target.suffix.lower() == ".csv":
            tables[item["path"]] = list(csv.DictReader(io.StringIO(content.content.decode("utf-8-sig"))))
    return {"repo": repo, "revision": revision, "entries": entries, "tables": tables,
            "used_storage": info.get("usedStorage")}


def gcs_catalog(bucket, prefix, *, session=None):
    session = session or client()
    params = {"prefix": prefix, "maxResults": 1000}
    result = []
    while True:
        response = session.get(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o", params=params, timeout=120)
        response.raise_for_status()
        page = response.json()
        result.extend(page.get("items", []))
        if not page.get("nextPageToken"):
            break
        params["pageToken"] = page["nextPageToken"]
    return result


def s3_catalog(bucket, prefix, *, session=None, max_objects=10000):
    """List a bounded prefix; logs distinguish a capped catalog from full inventory."""
    import json
    import xml.etree.ElementTree as ET
    from requests import Session
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    if not isinstance(max_objects, int) or max_objects < 1:
        raise ValueError("max_objects must be a positive integer")
    session = session or client()
    if isinstance(session, Session):
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"], respect_retry_after_header=True)))
    params = {"list-type": "2", "prefix": prefix, "max-keys": min(1000, max_objects)}
    result, capped = [], False
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    print(json.dumps({"stage": "s3_catalog_start", "bucket": bucket, "prefix": prefix,
                      "max_objects": max_objects}), flush=True)
    while True:
        params["max-keys"] = min(1000, max_objects - len(result))
        response = session.get(f"https://{bucket}.s3.amazonaws.com/", params=params, timeout=60)
        response.raise_for_status()
        page = ET.fromstring(response.content)
        objects = page.findall("s:Contents", ns)
        remaining = max_objects - len(result)
        for item in objects[:remaining]:
            result.append({"name": item.findtext("s:Key", namespaces=ns),
                           "size": int(item.findtext("s:Size", namespaces=ns)),
                           "etag": item.findtext("s:ETag", namespaces=ns)})
        truncated = page.findtext("s:IsTruncated", namespaces=ns) == "true"
        if len(result) >= max_objects:
            capped = truncated or len(objects) > remaining
            break
        if not truncated:
            break
        token = page.findtext("s:NextContinuationToken", namespaces=ns)
        if not token or token == params.get("continuation-token"):
            raise ValueError("Invalid or repeated S3 continuation token")
        params["continuation-token"] = token
    print(json.dumps({"stage": "s3_catalog_done", "bucket": bucket, "prefix": prefix,
                      "objects": len(result), "catalog_inventory_complete": not capped,
                      "cap_reached": capped}), flush=True)
    return result


def recording_identity(family, name):
    """Conservative acquisition-day sessions, including ONC array channels."""
    basename = Path(name).stem
    if family == "onc":
        match = re.search(r"([A-Z0-9]+)_(\d{8})T(\d{6})(?:\.(\d+))?Z", basename)
        if match:
            sensor, day, clock, fraction = match.groups()
            date = datetime.strptime(day + clock, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            return {"source_id": basename, "hydrophone_id": sensor,
                    "deployment_id": sensor, "session_id": f"{sensor}:{day}",
                    "recording_start_utc": date.isoformat()}
    if family == "orcasound":
        match = re.search(r"(rpi-[\w-]+?)_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})", basename)
        if match:
            sensor, *parts = match.groups()
            date = datetime(*map(int, parts), tzinfo=timezone.utc)
            return {"source_id": basename, "hydrophone_id": sensor,
                    "deployment_id": sensor, "session_id": f"{sensor}:{date:%Y%m%d}",
                    "recording_start_utc": date.isoformat()}
        epoch = re.match(r"(1\d{9})_", basename)
        if epoch:
            date = datetime.fromtimestamp(int(epoch[1]), tz=timezone.utc)
            return {"source_id": basename, "session_id": f"orcasound:{date:%Y%m%d}",
                    "recording_start_utc": date.isoformat()}
    return {"source_id": basename}


def hf_audio_rows(catalog, family):
    labels = {}
    for rows in catalog["tables"].values():
        for row in rows:
            filename = row.get("filename") or row.get("file_name")
            if filename:
                labels.setdefault(Path(filename).name, []).append(row)
    for item in catalog["entries"]:
        name = item["path"]
        _relative_metadata_path(name)
        if item["type"] != "file" or Path(name).suffix.lower() not in AUDIO_SUFFIXES:
            continue
        metadata = labels.get(Path(name).name, [])
        yield {"dataset": catalog["repo"].split("/")[-1], "source_family": family,
               "path_in_source": name, "url": f"https://huggingface.co/datasets/{catalog['repo']}/resolve/{catalog['revision']}/{quote(name)}",
               "revision": catalog["revision"], "bytes": item.get("size"),
               "sha256": (item.get("lfs") or {}).get("oid"), "annotations": metadata,
               **recording_identity(family, name)}
