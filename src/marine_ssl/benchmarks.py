"""Resolve benchmark download plans without downloading audio or changing splits.

Run metadata discovery in an MLS CPU job. Returned rows are an explicit allowlist
for the downloader; a successful plan is not an integrity/availability verdict.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import PurePosixPath
from urllib.parse import quote, unquote, urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"}


def metadata_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "marine-ssl-data/0.1 (metadata inventory)"
    return session


def _json(session, url, **params):
    response = session.get(url, params=params or None, timeout=(15, 120))
    response.raise_for_status()
    return response.json()


def _row(dataset, source_id, url, size=None, **metadata):
    return {"dataset": dataset, "source_id": source_id, "url": url,
            "size": size, **metadata}


def _hf_plan(dataset, spec, session):
    repo, revision = spec["repo"], spec["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"{dataset}: pin a 40-character HF commit SHA")
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000"
    rows = []
    while url:
        response = session.get(url, timeout=(15, 120))
        response.raise_for_status()
        for obj in response.json():
            if obj.get("type") != "file":
                continue
            name = obj["path"]
            suffix = PurePosixPath(name).suffix.lower()
            if suffix not in AUDIO_EXTENSIONS | {".csv", ".md", ".txt", ".json"}:
                continue
            lfs = obj.get("lfs") or {}
            rows.append(_row(dataset, f"hf:{repo}@{revision}/{name}",
                             f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(name, safe='/')}",
                             int(obj["size"]), relative_path=name,
                             source_family=spec["source_family"], sha256=lfs.get("oid"),
                             role="audio" if suffix in AUDIO_EXTENSIONS else "metadata"))
        url = response.links.get("next", {}).get("url")
        if url and urlsplit(url).netloc != "huggingface.co":
            raise ValueError("Unexpected HF pagination host")
    return rows


def _gcs_plan(dataset, spec, session):
    bucket, prefix = spec["bucket"], spec["prefix"]
    rows, token = [], None
    while True:
        params = {"prefix": prefix, "maxResults": 1000}
        if token:
            params["pageToken"] = token
        page = _json(session, f"https://storage.googleapis.com/storage/v1/b/{bucket}/o", **params)
        for obj in page.get("items", []):
            name = obj["name"]
            if name.endswith("/"):
                continue
            suffix = PurePosixPath(name).suffix.lower()
            provider = name[len(prefix):].split("/", 1)[0]
            family = {"onc": "onc", "orcasound": "orcasound"}.get(provider, spec["source_family"])
            generation = obj.get("generation")
            pinned_url = f"https://storage.googleapis.com/{bucket}/{quote(name, safe='/')}"
            if generation:
                pinned_url += "?generation=" + quote(str(generation), safe="")
            rows.append(_row(dataset, f"gs://{bucket}/{name}",
                             pinned_url,
                             int(obj["size"]), relative_path=name[len(prefix):],
                             source_family=family, provider=provider,
                             md5_base64=obj.get("md5Hash"), crc32c_base64=obj.get("crc32c"),
                             generation=obj.get("generation"),
                             role="audio" if suffix in AUDIO_EXTENSIONS else "metadata"))
        if len(rows) > spec.get("max_objects", 200000):
            raise ValueError(f"{dataset}: metadata object limit exceeded")
        token = page.get("nextPageToken")
        if not token:
            return rows


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href" and v)


def _voices_plan(dataset, spec, session):
    base = spec["url"].rstrip("/") + "/"
    origin = urlsplit(base)
    pending, seen, audio = [base], set(), set()
    while pending:
        url = pending.pop()
        if url in seen:
            continue
        if len(seen) >= spec.get("max_directories", 100):
            raise ValueError("Voices directory limit exceeded")
        seen.add(url)
        response = session.get(url, timeout=(15, 60))
        response.raise_for_status()
        parser = _Links()
        parser.feed(response.text)
        for href in parser.links:
            child = urljoin(url, href)
            parts = urlsplit(child)
            if (parts.scheme != origin.scheme or parts.netloc != origin.netloc
                    or not parts.path.startswith(origin.path) or parts.query or parts.fragment):
                continue
            if parts.path.endswith("/"):
                pending.append(child)
            elif PurePosixPath(parts.path).suffix.lower() in AUDIO_EXTENSIONS:
                audio.add(child)
    rows = []
    for url in sorted(audio):
        response = session.head(url, timeout=(15, 60), allow_redirects=True)
        response.raise_for_status()
        size = response.headers.get("Content-Length")
        rows.append(_row(dataset, url, url, int(size) if size else None,
                         relative_path=unquote(urlsplit(url).path[len(origin.path):]),
                         source_family=spec["source_family"], role="audio"))
    return rows


def plan_benchmarks(config: dict, session=None) -> dict:
    """Return per-dataset plans, unknown sizes, provenance, and explicit blockers.

    Accepts the parsed configs/benchmarks.yaml object. Errors are retained per
    dataset, allowing independent public sources to make progress. Credentials
    and authenticated IEEE requests deliberately stay outside this planner.
    """
    session = session or metadata_session()
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "datasets": {}}
    for dataset, spec in config["datasets"].items():
        entry = {"kind": spec["kind"], "license": spec.get("license"),
                 "source_page": spec.get("source_page"), "ready": False, "files": []}
        result["datasets"][dataset] = entry
        try:
            kind = spec["kind"]
            if kind == "hf":
                files = _hf_plan(dataset, spec, session)
            elif kind == "gcs":
                files = _gcs_plan(dataset, spec, session)
            elif kind == "voices_index":
                files = _voices_plan(dataset, spec, session)
            elif kind == "archive":
                obj = _json(session, spec["metadata_url"])
                pinned_url = spec["url"]
                if obj.get("generation"):
                    pinned_url += ("&" if "?" in pinned_url else "?") + "generation=" + quote(str(obj["generation"]), safe="")
                files = [_row(dataset, spec["url"], pinned_url, int(obj["size"]),
                              relative_path=PurePosixPath(urlsplit(spec["url"]).path).name,
                              md5_base64=obj.get("md5Hash"), source_family=spec["source_family"],
                              role="archive", generation=obj.get("generation"))]
            else:
                entry["status"] = spec.get("status", "requires_local_inventory")
                entry["notes"] = spec.get("notes", "")
                continue
            entry.update(files=files, status="planned_not_downloaded",
                         known_bytes=sum(row["size"] or 0 for row in files),
                         unknown_sizes=sum(row["size"] is None for row in files),
                         audio_files=sum(row["role"] == "audio" for row in files))
            if not files:
                raise ValueError("Source returned no downloadable files")
            if spec.get("notes"):
                entry["notes"] = spec["notes"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            # Public metadata URLs only; avoid request headers or credentials.
            entry.update(status="metadata_error", error=f"{type(exc).__name__}: {exc}")
    return result


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = plan_benchmarks(config)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
