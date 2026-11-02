"""Acquire an explicit benchmark plan, preserving native audio and annotations."""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import threading
import zipfile

from .io import client, download, read_jsonl, write_json, write_jsonl
from .sources import AUDIO_SUFFIXES


def confined(root, relative):
    relative = PurePosixPath(str(relative).replace("\\", "/"))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts or any(":" in part for part in relative.parts):
        raise ValueError("Unsafe source path")
    root = Path(root).resolve()
    result = root.joinpath(*relative.parts).resolve()
    result.relative_to(root)
    return result


def validate_audio(path):
    import soundfile as sf
    pcm = hashlib.sha256()
    with sf.SoundFile(path) as audio:
        frames, rate, channels = len(audio), audio.samplerate, audio.channels
        if not frames or not rate or not channels:
            raise ValueError("Empty or invalid audio")
        decoded = 0
        pcm.update(f"{rate}:{channels}:int32le\n".encode())
        for block in audio.blocks(blocksize=65536, dtype="int32", always_2d=True):
            decoded += len(block)
            pcm.update(block.astype("<i4", copy=False).tobytes())
        if decoded != frames:
            raise ValueError("Decoded frame count differs from header")
        return {"frames": frames, "sample_rate_hz": rate, "channels": channels,
                "duration_seconds": frames / rate, "subtype": audio.subtype,
                "pcm_sha256": pcm.hexdigest(), "full_decode_verified": True}


def verify_md5(path, expected):
    if not expected:
        return
    checksum = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    if base64.b64encode(checksum.digest()).decode() != expected:
        raise ValueError("Published MD5 mismatch")


def extract_zip(path, root, *, reserve_bytes=0):
    """Preflight paths, links, duplicates and expanded size; verify member CRCs."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    with zipfile.ZipFile(path) as archive:
        members = []
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            target = confined(root, entry.filename)
            if target in seen or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Duplicate or symlink ZIP entry")
            seen.add(target)
            members.append((entry, target))
        expanded = sum(entry.file_size for entry, _ in members)
        if shutil.disk_usage(root).free < reserve_bytes + expanded:
            raise RuntimeError("ZIP expanded size would cross free-space reserve")
        for entry, target in members:
            if shutil.disk_usage(root).free < reserve_bytes + entry.file_size:
                raise RuntimeError("Free-space reserve reached during ZIP extraction")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with archive.open(entry) as source, temporary.open("wb") as dest:
                shutil.copyfileobj(source, dest, 1024 * 1024)
            if temporary.stat().st_size != entry.file_size:
                raise ValueError("ZIP member size mismatch")
            os.replace(temporary, target)
            row = {"path": str(target), "archive_member": entry.filename, "bytes": entry.file_size}
            if target.suffix.lower() in AUDIO_SUFFIXES:
                row.update(validate_audio(target))
            rows.append(row)
    return rows


def acquire(plan, output, *, datasets=None, workers=8, reserve_bytes=400_000_000_000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = {name: entry for name, entry in plan["datasets"].items()
                if (not datasets or name in datasets) and entry.get("files")}
    for name, entry in selected.items():
        confined(output, name)
        for row in entry["files"]:
            if row.get("dataset") != name:
                raise ValueError("Plan row dataset differs from its dataset entry")
            if row.get("size") is None or not isinstance(row["size"], int) or row["size"] < 0:
                raise ValueError("Resolve a bounded source size before acquisition")
    required = sum(row.get("size") or 0 for entry in selected.values() for row in entry["files"])
    # Archives need room for both the transfer and extracted original files.
    required += sum(row.get("size") or 0 for entry in selected.values() for row in entry["files"]
                    if row.get("role") == "archive")
    free = shutil.disk_usage(output).free
    if required + reserve_bytes > free:
        raise RuntimeError(f"Capacity check failed: require {required}+{reserve_bytes} reserve, free {free}")
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(), "disk_free_before": free,
               "planned_bytes_with_archive_allowance": required, "reserve_bytes": reserve_bytes,
               "datasets": {}, "ready": False}
    stop = threading.Event()
    counts = Counter()
    lock = threading.Lock()

    def heartbeat():
        while not stop.wait(60):
            with lock:
                print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "progress": dict(counts)}), flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()

    def transfer(row):
        dataset_root = confined(output, row["dataset"])
        target = confined(dataset_root / "raw", row["relative_path"])
        if shutil.disk_usage(output).free < reserve_bytes + 2 * (row.get("size") or 0):
            raise RuntimeError("Free-space reserve reached; resume after capacity review")
        url = row["url"]
        if row.get("generation") and "storage.googleapis.com" in url and "?" not in url:
            url += "?generation=" + str(row["generation"])
        download(url, target, size=row.get("size"), sha256=row.get("sha256"))
        try:
            verify_md5(target, row.get("md5_base64"))
        except ValueError:
            # A receipt verifies local stability, not agreement with source MD5.
            # Invalidate it so the next run performs a fresh transfer.
            target.with_name(target.name + ".receipt.json").unlink(missing_ok=True)
            raise
        receipt = json.loads(target.with_name(target.name + ".receipt.json").read_text())
        result = {**row, "path": str(target), "sha256": receipt["sha256"],
                  "source_md5_verified": bool(row.get("md5_base64"))}
        if row.get("role") == "audio":
            result.update(validate_audio(target))
        elif row.get("role") == "archive":
            extracted = extract_zip(target, dataset_root / "extracted", reserve_bytes=reserve_bytes)
            write_jsonl(dataset_root / "extracted_manifest.jsonl", extracted)
            result["extracted_files"] = len(extracted)
            result["audio_files"] = sum(bool(item.get("full_decode_verified")) for item in extracted)
            result["duration_seconds"] = sum(item.get("duration_seconds", 0) for item in extracted)
            result["zip_crc_verified"] = True
        with lock:
            counts["completed_files"] += 1
            counts["completed_bytes"] += target.stat().st_size
        return result

    try:
        for name, entry in selected.items():
            rows, errors = [], []
            print(json.dumps({"dataset": name, "stage": "acquire", "files": len(entry["files"]),
                              "known_bytes": entry.get("known_bytes")}), flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(transfer, row): row for row in entry["files"]}
                for future in as_completed(futures):
                    try:
                        rows.append(future.result())
                    except Exception as exc:
                        row = futures[future]
                        errors.append({"source_id": row["source_id"], "error": f"{type(exc).__name__}: {exc}"})
                        with lock:
                            counts["errors"] += 1
            write_jsonl(output / name / "manifest.jsonl", sorted(rows, key=lambda row: row["source_id"]))
            status = {"planned_files": len(entry["files"]), "verified_files": len(rows), "errors": errors,
                      "duration_seconds": sum(row.get("duration_seconds", 0) for row in rows),
                      "transfer_and_decode_complete": not errors and len(rows) == len(entry["files"]),
                      "ready": False, "notes": entry.get("notes")}
            # Readiness additionally requires annotation/split/provenance validation.
            summary["datasets"][name] = status
            write_json(output / name / "validation.json", status)
            write_json(output / "acquisition.json", summary)
    finally:
        stop.set()
        thread.join(timeout=1)
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "acquisition.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reserve-gb", type=float, default=400)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be between 1 and 32")
    result = acquire(json.loads(Path(args.plan).read_text()), args.output, datasets=args.datasets,
                     workers=args.workers, reserve_bytes=int(args.reserve_gb * 1e9))
    raise SystemExit(any(entry["errors"] for entry in result["datasets"].values()))
