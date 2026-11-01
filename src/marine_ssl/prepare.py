"""Prepare verified 16 kHz mono five-second WAVs in per-recording TAR shards.

Run in an MLS CPU job. Workers prefetch a bounded number of original files;
deterministic streaming conversion commits quotas and duplicate checks serially.
"""

import argparse
import base64
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import uuid
import wave

from .io import digest, download, read_jsonl, write_json, write_jsonl

VERSION = 1


def _key(row):
    return hashlib.sha256(str(row["source_id"]).encode()).hexdigest()


def _signature(config):
    values = {name: config.get(name) for name in ("sample_rate", "channels", "sample_format",
                                                 "segment_seconds", "overlap_seconds")}
    values.update(version=VERSION, max_seconds=config.get("per_record_max_seconds", 3600),
                  quotas={key: (value["target_hours"], value.get("per_record_max_seconds")) for key, value in config["sources"].items()})
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _same_source(saved, row):
    return all(saved.get(key) == row.get(key) for key in
               ("source_id", "url", "revision", "sha256", "source_generation", "generation", "local_path"))


def _source(row, output, reserve_bytes):
    """Transfer only to owned scratch; existing raw originals are read-only."""
    if row.get("local_path"):
        path = Path(row["local_path"])
        if not path.is_file() or path.is_symlink():
            raise ValueError("Existing source must be a regular file")
        if row.get("sha256") and digest(path) != row["sha256"]:
            raise ValueError("Existing source SHA-256 mismatch")
        return path, None
    root = output / ".scratch" / row["source_key"] / _key(row)
    if root.is_symlink():
        raise ValueError("Scratch symlinks are not allowed")
    root.mkdir(parents=True, exist_ok=True)
    owner = root / "owner.json"
    if owner.exists():
        if json.loads(owner.read_text(encoding="utf-8")).get("source_id") != row["source_id"]:
            raise ValueError("Scratch ownership differs")
    elif any(root.iterdir()):
        raise ValueError("Refusing to overwrite scratch without an ownership record")
    else:
        write_json(owner, {"source_id": row["source_id"], "version": VERSION})
    size = row.get("bytes", row.get("size"))
    if size is None:
        raise ValueError("Remote source size required for bounded capacity planning")
    if shutil.disk_usage(output).free < reserve_bytes + int(size):
        raise RuntimeError("Free-space reserve reached")
    suffix = Path(str(row.get("path_in_source") or row.get("url", "")).split("?")[0]).suffix
    target = root / ("source" + suffix)
    url = row["url"]
    generation = row.get("source_generation") or row.get("generation")
    if generation and "storage.googleapis.com/" in url:
        url += ("&" if "?" in url else "?") + "generation=" + str(generation)
    download(url, target, size=int(size), sha256=row.get("sha256"))
    expected = row.get("source_md5_base64") or row.get("md5_base64")
    if expected:
        checksum = hashlib.md5()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                checksum.update(block)
        if base64.b64encode(checksum.digest()).decode() != expected:
            raise ValueError("Published source MD5 mismatch")
    return target, root


def _wav(pcm, rate):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


def _add(archive, name, value):
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(value), 0, 0o644
    archive.addfile(info, io.BytesIO(value))


def _convert(row, raw, scratch, remaining_frames, config, seen_segments, seen_pcm):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "stream=sample_rate,channels:format=duration",
                            "-of", "json", str(raw)], capture_output=True, text=True, check=True)
    header = json.loads(probe.stdout)
    native = header["streams"][0]
    if int(native["sample_rate"]) < config["sample_rate"]:
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-i", str(raw),
                        "-map", "0:a:0", "-f", "null", "-"], capture_output=True, check=True)
        return {"status": "native_rate_below_target", "native": native}, None
    rate, window = config["sample_rate"], config["sample_rate"] * config["segment_seconds"]
    limit = min(remaining_frames, int(config.get("per_record_max_seconds", 3600) * rate))
    temporary = scratch / ("prepared." + uuid.uuid4().hex + ".tar.part")
    selected, local_hashes, pcm_hash = [], set(), hashlib.sha256(b"16000:1:int16le\n")
    decoded_frames, selected_frames, duplicate_windows = 0, 0, 0
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-err_detect", "explode",
                                    "-threads", "1", "-i", str(raw), "-map", "0:a:0", "-vn",
                                    "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"],
                                   stdout=subprocess.PIPE, stderr=errors)
        try:
            with tarfile.open(temporary, "w") as archive:
                while True:
                    block = bytearray()
                    while len(block) < window * 2:
                        data = process.stdout.read(window * 2 - len(block))
                        if not data:
                            break
                        block.extend(data)
                    if not block:
                        break
                    if len(block) % 2:
                        raise ValueError("Partial PCM16 sample")
                    start = decoded_frames
                    decoded_frames += len(block) // 2
                    pcm_hash.update(block)
                    if len(block) != window * 2 or selected_frames + window > limit:
                        continue
                    checksum = hashlib.sha256(block).hexdigest()
                    if checksum in seen_segments or checksum in local_hashes:
                        duplicate_windows += 1
                        continue
                    local_hashes.add(checksum)
                    name = f"audio/{start:014d}.wav"
                    _add(archive, name, _wav(block, rate))
                    selected.append({"member": name, "start_frame": start, "frames": window, "pcm_sha256": checksum})
                    selected_frames += window
                if process.wait() != 0:
                    errors.seek(0)
                    raise ValueError("ffmpeg full decode failed: " + errors.read(2048).decode(errors="replace"))
                expected = float(header.get("format", {}).get("duration") or 0)
                if not decoded_frames or (expected and abs(decoded_frames / rate - expected) > max(0.5, expected * 0.001)):
                    raise ValueError("Full decoded duration differs from source metadata")
                result = {"schema_version": VERSION, "status": "prepared", "source": row,
                          "native": native, "source_sha256": digest(raw), "normalized_pcm_sha256": pcm_hash.hexdigest(),
                          "decoded_frames": decoded_frames, "sample_rate_hz": rate, "channels": 1, "subtype": "PCM_16",
                          "selected_frames": selected_frames, "duration_seconds": selected_frames / rate,
                          "segment_seconds": config["segment_seconds"], "segment_count": len(selected), "segments": selected,
                          "duplicate_windows_skipped": duplicate_windows, "dropped_tail_frames": decoded_frames % window,
                          "per_record_max_seconds": config.get("per_record_max_seconds", 3600), "full_decode_verified": True,
                          "ready_for_training": False}
                _add(archive, "recording.json", (json.dumps(result, ensure_ascii=False) + "\n").encode())
        except BaseException:
            process.kill()
            process.wait()
            if temporary.exists():
                temporary.unlink()
            raise
        finally:
            process.stdout.close()
    if pcm_hash.hexdigest() in seen_pcm or not selected:
        temporary.unlink()  # Exact owned intermediate, never an existing source.
        return {"status": "duplicate_or_no_complete_unique_windows", "source_id": row["source_id"]}, None
    return result, temporary


def prepare(config, candidates, output, *, workers=8):
    """Acquire until actual unique validated window counts reach source quotas."""
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    if (config["sample_rate"], config["channels"], config["sample_format"], config["segment_seconds"], config["overlap_seconds"]) != (16000, 1, "PCM_16", 5, 0):
        raise ValueError("This preparer supports only mono PCM16/16kHz, non-overlapping 5s windows")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be installed in the MLS image")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    signature, rate = _signature(config), config["sample_rate"]
    targets = {source: int(spec["target_hours"] * 3600 * rate) for source, spec in config["sources"].items()}
    if any(not re.fullmatch(r"[a-z0-9_-]+", source) for source in targets):
        raise ValueError("Unsafe source key")
    queues = {source: deque() for source in targets}
    unique = set()
    for row in candidates:
        source = row["source_key"]
        identity = (source, row["source_id"])
        if source not in queues or identity in unique:
            continue
        if (row.get("clearance_scope") != "source_metadata_only"
                or row.get("source_family") != config["sources"][source]["source_family"]):
            raise ValueError("Only planner-cleared pretrain candidates are accepted")
        unique.add(identity)
        queues[source].append(row)
    reserve = int(config.get("reserve_bytes", 400_000_000_000))
    largest = sorted((int(row.get("bytes", row.get("size")) or 0) for rows in queues.values() for row in rows), reverse=True)[:workers]
    existing_frames = Counter()
    for source, rows in queues.items():
        for row in rows:
            receipt = output / "receipts" / source / (_key(row) + ".json")
            if receipt.is_file():
                saved = json.loads(receipt.read_text(encoding="utf-8"))
                if saved.get("configuration_sha256") == signature and _same_source(saved["source"], row):
                    existing_frames[source] += saved["selected_frames"]
    largest_cap = max(spec.get("per_record_max_seconds", config.get("per_record_max_seconds", 3600)) for spec in config["sources"].values())
    required = sum(max(0, targets[key] - existing_frames[key]) for key in targets) * 2 + sum(largest) + workers * int(largest_cap * rate * 2)
    if shutil.disk_usage(output).free < reserve + required:
        raise RuntimeError(f"Capacity check failed: prepared+raw+temporary {required}, reserve {reserve}")
    lockfile = output / ".prepare.lock"
    lock = lockfile.open("x", encoding="utf-8")
    lock.write(json.dumps({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()})); lock.close()
    progress, totals, seen_segments, seen_pcm, manifest, issues = Counter(), Counter(), set(), set(), [], []
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(60):
            print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "progress": dict(progress),
                              "actual_hours": {key: totals[key] / rate / 3600 for key in targets}}), flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()

    source_order = deque(queues)

    def next_row():
        for _ in range(len(source_order)):
            source = source_order[0]
            source_order.rotate(-1)
            if queues[source] and totals[source] < targets[source]:
                return queues[source].popleft()
        return None

    def prefetch(row):
        receipt = output / "receipts" / row["source_key"] / (_key(row) + ".json")
        return (None, None) if receipt.is_file() else _source(row, output, reserve)

    def clean_raw(raw, scratch):
        if scratch:
            for path in (raw, raw.with_name(raw.name + ".receipt.json")):
                if path.is_file() and not path.is_symlink():
                    path.unlink()

    print(json.dumps({"stage": "prepare", "workers": workers, "configuration_sha256": signature,
                      "target_hours": {key: value / rate / 3600 for key, value in targets.items()},
                      "required_bytes_upper_estimate": required, "training_ready": False}), flush=True)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = deque()
            while True:
                while len(pending) < workers:
                    row = next_row()
                    if row is None:
                        break
                    pending.append((row, pool.submit(prefetch, row)))
                if not pending:
                    break
                row, future = pending.popleft()
                source, key = row["source_key"], _key(row)
                try:
                    raw, scratch = future.result()
                    if totals[source] >= targets[source]:
                        continue  # Retain bounded prefetched scratch for a future run.
                    shard = output / "shards" / source / (key + ".tar")
                    receipt_path = output / "receipts" / source / (key + ".json")
                    if receipt_path.exists():
                        result = json.loads(receipt_path.read_text(encoding="utf-8"))
                        if (result.get("configuration_sha256") != signature or not _same_source(result["source"], row)
                                or not shard.is_file() or digest(shard) != result.get("tar_sha256")):
                            raise ValueError("Existing prepared artifact is incompatible or corrupt; preserve and inspect")
                        hashes = {item["pcm_sha256"] for item in result["segments"]}
                        if hashes & seen_segments or result["normalized_pcm_sha256"] in seen_pcm:
                            raise ValueError("Existing prepared artifacts overlap; preserve and inspect")
                    else:
                        if shard.exists():
                            raise ValueError("Existing TAR lacks receipt; preserve and inspect")
                        (output / ".scratch").mkdir(exist_ok=True)
                        work = scratch or Path(tempfile.mkdtemp(prefix="prepare-", dir=output / ".scratch"))
                        effective = {**config, "per_record_max_seconds": config["sources"][source].get("per_record_max_seconds", config.get("per_record_max_seconds", 3600))}
                        temporary_bytes = min(targets[source] - totals[source], int(effective["per_record_max_seconds"] * rate)) * 2
                        if shutil.disk_usage(output).free < reserve + temporary_bytes:
                            raise RuntimeError("Free-space reserve reached before conversion")
                        result, temporary = _convert(row, raw, work, targets[source] - totals[source], effective, seen_segments, seen_pcm)
                        if temporary is None:
                            issues.append({"source_key": source, "source_id": row["source_id"], **result})
                            clean_raw(raw, scratch)
                            continue
                        shard.parent.mkdir(parents=True, exist_ok=True)
                        result.update(configuration_sha256=signature, tar_sha256=digest(temporary), tar_path=str(shard))
                        os.replace(temporary, shard)
                        write_json(receipt_path, result)
                        hashes = {item["pcm_sha256"] for item in result["segments"]}
                    if result["selected_frames"] > targets[source] - totals[source]:
                        raise ValueError("Existing receipt exceeds remaining quota; use a compatible output version")
                    seen_segments.update(hashes); seen_pcm.add(result["normalized_pcm_sha256"])
                    totals[source] += result["selected_frames"]
                    manifest.append({"source_key": source, "source_id": row["source_id"], "tar_path": str(shard),
                                     "receipt_path": str(receipt_path), "segment_count": result["segment_count"],
                                     "duration_seconds": result["duration_seconds"], "source": row})
                    progress["prepared_recordings"] += 1; progress["segments"] += result["segment_count"]
                    clean_raw(raw, scratch)
                except Exception as exc:
                    progress["errors"] += 1
                    issues.append({"source_key": source, "source_id": row["source_id"], "status": "error",
                                   "error": f"{type(exc).__name__}: {exc}"})
                    # Keep failed source/partials for inspection; stopping keeps
                    # retained raw files bounded by the prefetch worker count.
                    break
        summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "sources": {
            source: {"target_hours": targets[source] / rate / 3600, "actual_hours": totals[source] / rate / 3600,
                     "actual_frames": totals[source], "quota_met": totals[source] >= targets[source]} for source in targets},
            "counts": dict(progress), "training_ready": False,
            "notes": "Exact PCM/chunk dedup only; retain the source/session exclusions and unresolved benchmark protocol blockers."}
        write_jsonl(output / "prepared_manifest.jsonl", manifest)
        write_jsonl(output / "preparation_issues.jsonl", issues)
        write_json(output / "preparation.json", summary)
        print(json.dumps(summary), flush=True)
        return summary
    finally:
        stop.set(); thread.join(timeout=1); lockfile.unlink()


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "candidates", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    summary = prepare(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), read_jsonl(args.candidates), args.output, workers=args.workers)
    return int(not all(source["quota_met"] for source in summary["sources"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
