"""Prepare verified 16 kHz mono five-second WAVs in per-recording TAR shards.

Run in an MLS CPU job. Workers prefetch a bounded number of original files;
bounded concurrent conversion uses one coordinator for ordered quota/dedup commits.
"""

import argparse
from contextlib import contextmanager
from functools import lru_cache
import base64
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
import time
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


@lru_cache(maxsize=1)
def _ffmpeg_version():
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    return result.stdout.splitlines()[0]


@contextmanager
def _pcm_blocks(raw, config):
    """Full-source decode with bounded fast-path eligibility and streaming fallback."""
    rate, width = config["sample_rate"], config["sample_rate"] * config["segment_seconds"] * 2
    from .fast_audio import decode_pcm16
    fast = decode_pcm16(raw, target_rate=rate, max_seconds=config.get("per_record_max_seconds", 3600))
    if fast is not None:
        metadata, pcm = fast
        if pcm is not None:
            yield metadata, (memoryview(pcm)[offset:offset + width] for offset in range(0, len(pcm), width))
            return
        if metadata.get("status") == "native_rate_below_target":
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-i", str(raw),
                            "-map", "0:a:0", "-f", "null", "-"], capture_output=True, check=True)
            yield {**metadata, "full_decode_verified": True}, iter(())
            return
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "stream=sample_rate,channels:format=duration", "-of", "json", str(raw)],
                           capture_output=True, text=True, check=True)
    header = json.loads(probe.stdout)
    native = header["streams"][0]
    metadata = {"native": native, "backend": "ffmpeg", "versions": {"ffmpeg": _ffmpeg_version()},
                "sample_rate_hz": rate, "channels": 1, "full_decode_verified": True}
    if int(native["sample_rate"]) < rate:
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-i", str(raw),
                        "-map", "0:a:0", "-f", "null", "-"], capture_output=True, check=True)
        yield {**metadata, "status": "native_rate_below_target"}, iter(())
        return
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-err_detect", "explode",
                                    "-threads", "1", "-filter_threads", "1", "-i", str(raw), "-map", "0:a:0", "-vn",
                                    "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"],
                                   stdout=subprocess.PIPE, stderr=errors)
        frames = 0
        def blocks():
            nonlocal frames
            while True:
                block = bytearray()
                while len(block) < width:
                    value = process.stdout.read(width - len(block))
                    if not value:
                        break
                    block.extend(value)
                if not block:
                    return
                if len(block) % 2:
                    raise ValueError("Partial PCM16 sample")
                frames += len(block) // 2
                yield block
        try:
            yield metadata, blocks()
            if process.wait() != 0:
                errors.seek(0)
                raise ValueError("ffmpeg full decode failed: " + errors.read(2048).decode(errors="replace"))
            expected = float(header.get("format", {}).get("duration") or 0)
            if not frames or (expected and abs(frames / rate - expected) > max(0.5, expected * 0.001)):
                raise ValueError("Full decoded duration differs from source metadata")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()


def _convert(row, raw, scratch, remaining_frames, config, seen_segments, seen_pcm):
    rate, window = config["sample_rate"], config["sample_rate"] * config["segment_seconds"]
    limit = min(remaining_frames, int(config.get("per_record_max_seconds", 3600) * rate))
    temporary = scratch / ("prepared." + uuid.uuid4().hex + ".tar.part")
    selected, local_hashes = [], set()
    pcm_hash = hashlib.sha256(b"16000:1:int16le\n")
    decoded_frames = selected_frames = duplicate_windows = 0
    try:
        with _pcm_blocks(raw, config) as (metadata, blocks):
            if metadata.get("status") == "native_rate_below_target":
                return {"status": "native_rate_below_target", "native": metadata["native"],
                        "decoder_backend": metadata["backend"], "decoder_versions": metadata.get("versions", {})}, None
            with tarfile.open(temporary, "w") as archive:
                for block in blocks:
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
                if metadata.get("decoded_frames") is not None and decoded_frames != metadata["decoded_frames"]:
                    raise ValueError("Decoded frame count differs from decoder metadata")
                result = {"schema_version": VERSION, "status": "prepared", "source": row,
                          "native": metadata["native"], "source_sha256": digest(raw),
                          "normalized_pcm_sha256": pcm_hash.hexdigest(), "decoded_frames": decoded_frames,
                          "sample_rate_hz": rate, "channels": 1, "subtype": "PCM_16",
                          "selected_frames": selected_frames, "duration_seconds": selected_frames / rate,
                          "segment_seconds": config["segment_seconds"], "segment_count": len(selected), "segments": selected,
                          "duplicate_windows_skipped": duplicate_windows, "dropped_tail_frames": decoded_frames % window,
                          "per_record_max_seconds": config.get("per_record_max_seconds", 3600),
                          "full_decode_verified": True, "ready_for_training": False,
                          "decoder_backend": metadata["backend"], "decoder_versions": metadata.get("versions", {})}
                _add(archive, "recording.json", (json.dumps(result, ensure_ascii=False) + "\n").encode())
        if pcm_hash.hexdigest() in seen_pcm or not selected:
            temporary.unlink()
            return {"status": "duplicate_or_no_complete_unique_windows", "source_id": row["source_id"]}, None
        return result, temporary
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _trim_shard(result, temporary, kept, *, duplicate_count, quota_count):
    """Copy unchanged WAV entries only when coordinator exclusions require it."""
    result = {**result, "segments": kept, "segment_count": len(kept),
              "selected_frames": sum(segment["frames"] for segment in kept),
              "coordinator_exclusions": {"global_duplicate_windows": duplicate_count,
                                         "quota_trimmed_windows": quota_count}}
    result["duration_seconds"] = result["selected_frames"] / result.get("sample_rate_hz", 16000)
    trimmed = temporary.with_name("trimmed." + uuid.uuid4().hex + ".tar.part")
    try:
        with tarfile.open(temporary, "r") as source, tarfile.open(trimmed, "w") as target:
            for segment in kept:
                member = source.getmember(segment["member"])
                with source.extractfile(member) as stream:
                    target.addfile(member, stream)
            _add(target, "recording.json", (json.dumps(result, ensure_ascii=False) + "\n").encode())
        temporary.unlink()
        return result, trimmed
    except BaseException:
        if trimmed.exists():
            trimmed.unlink()
        raise


def prepare(config, candidates, output, *, workers=8, download_workers=None):
    """Parallel bounded transfer/decode, with ordered serial quota/dedup commits."""
    download_workers = workers if download_workers is None else download_workers
    if not 1 <= workers <= 32 or not 1 <= download_workers <= 64:
        raise ValueError("decode workers must be 1..32 and download workers 1..64")
    if (config["sample_rate"], config["channels"], config["sample_format"], config["segment_seconds"], config["overlap_seconds"]) != (16000, 1, "PCM_16", 5, 0):
        raise ValueError("Only mono PCM16/16kHz, non-overlapping five-second windows are supported")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required in the MLS image")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    signature, rate = _signature(config), config["sample_rate"]
    window = rate * config["segment_seconds"]
    targets = {source: int(spec["target_hours"] * 3600 * rate) for source, spec in config["sources"].items()}
    if any(not re.fullmatch(r"[a-z0-9_-]+", source) for source in targets):
        raise ValueError("Unsafe source key")
    rows_by_key, queues, identities = {}, {source: deque() for source in targets}, set()
    for row in candidates:
        source = row["source_key"]
        if source not in queues:
            continue
        identity = (row.get("source_family"), row.get("source_recording_id") or row["source_id"])
        if identity in identities:
            continue
        if row.get("clearance_scope") != "source_metadata_only" or row.get("source_family") != config["sources"][source]["source_family"]:
            raise ValueError("Only planner-cleared pretrain candidates are accepted")
        identities.add(identity); rows_by_key[(source, _key(row))] = row; queues[source].append(row)
    lockfile = output / ".prepare.lock"
    with lockfile.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}))
    progress, totals, manifest, issues = Counter(), Counter(), [], []
    seen_segments, seen_pcm, seen_raw = set(), set(), set()
    stop, last_checkpoint = threading.Event(), 0.0
    reserve = int(config.get("reserve_bytes", 400_000_000_000))
    stage = "verify_existing"

    def summary(status):
        return {"updated_utc": datetime.now(timezone.utc).isoformat(), "status": status,
                "sources": {key: {"actual_frames": totals[key], "actual_hours": totals[key] / rate / 3600,
                                  "target_hours": targets[key] / rate / 3600,
                                  "quota_met": totals[key] >= targets[key]} for key in targets},
                "counts": dict(progress), "decode_workers": workers, "download_workers": download_workers,
                "training_ready": False, "last_issue": issues[-1] if issues else None}

    def checkpoint(status, force=False):
        nonlocal last_checkpoint
        now = time.monotonic()
        if not force and now - last_checkpoint < 60:
            return
        last_checkpoint = now
        write_json(output / "preparation.json", summary(status))
        if force:
            write_jsonl(output / "preparation_issues.jsonl", issues)
            write_jsonl(output / "prepared_manifest.jsonl", manifest)

    def heartbeat():
        while not stop.wait(60):
            print(json.dumps({"stage": stage, **summary("running")}), flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()

    def clean_raw(task):
        raw, scratch = task.get("raw"), task.get("scratch")
        if raw is not None and scratch:
            for path in (raw, raw.with_name(raw.name + ".receipt.json")):
                if path.is_file() and not path.is_symlink():
                    path.unlink()

    def paths(row):
        source, key = row["source_key"], _key(row)
        return output / "shards" / source / (key + ".tar"), output / "receipts" / source / (key + ".json")

    def add_committed(row, result, shard, receipt, reused=False):
        hashes = {segment["pcm_sha256"] for segment in result["segments"]}
        if hashes & seen_segments or result["normalized_pcm_sha256"] in seen_pcm:
            raise ValueError("Existing prepared artifacts overlap; preserve and inspect")
        if result["selected_frames"] > targets[row["source_key"]] - totals[row["source_key"]]:
            raise ValueError("Existing receipt exceeds remaining quota; preserve and inspect")
        seen_segments.update(hashes); seen_pcm.add(result["normalized_pcm_sha256"])
        if result.get("source_sha256"):
            seen_raw.add(result["source_sha256"])
        totals[row["source_key"]] += result["selected_frames"]
        manifest.append({"source_key": row["source_key"], "source_id": row["source_id"],
                         "tar_path": str(shard), "receipt_path": str(receipt), "source": row,
                         "segment_count": result["segment_count"], "duration_seconds": result["duration_seconds"],
                         "decoder_backend": result.get("decoder_backend", "ffmpeg"),
                         "decoder_versions": result.get("decoder_versions", {"ffmpeg": "legacy-unrecorded"}),
                         "selection_order": result.get("selection_order"), "selected_at_utc": result.get("selected_at_utc")})
        progress["prepared_recordings"] += 1; progress["segments"] += result["segment_count"]
        progress["reused_recordings" if reused else "new_recordings"] += 1
        checkpoint("running")

    def verify_existing(item):
        row, receipt = item
        shard, expected = paths(row)
        result = json.loads(receipt.read_text(encoding="utf-8"))
        if receipt != expected or result.get("configuration_sha256") != signature or not _same_source(result["source"], row):
            raise ValueError("Existing prepared receipt is incompatible; preserve and inspect")
        if not shard.is_file() or digest(shard) != result.get("tar_sha256"):
            raise ValueError("Existing prepared TAR is missing or corrupt; preserve and inspect")
        if (len(result["segments"]) != result["segment_count"]
                or sum(segment.get("frames", window) for segment in result["segments"]) != result["selected_frames"]):
            raise ValueError("Existing receipt frame/segment counts differ")
        return row, result, shard, receipt

    def decode_task(task):
        row, raw = task["row"], task["raw"]
        shard, receipt = paths(row)
        if shard.exists() or receipt.exists():
            raise ValueError("Unclaimed prepared artifact exists; preserve and inspect")
        (output / ".scratch").mkdir(exist_ok=True)
        work = task["scratch"] or Path(tempfile.mkdtemp(prefix="prepare-", dir=output / ".scratch"))
        effective = {**config, "per_record_max_seconds": config["sources"][row["source_key"]].get(
            "per_record_max_seconds", config.get("per_record_max_seconds", 3600))}
        record_frames = int(effective["per_record_max_seconds"] * rate)
        allowance = record_frames * 2
        if shutil.disk_usage(output).free < reserve + allowance:
            raise RuntimeError("Free-space reserve reached before conversion")
        try:
            result, temporary = _convert(row, raw, work, record_frames, effective, frozenset(), frozenset())
            return {"result": result, "temporary": temporary}
        except (ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
            return {"error": exc, "isolated_decode_error": True}

    def commit(task, outcome):
        result, temporary, row = outcome["result"], outcome["temporary"], task["row"]
        if temporary is None:
            issues.append({"source_key": row["source_key"], "source_id": row["source_id"], **result})
            clean_raw(task); progress[result["status"]] += 1
            return
        duplicate_pcm = result["normalized_pcm_sha256"] in seen_pcm or (
            result.get("source_sha256") and result["source_sha256"] in seen_raw)
        remaining = targets[row["source_key"]] - totals[row["source_key"]]
        kept, duplicate_count, quota_count, kept_frames, local = [], 0, 0, 0, set()
        for segment in result["segments"]:
            checksum, frames = segment["pcm_sha256"], segment.get("frames", window)
            if duplicate_pcm or checksum in seen_segments or checksum in local:
                duplicate_count += 1; continue
            if kept_frames + frames > remaining:
                quota_count += 1; continue
            kept.append(segment); kept_frames += frames; local.add(checksum)
        if not kept:
            temporary.unlink(); clean_raw(task)
            progress["duplicate_or_quota_discarded"] += 1
            return
        if len(kept) != len(result["segments"]):
            result, temporary = _trim_shard(result, temporary, kept, duplicate_count=duplicate_count, quota_count=quota_count)
        else:
            result["coordinator_exclusions"] = {"global_duplicate_windows": 0, "quota_trimmed_windows": 0}
        shard, receipt = paths(row); shard.parent.mkdir(parents=True, exist_ok=True)
        result.update(configuration_sha256=signature, tar_sha256=digest(temporary), tar_path=str(shard),
                      selection_order=len(manifest), candidate_sequence=task["sequence"],
                      selected_at_utc=datetime.now(timezone.utc).isoformat())
        result.setdefault("decoder_backend", "ffmpeg")
        result.setdefault("decoder_versions", {"ffmpeg": "legacy-unrecorded"})
        os.replace(temporary, shard); write_json(receipt, result)
        add_committed(row, result, shard, receipt); clean_raw(task)

    print(json.dumps({"stage": "prepare", "configuration_sha256": signature, "decode_workers": workers,
                      "download_workers": download_workers, "training_ready": False}), flush=True)
    try:
        existing = []
        for source in targets:
            directory = output / "receipts" / source
            if directory.is_dir():
                for receipt in sorted(directory.glob("*.json")):
                    row = rows_by_key.get((source, receipt.stem))
                    if row is not None:
                        existing.append((row, receipt))
        reused_keys = set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row, result, shard, receipt in pool.map(verify_existing, existing):
                add_committed(row, result, shard, receipt, reused=True)
                reused_keys.add((row["source_key"], _key(row)))
        for source in queues:
            queues[source] = deque(row for row in queues[source] if (source, _key(row)) not in reused_keys)
        largest = sorted((int(row.get("bytes", row.get("size")) or 0) for rows in queues.values() for row in rows), reverse=True)[:download_workers + workers]
        largest_cap = max(spec.get("per_record_max_seconds", config.get("per_record_max_seconds", 3600)) for spec in config["sources"].values())
        required = sum(max(0, targets[key] - totals[key]) for key in targets) * 2 + sum(largest) + workers * int(largest_cap * rate * 2)
        if shutil.disk_usage(output).free < reserve + required:
            raise RuntimeError(f"Capacity check failed: remaining prepared/raw/temp {required}, reserve {reserve}")
        stage = "parallel_prepare"
        max_active = int(config.get("max_inflight_records", download_workers + workers))
        heavy_limit = int(config.get("heavy_download_workers_per_source", 2))
        heavy_bytes = int(config.get("heavy_source_bytes", 256 * 1024 ** 2))
        active, downloads, decodes, raw_ready = {}, {}, {}, deque()
        ready = {source: {} for source in targets}
        admitted, expected, inflight, heavy = Counter(), Counter(), Counter(), Counter()
        source_order, serial = deque(targets), 0
        consecutive_errors, abort = 0, False

        def release(task):
            active.pop(task["id"], None); source = task["row"]["source_key"]
            inflight[source] -= 1
            if task["heavy"]: heavy[source] -= 1

        def next_row():
            available = [key for key in source_order if queues[key] and totals[key] < targets[key]
                         and (int(queues[key][0].get("bytes", queues[key][0].get("size")) or 0) < heavy_bytes or heavy[key] < heavy_limit)]
            if not available: return None
            source = min(available, key=lambda key: inflight[key])
            while source_order[0] != source: source_order.rotate(-1)
            source_order.rotate(-1)
            return queues[source].popleft()

        with ThreadPoolExecutor(max_workers=download_workers) as transfers, ThreadPoolExecutor(max_workers=workers) as converters:
            while True:
                while len(downloads) < download_workers and len(active) < max_active and not abort:
                    row = next_row()
                    if row is None: break
                    source = row["source_key"]
                    task = {"id": serial, "row": row, "sequence": admitted[source],
                            "heavy": int(row.get("bytes", row.get("size")) or 0) >= heavy_bytes}
                    serial += 1; admitted[source] += 1; inflight[source] += 1
                    if task["heavy"]: heavy[source] += 1
                    active[task["id"]] = task
                    downloads[transfers.submit(_source, row, output, reserve)] = task
                while raw_ready and len(decodes) < workers and not abort:
                    task = raw_ready.popleft(); source = task["row"]["source_key"]
                    if totals[source] >= targets[source]:
                        ready[source][task["sequence"]] = (task, {"skipped_after_quota": True})
                    else:
                        decodes[converters.submit(decode_task, task)] = task
                for source in targets:
                    while expected[source] in ready[source] and not abort:
                        task, outcome = ready[source].pop(expected[source]); expected[source] += 1
                        if "error" in outcome:
                            exc = outcome["error"]; consecutive_errors += 1; progress["errors"] += 1
                            issue = {"source_key": source, "source_id": task["row"]["source_id"],
                                     "error": f"{type(exc).__name__}: {exc}", "status": "error",
                                     "isolated_decode_error": outcome.get("isolated_decode_error", False)}
                            issues.append(issue); print(json.dumps({"stage": "prepare_error", **issue}), flush=True)
                            checkpoint("error", force=True)
                            if issue["isolated_decode_error"]: clean_raw(task)
                            if not issue["isolated_decode_error"] or consecutive_errors >= 10: abort = True
                        elif not outcome.get("skipped_after_quota"):
                            try:
                                commit(task, outcome); consecutive_errors = 0
                            except Exception as exc:
                                progress["errors"] += 1
                                issue = {"source_key": source, "source_id": task["row"]["source_id"],
                                         "status": "commit_error", "error": f"{type(exc).__name__}: {exc}"}
                                issues.append(issue)
                                print(json.dumps({"stage": "prepare_error", **issue}), flush=True)
                                checkpoint("commit_error", force=True)
                                abort = True
                        release(task)
                if abort:
                    for future in list(downloads) + list(decodes): future.cancel()
                    print(json.dumps({"stage": "waiting_for_prefetch_shutdown", "downloads": len(downloads), "decodes": len(decodes)}), flush=True)
                    break
                pending = list(downloads) + list(decodes)
                if not pending:
                    if raw_ready: continue
                    if not active: break
                    raise RuntimeError("Ordered commit pipeline cannot make progress")
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(completed, key=lambda item: (downloads.get(item) or decodes[item])["id"]):
                    if future in downloads:
                        task = downloads.pop(future)
                        try:
                            task["raw"], task["scratch"] = future.result(); raw_ready.append(task)
                            progress["downloaded_recordings"] += 1
                        except Exception as exc:
                            ready[task["row"]["source_key"]][task["sequence"]] = (task, {"error": exc})
                    else:
                        task = decodes.pop(future)
                        try: outcome = future.result()
                        except Exception as exc: outcome = {"error": exc}
                        ready[task["row"]["source_key"]][task["sequence"]] = (task, outcome)
                checkpoint("running")
        result = summary("complete" if all(totals[key] >= targets[key] for key in targets) else "incomplete")
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        checkpoint(result["status"], force=True); write_json(output / "preparation.json", result)
        print(json.dumps(result), flush=True)
        return result
    except BaseException as exc:
        issues.append({"status": "fatal_error", "error": f"{type(exc).__name__}: {exc}"})
        print(json.dumps({"stage": "prepare_error", **issues[-1]}), flush=True)
        checkpoint("fatal_error", force=True)
        raise
    finally:
        stop.set(); thread.join(timeout=1); lockfile.unlink()


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "candidates", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent decoders")
    parser.add_argument("--download-workers", type=int, help="Concurrent transfers; defaults to --workers")
    args = parser.parse_args()
    result = prepare(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), read_jsonl(args.candidates),
                     args.output, workers=args.workers, download_workers=args.download_workers)
    return int(not all(source["quota_met"] for source in result["sources"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
