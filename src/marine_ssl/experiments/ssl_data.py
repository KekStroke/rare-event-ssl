"""Streaming prepared-TAR input and reproducible, session-disjoint SSL splits.

Index construction reads metadata only and belongs in an MLS CPU job for the
full corpus. Reading audio never extracts members or creates an audio cache.
"""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tarfile
import uuid
import wave


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _rows(path, digest=None):
    with Path(path).open("rb") as handle:
        for number, line in enumerate(handle, 1):
            if digest is not None:
                digest.update(line)
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            yield number, value


def _identity(row):
    source = row.get("source") or {}
    family = source.get("source_family") or row.get("source_family") or row.get("source_key")
    recording = source.get("source_recording_id") or row.get("source_recording_id") or row.get("source_id") or source.get("source_id")
    if not family or not recording:
        raise ValueError("a source family and original recording identity are required")
    # Never split one explicit acquisition session/deployment into individual
    # windows. A recording is the fallback when no broader group is recorded.
    field = next((name for name in ("session_id", "deployment_id") if source.get(name)), None)
    group = f"{field}:{source[field]}" if field else f"recording:{recording}"
    return str(family), str(recording), json.dumps([str(family), group], ensure_ascii=False, separators=(",", ":"))


def build_ssl_index(prepared_manifest, output_dir, *, validation_fraction: float,
                    seed: int, sample_rate_hz: int = 16000, frames_per_sample: int = 80000) -> dict:
    """Write train/validation shard JSONLs plus fingerprinted ``split.json``.

    Two streaming passes retain only O(group count) bookkeeping, not one entry
    per WAV. Groups are ranked by seeded SHA256 independently per source family;
    ``floor(groups*fraction + .5)`` groups enter validation. Small sources may
    have an empty split, which is explicitly reported rather than overridden.
    The original manifest and each emitted index receive SHA256 fingerprints.
    No TAR/audio is opened and no audio-integrity claim is made by this step.
    """
    fraction = float(validation_fraction)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("validation_fraction must be in [0,1]")
    if sample_rate_hz <= 0 or frames_per_sample <= 0:
        raise ValueError("positive sample rate and frame count required")
    groups, seen_paths = {}, set()
    first_hash = hashlib.sha256()
    total_shards = total_samples = 0
    for _, row in _rows(prepared_manifest, first_hash):
        family, _, group = _identity(row)
        tar_path = str(row["tar_path"])
        if tar_path in seen_paths:
            raise ValueError(f"duplicate TAR path: {tar_path}")
        seen_paths.add(tar_path)
        count = row["segment_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("segment_count must be a positive integer")
        if "duration_seconds" in row and not math.isclose(float(row["duration_seconds"]), count * frames_per_sample / sample_rate_hz, abs_tol=1e-6):
            raise ValueError(f"segment duration/count mismatch: {tar_path}")
        groups[group] = family
        total_shards += 1
        total_samples += count
    del seen_paths
    strata = defaultdict(list)
    for group, family in groups.items():
        strata[family].append(group)
    assignments, source_groups = {}, {}
    for family, members in sorted(strata.items()):
        ordered = sorted(members, key=lambda group: (hashlib.sha256(_json_bytes([seed, group])).hexdigest(), group))
        validation_count = math.floor(len(ordered) * fraction + .5)
        for i, group in enumerate(ordered):
            assignments[group] = "validation" if i < validation_count else "train"
        source_groups[family] = {"train": len(ordered) - validation_count, "validation": validation_count}
    assignment_hash = hashlib.sha256()
    for group, split in sorted(assignments.items()):
        assignment_hash.update(_json_bytes([group, split]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    paths = {split: output / f"{split}.jsonl" for split in ("train", "validation")}
    temporary = {split: output / f".{split}.{token}.part" for split in paths}
    handles, hashes = {}, {split: hashlib.sha256() for split in paths}
    counts = {split: Counter(shards=0, samples=0) for split in paths}
    by_source = defaultdict(lambda: {split: Counter(shards=0, samples=0) for split in paths})
    second_hash = hashlib.sha256()
    try:
        handles = {split: path.open("wb") for split, path in temporary.items()}
        for line_number, row in _rows(prepared_manifest, second_hash):
            family, recording, group = _identity(row)
            split = assignments[group]
            item = {**row, "source_family": family, "recording_id": recording,
                    "group_id": group, "split": split, "manifest_line": line_number,
                    "sample_rate_hz": sample_rate_hz, "frames_per_sample": frames_per_sample}
            content = _json_bytes(item)
            handles[split].write(content)
            hashes[split].update(content)
            key = str(row.get("source_key") or family)
            for counter in (counts[split], by_source[key][split]):
                counter.update(shards=1, samples=row["segment_count"])
        if first_hash.digest() != second_hash.digest():
            raise ValueError("prepared manifest changed while building the split")
        for handle in handles.values():
            handle.close()
        handles.clear()
        summary = {"schema_version": 1, "prepared_manifest": str(Path(prepared_manifest).absolute()),
                   "prepared_manifest_sha256": first_hash.hexdigest(),
                   "group_assignment_sha256": assignment_hash.hexdigest(),
                   "validation_fraction": fraction, "seed": seed,
                   "grouping": "source_family + session_id, else deployment_id, else original recording",
                   "stratification": "source_family", "group_counts": source_groups,
                   "total_shards": total_shards, "total_samples": total_samples,
                   "sample_rate_hz": sample_rate_hz, "frames_per_sample": frames_per_sample,
                   "audio_validated": False,
                   "splits": {split: {**dict(counts[split]), "path": str(paths[split].absolute()),
                                       "sha256": hashes[split].hexdigest()} for split in paths},
                   "by_source": {key: {split: dict(value) for split, value in split_counts.items()}
                                 for key, split_counts in sorted(by_source.items())},
                   "empty_source_splits": [{"source_family": family, "split": split}
                                           for family, values in source_groups.items()
                                           for split, count in values.items() if count == 0]}
        for split in paths:
            os.replace(temporary[split], paths[split])
        summary_part = output / f".split.{token}.part"
        summary_part.write_bytes(_json_bytes(summary))
        os.replace(summary_part, output / "split.json")
        return summary
    finally:
        for handle in handles.values():
            handle.close()
        for path in temporary.values():
            if path.exists():
                path.unlink()


def _shuffle(rows, rng, size):
    if size < 1:
        yield from rows
        return
    buffer = []
    for row in rows:
        if len(buffer) < size:
            buffer.append(row)
        else:
            position = rng.randrange(len(buffer))
            yield buffer[position]
            buffer[position] = row
    rng.shuffle(buffer)
    yield from buffer


def iter_tar_samples(index: str | Path | Iterable[Mapping], *, epoch: int = 0, seed: int = 0,
                     rank: int = 0, world_size: int = 1, worker_id: int = 0,
                     num_workers: int = 1, shard_shuffle_buffer: int = 256,
                     sample_rate_hz: int = 16000, frames_per_sample: int = 80000):
    """Yield validated native PCM16 samples, with bounded shard/audio buffers.

    ``waveform`` is a signed-16-bit array, not float-normalized. Stable sample IDs
    use original source identity plus member name. Epoch shuffling happens
    before rank/worker modulo partitioning, so no shard is read by two workers
    when all ranks use the same seed, epoch and worker count. Input JSONL is
    streamed by every worker; WAV audio is read only by its assigned worker.
    """
    if world_size < 1 or num_workers < 1 or not 0 <= rank < world_size or not 0 <= worker_id < num_workers:
        raise ValueError("invalid rank/worker partition")
    if shard_shuffle_buffer < 0:
        raise ValueError("shard_shuffle_buffer must be nonnegative")
    rows = (row for _, row in _rows(index)) if isinstance(index, (str, Path)) else iter(index)
    epoch_seed = int.from_bytes(hashlib.sha256(_json_bytes([seed, epoch])).digest()[:8], "big")
    rows = _shuffle(rows, random.Random(epoch_seed), shard_shuffle_buffer)
    partitions, partition = world_size * num_workers, rank * num_workers + worker_id
    for position, row in enumerate(rows):
        if position % partitions != partition:
            continue
        family, recording, default_group = _identity(row)
        rate, frames = sample_rate_hz, frames_per_sample
        if (int(row.get("sample_rate_hz", rate)), int(row.get("frames_per_sample", frames))) != (rate, frames):
            raise ValueError("index audio contract differs from the requested sample rate/frame count")
        group = row.get("group_id", default_group)
        count = 0
        with tarfile.open(row["tar_path"], "r|") as archive:
            for member in archive:
                if not member.name.lower().endswith(".wav"):
                    continue
                if not member.isfile() or member.size > frames * 2 + 65536:
                    raise ValueError(f"unexpected WAV member: {row['tar_path']}:{member.name}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"unreadable WAV member: {member.name}")
                with extracted:
                    payload = extracted.read()
                with wave.open(io.BytesIO(payload), "rb") as audio:
                    if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes(), audio.getcomptype()) != (rate, 1, 2, frames, "NONE"):
                        raise ValueError(f"WAV contract violation: {row['tar_path']}:{member.name}")
                    pcm = audio.readframes(frames + 1)
                if len(pcm) != frames * 2:
                    raise ValueError(f"truncated WAV member: {member.name}")
                waveform = array("h")
                waveform.frombytes(pcm)
                if sys.byteorder != "little":
                    waveform.byteswap()
                start = re.fullmatch(r"audio/(\d+)\.wav", member.name)
                identity = _json_bytes([family, recording, member.name])
                yield {"sample_id": hashlib.sha256(identity).hexdigest(), "source_key": row.get("source_key", family),
                       "source_family": family, "source_id": row.get("source_id", recording),
                       "recording_id": recording, "group_id": group, "split": row.get("split"),
                       "tar_path": row["tar_path"], "member": member.name,
                       "start_frame": int(start.group(1)) if start else None,
                       "sample_rate_hz": rate, "frames": frames, "waveform": waveform,
                       "source": row.get("source", {})}
                count += 1
                # Streaming iteration need not retain TarInfo objects.
                archive.members.clear()
        if count != row["segment_count"]:
            raise ValueError(f"TAR segment count mismatch: {row['tar_path']}: {count} != {row['segment_count']}")


try:
    from torch.utils.data import IterableDataset as _IterableDataset
except ImportError:
    _IterableDataset = object


class TarAudioDataset(_IterableDataset):
    """Optional spawn-compatible torch wrapper; use persistent_workers=False.

    Call ``set_epoch`` before creating each epoch's iterator/workers. Keeping
    workers across epochs would retain their earlier epoch value. Default
    collation of the variable provenance dictionaries is intentionally avoided:
    use batch_size=None or a training-specific collate function.
    """
    def __init__(self, index, *, epoch=0, seed=0, rank=0, world_size=1, shard_shuffle_buffer=256):
        if _IterableDataset is object:
            raise RuntimeError("TarAudioDataset requires PyTorch; the TAR reader does not")
        super().__init__()
        self.index = str(index)
        self.epoch, self.seed, self.rank, self.world_size = epoch, seed, rank, world_size
        self.shard_shuffle_buffer = shard_shuffle_buffer

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        import torch
        from torch.utils.data import get_worker_info
        worker = get_worker_info()
        for sample in iter_tar_samples(self.index, epoch=self.epoch, seed=self.seed,
                                       rank=self.rank, world_size=self.world_size,
                                       worker_id=worker.id if worker else 0,
                                       num_workers=worker.num_workers if worker else 1,
                                       shard_shuffle_buffer=self.shard_shuffle_buffer):
            sample["waveform"] = torch.tensor(sample["waveform"], dtype=torch.float32) / 32768.0
            yield sample
