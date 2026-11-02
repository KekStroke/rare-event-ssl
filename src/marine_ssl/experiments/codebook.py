"""Train-only AVES block-six targets: GPU sample, CPU fit, GPU SQLite labels.

Run each stage in an MLS job. Only the bounded codebook sample retains float
features; full-corpus targets are uint8 SQLite BLOBs, one row per recording.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import time
import wave

from .ssl_data import _identity, _json_bytes, _rows, iter_tar_samples


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def fingerprint(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def read_config(path):
    import yaml
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    return config


def index_info(config, split):
    path = Path(config[f"{split}_index"])
    summary = json.loads(Path(config["split_summary"]).read_text(encoding="utf-8"))
    checksum = digest(path)
    if checksum != summary["splits"][split]["sha256"]:
        raise ValueError(f"{split} index fingerprint differs from split.json")
    count = recordings = 0
    for _, row in _rows(path):
        if row.get("split") != split:
            raise ValueError(f"non-{split} row in {path}")
        count += row["segment_count"]
        recordings += 1
    if (count, recordings) != (summary["splits"][split]["samples"], summary["splits"][split]["shards"]):
        raise ValueError("index counts differ from split.json")
    return {"path": str(path), "sha256": checksum, "samples": count, "recordings": recordings,
            "prepared_manifest_sha256": summary["prepared_manifest_sha256"]}


def select_ordinals(total, *, seed, nshards=16, clips_per_shard=320):
    """Historical contiguous-shard sampling, restricted to the train index."""
    import numpy as np
    if total < 1 or nshards < 1 or clips_per_shard < 1:
        raise ValueError("positive sample/shard counts required")
    selected = []
    for rank in range(nshards):
        start, end = round(total * rank / nshards), round(total * (rank + 1) / nshards)
        choice = np.random.default_rng(seed + rank).choice(end - start, min(clips_per_shard, end - start), replace=False)
        selected.extend(start + int(i) for i in choice)
    return sorted(selected)


def _selected_audio(index, ordinals):
    """Seek to selected members; do not decode all clips in selected long TARs."""
    import numpy as np
    wanted, offset = iter(ordinals), 0
    next_index = next(wanted, None)
    for _, row in _rows(index):
        count = row["segment_count"]
        local = []
        while next_index is not None and next_index < offset + count:
            local.append(next_index - offset)
            next_index = next(wanted, None)
        offset += count
        if not local:
            continue
        family, recording, group = _identity(row)
        with tarfile.open(row["tar_path"], "r:") as archive:
            members = [member for member in archive.getmembers() if member.name.lower().endswith(".wav")]
            if len(members) != count:
                raise ValueError("TAR member count differs from manifest")
            for i in local:
                member = members[i]
                if not member.isfile() or member.size > 225536:
                    raise ValueError("unexpected sampled WAV member")
                with archive.extractfile(member) as stream:
                    payload = stream.read()
                with wave.open(io.BytesIO(payload), "rb") as audio:
                    if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes(), audio.getcomptype()) != (16000, 1, 2, 80000, "NONE"):
                        raise ValueError("sampled WAV violates native audio contract")
                    pcm = audio.readframes(80001)
                if len(pcm) != 160000:
                    raise ValueError("truncated sampled WAV")
                yield {"sample_id": fingerprint([family, recording, member.name]), "tar_path": row["tar_path"],
                       "member": member.name, "group_id": group, "recording_id": recording,
                       "source_family": family, "waveform": np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0}
        if next_index is None:
            break


def _teacher(config):
    from .encoders import load_encoder
    import torch
    if config["teacher"]["kind"] != "aves_torchaudio":
        raise ValueError("historical target teacher must be published AVES TorchAudio")
    teacher = load_encoder(config["teacher"], config.get("device", "cuda"))
    if teacher.normalize_waveform:
        raise ValueError("historical AVES target waveforms must not be standardized")
    teacher.model.eval().requires_grad_(False)
    return teacher


def _teacher_batch(teacher, samples):
    import numpy as np
    import torch
    waveforms = torch.from_numpy(np.stack([sample["waveform"] for sample in samples])).to(teacher.device)
    with torch.inference_mode():
        layers, _ = teacher.model.extract_features(waveforms, num_layers=6)
        features = layers[-1].float()
    if tuple(features.shape) != (len(samples), 249, 768) or not bool(torch.isfinite(features).all()):
        raise ValueError("AVES block-six target features have invalid shape/values")
    return features


def sample_features(config):
    import numpy as np
    info = index_info(config, "train")
    params = {"seed": int(config.get("seed", 20261103)), "nshards": 16, "clips_per_shard": 320}
    signature = fingerprint({"train": info, "teacher": config["teacher"], "sampling": params, "block": 6})
    output = Path(config["sample_dir"]); output.mkdir(parents=True, exist_ok=True)
    ready = output / "sample.ready.json"
    if ready.exists():
        result = json.loads(ready.read_text())
        if (result["signature"] != signature or digest(output / "features.npy") != result["features_sha256"]
                or digest(output / "sampled_clips.jsonl") != result["sampled_clips_sha256"]):
            raise ValueError("existing codebook feature sample is incompatible")
        return result
    selected = select_ordinals(info["samples"], **params)
    print(json.dumps({"stage": "sample", "clips": len(selected), "float_feature_bytes": len(selected) * 249 * 768 * 4}), flush=True)
    teacher = _teacher(config)
    features_path = output / "features.npy.part"
    features = np.lib.format.open_memmap(features_path, mode="w+", dtype="float32", shape=(len(selected) * 249, 768))
    records, batch, completed, last_log = [], [], 0, time.monotonic()
    def consume():
        nonlocal completed, last_log
        values = _teacher_batch(teacher, batch).cpu().numpy()
        features[completed * 249:(completed + len(batch)) * 249] = values.reshape(-1, 768)
        records.extend({key: value for key, value in sample.items() if key != "waveform"} for sample in batch)
        completed += len(batch)
        batch.clear()
        if time.monotonic() - last_log >= 60:
            print(json.dumps({"stage": "sample", "completed_clips": completed}), flush=True)
            last_log = time.monotonic()
    for sample in _selected_audio(info["path"], selected):
        batch.append(sample)
        if len(batch) >= int(config.get("teacher_batch_size", 64)):
            consume()
    if batch:
        consume()
    if completed != len(selected):
        raise ValueError("sample selection count mismatch")
    features.flush(); del features
    features_path.replace(output / "features.npy")
    (output / "sampled_clips.jsonl").write_bytes(b"".join(_json_bytes(row) for row in records))
    result = {"signature": signature, "train": info, "teacher": config["teacher"], "teacher_block": 6,
              "sampling": params, "clips": completed, "frames": completed * 249, "dimension": 768,
              "features_sha256": digest(output / "features.npy"),
              "sampled_clips_sha256": digest(output / "sampled_clips.jsonl")}
    write_json(ready, result)
    return result


def fit_codebook(config):
    """CPU-only MiniBatchKMeans with explicit Fairseq v0.12.2 defaults."""
    import numpy as np
    import sklearn
    from sklearn.cluster import MiniBatchKMeans
    sample_dir, output = Path(config["sample_dir"]), Path(config["codebook_dir"])
    sample = json.loads((sample_dir / "sample.ready.json").read_text())
    if digest(sample_dir / "features.npy") != sample["features_sha256"]:
        raise ValueError("sample feature checksum mismatch")
    if index_info(config, "train")["sha256"] != sample["train"]["sha256"]:
        raise ValueError("sample was not derived from this train split")
    params = {"n_clusters": 200, "init": "k-means++", "max_iter": 100, "batch_size": 10000,
              "tol": 0.0, "max_no_improvement": 100, "n_init": 20, "reassignment_ratio": 0.0,
              "random_state": int(config.get("seed", 20261103)), "compute_labels": False}
    signature = fingerprint({"sample": sample["signature"], "features": sample["features_sha256"], "params": params})
    output.mkdir(parents=True, exist_ok=True)
    ready = output / "codebook.ready.json"
    if ready.exists():
        result = json.loads(ready.read_text())
        if result["signature"] != signature or digest(output / "centroids.npy") != result["centroids_sha256"]:
            raise ValueError("existing codebook is incompatible")
        return result
    features = np.load(sample_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    if features.shape != (sample["frames"], 768) or features.dtype != np.float32:
        raise ValueError("invalid sampled feature matrix")
    print(json.dumps({"stage": "fit", "frames": len(features), "params": params}), flush=True)
    model = MiniBatchKMeans(**params).fit(features)
    with (output / "centroids.npy.part").open("wb") as handle:
        np.save(handle, model.cluster_centers_.astype("float32"), allow_pickle=False)
    (output / "centroids.npy.part").replace(output / "centroids.npy")
    (output / "dict.km.txt").write_text("".join(f"{i} 1\n" for i in range(200)))
    result = {"signature": signature, "sample": sample, "params": params, "sklearn_version": sklearn.__version__,
              "centroids_sha256": digest(output / "centroids.npy"), "inertia": float(model.inertia_),
              "upstream_defaults": "https://github.com/facebookresearch/fairseq/blob/v0.12.2/examples/hubert/simple_kmeans/learn_kmeans.py"}
    write_json(ready, result)
    return result


def label_corpus(config):
    import numpy as np
    import torch
    infos = {split: index_info(config, split) for split in ("train", "validation")}
    codebook_dir, output = Path(config["codebook_dir"]), Path(config["labels_dir"])
    codebook = json.loads((codebook_dir / "codebook.ready.json").read_text())
    if codebook["sample"]["train"]["sha256"] != infos["train"]["sha256"]:
        raise ValueError("centroids were not fitted on this training split")
    if codebook["sample"]["teacher"] != config["teacher"]:
        raise ValueError("label teacher differs from the codebook feature teacher")
    if digest(codebook_dir / "centroids.npy") != codebook["centroids_sha256"]:
        raise ValueError("centroid checksum mismatch")
    signature = fingerprint({"indexes": infos, "teacher": config["teacher"], "codebook": codebook["signature"], "block": 6})
    output.mkdir(parents=True, exist_ok=True)
    database, ready = output / "labels.sqlite", output / "labels.ready.json"
    if ready.exists():
        result = json.loads(ready.read_text())
        if result["signature"] != signature or digest(database) != result["database_sha256"]:
            raise ValueError("existing labels are incompatible")
        return result
    connection = sqlite3.connect(database, timeout=60)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS recordings (tar_path TEXT PRIMARY KEY, source_id TEXT, members TEXT NOT NULL, labels BLOB NOT NULL, checksum TEXT NOT NULL)")
    prior = connection.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
    if prior and prior[0] != signature:
        connection.close(); raise ValueError("partial label database has a different provenance signature")
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('signature', ?)", (signature,))
    connection.commit()
    complete = set()
    for path, members_json, raw, checksum in connection.execute("SELECT tar_path,members,labels,checksum FROM recordings"):
        if len(raw) != len(json.loads(members_json)) * 249 or hashlib.sha256(raw).hexdigest() != checksum:
            connection.close(); raise ValueError("corrupt committed recording targets: " + path)
        complete.add(path)
    teacher = _teacher(config)
    centers = torch.from_numpy(np.load(codebook_dir / "centroids.npy", allow_pickle=False)).to(teacher.device)
    if tuple(centers.shape) != (200, 768) or not bool(torch.isfinite(centers).all()):
        raise ValueError("invalid K200 centers")
    pending, validated, batch = {}, set(), []
    writes, last_commit, last_log = 0, time.monotonic(), time.monotonic()
    done_samples = 0
    def maybe_commit():
        nonlocal writes, last_commit
        now = time.monotonic()
        if writes and (writes >= 64 or now - last_commit >= 30):
            connection.commit(); writes = 0; last_commit = now
    def finalize(path):
        nonlocal writes
        record = pending.get(path)
        # A yielded last-expected sample is insufficient: the reader must also
        # finish its end-of-TAR count/integrity checks before durable admission.
        if record is None or not record["all_labels_ready"] or path not in validated:
            return
        if len(record["members"]) != record["count"]:
            raise ValueError("recording target count differs from validated TAR")
        raw = np.stack(record["labels"]).tobytes()
        connection.execute("INSERT INTO recordings VALUES (?,?,?,?,?)", (path, record["source_id"],
                           json.dumps(record["members"]), raw, hashlib.sha256(raw).hexdigest()))
        del pending[path]; validated.remove(path)
        complete.add(path); writes += 1
    def consume():
        nonlocal writes, last_commit, last_log, done_samples
        values = _teacher_batch(teacher, batch)
        distance = values.square().sum(-1, keepdim=True) + centers.square().sum(-1) - 2 * values @ centers.T
        labels = distance.argmin(-1).to(torch.uint8).cpu().numpy()
        for sample, target in zip(batch, labels):
            record = pending.setdefault(sample["tar_path"], {"members": [], "labels": [], "source_id": sample["source_id"],
                                        "count": sample["recording_segment_count"], "all_labels_ready": False})
            record["members"].append(sample["member"]); record["labels"].append(target)
            if sample["last_member"]:
                record["all_labels_ready"] = True
                finalize(sample["tar_path"])
            done_samples += 1
        batch.clear()
        now = time.monotonic()
        maybe_commit()
        if now - last_log >= 60:
            print(json.dumps({"stage": "label", "recordings_complete": len(complete), "new_samples": done_samples}), flush=True)
            last_log = now
    try:
        for split, info in infos.items():
            for _, row in _rows(info["path"]):
                if row["tar_path"] in complete:
                    continue
                for i, sample in enumerate(iter_tar_samples([row], shard_shuffle_buffer=0)):
                    sample["waveform"] = np.asarray(sample["waveform"], dtype="float32") / 32768.0
                    sample["last_member"] = i + 1 == row["segment_count"]
                    sample["recording_segment_count"] = row["segment_count"]
                    batch.append(sample)
                    if len(batch) >= int(config.get("teacher_batch_size", 64)):
                        consume()
                validated.add(row["tar_path"])
                finalize(row["tar_path"])
                maybe_commit()
        if batch:
            consume()
        if pending:
            raise ValueError("incomplete recording label buffers")
        expected = sum(info["recordings"] for info in infos.values())
        if len(complete) != expected:
            raise ValueError("label database recording count mismatch")
        connection.commit()
        result = {"signature": signature, "indexes": infos, "teacher": config["teacher"],
                  "codebook_signature": codebook["signature"], "teacher_block": 6, "label_frames": 249,
                  "num_classes": 200, "recordings": expected, "samples": sum(info["samples"] for info in infos.values())}
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('complete', ?)", (json.dumps(result),))
        connection.commit()
    finally:
        connection.close()
    result["database_sha256"] = digest(database)
    write_json(ready, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sample", "fit", "label"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result = {"sample": sample_features, "fit": fit_codebook, "label": label_corpus}[args.stage](read_config(args.config))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
