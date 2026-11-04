"""Bounded, resumable feature cache; one SQLite file, no per-clip files.

Only the owning MLS process writes this cache. Every completed item is committed
atomically; incomplete items are recomputed. Audio is decoded in five-second
pieces and receives identical physical preprocessing for every encoder.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np

from .datasets import stable_hash


PREPROCESSING = {"version": 2, "sample_rate": 16000, "channel": 0,
                 "context_seconds": 5.0, "padding": "only_conv_minimum_400_samples",
                 "resampler": "scipy.signal.resample_poly", "pooling": "frame_mean_and_std",
                 "frame_storage": "float16", "waveform_normalization": "registered_encoder_spec"}
FRAME_BLOCK_FRAMES = 8192


def feature_identity(row, layers, temporal):
    return stable_hash({"path": row["path"], "start": row["start_seconds"],
                        "end": row["end_seconds"], "pcm_sha256": row.get("pcm_sha256"),
                        "sha256": row.get("sha256"), "layers": list(layers), "temporal": temporal})


def pool_frames(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite nonempty T x D features")
    return np.concatenate((values.mean(axis=0), values.std(axis=0))).astype(np.float32)


def _pack(data):
    stream = io.BytesIO()
    np.savez(stream, **data)
    return stream.getvalue()


def _unpack(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class FeatureStore:
    def __init__(self, path, model_spec, *, layers=(6, 9, 12), encoder=None,
                 max_bytes=128 * 1024**3, readonly=False):
        self.path = Path(path)
        self.layers, self.encoder, self.max_bytes = list(layers), encoder, max_bytes
        self.readonly = readonly
        identity = {"model": asdict(model_spec) if hasattr(model_spec, "__dataclass_fields__") else model_spec,
                    "preprocessing": PREPROCESSING, "layers": self.layers}
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path.resolve().as_uri() + ("?mode=ro" if readonly else "?mode=rwc"), uri=True, timeout=60)
        if not readonly:
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS features (id TEXT PRIMARY KEY, data BLOB NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS frames (id TEXT, layer INTEGER, block INTEGER, data BLOB NOT NULL, PRIMARY KEY(id,layer,block))")
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('identity',?)", (json.dumps(identity, sort_keys=True),))
            self.db.commit()
        recorded = self.db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if not recorded or json.loads(recorded[0]) != identity:
            self.db.close()
            raise ValueError("Feature cache identity differs from registered model/preprocessing")
        self.identity = identity
        self.computed, self.hits = 0, 0

    def close(self):
        self.db.close()

    def get(self, row, *, temporal=False, frame_layers=None):
        key = feature_identity(row, self.layers, temporal)
        found = self.db.execute("SELECT data FROM features WHERE id=?", (key,)).fetchone()
        if found:
            self.hits += 1
            data = _unpack(found[0])
            if temporal:
                for layer in (self.layers if frame_layers is None else frame_layers):
                    blocks = [part[0] for part in self.db.execute("SELECT data FROM frames WHERE id=? AND layer=? ORDER BY block", (key, layer))]
                    if not blocks:
                        raise ValueError("Committed temporal item lacks bounded frame blocks")
                    data[f"frames_{layer}"] = np.concatenate([_unpack(part)["values"] for part in blocks])
            return data
        if self.readonly or self.encoder is None:
            raise KeyError("Missing committed feature item " + key)
        data = self._extract(row, temporal)
        frames = {layer: data[f"frames_{layer}"] for layer in self.layers} if temporal else {}
        payload = _pack({k: v for k, v in data.items() if not k.startswith("frames_")})
        retained_bytes = len(payload) + sum(array.nbytes for array in frames.values())
        # Include rollback-journal space in the operational disk budget.
        if self.path.stat().st_size + 2 * retained_bytes > self.max_bytes:
            raise RuntimeError("Feature-cache storage budget exceeded; retained items are preserved")
        with self.db:
            for layer, values in frames.items():
                for block, start in enumerate(range(0, len(values), FRAME_BLOCK_FRAMES)):
                    self.db.execute("INSERT INTO frames VALUES (?,?,?,?)", (key, layer, block,
                        _pack({"values": values[start:start + FRAME_BLOCK_FRAMES]})))
            self.db.execute("INSERT INTO features VALUES (?,?)", (key, payload))
        self.computed += 1
        return data

    def _extract(self, row, temporal):
        import soundfile as sf
        from scipy.signal import resample_poly
        import torch
        chunks = {layer: [] for layer in self.layers}
        pooled_chunks = {layer: [] for layer in self.layers}
        sums, squares, counts = {}, {}, {}
        grids, chunk_bounds, padding = [], [], []
        start, end = float(row["start_seconds"]), float(row["end_seconds"])
        with sf.SoundFile(row["path"]) as audio:
            rate = audio.samplerate
            lo, hi = round(start * rate), min(round(end * rate), len(audio))
            if hi <= lo or abs(hi / rate - end) > 1 / rate + 1e-8:
                raise ValueError("Verified row interval no longer matches decoded source")
            audio.seek(lo)
            for first in range(lo, hi, round(5 * rate)):
                wave = audio.read(min(round(5 * rate), hi - first), dtype="float32", always_2d=True)[:, 0]
                divisor = math.gcd(rate, 16000)
                if rate != 16000:
                    wave = resample_poly(wave, 16000 // divisor, rate // divisor).astype(np.float32)
                output = self.encoder.extract_layers(torch.from_numpy(wave.copy()), layers=self.layers)
                meta = output.metadata
                n = meta.feature_frames
                # Short clips retain one explicitly padded feature; never invent a time outside the row.
                times = first / rate + meta.center_offset_seconds + np.arange(n) * meta.frame_hop_seconds
                times = np.minimum(times, end)
                grids.append(times.astype(np.float64))
                chunk_bounds.append((first / rate, min(end, first / rate + len(wave) / 16000)))
                padding.append(meta.padding_samples)
                for layer, values in output.layers.items():
                    values = values.detach().cpu().numpy().astype(np.float32)
                    pooled_chunks[layer].append(pool_frames(values))
                    sums[layer] = sums.get(layer, 0) + values.sum(axis=0, dtype=np.float64)
                    squares[layer] = squares.get(layer, 0) + np.square(values.astype(np.float64)).sum(axis=0)
                    counts[layer] = counts.get(layer, 0) + len(values)
                    if temporal:
                        chunks[layer].append(values.astype(np.float16))
        data = {"times": np.concatenate(grids), "chunk_bounds": np.asarray(chunk_bounds),
                "padding_samples": np.asarray(padding), "duration_seconds": np.asarray(end - start)}
        for layer, values in chunks.items():
            mean = sums[layer] / counts[layer]
            std = np.sqrt(np.maximum(0, squares[layer] / counts[layer] - mean**2))
            data[f"pooled_{layer}"] = np.concatenate((mean, std)).astype(np.float32)
            data[f"chunks_{layer}"] = np.stack(pooled_chunks[layer])
            if temporal:
                data[f"frames_{layer}"] = np.concatenate(values)
        return data

    def temporal_loader(self, row, layers):
        data = self.get(row, temporal=True, frame_layers=layers)
        return {layer: {"features": data[f"frames_{layer}"], "times": data["times"],
                        "duration_seconds": float(data["duration_seconds"]),
                        "frame_hop_seconds": .02, "context_seconds": 5.0} for layer in layers}

    def prepare(self, cohorts):
        seen, total = set(), 0
        for cohort in cohorts.values():
            if not cohort["ready"]:
                continue
            temporal = cohort["task"] in {"avacha_field", "manual", "dominica"}
            for row in cohort["rows"]:
                key = feature_identity(row, self.layers, temporal)
                if key not in seen:
                    seen.add(key)
                    total += 1
        done, tick, started = 0, time.monotonic(), time.monotonic()
        seen.clear()
        for cohort in cohorts.values():
            if not cohort["ready"]:
                continue
            temporal = cohort["task"] in {"avacha_field", "manual", "dominica"}
            for row in cohort["rows"]:
                key = feature_identity(row, self.layers, temporal)
                if key in seen:
                    continue
                self.get(row, temporal=temporal)
                seen.add(key)
                done += 1
                if time.monotonic() - tick > 60 or done == total:
                    print(json.dumps({"stage": "features", "done": done, "total": total,
                                      "computed": self.computed, "cache_hits": self.hits,
                                      "elapsed_seconds": time.monotonic() - started,
                                      "cache_bytes": self.path.stat().st_size}), flush=True)
                    tick = time.monotonic()
