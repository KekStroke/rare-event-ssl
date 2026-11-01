"""Actual WAV/TAR coordinator tests; no external audio, network or FFmpeg."""
import hashlib
import importlib
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

m = importlib.import_module("marine_ssl.prepare")


class ParallelPrepareTests(unittest.TestCase):
    def config(self, seconds):
        return {"sample_rate": 16000, "channels": 1, "sample_format": "PCM_16",
                "segment_seconds": 5, "overlap_seconds": 0, "reserve_bytes": 0,
                "sources": {key: {"source_family": key, "target_hours": value / 3600}
                            for key, value in seconds.items()}}

    def row(self, source, name, labels):
        return {"source_key": source, "source_family": source, "source_id": name,
                "clearance_scope": "source_metadata_only", "bytes": 1, "test_windows": labels}

    def convert(self, row, raw, scratch, *args):
        blocks = [bytes([ord(label), 0]) * 80000 for label in row["test_windows"]]
        segments = [{"member": f"audio/{i * 80000:014d}.wav", "start_frame": i * 80000,
                     "frames": 80000, "pcm_sha256": hashlib.sha256(block).hexdigest()}
                    for i, block in enumerate(blocks)]
        result = {"source": row, "source_sha256": hashlib.sha256(row["source_id"].encode()).hexdigest(),
                  "normalized_pcm_sha256": hashlib.sha256(b"".join(blocks)).hexdigest(),
                  "selected_frames": 80000 * len(blocks), "duration_seconds": 5 * len(blocks),
                  "sample_rate_hz": 16000, "segment_count": len(segments), "segments": segments,
                  "decoder_backend": "mock-pcm16", "decoder_versions": {"mock": "1"}}
        path = scratch / "mock.tar.part"
        with tarfile.open(path, "w") as archive:
            for segment, block in zip(segments, blocks):
                m._add(archive, segment["member"], m._wav(block, 16000))
            m._add(archive, "recording.json", json.dumps(result).encode())
        return result, path

    def run_prepare(self, config, rows, output, convert=None, workers=2, download_workers=4):
        with patch.object(m, "_source", return_value=(Path("unused-raw"), None)), \
                patch.object(m, "_convert", side_effect=convert or self.convert), \
                patch.object(m.shutil, "which", return_value="mock-tool"):
            return m.prepare(config, rows, output, workers=workers, download_workers=download_workers)

    def test_parallel_decode_ordered_dedup_trim_and_legacy_receipt_resume(self):
        rows = [self.row("a", "a0", "AB"), self.row("a", "a1", "BC")]
        config = self.config({"a": 15})
        lock, active, maximum, finished = threading.Lock(), 0, 0, []
        def convert(row, *args):
            nonlocal active, maximum
            with lock:
                active += 1; maximum = max(maximum, active)
            try:
                if row["source_id"] == "a0": time.sleep(0.08)
                result = self.convert(row, *args)
                finished.append(row["source_id"])
                return result
            finally:
                with lock: active -= 1
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = self.run_prepare(config, rows, output, convert)
            self.assertEqual(result["sources"]["a"]["actual_frames"], 240000)
            self.assertEqual(finished, ["a1", "a0"])
            self.assertEqual(maximum, 2)
            receipts = {json.loads(p.read_text())["source"]["source_id"]: (p, json.loads(p.read_text()))
                        for p in (output / "receipts" / "a").glob("*.json")}
            self.assertEqual(receipts["a0"][1]["selection_order"], 0)
            self.assertEqual(receipts["a1"][1]["selection_order"], 1)
            trimmed = receipts["a1"][1]
            self.assertEqual(trimmed["segment_count"], 1)
            self.assertEqual(trimmed["coordinator_exclusions"]["global_duplicate_windows"], 1)
            with tarfile.open(trimmed["tar_path"]) as archive:
                self.assertEqual(sum(member.name.endswith(".wav") for member in archive), 1)
            legacy_path, legacy = receipts["a0"]
            legacy.pop("decoder_backend"); legacy.pop("decoder_versions")
            legacy_path.write_text(json.dumps(legacy))
            before = {path: path.read_bytes() for path, _ in receipts.values()}
            with patch.object(m, "_source", side_effect=AssertionError("must not redownload")), \
                    patch.object(m, "_convert", side_effect=AssertionError("must not decode")), \
                    patch.object(m.shutil, "which", return_value="mock-tool"):
                resumed = m.prepare(config, rows, output, workers=1, download_workers=3)
            self.assertEqual(resumed["sources"]["a"]["actual_frames"], 240000)
            self.assertEqual(resumed["counts"]["reused_recordings"], 2)
            self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))
            manifest = [json.loads(line) for line in (output / "prepared_manifest.jsonl").read_text().splitlines()]
            self.assertEqual(next(row for row in manifest if row["source_id"] == "a0")["decoder_backend"], "ffmpeg")

    def test_cross_source_dedup_and_last_quota_are_applied_before_commit(self):
        rows = [self.row("a", "a0", "A"), self.row("b", "b0", "ADE")]
        config = self.config({"a": 5, "b": 5})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            def convert(row, *args):
                if row["source_key"] == "b":
                    deadline = time.monotonic() + 3
                    while not list((output / "receipts" / "a").glob("*.json")):
                        if time.monotonic() > deadline: raise AssertionError("source a was blocked")
                        time.sleep(0.005)
                return self.convert(row, *args)
            result = self.run_prepare(config, rows, output, convert)
            self.assertTrue(all(source["actual_frames"] == 80000 for source in result["sources"].values()))
            receipt = json.loads(next((output / "receipts" / "b").glob("*.json")).read_text())
            self.assertEqual(receipt["segment_count"], 1)
            self.assertEqual(receipt["coordinator_exclusions"], {"global_duplicate_windows": 1, "quota_trimmed_windows": 1})
            self.assertEqual(receipt["segments"][0]["start_frame"], 80000)
            with tarfile.open(receipt["tar_path"]) as archive:
                self.assertEqual(sum(member.name.endswith(".wav") for member in archive), 1)

    def test_transfer_error_is_persisted_before_other_download_finishes(self):
        rows = [self.row("a", "failed", "A"), self.row("b", "waiting", "B")]
        config = self.config({"a": 5, "b": 5})
        observed = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            def source(row, *args):
                if row["source_id"] == "failed":
                    raise RuntimeError("simulated transfer failure")
                deadline = time.monotonic() + 3
                while not (output / "preparation_issues.jsonl").exists():
                    if time.monotonic() > deadline: raise AssertionError("error was hidden behind pool wait")
                    time.sleep(0.005)
                observed.append(json.loads((output / "preparation_issues.jsonl").read_text().splitlines()[0])["error"])
                return Path("unused-raw"), None
            with patch.object(m, "_source", side_effect=source), \
                    patch.object(m.shutil, "which", return_value="mock-tool"):
                result = m.prepare(config, rows, output, workers=2, download_workers=2)
            self.assertFalse(any(source["quota_met"] for source in result["sources"].values()))
            self.assertFalse((output / ".prepare.lock").exists())
        self.assertEqual(observed, ["RuntimeError: simulated transfer failure"])

    def test_legacy_header_repair_is_opted_in_only_for_orcasound_local_clock(self):
        import marine_ssl.fast_audio as fast_audio
        cases = [("orcasound", "unspecified_local_clock", True),
                 ("orcasound", "utc", False), ("pacific_sound", "unspecified_local_clock", False)]
        config = self.config({"a": 5})
        observed = []
        repair = {"missing_declared_frames": 2, "pcm_frames_retained": 220500, "source_fully_present": False}
        quality = {"retained_physical_pcm_prefix": True}
        def decode(raw, **kwargs):
            enabled = kwargs["allow_header_repair"]
            observed.append(enabled)
            metadata = {"native": {"sample_rate": "44100", "channels": 1, "frames": 220500},
                        "backend": "mock-repair", "versions": {"mock": "1"}, "decoded_frames": 80000,
                        "full_decode_verified": True}
            if enabled:
                metadata.update(container_repair=repair, quality=quality)
            return metadata, bytes([1, 0]) * 80000
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); raw = root / "original.wav"; raw.write_bytes(b"immutable original container")
            original = raw.read_bytes()
            for number, (family, clock, enabled) in enumerate(cases):
                work = root / str(number); work.mkdir()
                row = {"source_family": family, "time_basis": clock, "source_id": str(number)}
                with patch.object(fast_audio, "decode_pcm16", side_effect=decode):
                    result, shard = m._convert(row, raw, work, 80000, config, set(), set())
                self.assertEqual(result["container_repair"], repair if enabled else None)
                self.assertEqual(result["quality"], quality if enabled else {})
                with tarfile.open(shard) as archive:
                    recorded = json.loads(archive.extractfile("recording.json").read())
                    self.assertEqual(recorded["container_repair"], result["container_repair"])
                    self.assertEqual(recorded["quality"], result["quality"])
                self.assertEqual(raw.read_bytes(), original)
        self.assertEqual(observed, [True, False, False])

    def test_drained_batch_refills_until_all_candidates_are_exhausted(self):
        rows = [self.row("a", f"a{number:03d}", chr(33 + number)) for number in range(100)]
        rows[50]["test_windows"] = rows[49]["test_windows"]
        config = self.config({"a": 600})
        config["max_inflight_records"] = 1  # Deterministically drains before refill.
        converted = []
        def convert(row, *args):
            converted.append(row["source_id"])
            if row["source_id"] == "a005":
                return {"status": "duplicate_or_no_complete_unique_windows", "source_id": row["source_id"]}, None
            return self.convert(row, *args)
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_prepare(config, rows, Path(temporary), convert)
        self.assertEqual(converted, [row["source_id"] for row in rows])
        self.assertEqual(result["counts"]["downloaded_recordings"], 100)
        self.assertEqual(result["counts"]["prepared_recordings"], 98)
        self.assertEqual(result["sources"]["a"]["actual_frames"], 98 * 80000)
        self.assertFalse(result["sources"]["a"]["quota_met"])
        self.assertEqual(result["status"], "incomplete")

    def test_range_prefix_is_pacific_only_falls_back_and_preserves_hash_scope(self):
        import marine_ssl.remote_wav as remote_wav
        import marine_ssl.fast_audio as fast_audio
        config = self.config({"a": 5})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def row(family):
                return {"source_key": family, "source_family": family, "source_id": "original-id",
                        "url": "https://public.example/recording.wav?signature=not-for-owner",
                        "path_in_source": "2020/09/recording.wav", "revision": "pinned",
                        "source_etag": "etag", "bytes": 100000, "sha256": "a" * 64,
                        "use_range_prefix": True, "per_record_max_seconds": 21600}
            acquisition = {"source_etag": "etag", "byte_range": [44, 2043], "source_sha256_verified": False}
            def ranged(origin, path, seconds):
                self.assertEqual(seconds, 21600)
                path.write_bytes(b"materialized source prefix")
                return dict(acquisition)
            def full(url, path, **kwargs):
                path.write_bytes(b"full source")
                return path
            original = row("pacific_sound")
            with patch.object(remote_wav, "acquire_wav_prefix", side_effect=ranged) as selected, \
                    patch.object(m, "download", side_effect=full) as download:
                raw, scratch = m._source(original, root / "range", 0)
                self.assertEqual(selected.call_count, 1); download.assert_not_called()
            owner = json.loads((scratch / "owner.json").read_text())
            self.assertEqual(owner["path_in_source"], "2020/09/recording.wav")
            self.assertEqual(owner["expected_bytes"], 100000)
            self.assertNotIn("not-for-owner", json.dumps(owner))
            metadata = {"native": {"sample_rate": "16000", "channels": 1}, "backend": "mock",
                        "versions": {}, "decoded_frames": 80000, "full_decode_verified": True}
            with patch.object(fast_audio, "decode_pcm16", return_value=(metadata, bytes([1, 0]) * 80000)):
                result, _ = m._convert(original, raw, scratch, 80000, config, set(), set())
            self.assertIsNone(result["source_sha256"])
            self.assertFalse(result["source_sha256_verified"])
            self.assertEqual(result["materialized_source_sha256"], m.digest(raw))
            self.assertEqual(result["source"]["sha256"], "a" * 64)
            for number, family in enumerate(("orcasound", "pacific_sound")):
                origin = row(family)
                with patch.object(remote_wav, "acquire_wav_prefix", return_value=None) as selected, \
                        patch.object(m, "download", side_effect=full) as download:
                    materialized, _ = m._source(origin, root / str(number), 0)
                    self.assertEqual(download.call_count, 1)
                    self.assertEqual(selected.call_count, int(family == "pacific_sound"))
                    self.assertNotIn("source_acquisition", origin)
                    self.assertEqual(materialized.read_bytes(), b"full source")
