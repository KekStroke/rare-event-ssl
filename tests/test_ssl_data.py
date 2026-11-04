import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import wave

from marine_ssl.experiments.ssl_data import build_ssl_index, iter_tar_samples


def wav_bytes(rate=16000, frames=80000, channels=1):
    handle = io.BytesIO()
    with wave.open(handle, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\x01\x00" * frames * channels)
    return handle.getvalue()


def tar_row(root, number, *, session=None, rate=16000, count=1, declared_count=None):
    path = root / f"{number}.tar"
    with tarfile.open(path, "w") as archive:
        for member, payload in [(f"audio/{i * 80000:014d}.wav", wav_bytes(rate=rate)) for i in range(count)] + [("recording.json", b'{"ignored":true}')]:
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return {"tar_path": str(path), "source_key": "orc", "source_id": f"record-{number}",
            "segment_count": count if declared_count is None else declared_count,
            "duration_seconds": 5 * (count if declared_count is None else declared_count),
            "source": {"source_family": "orcasound", "session_id": session or f"session-{number}"}}


class SSLDataTests(unittest.TestCase):
    def test_native_reader_ignores_metadata_and_has_stable_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            row = tar_row(Path(folder), 0, count=2)
            samples = list(iter_tar_samples([row], shard_shuffle_buffer=1))
            self.assertEqual(len(samples), 2)
            self.assertEqual(len(samples[0]["waveform"]), 80000)
            self.assertEqual(samples[0]["waveform"][0], 1)
            self.assertEqual(samples[1]["start_frame"], 80000)
            self.assertNotEqual(samples[0]["sample_id"], samples[1]["sample_id"])
            self.assertEqual(samples[0]["sample_id"], next(iter_tar_samples([row], epoch=3))["sample_id"])
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["0.tar"])

    def test_contract_and_count_mismatches_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wrong_rate = tar_row(root, 1, rate=8000)
            with self.assertRaisesRegex(ValueError, "contract"):
                list(iter_tar_samples([wrong_rate]))
            with self.assertRaisesRegex(ValueError, "contract"):
                list(iter_tar_samples([{**wrong_rate, "sample_rate_hz": 8000}]))
            wrong_count = tar_row(root, 2, declared_count=2)
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                list(iter_tar_samples([wrong_count]))

    def test_shuffled_worker_partition_is_disjoint_and_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [tar_row(Path(folder), i) for i in range(12)]
            order = lambda **kwargs: [s["sample_id"] for s in iter_tar_samples(rows, seed=17, epoch=2, shard_shuffle_buffer=3, **kwargs)]
            whole = order()
            parts = [order(rank=r, world_size=2, worker_id=w, num_workers=2) for r in range(2) for w in range(2)]
            self.assertEqual(len(set(sum(parts, []))), 12)
            self.assertEqual(set(sum(parts, [])), set(whole))
            self.assertEqual(order(), whole)
            changed = [s["sample_id"] for s in iter_tar_samples(rows, seed=17, epoch=3, shard_shuffle_buffer=3)]
            self.assertNotEqual(changed, whole)

    def test_group_split_is_source_stratified_and_fingerprinted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # No real TAR read is needed to construct an index.
            rows = [{"tar_path": f"/never-open/{i}.tar", "source_key": family,
                     "source_id": f"rec-{i}", "segment_count": 2, "duration_seconds": 10,
                     "source": {"source_family": family, "session_id": f"session-{i // 2}"}}
                    for family in ("onc", "orcasound") for i in range(8)]
            # Paths must be globally unique even across source families.
            for row in rows:
                row["tar_path"] = f"/{row['source_key']}" + row["tar_path"]
            manifest = root / "prepared.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            summary = build_ssl_index(manifest, root / "index", validation_fraction=.5, seed=11)
            self.assertEqual(summary["total_samples"], 32)
            self.assertFalse(summary["audio_validated"])
            self.assertEqual(summary["prepared_manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
            memberships = {}
            for split in ("train", "validation"):
                path = root / "index" / f"{split}.jsonl"
                self.assertEqual(summary["splits"][split]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    self.assertEqual(memberships.setdefault(row["group_id"], split), split)
            self.assertEqual(summary["group_counts"]["onc"], {"train": 2, "validation": 2})
            # Membership is stable under metadata-row order changes.
            manifest.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)), encoding="utf-8")
            second = build_ssl_index(manifest, root / "other", validation_fraction=.5, seed=11)
            self.assertEqual(summary["group_assignment_sha256"], second["group_assignment_sha256"])

    def test_duplicate_shards_rejected_and_empty_split_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = {"tar_path": "/never-open/a.tar", "source_key": "onc", "source_id": "a", "segment_count": 1}
            manifest = root / "prepared.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            summary = build_ssl_index(manifest, root / "ok", validation_fraction=.01, seed=1)
            self.assertEqual(summary["empty_source_splits"], [{"source_family": "onc", "split": "validation"}])
            manifest.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate TAR"):
                build_ssl_index(manifest, root / "bad", validation_fraction=.1, seed=1)


if __name__ == "__main__":
    unittest.main()
