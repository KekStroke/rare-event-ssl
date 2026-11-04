import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch
import wave

from marine_ssl.experiments.codebook import digest, index_info


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def fixture(root):
    indexes = {}
    for split in ("train", "validation"):
        tar_path = root / f"{split}.tar"
        payload = io.BytesIO()
        with wave.open(payload, "wb") as audio:
            audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
            audio.writeframes(b"\x01\x00" * 80000)
        with tarfile.open(tar_path, "w") as archive:
            member = tarfile.TarInfo("audio/00000000000000.wav")
            member.size = len(payload.getvalue()); archive.addfile(member, io.BytesIO(payload.getvalue()))
        row = {"tar_path": str(tar_path), "source_id": split, "source_key": "onc", "split": split,
               "source": {"source_family": "onc", "session_id": split}, "segment_count": 1}
        path = root / f"{split}.jsonl"; path.write_text(json.dumps(row) + "\n")
        indexes[split] = {"path": str(path), "samples": 1, "shards": 1, "sha256": digest(path)}
    summary = root / "split.json"
    summary.write_text(json.dumps({"splits": indexes, "prepared_manifest_sha256": "pinned"}))
    return {"train_index": indexes["train"]["path"], "validation_index": indexes["validation"]["path"],
            "split_summary": str(summary), "teacher": {"kind": "aves_torchaudio", "checkpoint": "registered"},
            "codebook_dir": str(root / "codebook"), "labels_dir": str(root / "labels"), "teacher_batch_size": 1,
            "device": "cpu"}


class CodebookIndexTests(unittest.TestCase):
    def test_train_index_rejects_held_out_rows_and_fingerprint_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config = fixture(root)
            self.assertEqual(index_info(config, "train")["samples"], 1)
            path = Path(config["train_index"])
            path.write_text(path.read_text().replace('"train"', '"validation"'))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                index_info(config, "train")


@unittest.skipUnless(HAS_NUMPY, "NumPy required")
class CodebookSamplingTests(unittest.TestCase):
    def test_train_ordinals_stratified_bounded_and_seeded(self):
        from marine_ssl.experiments.codebook import select_ordinals
        first = select_ordinals(100000, seed=13)
        self.assertEqual(len(first), 5120)
        self.assertEqual(len(set(first)), 5120)
        self.assertEqual(first, select_ordinals(100000, seed=13))
        self.assertNotEqual(first, select_ordinals(100000, seed=14))
        self.assertEqual(select_ordinals(3, seed=13), [0, 1, 2])


@unittest.skipUnless(HAS_NUMPY and HAS_TORCH, "NumPy/PyTorch required")
class TargetResumeTests(unittest.TestCase):
    def test_corrupt_tar_cannot_be_committed_or_skipped_on_resume(self):
        import numpy as np
        import torch
        from marine_ssl.experiments.codebook import label_corpus
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config = fixture(root)
            row = json.loads(Path(config["train_index"]).read_text())
            with tarfile.open(row["tar_path"], "r") as archive:
                payload = archive.extractfile("audio/00000000000000.wav").read()
            with tarfile.open(row["tar_path"], "a") as archive:
                extra = tarfile.TarInfo("audio/00000000080000.wav")
                extra.size = len(payload); archive.addfile(extra, io.BytesIO(payload))
            codebook = Path(config["codebook_dir"]); codebook.mkdir()
            np.save(codebook / "centroids.npy", np.zeros((200, 768), dtype="float32"))
            (codebook / "codebook.ready.json").write_text(json.dumps({"signature": "centroids", "sample": {
                "train": index_info(config, "train"), "teacher": config["teacher"]},
                "centroids_sha256": digest(codebook / "centroids.npy")}))
            for _ in range(2):
                clock = itertools.count(0, 31)
                with patch("marine_ssl.experiments.codebook._teacher", return_value=types.SimpleNamespace(device=torch.device("cpu"))), \
                     patch("marine_ssl.experiments.codebook._teacher_batch", return_value=torch.zeros(1, 249, 768)), \
                     patch("marine_ssl.experiments.codebook.time.monotonic", side_effect=lambda: next(clock)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "count mismatch"):
                        label_corpus(config)
                database = Path(config["labels_dir"]) / "labels.sqlite"
                with sqlite3.connect(database) as connection:
                    self.assertEqual(connection.execute("SELECT count(*) FROM recordings").fetchone()[0], 0)
                self.assertFalse((Path(config["labels_dir"]) / "labels.ready.json").exists())

    def test_sqlite_targets_resume_committed_recordings_without_relabeling(self):
        import numpy as np
        import torch
        from marine_ssl.experiments.codebook import label_corpus
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config = fixture(root)
            codebook = Path(config["codebook_dir"]); codebook.mkdir()
            np.save(codebook / "centroids.npy", np.zeros((200, 768), dtype="float32"))
            (codebook / "codebook.ready.json").write_text(json.dumps({"signature": "centroids", "sample": {
                "train": index_info(config, "train"), "teacher": config["teacher"]},
                "centroids_sha256": digest(codebook / "centroids.npy")}))
            teacher = types.SimpleNamespace(device=torch.device("cpu"))
            clock = iter(range(0, 100000, 31))
            with patch("marine_ssl.experiments.codebook._teacher", return_value=teacher), \
                 patch("marine_ssl.experiments.codebook._teacher_batch", side_effect=[torch.zeros(1, 249, 768), RuntimeError("interrupted")]), \
                 patch("marine_ssl.experiments.codebook.time.monotonic", side_effect=lambda: next(clock)), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    label_corpus(config)
            database = Path(config["labels_dir"]) / "labels.sqlite"
            with sqlite3.connect(database) as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM recordings").fetchone()[0], 1)
            with patch("marine_ssl.experiments.codebook._teacher", return_value=teacher), \
                 patch("marine_ssl.experiments.codebook._teacher_batch", return_value=torch.zeros(1, 249, 768)) as extract:
                result = label_corpus(config)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(result["recordings"], 2)
            self.assertEqual(result["database_sha256"], digest(database))


if __name__ == "__main__":
    unittest.main()
