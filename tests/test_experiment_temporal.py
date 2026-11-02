import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from marine_ssl.experiments.temporal import (
    _intervals, _resolved, _targets, evaluate_temporal,
)


def row(split, *, fold=None, point=False, task="dominica"):
    identity = f"{fold}-{split}"
    events = [{"time_seconds": .504, "label": "click"}] if point else [
        {"start_seconds": .4, "end_seconds": .8, "label": "sound" if task == "manual" else "K1"}]
    return {"row_id": identity, "path": f"/fixture/{identity}.wav", "recording_id": identity,
            "group": "source-" + identity, "split": split, "fold": fold,
            "start_seconds": 0., "end_seconds": 2., "events": events, "sample_rate": 156250 if point else 48000}


def config():
    return {"C": [.1, 1, 10], "max_iter": 1000, "temporal": {
        "layers": [1, 2], "embedding_dim": 2, "max_train_frames": 200,
        "min_positive_frames_per_class": 2, "max_iter": 200,
        "threshold_grid": [.4, .6, 1.000001], "report_ious": [.5]}}


class Loader:
    def __init__(self, point=False):
        self.calls = []
        self.point = point

    def __call__(self, item, layers):
        self.calls.extend((item["split"], layer) for layer in layers)
        times = np.arange(.01, 2., .02)
        active = np.zeros(len(times), dtype=bool)
        if self.point:
            active[np.argmin(abs(times - .504))] = True
        else:
            active = (times >= .4) & (times < .8)
        informative = np.stack([np.where(active, 3., -3.), np.zeros(len(times))], axis=1).astype(np.float32)
        return {layer: {"features": informative if layer == 1 else np.zeros_like(informative),
                        "times": times, "duration_seconds": 2., "frame_hop_seconds": .02,
                        "context_seconds": 5.} for layer in layers}


class ExperimentTemporalTests(unittest.TestCase):
    def test_point_evaluation_is_coarse_and_test_never_selects_layer_or_threshold(self):
        rows = [row(name, point=True) for name in ("train", "val", "test")]
        dataset = {"task": "dominica", "ready": True, "rows": rows, "fingerprint": "frozen"}
        with tempfile.TemporaryDirectory() as directory:
            loader = Loader(point=True)
            first = evaluate_temporal(dataset, loader, Path(directory) / "first", config())
            chosen = first["folds"][0]
            self.assertEqual(chosen["selected_layer"], 1)
            self.assertNotIn(("test", 2), loader.calls)
            self.assertEqual(chosen["head"]["training_recordings"], ["None-train"])
            self.assertFalse(chosen["resolution"]["subframe_localization"])
            self.assertEqual(chosen["resolution"]["observed_hops_seconds"], [.02])
            self.assertEqual(chosen["test"]["onset_1ms"]["pooled"]["true_positive"], 0)
            self.assertEqual(chosen["test"]["onset_5ms"]["pooled"]["true_positive"], 0)
            self.assertEqual(chosen["test"]["onset_10ms"]["pooled"]["true_positive"], 1)
            self.assertEqual(chosen["resolution"]["nearest_grid_recall_upper_bound"], {"1ms": 0., "5ms": 0., "10ms": 1.})
            # Change only held-out labels. No validation choice may change.
            rows[2]["events"][0]["time_seconds"] = 1.004
            second = evaluate_temporal(dataset, Loader(point=True), Path(directory) / "second", config())
            self.assertEqual(chosen["selection"], second["folds"][0]["selection"])
            self.assertEqual(chosen["threshold"], second["folds"][0]["threshold"])
            self.assertEqual(second["folds"][0]["test"]["onset_10ms"]["pooled"]["true_positive"], 0)
            saved = [json.loads(line) for line in Path(first["prediction_artifact"]).read_text().splitlines()]
            self.assertEqual(saved[0]["group"], "source-None-test")
            self.assertTrue((Path(directory) / "first/head_fold_0_layer_1.npz").is_file())

    def test_field_folds_use_separate_heads_and_dev_far_thresholds(self):
        rows = [row(split, fold=fold, task="avacha_field") for fold in range(3)
                for split in ("train", "val", "test")]
        dataset = {"task": "avacha_field", "ready": True, "rows": rows,
                   "fingerprint": "field", "protocol": {"classes": ["K1"]}}
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_temporal(dataset, Loader(), directory, config())
            self.assertEqual(len(result["folds"]), 3)
            self.assertEqual(result["test_recording_count"], 3)
            self.assertEqual(result["test_exposure_seconds"], 6.)
            self.assertEqual(result["test"]["iou_0.5"]["pooled"]["true_positive"], 3)
            self.assertEqual(result["test"]["iou_0.5"]["pooled"]["f1"], 1.)
            self.assertEqual(result["recall_at_dev_far"]["100"]["recall"], 1.)
            for fold in result["folds"]:
                self.assertEqual(fold["test"]["iou_0.5"]["pooled"]["f1"], 1.)
                for budget, point in fold["recall_at_dev_far"].items():
                    self.assertLessEqual(point["validation"]["false_alarms_per_hour"], float(budget))
                    self.assertEqual(point["test"]["recall"], 1.)
                self.assertFalse(set(fold["head"]["training_recordings"]) & set(fold["test_recordings"]))

    def test_multilabel_targets_preserve_overlap_and_do_not_bridge_context_gaps(self):
        item = row("train", task="avacha_field")
        item["events"].append({"start_seconds": .5, "end_seconds": .7, "label": "K2"})
        times = np.array([.45, .55, .65, .75])
        cfg = _resolved({"embedding_dim": 2})
        targets, _, _, _, _ = _targets(item, times, ["K1", "K2"], False, {}, cfg)
        np.testing.assert_array_equal(targets, [[1, 0], [1, 1], [1, 1], [1, 0]])
        item["end_seconds"] = 6.
        decoded = _intervals(np.array([.9, .9]), np.array([4.97, 5.01]), .02, item, "K1", cfg)
        self.assertEqual(len(decoded), 2)
        self.assertLess(decoded[0]["end"], decoded[1]["start"])

    def test_manual_artifact_audit_requires_exact_pinned_csv_and_remains_auxiliary(self):
        rows = [row(name, task="manual") for name in ("train", "val", "test")]
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "manifest.csv"
            text = "audio,start_s,end_s,label\n" + "".join(
                f"None-{split}.wav,.4,.8,sound\nNone-{split}.wav,1.2,1.4,artifact\n"
                for split in ("train", "val", "test"))
            csv_path.write_text(text, encoding="utf-8")
            dataset = {"task": "manual", "ready": True, "auxiliary": True, "rows": rows,
                "fingerprint": "manual", "input_metadata_sha256": {str(csv_path): hashlib.sha256(csv_path.read_bytes()).hexdigest()}}
            result = evaluate_temporal(dataset, Loader(), Path(directory) / "output", config())
            self.assertTrue(result["auxiliary"])
            artifact = result["folds"][0]["artifact_metrics"]
            self.assertEqual(artifact["status"], "ok")
            self.assertEqual(artifact["artifact_only_frames"], 10)
            self.assertEqual(artifact["sound_false_positive_rate"], 0.)
            csv_path.write_text(text + "changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                evaluate_temporal(dataset, Loader(), Path(directory) / "mismatch", config())

    def test_leakage_missing_grids_and_not_ready_inputs_do_not_fit(self):
        rows = [row(name, point=True) for name in ("train", "val", "test")]
        dataset = {"task": "dominica", "ready": False, "rows": rows, "reasons": ["blocked"]}
        with tempfile.TemporaryDirectory() as directory:
            loader = Loader(point=True)
            result = evaluate_temporal(dataset, loader, Path(directory) / "blocked", config())
            self.assertEqual(result["status"], "not_ready")
            self.assertEqual(loader.calls, [])
            dataset["ready"] = True
            rows[1]["group"] = rows[0]["group"]
            with self.assertRaisesRegex(ValueError, "crosses"):
                evaluate_temporal(dataset, loader, Path(directory) / "leak", config())
            rows[1]["group"] = "val-only"
            def wrong_grid(item, layers):
                result = loader(item, layers)
                result[layers[0]]["times"] += 10
                return result
            with self.assertRaisesRegex(ValueError, "ABSOLUTE"):
                evaluate_temporal(dataset, wrong_grid, Path(directory) / "grid", config())


if __name__ == "__main__":
    unittest.main()
