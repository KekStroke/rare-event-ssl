import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from marine_ssl.experiments.compare_temporal import (
    _ap_histogram, _weighted_ap, paired_temporal_comparison,
)
from marine_ssl.experiments.metrics import event_metrics
from marine_ssl.experiments.temporal import _resolved


def record(identity, group, references, predictions, *, label="click", fold=None):
    def events(values, scored):
        return [{"start": start, "end": start, "label": label, "recording_id": identity,
                 **({"score": score} if scored else {})} for start, score in values]
    return {"row_id": identity, "recording_id": identity, "group": group, "fold": fold,
            "duration_seconds": 60., "references": events(references, False),
            "predictions": events(predictions, True), "call_predictions": [],
            "budget_thresholds": {"50": .5, "100": .5, "300": .5}}


def metadata(task="dominica"):
    return {"task": task, "status": "completed", "dataset_fingerprint": "same-cohort",
            "labels": ["sound"] if task == "manual" else ["K1"] if task == "avacha_field" else ["click"],
            "config": _resolved({}), "folds": [{"fold": None, "selection": {"data": "validation_only"},
                "recall_at_dev_far": {str(b): {"validation": {"threshold": .5}} for b in (50, 100, 300)}}]}


def save(path, rows, meta):
    path.mkdir(parents=True, exist_ok=True)
    (path / "temporal_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "metrics.json").write_text(json.dumps(meta))
    return path


class TemporalComparisonTests(unittest.TestCase):
    def test_weighted_ap_equals_unique_recording_replication_with_ties_and_rerouting(self):
        rows = [record("a", "group-a", [(0., None), (.01, None)], [(.004, .9), (0., .8), (0., .7)]),
                record("b", "group-b", [(1., None)], [(1., .8)])]
        artifact = {"rows": {r["row_id"]: r for r in rows}}
        hist = _ap_histogram(artifact, ["group-a", "group-b"], tolerance=.006)
        for counts in ([1, 1], [2, 1], [0, 3], [3, 0]):
            refs, preds = [], []
            for original, count in zip(rows, counts):
                for replica in range(count):
                    unique = f"{original['recording_id']}:replica:{replica}"
                    refs.extend({**e, "recording_id": unique} for e in original["references"])
                    preds.extend({**e, "recording_id": unique} for e in original["predictions"])
            expected = event_metrics(refs, preds, onset_tolerance_seconds=.006)["average_precision"]
            self.assertAlmostEqual(_weighted_ap(hist, np.asarray(counts)), expected)
        self.assertIsNone(_weighted_ap(hist, np.array([0, 0])))

    def test_public_point_comparison_is_paired_seeded_and_reports_secondary_budget(self):
        reference = [record("a", "source-a", [(1., None)], [(1.006, .9)]),
                     record("b", "source-b", [(1., None)], [(1.02, .8)])]
        candidate = copy.deepcopy(reference)
        candidate[1]["predictions"][0]["start"] = candidate[1]["predictions"][0]["end"] = 1.006
        with tempfile.TemporaryDirectory() as directory:
            a = save(Path(directory) / "a", reference, metadata())
            b = save(Path(directory) / "b", candidate, metadata())
            result = paired_temporal_comparison(a, b, n_resamples=100, seed=7)
            self.assertEqual(result, paired_temporal_comparison(a / "temporal_predictions.jsonl", b, n_resamples=100, seed=7))
            primary = result["metrics"]["ap_onset_10ms"]
            self.assertEqual(primary["reference"], .5)
            self.assertEqual(primary["candidate"], 1.)
            self.assertEqual(primary["difference_candidate_minus_reference"], .5)
            self.assertEqual(result["bootstrap_groups"], 2)
            self.assertEqual(primary["undefined_resamples"], 0)
            with patch("marine_ssl.experiments.compare_temporal.SECONDARY_AP_OPERATION_BUDGET", 0):
                bounded = paired_temporal_comparison(a, b, n_resamples=10)
            self.assertEqual(bounded["metrics"]["ap_onset_1ms"]["status"], "secondary_ci_not_computed_budget_limit")
            self.assertIsNone(bounded["metrics"]["ap_onset_5ms"]["paired_source_bootstrap_95_percent"])

    def test_field_budget_bootstrap_uses_saved_dev_thresholds_and_fast_record_counts(self):
        reference = [record("a", "source-a", [(1., None)], [(1., .9)], label="K1"),
                     record("b", "source-b", [(1., None)], [(1., .1)], label="K1")]
        for item in reference:
            for event in item["references"] + item["predictions"]:
                event["end"] = 2.
            item["call_predictions"] = [{**event, "label": "call"} for event in item["predictions"]]
        candidate = copy.deepcopy(reference)
        candidate[1]["call_predictions"][0]["score"] = .9
        with tempfile.TemporaryDirectory() as directory:
            a = save(Path(directory) / "a", reference, metadata("avacha_field"))
            b = save(Path(directory) / "b", candidate, metadata("avacha_field"))
            result = paired_temporal_comparison(a, b, n_resamples=40)
            self.assertEqual(len(result["primary_metrics"]), 3)
            for metric in result["metrics"].values():
                self.assertEqual(metric["reference"], .5)
                self.assertEqual(metric["candidate"], 1.)
                self.assertEqual(metric["candidate_test_false_alarms_per_hour"], 0.)
                self.assertEqual(metric["candidate_operating_counts"]["exposure_seconds"], 120.)
            candidate[0]["budget_thresholds"]["50"] = .7
            save(b, candidate, metadata("avacha_field"))
            with self.assertRaisesRegex(ValueError, "threshold"):
                paired_temporal_comparison(a, b, n_resamples=2)

    def test_manual_interval_ap_missing_and_undefined_are_explicit(self):
        rows = [record("a", "source-a", [(1., None)], [(1., .9)], label="sound")]
        for event in rows[0]["references"] + rows[0]["predictions"]:
            event["end"] = 2.
        with tempfile.TemporaryDirectory() as directory:
            a = save(Path(directory) / "a", rows, metadata("manual"))
            b = save(Path(directory) / "b", rows, metadata("manual"))
            result = paired_temporal_comparison(a, b, n_resamples=20)
            self.assertEqual(result["metrics"]["ap_iou_0.5"]["paired_source_bootstrap_95_percent"], [0., 0.])
            missing = paired_temporal_comparison(a, Path(directory) / "missing", n_resamples=2)
            self.assertEqual(missing["status"], "missing_artifact")
            self.assertTrue(missing["candidate_missing"])
            rows[0]["references"] = []
            save(a, rows, metadata("manual")); save(b, rows, metadata("manual"))
            undefined = paired_temporal_comparison(a, b, n_resamples=5)
            self.assertEqual(undefined["status"], "undefined_primary_or_resamples")
            self.assertEqual(undefined["metrics"]["ap_iou_0.5"]["undefined_resamples"], 5)

    def test_mismatched_truth_groups_criteria_and_duplicate_rows_fail_closed(self):
        rows = [record("a", "source-a", [(1., None)], [(1., .9)])]
        with tempfile.TemporaryDirectory() as directory:
            a = save(Path(directory) / "a", rows, metadata())
            b = Path(directory) / "b"
            for field, replacement in (("group", "different"), ("recording_id", "different")):
                changed = copy.deepcopy(rows); changed[0][field] = replacement
                save(b, changed, metadata())
                with self.assertRaises(ValueError):
                    paired_temporal_comparison(a, b, n_resamples=2)
            changed = copy.deepcopy(rows); changed[0]["references"][0]["start"] = .9
            save(b, changed, metadata())
            with self.assertRaisesRegex(ValueError, "truth"):
                paired_temporal_comparison(a, b, n_resamples=2)
            meta = metadata(); meta["config"]["selection_tolerance_seconds"] = .005
            save(b, rows, meta)
            with self.assertRaisesRegex(ValueError, "config"):
                paired_temporal_comparison(a, b, n_resamples=2)
            save(b, rows + rows, metadata())
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                paired_temporal_comparison(a, b, n_resamples=2)


if __name__ == "__main__":
    unittest.main()
