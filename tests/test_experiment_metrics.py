import unittest

from marine_ssl.experiments.metrics import (
    binary_metrics, event_metrics, match_events, multiclass_metrics,
    paired_recording_bootstrap_ci,
)


class ExperimentMetricsTests(unittest.TestCase):
    def test_multiclass_absent_class_is_explicit_not_zero(self):
        result = multiclass_metrics([0, 0, 1], [0, 1, 1], labels=[0, 1, 2])
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertAlmostEqual(result["balanced_accuracy"], .75)
        self.assertEqual(result["undefined_f1_labels"], [2])
        self.assertEqual(result["per_class"][0]["support"], 2)
        self.assertIsNone(multiclass_metrics([], [])["accuracy"])

    def test_binary_ties_and_fixed_threshold(self):
        result = binary_metrics([1, 0], [.5, .5], threshold=.7)
        self.assertEqual(result["average_precision"], .5)
        self.assertEqual(result["roc_auc"], .5)
        self.assertEqual(result["f1"], 0)
        self.assertEqual(result["false_negative"], 1)
        self.assertIsNone(result["precision"])
        self.assertAlmostEqual(binary_metrics([1, 0, 1], [.9, .8, .7], threshold=.75)["average_precision"], 5 / 6)

    def test_binary_no_positive_and_empty_are_undefined(self):
        result = binary_metrics([0, 0], [.2, .8], threshold=.5)
        self.assertIsNone(result["average_precision"])
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(binary_metrics([], [], threshold=.5)["f1"])
        with self.assertRaises(ValueError):
            binary_metrics([1], [float("nan")], threshold=.5)

    def test_event_matching_reroutes_non_greedy_and_respects_recordings(self):
        refs = [{"start": 0}, {"start": .01}]
        preds = [{"start": .004}, {"start": 0}]
        result = match_events(refs, preds, onset_tolerance_seconds=.006)
        self.assertEqual(len(result["matches"]), 2)
        self.assertEqual(match_events([{"start": 0, "recording_id": "a"}],
                                     [{"start": 0, "recording_id": "b"}],
                                     onset_tolerance_seconds=.001)["matches"], [])
        self.assertEqual(match_events([{"start": 0, "label": "whistle"}],
                                     [{"start": 0, "label": "click"}],
                                     onset_tolerance_seconds=.001)["matches"], [])

    def test_event_ap_duplicates_ties_and_fixed_threshold(self):
        refs = [{"start": 0}, {"start": 1}]
        preds = [{"start": 0, "score": .9}, {"start": 0, "score": .8}, {"start": 1, "score": .7}]
        result = event_metrics(refs, preds, onset_tolerance_seconds=.001, duration_seconds=3600, threshold=.8)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertEqual((result["true_positive"], result["false_positive"], result["false_negative"]), (1, 1, 1))
        self.assertEqual(result["false_alarms_per_hour"], 1)
        tied = event_metrics(refs[:1], [{"start": 0, "score": .5}] * 2, onset_tolerance_seconds=.001)
        self.assertEqual(tied["average_precision"], .5)

    def test_grouped_point_windows_preserve_epsilon_and_interval_constraints(self):
        refs = [{"start": i / 100, "recording_id": f"recording-{r}", "label": "click"}
                for r in range(100) for i in range(100)]
        preds = [{"start": .504, "recording_id": f"recording-{r}", "label": "click"}
                 for r in range(100)]
        result = match_events(refs, preds, onset_tolerance_seconds=.004)
        self.assertEqual(len(result["matches"]), 100)
        self.assertTrue(all(refs[m["reference_index"]]["recording_id"] == preds[m["prediction_index"]]["recording_id"]
                            for m in result["matches"]))
        # The numerical tolerance margin remains part of eligibility.
        self.assertEqual(len(match_events([{"start": 0}], [{"start": .01 + .5e-12}],
                                          onset_tolerance_seconds=.01)["matches"]), 1)
        self.assertEqual(match_events([{"start": 0}], [{"start": .01 + 2e-12}],
                                      onset_tolerance_seconds=.01)["matches"], [])
        self.assertEqual(match_events([{"start": 0, "end": 1}], [{"start": 0, "end": 10}],
                                      onset_tolerance_seconds=.01, min_iou=.5)["matches"], [])

    def test_click_tolerances_and_iou_and_empty(self):
        for tolerance, expected in [(0.001, 0), (.005, 1), (.01, 1)]:
            result = event_metrics([{"start": 1}], [{"start": 1.004, "score": 1}], onset_tolerance_seconds=tolerance)
            self.assertEqual(result["true_positive"], expected)
        self.assertEqual(len(match_events([{"start": 0, "end": 2}], [{"start": 1, "end": 3}], min_iou=.3)["matches"]), 1)
        self.assertEqual(match_events([{"start": 0, "end": 2}], [{"start": 1, "end": 3}], min_iou=.5)["matches"], [])
        self.assertIsNone(event_metrics([], [], onset_tolerance_seconds=.01)["average_precision"])
        self.assertEqual(event_metrics([{"start": 0}], [], onset_tolerance_seconds=.01)["average_precision"], 0)
        with self.assertRaises(ValueError):
            match_events([], [])

    def test_bootstrap_pairs_clusters_and_is_seeded(self):
        rows = {"a": (1, 0), "b": (0, 1), "c": (1, .5)}
        statistic = lambda sample: sum(a - b for a, b in sample) / len(sample)
        first = paired_recording_bootstrap_ci(rows, statistic, n_resamples=100, seed=17)
        self.assertEqual(first, paired_recording_bootstrap_ci(rows, statistic, n_resamples=100, seed=17))
        self.assertAlmostEqual(first["estimate"], 1 / 6)
        self.assertLessEqual(first["lower"], first["estimate"])
        self.assertGreaterEqual(first["upper"], first["estimate"])
        undefined = paired_recording_bootstrap_ci({"a": 1}, lambda _: None, n_resamples=4)
        self.assertEqual(undefined["undefined_resamples"], 4)
        self.assertIsNone(undefined["lower"])


if __name__ == "__main__":
    unittest.main()
