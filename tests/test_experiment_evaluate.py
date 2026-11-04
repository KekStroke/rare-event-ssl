import tempfile
from pathlib import Path
import unittest

import numpy as np

from marine_ssl.experiments.evaluate import evaluate_external, fit_task, selection_metric, validation_threshold


class FakeStore:
    identity = {"model": "fixed"}

    def get(self, row):
        x = np.asarray([row["label"] * 2 - 1, row["i"] * .001], dtype=np.float32)
        return {"pooled_6": x, "chunks_6": x[None]}


class HeadTests(unittest.TestCase):
    def test_external_reuses_head_and_rejects_other_model(self):
        rows = [{"row_id": str(i), "recording_id": str(i), "group": str(i), "split": split,
                 "label": i % 2, "i": i} for i, split in enumerate(["train"] * 8 + ["val"] * 4 + ["test"] * 4)]
        cohort = {"task": "avacha_presence", "rows": rows, "fingerprint": "cohort", "notes": [], "protocol": {}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = fit_task(cohort, FakeStore(), root / "train", {"layers": [6], "C": [1], "seed": 42})
            self.assertEqual(result["test"]["average_precision"], 1)
            external = {**cohort, "task": "onc", "rows": rows[-4:], "protocol": {"requires_head_from": "avacha_presence"}}
            result = evaluate_external(external, FakeStore(), root / "external", root / "train/head.joblib")
            self.assertEqual(result["test"]["average_precision"], 1)
            other = FakeStore()
            other.identity = {"model": "different"}
            with self.assertRaises(ValueError):
                evaluate_external(external, other, root / "bad", root / "train/head.joblib")

    def test_threshold_comes_from_validation_only(self):
        threshold = validation_threshold([0, 1, 1], [.1, .6, .9])
        self.assertEqual(threshold, .6)

    def test_validation_class_universe_is_fixed_when_one_class_has_no_support(self):
        metrics = {"macro_f1": .4, "per_class": [{"label": "K1", "f1": .8}, {"label": "K27", "f1": 0.0}]}
        cohort = {"protocol": {"selection_labels": ["K1"]}}
        self.assertEqual(selection_metric(metrics, cohort, binary=False), .8)


if __name__ == "__main__":
    unittest.main()
