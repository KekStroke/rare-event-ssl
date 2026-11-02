import unittest

from marine_ssl.experiments.compare import paired_comparison


class ComparisonTests(unittest.TestCase):
    def test_identical_models_have_zero_paired_difference_even_with_clusters(self):
        rows = [{"row_id": str(i), "group": str(i // 3), "truth": i % 2,
                 "prediction": i % 2, "score": float(i % 2)} for i in range(12)]
        result = paired_comparison(rows, rows, classes=[0, 1], binary=True, n_resamples=25)
        self.assertEqual(result["paired_source_bootstrap_95_percent"], [0., 0.])
        self.assertEqual(result["bootstrap_groups"], 4)
        with self.assertRaises(ValueError):
            paired_comparison(rows, rows[:-1], classes=[0, 1], binary=True, n_resamples=2)

    def test_multiclass_delta_is_same_cohort(self):
        rows = [{"row_id": str(i), "group": str(i), "truth": i % 2,
                 "prediction": 0, "score": None} for i in range(10)]
        improved = [{**row, "prediction": row["truth"]} for row in rows]
        result = paired_comparison(rows, improved, classes=[0, 1], binary=False, n_resamples=30)
        self.assertGreater(result["difference_candidate_minus_reference"], .6)


if __name__ == "__main__":
    unittest.main()
