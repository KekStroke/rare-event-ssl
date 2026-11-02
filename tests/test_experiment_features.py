import tempfile
from pathlib import Path
import unittest

import numpy as np

from marine_ssl.experiments.features import FRAME_BLOCK_FRAMES, FeatureStore, feature_identity, pool_frames


class FeatureTests(unittest.TestCase):
    def test_context_identity_ignores_task_and_split_but_preserves_content(self):
        row = {"path": "/audio.wav", "start_seconds": 0, "end_seconds": 5, "pcm_sha256": "a"}
        key = feature_identity(row, [6, 12], False)
        self.assertEqual(key, feature_identity({**row, "split": "test", "label": 1}, [6, 12], False))
        self.assertNotEqual(key, feature_identity({**row, "pcm_sha256": "b"}, [6, 12], False))
        self.assertNotEqual(key, feature_identity(row, [6, 12], True))

    def test_atomic_cache_resume_and_model_identity(self):
        row = {"path": "/fake.wav", "start_seconds": 0, "end_seconds": 5}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.sqlite"
            store = FeatureStore(path, {"sha256": "a"}, encoder=object())
            store._extract = lambda row, temporal: {"pooled_6": np.arange(4, dtype=np.float32)}
            np.testing.assert_array_equal(store.get(row)["pooled_6"], np.arange(4))
            store.close()
            resumed = FeatureStore(path, {"sha256": "a"}, readonly=True)
            np.testing.assert_array_equal(resumed.get(row)["pooled_6"], np.arange(4))
            resumed.close()
            with self.assertRaises(ValueError):
                FeatureStore(path, {"sha256": "different"}, readonly=True)

    def test_population_pooling_and_nonfinite_rejection(self):
        np.testing.assert_array_equal(pool_frames([[0, 2], [2, 4]]), [1, 3, 1, 1])
        with self.assertRaises(ValueError):
            pool_frames([[float("nan")]])

    def test_temporal_arrays_use_bounded_blocks_and_resume_exactly(self):
        row = {"path": "/fake-long.wav", "start_seconds": 0, "end_seconds": 5115}
        values = np.arange((FRAME_BLOCK_FRAMES + 3) * 2, dtype=np.float16).reshape(-1, 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.sqlite"
            store = FeatureStore(path, {"sha256": "a"}, layers=[6], encoder=object())
            store._extract = lambda row, temporal: {"pooled_6": np.zeros(4), "frames_6": values}
            store.get(row, temporal=True)
            self.assertEqual(store.db.execute("SELECT count(*) FROM frames").fetchone()[0], 2)
            self.assertLess(store.db.execute("SELECT max(length(data)) FROM frames").fetchone()[0], 10**9)
            store.close()
            resumed = FeatureStore(path, {"sha256": "a"}, layers=[6], readonly=True)
            np.testing.assert_array_equal(resumed.get(row, temporal=True)["frames_6"], values)
            resumed.close()


if __name__ == "__main__":
    unittest.main()
