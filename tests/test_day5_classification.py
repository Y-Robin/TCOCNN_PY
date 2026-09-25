import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Evaluation Seminar" / "Day_05" / "day5_utils.py"
SPEC = importlib.util.spec_from_file_location("day5_utils", MODULE_PATH)
day5 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = day5
SPEC.loader.exec_module(day5)


class Day5CsvTests(unittest.TestCase):
    def test_split_loader_ignores_first_cycle_and_builds_tcocnn_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            for class_index, class_name in enumerate(day5.CLASS_NAMES):
                values = np.arange(4 * 12, dtype=np.float32).reshape(4, 12)
                values += class_index * 1000
                np.savetxt(
                    data_dir / f"train_{class_name}.csv",
                    values,
                    delimiter=",",
                )

            X, y, sources = day5.load_binary_split(
                data_dir,
                "train",
                expected_samples=12,
            )

        self.assertEqual(X.shape, (6, 1, 12, 1))
        np.testing.assert_array_equal(y, [0, 0, 0, 1, 1, 1])
        self.assertEqual(sources.count("train_water.csv"), 3)
        self.assertEqual(float(X[0, 0, 0, 0]), 12.0)

    def test_log_zscore_is_applied_independently_per_sample(self):
        X = np.arange(1, 49, dtype=np.float32).reshape(4, 1, 12, 1)
        transformed = day5.log_zscore_per_sample(X)
        self.assertEqual(transformed.shape, X.shape)
        sample_mean = transformed.mean(axis=(1, 2, 3))
        sample_std = transformed.std(axis=(1, 2, 3))
        np.testing.assert_allclose(sample_mean, 0.0, atol=1e-6)
        np.testing.assert_allclose(sample_std, 1.0, atol=1e-6)

    def test_log_zscore_rejects_non_positive_values(self):
        X = np.ones((2, 1, 12, 1), dtype=np.float32)
        X[0, 0, 0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            day5.log_zscore_per_sample(X)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_stratified_split_keeps_both_classes(self):
        X = np.zeros((10, 1, 12, 1), dtype=np.float32)
        y = np.array([0] * 5 + [1] * 5)
        _, X_validation, _, y_validation = day5.stratified_train_validation_split(
            X, y
        )
        self.assertEqual(X_validation.shape[0], 2)
        self.assertEqual(set(y_validation), {0, 1})


if __name__ == "__main__":
    unittest.main()
