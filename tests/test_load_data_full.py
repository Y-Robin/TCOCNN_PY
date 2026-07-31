import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
from scipy.io import savemat


matplotlib.use("Agg")
PREPROCESSING_DIRECTORY = Path(__file__).resolve().parents[1] / "PreProcessing"
sys.path.insert(0, str(PREPROCESSING_DIRECTORY))

import loadDataFull


class LoadDataFullNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        directory = Path(self.temporary_directory.name)
        rows, features = 80, 12
        sensor = np.arange(1, rows * features + 1, dtype=np.float64).reshape(
            rows, features
        )
        ranges = np.repeat(np.arange(4, 44, dtype=np.float64), 2).reshape(-1, 1)
        target = np.linspace(1.0, 300.0, rows, dtype=np.float64).reshape(-1, 1)
        self.sensor_path = directory / "sensor.mat"
        self.target_path = directory / "target.mat"
        savemat(self.sensor_path, {"sensor": sensor})
        # loadDataFull transposes target gases and therefore expects the
        # MATLAB-style 1xN target layout used by the project datasets.
        savemat(self.target_path, {"acetone": target.T, "range": ranges.T})
        self.base_config = {
            "fileNameDataAll": [str(self.sensor_path)],
            "fileNameTargetAll": [str(self.target_path)],
            "targetGas": "acetone",
            "loadMethod": 1,
            "dataSize": (1, features),
            "randomFlag": False,
            "rng_val": 42,
            "OcclusionFlag": False,
            "saveFlag": False,
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def load_mode(self, mode):
        data, _ = loadDataFull.load_Data_Full(
            {**self.base_config, "normFlag": mode}
        )
        return data["train"]

    def test_boolean_modes_remain_backward_compatible(self):
        np.testing.assert_allclose(
            self.load_mode(True), self.load_mode("standard"), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            self.load_mode(False), self.load_mode("none"), rtol=0, atol=0
        )

    def test_log_modes_are_finite_and_distinct(self):
        unchanged = self.load_mode("none")
        logged = self.load_mode("log1p")
        logged_standardized = self.load_mode("log1p_standard")
        self.assertTrue(np.isfinite(logged).all())
        self.assertTrue(np.isfinite(logged_standardized).all())
        self.assertFalse(np.allclose(unchanged, logged))
        self.assertFalse(np.allclose(logged, logged_standardized))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load_mode("unknown")


if __name__ == "__main__":
    unittest.main()
