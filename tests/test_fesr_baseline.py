import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "Evaluation Seminar" / "Day_02"
sys.path.insert(0, str(MODULE_DIR))

import dataset_pipeline as pipeline
import fesr_baseline as fesr


class FesrFeatureTests(unittest.TestCase):
    def test_one_channel_produces_240_features(self):
        cycle = np.arange(pipeline.CYCLE_SAMPLES, dtype=float)
        X = np.stack((cycle, cycle + 5))[:, None, :]
        features, names = fesr.extract_fesr_features(X)
        self.assertEqual(features.shape, (2, 240))
        self.assertEqual(len(names), 240)
        self.assertAlmostEqual(features[0, 0], np.mean(np.arange(12)))
        self.assertAlmostEqual(features[0, 1], pipeline.SAMPLE_RATE_HZ)

    def test_wrong_sensor_dimension_is_rejected(self):
        X = np.zeros((3, 4, pipeline.CYCLE_SAMPLES))
        with self.assertRaises(ValueError):
            fesr.extract_fesr_features(X)

    def test_rfe_places_predictive_feature_first(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(100, 5))
        y = 7 * X[:, 3] + rng.normal(scale=0.01, size=100)
        ranking = fesr.rfe_lsr_ranking(X, y)
        self.assertEqual(ranking[0], 3)

    def test_pearson_places_predictive_feature_first(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(100, 5))
        y = -4 * X[:, 2] + rng.normal(scale=0.01, size=100)
        ranking = fesr.pearson_ranking(X, y)
        self.assertEqual(ranking[0], 2)


if __name__ == "__main__":
    unittest.main()
