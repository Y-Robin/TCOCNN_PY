import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Evaluation Seminar" / "Day_02" / "dataset_pipeline.py"
SPEC = importlib.util.spec_from_file_location("dataset_pipeline", MODULE_PATH)
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class TemperatureBoundaryTests(unittest.TestCase):
    def test_boundary_points_cover_only_phase_edges(self):
        points = pipeline.temperature_boundary_points()
        self.assertEqual(len(points), 48)
        self.assertEqual(points[0].index, 0)
        self.assertEqual(points[-1].index, 1439)
        self.assertEqual({point.edge for point in points}, {"start", "end"})
        self.assertEqual(
            {point.temperature_c for point in points},
            {400, *range(100, 400, 25)},
        )
        profile = pipeline.temperature_profile()
        self.assertEqual(profile.shape, (1440,))
        self.assertTrue(np.all(profile[:50] == 400))
        self.assertTrue(np.all(profile[50:120] == 100))

    def test_point_dataset_keeps_singleton_axes(self):
        n = 8
        targets = {
            name: np.linspace(1, 10, n)
            for name in pipeline.INTERFERENCE_TARGETS
        }
        targets["range"] = np.arange(n)
        fake_splits = {
            split: {
                "X": np.zeros((n, 1, pipeline.CYCLE_SAMPLES), dtype=np.float32),
                "targets": targets,
                "source_indices": np.arange(n),
                "source": "test",
            }
            for split in pipeline.SPLIT_NAMES
        }
        score = pipeline.PointScore(
            "acetone", 49, 400, "high", 0, "end", 0.5, 0.5,
            "water", 0.2, 0.2, 0.3,
        )
        dataset = pipeline.build_point_dataset(
            fake_splits, "acetone", score
        )
        for split in pipeline.SPLIT_NAMES:
            self.assertEqual(dataset[f"X_{split}"].shape, (n, 1, 1))

        combined = pipeline.build_multi_point_dataset(
            fake_splits, "acetone", [0, 49, 50, 119]
        )
        for split in pipeline.SPLIT_NAMES:
            self.assertEqual(combined[f"X_{split}"].shape, (n, 1, 4))


if __name__ == "__main__":
    unittest.main()
