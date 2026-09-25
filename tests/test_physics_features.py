import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "Evaluation Seminar" / "Day_02"
sys.path.insert(0, str(MODULE_DIR))

import dataset_pipeline as pipeline
import physics_features as physics


class PhaseFeatureTests(unittest.TestCase):
    def test_phase_layout_covers_cycle(self):
        phases = physics.temperature_phases()
        self.assertEqual(len(phases), 24)
        self.assertEqual(phases[0].start, 0)
        self.assertEqual(phases[-1].stop, pipeline.CYCLE_SAMPLES)
        self.assertTrue(all(a.stop == b.start for a, b in zip(phases[:-1], phases[1:])))

    def test_tau63_recovers_first_order_response(self):
        cycles = np.zeros((1, 1, pipeline.CYCLE_SAMPLES), dtype=float)
        tau_true = 1.5
        phase = physics.temperature_phases()[0]
        t = np.arange(phase.stop - phase.start) / pipeline.SAMPLE_RATE_HZ
        cycles[0, 0, phase.start:phase.stop] = 100 + 50 * (1 - np.exp(-t / tau_true))
        features, names = physics.extract_phase_features(cycles)
        tau = features[0, names.index(phase.label + "__tau63_s")]
        self.assertAlmostEqual(tau, tau_true, delta=0.35)
        self.assertEqual(features.shape, (1, 24 * 7))


class AlaFeatureTests(unittest.TestCase):
    def test_breakpoints_and_feature_shape(self):
        x = np.arange(pipeline.CYCLE_SAMPLES)
        reference = 0.01 * x + 20 * np.sin(x / 35)
        breakpoints = physics.learn_ala_breakpoints(reference, n_segments=20)
        self.assertEqual(len(breakpoints), 21)
        self.assertEqual(breakpoints[0], 0)
        self.assertEqual(breakpoints[-1], pipeline.CYCLE_SAMPLES - 1)
        self.assertTrue(np.all(np.diff(breakpoints) > 0))

        cycles = np.stack((reference, reference + 2))[..., None, :]
        features, names = physics.extract_ala_features(cycles, breakpoints)
        self.assertEqual(features.shape, (2, 40))
        self.assertEqual(len(names), 40)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
