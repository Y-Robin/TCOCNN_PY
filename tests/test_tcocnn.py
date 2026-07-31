import sys
import unittest
from pathlib import Path

import numpy as np
import torch


NETWORKS_DIRECTORY = Path(__file__).resolve().parents[1] / "Networks"
sys.path.insert(0, str(NETWORKS_DIRECTORY))

from TCOCNN import TCOCNNClass
from TCOCNNs import TCOCNNsClass


SMALL_PARAMS = {
    "n_filter": 4,
    "section_depth": 2,
    "kernel": 5,
    "stride": 2,
    "num_neurons": 8,
    "drop_out": 0.1,
}


class TCOCNNCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(42)
        cls.data = generator.normal(size=(8, 4, 64, 1)).astype(np.float32)
        cls.targets = generator.normal(size=(8, 1)).astype(np.float32)

    def test_regression_training_prediction_copy_and_explanations(self):
        model = TCOCNNClass((4, 64, 1), 1, regression=True, device="cpu")
        model.build_net(SMALL_PARAMS)
        model.compile_model(1e-3)
        model.train(
            self.data,
            self.targets,
            validation_data=(self.data[:2], self.targets[:2]),
            epochs=1,
            batch_size=4,
        )

        predictions = model.predict(self.data[:2])
        self.assertEqual(predictions.shape, (2, 1))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertIn("loss", model.history.history)
        self.assertIn("val_loss", model.history.history)

        copied_model = model.copy()
        np.testing.assert_allclose(
            predictions,
            copied_model.predict(self.data[:2]),
            rtol=1e-6,
            atol=1e-6,
        )
        copied_model.retrain(
            self.data,
            self.targets,
            epochs=1,
            batch_size=4,
            new_learning_rate=5e-4,
        )

        gradient_map = model.get_gradient_map(self.data[:2], window_size=3)
        self.assertEqual(gradient_map.shape, self.data[:2].shape)
        self.assertTrue(np.isfinite(gradient_map).all())

        occlusion_map = model.custom_occlusion(self.data, self.data[:1])
        self.assertEqual(occlusion_map.shape, self.data[:1].shape)
        self.assertTrue(np.isfinite(occlusion_map).all())

    def test_pooling_variant_and_classification_probabilities(self):
        labels = np.arange(self.data.shape[0]) % 3
        model = TCOCNNsClass((4, 64, 1), 3, regression=False, device="cpu")
        model.build_net(SMALL_PARAMS)
        model.compile_model(1e-3)
        model.train(self.data, labels, epochs=1, batch_size=4)

        probabilities = model.predict(self.data[:3])
        self.assertEqual(probabilities.shape, (3, 3))
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(3),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_original_notebook_input_shape(self):
        params = {
            **SMALL_PARAMS,
            "section_depth": 4,
            "kernel": 76,
            "stride": 31,
        }
        model = TCOCNNClass((4, 1440, 1), 1, regression=True, device="cpu")
        model.build_net(params)
        model.compile_model()
        predictions = model.predict(
            np.zeros((2, 4, 1440, 1), dtype=np.float32)
        )
        self.assertEqual(predictions.shape, (2, 1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_automatic_cuda_selection_and_training(self):
        model = TCOCNNClass((4, 64, 1), 1, regression=True)
        self.assertEqual(model.device.type, "cuda")
        model.build_net(SMALL_PARAMS)
        model.compile_model(1e-3)
        model.train(self.data, self.targets, epochs=1, batch_size=4)
        predictions = model.predict(self.data[:2])
        self.assertEqual(predictions.shape, (2, 1))
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
