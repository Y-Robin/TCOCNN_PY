import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
from skopt.space import Categorical, Real


NETWORKS_DIRECTORY = Path(__file__).resolve().parents[1] / "Networks"
sys.path.insert(0, str(NETWORKS_DIRECTORY))

from TCOCNN import TCOCNNClass
from TCOCNNs import TCOCNNsClass
from TCOCNNv2 import TCOCNNv2Class
from TCOCNNv3 import TCOCNNv3Class
from _torch_tcocnn import ResidualConvBlock, SamePadConv2d


SMALL_PARAMS = {
    "n_filter": 4,
    "section_depth": 2,
    "kernel": 5,
    "stride": 2,
    "num_neurons": 8,
    "drop_out": 0.1,
}

SMALL_V3_PARAMS = {
    **SMALL_PARAMS,
    "convs_per_block": 3,
    "channel_growth": 2,
    "residual": True,
}


class TCOCNNCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(42)
        cls.data = generator.normal(size=(8, 4, 64, 1)).astype(np.float32)
        cls.targets = generator.normal(size=(8, 1)).astype(np.float32)

    def test_regression_training_prediction_copy_and_explanations(self):
        model = TCOCNNClass((4, 64, 1), 1, regression=True, device="cpu")
        self.assertTrue(model.recalibrate_batchnorm_after_epoch)
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
        self.assertTrue(model.recalibrate_batchnorm_after_epoch)
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

    def test_v2_uses_paired_convolutions_and_global_average_pooling(self):
        model = TCOCNNv2Class((4, 64, 1), 1, regression=True, device="cpu")
        self.assertTrue(model.recalibrate_batchnorm_after_epoch)
        model.build_net(SMALL_PARAMS)
        module = model._require_model()

        convolutions = [
            layer for layer in module.features if isinstance(layer, SamePadConv2d)
        ]
        pools = [
            layer
            for layer in module.features
            if isinstance(layer, torch.nn.MaxPool2d)
        ]
        self.assertEqual(len(convolutions), 2 * SMALL_PARAMS["section_depth"])
        self.assertEqual(len(pools), SMALL_PARAMS["section_depth"])
        self.assertTrue(all(layer.stride == (1, 1) for layer in convolutions))
        for section_index in range(SMALL_PARAMS["section_depth"]):
            first, second = convolutions[2 * section_index : 2 * section_index + 2]
            self.assertEqual(first.conv.out_channels, second.conv.out_channels)
        self.assertIsInstance(module.global_pool, torch.nn.AdaptiveAvgPool2d)
        self.assertEqual(module.fc1.in_features, convolutions[-1].conv.out_channels)

        model.compile_model(1e-3)
        model.train(self.data, self.targets, epochs=1, batch_size=4)
        predictions = model.predict(self.data[:2])
        self.assertEqual(predictions.shape, (2, 1))
        self.assertTrue(np.isfinite(predictions).all())
        np.testing.assert_allclose(
            predictions,
            model.copy().predict(self.data[:2]),
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

    def test_v3_uses_configurable_residual_blocks_and_global_average_pooling(self):
        model = TCOCNNv3Class((4, 64, 1), 1, regression=True, device="cpu")
        model.build_net(SMALL_V3_PARAMS)
        module = model._require_model()

        blocks = [
            layer for layer in module.features if isinstance(layer, ResidualConvBlock)
        ]
        pools = [
            layer for layer in module.features if isinstance(layer, torch.nn.MaxPool2d)
        ]
        self.assertEqual(len(blocks), SMALL_V3_PARAMS["section_depth"])
        self.assertEqual(len(pools), SMALL_V3_PARAMS["section_depth"])
        self.assertTrue(all(block.use_residual for block in blocks))
        self.assertTrue(
            all(
                len(block.convolutions) == SMALL_V3_PARAMS["convs_per_block"]
                for block in blocks
            )
        )
        self.assertEqual(blocks[0].convolutions[0].conv.out_channels, 4)
        self.assertEqual(blocks[1].convolutions[0].conv.out_channels, 6)
        self.assertTrue(
            all(
                convolution.stride == (1, 1)
                for block in blocks
                for convolution in block.convolutions
            )
        )
        self.assertIsInstance(module.global_pool, torch.nn.AdaptiveAvgPool2d)

        model.compile_model(1e-3)
        with mock.patch.object(
            model, "_recalibrate_batchnorm", wraps=model._recalibrate_batchnorm
        ) as recalibrate:
            model.train(self.data, self.targets, epochs=1, batch_size=4)
        recalibrate.assert_called_once_with(self.data, 4)
        predictions = model.predict(self.data[:2])
        self.assertEqual(predictions.shape, (2, 1))
        np.testing.assert_allclose(
            predictions,
            model.copy().predict(self.data[:2]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_hyperparameter_optimization_restores_best_validation_model(self):
        search_space = [
            Categorical([3, 4], name="n_filter"),
            Categorical([1, 2], name="section_depth"),
            Categorical([3, 5], name="kernel"),
            Categorical([2, 3], name="stride"),
            Categorical([6, 8], name="num_neurons"),
            Real(0.0, 0.2, name="drop_out"),
            Real(5e-4, 1e-3, name="initial_learning_rate"),
            Categorical([2, 4], name="batch_size"),
        ]
        model = TCOCNNClass((4, 64, 1), 1, regression=True, device="cpu")
        result = model.optimize_model(
            self.data[:6],
            self.targets[:6] * 100.0,
            self.data[6:],
            self.targets[6:] * 100.0,
            num_epochs=2,
            trial_epochs=1,
            search_space=search_space,
            random_state=7,
            plot_results=False,
        )

        self.assertEqual(len(result.func_vals), 2)
        self.assertEqual(len(model.optimization_trials), 2)
        self.assertIsNotNone(model.best_optim_params)
        self.assertIsNotNone(model.best_batch_size)
        self.assertTrue(np.isfinite(model.best_validation_rmse))
        self.assertEqual(model.predict(self.data[6:]).shape, (2, 1))

    def test_v2_hyperparameter_optimization_restores_best_validation_model(self):
        search_space = [
            Categorical([3, 4], name="n_filter"),
            Categorical([1, 2], name="section_depth"),
            Categorical([3, 5], name="kernel"),
            Categorical([2, 3], name="stride"),
            Categorical([6, 8], name="num_neurons"),
            Real(0.0, 0.2, name="drop_out"),
            Real(5e-4, 1e-3, name="initial_learning_rate"),
            Categorical([2, 4], name="batch_size"),
        ]
        model = TCOCNNv2Class((4, 64, 1), 1, regression=True, device="cpu")
        result = model.optimize_model(
            self.data[:6],
            self.targets[:6] * 100.0,
            self.data[6:],
            self.targets[6:] * 100.0,
            num_epochs=2,
            trial_epochs=1,
            search_space=search_space,
            random_state=11,
            plot_results=False,
        )

        self.assertEqual(len(result.func_vals), 2)
        self.assertEqual(model.architecture, "v2")
        self.assertIsNotNone(model.best_optim_params)
        self.assertIsNotNone(model.best_batch_size)
        self.assertTrue(np.isfinite(model.best_validation_rmse))
        self.assertEqual(model.predict(self.data[6:]).shape, (2, 1))

    def test_v3_hyperparameter_optimization_uses_extended_search_space(self):
        search_space = [
            Categorical([3, 4], name="n_filter"),
            Categorical([1, 2], name="section_depth"),
            Categorical([3, 5], name="kernel"),
            Categorical([2, 3], name="stride"),
            Categorical([1, 2], name="convs_per_block"),
            Categorical([0, 2], name="channel_growth"),
            Categorical([False, True], name="residual"),
            Categorical([6, 8], name="num_neurons"),
            Real(0.0, 0.2, name="drop_out"),
            Real(5e-4, 1e-3, name="initial_learning_rate"),
            Categorical([2, 4], name="batch_size"),
        ]
        model = TCOCNNv3Class((4, 64, 1), 1, regression=True, device="cpu")
        result = model.optimize_model(
            self.data[:6],
            self.targets[:6] * 100.0,
            self.data[6:],
            self.targets[6:] * 100.0,
            num_epochs=2,
            trial_epochs=1,
            search_space=search_space,
            random_state=13,
            plot_results=False,
        )

        self.assertEqual(len(result.func_vals), 2)
        self.assertEqual(model.architecture, "v3")
        self.assertIn("convs_per_block", model.best_optim_params)
        self.assertIn("residual", model.best_optim_params)
        self.assertTrue(np.isfinite(model.best_validation_rmse))
        self.assertEqual(model.predict(self.data[6:]).shape, (2, 1))

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
