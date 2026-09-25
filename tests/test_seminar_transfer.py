"""Regression checks for frozen-feature transfer and signed sensor attributions."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for path in ['Networks', 'Evaluation Seminar/Day_03', 'Evaluation Seminar/Day_04']:
    sys.path.insert(0, str(ROOT / path))
from day4_utils import transfer_subset, fit_network, gradient_and_cam, occlusion, DEFAULT_PARAMS
from transfer_workflow import (
    DirectStandardization, PiecewiseStandardization, paired_source_mean,
)


def test_transfer_budget_keeps_groups_and_is_nested():
    groups = np.repeat(np.arange(20), 3)
    split = {'groups': groups, 'X_z': np.zeros((60, 1, 32, 1), np.float32),
             'y_z': np.zeros((60, 1), np.float32)}
    domain = {'train': split, 'val': split}
    small = transfer_subset(domain, 4)['train']['groups']
    large = transfer_subset(domain, 10)['train']['groups']
    assert set(small) <= set(large)
    assert len(small) == 12 and len(large) == 30
    assert set(np.unique(small, return_counts=True)[1]) == {3}


def test_paired_source_mean_uses_one_deterministic_reference_per_target_cycle():
    rows = np.array([10, 20])
    groups = np.array([1, 2])
    target_x = np.array([1., 2.], dtype=np.float32)[:, None, None, None]
    subset = {'train': {'X_z': target_x, 'rows': rows, 'groups': groups}}
    domains = {
        sensor: {'train': {'X_z': target_x + offset, 'rows': rows, 'groups': groups}}
        for sensor, offset in [('a', 1.), ('b', 3.)]
    }
    np.testing.assert_allclose(
        paired_source_mean(domains, ['a', 'b'], subset), target_x + 2.
    )


def test_ds_and_pds_recover_synthetic_sensor_transfer():
    rng = np.random.default_rng(23)
    train = rng.normal(size=(40, 4, 1440, 1)).astype(np.float32)
    test = rng.normal(size=(10, 4, 1440, 1)).astype(np.float32)
    transform = lambda values: 1.7 * values - .3
    ds = DirectStandardization(1e-6).fit(train, transform(train))
    np.testing.assert_allclose(ds.transform(train), transform(train), atol=2e-4)
    pds = PiecewiseStandardization(1e-6, radius=1).fit(train, transform(train))
    np.testing.assert_allclose(pds.transform(test), transform(test), atol=2e-4)

def test_head_only_preserves_backbone_and_batchnorm_buffers():
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 1, 32, 1)).astype(np.float32)
    y = X.mean(axis=(1, 2, 3))[:, None]
    domain = {'train': {'X_z': X[:16], 'y_z': y[:16]}, 'val': {'X_z': X[16:], 'y_z': y[16:]}}
    params = {**DEFAULT_PARAMS, 'n_filter': 4, 'section_depth': 1,
              'num_neurons': 8, 'convs_per_block': 1, 'kernel': 3, 'batch_size': 8}
    source, _, _ = fit_network(domain, params, epochs=2)
    initial = {k: v.detach().cpu().clone() for k, v in source.model.state_dict().items()}
    adapted, _, _ = fit_network(domain, params, epochs=2, initial=initial, head_only=True)
    state = adapted.model.state_dict()
    assert all(torch.equal(value, state[key].cpu()) for key, value in initial.items() if key.startswith('features.'))
    assert any(not torch.equal(value, state[key].cpu()) for key, value in initial.items() if key.startswith('fc'))


def test_day4_training_recalibrates_batchnorm_unless_frozen():
    torch.set_num_threads(2)
    rng = np.random.default_rng(17)
    X = rng.normal(size=(16, 1, 32, 1)).astype(np.float32)
    y = X.mean(axis=(1, 2, 3))[:, None]
    domain = {'train': {'X_z': X[:12], 'y_z': y[:12]},
              'val': {'X_z': X[12:], 'y_z': y[12:]}}
    params = {**DEFAULT_PARAMS, 'n_filter': 4, 'section_depth': 1,
              'num_neurons': 8, 'convs_per_block': 1, 'kernel': 3, 'batch_size': 4}
    _, _, info = fit_network(domain, params, epochs=1)
    assert info['batchnorm_recalibrated_after_epoch'] is True
    _, _, frozen_info = fit_network(domain, params, epochs=1, freeze_batchnorm=True)
    assert frozen_info['batchnorm_recalibrated_after_epoch'] is False


class MeanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Identity()
    def forward(self, x): return self.features(x).mean(dim=(1, 2, 3))[:, None]


class MeanWrapper:
    device = torch.device('cpu')
    model = MeanModel()
    def predict(self, x): return np.mean(x, axis=(1, 2, 3))[:, None]


def test_signed_occlusion_has_correct_units_and_preserves_input():
    x = np.ones((2, 1, 120, 1), np.float32)
    before = x.copy()
    effects, starts = occlusion(MeanWrapper(), x, np.zeros((1, 1, 120, 1), np.float32), width=30, step=30, y_std=10.)
    np.testing.assert_allclose(effects, 2.5)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(starts, [0, 30, 60, 90])


def test_gradients_match_analytic_derivative_and_cam_remains_signed():
    x = -np.ones((2, 1, 120, 1), np.float32)
    gradient, cam = gradient_and_cam(MeanWrapper(), x, y_std=12.)
    np.testing.assert_allclose(gradient, .1, rtol=1e-6)
    assert gradient.shape == cam.shape == (2, 120)
    assert np.all(cam < 0), 'Signed regression Grad-CAM must not discard negative responses.'

if __name__ == '__main__':
    for name, function in sorted(list(globals().items())):
        if name.startswith('test_') and callable(function):
            function()
            print('PASS', name, flush=True)
