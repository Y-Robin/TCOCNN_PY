"""Single-channel, sensor-to-sensor transfer and explainability helpers."""
from pathlib import Path
import copy
import json
import numpy as np
from scipy.io import loadmat, whosmat
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from TCOCNNv3 import TCOCNNv3Class
from day3_utils import project_root, fit_input_zscore, apply_zscore, fit_target_zscore

DEFAULT_PARAMS = dict(n_filter=32, section_depth=3, kernel=9, stride=4,
                      num_neurons=128, drop_out=.15, initial_learning_rate=5e-4,
                      convs_per_block=2, channel_growth=16, residual=True, batch_size=64)

GAS_LABELS = {
    'acetone': 'Acetone',
    'co': 'Carbon monoxide',
    'carbon_monoxide': 'Carbon monoxide',
    'ethanol': 'Ethanol',
    'ethylacetate': 'Ethyl acetate',
    'ethyl_acetate': 'Ethyl acetate',
    'formaldehyde': 'Formaldehyde',
    'hydrogen': 'Hydrogen',
    'toluene': 'Toluene',
}

def gas_label(name):
    """Human-readable label while keeping the dataset key explicit."""
    return GAS_LABELS.get(str(name).lower(), str(name).replace('_', ' ').title())


def network_summary_rows(wrapper):
    """Return an auditable layer/parameter table for the built PyTorch network."""
    rows = []
    total = 0
    trainable = 0
    for name, module in wrapper.model.named_modules():
        if name == '':
            continue
        own = sum(p.numel() for p in module.parameters(recurse=False))
        own_trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        if own == 0 and any(True for _ in module.children()):
            continue
        total += own
        trainable += own_trainable
        rows.append({'layer': name, 'type': module.__class__.__name__,
                     'parameters': int(own), 'trainable': int(own_trainable)})
    rows.append({'layer': 'TOTAL', 'type': wrapper.model.__class__.__name__,
                 'parameters': int(total), 'trainable': int(trainable)})
    return rows


def load_sensor_domains(channel=0, calibration=1, gas='acetone'):
    """Load the same channel of seven devices with shared, pre-existing UGM splits.

    Restrict to one calibration so sensor transfer is not confounded by ageing.
    Read only the requested MATLAB variable; the other three channels stay unloaded.
    """
    if channel not in range(4) or calibration not in (1, 2):
        raise ValueError('Choose channel 0..3 and calibration 1 or 2.')
    folder = project_root() / 'Data' / 'Dennis' / 'Prepared'
    target = loadmat(folder / 'dennis_targets_leakage_safe.mat', simplify_cells=True)
    paths = sorted(p for p in folder.glob('dennis_*.mat') if p.stem.split('_')[-1].isdigit())
    if len(paths) != 7:
        raise ValueError('Expected seven prepared sensor files in Data/Dennis/Prepared.')
    if gas not in target:
        available = sorted(k for k in target.keys() if not str(k).startswith('__'))
        raise KeyError(f'Gas target {gas!r} not found. Available targets include: {available}')
    y = np.asarray(target[gas]).ravel()
    group = np.asarray(target['range']).ravel()
    split = np.asarray(target['split_code']).ravel()
    chosen = np.asarray(target['calibration']).ravel() == calibration
    valid = chosen & np.isfinite(y) & np.isfinite(group) & np.isin(split, [1, 2, 3])
    arrays = {}
    for path in paths:
        keys = [name for name, shape, kind in whosmat(path) if name.endswith(f'_sensor{channel}')]
        if len(keys) != 1:
            raise ValueError('Single-channel variable could not be resolved.')
        values = np.asarray(loadmat(path, variable_names=keys)[keys[0]], dtype=np.float32)
        if values.shape != (len(y), 1440):
            raise ValueError(f'Unexpected shape for sensor {path.stem.split("_")[-1]}.')
        valid &= np.isfinite(values).all(axis=1)
        arrays[path.stem.split('_')[-1]] = values
    masks = {name: valid & (split == code) for name, code in [('train', 1), ('val', 2), ('test', 3)]}
    groups = {name: set(group[mask]) for name, mask in masks.items()}
    if any(groups[a] & groups[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise AssertionError('A UGM occurs in multiple splits.')
    if any(not mask.any() for mask in masks.values()):
        raise ValueError('An empty split remains after the finite-data check.')
    domains = {sensor: {name: {'X': values[mask, None, :, None].copy(),
                              'y': y[mask].astype(np.float32),
                              'groups': group[mask].astype(int), 'rows': np.flatnonzero(mask)}
                        for name, mask in masks.items()} for sensor, values in arrays.items()}
    audit = dict(gas=gas, gas_label=gas_label(gas), channel=channel, calibration=calibration, sensors=list(domains),
                 discarded_rows_in_calibration=int(chosen.sum() - valid.sum()),
                 rows={name: int(mask.sum()) for name, mask in masks.items()},
                 groups={name: len(value) for name, value in groups.items()})
    return domains, audit


def source_scaler(source):
    mean, std = fit_input_zscore(source['train']['X'])
    y_mean, y_std = fit_target_zscore(source['train']['y'])
    return dict(x_mean=mean, x_std=std, y_mean=y_mean, y_std=y_std)


def scale_domain(domain, scaler):
    return {name: {**split, 'X_z': apply_zscore(split['X'], scaler['x_mean'], scaler['x_std']),
                   'y_z': ((split['y'] - scaler['y_mean']) / scaler['y_std'])[:, None].astype(np.float32)}
            for name, split in domain.items()}


def transfer_subset(domain, n_groups, seed=42):
    """Nested random UGM subsets; every cycle belonging to the selected UGMs is retained."""
    order = np.random.default_rng(seed).permutation(np.unique(domain['train']['groups']))
    if not 1 <= n_groups <= len(order):
        raise ValueError('Transfer budget exceeds the available training UGMs.')
    indices = np.flatnonzero(np.isin(domain['train']['groups'], order[:n_groups]))
    subset = {**domain, 'train': {key: value[indices] for key, value in domain['train'].items()}}
    if len(np.unique(subset['train']['groups'])) != n_groups:
        raise AssertionError('Transfer subset does not contain the requested number of UGMs.')
    return subset


def _set_batchnorm_eval(model):
    """Freeze running mean/variance while leaving affine BN parameters trainable."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def fit_network(domain, params=None, epochs=60, initial=None, head_only=False,
                freeze_batchnorm=False, seed=42, restore_best=True,
                use_validation=True, lr_schedule='step', verbose=True):
    """Train a TCOCNN under explicit model-selection boundaries.

    Parameters
    ----------
    use_validation:
        If False, the validation split is never read.  This is used for the final
        seventh-sensor adaptation after LR and epoch have already been selected on
        the six development sensors.
    restore_best:
        If True, restore the best *UGM-mean validation* checkpoint after all epochs.
        This is used for source-model development only, never for final target
        Scratch/Head-only/Fine-tune adaptation.
    lr_schedule:
        ``'step'`` keeps the source-model StepLR schedule. ``'constant'`` is used in
        transfer LR searches so a candidate learning rate means exactly that rate.
    freeze_batchnorm:
        Keep running statistics fixed.  Recommended for low-data fine-tuning and
        automatically implied by head-only adaptation of the frozen feature stack.
    """
    params = dict(DEFAULT_PARAMS if params is None else params)
    if restore_best and not use_validation:
        raise ValueError('restore_best=True requires validation data.')
    if lr_schedule not in {'step', 'constant'}:
        raise ValueError("lr_schedule must be 'step' or 'constant'.")
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    wrapper = TCOCNNv3Class(domain['train']['X_z'].shape[1:], 1, regression=True)
    wrapper.build_net(params)
    model = wrapper.model
    if initial is not None:
        model.load_state_dict(initial)
    if head_only:
        if initial is None:
            raise ValueError('Head-only adaptation requires pretrained weights.')
        for p in model.features.parameters():
            p.requires_grad_(False)
        freeze_batchnorm = True
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=params['initial_learning_rate'], weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=.5)
                 if lr_schedule == 'step' else None)

    def tensors(split):
        return (torch.from_numpy(split['X_z'].transpose(0, 3, 1, 2).copy()),
                torch.from_numpy(split['y_z']))

    split_names = ['train', 'val'] if use_validation else ['train']
    loaders = {name: DataLoader(TensorDataset(*tensors(domain[name])),
                                batch_size=params.get('batch_size', 64),
                                shuffle=(name == 'train')) for name in split_names}
    history = {'loss': [], 'learning_rate': []}
    if use_validation:
        history.update({'val_loss': [], 'val_group_mse': []})
    best, best_epoch, best_state = np.inf, 0, None

    for epoch in range(1, epochs + 1):
        history['learning_rate'].append(float(optimizer.param_groups[0]['lr']))
        val_predictions = []
        for name in split_names:
            model.train(name == 'train')
            if head_only:
                model.features.eval()
            if name == 'train' and freeze_batchnorm:
                _set_batchnorm_eval(model)
            total = 0.
            with torch.set_grad_enabled(name == 'train'):
                for x, y in loaders[name]:
                    x, y = x.to(wrapper.device), y.to(wrapper.device)
                    if name == 'train':
                        optimizer.zero_grad(set_to_none=True)
                    output = model(x)
                    loss = nn.functional.mse_loss(output, y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss.')
                    if name == 'train':
                        loss.backward(); optimizer.step()
                    total += loss.item() * len(x)
                    if name == 'val':
                        val_predictions.append(output.detach().cpu().numpy().ravel())
            history['loss' if name == 'train' else 'val_loss'].append(total / len(loaders[name].dataset))
            if (
                name == 'train'
                and not freeze_batchnorm
                and wrapper.recalibrate_batchnorm_after_epoch
            ):
                wrapper._recalibrate_batchnorm(
                    domain['train']['X_z'], params.get('batch_size', 64)
                )

        if use_validation:
            val_pred = np.concatenate(val_predictions)
            val_truth = domain['val']['y_z'].ravel()
            val_groups = domain['val'].get('eval_groups', domain['val'].get('groups', np.arange(len(val_truth))))
            _, inverse = np.unique(val_groups, return_inverse=True)
            counts = np.bincount(inverse)
            group_errors = np.bincount(inverse, weights=val_pred - val_truth) / counts
            group_mse = float(np.mean(group_errors ** 2))
            history['val_group_mse'].append(group_mse)
            if group_mse < best:
                best, best_epoch = group_mse, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if scheduler is not None:
            scheduler.step()
        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            lr_now = history['learning_rate'][-1]
            if use_validation:
                print(f'Epoch {epoch}/{epochs}: train_MSE(z)={history["loss"][-1]:.4f}, '
                      f'val_UGM_RMSE(z)={np.sqrt(history["val_group_mse"][-1]):.4f}, lr={lr_now:.2e}', flush=True)
            else:
                print(f'Epoch {epoch}/{epochs}: train_MSE(z)={history["loss"][-1]:.4f}, lr={lr_now:.2e} '
                      '[target validation not read]', flush=True)

    if restore_best:
        model.load_state_dict(best_state)
        state = best_state
    else:
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epochs
    model.eval()
    if head_only:
        for key, value in state.items():
            if key.startswith('features.') and not torch.equal(value, initial[key].cpu()):
                raise AssertionError('Frozen backbone parameters or BatchNorm buffers changed.')

    return wrapper, history, {
        'best_epoch': int(best_epoch),
        'restored_best': bool(restore_best),
        'used_validation': bool(use_validation),
        'selection_metric': ('UGM-mean validation MSE' if restore_best else
                             'fixed development-selected epoch; target validation not read' if not use_validation else
                             'fixed epoch scored on development validation'),
        'optimizer': 'AdamW',
        'weight_decay': 1e-4,
        'scheduler': 'StepLR(step_size=30, gamma=0.5)' if scheduler is not None else 'constant learning rate',
        'initial_learning_rate': params['initial_learning_rate'],
        'batch_size': params.get('batch_size', 64),
        'batchnorm_running_stats_frozen': bool(freeze_batchnorm),
        'batchnorm_recalibrated_after_epoch': bool(
            wrapper.recalibrate_batchnorm_after_epoch and not freeze_batchnorm
        ),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def predict_normalized(wrapper, X_z, batch_size=256):
    """Canonical inference path used by transfer evaluation and XAI."""
    array = np.asarray(X_z, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f'Expected X_z with four dimensions, got {array.shape}.')
    tensor = torch.from_numpy(array.transpose(0, 3, 1, 2).copy())
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    model = wrapper.model
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for (x,) in loader:
            out.append(model(x.to(wrapper.device)).detach().cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(out, axis=0)


def predict_ppb(wrapper, split, scaler):
    return predict_normalized(wrapper, split['X_z']).ravel() * scaler['y_std'] + scaler['y_mean']


def gradient_and_cam(wrapper, X_z, y_std=1.):
    """Return input gradients and signed regression Grad-CAM per time sample.

    Both results are expressed in physical output units. Unlike classification
    Grad-CAM, the map deliberately keeps its sign: negative values are meaningful
    for a regressor and must not be discarded by a ReLU.
    """
    array = np.asarray(X_z, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f'Expected X_z with four dimensions, got {array.shape}.')
    if not hasattr(wrapper.model, 'features'):
        raise ValueError('Grad-CAM requires a model with a features module.')

    tensor = torch.from_numpy(array.transpose(0, 3, 1, 2).copy()).to(wrapper.device)
    tensor.requires_grad_(True)
    model = wrapper.model
    was_training = model.training
    model.eval()
    captured = {}

    def capture_features(_module, _inputs, output):
        captured['activation'] = output
        output.retain_grad()

    handle = model.features.register_forward_hook(capture_features)
    try:
        model.zero_grad(set_to_none=True)
        output = model(tensor)
        output.sum().backward()
        activation = captured.get('activation')
        if activation is None or activation.grad is None or tensor.grad is None:
            raise RuntimeError('Could not obtain gradients from the feature backbone.')

        scale = float(np.asarray(y_std).reshape(()))
        input_gradient = (tensor.grad * scale).mean(dim=(1, 2))
        feature_gradient = activation.grad * scale
        weights = feature_gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = nn.functional.interpolate(
            cam, size=tensor.shape[2:], mode='bilinear', align_corners=False
        ).mean(dim=(1, 2))
        return (input_gradient.detach().cpu().numpy(),
                cam.detach().cpu().numpy())
    finally:
        handle.remove()
        model.train(was_training)


def save_checkpoint(folder, wrapper, scaler, metadata, reference_cycle):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in wrapper.model.state_dict().items()}, folder / 'weights.pt')
    preprocessing = {key: value for key, value in scaler.items() if key != 'reference_cycle'}
    np.savez(folder / 'preprocessing.npz', **preprocessing, reference_cycle=reference_cycle)
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def load_checkpoint(folder):
    folder = Path(folder)
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    with np.load(folder / 'preprocessing.npz') as archive:
        scaler = dict(archive)
    wrapper = TCOCNNv3Class(tuple(metadata.get('input_shape', (1, 1440, 1))), 1, regression=True)
    wrapper.build_net(metadata['params'])
    wrapper.model.load_state_dict(torch.load(folder / 'weights.pt', map_location=wrapper.device, weights_only=True))
    wrapper.model.eval()
    return wrapper, scaler, metadata


def occlusion(wrapper, X_z, reference, width=60, step=60, y_std=1.):
    """Signed output change for each replaced window, in physical output units."""
    X_z = np.asarray(X_z, dtype=np.float32)
    starts = list(range(0, X_z.shape[2] - width + 1, step))
    if not starts or starts[-1] + width != X_z.shape[2]:
        raise ValueError('Choose a window/step that covers the complete cycle.')
    original = predict_normalized(wrapper, X_z).ravel()
    changes = []
    for start in starts:
        altered = X_z.copy()
        altered[:, :, start:start + width] = reference[:, :, start:start + width]
        changes.append((original - predict_normalized(wrapper, altered).ravel()) * y_std)
    return np.stack(changes, axis=1), np.asarray(starts)
