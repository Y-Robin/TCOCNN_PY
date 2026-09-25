"""Shared data loading, splitting, and scaling helpers for Day 3."""
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SENSOR_GROUP = "sensorA"
N_SUBSENSORS = 4
CYCLE_SAMPLES = 1440
TARGET = "acetone"
SEED = 42


def project_root() -> Path:
    """Find the repository root from either JupyterLab or a script."""
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "Data" / "fullData.mat").exists():
            return candidate
    raise FileNotFoundError("Data/fullData.mat was not found. Run the Day 2 data setup first.")


def _load_official_part(part: str):
    """Load all four sub-sensors and all targets from one official HDF5 split."""
    with h5py.File(project_root() / "Data" / "fullData.mat", "r") as handle:
        references = handle[f"{SENSOR_GROUP}_{part}"]
        channels = [
            np.asarray(handle[references[index, 0]]).T.astype(np.float32)
            for index in range(N_SUBSENSORS)
        ]
        targets_group = handle[f"targets_{part}"]
        targets = {name: np.asarray(targets_group[name]).ravel() for name in targets_group}
    # Shape: samples × parallel sub-sensors × time × one input channel.
    X = np.stack(channels, axis=1)[..., np.newaxis]
    if X.shape[1:] != (N_SUBSENSORS, CYCLE_SAMPLES, 1):
        raise AssertionError(f"Unexpected four-sensor shape: {X.shape}")
    return X, targets


def load_four_sensor_splits(target: str = TARGET):
    """Create leakage-safe train/validation/test splits grouped by UGM ID."""
    X_train, targets_train = _load_official_part("train")
    X_test, targets_test = _load_official_part("test")
    X = np.concatenate([X_train, X_test])
    targets = {name: np.concatenate([targets_train[name], targets_test[name]]) for name in targets_train}
    groups = targets["range"].astype(int)

    # Day 3 trains within the base concentration region; 501+ remains extrapolation data.
    base = np.flatnonzero(groups <= 500)
    extra = np.flatnonzero(groups >= 501)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    development_pos, test_pos = next(outer.split(base, groups=groups[base]))
    development, test = base[development_pos], base[test_pos]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_pos, val_pos = next(inner.split(development, groups=groups[development]))
    indices = {"train": development[train_pos], "val": development[val_pos], "test": test, "test_extra": extra}
    return {
        name: {"X": X[idx], "y": targets[target][idx].astype(np.float32), "groups": groups[idx]}
        for name, idx in indices.items()
    }


def fit_input_zscore(X_train: np.ndarray):
    """Learn one mean/std per sub-sensor from training samples and time points."""
    mean = X_train.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    std = X_train.std(axis=(0, 2), keepdims=True, dtype=np.float64)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((values - mean) / std).astype(np.float32)


def fit_target_zscore(y_train: np.ndarray):
    """Learn optional target scaling; retain both values for inverse transformation."""
    mean, std = float(np.mean(y_train)), float(np.std(y_train))
    return mean, std if std > 0 else 1.0


def prepare_model_data(target: str = TARGET):
    """Load splits and apply train-only input and output Z-score transformations."""
    splits = load_four_sensor_splits(target)
    x_mean, x_std = fit_input_zscore(splits["train"]["X"])
    y_mean, y_std = fit_target_zscore(splits["train"]["y"])
    for split in splits.values():
        split["X_z"] = apply_zscore(split["X"], x_mean, x_std)
        split["y_z"] = ((split["y"] - y_mean) / y_std).astype(np.float32)[:, None]
    return splits, {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def train_experiment(splits, params, epochs=100, seed=42, model_class=None):
    """Train for the declared budget and restore the best validation epoch."""
    import time
    import torch
    from TCOCNN import TCOCNNClass
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_class = TCOCNNClass if model_class is None else model_class
    model = model_class(splits['train']['X_z'].shape[1:], 1, regression=True)
    model.build_net(params)
    model.compile_model(params['initial_learning_rate'])
    # A slower decay keeps useful learning rates throughout the longer run.
    model.lr_scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=30, gamma=0.5)
    history = {'loss': [], 'val_loss': []}
    best_loss, best_state, best_epoch = float('inf'), None, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(splits['train']['X_z'], splits['train']['y_z'],
                    validation_data=(splits['val']['X_z'], splits['val']['y_z']),
                    epochs=1, batch_size=params.get('batch_size', 64))
        for key in history:
            history[key].append(model.history.history[key][0])
        loss = history['val_loss'][-1]
        if not np.isfinite(loss) or not np.isfinite(history['loss'][-1]):
            raise FloatingPointError('Non-finite loss; inspect scaling and learning rate.')
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f'Epoch {epoch}/{epochs}: train={history["loss"][-1]:.4f}, val={loss:.4f}', flush=True)
    model.model.load_state_dict(best_state)
    model.history.history = history
    return model, {'best_epoch': best_epoch, 'epochs': epochs, 'seed': seed,
                   'seconds': time.perf_counter() - started,
                   'parameters': sum(p.numel() for p in model.model.parameters())}


def regression_metrics(truth, prediction):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    truth, prediction = np.asarray(truth).ravel(), np.asarray(prediction).ravel()
    return {'RMSE_ppb': float(np.sqrt(mean_squared_error(truth, prediction))),
            'MAE_ppb': float(mean_absolute_error(truth, prediction)),
            'R2': float(r2_score(truth, prediction)),
            'bias_ppb': float(np.mean(prediction - truth))}


def plot_comparison(truth, predictions, title):
    """Use common limits so prediction and residual panels are comparable."""
    import matplotlib.pyplot as plt
    truth = np.asarray(truth).ravel()
    predictions = {name: np.asarray(p).ravel() for name, p in predictions.items()}
    low = min(truth.min(), *(p.min() for p in predictions.values()))
    high = max(truth.max(), *(p.max() for p in predictions.values()))
    margin = max((high - low) * .04, 1.)
    limits = [low - margin, high + margin]
    residual_limit = max(1., *(float(np.abs(p - truth).max()) for p in predictions.values())) * 1.05
    fig, axes = plt.subplots(2, len(predictions), figsize=(5 * len(predictions), 8), squeeze=False)
    for column, (name, prediction) in enumerate(predictions.items()):
        ax, residual = axes[:, column]
        metrics = regression_metrics(truth, prediction)
        ax.scatter(truth, prediction, alpha=.35, s=14)
        ax.plot(limits, limits, 'k--')
        ax.set(xlim=limits, ylim=limits, xlabel='True concentration [ppb]', ylabel='Predicted concentration [ppb]',
               title=f'{name}\nRMSE={metrics["RMSE_ppb"]:.1f} ppb; R²={metrics["R2"]:.3f}')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=.3)
        residual.scatter(truth, prediction - truth, alpha=.35, s=14)
        residual.axhline(0, color='black', linestyle='--')
        residual.grid(True, alpha=.3)
        residual.set(xlim=limits, ylim=(-residual_limit, residual_limit), xlabel='True concentration [ppb]', ylabel='Prediction − truth [ppb]')
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def show_results(rows):
    """Render a readable metrics table without an extra dataframe dependency."""
    from html import escape
    from IPython.display import HTML, display
    columns = list(dict.fromkeys(key for row in rows for key in row))
    header = ''.join(f'<th>{escape(str(key))}</th>' for key in columns)
    body = ''
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key, '')
            values.append(f'{value:.3f}' if isinstance(value, (float, np.floating)) else str(value))
        body += '<tr>' + ''.join(f'<td>{escape(value)}</td>' for value in values) + '</tr>'
    display(HTML(f'<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'))