"""Utilities for the Day 5 water-vs-H2 classification notebook."""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Iterable

import numpy as np

CLASS_NAMES = ("water", "h2")
EXPECTED_CYCLE_SAMPLES = 1440
SEED = 42


def load_cycle_csv(
    path: str | Path,
    *,
    skip_first_cycle: bool = True,
    expected_samples: int | None = EXPECTED_CYCLE_SAMPLES,
) -> np.ndarray:
    """Load a headerless CSV whose rows are complete temperature cycles."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    values = np.loadtxt(
        path, delimiter=",", dtype=np.float32, ndmin=2, encoding="utf-8-sig"
    )
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D cycle table in {path}, got {values.shape}.")
    if expected_samples is not None and values.shape[1] != expected_samples:
        raise ValueError(
            f"{path.name} has {values.shape[1]} values per cycle; "
            f"expected {expected_samples}."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains missing or non-finite values.")
    if skip_first_cycle:
        if values.shape[0] < 2:
            raise ValueError(
                f"{path.name} contains only {values.shape[0]} cycle(s). "
                "At least two are required when the first cycle is ignored."
            )
        values = values[1:]
    return values.astype(np.float32, copy=False)


def load_binary_split(
    data_dir: str | Path,
    split: str,
    *,
    class_names: Iterable[str] = CLASS_NAMES,
    skip_first_cycle: bool = True,
    expected_samples: int | None = EXPECTED_CYCLE_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load split_water.csv and split_h2.csv for TCOCNN."""
    data_dir = Path(data_dir)
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sources: list[str] = []
    names = tuple(class_names)
    if len(names) != 2:
        raise ValueError("This seminar exercise expects exactly two classes.")

    for class_index, class_name in enumerate(names):
        path = data_dir / f"{split}_{class_name}.csv"
        cycles = load_cycle_csv(
            path,
            skip_first_cycle=skip_first_cycle,
            expected_samples=expected_samples,
        )
        matrices.append(cycles)
        labels.append(np.full(cycles.shape[0], class_index, dtype=np.int64))
        sources.extend([path.name] * cycles.shape[0])

    X = np.concatenate(matrices, axis=0)[:, np.newaxis, :, np.newaxis]
    y = np.concatenate(labels, axis=0)
    return X, y, sources


def stratified_train_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    validation_fraction: float = 0.2,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make a reproducible split while keeping both classes in validation."""
    from sklearn.model_selection import train_test_split

    counts = np.bincount(np.asarray(y, dtype=np.int64))
    if counts.size != 2 or np.any(counts < 2):
        raise ValueError(
            "Each class needs at least two usable training cycles for a "
            "stratified train/validation split."
        )
    validation_size = max(2, int(np.ceil(len(y) * validation_fraction)))
    validation_size = min(validation_size, len(y) - 2)
    return train_test_split(
        X, y, test_size=validation_size, random_state=seed, stratify=y
    )


def log_zscore_per_sample(X: np.ndarray) -> np.ndarray:
    """Apply log10 and then a separate Z-score to every complete cycle."""
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 4:
        raise ValueError(
            "Expected TCOCNN input with shape (samples, sensors, time, channels)."
        )
    if np.any(values <= 0):
        raise ValueError(
            "The log10 transformation requires strictly positive sensor values."
        )

    logged = np.log10(values)
    sample_axes = tuple(range(1, logged.ndim))
    mean = logged.mean(axis=sample_axes, keepdims=True, dtype=np.float64)
    std = logged.std(axis=sample_axes, keepdims=True, dtype=np.float64)
    std[std == 0] = 1.0
    return ((logged - mean) / std).astype(np.float32)


def set_seed(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_classifier(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    epochs: int = 40,
    batch_size: int = 4,
) -> tuple[dict[str, list[float]], int]:
    """Train one epoch at a time and restore the best validation checkpoint."""
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train(
            X_train,
            y_train,
            validation_data=(X_validation, y_validation),
            epochs=1,
            batch_size=batch_size,
        )
        for key in history:
            history[key].append(float(model.history.history[key][0]))
        validation_loss = history["val_loss"][-1]
        if not np.isfinite(validation_loss):
            raise FloatingPointError("Non-finite validation loss encountered.")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:>3}/{epochs}: "
                f"loss={history['loss'][-1]:.4f}, "
                f"val_loss={validation_loss:.4f}, "
                f"val_accuracy={history['val_accuracy'][-1]:.3f}"
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.model.load_state_dict(best_state)
    model.history.history = copy.deepcopy(history)
    return history, best_epoch


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

    prediction = np.asarray(probabilities).argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]),
    }
