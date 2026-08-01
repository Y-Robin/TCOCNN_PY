"""Shared public-data model and deterministic split for the export example."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_diabetes
from torch import nn


SEED = 42


class TinyDiabetesRegressor(nn.Module):
    """Small MLP used only to demonstrate the complete export workflow."""

    def __init__(self, input_features: int = 10, hidden_features: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(values)))


@dataclass(frozen=True)
class PublicData:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray


def set_deterministic_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_public_data(seed: int = SEED) -> PublicData:
    """Load bundled public data and create a deterministic 70/15/15 split."""
    dataset = load_diabetes()
    inputs = np.asarray(dataset.data, dtype=np.float32)
    targets = np.asarray(dataset.target, dtype=np.float32).reshape(-1, 1)
    generator = np.random.default_rng(seed)
    indices = generator.permutation(inputs.shape[0])
    train_end = int(0.70 * indices.size)
    validation_end = int(0.85 * indices.size)
    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]

    mean = inputs[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = inputs[train_indices].std(axis=0, dtype=np.float64).astype(np.float32)
    if np.any(std == 0.0):
        raise ValueError("A training feature has zero standard deviation.")
    normalized = (inputs - mean) / std

    return PublicData(
        train_x=np.ascontiguousarray(normalized[train_indices], dtype=np.float32),
        train_y=np.ascontiguousarray(targets[train_indices], dtype=np.float32),
        val_x=np.ascontiguousarray(normalized[validation_indices], dtype=np.float32),
        val_y=np.ascontiguousarray(targets[validation_indices], dtype=np.float32),
        test_x=np.ascontiguousarray(normalized[test_indices], dtype=np.float32),
        test_y=np.ascontiguousarray(targets[test_indices], dtype=np.float32),
        feature_names=[str(name) for name in dataset.feature_names],
        mean=mean,
        std=std,
    )


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    difference = predicted_values - target_values
    residual = float(np.sum(difference**2))
    total = float(np.sum((target_values - target_values.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(difference))),
        "RMSE": float(np.sqrt(np.mean(difference**2))),
        "R2": float(1.0 - residual / total),
    }


def predict_pytorch(model: nn.Module, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(values)).cpu().numpy()
