"""DAV3E-inspired FESR baseline for exactly one sensor channel."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

from dataset_pipeline import CYCLE_SAMPLES, SAMPLE_RATE_HZ, SPLIT_NAMES

N_SEGMENTS = 120
FEATURE_COUNTS = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 120, 180, 240)
PLS_COMPONENTS = (1, 2, 3, 4, 6, 8, 12, 16, 20)


@dataclass
class FesrResult:
    selected_indices: np.ndarray
    selected_names: list[str]
    ranking: np.ndarray
    n_features: int
    n_components: int
    validation_rmse: float
    predictions: dict[str, np.ndarray]
    model: PLSRegression
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    selection_method: str


def extract_fesr_features(
    X: np.ndarray,
    n_segments: int = N_SEGMENTS,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, list[str]]:
    """Equidistant segment means and slopes.

    With 1,440 samples and 120 segments, each segment contains 12 samples.
    Bei einem Kanal entstehen 120 * 2 = 240 Merkmale.
    """
    values = np.asarray(X, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (1, CYCLE_SAMPLES):
        raise ValueError(
            f"Expected (n, 1, {CYCLE_SAMPLES}), received: {values.shape}"
        )
    if CYCLE_SAMPLES % n_segments:
        raise ValueError("The segment count must divide the cycle without a remainder")

    raw = values[:, 0, :]
    segment_width = CYCLE_SAMPLES // n_segments
    columns: list[np.ndarray] = []
    names: list[str] = []
    for segment in range(n_segments):
        start = segment * segment_width
        stop = start + segment_width
        block = raw[:, start:stop]
        t = np.arange(segment_width, dtype=float) / sample_rate_hz
        centered = t - t.mean()
        slope = (block @ centered) / np.sum(centered**2)
        columns.extend((block.mean(axis=1), slope))
        names.extend(
            (
                f"segment{segment:03d}_{start:04d}-{stop - 1:04d}__mean",
                f"segment{segment:03d}_{start:04d}-{stop - 1:04d}__slope_per_s",
            )
        )
    return np.column_stack(columns), names


def standardize_from_train(
    features: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    mean = features["train"].mean(axis=0)
    scale = features["train"].std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = {
        split: (np.asarray(features[split], dtype=float) - mean) / scale
        for split in SPLIT_NAMES
    }
    return standardized, mean, scale


def rfe_lsr_ranking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """RFE-LSR: wiederholt kleinsten absoluten Ridge-Koeffizienten entfernen."""
    selector = RFE(
        estimator=Ridge(alpha=ridge_alpha),
        n_features_to_select=1,
        step=1,
    )
    selector.fit(X_train, y_train)
    return np.argsort(selector.ranking_, kind="stable")


def pearson_ranking(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Absteigende Rangfolge nach absoluter Pearson-Korrelation."""
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    y_centered = y - y.mean()
    y_norm = np.sqrt(np.sum(y_centered**2))
    X_centered = X - X.mean(axis=0)
    denominator = np.sqrt(np.sum(X_centered**2, axis=0)) * y_norm
    correlation = np.divide(
        X_centered.T @ y_centered,
        denominator,
        out=np.zeros(X.shape[1], dtype=float),
        where=denominator > 1e-12,
    )
    return np.argsort(-np.abs(correlation), kind="stable")


def feature_ranking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selection_method: str,
) -> np.ndarray:
    if selection_method == "rfe_lsr":
        return rfe_lsr_ranking(X_train, y_train)
    if selection_method == "pearson":
        return pearson_ranking(X_train, y_train)
    raise ValueError(f"Unknown feature selection: {selection_method}")


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    residual = truth - np.asarray(prediction, dtype=float)
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "R2": 1.0 - float(np.sum(residual**2)) / denominator,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(residual))),
    }


def fit_fesr(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    feature_names: list[str],
    feature_counts: tuple[int, ...] = FEATURE_COUNTS,
    component_counts: tuple[int, ...] = PLS_COMPONENTS,
    selection_method: str = "rfe_lsr",
) -> FesrResult:
    """Rank on training, tune on validation, and leave final tests untouched."""
    scaled, mean, scale = standardize_from_train(features)
    y_train = np.asarray(targets["train"], dtype=float)
    y_val = np.asarray(targets["val"], dtype=float)
    ranking = feature_ranking(
        scaled["train"], y_train, selection_method=selection_method
    )

    best: tuple[float, int, int] | None = None
    for n_features in feature_counts:
        if n_features > scaled["train"].shape[1]:
            continue
        selected = ranking[:n_features]
        for n_components in component_counts:
            if n_components > n_features:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = PLSRegression(
                    n_components=n_components,
                    scale=False,
                    max_iter=1000,
                    tol=1e-7,
                )
                model.fit(scaled["train"][:, selected], y_train)
            prediction = model.predict(scaled["val"][:, selected]).reshape(-1)
            rmse = regression_metrics(y_val, prediction)["RMSE"]
            candidate = (rmse, n_features, n_components)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        raise RuntimeError("No valid FESR configuration found")
    validation_rmse, n_features, n_components = best
    selected = ranking[:n_features]
    model = PLSRegression(
        n_components=n_components,
        scale=False,
        max_iter=1000,
        tol=1e-7,
    )
    model.fit(scaled["train"][:, selected], y_train)
    predictions = {
        split: model.predict(scaled[split][:, selected]).reshape(-1)
        for split in SPLIT_NAMES
    }
    return FesrResult(
        selected_indices=selected,
        selected_names=[feature_names[index] for index in selected],
        ranking=ranking,
        n_features=n_features,
        n_components=n_components,
        validation_rmse=validation_rmse,
        predictions=predictions,
        model=model,
        feature_mean=mean,
        feature_scale=scale,
        selection_method=selection_method,
    )
