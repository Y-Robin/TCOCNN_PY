"""Train and evaluate the acetone model for one of sensors A, B, or C."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "PreProcessing"))
sys.path.insert(0, str(REPOSITORY_ROOT / "Networks"))

import loadDataFull  # noqa: E402
from TCOCNN import TCOCNNClass  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate acetone regression and create a test scatterplot."
    )
    parser.add_argument(
        "--sensor",
        choices=("A", "B", "C", "a", "b", "c"),
        default="A",
        help="Sensor to evaluate (default: A).",
    )
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/acetone_sensor_<letter>).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_acetone_data(sensor: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data_dir = REPOSITORY_ROOT / "Data"
    settings = {
        "fileNameDataAll": [str(data_dir / f"sensor{sensor}.mat")],
        "fileNameTargetAll": [str(data_dir / "targets.mat")],
        "targetGas": "acetone",
        "loadMethod": 1,
        "dataSize": [4, 1440],
        "Outputsize": 1,
        "Regression": True,
        "saveFlag": False,
        "randomFlag": False,
        "normFlag": True,
        "OcclusionFlag": False,
        "rng_val": 0,
    }
    return loadDataFull.load_Data_Full(settings)


def save_predictions(path: Path, actual: np.ndarray, predicted: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample", "actual_acetone", "predicted_acetone", "residual"))
        for index, (actual_value, predicted_value) in enumerate(
            zip(actual, predicted, strict=True)
        ):
            writer.writerow(
                (
                    index,
                    float(actual_value),
                    float(predicted_value),
                    float(predicted_value - actual_value),
                )
            )


def save_scatterplot(
    path: Path,
    sensor: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics: dict[str, float | int | str],
) -> None:
    limit_min = float(min(actual.min(), predicted.min()))
    limit_max = float(max(actual.max(), predicted.max()))
    padding = max((limit_max - limit_min) * 0.05, 0.01)
    limits = (limit_min - padding, limit_max + padding)

    fig, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(actual, predicted, alpha=0.65, s=28, edgecolors="none")
    axis.plot(limits, limits, color="crimson", linewidth=1.5, label="Ideale Vorhersage")
    axis.set(
        xlabel="Tatsächliche Acetonkonzentration",
        ylabel="Vorhergesagte Acetonkonzentration",
        title=f"Aceton – Sensor {sensor} – Testdaten",
        xlim=limits,
        ylim=limits,
    )
    axis.text(
        0.03,
        0.97,
        f"n = {metrics['test_samples']}\nRMSE = {metrics['rmse']:.4f}\n"
        f"MAE = {metrics['mae']:.4f}\nR² = {metrics['r2']:.4f}",
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sensor = args.sensor.upper()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    set_seed(args.seed)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPOSITORY_ROOT / "output" / f"acetone_sensor_{sensor.lower()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data, target = load_acetone_data(sensor)
    with (REPOSITORY_ROOT / "Evaluation" / "acetoneParams.json").open(
        encoding="utf-8"
    ) as handle:
        parameters = json.load(handle)

    model = TCOCNNClass((4, 1440, 1), 1, regression=True)
    model.build_net(parameters)
    model.compile_model(parameters["initial_learning_rate"])
    model.train(
        data["train"],
        target["train"],
        validation_data=(data["val"], target["val"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    actual = np.asarray(target["test"], dtype=np.float64).reshape(-1)
    predicted = np.asarray(model.predict(data["test"]), dtype=np.float64).reshape(-1)
    metrics: dict[str, float | int | str] = {
        "gas": "acetone",
        "sensor": sensor,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_samples": int(target["train"].shape[0]),
        "validation_samples": int(target["val"].shape[0]),
        "test_samples": int(actual.shape[0]),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }

    metrics_path = output_dir / "test_metrics.json"
    predictions_path = output_dir / "test_predictions.csv"
    plot_path = output_dir / "test_scatter.png"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_predictions(predictions_path, actual, predicted)
    save_scatterplot(plot_path, sensor, actual, predicted, metrics)

    print(json.dumps(metrics, indent=2))
    print(f"Scatterplot: {plot_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
