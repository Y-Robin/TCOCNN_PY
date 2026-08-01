"""Train a tiny model on scikit-learn's bundled public diabetes dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from dummy_model import (
    SEED,
    TinyDiabetesRegressor,
    load_public_data,
    predict_pytorch,
    regression_metrics,
    set_deterministic_seed,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
CHECKPOINT = EXAMPLE_DIR / "tiny_diabetes_regressor.pt"
TRAINING_REPORT = EXAMPLE_DIR / "tiny_diabetes_training.json"


def main() -> None:
    set_deterministic_seed()
    data = load_public_data()
    model = TinyDiabetesRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    train_x = torch.from_numpy(data.train_x)
    train_y = torch.from_numpy(data.train_y)
    val_x = torch.from_numpy(data.val_x)
    val_y = torch.from_numpy(data.val_y)
    history: list[dict[str, float | int]] = []

    for epoch in range(600):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_x)
        loss = criterion(prediction, train_y)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == 599:
            model.eval()
            with torch.inference_mode():
                validation_loss = criterion(model(val_x), val_y)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_mse": float(loss.item()),
                    "validation_mse": float(validation_loss.item()),
                }
            )

    metrics = {
        split: regression_metrics(target, predict_pytorch(model, values))
        for split, values, target in (
            ("train", data.train_x, data.train_y),
            ("val", data.val_x, data.val_y),
            ("test", data.test_x, data.test_y),
        )
    }
    checkpoint = {
        "format": "public_example_state_dict_bundle",
        "state_dict": model.state_dict(),
        "model_class": "TinyDiabetesRegressor",
        "input_features": 10,
        "hidden_features": 16,
        "output_features": 1,
        "seed": SEED,
        "epochs": 600,
        "feature_names": data.feature_names,
        "preprocessing_mean": data.mean,
        "preprocessing_std": data.std,
        "target_scaled": False,
        "metrics": metrics,
    }
    torch.save(checkpoint, CHECKPOINT)
    report = {
        "dataset": "sklearn.datasets.load_diabetes",
        "public_dataset": True,
        "distribution": "bundled with scikit-learn; no network download",
        "samples": {
            "train": int(data.train_x.shape[0]),
            "val": int(data.val_x.shape[0]),
            "test": int(data.test_x.shape[0]),
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "target_scaled": False,
        "metrics": metrics,
        "history": history,
    }
    TRAINING_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint: {CHECKPOINT.relative_to(ROOT)}")
    print(f"Parameters: {report['parameters']}")
    print("Test metrics:", metrics["test"])


if __name__ == "__main__":
    main()
