"""Compare public-example PyTorch and fully INT8 TFLite predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
WORK_DIR = EXAMPLE_DIR / ".work"
ARTIFACT_DIR = ROOT / "artifacts" / "example"
CHECKPOINT = EXAMPLE_DIR / "tiny_diabetes_regressor.pt"
TFLITE_PATH = ARTIFACT_DIR / "tiny_diabetes_regressor_int8.tflite"
METADATA_PATH = ARTIFACT_DIR / "tiny_diabetes_regressor_metadata.json"
PLOT_PATH = ARTIFACT_DIR / "tiny_diabetes_model_comparison.png"
PREDICTIONS_PATH = ARTIFACT_DIR / "tiny_diabetes_test_predictions.csv"
CONVERTER_PYTHON = ROOT / "export_int8" / ".venv-tflite" / "Scripts" / "python.exe"


def tflite_worker(input_path: Path, output_path: Path) -> None:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    values = np.load(input_path)
    interpreter = Interpreter(
        model_path=str(TFLITE_PATH),
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    predictions = np.empty((values.shape[0], 1), dtype=np.float32)
    for index in range(values.shape[0]):
        quantized = np.clip(
            np.rint(values[index : index + 1] / input_scale + input_zero_point),
            -128,
            127,
        ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        predictions[index] = (output.astype(np.float32) - output_zero_point) * output_scale
    np.save(output_path, predictions, allow_pickle=False)


def create_plot(target: np.ndarray, pytorch: np.ndarray, int8: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = target.reshape(-1)
    pytorch = pytorch.reshape(-1)
    int8 = int8.reshape(-1)
    limits = (
        float(min(target.min(), pytorch.min(), int8.min())),
        float(max(target.max(), pytorch.max(), int8.max())),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    index = np.arange(target.size)
    axes[0].plot(index, target, label="Target", color="black", linewidth=1.4)
    axes[0].plot(index, pytorch, label="PyTorch", alpha=0.8)
    axes[0].plot(index, int8, label="TFLite INT8", alpha=0.8)
    axes[0].set(title="Public test split", xlabel="Test sample", ylabel="Disease progression")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].scatter(target, pytorch, label="PyTorch", alpha=0.7)
    axes[1].scatter(target, int8, label="TFLite INT8", alpha=0.7)
    axes[1].plot(limits, limits, "k--", linewidth=1, label="Ideal")
    axes[1].set(
        title="Prediction versus target",
        xlabel="Target",
        ylabel="Prediction",
        xlim=limits,
        ylim=limits,
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].scatter(pytorch, int8, alpha=0.75)
    axes[2].plot(limits, limits, "k--", linewidth=1)
    axes[2].set(
        title="INT8 agreement",
        xlabel="PyTorch prediction",
        ylabel="INT8 prediction",
        xlim=limits,
        ylim=limits,
    )
    axes[2].grid(alpha=0.25)
    figure.savefig(PLOT_PATH, dpi=170)
    plt.close(figure)


def main() -> None:
    import torch

    from dummy_model import (
        TinyDiabetesRegressor,
        load_public_data,
        predict_pytorch,
        regression_metrics,
    )

    if not TFLITE_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("Run export_dummy_model.py before comparison.")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = TinyDiabetesRegressor(
        int(checkpoint["input_features"]), int(checkpoint["hidden_features"])
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    data = load_public_data(int(checkpoint["seed"]))
    pytorch_prediction = predict_pytorch(model, data.test_x)
    input_path = WORK_DIR / "comparison_test_input.npy"
    output_path = WORK_DIR / "comparison_test_int8.npy"
    np.save(input_path, data.test_x, allow_pickle=False)
    subprocess.run(
        [
            str(CONVERTER_PYTHON),
            str(Path(__file__).resolve()),
            "--worker",
            str(input_path),
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
    )
    int8_prediction = np.load(output_path)
    comparison = {
        "samples": int(data.test_x.shape[0]),
        "pytorch": regression_metrics(data.test_y, pytorch_prediction),
        "tflite_int8": regression_metrics(data.test_y, int8_prediction),
        "int8_vs_pytorch": regression_metrics(pytorch_prediction, int8_prediction),
        "prediction_correlation": float(
            np.corrcoef(pytorch_prediction.reshape(-1), int8_prediction.reshape(-1))[0, 1]
        ),
        "runtime": "ai_edge_litert BUILTIN_REF, delegates disabled",
    }
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata["comparison"] = comparison
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table = np.column_stack(
        [data.test_y.reshape(-1), pytorch_prediction.reshape(-1), int8_prediction.reshape(-1)]
    )
    np.savetxt(
        PREDICTIONS_PATH,
        table,
        delimiter=",",
        header="target,pytorch,tflite_int8",
        comments="",
    )
    create_plot(data.test_y, pytorch_prediction, int8_prediction)
    print("PyTorch:", comparison["pytorch"])
    print("TFLite INT8:", comparison["tflite_int8"])
    print("INT8 versus PyTorch:", comparison["int8_vs_pytorch"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, metavar=("INPUT", "OUTPUT"))
    arguments = parser.parse_args()
    if arguments.worker:
        tflite_worker(*(Path(item).resolve() for item in arguments.worker))
    else:
        main()
