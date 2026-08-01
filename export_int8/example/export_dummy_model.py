"""Export the trained public example model to fully integer TFLite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
WORK_DIR = EXAMPLE_DIR / ".work"
ARTIFACT_DIR = ROOT / "artifacts" / "example"
CHECKPOINT = EXAMPLE_DIR / "tiny_diabetes_regressor.pt"
STEM = "tiny_diabetes_regressor"
ONNX_PATH = ARTIFACT_DIR / f"{STEM}.onnx"
TFLITE_PATH = ARTIFACT_DIR / f"{STEM}_int8.tflite"
METADATA_PATH = ARTIFACT_DIR / f"{STEM}_metadata.json"
PREPROCESSING_PATH = ARTIFACT_DIR / f"{STEM}_preprocessing.json"
CPP_PATH = ARTIFACT_DIR / "dummy_model_data.cpp"
HEADER_PATH = ARTIFACT_DIR / "dummy_model_data.h"
CONVERTER_PYTHON = ROOT / "export_int8" / ".venv-tflite" / "Scripts" / "python.exe"
CALIBRATION_PATH = WORK_DIR / "representative_train.npy"
WEIGHTS_PATH = WORK_DIR / "weights.npz"
PROBE_PATH = WORK_DIR / "probe.npy"
REFERENCE_PATH = WORK_DIR / "pytorch_reference.npy"
WORKER_REPORT_PATH = WORK_DIR / "conversion_report.json"


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model():
    import torch
    from dummy_model import TinyDiabetesRegressor

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Missing {CHECKPOINT}. Run train_dummy_model.py first."
        )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = TinyDiabetesRegressor(
        int(checkpoint["input_features"]),
        int(checkpoint["hidden_features"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return checkpoint, model


def export_onnx(model) -> None:
    import onnx
    import torch

    dummy = torch.zeros((1, 10), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(ONNX_PATH),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            export_params=True,
            do_constant_folding=True,
            dynamo=False,
        )
    graph = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(graph, full_check=True)


def conversion_worker() -> None:
    import onnxruntime as ort
    import tensorflow as tf
    from ai_edge_litert.interpreter import Interpreter

    weights = np.load(WEIGHTS_PATH)
    calibration = np.load(CALIBRATION_PATH, mmap_mode="r")
    probe = np.load(PROBE_PATH)
    reference = np.load(REFERENCE_PATH)

    network_input = tf.keras.Input(batch_shape=(1, 10), dtype=tf.float32, name="input")
    dense1 = tf.keras.layers.Dense(16, name="fc1")
    values = dense1(network_input)
    dense1.set_weights([weights["fc1.weight"].T, weights["fc1.bias"]])
    values = tf.keras.layers.ReLU(name="relu")(values)
    dense2 = tf.keras.layers.Dense(1, name="output")
    network_output = dense2(values)
    dense2.set_weights([weights["fc2.weight"].T, weights["fc2.bias"]])
    model = tf.keras.Model(network_input, network_output, name="tiny_diabetes_regressor")

    tensorflow_prediction = np.concatenate(
        [model(probe[index : index + 1], training=False).numpy() for index in range(probe.shape[0])]
    )
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_prediction = np.concatenate(
        [session.run(None, {"input": probe[index : index + 1]})[0] for index in range(probe.shape[0])]
    )
    tf_error = float(np.max(np.abs(tensorflow_prediction - reference)))
    onnx_error = float(np.max(np.abs(onnx_prediction - reference)))
    if tf_error > 1e-4 or onnx_error > 1e-4:
        raise RuntimeError(f"Float parity failed: TensorFlow={tf_error}, ONNX={onnx_error}")

    def representative_dataset():
        for index in range(calibration.shape[0]):
            yield [np.asarray(calibration[index : index + 1], dtype=np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    TFLITE_PATH.write_bytes(converter.convert())

    interpreter = Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise RuntimeError("Example model does not have INT8 input and output.")
    dtype_counts: dict[str, int] = {}
    for detail in interpreter.get_tensor_details():
        dtype = np.dtype(detail["dtype"]).name
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    if any(dtype.startswith("float") for dtype in dtype_counts):
        raise RuntimeError(f"Float tensors remain in TFLite graph: {dtype_counts}")

    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    write_json(
        WORKER_REPORT_PATH,
        {
            "float_parity": {
                "samples": int(probe.shape[0]),
                "tensorflow_max_abs_error": tf_error,
                "onnxruntime_max_abs_error": onnx_error,
                "tolerance": 1e-4,
                "passed": True,
            },
            "input": {
                "shape": input_detail["shape"],
                "dtype": np.dtype(input_detail["dtype"]).name,
                "scale": input_scale,
                "zero_point": input_zero_point,
            },
            "output": {
                "shape": output_detail["shape"],
                "dtype": np.dtype(output_detail["dtype"]).name,
                "scale": output_scale,
                "zero_point": output_zero_point,
            },
            "tensor_dtype_counts": dtype_counts,
            "fully_integer_verified": True,
        },
    )


def write_cpp_array() -> dict[str, Any]:
    payload = TFLITE_PATH.read_bytes()
    HEADER_PATH.write_text(
        "#pragma once\n\n"
        "extern const unsigned char g_dummy_model_data[];\n"
        "extern const unsigned int g_dummy_model_data_len;\n",
        encoding="utf-8",
    )
    rows = []
    for offset in range(0, len(payload), 12):
        rows.append(
            "    " + ", ".join(f"0x{byte:02x}" for byte in payload[offset : offset + 12]) + ","
        )
    CPP_PATH.write_text(
        '#include "dummy_model_data.h"\n\n'
        "alignas(16) const unsigned char g_dummy_model_data[] = {\n"
        + "\n".join(rows)
        + "\n};\n\n"
        "const unsigned int g_dummy_model_data_len =\n"
        "    sizeof(g_dummy_model_data);\n",
        encoding="utf-8",
    )
    encoded_bytes = len(re.findall(r"0x[0-9a-f]{2}", CPP_PATH.read_text(encoding="utf-8")))
    if encoded_bytes != len(payload):
        raise RuntimeError("Generated C++ byte count does not match TFLite file size.")
    return {"array_bytes": encoded_bytes, "file_bytes": len(payload), "exact_match": True}


def main() -> None:
    import torch
    from dummy_model import load_public_data, predict_pytorch

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint, model = load_model()
    data = load_public_data(int(checkpoint["seed"]))
    export_onnx(model)
    np.save(CALIBRATION_PATH, data.train_x, allow_pickle=False)
    np.savez(
        WEIGHTS_PATH,
        **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()},
    )
    probe = data.test_x[:32]
    np.save(PROBE_PATH, probe, allow_pickle=False)
    np.save(REFERENCE_PATH, predict_pytorch(model, probe), allow_pickle=False)
    if not CONVERTER_PYTHON.is_file():
        raise FileNotFoundError(f"Missing converter environment: {CONVERTER_PYTHON}")
    subprocess.run(
        [str(CONVERTER_PYTHON), str(Path(__file__).resolve()), "--worker"],
        cwd=ROOT,
        check=True,
    )
    worker_report = json.loads(WORKER_REPORT_PATH.read_text(encoding="utf-8"))
    cpp_report = write_cpp_array()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    preprocessing = {
        "dataset": "sklearn.datasets.load_diabetes",
        "public_dataset": True,
        "feature_order": data.feature_names,
        "input_shape": [1, 10],
        "operations": [
            {
                "name": "standardize",
                "formula": "(x - training_mean) / training_std",
                "mean": data.mean,
                "std": data.std,
                "ddof": 0,
            }
        ],
        "target_scaled": False,
    }
    metadata = {
        "example_only": True,
        "public_dataset": True,
        "dataset": "sklearn.datasets.load_diabetes",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "retrained_during_export": False,
        "architecture": "Linear(10,16) -> ReLU -> Linear(16,1)",
        "parameters": parameter_count,
        "representative_samples": int(data.train_x.shape[0]),
        "representative_split": "train",
        "onnx": {
            "path": ONNX_PATH.name,
            "opset": 17,
            "size_bytes": ONNX_PATH.stat().st_size,
            "sha256": sha256(ONNX_PATH),
        },
        "tflite_int8": {
            **worker_report,
            "path": TFLITE_PATH.name,
            "size_bytes": TFLITE_PATH.stat().st_size,
            "sha256": sha256(TFLITE_PATH),
            "weights_dtype": "int8",
            "activations_dtype": "int8",
        },
        "cpp_array": cpp_report,
        "training_metrics": checkpoint["metrics"],
        "comparison": None,
    }
    write_json(PREPROCESSING_PATH, preprocessing)
    write_json(METADATA_PATH, metadata)
    print(f"Exported: {TFLITE_PATH.relative_to(ROOT)}")
    print(f"Parameters: {parameter_count}")
    print(f"TFLite size: {TFLITE_PATH.stat().st_size} bytes")
    print("Input quantization:", worker_report["input"])
    print("Output quantization:", worker_report["output"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    arguments = parser.parse_args()
    if arguments.worker:
        conversion_worker()
    else:
        main()
