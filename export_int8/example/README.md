# Runnable public INT8 export example

This directory contains a complete, small, commit-safe example. It trains a
193-parameter PyTorch regressor on scikit-learn's public diabetes dataset,
exports it to ONNX, converts it to a fully integer TFLite model, compares INT8
with PyTorch, creates a plot, and packages the model as a C++ byte array.

The dataset is bundled with scikit-learn, so the example requires no partner
files, credentials, or network download. It contains 442 observations with ten
numeric baseline features and a continuous disease-progression target.

## Model

```text
10 inputs -> Linear(10, 16) -> ReLU -> Linear(16, 1)
```

- deterministic seed: `42`
- split: 70% train, 15% validation, 15% test
- preprocessing: feature-wise mean/std fitted on the training split only
- target scaling: none
- representative INT8 calibration: all real training-split inputs

## Run the complete example

From the repository root:

```powershell
.venv\Scripts\python.exe export_int8\example\train_dummy_model.py
.venv\Scripts\python.exe export_int8\example\export_dummy_model.py
.venv\Scripts\python.exe export_int8\example\compare_dummy_models.py
```

The first command deliberately trains the public dummy model. The export command
only restores that checkpoint and does not retrain it. TensorFlow and LiteRT are
invoked automatically through `export_int8/.venv-tflite`.

## Outputs

Training outputs remain here:

- `tiny_diabetes_regressor.pt`
- `tiny_diabetes_training.json`

Deployment outputs are written to `artifacts/example/`:

- float ONNX and fully INT8 TFLite models;
- preprocessing and metadata JSON;
- PyTorch/INT8 comparison CSV and PNG;
- verified `dummy_model_data.cpp` and `dummy_model_data.h`.

Unlike real model folders, this example and its generated artifacts are allowed
in Git because every input comes from the bundled public dataset.
