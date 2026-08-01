# INT8 model export

This directory contains the general environment definition and model-specific
PyTorch-to-TFLite export workspaces. Real model folders are private by default;
only this documentation and `example/` are intended for Git.

For a complete runnable reference, see `example/README.md`. It trains and
exports a tiny model using only scikit-learn's bundled public diabetes dataset.

## Directory layout

```text
export_int8/
|-- README.md
|-- requirements.txt
|-- .venv-tflite/            # local TensorFlow/LiteRT environment
|-- example/                  # complete runnable public-data workflow
`-- <model-or-partner>/       # private export scripts and temporary files
```

The matching generated files belong in `artifacts/<model-or-partner>/`. Keeping
the scripts and outputs under the same name makes it clear which pipeline
created each deployment package.

## Environments

The project intentionally uses two Python environments:

- `../.venv` contains PyTorch, the data loader, notebooks, and project code.
- `.venv-tflite` contains TensorFlow, ONNX Runtime, and the LiteRT interpreter.

Create the converter environment from the repository root:

```powershell
py -3.11 -m venv export_int8\.venv-tflite
export_int8\.venv-tflite\Scripts\python.exe -m pip install --upgrade pip
export_int8\.venv-tflite\Scripts\python.exe -m pip install -r export_int8\requirements.txt
```

The model-specific export script is started with the main PyTorch environment.
It invokes `.venv-tflite` automatically for conversion and INT8 inference:

```powershell
.venv\Scripts\python.exe export_int8\<model-or-partner>\export_model.py
.venv\Scripts\python.exe export_int8\<model-or-partner>\compare_models.py
```

For VS Code, keep `../.venv/Scripts/python.exe` selected for notebooks and the
main scripts. Select `.venv-tflite/Scripts/python.exe` only when debugging the
TensorFlow worker itself.

## Adding another model

1. Copy `example/export_config.example.json` into a new descriptive subfolder.
2. Add model-specific export and comparison scripts to that subfolder.
3. Write results only to `artifacts/<same-name>/`.
4. Use training data only for representative INT8 calibration.
5. Validate the float conversion against PyTorch before accepting INT8 output.
6. Record preprocessing, tensor shapes, quantization, metrics, and hashes as
   machine-readable metadata.

New real subfolders are ignored automatically. Do not add an exception for a
partner folder to `.gitignore`.
