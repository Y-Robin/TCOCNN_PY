# TCOCNN

TCOCNN is a convolutional neural-network toolbox for analysing metal-oxide
semiconductor (MOS) gas-sensor data. The Python implementation uses PyTorch
while retaining the NumPy- and notebook-facing API of the original project.

## Repository structure

```text
TCOCNN_PY/
|-- Data/                    # local/private datasets and prepared MATLAB files
|-- Evaluation/              # public examples plus private evaluation folders
|-- Networks/                # PyTorch TCOCNN implementations
|-- PreProcessing/           # loading, splitting, and normalization
|-- export_int8/             # INT8 export tooling and private model workspaces
|-- artifacts/               # grouped deployment packages
|-- tests/                   # compatibility and loader tests
`-- requirements.txt         # main PyTorch/notebook environment
```

Real datasets, partner evaluations, checkpoints, model-specific export scripts,
deployment models, metrics, and plots are local-only. The runnable examples
under `export_int8/example/` and `artifacts/example/` use only scikit-learn's
bundled public diabetes dataset and expose no partner values.

## Main installation

Python 3.10 or newer is required. From the repository root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start JupyterLab from the same environment:

```powershell
python -m jupyter lab
```

On Windows, `requirements.txt` installs the CUDA 13.0 build of PyTorch. The
network uses CUDA when `torch.cuda.is_available()` is true and otherwise uses
the CPU. Verify the active installation with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

The CUDA 13.0 setup was verified with an NVIDIA RTX 3080. Other platforms
should use a suitable PyTorch package for their operating system and hardware.

## Notebook compatibility

Notebooks continue to pass NumPy arrays in channels-last form:

```text
(samples, height, width, channels)
```

The compatibility layer converts inputs internally to PyTorch's channels-first
format, manages device placement and batching, and returns NumPy predictions.
The established `TCOCNNClass` API remains available, including:

- `build_net`, `compile_model`, `train`, and `predict`;
- `copy`, `retrain`, and `optimize_model`;
- `custom_occlusion` and `get_gradient_map`.

Run the tests from the repository root:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## INT8/TFLite deployment export

Deployment conversion has a separate environment so TensorFlow's dependency
constraints do not modify the PyTorch notebook environment:

```powershell
py -3.11 -m venv export_int8\.venv-tflite
export_int8\.venv-tflite\Scripts\python.exe -m pip install --upgrade pip
export_int8\.venv-tflite\Scripts\python.exe -m pip install -r export_int8\requirements.txt
```

Model-specific scripts live in private folders such as
`export_int8/<model-or-partner>/`; their outputs go to the matching
`artifacts/<model-or-partner>/` folder. Start export and comparison with the
main PyTorch environment. The scripts call `.venv-tflite` themselves when
TensorFlow or LiteRT is required:

```powershell
.venv\Scripts\python.exe export_int8\<model-or-partner>\export_model.py
.venv\Scripts\python.exe export_int8\<model-or-partner>\compare_models.py
```

See [export_int8/README.md](export_int8/README.md) for the complete workflow and
[artifacts/README.md](artifacts/README.md) for deployment-package contents.

A fully runnable public example is available under `export_int8/example/`. It
trains a 193-parameter model on scikit-learn's bundled diabetes dataset and then
executes the same ONNX, full-INT8, comparison, metadata, plot, and C++ packaging
stages used by private model exports.

## C++ model arrays

A generated C++ package normally contains:

- `model_data.cpp`: the unchanged TFLite bytes as an aligned array;
- `model_data.h`: declarations for the array and its byte length;
- a model-specific README: shapes, preprocessing, quantization, runtime notes,
  validation metrics, and an integration prompt.

Only the `.cpp` and `.h` files are compiler inputs. Keep the `.tflite`, ONNX,
JSON metadata, preprocessing description, plots, and README as the traceable
reference package. An embedded application must reproduce the documented
feature order and preprocessing exactly.

This repository deliberately does not infer or generate board-specific Tensor
Arena sizes, sensor integration, measurement cycles, communication logic, or
firmware behavior as part of model conversion.

## Published examples

### Transfer learning

The transfer-learning example uses the dataset from the
[transfer-learning study](https://www.mdpi.com/2073-4433/13/10/1614):

1. Download the dataset from [Zenodo](https://zenodo.org/record/6821340).
2. Place `fullData.mat` in `Data/`.
3. Run `Evaluation/PrepareDataset.ipynb`.
4. Run `Evaluation/testBuiltModel.ipynb`.

### Drift and field tests

The drift and field-test example uses the corresponding
[published study](https://www.mdpi.com/2073-4433/12/11/1487):

1. Download the dataset from [Zenodo](https://zenodo.org/records/4593853).
2. Place it in `Data/Field/`.
3. Run `Evaluation/Field/PrepareDatasetField.ipynb`.
4. Run `Evaluation/Field/testBuiltModelField.ipynb`.

Additional public evaluation notebooks include the
[VOC study](https://www.mdpi.com/2073-4433/14/7/1123) under
`Evaluation/VOC4IAQ/`.

## Privacy and Git rules

The repository uses deny-by-default rules for private work:

- all `.mat` files and contents below `Data/` are ignored;
- unknown/new subfolders below `Evaluation/` are ignored;
- nested `Results/` folders and parameter-set files are ignored;
- every real subfolder below `export_int8/` is ignored except `example/`;
- every real subfolder below `artifacts/` is ignored except `example/`;
- Python environments, including `.venv-tflite`, are ignored.

Do not add partner folders as `.gitignore` exceptions. Git ignore rules also do
not remove files that were tracked previously; such files must be removed from
Git's index once while retaining the local copy.

## License

This project is licensed under the
[GNU Affero General Public License v3.0](LICENSE.txt).
