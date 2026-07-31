# TCOCNN

TCOCNN is a convolutional neural-network toolbox for analysing metal-oxide
semiconductor (MOS) gas-sensor data in air-quality applications. The repository
contains the network implementation, preprocessing methods, and Jupyter
notebooks used for training and evaluation.

## Repository structure

```text
TCOCNN_PY/
├── Data/           # local datasets and generated MATLAB files
├── Evaluation/     # preparation, training, and evaluation notebooks
│   └── Results/    # local evaluation outputs
├── Networks/       # TCOCNN implementations
└── PreProcessing/  # data loading and standardisation methods
```

## Installation

TCOCNN uses PyTorch and keeps the existing notebook-facing API. Python 3.10 or
newer is required. Create a virtual environment and install all dependencies
from `requirements.txt`:

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
network automatically uses an NVIDIA GPU when `torch.cuda.is_available()`
returns `True`; otherwise it runs on the CPU. Check the active PyTorch
installation with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

The CUDA 13.0 setup was verified with an NVIDIA RTX 3080. For another operating
system or compute platform, use the selector on the
[official PyTorch installation page](https://pytorch.org/get-started/locally/).

## Notebook compatibility

The existing notebooks continue to use NumPy arrays with the original
channels-last layout `(samples, height, width, channels)`. The compatibility
layer converts these arrays to PyTorch tensors internally, moves batches to the
selected device, and converts predictions back to NumPy arrays.

The established `TCOCNNClass` methods remain available:

- `build_net`
- `compile_model`
- `train`
- `predict`
- `copy` and `retrain`
- `optimize_model`
- `custom_occlusion`
- `get_gradient_map`

Run the compatibility tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

## Example 1: transfer learning

This example uses the dataset associated with the
[transfer-learning study](https://www.mdpi.com/2073-4433/13/10/1614).

1. Download the dataset from [Zenodo](https://zenodo.org/record/6821340).
2. Place `fullData.mat` in `Data/`.
3. Run `Evaluation/PrepareDataset.ipynb`.
4. Run `Evaluation/testBuiltModel.ipynb`.

The remaining evaluation notebooks can then be used to test the other methods.

## Example 2: drift and field tests

This example uses the dataset associated with the
[drift and field-test study](https://www.mdpi.com/2073-4433/12/11/1487).

1. Download the dataset from [Zenodo](https://zenodo.org/records/4593853).
2. Place the downloaded data in `Data/Field/`.
3. Run `Evaluation/Field/PrepareDatasetField.ipynb`.
4. Run `Evaluation/Field/testBuiltModelField.ipynb`.

## Additional evaluation

Further evaluation notebooks are included for the
[VOC study](https://www.mdpi.com/2073-4433/14/7/1123), including the notebooks
under `Evaluation/VOC4IAQ/`.

## Local data and generated files

Datasets and generated results are intentionally excluded from Git:

- all `.mat` files;
- everything inside `Data/`;
- everything inside `Evaluation/Results/`, including nested directories; and
- generated JSON parameter sets whose names contain `Params` or `Parameters`,
  as well as `ParameterSets` directories.

The `Data/` and `Evaluation/Results/` directories remain in the repository
through placeholder files. Save new evaluation outputs below
`Evaluation/Results/` so they are not accidentally committed.

> Git ignore rules do not remove files that are already tracked. Existing
> parameter files must be removed from Git's index once while being retained
> locally.

## License

This project is licensed under the
[GNU Affero General Public License v3.0](LICENSE.txt).
