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

## Requirements

The toolbox was developed with TensorFlow 2.6 and Keras 2.6. It also uses:

- NumPy
- SciPy
- h5py
- scikit-learn
- scikit-optimize
- Matplotlib
- Jupyter Notebook or JupyterLab

Use a Python version that is compatible with TensorFlow 2.6 when reproducing the
original environment.

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
