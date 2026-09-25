# Machine Learning Seminar

The seminar is organized by teaching day:

- `Day_01/` contains two self-contained, English introductions using fictional data.
- `Day_02/` contains the existing applied gas-sensor course and its reusable pipeline.
- `Day_03/` introduces TCOCNN architectures, four-sensor preprocessing, training,
  and hyperparameter optimization.
- `Day_02/Legacy_Previous_Course/` preserves the earlier notebook series as reference.

## Day 1 — general Python machine learning

1. `01_SciPy_and_Classical_ML_Foundations.ipynb` — preprocessing, regression,
   classification, SciPy optimization, scikit-learn estimators, an
   imputer–scaler–model pipeline, explicit L1/L2 regularization, and manual tuning.
2. `02_PyTorch_ML_Foundations.ipynb` — tensors, autograd, mini-batches, a fully
   explained training loop, weight decay, dropout, early stopping, manual
   hyperparameter search, regression, and classification.

Both notebooks run without downloads and deliberately avoid high-level tuning
helpers so learners can see what happens inside model selection.

## Day 2 — applied MOS gas-sensor machine learning

The existing course uses Sensor A, sub-sensor/channel 0 from the public
**Transfer Learning Dataset for Metal Oxide Semiconductor Gas Sensors**.

- Dataset: <https://doi.org/10.5281/zenodo.6821340>
- Paper: <https://doi.org/10.3390/atmos13101614>

Day 2 covers raw temperature-cycle data, single-point calibration, power-law
models, physical features, overfitting and group leakage, dimensionality, and a
full Feature Extraction, Selection and Regression (FESR) baseline.

## Day 3 — TCOCNN

Day 3 explains the PyTorch components of a TCOCNN, train-only input and target
Z-scores, parallel processing of four parallel sensor channels, a complete real-data
training example, and grid, random, and Bayesian search experiments.

## Day 5 — water/H2 classification

Day 5 uses TCOCNNv3 as a binary classifier for cycle-wise CSV data. It ignores
the first settling cycle in every file, applies log10 and a per-cycle Z-score,
restores the best validation checkpoint, and evaluates the frozen model on
separate water and H2 test files.

## Start

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
jupyter lab "Evaluation Seminar"
```

Install `requirements-seminar.txt` if the environment is not available. The
materials target Python 3.11 and use NumPy, SciPy, Matplotlib, scikit-learn,
scikit-optimize, h5py, and PyTorch.

## Additional practical notebooks

- Day 2/09: FESR as a scikit-learn Pipeline with grouped model selection.
- Day 3/05: TCOCNN v3 residual ablation and Bayesian hyperparameter search.
- Day 4/01: one-channel transfer between seven sensors.
- Day 4/02: occlusion, input gradients and signed regression Grad-CAM.
- Day 5/01: binary water-vs-H2 classification from train/test CSV files.

The Day 4 notebooks run in order; Notebook 01 saves the checkpoint for Notebook 02.
See `Day_04/README.md` for data requirements, budgets and split boundaries.
