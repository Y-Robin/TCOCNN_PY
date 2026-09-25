# Day 4 - Calibration transfer and occlusion XAI

First inspect the exact data scope and then inspect or rerun the six-source
optimization in:

1. `00_1_Datenuebersicht_Global6_HPO.ipynb`
2. `00_Global6_Hyperparameter_Optimization.ipynb`

The underlying reproducible command-line implementation is:

```bash
python "Evaluation Seminar/Day_04/00_Global6_Hyperparameter_Optimization.py"
```

Then continue with:

1. `01_Transfer_Learning_Between_Sensors.ipynb`
2. `02_XAI_Occlusion_and_Cycle_Buildup.ipynb`

The notebooks expect the repository-level `Networks/` and `Data/` folders to be
available. In particular, the seven prepared devices and target metadata must be
present under `Data/Dennis/Prepared`.

For the complete LOSO-selected transfer experiment run:

```bash
python "Evaluation Seminar/Day_04/calibration_transfer_sensor0.py"
```

Pass the optimized Global-6 configuration explicitly with:

```bash
python "Evaluation Seminar/Day_04/calibration_transfer_sensor0.py" --global-params "artifacts/seminar_day4_global6_hpo/best_hyperparameters.json"
```

This script always pools devices 1-6 as source observations, transfers to device
7, and uses only subsensor/channel 0 from every device. It asserts the model input
shape `(1, 1440, 1)` so devices cannot accidentally be encoded as input channels.
The historical `calibration_transfer_all_subsensors.py` filename remains only as
a compatibility entry point for this same single-subsensor experiment.

## Notebook 01: six-source model -> seventh sensor

The experiment is **acetone [ppb]**, calibration 1, and exclusively channel /
subsensor 0. Sensors 1-6 contribute observations to the pooled `Global-6`
training set; they are never encoded as model input channels. Sensor 7 is the
only transfer target, and every split is checked to have input shape
`(n, 1, 1440, 1)`.

The compared methods are `Global-6`, `Global + target`, `Scratch`,
`Head-only`, `Fine-tune`, `DS`, and `PDS`. Hyperparameters are selected
only on the six development sensors. In each LOSO fold, five devices train the
fold source model and the sixth acts as pseudo-target. Learning rate and epoch
are selected for all neural transfer methods; alpha is selected for DS and alpha
plus window size for PDS. Selection minimizes mean held-out UGM RMSE across all
six folds and is repeated for budgets of 5, 10, 20, 40, 80, 120, and 137 UGMs.

For DS/PDS, a pseudo-target cycle is paired with the sample-wise mean of the same
measurement row from the five fold-source devices. In the final transfer it is
paired with the corresponding mean from all six source devices. Sensor 7 does
not participate in hyperparameter selection.

Generated artifacts are written under
`artifacts/seminar_day4_global6_sensor0/`, including LOSO search records,
selected settings, final metrics, source and fine-tuned checkpoints, and plots.
The full default LOSO run is computationally expensive; the notebook includes a
one-fold smoke-test configuration for technical checks.

## Notebook 02: occlusion only, Top-1 -> full cycle

The XAI notebook intentionally uses **occlusion only**. It loads the saved
fine-tuned target model and the original six-source checkpoint.

The 1440-sample signal is treated as high/low pairs with 50 high + 70 low samples.
The number of pairs is derived from signal length and checked. For every pair the
notebook reports high-only, low-only and complete-pair occlusion effects. Ranking
uses only the adaptation-training UGMs.

Then the same fixed ranking is evaluated cumulatively: Top 1, Top 2, ... up to the
complete signal. Two views are compared:

- masking all non-selected pairs in the already trained model;
- building a reduced input from the selected pairs and re-adapting a fresh copy of
  the same six-source model with the transfer settings selected in Notebook 01.

Reduced-input fine-tuning reuses the same constant LR, fixed epoch count and frozen
BatchNorm running statistics. XAI validation selects the final `k`; test data only
report the frozen curve. XAI test output also uses UGM metrics only.
