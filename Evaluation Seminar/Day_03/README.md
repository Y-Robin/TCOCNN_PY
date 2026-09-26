# Day 3 — TCOCNN with one customer sub-sensor

Every notebook uses Sensor A, sub-sensor 0 only. The model input has shape
`(samples, 1, 1440, 1)`; no other sub-sensor is loaded, plotted, scaled, or trained.

Run the notebooks in this order:

1. `01_TCOCNN_Architecture_and_Components.ipynb` — a small synthetic TCOCNN,
   tensor shapes, the one-row input matrix, and first-layer activations.
2. `02_Preprocessing_One_Subsensor.ipynb` — grouped real-data splits, target
   distributions, train-only input/target scaling, and raw/scaled signal views.
3. `03_Train_a_TCOCNN_on_Sensor_Data.ipynb` — one fixed v3 architecture,
   40 training epochs, learning curves, parity plots, and residuals.
4. `04_TCOCNN_Hyperparameter_Optimization.ipynb` — grid, random, and Bayesian
   search with four trials per strategy and 25 epochs per trial.
5. `05_TCOCNN_v3_Hyperparameter_Optimization.ipynb` — Plain/V1 comparison,
   residual ablation, and six Bayesian v3 trials with 30 epochs each.
6. `05b_TCOCNN_v3_Low_Learning_Rate_Search.ipynb` — the same controlled v3
   experiment with a lower learning-rate search range.

Notebook 01 is synthetic. Notebooks 02–05b use `Data/fullData.mat`. All split,
scaling, and checkpoint decisions use training and validation data only. The
regular test and `test_extra` remain final evaluations. Every notebook is stored
with executed default outputs and additional visual diagnostics.
