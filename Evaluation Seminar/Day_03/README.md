# Day 3 — TCOCNN

1. `01_TCOCNN_Architecture_and_Components.ipynb`
2. `02_Preprocessing_and_Parallel_Sensors.ipynb`
3. `03_Train_a_TCOCNN_on_Sensor_Data.ipynb`
4. `04_TCOCNN_Hyperparameter_Optimization.ipynb`

Run notebooks in order. Notebook 01 is synthetic; notebooks 02–04 use
`Data/fullData.mat`. All preprocessing statistics are learned from training data
only. Training compares 30 and 100 epochs with 32/64 filters and 256/512 dense units.
Search uses eight trials per method and 60 epochs per trial, restoring the best
validation checkpoint. Scatterplots and residuals compare validation predictions;
the training notebook evaluates the frozen winner on both held-out splits.
Long runs may take substantial time on CPU.

5. `05_TCOCNN_v3_Hyperparameter_Optimization.ipynb`

The v3 extension compares a plain reference, a controlled residual ablation, and
six Bayesian trials. Every run receives 60 epochs. Validation scatterplots,
parameter counts, runtime, and held-out evaluation make the tradeoffs visible.
