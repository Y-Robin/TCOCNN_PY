# Day 4 — Calibration transfer and XAI

All notebooks are English and use subsensor 0 only.

1. 00_1_Global6_Data_Overview.ipynb
2. 00_Global6_Hyperparameter_Optimization.ipynb
3. 01_Transfer_Learning_Between_Sensors.ipynb
4. 02_XAI_Occlusion_and_Cycle_Buildup.ipynb

Notebook 00 saves the Global-6 parameters and weights once. Notebook 01 selects
the transfer learning rate and DS/PDS settings on one example slave, freezes
them, and evaluates nested sensor-7 transfer budgets of 5, 10, 20, and 40 UGMs.
DS/PDS use all individual master/slave regression rows without a master mean.

Notebook 02 reloads the exact 40-UGM transfer subset. Occlusion identifies the
most important high/low pair. A nested contiguous window then grows one adjacent
pair at a time, and a new transfer model is trained for every window. This shows
how much continuous measurement time is needed without fragmented inputs.
