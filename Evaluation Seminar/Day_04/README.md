# Day 4 — Calibration transfer and XAI

All notebooks are English and use subsensor 0 only.

1. 00_1_Global6_Data_Overview.ipynb
2. 00_Global6_Hyperparameter_Optimization.ipynb
3. 01_Transfer_Learning_Between_Sensors.ipynb
4. 02_XAI_Occlusion_and_Cycle_Buildup.ipynb

Notebook 00 saves the Global-6 parameters and weights once. Notebook 01 selects
the transfer learning rate and DS/PDS settings on one example slave, freezes
them, and evaluates nested sensor-7 transfer budgets of 5, 10, 20, and 40 UGMs.
DS/PDS use all individual master/slave regression rows. The diagnostic in
Notebook 01 demonstrates that this shared least-squares fit still targets a
master-mean compromise mathematically. It compares inference on actual master
signals, their mean, and individual master mappings. Sensor 6 was included in
Global-6 pretraining, so it is an example device, not an independent held-out
device for model selection.

Notebook 02 reloads the exact 40-UGM transfer subset. Occlusion identifies the
most important high/low pair. A nested contiguous window then grows one adjacent
pair at a time. Every model retains 1440 input positions. Outside the measured
window, values are replaced by the same mean computed from adaptation training
cycles only. Each model is retrained for up to 30 epochs; validation selects
between learning rates 1e-6 and 1e-5 and selects the checkpoint. Test is evaluated
after these choices are frozen. Training curves and actual masked-input plots
make underfitting and input construction visible. This is a bounded comparison,
not a claim that the shortest windows contain no predictive information.

The XAI notebook also includes Ridge and fresh-CNN one-pair controls. Removal
importance is not standalone predictive sufficiency; the selected nested windows
are not an exhaustive search over every possible contiguous window.
