# Day 2 — Applied machine learning for MOS gas-sensor data

The notebooks in this folder form the current Day 2 course. They use the public dataset from
[Zenodo DOI 10.5281/zenodo.6821340](https://doi.org/10.5281/zenodo.6821340).

## Fixed data rules

- Load Sensor A, channel/sub-sensor 0 only; the sensor dimension always has length one.
- Split UGM 1–500 by complete UGM groups into 64% train, 16% validation, and 20% test.
- Reserve UGM 501–906 exclusively for `test_extra`.
- Never place cycles from one UGM ID in different grouped splits.
- Keep all concentrations; there is no 300 ppb filter.
- Use training data only for point selection, correlations, scaling, and learned boundaries.
- Provide both the values stored on Zenodo and an additional `log1p` transformation.

Zenodo already describes the stored channel as logarithmic sensor resistance. The `log1p`
variant is therefore an additional mathematical transformation, not a reconstruction of the
original resistance.

## Notebook sequence

1. `01_Dataset_and_Single_Raw_Points.ipynb` — source data, grouped splits, temperature-boundary
   points, training-only correlation and selectivity screening, and exports.
2. `02_Visualizing_the_Data.ipynb` — target distributions, target correlation matrix, raw-cycle
   bands, examples, all 48 boundary scatter plots, and single-point baseline metrics.
3. `03_Single_Feature_Power_Law_vs_Linear_Regression.ipynb` — raw linear, log1p linear, and
   empirical power-law calibration on one validation-selected point.
4. `04_Multiple_Features_Power_Law_vs_Linear.ipynb` — greedy multi-point selection and the same
   three model families using one shared feature matrix.
5. `05_Physics_Informed_Features_Time_Constants_ALA.ipynb` — phase dynamics, tau63, Adaptive Linear
   Approximation, Ridge models, and interpretable feature correlations.
6. `06_Overfitting_Random_Split_vs_Unseen_UGMs.ipynb` — deliberate UGM leakage versus an
   honest unknown-UGM test and an ordered model-complexity experiment.
7. `07_Curse_of_Dimensionality.ipynb` — progressively overloaded k-NN, distance contrast, and
   1,000 irrelevant control features.
8. `08_FESR_Baseline_for_All_Gases.ipynb` — the shared single-channel FESR baseline for all ten gases:
   equidistant/ALA extraction, Pearson/RFE-LSR selection, and PLS regression.
9. `09_FESR_with_scikit-learn_Pipeline.ipynb` — fold-safe automatic segmentation and FESR inside a
   scikit-learn pipeline with grouped cross-validation.

Every notebook now includes direct visual checks of its inputs, split structure, feature space,
selection path, or model behavior. Shared English explanations remain centralized in
`notebook_guides.py`.

## Shared Python modules and exports

- `dataset_pipeline.py` loads, validates, splits, transforms, and exports the dataset.
- `physics_features.py` extracts phase and ALA features from one sensor channel.
- `fesr_baseline.py` implements FESR segmentation, RFE-LSR selection, and PLS regression.
- `automatic_fesr.py` implements fold-learned automatic segmentation.
- `notebook_guides.py` contains the English software and parameter explanations.

Run the export pipeline from the repository root:

```powershell
.\.venv\Scripts\python.exe "Evaluation Seminar\Day_02\dataset_pipeline.py"
```

Point datasets are written to `Data/seminar_point_datasets`; FESR results are written to
`Data/seminar_baselines`. Both regular test splits remain evaluation-only.
