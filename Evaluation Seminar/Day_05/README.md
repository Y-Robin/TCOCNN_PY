# Day 5 — Water/H2 classification

01_Water_vs_H2_Classification_with_TCOCNNv3.ipynb trains a binary TCOCNNv3
classifier from four headerless CSV files:

- train_water.csv
- train_h2.csv
- test_water.csv
- test_h2.csv

Each row must contain one complete cycle with 1,440 comma-separated values. The
first row of every file is treated as a settling cycle and ignored. Put the CSVs
in Day_05/data/ or change DATA_DIR in the configuration cell.

Before splitting and training, every cycle is transformed independently with
log10 followed by a per-cycle Z-score. The notebook visualizes both the raw and
the transformed cycles.

The notebook uses the class names directly. TCOCNNv3 supports classification via
regression=False, so artificial concentration targets (water = 400 ppb and H2 =
20,000 ppb) are not needed.
