# Generated public example artifacts

These files are produced by the runnable workflow in `export_int8/example/`.
They contain only a tiny model trained on scikit-learn's bundled public diabetes
dataset and are safe to keep in Git.

Expected generated files:

```text
tiny_diabetes_regressor.onnx
tiny_diabetes_regressor_int8.tflite
tiny_diabetes_regressor_metadata.json
tiny_diabetes_regressor_preprocessing.json
tiny_diabetes_test_predictions.csv
tiny_diabetes_model_comparison.png
dummy_model_data.cpp
dummy_model_data.h
```

The example demonstrates the file formats and verification workflow only. It is
not a gas-sensor model and must not be substituted for a real TCOCNN deployment.
