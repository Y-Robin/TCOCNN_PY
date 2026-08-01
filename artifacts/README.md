# Deployment artifacts

Generated deployment packages are grouped by the same model or partner name as
their scripts below `export_int8/`.

```text
artifacts/
|-- README.md
|-- example/                  # runnable public dataset/model example
`-- <model-or-partner>/       # private models, metadata, arrays, and plots
```

For a C++ deployment that embeds the model, the compiler normally needs only
`model_data.cpp` and `model_data.h`. The `.tflite`, metadata, preprocessing JSON,
comparison plots, and source-format model should still be retained as the
traceable deployment package.

Every private subfolder is ignored by Git automatically. The generated files in
`example/` are the sole exception because their model is trained exclusively on
scikit-learn's bundled public diabetes dataset. Never place a partner model,
partner metric, or private calibration range in `example/`.
