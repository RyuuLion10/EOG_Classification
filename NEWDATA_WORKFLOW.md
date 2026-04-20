# newData Workflow

## 1. Build the paired 2-channel dataset

```powershell
.venv\Scripts\python prepare_newdata_dataset.py --overwrite
```

This converts `../newData/*.txt` into:

```text
dataset/
  blink/
  down/
  left/
  right/
  up/
```

Each CSV contains:

```csv
horizontal,vertical
129,127
135,125
...
```

Channel order is fixed to:

- column 1 = `horizontal`
- column 2 = `vertical`

Skipped source files are recorded in `dataset/conversion_summary.json`.

## 2. Train the 5-class model and export INT8 TFLite

```powershell
.venv\Scripts\python train_eog_1dcnn.py `
  --data-dir dataset `
  --output-dir artifacts `
  --epochs 50 `
  --batch-size 16 `
  --export-tflite
```

Outputs:

- `artifacts/eog_1dcnn.keras`
- `artifacts/eog_1dcnn_int8.tflite`
- `artifacts/metadata.json`

## 3. Evaluate the trained model

```powershell
.venv\Scripts\python evaluate_eog_model.py `
  --data-dir dataset `
  --model-path artifacts\eog_1dcnn.keras `
  --output-dir artifacts
```

Outputs:

- `artifacts/classification_report.json`
- `artifacts/confusion_matrix.csv`

## 4. Generate M55M1 deployment artifacts

```powershell
.venv\Scripts\python m55m1_deploy\tools\prepare_eog_m55m1.py `
  --model artifacts\eog_1dcnn_int8.tflite `
  --summary m55m1_deploy\eog_model_summary.json
```

Outputs:

- `m55m1_deploy/Model/NN_Model_INT8.tflite.cpp`
- `m55m1_deploy/Model/include/EogModelConfig.hpp`
- `m55m1_deploy/eog_model_summary.json`

## 5. Copy to the BSP / Keil project

Copy these generated files into your `NN_ModelInference` project on the M55M1 BSP side:

- `m55m1_deploy/main.cpp`
- `m55m1_deploy/BoardInit.cpp`
- `m55m1_deploy/Model/NNModel.cpp`
- `m55m1_deploy/Model/include/NNModel.hpp`
- `m55m1_deploy/Model/include/EogModelConfig.hpp`
- `m55m1_deploy/Model/NN_Model_INT8.tflite.cpp`

## Current class order

The current 5-class output order is:

1. `blink`
2. `down`
3. `left`
4. `right`
5. `up`
