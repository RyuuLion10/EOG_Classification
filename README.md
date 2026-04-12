# EOG Classification

This repository contains an end-to-end EOG classification workflow:

- Python training and evaluation for a 5-class EOG model
- TensorFlow Lite INT8 export
- Ethos-U / ARM M55M1 deployment files
- TriBLE / Trianswer UART streaming integration notes

The current target classes are:

- `down`
- `forward`
- `left`
- `right`
- `up`

## Project Layout

Repo root:

- [train_eog_1dcnn.py](C:/Users/User/Desktop/EOG/EOG_Classification/train_eog_1dcnn.py): train the 1D-CNN model
- [evaluate_eog_model.py](C:/Users/User/Desktop/EOG/EOG_Classification/evaluate_eog_model.py): evaluate the trained model
- [prepare_eyecon_dataset.py](C:/Users/User/Desktop/EOG/EOG_Classification/prepare_eyecon_dataset.py): convert the raw EyeCon dataset into per-sample CSV files
- [eog_1dcnn_int8.tflite](C:/Users/User/Desktop/EOG/EOG_Classification/eog_1dcnn_int8.tflite): exported INT8 TFLite model
- [requirements.txt](C:/Users/User/Desktop/EOG/EOG_Classification/requirements.txt): Python dependencies

Deployment files:

- [m55m1_deploy](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy)

Trianswer sample files:

- [trianswer_samples](C:/Users/User/Desktop/EOG/EOG_Classification/trianswer_samples)

## Python Pipeline

The training pipeline is based on 2-channel EOG data with this final model input shape:

```python
X.shape == (N, 256, 2)
```

Typical preprocessing assumptions used in this project:

- band-pass filtering
- baseline handling
- resampling to `256` time steps
- 2 channels ordered as:
  - `horizontal`
  - `vertical`

## Environment Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you also want to regenerate the Ethos-U model:

```powershell
pip install ethos-u-vela
```

## EyeCon Dataset Preparation

If you have the raw EyeCon dataset extracted locally, prepare it with:

```powershell
python prepare_eyecon_dataset.py --raw-dir raw_eyecon/DATASET --output-dir dataset --overwrite
```

The expected raw dataset structure is similar to:

```text
raw_eyecon/
  DATASET/
    S1/
      EOG.mat
      TargetGA.mat
      ControlSignal.mat
    S2/
      ...
```

Prepared dataset output is expected to look like:

```text
dataset/
  up/
    sample_001.csv
  down/
    sample_001.csv
  left/
    sample_001.csv
  right/
    sample_001.csv
  forward/
    sample_001.csv
```

Each sample CSV should contain two columns:

```csv
horizontal,vertical
0.12,-0.03
0.11,-0.04
...
```

## Training

Train with default settings:

```powershell
python train_eog_1dcnn.py
```

Example with explicit arguments:

```powershell
python train_eog_1dcnn.py --data-dir dataset --output-dir artifacts --epochs 50 --batch-size 32 --fs 176 --target-len 256
```

Typical outputs:

- `artifacts/eog_1dcnn.keras`
- `artifacts/metadata.json`

## Evaluation

Run:

```powershell
python evaluate_eog_model.py
```

## TFLite Model

This repo currently includes:

- [eog_1dcnn_int8.tflite](C:/Users/User/Desktop/EOG/EOG_Classification/eog_1dcnn_int8.tflite)

That model is the one used for the M55M1 deployment flow.

## ARM M55M1 Deployment

The M55M1 deployment files are under:

- [m55m1_deploy](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy)

Important documents:

- [README_M55M1.md](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/README_M55M1.md): full rebuild guide
- [README_M55M1_HANDOFF.md](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/README_M55M1_HANDOFF.md): handoff version with checkpoints
- [TRIANSWER_UART_INTEGRATION.md](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/TRIANSWER_UART_INTEGRATION.md): UART data path and wiring notes

Current M55M1 deployment status:

- target board: `NuMaker-M55M1`
- target accelerator: `Ethos-U55-256`
- model input: `1 x 256 x 2`
- model output: `1 x 5`
- board input stream: `UART1`
- board debug output: USB serial debug UART

## Vela Conversion

For the working M55M1 target, use:

```powershell
vela eog_1dcnn_int8.tflite `
  --accelerator-config ethos-u55-256 `
  --system-config Ethos_U55_High_End_Embedded `
  --memory-mode Shared_Sram `
  --output-dir m55m1_deploy\vela_out_u55_256
```

Then generate the C++ model source:

```powershell
python m55m1_deploy\tools\prepare_eog_m55m1.py `
  --model m55m1_deploy\vela_out_u55_256\eog_1dcnn_int8_vela.tflite `
  --summary m55m1_deploy\vela_out_u55_256\eog_model_summary_u55_256.json
```

## TriBLE / Trianswer Integration

Recommended runtime architecture:

```text
TriBLE / Trianswer -> UART1 -> M55M1 -> Ethos-U inference -> debug UART output
```

Current expected channel meaning:

- `CH1 = horizontal`
- `CH2 = vertical`

Current board-side UART stream format:

```text
0.152,-0.041
```

or:

```text
0.152 -0.041
```

Each line is one 2-channel EOG sample.

The M55M1 firmware:

- fills the first `256` samples into the window
- runs inference every `32` new samples afterward

## Trianswer Sample Files

Sample merged files are stored under:

- [trianswer_samples/baseline.txt](C:/Users/User/Desktop/EOG/EOG_Classification/trianswer_samples/baseline.txt)
- [trianswer_samples/baseline_meta.txt](C:/Users/User/Desktop/EOG/EOG_Classification/trianswer_samples/baseline_meta.txt)

These are useful for checking:

- channel order
- raw export format
- recording duration

## Notes

- If you rebuild the deployment on another PC, start with [README_M55M1.md](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/README_M55M1.md)
- If you are handing the project to someone else, use [README_M55M1_HANDOFF.md](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/README_M55M1_HANDOFF.md)
- If runtime inference fails with an NPU mismatch, make sure Vela was run for `ethos-u55-256`
