# EOG to ARM M55M1 Rebuild Guide

This document explains how to rebuild the full project on another Windows PC, including:

- Python environment for the EOG model pipeline
- Vela conversion for Ethos-U
- Generation of the C++ model blob for M55M1
- Keil project build and flash
- TriBLE/Trianswer UART streaming into the M55M1 board

This guide reflects the current working project state in this workspace.

## What This Project Does

The project deploys the INT8 TensorFlow Lite model [eog_1dcnn_int8.tflite](C:/Users/User/Desktop/EOG/EOG_Classification/eog_1dcnn_int8.tflite) onto a Nuvoton `NuMaker-M55M1` board and runs inference on `Ethos-U`.

Current working configuration:

- Model input: `1 x 256 x 2`
- Model output: `1 x 5`
- Class order:
  - `down`
  - `forward`
  - `left`
  - `right`
  - `up`
- NPU target: `Ethos-U55-256`
- Board runtime mode: `Ethos-U / Vela model`
- Sensor stream input: `UART1`
- Debug output: board USB debug UART

## Important Paths

Repository root:

- [EOG_Classification](C:/Users/User/Desktop/EOG/EOG_Classification)

Main deployment files:

- [m55m1_deploy/main.cpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/main.cpp)
- [m55m1_deploy/BoardInit.cpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/BoardInit.cpp)
- [m55m1_deploy/Model/NNModel.cpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/Model/NNModel.cpp)
- [m55m1_deploy/Model/include/NNModel.hpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/Model/include/NNModel.hpp)
- [m55m1_deploy/Model/include/EogModelConfig.hpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/Model/include/EogModelConfig.hpp)
- [m55m1_deploy/tools/prepare_eog_m55m1.py](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/tools/prepare_eog_m55m1.py)
- [m55m1_deploy/vela_out_u55_256/eog_1dcnn_int8_vela.tflite](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/vela_out_u55_256/eog_1dcnn_int8_vela.tflite)
- [m55m1_deploy/TRIANSWER_UART_INTEGRATION.md](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/TRIANSWER_UART_INTEGRATION.md)

Working BSP-local Keil project:

- [NN_ModelInference.uvprojx](C:/Users/User/Desktop/ML_M55M1_SampleCode/M55M1BSP-3.01.003/SampleCode/MachineLearning/NN_ModelInference/KEIL/NN_ModelInference.uvprojx)

## Software You Need on the New PC

Install these first:

1. Python `3.10+`
2. Keil MDK / uVision 5
3. Nu-Link driver
4. Nuvoton `ML_M55M1_SampleCode`
5. `ethos-u-vela`
6. Tera Term or another serial terminal

Recommended folder layout:

```text
C:\
  Users\
    <YourUser>\
      Desktop\
        EOG\
          EOG_Classification\
        ML_M55M1_SampleCode\
          M55M1BSP-3.01.003\
```

The exact drive letter does not matter, but the README assumes:

- one copy of this repo
- one copy of `ML_M55M1_SampleCode`
- Keil can open the BSP-local `NN_ModelInference` project

## Step 1: Clone or Copy This Repo

Copy this project folder to the new PC:

- [EOG_Classification](C:/Users/User/Desktop/EOG/EOG_Classification)

If you use Git:

```powershell
git clone <your-repo-url>
cd EOG_Classification
```

## Step 2: Set Up Python

From the repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install ethos-u-vela
```

Current Python requirements from [requirements.txt](C:/Users/User/Desktop/EOG/EOG_Classification/requirements.txt):

- `tensorflow`
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`

## Step 3: Prepare the M55M1 BSP

Install or copy the Nuvoton BSP to:

- `C:\Users\<YourUser>\Desktop\ML_M55M1_SampleCode\M55M1BSP-3.01.003`

This deployment depends on:

- `ThirdParty\tflite_micro`
- `Library`
- the existing `SampleCode\MachineLearning\ImageClassification` and `NN_ModelInference` project structure

If you do not already have the BSP-local inference project, create it by copying the Nuvoton sample tree and then replacing files with the versions from this repo.

The expected project folder is:

- `...\M55M1BSP-3.01.003\SampleCode\MachineLearning\NN_ModelInference`

## Step 4: Convert the Model with Vela

The working accelerator target is `ethos-u55-256`.

From the repo root:

```powershell
vela eog_1dcnn_int8.tflite `
  --accelerator-config ethos-u55-256 `
  --system-config Ethos_U55_High_End_Embedded `
  --memory-mode Shared_Sram `
  --output-dir m55m1_deploy\vela_out_u55_256
```

Expected output:

- `m55m1_deploy\vela_out_u55_256\eog_1dcnn_int8_vela.tflite`
- Vela summary CSV

Important:

- Do not use `ethos-u55-128` for this board
- If you compile for the wrong MAC configuration, runtime will fail with an NPU config mismatch

## Step 5: Generate the C++ Model Blob

Run:

```powershell
python m55m1_deploy\tools\prepare_eog_m55m1.py `
  --model m55m1_deploy\vela_out_u55_256\eog_1dcnn_int8_vela.tflite `
  --summary m55m1_deploy\vela_out_u55_256\eog_model_summary_u55_256.json
```

This generates:

- [m55m1_deploy/Model/NN_Model_INT8.tflite.cpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/Model/NN_Model_INT8.tflite.cpp)
- summary JSON for verification

## Step 6: Copy Deployment Files into the BSP Project

Copy these files from this repo into the BSP-local Keil project:

- [m55m1_deploy/main.cpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/main.cpp)
  -> `...\NN_ModelInference\main.cpp`
- [m55m1_deploy/BoardInit.cpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/BoardInit.cpp)
  -> `...\NN_ModelInference\BoardInit.cpp`
- [m55m1_deploy/Model/NNModel.cpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/Model/NNModel.cpp)
  -> `...\NN_ModelInference\Model\NNModel.cpp`
- [m55m1_deploy/Model/include/NNModel.hpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/Model/include/NNModel.hpp)
  -> `...\NN_ModelInference\Model\include\NNModel.hpp`
- [m55m1_deploy/Model/include/EogModelConfig.hpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/Model/include/EogModelConfig.hpp)
  -> `...\NN_ModelInference\Model\include\EogModelConfig.hpp`
- [m55m1_deploy/Model/NN_Model_INT8.tflite.cpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/Model/NN_Model_INT8.tflite.cpp)
  -> `...\NN_ModelInference\Model\NN_Model_INT8.tflite.cpp`

If you already copied the BSP-local project from another machine, still regenerate and overwrite `NN_Model_INT8.tflite.cpp` if the model changed.

## Step 7: Keil Project Expectations

The working Keil project is:

- [NN_ModelInference.uvprojx](C:/Users/User/Desktop/ML_M55M1_SampleCode/M55M1BSP-3.01.003/SampleCode/MachineLearning/NN_ModelInference/KEIL/NN_ModelInference.uvprojx)

The project should already include:

- `ARM_NPU`
- `EOG_USE_ETHOS_U`
- `TF_LITE_STATIC_MEMORY`
- `CMSIS_NN`
- `ACTIVATION_BUF_SZ=0x00040000`

It should also include the Ethos-U driver sources, TFLM, and the local model files.

If you are rebuilding manually from scratch, the easiest path is:

1. Start from the Nuvoton `ImageClassification` sample
2. Duplicate it as `NN_ModelInference`
3. Replace app/model files with the ones from this repo
4. Keep the Ethos-U, TFLM, CMSIS, and BSP include paths from the working project

## Step 8: Build from Keil GUI

Open:

- [NN_ModelInference.uvprojx](C:/Users/User/Desktop/ML_M55M1_SampleCode/M55M1BSP-3.01.003/SampleCode/MachineLearning/NN_ModelInference/KEIL/NN_ModelInference.uvprojx)

Then:

1. Select target `NN_ModelInference`
2. Build
3. Confirm the output is `0 Error(s), 0 Warning(s)`

## Step 9: Build from Command Line

If `UV4.exe` is installed:

```powershell
& 'C:\Keil_v5\UV4\UV4.exe' -b `
  'C:\Users\User\Desktop\ML_M55M1_SampleCode\M55M1BSP-3.01.003\SampleCode\MachineLearning\NN_ModelInference\KEIL\NN_ModelInference.uvprojx' `
  -t 'NN_ModelInference'
```

Expected outputs:

- `...\KEIL\Objects\NN_ModelInference.axf`
- `...\KEIL\release`
- build log at:
  - `...\KEIL\Objects\NN_ModelInference.build_log.htm`

## Step 10: Flash the Board

Connect the NuMaker-M55M1 board through Nu-Link and run:

```powershell
& 'C:\Keil_v5\UV4\UV4.exe' -f `
  'C:\Users\User\Desktop\ML_M55M1_SampleCode\M55M1BSP-3.01.003\SampleCode\MachineLearning\NN_ModelInference\KEIL\NN_ModelInference.uvprojx' `
  -t 'NN_ModelInference'
```

If flash completes but the app does not auto-run, press the board `reset` button.

## Step 11: Verify Board Boot on UART

Open the debug serial port with:

- `115200 8N1`

Expected boot messages include:

- `BoardInit: complete`
- `Ethos-U device initialised`
- `Target system: M55M1`
- `Mode       : Ethos-U / Vela model`

If you see an error like:

```text
NPU config mismatch. npu.macs_per_cc=8, optimizer.macs_per_cc=7
```

then the model was compiled for the wrong accelerator. Re-run Vela using `ethos-u55-256`.

## Runtime Data Path

Current runtime behavior:

- Debug UART prints logs and inference results
- Sensor input comes from `UART1`
- `UART1` pin map:
  - `PB2 = RX`
  - `PB3 = TX`
- Sensor UART baud:
  - `115200 8N1`

See also:

- [TRIANSWER_UART_INTEGRATION.md](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/TRIANSWER_UART_INTEGRATION.md)

## UART Stream Format

The board expects one 2-channel sample per line:

```text
0.152,-0.041
```

or:

```text
0.152 -0.041
```

Behavior:

- first `256` frames fill the inference window
- then the board runs inference every `32` new frames
- results are printed to the debug UART

Supported command lines on `UART1`:

- `help`
- `reset`
- `infer`

## TriBLE / Trianswer Integration

Recommended architecture:

```text
TriBLE/Trianswer -> UART1 -> M55M1 -> Ethos-U inference -> debug UART output
```

Recommended wiring:

- `Trianswer TX` -> `M55M1 PB2`
- `Trianswer RX` -> `M55M1 PB3`
- `Trianswer GND` -> `M55M1 GND`

Use `3.3V TTL` only.

Do not connect `5V UART` signals directly.

## Input Data Requirements

The model was trained on preprocessed EOG data, not raw arbitrary ADC counts.

Your sensor-side stream should match the same domain used in training as closely as possible:

- band-pass filtered
- baseline handled
- aligned channel meaning:
  - CH1 = horizontal
  - CH2 = vertical
- same semantic class definitions

Model-side constants are in:

- [EogModelConfig.hpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/Model/include/EogModelConfig.hpp)

Current quantization:

- input scale: `0.03121989034116268`
- input zero point: `-24`
- output scale: `0.00390625`
- output zero point: `-128`

## Rebuilding the Dataset Side

Python-side dataset/training tools remain in the repo root:

- [prepare_eyecon_dataset.py](C:/Users/User/Desktop/EOG/EOG_Classification/prepare_eyecon_dataset.py)
- [train_eog_1dcnn.py](C:/Users/User/Desktop/EOG/EOG_Classification/train_eog_1dcnn.py)
- [evaluate_eog_model.py](C:/Users/User/Desktop/EOG/EOG_Classification/evaluate_eog_model.py)

The pipeline described in the repo README expects shape:

- `(N, 256, 2)`

If you retrain the model, repeat:

1. retrain/export TFLite
2. run Vela for `ethos-u55-256`
3. regenerate `NN_Model_INT8.tflite.cpp`
4. rebuild and flash Keil project

## Files Worth Backing Up

If you want the fastest restore on another PC, keep these together:

- this repo
- the BSP-local `NN_ModelInference` project folder
- Keil project file
- generated Vela output
- generated `NN_Model_INT8.tflite.cpp`

That lets you rebuild even if you do not want to re-run every preparation step immediately.

## Known Good Outputs

When everything is correct:

- Keil build finishes with `0 Error(s), 0 Warning(s)`
- board boots and prints Ethos-U init info
- runtime shows:
  - `Operator 0: ethos-u`
  - `Mode       : Ethos-U / Vela model`
- inference results print on the debug UART

## Troubleshooting

- Build fails on missing TFLM or BSP headers:
  - confirm `ML_M55M1_SampleCode` is installed and include paths are valid
- Flash succeeds but no UART output:
  - check COM port
  - check `115200 8N1`
  - press reset
- UART output appears but inference fails:
  - verify Vela target is `ethos-u55-256`
- Results are unstable:
  - verify your input stream is preprocessed consistently with training
  - verify channel order is horizontal then vertical
  - verify class definitions are consistent

## Suggested Restore Checklist

On a new PC, do these in order:

1. Install Python, Keil, Nu-Link, BSP, and Vela
2. Copy this repo
3. Run Python environment setup
4. Run Vela for `ethos-u55-256`
5. Run `prepare_eog_m55m1.py`
6. Copy deployment files into BSP-local `NN_ModelInference`
7. Open or build the Keil project
8. Flash the board
9. Open UART
10. Feed TriBLE/Trianswer data through `UART1`

## Notes

This README documents the current working deployment, not a generic Nuvoton tutorial.
If you move folders on the new PC, update the path assumptions in:

- Keil include paths
- the Python script `prepare_eog_m55m1.py`
- any local automation scripts you create
