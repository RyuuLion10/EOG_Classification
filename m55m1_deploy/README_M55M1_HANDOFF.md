# M55M1 Rebuild Handoff Guide

This guide is written for another engineer to rebuild the project on a new Windows PC with minimal guesswork.

Use this together with:

- [README_M55M1.md](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/README_M55M1.md)
- [TRIANSWER_UART_INTEGRATION.md](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/TRIANSWER_UART_INTEGRATION.md)

This version adds:

- path checkpoints
- expected outputs
- suggested screenshot points
- a practical handoff checklist

## Goal

Rebuild a working M55M1 firmware that:

- runs the EOG INT8 model on `Ethos-U`
- receives `horizontal,vertical` EOG samples from `UART1`
- prints inference results to the debug UART

## Final Working Targets

The rebuild is successful when all of these are true:

- Keil project opens without missing-file errors
- Keil build ends with `0 Error(s), 0 Warning(s)`
- board flashes successfully through Nu-Link
- board boot log prints `Ethos-U device initialised`
- runtime log prints `Mode       : Ethos-U / Vela model`
- `UART1` accepts streaming samples
- debug UART prints `Prediction scores:` and `Predicted class:`

## Step 0: Prepare Folder Layout

Recommended directory layout:

```text
C:\Users\<User>\Desktop\
  EOG\
    EOG_Classification\
  ML_M55M1_SampleCode\
    M55M1BSP-3.01.003\
```

Checkpoint:

- repo exists:
  - [EOG_Classification](C:/Users/User/Desktop/EOG/EOG_Classification)
- BSP exists:
  - `C:\Users\<User>\Desktop\ML_M55M1_SampleCode\M55M1BSP-3.01.003`

Suggested screenshot:

- File Explorer showing both `EOG_Classification` and `ML_M55M1_SampleCode`

## Step 1: Install Tools

Required software:

- Python `3.10+`
- Keil MDK / uVision 5
- Nu-Link driver
- Tera Term
- `ethos-u-vela`

Checkpoint:

- `python --version` works
- Keil exists at something like:
  - `C:\Keil_v5\UV4\UV4.exe`
- Nu-Link can see the board when connected

Suggested screenshots:

- `python --version`
- Keil main window
- Device Manager showing Nu-Link or board COM port

## Step 2: Create Python Environment

From the repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install ethos-u-vela
```

Checkpoint:

- `.venv` folder exists
- `pip show ethos-u-vela` returns package info

Suggested screenshot:

- terminal after `pip install ethos-u-vela`

## Step 3: Confirm the Model Exists

Required model file:

- [eog_1dcnn_int8.tflite](C:/Users/User/Desktop/EOG/EOG_Classification/eog_1dcnn_int8.tflite)

Checkpoint:

- file exists
- size is non-zero

Suggested screenshot:

- File Explorer properties of `eog_1dcnn_int8.tflite`

## Step 4: Run Vela for the Correct Accelerator

Run from repo root:

```powershell
vela eog_1dcnn_int8.tflite `
  --accelerator-config ethos-u55-256 `
  --system-config Ethos_U55_High_End_Embedded `
  --memory-mode Shared_Sram `
  --output-dir m55m1_deploy\vela_out_u55_256
```

Important:

- The correct accelerator is `ethos-u55-256`
- Using `ethos-u55-128` will build, but inference will fail at runtime

Checkpoint:

- output file exists:
  - [eog_1dcnn_int8_vela.tflite](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/vela_out_u55_256/eog_1dcnn_int8_vela.tflite)
- Vela summary CSV exists in the same folder

Suggested screenshot:

- terminal showing Vela completed successfully

## Step 5: Generate the C++ Model Source

Run:

```powershell
python m55m1_deploy\tools\prepare_eog_m55m1.py `
  --model m55m1_deploy\vela_out_u55_256\eog_1dcnn_int8_vela.tflite `
  --summary m55m1_deploy\vela_out_u55_256\eog_model_summary_u55_256.json
```

Checkpoint:

- generated model source exists:
  - [NN_Model_INT8.tflite.cpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/Model/NN_Model_INT8.tflite.cpp)
- summary exists:
  - [eog_model_summary_u55_256.json](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/vela_out_u55_256/eog_model_summary_u55_256.json)

Quick verification:

- input shape should be `1, 256, 2`
- output shape should be `1, 5`

Suggested screenshot:

- the JSON file opened and showing input/output shapes

## Step 6: Prepare the BSP-local Keil Project

Expected project path:

- [NN_ModelInference.uvprojx](C:/Users/User/Desktop/ML_M55M1_SampleCode/M55M1BSP-3.01.003/SampleCode/MachineLearning/NN_ModelInference/KEIL/NN_ModelInference.uvprojx)

If the project is not present yet:

1. create or copy `NN_ModelInference` under:
   - `...\SampleCode\MachineLearning\`
2. make sure the subfolders exist:
   - `Device`
   - `KEIL`
   - `Model`
   - `NPU`

Checkpoint:

- `NN_ModelInference` folder exists
- `KEIL\NN_ModelInference.uvprojx` exists

Suggested screenshot:

- File Explorer inside `...\SampleCode\MachineLearning\NN_ModelInference`

## Step 7: Copy the Deployment Files

Copy these files from the repo into the BSP-local Keil project:

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

Checkpoint:

- the copied `main.cpp` is newer than the original Nuvoton sample
- the copied `BoardInit.cpp` contains:
  - `Sensor UART: UART1 @ 115200 baud`
- the copied `main.cpp` contains:
  - `Sensor input UART: UART1 (PB2=RX, PB3=TX), 115200 8N1`

Suggested screenshots:

- `BoardInit.cpp` open in editor
- `main.cpp` open in editor

## Step 8: Open Keil and Check Project Settings

Open:

- [NN_ModelInference.uvprojx](C:/Users/User/Desktop/ML_M55M1_SampleCode/M55M1BSP-3.01.003/SampleCode/MachineLearning/NN_ModelInference/KEIL/NN_ModelInference.uvprojx)

Check these:

- target selected: `NN_ModelInference`
- model file appears in project tree:
  - `NN_Model_INT8.tflite.cpp`
- app files appear in project tree:
  - `main.cpp`
  - `BoardInit.cpp`
  - `NNModel.cpp`

Also check preprocessor defines include:

- `ARM_NPU`
- `EOG_USE_ETHOS_U`

Checkpoint:

- project tree resolves all files without red missing-file icons

Suggested screenshots:

- Keil project tree
- target options page showing preprocessor defines

## Step 9: Build

You can use the Keil GUI or command line.

Command line:

```powershell
& 'C:\Keil_v5\UV4\UV4.exe' -b `
  'C:\Users\User\Desktop\ML_M55M1_SampleCode\M55M1BSP-3.01.003\SampleCode\MachineLearning\NN_ModelInference\KEIL\NN_ModelInference.uvprojx' `
  -t 'NN_ModelInference'
```

Checkpoint:

- build log exists:
  - `...\KEIL\Objects\NN_ModelInference.build_log.htm`
- final line includes:
  - `0 Error(s), 0 Warning(s).`
- output AXF exists:
  - `...\KEIL\Objects\NN_ModelInference.axf`

Suggested screenshot:

- build log showing `0 Error(s), 0 Warning(s)`

## Step 10: Flash the Board

Command line:

```powershell
& 'C:\Keil_v5\UV4\UV4.exe' -f `
  'C:\Users\User\Desktop\ML_M55M1_SampleCode\M55M1BSP-3.01.003\SampleCode\MachineLearning\NN_ModelInference\KEIL\NN_ModelInference.uvprojx' `
  -t 'NN_ModelInference'
```

Checkpoint:

- no flash error popup in Keil
- board enumerates over USB serial

If needed:

- press the board `reset` button after flashing

Suggested screenshot:

- Keil flash dialog or successful flash log

## Step 11: Open the Debug UART

Use Tera Term or another serial terminal.

Settings:

- baud: `115200`
- data: `8`
- parity: `None`
- stop: `1`
- flow control: `None`

Checkpoint:

- boot log appears after reset

Expected lines:

```text
INFO - BoardInit: complete
INFO - Ethos-U device initialised
INFO - Target system: M55M1
=== EOG 1D-CNN on M55M1 ===
Mode       : Ethos-U / Vela model
```

Suggested screenshot:

- Tera Term showing the successful boot log

## Step 12: Connect TriBLE / Trianswer

Working UART input mapping:

- `Trianswer TX` -> `M55M1 PB2 (UART1_RX)`
- `Trianswer RX` -> `M55M1 PB3 (UART1_TX)` optional
- `Trianswer GND` -> `M55M1 GND`

Electrical requirement:

- `3.3V TTL`

Checkpoint:

- wiring matches [TRIANSWER_UART_INTEGRATION.md](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/TRIANSWER_UART_INTEGRATION.md)

Suggested screenshot:

- a photo of the UART wiring

## Step 13: Feed Data

The board expects one line per sample:

```text
0.152,-0.041
```

or:

```text
0.152 -0.041
```

Current channel meaning:

- first value = `horizontal`
- second value = `vertical`

Runtime behavior:

- first `256` frames fill the window
- after that, inference runs every `32` new frames

Expected runtime messages:

```text
Buffered 32 / 256 frames
Buffered 64 / 256 frames
...
Prediction scores:
Predicted class: ...
```

Suggested screenshot:

- Tera Term showing buffering messages
- Tera Term showing a predicted class output

## Step 14: Common Failure Cases

If Keil builds but runtime fails with:

```text
NPU config mismatch. npu.macs_per_cc=8, optimizer.macs_per_cc=7
```

Cause:

- model compiled for the wrong Vela target

Fix:

- rebuild with `ethos-u55-256`

If there is no UART output:

- check COM port
- check `115200 8N1`
- press reset
- verify flash actually completed

If there is UART output but no predictions:

- confirm `UART1` is receiving data
- confirm the stream is one sample per line
- confirm the data is not raw unsupported format

If predictions are unstable:

- check channel order:
  - CH1 must be horizontal
  - CH2 must be vertical
- check preprocessing consistency with training

## Handoff Package Checklist

Before handing this to another person, make sure these are included:

- repo folder:
  - [EOG_Classification](C:/Users/User/Desktop/EOG/EOG_Classification)
- BSP-local project folder:
  - `...\SampleCode\MachineLearning\NN_ModelInference`
- generated Vela model:
  - [eog_1dcnn_int8_vela.tflite](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/vela_out_u55_256/eog_1dcnn_int8_vela.tflite)
- generated model blob:
  - [NN_Model_INT8.tflite.cpp](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/Model/NN_Model_INT8.tflite.cpp)
- this handoff guide:
  - [README_M55M1_HANDOFF.md](C:/Users/User/Desktop/EOG/EOG_Classification/m55m1_deploy/README_M55M1_HANDOFF.md)

## Quick Verification Summary

The shortest path to confirm the rebuild worked:

1. build in Keil
2. check `0 Error(s), 0 Warning(s)`
3. flash
4. reset board
5. open debug UART
6. confirm `Ethos-U device initialised`
7. send data into `UART1`
8. confirm `Predicted class:`
