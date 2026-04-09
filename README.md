# EOG 1D-CNN Pipeline

這個專案是給 University of Malta 公開雙通道 EOG 資料使用的最小可跑版本。
目標是先把資料整理、前處理、TensorFlow 1D-CNN 訓練與模型儲存整條鏈打通。

## 目前最佳結果

目前最佳版本使用：

- `ControlSignal` 切出主要 target saccade 段
- `forward` 類別做保守抽樣，避免嚴重類別失衡
- `0.5–20 Hz` band-pass + baseline removal + resample 到 `256`
- 1D-CNN 輸入 shape: `(N, 256, 2)`

在目前整理後的資料集上：

- dataset size: `2246`
- class counts: `down 244 / forward 449 / left 682 / right 649 / up 222`
- test accuracy: `0.9267`

測試集 confusion matrix 摘要：

```text
         down  forward  left  right  up
down       34        0    11      4   0
forward     0       90     0      0   0
left        0        0   137      0   0
right       2        1     0    123   4
up          1        0     8      2  33
```

重點觀察：

- `forward` 已不再大量干擾 `up/down`
- `left/right` 幾乎已經分得很乾淨
- `up/down` 仍有一部分會混到 `left`，尤其是 `down -> left` 和 `up -> left`
- 目前 `up` recall 約 `0.75`，`down` recall 約 `0.694`

## 資料夾結構

把每個 trial 存成一個 CSV，並放在對應類別底下：

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

每個 CSV 建議是兩欄：

```csv
horizontal,vertical
0.12,-0.03
0.11,-0.04
...
```

如果原始檔案是三欄，也支援把第 1 欄當時間、第 2 和第 3 欄當 `horizontal` / `vertical`。

## 安裝

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## EyeCon 原始資料整理

先把 University of Malta 官方 `DATASET.zip` 解壓後放成這種結構：

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

然後執行：

```powershell
python prepare_eyecon_dataset.py --raw-dir raw_eyecon/DATASET --output-dir dataset --overwrite
```

這支腳本會：

- 自動找出 `EOG.mat` 和 `TargetGA.mat` 內的主要 numeric array
- 嘗試把 EOG 資料整理成 `(trial, time, 2)`
- 依 `TargetGA` 的水平/垂直目標角度映射到 `up/down/left/right/forward`
- 匯出成 `dataset/<class>/sample_xxxx.csv`

目前 `forward` 類別採用實用近似法：

- 若目標角度接近 `(0, 0)`，直接標成 `forward`
- 否則另外從每個 trial 開頭切一段 center-gaze 訊號另存成 `forward`

這是為了先把五類資料鏈打通。因為 Dataset 1 原始 trial 本身是「中心 -> 隨機目標 -> 回中心 -> blink」，不是天生就已經切成五個完整 trial 類別。

## 前處理設定

- Band-pass: `0.5–20 Hz`
- 每通道減平均
- 重採樣到 `256` 點
- 每個 window 做 per-channel z-score

輸入模型前的資料 shape 會是：

```python
X.shape == (N, 256, 2)
y.shape == (N,)
```

## 訓練

```powershell
python train_eog_1dcnn.py
```

可選參數：

```powershell
python train_eog_1dcnn.py --data-dir dataset --output-dir artifacts --epochs 50 --batch-size 32 --fs 176 --target-len 256
```

## 輸出

訓練完成後會在 `artifacts/` 生成：

- `eog_1dcnn.keras`
- `metadata.json`

## 注意

- `filtfilt` 需要訊號長度大於 padding 長度，太短的 trial 會報錯。
- `train_test_split(..., stratify=y)` 需要每個類別至少有足夠樣本，建議每類先放至少數個 trial。
- 目前先專注在五類：`up / down / left / right / forward`。
