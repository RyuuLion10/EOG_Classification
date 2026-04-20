import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


CLASS_NAMES = ["blink", "down", "left", "right", "up"]
DEFAULT_DATA_DIR = "dataset"
DEFAULT_ARTIFACT_DIR = "artifacts"
ORIG_FS = 176
TARGET_LEN = 256
LOWCUT = 0.5
HIGHCUT = 20.0
EPS = 1e-8


def bandpass_filter(x, fs, lowcut=LOWCUT, highcut=HIGHCUT, order=3):
    nyquist = fs * 0.5
    if highcut >= nyquist:
        raise ValueError(
            f"highcut={highcut} must be smaller than Nyquist frequency {nyquist}"
        )
    b, a = butter(order, [lowcut, highcut], btype="bandpass", fs=fs)
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        raise ValueError(
            f"Signal length {len(x)} is too short for filtfilt padlen {padlen}"
        )
    return filtfilt(b, a, x)


def preprocess_trial(x, fs=ORIG_FS, target_len=TARGET_LEN):
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected shape (T, 2), got {x.shape}")

    x = x.astype(np.float32)

    x0 = bandpass_filter(x[:, 0], fs)
    x1 = bandpass_filter(x[:, 1], fs)
    x = np.stack([x0, x1], axis=-1)

    x = x - np.mean(x, axis=0, keepdims=True)
    x = resample(x, target_len, axis=0)

    std = np.std(x, axis=0, keepdims=True) + EPS
    x = x / std
    return x.astype(np.float32)


def load_one_csv(path):
    df = pd.read_csv(path)
    if df.empty or df.select_dtypes(include=[np.number]).shape[1] < 2:
        df = pd.read_csv(path, header=None)
    if df.empty:
        raise ValueError(f"{path} is empty")

    cols = [str(c).strip().lower() for c in df.columns]
    if "horizontal" in cols and "vertical" in cols:
        h = df.iloc[:, cols.index("horizontal")].to_numpy()
        v = df.iloc[:, cols.index("vertical")].to_numpy()
        x = np.stack([h, v], axis=-1)
    else:
        arr = df.select_dtypes(include=[np.number]).to_numpy()
        if arr.shape[1] == 2:
            x = arr
        elif arr.shape[1] >= 3:
            x = arr[:, 1:3]
        else:
            raise ValueError(f"Cannot parse {path}, numeric columns < 2")

    if np.isnan(x).any():
        raise ValueError(f"NaN detected in {path}")

    return x


def load_dataset(data_dir, fs=ORIG_FS, target_len=TARGET_LEN):
    X, y, file_paths = [], [], []

    for label in CLASS_NAMES:
        class_dir = os.path.join(data_dir, label)
        files = sorted(glob.glob(os.path.join(class_dir, "*.csv")))
        if not files:
            print(f"[warn] No CSV files found in {class_dir}")

        for file_path in files:
            raw = load_one_csv(file_path)
            x = preprocess_trial(raw, fs=fs, target_len=target_len)
            X.append(x)
            y.append(label)
            file_paths.append(file_path)

    if not X:
        raise ValueError(
            f"No samples were loaded from {data_dir}. "
            "Put CSV files under dataset/<class_name>/"
        )

    X = np.stack(X, axis=0)
    y = np.array(y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le, file_paths


def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    unique_labels, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    if min_count < 3:
        class_counts = {
            int(label): int(count) for label, count in zip(unique_labels, counts)
        }
        raise ValueError(
            "Each class needs at least 3 samples for train/val/test stratified split. "
            f"Current encoded class counts: {class_counts}"
        )

    temp_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=random_state,
        stratify=y,
    )

    relative_test_size = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape=(TARGET_LEN, 2), num_classes=5):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(16, 7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def representative_dataset(X, max_samples=100):
    limit = min(len(X), max_samples)
    for i in range(limit):
        yield [X[i : i + 1].astype(np.float32)]


def export_int8_tflite(model, X_calib, tflite_path):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(X_calib)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    out_path = Path(tflite_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    return out_path


def save_metadata(output_dir, label_encoder, history, test_metrics, args, tflite_path=None):
    metadata = {
        "classes": list(label_encoder.classes_),
        "target_len": args.target_len,
        "sampling_rate": args.fs,
        "bandpass_hz": [LOWCUT, HIGHCUT],
        "epochs_ran": len(history.history.get("loss", [])),
        "test_loss": float(test_metrics[0]),
        "test_accuracy": float(test_metrics[1]),
    }
    if tflite_path is not None:
        metadata["tflite_path"] = str(tflite_path)

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 1D-CNN on 2-channel EOG CSV trials."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--fs", type=int, default=ORIG_FS)
    parser.add_argument("--target-len", type=int, default=TARGET_LEN)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument(
        "--tflite-path",
        default=os.path.join(DEFAULT_ARTIFACT_DIR, "eog_1dcnn_int8.tflite"),
    )
    return parser.parse_args()


def main():
    import tensorflow as tf

    args = parse_args()
    ensure_dir(args.output_dir)

    X, y, label_encoder, _ = load_dataset(
        args.data_dir, fs=args.fs, target_len=args.target_len
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("classes:", list(label_encoder.classes_))

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, random_state=args.random_state
    )

    print("train:", X_train.shape, y_train.shape)
    print("val:", X_val.shape, y_val.shape)
    print("test:", X_test.shape, y_test.shape)

    model = build_model(
        input_shape=(args.target_len, X.shape[-1]),
        num_classes=len(label_encoder.classes_),
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    test_metrics = model.evaluate(X_test, y_test, verbose=1)
    print("Test accuracy:", test_metrics[1])

    model_path = os.path.join(args.output_dir, "eog_1dcnn.keras")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    tflite_path = None
    if args.export_tflite:
        tflite_path = export_int8_tflite(model, X_train, args.tflite_path)
        print(f"Saved INT8 TFLite model to {tflite_path}")

    save_metadata(output_dir=args.output_dir,
                  label_encoder=label_encoder,
                  history=history,
                  test_metrics=test_metrics,
                  args=args,
                  tflite_path=tflite_path)


if __name__ == "__main__":
    main()
