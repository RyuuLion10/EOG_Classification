import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from train_eog_1dcnn import ensure_dir, load_dataset, split_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained EOG 1D-CNN model with confusion matrix and classification report."
    )
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--model-path", default=os.path.join("artifacts", "eog_1dcnn.keras"))
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--fs", type=int, default=176)
    parser.add_argument("--target-len", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def save_report(output_dir, labels, report_dict, cm):
    report_path = os.path.join(output_dir, "classification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"), encoding="utf-8")


def main():
    import tensorflow as tf

    args = parse_args()
    ensure_dir(args.output_dir)

    X, y, label_encoder, _ = load_dataset(
        args.data_dir, fs=args.fs, target_len=args.target_len
    )
    _, _, X_test, _, _, y_test = split_dataset(X, y, random_state=args.random_state)

    model = tf.keras.models.load_model(args.model_path)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    labels = list(label_encoder.classes_)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=labels,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    print("\nClassification report:")
    print(report_text)

    save_report(args.output_dir, labels, report_dict, cm)


if __name__ == "__main__":
    main()
