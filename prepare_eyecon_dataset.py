import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


CLASS_NAMES = ["up", "down", "left", "right", "forward"]
META_KEYS = {"__header__", "__version__", "__globals__"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert University of Malta EyeCon Dataset 1 MAT files into dataset/<class>/sample_xxx.csv."
    )
    parser.add_argument(
        "--raw-dir",
        default="raw_eyecon/DATASET",
        help="Folder containing subject directories such as S1, S2, ...",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Output folder containing up/down/left/right/forward subfolders.",
    )
    parser.add_argument(
        "--forward-threshold",
        type=float,
        default=1.0,
        help="If sqrt(horizontal^2 + vertical^2) <= threshold, label as forward.",
    )
    parser.add_argument(
        "--forward-segment-samples",
        type=int,
        default=176,
        help=(
            "If the raw target file only gives one label per trial, create a forward sample "
            "from the first N points of each trial."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing CSV files in output class folders before exporting.",
    )
    parser.add_argument(
        "--forward-max-count",
        type=int,
        default=None,
        help=(
            "Maximum number of forward samples to export. "
            "Default: rounded mean count of non-forward classes."
        ),
    )
    parser.add_argument(
        "--forward-crop-ratio",
        type=float,
        default=0.5,
        help="Keep only the leading portion of the center/return segment for forward candidates.",
    )
    return parser.parse_args()


def matlab_numeric_items(path):
    data = loadmat(path, squeeze_me=False, struct_as_record=False)
    items = {}
    for key, value in data.items():
        if key in META_KEYS:
            continue
        arr = np.asarray(value)
        if arr.dtype == np.object_:
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        items[key] = arr
    if not items:
        raise ValueError(f"No numeric arrays found in {path}")
    return items


def choose_main_array(items):
    ranked = sorted(
        items.items(),
        key=lambda kv: (np.asarray(kv[1]).size, np.asarray(kv[1]).ndim),
        reverse=True,
    )
    return ranked[0]


def find_preferred_array(items, preferred_names):
    normalized = {k.lower(): (k, v) for k, v in items.items()}
    for preferred in preferred_names:
        if preferred in normalized:
            return normalized[preferred]
    return choose_main_array(items)


def find_target_array(items):
    return find_preferred_array(items, ("targetga", "target_ga", "target", "ga"))


def find_control_array(items):
    return find_preferred_array(items, ("controlsignal", "control_signal", "control"))


def ensure_class_dirs(output_dir, overwrite=False):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_NAMES:
        class_dir = output_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for csv_path in class_dir.glob("*.csv"):
                csv_path.unlink()


def get_subject_dirs(raw_dir):
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    subject_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.upper().startswith("S")
    )
    if not subject_dirs:
        raise ValueError(f"No subject folders like S1/S2 found under {raw_dir}")
    return subject_dirs


def move_axis(arr, source_axis, target_axis):
    return np.moveaxis(arr, source_axis, target_axis)


def normalize_eog_trials(arr, expected_trials=None):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        raise ValueError(f"EOG array must have at least 2 dimensions, got {arr.shape}")

    if arr.ndim == 2:
        if 2 in arr.shape:
            if arr.shape[-1] == 2:
                return arr[np.newaxis, :, :].astype(np.float32)
            if arr.shape[0] == 2:
                return arr.T[np.newaxis, :, :].astype(np.float32)
        raise ValueError(
            "2D EOG array does not expose a clear 2-channel axis. "
            f"Observed shape: {arr.shape}"
        )

    if arr.ndim != 3:
        raise ValueError(f"Unsupported EOG array shape: {arr.shape}")

    candidate_axes = [axis for axis, size in enumerate(arr.shape) if size == 2]
    if not candidate_axes:
        raise ValueError(f"Could not find a 2-channel axis in EOG array shape {arr.shape}")

    for channel_axis in candidate_axes:
        candidate = move_axis(arr, channel_axis, -1)
        leading_shape = candidate.shape[:-1]
        if expected_trials is not None:
            for axis, size in enumerate(leading_shape):
                if size == expected_trials:
                    trial_first = move_axis(candidate, axis, 0)
                    return trial_first.astype(np.float32)
        trial_axis = int(np.argmin(leading_shape))
        trial_first = move_axis(candidate, trial_axis, 0)
        return trial_first.astype(np.float32)

    raise ValueError(f"Failed to normalize EOG array shape {arr.shape}")


def normalize_continuous_eog(arr):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected continuous EOG to be 2D, got {arr.shape}")

    if arr.shape[0] == 2:
        return arr.T.astype(np.float32)
    if arr.shape[1] == 2:
        return arr.astype(np.float32)

    raise ValueError(f"Could not infer 2-channel layout from EOG shape {arr.shape}")


def normalize_trial_targets(arr, expected_trials):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        if arr.shape[0] != expected_trials:
            raise ValueError(
                f"Target vector length {arr.shape[0]} does not match expected trials {expected_trials}"
            )
        return np.stack([arr, np.zeros_like(arr)], axis=-1).astype(np.float32)

    if arr.ndim == 2:
        if expected_trials in arr.shape and 2 in arr.shape:
            if arr.shape[0] == expected_trials and arr.shape[1] >= 2:
                return arr[:, :2].astype(np.float32)
            if arr.shape[1] == expected_trials and arr.shape[0] >= 2:
                return arr[:2, :].T.astype(np.float32)

        if arr.shape[0] == expected_trials:
            if arr.shape[1] == 1:
                return np.concatenate([arr, np.zeros_like(arr)], axis=1).astype(np.float32)
            return arr[:, :2].astype(np.float32)
        if arr.shape[1] == expected_trials:
            transposed = arr.T
            if transposed.shape[1] == 1:
                return np.concatenate(
                    [transposed, np.zeros_like(transposed)], axis=1
                ).astype(np.float32)
            return transposed[:, :2].astype(np.float32)

    raise ValueError(
        "Could not align target array with trial count. "
        f"Target shape: {arr.shape}, expected trials: {expected_trials}"
    )


def extract_trial_bounds_from_control(control_arr):
    control = np.asarray(control_arr).squeeze()
    if control.ndim != 1:
        raise ValueError(f"Expected 1D control signal, got {control.shape}")

    changes = np.where(np.diff(control) != 0)[0] + 1
    seg_starts = np.r_[0, changes]
    seg_ends = np.r_[changes, len(control)]
    seg_values = control[seg_starts]

    valid_mask = seg_values != 30
    seg_starts = seg_starts[valid_mask]
    seg_ends = seg_ends[valid_mask]
    seg_values = seg_values[valid_mask]

    if len(seg_values) % 3 != 0:
        raise ValueError(
            f"Expected control signal segments in groups of 3, got {len(seg_values)}"
        )

    trial_bounds = []
    for idx in range(0, len(seg_values), 3):
        values = seg_values[idx : idx + 3].tolist()
        starts = seg_starts[idx : idx + 3]
        ends = seg_ends[idx : idx + 3]
        if values != [1, 2, 3]:
            raise ValueError(
                f"Unexpected control pattern at group {idx // 3}: {values}, expected [1, 2, 3]"
            )
        trial_bounds.append(
            {
                "trial_start": int(starts[0]),
                "trial_end": int(ends[2]),
                "segment_1_start": int(starts[0]),
                "segment_1_end": int(ends[0]),
                "segment_2_start": int(starts[1]),
                "segment_2_end": int(ends[1]),
                "segment_3_start": int(starts[2]),
                "segment_3_end": int(ends[2]),
                "forward_start": int(starts[0]),
                "forward_end": int(ends[0]),
            }
        )
    return trial_bounds


def direction_from_target(horizontal, vertical, forward_threshold):
    magnitude = float(np.sqrt(horizontal ** 2 + vertical ** 2))
    if magnitude <= forward_threshold:
        return "forward"
    if abs(horizontal) >= abs(vertical):
        return "right" if horizontal > 0 else "left"
    return "up" if vertical > 0 else "down"


def write_sample_csv(signal, output_dir, label, sample_index):
    df = pd.DataFrame(signal, columns=["horizontal", "vertical"])
    out_path = Path(output_dir) / label / f"sample_{sample_index:04d}.csv"
    df.to_csv(out_path, index=False)


def buffer_forward_candidate(forward_candidates, signal, subject_name, summary):
    if len(signal) >= 16:
        forward_candidates.append(
            {
                "subject_name": subject_name,
                "signal": signal.copy(),
            }
        )
        summary["forward_candidates"] += 1


def crop_leading_portion(signal, ratio):
    if len(signal) == 0:
        return signal
    keep = max(16, int(round(len(signal) * ratio)))
    keep = min(len(signal), keep)
    return signal[:keep]


def export_subject_trials(
    subject_name,
    eog_trials,
    trial_targets,
    output_dir,
    counters,
    forward_threshold,
    forward_segment_samples,
    forward_candidates,
):
    summary = defaultdict(int)

    if len(eog_trials) != len(trial_targets):
        raise ValueError(
            f"{subject_name}: number of EOG trials ({len(eog_trials)}) does not match "
            f"target rows ({len(trial_targets)})"
        )

    for trial_idx, (signal, target_pair) in enumerate(zip(eog_trials, trial_targets), start=1):
        label = direction_from_target(target_pair[0], target_pair[1], forward_threshold)
        if label == "forward":
            buffer_forward_candidate(forward_candidates, signal, subject_name, summary)
        else:
            counters[label] += 1
            write_sample_csv(signal, output_dir, label, counters[label])
            summary[label] += 1

        forward_len = min(forward_segment_samples, signal.shape[0])
        if forward_len >= 16:
            buffer_forward_candidate(
                forward_candidates, signal[:forward_len], subject_name, summary
            )

    return dict(summary)


def export_subject_from_continuous(
    subject_name,
    continuous_eog,
    trial_bounds,
    target_arr,
    output_dir,
    counters,
    forward_threshold,
    forward_candidates,
    forward_crop_ratio,
):
    targets = np.asarray(target_arr)
    if targets.shape[0] != len(trial_bounds) * 2:
        raise ValueError(
            f"{subject_name}: expected TargetGA rows to be 2x trial count. "
            f"Got targets={targets.shape[0]}, trials={len(trial_bounds)}"
        )

    summary = defaultdict(int)
    for trial_idx, bounds in enumerate(trial_bounds):
        target_pair = targets[trial_idx * 2]
        return_pair = targets[trial_idx * 2 + 1]

        label = direction_from_target(target_pair[0], target_pair[1], forward_threshold)
        signal = continuous_eog[bounds["segment_1_start"] : bounds["segment_1_end"]]
        if label == "forward":
            buffer_forward_candidate(forward_candidates, signal, subject_name, summary)
        else:
            counters[label] += 1
            write_sample_csv(signal, output_dir, label, counters[label])
            summary[label] += 1

        if np.linalg.norm(return_pair) <= forward_threshold + 1e-8:
            forward_signal = continuous_eog[bounds["segment_2_start"] : bounds["segment_2_end"]]
            forward_signal = crop_leading_portion(forward_signal, forward_crop_ratio)
            buffer_forward_candidate(
                forward_candidates, forward_signal, subject_name, summary
            )

    return dict(summary)


def export_balanced_forward_samples(
    forward_candidates, output_dir, counters, subject_summaries, forward_max_count=None
):
    non_forward_counts = [counters[label] for label in CLASS_NAMES if label != "forward"]
    if not non_forward_counts:
        desired_count = 0
    elif forward_max_count is not None:
        desired_count = max(0, min(forward_max_count, len(forward_candidates)))
    else:
        desired_count = min(
            int(round(np.mean(non_forward_counts))), len(forward_candidates)
        )

    for candidate in forward_candidates[:desired_count]:
        counters["forward"] += 1
        write_sample_csv(candidate["signal"], output_dir, "forward", counters["forward"])
        subject_summaries.setdefault(candidate["subject_name"], {})
        subject_summaries[candidate["subject_name"]]["forward"] = (
            subject_summaries[candidate["subject_name"]].get("forward", 0) + 1
        )

    return desired_count


def convert_dataset(
    raw_dir,
    output_dir,
    forward_threshold,
    forward_segment_samples,
    overwrite,
    forward_max_count=None,
    forward_crop_ratio=0.5,
):
    ensure_class_dirs(output_dir, overwrite=overwrite)
    subject_dirs = get_subject_dirs(raw_dir)
    counters = defaultdict(int)
    subject_summaries = {}
    forward_candidates = []

    for subject_dir in subject_dirs:
        eog_path = subject_dir / "EOG.mat"
        target_path = subject_dir / "TargetGA.mat"
        control_path = subject_dir / "ControlSignal.mat"
        if not eog_path.exists() or not target_path.exists():
            print(f"[skip] {subject_dir.name}: missing EOG.mat or TargetGA.mat")
            continue

        eog_key, eog_arr = choose_main_array(matlab_numeric_items(eog_path))
        target_key, target_arr = find_target_array(matlab_numeric_items(target_path))
        control_key, control_arr = find_control_array(matlab_numeric_items(control_path))

        print(
            f"[info] {subject_dir.name}: using EOG variable '{eog_key}' shape {np.asarray(eog_arr).shape}; "
            f"Target variable '{target_key}' shape {np.asarray(target_arr).shape}; "
            f"Control variable '{control_key}' shape {np.asarray(control_arr).shape}"
        )

        eog_arr_np = np.asarray(eog_arr)
        target_arr_np = np.asarray(target_arr)

        if eog_arr_np.ndim == 2 and 2 not in np.squeeze(target_arr_np).shape:
            eog_trials = normalize_eog_trials(eog_arr_np)
            trial_targets = normalize_trial_targets(target_arr_np, expected_trials=len(eog_trials))
            subject_summary = export_subject_trials(
                subject_name=subject_dir.name,
                eog_trials=eog_trials,
                trial_targets=trial_targets,
                output_dir=output_dir,
                counters=counters,
                forward_threshold=forward_threshold,
                forward_segment_samples=forward_segment_samples,
                forward_candidates=forward_candidates,
            )
        else:
            continuous_eog = normalize_continuous_eog(eog_arr_np)
            trial_bounds = extract_trial_bounds_from_control(control_arr)
            subject_summary = export_subject_from_continuous(
                subject_name=subject_dir.name,
                continuous_eog=continuous_eog,
                trial_bounds=trial_bounds,
                target_arr=target_arr_np,
                output_dir=output_dir,
                counters=counters,
                forward_threshold=forward_threshold,
                forward_candidates=forward_candidates,
                forward_crop_ratio=forward_crop_ratio,
            )
        subject_summaries[subject_dir.name] = subject_summary

    selected_forward = export_balanced_forward_samples(
        forward_candidates=forward_candidates,
        output_dir=output_dir,
        counters=counters,
        subject_summaries=subject_summaries,
        forward_max_count=forward_max_count,
    )
    print(
        f"[info] forward candidates collected: {len(forward_candidates)}, "
        f"exported: {selected_forward}"
    )

    return dict(counters), subject_summaries


def main():
    args = parse_args()
    totals, subject_summaries = convert_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        forward_threshold=args.forward_threshold,
        forward_segment_samples=args.forward_segment_samples,
        overwrite=args.overwrite,
        forward_max_count=args.forward_max_count,
        forward_crop_ratio=args.forward_crop_ratio,
    )

    print("\nConversion summary")
    for subject_name, summary in subject_summaries.items():
        print(f"  {subject_name}: {summary}")
    print(f"  total: {totals}")


if __name__ == "__main__":
    main()
