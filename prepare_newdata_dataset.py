import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


CLASS_NAMES = ["blink", "down", "left", "right", "up"]
PAIR_PATTERN = re.compile(r"^(?P<label>[A-Za-z]+)(?P<index>\d+)(?P<channel>[hv])$")
DEFAULT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "newData"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "dataset"


def read_signal(path: Path) -> list[float]:
    values = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                values.append(float(text))
            except ValueError as exc:
                raise ValueError(f"{path} line {line_no} is not numeric: {text!r}") from exc
    if not values:
        raise ValueError(f"{path} is empty")
    return values


def collect_pairs(source_dir: Path):
    grouped = defaultdict(dict)
    skipped = []

    for path in sorted(source_dir.glob("*.txt")):
        match = PAIR_PATTERN.match(path.stem)
        if not match:
            skipped.append({"file": path.name, "reason": "filename does not match <label><index><h|v>"})
            continue

        label = match.group("label").lower()
        if label not in CLASS_NAMES:
            skipped.append({"file": path.name, "reason": f"unsupported label {label!r}"})
            continue

        sample_id = f"{label}{int(match.group('index')):02d}"
        channel = match.group("channel")
        grouped[(label, sample_id)][channel] = path

    complete_pairs = []
    for (label, sample_id), pair in sorted(grouped.items()):
        if set(pair) != {"h", "v"}:
            skipped.append(
                {
                    "file": ", ".join(sorted(p.name for p in pair.values())),
                    "reason": f"incomplete pair for {sample_id}",
                }
            )
            continue
        complete_pairs.append((label, sample_id, pair["h"], pair["v"]))

    return complete_pairs, skipped


def convert_dataset(source_dir: Path, output_dir: Path, overwrite: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"source_dir": str(source_dir), "output_dir": str(output_dir), "classes": {}, "skipped": []}

    pairs, skipped = collect_pairs(source_dir)
    summary["skipped"].extend(skipped)

    for label in CLASS_NAMES:
        (output_dir / label).mkdir(parents=True, exist_ok=True)
        summary["classes"][label] = {"samples": 0}

    for label, sample_id, h_path, v_path in pairs:
        horizontal = read_signal(h_path)
        vertical = read_signal(v_path)
        if len(horizontal) != len(vertical):
            summary["skipped"].append(
                {
                    "file": f"{h_path.name}, {v_path.name}",
                    "reason": f"length mismatch: h={len(horizontal)}, v={len(vertical)}",
                }
            )
            continue

        df = pd.DataFrame(
            {
                "horizontal": horizontal,
                "vertical": vertical,
            }
        )
        out_path = output_dir / label / f"{sample_id}.csv"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} already exists. Use --overwrite to replace it.")
        df.to_csv(out_path, index=False)
        summary["classes"][label]["samples"] += 1

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert paired newData text files into 2-column EOG CSV samples."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"source-dir not found: {args.source_dir}")

    summary = convert_dataset(args.source_dir, args.output_dir, args.overwrite)
    summary_path = args.output_dir / "conversion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Converted dataset written to: {args.output_dir}")
    for label in CLASS_NAMES:
        print(f"  {label:<5} -> {summary['classes'][label]['samples']} samples")
    if summary["skipped"]:
        print("Skipped files:")
        for item in summary["skipped"]:
            print(f"  - {item['file']}: {item['reason']}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
