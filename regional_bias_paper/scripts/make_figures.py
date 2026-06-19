#!/usr/bin/env python3
"""Create matplotlib figures from all_metrics.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def grouped_bar(rows: List[Dict[str, str]], metric: str, title: str, ylabel: str, path: Path) -> None:
    models = sorted({row["model"] for row in rows})
    series_keys = sorted({(row["dataset"], row["finetune"]) for row in rows})
    values: Dict[Tuple[str, str], Dict[str, Optional[float]]] = defaultdict(dict)
    for row in rows:
        values[(row["dataset"], row["finetune"])][row["model"]] = as_float(row.get(metric))

    x_positions = list(range(len(models)))
    width = 0.8 / max(len(series_keys), 1)

    plt.figure(figsize=(12, 6))
    for index, key in enumerate(series_keys):
        offsets = [x + (index - (len(series_keys) - 1) / 2) * width for x in x_positions]
        heights = [values[key].get(model) for model in models]
        numeric_heights = [height if height is not None else 0 for height in heights]
        label = f"{key[0]}, finetune={key[1]}"
        plt.bar(offsets, numeric_heights, width=width, label=label)

    plt.xticks(x_positions, models, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    save_current(path)


def paired_scatter(rows: List[Dict[str, str]], metric: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], row["model"])][row["finetune"]] = row

    plt.figure(figsize=(7, 7))
    all_values: List[float] = []
    for (dataset, model), by_setting in sorted(grouped.items()):
        pre = by_setting.get("no")
        fine = by_setting.get("yes")
        if pre is None or fine is None:
            continue
        x_value = as_float(pre.get(metric))
        y_value = as_float(fine.get(metric))
        if x_value is None or y_value is None:
            continue
        all_values.extend([x_value, y_value])
        plt.scatter(x_value, y_value, label=f"{dataset}-{model}", alpha=0.8)
        plt.annotate(model, (x_value, y_value), fontsize=7, alpha=0.8)

    if all_values:
        low = min(all_values)
        high = max(all_values)
        padding = (high - low) * 0.05 if high > low else 0.01
        plt.plot([low - padding, high + padding], [low - padding, high + padding], linestyle="--", linewidth=1)
        plt.xlim(low - padding, high + padding)
        plt.ylim(low - padding, high + padding)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    save_current(path)


def acc_gap_scatter(rows: List[Dict[str, str]], path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for setting in sorted({row["finetune"] for row in rows}):
        subset = [row for row in rows if row["finetune"] == setting]
        x_values = [as_float(row.get("object_acc")) for row in subset]
        y_values = [as_float(row.get("region_bias_gap")) for row in subset]
        x_clean = [x for x, y in zip(x_values, y_values) if x is not None and y is not None]
        y_clean = [y for x, y in zip(x_values, y_values) if x is not None and y is not None]
        plt.scatter(x_clean, y_clean, label=f"finetune={setting}", alpha=0.8)

    for row in rows:
        x_value = as_float(row.get("object_acc"))
        y_value = as_float(row.get("region_bias_gap"))
        if x_value is not None and y_value is not None:
            plt.annotate(row["model"], (x_value, y_value), fontsize=7, alpha=0.7)

    plt.xlabel("Object accuracy")
    plt.ylabel("Regional bias gap")
    plt.title("Object accuracy versus regional bias gap")
    plt.legend()
    plt.grid(alpha=0.3)
    save_current(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to derived/processed/all_metrics.csv.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)

    grouped_bar(
        rows,
        "object_acc",
        "Object accuracy by model, dataset, and evaluation setting",
        "Object accuracy",
        args.output_dir / "object_acc_by_model_dataset.png",
    )
    grouped_bar(
        rows,
        "region_bias_gap",
        "Regional bias gap by model, dataset, and evaluation setting",
        "Regional bias gap",
        args.output_dir / "region_bias_gap_by_model_dataset.png",
    )
    paired_scatter(
        rows,
        "object_acc",
        "Pretrained versus finetuned object accuracy",
        "Pretrained object accuracy",
        "Finetuned object accuracy",
        args.output_dir / "pretrained_vs_finetuned_object_acc.png",
    )
    paired_scatter(
        rows,
        "region_bias_gap",
        "Pretrained versus finetuned regional bias gap",
        "Pretrained regional bias gap",
        "Finetuned regional bias gap",
        args.output_dir / "pretrained_vs_finetuned_region_bias_gap.png",
    )
    acc_gap_scatter(rows, args.output_dir / "object_acc_vs_region_bias_gap.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

