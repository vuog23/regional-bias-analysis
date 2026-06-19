#!/usr/bin/env python3
"""Compute summary and paired disparity metrics from all_metrics.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def fmt(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else value


def write_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fieldnames})


def grouped_average(rows: List[Dict[str, str]], group_fields: Tuple[str, ...], metric: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], List[Optional[float]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(as_float(row.get(metric)))

    output: List[Dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        record = {field: key[index] for index, field in enumerate(group_fields)}
        record[f"avg_{metric}"] = mean(values)
        record["n"] = len(values)
        output.append(record)
    return output


def model_rankings(rows: List[Dict[str, str]], metric: str, ascending: bool) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Optional[float]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(as_float(row.get(metric)))

    output = []
    for model, values in grouped.items():
        output.append({"model": model, f"avg_{metric}": mean(values), "n": len(values)})
    output.sort(key=lambda row: (row[f"avg_{metric}"] is None, row[f"avg_{metric}"] if ascending else -(row[f"avg_{metric}"] or 0)))
    return output


def paired_differences(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], row["model"])][row["finetune"]] = row

    output: List[Dict[str, Any]] = []
    for (dataset, model), by_setting in sorted(grouped.items()):
        fine = by_setting.get("yes")
        pre = by_setting.get("no")
        if fine is None or pre is None:
            continue
        fine_acc = as_float(fine.get("object_acc"))
        pre_acc = as_float(pre.get("object_acc"))
        fine_gap = as_float(fine.get("region_bias_gap"))
        pre_gap = as_float(pre.get("region_bias_gap"))
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "object_acc_pretrained": pre_acc,
                "object_acc_finetuned": fine_acc,
                "object_acc_delta_finetuned_minus_pretrained": None if fine_acc is None or pre_acc is None else fine_acc - pre_acc,
                "region_bias_gap_pretrained": pre_gap,
                "region_bias_gap_finetuned": fine_gap,
                "region_bias_gap_delta_finetuned_minus_pretrained": None if fine_gap is None or pre_gap is None else fine_gap - pre_gap,
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to derived/processed/all_metrics.csv.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for summary CSV outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)

    outputs = [
        (
            "avg_object_acc_by_dataset.csv",
            grouped_average(rows, ("dataset",), "object_acc"),
            ["dataset", "avg_object_acc", "n"],
        ),
        (
            "avg_object_acc_by_model.csv",
            grouped_average(rows, ("model",), "object_acc"),
            ["model", "avg_object_acc", "n"],
        ),
        (
            "avg_region_bias_gap_by_dataset.csv",
            grouped_average(rows, ("dataset",), "region_bias_gap"),
            ["dataset", "avg_region_bias_gap", "n"],
        ),
        (
            "avg_region_bias_gap_by_model.csv",
            grouped_average(rows, ("model",), "region_bias_gap"),
            ["model", "avg_region_bias_gap", "n"],
        ),
        (
            "best_models_by_object_acc.csv",
            model_rankings(rows, "object_acc", ascending=False),
            ["model", "avg_object_acc", "n"],
        ),
        (
            "best_models_by_lowest_region_bias_gap.csv",
            model_rankings(rows, "region_bias_gap", ascending=True),
            ["model", "avg_region_bias_gap", "n"],
        ),
        (
            "finetuned_vs_pretrained_differences.csv",
            paired_differences(rows),
            [
                "dataset",
                "model",
                "object_acc_pretrained",
                "object_acc_finetuned",
                "object_acc_delta_finetuned_minus_pretrained",
                "region_bias_gap_pretrained",
                "region_bias_gap_finetuned",
                "region_bias_gap_delta_finetuned_minus_pretrained",
            ],
        ),
    ]

    for filename, table_rows, fieldnames in outputs:
        path = args.output_dir / filename
        write_rows(path, table_rows, fieldnames)
        print(f"Wrote {len(table_rows)} rows to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

