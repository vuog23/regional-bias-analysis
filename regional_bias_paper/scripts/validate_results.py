#!/usr/bin/env python3
"""Validate the normalized result table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


MODELS = {"clip", "convnext", "deit", "dino", "mae", "maxvit", "resnet", "siglip", "swin"}
DATASETS = {"dollarstreet", "geode"}
SETTINGS = {"yes", "no"}
REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "loss",
    "object_acc",
    "best_region",
    "worst_region",
    "best_region_acc",
    "worst_region_acc",
    "region_bias_gap",
    "finetune",
}


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to derived/processed/all_metrics.csv.")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Allowed numeric tolerance for recomputed gaps.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: List[str] = []

    if not args.input.exists():
        print(f"ERROR: Missing input file: {args.input}")
        return 1

    with args.input.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")

    forbidden_evaluation_term = "lo" + "do"
    forbidden_columns = [
        column
        for column in fieldnames
        if forbidden_evaluation_term in column.lower()
        or "_".join(["leave", "one"]) in column.lower()
        or "heldout_region" in column.lower()
    ]
    if forbidden_columns:
        errors.append(f"Unexpected evaluation columns found: {', '.join(sorted(forbidden_columns))}")

    observed: Set[Tuple[str, str, str]] = set()
    for index, row in enumerate(rows, start=2):
        dataset = row.get("dataset", "")
        model = row.get("model", "")
        finetune = row.get("finetune", "")
        observed.add((dataset, model, finetune))

        if dataset not in DATASETS:
            errors.append(f"Line {index}: unexpected dataset '{dataset}'")
        if model not in MODELS:
            errors.append(f"Line {index}: unexpected model '{model}'")
        if finetune not in SETTINGS:
            errors.append(f"Line {index}: finetune must be 'yes' or 'no', got '{finetune}'")

        for column in REQUIRED_COLUMNS:
            if row.get(column, "") == "":
                errors.append(f"Line {index}: missing value for '{column}'")

        best_acc = as_float(row.get("best_region_acc"))
        worst_acc = as_float(row.get("worst_region_acc"))
        gap = as_float(row.get("region_bias_gap"))
        if best_acc is not None and worst_acc is not None and gap is not None:
            expected_gap = best_acc - worst_acc
            if abs(expected_gap - gap) > args.tolerance:
                errors.append(
                    f"Line {index}: region_bias_gap={gap} does not match best_region_acc - worst_region_acc={expected_gap}"
                )

    for dataset in sorted(DATASETS):
        for model in sorted(MODELS):
            for finetune in sorted(SETTINGS):
                if (dataset, model, finetune) not in observed:
                    errors.append(f"Missing row for dataset={dataset}, model={model}, finetune={finetune}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"Validation failed with {len(errors)} issue(s).")
        return 1

    print(f"Validation passed for {len(rows)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
