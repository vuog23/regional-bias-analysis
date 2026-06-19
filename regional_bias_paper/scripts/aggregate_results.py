#!/usr/bin/env python3
"""Aggregate external result JSON files into one reproducible metrics table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MODELS = ["clip", "convnext", "deit", "dino", "mae", "maxvit", "resnet", "siglip", "swin"]
DATASETS = ["dollarstreet", "geode"]
COLUMNS = [
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
]


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        warn(f"Missing file: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        warn(f"Invalid JSON in {path}: {exc}")
        return None
    if not isinstance(data, dict):
        warn(f"Expected JSON object in {path}")
        return None
    return data


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def first_present(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def prefer_explicit_or_computed(explicit_value: Any, computed_value: Optional[float]) -> Optional[float]:
    explicit_float = as_float(explicit_value)
    return computed_value if explicit_float is None else explicit_float


def test_split_or_top_level(raw: Dict[str, Any], source: Path) -> Dict[str, Any]:
    splits = raw.get("splits")
    if isinstance(splits, dict):
        test_split = splits.get("test")
        if isinstance(test_split, dict):
            return test_split
        warn(f"No 'test' split found in {source}; using top-level fields")
    return raw


def normalize_region_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    region_acc: Dict[str, float] = {}
    for region, acc in value.items():
        numeric = as_float(acc)
        if numeric is not None:
            region_acc[str(region)] = numeric
    return region_acc


def best_and_worst_from_regions(region_acc: Dict[str, float]) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[float]]:
    if not region_acc:
        return None, None, None, None
    best_region = max(region_acc, key=region_acc.get)
    worst_region = min(region_acc, key=region_acc.get)
    return best_region, worst_region, region_acc[best_region], region_acc[worst_region]


def extract_metrics(
    raw: Dict[str, Any],
    dataset: str,
    model: str,
    finetune: str,
    source: Path,
) -> Dict[str, Any]:
    metrics = test_split_or_top_level(raw, source)

    region_map = normalize_region_map(
        first_present(
            metrics,
            [
                "region_object_acc",
                "regional_object_acc",
                "region_acc",
                "region_accuracy",
                "region_accuracies",
            ],
        )
    )

    computed_best_region, computed_worst_region, computed_best_acc, computed_worst_acc = best_and_worst_from_regions(region_map)

    best_region = first_present(metrics, ["best_region"]) or computed_best_region
    worst_region = first_present(metrics, ["worst_region"]) or computed_worst_region
    best_region_acc = prefer_explicit_or_computed(first_present(metrics, ["best_region_acc"]), computed_best_acc)
    worst_region_acc = prefer_explicit_or_computed(first_present(metrics, ["worst_region_acc"]), computed_worst_acc)
    region_bias_gap = as_float(first_present(metrics, ["region_bias_gap", "regional_bias_gap", "bias_gap"]))

    if region_bias_gap is None and best_region_acc is not None and worst_region_acc is not None:
        region_bias_gap = best_region_acc - worst_region_acc

    row = {
        "dataset": dataset,
        "model": model,
        "loss": as_float(first_present(metrics, ["loss", "test_loss"])),
        "object_acc": as_float(first_present(metrics, ["object_acc", "accuracy", "acc", "top1_acc"])),
        "best_region": best_region,
        "worst_region": worst_region,
        "best_region_acc": best_region_acc,
        "worst_region_acc": worst_region_acc,
        "region_bias_gap": region_bias_gap,
        "finetune": finetune,
    }

    for key in ["loss", "object_acc", "best_region", "worst_region", "best_region_acc", "worst_region_acc", "region_bias_gap"]:
        if row[key] is None:
            warn(f"Missing key or value '{key}' for dataset={dataset}, model={model}, finetune={finetune}, source={source}")

    return row


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in COLUMNS})


def write_json(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def aggregate(results_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for dataset in DATASETS:
        for model in MODELS:
            finetuned_path = results_root / dataset / model / "test_results.json"
            finetuned_json = load_json(finetuned_path)
            if finetuned_json is not None:
                rows.append(extract_metrics(finetuned_json, dataset, model, "yes", finetuned_path))

            pretrained_path = results_root / "imagenet" / f"{dataset}_{model}.json"
            pretrained_json = load_json(pretrained_path)
            if pretrained_json is not None:
                rows.append(extract_metrics(pretrained_json, dataset, model, "no", pretrained_path))

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path, help="External results root directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Paper workspace output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = aggregate(args.results_root)

    processed_dir = args.output_dir / "processed"
    csv_path = processed_dir / "all_metrics.csv"
    json_path = processed_dir / "all_metrics.json"
    write_csv(rows, csv_path)
    write_json(rows, json_path)

    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote {len(rows)} rows to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
