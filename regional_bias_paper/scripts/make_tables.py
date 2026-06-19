#!/usr/bin/env python3
"""Create paper-ready CSV and LaTeX tables from all_metrics.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


METRIC_COLUMNS = [
    "dataset",
    "model",
    "finetune",
    "loss",
    "object_acc",
    "best_region",
    "worst_region",
    "best_region_acc",
    "worst_region_acc",
    "region_bias_gap",
]


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


def format_value(value: Any) -> str:
    numeric = as_float(value)
    if numeric is not None:
        return f"{numeric:.4f}"
    return "" if value is None else str(value)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fieldnames})


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("$", "\\$")
    )


def write_latex(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" * len(fieldnames)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[ht]\n")
        handle.write("\\centering\n")
        handle.write(f"\\caption{{{latex_escape(caption)}}}\n")
        handle.write(f"\\label{{{label.replace('_', '-')}}}\n")
        handle.write("\\resizebox{\\textwidth}{!}{%\n")
        handle.write(f"\\begin{{tabular}}{{{alignment}}}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join(latex_escape(field.replace("_", " ")) for field in fieldnames) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(" & ".join(latex_escape(format_value(row.get(field))) for field in fieldnames) + " \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write("}%\n")
        handle.write("\\end{table}\n")


def sorted_metrics(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda row: (row["dataset"], row["model"], row["finetune"]))


def paired_comparison(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], row["model"])][row["finetune"]] = row

    output = []
    for (dataset, model), by_setting in sorted(grouped.items()):
        fine = by_setting.get("yes")
        pre = by_setting.get("no")
        if fine is None or pre is None:
            continue
        pre_acc = as_float(pre.get("object_acc"))
        fine_acc = as_float(fine.get("object_acc"))
        pre_gap = as_float(pre.get("region_bias_gap"))
        fine_gap = as_float(fine.get("region_bias_gap"))
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "object_acc_pretrained": pre_acc,
                "object_acc_finetuned": fine_acc,
                "object_acc_delta": None if pre_acc is None or fine_acc is None else fine_acc - pre_acc,
                "region_bias_gap_pretrained": pre_gap,
                "region_bias_gap_finetuned": fine_gap,
                "region_bias_gap_delta": None if pre_gap is None or fine_gap is None else fine_gap - pre_gap,
            }
        )
    return output


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    return None if not numeric else sum(numeric) / len(numeric)


def summary(rows: List[Dict[str, str]], fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)

    output = []
    for key, group_rows in sorted(grouped.items()):
        record = {field: key[index] for index, field in enumerate(fields)}
        record["mean_object_acc"] = mean(as_float(row.get("object_acc")) for row in group_rows)
        record["mean_worst_region_acc"] = mean(as_float(row.get("worst_region_acc")) for row in group_rows)
        record["mean_best_region_acc"] = mean(as_float(row.get("best_region_acc")) for row in group_rows)
        record["mean_region_bias_gap"] = mean(as_float(row.get("region_bias_gap")) for row in group_rows)
        record["n"] = len(group_rows)
        output.append(record)
    return output


def write_table_set(output_dir: Path, latex_dir: Path, stem: str, rows: List[Dict[str, Any]], fieldnames: List[str], caption: str) -> None:
    write_csv(output_dir / f"{stem}.csv", rows, fieldnames)
    write_latex(latex_dir / f"{stem}.tex", rows, fieldnames, caption, f"tab:{stem}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to derived/processed/all_metrics.csv.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated CSV tables.")
    parser.add_argument("--latex-dir", required=True, type=Path, help="Directory for generated LaTeX tables.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)

    overall = sorted_metrics(rows)
    finetuned = sorted_metrics(row for row in rows if row["finetune"] == "yes")
    pretrained = sorted_metrics(row for row in rows if row["finetune"] == "no")
    comparison = paired_comparison(rows)
    gap_ranking = sorted(rows, key=lambda row: (as_float(row.get("region_bias_gap")) is None, as_float(row.get("region_bias_gap")) or 0))
    best_worst = sorted_metrics(rows)
    dataset_summary = summary(rows, ("dataset", "finetune"))
    model_summary = summary(rows, ("model", "finetune"))

    table_specs = [
        ("overall_metrics_table", overall, METRIC_COLUMNS, "Overall metrics by dataset, model, and evaluation setting"),
        ("finetuned_only_table", finetuned, METRIC_COLUMNS, "Finetuned evaluation metrics"),
        ("pretrained_only_table", pretrained, METRIC_COLUMNS, "Pretrained ImageNet evaluation metrics"),
        (
            "pretrained_vs_finetuned_comparison_table",
            comparison,
            [
                "dataset",
                "model",
                "object_acc_pretrained",
                "object_acc_finetuned",
                "object_acc_delta",
                "region_bias_gap_pretrained",
                "region_bias_gap_finetuned",
                "region_bias_gap_delta",
            ],
            "Pretrained versus finetuned comparison",
        ),
        ("region_bias_gap_ranking_table", gap_ranking, METRIC_COLUMNS, "Regional bias gap ranking"),
        (
            "best_worst_region_table",
            best_worst,
            [
                "dataset",
                "model",
                "finetune",
                "best_region",
                "best_region_acc",
                "worst_region",
                "worst_region_acc",
                "region_bias_gap",
            ],
            "Best and worst region metrics",
        ),
        (
            "dataset_level_summary_table",
            dataset_summary,
            ["dataset", "finetune", "mean_object_acc", "mean_worst_region_acc", "mean_best_region_acc", "mean_region_bias_gap", "n"],
            "Dataset-level summary",
        ),
        (
            "model_level_summary_table",
            model_summary,
            ["model", "finetune", "mean_object_acc", "mean_worst_region_acc", "mean_best_region_acc", "mean_region_bias_gap", "n"],
            "Model-level summary",
        ),
    ]

    for stem, table_rows, fieldnames, caption in table_specs:
        write_table_set(args.output_dir, args.latex_dir, stem, table_rows, fieldnames, caption)
        print(f"Wrote {stem} ({len(table_rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
