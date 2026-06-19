# Prompt: Make Tables

Create paper-ready tables from `derived/processed/all_metrics.csv`.

Rules:

- Do not invent results.
- Use only normalized metrics from the derived table.
- Prefer object_acc, worst_region_acc, best_region_acc, and region_bias_gap.
- Report pretrained and finetuned settings separately and comparatively.

Run:

```bash
python scripts/compute_disparity_metrics.py --input ./derived/processed/all_metrics.csv --output-dir ./derived/tables
python scripts/make_tables.py --input ./derived/processed/all_metrics.csv --output-dir ./derived/tables --latex-dir ./paper/tables
```

Expected outputs:

- overall metrics table
- finetuned-only table
- pretrained-only table
- pretrained versus finetuned comparison table
- region bias gap ranking table
- best/worst region table
- dataset-level summary table
- model-level summary table

Summarize only evidence-backed table contents.

