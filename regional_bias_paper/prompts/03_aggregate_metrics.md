# Prompt: Aggregate Metrics

Aggregate external JSON results into the normalized paper table.

Rules:

- Do not modify external results.
- Do not invent missing metrics.
- Compute `region_bias_gap` as `best_region_acc - worst_region_acc` only when it is absent and both values are available.
- Keep `finetune=yes` for finetuned results and `finetune=no` for pretrained ImageNet results.

Run:

```bash
python scripts/aggregate_results.py --results-root "D:/Project/DAP_paper/results" --output-dir ./derived
python scripts/validate_results.py --input ./derived/processed/all_metrics.csv
```

Then summarize:

- number of rows generated
- missing files
- missing keys
- validation errors
- next required actions

