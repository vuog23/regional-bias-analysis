# Prompt: Draft Results

Draft the Results section from generated tables only.

Allowed empirical sources:

- `derived/processed/all_metrics.csv`
- `derived/tables/*.csv`
- generated figures in `derived/figures/`

Rules:

- Do not invent results.
- Do not infer causality.
- Prefer object_acc, worst_region_acc, best_region_acc, and region_bias_gap.
- State exact model, dataset, and evaluation setting for every result.
- Use cautious language for patterns.
- Write `TODO: evidence needed` if a table or metric is missing.

Cover:

- object accuracy
- best region
- worst region
- best-region accuracy
- worst-region accuracy
- regional bias gap
- pretrained versus finetuned comparisons

