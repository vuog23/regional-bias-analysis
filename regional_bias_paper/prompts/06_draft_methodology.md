# Prompt: Draft Methodology

Draft the Methodology section from verified evidence only.

Inputs:

- `scripts/aggregate_results.py`
- `scripts/validate_results.py`
- `derived/processed/all_metrics.csv`
- `external_sources.md`
- verified config details from allowed YAML/Python files

Rules:

- Do not invent training details.
- Do not invent dataset statistics.
- Use `TODO: evidence needed` for missing setup information.
- Use formal academic writing.
- Avoid causal claims.
- Explain metrics clearly:
  - object_acc
  - best_region_acc
  - worst_region_acc
  - region_bias_gap

Output:

- Evidence-backed methodology prose.
- A list of unresolved TODOs.

