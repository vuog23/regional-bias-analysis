# name

result-auditor

# description

Audit external GeoDE and Dollar Street result files and derived metric tables for completeness, schema consistency, and reproducibility.

# instructions

- Treat this as a paper-analysis task, not a training task.
- Only read external JSON, YAML, and Python files needed for the audit.
- Do not move, copy, delete, overwrite, or rename external files.
- Do not read or copy checkpoint files.
- Do not invent evidence.
- Do not invent result values, model performance, dataset statistics, or missing metrics.
- If evidence is missing, write `TODO: evidence needed`.
- Check that all expected models and datasets are represented.
- Check that both pretrained and finetuned settings are represented.
- Check that `region_bias_gap` is either present or computable as `best_region_acc - worst_region_acc`.
- Prefer object_acc, worst_region_acc, best_region_acc, and region_bias_gap when summarizing findings.
- Use cautious wording and avoid causal claims.

