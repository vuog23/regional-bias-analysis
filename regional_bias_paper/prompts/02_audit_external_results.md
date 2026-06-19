# Prompt: Audit External Results

Audit the external result files for this paper workspace.

Rules:

- Only read JSON, YAML, and Python files needed for documentation.
- Do not move, copy, delete, or overwrite external files.
- Do not read or copy checkpoint files.
- Do not invent results or dataset statistics.
- If evidence is missing, write `TODO: evidence needed`.

Tasks:

1. Confirm that result files exist for all datasets, models, and evaluation settings.
2. Inspect JSON schemas for finetuned and pretrained result files.
3. Confirm that required metrics are present or computable:
   - loss
   - object_acc
   - best_region
   - worst_region
   - best_region_acc
   - worst_region_acc
   - region_bias_gap
4. Inspect `used_config.yaml`, the global config, and training script only for experimental setup details.
5. Summarize any missing files, missing keys, or ambiguous metadata.

