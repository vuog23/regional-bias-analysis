# Prompt: Make Figures

Create figures from `derived/processed/all_metrics.csv`.

Rules:

- Use matplotlib.
- Do not use seaborn.
- Do not hard-code colors.
- Do not invent missing metrics.
- Use clear labels and titles.

Run:

```bash
python scripts/make_figures.py --input ./derived/processed/all_metrics.csv --output-dir ./derived/figures
```

Expected figures:

1. object_acc by model and dataset
2. region_bias_gap by model and dataset
3. pretrained versus finetuned object_acc
4. pretrained versus finetuned region_bias_gap
5. object_acc versus region_bias_gap scatter plot

After generating figures, check whether the visual patterns are supported by the data before drafting prose.

