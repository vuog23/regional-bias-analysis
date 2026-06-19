# Reproducibility Checklist

## Inputs

- [ ] External results root exists.
- [ ] External global config file exists.
- [ ] External training script exists.
- [ ] Finetuned `test_results.json` files exist for all 9 models and both datasets.
- [ ] Pretrained ImageNet result JSON files exist for all 9 models and both datasets.
- [ ] No checkpoint files are copied into this workspace.

## Derived Metrics

- [ ] Run `scripts/aggregate_results.py`.
- [ ] Confirm `derived/processed/all_metrics.csv` exists.
- [ ] Confirm `derived/processed/all_metrics.json` exists.
- [ ] Run `scripts/validate_results.py`.
- [ ] Confirm `region_bias_gap` equals `best_region_acc - worst_region_acc` within tolerance.
- [ ] Confirm `finetune` values are only `yes` and `no`.

## Tables and Figures

- [ ] Run `scripts/compute_disparity_metrics.py`.
- [ ] Run `scripts/make_tables.py`.
- [ ] Run `scripts/make_figures.py`.
- [ ] Check generated CSV tables in `derived/tables/`.
- [ ] Check generated LaTeX tables in `paper/tables/`.
- [ ] Check figures in `derived/figures/`.

## Literature and Citations

- [ ] Run or update the Deep Research prompt.
- [ ] Update `literature/paper_inventory.csv`.
- [ ] Update `literature/related_work_matrix.csv`.
- [ ] Add only verified BibTeX entries to `paper/references.bib` or `literature/bibtex_raw.bib`.
- [ ] Run `scripts/validate_literature.py`.
- [ ] Run `scripts/validate_citations.py`.

## Paper Draft

- [ ] Convert section outlines into evidence-backed prose.
- [ ] Replace every TODO with either evidence or an explicit limitation.
- [ ] Compile `paper/main.tex`.
- [ ] Review tables, figures, and citations.

