# Regional Bias Paper Workspace

This workspace supports a reproducible research paper on regional bias in computer vision models evaluated on GeoDE and Dollar Street. It is for paper analysis and writing only; it is not a model-training workspace.

## Scope

The paper compares two evaluation settings:

1. Finetuned evaluation from `results/{dataset}/{model}/test_results.json`
2. Pretrained ImageNet evaluation from `results/imagenet/{dataset}_{model}.json`

Supported datasets: `dollarstreet`, `geode`.

Supported models: `clip`, `convnext`, `deit`, `dino`, `mae`, `maxvit`, `resnet`, `siglip`, `swin`.

## External Inputs

External files are documented in `external_sources.md` and should only be read, never modified by this paper workspace.

Use `project_paths.example.yaml` as a template for local paths. Scripts accept command-line arguments so paths are not hard-coded in code.

## Common Commands

From this directory:

```bash
python scripts/aggregate_results.py --results-root "D:/Project/DAP_paper/results" --output-dir ./derived
python scripts/validate_results.py --input ./derived/processed/all_metrics.csv
python scripts/compute_disparity_metrics.py --input ./derived/processed/all_metrics.csv --output-dir ./derived/tables
python scripts/make_tables.py --input ./derived/processed/all_metrics.csv --output-dir ./derived/tables --latex-dir ./paper/tables
python scripts/make_figures.py --input ./derived/processed/all_metrics.csv --output-dir ./derived/figures
python scripts/validate_literature.py --root .
python scripts/validate_citations.py --paper-dir ./paper --bib ./paper/references.bib
python scripts/compile_paper.py --paper-dir ./paper
```

## Writing Rules

The section files in `paper/sections/` are structured outlines only. Do not write the full paper body until the derived tables, figures, and citation evidence have been checked.

Empirical claims must be traceable to `derived/processed/all_metrics.csv`, `derived/tables/`, or external result JSON files. Literature claims must be traceable to `literature/`, `notes/`, or BibTeX entries in `paper/references.bib` or `literature/bibtex_raw.bib`.

