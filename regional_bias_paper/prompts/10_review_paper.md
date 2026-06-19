# Prompt: Review Paper

Review the paper for evidence, reproducibility, and citation integrity.

Rules:

- Findings first, ordered by severity.
- Do not rewrite the paper unless asked.
- Do not invent citations or results.
- Flag unsupported empirical claims.
- Flag unsupported literature claims.
- Flag overclaiming or causal language.
- Confirm that every citation key used in LaTeX exists in `paper/references.bib`.
- Confirm that every empirical claim traces to external JSON files or derived tables.

Run before review:

```bash
python scripts/validate_results.py --input ./derived/processed/all_metrics.csv
python scripts/validate_literature.py --root .
python scripts/validate_citations.py --paper-dir ./paper --bib ./paper/references.bib
```

Report:

- citation issues
- unsupported claims
- missing TODO evidence
- reproducibility gaps
- table/figure consistency issues
- suggested fixes

