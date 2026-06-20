# Agent Rules

- This paper analyzes regional bias on GeoDE and Dollar Street.
- There is no LODO.
- Do not invent citations.
- Do not invent results.
- Do not invent dataset statistics.
- Do not invent model performance.
- Every empirical claim must come from external results JSON files or derived tables.
- Every literature claim must come from literature/, notes/, or paper/references.bib.
- If evidence is missing, write TODO: evidence needed.
- Use formal academic writing.
- Avoid overclaiming.
- Prefer object_acc, worst_region_acc, best_region_acc, and region_bias_gap when discussing results.
- Do not rely only on overall object accuracy.
- Do not cite a paper unless it exists in paper/references.bib or literature/bibtex_raw.bib.
- Do not create fake BibTeX entries.
- Do not summarize a paper from memory if the paper was not found or provided.
- Use cautious language such as suggests, indicates, may reflect, is consistent with, and may be associated with.
- Do not claim causality unless the available evidence directly supports it.
- Do not move, copy, delete, overwrite, or rename external result, config, training, dataset, or checkpoint files.
- Only read external JSON, YAML, and Python files when needed for reproducibility documentation.


Literature and insight rule:

The paper should not only cite model papers. It must also cite strictly relevant work on geographic bias, dataset bias, domain shift, worst-group robustness, fairness metrics, visual dataset representation, GeoDE, Dollar Street, and evaluation beyond aggregate accuracy.

Citations should be broad but not inflated. Use many citations only when they directly support the claims being made.

For results, the agent should search for non-obvious insights using object accuracy, worst-region accuracy, best-region accuracy, region bias gap, pretrained-vs-finetuned changes, dataset differences, and model-family differences.

Creative plots are encouraged, but every insight must be tied to actual metrics and must include a caveat.
