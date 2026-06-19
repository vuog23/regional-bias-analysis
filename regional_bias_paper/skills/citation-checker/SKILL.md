# name

citation-checker

# description

Validate citation keys, BibTeX provenance, and literature-claim traceability for the paper workspace.

# instructions

- Do not invent citations.
- Do not create fake BibTeX entries.
- Do not cite a paper unless it exists in `paper/references.bib` or `literature/bibtex_raw.bib`.
- Do not summarize a paper from memory if it was not found or provided.
- If citation evidence is missing, write `TODO: evidence needed`.
- Run or replicate `scripts/validate_citations.py` to check LaTeX citation keys.
- Run or replicate `scripts/validate_literature.py` to check citation-bank keys.
- Report missing citation keys clearly.
- Recommend adding BibTeX only after source metadata is verified.

