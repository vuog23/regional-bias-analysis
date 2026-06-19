#!/usr/bin/env python3
"""Validate literature tracking files and citation-bank keys."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Set


REQUIRED_FILES = [
    "literature/paper_inventory.csv",
    "literature/related_work_matrix.csv",
    "literature/citation_bank.md",
    "literature/claim_bank.md",
    "paper/references.bib",
]


def bib_keys(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))


def citation_bank_keys(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*-\s*citation key:\s*([A-Za-z0-9:_-]+)\s*$", line)
        if match:
            keys.add(match.group(1))
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Paper workspace root.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    errors: List[str] = []

    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.exists():
            errors.append(f"Missing required literature file: {path}")

    available_keys = bib_keys(root / "paper" / "references.bib") | bib_keys(root / "literature" / "bibtex_raw.bib")
    cited_keys = citation_bank_keys(root / "literature" / "citation_bank.md")
    missing = sorted(cited_keys - available_keys)
    if missing:
        errors.append(f"Citation-bank keys missing from BibTeX files: {', '.join(missing)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"Literature validation failed with {len(errors)} issue(s).")
        return 1

    print(f"Literature validation passed. Checked {len(cited_keys)} citation-bank keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

