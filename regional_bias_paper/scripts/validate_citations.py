#!/usr/bin/env python3
"""Validate that LaTeX citation keys exist in paper/references.bib."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Set


def bib_keys(path: Path) -> Set[str]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))


def tex_citation_keys(paper_dir: Path) -> Set[str]:
    keys: Set[str] = set()
    pattern = re.compile(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}")
    for path in paper_dir.rglob("*.tex"):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            for key in match.group(1).split(","):
                clean = key.strip()
                if clean:
                    keys.add(clean)
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", required=True, type=Path, help="Paper directory containing .tex files.")
    parser.add_argument("--bib", required=True, type=Path, help="Path to paper/references.bib.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    available = bib_keys(args.bib)
    used = tex_citation_keys(args.paper_dir)
    missing = sorted(used - available)

    if missing:
        for key in missing:
            print(f"ERROR: Missing BibTeX entry for citation key: {key}")
        print(f"Citation validation failed with {len(missing)} missing key(s).")
        return 1

    print(f"Citation validation passed. Checked {len(used)} citation key(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

