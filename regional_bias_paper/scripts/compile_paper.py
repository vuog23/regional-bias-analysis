#!/usr/bin/env python3
"""Compile paper/main.tex with latexmk."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def default_paper_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "paper"


def print_latex_errors(log_path: Path) -> None:
    if not log_path.exists():
        print(f"ERROR: LaTeX failed and no log file was found at {log_path}")
        return

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    error_blocks: list[str] = []
    for index, line in enumerate(lines):
        if line.startswith("!"):
            start = max(0, index - 2)
            end = min(len(lines), index + 8)
            error_blocks.append("\n".join(lines[start:end]))

    if error_blocks:
        print("LaTeX error excerpts:")
        for block in error_blocks:
            print("-" * 72)
            print(block)
    else:
        print("LaTeX failed. Last 80 log lines:")
        print("-" * 72)
        print("\n".join(lines[-80:]))


def run_compiler(command: list[str], paper_dir: Path) -> subprocess.CompletedProcess[str]:
    print(f"Running in {paper_dir}: {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=paper_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def run_direct_pdf_build(paper_dir: Path) -> subprocess.CompletedProcess[str]:
    """Fallback for local MiKTeX installs where latexmk exists but Perl does not."""
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if pdflatex is None:
        return subprocess.CompletedProcess(["pdflatex"], 1, "pdflatex was not found on PATH.\n")

    commands: list[list[str]] = [
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]
    if bibtex is not None:
        commands.append([bibtex, "main"])
    commands.extend(
        [
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ]
    )

    combined_output: list[str] = []
    last_completed: subprocess.CompletedProcess[str] | None = None
    for command in commands:
        last_completed = run_compiler(command, paper_dir)
        if last_completed.stdout:
            combined_output.append(last_completed.stdout)
        if last_completed.returncode != 0:
            return subprocess.CompletedProcess(
                command,
                last_completed.returncode,
                "\n".join(combined_output),
            )

    return subprocess.CompletedProcess(
        commands[-1],
        0,
        "\n".join(combined_output),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=default_paper_dir(),
        help="Directory containing main.tex. Defaults to this workspace's paper/ directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parents[1]
    paper_dir = args.paper_dir.resolve()
    main_tex = paper_dir / "main.tex"
    pdf_path = paper_dir / "main.pdf"

    if not main_tex.exists():
        print(f"ERROR: Missing LaTeX entrypoint: {main_tex}")
        return 1

    latexmk = shutil.which("latexmk")
    if latexmk is not None:
        command = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    else:
        local_tectonic = workspace_root / ".tools" / "tectonic.exe"
        if not local_tectonic.exists():
            print("ERROR: latexmk was not found on PATH, and no local Tectonic fallback exists at .tools/tectonic.exe.")
            return 1
        command = [str(local_tectonic), "--keep-logs", "main.tex"]

    completed = run_compiler(command, paper_dir)
    if (
        completed.returncode != 0
        and "could not find the script engine 'perl'" in (completed.stdout or "").lower()
    ):
        print("latexmk is installed, but MiKTeX cannot run it because Perl is missing.")
        print("Falling back to direct pdflatex/bibtex compilation.")
        completed = run_direct_pdf_build(paper_dir)

    if completed.returncode != 0:
        print("ERROR: LaTeX compilation failed.")
        if completed.stdout:
            print("compiler output:")
            print("-" * 72)
            print(completed.stdout.strip())
        print_latex_errors(paper_dir / "main.log")
        return completed.returncode

    if not pdf_path.exists():
        print(f"ERROR: latexmk completed but expected PDF was not found: {pdf_path}")
        return 1

    try:
        print(pdf_path.relative_to(workspace_root).as_posix())
    except ValueError:
        print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
