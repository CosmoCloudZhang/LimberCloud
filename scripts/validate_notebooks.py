#!/usr/bin/env python3
"""Validate notebook naming, runtime setup, JSON, and Python syntax."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = REPOSITORY_ROOT / "notebooks"
FORBIDDEN_CODE_TOKENS = (
    "/pscratch/",
    "/global/cfs/",
    "sys.path.insert",
    "from PROJECTION import",
    "COSMOLOGY.json",
    "SURVEY.json",
    "DENSITY.json",
    "GALAXY.json",
    "MAGNIFICATION.json",
    "ALIGNMENT.json",
)
NOTEBOOK_NAME_PATTERN = re.compile(
    r"^[A-Z][A-Za-z0-9]*(?:_[A-Z][A-Za-z0-9]*)*\.ipynb$"
)


def main() -> int:
    failures = []
    notebooks = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    for notebook_path in notebooks:
        if not NOTEBOOK_NAME_PATTERN.fullmatch(notebook_path.name):
            failures.append(
                f"{notebook_path}: filename must use Title_Case_With_Underscores"
            )

        try:
            notebook = json.loads(notebook_path.read_text())
        except json.JSONDecodeError as error:
            failures.append(f"{notebook_path}: invalid JSON: {error}")
            continue

        code_cells = [
            cell
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        for token in FORBIDDEN_CODE_TOKENS:
            if token in code:
                failures.append(f"{notebook_path}: obsolete code token {token!r}")

        for cell_number, cell in enumerate(code_cells, start=1):
            source = "".join(cell.get("source", []))
            try:
                ast.parse(source)
            except SyntaxError as error:
                failures.append(
                    f"{notebook_path}: code cell {cell_number}: "
                    f"{error.msg} at line {error.lineno}"
                )

        if "ProjectPaths.from_root" not in code:
            failures.append(f"{notebook_path}: missing ProjectPaths runtime setup")

    if failures:
        print("\n".join(failures))
        return 1

    print(f"Validated {len(notebooks)} notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
