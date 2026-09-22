"""Verify that OneCovariance can start in the interpreter that will run it.

A separate source directory is not a separately selected interpreter. The
covariance launcher invokes the external ``covariance.py`` with whichever
Python it activated, so this check resolves that executable's import-time
dependencies in that same interpreter before an expensive preparation step.

The module list is read from the source with ``ast``; nothing from the external
checkout is executed here, so an unrelated runtime failure inside OneCovariance
cannot masquerade as a dependency result.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path


def imported_modules(path: Path) -> set[str]:
    """Return the top-level modules imported by one Python file.

    Args:
        path: Source file to parse.

    Returns:
        set[str]: Top-level module names. Relative imports are skipped because
        they resolve inside the external checkout itself.
    """

    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def main(script: str) -> int:
    """Report the executable, the interpreter and any missing dependency.

    Args:
        script (str): Path to the external ``covariance.py``.

    Returns:
        int: Zero when every import-time dependency resolves.
    """

    path = Path(script)
    if not path.is_file():
        print(f"OneCovariance executable is missing: {path}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(path.parent))
    missing = sorted(
        name for name in imported_modules(path) if importlib.util.find_spec(name) is None
    )
    print(f"OneCovariance executable: {path}")
    print(f"OneCovariance interpreter: {sys.executable}")
    if missing:
        print(
            "Missing OneCovariance dependencies in this interpreter: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "Install them in the selected research environment, or set "
            "LIMBERCLOUD_ONECOVARIANCE_PYTHON to a deliberately separate "
            "environment and record that choice.",
            file=sys.stderr,
        )
        return 1
    print("OneCovariance dependency check passed")
    return 0


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="OneCovariance startup check")
    PARSER.add_argument("script", help="Path to the external covariance.py")
    raise SystemExit(main(PARSER.parse_args().script))
