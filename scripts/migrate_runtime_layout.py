#!/usr/bin/env python3
"""Copy verified legacy inputs into the canonical runtime layout.

The command is a dry run unless ``--execute`` is supplied. It never removes
legacy files and never overwrites a different canonical file.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


CONFIGURATION_FILES = {
    "COSMOLOGY.json": "cosmology.json",
    "SURVEY.json": "survey.json",
    "DENSITY.json": "number_density.json",
    "GALAXY.json": "galaxy_bias.json",
    "MAGNIFICATION.json": "magnification_bias.json",
    "ALIGNMENT.json": "intrinsic_alignment.json",
}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path, execute: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing legacy input: {source}")

    if destination.exists():
        if not destination.is_file() or file_digest(source) != file_digest(destination):
            raise RuntimeError(f"Refusing to overwrite different destination: {destination}")
        print(f"verified {destination}")
        return

    print(f"copy {source} -> {destination}")
    if execute:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if file_digest(source) != file_digest(destination):
            raise RuntimeError(f"Checksum verification failed: {destination}")


def migrate_data(root: Path, execute: bool) -> None:
    source_root = root / "DATA"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing legacy data directory: {source_root}")

    for source in sorted(source_root.rglob("*")):
        if source.is_file():
            copy_verified(source, root / "data" / source.relative_to(source_root), execute)


def migrate_config(root: Path, execute: bool) -> None:
    for legacy_name, canonical_name in CONFIGURATION_FILES.items():
        copy_verified(
            root / "INFO" / legacy_name,
            root / "config" / canonical_name,
            execute,
        )


def create_output_directories(root: Path, execute: bool) -> None:
    directories = (
        root / "results" / "spectra" / "CCL",
        root / "results" / "spectra" / "NUMBA",
        root / "results" / "spectra" / "JAX" / "CPU",
        root / "results" / "spectra" / "JAX" / "GPU",
        root / "results" / "covariance",
        root / "results" / "validation" / "spectra",
        root / "plots",
        root / "logs",
    )
    for directory in directories:
        print(f"create {directory}")
        if execute:
            directory.mkdir(parents=True, exist_ok=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="External LimberCloud runtime root")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform verified copies; otherwise only print the plan",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    root = arguments.root.expanduser().resolve()
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError(f"Refusing unsafe runtime root: {root}")

    migrate_data(root, arguments.execute)
    migrate_config(root, arguments.execute)
    create_output_directories(root, arguments.execute)
    print("Migration verification completed." if arguments.execute else "Dry run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
