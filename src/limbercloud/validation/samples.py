"""Canonical seeded cosmology table.

Fiducial sample 0 is copied from the supplied parameters and consumes no random
draw. Sampled IDs ``1..N`` use one ``numpy.random.default_rng`` stream. Draws
are parameter-major: each sampled column is one vectorised uniform call, in
``SAMPLED_PARAMETERS`` order. ``WA`` and ``OMEGA_K`` stay fixed where they are
currently zero, and ``OMEGA_GAMMA`` stays fixed at its nonzero fiducial value
as the declared radiation convention. None of them consume a draw. Restart
loads this table by sample ID.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy

from limbercloud.validation.contract import (
    CAMPAIGN_SAMPLED_COUNT,
    FIDUCIAL_SAMPLE_ID,
    FIXED_NONZERO_PARAMETERS,
    FIXED_ZERO_PARAMETERS,
    PRIMARY_PARAMETERS,
    RELATIVE_HALF_WIDTH,
    SAMPLED_PARAMETERS,
    multiplicative_bounds,
)
from limbercloud.validation.contract import SCHEMA_VERSION as EVALUATION_SCHEMA_VERSION
from limbercloud.validation.cosmology import (
    EffectiveCosmologyKwargs,
    effective_cosmology_kwargs,
)

TABLE_SCHEMA_VERSION = "limbercloud.cosmology-table.v1"
TABLE_FILENAME = "Cosmologies.npz"
TABLE_MANIFEST_FILENAME = "Manifest.json"


class SampleTableError(ValueError):
    """Raised when a cosmology table is incomplete or inconsistent."""


@dataclass(frozen=True)
class CosmologyTable:
    """Immutable cosmology samples shared by every backend.

    Args:
        sample_id: Integer IDs. The fiducial row is 0.
        is_fiducial: True only for sample 0.
        parameter_names: Column order.
        values: Parameter values shaped ``(sample, parameter)``.
        bounds: Sorted ``(lower, upper)`` for every primary parameter.
        seed: Integer seed passed to ``default_rng``.
        bit_generator: Bit-generator class name.
        numpy_version: NumPy version that drew the table.
        content_hash: SHA-256 of the canonical array payload.
        half_width: Fractional half-width actually used for the sampled bounds.
        solver_fingerprint: Hash of the effective solver specification that
            every consumer of this table must reuse.
    """

    sample_id: numpy.ndarray
    is_fiducial: numpy.ndarray
    parameter_names: tuple[str, ...]
    values: numpy.ndarray
    bounds: dict[str, tuple[float, float]]
    seed: int
    bit_generator: str
    numpy_version: str
    content_hash: str
    half_width: float = RELATIVE_HALF_WIDTH
    solver_fingerprint: str = ""

    def row_dict(self, sample_id: int) -> dict[str, float]:
        """Return one sample by ID.

        Args:
            sample_id (int): Fiducial 0 or a sampled ID. File position is not
                an identifier; the ID column is.

        Returns:
            dict[str, float]: Parameter name to value.
        """

        ids = numpy.asarray(self.sample_id)
        matches = numpy.flatnonzero(ids == int(sample_id))
        if matches.size != 1:
            raise SampleTableError(f"Sample ID {sample_id} is not unique in the table")
        row = self.values[int(matches[0])]
        return {
            name: float(row[index]) for index, name in enumerate(self.parameter_names)
        }


def content_hash(
    sample_id: numpy.ndarray, parameter_names: tuple[str, ...], values: numpy.ndarray
) -> str:
    """Hash sample IDs, parameter order and values.

    Args:
        sample_id (numpy.ndarray): Integer sample IDs.
        parameter_names: Column names in stored order.
        values (numpy.ndarray): Float64 parameter table.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """

    import hashlib

    hasher = hashlib.sha256()
    hasher.update(b"limbercloud.cosmology-table.v1\n")
    hasher.update(",".join(parameter_names).encode("utf-8"))
    hasher.update(b"\n")
    hasher.update(numpy.ascontiguousarray(sample_id, dtype=numpy.int64).tobytes())
    hasher.update(numpy.ascontiguousarray(values, dtype=numpy.float64).tobytes())
    return hasher.hexdigest()


def _require_fixed_zero(name: str, value: float) -> None:
    if float(value) != 0.0:
        raise SampleTableError(
            f"{name}={value} is nonzero. This study keeps {name} fixed only where "
            "the fiducial value is zero and does not invent an additive sampling range."
        )


def generate_cosmology_table(
    fiducial: dict[str, float],
    *,
    seed: int,
    sampled_count: int,
    half_width: float = RELATIVE_HALF_WIDTH,
) -> CosmologyTable:
    """Draw one shared table. The fiducial row consumes no random number.

    Args:
        fiducial: Fiducial parameters including every primary name.
        seed (int): Seed for ``numpy.random.default_rng``.
        sampled_count (int): Number of non-fiducial rows. The paper campaign
            passes 1000 explicitly. Zero stores the fiducial only.
        half_width (float): Fractional half-width for nonzero sampled parameters.
            The proposed default is ±10%.

    Returns:
        CosmologyTable: Fiducial row first, then sampled IDs ``1..N``.
    """

    if int(sampled_count) < 0:
        raise SampleTableError(f"sampled_count must be >= 0; got {sampled_count}")
    missing = [name for name in PRIMARY_PARAMETERS if name not in fiducial]
    if missing:
        raise SampleTableError(f"Fiducial parameters missing: {', '.join(missing)}")

    bounds: dict[str, tuple[float, float]] = {}
    for name in FIXED_ZERO_PARAMETERS:
        _require_fixed_zero(name, float(fiducial[name]))
        bounds[name] = (0.0, 0.0)
    for name in FIXED_NONZERO_PARAMETERS:
        value = float(fiducial[name])
        bounds[name] = (value, value)
    for name in SAMPLED_PARAMETERS:
        value = float(fiducial[name])
        if value == 0.0:
            raise SampleTableError(
                f"Sampled parameter {name} is zero. Refusing a multiplicative range; "
                "declare it fixed explicitly if that is the scientific choice."
            )
        bounds[name] = multiplicative_bounds(value, half_width)

    rng = numpy.random.default_rng(int(seed))
    row_count = int(sampled_count) + 1
    values = numpy.empty((row_count, len(PRIMARY_PARAMETERS)), dtype=numpy.float64)
    for column, name in enumerate(PRIMARY_PARAMETERS):
        values[0, column] = float(fiducial[name])
        if name in FIXED_ZERO_PARAMETERS or name in FIXED_NONZERO_PARAMETERS:
            values[1:, column] = float(fiducial[name])
            continue
        low, high = bounds[name]
        if sampled_count:
            values[1:, column] = rng.uniform(low, high, size=int(sampled_count))

    from limbercloud.validation.cosmology import FIDUCIAL_SOLVER

    sample_id = numpy.arange(row_count, dtype=numpy.int64)
    is_fiducial = sample_id == FIDUCIAL_SAMPLE_ID
    names = tuple(PRIMARY_PARAMETERS)
    return CosmologyTable(
        sample_id=sample_id,
        is_fiducial=is_fiducial,
        parameter_names=names,
        values=values,
        bounds=bounds,
        seed=int(seed),
        bit_generator=type(rng.bit_generator).__name__,
        numpy_version=numpy.__version__,
        content_hash=content_hash(sample_id, names, values),
        half_width=float(half_width),
        solver_fingerprint=FIDUCIAL_SOLVER.fingerprint(),
    )


def evaluation_sample_ids(*, sample_count: int | None) -> list[int]:
    """List sample IDs for one evaluation request.

    Sample 0, the fiducial, is always included. ``sample_count`` adds IDs
    1..N after it.

    Args:
        sample_count: Extra rows after the fiducial. ``None`` and ``0`` both
            mean the fiducial alone.

    Returns:
        list[int]: ``[0]`` or ``[0, 1, ..., N]``.
    """

    from limbercloud.experiments.sample_controls import resolve_sample_count

    count = resolve_sample_count(sample_count)
    return [FIDUCIAL_SAMPLE_ID, *range(1, count + 1)]


def assert_campaign_request(sample_ids: list[int]) -> None:
    """Check the explicit paper-campaign ID list.

    Args:
        sample_ids: IDs a runner is about to evaluate.

    Raises:
        SampleTableError: When the list is not fiducial 0 plus 1–1000.
    """

    expected = [FIDUCIAL_SAMPLE_ID, *range(1, CAMPAIGN_SAMPLED_COUNT + 1)]
    if sample_ids != expected:
        raise SampleTableError(
            "The paper campaign must request fiducial sample 0 plus sampled IDs "
            f"1–{CAMPAIGN_SAMPLED_COUNT}"
        )


def select_rows(table: CosmologyTable, sample_ids: list[int]) -> list[dict[str, float]]:
    """Load rows by sample ID. This never draws new parameters.

    Args:
        table: Persisted cosmology table.
        sample_ids: IDs to return, in the requested order.

    Returns:
        list[dict[str, float]]: One parameter dictionary per requested ID.
    """

    return [table.row_dict(sample_id) for sample_id in sample_ids]


def ccl_cosmology_kwargs(row: dict[str, float]) -> EffectiveCosmologyKwargs:
    """Build the CCL constructor arguments from one table row.

    The settings come from :mod:`limbercloud.validation.cosmology`, so the
    sampled constructor and the fixed nuisance generators share one effective
    model: explicit ``Omega_g`` and CAMB ``kmax=100``.

    Args:
        row: Parameter dictionary from ``CosmologyTable.row_dict``.

    Returns:
        EffectiveCosmologyKwargs: Keyword arguments for ``pyccl.Cosmology``,
        excluding any fresh random draw.
    """

    return effective_cosmology_kwargs(row)


def sampled_parameter_rows(
    *,
    sample_count: int,
    sample_table: str | Path | None,
) -> list[dict[str, float]]:
    """Return the fiducial row, then ``sample_count`` sampled rows.

    See ``timing_loop_rows``. The name is the driver-facing wrapper.
    """

    return timing_loop_rows(sample_count=sample_count, sample_table=sample_table)


def timing_loop_rows(
    *,
    sample_count: int,
    sample_table: str | Path | None,
) -> list[dict[str, float]]:
    """Return the cosmologies one timing driver evaluates.

    Index 0 is the fiducial. The following rows are sample IDs ``1..N``.
    The cumulative timing file records the sampled rows only; the fiducial is
    evaluated first and is excluded from those checkpoints.

    Args:
        sample_count (int): Resolved count of rows after the fiducial.
        sample_table: Directory or ``Cosmologies.npz`` path. Required for the
            fiducial as well as for any extra rows.

    Returns:
        list[dict[str, float]]: Row 0, then rows ``1..sample_count``.
    """

    if sample_table is None:
        raise SampleTableError(
            "Unseeded cosmology draws are retired. Pass --sample-table pointing at "
            "a canonical Cosmologies.npz directory. Sample 0 of that table is the "
            "fiducial, and sample IDs are never re-drawn."
        )
    table = load_cosmology_table(sample_table)
    return select_rows(table, [0, *range(1, int(sample_count) + 1)])


def _table_directory(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate
    if candidate.name == TABLE_FILENAME:
        return candidate.parent
    raise SampleTableError(
        f"Cosmology table path {candidate} is not a directory or {TABLE_FILENAME}"
    )


def save_cosmology_table(directory: str | Path, table: CosmologyTable) -> Path:
    """Write ``Cosmologies.npz`` and publish ``Manifest.json`` last.

    Args:
        directory: Destination directory. Created if needed.
        table: Table to persist.

    Returns:
        Path: Directory containing the table and its manifest.
    """

    from limbercloud.io.artifacts import publish_file, sha256_file

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = destination / TABLE_FILENAME
    temporary = destination / f".{TABLE_FILENAME}.{os.getpid()}.partial.npz"
    numpy.savez(
        temporary,
        sample_id=numpy.ascontiguousarray(table.sample_id, dtype=numpy.int64),
        is_fiducial=numpy.ascontiguousarray(table.is_fiducial, dtype=numpy.bool_),
        parameter_names=numpy.asarray(table.parameter_names),
        values=numpy.ascontiguousarray(table.values, dtype=numpy.float64),
    )
    # numpy.savez appends .npz when the name does not already end with it.
    written = (
        temporary
        if temporary.is_file()
        else temporary.with_suffix(temporary.suffix + ".npz")
    )
    if not written.is_file():
        raise SampleTableError(
            f"Failed to write temporary cosmology table in {destination}"
        )
    published = publish_file(written, arrays)
    manifest = {
        "schema_version": TABLE_SCHEMA_VERSION,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "seed": table.seed,
        "bit_generator": table.bit_generator,
        "numpy_version": table.numpy_version,
        "rng": "numpy.random.default_rng",
        "draw_policy": (
            "Fiducial sample 0 copies the supplied parameters and consumes no draw. "
            "Each sampled parameter then consumes one vectorised uniform draw, in "
            "SAMPLED_PARAMETERS order. WA, OMEGA_K and OMEGA_GAMMA are fixed and "
            "consume no draw. Restart loads sample IDs from this file."
        ),
        "half_width": float(table.half_width),
        "solver_fingerprint": table.solver_fingerprint,
        "parameter_names": list(table.parameter_names),
        "sampled_parameters": list(SAMPLED_PARAMETERS),
        "fixed_zero_parameters": list(FIXED_ZERO_PARAMETERS),
        "fixed_nonzero_parameters": list(FIXED_NONZERO_PARAMETERS),
        "bounds": {
            name: {"lower": low, "upper": high}
            for name, (low, high) in table.bounds.items()
        },
        "sample_id_first": int(table.sample_id[0]),
        "sample_id_last": int(table.sample_id[-1]),
        "sampled_count": int(table.sample_id.size - 1),
        "content_hash": table.content_hash,
        "products": {
            TABLE_FILENAME: {"sha256": sha256_file(published)},
        },
    }
    manifest_path = destination / TABLE_MANIFEST_FILENAME
    temporary_manifest = destination / f".{TABLE_MANIFEST_FILENAME}.partial"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    publish_file(temporary_manifest, manifest_path)
    return destination


def load_cosmology_table(path: str | Path) -> CosmologyTable:
    """Load a completed cosmology table and check its manifest.

    Args:
        path: Table directory or ``Cosmologies.npz`` path.

    Returns:
        CosmologyTable: Validated table. Incomplete manifests are rejected.
    """

    from limbercloud.io.artifacts import sha256_file

    directory = _table_directory(path)
    manifest_path = directory / TABLE_MANIFEST_FILENAME
    arrays_path = directory / TABLE_FILENAME
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise SampleTableError(f"Cosmology table in {directory} is incomplete")
    manifest = json.loads(manifest_path.read_text())
    for key, expected in (
        ("schema_version", TABLE_SCHEMA_VERSION),
        ("evaluation_schema_version", EVALUATION_SCHEMA_VERSION),
    ):
        if manifest.get(key) != expected:
            raise SampleTableError(
                f"Cosmology table {key}={manifest.get(key)!r} is unsupported; "
                f"expected {expected!r}. Regenerate with generate_samples.py."
            )
    if manifest.get("status") != "complete":
        raise SampleTableError(
            f"Cosmology table manifest in {directory} is not complete"
        )
    recorded = manifest.get("products", {}).get(TABLE_FILENAME, {}).get("sha256")
    actual = sha256_file(arrays_path)
    if recorded != actual:
        raise SampleTableError("Cosmology table checksum does not match its manifest")
    with numpy.load(arrays_path, allow_pickle=False) as payload:
        sample_id = numpy.asarray(payload["sample_id"], dtype=numpy.int64)
        is_fiducial = numpy.asarray(payload["is_fiducial"], dtype=numpy.bool_)
        parameter_names = tuple(str(name) for name in payload["parameter_names"])
        values = numpy.asarray(payload["values"], dtype=numpy.float64)
    digest = content_hash(sample_id, parameter_names, values)
    if digest != manifest.get("content_hash"):
        raise SampleTableError(
            "Cosmology table content hash does not match its manifest"
        )
    bounds = {
        name: (float(item["lower"]), float(item["upper"]))
        for name, item in manifest["bounds"].items()
    }
    return CosmologyTable(
        sample_id=sample_id,
        is_fiducial=is_fiducial,
        parameter_names=parameter_names,
        values=values,
        bounds=bounds,
        seed=int(manifest["seed"]),
        bit_generator=str(manifest["bit_generator"]),
        numpy_version=str(manifest["numpy_version"]),
        content_hash=digest,
        half_width=float(manifest.get("half_width", RELATIVE_HALF_WIDTH)),
        solver_fingerprint=str(manifest.get("solver_fingerprint", "")),
    )
