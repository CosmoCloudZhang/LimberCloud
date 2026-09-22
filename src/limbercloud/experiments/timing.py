"""Small, self-describing timing products for fiducial and sampled evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy

from limbercloud.io.artifacts import timing_basename

TIMING_SCHEMA = "limbercloud.timing.v1"


@dataclass(frozen=True)
class TimingSeries:
    """Actual evaluation counts and seconds, bound to one cosmology table."""

    counts: numpy.ndarray
    seconds: numpy.ndarray
    metadata: dict


def write_timing_products(
    directory,
    configuration,
    *,
    family,
    sample_table_hash,
    counts,
    fiducial,
    sampled,
    interpolation=None,
):
    """Write sample 0 separately from cumulative samples 1..N.

    ``fiducial`` and ``sampled`` map uppercase stage names to a scalar and an
    array, respectively; the empty stage denotes the total. Zero sampled
    counts leave previous Cosmology products intact. The fiducial includes
    any first-evaluation compilation or initialization; sampled timers reset
    after that evaluation. These text files are replaced in place.
    """
    if not sample_table_hash:
        raise ValueError("Timing products require the sample-table content hash")
    if set(fiducial) != set(sampled):
        raise ValueError("Fiducial and sampled timing stages must agree")
    counts = numpy.asarray(counts)
    if counts.ndim != 1 or (
        counts.size
        and (
            not numpy.all(numpy.isfinite(counts))
            or numpy.any(counts < 1)
            or numpy.any(counts != numpy.floor(counts))
            or numpy.any(numpy.diff(counts) <= 0)
        )
    ):
        raise ValueError(
            "Sampled checkpoint counts must be increasing positive integers"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    products = []
    for population in ("Fiducial", "Cosmology"):
        if population == "Cosmology" and counts.size == 0:
            continue
        checkpoint_counts = numpy.array([0]) if population == "Fiducial" else counts
        for stage in fiducial:
            seconds = numpy.atleast_1d(
                fiducial[stage] if population == "Fiducial" else sampled[stage]
            ).astype(numpy.float64)
            if seconds.shape != checkpoint_counts.shape:
                raise ValueError(
                    f"{population} {stage or 'total'} timing/count mismatch"
                )
            if not numpy.all(numpy.isfinite(seconds)) or numpy.any(seconds < 0):
                raise ValueError("Timings must be finite and non-negative")
            if numpy.any(numpy.diff(seconds) < 0):
                raise ValueError("Cumulative timings must be non-decreasing")
            metadata = {
                "schema_version": TIMING_SCHEMA,
                "population": population,
                "sample_table_hash": sample_table_hash,
                "sample_id_first": 0 if population == "Fiducial" else 1,
                "sample_id_last": int(checkpoint_counts[-1]),
                "configuration": configuration,
                "family": family,
                "stage": stage or "TOTAL",
                "timing_convention": (
                    "sample 0; includes first-evaluation compilation/initialization"
                    if population == "Fiducial"
                    else "cumulative samples 1..N; excludes sample 0"
                ),
            }
            path = directory / timing_basename(
                configuration,
                f"_{stage}" if stage else "",
                interpolation,
                family=family,
                population=population,
            )
            numpy.savetxt(
                path,
                numpy.column_stack((checkpoint_counts, seconds)),
                fmt=["%d", "%.18e"],
                header=json.dumps(metadata, sort_keys=True) + "\nsample_count seconds",
            )
            products.append(path)
    return products


def load_cosmology_timing(directory, configuration, suffix="", *, family):
    """Read explicit sampled timings; never fall back to fiducial/legacy files."""
    path = Path(directory) / timing_basename(configuration, suffix, family=family)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing sampled timing product: {path}. Run with --sample-count > 0; "
            "Fiducial products are not sampled benchmarks."
        )
    try:
        with path.open() as handle:
            first_line = handle.readline()
        if not first_line.startswith("# "):
            raise ValueError("missing metadata header")
        metadata = json.loads(first_line[2:])
        data = numpy.loadtxt(path, ndmin=2)
    except (ValueError, OSError) as error:
        raise ValueError(f"Invalid sampled timing product {path}: {error}") from error
    expected = {
        "schema_version": TIMING_SCHEMA,
        "population": "Cosmology",
        "configuration": configuration,
        "family": family,
        "stage": suffix.removeprefix("_") or "TOTAL",
        "sample_id_first": 1,
        "timing_convention": "cumulative samples 1..N; excludes sample 0",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Timing product {path}: incompatible {key}")
    if not metadata.get("sample_table_hash"):
        raise ValueError(f"Timing product {path}: missing sample-table hash")
    if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] == 0:
        raise ValueError(f"Timing product {path}: expected count/seconds columns")
    counts, seconds = data.T
    if (
        not numpy.all(numpy.isfinite(data))
        or numpy.any(counts < 1)
        or numpy.any(counts != numpy.floor(counts))
        or numpy.any(numpy.diff(counts) <= 0)
        or numpy.any(seconds < 0)
        or numpy.any(numpy.diff(seconds) < 0)
        or metadata.get("sample_id_last") != int(counts[-1])
    ):
        raise ValueError(f"Timing product {path}: invalid counts or cumulative seconds")
    return TimingSeries(counts.astype(numpy.int64), seconds, metadata)


def require_matching_timings(*series):
    """Require identical table selection and actual counts for a comparison."""
    if not series:
        raise ValueError("A timing comparison requires at least one series")
    reference = series[0]
    for candidate in series[1:]:
        if not numpy.array_equal(candidate.counts, reference.counts):
            raise ValueError(
                "Benchmark products have different sampled checkpoint counts"
            )
        for key in ("sample_table_hash", "sample_id_first", "sample_id_last"):
            if candidate.metadata[key] != reference.metadata[key]:
                raise ValueError(f"Benchmark products have different {key}")
    return reference.counts
