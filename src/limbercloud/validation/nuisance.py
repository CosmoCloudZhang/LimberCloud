"""Validated loaders for the fixed fiducial nuisance tables.

Intrinsic-alignment and galaxy-bias arrays are generated once at the fiducial
cosmology and reused for every sampled cosmology. Only the background, power
and geometric prefactors follow the active sample. Every driver loads them
through this module so a stale ``eta_IA = 0.5`` array, a mismatched redshift
axis or an untagged table cannot silently enter an accepted product.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy

from limbercloud.validation.contract import (
    NUISANCE_COSMOLOGY_POLICY,
    require_accepted_eta,
)
from limbercloud.validation.cosmology import (
    FIDUCIAL_SOLVER,
    model_fingerprint,
    parameter_hash,
)


class NuisanceArtifactError(ValueError):
    """Raised when a nuisance table is stale, untagged or inconsistent."""


@dataclass(frozen=True)
class NuisanceProvenance:
    """Identity of one fixed fiducial nuisance table.

    Args:
        name: ``intrinsic_alignment``, ``galaxy_bias`` or ``magnification_bias``.
        policy: Nuisance cosmology policy recorded by the generator.
        fiducial_input_hash: Hash of the primary fiducial parameters.
        generating_model_fingerprint: Hash of parameters plus solver settings.
        solver_fingerprint: Hash of the solver settings alone.
    """

    name: str
    policy: str
    fiducial_input_hash: str
    generating_model_fingerprint: str
    solver_fingerprint: str

    def as_dict(self) -> dict[str, str]:
        """Return the JSON-ready provenance stored with a product."""

        return {
            "name": self.name,
            "policy": self.policy,
            "fiducial_input_hash": self.fiducial_input_hash,
            "generating_model_fingerprint": self.generating_model_fingerprint,
            "solver_fingerprint": self.solver_fingerprint,
        }


def require_nuisance_compatibility(table, *provenances, solver=FIDUCIAL_SOLVER) -> None:
    """Bind fixed nuisance tables to sample 0 and the selected solver.

    Args:
        table: Canonical ``CosmologyTable`` loaded by the schema-aware loader.
        provenances: Generating identities returned by the nuisance loaders.
        solver: Solver specification used for the evaluation.

    Raises:
        NuisanceArtifactError: When a recorded identity differs from the
            selected model. Sampled rows are deliberately not compared because
            these nuisance functions remain fixed at the fiducial cosmology.
    """

    solver_hash = solver.fingerprint()
    if table.solver_fingerprint != solver_hash:
        raise NuisanceArtifactError(
            "Cosmology table solver_fingerprint differs from the selected solver. "
            "Regenerate the sample table with generate_samples.py."
        )
    fiducial = table.row_dict(0)
    if not numpy.array_equal(table.is_fiducial, numpy.asarray(table.sample_id) == 0):
        raise NuisanceArtifactError("Cosmology table must mark only sample 0 as fiducial")
    expected = {
        "policy": NUISANCE_COSMOLOGY_POLICY,
        "fiducial_input_hash": parameter_hash(fiducial),
        "generating_model_fingerprint": model_fingerprint(fiducial, solver),
        "solver_fingerprint": solver_hash,
    }
    for provenance in provenances:
        for field, value in expected.items():
            if getattr(provenance, field) != value:
                raise NuisanceArtifactError(
                    f"{provenance.name} {field} differs from canonical sample 0 "
                    f"and the selected solver. Regenerate it with "
                    f"scripts/generate_config/{provenance.name}.py."
                )


def _read_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_file():
        raise NuisanceArtifactError(f"Nuisance table {candidate} does not exist")
    with candidate.open("r") as handle:
        return json.load(handle)


def _require_policy(record, name: str, key: str) -> str:
    policy = str(record.get(key, ""))
    if policy != NUISANCE_COSMOLOGY_POLICY:
        raise NuisanceArtifactError(
            f"{name} records nuisance policy {policy!r}; accepted products require "
            f"{NUISANCE_COSMOLOGY_POLICY!r}. Regenerate the table."
        )
    return policy


def _require_provenance(record, name: str, prefix: str) -> NuisanceProvenance:
    fields = {
        "fiducial_input_hash": f"{prefix}fiducial_input_hash",
        "generating_model_fingerprint": f"{prefix}generating_model_fingerprint",
        "solver_fingerprint": f"{prefix}solver_fingerprint",
    }
    missing = [key for key in fields.values() if not record.get(key)]
    if missing:
        raise NuisanceArtifactError(
            f"{name} is missing generating-model provenance: {', '.join(missing)}. "
            "Regenerate it with the current generator; adding metadata by hand does "
            "not reproduce the array."
        )
    policy_key = f"{prefix}policy" if f"{prefix}policy" in record else "nuisance_cosmology_policy"
    return NuisanceProvenance(
        name=name,
        policy=_require_policy(record, name, policy_key),
        fiducial_input_hash=str(record[fields["fiducial_input_hash"]]),
        generating_model_fingerprint=str(record[fields["generating_model_fingerprint"]]),
        solver_fingerprint=str(record[fields["solver_fingerprint"]]),
    )


def _require_redshift_axis(record, key: str, redshift, name: str) -> None:
    if key not in record:
        raise NuisanceArtifactError(
            f"{name} stores no redshift axis under {key!r}; it cannot be validated "
            "against the evaluation grid."
        )
    stored = numpy.asarray(record[key], dtype=numpy.float64)
    grid = numpy.asarray(redshift, dtype=numpy.float64)
    if stored.shape != grid.shape or not numpy.allclose(stored, grid, rtol=0.0, atol=1e-12):
        raise NuisanceArtifactError(
            f"{name} was tabulated on {stored.size} redshifts that do not match the "
            f"{grid.size} evaluation redshifts"
        )


def load_alignment(path: str | Path, redshift) -> tuple[numpy.ndarray, NuisanceProvenance]:
    """Load the intrinsic-alignment amplitude and validate its identity.

    Args:
        path: ``config/intrinsic_alignment.json``.
        redshift: Evaluation redshift grid the array must match.

    Returns:
        tuple[numpy.ndarray, NuisanceProvenance]: Float64 signed amplitude
        ``A(z)`` and its provenance.

    Raises:
        NuisanceArtifactError: When the grid, policy or provenance disagree.
        UnresolvedScienceDecision: When ``eta_IA`` is missing, unresolved or is
        not the adopted campaign value.
    """

    record = _read_json(path)
    name = "intrinsic_alignment"
    require_accepted_eta(record)
    _require_redshift_axis(record, "redshift", redshift, name)
    provenance = _require_provenance(record, name, "")
    amplitude = numpy.asarray(record["A"], dtype=numpy.float64)
    if amplitude.shape != numpy.asarray(redshift, dtype=numpy.float64).shape:
        raise NuisanceArtifactError(
            f"{name} amplitude has {amplitude.size} samples on a "
            f"{numpy.size(redshift)}-point redshift grid"
        )
    if not numpy.all(numpy.isfinite(amplitude)):
        raise NuisanceArtifactError(f"{name} amplitude contains nonfinite values")
    return amplitude, provenance


def load_galaxy_bias(
    path: str | Path,
    survey: str,
    redshift,
) -> tuple[numpy.ndarray, NuisanceProvenance]:
    """Load the linear galaxy bias of one survey and validate its identity.

    Args:
        path: ``config/galaxy_bias.json``.
        survey (str): ``Y1`` or ``Y10``.
        redshift: Evaluation redshift grid the array must match.

    Returns:
        tuple[numpy.ndarray, NuisanceProvenance]: Float64 ``b(z)`` and its
        provenance.
    """

    record = _read_json(path)
    name = "galaxy_bias"
    if survey not in record:
        raise NuisanceArtifactError(f"{name} has no entry for survey {survey!r}")
    _require_redshift_axis(record, "_redshift", redshift, name)
    provenance = _require_provenance(record, name, "_")
    bias = numpy.asarray(record[survey], dtype=numpy.float64)
    if bias.shape != numpy.asarray(redshift, dtype=numpy.float64).shape:
        raise NuisanceArtifactError(
            f"{name}[{survey}] has {bias.size} samples on a "
            f"{numpy.size(redshift)}-point redshift grid"
        )
    if not numpy.all(numpy.isfinite(bias)):
        raise NuisanceArtifactError(f"{name}[{survey}] contains nonfinite values")
    return bias, provenance


def load_magnification_slope(path: str | Path, survey: str) -> numpy.ndarray:
    """Load the per-bin number-count slope ``s`` of one survey.

    Args:
        path: ``config/magnification_bias.json``.
        survey (str): ``Y1`` or ``Y10``.

    Returns:
        numpy.ndarray: Float64 slopes, one per lens bin. The analytical response
        ``q = 5s - 2`` is applied by the assembly helpers, not here.
    """

    record = _read_json(path)
    name = "magnification_bias"
    if record.get("_quantity") != "magnification_slope_s":
        raise NuisanceArtifactError(
            f"{name} does not declare _quantity='magnification_slope_s'; the stored "
            "numbers cannot be identified as slope or response"
        )
    if survey not in record:
        raise NuisanceArtifactError(f"{name} has no entry for survey {survey!r}")
    slopes = numpy.asarray(record[survey], dtype=numpy.float64)
    if slopes.ndim != 1 or slopes.size == 0:
        raise NuisanceArtifactError(f"{name}[{survey}] must be a nonempty 1-d slope list")
    if not numpy.all(numpy.isfinite(slopes)):
        raise NuisanceArtifactError(f"{name}[{survey}] contains nonfinite values")
    return slopes
