"""The run identities Phase 2 enforces, in three separate layers.

**Shared science** is what makes two products comparable: the canonical sample
table, the physics and input hashes, the eta and nuisance policy, the endpoint
policy, the radial grid and the angular operator, the probe set and the pair
orientation. Different methods share it.

**Producer workload** is what makes a resume legal: the shared science plus the
method family, device and radial order, the quadrature and timing boundary, the
numerical dependency signature and a deterministic compute-source manifest of
the code that will run. Different methods keep different producer identities.

**Execution records** are append-only facts about an attempt: scheduler job,
host, start time, output locations, checksums and status. They refer to the
producer identity; nothing in them feeds back into it. Neither do later reports
or documentation, so the fingerprint does not hash itself and a docs-only edit
cannot break a resume.

Constructors validate rather than store inconsistent fields, so an artifact
that is missing a required identity cannot silently resume as a new product.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from limbercloud.validation.contract import (
    ETA_IA_ADOPTED_VALUE,
    NN_FINAL_DIAGONAL_POLICY,
    NUISANCE_COSMOLOGY_POLICY,
    configuration_probes,
)
from limbercloud.validation.estimator import AngularContract, canonical_angular_contract
from limbercloud.validation.method import MethodIdentity

RUN_IDENTITY_SCHEMA_VERSION = "limbercloud.run-identity.v1"

SURVEYS = ("Y1", "Y10")

# Source files whose content defines what a producer computes. Reports,
# documentation and notebooks are provenance, not compute inputs.
COMPUTE_SOURCE_ROOTS = ("src/limbercloud", "experiments/spectra")
COMPUTE_SOURCE_SUFFIXES = (".py",)


class RunIdentityError(ValueError):
    """Raised when a run identity is incomplete or internally inconsistent."""


def _canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _digest(label: str, payload) -> str:
    hasher = hashlib.sha256()
    hasher.update(label.encode("utf-8"))
    hasher.update(b"\n")
    hasher.update(_canonical_json(payload).encode("utf-8"))
    return hasher.hexdigest()


def compute_source_manifest(
    root: str | Path,
    *,
    extra_files: Sequence[str | Path] = (),
) -> dict[str, str]:
    """Hash every source file that determines what a producer computes.

    Tracked, staged, unstaged and untracked files under the declared roots all
    contribute, because an uncommitted edit changes the computation just as a
    commit does. The scope is declared here rather than taken from Git status,
    which keeps the manifest reproducible from a plain checkout.

    Args:
        root: Repository root.
        extra_files: Additional files to include, relative to ``root``.

    Returns:
        dict[str, str]: Relative POSIX path to SHA-256 digest, sorted by path.
    """

    base = Path(root).resolve()
    paths: set[Path] = set()
    for relative in COMPUTE_SOURCE_ROOTS:
        directory = base / relative
        if not directory.is_dir():
            continue
        for suffix in COMPUTE_SOURCE_SUFFIXES:
            paths.update(
                candidate
                for candidate in directory.rglob(f"*{suffix}")
                if "__pycache__" not in candidate.parts
            )
    for relative in extra_files:
        candidate = base / relative
        if candidate.is_file():
            paths.add(candidate)
    manifest: dict[str, str] = {}
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[path.relative_to(base).as_posix()] = digest
    return manifest


def source_fingerprint(manifest: Mapping[str, str]) -> str:
    """Return one deterministic digest of a compute-source manifest.

    Args:
        manifest: Relative path to file digest.

    Returns:
        str: Hexadecimal SHA-256 over the canonical serialisation.
    """

    if not manifest:
        raise RunIdentityError("A compute-source manifest cannot be empty")
    return _digest("limbercloud.compute-source", dict(sorted(manifest.items())))


@dataclass(frozen=True)
class SharedScience:
    """What two products must agree on before they may be compared.

    Args:
        survey: ``Y1`` or ``Y10``.
        configuration: ``Single``, ``Double`` or ``Triple``.
        sample_table_hash: Content hash of the canonical cosmology table.
        solver_fingerprint: Effective cosmology and solver specification.
        nuisance_hashes: Generating-model fingerprint of each nuisance table.
        radial_grid: Radial grid description, including node count and range.
        angular_contract: The shared 21-node, 20-band operator.
        eta_ia: Adopted intrinsic-alignment slope.
        nuisance_policy: How nuisance tables vary across samples.
        endpoint_policy: NN terminal-basis policy.
        pair_orientation: Declared leg order of each probe.
    """

    survey: str
    configuration: str
    sample_table_hash: str
    solver_fingerprint: str
    nuisance_hashes: Mapping[str, str]
    radial_grid: Mapping[str, object]
    angular_contract: AngularContract = field(default_factory=canonical_angular_contract)
    eta_ia: float = ETA_IA_ADOPTED_VALUE
    nuisance_policy: str = NUISANCE_COSMOLOGY_POLICY
    endpoint_policy: str = NN_FINAL_DIAGONAL_POLICY
    pair_orientation: str = "TE is lens-first and source-second"

    def __post_init__(self) -> None:
        if self.survey not in SURVEYS:
            raise RunIdentityError(f"Unknown survey {self.survey!r}; expected one of: Y1, Y10")
        try:
            configuration_probes(self.configuration)
        except ValueError as error:
            raise RunIdentityError(str(error)) from error
        for name, value in (
            ("sample_table_hash", self.sample_table_hash),
            ("solver_fingerprint", self.solver_fingerprint),
        ):
            if not value:
                raise RunIdentityError(f"Shared science requires {name}")
        if not self.nuisance_hashes:
            raise RunIdentityError(
                "Shared science requires the generating-model fingerprint of every "
                "nuisance table"
            )
        missing = [name for name, value in self.nuisance_hashes.items() if not value]
        if missing:
            raise RunIdentityError(
                "Nuisance fingerprints are empty for: " + ", ".join(sorted(missing))
            )
        if float(self.eta_ia) != ETA_IA_ADOPTED_VALUE:
            raise RunIdentityError(
                f"The accepted campaign eta_IA is {ETA_IA_ADOPTED_VALUE}; got {self.eta_ia}"
            )
        if self.nuisance_policy != NUISANCE_COSMOLOGY_POLICY:
            raise RunIdentityError(
                f"Accepted products use nuisance policy {NUISANCE_COSMOLOGY_POLICY!r}"
            )
        if self.endpoint_policy != NN_FINAL_DIAGONAL_POLICY:
            raise RunIdentityError(
                f"Accepted products use endpoint policy {NN_FINAL_DIAGONAL_POLICY!r}"
            )
        for key in ("node_count", "minimum", "maximum"):
            if key not in self.radial_grid:
                raise RunIdentityError(f"The radial grid description requires {key!r}")

    @property
    def probes(self) -> tuple[str, ...]:
        """Return the probes this configuration publishes."""

        return configuration_probes(self.configuration)

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-ready shared-science description."""

        return {
            "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
            "survey": self.survey,
            "configuration": self.configuration,
            "probes": list(self.probes),
            "sample_table_hash": self.sample_table_hash,
            "solver_fingerprint": self.solver_fingerprint,
            "nuisance_hashes": dict(sorted(self.nuisance_hashes.items())),
            "radial_grid": dict(sorted(self.radial_grid.items())),
            "angular_operator": self.angular_contract.as_dict(),
            "eta_ia": float(self.eta_ia),
            "nuisance_policy": self.nuisance_policy,
            "endpoint_policy": self.endpoint_policy,
            "pair_orientation": self.pair_orientation,
        }

    def fingerprint(self) -> str:
        """Return the digest two methods must share to be comparable."""

        return _digest("limbercloud.shared-science", self.as_dict())


@dataclass(frozen=True)
class ProducerWorkload:
    """What a resume must match exactly before it may continue a run.

    Args:
        shared: The shared-science contract.
        method: Validated family, device and radial order.
        requested_sample_ids: Immutable requested IDs, in evaluation order.
        quadrature: Quadrature description of the numerical policy.
        timing_boundary: Where the compute timer starts and stops.
        source_manifest: Relative path to digest for every compute source file.
        dependency_signature: Versions and build settings of the numerical
            runtimes. A changed runtime cannot resume under an unchanged
            source digest.
    """

    shared: SharedScience
    method: MethodIdentity
    requested_sample_ids: tuple[int, ...]
    quadrature: Mapping[str, object]
    timing_boundary: str
    source_manifest: Mapping[str, str]
    dependency_signature: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.method, MethodIdentity):
            raise RunIdentityError("method must be a validated MethodIdentity")
        identifiers = tuple(int(value) for value in self.requested_sample_ids)
        if not identifiers:
            raise RunIdentityError("A workload must request at least one sample ID")
        if len(set(identifiers)) != len(identifiers):
            raise RunIdentityError("Requested sample IDs contain duplicates")
        if any(value < 0 for value in identifiers):
            raise RunIdentityError("Sample IDs must be non-negative")
        object.__setattr__(self, "requested_sample_ids", identifiers)
        if not self.timing_boundary:
            raise RunIdentityError("A workload must declare its timing boundary")
        if not self.quadrature:
            raise RunIdentityError("A workload must declare its quadrature policy")
        if not self.dependency_signature:
            raise RunIdentityError(
                "A workload must record the numerical dependency signature; a changed "
                "runtime must not resume under an unchanged source digest"
            )
        # Raises for an empty manifest.
        source_fingerprint(self.source_manifest)

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-ready workload description."""

        return {
            "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
            "shared_science_fingerprint": self.shared.fingerprint(),
            "method": self.method.as_dict(),
            "requested_sample_ids": list(self.requested_sample_ids),
            "quadrature": dict(sorted(self.quadrature.items())),
            "timing_boundary": self.timing_boundary,
            "source_fingerprint": source_fingerprint(self.source_manifest),
            "dependency_signature": dict(sorted(self.dependency_signature.items())),
        }

    def fingerprint(self) -> str:
        """Return the digest a resume must reproduce exactly."""

        return _digest("limbercloud.producer-workload", self.as_dict())

    def is_comparable_with(self, other: "ProducerWorkload") -> bool:
        """Return whether two producers computed comparable shared science.

        Args:
            other: Another producer workload.

        Returns:
            bool: True when the shared-science fingerprints agree. Different
            methods are expected to differ in everything else.
        """

        return self.shared.fingerprint() == other.shared.fingerprint()

    def may_resume(self, other: "ProducerWorkload") -> bool:
        """Return whether ``other`` may continue this exact workload.

        Args:
            other: Workload recorded by the earlier attempt.

        Returns:
            bool: True only when the full workload fingerprints agree.
        """

        return self.fingerprint() == other.fingerprint()


@dataclass(frozen=True)
class ExecutionRecord:
    """One append-only attempt. Nothing here changes a producer fingerprint.

    Args:
        producer_fingerprint: The workload this attempt belongs to.
        head_commit: Repository HEAD, recorded as provenance only.
        scheduler_job: Scheduler job identifier.
        host: Host that ran the attempt.
        process_id: Operating-system process identifier.
        started_at: ISO-8601 start time.
        output_paths: Where the attempt wrote, keyed by product name.
        output_checksums: Digests of those products.
        status: ``attempted``, ``completed`` or ``failed``.
    """

    producer_fingerprint: str
    head_commit: str
    scheduler_job: str
    host: str
    process_id: int
    started_at: str
    output_paths: Mapping[str, str] = field(default_factory=dict)
    output_checksums: Mapping[str, str] = field(default_factory=dict)
    status: str = "attempted"

    STATUSES = ("attempted", "completed", "failed")

    def __post_init__(self) -> None:
        if not self.producer_fingerprint:
            raise RunIdentityError("An execution record must name its producer workload")
        if self.status not in self.STATUSES:
            choices = ", ".join(self.STATUSES)
            raise RunIdentityError(f"Unknown status {self.status!r}; expected one of: {choices}")

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-ready execution record."""

        return {
            "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
            "producer_fingerprint": self.producer_fingerprint,
            "head_commit": self.head_commit,
            "scheduler_job": self.scheduler_job,
            "host": self.host,
            "process_id": int(self.process_id),
            "started_at": self.started_at,
            "output_paths": dict(sorted(self.output_paths.items())),
            "output_checksums": dict(sorted(self.output_checksums.items())),
            "status": self.status,
        }


def numerical_dependency_signature() -> dict[str, str]:
    """Return the installed numerical runtime versions and precision settings.

    Versions come from the installed distribution metadata, so building the
    signature never imports a site-linked or MPI-initialising backend.

    Returns:
        dict[str, str]: Versions of the packages whose builds change results,
        plus the float64 setting JAX runs under. Missing packages are recorded
        as ``unavailable`` rather than omitted, so an absent backend is visible
        instead of silently matching.
    """

    import importlib.metadata

    distributions = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pyccl": "pyccl",
        "camb": "camb",
        "numba": "numba",
        "jax": "jax",
        "jaxlib": "jaxlib",
    }
    signature: dict[str, str] = {}
    for name, distribution in distributions.items():
        try:
            signature[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            signature[name] = "unavailable"
    signature["float64"] = "enabled"
    return signature
