"""HDF5 spectrum checkpoints and run manifests.

One writer owns an artifact namespace. A sample checkpoint is written to a
temporary HDF5 file, closed, reopened and validated, then renamed on the
destination filesystem. A cross-filesystem move is not atomic: bytes staged on
another device are copied to a temporary file on the destination, checksummed,
and only then renamed. The run manifest is published last. Readers accept
completed manifests only. HDF5 in-place writes are not treated as transactions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy

from limbercloud.validation.contract import CONFIGURATION_PROBES, SPECTRA_SCHEMA_VERSION
from limbercloud.validation.estimator import EllEstimator, EstimatorMismatch
from limbercloud.validation.method import MethodIdentity

MANIFEST_STATUS_COMPLETE = "complete"

SURVEYS = ("Y1", "Y10")


class ArtifactError(ValueError):
    """Raised when an artifact is incomplete, mismatched or already owned."""


def _validate_survey_token(survey: str) -> str:
    token = str(survey).strip().upper()
    if token not in SURVEYS:
        choices = ", ".join(SURVEYS)
        raise ArtifactError(f"Unknown survey {survey!r}; expected one of: {choices}")
    return token


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        str: Hexadecimal digest of the file bytes.
    """

    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def same_filesystem(left: str | Path, right: str | Path) -> bool:
    """Return whether two existing directories share a device.

    Args:
        left: Existing directory.
        right: Existing directory.

    Returns:
        bool: True when ``stat().st_dev`` matches. A rename across devices is
        not an atomic publication.
    """

    return os.stat(left).st_dev == os.stat(right).st_dev


def publish_file(source: str | Path, destination: str | Path) -> Path:
    """Publish one finished file by a same-filesystem rename.

    Args:
        source: Temporary file that has already been closed and validated.
        destination: Final path. Its parent directory is created if needed.

    Returns:
        Path: The published destination.

    Raises:
        ArtifactError: When a cross-filesystem copy does not preserve the
        checksum, or the published bytes do not match the source digest.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(source_path)
    if same_filesystem(source_path.parent, destination_path.parent):
        os.replace(source_path, destination_path)
    else:
        staging = (
            destination_path.parent / f".{destination_path.name}.{os.getpid()}.partial"
        )
        shutil.copyfile(source_path, staging)
        if sha256_file(staging) != digest:
            staging.unlink(missing_ok=True)
            raise ArtifactError(
                f"Cross-filesystem copy to {staging} did not match the source checksum"
            )
        os.replace(staging, destination_path)
        source_path.unlink(missing_ok=True)
    if sha256_file(destination_path) != digest:
        raise ArtifactError(
            f"Published file {destination_path} failed checksum validation"
        )
    return destination_path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class NamespaceLock:
    """Exclusive writer lock for one artifact directory.

    Args:
        directory: Namespace directory. The lock file lives inside it.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.path = self.directory / ".writer.lock"
        self._held = False

    def acquire(self, *, resume_stale: bool = False) -> None:
        """Create the lock or refuse a second writer.

        Args:
            resume_stale (bool): When true, a lock whose recorded host matches
                this machine and whose PID is not alive may be removed. Locks
                from another host are never stolen.

        Raises:
            ArtifactError: When another writer holds the namespace.
        """

        self.directory.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._release_stale(resume_stale)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(self.path, flags, 0o644)
        except FileExistsError as error:
            raise ArtifactError(
                f"Artifact namespace {self.directory} already has a writer"
            ) from error
        payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname()}) + "\n"
        os.write(descriptor, payload.encode("utf-8"))
        os.close(descriptor)
        self._held = True

    def _release_stale(self, resume_stale: bool) -> None:
        try:
            recorded = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(f"Unreadable writer lock {self.path}") from error
        same_host = recorded.get("host") == socket.gethostname()
        alive = _pid_alive(int(recorded.get("pid", -1)))
        if same_host and not alive and resume_stale:
            self.path.unlink()
            return
        raise ArtifactError(
            f"Artifact namespace {self.directory} already has a writer "
            f"(host={recorded.get('host')}, pid={recorded.get('pid')})"
        )

    @property
    def is_held(self) -> bool:
        """Return whether this object currently holds the namespace lock."""

        return self._held

    def release(self) -> None:
        """Remove the lock if this process holds it."""

        if self._held and self.path.exists():
            self.path.unlink()
        self._held = False

    def __enter__(self) -> "NamespaceLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


@dataclass(frozen=True)
class ArtifactIdentity:
    """Identity that every checkpoint in one namespace must repeat.

    The family, device and radial order are validated together, so a NUMERIC
    order cannot be attached to a CCL, NUMBA or JAX product and JAX cannot omit
    its device. Supplying ``estimator`` additionally lets readers check the
    fingerprint against the coordinates actually stored in a file.

    Args:
        run_id: Run directory name.
        survey: ``Y1`` or ``Y10``.
        family: ``CCL``, ``NUMBA``, ``JAX`` or ``NUMERIC``.
        configuration: ``Single``, ``Double`` or ``Triple``.
        sample_table_hash: Content hash of the shared cosmology table.
        estimator_fingerprint: Ell-estimator fingerprint.
        device: ``CPU`` or ``GPU``. JAX must choose; other families are CPU.
        interpolation: NUMERIC order, or empty.
        estimator: Optional estimator whose fingerprint must equal
            ``estimator_fingerprint`` and whose coordinates are compared with
            the stored ``ell`` dataset.
    """

    run_id: str
    survey: str
    family: str
    configuration: str
    sample_table_hash: str
    estimator_fingerprint: str
    device: str = ""
    interpolation: str = ""
    estimator: EllEstimator | None = None

    def __post_init__(self) -> None:
        method = MethodIdentity.create(
            self.family, self.device or None, self.interpolation or None
        )
        object.__setattr__(self, "family", method.family)
        object.__setattr__(self, "device", method.device)
        object.__setattr__(self, "interpolation", method.interpolation)
        if _validate_survey_token(self.survey) != self.survey:
            object.__setattr__(self, "survey", _validate_survey_token(self.survey))
        if self.configuration not in CONFIGURATION_PROBES:
            choices = ", ".join(CONFIGURATION_PROBES)
            raise ArtifactError(
                f"Unknown configuration {self.configuration!r}; expected one of: {choices}"
            )
        if not self.sample_table_hash:
            raise ArtifactError("An artifact identity requires the sample-table hash")
        if not self.estimator_fingerprint:
            raise ArtifactError(
                "An artifact identity requires the estimator fingerprint"
            )
        if (
            self.estimator is not None
            and self.estimator.fingerprint() != self.estimator_fingerprint
        ):
            raise ArtifactError(
                "The supplied estimator does not reproduce estimator_fingerprint"
            )

    @property
    def method(self) -> MethodIdentity:
        """Return the validated family/device/order identity."""

        return MethodIdentity(self.family, self.device, self.interpolation)

    def as_dict(self) -> dict[str, str]:
        """Return the identity fields stored as HDF5 attributes."""

        return {
            "schema_version": SPECTRA_SCHEMA_VERSION,
            "run_id": self.run_id,
            "survey": self.survey,
            "family": self.family,
            "configuration": self.configuration,
            "sample_table_hash": self.sample_table_hash,
            "estimator_fingerprint": self.estimator_fingerprint,
            "device": self.device,
            "interpolation": self.interpolation,
        }


@dataclass(frozen=True)
class SampleCheckpoint:
    """One completed sample. Coefficient tensors are not stored.

    Args:
        sample_id: Stable sample ID.
        is_fiducial: True only for sample 0.
        probe: ``EE``, ``TE`` or ``TT``.
        cosmology: Parameter values for this sample.
        parameter_names: Column names matching ``cosmology``.
        ell: Multipole coordinate.
        pair_i: First bin index of each pair.
        pair_j: Second bin index of each pair.
        cl: Float64 spectrum shaped ``(ell, pair)``.
        stage_seconds: Non-overlapping stage durations in seconds.
        estimator_name: Estimator label, stored beside its fingerprint.
    """

    sample_id: int
    is_fiducial: bool
    probe: str
    cosmology: numpy.ndarray
    parameter_names: tuple[str, ...]
    ell: numpy.ndarray
    pair_i: numpy.ndarray
    pair_j: numpy.ndarray
    cl: numpy.ndarray
    stage_seconds: Mapping[str, float]
    estimator_name: str


def _require_h5py():
    try:
        import h5py
    except ImportError as error:
        raise ArtifactError(
            "h5py is required to read or write spectrum checkpoints"
        ) from error
    return h5py


def checkpoint_name(sample_id: int, probe: str) -> str:
    """Return the published shard filename for one sample and probe.

    Args:
        sample_id (int): Sample ID, formatted with six digits.
        probe (str): ``EE``, ``TE`` or ``TT``.

    Returns:
        str: ``sample_000001_EE.h5``.
    """

    return f"sample_{int(sample_id):06d}_{probe}.h5"


def _require_declared_coordinates(
    identity: ArtifactIdentity, ell: numpy.ndarray
) -> None:
    """Compare stored multipoles with the identity's declared estimator.

    Args:
        identity: Namespace identity, optionally carrying its estimator.
        ell (numpy.ndarray): Coordinates stored beside the spectra.

    Raises:
        ArtifactError: When the estimator names different coordinates. A
        fingerprint that is never compared with the real array proves nothing.
    """

    if identity.estimator is None:
        return
    try:
        identity.estimator.validate_coordinates(ell)
    except EstimatorMismatch as error:
        raise ArtifactError(str(error)) from error


def _write_checkpoint_file(
    path: Path, record: SampleCheckpoint, identity: ArtifactIdentity
) -> None:
    h5py = _require_h5py()
    cl = numpy.asarray(record.cl, dtype=numpy.float64)
    ell = numpy.asarray(record.ell, dtype=numpy.float64)
    pair_i = numpy.asarray(record.pair_i, dtype=numpy.int32)
    pair_j = numpy.asarray(record.pair_j, dtype=numpy.int32)
    if (
        cl.ndim != 2
        or cl.shape != (ell.size, pair_i.size)
        or pair_j.size != pair_i.size
    ):
        raise ArtifactError(
            f"Checkpoint cl shape {cl.shape} does not match ell {ell.size} and pairs {pair_i.size}"
        )
    cosmology = numpy.asarray(record.cosmology, dtype=numpy.float64)
    if cosmology.shape != (len(record.parameter_names),):
        raise ArtifactError("Cosmology vector does not match parameter_names")
    _require_declared_coordinates(identity, ell)
    with h5py.File(path, "w") as handle:
        for key, value in identity.as_dict().items():
            handle.attrs[key] = value
        handle.attrs["probe"] = record.probe
        handle.attrs["estimator_name"] = record.estimator_name
        handle.attrs["status"] = MANIFEST_STATUS_COMPLETE
        handle.attrs["sample_id"] = int(record.sample_id)
        handle.attrs["is_fiducial"] = bool(record.is_fiducial)
        handle.create_dataset(
            "sample_id", data=numpy.asarray(record.sample_id, dtype=numpy.int64)
        )
        handle.create_dataset(
            "is_fiducial", data=numpy.asarray(record.is_fiducial, dtype=numpy.bool_)
        )
        handle.create_dataset("cosmology", data=cosmology)
        handle.create_dataset(
            "parameter_names",
            data=numpy.asarray(
                record.parameter_names, dtype=h5py.string_dtype(encoding="utf-8")
            ),
        )
        handle.create_dataset("ell", data=ell)
        handle.create_dataset("pair_i", data=pair_i)
        handle.create_dataset("pair_j", data=pair_j)
        handle.create_dataset(
            "cl",
            data=cl,
            chunks=cl.shape,
            compression="gzip",
            compression_opts=4,
        )
        stages = handle.create_group("stage_seconds")
        for name, seconds in record.stage_seconds.items():
            stages.attrs[str(name)] = float(seconds)
        raw = handle.create_group("sampled")
        raw.create_dataset("ell", data=ell)
        raw.create_dataset("cl", data=cl)
        raw.attrs["role"] = "raw sampled spectra"
        # Bandpowers are a different estimator and are absent until requested.


def validate_checkpoint_file(
    path: str | Path, identity: ArtifactIdentity
) -> SampleCheckpoint:
    """Reopen a checkpoint and reject identity or shape mismatches.

    Args:
        path: HDF5 shard.
        identity: Expected namespace identity.

    Returns:
        SampleCheckpoint: The validated record. An attribute claiming completion
        is not accepted unless the datasets reload and match ``identity``.
    """

    h5py = _require_h5py()
    path = Path(path)
    try:
        handle = h5py.File(path, "r")
    except OSError as error:
        raise ArtifactError(f"Unreadable checkpoint {path}") from error
    with handle:
        if handle.attrs.get("status") != MANIFEST_STATUS_COMPLETE:
            raise ArtifactError(f"Checkpoint {path} is not a completed shard")
        for key, expected in identity.as_dict().items():
            actual = handle.attrs.get(key)
            if str(actual) != str(expected):
                raise ArtifactError(
                    f"Checkpoint {path} {key}={actual!r} does not match {expected!r}"
                )
        cl = numpy.asarray(handle["cl"], dtype=numpy.float64)
        ell = numpy.asarray(handle["ell"], dtype=numpy.float64)
        pair_i = numpy.asarray(handle["pair_i"], dtype=numpy.int32)
        pair_j = numpy.asarray(handle["pair_j"], dtype=numpy.int32)
        if "coefficients" in handle:
            raise ArtifactError(
                "Spectrum checkpoints must not store coefficient tensors"
            )
        if cl.shape != (ell.size, pair_i.size) or pair_j.shape != pair_i.shape:
            raise ArtifactError(f"Checkpoint {path} has inconsistent axes")
        _require_declared_coordinates(identity, ell)
        names = tuple(str(item) for item in handle["parameter_names"].asstr())
        stages = {
            str(key): float(value)
            for key, value in handle["stage_seconds"].attrs.items()
        }
        return SampleCheckpoint(
            sample_id=int(handle.attrs["sample_id"]),
            is_fiducial=bool(handle.attrs["is_fiducial"]),
            probe=str(handle.attrs["probe"]),
            cosmology=numpy.asarray(handle["cosmology"], dtype=numpy.float64),
            parameter_names=names,
            ell=ell,
            pair_i=pair_i,
            pair_j=pair_j,
            cl=cl,
            stage_seconds=stages,
            estimator_name=str(handle.attrs["estimator_name"]),
        )


def _require_lock(
    namespace_lock: NamespaceLock, namespace: str | Path | None = None
) -> None:
    """Require a held lock that owns the exact namespace being written.

    Args:
        namespace_lock: Lock supplied by the caller.
        namespace: Directory about to be written. A lock on another directory
            does not authorise writes here.

    Raises:
        ArtifactError: When no lock is held or it owns a different namespace.
    """

    if not isinstance(namespace_lock, NamespaceLock) or not namespace_lock.is_held:
        raise ArtifactError("This namespace requires the holding writer lock")
    if namespace is None:
        return
    target = Path(namespace).expanduser().resolve()
    owned = Path(namespace_lock.directory).expanduser().resolve()
    if target != owned:
        raise ArtifactError(
            f"The held writer lock owns {owned}, not the write namespace {target}"
        )


def write_sample_checkpoint(
    namespace: str | Path,
    record: SampleCheckpoint,
    identity: ArtifactIdentity,
    namespace_lock: NamespaceLock,
    *,
    staging_directory: str | Path | None = None,
) -> Path:
    """Write, validate and publish one sample checkpoint.

    Args:
        namespace: Destination run directory on the durable filesystem.
        record: Sample payload. Must not include coefficient tensors.
        identity: Namespace identity written into the file and checked on reopen.
        namespace_lock: Lock already held by this writer. A second writer cannot
            publish into the namespace.
        staging_directory: Optional directory on another filesystem. The file is
            copied to a temporary destination path before publication.

    Returns:
        Path: Published shard. A temporary or invalid file is not returned.
    """

    _require_lock(namespace_lock, namespace)
    destination_dir = Path(namespace) / "checkpoints"
    destination_dir.mkdir(parents=True, exist_ok=True)
    final_path = destination_dir / checkpoint_name(record.sample_id, record.probe)
    if final_path.exists():
        try:
            existing = validate_checkpoint_file(final_path, identity)
        except ArtifactError:
            existing = None
        if existing is not None:
            raise ArtifactError(
                f"Sample {record.sample_id} {record.probe} is already published in {final_path}"
            )
    staging_root = (
        Path(staging_directory) if staging_directory is not None else destination_dir
    )
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = staging_root / f".{final_path.name}.{os.getpid()}.partial"
    _write_checkpoint_file(temporary, record, identity)
    validate_checkpoint_file(temporary, identity)
    return publish_file(temporary, final_path)


def write_failure_record(
    namespace: str | Path,
    *,
    sample_id: int,
    probe: str,
    message: str,
    identity: ArtifactIdentity,
    namespace_lock: NamespaceLock,
) -> Path:
    """Publish a failure sidecar without drawing a replacement sample.

    Args:
        namespace: Run directory.
        sample_id (int): Sample that failed.
        probe (str): Probe being evaluated.
        message (str): Failure text stored in the sidecar.
        identity: Namespace identity.

    Returns:
        Path: Published JSON sidecar.
    """

    _require_lock(namespace_lock, namespace)
    destination_dir = Path(namespace) / "checkpoints"
    destination_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "sample_id": int(sample_id),
        "probe": probe,
        "message": message,
        **identity.as_dict(),
    }
    temporary = (
        destination_dir / f".sample_{int(sample_id):06d}_{probe}.failed.json.partial"
    )
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return publish_file(
        temporary, destination_dir / f"sample_{int(sample_id):06d}_{probe}.failed.json"
    )


def completed_sample_ids(
    namespace: str | Path,
    identity: ArtifactIdentity,
    probe: str,
) -> list[int]:
    """Return sample IDs whose shards reopen and match ``identity``.

    Args:
        namespace: Run directory.
        identity: Expected identity.
        probe (str): Probe filename token.

    Returns:
        list[int]: Validated sample IDs. Partial files, corrupt HDF5 and
        identity mismatches are omitted. They are not completed samples.
    """

    directory = Path(namespace) / "checkpoints"
    if not directory.is_dir():
        return []
    found: list[int] = []
    for path in sorted(directory.glob(f"sample_*_{probe}.h5")):
        try:
            record = validate_checkpoint_file(path, identity)
        except ArtifactError:
            continue
        if record.probe == probe:
            found.append(record.sample_id)
    return found


def pending_sample_ids(requested: Sequence[int], completed: Sequence[int]) -> list[int]:
    """Return requested IDs that do not yet have a validated shard.

    Args:
        requested: Sample IDs in evaluation order.
        completed: IDs already validated.

    Returns:
        list[int]: Remaining IDs. This does not draw replacements.
    """

    done = set(int(sample_id) for sample_id in completed)
    return [int(sample_id) for sample_id in requested if int(sample_id) not in done]


def _order_token(family: str | None, interpolation: str | None) -> str:
    """Return the filename order token after checking it against the family.

    Args:
        family: Producing family. Required whenever ``interpolation`` is given.
        interpolation: Radial order. Only NUMERIC products carry one.

    Returns:
        str: ``"_LINEAR"``-style token for NUMERIC, otherwise an empty string.
        CCL, NUMBA and JAX names carry no radial-order token.
    """

    supplied = interpolation is not None and str(interpolation).strip() != ""
    if family is None:
        if supplied:
            raise ArtifactError(
                f"Interpolation {interpolation!r} needs an explicit family; radial "
                "orders belong to NUMERIC filenames only"
            )
        return ""
    method = MethodIdentity.create(
        family, None if family.upper() != "JAX" else "CPU", interpolation
    )
    return f"_{method.interpolation}" if method.interpolation else ""


def spectra_basename(
    configuration: str,
    probe: str,
    interpolation: str | None = None,
    *,
    family: str | None = None,
) -> str:
    """Return a consolidated spectrum filename.

    Host CPU count is a Slurm runtime setting. It is not part of the filename.

    Args:
        configuration (str): Title-case configuration.
        probe (str): ``EE``, ``TE`` or ``TT``.
        interpolation: NUMERIC order. It requires ``family='NUMERIC'``.
        family: Producing family. Required whenever an order is supplied.

    Returns:
        str: For example ``Spectra_Triple_EE.h5``, or
        ``Spectra_Triple_LINEAR_EE.h5`` for NUMERIC.
    """

    order = _order_token(family, interpolation)
    return f"Spectra_{configuration}{order}_{probe}.h5"


def timing_basename(
    configuration: str,
    suffix: str = "",
    interpolation: str | None = None,
    *,
    family: str | None = None,
    population: str = "Cosmology",
) -> str:
    """Return a timing filename with an explicit evaluation population.

    ``Fiducial`` identifies sample 0; ``Cosmology`` identifies cumulative
    samples 1..N. Uppercase stage tokens precede that population token. Only
    NUMERIC carries an interpolation order; host CPU count is never included.
    """
    if population not in {"Fiducial", "Cosmology"}:
        raise ArtifactError("Timing population must be Fiducial or Cosmology")
    if suffix not in {"", "_COSMOLOGY", "_CELL", "_COEFFICIENT", "_PROJECTION"}:
        raise ArtifactError(f"Unknown timing stage suffix {suffix!r}")
    order = _order_token(family, interpolation)
    return f"Time_{configuration}{order}{suffix}_{population}.txt"


def samples_timing_basename(
    configuration: str,
    interpolation: str | None = None,
    *,
    family: str | None = None,
) -> str:
    """Return the per-sample stage-duration HDF5 name.

    Args:
        configuration (str): Title-case configuration.
        interpolation: NUMERIC order. It requires ``family='NUMERIC'``.
        family: Producing family. Required whenever an order is supplied.

    Returns:
        str: For example ``Time_Triple_SAMPLES.h5``.
    """

    order = _order_token(family, interpolation)
    return f"Time_{configuration}{order}_SAMPLES.h5"


def manifest_basename(
    configuration: str,
    interpolation: str | None = None,
    *,
    family: str | None = None,
) -> str:
    """Return the run-manifest filename published after product validation.

    Args:
        configuration (str): Title-case configuration.
        interpolation: NUMERIC order. It requires ``family='NUMERIC'``.
        family: Producing family. Required whenever an order is supplied.

    Returns:
        str: For example ``Manifest_Triple.json``.
    """

    order = _order_token(family, interpolation)
    return f"Manifest_{configuration}{order}.json"


def consolidate_probe(
    namespace: str | Path,
    identity: ArtifactIdentity,
    namespace_lock: NamespaceLock,
    *,
    probe: str,
    configuration: str,
    sample_ids: Sequence[int],
) -> Path:
    """Stack validated shards into a temporary final file and publish it.

    Args:
        namespace: Run directory containing ``checkpoints/``.
        identity: Identity required of every shard and of the final file.
        probe (str): Probe to consolidate.
        configuration (str): Title-case configuration label.
        sample_ids: IDs that must be present. Duplicates or gaps are rejected.
            Order in the file follows these IDs. Fiducial-first is conventional
            when ID 0 is included, but readers select on ``sample_id``.

    Returns:
        Path: Published consolidated HDF5 file.         Shards are left in place.
    """

    _require_lock(namespace_lock, namespace)
    h5py = _require_h5py()
    if len(sample_ids) != len(set(int(item) for item in sample_ids)):
        raise ArtifactError("Consolidated sample IDs contain duplicates")
    records = []
    checkpoint_dir = Path(namespace) / "checkpoints"
    for sample_id in sample_ids:
        path = checkpoint_dir / checkpoint_name(int(sample_id), probe)
        records.append(validate_checkpoint_file(path, identity))
    if not records:
        raise ArtifactError("Refusing to publish an empty consolidated spectrum")
    ell = records[0].ell
    pair_i = records[0].pair_i
    pair_j = records[0].pair_j
    names = records[0].parameter_names
    for record in records[1:]:
        if not numpy.array_equal(record.ell, ell) or not numpy.array_equal(
            record.pair_i, pair_i
        ):
            raise ArtifactError("Checkpoint ell or pair axes disagree")
        if record.parameter_names != names:
            raise ArtifactError("Checkpoint parameter names disagree")
    cl = numpy.stack([record.cl for record in records], axis=0)
    sample_id = numpy.asarray(
        [record.sample_id for record in records], dtype=numpy.int64
    )
    is_fiducial = numpy.asarray(
        [record.is_fiducial for record in records], dtype=numpy.bool_
    )
    cosmology = numpy.stack([record.cosmology for record in records], axis=0)
    destination = Path(namespace) / spectra_basename(
        configuration,
        probe,
        identity.interpolation or None,
        family=identity.family,
    )
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.partial"
    with h5py.File(temporary, "w") as handle:
        for key, value in identity.as_dict().items():
            handle.attrs[key] = value
        handle.attrs["probe"] = probe
        handle.attrs["estimator_name"] = records[0].estimator_name
        handle.attrs["status"] = "validated-not-published"
        handle.create_dataset("sample_id", data=sample_id)
        handle.create_dataset("is_fiducial", data=is_fiducial)
        handle.create_dataset(
            "parameter_names",
            data=numpy.asarray(names, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        handle.create_dataset(
            "cosmology", data=numpy.asarray(cosmology, dtype=numpy.float64)
        )
        handle.create_dataset("ell", data=numpy.asarray(ell, dtype=numpy.float64))
        handle.create_dataset("pair_i", data=numpy.asarray(pair_i, dtype=numpy.int32))
        handle.create_dataset("pair_j", data=numpy.asarray(pair_j, dtype=numpy.int32))
        handle.create_dataset(
            "cl",
            data=numpy.asarray(cl, dtype=numpy.float64),
            chunks=(1, cl.shape[1], cl.shape[2]),
            compression="gzip",
            compression_opts=4,
        )
        sampled = handle.create_group("sampled")
        sampled.attrs["role"] = "raw sampled spectra"
        sampled.create_dataset("ell", data=numpy.asarray(ell, dtype=numpy.float64))
    _validate_consolidated(temporary, identity, probe, sample_ids)
    return publish_file(temporary, destination)


def _validate_consolidated(
    path: Path,
    identity: ArtifactIdentity,
    probe: str,
    sample_ids: Sequence[int],
) -> None:
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        for key, expected in identity.as_dict().items():
            if str(handle.attrs.get(key)) != str(expected):
                raise ArtifactError(
                    f"Consolidated file {path} failed identity check {key}"
                )
        if str(handle.attrs.get("probe")) != probe:
            raise ArtifactError(f"Consolidated file {path} probe mismatch")
        stored_ids = numpy.asarray(handle["sample_id"], dtype=numpy.int64)
        if stored_ids.tolist() != [int(item) for item in sample_ids]:
            raise ArtifactError(
                f"Consolidated file {path} sample IDs do not match the request"
            )
        cl = numpy.asarray(handle["cl"], dtype=numpy.float64)
        if cl.shape[0] != stored_ids.size:
            raise ArtifactError(
                f"Consolidated file {path} cl axis 0 is not the sample axis"
            )
        if "coefficients" in handle:
            raise ArtifactError(
                "Consolidated spectra must not store coefficient tensors"
            )
        if "bandpower" in handle and "sampled" not in handle:
            raise ArtifactError("Bandpowers cannot stand in for raw sampled spectra")


def write_manifest(
    namespace: str | Path,
    namespace_lock: NamespaceLock,
    *,
    identity: ArtifactIdentity,
    configuration: str,
    products: Mapping[str, Path],
    completed_sample_ids: Sequence[int],
    failed_sample_ids: Sequence[int],
    eta_ia: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Publish the run manifest after the listed products validate.

    Args:
        namespace: Run directory.
        identity: Namespace identity.
        configuration (str): Title-case configuration.
        products: Filename to path. Each path is checksummed.
        completed_sample_ids: Sample IDs present in the products.
        failed_sample_ids: Sample IDs with published failure records. They are
            not dropped from the manifest.
        eta_ia: IA decision record. An unresolved status is stored as-is.
        extra: Additional JSON-ready provenance.

    Returns:
        Path: Published manifest. It is the last file written.
    """

    _require_lock(namespace_lock, namespace)
    product_records = {}
    for name, path in products.items():
        file_path = Path(path)
        if not file_path.is_file():
            raise ArtifactError(f"Cannot publish a manifest missing {file_path}")
        product_records[name] = {
            "path": file_path.name,
            "sha256": sha256_file(file_path),
        }
    payload = {
        "status": MANIFEST_STATUS_COMPLETE,
        "identity": identity.as_dict(),
        "configuration": configuration,
        "products": product_records,
        "completed_sample_ids": [int(item) for item in completed_sample_ids],
        "failed_sample_ids": [int(item) for item in failed_sample_ids],
        "eta_ia": dict(eta_ia),
        "compression": {"cl": "gzip", "level": 4, "plugin": "hdf5-standard"},
        "coefficient_tensors_retained": False,
    }
    if extra:
        payload["extra"] = dict(extra)
    destination = Path(namespace) / manifest_basename(
        configuration,
        identity.interpolation or None,
        family=identity.family,
    )
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.partial"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return publish_file(temporary, destination)


def read_completed_manifest(path: str | Path) -> dict:
    """Load a manifest and require completed status plus matching checksums.

    Args:
        path: Manifest JSON path.

    Returns:
        dict: Parsed manifest. Missing products and checksum mismatches raise.
    """

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ArtifactError(f"Manifest {manifest_path} does not exist")
    payload = json.loads(manifest_path.read_text())
    if payload.get("status") != MANIFEST_STATUS_COMPLETE:
        raise ArtifactError(f"Manifest {manifest_path} is not complete")
    for name, record in payload.get("products", {}).items():
        product = manifest_path.parent / record["path"]
        if not product.is_file():
            raise ArtifactError(f"Manifest product {name} is missing at {product}")
        if sha256_file(product) != record["sha256"]:
            raise ArtifactError(f"Manifest product {name} checksum mismatch")
    return payload


def read_fiducial_spectrum(manifest_path: str | Path, probe: str) -> numpy.ndarray:
    """Read the fiducial row of one probe by sample ID.

    Args:
        manifest_path: Completed manifest.
        probe (str): ``EE``, ``TE`` or ``TT``.

    Returns:
        numpy.ndarray: Float64 ``cl`` row for ``is_fiducial`` / sample ID 0,
        without loading every row into a new array beyond the HDF5 selection.
    """

    h5py = _require_h5py()
    manifest = read_completed_manifest(manifest_path)
    matches = [
        (name, record)
        for name, record in manifest["products"].items()
        if name.endswith(f"_{probe}.h5") or record["path"].endswith(f"_{probe}.h5")
    ]
    if len(matches) != 1:
        raise ArtifactError(f"Manifest does not identify one {probe} spectrum product")
    path = Path(manifest_path).parent / matches[0][1]["path"]
    with h5py.File(path, "r") as handle:
        sample_id = numpy.asarray(handle["sample_id"], dtype=numpy.int64)
        fiducial = numpy.flatnonzero(sample_id == 0)
        if fiducial.size != 1:
            raise ArtifactError(
                f"{path} does not contain a single fiducial sample ID 0"
            )
        return numpy.asarray(handle["cl"][int(fiducial[0])], dtype=numpy.float64)
