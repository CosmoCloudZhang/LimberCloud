"""The current angular contract: 21 raw multipoles and 20 bandpowers.

Every method evaluates float64 ``numpy.geomspace(20, 2000, 21)``. One shared
operator turns those 21 raw samples into 20 bandpowers: a **natural** cubic
spline of ``ell * C_ell`` against ``log(ell)``, integrated across each log
interval and divided by the corresponding linear ``delta_ell``. The source
notebook fixes ``bc_type='natural'``; SciPy's default not-a-knot boundary is a
different operator and is not accepted as compatible.

The 20 bandpowers are the residual and covariance vector. The 21 raw samples
are retained so the spline can be reproduced; they are not a second comparison
vector. Geometric bin centres are display coordinates only. The historical
20-centre CCL grid and the 101-point covariance grid remain available for
reading old products and are never the current run default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy

CANONICAL_ELL_MIN = 20.0
CANONICAL_ELL_MAX = 2000.0
CANONICAL_NODE_COUNT = 21
CANONICAL_BAND_COUNT = CANONICAL_NODE_COUNT - 1

ANGULAR_OPERATOR_VERSION = "limbercloud.angular.natural-spline.v1"
ANGULAR_TRANSFORM = "ell*C_ell versus log(ell)"
ANGULAR_BOUNDARY_CONDITION = "natural"
ANGULAR_NORMALIZATION = "integral over log interval divided by linear delta_ell"
ANGULAR_DTYPE = "float64"
ANGULAR_AXIS = "last"

# Current estimator names.
ESTIMATOR_RAW_NODES = "raw_nodes"
ESTIMATOR_BANDPOWERS = "natural_spline_bandpowers"

# Historical names retained so old products can still be labelled and refused.
ESTIMATOR_SAMPLED_NODES = "sampled_nodes"
ESTIMATOR_GEOMETRIC_CENTRES = "geometric_centres"
ESTIMATOR_UNIFORM_DELL = "uniform_dell_band_average"

ESTIMATOR_NAMES = (
    ESTIMATOR_RAW_NODES,
    ESTIMATOR_BANDPOWERS,
    ESTIMATOR_SAMPLED_NODES,
    ESTIMATOR_GEOMETRIC_CENTRES,
    ESTIMATOR_UNIFORM_DELL,
)

LEGACY_ESTIMATOR_NAMES = (
    ESTIMATOR_SAMPLED_NODES,
    ESTIMATOR_GEOMETRIC_CENTRES,
    ESTIMATOR_UNIFORM_DELL,
)

# Historical aliases. ``LIMBERCLOUD_EDGE_COUNT`` is the current node count.
SAMPLED_ELL_MIN = CANONICAL_ELL_MIN
SAMPLED_ELL_MAX = CANONICAL_ELL_MAX
LIMBERCLOUD_EDGE_COUNT = CANONICAL_NODE_COUNT
CCL_CENTRE_COUNT = CANONICAL_BAND_COUNT
LEGACY_COVARIANCE_INPUT_COUNT = 101
COVARIANCE_INPUT_COUNT = LEGACY_COVARIANCE_INPUT_COUNT


class EstimatorMismatch(ValueError):
    """Raised when two spectra do not share an estimator fingerprint."""


class AngularContractError(ValueError):
    """Raised for invalid multipole coordinates, shapes or spectra."""


def canonical_ell_nodes(
    ell_min: float = CANONICAL_ELL_MIN,
    ell_max: float = CANONICAL_ELL_MAX,
    count: int = CANONICAL_NODE_COUNT,
) -> numpy.ndarray:
    """Return the shared raw multipole nodes.

    Args:
        ell_min (float): First node. The campaign value is 20.
        ell_max (float): Last node. The campaign value is 2000.
        count (int): Node count. The campaign value is 21.

    Returns:
        numpy.ndarray: Float64 geomspace nodes. These are also the 20 band
        edges; ``C_ell`` is a point sample here, not a bandpower.
    """

    return numpy.geomspace(float(ell_min), float(ell_max), int(count)).astype(numpy.float64)


def limbercloud_edges(
    ell_min: float = CANONICAL_ELL_MIN,
    ell_max: float = CANONICAL_ELL_MAX,
    count: int = CANONICAL_NODE_COUNT,
) -> numpy.ndarray:
    """Return :func:`canonical_ell_nodes` under its historical name.

    Args:
        ell_min (float): First node.
        ell_max (float): Last node.
        count (int): Number of nodes.

    Returns:
        numpy.ndarray: Float64 geomspace nodes.
    """

    return canonical_ell_nodes(ell_min, ell_max, count)


def validate_ell_nodes(ell_nodes) -> numpy.ndarray:
    """Return validated float64 multipole nodes.

    Args:
        ell_nodes: Candidate coordinates.

    Returns:
        numpy.ndarray: Float64 1-d array with at least two samples.

    Raises:
        AngularContractError: When the coordinates are not one-dimensional,
        finite, positive and strictly increasing.
    """

    ell = numpy.asarray(ell_nodes, dtype=numpy.float64)
    if ell.ndim != 1 or ell.size < 2:
        raise AngularContractError(
            f"ell must be a 1-d array with at least two samples; got shape {ell.shape}"
        )
    if not numpy.all(numpy.isfinite(ell)):
        raise AngularContractError("ell contains nonfinite values")
    if not numpy.all(ell > 0.0):
        raise AngularContractError("ell must be strictly positive for a log transform")
    if not numpy.all(numpy.diff(ell) > 0.0):
        raise AngularContractError("ell must be strictly increasing")
    return ell


def validate_spectra(cl_nodes, ell_size: int) -> numpy.ndarray:
    """Return validated float64 spectra sampled on ``ell_size`` nodes.

    Args:
        cl_nodes: Spectra whose last axis runs over the multipole nodes.
        ell_size (int): Expected length of that last axis.

    Returns:
        numpy.ndarray: Float64 array. Signed and exactly zero spectra are
        legitimate and are not rejected.

    Raises:
        AngularContractError: When the last axis does not match or the values
        are not all finite.
    """

    cl = numpy.asarray(cl_nodes, dtype=numpy.float64)
    if cl.ndim < 1 or cl.shape[-1] != int(ell_size):
        raise AngularContractError(
            f"spectra last axis {cl.shape[-1] if cl.ndim else None} does not match "
            f"{int(ell_size)} ell nodes"
        )
    if not numpy.all(numpy.isfinite(cl)):
        raise AngularContractError("spectra contain nonfinite values")
    return cl


def band_display_centres(edges) -> numpy.ndarray:
    """Return geometric bin centres used as display coordinates.

    Args:
        edges: Increasing bin edges, length ``N+1``.

    Returns:
        numpy.ndarray: ``sqrt(edge_i * edge_{i+1})``, length ``N``. These label
        the 20 bandpowers on a plot; they never select extra evaluations.
    """

    values = validate_ell_nodes(edges)
    return numpy.sqrt(values[1:] * values[:-1])


def geometric_centres(edges) -> numpy.ndarray:
    """Return geometric bin centres under their historical name.

    Args:
        edges: Increasing bin edges, length ``N+1``.

    Returns:
        numpy.ndarray: ``sqrt(edge_i * edge_{i+1})``, length ``N``.
    """

    return band_display_centres(edges)


def legacy_covariance_input_ell(
    ell_min: float = CANONICAL_ELL_MIN,
    ell_max: float = CANONICAL_ELL_MAX,
    count: int = LEGACY_COVARIANCE_INPUT_COUNT,
) -> numpy.ndarray:
    """Return the retired 101-point covariance-input grid.

    Args:
        ell_min (float): First sample.
        ell_max (float): Last sample.
        count (int): Sample count. 101 is the historical value.

    Returns:
        numpy.ndarray: Float64 geomspace samples. This grid is legacy-only: the
        current covariance input is the 20 bandpowers reconstructed from the 21
        raw nodes.
    """

    return canonical_ell_nodes(ell_min, ell_max, count)


def covariance_input_ell(
    ell_min: float = CANONICAL_ELL_MIN,
    ell_max: float = CANONICAL_ELL_MAX,
    count: int = LEGACY_COVARIANCE_INPUT_COUNT,
) -> numpy.ndarray:
    """Return :func:`legacy_covariance_input_ell` under its historical name.

    Args:
        ell_min (float): First sample.
        ell_max (float): Last sample.
        count (int): Sample count.

    Returns:
        numpy.ndarray: Float64 geomspace samples, legacy-only.
    """

    return legacy_covariance_input_ell(ell_min, ell_max, count)


def _natural_spline(log_ell: numpy.ndarray, product: numpy.ndarray):
    import scipy.interpolate

    return scipy.interpolate.CubicSpline(
        log_ell, product, axis=-1, bc_type=ANGULAR_BOUNDARY_CONDITION
    )


def natural_spline_bandpowers(ell_nodes, cl_nodes) -> numpy.ndarray:
    """Average sampled spectra with the notebook's natural-spline operator.

    A natural cubic spline interpolates ``ell * C_ell`` against ``log(ell)``.
    Each band integral runs over one log interval and is divided by the linear
    bin width, which equals ``(1/deltaell) * integral C(ell) dell`` for the
    reconstructed ``C(ell) = spline(log ell) / ell``.

    Args:
        ell_nodes: Strictly increasing positive multipoles, length ``N``.
        cl_nodes: Spectra sampled on those nodes; the last axis has length
            ``N``. Leading axes are batch axes and are preserved.

    Returns:
        numpy.ndarray: Float64 bandpowers on the ``N-1`` intervals, shaped
        ``cl_nodes.shape[:-1] + (N-1,)``.
    """

    ell = validate_ell_nodes(ell_nodes)
    cl = validate_spectra(cl_nodes, ell.size)
    log_ell = numpy.log(ell)
    spline = _natural_spline(log_ell, ell * cl)
    widths = numpy.diff(ell)
    bandpowers = numpy.empty(cl.shape[:-1] + (ell.size - 1,), dtype=numpy.float64)
    for index in range(ell.size - 1):
        integral = spline.integrate(log_ell[index], log_ell[index + 1])
        bandpowers[..., index] = numpy.asarray(integral) / widths[index]
    return bandpowers


def bandpower_operator_matrix(ell_nodes) -> numpy.ndarray:
    """Return the ``(N-1, N)`` matrix of the natural-spline band operator.

    The operator is linear in ``cl_nodes``, so it has an exact matrix form.
    ``matrix @ cl`` reproduces :func:`natural_spline_bandpowers` for signed
    vectors and for batches.

    Args:
        ell_nodes: Strictly increasing positive multipoles, length ``N``.

    Returns:
        numpy.ndarray: Float64 weights. Row ``b`` holds the contribution of each
        raw node to bandpower ``b``.
    """

    ell = validate_ell_nodes(ell_nodes)
    basis = numpy.eye(ell.size, dtype=numpy.float64)
    return natural_spline_bandpowers(ell, basis).T.copy()


def uniform_dell_band_average(ell_nodes, cl_nodes) -> numpy.ndarray:
    """Return :func:`natural_spline_bandpowers` under its historical name.

    Args:
        ell_nodes: Strictly increasing multipoles, length ``N``.
        cl_nodes: Spectra sampled on those nodes.

    Returns:
        numpy.ndarray: Float64 bandpowers on the ``N-1`` intervals. The boundary
        condition is natural, matching the notebook, not SciPy's default.
    """

    return natural_spline_bandpowers(ell_nodes, cl_nodes)


@dataclass(frozen=True)
class EllEstimator:
    """Named multipole coordinate of one spectrum product.

    Args:
        name: One of :data:`ESTIMATOR_NAMES`.
        ell: Multipoles at which ``cl`` is stored.
        edges: Bin edges when the estimator is defined from them.
        boundary_condition: Spline boundary condition, or an empty string for
            estimators that do not spline. ``natural`` and ``not-a-knot`` are
            different operators and produce different fingerprints.
        operator_version: Versioned operator label, so a historical helper
            cannot present itself as the current contract.
    """

    name: str
    ell: tuple[float, ...]
    edges: tuple[float, ...] | None = None
    boundary_condition: str = ""
    operator_version: str = ""

    def __post_init__(self) -> None:
        if self.name not in ESTIMATOR_NAMES:
            choices = ", ".join(ESTIMATOR_NAMES)
            raise ValueError(f"Unknown estimator {self.name!r}; expected one of: {choices}")
        if len(self.ell) < 1:
            raise ValueError("ell must contain at least one sample")
        values = numpy.asarray(self.ell, dtype=numpy.float64)
        if not numpy.all(numpy.isfinite(values)):
            raise ValueError("Estimator coordinates must be finite")

    @property
    def is_legacy(self) -> bool:
        """Return whether this estimator only reads historical products."""

        return self.name in LEGACY_ESTIMATOR_NAMES

    def fingerprint(self) -> str:
        """Return a stable hash of the estimator identity and coordinates.

        Returns:
            str: Hexadecimal SHA-256 digest over the name, boundary condition,
            operator version and exact float64 coordinate bytes.
        """

        digest = hashlib.sha256()
        digest.update(self.name.encode("utf-8"))
        digest.update(b"\n")
        digest.update(self.boundary_condition.encode("utf-8"))
        digest.update(b"\n")
        digest.update(self.operator_version.encode("utf-8"))
        digest.update(b"\n")
        digest.update(numpy.asarray(self.ell, dtype=numpy.float64).tobytes())
        if self.edges is not None:
            digest.update(b"\n")
            digest.update(numpy.asarray(self.edges, dtype=numpy.float64).tobytes())
        return digest.hexdigest()

    def validate_coordinates(self, ell) -> None:
        """Reject spectra stored on coordinates this estimator does not name.

        Args:
            ell: Coordinates actually stored beside the spectra.

        Raises:
            EstimatorMismatch: When the stored coordinates differ from the
            declared ones. A fingerprint that is never compared with the real
            array proves nothing.
        """

        stored = numpy.asarray(ell, dtype=numpy.float64)
        declared = numpy.asarray(self.ell, dtype=numpy.float64)
        if stored.shape != declared.shape or not numpy.array_equal(stored, declared):
            raise EstimatorMismatch(
                f"Estimator {self.name} declares {declared.size} coordinates that do "
                f"not match the {stored.size} stored values"
            )


@dataclass(frozen=True)
class AngularContract:
    """The complete current angular operator, validated and fingerprinted.

    Args:
        nodes: The 21 raw multipoles, in stored order.
        edges: The 21 band edges. They coincide with ``nodes``.
        centres: The 20 geometric display centres.
        transform: Splined quantity and abscissa.
        boundary_condition: Cubic-spline boundary condition.
        normalization: How each band integral is normalised.
        dtype: Storage dtype of every coordinate and spectrum.
        axis: Which array axis runs over multipoles.
        version: Operator and schema version.
    """

    nodes: tuple[float, ...]
    edges: tuple[float, ...]
    centres: tuple[float, ...]
    transform: str = ANGULAR_TRANSFORM
    boundary_condition: str = ANGULAR_BOUNDARY_CONDITION
    normalization: str = ANGULAR_NORMALIZATION
    dtype: str = ANGULAR_DTYPE
    axis: str = ANGULAR_AXIS
    version: str = ANGULAR_OPERATOR_VERSION

    def __post_init__(self) -> None:
        nodes = validate_ell_nodes(self.nodes)
        edges = validate_ell_nodes(self.edges)
        if not numpy.array_equal(nodes, edges):
            raise AngularContractError("The raw nodes and the band edges must coincide")
        if len(self.centres) != nodes.size - 1:
            raise AngularContractError(
                f"{len(self.centres)} display centres do not bound "
                f"{nodes.size - 1} intervals"
            )
        if self.boundary_condition != ANGULAR_BOUNDARY_CONDITION:
            raise AngularContractError(
                f"The current contract is a {ANGULAR_BOUNDARY_CONDITION!r} cubic spline; "
                f"got {self.boundary_condition!r}"
            )

    @property
    def node_count(self) -> int:
        """Return the number of raw multipole samples."""

        return len(self.nodes)

    @property
    def band_count(self) -> int:
        """Return the number of bandpowers."""

        return len(self.nodes) - 1

    @property
    def ell(self) -> numpy.ndarray:
        """Return the raw nodes as a float64 array."""

        return numpy.asarray(self.nodes, dtype=numpy.float64)

    @property
    def band_widths(self) -> numpy.ndarray:
        """Return the linear ``delta_ell`` of each band."""

        return numpy.diff(self.ell)

    def raw_estimator(self) -> EllEstimator:
        """Return the estimator naming the 21 raw samples."""

        return EllEstimator(
            ESTIMATOR_RAW_NODES,
            self.nodes,
            self.edges,
            boundary_condition="",
            operator_version=self.version,
        )

    def band_estimator(self) -> EllEstimator:
        """Return the estimator naming the 20 bandpowers."""

        return EllEstimator(
            ESTIMATOR_BANDPOWERS,
            self.centres,
            self.edges,
            boundary_condition=self.boundary_condition,
            operator_version=self.version,
        )

    def validate_nodes(self, ell) -> numpy.ndarray:
        """Reject coordinates that are not this contract's raw nodes.

        Args:
            ell: Coordinates to check.

        Returns:
            numpy.ndarray: The validated float64 nodes.

        Raises:
            AngularContractError: When the coordinates differ from the contract.
        """

        values = validate_ell_nodes(ell)
        expected = self.ell
        if values.shape != expected.shape or not numpy.array_equal(values, expected):
            raise AngularContractError(
                f"Spectra are stored on {values.size} multipoles that are not the "
                f"{expected.size} contract nodes"
            )
        return values

    def bandpowers(self, cl_nodes) -> numpy.ndarray:
        """Apply the shared operator to raw spectra on this contract's nodes.

        Args:
            cl_nodes: Spectra whose last axis has length ``node_count``.

        Returns:
            numpy.ndarray: Float64 bandpowers whose last axis has length
            ``band_count``.
        """

        cl = validate_spectra(cl_nodes, self.node_count)
        return natural_spline_bandpowers(self.ell, cl)

    def operator_matrix(self) -> numpy.ndarray:
        """Return the ``(band_count, node_count)`` weights of the operator."""

        return bandpower_operator_matrix(self.ell)

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-ready description persisted beside every product."""

        return {
            "version": self.version,
            "node_count": self.node_count,
            "band_count": self.band_count,
            "nodes": [float(value) for value in self.nodes],
            "edges": [float(value) for value in self.edges],
            "centres": [float(value) for value in self.centres],
            "transform": self.transform,
            "boundary_condition": self.boundary_condition,
            "normalization": self.normalization,
            "dtype": self.dtype,
            "axis": self.axis,
            "raw_estimator": ESTIMATOR_RAW_NODES,
            "band_estimator": ESTIMATOR_BANDPOWERS,
        }

    def fingerprint(self) -> str:
        """Return a deterministic hash of the whole operator description.

        Returns:
            str: Hexadecimal SHA-256 digest. The boundary condition and the
            operator version are included, so the historical not-a-knot helper
            cannot pass as compatible.
        """

        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_angular_contract(
    ell_min: float = CANONICAL_ELL_MIN,
    ell_max: float = CANONICAL_ELL_MAX,
    count: int = CANONICAL_NODE_COUNT,
) -> AngularContract:
    """Return the current 21-node, 20-band natural-spline contract.

    Args:
        ell_min (float): First node. The campaign value is 20.
        ell_max (float): Last node. The campaign value is 2000.
        count (int): Node count. The campaign value is 21.

    Returns:
        AngularContract: Validated operator shared by every method.
    """

    nodes = canonical_ell_nodes(ell_min, ell_max, count)
    centres = band_display_centres(nodes)
    return AngularContract(
        nodes=tuple(float(value) for value in nodes),
        edges=tuple(float(value) for value in nodes),
        centres=tuple(float(value) for value in centres),
    )


def canonical_raw_estimator() -> EllEstimator:
    """Return the estimator of the 21 shared raw multipole samples."""

    return canonical_angular_contract().raw_estimator()


def canonical_band_estimator() -> EllEstimator:
    """Return the estimator of the 20 shared bandpowers."""

    return canonical_angular_contract().band_estimator()


def historical_limbercloud_estimator() -> EllEstimator:
    """Return the historical 21-edge sampled-node estimator.

    Returns:
        EllEstimator: Labelled ``sampled_nodes`` with no operator version. It
        shares coordinates with the current raw estimator but is a different
        identity, so historical products cannot resume as current ones.
    """

    edges = canonical_ell_nodes()
    values = tuple(float(value) for value in edges)
    return EllEstimator(ESTIMATOR_SAMPLED_NODES, values, values)


def historical_ccl_estimator() -> EllEstimator:
    """Return the historical 20-centre estimator of the old CCL drivers.

    Returns:
        EllEstimator: Legacy-only. Current CCL runners evaluate the same 21
        nodes as the analytical methods.
    """

    edges = canonical_ell_nodes()
    centres = band_display_centres(edges)
    return EllEstimator(
        ESTIMATOR_GEOMETRIC_CENTRES,
        tuple(float(value) for value in centres),
        tuple(float(value) for value in edges),
    )


def assert_same_estimator(left: EllEstimator, right: EllEstimator) -> None:
    """Reject spectra that do not share an estimator fingerprint.

    Args:
        left: First estimator.
        right: Second estimator.

    Raises:
        EstimatorMismatch: When the fingerprints differ. Twenty CCL centres are
        not the same coordinate as twenty-one raw nodes, and a not-a-knot
        bandpower is not the current natural-spline bandpower.
    """

    if left.fingerprint() != right.fingerprint():
        raise EstimatorMismatch(
            f"Estimator mismatch: {left.name} ({len(left.ell)} samples, "
            f"boundary={left.boundary_condition or 'none'}) vs "
            f"{right.name} ({len(right.ell)} samples, "
            f"boundary={right.boundary_condition or 'none'})"
        )
