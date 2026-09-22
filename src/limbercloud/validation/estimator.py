"""Angular-estimator identities.

Historical CCL experiment drivers evaluate 20 geometric bin centres. LimberCloud
drivers evaluate 21 geomspace edges and do not then form band averages. Notebooks
additionally average ``ell * C_ell`` uniformly in ``ell``. Those are different
estimators. Shared spectra must name one of them and refuse mixed comparisons.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy

SAMPLED_ELL_MIN = 20.0
SAMPLED_ELL_MAX = 2000.0
LIMBERCLOUD_EDGE_COUNT = 21
CCL_CENTRE_COUNT = 20
COVARIANCE_INPUT_COUNT = 101

ESTIMATOR_SAMPLED_NODES = "sampled_nodes"
ESTIMATOR_GEOMETRIC_CENTRES = "geometric_centres"
ESTIMATOR_UNIFORM_DELL = "uniform_dell_band_average"


class EstimatorMismatch(ValueError):
    """Raised when two spectra do not share an estimator fingerprint."""


def limbercloud_edges(
    ell_min: float = SAMPLED_ELL_MIN,
    ell_max: float = SAMPLED_ELL_MAX,
    count: int = LIMBERCLOUD_EDGE_COUNT,
) -> numpy.ndarray:
    """Return the historical LimberCloud multipole nodes.

    Args:
        ell_min (float): First node.
        ell_max (float): Last node.
        count (int): Number of nodes. The experiment drivers use 21.

    Returns:
        numpy.ndarray: Float64 geomspace nodes.
    """

    return numpy.geomspace(float(ell_min), float(ell_max), int(count)).astype(numpy.float64)


def geometric_centres(edges: numpy.ndarray) -> numpy.ndarray:
    """Return geometric bin centres used by the historical CCL drivers.

    Args:
        edges (numpy.ndarray): Increasing bin edges, length ``N+1``.

    Returns:
        numpy.ndarray: ``sqrt(edge_i * edge_{i+1})``, length ``N``.
    """

    values = numpy.asarray(edges, dtype=numpy.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("edges must be a 1-d array with at least two samples")
    if not numpy.all(numpy.diff(values) > 0):
        raise ValueError("edges must be strictly increasing")
    return numpy.sqrt(values[1:] * values[:-1])


def covariance_input_ell(
    ell_min: float = SAMPLED_ELL_MIN,
    ell_max: float = SAMPLED_ELL_MAX,
    count: int = COVARIANCE_INPUT_COUNT,
) -> numpy.ndarray:
    """Return the compatibility raw-ell grid for covariance input.

    Args:
        ell_min (float): First sample.
        ell_max (float): Last sample.
        count (int): Sample count. 101 is the current compatibility default,
            not a universal covariance requirement.

    Returns:
        numpy.ndarray: Float64 geomspace samples. These are not bandpowers.
    """

    return limbercloud_edges(ell_min, ell_max, count)


def uniform_dell_band_average(ell_nodes: numpy.ndarray, cl_nodes: numpy.ndarray) -> numpy.ndarray:
    """Average a sampled spectrum with the notebook's uniform-``dell`` window.

    The spline is built on ``log(ell)`` for the product ``ell * C_ell``. Each
    bin integral is divided by the linear bin width. This is a bandpower, not
    a point sample.

    Args:
        ell_nodes (numpy.ndarray): Strictly increasing multipoles, length ``N``.
        cl_nodes (numpy.ndarray): Spectra sampled on those nodes. The last axis
            has length ``N``.

    Returns:
        numpy.ndarray: Bandpowers on the ``N-1`` intervals, float64.
    """

    import scipy.interpolate

    ell = numpy.asarray(ell_nodes, dtype=numpy.float64)
    cl = numpy.asarray(cl_nodes, dtype=numpy.float64)
    if ell.ndim != 1 or ell.size < 2:
        raise ValueError("ell_nodes must contain at least two increasing samples")
    if not numpy.all(numpy.diff(ell) > 0):
        raise ValueError("ell_nodes must be strictly increasing")
    if cl.shape[-1] != ell.size:
        raise ValueError(
            f"cl last axis {cl.shape[-1]} does not match {ell.size} ell nodes"
        )
    log_ell = numpy.log(ell)
    spline = scipy.interpolate.CubicSpline(log_ell, ell * cl, axis=-1)
    widths = numpy.diff(ell)
    bandpowers = numpy.empty(cl.shape[:-1] + (ell.size - 1,), dtype=numpy.float64)
    for index in range(ell.size - 1):
        bandpowers[..., index] = spline.integrate(log_ell[index], log_ell[index + 1]) / widths[index]
    return bandpowers


@dataclass(frozen=True)
class EllEstimator:
    """Named multipole coordinate used by a spectrum product.

    Args:
        name: ``sampled_nodes``, ``geometric_centres`` or
            ``uniform_dell_band_average``.
        ell: Multipoles at which ``cl`` is stored.
        edges: Bin edges when the estimator is defined from them.
    """

    name: str
    ell: tuple[float, ...]
    edges: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.name not in {
            ESTIMATOR_SAMPLED_NODES,
            ESTIMATOR_GEOMETRIC_CENTRES,
            ESTIMATOR_UNIFORM_DELL,
        }:
            raise ValueError(f"Unknown estimator {self.name!r}")
        if len(self.ell) < 1:
            raise ValueError("ell must contain at least one sample")

    def fingerprint(self) -> str:
        """Return a stable hash of the estimator name and coordinates.

        Returns:
            str: Hexadecimal SHA-256 digest.
        """

        payload = numpy.asarray(self.ell, dtype=numpy.float64).tobytes()
        if self.edges is not None:
            payload += numpy.asarray(self.edges, dtype=numpy.float64).tobytes()
        digest = hashlib.sha256(self.name.encode("utf-8") + b"\n" + payload)
        return digest.hexdigest()


def historical_limbercloud_estimator() -> EllEstimator:
    """Return the 21-edge sampled-node estimator used by Numba and JAX drivers."""

    edges = limbercloud_edges()
    return EllEstimator(ESTIMATOR_SAMPLED_NODES, tuple(float(value) for value in edges), tuple(float(value) for value in edges))


def historical_ccl_estimator() -> EllEstimator:
    """Return the 20-centre estimator used by the CCL experiment drivers."""

    edges = limbercloud_edges()
    centres = geometric_centres(edges)
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
        not the same coordinate as twenty-one LimberCloud nodes.
    """

    if left.fingerprint() != right.fingerprint():
        raise EstimatorMismatch(
            f"Estimator mismatch: {left.name} ({len(left.ell)} samples) vs "
            f"{right.name} ({len(right.ell)} samples)"
        )
