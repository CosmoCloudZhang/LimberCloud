"""Component assembly and active-cosmology prefactors.

CCL number-count tracers take the magnification slope ``s``. The analytical
count response is ``q = 5s - 2``. Magnification-shear and magnification-intrinsic
terms both use the response-weighted lens distribution. Lensing amplitude and
the radial density conversion use the cosmology being evaluated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy
import scipy.constants

from limbercloud.validation.contract import PROBE_COMPONENTS, configuration_probes

# Match the historical runner: H0 = h * 100 km/s/Mpc = h * 1e5 m/s/Mpc.
_H0_PER_H = 100_000.0


def hubble_distance_factor(h: float) -> float:
    """Convert dimensionless ``h`` to ``H0/c`` in inverse megaparsecs.

    Args:
        h (float): Hubble parameter in units of 100 km/s/Mpc.

    Returns:
        float: ``H0/c`` using the same metre-per-second conversion as the runners.
    """

    return float(h) * _H0_PER_H / float(scipy.constants.c)


def active_lensing_amplitude(omega_m: float, h: float) -> float:
    """Return the Limber lensing prefactor for the active cosmology.

    Args:
        omega_m (float): Matter density parameter of the cosmology being evaluated.
        h (float): Dimensionless Hubble parameter of that same cosmology.

    Returns:
        float: ``(3/2) * Omega_m * (H0/c)^2``.
    """

    return 1.5 * float(omega_m) * hubble_distance_factor(h) ** 2


def radial_density_weight(psi: numpy.ndarray, h_over_h0: numpy.ndarray, h: float) -> numpy.ndarray:
    """Convert a redshift distribution to a comoving radial weight.

    Args:
        psi (numpy.ndarray): Redshift distribution samples.
        h_over_h0 (numpy.ndarray): ``E(z)`` of the active cosmology, broadcastable
            onto ``psi``.
        h (float): Dimensionless Hubble parameter of the active cosmology.

    Returns:
        numpy.ndarray: ``psi * E(z) * H0/c`` as float64.
    """

    return numpy.asarray(psi, dtype=numpy.float64) * numpy.asarray(
        h_over_h0, dtype=numpy.float64
    ) * hubble_distance_factor(h)


def magnification_response_q(slope_s: numpy.ndarray | Sequence[float]) -> numpy.ndarray:
    """Return the analytical magnification response ``q = 5s - 2``.

    Args:
        slope_s: Number-count slope samples stored by the magnification generator.

    Returns:
        numpy.ndarray: Response ``q`` as float64, same shape as ``slope_s``.
    """

    slopes = numpy.asarray(slope_s, dtype=numpy.float64)
    return 5.0 * slopes - 2.0


def ccl_magnification_bias(slope_s: numpy.ndarray | Sequence[float]) -> numpy.ndarray:
    """Return the slope array CCL expects as ``mag_bias``.

    Args:
        slope_s: Stored number-count slopes. This function does not apply
            ``5s-2``.

    Returns:
        numpy.ndarray: Float64 copy of ``s``.
    """

    return numpy.asarray(slope_s, dtype=numpy.float64).copy()


def analytical_magnification_response(
    slope_s: numpy.ndarray | Sequence[float],
) -> numpy.ndarray:
    """Return ``q`` for the analytical magnification-weighted distribution.

    Args:
        slope_s: Stored number-count slopes.

    Returns:
        numpy.ndarray: ``q = 5s - 2``.
    """

    return magnification_response_q(slope_s)


def magnification_weighted_lens(
    lens_phi: numpy.ndarray,
    response_q: numpy.ndarray | Sequence[float],
) -> numpy.ndarray:
    """Apply one bin-constant magnification response to each lens distribution.

    Args:
        lens_phi (numpy.ndarray): Lens weights shaped ``(bin, radial)``.
        response_q: Analytical response per lens bin. Both the MS and MI lens
            distributions use this weight.

    Returns:
        numpy.ndarray: ``lens_phi * q`` as float64.
    """

    phi = numpy.asarray(lens_phi, dtype=numpy.float64)
    response = numpy.asarray(response_q, dtype=numpy.float64)
    if phi.ndim != 2:
        raise ValueError(f"lens_phi must have shape (bin, radial); got {phi.shape}")
    if response.shape != (phi.shape[0],):
        raise ValueError(
            f"response_q length {response.shape} does not match {phi.shape[0]} lens bins"
        )
    return phi * response[:, numpy.newaxis]


def assemble_probe(
    probe: str,
    components: Mapping[str, numpy.ndarray],
) -> numpy.ndarray:
    """Sum the named components of one probe.

    Args:
        probe (str): ``EE``, ``TE`` or ``TT``.
        components: Component spectra keyed by ``SS``, ``MS`` and so on.
            Every component of ``probe`` is required. Magnification components
            must already use the response-weighted lens distribution.

    Returns:
        numpy.ndarray: Float64 sum, the final probe spectrum.
    """

    try:
        names = PROBE_COMPONENTS[probe]
    except KeyError as error:
        raise ValueError(f"Unknown probe {probe!r}") from error
    missing = [name for name in names if name not in components]
    if missing:
        raise KeyError(f"Probe {probe} is missing components: {', '.join(missing)}")
    total = numpy.array(components[names[0]], dtype=numpy.float64, copy=True)
    for name in names[1:]:
        total = total + numpy.asarray(components[name], dtype=numpy.float64)
    return total


def assemble_configuration(
    configuration: str,
    components: Mapping[str, Mapping[str, numpy.ndarray]],
) -> dict[str, numpy.ndarray]:
    """Assemble the probes required by Single, Double or Triple.

    Args:
        configuration (str): Title-case configuration label.
        components: Mapping of probe name to that probe's component spectra.

    Returns:
        dict[str, numpy.ndarray]: Final spectra for the requested probes only.
    """

    assembled: dict[str, numpy.ndarray] = {}
    for probe in configuration_probes(configuration):
        if probe not in components:
            raise KeyError(f"Configuration {configuration} requires probe {probe}")
        assembled[probe] = assemble_probe(probe, components[probe])
    return assembled
