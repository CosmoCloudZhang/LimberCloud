"""Component assembly and active-cosmology prefactors.

CCL number-count tracers take the magnification slope ``s``. The analytical
count response is ``q = 5s - 2``. Magnification-shear and magnification-intrinsic
terms both use the response-weighted lens distribution. Lensing amplitude and
the radial density conversion use the cosmology being evaluated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy
from scipy.constants import c as speed_of_light

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

    return float(h) * _H0_PER_H / float(speed_of_light)


def active_lensing_amplitude(omega_m: float, h: float) -> float:
    """Return the Limber lensing prefactor for the active cosmology.

    Args:
        omega_m (float): Matter density parameter of the cosmology being evaluated.
        h (float): Dimensionless Hubble parameter of that same cosmology.

    Returns:
        float: ``(3/2) * Omega_m * (H0/c)^2``.
    """

    return 1.5 * float(omega_m) * hubble_distance_factor(h) ** 2


def radial_density_weight(
    psi: numpy.ndarray, h_over_h0: numpy.ndarray, h: float
) -> numpy.ndarray:
    """Convert a redshift distribution to a comoving radial weight.

    Args:
        psi (numpy.ndarray): Redshift distribution samples.
        h_over_h0 (numpy.ndarray): ``E(z)`` of the active cosmology, broadcastable
            onto ``psi``.
        h (float): Dimensionless Hubble parameter of the active cosmology.

    Returns:
        numpy.ndarray: ``psi * E(z) * H0/c`` as float64.
    """

    return (
        numpy.asarray(psi, dtype=numpy.float64)
        * numpy.asarray(h_over_h0, dtype=numpy.float64)
        * hubble_distance_factor(h)
    )


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


def component_activity(alignment_amplitude=None, response_q=None) -> dict[str, bool]:
    """Identify wholly disabled terms before coefficient construction.

    Args:
        alignment_amplitude: Loaded signed IA amplitude across the radial grid;
            ``None`` or identically zero disables intrinsic-alignment terms.
            The redshift slope eta is not an amplitude/off switch.
        response_q: Loaded per-bin magnification response ``5*s-2``; ``None``
            or identically zero disables magnification terms. Individual zero
            bins retain the existing response-weighted leg/pair treatment.

    Returns:
        dict[str, bool]: Activity for every named EE/TE/TT component. Exact
        zeros are used; small nonzero or signed signals remain active.
    """

    active = []
    for name, values in (("alignment_amplitude", alignment_amplitude), ("response_q", response_q)):
        if values is None:
            active.append(False)
            continue
        array = numpy.asarray(values, dtype=numpy.float64)
        if array.size == 0 or not numpy.all(numpy.isfinite(array)):
            raise ValueError(f"{name} must contain finite, nonempty values")
        active.append(bool(numpy.any(array != 0.0)))
    ia_active, magnification_active = active
    return {
        component: ("I" not in component or ia_active)
        and ("M" not in component or magnification_active)
        for components in PROBE_COMPONENTS.values()
        for component in components
    }


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
        term = numpy.asarray(components[name], dtype=numpy.float64)
        if term.shape != total.shape:
            raise ValueError(
                f"Probe {probe} component {name} has shape {term.shape}, not "
                f"{total.shape}. Broadcasting incompatible component grids would "
                "silently change the spectrum."
            )
        total = total + term
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
