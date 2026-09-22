"""Matched-integrand quadrature and the separate NUMERIC comparison.

The analytical number-count coefficients integrate
``P(chi) * hat_i * hat_j / chi^2``. On the observer interval the adopted power
is ``P = P_1 (chi / chi_1)^3``; later intervals use power linear in ``chi``.
The lensing families additionally linearise ``1+z``. NUMERIC linear interpolation
instead interpolates the scale factor ``a(chi)`` and does not special-case the
observer power, so it is an approximation comparison rather than a quadrature
of that integrand.
"""

from __future__ import annotations

import scipy.integrate

from limbercloud.validation.contract import NN_OBSERVER_FACTORS

NUMERIC_SCIPY_KIND = {
    "linear": "slinear",
    "quadratic": "quadratic",
    "cubic": "cubic",
}


def numeric_interpolation_contract(order: str) -> dict[str, str]:
    """Describe one NUMERIC setting without treating it as the analytic integrand.

    Args:
        order (str): ``linear``, ``quadratic`` or ``cubic``.

    Returns:
        dict[str, str]: SciPy kind and the quantities that setting interpolates.
    """

    key = order.strip().lower()
    try:
        kind = NUMERIC_SCIPY_KIND[key]
    except KeyError as error:
        choices = ", ".join(NUMERIC_SCIPY_KIND)
        raise ValueError(f"Unknown NUMERIC order {order!r}; expected one of: {choices}") from error
    return {
        "order": key,
        "scipy_kind": kind,
        "distribution": "interp1d of phi(chi)",
        "scale_factor": "interp1d of a(chi)=1/(1+z); not linear 1+z",
        "power": "interp1d of P(chi), including the first interval",
        "observer_power": "no cubic P=P1*(chi/chi1)^3 special case",
        "role": "approximation comparison, not matched-integrand quadrature",
    }


def scale_factor_from_linear_one_plus_z(
    x: float,
    redshift_left: float,
    redshift_right: float,
) -> float:
    """Evaluate ``a`` when ``1+z`` is linear in the interval coordinate.

    Args:
        x (float): Interval coordinate in ``[0, 1]``.
        redshift_left (float): Redshift at ``x=0``.
        redshift_right (float): Redshift at ``x=1``.

    Returns:
        float: ``1 / (1+z)`` at ``x``.
    """

    one_plus_z = (1.0 + float(redshift_left)) * (1.0 - float(x)) + (
        1.0 + float(redshift_right)
    ) * float(x)
    return 1.0 / one_plus_z


def scale_factor_from_linear_a(
    x: float,
    redshift_left: float,
    redshift_right: float,
) -> float:
    """Evaluate the NUMERIC linear interpolant of ``a`` itself.

    Args:
        x (float): Interval coordinate in ``[0, 1]``.
        redshift_left (float): Redshift at ``x=0``.
        redshift_right (float): Redshift at ``x=1``.

    Returns:
        float: Linear interpolation of the endpoint scale factors.
    """

    a_left = 1.0 / (1.0 + float(redshift_left))
    a_right = 1.0 / (1.0 + float(redshift_right))
    return a_left * (1.0 - float(x)) + a_right * float(x)


def _hats(chi: float, chi_left: float, chi_right: float) -> tuple[float, float]:
    if chi_left == 0.0:
        rising = chi / chi_right
        return 1.0 - rising, rising
    width = chi_right - chi_left
    falling = (chi_right - chi) / width
    rising = (chi - chi_left) / width
    return falling, rising


def _power(chi: float, chi_left: float, chi_right: float, power_left: float, power_right: float) -> float:
    if chi_left == 0.0:
        return float(power_right) * (chi / chi_right) ** 3
    x = (chi - chi_left) / (chi_right - chi_left)
    return float(power_left) * (1.0 - x) + float(power_right) * x


def nn_hat_product(which: str, falling: float, rising: float) -> float:
    """Return one NN hat product.

    Args:
        which (str): ``element1`` (falling-falling), ``element2`` (cross) or
            ``element3`` (rising-rising).
        falling (float): Left-node hat.
        rising (float): Right-node hat.

    Returns:
        float: Hat product entering the NN integrand.
    """

    if which == "element1":
        return falling * falling
    if which == "element2":
        return falling * rising
    if which == "element3":
        return rising * rising
    raise ValueError(f"Unknown NN element {which!r}")


def nn_interval_quadrature(
    chi_left: float,
    chi_right: float,
    power_left: float,
    power_right: float,
    which: str,
) -> float:
    """Integrate the analytical NN integrand on one interval.

    The observer interval (``chi_left == 0``) uses cubic power. Any later
    interval uses power linear in comoving distance. The quadrature is
    independent of the closed-form coefficient implementation.

    Args:
        chi_left (float): Left edge. Zero selects the cubic observer policy.
        chi_right (float): Right edge, strictly larger than ``chi_left``.
        power_left (float): Power at the left edge. Ignored for the cubic
            observer policy, matching ``P = P_right * (chi/chi_right)^3``.
        power_right (float): Power at the right edge.
        which (str): ``element1``, ``element2`` or ``element3``.

    Returns:
        float: Integral of ``P * hat_i * hat_j / chi^2`` across the interval.
    """

    chi_left = float(chi_left)
    chi_right = float(chi_right)
    if not chi_right > chi_left:
        raise ValueError("chi_right must be greater than chi_left")
    if chi_left < 0:
        raise ValueError("chi_left must be >= 0")
    if which not in NN_OBSERVER_FACTORS:
        raise ValueError(f"Unknown NN element {which!r}")

    def integrand(chi: float) -> float:
        falling, rising = _hats(chi, chi_left, chi_right)
        power = _power(chi, chi_left, chi_right, power_left, power_right)
        return power * nn_hat_product(which, falling, rising) / chi**2

    if chi_left == 0.0:
        # P/chi^2 is O(chi) at the origin under the cubic policy, so the
        # substituted coordinate x=chi/chi_right is regular.
        def regularised(x: float) -> float:
            return integrand(x * chi_right) * chi_right

        value, _ = scipy.integrate.quad(regularised, 0.0, 1.0, epsabs=1e-12, limit=200)
        return float(value)

    value, _ = scipy.integrate.quad(integrand, chi_left, chi_right, epsabs=1e-12, limit=200)
    return float(value)


def nn_observer_closed_form(chi_right: float, power_right: float, which: str) -> float:
    """Return ``(P_1/chi_1)`` times the cubic-policy rational factor.

    Args:
        chi_right (float): Right edge of the observer interval.
        power_right (float): Power at that edge.
        which (str): ``element1``, ``element2`` or ``element3``.

    Returns:
        float: Exact observer-interval integral for the cubic power policy.
        ``element3`` uses ``1/4``, not the withdrawn linear-power value ``1/2``.
    """

    return float(power_right) / float(chi_right) * NN_OBSERVER_FACTORS[which]
