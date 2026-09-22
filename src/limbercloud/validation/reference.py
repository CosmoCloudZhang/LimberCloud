"""Independent reference calculations for the analytical coefficient kernels.

Nothing here calls the closed forms under test. Hats, the declared power law,
the linearised ``1+z`` and the lensing source integrals are reconstructed from
their definitions and integrated with SciPy quadrature, so a formula error and
an assembly error both show up as a disagreement.

The analytical number-count coefficients integrate ``P(chi) hat_i hat_j /
chi^2``. On the observer interval the adopted power is ``P = P_1 (chi/chi_1)^3``;
later intervals use power linear in ``chi``. The lensing families additionally
linearise ``1+z`` across each interval. NUMERIC linear interpolation instead
interpolates the scale factor ``a(chi)`` and does not special-case the observer
power, so it is an approximation comparison rather than a quadrature of that
integrand.
"""

from __future__ import annotations

import numpy
from scipy.integrate import quad

from limbercloud.validation.contract import NN_OBSERVER_FACTORS

NUMERIC_SCIPY_KIND = {
    "linear": "slinear",
    "quadratic": "quadratic",
    "cubic": "cubic",
}

# Absolute and relative tolerances for reference comparisons. They are set by the
# adaptive quadrature request below and by the cancellation scale of the
# integrands, not by whatever the implementation happens to return.
QUADRATURE_ABSOLUTE_TOLERANCE = 1e-12
QUADRATURE_RELATIVE_TOLERANCE = 1e-10
QUADRATURE_SUBDIVISION_LIMIT = 400


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
        raise ValueError(
            f"Unknown NUMERIC order {order!r}; expected one of: {choices}"
        ) from error
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

    return 1.0 / one_plus_z_linear(x, redshift_left, redshift_right)


def one_plus_z_linear(x: float, redshift_left: float, redshift_right: float) -> float:
    """Evaluate the linearised ``1+z`` used by the lensing kernels.

    Args:
        x (float): Interval coordinate in ``[0, 1]``.
        redshift_left (float): Redshift at ``x=0``.
        redshift_right (float): Redshift at ``x=1``.

    Returns:
        float: ``1+z`` interpolated linearly between the interval endpoints.
    """

    return (1.0 + float(redshift_left)) * (1.0 - float(x)) + (
        1.0 + float(redshift_right)
    ) * float(x)


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


def hat(chi, node_index: int, chi_grid) -> float:
    """Evaluate the piecewise-linear basis function of one radial node.

    Args:
        chi: Radial coordinate.
        node_index (int): Node whose hat is evaluated, ``0..N``.
        chi_grid: Increasing node positions, length ``N+1``.

    Returns:
        float: Hat value. It is zero outside ``[chi[i-1], chi[i+1]]``; the first
        and last nodes carry the corresponding one-sided halves. Density is zero
        outside the finite domain, not at its last node.
    """

    grid = numpy.asarray(chi_grid, dtype=numpy.float64)
    last = grid.size - 1
    value = numpy.zeros_like(numpy.asarray(chi, dtype=numpy.float64))
    position = numpy.asarray(chi, dtype=numpy.float64)
    if node_index > 0:
        left, right = grid[node_index - 1], grid[node_index]
        inside = (position >= left) & (position <= right)
        value = numpy.where(inside, (position - left) / (right - left), value)
    if node_index < last:
        left, right = grid[node_index], grid[node_index + 1]
        inside = (position > left) & (position <= right)
        value = numpy.where(inside, (right - position) / (right - left), value)
    return float(value) if numpy.ndim(chi) == 0 else value


def _hats(chi: float, chi_left: float, chi_right: float) -> tuple[float, float]:
    if chi_left == 0.0:
        rising = chi / chi_right
        return 1.0 - rising, rising
    width = chi_right - chi_left
    falling = (chi_right - chi) / width
    rising = (chi - chi_left) / width
    return falling, rising


def _power(
    chi: float, chi_left: float, chi_right: float, power_left: float, power_right: float
) -> float:
    if chi_left == 0.0:
        return float(power_right) * (chi / chi_right) ** 3
    x = (chi - chi_left) / (chi_right - chi_left)
    return float(power_left) * (1.0 - x) + float(power_right) * x


def declared_power(chi, chi_grid, power_nodes) -> float:
    """Evaluate the declared radial power law of the analytical kernels.

    Args:
        chi: Radial coordinate inside the grid.
        chi_grid: Increasing node positions.
        power_nodes: Power at each node, same length as ``chi_grid``.

    Returns:
        float: ``P_1 (chi/chi_1)^3`` on an observer interval starting at zero,
        and power linear in ``chi`` on every later interval.
    """

    grid = numpy.asarray(chi_grid, dtype=numpy.float64)
    nodes = numpy.asarray(power_nodes, dtype=numpy.float64)
    position = float(chi)
    index = int(
        numpy.clip(
            numpy.searchsorted(grid, position, side="left") - 1, 0, grid.size - 2
        )
    )
    return _power(
        position, grid[index], grid[index + 1], nodes[index], nodes[index + 1]
    )


def _quad(function, lower: float, upper: float) -> float:
    value, _ = quad(
        function,
        float(lower),
        float(upper),
        epsabs=QUADRATURE_ABSOLUTE_TOLERANCE,
        epsrel=QUADRATURE_RELATIVE_TOLERANCE,
        limit=QUADRATURE_SUBDIVISION_LIMIT,
    )
    return float(value)


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

        return _quad(regularised, 0.0, 1.0)

    return _quad(integrand, chi_left, chi_right)


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


def terminal_source_integral(x: float, lower: float, upper: float) -> float:
    """Return the final-interval lensing source integral ``H_N(x)``.

    For the final interval ``[a, b]`` the source basis at node ``N`` is the
    rising hat ``(u-a)/(b-a)``. The lensing efficiency integrates sources above
    the evaluation point, so for ``a <= x <= b`` the lower limit is ``x``:

    ``H_N(x) = integral_x^b [(u-a)/(b-a)] [(u-x)/u] du``
    ``      = [(b-x)^2/2 - a*(b-x) + a*x*log(b/x)] / (b-a)``

    The expression valid below ``a`` cannot be used unchanged inside ``[a, b]``.

    Args:
        x (float): Evaluation point in ``[a, b]``.
        lower (float): ``a``, the start of the final interval.
        upper (float): ``b``, the end of the final interval.

    Returns:
        float: The source integral. It is exactly zero at ``x = b`` and is
        continuous with :func:`full_source_integral` at ``x = a``.
    """

    a = float(lower)
    b = float(upper)
    position = float(x)
    if not b > a:
        raise ValueError("The final interval requires upper > lower")
    if position < a or position > b:
        raise ValueError(f"x={position} is outside the final interval [{a}, {b}]")
    return _rising_source_segment(position, position, b, a, b)


def full_source_integral(x: float, lower: float, upper: float) -> float:
    """Return the lensing source integral when the whole hat lies above ``x``.

    Args:
        x (float): Evaluation point at or below ``lower``.
        lower (float): ``a``, the start of the source interval.
        upper (float): ``b``, the end of the source interval.

    Returns:
        float: ``integral_a^b [(u-a)/(b-a)] [(u-x)/u] du``, which is linear in
        ``x``. This is the branch the current NS/SN/SS terminal elements use.
    """

    a = float(lower)
    b = float(upper)
    position = float(x)
    if not b > a:
        raise ValueError("The source interval requires upper > lower")
    if position > a:
        raise ValueError(
            f"x={position} lies inside [{a}, {b}]; use terminal_source_integral instead"
        )
    return _rising_source_segment(position, a, b, a, b)


def source_integral_quadrature(x: float, lower: float, upper: float) -> float:
    """Integrate the rising-hat lensing source kernel directly.

    Args:
        x (float): Evaluation point, at or below ``upper``.
        lower (float): ``a``, the start of the source interval.
        upper (float): ``b``, the end of the source interval.

    Returns:
        float: ``integral_max(x,a)^b [(u-a)/(b-a)] [(u-x)/u] du`` evaluated by
        adaptive quadrature, independent of either closed form.
    """

    a = float(lower)
    b = float(upper)
    position = float(x)
    start = max(position, a)
    if start >= b:
        return 0.0

    return _rising_source_segment(position, start, b, a, b)


def _rising_source_segment(
    x: float, start: float, stop: float, left: float, right: float
) -> float:
    """Integrate a rising source half in a stable unit-interval coordinate."""

    if stop <= start:
        return 0.0
    gap = stop - start

    def integrand(v: float) -> float:
        offset = gap * v
        u = start + offset
        geometry = 1.0 if x == 0.0 else ((start - x) + offset) / u
        return ((start - left) + offset) / (right - left) * geometry

    return gap * _quad(integrand, 0.0, 1.0)


def _falling_source_segment(
    x: float, start: float, stop: float, left: float, right: float
) -> float:
    """Integrate a falling source half in a stable unit-interval coordinate."""

    if stop <= start:
        return 0.0
    gap = stop - start

    def integrand(v: float) -> float:
        offset = gap * v
        u = start + offset
        geometry = 1.0 if x == 0.0 else ((start - x) + offset) / u
        return ((right - start) - offset) / (right - left) * geometry

    return gap * _quad(integrand, 0.0, 1.0)


def lensing_efficiency(x: float, node_index: int, chi_grid) -> float:
    """Return the nodal lensing efficiency ``G_j(x)`` by independent quadrature.

    Each source half uses its original moving-limit integral, mapped onto a
    unit interval to avoid cancellation on narrow supports. No production
    closed form or stable series is reused.

    Args:
        x (float): Evaluation point.
        node_index (int): Source node whose hat is integrated.
        chi_grid: Increasing node positions.

    Returns:
        float: ``integral_x^{chi_max} hat_j(u) (u-x)/u du``. A hat straddling
        ``x`` is only partly counted, which is exactly the terminal case the
        below-interval expression cannot represent.
    """

    grid = numpy.asarray(chi_grid, dtype=numpy.float64)
    position = float(x)
    last = grid.size - 1
    total = 0.0
    if node_index > 0:
        left, right = float(grid[node_index - 1]), float(grid[node_index])
        total += _rising_source_segment(
            position, max(position, left), right, left, right
        )
    if node_index < last:
        left, right = float(grid[node_index]), float(grid[node_index + 1])
        total += _falling_source_segment(
            position, max(position, left), right, left, right
        )
    return total
