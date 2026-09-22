"""Final-interval lensing contributions, where the source hat straddles x.

The ordinary NS/SN/SS terminal elements assume the whole node-N source hat lies
above the evaluation point, which makes the source integral linear in ``x``.
That is true only while ``x <= chi[N-1]``. On the last interval ``[a, b]`` the
evaluation point lies inside the source support, so the lower limit of the
source integral is ``x`` itself:

``H_N(x) = integral_x^b [(u-a)/(b-a)] [(u-x)/u] du``
``      = [(b-x)^2/2 - a*(b-x) + a*x*log(b/x)]/(b-a)``

``F_N(x) = integral_x^b [(b-u)/(b-a)] [(u-x)/u] du``
``      = [(b^2-x^2)/2 - b*x*log(b/x)]/(b-a)``

``F_N`` is the matching partial integral of the node ``N-1`` hat, whose falling
half also lies on ``[a, b]``.

Closed forms for these terminal integrals exist but carry ``(b-a)^-4`` and
``(b-a)^-5`` prefactors that cancel against the logarithm, so they lose all
significance on narrow terminal intervals. The same derived integrands are
therefore evaluated with a fixed Gauss-Legendre rule. The integrands are
analytic on ``[a, b]`` for ``a > 0``, so the rule converges geometrically and
reproduces adaptive quadrature to machine precision, while staying stable for
any node spacing. The cost is one fixed rule on one interval per coefficient
build, not per node pair.
"""

import numba
import numpy

TERMINAL_QUADRATURE_ORDER = 48

_TERMINAL_NODE, _TERMINAL_WEIGHT = numpy.polynomial.legendre.leggauss(
    TERMINAL_QUADRATURE_ORDER
)


@numba.njit(cache=True)
def terminal_rising_source(chi, chi_lower, chi_upper):
    """Return the node-N source integral for an evaluation point inside [a, b]."""

    gap = chi_upper - chi
    return (
        0.5 * gap * gap - chi_lower * gap + chi_lower * chi * numpy.log(chi_upper / chi)
    ) / (chi_upper - chi_lower)


@numba.njit(cache=True)
def terminal_falling_source(chi, chi_lower, chi_upper):
    """Return the node-(N-1) source integral for an evaluation point inside [a, b]."""

    return (
        0.5 * (chi_upper * chi_upper - chi * chi)
        - chi_upper * chi * numpy.log(chi_upper / chi)
    ) / (chi_upper - chi_lower)


@numba.njit(cache=True)
def density_terminal(chi1, chi2, power1, power2, redshift1, redshift2, rising):
    """Integrate one density hat against the terminal lensing source integral.

    Args:
        chi1: Start ``a`` of the final interval.
        chi2: End ``b`` of the final interval.
        power1: Power at ``a``, shaped over multipoles.
        power2: Power at ``b``, shaped over multipoles.
        redshift1: Redshift at ``a``.
        redshift2: Redshift at ``b``.
        rising: True for the node-N density hat, false for node ``N-1``.

    Returns:
        Integral of ``P(x) (1+z)(x) / x * hat(x) * H_N(x)`` over ``[a, b]``.
    """

    half = 0.5 * (chi2 - chi1)
    middle = 0.5 * (chi2 + chi1)
    total = numpy.zeros_like(power1)
    for index in range(_TERMINAL_NODE.shape[0]):
        chi = middle + half * _TERMINAL_NODE[index]
        fraction = (chi - chi1) / (chi2 - chi1)
        power = power1 * (1.0 - fraction) + power2 * fraction
        redshift = (1.0 + redshift1) * (1.0 - fraction) + (1.0 + redshift2) * fraction
        basis = fraction if rising else 1.0 - fraction
        source = terminal_rising_source(chi, chi1, chi2)
        total = total + _TERMINAL_WEIGHT[index] * half * power * redshift / chi * basis * source
    return total


@numba.njit(cache=True)
def lensing_terminal(chi1, chi2, power1, power2, redshift1, redshift2, both_rising):
    """Integrate two terminal lensing source integrals against each other.

    Args:
        chi1: Start ``a`` of the final interval.
        chi2: End ``b`` of the final interval.
        power1: Power at ``a``, shaped over multipoles.
        power2: Power at ``b``, shaped over multipoles.
        redshift1: Redshift at ``a``.
        redshift2: Redshift at ``b``.
        both_rising: True for ``H_N * H_N``, false for the cross term
            ``F_N * H_N`` between nodes ``N-1`` and ``N``.

    Returns:
        Integral of ``P(x) (1+z)(x)^2 * G_i(x) * G_j(x)`` over ``[a, b]``.
    """

    half = 0.5 * (chi2 - chi1)
    middle = 0.5 * (chi2 + chi1)
    total = numpy.zeros_like(power1)
    for index in range(_TERMINAL_NODE.shape[0]):
        chi = middle + half * _TERMINAL_NODE[index]
        fraction = (chi - chi1) / (chi2 - chi1)
        power = power1 * (1.0 - fraction) + power2 * fraction
        redshift = (1.0 + redshift1) * (1.0 - fraction) + (1.0 + redshift2) * fraction
        rising = terminal_rising_source(chi, chi1, chi2)
        other = rising if both_rising else terminal_falling_source(chi, chi1, chi2)
        total = total + _TERMINAL_WEIGHT[index] * half * power * redshift * redshift * other * rising
    return total
