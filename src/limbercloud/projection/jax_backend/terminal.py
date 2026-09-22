"""Final-interval lensing contributions, where the source hat straddles x.

This mirrors :mod:`limbercloud.projection.numba_backend.terminal`. The ordinary
NS/SN/SS terminal elements assume the whole node-N source hat lies above the
evaluation point, which makes the source integral linear in ``x``. That holds
only while ``x <= chi[N-1]``. On the last interval ``[a, b]`` the evaluation
point lies inside the source support, so the source integral starts at ``x``:

``H_N(x) = [(b-x)^2/2 - a*(b-x) + a*x*log(b/x)]/(b-a)``
``F_N(x) = [(b^2-x^2)/2 - b*x*log(b/x)]/(b-a)``

``F_N`` is the matching partial integral of the node ``N-1`` hat. Both
integrands are analytic on ``[a, b]`` for ``a > 0``, so the fixed Gauss-Legendre
rule below reproduces adaptive quadrature to machine precision and stays stable
for any node spacing, unlike the closed forms whose ``(b-a)^-4`` prefactors
cancel against the logarithm.
"""

import jax
import jax.numpy as jnp
import numpy
from jax import config

config.update("jax_enable_x64", True)

TERMINAL_QUADRATURE_ORDER = 48

_node, _weight = numpy.polynomial.legendre.leggauss(TERMINAL_QUADRATURE_ORDER)
_TERMINAL_NODE = jnp.asarray(_node, dtype=jnp.float64)
_TERMINAL_WEIGHT = jnp.asarray(_weight, dtype=jnp.float64)


def _rising_source(chi, chi1, chi2):
    gap = chi2 - chi
    return (0.5 * gap * gap - chi1 * gap + chi1 * chi * jnp.log(chi2 / chi)) / (chi2 - chi1)


def _falling_source(chi, chi1, chi2):
    return (0.5 * (chi2 * chi2 - chi * chi) - chi2 * chi * jnp.log(chi2 / chi)) / (chi2 - chi1)


def _quadrature(chi1, chi2, power1, power2, redshift1, redshift2, kernel):
    half = 0.5 * (chi2 - chi1)
    middle = 0.5 * (chi2 + chi1)
    chi = middle + half * _TERMINAL_NODE
    fraction = (chi - chi1) / (chi2 - chi1)
    power = power1[None, :] * (1.0 - fraction)[:, None] + power2[None, :] * fraction[:, None]
    redshift = (1.0 + redshift1) * (1.0 - fraction) + (1.0 + redshift2) * fraction
    weight = kernel(chi, fraction, redshift, chi1, chi2)
    return half * jnp.einsum('q,q,qe->e', _TERMINAL_WEIGHT, weight, power)


@jax.jit
def density_terminal_falling(chi1, chi2, power1, power2, redshift1, redshift2):
    """Integrate the node-(N-1) density hat against ``H_N``."""

    def kernel(chi, fraction, redshift, lower, upper):
        return redshift / chi * (1.0 - fraction) * _rising_source(chi, lower, upper)

    return _quadrature(chi1, chi2, power1, power2, redshift1, redshift2, kernel)


@jax.jit
def density_terminal_rising(chi1, chi2, power1, power2, redshift1, redshift2):
    """Integrate the node-N density hat against ``H_N``."""

    def kernel(chi, fraction, redshift, lower, upper):
        return redshift / chi * fraction * _rising_source(chi, lower, upper)

    return _quadrature(chi1, chi2, power1, power2, redshift1, redshift2, kernel)


@jax.jit
def lensing_terminal_cross(chi1, chi2, power1, power2, redshift1, redshift2):
    """Integrate ``F_N * H_N``, the node ``N-1`` by node ``N`` lensing cross term."""

    def kernel(chi, fraction, redshift, lower, upper):
        return redshift * redshift * _falling_source(chi, lower, upper) * _rising_source(chi, lower, upper)

    return _quadrature(chi1, chi2, power1, power2, redshift1, redshift2, kernel)


@jax.jit
def lensing_terminal_diagonal(chi1, chi2, power1, power2, redshift1, redshift2):
    """Integrate ``H_N * H_N``, the node-N lensing diagonal."""

    def kernel(chi, fraction, redshift, lower, upper):
        rising = _rising_source(chi, lower, upper)
        return redshift * redshift * rising * rising

    return _quadrature(chi1, chi2, power1, power2, redshift1, redshift2, kernel)
