import jax
import jax.numpy as jnp
from jax import config, lax

from limbercloud.projection.jax_backend.local import _series_terms

config.update("jax_enable_x64", True)

# Element 1
@jax.jit
def _element1_nonzero(chi1, chi2, power1, power2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / jnp.where(power2 == 0, 1.0, power2)

    def true_branch(_):
        return jnp.full_like(p, 1 / 12)
    def false_branch(_):
        formula = ((a * (2 * ( - 2 + a) * a + 6 * p - a * (3 + a) * p)) / (2 * (-chi1 / chi2)) + (2 * a - 3 * p) * jnp.log(chi1 / chi2)) / a ** 3
        return formula
    formula = lax.cond(chi1 == 0.0, true_branch, false_branch, operand=None)

    element = power2 * formula / chi2
    return element

# Element 2
@jax.jit
def _element2_nonzero(chi1, chi2, power1, power2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / jnp.where(power2 == 0, 1.0, power2)

    def true_branch(_):
        return jnp.full_like(p, 1 / 12)
    def false_branch(_):
        formula = (( - a) * ( - 6 * p + a * (4 + p)) + 2 * (a ** 2 + 3 * p - 2 * a * (1 + p)) * jnp.log(chi1 / chi2)) / (2 * a ** 3)
        return formula
    formula = lax.cond(chi1 == 0.0, true_branch, false_branch, operand=None)

    element = power2 * formula / chi2
    return element

# Element 3
@jax.jit
def _element3_nonzero(chi1, chi2, power1, power2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / jnp.where(power2 == 0, 1.0, power2)

    def true_branch(_):
        return jnp.full_like(p, 1 / 4)
    def false_branch(_):
        formula = ((1 / 2) * a * ( - 6 * p + a * (4 - 2 * a + 5 * p)) - (-chi1 / chi2) * ( - 3 * p + a * (2 + p)) * jnp.log(chi1 / chi2)) / a ** 3
        return formula
    formula = lax.cond(chi1 == 0.0, true_branch, false_branch, operand=None)

    element = power2 * formula / chi2
    return element

@jax.jit
def _nn_series(chi1, chi2, power1, power2, which):
    a = (chi2-chi1)/chi2
    def step(m, total):
        m0 = power2/(m+1)+(power1-power2)/(m+2)
        m1 = power2/(m+2)+(power1-power2)/(m+3)
        m2 = power2/(m+3)+(power1-power2)/(m+4)
        value = jnp.where(which == 1, m2, jnp.where(which == 2, m1-m2, m0-2*m1+m2))
        return total + (m+1)*a**m*value
    return a/chi2*lax.fori_loop(0, _series_terms(a), step, jnp.zeros_like(power1))


@jax.jit
def element1(chi1, chi2, power1, power2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    a = (chi2-chi1)/chi2
    def regular(_):
        return _nn_series(chi1, chi2, power1, power2, 1)
    def other(_):
        return _element1_zero_safe(chi1, chi2, power1, power2)
    return lax.cond((chi1 > 0) & (a <= 0.75), regular, other, None)


@jax.jit
def _element1_zero_safe(chi1, chi2, power1, power2):
    zero = power2 == 0
    safe_power2 = jnp.where(zero, 1.0, power2)
    regular = _element1_nonzero(chi1, chi2, power1, safe_power2)
    def corrected(_):
        left = _element1_nonzero(chi1, chi2, jnp.ones_like(power1), jnp.ones_like(power1)) - _element1_nonzero(chi1, chi2, jnp.zeros_like(power1), jnp.ones_like(power2))
        left = jnp.where(chi1 == 0.0, 0.0, left)
        return jnp.where(zero, power1*left, regular)
    return lax.cond(jnp.any(zero), corrected, lambda _: regular, None)

@jax.jit
def element2(chi1, chi2, power1, power2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    a = (chi2-chi1)/chi2
    def regular(_):
        return _nn_series(chi1, chi2, power1, power2, 2)
    def other(_):
        return _element2_zero_safe(chi1, chi2, power1, power2)
    return lax.cond((chi1 > 0) & (a <= 0.75), regular, other, None)


@jax.jit
def _element2_zero_safe(chi1, chi2, power1, power2):
    zero = power2 == 0
    safe_power2 = jnp.where(zero, 1.0, power2)
    regular = _element2_nonzero(chi1, chi2, power1, safe_power2)
    def corrected(_):
        left = _element2_nonzero(chi1, chi2, jnp.ones_like(power1), jnp.ones_like(power1)) - _element2_nonzero(chi1, chi2, jnp.zeros_like(power1), jnp.ones_like(power2))
        left = jnp.where(chi1 == 0.0, 0.0, left)
        return jnp.where(zero, power1*left, regular)
    return lax.cond(jnp.any(zero), corrected, lambda _: regular, None)

@jax.jit
def element3(chi1, chi2, power1, power2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    a = (chi2-chi1)/chi2
    def regular(_):
        return _nn_series(chi1, chi2, power1, power2, 3)
    def other(_):
        return _element3_zero_safe(chi1, chi2, power1, power2)
    return lax.cond((chi1 > 0) & (a <= 0.75), regular, other, None)


@jax.jit
def _element3_zero_safe(chi1, chi2, power1, power2):
    zero = power2 == 0
    safe_power2 = jnp.where(zero, 1.0, power2)
    regular = _element3_nonzero(chi1, chi2, power1, safe_power2)
    def corrected(_):
        left = _element3_nonzero(chi1, chi2, jnp.ones_like(power1), jnp.ones_like(power1)) - _element3_nonzero(chi1, chi2, jnp.zeros_like(power1), jnp.ones_like(power2))
        left = jnp.where(chi1 == 0.0, 0.0, left)
        return jnp.where(zero, power1*left, regular)
    return lax.cond(jnp.any(zero), corrected, lambda _: regular, None)

# Coefficient
# There are N intervals and N+1 nodes. Every interval stores all four hat
# placements, including the rising-hat diagonal of the final interval. The
# density vanishes outside the finite domain, not at its last node, so the
# node-N basis function is supported across [chi[N-1], chi[N]].
@jax.jit
def coefficient(chi_grid, power_grid):
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    coefficients = jnp.zeros((grid_size + 1, grid_size + 1, ell_size + 1))

    def accumulate_step(n, coefficients):
        value1 = element1(chi_grid[n], chi_grid[n + 1], power_grid[:, n], power_grid[:, n + 1])
        value2 = element2(chi_grid[n], chi_grid[n + 1], power_grid[:, n], power_grid[:, n + 1])
        value3 = element3(chi_grid[n], chi_grid[n + 1], power_grid[:, n], power_grid[:, n + 1])

        coefficients = coefficients.at[n, n, :].add(value1)
        coefficients = coefficients.at[n, n + 1, :].add(value2)
        coefficients = coefficients.at[n + 1, n, :].add(value2)
        coefficients = coefficients.at[n + 1, n + 1, :].add(value3)

        return coefficients

    return lax.fori_loop(0, grid_size, accumulate_step, coefficients)

# Spectra
@jax.jit
def spectra(factor, phi_a_grid, phi_b_grid, chi_grid, power_grid):
    coefficients = coefficient(chi_grid, power_grid)
    # jnp.einsum takes no dtype argument; float64 comes from jax_enable_x64.
    return factor * jnp.einsum('mi,nj,ijl->mnl', phi_a_grid, phi_b_grid, coefficients)
