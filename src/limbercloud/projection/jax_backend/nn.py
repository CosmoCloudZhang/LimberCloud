import jax
from jax import lax
from jax import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)

# Element 1
@jax.jit
def element1(chi1, chi2, power1, power2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / power2
    
    def true_branch(_):
        return jnp.full_like(p, 1 / 12)
    def false_branch(_):
        formula = ((a * (2 * ( - 2 + a) * a + 6 * p - a * (3 + a) * p)) / (2 * ( - 1 + a)) + (2 * a - 3 * p) * jnp.log(1 - a)) / a ** 3
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    element = power2 * formula / chi2
    return element

# Element 2
@jax.jit
def element2(chi1, chi2, power1, power2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / power2
    
    def true_branch(_):
        return jnp.full_like(p, 1 / 12)
    def false_branch(_):
        formula = (( - a) * ( - 6 * p + a * (4 + p)) + 2 * (a ** 2 + 3 * p - 2 * a * (1 + p)) * jnp.log(1 - a)) / (2 * a ** 3)
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    element = power2 * formula / chi2
    return element

# Element 3
@jax.jit
def element3(chi1, chi2, power1, power2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / power2
    
    def true_branch(_):
        return jnp.full_like(p, 1 / 4)
    def false_branch(_):
        formula = ((1 / 2) * a * ( - 6 * p + a * (4 - 2 * a + 5 * p)) - ( - 1 + a) * ( - 3 * p + a * (2 + p)) * jnp.log(1 - a)) / a ** 3
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    element = power2 * formula / chi2
    return element

# Coefficient
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
        coefficients = coefficients.at[n + 1, n + 1, :].add(
            jnp.where(n + 1 < grid_size, value3, jnp.zeros_like(value3))
        )
        
        return coefficients
    
    return lax.fori_loop(0, grid_size, accumulate_step, coefficients)

# Spectra
@jax.jit
def spectra(factor, phi_a_grid, phi_b_grid, chi_grid, power_grid):
    coefficients = coefficient(chi_grid, power_grid)
    return factor * jnp.einsum('mi,nj,ijl->mnl', phi_a_grid, phi_b_grid, coefficients, dtype=jnp.float64)
