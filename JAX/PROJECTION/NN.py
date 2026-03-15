import jax
from jax import lax
from jax import vmap
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
    formula = lax.condition(a == 1.0, true_branch, false_branch, operand=None)
    
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
    formula = lax.condition(a == 1.0, true_branch, false_branch, operand=None)
    
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
    formula = lax.condition(a == 1.0, true_branch, false_branch, operand=None)
    
    element = power2 * formula / chi2
    return element

# Element
@jax.jit
def element(n, i, j, chi_grid, power_grid):
    grid_size = chi_grid.shape[0] - 1
    zeros_vector = jnp.zeros_like(power_grid[:, 0])
    
    condition0 = ((i < n) & (n < grid_size)) | ((j < n) & (n < grid_size))
    condition1 = (n == i) & (n == j) & (n < grid_size)
    condition2 = ( ((n == i) & (n + 1 == j)) | ((n + 1 == i) & (n == j)) ) & (n < grid_size)
    condition3 = (n + 1 == i) & (n + 1 == j) & (n + 1 < grid_size)
    
    branch_index = jnp.select(
        [condition0, condition1, condition2, condition3],
        [0, 1, 2, 3],
        default=0
    )
    
    def return_zeros(_):
        return zeros_vector
    
    def compute_element1(_):
        return element1(chi_grid[n], chi_grid[n + 1], power_grid[:, n], power_grid[:, n + 1])
    
    def compute_element2(_):
        return element2(chi_grid[n], chi_grid[n + 1], power_grid[:, n], power_grid[:, n + 1])
    
    def compute_element3(_):
        return element3(chi_grid[n], chi_grid[n + 1], power_grid[:, n], power_grid[:, n + 1])
    
    branches = [return_zeros, compute_element1, compute_element2, compute_element3]
    return lax.switch(branch_index, branches, None)

# Coefficient
@jax.jit
def coefficient(chi_grid, power_grid):
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    
    coefficients = jnp.zeros((grid_size + 1, grid_size + 1, ell_size + 1), dtype=power_grid.dtype)
    indices = jnp.arange(grid_size + 1, dtype=jnp.int32)
    
    def compute_elements_for_n(n):
        def compute_row(i):
            def compute_entry(j):
                return element(n, i, j, chi_grid, power_grid)
            return vmap(compute_entry, in_axes=(0,))(indices)
        return vmap(compute_row, in_axes=(0,))(indices)
    
    def accumulate_step(n, accumulated):
        return accumulated + compute_elements_for_n(n)
    
    return lax.fori_loop(0, grid_size, accumulate_step, coefficients)

# Spectra
@jax.jit
def spectra(factor, phi_a_grid, phi_b_grid, chi_grid, power_grid):
    coefficients = coefficient(chi_grid, power_grid)
    spectrum = factor * jnp.einsum('mi,nj,ijl->mnl', phi_a_grid, phi_b_grid, coefficients)
    return spectrum
