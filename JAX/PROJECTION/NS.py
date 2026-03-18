import jax
from jax import lax
from jax import vmap
from jax import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)

# Element 1
@jax.jit
def element1(chi1, chi2, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (63 - 41 * z) / 25200)
    def false_branch(_):
        formula = (1 / (720 * a ** 4)) * (a * (10 * a * (12 * a * ( - 6 + ( - 3 + a) * a) + (60 + a * (30 + (20 - 9 * a) * a)) * p) +
(10 * a * (60 + a * (30 + (20 - 9 * a) * a)) + 9 * ( - 60 + a * ( - 30 + a * ( - 20 + a * ( - 15 + 8 * a)))) * p) * z) +
60 * ( - 12 * a ** 2 - 9 * p * z + 10 * a * (p + z) + a ** 4 * (6 - 4 * p - 4 * z + 3 * p * z)) * jnp.log(1 - a))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Element 2
@jax.jit
def element2(chi1, chi2, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (21 - 11 * z) / 12600)
    def false_branch(_):
        formula = (1 / (720 * a ** 4)) * (a * (540 * p * z + 5 * a ** 3 * ( - 144 + 32 * p + 32 * z - 13 * p * z) + 6 * a ** 4 * (10 - 5 * p - 5 * z + 3 * p * z) - 60 * a ** 2 * ( - 12 - 7 * z + p * ( - 7 + 2 * z)) -
30 * a * (20 * z + p * (20 + 11 * z))) + 60 * ( - 18 * a ** 3 + a ** 4 * (6 + p * ( - 2 + z) - 2 * z) + 9 * p * z + 12 * a ** 2 * (1 + p + z) - 10 * a * (p + z + p * z)) * jnp.log(1 - a))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Element 3
@jax.jit
def element3(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5 - chi3) / (2 * chi2)
    c = chi3 * jnp.log(chi4 / chi3) / (chi4 - chi3) - chi5 * jnp.log(chi5 / chi4) / (chi5 - chi4)
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (1 / 60) * (5 * b + 3 * c - (2 * b + c) * z))
    def false_branch(_):
        formula = (a * ( - 12 * b * p * z + 2 * a ** 2 * b * ( - 6 + 3 * p + 3 * z - 2 * p * z) + 6 * a * b * (2 * p + 2 * z - p * z) + a ** 3 * c * (6 - 4 * p - 4 * z + 3 * p * z)) - 12 * b * (a - p) * (a - z) * jnp.log(1 - a)) / (12 * a ** 3)
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Element 4
@jax.jit
def element4(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5 - chi3) / (2 * chi2)
    c = chi3 * jnp.log(chi4 / chi3) / (chi4 - chi3) - chi5 * jnp.log(chi5 / chi4) / (chi5 - chi4)
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (1 / 60) * ( - 2 * c * ( - 6 + z) - 3 * b * ( - 5 + z)))
    def false_branch(_):
        formula = (a * (a ** 3 * c * (6 + p * ( - 2 + z) - 2 * z) + 12 * b * p * z + 2 * a ** 2 * b * (6 + 3 * p + 3 * z - p * z) - 6 * a * b * (2 * p + (2 + p) * z)) -
12 * ( - 1 + a) * b * (a - p) * (a - z) * jnp.log(1 - a)) / (12 * a ** 3)
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Element 5
@jax.jit
def element5(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (7 * b * (7 + 5 * b) - 2 * b * (9 + 7 * b) * z + 14 * (1 + b) * ( - 3 + z) * jnp.log(1 + b)) / (840 * b))
    def false_branch(_):
        formula = - ((1 / (720 * a ** 4 * b)) * (a * b * (10 * a * (6 * a * ( - 12 + 5 * a ** 2 + 6 * a * (2 + b)) + 60 * p - a * (66 + 28 * a + 17 * a ** 2 + 18 * (2 + a) * b) * p) +
( - 10 * a * ( - 60 + a * (66 + 28 * a + 17 * a ** 2 + 18 * (2 + a) * b)) + 3 * ( - 180 + a * (30 * (7 + 4 * b) + a * (90 + 60 * b + a * (55 + 39 * a + 40 * b)))) * p) * z) +
60 * (b * (6 * a ** 3 * (3 + b) + 10 * a * p - 9 * p * z + a * (10 + 15 * p + 6 * b * p) * z - 2 * a ** 2 * (6 + 8 * p + 3 * b * p + 8 * z + 3 * b * z) + a ** 5 * ( - 6 + 4 * p + 4 * z - 3 * p * z) +
a ** 4 * (6 - 4 * p - 4 * z + 3 * p * z)) * jnp.log(1 - a) + a ** 5 * (1 + b) * (6 - 4 * z + p * ( - 4 + 3 * z)) * jnp.log(1 + b))))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Element 6
@jax.jit
def element6(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (1 / 840) * (175 - 21 * b * ( - 5 + z) - 31 * z + (28 * (1 + b) * ( - 6 + z) * jnp.log(1 + b)) / b))
    def false_branch(_):
        formula = (1 / (720 * a ** 4)) * (60 * ( - 1 + a) * ( - 6 * a ** 3 * (4 + b) + a ** 4 * (6 + p * ( - 2 + z) - 2 * z) + 9 * p * z + 6 * a ** 2 * (2 + (3 + b) * p + (3 + b) * z) -
2 * a * (5 * z + p * (5 + 8 * z + 3 * b * z))) * jnp.log(1 - a) + a * ( - 540 * p * z + a ** 4 * ( - 600 + p * (130 - 53 * z) + 130 * z) + 30 * a * (20 * z + p * (20 + 41 * z + 12 * b * z)) +
5 * a ** 3 * (360 + 88 * p + 88 * z - 23 * p * z + 12 * b * (6 + 3 * p + 3 * z - p * z)) - 30 * a ** 2 * (24 + 46 * z + 12 * b * z + p * (46 + 13 * z + 6 * b * (2 + z))) -
(60 * a ** 4 * (1 + b) * (6 + p * ( - 2 + z) - 2 * z) * jnp.log(1 + b)) / b))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Element 7
@jax.jit
def element7(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi4 - chi3) / (2 * chi2)
    c = chi3 * jnp.log(chi4 / chi3) / (chi4 - chi3) - 1
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (1 / 60) * (5 * b + 3 * c - (2 * b + c) * z))
    def false_branch(_):
        formula = (a * ( - 12 * b * p * z + 2 * a ** 2 * b * ( - 6 + 3 * p + 3 * z - 2 * p * z) + 6 * a * b * (2 * p + 2 * z - p * z) + a ** 3 * c * (6 - 4 * p - 4 * z + 3 * p * z)) - 12 * b * (a - p) * (a - z) * jnp.log(1 - a)) / (12 * a ** 3)
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Element 8
@jax.jit
def element8(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi4 - chi3) / (2 * chi2)
    c = chi3 * jnp.log(chi4 / chi3) / (chi4 - chi3) - 1
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (1 / 60) * ( - 2 * c * ( - 6 + z) - 3 * b * ( - 5 + z)))
    def false_branch(_):
        formula = (a * (a ** 3 * c * (6 + p * ( - 2 + z) - 2 * z) + 12 * b * p * z + 2 * a ** 2 * b * (6 + 3 * p + 3 * z - p * z) - 6 * a * b * (2 * p + (2 + p) * z)) -
12 * ( - 1 + a) * b * (a - p) * (a - z) * jnp.log(1 - a)) / (12 * a ** 3)
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Coefficient
@jax.jit
def coefficient(chi_grid, power_grid, redshift_grid):
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    coefficients = jnp.zeros((grid_size + 1, grid_size + 1, ell_size + 1))
    
    k_indices = jnp.arange(grid_size + 1, dtype=jnp.int32)
    def accumulate_step(n, coefficients):
        valid = (n + 1 < grid_size)
        
        value1 = element1(chi_grid[n], chi_grid[n + 1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        coefficients = coefficients.at[n, n, :].add(value1)
        
        value2 = element2(chi_grid[n], chi_grid[n + 1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        coefficients = coefficients.at[n, n + 1, :].add(jnp.where(valid, value2, jnp.zeros_like(value2)))
        
        def compute_element3(j):
            return element3(chi_grid[n], chi_grid[n + 1], chi_grid[j - 1], chi_grid[j], chi_grid[j + 1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        value3_all = vmap(compute_element3)(k_indices)
        value3_mask = (k_indices >= n + 2) & (k_indices < grid_size) & valid
        coefficients = coefficients.at[n, :, :].add(jnp.where(value3_mask[:, None], value3_all, 0.0))
        
        def compute_element4(j):
            return element4(chi_grid[n], chi_grid[n + 1], chi_grid[j - 1], chi_grid[j], chi_grid[j + 1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        value4_all = vmap(compute_element4)(k_indices)
        value4_mask = (k_indices >= n + 2) & (k_indices < grid_size) & valid
        coefficients = coefficients.at[n + 1, :, :].add(jnp.where(value4_mask[:, None], value4_all, 0.0))
        
        value5 = element5(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        coefficients = coefficients.at[n + 1, n, :].add(jnp.where(valid, value5, jnp.zeros_like(value5)))
        
        value6 = element6(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        coefficients = coefficients.at[n + 1, n + 1, :].add(jnp.where(valid, value6, jnp.zeros_like(value6)))
        
        value7 = element7(chi_grid[n], chi_grid[n + 1], chi_grid[-2], chi_grid[-1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        coefficients = coefficients.at[grid_size, n, :].add(value7)
        
        value8 = element8(chi_grid[n], chi_grid[n + 1], chi_grid[-2], chi_grid[-1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        coefficients = coefficients.at[grid_size, n + 1, :].add(jnp.where(valid, value8, jnp.zeros_like(value8)))
        
        return coefficients
    
    return lax.fori_loop(0, grid_size, accumulate_step, coefficients)

# Spectra
@jax.jit
def spectra(factor, phi_a_grid, phi_b_grid, chi_grid, power_grid, redshift_grid):
    coefficients = coefficient(chi_grid, power_grid, redshift_grid)
    return factor * jnp.einsum('mi,nj,ijl->mnl', phi_a_grid, phi_b_grid, coefficients)
