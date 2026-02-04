import jax
from jax import lax
from jax import vmap
from jax import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)


@jax.jit
def element1(chi1, chi2, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (63 - 41 * z) / 25200)
    #else:
    def false_branch(_):
        formula =(1 / (720 * a**4)) * (a * (10 * a * (12 * a * ( - 6 + ( - 3 + a) * a) + (60 + a * (30 + (20 - 9 * a) * a)) * y) + 
(10 * a * (60 + a * (30 + (20 - 9 * a) * a)) + 9 * ( - 60 + a * ( - 30 + a * ( - 20 + a * ( - 15 + 8 * a)))) * y) * z) + 
60 * ( - 12 * a**2 - 9 * y * z + 10 * a * (y + z) + a**4 * (6 - 4 * y - 4 * z + 3 * y * z)) * jnp.log(1 - a))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2* power2 * (1 + redshift2) * formula
    return coefficient


@jax.jit
def element2(chi1, chi2, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (21 - 11 * z) / 12600)
    #else:
    def false_branch(_):
        formula = (1 / (720 * a**4)) * (a * (540 * y * z + 5 * a**3 * ( - 144 + 32 * y + 32 * z - 13 * y * z) + 6 * a**4 * (10 - 5 * y - 5 * z + 3 * y * z) - 60 * a**2 * ( - 12 - 7 * z + y * ( - 7 + 2 * z)) - 30 * a * (20 * z + y * (20 + 11 * z))) + 60 * ( - 18 * a**3 + a**4 * (6 + y * ( - 2 + z) - 2 * z) + 9 * y * z + 12 * a**2 * (1 + y + z) - 10 * a * (y + z + y * z)) * jnp.log(1 - a))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2* power2 * (1 + redshift2) * formula
    return coefficient


@jax.jit
def element3(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5-chi3)/(2*chi2)
    c = chi3*jnp.log(chi4/chi3)/(chi4-chi3)-chi5*jnp.log(chi5/chi4)/(chi5-chi4)
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (1 / 60) * (5 * b + 3 * c - (2 * b + c) * z))
    #else:
    def false_branch(_):
        formula = (a * ( - 12 * b * y * z + 2 * a**2 * b * ( - 6 + 3 * y + 3 * z - 2 * y * z) + 6 * a * b * (2 * y + 2 * z - y * z) + a**3 * c * (6 - 4 * y - 4 * z + 3 * y * z)) - 12 * b * (a - y) * (a - z) * jnp.log(1 - a)) / (12 * a**3)
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2* power2 * (1 + redshift2) * formula
    return coefficient


@jax.jit
def element4(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5-chi3)/(2*chi2)
    c = chi3*jnp.log(chi4/chi3)/(chi4-chi3)-chi5*jnp.log(chi5/chi4)/(chi5-chi4)
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (1 / 60) * ( - 2 * c * ( - 6 + z) - 3 * b * ( - 5 + z)))
    #else:
    def false_branch(_):
        formula = (a * (a**3 * c * (6 + y * ( - 2 + z) - 2 * z) + 12 * b * y * z + 2 * a**2 * b * (6 + 3 * y + 3 * z - y * z) - 6 * a * b * (2 * y + (2 + y) * z)) - 12 * ( - 1 + a) * b * (a - y) * (a - z) * jnp.log(1 - a)) / (12 * a**3)
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2* power2 * (1 + redshift2) * formula
    return coefficient


@jax.jit 
def element5(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (7 * b * (7 + 5 * b) - 2 * b * (9 + 7 * b) * z + 14 * (1 + b) * ( - 3 + z) * jnp.log(1 + b)) / (840 * b))
    #else:
    def false_branch(_):
        formula =  - ((1 / (720 * a**4 * b)) * (a * b * (10 * a * (6 * a * ( - 12 + 5 * a**2 + 6 * a * (2 + b)) + 60 * y - a * (66 + 28 * a + 17 * a**2 + 18 * (2 + a) * b) * y) + 
( - 10 * a * ( - 60 + a * (66 + 28 * a + 17 * a**2 + 18 * (2 + a) * b)) + 3 * ( - 180 + a * (30 * (7 + 4 * b) + a * (90 + 60 * b + a * (55 + 39 * a + 40 * b)))) * y) * z) + 
60 * (b * (6 * a**3 * (3 + b) + 10 * a * y - 9 * y * z + a * (10 + 15 * y + 6 * b * y) * z - 2 * a**2 * (6 + 8 * y + 3 * b * y + 8 * z + 3 * b * z) + a**5 * ( - 6 + 4 * y + 4 * z - 3 * y * z) + 
a**4 * (6 - 4 * y - 4 * z + 3 * y * z)) * jnp.log(1 - a) + a**5 * (1 + b) * (6 - 4 * z + y * ( - 4 + 3 * z)) * jnp.log(1 + b))))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2* power2 * (1 + redshift2) * formula
    return coefficient


@jax.jit
def element6(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (1 / 840) * (175 - 21 * b * ( - 5 + z) - 31 * z + (28 * (1 + b) * ( - 6 + z) * jnp.log(1 + b)) / b))
    #else:
    def false_branch(_):
        formula = (1 / (720 * a**4)) * (60 * ( - 1 + a) * ( - 6 * a**3 * (4 + b) + a**4 * (6 + y * ( - 2 + z) - 2 * z) + 9 * y * z + 6 * a**2 * (2 + (3 + b) * y + (3 + b) * z) - 2 * a * (5 * z + y * (5 + 8 * z + 3 * b * z))) * jnp.log(1 - a) + a * ( - 540 * y * z + a**4 * ( - 600 + y * (130 - 53 * z) + 130 * z) + 30 * a * (20 * z + y * (20 + 41 * z + 12 * b * z)) + 5 * a**3 * (360 + 88 * y + 88 * z - 23 * y * z + 12 * b * (6 + 3 * y + 3 * z - y * z)) - 30 * a**2 * (24 + 46 * z + 12 * b * z + y * (46 + 13 * z + 6 * b * (2 + z))) - (60 * a**4 * (1 + b) * (6 + y * ( - 2 + z) - 2 * z) * jnp.log(1 + b)) / b))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2* power2 * (1 + redshift2) * formula
    return coefficient


@jax.jit
def element7(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi4-chi3)/(2*chi2)
    c = chi3*jnp.log(chi4/chi3)/(chi4-chi3)-1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (1 / 60) * (5 * b + 3 * c - (2 * b + c) * z))
    #else:
    def false_branch(_): 
        formula =   (a * ( - 12 * b * y * z + 2 * a**2 * b * ( - 6 + 3 * y + 3 * z - 2 * y * z) + 6 * a * b * (2 * y + 2 * z - y * z) + a**3 * c * (6 - 4 * y - 4 * z + 3 * y * z)) - 12 * b * (a - y) * (a - z) * jnp.log(1 - a)) / (12 * a**3)
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2* power2 * (1 + redshift2) * formula
    return coefficient


@jax.jit
def element8(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi4-chi3)/(2*chi2)
    c = chi3*jnp.log(chi4/chi3)/(chi4-chi3)-1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (1 / 60) * ( - 2 * c * ( - 6 + z) - 3 * b * ( - 5 + z)))
        
    #else:
    def false_branch(_):
        formula = (a * (a**3 * c * (6 + y * ( - 2 + z) - 2 * z) + 12 * b * y * z + 2 * a**2 * b * (6 + 3 * y + 3 * z - y * z) - 6 * a * b * (2 * y + (2 + y) * z)) - 12 * ( - 1 + a) * b * (a - y) * (a - z) * jnp.log(1 - a)) / (12 * a**3)
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2* power2 * (1 + redshift2) * formula
    return coefficient


@jax.jit
def element(n, i, j, chi_grid, power_grid, redshift_grid):
    grid_size = chi_grid.shape[0] - 1
    zeros_vec = jnp.zeros_like(power_grid[:, 0])
    operand = (n, i, j, chi_grid, power_grid, redshift_grid)
    
    cond1 = ((i < n) & (n < grid_size)) | ((j < n) & (n < grid_size))
    cond2 = (n == j) & (n == i) & (n < grid_size)
    cond3 = (n == j) & (n + 1 == i) & (n + 1 < grid_size)
    cond4 = (n + 1 < j) & (j < grid_size) & (n == i) & (n < grid_size)
    cond5 = (n + 1 < j) & (j < grid_size) & (n + 1 == i) & (n + 1 < grid_size)
    cond6 = (n + 1 == j) & (n == i) & (n + 2 < chi_grid.shape[0])  
    cond7 = (n + 1 == j) & (n + 1 == i) & (n + 2 < chi_grid.shape[0])
    cond8 = (j == grid_size) & (n == i) & (grid_size >= 1) & (n < grid_size)
    cond9 = (j == grid_size) & (n + 1 == i) & (grid_size >= 1) & (n + 1 < grid_size)
    
    def zeros_fn(_): return zeros_vec
    
    def e1_fn(op):  
        n_, i_, j_, chi, pw, z = op
        return element1(chi[n_], chi[n_ + 1], pw[:, n_], pw[:, n_ + 1], z[n_], z[n_ + 1])
    
    def e2_fn(op):  
        n_, i_, j_, chi, pw, z = op
        return element2(chi[n_], chi[n_ + 1], pw[:, n_], pw[:, n_ + 1], z[n_], z[n_ + 1])
    
    def e3_fn(op):  
        n_, i_, j_, chi, pw, z = op
        return element3(chi[n_], chi[n_ + 1], chi[j_-1], chi[j_], chi[j_+1], pw[:, n_], pw[:, n_+1], z[n_], z[n_+1])
    
    def e4_fn(op):  
        n_, i_, j_, chi, pw, z = op
        return element4(chi[n_], chi[n_ + 1], chi[j_-1], chi[j_], chi[j_+1], pw[:, n_], pw[:, n_+1], z[n_], z[n_+1])
    
    def e5_fn(op):  
        n_, i_, j_, chi, pw, z = op
        return element5(chi[n_], chi[n_ + 1], chi[n_ + 2], pw[:, n_], pw[:, n_+1], z[n_], z[n_+1])
    
    def e6_fn(op):  
        n_, i_, j_, chi, pw, z = op
        return element6(chi[n_], chi[n_ + 1], chi[n_ + 2], pw[:, n_], pw[:, n_+1], z[n_], z[n_+1])
    
    def e7_fn(op):  
        n_, i_, j_, chi, pw, z = op
        return element7(chi[n_], chi[n_ + 1], chi[-2], chi[-1], pw[:, n_], pw[:, n_+1], z[n_], z[n_+1])
    
    def e8_fn(op):  
        n_, i_, j_, chi, pw, z = op
        return element8(chi[n_], chi[n_ + 1], chi[-2], chi[-1], pw[:, n_], pw[:, n_+1], z[n_], z[n_+1])
    
    out = lax.cond(
        cond1, zeros_fn,
        lambda op: lax.cond(cond2, e1_fn,
        lambda op: lax.cond(cond3, e2_fn,
        lambda op: lax.cond(cond4, e3_fn,
        lambda op: lax.cond(cond5, e4_fn,
        lambda op: lax.cond(cond6, e5_fn,
        lambda op: lax.cond(cond7, e6_fn,
        lambda op: lax.cond(cond8, e7_fn,
        lambda op: lax.cond(cond9, e8_fn,
        zeros_fn, op), op), op), op), op), op), op), op), operand
    )
    
    return out


@jax.jit
def coefficient(chi_grid, power_grid, redshift_grid):
    grid_size = chi_grid.shape[0] - 1
    ell_size  = power_grid.shape[0] - 1
    
    coeff0 = jnp.zeros((grid_size + 1, grid_size + 1, ell_size + 1), dtype=power_grid.dtype)
    ij = jnp.arange(grid_size + 1, dtype=jnp.int32)
    index_i, index_j = jnp.meshgrid(ij, ij, indexing='ij')   
    
    # for n in range(grid_size):
    def all_e_for_n(n):
        # for i in range(n, grid_size + 1):
        def e_for_i(i):
            # for j in range(n, grid_size + 1):
            def e_for_j(j):
                return element(n, i, j, chi_grid, power_grid, redshift_grid)  
            return vmap(e_for_j, in_axes=(0,))(ij)  
        e_ij = vmap(e_for_i, in_axes=(0,))(ij)      
        mask = (index_i >= n) & (index_j >= n)                       
        return jnp.where(mask[..., None], e_ij, 0)
    
    def body_n(n, coeff):
        return coeff + all_e_for_n(n)
    
    coefficients = lax.fori_loop(0, grid_size, body_n, coeff0)
    return coefficients


@jax.jit
def function(amplitude, phi_grid1, phi_grid2, chi_grid, power_grid, redshift_grid):
    
    amplitude    = jnp.asarray(amplitude)
    phi_grid1    = jnp.asarray(phi_grid1)      
    phi_grid2    = jnp.asarray(phi_grid2)      
    chi_grid     = jnp.asarray(chi_grid)
    power_grid   = jnp.asarray(power_grid)
    redshift_grid= jnp.asarray(redshift_grid)
    
    coeff = coefficient(chi_grid, power_grid, redshift_grid)   
    #for m1,m2,i,j in range()
    functions = amplitude * jnp.einsum('mi,nj,ijl->mnl', phi_grid1, phi_grid2, coeff)
    return functions