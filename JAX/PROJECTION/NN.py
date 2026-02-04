import jax
from jax import lax
from jax import vmap
from jax import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)


@jax.jit
def element1(chi1, chi2, power1, power2):
    a = 1.0 - chi1 / chi2          
    y = 1.0 - power1 / power2      
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, 1 / 12)
    #else:
    def false_branch(_):
        formula = ((a * (2 * ( - 2 + a) * a + 6 * y - a * (3 + a) * y)) / (2 * ( - 1 + a)) + (2 * a - 3 * y) * jnp.log(1 - a)) / a**3
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = power2 * formula / chi2   
    return coefficient


@jax.jit
def element2(chi1, chi2, power1, power2):
    a = 1.0 - chi1 / chi2          
    y = 1.0 - power1 / power2      
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, 1 / 12)
    #else:
    def false_branch(_):
        formula = (( - a) * ( - 6 * y + a * (4 + y)) + 2 * (a**2 + 3 * y - 2 * a * (1 + y)) * jnp.log(1 - a)) / (2 * a**3)
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = power2 * formula / chi2   
    return coefficient


@jax.jit
def element3(chi1, chi2, power1, power2):
    a = 1.0 - chi1 / chi2          
    y = 1.0 - power1 / power2      
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, 1 / 4)
    #else:
    def false_branch(_):
        formula =  ((1 / 2) * a * ( - 6 * y + a * (4 - 2 * a + 5 * y)) - ( - 1 + a) * ( - 3 * y + a * (2 + y)) * jnp.log(1 - a)) / a**3
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = power2 * formula / chi2  
    return coefficient


@jax.jit
def element(n, i, j, chi_grid, power_grid, redshift_grid):
    
    grid_size = chi_grid.shape[0] - 1
    zeros_vec = jnp.zeros_like(power_grid[:, 0])  
    
    operand = (n, i, j, chi_grid, power_grid)
    
    cond1 = ((i < n) & (n < grid_size)) | ((j < n) & (n < grid_size))
    cond2 = (n == i) & (n == j) & (n < grid_size)
    cond3 = ( ((n == i) & (n + 1 == j) & (n + 1 < grid_size)) | ((n + 1 == i) & (n == j) & (n + 1 < grid_size)) )
    cond4 = (n + 1 == i) & (n + 1 == j) & (n + 1 < grid_size)
    
    def zeros_fn(_):
        return zeros_vec
    
    def e1_fn(op):
        n_, i_, j_, chi, pw = op
        return element1(chi[n_], chi[n_ + 1], pw[:, n_], pw[:, n_ + 1])
    
    def e2_fn(op):
        n_, i_, j_, chi, pw = op
        return element2(chi[n_], chi[n_ + 1], pw[:, n_], pw[:, n_ + 1])
    
    def e3_fn(op):
        n_, i_, j_, chi, pw = op
        return element3(chi[n_], chi[n_ + 1], pw[:, n_], pw[:, n_ + 1])
    
    out = lax.cond(
        cond1, zeros_fn,                                    
        lambda op: lax.cond(
            cond2, e1_fn,                                   
            lambda op2: lax.cond(
                cond3, e2_fn,                               
                lambda op3: lax.cond(
                    cond4, e3_fn,                           
                    zeros_fn,                                
                    op3
                ),
                op2
            ),
            op
        ),
        operand
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