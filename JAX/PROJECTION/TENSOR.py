import jax
import jax.numpy as jnp

# Spectra
@jax.jit
def spectra(factor, phi_a_grid, phi_b_grid, coefficients):
    return factor * jnp.einsum('mi,nj,ijl->mnl', phi_a_grid, phi_b_grid, coefficients)