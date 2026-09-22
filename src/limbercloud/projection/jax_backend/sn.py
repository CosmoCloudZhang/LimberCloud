"""SN coefficients are the transpose of the corresponding NS coefficients."""

import jax
import jax.numpy as jnp

from limbercloud.projection.jax_backend import ns

element1 = ns.element1
element2 = ns.element2
element3 = ns.element3
element4 = ns.element4
element5 = ns.element5
element6 = ns.element6
element7 = ns.element7
element8 = ns.element8
element9 = ns.element9
element10 = ns.element10



@jax.jit
def coefficient(chi_grid, power_grid, redshift_grid):
    """Exchange the density/source node axes without duplicating case dispatch."""
    return jnp.transpose(ns.coefficient(chi_grid, power_grid, redshift_grid), (1, 0, 2))


@jax.jit
def spectra(factor, phi_a_grid, phi_b_grid, chi_grid, power_grid, redshift_grid):
    coefficients = coefficient(chi_grid, power_grid, redshift_grid)
    return factor * jnp.einsum('mi,nj,ijl->mnl', phi_a_grid, phi_b_grid, coefficients)
