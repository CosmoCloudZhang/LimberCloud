import numpy

# Spectra
def spectra(factor, phi_a_grid, phi_b_grid, coefficients):
    return factor * numpy.einsum('mi,nj,ijl->mnl', phi_a_grid, phi_b_grid, coefficients, dtype=numpy.float64)