import numba
import numpy

# Element 1
@numba.njit(cache=True)
def element1(chi1, chi2, power1, power2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / power2
    
    if a == 1:
        formula = numpy.full_like(p, 1 / 12)
    else: 
        formula = ((a * (2 * ( - 2 + a) * a + 6 * p - a * (3 + a) * p)) / (2 * ( - 1 + a)) + (2 * a - 3 * p) * numpy.log(1 - a)) / a ** 3
    
    element = power2 * formula / chi2
    return element

# Element 2
@numba.njit(cache=True)
def element2(chi1, chi2, power1, power2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / power2
    
    if a == 1:
        formula = numpy.full_like(p, 1 / 12)
    else: 
        formula = (( - a) * ( - 6 * p + a * (4 + p)) + 2 * (a ** 2 + 3 * p - 2 * a * (1 + p)) * numpy.log(1 - a)) / (2 * a ** 3)
    
    element = power2 * formula / chi2
    return element

# Element 3
@numba.njit(cache=True)
def element3(chi1, chi2, power1, power2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / power2
    
    if a == 1:
        formula = numpy.full_like(p, 1 / 4)
    else: 
        formula = ((1 / 2) * a * ( - 6 * p + a * (4 - 2 * a + 5 * p)) - ( - 1 + a) * ( - 3 * p + a * (2 + p)) * numpy.log(1 - a)) / a ** 3
    
    element = power2 * formula / chi2
    return element

# Coefficient
@numba.njit(cache=True)
def coefficient(chi_grid, power_grid):
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    coefficients = numpy.zeros((grid_size + 1, grid_size + 1, ell_size + 1), dtype=numpy.float64)
    
    # Loop
    for n in range(grid_size):
        # Element 1
        if n < grid_size:
            element = element1(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1])  
            coefficients[n, n, :] += element
        # Element 2
        if n < grid_size:
            element = element2(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1])
            coefficients[n, n + 1, :] += element
            coefficients[n + 1, n, :] += element
        # Element 3
        if n + 1 < grid_size:
            element = element3(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1])
            coefficients[n + 1, n + 1, :] += element
    return coefficients

# Spectra
def spectra(factor, amplitude, phi_a_grid, phi_b_grid, chi_grid, power_grid):
    coefficients = coefficient(chi_grid, power_grid)
    return factor * amplitude * numpy.einsum('ijk,ai,bj->abk', coefficients, phi_a_grid, phi_b_grid, dtype=numpy.float64)