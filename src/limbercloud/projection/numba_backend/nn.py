import numba
import numpy

from limbercloud.projection.numba_backend.local import _series_terms


# Element 1
@numba.njit(cache=True)
def _element1_nonzero(chi1, chi2, power1, power2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, 1 / 12)
    else:
        formula = ((a * (2 * ( - 2 + a) * a + 6 * p - a * (3 + a) * p)) / (2 * (-chi1 / chi2)) + (2 * a - 3 * p) * numpy.log(chi1 / chi2)) / a ** 3

    element = power2 * formula / chi2
    return element

# Element 2
@numba.njit(cache=True)
def _element2_nonzero(chi1, chi2, power1, power2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, 1 / 12)
    else:
        formula = (( - a) * ( - 6 * p + a * (4 + p)) + 2 * (a ** 2 + 3 * p - 2 * a * (1 + p)) * numpy.log(chi1 / chi2)) / (2 * a ** 3)

    element = power2 * formula / chi2
    return element

# Element 3
@numba.njit(cache=True)
def _element3_nonzero(chi1, chi2, power1, power2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, 1 / 4)
    else:
        formula = ((1 / 2) * a * ( - 6 * p + a * (4 - 2 * a + 5 * p)) - (-chi1 / chi2) * ( - 3 * p + a * (2 + p)) * numpy.log(chi1 / chi2)) / a ** 3

    element = power2 * formula / chi2
    return element

@numba.njit(cache=True)
def _nn_series(chi1, chi2, power1, power2, which):
    a = (chi2-chi1)/chi2
    total = numpy.zeros_like(power1)
    for m in range(_series_terms(a)):
        m0 = power2/(m+1)+(power1-power2)/(m+2)
        m1 = power2/(m+2)+(power1-power2)/(m+3)
        m2 = power2/(m+3)+(power1-power2)/(m+4)
        value = m2 if which == 1 else (m1-m2 if which == 2 else m0-2*m1+m2)
        total += (m+1)*a**m*value
    return a/chi2*total

@numba.njit(cache=True)
def element1(chi1, chi2, power1, power2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    a = (chi2-chi1)/chi2
    if chi1 > 0 and a <= 0.75:
        return _nn_series(chi1, chi2, power1, power2, 1)
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element1_nonzero(chi1, chi2, power1, safe_power2)
    if numpy.any(zero):
        left = _element1_nonzero(chi1, chi2, numpy.ones_like(power1), numpy.ones_like(power1)) - _element1_nonzero(chi1, chi2, numpy.zeros_like(power1), numpy.ones_like(power2))
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element2(chi1, chi2, power1, power2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    a = (chi2-chi1)/chi2
    if chi1 > 0 and a <= 0.75:
        return _nn_series(chi1, chi2, power1, power2, 2)
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element2_nonzero(chi1, chi2, power1, safe_power2)
    if numpy.any(zero):
        left = _element2_nonzero(chi1, chi2, numpy.ones_like(power1), numpy.ones_like(power1)) - _element2_nonzero(chi1, chi2, numpy.zeros_like(power1), numpy.ones_like(power2))
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element3(chi1, chi2, power1, power2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    a = (chi2-chi1)/chi2
    if chi1 > 0 and a <= 0.75:
        return _nn_series(chi1, chi2, power1, power2, 3)
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element3_nonzero(chi1, chi2, power1, safe_power2)
    if numpy.any(zero):
        left = _element3_nonzero(chi1, chi2, numpy.ones_like(power1), numpy.ones_like(power1)) - _element3_nonzero(chi1, chi2, numpy.zeros_like(power1), numpy.ones_like(power2))
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

# Coefficient
# There are N intervals and N+1 nodes. Every interval stores all four hat
# placements, including the rising-hat diagonal of the final interval. The
# density vanishes outside the finite domain, not at its last node, so the
# node-N basis function is supported across [chi[N-1], chi[N]].
@numba.njit(cache=True)
def coefficient(chi_grid, power_grid):
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    coefficients = numpy.zeros((grid_size + 1, grid_size + 1, ell_size + 1), dtype=numpy.float64)

    # Loop
    for n in range(grid_size):
        # Element 1
        element = element1(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1])
        coefficients[n, n, :] += element
        # Element 2
        element = element2(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1])
        coefficients[n, n + 1, :] += element
        coefficients[n + 1, n, :] += element
        # Element 3
        element = element3(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1])
        coefficients[n + 1, n + 1, :] += element
    return coefficients

# Spectra
def spectra(factor, phi_a_grid, phi_b_grid, chi_grid, power_grid):
    coefficients = coefficient(chi_grid, power_grid)
    return factor * numpy.einsum('mi,nj,ijl->mnl', phi_a_grid, phi_b_grid, coefficients, dtype=numpy.float64)
