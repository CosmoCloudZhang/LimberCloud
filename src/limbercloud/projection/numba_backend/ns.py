import numba
import numpy

from limbercloud.projection.numba_backend import local


# Element 1
@numba.njit(cache=True)
def element1(chi1, chi2, power1, power2, redshift1, redshift2):
    return local.ns1(chi1, chi2, power1, power2, redshift1, redshift2)

# Element 2
@numba.njit(cache=True)
def element2(chi1, chi2, power1, power2, redshift1, redshift2):
    return local.ns2(chi1, chi2, power1, power2, redshift1, redshift2)

# Element 3
@numba.njit(cache=True)
def _element3_nonzero(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5 - chi3) / (2 * chi2)
    c = chi3 * numpy.log(chi4 / chi3) / (chi4 - chi3) - chi5 * numpy.log(chi5 / chi4) / (chi5 - chi4)
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (1 / 60) * (5 * b + 3 * c - (2 * b + c) * z))
    else:
        formula = (a * ( - 12 * b * p * z + 2 * a ** 2 * b * ( - 6 + 3 * p + 3 * z - 2 * p * z) + 6 * a * b * (2 * p + 2 * z - p * z) + a ** 3 * c * (6 - 4 * p - 4 * z + 3 * p * z)) - 12 * b * (a - p) * (a - z) * numpy.log(chi1 / chi2)) / (12 * a ** 3)

    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Element 4
@numba.njit(cache=True)
def _element4_nonzero(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5 - chi3) / (2 * chi2)
    c = chi3 * numpy.log(chi4 / chi3) / (chi4 - chi3) - chi5 * numpy.log(chi5 / chi4) / (chi5 - chi4)
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (1 / 60) * ( - 2 * c * ( - 6 + z) - 3 * b * ( - 5 + z)))
    else:
        formula = (a * (a ** 3 * c * (6 + p * ( - 2 + z) - 2 * z) + 12 * b * p * z + 2 * a ** 2 * b * (6 + 3 * p + 3 * z - p * z) - 6 * a * b * (2 * p + (2 + p) * z)) -
12 * (-chi1 / chi2) * b * (a - p) * (a - z) * numpy.log(chi1 / chi2)) / (12 * a ** 3)

    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Element 5
@numba.njit(cache=True)
def _element5_nonzero(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (7 * b * (7 + 5 * b) - 2 * b * (9 + 7 * b) * z + 14 * (1 + b) * ( - 3 + z) * numpy.log(1 + b)) / (840 * b))
    else:
        formula = - ((1 / (720 * a ** 4 * b)) * (a * b * (10 * a * (6 * a * ( - 12 + 5 * a ** 2 + 6 * a * (2 + b)) + 60 * p - a * (66 + 28 * a + 17 * a ** 2 + 18 * (2 + a) * b) * p) +
( - 10 * a * ( - 60 + a * (66 + 28 * a + 17 * a ** 2 + 18 * (2 + a) * b)) + 3 * ( - 180 + a * (30 * (7 + 4 * b) + a * (90 + 60 * b + a * (55 + 39 * a + 40 * b)))) * p) * z) +
60 * (b * (6 * a ** 3 * (3 + b) + 10 * a * p - 9 * p * z + a * (10 + 15 * p + 6 * b * p) * z - 2 * a ** 2 * (6 + 8 * p + 3 * b * p + 8 * z + 3 * b * z) + a ** 5 * ( - 6 + 4 * p + 4 * z - 3 * p * z) +
a ** 4 * (6 - 4 * p - 4 * z + 3 * p * z)) * numpy.log(chi1 / chi2) + a ** 5 * (1 + b) * (6 - 4 * z + p * ( - 4 + 3 * z)) * numpy.log(1 + b))))

    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Element 6
@numba.njit(cache=True)
def _element6_nonzero(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (1 / 840) * (175 - 21 * b * ( - 5 + z) - 31 * z + (28 * (1 + b) * ( - 6 + z) * numpy.log(1 + b)) / b))
    else:
        formula = (1 / (720 * a ** 4)) * (60 * (-chi1 / chi2) * ( - 6 * a ** 3 * (4 + b) + a ** 4 * (6 + p * ( - 2 + z) - 2 * z) + 9 * p * z + 6 * a ** 2 * (2 + (3 + b) * p + (3 + b) * z) -
2 * a * (5 * z + p * (5 + 8 * z + 3 * b * z))) * numpy.log(chi1 / chi2) + a * ( - 540 * p * z + a ** 4 * ( - 600 + p * (130 - 53 * z) + 130 * z) + 30 * a * (20 * z + p * (20 + 41 * z + 12 * b * z)) +
5 * a ** 3 * (360 + 88 * p + 88 * z - 23 * p * z + 12 * b * (6 + 3 * p + 3 * z - p * z)) - 30 * a ** 2 * (24 + 46 * z + 12 * b * z + p * (46 + 13 * z + 6 * b * (2 + z))) -
(60 * a ** 4 * (1 + b) * (6 + p * ( - 2 + z) - 2 * z) * numpy.log(1 + b)) / b))

    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Element 7
@numba.njit(cache=True)
def _element7_nonzero(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi4 - chi3) / (2 * chi2)
    c = chi3 * numpy.log(chi4 / chi3) / (chi4 - chi3) - 1
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (1 / 60) * (5 * b + 3 * c - (2 * b + c) * z))
    else:
        formula = (a * ( - 12 * b * p * z + 2 * a ** 2 * b * ( - 6 + 3 * p + 3 * z - 2 * p * z) + 6 * a * b * (2 * p + 2 * z - p * z) + a ** 3 * c * (6 - 4 * p - 4 * z + 3 * p * z)) - 12 * b * (a - p) * (a - z) * numpy.log(chi1 / chi2)) / (12 * a ** 3)

    element = chi2 * power2 * (1 + redshift2) * formula
    return element

# Element 8
@numba.njit(cache=True)
def _element8_nonzero(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi4 - chi3) / (2 * chi2)
    c = chi3 * numpy.log(chi4 / chi3) / (chi4 - chi3) - 1
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (1 / 60) * ( - 2 * c * ( - 6 + z) - 3 * b * ( - 5 + z)))
    else:
        formula = (a * (a ** 3 * c * (6 + p * ( - 2 + z) - 2 * z) + 12 * b * p * z + 2 * a ** 2 * b * (6 + 3 * p + 3 * z - p * z) - 6 * a * b * (2 * p + (2 + p) * z)) -
12 * (-chi1 / chi2) * b * (a - p) * (a - z) * numpy.log(chi1 / chi2)) / (12 * a ** 3)

    element = chi2 * power2 * (1 + redshift2) * formula
    return element

@numba.njit(cache=True)
def element3(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element3_nonzero(chi1, chi2, chi3, chi4, chi5, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element3_nonzero(chi1, chi2, chi3, chi4, chi5, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element3_nonzero(chi1, chi2, chi3, chi4, chi5, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element4(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element4_nonzero(chi1, chi2, chi3, chi4, chi5, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element4_nonzero(chi1, chi2, chi3, chi4, chi5, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element4_nonzero(chi1, chi2, chi3, chi4, chi5, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element5(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element5_nonzero(chi1, chi2, chi3, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element5_nonzero(chi1, chi2, chi3, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element5_nonzero(chi1, chi2, chi3, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element6(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element6_nonzero(chi1, chi2, chi3, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element6_nonzero(chi1, chi2, chi3, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element6_nonzero(chi1, chi2, chi3, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element7(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element7_nonzero(chi1, chi2, chi3, chi4, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element7_nonzero(chi1, chi2, chi3, chi4, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element7_nonzero(chi1, chi2, chi3, chi4, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element8(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element8_nonzero(chi1, chi2, chi3, chi4, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element8_nonzero(chi1, chi2, chi3, chi4, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element8_nonzero(chi1, chi2, chi3, chi4, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element9(chi1, chi2, power1, power2, redshift1, redshift2):
    """New final-interval NS B09; cubic observer handled separately."""
    return local.ns9(chi1, chi2, power1, power2, redshift1, redshift2)

@numba.njit(cache=True)
def element10(chi1, chi2, power1, power2, redshift1, redshift2):
    """New final-interval NS B10; cubic observer handled separately."""
    return local.ns10(chi1, chi2, power1, power2, redshift1, redshift2)

# Coefficient
@numba.njit(cache=True, parallel=True)
def coefficient(chi_grid, power_grid, redshift_grid):
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    coefficients = numpy.zeros((grid_size + 1, grid_size + 1, ell_size + 1), dtype=numpy.float64)

    # Loop
    for n in range(grid_size):
        # Element 1
        if n < grid_size:
            element = element1(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n, n, :] += element
        # Element 2. The node n+1 density hat and the partly covered node n
        # source hat both live on this interval, so the terminal density row
        # n+1 = N needs no next node and is stored on every interval.
        if n < grid_size:
            element = element2(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n + 1, n, :] += element
        # Element 3
        if n + 1 < grid_size:
            for k in numba.prange(n + 2, grid_size):
                element = element3(chi_grid[n], chi_grid[n + 1], chi_grid[k - 1], chi_grid[k], chi_grid[k + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
                coefficients[n, k, :] += element
        # Element 4
        if n + 1 < grid_size:
            for k in numba.prange(n + 2, grid_size):
                element = element4(chi_grid[n], chi_grid[n + 1], chi_grid[k - 1], chi_grid[k], chi_grid[k + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
                coefficients[n + 1, k, :] += element
        # Element 5
        if n + 1 < grid_size:
            element = element5(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n, n + 1, :] += element
        # Element 6
        if n + 1 < grid_size:
            element = element6(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n + 1, n + 1, :] += element
        # Element 7 and Element 8 place the terminal source node N. Below the
        # final interval the whole node-N hat lies above the evaluation point
        # and the source integral is linear in x. On the final interval the
        # evaluation point is inside that hat, so both placements switch to the
        # truncated source integral H_N(x).
        if n + 1 < grid_size:
            element = element7(chi_grid[n], chi_grid[n + 1], chi_grid[grid_size - 1], chi_grid[grid_size], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n, grid_size, :] += element
            element = element8(chi_grid[n], chi_grid[n + 1], chi_grid[grid_size - 1], chi_grid[grid_size], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n + 1, grid_size, :] += element
        else:
            element = element9(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n, grid_size, :] += element
            element = element10(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n + 1, grid_size, :] += element
    return coefficients

# Spectra
def spectra(factor, phi_a_grid, phi_b_grid, chi_grid, power_grid, redshift_grid):
    coefficients = coefficient(chi_grid, power_grid, redshift_grid)
    return factor * numpy.einsum('mi,nj,ijl->mnl', phi_a_grid, phi_b_grid, coefficients, dtype=numpy.float64)
