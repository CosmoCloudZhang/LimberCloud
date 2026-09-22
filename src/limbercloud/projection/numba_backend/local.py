"""Analytic local hat integrals with stable narrow-interval series.

The ordinary closed forms integrate polynomial/logarithmic moments exactly.
For a <= 3/4, convergent analytic series avoid cancellation. At most 160 terms
have a geometric tail below 1.1e-19 before the bounded moment factors.
These are integrated power series, not samples or quadrature. Observer
intervals use separately integrated cubic power. Endpoint powers enter
linearly, so signed and zero endpoints require no division by power.

See NS B09/B10 and SS B11/B12 derivation notebooks for the construction.
"""

import numba
import numpy

_SERIES_TERMS = 160
_D = numpy.array([1.0 / ((m + 2) * (m + 3)) for m in range(_SERIES_TERMS)])
_E = numpy.convolve(_D, _D)[:_SERIES_TERMS]


@numba.njit(cache=True)
def _series_terms(a):
    # a**K <= 2**-64 in each regime; at a=3/4, K=160 is smaller still.
    if a <= 0.0625:
        return 16
    if a <= 0.25:
        return 32
    if a <= 0.5:
        return 64
    return 160


@numba.njit(cache=True)
def _moment(n, power1, power2, redshift1, redshift2, squared):
    """Integrate s**n times endpoint-linear P and (1+z)**(1 or 2)."""
    r = 1.0 + redshift2
    dr = redshift1 - redshift2
    dp = power1 - power2
    if squared:
        return (power2*r*r/(n+1) + (dp*r*r + 2*power2*r*dr)/(n+2)
                + (2*dp*r*dr + power2*dr*dr)/(n+3) + dp*dr*dr/(n+4))
    return power2*r/(n+1) + (dp*r+power2*dr)/(n+2) + dp*dr/(n+3)


@numba.njit(cache=True)
def _density_series(a, power1, power2, redshift1, redshift2, source_rising, density_rising):
    total = numpy.zeros_like(power1)
    for m in range(_series_terms(a)):
        # F/t = a^2 sum_m a^m (1/2 - 1/(m+3)) s^(m+3).
        degree = m + 3
        weight = _moment(degree + 1, power1, power2, redshift1, redshift2, False)
        if density_rising:
            weight = _moment(degree, power1, power2, redshift1, redshift2, False) - weight
        value = (0.5 - 1.0/(m+3))*weight
        if source_rising:
            degree = m + 2
            other = _moment(degree + 1, power1, power2, redshift1, redshift2, False)
            if density_rising:
                other = _moment(degree, power1, power2, redshift1, redshift2, False) - other
            value = 0.5*other - (1-a)*value
        total += a**m*value
    return a**3*total


@numba.njit(cache=True)
def _lensing_series(a, power1, power2, redshift1, redshift2, rising_count):
    total = numpy.zeros_like(power1)
    for m in range(_series_terms(a)):
        ff = _E[m]*_moment(m+6, power1, power2, redshift1, redshift2, True)
        if rising_count == 0:
            value = ff
        else:
            sf = _D[m]*_moment(m+5, power1, power2, redshift1, redshift2, True)
            value = 0.5*sf - (1-a)*ff
            if rising_count == 2:
                value = -(1-a)*sf + (1-a)**2*ff
        total += a**m*value
    if rising_count == 2:
        total += 0.25*_moment(4, power1, power2, redshift1, redshift2, True)
    return a**5*total


@numba.njit(cache=True)
def ns1(chi1, chi2, power1, power2, redshift1, redshift2):
    """Evaluate NS B01 with stable local-source moments."""
    a = (chi2 - chi1)/chi2
    z = (redshift2 - redshift1)/(1 + redshift2)
    if chi1 == 0.0:
        return chi2*power2*(1 + redshift2)*(-1/25200*(41*z - 63))
    if a <= 0.75:
        return chi2*_density_series(a, power1, power2, redshift1, redshift2, False, False)
    left = -1/720*(72*a**5*z - 90*a**5 + 180*a**4*z*numpy.log(chi1 / chi2) - 135*a**4*z - 240*a**4*numpy.log(chi1 / chi2) + 200*a**4 - 180*a**3*z + 300*a**3 - 270*a**2*z + 600*a**2 - 540*a*z + 600*a*numpy.log(chi1 / chi2) - 540*z*numpy.log(chi1 / chi2))/a**4
    right = -1/720*(18*a**5*z - 30*a**5 + 60*a**4*z*numpy.log(chi1 / chi2) - 65*a**4*z - 120*a**4*numpy.log(chi1 / chi2) + 160*a**4 - 120*a**3*z + 420*a**3 - 330*a**2*z + 720*a**2*numpy.log(chi1 / chi2) - 600*a**2 - 600*a*z*numpy.log(chi1 / chi2) + 540*a*z - 600*a*numpy.log(chi1 / chi2) + 540*z*numpy.log(chi1 / chi2))/a**4
    return chi2*(1 + redshift2)*(power1*left + power2*right)


@numba.njit(cache=True)
def ns2(chi1, chi2, power1, power2, redshift1, redshift2):
    """Evaluate NS B02 with stable local-source moments."""
    a = (chi2 - chi1)/chi2
    z = (redshift2 - redshift1)/(1 + redshift2)
    if chi1 == 0.0:
        return chi2*power2*(1 + redshift2)*(-1/12600*(11*z - 21))
    if a <= 0.75:
        return chi2*_density_series(a, power1, power2, redshift1, redshift2, False, True)
    left = -1/720*(18*a**5*z - 30*a**5 + 60*a**4*z*numpy.log(chi1 / chi2) - 65*a**4*z - 120*a**4*numpy.log(chi1 / chi2) + 160*a**4 - 120*a**3*z + 420*a**3 - 330*a**2*z + 720*a**2*numpy.log(chi1 / chi2) - 600*a**2 - 600*a*z*numpy.log(chi1 / chi2) + 540*a*z - 600*a*numpy.log(chi1 / chi2) + 540*z*numpy.log(chi1 / chi2))/a**4
    right = -1/720*(12*a**5*z - 30*a**5 + 60*a**4*z*numpy.log(chi1 / chi2) - 95*a**4*z - 240*a**4*numpy.log(chi1 / chi2) + 560*a**4 - 300*a**3*z + 1080*a**3*numpy.log(chi1 / chi2) - 1140*a**3 - 720*a**2*z*numpy.log(chi1 / chi2) + 930*a**2*z - 1440*a**2*numpy.log(chi1 / chi2) + 600*a**2 + 1200*a*z*numpy.log(chi1 / chi2) - 540*a*z + 600*a*numpy.log(chi1 / chi2) - 540*z*numpy.log(chi1 / chi2))/a**4
    return chi2*(1 + redshift2)*(power1*left + power2*right)


@numba.njit(cache=True)
def ns9(chi1, chi2, power1, power2, redshift1, redshift2):
    """Evaluate NS B09 with stable local-source moments."""
    a = (chi2 - chi1)/chi2
    z = (redshift2 - redshift1)/(1 + redshift2)
    if chi1 == 0.0:
        return chi2*power2*(1 + redshift2)*(-1/840*(4*z - 7))
    if a <= 0.75:
        return chi2*_density_series(a, power1, power2, redshift1, redshift2, True, False)
    left = -1/720*(180*a**5*z*numpy.log(chi1 / chi2) - 297*a**5*z - 240*a**5*numpy.log(chi1 / chi2) + 410*a**5 - 180*a**4*z*numpy.log(chi1 / chi2) - 165*a**4*z + 240*a**4*numpy.log(chi1 / chi2) + 280*a**4 - 270*a**3*z + 660*a**3 - 630*a**2*z + 960*a**2*numpy.log(chi1 / chi2) - 600*a**2 - 900*a*z*numpy.log(chi1 / chi2) + 540*a*z - 600*a*numpy.log(chi1 / chi2) + 540*z*numpy.log(chi1 / chi2))/a**4
    right = -1/720*(60*a**5*z*numpy.log(chi1 / chi2) - 113*a**5*z - 120*a**5*numpy.log(chi1 / chi2) + 250*a**5 - 60*a**4*z*numpy.log(chi1 / chi2) - 115*a**4*z + 120*a**4*numpy.log(chi1 / chi2) + 440*a**4 - 390*a**3*z + 1080*a**3*numpy.log(chi1 / chi2) - 1380*a**3 - 960*a**2*z*numpy.log(chi1 / chi2) + 1230*a**2*z - 1680*a**2*numpy.log(chi1 / chi2) + 600*a**2 + 1500*a*z*numpy.log(chi1 / chi2) - 540*a*z + 600*a*numpy.log(chi1 / chi2) - 540*z*numpy.log(chi1 / chi2))/a**4
    return chi2*(1 + redshift2)*(power1*left + power2*right)


@numba.njit(cache=True)
def ns10(chi1, chi2, power1, power2, redshift1, redshift2):
    """Evaluate NS B10 with stable local-source moments."""
    a = (chi2 - chi1)/chi2
    z = (redshift2 - redshift1)/(1 + redshift2)
    if chi1 == 0.0:
        return chi2*power2*(1 + redshift2)*(-1/840*(3*z - 7))
    if a <= 0.75:
        return chi2*_density_series(a, power1, power2, redshift1, redshift2, True, True)
    left = -1/720*(60*a**5*z*numpy.log(chi1 / chi2) - 113*a**5*z - 120*a**5*numpy.log(chi1 / chi2) + 250*a**5 - 60*a**4*z*numpy.log(chi1 / chi2) - 115*a**4*z + 120*a**4*numpy.log(chi1 / chi2) + 440*a**4 - 390*a**3*z + 1080*a**3*numpy.log(chi1 / chi2) - 1380*a**3 - 960*a**2*z*numpy.log(chi1 / chi2) + 1230*a**2*z - 1680*a**2*numpy.log(chi1 / chi2) + 600*a**2 + 1500*a*z*numpy.log(chi1 / chi2) - 540*a*z + 600*a*numpy.log(chi1 / chi2) - 540*z*numpy.log(chi1 / chi2))/a**4
    right = -1/720*(60*a**5*z*numpy.log(chi1 / chi2) - 137*a**5*z - 240*a**5*numpy.log(chi1 / chi2) + 710*a**5 - 60*a**4*z*numpy.log(chi1 / chi2) - 325*a**4*z + 1680*a**4*numpy.log(chi1 / chi2) - 2240*a**4 - 1080*a**3*z*numpy.log(chi1 / chi2) + 1770*a**3*z - 3240*a**3*numpy.log(chi1 / chi2) + 2100*a**3 + 2640*a**2*z*numpy.log(chi1 / chi2) - 1830*a**2*z + 2400*a**2*numpy.log(chi1 / chi2) - 600*a**2 - 2100*a*z*numpy.log(chi1 / chi2) + 540*a*z - 600*a*numpy.log(chi1 / chi2) + 540*z*numpy.log(chi1 / chi2))/a**4
    return chi2*(1 + redshift2)*(power1*left + power2*right)


@numba.njit(cache=True)
def ss1(chi1, chi2, power1, power2, redshift1, redshift2):
    """Evaluate SS B01 with stable local-source moments."""
    a = (chi2 - chi1)/chi2
    z = (redshift2 - redshift1)/(1 + redshift2)
    if chi1 == 0.0:
        return chi2**3*power2*(1 + redshift2)**2*((1/177811200)*(7087*z**2 - 20184*z + 14952))
    if a <= 0.75:
        return chi2**3*_lensing_series(a, power1, power2, redshift1, redshift2, 0)
    left = (1/5292000)*(165375*a**8*z**2 - 378000*a**8*z + 220500*a**8 + 756000*a**7*z**2*numpy.log(chi1 / chi2) - 864000*a**7*z**2 - 1764000*a**7*z*numpy.log(chi1 / chi2) + 2058000*a**7*z + 1058400*a**7*numpy.log(chi1 / chi2) - 1270080*a**7 + 882000*a**6*z**2*numpy.log(chi1 / chi2)**2 - 2940000*a**6*z**2*numpy.log(chi1 / chi2) + 1246000*a**6*z**2 - 2116800*a**6*z*numpy.log(chi1 / chi2)**2 + 7197120*a**6*z*numpy.log(chi1 / chi2) - 3203424*a**6*z + 1323000*a**6*numpy.log(chi1 / chi2)**2 - 4630500*a**6*numpy.log(chi1 / chi2) + 2216025*a**6 - 2116800*a**5*z**2*numpy.log(chi1 / chi2)**2 + 2610720*a**5*z**2*numpy.log(chi1 / chi2) - 85344*a**5*z**2 + 5292000*a**5*z*numpy.log(chi1 / chi2)**2 - 6879600*a**5*z*numpy.log(chi1 / chi2) + 361620*a**5*z - 3528000*a**5*numpy.log(chi1 / chi2)**2 + 4998000*a**5*numpy.log(chi1 / chi2) - 475300*a**5 + 1323000*a**4*z**2*numpy.log(chi1 / chi2)**2 - 44100*a**4*z**2*numpy.log(chi1 / chi2) - 95655*a**4*z**2 - 3528000*a**4*z*numpy.log(chi1 / chi2)**2 + 235200*a**4*z*numpy.log(chi1 / chi2) + 403760*a**4*z + 2646000*a**4*numpy.log(chi1 / chi2)**2 - 441000*a**4*numpy.log(chi1 / chi2) - 492450*a**4 - 58800*a**3*z**2*numpy.log(chi1 / chi2) - 107940*a**3*z**2 + 352800*a**3*z*numpy.log(chi1 / chi2) + 429240*a**3*z - 882000*a**3*numpy.log(chi1 / chi2) - 102900*a**3 - 88200*a**2*z**2*numpy.log(chi1 / chi2) - 117810*a**2*z**2 + 705600*a**2*z*numpy.log(chi1 / chi2) + 152880*a**2*z - 441000*a**2*numpy.log(chi1 / chi2)**2 - 102900*a**2*numpy.log(chi1 / chi2) - 176400*a*z**2*numpy.log(chi1 / chi2) - 59220*a*z**2 + 352800*a*z*numpy.log(chi1 / chi2)**2 + 152880*a*z*numpy.log(chi1 / chi2) - 88200*z**2*numpy.log(chi1 / chi2)**2 - 59220*z**2*numpy.log(chi1 / chi2))/a**5
    right = (1/5292000)*(23625*a**8*z**2 - 63000*a**8*z + 44100*a**8 + 126000*a**7*z**2*numpy.log(chi1 / chi2) - 165000*a**7*z**2 - 352800*a**7*z*numpy.log(chi1 / chi2) + 482160*a**7*z + 264600*a**7*numpy.log(chi1 / chi2) - 383670*a**7 + 176400*a**6*z**2*numpy.log(chi1 / chi2)**2 - 658560*a**6*z**2*numpy.log(chi1 / chi2) + 355712*a**6*z**2 - 529200*a**6*z*numpy.log(chi1 / chi2)**2 + 2063880*a**6*z*numpy.log(chi1 / chi2) - 1228626*a**6*z + 441000*a**6*numpy.log(chi1 / chi2)**2 - 1837500*a**6*numpy.log(chi1 / chi2) + 1262975*a**6 - 529200*a**5*z**2*numpy.log(chi1 / chi2)**2 + 829080*a**5*z**2*numpy.log(chi1 / chi2) - 95466*a**5*z**2 + 1764000*a**5*z*numpy.log(chi1 / chi2)**2 - 3116400*a**5*z*numpy.log(chi1 / chi2) + 588980*a**5*z - 1764000*a**5*numpy.log(chi1 / chi2)**2 + 3822000*a**5*numpy.log(chi1 / chi2) - 1362200*a**5 + 441000*a**4*z**2*numpy.log(chi1 / chi2)**2 - 73500*a**4*z**2*numpy.log(chi1 / chi2) - 106225*a**4*z**2 - 1764000*a**4*z*numpy.log(chi1 / chi2)**2 + 646800*a**4*z*numpy.log(chi1 / chi2) + 581140*a**4*z + 2646000*a**4*numpy.log(chi1 / chi2)**2 - 3087000*a**4*numpy.log(chi1 / chi2) + 345450*a**4 - 117600*a**3*z**2*numpy.log(chi1 / chi2) - 106680*a**3*z**2 + 1411200*a**3*z*numpy.log(chi1 / chi2) - 223440*a**3*z - 1764000*a**3*numpy.log(chi1 / chi2)**2 + 735000*a**3*numpy.log(chi1 / chi2) + 102900*a**3 - 264600*a**2*z**2*numpy.log(chi1 / chi2) + 41370*a**2*z**2 + 882000*a**2*z*numpy.log(chi1 / chi2)**2 - 499800*a**2*z*numpy.log(chi1 / chi2) - 152880*a**2*z + 441000*a**2*numpy.log(chi1 / chi2)**2 + 102900*a**2*numpy.log(chi1 / chi2) - 176400*a*z**2*numpy.log(chi1 / chi2)**2 + 99960*a*z**2*numpy.log(chi1 / chi2) + 59220*a*z**2 - 352800*a*z*numpy.log(chi1 / chi2)**2 - 152880*a*z*numpy.log(chi1 / chi2) + 88200*z**2*numpy.log(chi1 / chi2)**2 + 59220*z**2*numpy.log(chi1 / chi2))/a**5
    return chi2**3*(1 + redshift2)**2*(power1*left + power2*right)


@numba.njit(cache=True)
def ss11(chi1, chi2, power1, power2, redshift1, redshift2):
    """Evaluate SS B11 with stable local-source moments."""
    a = (chi2 - chi1)/chi2
    z = (redshift2 - redshift1)/(1 + redshift2)
    if chi1 == 0.0:
        return chi2**3*power2*(1 + redshift2)**2*((1/6350400)*(652*z**2 - 1989*z + 1602))
    if a <= 0.75:
        return chi2**3*_lensing_series(a, power1, power2, redshift1, redshift2, 1)
    left = (1/5292000)*(378000*a**8*z**2*numpy.log(chi1 / chi2) - 597375*a**8*z**2 - 882000*a**8*z*numpy.log(chi1 / chi2) + 1407000*a**8*z + 529200*a**8*numpy.log(chi1 / chi2) - 855540*a**8 + 882000*a**7*z**2*numpy.log(chi1 / chi2)**2 - 3255000*a**7*z**2*numpy.log(chi1 / chi2) + 2099500*a**7*z**2 - 2116800*a**7*z*numpy.log(chi1 / chi2)**2 + 7902720*a**7*z*numpy.log(chi1 / chi2) - 5226144*a**7*z + 1323000*a**7*numpy.log(chi1 / chi2)**2 - 5027400*a**7*numpy.log(chi1 / chi2) + 3453030*a**7 - 2998800*a**6*z**2*numpy.log(chi1 / chi2)**2 + 5550720*a**6*z**2*numpy.log(chi1 / chi2) - 1343944*a**6*z**2 + 7408800*a**6*z*numpy.log(chi1 / chi2)**2 - 14076720*a**6*z*numpy.log(chi1 / chi2) + 3609144*a**6*z - 4851000*a**6*numpy.log(chi1 / chi2)**2 + 9628500*a**6*numpy.log(chi1 / chi2) - 2735425*a**6 + 3439800*a**5*z**2*numpy.log(chi1 / chi2)**2 - 2654820*a**5*z**2*numpy.log(chi1 / chi2) - 26061*a**5*z**2 - 8820000*a**5*z*numpy.log(chi1 / chi2)**2 + 7114800*a**5*z*numpy.log(chi1 / chi2) + 100940*a**5*z + 6174000*a**5*numpy.log(chi1 / chi2)**2 - 5439000*a**5*numpy.log(chi1 / chi2) - 83300*a**5 - 1323000*a**4*z**2*numpy.log(chi1 / chi2)**2 - 14700*a**4*z**2*numpy.log(chi1 / chi2) - 33285*a**4*z**2 + 3528000*a**4*z*numpy.log(chi1 / chi2)**2 + 117600*a**4*z*numpy.log(chi1 / chi2) + 113680*a**4*z - 2646000*a**4*numpy.log(chi1 / chi2)**2 - 441000*a**4*numpy.log(chi1 / chi2) + 257250*a**4 - 29400*a**3*z**2*numpy.log(chi1 / chi2) - 41370*a**3*z**2 + 352800*a**3*z*numpy.log(chi1 / chi2) - 99960*a**3*z - 441000*a**3*numpy.log(chi1 / chi2)**2 + 646800*a**3*numpy.log(chi1 / chi2) + 102900*a**3 - 88200*a**2*z**2*numpy.log(chi1 / chi2) - 4410*a**2*z**2 + 352800*a**2*z*numpy.log(chi1 / chi2)**2 - 376320*a**2*z*numpy.log(chi1 / chi2) - 152880*a**2*z + 441000*a**2*numpy.log(chi1 / chi2)**2 + 102900*a**2*numpy.log(chi1 / chi2) - 88200*a*z**2*numpy.log(chi1 / chi2)**2 + 54180*a*z**2*numpy.log(chi1 / chi2) + 59220*a*z**2 - 352800*a*z*numpy.log(chi1 / chi2)**2 - 152880*a*z*numpy.log(chi1 / chi2) + 88200*z**2*numpy.log(chi1 / chi2)**2 + 59220*z**2*numpy.log(chi1 / chi2))/a**5
    right = (1/5292000)*(63000*a**8*z**2*numpy.log(chi1 / chi2) - 106125*a**8*z**2 - 176400*a**8*z*numpy.log(chi1 / chi2) + 304080*a**8*z + 132300*a**8*numpy.log(chi1 / chi2) - 235935*a**8 + 176400*a**7*z**2*numpy.log(chi1 / chi2)**2 - 696360*a**7*z**2*numpy.log(chi1 / chi2) + 513572*a**7*z**2 - 529200*a**7*z*numpy.log(chi1 / chi2)**2 + 2152080*a**7*z*numpy.log(chi1 / chi2) - 1679916*a**7*z + 441000*a**7*numpy.log(chi1 / chi2)**2 - 1881600*a**7*numpy.log(chi1 / chi2) + 1606220*a**7 - 705600*a**6*z**2*numpy.log(chi1 / chi2)**2 + 1487640*a**6*z**2*numpy.log(chi1 / chi2) - 460628*a**6*z**2 + 2293200*a**6*z*numpy.log(chi1 / chi2)**2 - 5180280*a**6*z*numpy.log(chi1 / chi2) + 1861706*a**6*z - 2205000*a**6*numpy.log(chi1 / chi2)**2 + 5659500*a**6*numpy.log(chi1 / chi2) - 2691325*a**6 + 970200*a**5*z**2*numpy.log(chi1 / chi2)**2 - 902580*a**5*z**2*numpy.log(chi1 / chi2) - 24409*a**5*z**2 - 3528000*a**5*z*numpy.log(chi1 / chi2)**2 + 3763200*a**5*z*numpy.log(chi1 / chi2) + 65660*a**5*z + 4410000*a**5*numpy.log(chi1 / chi2)**2 - 6909000*a**5*numpy.log(chi1 / chi2) + 1553300*a**5 - 441000*a**4*z**2*numpy.log(chi1 / chi2)**2 - 44100*a**4*z**2*numpy.log(chi1 / chi2) - 23555*a**4*z**2 + 1764000*a**4*z*numpy.log(chi1 / chi2)**2 + 764400*a**4*z*numpy.log(chi1 / chi2) - 628180*a**4*z - 4410000*a**4*numpy.log(chi1 / chi2)**2 + 3601500*a**4*numpy.log(chi1 / chi2) - 110250*a**4 - 147000*a**3*z**2*numpy.log(chi1 / chi2) + 91350*a**3*z**2 + 882000*a**3*z*numpy.log(chi1 / chi2)**2 - 1646400*a**3*z*numpy.log(chi1 / chi2) - 105840*a**3*z + 2205000*a**3*numpy.log(chi1 / chi2)**2 - 499800*a**3*numpy.log(chi1 / chi2) - 102900*a**3 - 176400*a**2*z**2*numpy.log(chi1 / chi2)**2 + 276360*a**2*z**2*numpy.log(chi1 / chi2) + 80850*a**2*z**2 - 1234800*a**2*z*numpy.log(chi1 / chi2)**2 + 170520*a**2*z*numpy.log(chi1 / chi2) + 152880*a**2*z - 441000*a**2*numpy.log(chi1 / chi2)**2 - 102900*a**2*numpy.log(chi1 / chi2) + 264600*a*z**2*numpy.log(chi1 / chi2)**2 + 22260*a*z**2*numpy.log(chi1 / chi2) - 59220*a*z**2 + 352800*a*z*numpy.log(chi1 / chi2)**2 + 152880*a*z*numpy.log(chi1 / chi2) - 88200*z**2*numpy.log(chi1 / chi2)**2 - 59220*z**2*numpy.log(chi1 / chi2))/a**5
    return chi2**3*(1 + redshift2)**2*(power1*left + power2*right)


@numba.njit(cache=True)
def ss12(chi1, chi2, power1, power2, redshift1, redshift2):
    """Evaluate SS B12 with stable local-source moments."""
    a = (chi2 - chi1)/chi2
    z = (redshift2 - redshift1)/(1 + redshift2)
    if chi1 == 0.0:
        return chi2**3*power2*(1 + redshift2)**2*((1/10080)*(3*z**2 - 10*z + 9))
    if a <= 0.75:
        return chi2**3*_lensing_series(a, power1, power2, redshift1, redshift2, 2)
    left = (1/5292000)*(882000*a**8*z**2*numpy.log(chi1 / chi2)**2 - 2814000*a**8*z**2*numpy.log(chi1 / chi2) + 2254375*a**8*z**2 - 2116800*a**8*z*numpy.log(chi1 / chi2)**2 + 6844320*a**8*z*numpy.log(chi1 / chi2) - 5568864*a**8*z + 1323000*a**8*numpy.log(chi1 / chi2)**2 - 4365900*a**8*numpy.log(chi1 / chi2) + 3640455*a**8 - 3880800*a**7*z**2*numpy.log(chi1 / chi2)**2 + 8364720*a**7*z**2*numpy.log(chi1 / chi2) - 3445544*a**7*z**2 + 9525600*a**7*z*numpy.log(chi1 / chi2)**2 - 20921040*a**7*z*numpy.log(chi1 / chi2) + 8844108*a**7*z - 6174000*a**7*numpy.log(chi1 / chi2)**2 + 13994400*a**7*numpy.log(chi1 / chi2) - 6199480*a**7 + 6438600*a**6*z**2*numpy.log(chi1 / chi2)**2 - 8205540*a**6*z**2*numpy.log(chi1 / chi2) + 1314733*a**6*z**2 - 16228800*a**6*z*numpy.log(chi1 / chi2)**2 + 21191520*a**6*z*numpy.log(chi1 / chi2) - 3493504*a**6*z + 11025000*a**6*numpy.log(chi1 / chi2)**2 - 15067500*a**6*numpy.log(chi1 / chi2) + 2630075*a**6 - 4762800*a**5*z**2*numpy.log(chi1 / chi2)**2 + 2640120*a**5*z**2*numpy.log(chi1 / chi2) - 12474*a**5*z**2 + 12348000*a**5*z*numpy.log(chi1 / chi2)**2 - 6997200*a**5*z*numpy.log(chi1 / chi2) + 42140*a**5*z - 8820000*a**5*numpy.log(chi1 / chi2)**2 + 4998000*a**5*numpy.log(chi1 / chi2) + 274400*a**5 + 1323000*a**4*z**2*numpy.log(chi1 / chi2)**2 - 14700*a**4*z**2*numpy.log(chi1 / chi2) - 18585*a**4*z**2 - 3528000*a**4*z*numpy.log(chi1 / chi2)**2 + 235200*a**4*z*numpy.log(chi1 / chi2) - 125440*a**4*z + 2205000*a**4*numpy.log(chi1 / chi2)**2 + 955500*a**4*numpy.log(chi1 / chi2) - 22050*a**4 - 58800*a**3*z**2*numpy.log(chi1 / chi2) + 5460*a**3*z**2 + 352800*a**3*z*numpy.log(chi1 / chi2)**2 - 552720*a**3*z*numpy.log(chi1 / chi2) - 229320*a**3*z + 882000*a**3*numpy.log(chi1 / chi2)**2 - 411600*a**3*numpy.log(chi1 / chi2) - 102900*a**3 - 88200*a**2*z**2*numpy.log(chi1 / chi2)**2 + 79380*a**2*z**2*numpy.log(chi1 / chi2) + 126630*a**2*z**2 - 705600*a**2*z*numpy.log(chi1 / chi2)**2 + 47040*a**2*z*numpy.log(chi1 / chi2) + 152880*a**2*z - 441000*a**2*numpy.log(chi1 / chi2)**2 - 102900*a**2*numpy.log(chi1 / chi2) + 176400*a*z**2*numpy.log(chi1 / chi2)**2 + 68040*a*z**2*numpy.log(chi1 / chi2) - 59220*a*z**2 + 352800*a*z*numpy.log(chi1 / chi2)**2 + 152880*a*z*numpy.log(chi1 / chi2) - 88200*z**2*numpy.log(chi1 / chi2)**2 - 59220*z**2*numpy.log(chi1 / chi2))/a**5
    right = (1/5292000)*(176400*a**8*z**2*numpy.log(chi1 / chi2)**2 - 608160*a**8*z**2*numpy.log(chi1 / chi2) + 530057*a**8*z**2 - 529200*a**8*z*numpy.log(chi1 / chi2)**2 + 1887480*a**8*z*numpy.log(chi1 / chi2) - 1712046*a**8*z + 441000*a**8*numpy.log(chi1 / chi2)**2 - 1661100*a**8*numpy.log(chi1 / chi2) + 1609895*a**8 - 882000*a**7*z**2*numpy.log(chi1 / chi2)**2 + 2095800*a**7*z**2*numpy.log(chi1 / chi2) - 976510*a**7*z**2 + 2822400*a**7*z*numpy.log(chi1 / chi2)**2 - 7067760*a**7*z*numpy.log(chi1 / chi2) + 3554852*a**7*z - 2646000*a**7*numpy.log(chi1 / chi2)**2 + 7320600*a**7*numpy.log(chi1 / chi2) - 4323270*a**7 + 1675800*a**6*z**2*numpy.log(chi1 / chi2)**2 - 2390220*a**6*z**2*numpy.log(chi1 / chi2) + 432019*a**6*z**2 - 5821200*a**6*z*numpy.log(chi1 / chi2)**2 + 8943480*a**6*z*numpy.log(chi1 / chi2) - 1766646*a**6*z + 6615000*a**6*numpy.log(chi1 / chi2)**2 - 12568500*a**6*numpy.log(chi1 / chi2) + 4156425*a**6 - 1411200*a**5*z**2*numpy.log(chi1 / chi2)**2 + 858480*a**5*z**2*numpy.log(chi1 / chi2) - 8596*a**5*z**2 + 5292000*a**5*z*numpy.log(chi1 / chi2)**2 - 2998800*a**5*z*numpy.log(chi1 / chi2) - 590940*a**5*z - 8820000*a**5*numpy.log(chi1 / chi2)**2 + 10290000*a**5*numpy.log(chi1 / chi2) - 1376900*a**5 + 441000*a**4*z**2*numpy.log(chi1 / chi2)**2 - 102900*a**4*z**2*numpy.log(chi1 / chi2) + 81305*a**4*z**2 - 882000*a**4*z*numpy.log(chi1 / chi2)**2 - 2146200*a**4*z*numpy.log(chi1 / chi2) + 169540*a**4*z + 6615000*a**4*numpy.log(chi1 / chi2)**2 - 3748500*a**4*numpy.log(chi1 / chi2) - 124950*a**4 - 176400*a**3*z**2*numpy.log(chi1 / chi2)**2 + 335160*a**3*z**2*numpy.log(chi1 / chi2) + 109200*a**3*z**2 - 2116800*a**3*z*numpy.log(chi1 / chi2)**2 + 1375920*a**3*z*numpy.log(chi1 / chi2) + 435120*a**3*z - 2646000*a**3*numpy.log(chi1 / chi2)**2 + 264600*a**3*numpy.log(chi1 / chi2) + 102900*a**3 + 441000*a**2*z**2*numpy.log(chi1 / chi2)**2 - 102900*a**2*z**2*numpy.log(chi1 / chi2) - 203070*a**2*z**2 + 1587600*a**2*z*numpy.log(chi1 / chi2)**2 + 158760*a**2*z*numpy.log(chi1 / chi2) - 152880*a**2*z + 441000*a**2*numpy.log(chi1 / chi2)**2 + 102900*a**2*numpy.log(chi1 / chi2) - 352800*a*z**2*numpy.log(chi1 / chi2)**2 - 144480*a*z**2*numpy.log(chi1 / chi2) + 59220*a*z**2 - 352800*a*z*numpy.log(chi1 / chi2)**2 - 152880*a*z*numpy.log(chi1 / chi2) + 88200*z**2*numpy.log(chi1 / chi2)**2 + 59220*z**2*numpy.log(chi1 / chi2))/a**5
    return chi2**3*(1 + redshift2)**2*(power1*left + power2*right)

