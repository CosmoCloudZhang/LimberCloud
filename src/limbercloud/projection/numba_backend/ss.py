import numba
import numpy

from limbercloud.projection.numba_backend import local


# Element 1
@numba.njit(cache=True)
def element1(chi1, chi2, power1, power2, redshift1, redshift2):
    return local.ss1(chi1, chi2, power1, power2, redshift1, redshift2)

# Element 2
@numba.njit(cache=True)
def _element2_nonzero(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (b * (6642 + z * ( - 6669 + 1867 * z) + 18 * b * (294 + z * ( - 308 + 89 * z))) - 45 * (1 + b) * (112 + z * ( - 104 + 27 * z)) * numpy.log(1 + b)) / (6350400 * b))
    else:
        formula = - ((1 / (4 * a ** 5 * b)) * ((1 / 15) * (-chi1 / chi2) ** 4 * b * (5 * a ** 2 * (p + a * ( - 4 + 3 * p)) - 2 * a * (2 * p + a * ( - 5 + 6 * p + 3 * a * ( - 5 + 4 * p))) * z +
( - 2 * a * (1 + 3 * a + 6 * a ** 2) + p + a * (3 + 2 * a * (3 + 5 * a)) * p) * z ** 2) * numpy.log(chi1 / chi2) ** 2 +
(1 / 3150) * ((-chi1 / chi2) ** 2 * numpy.log(chi1 / chi2) * (b * (35 * a ** 2 * ( - 5 * a * (2 + a * (23 + a * ( - 52 + 9 * a) - 18 * b)) + (7 + a * (28 + a * (19 + 36 * ( - 5 + a) * a - 60 * b) - 30 * b)) * p) + 14 * a * (5 * a * (7 + a * (28 + a * (19 + 36 * ( - 5 + a) * a - 60 * b) - 30 * b)) - 26 * p + a * ( - 56 + 2 * a * ( - 13 + 3 * a * (4 + (114 - 25 * a) * a)) +
75 * (1 + a * (2 + 3 * a)) * b) * p) * z + (7 * a * ( - 26 - 2 * a * (28 + a * (13 + 3 * a * ( - 4 + a * ( - 114 + 25 * a)))) + 75 * a * (1 + a * (2 + 3 * a)) * b) +
(141 + a * (201 + 51 * a - 169 * a ** 2 - 424 * a ** 3 - 3850 * a ** 4 + 900 * a ** 5 - 315 * (1 + a * (2 + a * (3 + 4 * a))) * b)) * p) * z ** 2) -
210 * (-chi1 / chi2) * a * (1 + b) * (5 * a ** 2 * (p + a * ( - 4 + 3 * p)) - 2 * a * (2 * p + a * ( - 5 + 6 * p + 3 * a * ( - 5 + 4 * p))) * z +
( - 2 * a * (1 + 3 * a + 6 * a ** 2) + p + a * (3 + 2 * a * (3 + 5 * a)) * p) * z ** 2) * numpy.log(1 + b))) -
(1 / 1323000) * (a * (b * (245 * a ** 2 * ( - 420 * p + 20 * a ** 2 * ( - 60 + 45 * b * ( - 6 + p) + 62 * p) + 9 * a ** 5 * ( - 195 + 148 * p) + 5 * a**3 * ( - 830 + 60 * b * (27 - 16 * p) + 193 * p) +
150 * a * (4 + (5 + 12 * b) * p) + 2 * a ** 4 * (3725 - 2322 * p + 225 * b * ( - 4 + 3 * p))) -
14 * a * (35 * a * (420 + a * ( - 150 * (5 + 12 * b) + a * ( - 20 * (62 + 45 * b) + a * ( - 965 + 4800 * b - 18 * a * ( - 258 + 74 * a + 75 * b))))) +
( - 10920 + 2 * a * (9030 + a * (10360 + a * (7805 + 6 * a * (1008 + a * ( - 9688 + 3125 * a))))) + 525 * a * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b) * p) * z + (7 * a * (10920 - 2 * a * (9030 + a * (10360 + a * (7805 + 6 * a * (1008 + a * ( - 9688 + 3125 * a))))) -
525 * a * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b) + ( - 59220 + a * (92610 + 85470 * a + 62685 * a ** 2 + 48111 * a ** 3 + 38584 * a ** 4 - 629500 * a ** 5 + 219375 * a**6 + 2205 * (60 + a * (30 + a * (20 + a * (15 + 4 * a * ( - 72 + 25 * a))))) * b)) * p) * z**2) +
210 * a * (1 + b) * ( - 210 * a * (p * ( - 8 + z) - 4 * z) * z - 420 * p * z ** 2 - 35 * a ** 3 * (30 * ( - 8 + p) + 4 * (15 - 4 * p) * z + ( - 8 + 3 * p) * z ** 2) +
14 * a ** 4 * ( - 1500 + 850 * p + 5 * (340 - 117 * z) * z + 6 * p * z * ( - 195 + 74 * z)) - 140 * a ** 2 * ( - 3 * ( - 10 + z) * z + p * (15 + ( - 6 + z) * z)) +
30 * a ** 6 * (4 * p * (21 + 5 * z * ( - 7 + 3 * z)) - 7 * (15 + 2 * z * ( - 12 + 5 * z))) - 7 * a**5 * ( - 2200 + 18 * (175 - 68 * z) * z + p * (1575 + 8 * z * ( - 306 + 125 * z)))) * numpy.log(1 + b)))))

    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 3
@numba.njit(cache=True)
def _element3_nonzero(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5 - chi3) / (2 * chi2)
    c = chi3 * numpy.log(chi4 / chi3) / (chi4 - chi3) - chi5 * numpy.log(chi5 / chi4) / (chi5 - chi4)
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (5 * c * (112 + z * ( - 104 + 27 * z)) + 4 * b * (294 + z * ( - 308 + 89 * z))) / 705600)
    else:
        formula = - ((1 / (2 * a ** 4)) * ((1 / 12600) * (a * (35 * a ** 2 * (10 * a * (6 * (6 + a * ( - 9 + 2 * a)) * b + (24 + a * ( - 60 + (44 - 9 * a) * a)) * c) -
(10 * (12 + a * (6 + a * ( - 32 + 9 * a))) * b + (60 + a * (30 + a * ( - 340 + 9 * (35 - 8 * a) * a))) * c) * p) -
14 * a * (50 * a * (12 + a * (6 + a * ( - 32 + 9 * a))) * b - 5 * a * ( - 60 + a * ( - 30 + a * (340 + 9 * a * ( - 35 + 8 * a)))) * c -
(5 * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 2 * (60 + a * (30 + a * (20 - 3 * a * (195 - 204 * a + 50 * a ** 2)))) * c) * p) * z +
(7 * a * (5 * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 2 * (60 + a * (30 + a * (20 - 3 * a * (195 - 204 * a + 50 * a ** 2)))) * c) -
(21 * (60 + a * (30 + a * (20 + a * (15 + 4 * a * ( - 72 + 25 * a))))) * b + (420 + a * (210 + a * (140 + a * (105 - 8 * a * (777 + 25 * a * ( - 35 + 9 * a)))))) * c) * p) * z ** 2)) + (1 / 30) * (-chi1 / chi2) ** 2 * (5 * a ** 2 * (6 * a * b - 4 * (-chi1 / chi2) * a * c - ((2 + 4 * a) * b + c + (2 - 3 * a) * a * c) * p) +
2 * a * ( - 10 * a * (1 + 2 * a) * b + 5 * (-chi1 / chi2) * a * (1 + 3 * a) * c + 5 * (1 + a * (2 + 3 * a)) * b * p - 2 * (-chi1 / chi2) * (1 + 3 * a + 6 * a ** 2) * c * p) * z +
(5 * a * (1 + a * (2 + 3 * a)) * b - 2 * (-chi1 / chi2) * a * (1 + 3 * a + 6 * a ** 2) * c - (3 * (1 + a * (2 + a * (3 + 4 * a))) * b + c + a * (2 + a * (3 + 2 * (2 - 5 * a) * a)) * c) * p) * z ** 2) * numpy.log(chi1 / chi2)))

    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 4
@numba.njit(cache=True)
def _element4_nonzero(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi4 - chi3) / (2 * chi2)
    c = chi3 * numpy.log(chi4 / chi3) / (chi4 - chi3) - 1
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (5 * c * (112 + z * ( - 104 + 27 * z)) + 4 * b * (294 + z * ( - 308 + 89 * z))) / 705600)
    else:
        formula = - ((1 / (2 * a ** 4)) * ((1 / 12600) * (a * (35 * a ** 2 * (10 * a * (6 * (6 + a * ( - 9 + 2 * a)) * b + (24 + a * ( - 60 + (44 - 9 * a) * a)) * c) -
(10 * (12 + a * (6 + a * ( - 32 + 9 * a))) * b + (60 + a * (30 + a * ( - 340 + 9 * (35 - 8 * a) * a))) * c) * p) -
14 * a * (50 * a * (12 + a * (6 + a * ( - 32 + 9 * a))) * b - 5 * a * ( - 60 + a * ( - 30 + a * (340 + 9 * a * ( - 35 + 8 * a)))) * c -
(5 * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 2 * (60 + a * (30 + a * (20 - 3 * a * (195 - 204 * a + 50 * a ** 2)))) * c) * p) * z +
(7 * a * (5 * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 2 * (60 + a * (30 + a * (20 - 3 * a * (195 - 204 * a + 50 * a ** 2)))) * c) -
(21 * (60 + a * (30 + a * (20 + a * (15 + 4 * a * ( - 72 + 25 * a))))) * b + (420 + a * (210 + a * (140 + a * (105 - 8 * a * (777 + 25 * a * ( - 35 + 9 * a)))))) * c) * p) * z ** 2)) + (1 / 30) * (-chi1 / chi2) ** 2 * (5 * a ** 2 * (6 * a * b - 4 * (-chi1 / chi2) * a * c - ((2 + 4 * a) * b + c + (2 - 3 * a) * a * c) * p) +
2 * a * ( - 10 * a * (1 + 2 * a) * b + 5 * (-chi1 / chi2) * a * (1 + 3 * a) * c + 5 * (1 + a * (2 + 3 * a)) * b * p - 2 * (-chi1 / chi2) * (1 + 3 * a + 6 * a ** 2) * c * p) * z +
(5 * a * (1 + a * (2 + 3 * a)) * b - 2 * (-chi1 / chi2) * a * (1 + 3 * a + 6 * a ** 2) * c - (3 * (1 + a * (2 + a * (3 + 4 * a))) * b + c + a * (2 + a * (3 + 2 * (2 - 5 * a) * a)) * c) * p) * z ** 2) * numpy.log(chi1 / chi2)))

    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 5
@numba.njit(cache=True)
def _element5_nonzero(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (1 / (10080 * b ** 2)) * (b ** 2 * (1785 + z * ( - 562 + 79 * z) + 42 * b ** 2 * (15 + ( - 6 + z) * z) + 6 * b * (350 + z * ( - 124 + 19 * z))) -
8 * b * (1 + b) * (432 + z * ( - 129 + 17 * z) + 12 * b * (21 + ( - 7 + z) * z)) * numpy.log(1 + b) + 60 * (1 + b) ** 2 * (28 + ( - 8 + z) * z) * numpy.log(1 + b) ** 2))
    else:
        formula = (1 / (5292000 * a ** 5 * b ** 2)) * (a * b ** 2 * (245 * a ** 2 * (420 * p + a * ( - 600 + a ** 4 * (4030 - 2439 * p) - 90 * (39 + 40 * b) * p + 5 * a ** 2 * ( - 220 - 60 * b * ( - 2 * ( - 9 + p) + 9 * b * ( - 2 + p)) + 133 * p) +
a ** 3 * ( - 2150 + 300 * b * (30 - 17 * p) + 404 * p) + 20 * a * (495 + 34 * p + 90 * b * (6 + p)))) +
14 * a * ( - 10920 * p - 70 * a ** 3 * ( - 340 + 150 * b * ( - 6 + p) + 163 * p) + a ** 6 * ( - 85365 + 60096 * p) + 420 * a * (35 + 3 * (53 + 50 * b) * p) -
70 * a ** 2 * (1755 + 232 * p + 450 * b * (4 + p)) + 14 * a ** 5 * (1010 - 303 * p + 75 * b * ( - 170 + 117 * p)) +
7 * a ** 4 * (3325 - 952 * p + 750 * b * (4 - p + 6 * b * ( - 3 + 2 * p)))) * z +
(a ** 7 * (420672 - 322375 * p) + 59220 * p + 35 * a ** 3 * ( - 3248 + 1260 * b * ( - 5 + p) + 1371 * p) + a ** 6 * ( - 29694 + 4410 * b * (195 - 148 * p) + 11624 * p) -
210 * a * (364 + 3 * (481 + 420 * b) * p) + 14 * a ** 4 * ( - 5705 + 1941 * p + 525 * b * ( - 10 + 3 * p)) + 420 * a ** 2 * (1113 + 197 * p + 105 * b * (10 + 3 * p)) -
7 * a ** 5 * (6664 - 2441 * p + 210 * b * (25 - 9 * p + 75 * b * ( - 4 + 3 * p)))) * z ** 2) +
420 * ( - 210 * (-chi1 / chi2) ** 5 * b ** 2 * (5 * a ** 2 * (p + a * ( - 4 + 3 * p)) - 2 * a * (2 * p + a * ( - 5 + 6 * p + 3 * a * ( - 5 + 4 * p))) * z +
( - 2 * a * (1 + 3 * a + 6 * a ** 2) + p + a * (3 + 2 * a * (3 + 5 * a)) * p) * z ** 2) * numpy.log(chi1 / chi2) ** 2 + (-chi1 / chi2) ** 3 * b * numpy.log(chi1 / chi2) * (b * (35 * a ** 2 * ( - 10 * a * ( - 1 + a * (2 + 17 * a + 18 * b)) + ( - 7 + a * (11 + 60 * b + a * (59 + 117 * a + 120 * b))) * p) -
14 * a * ( - 5 * a * ( - 7 + a * (11 + 60 * b + a * (59 + 117 * a + 120 * b))) + 2 * ( - 13 + a * (17 + 75 * b + a * (77 + 150 * b + 3 * a * (49 + 74 * a + 75 * b)))) * p) * z +
( - 14 * a * ( - 13 + a * (17 + 75 * b + a * (77 + 150 * b + 3 * a * (49 + 74 * a + 75 * b)))) +
( - 141 + a * (159 + 669 * a + 1249 * a ** 2 + 1864 * a ** 3 + 2500 * a ** 4 + 630 * (1 + a * (2 + a * (3 + 4 * a))) * b)) * p) * z ** 2) +
420 * (-chi1 / chi2) * a * (1 + b) * (5 * a ** 2 * (p + a * ( - 4 + 3 * p)) - 2 * a * (2 * p + a * ( - 5 + 6 * p + 3 * a * ( - 5 + 4 * p))) * z +
( - 2 * a * (1 + 3 * a + 6 * a ** 2) + p + a * (3 + 2 * a * (3 + 5 * a)) * p) * z ** 2) * numpy.log(1 + b)) -
a ** 2 * (1 + b) * numpy.log(1 + b) * (b * ( - 420 * p * z ** 2 + 210 * a * z * (4 * z + p * (8 + z)) + a ** 6 * ( - 5950 + 4095 * p + 42 * (195 - 74 * z) * z + 4 * p * z * ( - 1554 + 625 * z)) +
70 * a ** 2 * ( - 6 * z * (10 + z) + p * ( - 30 + ( - 12 + z) * z)) + 35 * a ** 3 * (240 - 4 * ( - 15 + z) * z + p * (30 + ( - 8 + z) * z)) +
7 * a ** 5 * ( - 25 * ( - 64 + 35 * p + 70 * z) + 2 * z * (p * (594 - 224 * z) + 297 * z) + 30 * b * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z))) +
7 * a ** 4 * (50 * ( - 12 + p) - 20 * ( - 5 + p) * z + ( - 10 + 3 * p) * z ** 2 - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + p * (6 + z * ( - 8 + 3 * z))))) +
210 * a ** 4 * (1 + b) * ( - 20 * (3 + ( - 3 + a) * a) + 5 * (6 + a * ( - 8 + 3 * a)) * p + 2 * (30 + 5 * a * ( - 8 + 3 * a) - 20 * p + 6 * (5 - 2 * a) * a * p) * z +
(5 * ( - 4 + 3 * p) + 2 * a * (15 - 12 * p + a * ( - 6 + 5 * p))) * z ** 2) * numpy.log(1 + b))))

    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 6
@numba.njit(cache=True)
def _element6_nonzero(chi1, chi2, chi3, chi4, chi5, chi6, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    c = (chi6 - chi4) / (2 * chi2)
    d = chi4 * numpy.log(chi5 / chi4) / (chi5 - chi4) - chi6 * numpy.log(chi6 / chi5) / (chi6 - chi5)
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (1 / (5040 * b)) * (b * (2 * d * (432 + z * ( - 129 + 17 * z) + 12 * b * (21 + ( - 7 + z) * z)) + 3 * c * (350 + z * ( - 124 + 19 * z) + 14 * b * (15 + ( - 6 + z) * z))) -
6 * (1 + b) * (5 * d * (28 + ( - 8 + z) * z) + 8 * c * (21 + ( - 7 + z) * z)) * numpy.log(1 + b)))
    else:
        formula = (1 / (25200 * a ** 4 * b)) * (a * b * ( - 420 * (3 * c + d) * p * z ** 2 + 70 * a ** 2 * ( - 30 * (2 * c + d) * p - 6 * (5 * c * (4 + p) + 2 * d * (5 + p)) * z + (d * ( - 6 + p) + 3 * c * ( - 5 + p)) * z ** 2) +
210 * a * z * (8 * d * p + 10 * c * z + d * (4 + p) * z + c * p * (20 + 3 * z)) + a ** 6 * d * ( - 5950 + 4095 * p + 42 * (195 - 74 * z) * z + 4 * p * z * ( - 1554 + 625 * z)) +
35 * a ** 3 * (c * (60 * (6 + p) - 20 * ( - 6 + p) * z + ( - 10 + 3 * p) * z ** 2) + d * (240 - 4 * ( - 15 + z) * z + p * (30 + ( - 8 + z) * z))) +
7 * a ** 5 * (c * (1500 - 850 * p + 6 * p * (195 - 74 * z) * z + 5 * z * ( - 340 + 117 * z)) + d * ( - 25 * ( - 64 + 35 * p + 70 * z) + 2 * z * (p * (594 - 224 * z) + 297 * z) +
30 * b * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)))) +
7 * a ** 4 * (d * (50 * ( - 12 + p) - 20 * ( - 5 + p) * z + ( - 10 + 3 * p) * z ** 2 - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + p * (6 + z * ( - 8 + 3 * z)))) +
c * (100 * ( - 9 + p + 2 * z) + z * ( - 25 * z + p * ( - 50 + 9 * z)) - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + p * (6 + z * ( - 8 + 3 * z)))))) -
420 * ((-chi1 / chi2) ** 3 * b * (5 * a ** 2 * (6 * a * c - 4 * (-chi1 / chi2) * a * d - ((2 + 4 * a) * c + d + (2 - 3 * a) * a * d) * p) +
2 * a * ( - 10 * a * (1 + 2 * a) * c + 5 * (-chi1 / chi2) * a * (1 + 3 * a) * d + 5 * (1 + a * (2 + 3 * a)) * c * p - 2 * (-chi1 / chi2) * (1 + 3 * a + 6 * a ** 2) * d * p) * z +
(5 * a * (1 + a * (2 + 3 * a)) * c - 2 * (-chi1 / chi2) * a * (1 + 3 * a + 6 * a ** 2) * d - (3 * (1 + a * (2 + a * (3 + 4 * a))) * c + d + a * (2 + a * (3 + 2 * (2 - 5 * a) * a)) * d) * p) * z ** 2) * numpy.log(chi1 / chi2) + a ** 5 * (1 + b) * (d * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * p + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * p + 6 * a * ( - 5 + 2 * a) * p) * z +
(20 - 15 * p + 2 * a * ( - 15 + 6 * a + 12 * p - 5 * a * p)) * z ** 2) + c * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * p * (6 + z * ( - 8 + 3 * z)))) * numpy.log(1 + b)))

    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 7
@numba.njit(cache=True)
def _element7_nonzero(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    c = (chi5 - chi4) / (2 * chi2)
    d = chi4 * numpy.log(chi5 / chi4) / (chi5 - chi4) - 1
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (1 / (5040 * b)) * (b * (2 * d * (432 + z * ( - 129 + 17 * z) + 12 * b * (21 + ( - 7 + z) * z)) + 3 * c * (350 + z * ( - 124 + 19 * z) + 14 * b * (15 + ( - 6 + z) * z))) -
6 * (1 + b) * (5 * d * (28 + ( - 8 + z) * z) + 8 * c * (21 + ( - 7 + z) * z)) * numpy.log(1 + b)))
    else:
        formula = (1 / (25200 * a ** 4 * b)) * (a * b * ( - 420 * (3 * c + d) * p * z ** 2 + 70 * a ** 2 * ( - 30 * (2 * c + d) * p - 6 * (5 * c * (4 + p) + 2 * d * (5 + p)) * z + (d * ( - 6 + p) + 3 * c * ( - 5 + p)) * z ** 2) +
210 * a * z * (8 * d * p + 10 * c * z + d * (4 + p) * z + c * p * (20 + 3 * z)) + a ** 6 * d * ( - 5950 + 4095 * p + 42 * (195 - 74 * z) * z + 4 * p * z * ( - 1554 + 625 * z)) +
35 * a ** 3 * (c * (60 * (6 + p) - 20 * ( - 6 + p) * z + ( - 10 + 3 * p) * z ** 2) + d * (240 - 4 * ( - 15 + z) * z + p * (30 + ( - 8 + z) * z))) +
7 * a ** 5 * (c * (1500 - 850 * p + 6 * p * (195 - 74 * z) * z + 5 * z * ( - 340 + 117 * z)) + d * ( - 25 * ( - 64 + 35 * p + 70 * z) + 2 * z * (p * (594 - 224 * z) + 297 * z) +
30 * b * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)))) +
7 * a ** 4 * (d * (50 * ( - 12 + p) - 20 * ( - 5 + p) * z + ( - 10 + 3 * p) * z ** 2 - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + p * (6 + z * ( - 8 + 3 * z)))) +
c * (100 * ( - 9 + p + 2 * z) + z * ( - 25 * z + p * ( - 50 + 9 * z)) - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + p * (6 + z * ( - 8 + 3 * z)))))) -
420 * ((-chi1 / chi2) ** 3 * b * (5 * a ** 2 * (6 * a * c - 4 * (-chi1 / chi2) * a * d - ((2 + 4 * a) * c + d + (2 - 3 * a) * a * d) * p) +
2 * a * ( - 10 * a * (1 + 2 * a) * c + 5 * (-chi1 / chi2) * a * (1 + 3 * a) * d + 5 * (1 + a * (2 + 3 * a)) * c * p - 2 * (-chi1 / chi2) * (1 + 3 * a + 6 * a ** 2) * d * p) * z +
(5 * a * (1 + a * (2 + 3 * a)) * c - 2 * (-chi1 / chi2) * a * (1 + 3 * a + 6 * a ** 2) * d - (3 * (1 + a * (2 + a * (3 + 4 * a))) * c + d + a * (2 + a * (3 + 2 * (2 - 5 * a) * a)) * d) * p) * z ** 2) * numpy.log(chi1 / chi2) + a ** 5 * (1 + b) * (d * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * p + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * p + 6 * a * ( - 5 + 2 * a) * p) * z +
(20 - 15 * p + 2 * a * ( - 15 + 6 * a + 12 * p - 5 * a * p)) * z ** 2) + c * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * p * (6 + z * ( - 8 + 3 * z)))) * numpy.log(1 + b)))

    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 8
@numba.njit(cache=True)
def _element8_nonzero(chi1, chi2, chi3, chi4, chi5, chi6, chi7, chi8, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5 - chi3) / (2 * chi2)
    c = chi3 * numpy.log(chi4 / chi3) / (chi4 - chi3) - chi5 * numpy.log(chi5 / chi4) / (chi5 - chi4)
    d = (chi8 - chi6) / (2 * chi2)
    e = chi6 * numpy.log(chi7 / chi6) / (chi7 - chi6) - chi8 * numpy.log(chi8 / chi7) / (chi8 - chi7)
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (1 / 840) * (5 * c * e * (28 + ( - 8 + z) * z) + 8 * c * d * (21 + ( - 7 + z) * z) + 8 * b * e * (21 + ( - 7 + z) * z) + 14 * b * d * (15 + ( - 6 + z) * z)))
    else:
        formula = (1 / 60) * a * (b * ( - 30 * e * ( - 2 + p + 2 * z) + 5 * e * z * (p * (8 - 3 * z) + 4 * z) + 20 * d * (3 + ( - 3 + z) * z) + a * e * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * d * p * (6 + z * ( - 8 + 3 * z))) + c * (e * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * p + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * p + 6 * a * ( - 5 + 2 * a) * p) * z +
(20 - 15 * p + 2 * a * ( - 15 + 6 * a + 12 * p - 5 * a * p)) * z ** 2) + d * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * p * (6 + z * ( - 8 + 3 * z)))))

    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 9
@numba.njit(cache=True)
def _element9_nonzero(chi1, chi2, chi3, chi4, chi5, chi6, chi7, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5 - chi3) / (2 * chi2)
    c = chi3 * numpy.log(chi4 / chi3) / (chi4 - chi3) - chi5 * numpy.log(chi5 / chi4) / (chi5 - chi4)
    d = (chi7 - chi6) / (2 * chi2)
    e = chi6 * numpy.log(chi7 / chi6) / (chi7 - chi6) - 1
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (1 / 840) * (5 * c * e * (28 + ( - 8 + z) * z) + 8 * c * d * (21 + ( - 7 + z) * z) + 8 * b * e * (21 + ( - 7 + z) * z) + 14 * b * d * (15 + ( - 6 + z) * z)))
    else:
        formula = (1 / 60) * a * (b * ( - 30 * e * ( - 2 + p + 2 * z) + 5 * e * z * (p * (8 - 3 * z) + 4 * z) + 20 * d * (3 + ( - 3 + z) * z) + a * e * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * d * p * (6 + z * ( - 8 + 3 * z))) + c * (e * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * p + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * p + 6 * a * ( - 5 + 2 * a) * p) * z +
(20 - 15 * p + 2 * a * ( - 15 + 6 * a + 12 * p - 5 * a * p)) * z ** 2) + d * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * p * (6 + z * ( - 8 + 3 * z)))))

    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 10
@numba.njit(cache=True)
def _element10_nonzero(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi4 - chi3) / (2 * chi2)
    c = chi3 * numpy.log(chi4 / chi3) / (chi4 - chi3) - 1
    p = 1 - power1 / numpy.where(power2 == 0, 1.0, power2)
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if chi1 == 0.0:
        formula = numpy.full_like(p, (1 / 840) * (5 * c ** 2 * (28 + ( - 8 + z) * z) + 16 * b * c * (21 + ( - 7 + z) * z) + 14 * b ** 2 * (15 + ( - 6 + z) * z)))
    else:
        formula = (1 / 60) * a * (c ** 2 * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * p + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * p + 6 * a * ( - 5 + 2 * a) * p) * z +
(20 - 15 * p + 2 * a * ( - 15 + 6 * a + 12 * p - 5 * a * p)) * z ** 2) + 2 * b * c * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * p * (6 + z * ( - 8 + 3 * z))) - 5 * b ** 2 * ( - 4 * (3 + ( - 3 + z) * z) + p * (6 + z * ( - 8 + 3 * z))))

    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

@numba.njit(cache=True)
def element2(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element2_nonzero(chi1, chi2, chi3, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element2_nonzero(chi1, chi2, chi3, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element2_nonzero(chi1, chi2, chi3, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

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
def element4(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element4_nonzero(chi1, chi2, chi3, chi4, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element4_nonzero(chi1, chi2, chi3, chi4, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element4_nonzero(chi1, chi2, chi3, chi4, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
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
def element6(chi1, chi2, chi3, chi4, chi5, chi6, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element6_nonzero(chi1, chi2, chi3, chi4, chi5, chi6, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element6_nonzero(chi1, chi2, chi3, chi4, chi5, chi6, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element6_nonzero(chi1, chi2, chi3, chi4, chi5, chi6, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element7(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element7_nonzero(chi1, chi2, chi3, chi4, chi5, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element7_nonzero(chi1, chi2, chi3, chi4, chi5, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element7_nonzero(chi1, chi2, chi3, chi4, chi5, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element8(chi1, chi2, chi3, chi4, chi5, chi6, chi7, chi8, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element8_nonzero(chi1, chi2, chi3, chi4, chi5, chi6, chi7, chi8, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element8_nonzero(chi1, chi2, chi3, chi4, chi5, chi6, chi7, chi8, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element8_nonzero(chi1, chi2, chi3, chi4, chi5, chi6, chi7, chi8, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element9(chi1, chi2, chi3, chi4, chi5, chi6, chi7, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element9_nonzero(chi1, chi2, chi3, chi4, chi5, chi6, chi7, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element9_nonzero(chi1, chi2, chi3, chi4, chi5, chi6, chi7, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element9_nonzero(chi1, chi2, chi3, chi4, chi5, chi6, chi7, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element10(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    """Preserve endpoint linearity when an ordinary right power is zero."""
    zero = power2 == 0
    safe_power2 = numpy.where(zero, 1.0, power2)
    regular = _element10_nonzero(chi1, chi2, chi3, chi4, power1, safe_power2, redshift1, redshift2)
    if numpy.any(zero):
        left = _element10_nonzero(chi1, chi2, chi3, chi4, numpy.ones_like(power1), numpy.ones_like(power1), redshift1, redshift2) - _element10_nonzero(chi1, chi2, chi3, chi4, numpy.zeros_like(power1), numpy.ones_like(power2), redshift1, redshift2)
        if chi1 == 0.0:
            left = numpy.zeros_like(left)
        return numpy.where(zero, power1*left, regular)
    return regular

@numba.njit(cache=True)
def element11(chi1, chi2, power1, power2, redshift1, redshift2):
    """New final-interval SS B11; cubic observer handled separately."""
    return local.ss11(chi1, chi2, power1, power2, redshift1, redshift2)

@numba.njit(cache=True)
def element12(chi1, chi2, power1, power2, redshift1, redshift2):
    """New final-interval SS B12; cubic observer handled separately."""
    return local.ss12(chi1, chi2, power1, power2, redshift1, redshift2)

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
        # Element 2
        if n + 1 < grid_size:
            element = element2(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n, n + 1, :] += element
            coefficients[n + 1, n, :] += element
        # Element 3
        if n + 1 < grid_size:
            for k in numba.prange(n + 2, grid_size):
                element = element3(chi_grid[n], chi_grid[n + 1], chi_grid[k - 1], chi_grid[k], chi_grid[k + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
                coefficients[n, k, :] += element
                coefficients[k, n, :] += element
        # Element 4. Below the final interval the whole node-N source hat lies
        # above the evaluation point. On the final interval the evaluation
        # point is inside both terminal hats, so the cross term becomes the
        # product of the two truncated source integrals F_N and H_N.
        if n + 1 < grid_size:
            element = element4(chi_grid[n], chi_grid[n + 1], chi_grid[grid_size - 1], chi_grid[grid_size], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n, grid_size, :] += element
            coefficients[grid_size, n, :] += element
        else:
            element = element11(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n, grid_size, :] += element
            coefficients[grid_size, n, :] += element
        # Element 5
        if n + 1 < grid_size:
            element = element5(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n + 1, n + 1, :] += element
        # Element 6
        if n + 1 < grid_size:
            for k in numba.prange(n + 2, grid_size):
                element = element6(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], chi_grid[k - 1], chi_grid[k], chi_grid[k + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
                coefficients[n + 1, k, :] += element
                coefficients[k, n + 1, :] += element
        # Element 7
        if n + 1 < grid_size:
            element = element7(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], chi_grid[grid_size - 1], chi_grid[grid_size], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[n + 1, grid_size, :] += element
            coefficients[grid_size, n + 1, :] += element
        # Element 8
        if n + 1 < grid_size:
            for i in numba.prange(n + 2, grid_size):
                for j in range(n + 2, grid_size):
                    element = element8(chi_grid[n], chi_grid[n + 1], chi_grid[i - 1], chi_grid[i], chi_grid[i + 1], chi_grid[j - 1], chi_grid[j], chi_grid[j + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
                    coefficients[i, j, :] += element
        # Element 9
        if n + 1 < grid_size:
            for k in numba.prange(n + 2, grid_size):
                element = element9(chi_grid[n], chi_grid[n + 1], chi_grid[k - 1], chi_grid[k], chi_grid[k + 1], chi_grid[grid_size - 1], chi_grid[grid_size], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
                coefficients[k, grid_size, :] += element
                coefficients[grid_size, k, :] += element
        # Element 10. The same terminal distinction applies to the node-N
        # diagonal: on the final interval it is the square of H_N(x).
        if n + 1 < grid_size:
            element = element10(chi_grid[n], chi_grid[n + 1], chi_grid[grid_size - 1], chi_grid[grid_size], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[grid_size, grid_size, :] += element
        else:
            element = element12(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
            coefficients[grid_size, grid_size, :] += element
    return coefficients

# Spectra
def spectra(factor, phi_a_grid, phi_b_grid, chi_grid, power_grid, redshift_grid):
    coefficients = coefficient(chi_grid, power_grid, redshift_grid)
    return factor * numpy.einsum('mi,nj,ijl->mnl', phi_a_grid, phi_b_grid, coefficients, dtype=numpy.float64)
