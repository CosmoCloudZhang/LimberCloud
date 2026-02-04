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
        return jnp.full_like(power2, (14952 + z * ( - 20184 + 7087 * z)) / 177811200)
    #else:
    def false_branch(_):
        formula =(1 / (5292000 * a**5)) * (a * (245 * a**2 * (10 * a * ( - 60 + a * ( - 750 + a * (1420 + 27 * a * ( - 25 + 4 * a)))) + (420 + a * (2010 + a * (1940 - 9 * a * (1005 + 4 * a * ( - 144 + 25 * a))))) * y) + 
14 * a * ( - 35 * a * ( - 420 + a * ( - 2010 + a * ( - 1940 + 9 * a * (1005 + 4 * a * ( - 144 + 25 * a))))) + 
2 * ( - 5460 + a * ( - 15330 + a * ( - 14420 + 3 * a * ( - 4305 + 4 * a * (9534 + 125 * a * ( - 49 + 9 * a)))))) * y) * z + 
(14 * a * ( - 5460 + a * ( - 15330 + a * ( - 14420 + 3 * a * ( - 4305 + 4 * a * (9534 + 125 * a * ( - 49 + 9 * a)))))) + 
(59220 + a * (117810 + a * (107940 + a * (95655 + a * (85344 - 125 * a * (9968 + 27 * a * ( - 256 + 49 * a))))))) * y) * z**2) + 
420 * ( - 1 + a)**2 * jnp.log(1 - a) * (35 * a**2 * (7 * y + a * ( - 10 + 74 * y + a * ( - 260 + a * (90 - 72 * y) + 171 * y))) + 
14 * a * (5 * a * (7 + a * (74 + 9 * (19 - 8 * a) * a)) + 2 * ( - 13 + a * ( - 86 + 3 * a * ( - 63 + 2 * a * ( - 52 + 25 * a)))) * y) * z + 
(14 * a * ( - 13 + a * ( - 86 + 3 * a * ( - 63 + 2 * a * ( - 52 + 25 * a)))) + (141 + a * (702 + a * (1473 + 8 * a * (298 + 25 * (17 - 9 * a) * a)))) * y) * z**2 - 
210 * ( - 1 + a) * (5 * a**2 * (y + a * ( - 4 + 3 * y)) - 2 * a * (2 * y + a * ( - 5 + 6 * y + 3 * a * ( - 5 + 4 * y))) * z + 
( - 2 * a * (1 + 3 * a + 6 * a**2) + y + a * (3 + 2 * a * (3 + 5 * a)) * y) * z**2) * jnp.log(1 - a)))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2**3 * power2 * (1 + redshift2)**2 * formula
    return coefficient


@jax.jit
def element2(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (b * (6642 + z * ( - 6669 + 1867 * z) + 18 * b * (294 + z * ( - 308 + 89 * z))) - 45 * (1 + b) * (112 + z * ( - 104 + 27 * z)) * jnp.log(1 + b)) / (6350400 * b))
    #else:
    def false_branch(_):
        formula = - ((1 / (4 * a**5 * b)) * ((1 / 15) * ( - 1 + a)**4 * b * (5 * a**2 * (y + a * ( - 4 + 3 * y)) - 2 * a * (2 * y + a * ( - 5 + 6 * y + 3 * a * ( - 5 + 4 * y))) * z + 
( - 2 * a * (1 + 3 * a + 6 * a**2) + y + a * (3 + 2 * a * (3 + 5 * a)) * y) * z**2) * jnp.log(1 - a)**2 + 
(1 / 3150) * (( - 1 + a)**2 * jnp.log(1 - a) * (b * (35 * a**2 * ( - 5 * a * (2 + a * (23 + a * ( - 52 + 9 * a) - 18 * b)) + (7 + a * (28 + a * (19 + 36 * ( - 5 + a) * a - 60 * b) - 30 * b)) * y) + 14 * a * (5 * a * (7 + a * (28 + a * (19 + 36 * ( - 5 + a) * a - 60 * b) - 30 * b)) - 26 * y + a * ( - 56 + 2 * a * ( - 13 + 3 * a * (4 + (114 - 25 * a) * a)) + 
75 * (1 + a * (2 + 3 * a)) * b) * y) * z + (7 * a * ( - 26 - 2 * a * (28 + a * (13 + 3 * a * ( - 4 + a * ( - 114 + 25 * a)))) + 75 * a * (1 + a * (2 + 3 * a)) * b) + 
(141 + a * (201 + 51 * a - 169 * a**2 - 424 * a**3 - 3850 * a**4 + 900 * a**5 - 315 * (1 + a * (2 + a * (3 + 4 * a))) * b)) * y) * z**2) - 
210 * ( - 1 + a) * a * (1 + b) * (5 * a**2 * (y + a * ( - 4 + 3 * y)) - 2 * a * (2 * y + a * ( - 5 + 6 * y + 3 * a * ( - 5 + 4 * y))) * z + 
( - 2 * a * (1 + 3 * a + 6 * a**2) + y + a * (3 + 2 * a * (3 + 5 * a)) * y) * z**2) * jnp.log(1 + b))) - 
(1 / 1323000) * (a * (b * (245 * a**2 * ( - 420 * y + 20 * a**2 * ( - 60 + 45 * b * ( - 6 + y) + 62 * y) + 9 * a**5 * ( - 195 + 148 * y) + 5 * a**3 * ( - 830 + 60 * b * (27 - 16 * y) + 193 * y) + 
150 * a * (4 + (5 + 12 * b) * y) + 2 * a**4 * (3725 - 2322 * y + 225 * b * ( - 4 + 3 * y))) - 
14 * a * (35 * a * (420 + a * ( - 150 * (5 + 12 * b) + a * ( - 20 * (62 + 45 * b) + a * ( - 965 + 4800 * b - 18 * a * ( - 258 + 74 * a + 75 * b))))) + 
( - 10920 + 2 * a * (9030 + a * (10360 + a * (7805 + 6 * a * (1008 + a * ( - 9688 + 3125 * a))))) + 525 * a * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b) * y) * z + (7 * a * (10920 - 2 * a * (9030 + a * (10360 + a * (7805 + 6 * a * (1008 + a * ( - 9688 + 3125 * a))))) - 
525 * a * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b) + ( - 59220 + a * (92610 + 85470 * a + 62685 * a**2 + 48111 * a**3 + 38584 * a**4 - 629500 * a**5 + 219375 * a**6 + 2205 * (60 + a * (30 + a * (20 + a * (15 + 4 * a * ( - 72 + 25 * a))))) * b)) * y) * z**2) + 
210 * a * (1 + b) * ( - 210 * a * (y * ( - 8 + z) - 4 * z) * z - 420 * y * z**2 - 35 * a**3 * (30 * ( - 8 + y) + 4 * (15 - 4 * y) * z + ( - 8 + 3 * y) * z**2) + 
14 * a**4 * ( - 1500 + 850 * y + 5 * (340 - 117 * z) * z + 6 * y * z * ( - 195 + 74 * z)) - 140 * a**2 * ( - 3 * ( - 10 + z) * z + y * (15 + ( - 6 + z) * z)) + 
30 * a**6 * (4 * y * (21 + 5 * z * ( - 7 + 3 * z)) - 7 * (15 + 2 * z * ( - 12 + 5 * z))) - 7 * a**5 * ( - 2200 + 18 * (175 - 68 * z) * z + y * (1575 + 8 * z * ( - 306 + 125 * z)))) * jnp.log(1 + b)))))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2**3 * power2 * (1 + redshift2)**2 * formula
    
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
        return jnp.full_like(power2, (5 * c * (112 + z * ( - 104 + 27 * z)) + 4 * b * (294 + z * ( - 308 + 89 * z))) / 705600)
    #else:
    def false_branch(_):
        formula = - ((1 / (2 * a**4)) * ((1 / 12600) * (a * (35 * a**2 * (10 * a * (6 * (6 + a * ( - 9 + 2 * a)) * b + (24 + a * ( - 60 + (44 - 9 * a) * a)) * c) - 
(10 * (12 + a * (6 + a * ( - 32 + 9 * a))) * b + (60 + a * (30 + a * ( - 340 + 9 * (35 - 8 * a) * a))) * c) * y) - 
14 * a * (50 * a * (12 + a * (6 + a * ( - 32 + 9 * a))) * b - 5 * a * ( - 60 + a * ( - 30 + a * (340 + 9 * a * ( - 35 + 8 * a)))) * c - 
(5 * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 2 * (60 + a * (30 + a * (20 - 3 * a * (195 - 204 * a + 50 * a**2)))) * c) * y) * z + 
(7 * a * (5 * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 2 * (60 + a * (30 + a * (20 - 3 * a * (195 - 204 * a + 50 * a**2)))) * c) - 
(21 * (60 + a * (30 + a * (20 + a * (15 + 4 * a * ( - 72 + 25 * a))))) * b + (420 + a * (210 + a * (140 + a * (105 - 8 * a * (777 + 25 * a * ( - 35 + 9 * a)))))) * c) * y) * z**2)) + (1 / 30) * ( - 1 + a)**2 * (5 * a**2 * (6 * a * b - 4 * ( - 1 + a) * a * c - ((2 + 4 * a) * b + c + (2 - 3 * a) * a * c) * y) + 
2 * a * ( - 10 * a * (1 + 2 * a) * b + 5 * ( - 1 + a) * a * (1 + 3 * a) * c + 5 * (1 + a * (2 + 3 * a)) * b * y - 2 * ( - 1 + a) * (1 + 3 * a + 6 * a**2) * c * y) * z + 
(5 * a * (1 + a * (2 + 3 * a)) * b - 2 * ( - 1 + a) * a * (1 + 3 * a + 6 * a**2) * c - (3 * (1 + a * (2 + a * (3 + 4 * a))) * b + c + a * (2 + a * (3 + 2 * (2 - 5 * a) * a)) * c) * y) * z**2) * jnp.log(1 - a)))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2**3 * power2 * (1 + redshift2)**2 * formula
    return coefficient


@jax.jit
def element4(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi4-chi3)/(2*chi2)
    c = chi3*jnp.log(chi4/chi3)/(chi4-chi3)-1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (5 * c * (112 + z * ( - 104 + 27 * z)) + 4 * b * (294 + z * ( - 308 + 89 * z))) / 705600)
    #else:
    def false_branch(_):
        formula = - ((1 / (2 * a**4)) * ((1 / 12600) * (a * (35 * a**2 * (10 * a * (6 * (6 + a * ( - 9 + 2 * a)) * b + (24 + a * ( - 60 + (44 - 9 * a) * a)) * c) - 
(10 * (12 + a * (6 + a * ( - 32 + 9 * a))) * b + (60 + a * (30 + a * ( - 340 + 9 * (35 - 8 * a) * a))) * c) * y) - 
14 * a * (50 * a * (12 + a * (6 + a * ( - 32 + 9 * a))) * b - 5 * a * ( - 60 + a * ( - 30 + a * (340 + 9 * a * ( - 35 + 8 * a)))) * c - 
(5 * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 2 * (60 + a * (30 + a * (20 - 3 * a * (195 - 204 * a + 50 * a**2)))) * c) * y) * z + 
(7 * a * (5 * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 2 * (60 + a * (30 + a * (20 - 3 * a * (195 - 204 * a + 50 * a**2)))) * c) - 
(21 * (60 + a * (30 + a * (20 + a * (15 + 4 * a * ( - 72 + 25 * a))))) * b + (420 + a * (210 + a * (140 + a * (105 - 8 * a * (777 + 25 * a * ( - 35 + 9 * a)))))) * c) * y) * z**2)) + (1 / 30) * ( - 1 + a)**2 * (5 * a**2 * (6 * a * b - 4 * ( - 1 + a) * a * c - ((2 + 4 * a) * b + c + (2 - 3 * a) * a * c) * y) + 
2 * a * ( - 10 * a * (1 + 2 * a) * b + 5 * ( - 1 + a) * a * (1 + 3 * a) * c + 5 * (1 + a * (2 + 3 * a)) * b * y - 2 * ( - 1 + a) * (1 + 3 * a + 6 * a**2) * c * y) * z + 
(5 * a * (1 + a * (2 + 3 * a)) * b - 2 * ( - 1 + a) * a * (1 + 3 * a + 6 * a**2) * c - (3 * (1 + a * (2 + a * (3 + 4 * a))) * b + c + a * (2 + a * (3 + 2 * (2 - 5 * a) * a)) * c) * y) * z**2) * jnp.log(1 - a)))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2**3 * power2 * (1 + redshift2)**2 * formula
    return coefficient


@jax.jit
def element5(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (1 / (10080 * b**2)) * (b**2 * (1785 + z * ( - 562 + 79 * z) + 42 * b**2 * (15 + ( - 6 + z) * z) + 6 * b * (350 + z * ( - 124 + 19 * z))) - 8 * b * (1 + b) * (432 + z * ( - 129 + 17 * z) + 12 * b * (21 + ( - 7 + z) * z)) * jnp.log(1 + b) + 60 * (1 + b)**2 * (28 + ( - 8 + z) * z) * jnp.log(1 + b)**2))
    #else:
    def false_branch(_):
        formula =  (1 / (5292000 * a**5 * b**2)) * (a * b**2 * (245 * a**2 * (420 * y + a * ( - 600 + a**4 * (4030 - 2439 * y) - 90 * (39 + 40 * b) * y + 5 * a**2 * ( - 220 - 60 * b * ( - 2 * ( - 9 + y) + 9 * b * ( - 2 + y)) + 133 * y) + 
a**3 * ( - 2150 + 300 * b * (30 - 17 * y) + 404 * y) + 20 * a * (495 + 34 * y + 90 * b * (6 + y)))) + 
14 * a * ( - 10920 * y - 70 * a**3 * ( - 340 + 150 * b * ( - 6 + y) + 163 * y) + a**6 * ( - 85365 + 60096 * y) + 420 * a * (35 + 3 * (53 + 50 * b) * y) - 
70 * a**2 * (1755 + 232 * y + 450 * b * (4 + y)) + 14 * a**5 * (1010 - 303 * y + 75 * b * ( - 170 + 117 * y)) + 
7 * a**4 * (3325 - 952 * y + 750 * b * (4 - y + 6 * b * ( - 3 + 2 * y)))) * z + 
(a**7 * (420672 - 322375 * y) + 59220 * y + 35 * a**3 * ( - 3248 + 1260 * b * ( - 5 + y) + 1371 * y) + a**6 * ( - 29694 + 4410 * b * (195 - 148 * y) + 11624 * y) - 
210 * a * (364 + 3 * (481 + 420 * b) * y) + 14 * a**4 * ( - 5705 + 1941 * y + 525 * b * ( - 10 + 3 * y)) + 420 * a**2 * (1113 + 197 * y + 105 * b * (10 + 3 * y)) - 
7 * a**5 * (6664 - 2441 * y + 210 * b * (25 - 9 * y + 75 * b * ( - 4 + 3 * y)))) * z**2) + 
420 * ( - 210 * ( - 1 + a)**5 * b**2 * (5 * a**2 * (y + a * ( - 4 + 3 * y)) - 2 * a * (2 * y + a * ( - 5 + 6 * y + 3 * a * ( - 5 + 4 * y))) * z + 
( - 2 * a * (1 + 3 * a + 6 * a**2) + y + a * (3 + 2 * a * (3 + 5 * a)) * y) * z**2) * jnp.log(1 - a)**2 + ( - 1 + a)**3 * b * jnp.log(1 - a) * (b * (35 * a**2 * ( - 10 * a * ( - 1 + a * (2 + 17 * a + 18 * b)) + ( - 7 + a * (11 + 60 * b + a * (59 + 117 * a + 120 * b))) * y) - 
14 * a * ( - 5 * a * ( - 7 + a * (11 + 60 * b + a * (59 + 117 * a + 120 * b))) + 2 * ( - 13 + a * (17 + 75 * b + a * (77 + 150 * b + 3 * a * (49 + 74 * a + 75 * b)))) * y) * z + 
( - 14 * a * ( - 13 + a * (17 + 75 * b + a * (77 + 150 * b + 3 * a * (49 + 74 * a + 75 * b)))) + 
( - 141 + a * (159 + 669 * a + 1249 * a**2 + 1864 * a**3 + 2500 * a**4 + 630 * (1 + a * (2 + a * (3 + 4 * a))) * b)) * y) * z**2) + 
420 * ( - 1 + a) * a * (1 + b) * (5 * a**2 * (y + a * ( - 4 + 3 * y)) - 2 * a * (2 * y + a * ( - 5 + 6 * y + 3 * a * ( - 5 + 4 * y))) * z + 
( - 2 * a * (1 + 3 * a + 6 * a**2) + y + a * (3 + 2 * a * (3 + 5 * a)) * y) * z**2) * jnp.log(1 + b)) - 
a**2 * (1 + b) * jnp.log(1 + b) * (b * ( - 420 * y * z**2 + 210 * a * z * (4 * z + y * (8 + z)) + a**6 * ( - 5950 + 4095 * y + 42 * (195 - 74 * z) * z + 4 * y * z * ( - 1554 + 625 * z)) + 
70 * a**2 * ( - 6 * z * (10 + z) + y * ( - 30 + ( - 12 + z) * z)) + 35 * a**3 * (240 - 4 * ( - 15 + z) * z + y * (30 + ( - 8 + z) * z)) + 
7 * a**5 * ( - 25 * ( - 64 + 35 * y + 70 * z) + 2 * z * (y * (594 - 224 * z) + 297 * z) + 30 * b * ( - 30 + 20 * y + 5 * (8 - 3 * z) * z + 6 * y * z * ( - 5 + 2 * z))) + 
7 * a**4 * (50 * ( - 12 + y) - 20 * ( - 5 + y) * z + ( - 10 + 3 * y) * z**2 - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + y * (6 + z * ( - 8 + 3 * z))))) + 
210 * a**4 * (1 + b) * ( - 20 * (3 + ( - 3 + a) * a) + 5 * (6 + a * ( - 8 + 3 * a)) * y + 2 * (30 + 5 * a * ( - 8 + 3 * a) - 20 * y + 6 * (5 - 2 * a) * a * y) * z + 
(5 * ( - 4 + 3 * y) + 2 * a * (15 - 12 * y + a * ( - 6 + 5 * y))) * z**2) * jnp.log(1 + b))))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2**3 * power2 * (1 + redshift2)**2 * formula
    return coefficient


@jax.jit
def element6(chi1, chi2, chi3, chi4, chi5, chi6, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    c = (chi6-chi4)/(2*chi2)
    d = chi4*jnp.log(chi5/chi4)/(chi5-chi4)-chi6*jnp.log(chi6/chi5)/(chi6-chi5)
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (1 / (5040 * b)) * (b * (2 * d * (432 + z * ( - 129 + 17 * z) + 12 * b * (21 + ( - 7 + z) * z)) + 3 * c * (350 + z * ( - 124 + 19 * z) + 14 * b * (15 + ( - 6 + z) * z))) - 6 * (1 + b) * (5 * d * (28 + ( - 8 + z) * z) + 8 * c * (21 + ( - 7 + z) * z)) * jnp.log(1 + b)))
    #else:
    def false_branch(_):
        formula = (1 / (25200 * a**4 * b)) * (a * b * ( - 420 * (3 * c + d) * y * z**2 + 70 * a**2 * ( - 30 * (2 * c + d) * y - 6 * (5 * c * (4 + y) + 2 * d * (5 + y)) * z + (d * ( - 6 + y) + 3 * c * ( - 5 + y)) * z**2) + 
210 * a * z * (8 * d * y + 10 * c * z + d * (4 + y) * z + c * y * (20 + 3 * z)) + a**6 * d * ( - 5950 + 4095 * y + 42 * (195 - 74 * z) * z + 4 * y * z * ( - 1554 + 625 * z)) + 
35 * a**3 * (c * (60 * (6 + y) - 20 * ( - 6 + y) * z + ( - 10 + 3 * y) * z**2) + d * (240 - 4 * ( - 15 + z) * z + y * (30 + ( - 8 + z) * z))) + 
7 * a**5 * (c * (1500 - 850 * y + 6 * y * (195 - 74 * z) * z + 5 * z * ( - 340 + 117 * z)) + d * ( - 25 * ( - 64 + 35 * y + 70 * z) + 2 * z * (y * (594 - 224 * z) + 297 * z) + 
30 * b * ( - 30 + 20 * y + 5 * (8 - 3 * z) * z + 6 * y * z * ( - 5 + 2 * z)))) + 
7 * a**4 * (d * (50 * ( - 12 + y) - 20 * ( - 5 + y) * z + ( - 10 + 3 * y) * z**2 - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + y * (6 + z * ( - 8 + 3 * z)))) + 
c * (100 * ( - 9 + y + 2 * z) + z * ( - 25 * z + y * ( - 50 + 9 * z)) - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + y * (6 + z * ( - 8 + 3 * z)))))) - 
420 * (( - 1 + a)**3 * b * (5 * a**2 * (6 * a * c - 4 * ( - 1 + a) * a * d - ((2 + 4 * a) * c + d + (2 - 3 * a) * a * d) * y) + 
2 * a * ( - 10 * a * (1 + 2 * a) * c + 5 * ( - 1 + a) * a * (1 + 3 * a) * d + 5 * (1 + a * (2 + 3 * a)) * c * y - 2 * ( - 1 + a) * (1 + 3 * a + 6 * a**2) * d * y) * z + 
(5 * a * (1 + a * (2 + 3 * a)) * c - 2 * ( - 1 + a) * a * (1 + 3 * a + 6 * a**2) * d - (3 * (1 + a * (2 + a * (3 + 4 * a))) * c + d + a * (2 + a * (3 + 2 * (2 - 5 * a) * a)) * d) * y) * z**2) * jnp.log(1 - a) + a**5 * (1 + b) * (d * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * y + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * y + 6 * a * ( - 5 + 2 * a) * y) * z + 
(20 - 15 * y + 2 * a * ( - 15 + 6 * a + 12 * y - 5 * a * y)) * z**2) + c * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * y + 5 * (8 - 3 * z) * z + 6 * y * z * ( - 5 + 2 * z)) - 
5 * y * (6 + z * ( - 8 + 3 * z)))) * jnp.log(1 + b)))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2**3 * power2 * (1 + redshift2)**2 * formula
    return coefficient


@jax.jit
def element7(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    c = (chi5-chi4)/(2*chi2)
    d = chi4*jnp.log(chi5/chi4)/(chi5-chi4)-1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (1 / (5040 * b)) * (b * (2 * d * (432 + z * ( - 129 + 17 * z) + 12 * b * (21 + ( - 7 + z) * z)) + 3 * c * (350 + z * ( - 124 + 19 * z) + 14 * b * (15 + ( - 6 + z) * z))) - 
6 * (1 + b) * (5 * d * (28 + ( - 8 + z) * z) + 8 * c * (21 + ( - 7 + z) * z)) * jnp.log(1 + b)))
    #else:
    def false_branch(_):
        formula =  (1 / (25200 * a**4 * b)) * (a * b * ( - 420 * (3 * c + d) * y * z**2 + 70 * a**2 * ( - 30 * (2 * c + d) * y - 6 * (5 * c * (4 + y) + 2 * d * (5 + y)) * z + (d * ( - 6 + y) + 3 * c * ( - 5 + y)) * z**2) + 
210 * a * z * (8 * d * y + 10 * c * z + d * (4 + y) * z + c * y * (20 + 3 * z)) + a**6 * d * ( - 5950 + 4095 * y + 42 * (195 - 74 * z) * z + 4 * y * z * ( - 1554 + 625 * z)) + 
35 * a**3 * (c * (60 * (6 + y) - 20 * ( - 6 + y) * z + ( - 10 + 3 * y) * z**2) + d * (240 - 4 * ( - 15 + z) * z + y * (30 + ( - 8 + z) * z))) + 
7 * a**5 * (c * (1500 - 850 * y + 6 * y * (195 - 74 * z) * z + 5 * z * ( - 340 + 117 * z)) + d * ( - 25 * ( - 64 + 35 * y + 70 * z) + 2 * z * (y * (594 - 224 * z) + 297 * z) + 
30 * b * ( - 30 + 20 * y + 5 * (8 - 3 * z) * z + 6 * y * z * ( - 5 + 2 * z)))) + 
7 * a**4 * (d * (50 * ( - 12 + y) - 20 * ( - 5 + y) * z + ( - 10 + 3 * y) * z**2 - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + y * (6 + z * ( - 8 + 3 * z)))) + 
c * (100 * ( - 9 + y + 2 * z) + z * ( - 25 * z + y * ( - 50 + 9 * z)) - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + y * (6 + z * ( - 8 + 3 * z)))))) - 
420 * (( - 1 + a)**3 * b * (5 * a**2 * (6 * a * c - 4 * ( - 1 + a) * a * d - ((2 + 4 * a) * c + d + (2 - 3 * a) * a * d) * y) + 
2 * a * ( - 10 * a * (1 + 2 * a) * c + 5 * ( - 1 + a) * a * (1 + 3 * a) * d + 5 * (1 + a * (2 + 3 * a)) * c * y - 2 * ( - 1 + a) * (1 + 3 * a + 6 * a**2) * d * y) * z + 
(5 * a * (1 + a * (2 + 3 * a)) * c - 2 * ( - 1 + a) * a * (1 + 3 * a + 6 * a**2) * d - (3 * (1 + a * (2 + a * (3 + 4 * a))) * c + d + a * (2 + a * (3 + 2 * (2 - 5 * a) * a)) * d) * y) * z**2) * jnp.log(1 - a) + a**5 * (1 + b) * (d * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * y + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * y + 6 * a * ( - 5 + 2 * a) * y) * z + 
(20 - 15 * y + 2 * a * ( - 15 + 6 * a + 12 * y - 5 * a * y)) * z**2) + c * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * y + 5 * (8 - 3 * z) * z + 6 * y * z * ( - 5 + 2 * z)) - 
5 * y * (6 + z * ( - 8 + 3 * z)))) * jnp.log(1 + b)))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2**3 * power2 * (1 + redshift2)**2 * formula
    return coefficient


@jax.jit
def element8(chi1, chi2, chi3, chi4, chi5, chi6, chi7, chi8, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5-chi3)/(2*chi2)
    c = chi3*jnp.log(chi4/chi3)/(chi4-chi3)-chi5*jnp.log(chi5/chi4)/(chi5-chi4)
    d = (chi8-chi6)/(2*chi2)
    e = chi6*jnp.log(chi7/chi6)/(chi7-chi6)-chi8*jnp.log(chi8/chi7)/(chi8-chi7)
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (1 / 840) * (5 * c * e * (28 + ( - 8 + z) * z) + 8 * c * d * (21 + ( - 7 + z) * z) + 8 * b * e * (21 + ( - 7 + z) * z) + 14 * b * d * (15 + ( - 6 + z) * z)))
    #else:
    def false_branch(_):
        formula = (1 / 60) * a * (b * ( - 30 * e * ( - 2 + y + 2 * z) + 5 * e * z * (y * (8 - 3 * z) + 4 * z) + 20 * d * (3 + ( - 3 + z) * z) + a * e * ( - 30 + 20 * y + 5 * (8 - 3 * z) * z + 6 * y * z * ( - 5 + 2 * z)) - 
5 * d * y * (6 + z * ( - 8 + 3 * z))) + c * (e * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * y + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * y + 6 * a * ( - 5 + 2 * a) * y) * z + 
(20 - 15 * y + 2 * a * ( - 15 + 6 * a + 12 * y - 5 * a * y)) * z**2) + d * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * y + 5 * (8 - 3 * z) * z + 6 * y * z * ( - 5 + 2 * z)) - 
5 * y * (6 + z * ( - 8 + 3 * z)))))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2**3 * power2 * (1 + redshift2)**2 * formula
    return coefficient


@jax.jit
def element9(chi1, chi2, chi3, chi4, chi5, chi6, chi7, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5-chi3)/(2*chi2)
    c = chi3*jnp.log(chi4/chi3)/(chi4-chi3)-chi5*jnp.log(chi5/chi4)/(chi5-chi4)
    d = (chi7-chi6)/(2*chi2)
    e = chi6*jnp.log(chi7/chi6)/(chi7-chi6)-1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (1 / 840) * (5 * c * e * (28 + ( - 8 + z) * z) + 8 * c * d * (21 + ( - 7 + z) * z) + 8 * b * e * (21 + ( - 7 + z) * z) + 14 * b * d * (15 + ( - 6 + z) * z)))
    #else:
    def false_branch(_):
        formula = (1 / 60) * a * (b * ( - 30 * e * ( - 2 + y + 2 * z) + 5 * e * z * (y * (8 - 3 * z) + 4 * z) + 20 * d * (3 + ( - 3 + z) * z) + a * e * ( - 30 + 20 * y + 5 * (8 - 3 * z) * z + 6 * y * z * ( - 5 + 2 * z)) - 
5 * d * y * (6 + z * ( - 8 + 3 * z))) + c * (e * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * y + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * y + 6 * a * ( - 5 + 2 * a) * y) * z + 
(20 - 15 * y + 2 * a * ( - 15 + 6 * a + 12 * y - 5 * a * y)) * z**2) + d * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * y + 5 * (8 - 3 * z) * z + 6 * y * z * ( - 5 + 2 * z)) - 
5 * y * (6 + z * ( - 8 + 3 * z)))))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2**3 * power2 * (1 + redshift2)**2 * formula
    return coefficient


@jax.jit
def element10(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi4-chi3)/(2*chi2)
    c = chi3*jnp.log(chi4/chi3)/(chi4-chi3)-1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    #if a == 1:
    def true_branch(_):
        return jnp.full_like(power2, (1 / 840) * (5 * c**2 * (28 + ( - 8 + z) * z) + 16 * b * c * (21 + ( - 7 + z) * z) + 14 * b**2 * (15 + ( - 6 + z) * z)))
    #else:
    def false_branch(_):
        formula = (1 / 60) * a * (c**2 * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * y + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * y + 6 * a * ( - 5 + 2 * a) * y) * z + 
(20 - 15 * y + 2 * a * ( - 15 + 6 * a + 12 * y - 5 * a * y)) * z**2) + 2 * b * c * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * y + 5 * (8 - 3 * z) * z + 6 * y * z * ( - 5 + 2 * z)) - 
5 * y * (6 + z * ( - 8 + 3 * z))) - 5 * b**2 * ( - 4 * (3 + ( - 3 + z) * z) + y * (6 + z * ( - 8 + 3 * z))))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    coefficient = chi2**3 * power2 * (1 + redshift2)**2 * formula
    return coefficient


@jax.jit
def element(n, i, j, chi, pw, z):
    grid_size = chi.shape[0] - 1
    zeros_vec = jnp.zeros_like(pw[:, 0])
    operand = (n, i, j, chi, pw, z)
    
    cond1  = ((i < n) & (n < grid_size)) | ((j < n) & (n < grid_size))
    cond2  = ((n == i) & (i < grid_size)) & ((n == j) & (j < grid_size))
    cond3  = ( ((n == i) & (n + 1 == j) & (n + 2 < chi.shape[0])) | ((n + 1 == i) & (n == j) & (n + 2 < chi.shape[0])) )
    cond4  = ((n == i) & (n + 1 < j) & (j < grid_size))
    cond5  = ((n + 1 < i) & (i < grid_size) & (n == j))
    cond6  = ( ((n == i) & (j == grid_size)) | ((i == grid_size) & (n == j)) ) & (n < grid_size)
    cond7  = ((n + 1 == i) & (i < grid_size)) & ((n + 1 == j) & (j < grid_size)) & (n + 2 < chi.shape[0])
    cond8  = ((n + 1 == i) & (i < grid_size)) & (n + 1 < j) & (j < grid_size) & (n + 2 < chi.shape[0])
    cond9  = (n + 1 < i) & (i < grid_size) & ((n + 1 == j) & (j < grid_size)) & (n + 2 < chi.shape[0])
    cond10 = ( ((n + 1 == i) & (j == grid_size)) | ((i == grid_size) & (n + 1 == j)) ) & (n + 2 < chi.shape[0])
    cond11 = (n + 1 < i) & (i < grid_size) & (n + 1 < j) & (j < grid_size)
    cond12 = (n + 1 < i) & (i < grid_size) & (j == grid_size)
    cond13 = (i == grid_size) & (n + 1 < j) & (j < grid_size)
    cond14 = (n < grid_size) & (i == grid_size) & (j == grid_size)
    
    def zeros_fn(_): return zeros_vec
    
    def e1(op): 
        n, _, __, chi, pw, z = op
        return element1(chi[n], chi[n+1], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    def e2(op): 
        n, _, __, chi, pw, z = op
        return element2(chi[n], chi[n+1], chi[n+2], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    def e3(op): 
        n, _, j, chi, pw, z = op
        return element3(chi[n], chi[n+1], chi[j-1], chi[j], chi[j+1], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    def e4(op): 
        n, i, _, chi, pw, z = op
        return element3(chi[n], chi[n+1], chi[i-1], chi[i], chi[i+1], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    def e5(op): 
        n, _, __, chi, pw, z = op
        return element4(chi[n], chi[n+1], chi[-2], chi[-1], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    def e6(op): 
        n, _, __, chi, pw, z = op
        return element5(chi[n], chi[n+1], chi[n+2], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    def e7(op): 
        n, _, j, chi, pw, z = op
        return element6(chi[n], chi[n+1], chi[n+2], chi[j-1], chi[j], chi[j+1], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    def e8(op): 
        n, i, _, chi, pw, z = op
        return element6(chi[n], chi[n+1], chi[n+2], chi[i-1], chi[i], chi[i+1], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    def e9(op): 
        n, _, __, chi, pw, z = op
        return element7(chi[n], chi[n+1], chi[n+2], chi[-2], chi[-1], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    def e10(op): 
        n, i, j, chi, pw, z = op
        return element8(chi[n], chi[n+1], chi[i-1], chi[i], chi[i+1], chi[j-1], chi[j], chi[j+1], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    def e11(op): 
        n, i, _, chi, pw, z = op
        return element9(chi[n], chi[n+1], chi[i-1], chi[i], chi[i+1], chi[-2], chi[-1], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    def e12(op): 
        n, _, j, chi, pw, z = op
        return element9(chi[n], chi[n+1], chi[j-1], chi[j], chi[j+1], chi[-2], chi[-1], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    def e13(op): 
        n, _, __, chi, pw, z = op
        return element10(chi[n], chi[n+1], chi[-2], chi[-1], pw[:,n], pw[:,n+1], z[n], z[n+1])
    
    return lax.cond(cond1,  zeros_fn,
        lambda op: lax.cond(cond2,  e1,
        lambda op: lax.cond(cond3,  e2,
        lambda op: lax.cond(cond4,  e3,
        lambda op: lax.cond(cond5,  e4,
        lambda op: lax.cond(cond6,  e5,
        lambda op: lax.cond(cond7,  e6,
        lambda op: lax.cond(cond8,  e7,
        lambda op: lax.cond(cond9,  e8,
        lambda op: lax.cond(cond10, e9,
        lambda op: lax.cond(cond11, e10,
        lambda op: lax.cond(cond12, e11,
        lambda op: lax.cond(cond13, e12,
        lambda op: lax.cond(cond14, e13,
        zeros_fn, op), op), op), op), op), op), op), op), op), op), op), op), op), operand)


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