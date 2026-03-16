import jax
from jax import lax
from jax import vmap
from jax import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)

# Element 1
@jax.jit
def element1(chi1, chi2, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (14952 + z * ( - 20184 + 7087 * z)) / 177811200)
    def false_branch(_):
        formula = (1 / (5292000 * a ** 5)) * (a * (245 * a ** 2 * (10 * a * ( - 60 + a * ( - 750 + a * (1420 + 27 * a * ( - 25 + 4 * a)))) + (420 + a * (2010 + a * (1940 - 9 * a * (1005 + 4 * a * ( - 144 + 25 * a))))) * p) +
14 * a * ( - 35 * a * ( - 420 + a * ( - 2010 + a * ( - 1940 + 9 * a * (1005 + 4 * a * ( - 144 + 25 * a))))) +
2 * ( - 5460 + a * ( - 15330 + a * ( - 14420 + 3 * a * ( - 4305 + 4 * a * (9534 + 125 * a * ( - 49 + 9 * a)))))) * p) * z +
(14 * a * ( - 5460 + a * ( - 15330 + a * ( - 14420 + 3 * a * ( - 4305 + 4 * a * (9534 + 125 * a * ( - 49 + 9 * a)))))) +
(59220 + a * (117810 + a * (107940 + a * (95655 + a * (85344 - 125 * a * (9968 + 27 * a * ( - 256 + 49 * a))))))) * p) * z ** 2) +
420 * ( - 1 + a) ** 2 * jnp.log(1 - a) * (35 * a ** 2 * (7 * p + a * ( - 10 + 74 * p + a * ( - 260 + a * (90 - 72 * p) + 171 * p))) +
14 * a * (5 * a * (7 + a * (74 + 9 * (19 - 8 * a) * a)) + 2 * ( - 13 + a * ( - 86 + 3 * a * ( - 63 + 2 * a * ( - 52 + 25 * a)))) * p) * z +
(14 * a * ( - 13 + a * ( - 86 + 3 * a * ( - 63 + 2 * a * ( - 52 + 25 * a)))) + (141 + a * (702 + a * (1473 + 8 * a * (298 + 25 * (17 - 9 * a) * a)))) * p) * z ** 2 -
210 * ( - 1 + a) * (5 * a ** 2 * (p + a * ( - 4 + 3 * p)) - 2 * a * (2 * p + a * ( - 5 + 6 * p + 3 * a * ( - 5 + 4 * p))) * z +
( - 2 * a * (1 + 3 * a + 6 * a ** 2) + p + a * (3 + 2 * a * (3 + 5 * a)) * p) * z ** 2) * jnp.log(1 - a)))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 2
@jax.jit
def element2(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (b * (6642 + z * ( - 6669 + 1867 * z) + 18 * b * (294 + z * ( - 308 + 89 * z))) - 45 * (1 + b) * (112 + z * ( - 104 + 27 * z)) * jnp.log(1 + b)) / (6350400 * b))
    def false_branch(_):
        formula = - ((1 / (4 * a ** 5 * b)) * ((1 / 15) * ( - 1 + a) ** 4 * b * (5 * a ** 2 * (p + a * ( - 4 + 3 * p)) - 2 * a * (2 * p + a * ( - 5 + 6 * p + 3 * a * ( - 5 + 4 * p))) * z +
( - 2 * a * (1 + 3 * a + 6 * a ** 2) + p + a * (3 + 2 * a * (3 + 5 * a)) * p) * z ** 2) * jnp.log(1 - a) ** 2 +
(1 / 3150) * (( - 1 + a) ** 2 * jnp.log(1 - a) * (b * (35 * a ** 2 * ( - 5 * a * (2 + a * (23 + a * ( - 52 + 9 * a) - 18 * b)) + (7 + a * (28 + a * (19 + 36 * ( - 5 + a) * a - 60 * b) - 30 * b)) * p) + 14 * a * (5 * a * (7 + a * (28 + a * (19 + 36 * ( - 5 + a) * a - 60 * b) - 30 * b)) - 26 * p + a * ( - 56 + 2 * a * ( - 13 + 3 * a * (4 + (114 - 25 * a) * a)) +
75 * (1 + a * (2 + 3 * a)) * b) * p) * z + (7 * a * ( - 26 - 2 * a * (28 + a * (13 + 3 * a * ( - 4 + a * ( - 114 + 25 * a)))) + 75 * a * (1 + a * (2 + 3 * a)) * b) +
(141 + a * (201 + 51 * a - 169 * a ** 2 - 424 * a ** 3 - 3850 * a ** 4 + 900 * a ** 5 - 315 * (1 + a * (2 + a * (3 + 4 * a))) * b)) * p) * z ** 2) -
210 * ( - 1 + a) * a * (1 + b) * (5 * a ** 2 * (p + a * ( - 4 + 3 * p)) - 2 * a * (2 * p + a * ( - 5 + 6 * p + 3 * a * ( - 5 + 4 * p))) * z +
( - 2 * a * (1 + 3 * a + 6 * a ** 2) + p + a * (3 + 2 * a * (3 + 5 * a)) * p) * z ** 2) * jnp.log(1 + b))) -
(1 / 1323000) * (a * (b * (245 * a ** 2 * ( - 420 * p + 20 * a ** 2 * ( - 60 + 45 * b * ( - 6 + p) + 62 * p) + 9 * a ** 5 * ( - 195 + 148 * p) + 5 * a ** 3 * ( - 830 + 60 * b * (27 - 16 * p) + 193 * p) +
150 * a * (4 + (5 + 12 * b) * p) + 2 * a ** 4 * (3725 - 2322 * p + 225 * b * ( - 4 + 3 * p))) -
14 * a * (35 * a * (420 + a * ( - 150 * (5 + 12 * b) + a * ( - 20 * (62 + 45 * b) + a * ( - 965 + 4800 * b - 18 * a * ( - 258 + 74 * a + 75 * b))))) +
( - 10920 + 2 * a * (9030 + a * (10360 + a * (7805 + 6 * a * (1008 + a * ( - 9688 + 3125 * a))))) + 525 * a * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b) * p) * z + (7 * a * (10920 - 2 * a * (9030 + a * (10360 + a * (7805 + 6 * a * (1008 + a * ( - 9688 + 3125 * a))))) -
525 * a * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b) + ( - 59220 + a * (92610 + 85470 * a + 62685 * a ** 2 + 48111 * a ** 3 + 38584 * a ** 4 - 629500 * a ** 5 + 219375 * a ** 6 + 2205 * (60 + a * (30 + a * (20 + a * (15 + 4 * a * ( - 72 + 25 * a))))) * b)) * p) * z ** 2) +
210 * a * (1 + b) * ( - 210 * a * (p * ( - 8 + z) - 4 * z) * z - 420 * p * z ** 2 - 35 * a ** 3 * (30 * ( - 8 + p) + 4 * (15 - 4 * p) * z + ( - 8 + 3 * p) * z ** 2) +
14 * a ** 4 * ( - 1500 + 850 * p + 5 * (340 - 117 * z) * z + 6 * p * z * ( - 195 + 74 * z)) - 140 * a ** 2 * ( - 3 * ( - 10 + z) * z + p * (15 + ( - 6 + z) * z)) +
30 * a ** 6 * (4 * p * (21 + 5 * z * ( - 7 + 3 * z)) - 7 * (15 + 2 * z * ( - 12 + 5 * z))) - 7 * a ** 5 * ( - 2200 + 18 * (175 - 68 * z) * z + p * (1575 + 8 * z * ( - 306 + 125 * z)))) * jnp.log(1 + b)))))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 3
@jax.jit
def element3(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5 - chi3) / (2 * chi2)
    c = chi3 * jnp.log(chi4 / chi3) / (chi4 - chi3) - chi5 * jnp.log(chi5 / chi4) / (chi5 - chi4)
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (5 * c * (112 + z * ( - 104 + 27 * z)) + 4 * b * (294 + z * ( - 308 + 89 * z))) / 705600)
    def false_branch(_):
        formula = - ((1 / (2 * a ** 4)) * ((1 / 12600) * (a * (35 * a ** 2 * (10 * a * (6 * (6 + a * ( - 9 + 2 * a)) * b + (24 + a * ( - 60 + (44 - 9 * a) * a)) * c) -
(10 * (12 + a * (6 + a * ( - 32 + 9 * a))) * b + (60 + a * (30 + a * ( - 340 + 9 * (35 - 8 * a) * a))) * c) * p) -
14 * a * (50 * a * (12 + a * (6 + a * ( - 32 + 9 * a))) * b - 5 * a * ( - 60 + a * ( - 30 + a * (340 + 9 * a * ( - 35 + 8 * a)))) * c -
(5 * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 2 * (60 + a * (30 + a * (20 - 3 * a * (195 - 204 * a + 50 * a ** 2)))) * c) * p) * z +
(7 * a * (5 * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 2 * (60 + a * (30 + a * (20 - 3 * a * (195 - 204 * a + 50 * a ** 2)))) * c) -
(21 * (60 + a * (30 + a * (20 + a * (15 + 4 * a * ( - 72 + 25 * a))))) * b + (420 + a * (210 + a * (140 + a * (105 - 8 * a * (777 + 25 * a * ( - 35 + 9 * a)))))) * c) * p) * z ** 2)) + (1 / 30) * ( - 1 + a) ** 2 * (5 * a ** 2 * (6 * a * b - 4 * ( - 1 + a) * a * c - ((2 + 4 * a) * b + c + (2 - 3 * a) * a * c) * p) +
2 * a * ( - 10 * a * (1 + 2 * a) * b + 5 * ( - 1 + a) * a * (1 + 3 * a) * c + 5 * (1 + a * (2 + 3 * a)) * b * p - 2 * ( - 1 + a) * (1 + 3 * a + 6 * a ** 2) * c * p) * z +
(5 * a * (1 + a * (2 + 3 * a)) * b - 2 * ( - 1 + a) * a * (1 + 3 * a + 6 * a ** 2) * c - (3 * (1 + a * (2 + a * (3 + 4 * a))) * b + c + a * (2 + a * (3 + 2 * (2 - 5 * a) * a)) * c) * p) * z ** 2) * jnp.log(1 - a)))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 4
@jax.jit
def element4(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi4 - chi3) / (2 * chi2)
    c = chi3 * jnp.log(chi4 / chi3) / (chi4 - chi3) - 1
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (5 * c * (112 + z * ( - 104 + 27 * z)) + 4 * b * (294 + z * ( - 308 + 89 * z))) / 705600)
    def false_branch(_):
        formula = - ((1 / (2 * a ** 4)) * ((1 / 12600) * (a * (35 * a ** 2 * (10 * a * (6 * (6 + a * ( - 9 + 2 * a)) * b + (24 + a * ( - 60 + (44 - 9 * a) * a)) * c) -
(10 * (12 + a * (6 + a * ( - 32 + 9 * a))) * b + (60 + a * (30 + a * ( - 340 + 9 * (35 - 8 * a) * a))) * c) * p) -
14 * a * (50 * a * (12 + a * (6 + a * ( - 32 + 9 * a))) * b - 5 * a * ( - 60 + a * ( - 30 + a * (340 + 9 * a * ( - 35 + 8 * a)))) * c -
(5 * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 2 * (60 + a * (30 + a * (20 - 3 * a * (195 - 204 * a + 50 * a ** 2)))) * c) * p) * z +
(7 * a * (5 * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 2 * (60 + a * (30 + a * (20 - 3 * a * (195 - 204 * a + 50 * a ** 2)))) * c) -
(21 * (60 + a * (30 + a * (20 + a * (15 + 4 * a * ( - 72 + 25 * a))))) * b + (420 + a * (210 + a * (140 + a * (105 - 8 * a * (777 + 25 * a * ( - 35 + 9 * a)))))) * c) * p) * z ** 2)) + (1 / 30) * ( - 1 + a) ** 2 * (5 * a ** 2 * (6 * a * b - 4 * ( - 1 + a) * a * c - ((2 + 4 * a) * b + c + (2 - 3 * a) * a * c) * p) +
2 * a * ( - 10 * a * (1 + 2 * a) * b + 5 * ( - 1 + a) * a * (1 + 3 * a) * c + 5 * (1 + a * (2 + 3 * a)) * b * p - 2 * ( - 1 + a) * (1 + 3 * a + 6 * a ** 2) * c * p) * z +
(5 * a * (1 + a * (2 + 3 * a)) * b - 2 * ( - 1 + a) * a * (1 + 3 * a + 6 * a ** 2) * c - (3 * (1 + a * (2 + a * (3 + 4 * a))) * b + c + a * (2 + a * (3 + 2 * (2 - 5 * a) * a)) * c) * p) * z ** 2) * jnp.log(1 - a)))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 5
@jax.jit
def element5(chi1, chi2, chi3, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (1 / (10080 * b ** 2)) * (b ** 2 * (1785 + z * ( - 562 + 79 * z) + 42 * b ** 2 * (15 + ( - 6 + z) * z) + 6 * b * (350 + z * ( - 124 + 19 * z))) -
8 * b * (1 + b) * (432 + z * ( - 129 + 17 * z) + 12 * b * (21 + ( - 7 + z) * z)) * jnp.log(1 + b) + 60 * (1 + b) ** 2 * (28 + ( - 8 + z) * z) * jnp.log(1 + b) ** 2))
    def false_branch(_):
        formula = (1 / (5292000 * a ** 5 * b ** 2)) * (a * b ** 2 * (245 * a ** 2 * (420 * p + a * ( - 600 + a ** 4 * (4030 - 2439 * p) - 90 * (39 + 40 * b) * p + 5 * a ** 2 * ( - 220 - 60 * b * ( - 2 * ( - 9 + p) + 9 * b * ( - 2 + p)) + 133 * p) +
a ** 3 * ( - 2150 + 300 * b * (30 - 17 * p) + 404 * p) + 20 * a * (495 + 34 * p + 90 * b * (6 + p)))) +
14 * a * ( - 10920 * p - 70 * a ** 3 * ( - 340 + 150 * b * ( - 6 + p) + 163 * p) + a ** 6 * ( - 85365 + 60096 * p) + 420 * a * (35 + 3 * (53 + 50 * b) * p) -
70 * a ** 2 * (1755 + 232 * p + 450 * b * (4 + p)) + 14 * a ** 5 * (1010 - 303 * p + 75 * b * ( - 170 + 117 * p)) +
7 * a ** 4 * (3325 - 952 * p + 750 * b * (4 - p + 6 * b * ( - 3 + 2 * p)))) * z +
(a ** 7 * (420672 - 322375 * p) + 59220 * p + 35 * a ** 3 * ( - 3248 + 1260 * b * ( - 5 + p) + 1371 * p) + a ** 6 * ( - 29694 + 4410 * b * (195 - 148 * p) + 11624 * p) -
210 * a * (364 + 3 * (481 + 420 * b) * p) + 14 * a ** 4 * ( - 5705 + 1941 * p + 525 * b * ( - 10 + 3 * p)) + 420 * a ** 2 * (1113 + 197 * p + 105 * b * (10 + 3 * p)) -
7 * a ** 5 * (6664 - 2441 * p + 210 * b * (25 - 9 * p + 75 * b * ( - 4 + 3 * p)))) * z ** 2) +
420 * ( - 210 * ( - 1 + a) ** 5 * b ** 2 * (5 * a ** 2 * (p + a * ( - 4 + 3 * p)) - 2 * a * (2 * p + a * ( - 5 + 6 * p + 3 * a * ( - 5 + 4 * p))) * z +
( - 2 * a * (1 + 3 * a + 6 * a ** 2) + p + a * (3 + 2 * a * (3 + 5 * a)) * p) * z ** 2) * jnp.log(1 - a) ** 2 + ( - 1 + a) ** 3 * b * jnp.log(1 - a) * (b * (35 * a ** 2 * ( - 10 * a * ( - 1 + a * (2 + 17 * a + 18 * b)) + ( - 7 + a * (11 + 60 * b + a * (59 + 117 * a + 120 * b))) * p) -
14 * a * ( - 5 * a * ( - 7 + a * (11 + 60 * b + a * (59 + 117 * a + 120 * b))) + 2 * ( - 13 + a * (17 + 75 * b + a * (77 + 150 * b + 3 * a * (49 + 74 * a + 75 * b)))) * p) * z +
( - 14 * a * ( - 13 + a * (17 + 75 * b + a * (77 + 150 * b + 3 * a * (49 + 74 * a + 75 * b)))) +
( - 141 + a * (159 + 669 * a + 1249 * a ** 2 + 1864 * a ** 3 + 2500 * a ** 4 + 630 * (1 + a * (2 + a * (3 + 4 * a))) * b)) * p) * z ** 2) +
420 * ( - 1 + a) * a * (1 + b) * (5 * a ** 2 * (p + a * ( - 4 + 3 * p)) - 2 * a * (2 * p + a * ( - 5 + 6 * p + 3 * a * ( - 5 + 4 * p))) * z +
( - 2 * a * (1 + 3 * a + 6 * a ** 2) + p + a * (3 + 2 * a * (3 + 5 * a)) * p) * z ** 2) * jnp.log(1 + b)) -
a ** 2 * (1 + b) * jnp.log(1 + b) * (b * ( - 420 * p * z ** 2 + 210 * a * z * (4 * z + p * (8 + z)) + a ** 6 * ( - 5950 + 4095 * p + 42 * (195 - 74 * z) * z + 4 * p * z * ( - 1554 + 625 * z)) +
70 * a ** 2 * ( - 6 * z * (10 + z) + p * ( - 30 + ( - 12 + z) * z)) + 35 * a ** 3 * (240 - 4 * ( - 15 + z) * z + p * (30 + ( - 8 + z) * z)) +
7 * a ** 5 * ( - 25 * ( - 64 + 35 * p + 70 * z) + 2 * z * (p * (594 - 224 * z) + 297 * z) + 30 * b * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z))) +
7 * a ** 4 * (50 * ( - 12 + p) - 20 * ( - 5 + p) * z + ( - 10 + 3 * p) * z ** 2 - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + p * (6 + z * ( - 8 + 3 * z))))) +
210 * a ** 4 * (1 + b) * ( - 20 * (3 + ( - 3 + a) * a) + 5 * (6 + a * ( - 8 + 3 * a)) * p + 2 * (30 + 5 * a * ( - 8 + 3 * a) - 20 * p + 6 * (5 - 2 * a) * a * p) * z +
(5 * ( - 4 + 3 * p) + 2 * a * (15 - 12 * p + a * ( - 6 + 5 * p))) * z ** 2) * jnp.log(1 + b))))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 6
@jax.jit
def element6(chi1, chi2, chi3, chi4, chi5, chi6, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    c = (chi6 - chi4) / (2 * chi2)
    d = chi4 * jnp.log(chi5 / chi4) / (chi5 - chi4) - chi6 * jnp.log(chi6 / chi5) / (chi6 - chi5)
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (1 / (5040 * b)) * (b * (2 * d * (432 + z * ( - 129 + 17 * z) + 12 * b * (21 + ( - 7 + z) * z)) + 3 * c * (350 + z * ( - 124 + 19 * z) + 14 * b * (15 + ( - 6 + z) * z))) -
6 * (1 + b) * (5 * d * (28 + ( - 8 + z) * z) + 8 * c * (21 + ( - 7 + z) * z)) * jnp.log(1 + b)))
    def false_branch(_):
        formula = (1 / (25200 * a ** 4 * b)) * (a * b * ( - 420 * (3 * c + d) * p * z ** 2 + 70 * a ** 2 * ( - 30 * (2 * c + d) * p - 6 * (5 * c * (4 + p) + 2 * d * (5 + p)) * z + (d * ( - 6 + p) + 3 * c * ( - 5 + p)) * z ** 2) +
210 * a * z * (8 * d * p + 10 * c * z + d * (4 + p) * z + c * p * (20 + 3 * z)) + a ** 6 * d * ( - 5950 + 4095 * p + 42 * (195 - 74 * z) * z + 4 * p * z * ( - 1554 + 625 * z)) +
35 * a ** 3 * (c * (60 * (6 + p) - 20 * ( - 6 + p) * z + ( - 10 + 3 * p) * z ** 2) + d * (240 - 4 * ( - 15 + z) * z + p * (30 + ( - 8 + z) * z))) +
7 * a ** 5 * (c * (1500 - 850 * p + 6 * p * (195 - 74 * z) * z + 5 * z * ( - 340 + 117 * z)) + d * ( - 25 * ( - 64 + 35 * p + 70 * z) + 2 * z * (p * (594 - 224 * z) + 297 * z) +
30 * b * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)))) +
7 * a ** 4 * (d * (50 * ( - 12 + p) - 20 * ( - 5 + p) * z + ( - 10 + 3 * p) * z ** 2 - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + p * (6 + z * ( - 8 + 3 * z)))) +
c * (100 * ( - 9 + p + 2 * z) + z * ( - 25 * z + p * ( - 50 + 9 * z)) - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + p * (6 + z * ( - 8 + 3 * z)))))) -
420 * (( - 1 + a) ** 3 * b * (5 * a ** 2 * (6 * a * c - 4 * ( - 1 + a) * a * d - ((2 + 4 * a) * c + d + (2 - 3 * a) * a * d) * p) +
2 * a * ( - 10 * a * (1 + 2 * a) * c + 5 * ( - 1 + a) * a * (1 + 3 * a) * d + 5 * (1 + a * (2 + 3 * a)) * c * p - 2 * ( - 1 + a) * (1 + 3 * a + 6 * a ** 2) * d * p) * z +
(5 * a * (1 + a * (2 + 3 * a)) * c - 2 * ( - 1 + a) * a * (1 + 3 * a + 6 * a ** 2) * d - (3 * (1 + a * (2 + a * (3 + 4 * a))) * c + d + a * (2 + a * (3 + 2 * (2 - 5 * a) * a)) * d) * p) * z ** 2) * jnp.log(1 - a) + a ** 5 * (1 + b) * (d * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * p + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * p + 6 * a * ( - 5 + 2 * a) * p) * z +
(20 - 15 * p + 2 * a * ( - 15 + 6 * a + 12 * p - 5 * a * p)) * z ** 2) + c * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * p * (6 + z * ( - 8 + 3 * z)))) * jnp.log(1 + b)))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 7
@jax.jit
def element7(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    c = (chi5 - chi4) / (2 * chi2)
    d = chi4 * jnp.log(chi5 / chi4) / (chi5 - chi4) - 1
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (1 / (5040 * b)) * (b * (2 * d * (432 + z * ( - 129 + 17 * z) + 12 * b * (21 + ( - 7 + z) * z)) + 3 * c * (350 + z * ( - 124 + 19 * z) + 14 * b * (15 + ( - 6 + z) * z))) -
6 * (1 + b) * (5 * d * (28 + ( - 8 + z) * z) + 8 * c * (21 + ( - 7 + z) * z)) * jnp.log(1 + b)))
    def false_branch(_):
        formula = (1 / (25200 * a ** 4 * b)) * (a * b * ( - 420 * (3 * c + d) * p * z ** 2 + 70 * a ** 2 * ( - 30 * (2 * c + d) * p - 6 * (5 * c * (4 + p) + 2 * d * (5 + p)) * z + (d * ( - 6 + p) + 3 * c * ( - 5 + p)) * z ** 2) +
210 * a * z * (8 * d * p + 10 * c * z + d * (4 + p) * z + c * p * (20 + 3 * z)) + a ** 6 * d * ( - 5950 + 4095 * p + 42 * (195 - 74 * z) * z + 4 * p * z * ( - 1554 + 625 * z)) +
35 * a ** 3 * (c * (60 * (6 + p) - 20 * ( - 6 + p) * z + ( - 10 + 3 * p) * z ** 2) + d * (240 - 4 * ( - 15 + z) * z + p * (30 + ( - 8 + z) * z))) +
7 * a ** 5 * (c * (1500 - 850 * p + 6 * p * (195 - 74 * z) * z + 5 * z * ( - 340 + 117 * z)) + d * ( - 25 * ( - 64 + 35 * p + 70 * z) + 2 * z * (p * (594 - 224 * z) + 297 * z) +
30 * b * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)))) +
7 * a ** 4 * (d * (50 * ( - 12 + p) - 20 * ( - 5 + p) * z + ( - 10 + 3 * p) * z ** 2 - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + p * (6 + z * ( - 8 + 3 * z)))) +
c * (100 * ( - 9 + p + 2 * z) + z * ( - 25 * z + p * ( - 50 + 9 * z)) - 150 * b * ( - 4 * (3 + ( - 3 + z) * z) + p * (6 + z * ( - 8 + 3 * z)))))) -
420 * (( - 1 + a) ** 3 * b * (5 * a ** 2 * (6 * a * c - 4 * ( - 1 + a) * a * d - ((2 + 4 * a) * c + d + (2 - 3 * a) * a * d) * p) +
2 * a * ( - 10 * a * (1 + 2 * a) * c + 5 * ( - 1 + a) * a * (1 + 3 * a) * d + 5 * (1 + a * (2 + 3 * a)) * c * p - 2 * ( - 1 + a) * (1 + 3 * a + 6 * a ** 2) * d * p) * z +
(5 * a * (1 + a * (2 + 3 * a)) * c - 2 * ( - 1 + a) * a * (1 + 3 * a + 6 * a ** 2) * d - (3 * (1 + a * (2 + a * (3 + 4 * a))) * c + d + a * (2 + a * (3 + 2 * (2 - 5 * a) * a)) * d) * p) * z ** 2) * jnp.log(1 - a) + a ** 5 * (1 + b) * (d * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * p + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * p + 6 * a * ( - 5 + 2 * a) * p) * z +
(20 - 15 * p + 2 * a * ( - 15 + 6 * a + 12 * p - 5 * a * p)) * z ** 2) + c * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * p * (6 + z * ( - 8 + 3 * z)))) * jnp.log(1 + b)))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 8
@jax.jit
def element8(chi1, chi2, chi3, chi4, chi5, chi6, chi7, chi8, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5 - chi3) / (2 * chi2)
    c = chi3 * jnp.log(chi4 / chi3) / (chi4 - chi3) - chi5 * jnp.log(chi5 / chi4) / (chi5 - chi4)
    d = (chi8 - chi6) / (2 * chi2)
    e = chi6 * jnp.log(chi7 / chi6) / (chi7 - chi6) - chi8 * jnp.log(chi8 / chi7) / (chi8 - chi7)
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (1 / 840) * (5 * c * e * (28 + ( - 8 + z) * z) + 8 * c * d * (21 + ( - 7 + z) * z) + 8 * b * e * (21 + ( - 7 + z) * z) + 14 * b * d * (15 + ( - 6 + z) * z)))
    def false_branch(_):
        formula = (1 / 60) * a * (b * ( - 30 * e * ( - 2 + p + 2 * z) + 5 * e * z * (p * (8 - 3 * z) + 4 * z) + 20 * d * (3 + ( - 3 + z) * z) + a * e * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * d * p * (6 + z * ( - 8 + 3 * z))) + c * (e * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * p + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * p + 6 * a * ( - 5 + 2 * a) * p) * z +
(20 - 15 * p + 2 * a * ( - 15 + 6 * a + 12 * p - 5 * a * p)) * z ** 2) + d * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * p * (6 + z * ( - 8 + 3 * z)))))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 9
@jax.jit
def element9(chi1, chi2, chi3, chi4, chi5, chi6, chi7, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi5 - chi3) / (2 * chi2)
    c = chi3 * jnp.log(chi4 / chi3) / (chi4 - chi3) - chi5 * jnp.log(chi5 / chi4) / (chi5 - chi4)
    d = (chi7 - chi6) / (2 * chi2)
    e = chi6 * jnp.log(chi7 / chi6) / (chi7 - chi6) - 1
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (1 / 840) * (5 * c * e * (28 + ( - 8 + z) * z) + 8 * c * d * (21 + ( - 7 + z) * z) + 8 * b * e * (21 + ( - 7 + z) * z) + 14 * b * d * (15 + ( - 6 + z) * z)))
    def false_branch(_):
        formula = (1 / 60) * a * (b * ( - 30 * e * ( - 2 + p + 2 * z) + 5 * e * z * (p * (8 - 3 * z) + 4 * z) + 20 * d * (3 + ( - 3 + z) * z) + a * e * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * d * p * (6 + z * ( - 8 + 3 * z))) + c * (e * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * p + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * p + 6 * a * ( - 5 + 2 * a) * p) * z +
(20 - 15 * p + 2 * a * ( - 15 + 6 * a + 12 * p - 5 * a * p)) * z ** 2) + d * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * p * (6 + z * ( - 8 + 3 * z)))))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Element 10
@jax.jit
def element10(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):
    a = 1 - chi1 / chi2
    b = (chi4 - chi3) / (2 * chi2)
    c = chi3 * jnp.log(chi4 / chi3) / (chi4 - chi3) - 1
    p = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    
    def true_branch(_):
        return jnp.full_like(p, (1 / 840) * (5 * c ** 2 * (28 + ( - 8 + z) * z) + 16 * b * c * (21 + ( - 7 + z) * z) + 14 * b ** 2 * (15 + ( - 6 + z) * z)))
    def false_branch(_):
        formula = (1 / 60) * a * (c ** 2 * (20 * (3 + ( - 3 + a) * a) - 5 * (6 + a * ( - 8 + 3 * a)) * p + 2 * ( - 30 + 5 * (8 - 3 * a) * a + 20 * p + 6 * a * ( - 5 + 2 * a) * p) * z +
(20 - 15 * p + 2 * a * ( - 15 + 6 * a + 12 * p - 5 * a * p)) * z ** 2) + 2 * b * c * (20 * (3 + ( - 3 + z) * z) + a * ( - 30 + 20 * p + 5 * (8 - 3 * z) * z + 6 * p * z * ( - 5 + 2 * z)) -
5 * p * (6 + z * ( - 8 + 3 * z))) - 5 * b ** 2 * ( - 4 * (3 + ( - 3 + z) * z) + p * (6 + z * ( - 8 + 3 * z))))
        return formula
    formula = lax.cond(a == 1.0, true_branch, false_branch, operand=None)
    
    coefficient = chi2 ** 3 * power2 * (1 + redshift2) ** 2 * formula
    return coefficient

# Coefficient
@jax.jit
def coefficient(chi_grid, power_grid, redshift_grid):
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    coefficients = jnp.zeros((grid_size + 1, grid_size + 1, ell_size + 1), dtype=power_grid.dtype)
    
    k_indices = jnp.arange(grid_size + 1, dtype=jnp.int32)
    def accumulate_step(n, coefficients):
        valid = (n + 1 < grid_size)
        
        value1 = element1(chi_grid[n], chi_grid[n + 1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        coefficients = coefficients.at[n, n, :].add(value1)
        
        value2 = element2(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        coefficients = coefficients.at[n, n + 1, :].add(jnp.where(valid, value2, jnp.zeros_like(value2)))
        coefficients = coefficients.at[n + 1, n, :].add(jnp.where(valid, value2, jnp.zeros_like(value2)))
        
        def compute_value3(k):
            return element3(chi_grid[n], chi_grid[n + 1], chi_grid[k - 1], chi_grid[k], chi_grid[k + 1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        value3_all = vmap(compute_value3)(k_indices)
        value3_mask = (k_indices >= n + 2) & (k_indices < grid_size)
        coefficients = coefficients.at[n, :, :].add(jnp.where(value3_mask[:, None], value3_all, 0.0))
        coefficients = coefficients.at[:, n, :].add(jnp.where(value3_mask[:, None], value3_all, 0.0))
        
        value4 = element4(chi_grid[n], chi_grid[n + 1], chi_grid[-2], chi_grid[-1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        coefficients = coefficients.at[n, grid_size, :].add(value4)
        coefficients = coefficients.at[grid_size, n, :].add(value4)
        
        value5 = element5(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        coefficients = coefficients.at[n + 1, n + 1, :].add(jnp.where(valid, value5, jnp.zeros_like(value5)))
        
        def compute_element6(k):
            return element6(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2],chi_grid[k - 1], chi_grid[k], chi_grid[k + 1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        value6_all = vmap(compute_element6)(k_indices)
        value6_mask = (k_indices >= n + 2) & (k_indices < grid_size) & valid
        coefficients = coefficients.at[n + 1, :, :].add(jnp.where(value6_mask[:, None], value6_all, 0.0))
        coefficients = coefficients.at[:, n + 1, :].add(jnp.where(value6_mask[:, None], value6_all, 0.0))
        
        value7 = element7(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], chi_grid[-2], chi_grid[-1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        coefficients = coefficients.at[n + 1, grid_size, :].add(jnp.where(valid, value7, jnp.zeros_like(value7)))
        coefficients = coefficients.at[grid_size, n + 1, :].add(jnp.where(valid, value7, jnp.zeros_like(value7)))
        
        def compute_element8_row(i):
            def compute_element8_entry(j):
                return element8(chi_grid[n], chi_grid[n + 1], chi_grid[i - 1], chi_grid[i], chi_grid[i + 1], chi_grid[j - 1], chi_grid[j], chi_grid[j + 1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
            return vmap(compute_element8_entry)(k_indices)
        value8_all = vmap(compute_element8_row)(k_indices)
        value8_mask = ((k_indices[:, None] >= n + 2) & (k_indices[:, None] < grid_size) & (k_indices[None, :] >= n + 2) & (k_indices[None, :] < grid_size))
        coefficients = coefficients + jnp.where(value8_mask[:, :, None], value8_all, 0.0)
        
        def compute_element9(k):
            return element9(chi_grid[n], chi_grid[n + 1], chi_grid[k - 1], chi_grid[k], chi_grid[k + 1], chi_grid[-2], chi_grid[-1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        value9_all = vmap(compute_element9)(k_indices)
        value9_mask = (k_indices >= n + 2) & (k_indices < grid_size)
        coefficients = coefficients.at[:, grid_size, :].add(jnp.where(value9_mask[:, None], value9_all, 0.0))
        coefficients = coefficients.at[grid_size, :, :].add(jnp.where(value9_mask[:, None], value9_all, 0.0))
        
        value10 = element10(chi_grid[n], chi_grid[n + 1], chi_grid[-2], chi_grid[-1], power_grid[:, n], power_grid[:, n + 1], redshift_grid[n], redshift_grid[n + 1])
        coefficients = coefficients.at[grid_size, grid_size, :].add(value10)
        
        return coefficients
    
    return lax.fori_loop(0, grid_size, accumulate_step, coefficients)

# Spectra
@jax.jit
def spectra(factor, phi_a_grid, phi_b_grid, chi_grid, power_grid, redshift_grid):
    coefficients = coefficient(chi_grid, power_grid, redshift_grid)
    spectrum = factor * jnp.einsum('mi,nj,ijl->mnl', phi_a_grid, phi_b_grid, coefficients)
    return spectrum
