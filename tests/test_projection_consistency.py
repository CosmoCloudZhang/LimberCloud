"""Small cross-backend checks for the analytic projection formulas."""

import unittest

try:
    import jax  # noqa: F401
    import numba  # noqa: F401
    import numpy

    from limbercloud.projection import jax_backend, numba_backend
except ImportError:
    SCIENCE_DEPENDENCIES_AVAILABLE = False
else:
    SCIENCE_DEPENDENCIES_AVAILABLE = True


@unittest.skipUnless(
    SCIENCE_DEPENDENCIES_AVAILABLE,
    "NumPy, Numba, and JAX are required for cross-backend checks",
)
class ProjectionConsistencyTests(unittest.TestCase):
    @staticmethod
    def local_reference(family, number, left, right, power_left, power_right, z_left, z_right):
        """Integrate original source hats in dimensionless moving coordinates."""
        from scipy.integrate import quad

        width = right - left

        def source(r, rising):
            # G / width**2; distances are formed without subtracting nearby nodes.
            def integrand(v):
                hat = r + (1-r)*v if rising else (1-r)*(1-v)
                u = left + width*(r+(1-r)*v)
                return hat*v/u
            return (1-r)**2*quad(integrand, 0, 1, epsabs=2e-12, epsrel=2e-12)[0]

        def integrand(r):
            chi = left + width*r
            power = power_right*(chi/right)**3 if left == 0 else power_left*(1-r)+power_right*r
            rz = (1+z_left)*(1-r)+(1+z_right)*r
            if family == "NS":
                density = r if number in (2, 10) else 1-r
                return power*rz*density*source(r, number in (9, 10))/chi
            first = source(r, number == 12)
            second = source(r, number != 1)
            return power*rz**2*first*second

        exponent = 3 if family == "NS" else 5
        return width**exponent*quad(integrand, 0, 1, epsabs=2e-12, epsrel=2e-12)[0]

    def test_local_boundary_coefficients_against_moving_hat_integrals(self):
        for left, right in ((0.0, 1.0), (0.2, 1.0), (0.25, 1.0), (0.7, 1.3), (1.0, 1.0000001)):
            p1 = numpy.array([0.8, -0.4, 1.0, 0.0])
            p2 = numpy.array([1.1, 0.7, 0.0, 0.0])
            for family, numbers in (("NS", (1, 2, 9, 10)), ("SS", (1, 11, 12))):
                for number in numbers:
                    expected = numpy.array([
                        self.local_reference(family, number, left, right, x, y, 0.2, 0.5)
                        for x, y in zip(p1, p2, strict=True)
                    ])
                    for backend in (numba_backend, jax_backend):
                        with self.subTest(backend=backend.__name__, family=family, number=number, left=left, right=right):
                            function = getattr(getattr(backend, family), f"element{number}")
                            actual = numpy.asarray(function(left, right, p1, p2, 0.2, 0.5))
                            numpy.testing.assert_allclose(actual, expected, rtol=3e-10, atol=1e-60)

    def test_nn_zero_and_narrow_power_against_independent_quadrature(self):
        from limbercloud.validation.reference import nn_interval_quadrature

        for left, right in ((0.0, 1.0), (0.1, 1.0), (0.7, 1.3), (1.0, 1.00001)):
            p1 = numpy.array([0.0, 0.8, -0.6, 0.8])
            p2 = numpy.array([0.0, 0.0, 0.9, 1.1])
            for number in (1, 2, 3):
                name = f"element{number}"
                expected = numpy.array([nn_interval_quadrature(left, right, x, y, name) for x, y in zip(p1, p2, strict=True)])
                for backend in (numba_backend, jax_backend):
                    numpy.testing.assert_allclose(getattr(backend.NN, name)(left, right, p1, p2), expected, rtol=2e-9, atol=1e-30)

    def test_small_assembled_tensors_match_independent_hats(self):
        from scipy.integrate import quad

        from limbercloud.validation.reference import (
            declared_power,
            hat,
            lensing_efficiency,
        )

        for grid in (numpy.array([0.0, 1.0]), numpy.array([0.0, 0.7, 1.4, 2.0])):
            redshift = grid*0.2
            power = numpy.array([numpy.linspace(0.0, 1.1, len(grid)), numpy.zeros(len(grid))])
            power[1, 1:-1] = -0.4
            expected = {key: numpy.zeros((len(grid), len(grid), 2)) for key in ("NN", "NS", "SS")}
            for family in expected:
                for i in range(len(grid)):
                    for j in range(len(grid)):
                        for ell in range(2):
                            def integrand(x):
                                P = declared_power(x, grid, power[ell])
                                first = hat(x, i, grid) if family != "SS" else lensing_efficiency(x, i, grid)
                                second = hat(x, j, grid) if family == "NN" else lensing_efficiency(x, j, grid)
                                factor = 1/x**2 if family == "NN" else ((1+0.2*x)/x if family == "NS" else (1+0.2*x)**2)
                                return P*first*second*factor
                            expected[family][i, j, ell] = sum(quad(integrand, lower, upper, epsabs=1e-12, epsrel=2e-10)[0] for lower, upper in zip(grid[:-1], grid[1:], strict=True))
            for backend in (numba_backend, jax_backend):
                for family in expected:
                    module = getattr(backend, family)
                    actual = numpy.asarray(module.coefficient(grid, power) if family == "NN" else module.coefficient(grid, power, redshift))
                    numpy.testing.assert_allclose(actual, expected[family], rtol=3e-8, atol=3e-12)
                    if family == "SS":
                        numpy.testing.assert_allclose(actual, actual.transpose(1, 0, 2), rtol=0, atol=1e-15)
                ns = numpy.asarray(backend.NS.coefficient(grid, power, redshift))
                sn = numpy.asarray(backend.SN.coefficient(grid, power, redshift))
                numpy.testing.assert_allclose(sn, ns.transpose(1, 0, 2), rtol=0, atol=0)

    def test_nn_terminal_contraction_and_zero_terminal_fixture(self):
        grid = numpy.array([0.0, 1.0, 2.0])
        power = numpy.array([[0.0, 1.0, 1.0]])
        for backend in (numba_backend, jax_backend):
            final = numpy.array([[0.0, 0.0, 1.0]])
            result = numpy.asarray(backend.NN.spectra(1.0, final, final, grid, power))
            numpy.testing.assert_allclose(result, 1.5-2*numpy.log(2), rtol=2e-13, atol=1e-15)
            first = numpy.array([[1.0, 0.0, 0.0]])
            result = numpy.asarray(backend.NN.spectra(1.0, first, first, grid, power))
            numpy.testing.assert_allclose(result, 1/12, rtol=2e-13, atol=1e-15)

    def test_tiny_positive_left_endpoint_retains_ordinary_power(self):
        # 1-chi1/chi2 rounds to one here, but chi1 is not the observer.
        # These limits integrate constant ordinary power, independently of J.
        left = 1e-20
        power = numpy.ones(1)
        for backend in (numba_backend, jax_backend):
            numpy.testing.assert_allclose(backend.NN.element1(left, 1.0, power, power), 1/left, rtol=2e-13)
            for family, number, expected in (("NS", 9, -0.5*numpy.log(left)-11/12), ("NS", 10, 1/6), ("SS", 11, 43/1440), ("SS", 12, 1/20)):
                actual = getattr(getattr(backend, family), f"element{number}")(left, 1.0, power, power, 0.0, 0.0)
                numpy.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)

    def test_nn_element_matches_across_backends(self):
        power1 = numpy.array([0.8, 1.1, 1.6], dtype=numpy.float64)
        power2 = numpy.array([1.0, 1.4, 2.0], dtype=numpy.float64)

        expected = numba_backend.NN.element1(0.7, 1.3, power1, power2)
        actual = numpy.asarray(jax_backend.NN.element1(0.7, 1.3, power1, power2))

        numpy.testing.assert_allclose(actual, expected, rtol=1.0e-11, atol=1.0e-13)

    def test_redshift_dependent_elements_match_across_backends(self):
        power1 = numpy.array([0.8, 1.1, 1.6], dtype=numpy.float64)
        power2 = numpy.array([1.0, 1.4, 2.0], dtype=numpy.float64)
        arguments = (0.7, 1.3, power1, power2, 0.2, 0.5)

        for numba_module, jax_module in (
            (numba_backend.NS, jax_backend.NS),
            (numba_backend.SN, jax_backend.SN),
            (numba_backend.SS, jax_backend.SS),
        ):
            with self.subTest(module=numba_module.__name__):
                expected = numba_module.element1(*arguments)
                actual = numpy.asarray(jax_module.element1(*arguments))
                numpy.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=1.0e-11,
                    atol=1.0e-13,
                )


if __name__ == "__main__":
    unittest.main()
