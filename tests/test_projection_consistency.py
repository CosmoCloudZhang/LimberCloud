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
