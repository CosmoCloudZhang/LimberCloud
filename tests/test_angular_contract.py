"""The shared angular operator and the method identity.

These checks import NumPy and SciPy only. They never import h5py, Numba, JAX or
CCL, so they stay login-safe on a site-linked HDF5 installation.
"""

import unittest

import numpy
import scipy.interpolate

from limbercloud.validation.estimator import (
    ANGULAR_BOUNDARY_CONDITION,
    CANONICAL_BAND_COUNT,
    CANONICAL_NODE_COUNT,
    AngularContract,
    AngularContractError,
    EllEstimator,
    EstimatorMismatch,
    assert_same_estimator,
    band_display_centres,
    bandpower_operator_matrix,
    canonical_angular_contract,
    canonical_band_estimator,
    canonical_ell_nodes,
    canonical_raw_estimator,
    historical_ccl_estimator,
    historical_limbercloud_estimator,
    legacy_covariance_input_ell,
    natural_spline_bandpowers,
)
from limbercloud.validation.method import MethodError, MethodIdentity


def not_a_knot_bandpowers(ell, cl):
    """Return bandpowers under SciPy's default boundary condition."""

    log_ell = numpy.log(ell)
    spline = scipy.interpolate.CubicSpline(log_ell, ell * cl, axis=-1)
    return numpy.array(
        [
            spline.integrate(log_ell[index], log_ell[index + 1]) / (ell[index + 1] - ell[index])
            for index in range(ell.size - 1)
        ]
    )


class CanonicalContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = canonical_angular_contract()
        self.ell = self.contract.ell

    def test_twentyone_nodes_and_twenty_bands(self):
        self.assertEqual(self.contract.node_count, CANONICAL_NODE_COUNT)
        self.assertEqual(self.contract.band_count, CANONICAL_BAND_COUNT)
        self.assertEqual(self.ell.dtype, numpy.float64)
        numpy.testing.assert_allclose(self.ell[0], 20.0)
        numpy.testing.assert_allclose(self.ell[-1], 2000.0)
        numpy.testing.assert_array_equal(self.ell, canonical_ell_nodes())
        numpy.testing.assert_allclose(
            self.contract.centres, band_display_centres(self.ell)
        )

    def test_raw_and_band_datasets_stay_distinct(self):
        raw = canonical_raw_estimator()
        band = canonical_band_estimator()
        self.assertEqual(len(raw.ell), 21)
        self.assertEqual(len(band.ell), 20)
        self.assertNotEqual(raw.fingerprint(), band.fingerprint())
        with self.assertRaises(EstimatorMismatch):
            assert_same_estimator(raw, band)

    def test_boundary_condition_and_version_enter_the_fingerprint(self):
        contract = canonical_angular_contract()
        self.assertEqual(contract.fingerprint(), self.contract.fingerprint())
        self.assertEqual(contract.boundary_condition, ANGULAR_BOUNDARY_CONDITION)
        # The historical helper shares coordinates but is a different identity.
        self.assertNotEqual(
            historical_limbercloud_estimator().fingerprint(),
            canonical_raw_estimator().fingerprint(),
        )
        with self.assertRaises(AngularContractError):
            AngularContract(
                nodes=tuple(self.ell),
                edges=tuple(self.ell),
                centres=tuple(self.contract.centres),
                boundary_condition="not-a-knot",
            )

    def test_natural_spline_is_not_not_a_knot(self):
        log_ell = numpy.log(self.ell)
        # A curved, non-polynomial signal separates the boundary conditions.
        cl = numpy.sin(log_ell) * numpy.exp(-log_ell) / self.ell
        natural = self.contract.bandpowers(cl)
        default = not_a_knot_bandpowers(self.ell, cl)
        self.assertGreater(numpy.max(numpy.abs(natural - default) / numpy.abs(natural)), 1e-3)
        numpy.testing.assert_allclose(
            natural,
            natural_spline_bandpowers(self.ell, cl),
            rtol=0.0,
            atol=0.0,
        )

    def test_notebook_formula_is_reproduced_term_by_term(self):
        log_ell = numpy.log(self.ell)
        cl = numpy.cos(2.0 * log_ell) / self.ell**1.5
        spline = scipy.interpolate.CubicSpline(
            log_ell, self.ell * cl, axis=-1, bc_type="natural"
        )
        expected = numpy.array(
            [
                spline.integrate(log_ell[index], log_ell[index + 1])
                / (self.ell[index + 1] - self.ell[index])
                for index in range(CANONICAL_BAND_COUNT)
            ]
        )
        numpy.testing.assert_allclose(self.contract.bandpowers(cl), expected, rtol=0.0, atol=0.0)

    def test_exact_fixture_with_ell_cl_linear_in_log_ell(self):
        log_ell = numpy.log(self.ell)
        # ell*C linear in log(ell) is reproduced exactly by any cubic spline,
        # so the band integral has an elementary closed form.
        cl = (3.0 + 2.0 * log_ell) / self.ell
        expected = numpy.array(
            [
                (
                    3.0 * (log_ell[index + 1] - log_ell[index])
                    + (log_ell[index + 1] ** 2 - log_ell[index] ** 2)
                )
                / (self.ell[index + 1] - self.ell[index])
                for index in range(CANONICAL_BAND_COUNT)
            ]
        )
        numpy.testing.assert_allclose(self.contract.bandpowers(cl), expected, rtol=1e-12, atol=0.0)

    def test_constant_spectrum_is_only_approximately_preserved(self):
        # Splining ell*C against log(ell) does not reproduce a constant C
        # exactly. This measures that interpolation error; it does not
        # renormalise the operator to force agreement.
        cl = numpy.full(CANONICAL_NODE_COUNT, 2.5)
        bands = self.contract.bandpowers(cl)
        error = numpy.abs(bands - 2.5) / 2.5
        # The measured worst-band error on the campaign grid is 1.8e-3.
        self.assertLess(numpy.max(error), 2.5e-3)
        self.assertGreater(numpy.max(error), 1.0e-4)

    def test_operator_is_linear_and_matches_its_matrix_for_batches(self):
        log_ell = numpy.log(self.ell)
        first = numpy.sin(log_ell) / self.ell
        second = -numpy.cos(3.0 * log_ell) / self.ell**2
        matrix = bandpower_operator_matrix(self.ell)
        self.assertEqual(matrix.shape, (CANONICAL_BAND_COUNT, CANONICAL_NODE_COUNT))
        numpy.testing.assert_allclose(
            self.contract.bandpowers(1.5 * first - 4.0 * second),
            1.5 * self.contract.bandpowers(first) - 4.0 * self.contract.bandpowers(second),
            rtol=1e-12,
            atol=1e-300,
        )
        batch = numpy.stack([first, second, numpy.zeros_like(first)])
        bands = self.contract.bandpowers(batch)
        self.assertEqual(bands.shape, (3, CANONICAL_BAND_COUNT))
        numpy.testing.assert_allclose(bands, (matrix @ batch.T).T, rtol=1e-12, atol=1e-300)
        numpy.testing.assert_array_equal(bands[2], 0.0)

    def test_independent_integration_of_the_same_piecewise_polynomial(self):
        log_ell = numpy.log(self.ell)
        cl = numpy.exp(-0.3 * log_ell) * numpy.sin(1.7 * log_ell) / self.ell
        spline = scipy.interpolate.CubicSpline(
            log_ell, self.ell * cl, axis=-1, bc_type="natural"
        )
        # Integrate the stored coefficients directly instead of calling
        # CubicSpline.integrate.
        expected = numpy.empty(CANONICAL_BAND_COUNT)
        for index in range(CANONICAL_BAND_COUNT):
            width = log_ell[index + 1] - log_ell[index]
            powers = numpy.array([width**4 / 4.0, width**3 / 3.0, width**2 / 2.0, width])
            expected[index] = float(spline.c[:, index] @ powers) / (
                self.ell[index + 1] - self.ell[index]
            )
        numpy.testing.assert_allclose(self.contract.bandpowers(cl), expected, rtol=1e-11, atol=0.0)

    def test_signed_and_zero_spectra_are_legitimate(self):
        log_ell = numpy.log(self.ell)
        cl = numpy.sin(4.0 * log_ell) / self.ell
        numpy.testing.assert_allclose(
            self.contract.bandpowers(-cl), -self.contract.bandpowers(cl), rtol=1e-12, atol=1e-300
        )
        numpy.testing.assert_array_equal(
            self.contract.bandpowers(numpy.zeros(CANONICAL_NODE_COUNT)), 0.0
        )

    def test_invalid_coordinates_and_shapes_are_rejected(self):
        cl = numpy.ones(CANONICAL_NODE_COUNT)
        with self.assertRaises(AngularContractError):
            natural_spline_bandpowers([1.0, 1.0, 2.0], numpy.ones(3))
        with self.assertRaises(AngularContractError):
            natural_spline_bandpowers([2.0, 1.0], numpy.ones(2))
        with self.assertRaises(AngularContractError):
            natural_spline_bandpowers([0.0, 1.0], numpy.ones(2))
        with self.assertRaises(AngularContractError):
            natural_spline_bandpowers([1.0, numpy.nan], numpy.ones(2))
        with self.assertRaises(AngularContractError):
            self.contract.bandpowers(numpy.ones(CANONICAL_BAND_COUNT))
        with self.assertRaises(AngularContractError):
            self.contract.bandpowers(numpy.full(CANONICAL_NODE_COUNT, numpy.inf))
        with self.assertRaises(AngularContractError):
            self.contract.validate_nodes(legacy_covariance_input_ell())
        numpy.testing.assert_array_equal(self.contract.validate_nodes(self.ell), self.ell)
        self.assertEqual(self.contract.bandpowers(cl).shape, (CANONICAL_BAND_COUNT,))

    def test_estimator_validates_its_own_coordinates(self):
        estimator = canonical_raw_estimator()
        estimator.validate_coordinates(self.ell)
        with self.assertRaises(EstimatorMismatch):
            estimator.validate_coordinates(self.ell[:3])

    def test_legacy_coordinates_remain_readable_and_distinct(self):
        centres = historical_ccl_estimator()
        self.assertTrue(centres.is_legacy)
        self.assertEqual(len(centres.ell), 20)
        self.assertEqual(legacy_covariance_input_ell().size, 101)
        with self.assertRaises(EstimatorMismatch):
            assert_same_estimator(centres, canonical_band_estimator())
        with self.assertRaises(ValueError):
            EllEstimator("invented_estimator", (20.0, 30.0))


class MethodIdentityTests(unittest.TestCase):
    def test_numeric_requires_an_order_and_others_refuse_one(self):
        numeric = MethodIdentity.create("numeric", None, "linear")
        self.assertEqual(numeric.interpolation, "LINEAR")
        self.assertEqual(numeric.device, "CPU")
        self.assertTrue(numeric.selects_order)
        self.assertEqual(numeric.label(), "NUMERIC-LINEAR")
        with self.assertRaises(MethodError):
            MethodIdentity.create("NUMERIC")
        with self.assertRaises(MethodError):
            MethodIdentity.create("NUMERIC", None, "quintic")
        for family in ("CCL", "NUMBA", "JAX"):
            with self.subTest(family=family):
                with self.assertRaises(MethodError):
                    MethodIdentity.create(family, "CPU", "linear")

    def test_jax_requires_a_device_and_others_are_cpu(self):
        self.assertEqual(MethodIdentity.create("JAX", "gpu").device, "GPU")
        self.assertEqual(MethodIdentity.create("JAX", "GPU").label(), "JAX-GPU")
        with self.assertRaises(MethodError):
            MethodIdentity.create("JAX")
        for family in ("CCL", "NUMBA", "NUMERIC"):
            with self.subTest(family=family):
                order = "cubic" if family == "NUMERIC" else None
                self.assertEqual(MethodIdentity.create(family, None, order).device, "CPU")
                with self.assertRaises(MethodError):
                    MethodIdentity.create(family, "GPU", order)

    def test_unknown_families_are_rejected(self):
        for family in ("", "  ", "NUMPY", None):
            with self.subTest(family=family):
                with self.assertRaises(MethodError):
                    MethodIdentity.create(family)


if __name__ == "__main__":
    unittest.main()
