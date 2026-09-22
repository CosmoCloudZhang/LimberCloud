"""Fast checks for the shared sample table, estimators and HDF5 artifacts."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy

from limbercloud.experiments.sample_controls import resolve_sample_count
from limbercloud.io.artifacts import (
    ArtifactError,
    ArtifactIdentity,
    NamespaceLock,
    SampleCheckpoint,
    completed_sample_ids,
    consolidate_probe,
    pending_sample_ids,
    publish_file,
    read_completed_manifest,
    read_fiducial_spectrum,
    spectra_basename,
    timing_basename,
    write_failure_record,
    write_manifest,
    write_sample_checkpoint,
)
from limbercloud.validation.assembly import (
    active_lensing_amplitude,
    analytical_magnification_response,
    assemble_configuration,
    ccl_magnification_bias,
    magnification_weighted_lens,
)
from limbercloud.validation.contract import (
    NN_OBSERVER_FACTORS,
    EtaIADecision,
    UnresolvedScienceDecision,
    multiplicative_bounds,
)
from limbercloud.validation.estimator import (
    EllEstimator,
    EstimatorMismatch,
    assert_same_estimator,
    geometric_centres,
    historical_ccl_estimator,
    historical_limbercloud_estimator,
    uniform_dell_band_average,
)
from limbercloud.validation.evaluate import evaluate_configuration
from limbercloud.validation.reference import (
    nn_interval_quadrature,
    nn_observer_closed_form,
    numeric_interpolation_contract,
    scale_factor_from_linear_a,
    scale_factor_from_linear_one_plus_z,
)
from limbercloud.validation.samples import (
    SAMPLED_PARAMETERS,
    SampleTableError,
    assert_campaign_request,
    evaluation_sample_ids,
    generate_cosmology_table,
    load_cosmology_table,
    sampled_parameter_rows,
    save_cosmology_table,
    select_rows,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

FIDUCIAL = {
    "H": 0.6736,
    "W0": -1.0,
    "WA": 0.0,
    "NS": 0.9649,
    "AS": 2.083e-09,
    "M_NU": 0.06,
    "N_EFF": 3.046,
    "OMEGA_B": 0.05,
    "OMEGA_K": 0.0,
    "OMEGA_CDM": 0.26,
}


def _identity() -> ArtifactIdentity:
    return ArtifactIdentity(
        run_id="pilot",
        survey="Y1",
        family="NUMBA",
        configuration="Single",
        sample_table_hash="abc",
        estimator_fingerprint=historical_limbercloud_estimator().fingerprint(),
        device="",
        interpolation="",
    )


def _record(sample_id: int, scale: float) -> SampleCheckpoint:
    ell = numpy.array([20.0, 40.0, 80.0], dtype=numpy.float64)
    pair_i = numpy.array([0, 0, 1], dtype=numpy.int32)
    pair_j = numpy.array([0, 1, 1], dtype=numpy.int32)
    cl = numpy.full((ell.size, pair_i.size), scale, dtype=numpy.float64)
    cl[0, 0] = scale + 0.25
    return SampleCheckpoint(
        sample_id=sample_id,
        is_fiducial=sample_id == 0,
        probe="EE",
        cosmology=numpy.array([FIDUCIAL[name] for name in SAMPLED_PARAMETERS], dtype=numpy.float64),
        parameter_names=SAMPLED_PARAMETERS,
        ell=ell,
        pair_i=pair_i,
        pair_j=pair_j,
        cl=cl,
        stage_seconds={"cosmology": 0.1, "projection": 0.2},
        estimator_name="sampled_nodes",
    )


class SampleControlAndTableTests(unittest.TestCase):
    def test_tiny_run_controls_select_ids_without_draws(self):
        self.assertEqual(resolve_sample_count(None, False), 0)
        self.assertEqual(evaluation_sample_ids(sample_count=None, fiducial_only=False, include_fiducial=False), [])
        self.assertEqual(evaluation_sample_ids(sample_count=None, fiducial_only=True, include_fiducial=False), [0])
        self.assertEqual(evaluation_sample_ids(sample_count=2, fiducial_only=False, include_fiducial=False), [1, 2])
        self.assertEqual(
            evaluation_sample_ids(sample_count=2, fiducial_only=False, include_fiducial=True),
            [0, 1, 2],
        )
        with self.assertRaises(ValueError):
            evaluation_sample_ids(sample_count=1, fiducial_only=True, include_fiducial=False)
        assert_campaign_request([0, *range(1, 1001)])
        with self.assertRaises(SampleTableError):
            assert_campaign_request([1, 2, 3])

    def test_table_is_seeded_shared_and_resumed_by_id(self):
        first = generate_cosmology_table(FIDUCIAL, seed=11, sampled_count=4)
        second = generate_cosmology_table(FIDUCIAL, seed=11, sampled_count=4)
        other = generate_cosmology_table(FIDUCIAL, seed=12, sampled_count=4)
        numpy.testing.assert_array_equal(first.values, second.values)
        self.assertFalse(numpy.array_equal(first.values, other.values))
        numpy.testing.assert_array_equal(first.values[0], [FIDUCIAL[name] for name in first.parameter_names])
        self.assertTrue(bool(first.is_fiducial[0]))
        self.assertFalse(bool(first.is_fiducial[1]))
        numpy.testing.assert_array_equal(first.values[:, first.parameter_names.index("WA")], 0.0)
        numpy.testing.assert_array_equal(first.values[:, first.parameter_names.index("OMEGA_K")], 0.0)
        low, high = multiplicative_bounds(-1.0)
        self.assertLess(low, high)
        w0 = first.values[1:, first.parameter_names.index("W0")]
        self.assertTrue(numpy.all(w0 >= low) and numpy.all(w0 <= high))

        rng = numpy.random.default_rng(11)
        for name in SAMPLED_PARAMETERS:
            column = first.parameter_names.index(name)
            expected = rng.uniform(*multiplicative_bounds(FIDUCIAL[name]), size=4)
            numpy.testing.assert_allclose(first.values[1:, column], expected)

        with tempfile.TemporaryDirectory() as temporary:
            directory = save_cosmology_table(temporary, first)
            loaded = load_cosmology_table(directory)
            self.assertEqual(loaded.content_hash, first.content_hash)
            rows = select_rows(loaded, [2, 0])
            self.assertEqual(rows[0]["H"], float(first.values[2, first.parameter_names.index("H")]))
            self.assertEqual(rows[1]["W0"], -1.0)
            self.assertEqual(
                sampled_parameter_rows(sample_count=0, sample_table=None, fiducial_only=True),
                [],
            )
            with self.assertRaises(SampleTableError):
                sampled_parameter_rows(sample_count=1, sample_table=None)
            with self.assertRaises(SampleTableError):
                sampled_parameter_rows(sample_count=1, sample_table=directory, include_fiducial=True)

    def test_nonzero_fixed_parameter_is_refused(self):
        fiducial = dict(FIDUCIAL)
        fiducial["WA"] = 0.1
        with self.assertRaises(SampleTableError):
            generate_cosmology_table(fiducial, seed=1, sampled_count=1)

    def test_eta_disagreement_is_explicit(self):
        decision = EtaIADecision.unresolved()
        self.assertEqual(decision.generator_value, 0.5)
        self.assertEqual(decision.manuscript_value, 0.0)
        with self.assertRaises(UnresolvedScienceDecision):
            _ = decision.resolved_value
        self.assertEqual(EtaIADecision.explicit(0.2).resolved_value, 0.2)


class EquivalenceContractTests(unittest.TestCase):
    def test_magnification_slope_and_response_are_distinct(self):
        slopes = numpy.array([0.4, 0.8])
        numpy.testing.assert_array_equal(ccl_magnification_bias(slopes), slopes)
        response = analytical_magnification_response(slopes)
        numpy.testing.assert_allclose(response, [0.0, 2.0])
        lens = numpy.arange(6, dtype=numpy.float64).reshape(2, 3)
        weighted = magnification_weighted_lens(lens, response)
        numpy.testing.assert_array_equal(weighted[0], 0.0)
        numpy.testing.assert_allclose(weighted[1], 2.0 * lens[1])

    def test_active_prefactor_follows_the_sample(self):
        fiducial = active_lensing_amplitude(0.3, 0.67)
        shifted = active_lensing_amplitude(0.33, 0.7)
        self.assertGreater(abs(shifted - fiducial) / fiducial, 0.1)

    def test_configuration_assembly_sums_components(self):
        ones = numpy.ones((2, 2))
        components = {
            "EE": {name: ones for name in ("SS", "SI", "IS", "II")},
            "TE": {name: ones for name in ("MS", "MI", "GS", "GI")},
            "TT": {name: ones * 3 for name in ("MM", "MG", "GM", "GG")},
        }
        single = assemble_configuration("Single", components)
        self.assertEqual(set(single), {"EE"})
        numpy.testing.assert_array_equal(single["EE"], 4.0)
        triple = evaluate_configuration(
            "Triple",
            components,
            estimator=historical_limbercloud_estimator(),
            eta_ia=EtaIADecision.unresolved(),
        )
        numpy.testing.assert_array_equal(triple["TT"], 12.0)
        with self.assertRaises(EstimatorMismatch):
            evaluate_configuration(
                "Single",
                {"EE": components["EE"]},
                estimator=historical_limbercloud_estimator(),
                reference_estimator=historical_ccl_estimator(),
            )

    def test_historical_ell_estimators_do_not_match(self):
        edges = historical_limbercloud_estimator()
        centres = historical_ccl_estimator()
        self.assertEqual(len(edges.ell), 21)
        self.assertEqual(len(centres.ell), 20)
        with self.assertRaises(EstimatorMismatch):
            assert_same_estimator(edges, centres)
        numpy.testing.assert_allclose(geometric_centres(numpy.asarray(edges.ell)), centres.ell)

    def test_uniform_dell_window_is_not_a_node_sample(self):
        log_ell = numpy.array([0.0, 1.0, 2.0])
        ell = numpy.exp(log_ell)
        cl = log_ell / ell
        bands = uniform_dell_band_average(ell, cl)
        self.assertEqual(bands.shape, (2,))
        self.assertAlmostEqual(bands[0], 0.5 / (numpy.e - 1.0))

    def test_numeric_linear_is_not_the_analytic_integrand(self):
        contract = numeric_interpolation_contract("linear")
        self.assertEqual(contract["scipy_kind"], "slinear")
        self.assertIn("not linear 1+z", contract["scale_factor"])
        midpoint_a = scale_factor_from_linear_a(0.5, 0.0, 1.0)
        midpoint_z = scale_factor_from_linear_one_plus_z(0.5, 0.0, 1.0)
        self.assertNotAlmostEqual(midpoint_a, midpoint_z)

    def test_nn_observer_factor_remains_one_quarter(self):
        self.assertEqual(NN_OBSERVER_FACTORS["element1"], 1.0 / 12.0)
        self.assertEqual(NN_OBSERVER_FACTORS["element2"], 1.0 / 12.0)
        self.assertEqual(NN_OBSERVER_FACTORS["element3"], 0.25)
        self.assertNotEqual(NN_OBSERVER_FACTORS["element3"], 0.5)
        for name, factor in NN_OBSERVER_FACTORS.items():
            closed = nn_observer_closed_form(2.0, 3.0, name)
            quadrature = nn_interval_quadrature(0.0, 2.0, 99.0, 3.0, name)
            self.assertAlmostEqual(closed, 3.0 / 2.0 * factor)
            self.assertAlmostEqual(quadrature, closed, places=8)

    def test_known_driver_defects_are_wired_to_the_contract(self):
        spectra = REPOSITORY_ROOT / "experiments" / "spectra"
        for path in spectra.rglob("*.py"):
            if path.name == "generate_samples.py":
                continue
            text = path.read_text()
            relative = path.relative_to(spectra).as_posix()
            if "/double.py" in f"/{relative}" or relative.endswith("double.py") or relative.endswith("triple.py"):
                if relative.startswith("CCL/"):
                    self.assertIn("ccl_magnification_bias", text)
                else:
                    self.assertIn("analytical_magnification_response", text)
                    self.assertIn("factor_ms", text)
                    self.assertIn(
                        "factor=numpy.array(factor_ms, dtype=numpy.float64),\n"
                        "            phi_a_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis]",
                        text,
                    )
            if relative.startswith("JAX/") and relative.endswith("single.py"):
                self.assertIn("cell_data_ee = cell_data_ss + cell_data_si + cell_data_is + cell_data_ii", text)
            if relative.startswith("JAX/") and relative.endswith("triple.py"):
                self.assertIn("cell_data_te = cell_data_ms + cell_data_mi + cell_data_gs + cell_data_gi", text)
                self.assertIn("cell_data_tt = cell_data_mm + cell_data_mg + cell_data_gm + cell_data_gg", text)


class EndpointOracleTests(unittest.TestCase):
    def test_interior_nn_quadrature_matches_closed_form(self):
        try:
            from limbercloud.projection.numba_backend import nn as nn_backend
        except ImportError:
            self.skipTest("Numba projection backend is unavailable")
        power_left = numpy.array([1.0])
        power_right = numpy.array([1.7])
        for index, name in enumerate(("element1", "element2", "element3"), start=1):
            analytical = getattr(nn_backend, name)(1.0, 2.5, power_left, power_right)
            quadrature = nn_interval_quadrature(1.0, 2.5, 1.0, 1.7, name)
            numpy.testing.assert_allclose(analytical, quadrature, rtol=1e-8, atol=1e-10)
            self.assertEqual(index, index)

    def test_final_nn_diagonal_is_omitted_and_ns_subdiagonal_exists(self):
        try:
            from limbercloud.projection.numba_backend import nn as nn_backend
            from limbercloud.projection.numba_backend import ns as ns_backend
            from limbercloud.projection.numba_backend import ss as ss_backend
        except ImportError:
            self.skipTest("Numba projection backend is unavailable")
        chi = numpy.array([0.0, 1.0, 2.0], dtype=numpy.float64)
        power = numpy.ones((1, 3), dtype=numpy.float64)
        coefficients = nn_backend.coefficient(chi, power)
        self.assertEqual(coefficients[2, 2, 0], 0.0)
        missing = nn_interval_quadrature(1.0, 2.0, 1.0, 1.0, "element3")
        self.assertGreater(abs(missing), 0.0)
        observer = nn_backend.element3(0.0, 1.0, numpy.array([5.0]), numpy.array([1.0]))
        numpy.testing.assert_allclose(observer, nn_observer_closed_form(1.0, 1.0, "element3"))

        radial = numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float64)
        redshift = numpy.array([0.1, 0.3, 0.6, 1.0], dtype=numpy.float64)
        matter = numpy.array([[1.0, 1.2, 0.8, 1.1]], dtype=numpy.float64)
        ns_coefficients = ns_backend.coefficient(radial, matter, redshift)
        self.assertGreater(abs(ns_coefficients[1, 0, 0]), 0.0)
        ss_coefficients = ss_backend.coefficient(radial, matter, redshift)
        self.assertTrue(numpy.isfinite(ss_coefficients[3, 3, 0]))
        self.assertGreater(abs(ss_coefficients[3, 3, 0]), 0.0)


class ArtifactPublicationTests(unittest.TestCase):
    def test_checkpoint_resume_and_manifest(self):
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            namespace = Path(temporary) / "run"
            lock = NamespaceLock(namespace)
            lock.acquire()
            try:
                with self.assertRaises(ArtifactError):
                    NamespaceLock(namespace).acquire()
                write_sample_checkpoint(namespace, _record(0, 1.0), identity, lock)
                self.assertEqual(completed_sample_ids(namespace, identity, "EE"), [0])
                self.assertEqual(pending_sample_ids([0, 1], [0]), [1])
                with self.assertRaises(ArtifactError):
                    write_sample_checkpoint(namespace, _record(0, 9.0), identity, lock)
                corrupt = namespace / "checkpoints" / "sample_000001_EE.h5"
                corrupt.parent.mkdir(parents=True, exist_ok=True)
                corrupt.write_bytes(b"not an hdf5 file")
                self.assertEqual(completed_sample_ids(namespace, identity, "EE"), [0])
                write_sample_checkpoint(namespace, _record(1, 2.0), identity, lock)
                partial = namespace / "checkpoints" / ".sample_000002_EE.h5.partial"
                partial.write_bytes(b"interrupted")
                self.assertNotIn(2, completed_sample_ids(namespace, identity, "EE"))
                product = consolidate_probe(
                    namespace,
                    identity,
                    lock,
                    probe="EE",
                    configuration="Single",
                    allocation=4,
                    sample_ids=[0, 1],
                )
                self.assertEqual(product.name, spectra_basename("Single", 4, "EE"))
                import h5py

                with h5py.File(product, "r") as handle:
                    self.assertEqual(handle["cl"].shape, (2, 3, 3))
                    self.assertNotIn("coefficients", handle)
                    self.assertIn("sampled", handle)
                    self.assertNotIn("bandpower", handle)
                with self.assertRaises(ArtifactError):
                    read_completed_manifest(namespace / "Manifest_Single_4.json")
                write_failure_record(
                    namespace,
                    sample_id=2,
                    probe="EE",
                    message="projection failed",
                    identity=identity,
                    namespace_lock=lock,
                )
                manifest = write_manifest(
                    namespace,
                    lock,
                    identity=identity,
                    configuration="Single",
                    allocation=4,
                    products={product.name: product},
                    completed_sample_ids=[0, 1],
                    failed_sample_ids=[2],
                    eta_ia=EtaIADecision.unresolved().as_dict(),
                )
                loaded = read_completed_manifest(manifest)
                self.assertEqual(loaded["failed_sample_ids"], [2])
                self.assertEqual(loaded["eta_ia"]["status"], "unresolved")
                fiducial = read_fiducial_spectrum(manifest, "EE")
                numpy.testing.assert_allclose(fiducial, _record(0, 1.0).cl)
                product.write_bytes(product.read_bytes() + b"\0")
                with self.assertRaises(ArtifactError):
                    read_completed_manifest(manifest)
            finally:
                lock.release()

    def test_cross_filesystem_publication_checks_the_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"spectra")
            destination = Path(temporary) / "dest" / "spectra.bin"
            with mock.patch("limbercloud.io.artifacts.same_filesystem", return_value=False):
                published = publish_file(source, destination)
            self.assertEqual(published.read_bytes(), b"spectra")
            self.assertFalse(source.exists())

    def test_stale_lock_on_this_host_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = NamespaceLock(temporary)
            lock.acquire()
            lock.release()
            stale = NamespaceLock(temporary)
            stale.path.write_text(json.dumps({"pid": 2**22, "host": os.uname().nodename}) + "\n")
            # socket.gethostname may differ from uname nodename; write the same host the lock uses.
            import socket

            stale.path.write_text(json.dumps({"pid": 2**22, "host": socket.gethostname()}) + "\n")
            resumed = NamespaceLock(temporary)
            resumed.acquire(resume_stale=True)
            resumed.release()

    def test_timing_names_keep_order_and_legacy_shape(self):
        self.assertEqual(timing_basename("Triple", 128, "_COSMOLOGY"), "Time_Triple_128_COSMOLOGY.txt")
        self.assertEqual(
            timing_basename("Triple", 128, "_CELL", "linear"),
            "Time_Triple_128_LINEAR_CELL.txt",
        )
        estimator = EllEstimator("sampled_nodes", (20.0, 40.0))
        other = EllEstimator("geometric_centres", (20.0, 40.0))
        with self.assertRaises(EstimatorMismatch):
            assert_same_estimator(estimator, other)


if __name__ == "__main__":
    unittest.main()
