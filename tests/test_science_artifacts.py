"""Shared sample table, nuisance provenance, run guards and HDF5 artifacts.

This suite imports h5py. On an installation whose h5py is linked against a
site MPI build it is not automatically login-safe; run it through
``make test-science`` under a supported allocation.
"""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy

from limbercloud.experiments.run_guards import (
    RunControlError,
    prepare_result_directory,
    require_effective_work,
)
from limbercloud.experiments.sample_controls import resolve_sample_count
from limbercloud.io.artifacts import (
    ArtifactError,
    ArtifactIdentity,
    NamespaceLock,
    SampleCheckpoint,
    completed_sample_ids,
    consolidate_probe,
    manifest_basename,
    pending_sample_ids,
    publish_file,
    read_completed_manifest,
    read_fiducial_spectrum,
    samples_timing_basename,
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
    component_activity,
    magnification_weighted_lens,
)
from limbercloud.validation.contract import (
    A_IA_AMPLITUDE,
    ETA_IA_ADOPTED_VALUE,
    ETA_IA_HISTORICAL_GENERATOR_VALUE,
    NN_FINAL_DIAGONAL_POLICY,
    NN_OBSERVER_FACTORS,
    EtaIADecision,
    UnresolvedScienceDecision,
    multiplicative_bounds,
    require_accepted_eta,
)
from limbercloud.validation.cosmology import (
    FIDUCIAL_SOLVER,
    CosmologyContractError,
    effective_cosmology_kwargs,
    model_fingerprint,
    parameter_hash,
    sample_limber_power,
)
from limbercloud.validation.estimator import (
    EllEstimator,
    EstimatorMismatch,
    canonical_angular_contract,
    canonical_raw_estimator,
)
from limbercloud.validation.evaluate import (
    assemble_with_bandpowers,
    evaluate_configuration,
)
from limbercloud.validation.method import MethodError
from limbercloud.validation.nuisance import (
    NuisanceArtifactError,
    load_alignment,
    load_galaxy_bias,
    load_magnification_slope,
    require_nuisance_compatibility,
)
from limbercloud.validation.reference import (
    nn_interval_quadrature,
    nn_observer_closed_form,
    numeric_interpolation_contract,
    scale_factor_from_linear_a,
    scale_factor_from_linear_one_plus_z,
)
from limbercloud.validation.samples import (
    PRIMARY_PARAMETERS,
    SAMPLED_PARAMETERS,
    SampleTableError,
    assert_campaign_request,
    ccl_cosmology_kwargs,
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
    "OMEGA_GAMMA": 5.45023999774456e-05,
}

RECORD_ELL = numpy.array([20.0, 40.0, 80.0], dtype=numpy.float64)
RECORD_ESTIMATOR = EllEstimator(
    "raw_nodes",
    tuple(float(value) for value in RECORD_ELL),
    tuple(float(value) for value in RECORD_ELL),
    boundary_condition="",
    operator_version="test.fixture.v1",
)


def _identity(**overrides) -> ArtifactIdentity:
    fields = {
        "run_id": "pilot",
        "survey": "Y1",
        "family": "NUMBA",
        "configuration": "Single",
        "sample_table_hash": "abc",
        "estimator_fingerprint": RECORD_ESTIMATOR.fingerprint(),
        "estimator": RECORD_ESTIMATOR,
    }
    fields.update(overrides)
    return ArtifactIdentity(**fields)


def _record(sample_id: int, scale: float) -> SampleCheckpoint:
    pair_i = numpy.array([0, 0, 1], dtype=numpy.int32)
    pair_j = numpy.array([0, 1, 1], dtype=numpy.int32)
    cl = numpy.full((RECORD_ELL.size, pair_i.size), scale, dtype=numpy.float64)
    cl[0, 0] = scale + 0.25
    return SampleCheckpoint(
        sample_id=sample_id,
        is_fiducial=sample_id == 0,
        probe="EE",
        cosmology=numpy.array(
            [FIDUCIAL[name] for name in SAMPLED_PARAMETERS], dtype=numpy.float64
        ),
        parameter_names=SAMPLED_PARAMETERS,
        ell=RECORD_ELL,
        pair_i=pair_i,
        pair_j=pair_j,
        cl=cl,
        stage_seconds={"cosmology": 0.1, "projection": 0.2},
        estimator_name=RECORD_ESTIMATOR.name,
    )


class SampleControlAndTableTests(unittest.TestCase):
    def test_unsupported_table_schemas_are_rejected_before_use(self):
        table = generate_cosmology_table(FIDUCIAL, seed=11, sampled_count=1)
        with tempfile.TemporaryDirectory() as temporary:
            directory = save_cosmology_table(temporary, table)
            manifest_path = directory / "Manifest.json"
            original = json.loads(manifest_path.read_text())
            for key in ("schema_version", "evaluation_schema_version"):
                with self.subTest(key=key):
                    manifest_path.write_text(json.dumps(dict(original, **{key: "old"})))
                    with self.assertRaisesRegex(SampleTableError, key):
                        load_cosmology_table(directory)

    def test_tiny_run_controls_select_ids_without_draws(self):
        self.assertEqual(resolve_sample_count(None), 0)
        self.assertEqual(evaluation_sample_ids(sample_count=None), [0])
        self.assertEqual(evaluation_sample_ids(sample_count=0), [0])
        self.assertEqual(evaluation_sample_ids(sample_count=2), [0, 1, 2])
        with self.assertRaises(ValueError):
            evaluation_sample_ids(sample_count=-1)
        assert_campaign_request([0, *range(1, 1001)])
        with self.assertRaises(SampleTableError):
            assert_campaign_request([1, 2, 3])

    def test_table_is_seeded_shared_and_resumed_by_id(self):
        first = generate_cosmology_table(FIDUCIAL, seed=11, sampled_count=4)
        second = generate_cosmology_table(FIDUCIAL, seed=11, sampled_count=4)
        other = generate_cosmology_table(FIDUCIAL, seed=12, sampled_count=4)
        numpy.testing.assert_array_equal(first.values, second.values)
        self.assertFalse(numpy.array_equal(first.values, other.values))
        numpy.testing.assert_array_equal(
            first.values[0], [FIDUCIAL[name] for name in first.parameter_names]
        )
        self.assertTrue(bool(first.is_fiducial[0]))
        self.assertFalse(bool(first.is_fiducial[1]))
        numpy.testing.assert_array_equal(
            first.values[:, first.parameter_names.index("WA")], 0.0
        )
        numpy.testing.assert_array_equal(
            first.values[:, first.parameter_names.index("OMEGA_K")], 0.0
        )
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
            self.assertEqual(
                rows[0]["H"], float(first.values[2, first.parameter_names.index("H")])
            )
            self.assertEqual(rows[1]["W0"], -1.0)
            with self.assertRaises(SampleTableError):
                sampled_parameter_rows(sample_count=0, sample_table=None)
            fiducial_only = sampled_parameter_rows(
                sample_count=0, sample_table=directory
            )
            self.assertEqual(len(fiducial_only), 1)
            self.assertEqual(fiducial_only[0]["W0"], -1.0)
            sampled = sampled_parameter_rows(sample_count=2, sample_table=directory)
            self.assertEqual(len(sampled), 3)
            with self.assertRaises(SampleTableError):
                sampled_parameter_rows(sample_count=1, sample_table=None)

    def test_photon_density_is_fixed_and_recorded_rather_than_sampled(self):
        table = generate_cosmology_table(FIDUCIAL, seed=3, sampled_count=5)
        self.assertIn("OMEGA_GAMMA", PRIMARY_PARAMETERS)
        self.assertNotIn("OMEGA_GAMMA", SAMPLED_PARAMETERS)
        column = table.parameter_names.index("OMEGA_GAMMA")
        numpy.testing.assert_array_equal(
            table.values[:, column], FIDUCIAL["OMEGA_GAMMA"]
        )
        self.assertEqual(table.solver_fingerprint, FIDUCIAL_SOLVER.fingerprint())

    def test_nondefault_half_width_round_trips_through_the_manifest(self):
        table = generate_cosmology_table(
            FIDUCIAL, seed=5, sampled_count=3, half_width=0.25
        )
        self.assertEqual(table.half_width, 0.25)
        with tempfile.TemporaryDirectory() as temporary:
            directory = save_cosmology_table(temporary, table)
            manifest = json.loads((Path(directory) / "Manifest.json").read_text())
            self.assertEqual(manifest["half_width"], 0.25)
            self.assertEqual(load_cosmology_table(directory).half_width, 0.25)

    def test_nonzero_fixed_parameter_is_refused(self):
        fiducial = dict(FIDUCIAL)
        fiducial["WA"] = 0.1
        with self.assertRaises(SampleTableError):
            generate_cosmology_table(fiducial, seed=1, sampled_count=1)


class EffectiveCosmologyTests(unittest.TestCase):
    def test_limber_power_queries_only_positive_nodes_and_preserves_errors(self):
        calls = []

        def provider(*, cosmo, k, a):
            self.assertEqual(cosmo, "active")
            self.assertTrue(numpy.all(numpy.isfinite(k)))
            self.assertLess(float(numpy.max(k)), 100.0)
            calls.append((k.copy(), a))
            return 2.0 * k

        power = sample_limber_power(
            "active", [0.0, 1.0, 2.0], [1.0, 0.8, 0.5], [20.0, 40.0],
            power_provider=provider,
        )
        self.assertEqual(len(calls), 2)
        numpy.testing.assert_array_equal(power[:, 0], 0.0)
        numpy.testing.assert_allclose(power[:, 1], [41.0, 81.0])
        with self.assertRaisesRegex(CosmologyContractError, "Power provider"):
            sample_limber_power(
                None, [0.0, 1.0], [1.0, 0.5], [20.0],
                power_provider=lambda **kwargs: numpy.array([numpy.nan]),
            )

    def test_generators_and_the_sampled_constructor_share_one_model(self):
        table = generate_cosmology_table(FIDUCIAL, seed=7, sampled_count=2)
        sampled = ccl_cosmology_kwargs(table.row_dict(0))
        generator = effective_cosmology_kwargs(FIDUCIAL)
        self.assertEqual(sampled, generator)
        self.assertEqual(sampled["Omega_g"], FIDUCIAL["OMEGA_GAMMA"])
        self.assertEqual(sampled["extra_parameters"]["camb"]["kmax"], 100)

    def test_the_solver_specification_is_recorded_and_hashed(self):
        settings = FIDUCIAL_SOLVER.as_dict()
        self.assertIn("radiation_policy", settings)
        self.assertIn("temperature_policy", settings)
        self.assertIn("neutrino_policy", settings)
        self.assertEqual(settings["transfer_function"], "boltzmann_camb")
        self.assertEqual(len(FIDUCIAL_SOLVER.fingerprint()), 64)

    def test_the_model_fingerprint_separates_parameters_from_settings(self):
        shifted = dict(FIDUCIAL, H=0.7)
        self.assertNotEqual(parameter_hash(FIDUCIAL), parameter_hash(shifted))
        self.assertNotEqual(model_fingerprint(FIDUCIAL), model_fingerprint(shifted))
        # Derived keys that the generator JSON also carries do not change it.
        self.assertEqual(
            parameter_hash(FIDUCIAL), parameter_hash(dict(FIDUCIAL, OMEGA_M=0.3152))
        )
        with self.assertRaises(CosmologyContractError):
            effective_cosmology_kwargs({name: 1.0 for name in ("H", "W0")})


class IntrinsicAlignmentDecisionTests(unittest.TestCase):
    def test_eta_is_adopted_at_zero_with_the_historical_value_as_provenance(self):
        decision = EtaIADecision.adopted()
        self.assertEqual(decision.resolved_value, 0.0)
        self.assertEqual(ETA_IA_ADOPTED_VALUE, 0.0)
        self.assertTrue(decision.is_accepted)
        record = decision.as_dict()
        self.assertEqual(record["status"], "adopted")
        self.assertEqual(
            record["historical_generator_value"], ETA_IA_HISTORICAL_GENERATOR_VALUE
        )
        # The distinct amplitude is not changed with the slope.
        self.assertEqual(A_IA_AMPLITUDE, 0.5)

    def test_an_alternate_slope_is_explicitly_diagnostic(self):
        diagnostic = EtaIADecision.diagnostic(0.5)
        self.assertFalse(diagnostic.is_accepted)
        self.assertEqual(diagnostic.resolved_value, 0.5)
        with self.assertRaises(ValueError):
            EtaIADecision(value=0.5)

    def test_stale_or_unresolved_metadata_fails_acceptance(self):
        self.assertEqual(
            require_accepted_eta({"eta_pivot": 0.0, "eta_decision": "adopted"}), 0.0
        )
        for record in (
            {},
            {"eta_pivot": 0.5, "eta_decision": "adopted"},
            {"eta_pivot": 0.0, "eta_decision": "unresolved"},
            {"eta_pivot": 0.0},
        ):
            with self.subTest(record=record):
                with self.assertRaises(UnresolvedScienceDecision):
                    require_accepted_eta(record)


class NuisanceArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.redshift = numpy.linspace(0.0, 3.5, 351)

    def tearDown(self):
        self.temporary.cleanup()

    def write_alignment(self, **overrides):
        record = {
            "A": (-numpy.ones_like(self.redshift)).tolist(),
            "redshift": self.redshift.tolist(),
            "eta_pivot": 0.0,
            "eta_decision": "adopted",
            "nuisance_cosmology_policy": "fixed_tabulated_at_fiducial",
            "fiducial_input_hash": parameter_hash(FIDUCIAL),
            "generating_model_fingerprint": model_fingerprint(FIDUCIAL),
            "solver_fingerprint": FIDUCIAL_SOLVER.fingerprint(),
        }
        record.update(overrides)
        path = self.root / "intrinsic_alignment.json"
        path.write_text(json.dumps(record))
        return path

    def test_an_accepted_alignment_table_loads_with_its_provenance(self):
        amplitude, provenance = load_alignment(self.write_alignment(), self.redshift)
        self.assertEqual(amplitude.shape, self.redshift.shape)
        self.assertEqual(provenance.solver_fingerprint, FIDUCIAL_SOLVER.fingerprint())
        self.assertTrue(numpy.all(amplitude < 0.0))

    def test_fixed_provenance_binds_to_sample_zero_not_sampled_rows(self):
        _, provenance = load_alignment(self.write_alignment(), self.redshift)
        table = generate_cosmology_table(FIDUCIAL, seed=5, sampled_count=2)
        self.assertNotEqual(parameter_hash(table.row_dict(1)), parameter_hash(FIDUCIAL))
        require_nuisance_compatibility(table, provenance)
        require_nuisance_compatibility(table, replace(provenance, name="galaxy_bias"))
        for field in (
            "fiducial_input_hash", "generating_model_fingerprint", "solver_fingerprint"
        ):
            with self.subTest(field=field):
                mismatched = replace(provenance, **{field: "different-but-nonempty"})
                with self.assertRaisesRegex(NuisanceArtifactError, field):
                    require_nuisance_compatibility(table, mismatched)
        with self.assertRaisesRegex(NuisanceArtifactError, "solver_fingerprint"):
            require_nuisance_compatibility(
                replace(table, solver_fingerprint="different"), provenance
            )
        shifted = generate_cosmology_table(dict(FIDUCIAL, H=0.7), seed=5, sampled_count=0)
        with self.assertRaisesRegex(NuisanceArtifactError, "fiducial_input_hash"):
            require_nuisance_compatibility(shifted, provenance)
        with self.assertRaisesRegex(NuisanceArtifactError, "mark only sample 0"):
            require_nuisance_compatibility(
                replace(table, is_fiducial=numpy.array([False, True, False])), provenance
            )

    def test_disabled_components_use_amplitudes_and_preserve_active_legs(self):
        no_ia = component_activity(numpy.zeros(3), [1.0])
        self.assertEqual(
            {name for name, active in no_ia.items() if not active},
            {"SI", "IS", "II", "MI", "GI"},
        )
        no_mag = component_activity([-0.5, -0.5], analytical_magnification_response([0.4]))
        self.assertEqual(
            {name for name, active in no_mag.items() if not active},
            {"MS", "MI", "MM", "MG", "GM"},
        )
        both_off = component_activity([0.0], [0.0])
        self.assertEqual({name for name, active in both_off.items() if active}, {"SS", "GS", "GG"})
        self.assertTrue(all(component_activity([-1e-200], [-2.0]).values()))
        mixed = component_activity([-0.5], [0.0, 2.0])
        self.assertTrue(all(mixed.values()))
        weighted = magnification_weighted_lens(numpy.ones((2, 3)), [0.0, 2.0])
        numpy.testing.assert_array_equal(weighted[0], 0.0)
        numpy.testing.assert_array_equal(weighted[1], 2.0)
        for invalid in ([numpy.nan], [numpy.inf], []):
            with self.assertRaises(ValueError):
                component_activity(invalid, [0.0])

    def test_stale_eta_missing_provenance_and_grid_mismatch_are_refused(self):
        with self.assertRaises(UnresolvedScienceDecision):
            load_alignment(
                self.write_alignment(eta_pivot=0.5, eta_decision="adopted"),
                self.redshift,
            )
        with self.assertRaises(UnresolvedScienceDecision):
            load_alignment(
                self.write_alignment(eta_decision="unresolved"), self.redshift
            )
        with self.assertRaises(NuisanceArtifactError):
            load_alignment(
                self.write_alignment(generating_model_fingerprint=""), self.redshift
            )
        with self.assertRaises(NuisanceArtifactError):
            load_alignment(self.write_alignment(), numpy.linspace(0.0, 3.5, 100))
        with self.assertRaises(NuisanceArtifactError):
            load_alignment(
                self.write_alignment(nuisance_cosmology_policy="per_sample"),
                self.redshift,
            )

    def test_galaxy_bias_requires_its_redshift_axis_and_model_hash(self):
        record = {
            "Y1": numpy.ones_like(self.redshift).tolist(),
            "_redshift": self.redshift.tolist(),
            "_policy": "fixed_tabulated_at_fiducial",
            "_fiducial_input_hash": parameter_hash(FIDUCIAL),
            "_generating_model_fingerprint": model_fingerprint(FIDUCIAL),
            "_solver_fingerprint": FIDUCIAL_SOLVER.fingerprint(),
        }
        path = self.root / "galaxy_bias.json"
        path.write_text(json.dumps(record))
        bias, provenance = load_galaxy_bias(path, "Y1", self.redshift)
        self.assertEqual(bias.shape, self.redshift.shape)
        self.assertEqual(provenance.name, "galaxy_bias")
        with self.assertRaises(NuisanceArtifactError):
            load_galaxy_bias(path, "Y10", self.redshift)
        path.write_text(
            json.dumps(
                {key: value for key, value in record.items() if key != "_redshift"}
            )
        )
        with self.assertRaises(NuisanceArtifactError):
            load_galaxy_bias(path, "Y1", self.redshift)

    def test_magnification_slopes_must_declare_that_they_are_slopes(self):
        path = self.root / "magnification_bias.json"
        path.write_text(
            json.dumps({"Y1": [0.4, 0.8], "_quantity": "magnification_slope_s"})
        )
        numpy.testing.assert_allclose(load_magnification_slope(path, "Y1"), [0.4, 0.8])
        path.write_text(json.dumps({"Y1": [0.4, 0.8]}))
        with self.assertRaises(NuisanceArtifactError):
            load_magnification_slope(path, "Y1")


class RunGuardTests(unittest.TestCase):
    def test_the_fiducial_still_needs_the_saved_table(self):
        with self.assertRaises(RunControlError):
            require_effective_work(sample_count=0, sample_table=None)
        self.assertEqual(
            require_effective_work(sample_count=0, sample_table="/table"),
            0,
        )

    def test_a_real_run_needs_the_saved_table(self):
        with self.assertRaises(RunControlError):
            require_effective_work(sample_count=3, sample_table=None)
        self.assertEqual(
            require_effective_work(sample_count=3, sample_table="/table"),
            3,
        )

    def test_a_rerun_may_replace_timing_products_in_the_same_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "Y1"
            prepare_result_directory(directory)
            payload = "1.0\n2.0\n"
            (directory / "Time_Single.txt").write_text(payload)
            self.assertEqual(prepare_result_directory(directory), directory)
            self.assertEqual((directory / "Time_Single.txt").read_text(), payload)


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

    def test_nonuniform_slopes_weight_each_lens_bin_separately(self):
        slopes = numpy.array([0.4, 0.6, 1.0])
        response = analytical_magnification_response(slopes)
        numpy.testing.assert_allclose(response, [0.0, 1.0, 3.0])
        lens = numpy.ones((3, 4))
        weighted = magnification_weighted_lens(lens, response)
        numpy.testing.assert_allclose(weighted[:, 0], response)
        with self.assertRaises(ValueError):
            magnification_weighted_lens(lens, response[:2])

    def test_active_prefactor_follows_the_sample(self):
        fiducial = active_lensing_amplitude(0.3, 0.67)
        shifted = active_lensing_amplitude(0.33, 0.7)
        self.assertGreater(abs(shifted - fiducial) / fiducial, 0.1)

    def test_configuration_assembly_sums_components(self):
        ones = numpy.ones((2, 3))
        components = {
            "EE": {name: ones for name in ("SS", "SI", "IS", "II")},
            "TE": {name: ones for name in ("MS", "MI", "GS", "GI")},
            "TT": {name: ones * 3 for name in ("MM", "MG", "GM", "GG")},
        }
        single = assemble_configuration("Single", components)
        self.assertEqual(set(single), {"EE"})
        numpy.testing.assert_array_equal(single["EE"], 4.0)
        estimator = EllEstimator("raw_nodes", (20.0, 40.0, 80.0))
        triple = evaluate_configuration("Triple", components, estimator=estimator)
        numpy.testing.assert_array_equal(triple["TT"], 12.0)

    def test_incompatible_component_shapes_are_not_broadcast(self):
        components = {
            "EE": {
                "SS": numpy.ones((2, 3)),
                "SI": numpy.ones((2, 3)),
                "IS": numpy.ones((2, 3)),
                "II": numpy.ones((3,)),
            }
        }
        with self.assertRaises(ValueError):
            assemble_configuration("Single", components)

    def test_every_method_shares_the_same_coordinates_and_bandpowers(self):
        contract = canonical_angular_contract()
        log_ell = numpy.log(contract.ell)
        component = numpy.sin(log_ell) / contract.ell
        components = {
            probe: {name: component for name in names}
            for probe, names in (
                ("EE", ("SS", "SI", "IS", "II")),
                ("TE", ("MS", "MI", "GS", "GI")),
                ("TT", ("MM", "MG", "GM", "GG")),
            )
        }
        assembled = assemble_with_bandpowers("Triple", components)
        self.assertEqual(set(assembled.raw), {"EE", "TE", "TT"})
        for probe in assembled.raw:
            self.assertEqual(assembled.raw[probe].shape, (21,))
            self.assertEqual(assembled.bands[probe].shape, (20,))
        self.assertEqual(
            assembled.as_dict()["angular_fingerprint"], contract.fingerprint()
        )
        self.assertEqual(assembled.eta_ia.resolved_value, 0.0)
        with self.assertRaises(ValueError):
            evaluate_configuration(
                "Single",
                {"EE": {name: numpy.ones(5) for name in ("SS", "SI", "IS", "II")}},
                estimator=canonical_raw_estimator(),
            )

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

    def test_the_nn_policy_records_the_full_basis(self):
        self.assertEqual(
            NN_FINAL_DIAGONAL_POLICY, "full_basis_including_final_diagonal"
        )

    def test_known_driver_defects_are_wired_to_the_contract(self):
        spectra = REPOSITORY_ROOT / "experiments" / "spectra"
        for path in spectra.rglob("*.py"):
            if path.name == "generate_samples.py":
                continue
            text = path.read_text()
            relative = path.relative_to(spectra).as_posix()
            with self.subTest(path=relative):
                self.assertIn("require_effective_work", text)
                self.assertIn("prepare_result_directory", text)
                self.assertNotIn("result_folder.mkdir", text)
                if relative.startswith("CCL/"):
                    self.assertIn("canonical_ell_nodes", text)
                    self.assertNotIn("numpy.sqrt(ell_grid[1:]", text)
                else:
                    self.assertNotIn("canonical_ell_nodes", text)
                self.assertNotIn("--interpolation", text)
            if relative.endswith("double.py") or relative.endswith("triple.py"):
                if relative.startswith("CCL/"):
                    self.assertIn("ccl_magnification_bias", text)
                else:
                    self.assertIn("analytical_magnification_response", text)
                    self.assertIn("factor_ms", text)
            if relative.startswith("JAX/") and relative.endswith("single.py"):
                self.assertIn(
                    "cell_data_ee = cell_data_ss + cell_data_si + cell_data_is + cell_data_ii",
                    text,
                )
            if relative.startswith("JAX/") and relative.endswith("triple.py"):
                self.assertIn(
                    "cell_data_te = cell_data_ms + cell_data_mi + cell_data_gs + cell_data_gi",
                    text,
                )
                self.assertIn(
                    "cell_data_tt = cell_data_mm + cell_data_mg + cell_data_gm + cell_data_gg",
                    text,
                )


class ArtifactNamingTests(unittest.TestCase):
    def test_non_numeric_basenames_omit_the_allocation(self):
        self.assertEqual(
            timing_basename("Triple", "_COSMOLOGY"), "Time_Triple_COSMOLOGY_Cosmology.txt"
        )
        self.assertEqual(timing_basename("Triple", family="NUMBA"), "Time_Triple_Cosmology.txt")
        self.assertEqual(
            spectra_basename("Single", "EE", family="CCL"), "Spectra_Single_EE.h5"
        )
        self.assertEqual(
            samples_timing_basename("Double", family="JAX"), "Time_Double_SAMPLES.h5"
        )
        self.assertEqual(
            manifest_basename("Triple", family="NUMBA"), "Manifest_Triple.json"
        )

    def test_numeric_names_carry_the_order_token(self):
        self.assertEqual(
            timing_basename("Triple", "_CELL", "linear", family="NUMERIC"),
            "Time_Triple_LINEAR_CELL_Cosmology.txt",
        )
        self.assertEqual(
            spectra_basename("Triple", "EE", "cubic", family="NUMERIC"),
            "Spectra_Triple_CUBIC_EE.h5",
        )
        self.assertEqual(
            manifest_basename("Triple", "quadratic", family="NUMERIC"),
            "Manifest_Triple_QUADRATIC.json",
        )

    def test_an_order_is_refused_for_every_other_family(self):
        for family in ("CCL", "NUMBA", "JAX"):
            with self.subTest(family=family):
                with self.assertRaises(MethodError):
                    timing_basename("Triple", "", "linear", family=family)
                with self.assertRaises(MethodError):
                    spectra_basename("Triple", "EE", "linear", family=family)
        with self.assertRaises(ArtifactError):
            timing_basename("Triple", "", "linear")

    def test_identity_validates_the_method_combination(self):
        with self.assertRaises(MethodError):
            _identity(family="NUMBA", interpolation="LINEAR")
        with self.assertRaises(MethodError):
            _identity(family="JAX", device="")
        with self.assertRaises(MethodError):
            _identity(family="NUMERIC")
        numeric = _identity(family="NUMERIC", interpolation="linear")
        self.assertEqual(numeric.interpolation, "LINEAR")
        self.assertEqual(numeric.device, "CPU")
        self.assertEqual(_identity(family="ccl").family, "CCL")
        with self.assertRaises(ArtifactError):
            _identity(survey="Y2")
        with self.assertRaises(ArtifactError):
            _identity(configuration="Quadruple")
        with self.assertRaises(ArtifactError):
            _identity(estimator_fingerprint="not-the-estimator-digest")


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
                    sample_ids=[0, 1],
                )
                self.assertEqual(
                    product.name,
                    spectra_basename("Single", "EE", family=identity.family),
                )
                import h5py

                with h5py.File(product, "r") as handle:
                    self.assertEqual(handle["cl"].shape, (2, 3, 3))
                    self.assertNotIn("coefficients", handle)
                    self.assertIn("sampled", handle)
                    self.assertNotIn("bandpower", handle)
                with self.assertRaises(ArtifactError):
                    read_completed_manifest(namespace / "Manifest_Single.json")
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
                    products={product.name: product},
                    completed_sample_ids=[0, 1],
                    failed_sample_ids=[2],
                    eta_ia=EtaIADecision.adopted().as_dict(),
                )
                loaded = read_completed_manifest(manifest)
                self.assertEqual(loaded["failed_sample_ids"], [2])
                self.assertEqual(loaded["eta_ia"]["status"], "adopted")
                self.assertEqual(loaded["eta_ia"]["value"], 0.0)
                fiducial = read_fiducial_spectrum(manifest, "EE")
                numpy.testing.assert_allclose(fiducial, _record(0, 1.0).cl)
                product.write_bytes(product.read_bytes() + b"\0")
                with self.assertRaises(ArtifactError):
                    read_completed_manifest(manifest)
            finally:
                lock.release()

    def test_a_forged_estimator_fingerprint_is_caught_against_the_stored_axis(self):
        identity = _identity()
        record = _record(0, 1.0)
        mismatched = SampleCheckpoint(
            sample_id=record.sample_id,
            is_fiducial=record.is_fiducial,
            probe=record.probe,
            cosmology=record.cosmology,
            parameter_names=record.parameter_names,
            ell=numpy.array([20.0, 40.0, 81.0]),
            pair_i=record.pair_i,
            pair_j=record.pair_j,
            cl=record.cl,
            stage_seconds=record.stage_seconds,
            estimator_name=record.estimator_name,
        )
        with tempfile.TemporaryDirectory() as temporary:
            namespace = Path(temporary) / "run"
            lock = NamespaceLock(namespace)
            lock.acquire()
            try:
                with self.assertRaises(ArtifactError):
                    write_sample_checkpoint(namespace, mismatched, identity, lock)
            finally:
                lock.release()

    def test_a_lock_on_another_namespace_does_not_authorise_a_write(self):
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            owned = Path(temporary) / "owned"
            other = Path(temporary) / "other"
            lock = NamespaceLock(owned)
            lock.acquire()
            try:
                with self.assertRaises(ArtifactError):
                    write_sample_checkpoint(other, _record(0, 1.0), identity, lock)
                self.assertFalse((other / "checkpoints").exists())
            finally:
                lock.release()

    def test_cross_filesystem_publication_checks_the_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"spectra")
            destination = Path(temporary) / "dest" / "spectra.bin"
            with mock.patch(
                "limbercloud.io.artifacts.same_filesystem", return_value=False
            ):
                published = publish_file(source, destination)
            self.assertEqual(published.read_bytes(), b"spectra")
            self.assertFalse(source.exists())

    def test_stale_lock_on_this_host_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = NamespaceLock(temporary)
            lock.acquire()
            lock.release()
            stale = NamespaceLock(temporary)
            import socket

            stale.path.write_text(
                json.dumps({"pid": 2**22, "host": socket.gethostname()}) + "\n"
            )
            self.assertEqual(os.uname().nodename, os.uname().nodename)
            resumed = NamespaceLock(temporary)
            resumed.acquire(resume_stale=True)
            resumed.release()

    def test_estimator_mismatch_is_reported_with_both_names(self):
        estimator = EllEstimator("raw_nodes", (20.0, 40.0))
        other = EllEstimator("geometric_centres", (20.0, 40.0))
        with self.assertRaises(EstimatorMismatch):
            from limbercloud.validation.estimator import assert_same_estimator

            assert_same_estimator(estimator, other)


if __name__ == "__main__":
    unittest.main()
