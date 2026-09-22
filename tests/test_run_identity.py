"""The three run-identity layers Phase 2 enforces.

NumPy and SciPy only; no h5py, Numba, JAX or CCL import.
"""

import tempfile
import unittest
from pathlib import Path

from limbercloud.validation.method import MethodIdentity
from limbercloud.validation.run_identity import (
    ExecutionRecord,
    ProducerWorkload,
    RunIdentityError,
    SharedScience,
    compute_source_manifest,
    numerical_dependency_signature,
    source_fingerprint,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

RADIAL_GRID = {
    "node_count": 351,
    "minimum": 0.0,
    "maximum": 3.5,
    "variable": "redshift",
}
NUISANCE = {
    "intrinsic_alignment": "a" * 64,
    "galaxy_bias": "b" * 64,
    "magnification_bias": "c" * 64,
}
QUADRATURE = {"terminal_rule": "gauss-legendre", "terminal_order": 48}
TIMING = (
    "compute spans cosmology, coefficients, component assembly and the 21->20 operator"
)


def shared(**overrides) -> SharedScience:
    fields = {
        "survey": "Y1",
        "configuration": "Triple",
        "sample_table_hash": "d" * 64,
        "solver_fingerprint": "e" * 64,
        "nuisance_hashes": NUISANCE,
        "radial_grid": RADIAL_GRID,
    }
    fields.update(overrides)
    return SharedScience(**fields)


def workload(science=None, **overrides) -> ProducerWorkload:
    fields = {
        "shared": science if science is not None else shared(),
        "method": MethodIdentity.create("NUMBA"),
        "requested_sample_ids": (0, 1, 2),
        "quadrature": QUADRATURE,
        "timing_boundary": TIMING,
        "source_manifest": {"src/limbercloud/__init__.py": "f" * 64},
        "dependency_signature": {"numpy": "2.2.6", "float64": "enabled"},
    }
    fields.update(overrides)
    return ProducerWorkload(**fields)


class SharedScienceTests(unittest.TestCase):
    def test_the_shared_contract_carries_every_required_field(self):
        science = shared()
        payload = science.as_dict()
        for key in (
            "sample_table_hash",
            "solver_fingerprint",
            "nuisance_hashes",
            "radial_grid",
            "angular_operator",
            "eta_ia",
            "nuisance_policy",
            "endpoint_policy",
            "pair_orientation",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["probes"], ["EE", "TE", "TT"])
        self.assertEqual(payload["angular_operator"]["node_count"], 21)
        self.assertEqual(payload["angular_operator"]["band_count"], 20)
        self.assertEqual(payload["angular_operator"]["boundary_condition"], "natural")
        self.assertEqual(len(science.fingerprint()), 64)

    def test_inconsistent_fields_are_rejected_rather_than_stored(self):
        for overrides in (
            {"survey": "Y2"},
            {"configuration": "Quadruple"},
            {"sample_table_hash": ""},
            {"solver_fingerprint": ""},
            {"nuisance_hashes": {}},
            {"nuisance_hashes": {"galaxy_bias": ""}},
            {"eta_ia": 0.5},
            {"nuisance_policy": "per_sample"},
            {"endpoint_policy": "current_implementation_omits_final_diagonal"},
            {"radial_grid": {"node_count": 351}},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(RunIdentityError):
                    shared(**overrides)

    def test_changed_configuration_changes_the_fingerprint(self):
        self.assertNotEqual(
            shared().fingerprint(), shared(configuration="Single").fingerprint()
        )
        self.assertNotEqual(shared().fingerprint(), shared(survey="Y10").fingerprint())
        self.assertNotEqual(
            shared().fingerprint(),
            shared(nuisance_hashes=dict(NUISANCE, galaxy_bias="z" * 64)).fingerprint(),
        )


class ProducerWorkloadTests(unittest.TestCase):
    def test_different_methods_compare_but_do_not_resume_each_other(self):
        numba = workload()
        jax_cpu = workload(method=MethodIdentity.create("JAX", "CPU"))
        self.assertTrue(numba.is_comparable_with(jax_cpu))
        self.assertFalse(numba.may_resume(jax_cpu))
        self.assertTrue(numba.may_resume(workload()))

    def test_a_changed_runtime_cannot_resume_under_an_unchanged_source_digest(self):
        original = workload()
        upgraded = workload(
            dependency_signature={"numpy": "2.3.0", "float64": "enabled"}
        )
        self.assertFalse(original.may_resume(upgraded))
        self.assertTrue(original.is_comparable_with(upgraded))

    def test_changed_source_or_configuration_blocks_resume(self):
        original = workload()
        edited = workload(source_manifest={"src/limbercloud/__init__.py": "9" * 64})
        reconfigured = workload(science=shared(configuration="Single"))
        self.assertFalse(original.may_resume(edited))
        self.assertFalse(original.may_resume(reconfigured))
        self.assertFalse(original.is_comparable_with(reconfigured))

    def test_malformed_workloads_are_rejected(self):
        for overrides in (
            {"requested_sample_ids": ()},
            {"requested_sample_ids": (0, 0, 1)},
            {"requested_sample_ids": (0, -1)},
            {"quadrature": {}},
            {"timing_boundary": ""},
            {"source_manifest": {}},
            {"dependency_signature": {}},
            {"method": "NUMBA"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(RunIdentityError):
                    workload(**overrides)

    def test_numeric_workloads_carry_their_order(self):
        numeric = workload(method=MethodIdentity.create("NUMERIC", None, "cubic"))
        self.assertEqual(numeric.as_dict()["method"]["interpolation"], "CUBIC")
        self.assertNotEqual(numeric.fingerprint(), workload().fingerprint())
        self.assertTrue(numeric.is_comparable_with(workload()))


class ComputeSourceManifestTests(unittest.TestCase):
    def test_the_manifest_covers_source_and_excludes_reports(self):
        manifest = compute_source_manifest(REPOSITORY_ROOT)
        self.assertIn("src/limbercloud/validation/estimator.py", manifest)
        self.assertIn("experiments/spectra/NUMBA/Y1/single.py", manifest)
        self.assertFalse(any(path.startswith("revisions/") for path in manifest))
        self.assertFalse(any(path.startswith("documents/") for path in manifest))
        self.assertFalse(any("__pycache__" in path for path in manifest))
        self.assertEqual(len(source_fingerprint(manifest)), 64)

    def test_an_untracked_edit_changes_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src" / "limbercloud"
            source.mkdir(parents=True)
            (source / "__init__.py").write_text("value = 1\n")
            first = source_fingerprint(compute_source_manifest(root))
            (source / "extra.py").write_text("value = 2\n")
            second = source_fingerprint(compute_source_manifest(root))
            self.assertNotEqual(first, second)
            # A report written afterwards is provenance, not a compute input.
            reports = root / "revisions" / "2026-09" / "reports"
            reports.mkdir(parents=True)
            (reports / "REPORT.md").write_text("# report\n")
            self.assertEqual(second, source_fingerprint(compute_source_manifest(root)))

    def test_an_empty_manifest_is_refused(self):
        with self.assertRaises(RunIdentityError):
            source_fingerprint({})


class ExecutionRecordTests(unittest.TestCase):
    def test_records_refer_to_the_producer_without_feeding_back(self):
        producer = workload()
        record = ExecutionRecord(
            producer_fingerprint=producer.fingerprint(),
            head_commit="72bb8af",
            scheduler_job="58590099",
            host="nid001234",
            process_id=4242,
            started_at="2026-09-22T14:05:00+00:00",
            output_paths={"Spectra_Triple_EE.h5": "/cfs/run/Spectra_Triple_EE.h5"},
            output_checksums={"Spectra_Triple_EE.h5": "1" * 64},
            status="completed",
        )
        payload = record.as_dict()
        self.assertEqual(payload["producer_fingerprint"], producer.fingerprint())
        self.assertEqual(payload["head_commit"], "72bb8af")
        # Recording an attempt does not change what the producer computes.
        self.assertEqual(producer.fingerprint(), workload().fingerprint())
        for key in ("output_paths", "output_checksums", "scheduler_job", "head_commit"):
            self.assertNotIn(key, workload().as_dict())

    def test_malformed_records_are_rejected(self):
        with self.assertRaises(RunIdentityError):
            ExecutionRecord(
                producer_fingerprint="",
                head_commit="72bb8af",
                scheduler_job="1",
                host="host",
                process_id=1,
                started_at="2026-09-22T14:05:00+00:00",
            )
        with self.assertRaises(RunIdentityError):
            ExecutionRecord(
                producer_fingerprint="a" * 64,
                head_commit="72bb8af",
                scheduler_job="1",
                host="host",
                process_id=1,
                started_at="2026-09-22T14:05:00+00:00",
                status="probably-fine",
            )


class DependencySignatureTests(unittest.TestCase):
    def test_absent_backends_are_visible_rather_than_omitted(self):
        signature = numerical_dependency_signature()
        for name in ("numpy", "scipy", "pyccl", "camb", "numba", "jax", "jaxlib"):
            self.assertIn(name, signature)
        self.assertEqual(signature["float64"], "enabled")


if __name__ == "__main__":
    unittest.main()
