"""Tests for the stable experiment matrix and configuration aliases."""

import contextlib
import importlib.util
import io
import itertools
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy

from limbercloud import Configuration
from limbercloud.experiments import (
    build_checkpoint_counts,
    resolve_sample_count,
)
from limbercloud.experiments.timing import (
    load_cosmology_timing,
    require_matching_timings,
    write_timing_products,
)
from limbercloud.io.artifacts import timing_basename

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments"
GENERATOR_ROOT = REPOSITORY_ROOT / "scripts" / "generate_config"


class ExperimentContractTests(unittest.TestCase):
    def test_configuration_aliases(self):
        aliases = {
            "SINGLE": Configuration.SINGLE,
            "Single": Configuration.SINGLE,
            "single": Configuration.SINGLE,
            "DOUBLE": Configuration.DOUBLE,
            "Double": Configuration.DOUBLE,
            "TRIPLE": Configuration.TRIPLE,
            "Triple": Configuration.TRIPLE,
        }
        for value, expected in aliases.items():
            with self.subTest(value=value):
                self.assertIs(Configuration.parse(value), expected)

    def test_configuration_values(self):
        self.assertEqual(
            [configuration.value for configuration in Configuration],
            ["Single", "Double", "Triple"],
        )

    def test_complete_experiment_matrix_exists(self):
        for survey in ("Y1", "Y10"):
            for configuration in ("single", "double", "triple"):
                for backend in ("CCL", "NUMBA"):
                    with self.subTest(
                        backend=backend, survey=survey, configuration=configuration
                    ):
                        base = (
                            REPOSITORY_ROOT
                            / "experiments"
                            / "spectra"
                            / backend
                            / survey
                        )
                        self.assertTrue((base / f"{configuration}.py").is_file())
                        self.assertTrue((base / f"{configuration}.sh").is_file())

                for device in ("CPU", "GPU"):
                    with self.subTest(
                        backend="JAX",
                        device=device,
                        survey=survey,
                        configuration=configuration,
                    ):
                        base = (
                            REPOSITORY_ROOT
                            / "experiments"
                            / "spectra"
                            / "JAX"
                            / device
                            / survey
                        )
                        self.assertTrue((base / f"{configuration}.py").is_file())
                        self.assertTrue((base / f"{configuration}.sh").is_file())

    def test_batch_launchers_use_central_environment_setup(self):
        launchers = sorted(EXPERIMENT_ROOT.rglob("*.sh")) + sorted(
            GENERATOR_ROOT.glob("*.sh")
        )
        batch_launchers = [path for path in launchers if "#SBATCH" in path.read_text()]

        self.assertEqual(len(batch_launchers), 34)
        for path in batch_launchers:
            text = path.read_text()
            relative_path = path.relative_to(REPOSITORY_ROOT)
            with self.subTest(path=relative_path):
                self.assertIn("scripts/load_config.sh", text)
                self.assertIn("scripts/nersc/activate_venv.sh", text)
                self.assertNotIn("load_environment.sh", text)
                self.assertNotIn("LIMBERCLOUD_CONDA_ENV", text)
                self.assertNotIn("LIMBERCLOUD_REPO_ROOT", text)
                self.assertNotIn('source "${HOME}/.bashrc"', text)
                self.assertNotIn("${CosmoENV}", text)
                self.assertNotIn("--path=", text)

                module_profile = (
                    "gpu.sh"
                    if "experiments/spectra/JAX/GPU" in relative_path.as_posix()
                    else "cpu.sh"
                )
                self.assertIn(f"scripts/nersc/modules/{module_profile}", text)
                self.assertIn("PROJECT_ROOT", text)
                self.assertLess(
                    text.index("scripts/load_config.sh"),
                    text.index("scripts/nersc/activate_venv.sh"),
                )

    def test_run_all_launchers_preflight_local_environment(self):
        run_all_launchers = sorted(EXPERIMENT_ROOT.rglob("Run_All.sh"))

        self.assertEqual(len(run_all_launchers), 4)
        for path in run_all_launchers:
            text = path.read_text()
            with self.subTest(path=path.relative_to(REPOSITORY_ROOT)):
                self.assertIn("scripts/load_config.sh", text)
                self.assertNotIn("LIMBERCLOUD_REPO_ROOT", text)
                self.assertLess(
                    text.index("scripts/load_config.sh"),
                    text.index("sbatch"),
                )

    def test_covariance_launchers_use_canonical_onecovariance_root(self):
        launchers = sorted((EXPERIMENT_ROOT / "covariance").rglob("matrix.sh"))

        self.assertEqual(len(launchers), 2)
        for path in launchers:
            text = path.read_text()
            with self.subTest(path=path.relative_to(REPOSITORY_ROOT)):
                self.assertIn("limbercloud_require_onecovariance", text)
                self.assertIn("LIMBERCLOUD_ONECOVARIANCE_ROOT", text)

    def test_spectra_runners_expose_sample_controls_without_path(self):
        runners = [
            path
            for path in sorted((EXPERIMENT_ROOT / "spectra").rglob("*.py"))
            if path.name != "generate_samples.py"
        ]
        self.assertEqual(len(runners), 24)
        self.assertTrue((EXPERIMENT_ROOT / "spectra" / "generate_samples.py").is_file())
        for path in runners:
            text = path.read_text()
            with self.subTest(path=path.relative_to(REPOSITORY_ROOT)):
                self.assertIn("add_evaluation_arguments", text)
                self.assertIn("resolve_sample_count", text)
                self.assertIn("load_cosmology_table", text)
                self.assertIn("require_nuisance_compatibility", text)
                self.assertIn("component_activity", text)
                self.assertIn("write_timing_products", text)
                self.assertNotIn("numpy.random.uniform", text)
                self.assertNotIn("add_argument('--path'", text)
                self.assertNotIn("def main(tag, path", text)


class SampleControlTests(unittest.TestCase):
    def test_default_sample_count_is_the_fiducial_alone(self):
        self.assertEqual(resolve_sample_count(None), 0)
        self.assertEqual(resolve_sample_count(0), 0)

    def test_negative_sample_count_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_sample_count(-1)

    def test_campaign_sample_count(self):
        self.assertEqual(resolve_sample_count(1000), 1000)

    def test_checkpoint_counts_for_campaign(self):
        counts = build_checkpoint_counts(1000)
        self.assertEqual(int(counts[0]), 100)
        self.assertEqual(int(counts[-1]), 1000)
        self.assertEqual(counts.size, 10)

    def test_checkpoint_counts_for_tiny_runs(self):
        counts = build_checkpoint_counts(3)
        self.assertTrue((counts > 0).all())
        self.assertEqual(counts.tolist(), [3])
        self.assertEqual(build_checkpoint_counts(150).tolist(), [100, 150])
        self.assertEqual(
            build_checkpoint_counts(1010).tolist(), list(range(100, 1001, 100)) + [1010]
        )

    def test_checkpoint_counts_for_zero(self):
        counts = build_checkpoint_counts(0)
        self.assertEqual(counts.size, 0)


class TimingProductTests(unittest.TestCase):
    def write(self, directory, counts=(100,), table_hash="table-a", fiducial=7.0):
        return write_timing_products(
            directory,
            "Triple",
            family="NUMBA",
            sample_table_hash=table_hash,
            counts=counts,
            fiducial={"": fiducial},
            sampled={"": numpy.arange(1, len(counts) + 1, dtype=float)},
        )

    def test_fiducial_rerun_preserves_explicit_sampled_products(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "Time_Triple_128.txt"
            legacy.write_text("legacy\n")
            self.write(root, counts=(100, 150))
            sampled = root / timing_basename("Triple")
            previous = sampled.read_bytes()
            self.write(root, counts=(), fiducial=9.0)
            self.assertEqual(sampled.read_bytes(), previous)
            self.assertEqual(legacy.read_text(), "legacy\n")
            numpy.testing.assert_array_equal(
                numpy.loadtxt(root / timing_basename("Triple", population="Fiducial")),
                [0, 9.0],
            )
            result = load_cosmology_timing(root, "Triple", family="NUMBA")
            self.assertEqual(result.counts.tolist(), [100, 150])
            self.assertEqual(result.seconds.tolist(), [1.0, 2.0])

    def test_comparison_accepts_single_checkpoint_and_rejects_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root / "a")
            self.write(root / "b")
            first = load_cosmology_timing(root / "a", "Triple", family="NUMBA")
            second = load_cosmology_timing(root / "b", "Triple", family="NUMBA")
            self.assertEqual(require_matching_timings(first, second).tolist(), [100])
            self.write(root / "b", table_hash="table-b")
            second = load_cosmology_timing(root / "b", "Triple", family="NUMBA")
            with self.assertRaisesRegex(ValueError, "sample_table_hash"):
                require_matching_timings(first, second)
            self.write(root / "b", counts=(50,))
            second = load_cosmology_timing(root / "b", "Triple", family="NUMBA")
            with self.assertRaisesRegex(ValueError, "checkpoint counts"):
                require_matching_timings(first, second)

    def test_reader_rejects_fiducial_legacy_and_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, counts=())
            (root / "Time_Triple.txt").write_text("1\n")
            with self.assertRaisesRegex(FileNotFoundError, "sample-count > 0"):
                load_cosmology_timing(root, "Triple", family="NUMBA")
            self.write(root)
            path = root / timing_basename("Triple")
            text = path.read_text()
            path.write_text(
                text.replace('"population": "Cosmology"', '"population": "Fiducial"')
            )
            with self.assertRaisesRegex(ValueError, "population"):
                load_cosmology_timing(root, "Triple", family="NUMBA")
            path.write_text(text.rsplit("\n", 2)[0] + "\n100 nan\n")
            with self.assertRaisesRegex(ValueError, "invalid counts"):
                load_cosmology_timing(root, "Triple", family="NUMBA")

    def test_benchmark_reads_actual_single_checkpoint_without_legacy_fallback(self):
        # Stub plotting only: no LaTeX or graphical runtime is required here.
        for survey in ("Y1", "Y10"):
            with (
                self.subTest(survey=survey),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                for family, device in (
                    ("CCL", None),
                    ("NUMBA", None),
                    ("JAX", "CPU"),
                    ("JAX", "GPU"),
                ):
                    directory = root / "results" / "spectra" / family
                    if device:
                        directory /= device
                    directory /= survey
                    stages = (
                        {"": 1.0}
                        if family == "CCL"
                        else {
                            "": 3.0,
                            "COSMOLOGY": 1.0,
                            "COEFFICIENT": 1.0,
                            "PROJECTION": 1.0,
                        }
                    )
                    write_timing_products(
                        directory,
                        "Triple",
                        family=family,
                        sample_table_hash="shared",
                        counts=[50],
                        fiducial=stages,
                        sampled={key: [value] for key, value in stages.items()},
                    )
                pyplot = types.SimpleNamespace(rcParams={}, close=mock.Mock())
                figure = mock.Mock()
                axes = [mock.Mock() for _ in range(4)]
                pyplot.subplots = mock.Mock(return_value=(figure, axes))
                matplotlib = types.ModuleType("matplotlib")
                matplotlib.pyplot = pyplot
                with mock.patch.dict("sys.modules", {"matplotlib": matplotlib}):
                    driver = _load_driver(
                        EXPERIMENT_ROOT / "benchmarks" / survey / "benchmark.py"
                    )
                with contextlib.redirect_stdout(io.StringIO()):
                    driver.main(survey, "Triple", str(root))
                for axis in axes:
                    self.assertEqual(len(axis.loglog.call_args_list), 4)
                    for call in axis.loglog.call_args_list:
                        self.assertEqual(call.args[0].tolist(), [50])
                        self.assertEqual(call.args[1].shape, (1,))
                    self.assertEqual(
                        axis.loglog.call_args_list[0].kwargs["label"], r"$\mathtt{CCL}$"
                    )
                figure.savefig.assert_called_once()


def _load_driver(path):
    specification = importlib.util.spec_from_file_location("test_spectra_driver", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class _ReadyArray(numpy.ndarray):
    """NumPy stand-in for the JAX synchronization contract, without importing JAX."""

    def block_until_ready(self):
        return self


class RunnerTimingIntegrationTests(unittest.TestCase):
    def test_every_driver_runs_fiducial_first_and_skips_disabled_components(self):
        runners = [
            p
            for p in sorted((EXPERIMENT_ROOT / "spectra").rglob("*.py"))
            if p.name != "generate_samples.py"
        ]
        for path in runners:
            with (
                self.subTest(path=path.relative_to(REPOSITORY_ROOT)),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                survey = path.parent.name
                family = path.relative_to(EXPERIMENT_ROOT / "spectra").parts[0]
                label = path.stem.title()
                data = root / "data" / survey
                data.mkdir(parents=True)
                distributions = {
                    "redshift_range": numpy.array([0.0, 3.5]),
                    "bins": numpy.ones((2, 2)),
                }
                for name in ("lens", "source"):
                    numpy.save(data / f"lsst_{name}_bins.npy", distributions)
                (root / "config").mkdir()
                (root / "config" / "cosmology.json").write_text(
                    json.dumps({"OMEGA_M": 0.3, "H": 0.7})
                )
                evaluated = []
                tracers = []
                coefficient_calls = []

                def cosmology(**row):
                    evaluated.append(row["sample_id"])
                    return type(
                        "Cosmology",
                        (dict,),
                        {"h_over_h0": lambda self, a: numpy.ones_like(a)},
                    )(row)

                def tracer(**kwargs):
                    tracers.append(kwargs)
                    return kwargs

                ccl = types.ModuleType("pyccl")
                ccl.Cosmology = cosmology
                ccl.gsl_params = {}
                ccl.background = types.SimpleNamespace(
                    comoving_radial_distance=lambda cosmo, a: (1 / a - 1) * 1000
                )
                def power_provider(cosmo, k, a):
                    self.assertTrue(numpy.all(numpy.isfinite(k)))
                    self.assertTrue(numpy.all(k > 0))
                    self.assertLess(float(numpy.max(k)), 1e6)
                    return numpy.ones_like(k)

                ccl.power = types.SimpleNamespace(nonlin_matter_power=power_provider)
                ccl.tracers = types.SimpleNamespace(
                    WeakLensingTracer=tracer, NumberCountsTracer=tracer
                )
                ccl.cells = types.SimpleNamespace(
                    angular_cl=lambda **kwargs: numpy.ones_like(kwargs["ell"])
                )
                backend = types.ModuleType("test_backend")

                def coefficient(name, **kwargs):
                    self.assertTrue(numpy.any(kwargs["power_grid"] != 0))
                    self.assertTrue(numpy.all(kwargs["power_grid"][:, 0] == 0))
                    coefficient_calls.append(name)
                    return numpy.ones(1).view(_ReadyArray)

                for name in ("NN", "NS", "SN", "SS"):
                    setattr(
                        backend,
                        name,
                        types.SimpleNamespace(
                            coefficient=lambda name=name, **kwargs: coefficient(
                                name, **kwargs
                            )
                        ),
                    )
                backend.TENSOR = types.SimpleNamespace(
                    spectra=lambda **kwargs: numpy.ones(
                        (
                            kwargs["phi_a_grid"].shape[0],
                            kwargs["phi_b_grid"].shape[0],
                            kwargs["factor"].size,
                        )
                    ).view(_ReadyArray)
                )
                jax = types.ModuleType("jax")
                jax.config = types.SimpleNamespace(update=lambda *args: None)
                modules = {
                    "pyccl": ccl,
                    "jax": jax,
                    "limbercloud.projection.numba_backend": backend,
                    "limbercloud.projection.jax_backend": backend,
                }
                with (
                    mock.patch.dict("sys.modules", modules),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    driver = _load_driver(path)
                    provenance = types.SimpleNamespace(
                        generating_model_fingerprint="test"
                    )
                    driver.load_alignment = lambda *args: (numpy.zeros(351), provenance)
                    driver.load_galaxy_bias = lambda *args: (
                        numpy.ones(351),
                        provenance,
                    )
                    driver.load_magnification_slope = lambda *args: numpy.full(2, 0.4)
                    table = types.SimpleNamespace(content_hash="canonical")
                    driver.load_cosmology_table = mock.Mock(return_value=table)
                    driver.require_nuisance_compatibility = mock.Mock()
                    driver.select_rows = lambda table, ids: [
                        {"sample_id": i, "Omega_m": 0.3, "h": 0.7} for i in ids
                    ]
                    driver.ccl_cosmology_kwargs = lambda row: row
                    ticks = itertools.count()
                    driver.time = types.SimpleNamespace(time=lambda: float(next(ticks)))
                    driver.main(
                        survey, label, str(root), sample_count=1, sample_table="table"
                    )
                    self.assertEqual(evaluated, [0, 1])
                    driver.load_cosmology_table.assert_called_once_with("table")
                    driver.require_nuisance_compatibility.assert_called_once()
                    if family == "CCL":
                        self.assertTrue(tracers)
                        for value in tracers:
                            if "ia_bias" in value:
                                self.assertIsNone(value["ia_bias"])
                            if "mag_bias" in value:
                                self.assertIsNone(value["mag_bias"])
                    else:
                        expected = {
                            "Single": ["SS"],
                            "Double": ["NS", "NN"],
                            "Triple": ["SS", "NS", "NN"],
                        }[label]
                        self.assertEqual(coefficient_calls, expected * 2)
                    files = list(
                        (root / "results").rglob(f"Time_{label}_Cosmology.txt")
                    )
                    self.assertEqual(len(files), 1)
                    previous = files[0].read_bytes()
                    sampled = load_cosmology_timing(
                        files[0].parent, label, family=family
                    )
                    self.assertEqual(sampled.counts.tolist(), [1])
                    expected_seconds = 2.0 if family == "CCL" else 3.0
                    self.assertEqual(sampled.seconds.tolist(), [expected_seconds])
                    driver.main(
                        survey, label, str(root), sample_count=0, sample_table="table"
                    )
                    self.assertEqual(evaluated, [0, 1, 0])
                    self.assertEqual(files[0].read_bytes(), previous)
                    numpy.testing.assert_array_equal(
                        numpy.loadtxt(
                            files[0].parent
                            / timing_basename(label, population="Fiducial")
                        ),
                        [0, expected_seconds],
                    )


if __name__ == "__main__":
    unittest.main()
