"""Tests for the stable experiment matrix and configuration aliases."""

import unittest
from pathlib import Path

from limbercloud import Configuration
from limbercloud.experiments import (
    build_checkpoint_counts,
    resolve_sample_count,
)

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
                    with self.subTest(backend=backend, survey=survey, configuration=configuration):
                        base = REPOSITORY_ROOT / "experiments" / "spectra" / backend / survey
                        self.assertTrue((base / f"{configuration}.py").is_file())
                        self.assertTrue((base / f"{configuration}.sh").is_file())

                for device in ("CPU", "GPU"):
                    with self.subTest(backend="JAX", device=device, survey=survey, configuration=configuration):
                        base = REPOSITORY_ROOT / "experiments" / "spectra" / "JAX" / device / survey
                        self.assertTrue((base / f"{configuration}.py").is_file())
                        self.assertTrue((base / f"{configuration}.sh").is_file())

    def test_batch_launchers_use_central_environment_setup(self):
        launchers = sorted(EXPERIMENT_ROOT.rglob("*.sh")) + sorted(
            GENERATOR_ROOT.glob("*.sh")
        )
        batch_launchers = [
            path for path in launchers if "#SBATCH" in path.read_text()
        ]

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
        runners = sorted((EXPERIMENT_ROOT / "spectra").rglob("*.py"))
        self.assertEqual(len(runners), 24)
        for path in runners:
            text = path.read_text()
            with self.subTest(path=path.relative_to(REPOSITORY_ROOT)):
                self.assertIn("add_sample_control_arguments", text)
                self.assertIn("resolve_sample_count", text)
                self.assertNotIn("add_argument('--path'", text)
                self.assertNotIn("def main(tag, path", text)


class SampleControlTests(unittest.TestCase):
    def test_default_sample_count_is_zero(self):
        self.assertEqual(resolve_sample_count(None, False), 0)

    def test_fiducial_only_forces_zero(self):
        self.assertEqual(resolve_sample_count(None, True), 0)
        self.assertEqual(resolve_sample_count(0, True), 0)

    def test_fiducial_only_rejects_nonzero_sample_count(self):
        with self.assertRaises(ValueError):
            resolve_sample_count(1, True)

    def test_campaign_sample_count(self):
        self.assertEqual(resolve_sample_count(1000, False), 1000)

    def test_checkpoint_counts_for_campaign(self):
        counts = build_checkpoint_counts(1000)
        self.assertEqual(int(counts[0]), 100)
        self.assertEqual(int(counts[-1]), 1000)
        self.assertEqual(counts.size, 10)

    def test_checkpoint_counts_for_tiny_runs(self):
        counts = build_checkpoint_counts(3)
        self.assertTrue((counts > 0).all())
        self.assertEqual(int(counts[-1]), 3)

    def test_checkpoint_counts_for_zero(self):
        counts = build_checkpoint_counts(0)
        self.assertEqual(counts.size, 0)


if __name__ == "__main__":
    unittest.main()
