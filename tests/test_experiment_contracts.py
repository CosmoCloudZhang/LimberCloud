"""Tests for the stable experiment matrix and configuration aliases."""

import unittest
from pathlib import Path

from limbercloud import Configuration

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments"
NERSC_SCRIPT_ROOT = REPOSITORY_ROOT / "scripts" / "nersc"


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
            (NERSC_SCRIPT_ROOT / "generate_config").glob("*.sh")
        )
        batch_launchers = [
            path for path in launchers if "#SBATCH" in path.read_text()
        ]

        self.assertEqual(len(batch_launchers), 34)
        for path in batch_launchers:
            text = path.read_text()
            relative_path = path.relative_to(REPOSITORY_ROOT)
            with self.subTest(path=relative_path):
                self.assertIn("scripts/nersc/load_environment.sh", text)
                self.assertIn('conda activate "${LIMBERCLOUD_CONDA_ENV}"', text)
                self.assertNotIn('source "${HOME}/.bashrc"', text)
                self.assertNotIn("${CosmoENV}", text)

                module_profile = (
                    "gpu.sh"
                    if "experiments/spectra/JAX/GPU" in relative_path.as_posix()
                    else "cpu.sh"
                )
                self.assertIn(f"scripts/nersc/modules/{module_profile}", text)

                self.assertLess(
                    text.index("REPO_ROOT="),
                    text.index("scripts/nersc/load_environment.sh"),
                )
                self.assertLess(
                    text.index("scripts/nersc/load_environment.sh"),
                    text.index('conda activate "${LIMBERCLOUD_CONDA_ENV}"'),
                )

    def test_run_all_launchers_preflight_local_environment(self):
        run_all_launchers = sorted(EXPERIMENT_ROOT.rglob("run_all.sh"))

        self.assertEqual(len(run_all_launchers), 4)
        for path in run_all_launchers:
            text = path.read_text()
            with self.subTest(path=path.relative_to(REPOSITORY_ROOT)):
                self.assertIn("scripts/nersc/load_environment.sh", text)
                self.assertLess(
                    text.index("scripts/nersc/load_environment.sh"),
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


if __name__ == "__main__":
    unittest.main()
