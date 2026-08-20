"""Tests for the stable experiment matrix and configuration aliases."""

import unittest
from pathlib import Path

from limbercloud import Configuration


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
        self.assertEqual(
            [configuration.legacy_value for configuration in Configuration],
            ["SINGLE", "DOUBLE", "TRIPLE"],
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


if __name__ == "__main__":
    unittest.main()
