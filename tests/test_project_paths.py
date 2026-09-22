"""Tests for the canonical runtime filesystem contract."""

import tempfile
import unittest
from pathlib import Path

from limbercloud import ProjectPaths
from limbercloud.validation.method import MethodError


class ProjectPathsTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.runtime_root = (
            Path(self.temporary_directory.name) / "LimberCloud-runtime"
        ).resolve()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_canonical_runtime_paths(self):
        paths = ProjectPaths(self.runtime_root)

        self.assertEqual(paths.survey_data("Y1"), self.runtime_root / "data" / "Y1")
        self.assertEqual(
            paths.config_file("cosmology"),
            self.runtime_root / "config" / "cosmology.json",
        )
        self.assertEqual(
            paths.config_file("number_density"),
            self.runtime_root / "config" / "number_density.json",
        )
        self.assertEqual(
            paths.config_file("galaxy_bias"),
            self.runtime_root / "config" / "galaxy_bias.json",
        )
        self.assertEqual(
            paths.config_file("magnification_bias"),
            self.runtime_root / "config" / "magnification_bias.json",
        )
        self.assertEqual(
            paths.config_file("intrinsic_alignment"),
            self.runtime_root / "config" / "intrinsic_alignment.json",
        )
        self.assertEqual(paths.plots, self.runtime_root / "plots")
        self.assertEqual(
            paths.covariance_results("Y10"),
            self.runtime_root / "results" / "covariance" / "Y10",
        )
        self.assertEqual(
            paths.spectrum_results("NUMBA", "Y1"),
            self.runtime_root / "results" / "spectra" / "NUMBA" / "Y1",
        )
        self.assertEqual(
            paths.spectrum_results("JAX", "Y10", "CPU"),
            self.runtime_root
            / "results"
            / "spectra"
            / "JAX"
            / "CPU"
            / "Y10",
        )
        self.assertEqual(
            paths.spectrum_results("NUMERIC", "Y1", interpolation="linear", run_id="pilot"),
            self.runtime_root
            / "results"
            / "spectra"
            / "NUMERIC"
            / "LINEAR"
            / "Y1"
            / "pilot",
        )
        self.assertEqual(
            paths.spectrum_inputs("pilot"),
            self.runtime_root / "results" / "spectra" / "inputs" / "pilot",
        )
        self.assertEqual(
            paths.validation_results("Y1"),
            self.runtime_root / "results" / "validation" / "spectra" / "Y1",
        )
        self.assertEqual(
            paths.plot_group("kernel", "Y10"),
            self.runtime_root / "plots" / "kernel" / "Y10",
        )

    def test_invalid_path_components_are_rejected(self):
        paths = ProjectPaths(self.runtime_root)

        with self.assertRaises(ValueError):
            paths.survey_data("Y2")
        with self.assertRaises(ValueError):
            paths.spectrum_results("JAX", "Y1")
        with self.assertRaises(ValueError):
            paths.config_file("unknown")

    def test_a_radial_order_reaches_numeric_paths_only(self):
        paths = ProjectPaths(self.runtime_root)

        for backend, device in (("CCL", None), ("NUMBA", None), ("JAX", "CPU"), ("JAX", "GPU")):
            with self.subTest(backend=backend, device=device):
                with self.assertRaises(MethodError):
                    paths.spectrum_results(backend, "Y1", device, interpolation="linear")
        with self.assertRaises(MethodError):
            paths.spectrum_results("NUMERIC", "Y1")
        with self.assertRaises(MethodError):
            paths.spectrum_results("NUMERIC", "Y1", interpolation="quintic")
        with self.assertRaises(MethodError):
            paths.spectrum_results("NUMERIC", "Y1", "GPU", interpolation="linear")
        self.assertEqual(
            paths.spectrum_results("NUMERIC", "Y10", interpolation="CUBIC"),
            self.runtime_root / "results" / "spectra" / "NUMERIC" / "CUBIC" / "Y10",
        )

    def test_run_identifiers_stay_inside_the_family_root(self):
        paths = ProjectPaths(self.runtime_root)

        for run_id in ("", ".", "..", "a/b", "a\\b"):
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    paths.spectrum_results("NUMBA", "Y1", run_id=run_id)


if __name__ == "__main__":
    unittest.main()
