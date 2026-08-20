"""Tests for legacy and canonical filesystem contracts."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from limbercloud import ProjectPaths, RuntimeLayout


class RuntimeLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.runtime_root = (
            Path(self.temporary_directory.name) / "LimberCloud-runtime"
        ).resolve()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_legacy_runtime_layout(self):
        paths = ProjectPaths(self.runtime_root, RuntimeLayout.LEGACY)

        self.assertEqual(paths.survey_data("Y1"), self.runtime_root / "DATA" / "Y1")
        self.assertEqual(paths.config_file("cosmology"), self.runtime_root / "INFO" / "COSMOLOGY.json")
        self.assertEqual(paths.plots, self.runtime_root / "PLOT")
        self.assertEqual(paths.covariance_results("Y10"), self.runtime_root / "COVARIANCE" / "Y10")
        self.assertEqual(paths.spectrum_results("CCL", "Y1"), self.runtime_root / "PYTHON" / "CCL" / "Y1")
        self.assertEqual(paths.spectrum_results("JAX", "Y10", "GPU"), self.runtime_root / "JAX" / "GPU" / "Y10")
        self.assertEqual(paths.validation_results("Y1"), self.runtime_root / "PYTHON" / "CELL" / "Y1")
        self.assertEqual(paths.plot_group("kernel", "Y10"), self.runtime_root / "PLOT" / "KERNEL" / "Y10")

    def test_canonical_runtime_layout(self):
        paths = ProjectPaths(self.runtime_root, RuntimeLayout.CANONICAL)

        self.assertEqual(paths.survey_data("Y1"), self.runtime_root / "data" / "Y1")
        self.assertEqual(paths.config_file("cosmology"), self.runtime_root / "config" / "cosmology.json")
        self.assertEqual(paths.config_file("number_density"), self.runtime_root / "config" / "number_density.json")
        self.assertEqual(paths.config_file("galaxy_bias"), self.runtime_root / "config" / "galaxy_bias.json")
        self.assertEqual(paths.config_file("magnification_bias"), self.runtime_root / "config" / "magnification_bias.json")
        self.assertEqual(paths.config_file("intrinsic_alignment"), self.runtime_root / "config" / "intrinsic_alignment.json")
        self.assertEqual(paths.plots, self.runtime_root / "plots")
        self.assertEqual(paths.covariance_results("Y10"), self.runtime_root / "results" / "covariance" / "Y10")
        self.assertEqual(paths.spectrum_results("NUMBA", "Y1"), self.runtime_root / "results" / "spectra" / "NUMBA" / "Y1")
        self.assertEqual(paths.spectrum_results("JAX", "Y10", "CPU"), self.runtime_root / "results" / "spectra" / "JAX" / "CPU" / "Y10")
        self.assertEqual(paths.validation_results("Y1"), self.runtime_root / "results" / "validation" / "spectra" / "Y1")
        self.assertEqual(paths.plot_group("kernel", "Y10"), self.runtime_root / "plots" / "kernel" / "Y10")

    def test_layout_can_be_selected_from_environment(self):
        with mock.patch.dict(os.environ, {"LIMBERCLOUD_LAYOUT": "canonical"}):
            self.assertIs(
                ProjectPaths.from_root(self.runtime_root).layout,
                RuntimeLayout.CANONICAL,
            )

    def test_invalid_path_components_are_rejected(self):
        paths = ProjectPaths(self.runtime_root)

        with self.assertRaises(ValueError):
            paths.survey_data("Y2")
        with self.assertRaises(ValueError):
            paths.spectrum_results("JAX", "Y1")
        with self.assertRaises(ValueError):
            paths.config_file("unknown")


if __name__ == "__main__":
    unittest.main()
