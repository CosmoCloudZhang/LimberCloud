"""Tests for the non-destructive runtime-layout migration utility."""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.migrate_runtime_layout import (
    CONFIGURATION_FILES,
    create_output_directories,
    migrate_config,
    migrate_data,
)


class RuntimeMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.runtime_root = Path(self.temporary_directory.name) / "runtime"

        for survey in ("Y1", "Y10"):
            survey_directory = self.runtime_root / "DATA" / survey
            survey_directory.mkdir(parents=True)
            (survey_directory / "sample.txt").write_text(
                f"unchanged-{survey}\n",
                encoding="utf-8",
            )

        configuration_directory = self.runtime_root / "INFO"
        configuration_directory.mkdir()
        for legacy_name in CONFIGURATION_FILES:
            (configuration_directory / legacy_name).write_text(
                f'{{"source": "{legacy_name}"}}\n',
                encoding="utf-8",
            )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_dry_run_does_not_modify_runtime_root(self):
        before = sorted(
            str(path.relative_to(self.runtime_root))
            for path in self.runtime_root.rglob("*")
        )

        with redirect_stdout(StringIO()):
            migrate_data(self.runtime_root, execute=False)
            migrate_config(self.runtime_root, execute=False)
            create_output_directories(self.runtime_root, execute=False)

        after = sorted(
            str(path.relative_to(self.runtime_root))
            for path in self.runtime_root.rglob("*")
        )
        self.assertEqual(after, before)

    def test_execute_copies_inputs_and_creates_outputs(self):
        with redirect_stdout(StringIO()):
            migrate_data(self.runtime_root, execute=True)
            migrate_config(self.runtime_root, execute=True)
            create_output_directories(self.runtime_root, execute=True)

        for survey in ("Y1", "Y10"):
            source = self.runtime_root / "DATA" / survey / "sample.txt"
            destination = self.runtime_root / "data" / survey / "sample.txt"
            self.assertEqual(destination.read_bytes(), source.read_bytes())

        for legacy_name, canonical_name in CONFIGURATION_FILES.items():
            source = self.runtime_root / "INFO" / legacy_name
            destination = self.runtime_root / "config" / canonical_name
            self.assertEqual(destination.read_bytes(), source.read_bytes())

        self.assertTrue((self.runtime_root / "results" / "spectra" / "CCL").is_dir())
        self.assertTrue((self.runtime_root / "results" / "covariance").is_dir())
        self.assertTrue((self.runtime_root / "plots").is_dir())
        self.assertTrue((self.runtime_root / "logs").is_dir())

    def test_execute_refuses_to_overwrite_different_file(self):
        destination = self.runtime_root / "config" / "cosmology.json"
        destination.parent.mkdir()
        destination.write_text("different\n", encoding="utf-8")

        with redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                migrate_config(self.runtime_root, execute=True)


if __name__ == "__main__":
    unittest.main()
