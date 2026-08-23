"""Isolated contract tests for LimberCloud's NERSC environment helpers."""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOADER = REPOSITORY_ROOT / "scripts" / "nersc" / "load_environment.sh"
MODULES = REPOSITORY_ROOT / "scripts" / "nersc" / "modules"


class EnvironmentLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dotenv = self.root / ".env"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def clean_environment(self, **updates):
        environment = {
            "HOME": str(self.root),
            "LANG": "C",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }
        environment.update(updates)
        return environment

    def run_loader(self, shell_body, dotenv_text=None, **environment_updates):
        if dotenv_text is not None:
            self.dotenv.write_text(dotenv_text, encoding="utf-8")
            environment_updates.setdefault(
                "LIMBERCLOUD_ENV_FILE", str(self.dotenv)
            )
        command = (
            f"source {shlex.quote(str(LOADER))} || exit $?\n"
            f"{shell_body}"
        )
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", command],
            cwd=self.root,
            env=self.clean_environment(**environment_updates),
            check=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

    def test_loads_allowlisted_values_and_defaults_conda_name(self):
        result = self.run_loader(
            'printf "%s\\n%s\\n%s\\n" "$LIMBERCLOUD_RUNTIME_ROOT" '
            '"$LIMBERCLOUD_CONDA_ENV" "${UNRELATED_SETTING-unset}"',
            dotenv_text=(
                'export LIMBERCLOUD_RUNTIME_ROOT="/runtime path"\n'
                "UNRELATED_SETTING=not-exported\n"
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["/runtime path", "CosmoConda", "unset"],
        )

    def test_exported_canonical_values_override_dotenv(self):
        result = self.run_loader(
            'printf "%s\\n%s\\n" "$LIMBERCLOUD_RUNTIME_ROOT" '
            '"$LIMBERCLOUD_CONDA_ENV"',
            dotenv_text=(
                "LIMBERCLOUD_RUNTIME_ROOT=/from-file\n"
                "LIMBERCLOUD_CONDA_ENV=from-file\n"
            ),
            LIMBERCLOUD_RUNTIME_ROOT="/from-environment",
            LIMBERCLOUD_CONDA_ENV="environment-name",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["/from-environment", "environment-name"],
        )

    def test_does_not_interpolate_or_execute_dotenv_values(self):
        marker = self.root / "must-not-exist"
        result = self.run_loader(
            'printf "%s\\n" "$LIMBERCLOUD_RUNTIME_ROOT"',
            dotenv_text=(
                'LIMBERCLOUD_RUNTIME_ROOT="$(touch '
                f'{marker})"\n'
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            f"$(touch {marker})",
        )
        self.assertFalse(marker.exists())

    def test_requires_nonempty_runtime_root(self):
        result = self.run_loader(
            ":",
            dotenv_text="LIMBERCLOUD_RUNTIME_ROOT=\n",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LIMBERCLOUD_RUNTIME_ROOT is required", result.stderr)

    def test_accepts_conda_selector_without_inspecting_it(self):
        missing_environment = self.root / "this-environment-does-not-exist"
        result = self.run_loader(
            'printf "%s\\n" "$LIMBERCLOUD_CONDA_ENV"',
            dotenv_text=(
                "LIMBERCLOUD_RUNTIME_ROOT=/runtime\n"
                f"LIMBERCLOUD_CONDA_ENV={missing_environment}\n"
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(missing_environment))

    def test_explicit_missing_dotenv_is_an_error(self):
        missing_dotenv = self.root / "missing.env"
        result = self.run_loader(
            ":",
            LIMBERCLOUD_ENV_FILE=str(missing_dotenv),
            LIMBERCLOUD_RUNTIME_ROOT="/runtime",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LIMBERCLOUD_ENV_FILE does not exist", result.stderr)

    def test_default_missing_dotenv_is_allowed(self):
        isolated_loader = self.root / "project" / "scripts" / "nersc"
        isolated_loader.mkdir(parents=True)
        shutil.copy2(LOADER, isolated_loader / LOADER.name)
        command = (
            f"source {shlex.quote(str(isolated_loader / LOADER.name))}\n"
            'printf "%s\\n" "$LIMBERCLOUD_CONDA_ENV"'
        )
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", command],
            cwd=self.root,
            env=self.clean_environment(LIMBERCLOUD_RUNTIME_ROOT="/runtime"),
            check=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "CosmoConda")

    def test_legacy_conda_alias_is_supported_with_warning(self):
        result = self.run_loader(
            'printf "%s\\n" "$LIMBERCLOUD_CONDA_ENV"',
            dotenv_text="LIMBERCLOUD_RUNTIME_ROOT=/runtime\n",
            CosmoENV="legacy-environment",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "legacy-environment")
        self.assertIn("CosmoENV is deprecated", result.stderr)

    def test_legacy_script_alias_derives_and_validates_root(self):
        onecovariance_root = self.root / "OneCovariance checkout"
        onecovariance_root.mkdir()
        covariance_script = onecovariance_root / "covariance.py"
        covariance_script.write_text("# fixture\n", encoding="utf-8")
        result = self.run_loader(
            'limbercloud_require_onecovariance\n'
            'printf "%s\\n" "$LIMBERCLOUD_ONECOVARIANCE_ROOT"',
            dotenv_text=(
                "LIMBERCLOUD_RUNTIME_ROOT=/runtime\n"
                f'ONECOVARIANCE_SCRIPT="{covariance_script}"\n'
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(onecovariance_root))
        self.assertIn("ONECOVARIANCE_SCRIPT is deprecated", result.stderr)

    def test_legacy_root_alias_accepts_historical_script_path(self):
        onecovariance_root = self.root / "OneCovariance"
        onecovariance_root.mkdir()
        covariance_script = onecovariance_root / "covariance.py"
        covariance_script.write_text("# fixture\n", encoding="utf-8")
        result = self.run_loader(
            'limbercloud_require_onecovariance\n'
            'printf "%s\\n" "$LIMBERCLOUD_ONECOVARIANCE_ROOT"',
            dotenv_text=(
                "LIMBERCLOUD_RUNTIME_ROOT=/runtime\n"
                f"ONE_COVARIANCE_ROOT={covariance_script}\n"
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(onecovariance_root))
        self.assertIn("ONE_COVARIANCE_ROOT is deprecated", result.stderr)

    def test_onecovariance_validation_reports_missing_script(self):
        onecovariance_root = self.root / "OneCovariance"
        onecovariance_root.mkdir()
        result = self.run_loader(
            "limbercloud_require_onecovariance",
            dotenv_text=(
                "LIMBERCLOUD_RUNTIME_ROOT=/runtime\n"
                f"LIMBERCLOUD_ONECOVARIANCE_ROOT={onecovariance_root}\n"
            ),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("covariance.py was not found", result.stderr)

    def test_rejects_malformed_recognized_assignment_without_execution(self):
        marker = self.root / "must-not-exist"
        result = self.run_loader(
            ":",
            dotenv_text=(
                "LIMBERCLOUD_RUNTIME_ROOT=/runtime\n"
                f"touch {marker}\n"
            ),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid assignment", result.stderr)
        self.assertFalse(marker.exists())


class ModuleProfileTests(unittest.TestCase):
    def run_profile(self, profile):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            log = root / "module.log"
            command = (
                'module() { printf "%s\\n" "$*" >> "$MODULE_LOG"; }\n'
                f"source {shlex.quote(str(MODULES / profile))}"
            )
            result = subprocess.run(
                ["bash", "--noprofile", "--norc", "-c", command],
                cwd=root,
                env={
                    "HOME": str(root),
                    "LANG": "C",
                    "MODULE_LOG": str(log),
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                },
                check=False,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=True,
            )
            entries = log.read_text(encoding="utf-8").splitlines()
            return result, entries

    def test_cpu_profile_selects_cpu_before_common_modules(self):
        result, entries = self.run_profile("cpu.sh")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            entries,
            [
                "load cpu",
                "load conda",
                "load cray-mpich",
                "load PrgEnv-gnu",
                "load cray-hdf5-parallel",
            ],
        )

    def test_gpu_profile_selects_gpu_before_common_modules(self):
        result, entries = self.run_profile("gpu.sh")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            entries,
            [
                "load gpu",
                "load conda",
                "load cray-mpich",
                "load PrgEnv-gnu",
                "load cray-hdf5-parallel",
            ],
        )


if __name__ == "__main__":
    unittest.main()
