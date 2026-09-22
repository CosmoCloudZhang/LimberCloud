"""Isolated contract tests for LimberCloud configuration helpers."""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOADER = REPOSITORY_ROOT / "scripts" / "load_config.sh"
MODULES = REPOSITORY_ROOT / "scripts" / "nersc" / "modules"


class EnvironmentLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        # The loader reports physical paths from ``pwd -P``. A temporary
        # directory under a symlinked prefix such as macOS /var -> /private/var
        # spells the same directory differently, so normalise before comparing.
        self.root = Path(self.temporary_directory.name).resolve()
        self.project = self.root / "LimberCloud"
        (self.project / "scripts").mkdir(parents=True)
        (self.project / "src" / "limbercloud").mkdir(parents=True)
        shutil.copy2(LOADER, self.project / "scripts" / "load_config.sh")
        self.dotenv = self.project / ".env"

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
        loader = self.project / "scripts" / "load_config.sh"
        command = (
            f"source {shlex.quote(str(loader))} || exit $?\n"
            f"{shell_body}"
        )
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", command],
            cwd=self.project,
            env=self.clean_environment(**environment_updates),
            check=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

    def test_loads_allowlisted_values(self):
        result = self.run_loader(
            'printf "%s\\n%s\\n%s\\n" "$LIMBERCLOUD_RUNTIME_ROOT" '
            '"$PROJECT_ROOT" "${UNRELATED_SETTING-unset}"',
            dotenv_text=(
                'export LIMBERCLOUD_RUNTIME_ROOT="/runtime path"\n'
                "UNRELATED_SETTING=not-exported\n"
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["/runtime path", str(self.project), "unset"],
        )

    def test_exported_canonical_values_override_dotenv(self):
        result = self.run_loader(
            'printf "%s\\n" "$LIMBERCLOUD_RUNTIME_ROOT"',
            dotenv_text="LIMBERCLOUD_RUNTIME_ROOT=/from-file\n",
            LIMBERCLOUD_RUNTIME_ROOT="/from-environment",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "/from-environment")

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

    def test_rejects_obsolete_conda_selector(self):
        result = self.run_loader(
            ":",
            dotenv_text=(
                "LIMBERCLOUD_RUNTIME_ROOT=/runtime\n"
                "LIMBERCLOUD_CONDA_ENV=CosmoConda\n"
            ),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("obsolete key LIMBERCLOUD_CONDA_ENV", result.stderr)

    def test_default_missing_dotenv_is_allowed_with_exported_runtime(self):
        result = self.run_loader(
            'printf "%s\\n" "$LIMBERCLOUD_RUNTIME_ROOT"',
            LIMBERCLOUD_RUNTIME_ROOT="/runtime",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "/runtime")

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

    def test_nested_manuscript_git_is_not_selected_as_project_root(self):
        manuscript = self.project / "manuscript"
        manuscript.mkdir()
        subprocess.run(
            ["git", "init", "--quiet", str(manuscript)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        (manuscript / "notes.txt").write_text("fixture\n", encoding="utf-8")
        result = subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                (
                    f"cd {shlex.quote(str(manuscript))} && "
                    f"source {shlex.quote(str(self.project / 'scripts' / 'load_config.sh'))} && "
                    'printf "%s\\n" "$PROJECT_ROOT"'
                ),
            ],
            cwd=manuscript,
            env=self.clean_environment(LIMBERCLOUD_RUNTIME_ROOT="/runtime"),
            check=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(self.project))
        self.assertNotEqual(result.stdout.strip(), str(manuscript))


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

    def test_common_profile_disables_hdf5_file_locking(self):
        command = (
            'module() { :; }\n'
            f"source {shlex.quote(str(MODULES / 'common.sh'))}\n"
            'printf "%s\\n" "$HDF5_USE_FILE_LOCKING"'
        )
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", command],
            env={
                "HOME": str(REPOSITORY_ROOT),
                "LANG": "C",
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            },
            check=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "FALSE")


if __name__ == "__main__":
    unittest.main()
