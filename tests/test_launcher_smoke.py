"""Dry-run every shell launcher without invoking NERSC or Conda."""

import subprocess
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments"
GENERATOR_ROOT = REPOSITORY_ROOT / "scripts" / "generate_config"


class LauncherSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.runtime_root = self.root / "runtime"
        self.runtime_root.mkdir()
        self.onecovariance_root = self.root / "OneCovariance"
        self.onecovariance_root.mkdir()
        (self.onecovariance_root / "covariance.py").write_text(
            "# smoke-test fixture\n",
            encoding="utf-8",
        )
        self.command_log = self.root / "commands.log"
        self.stub_bin = self.root / "bin"
        self.stub_bin.mkdir()
        stub = """#!/usr/bin/env bash
printf '%s' "${0##*/}" >> "${LIMBERCLOUD_SMOKE_LOG}"
printf ' <%s>' "$@" >> "${LIMBERCLOUD_SMOKE_LOG}"
printf '\\n' >> "${LIMBERCLOUD_SMOKE_LOG}"
"""
        for command in ("conda", "mkdir", "module", "python", "sbatch", "srun"):
            path = self.stub_bin / command
            path.write_text(stub, encoding="utf-8")
            path.chmod(0o755)

        self.venv_prefix = (REPOSITORY_ROOT / ".venv").resolve()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def clean_environment(self):
        return {
            "HOME": str(self.root),
            "LANG": "C",
            "LIMBERCLOUD_RUNTIME_ROOT": str(self.runtime_root),
            "LIMBERCLOUD_ONECOVARIANCE_ROOT": str(self.onecovariance_root),
            "LIMBERCLOUD_SMOKE_LOG": str(self.command_log),
            "PATH": f"{self.stub_bin}:/usr/bin:/bin",
            "SLURM_CPUS_PER_TASK": "2",
        }

    def run_launcher(self, path):
        return subprocess.run(
            ["bash", "--noprofile", "--norc", str(path)],
            cwd=REPOSITORY_ROOT,
            env=self.clean_environment(),
            check=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

    def test_all_batch_launchers_reach_only_stubbed_commands(self):
        shell_files = sorted(EXPERIMENT_ROOT.rglob("*.sh")) + sorted(
            GENERATOR_ROOT.glob("*.sh")
        )
        launchers = [path for path in shell_files if "#SBATCH" in path.read_text()]

        self.assertEqual(len(launchers), 34)
        activate_token = f"conda <activate> <{self.venv_prefix}>"
        for path in launchers:
            self.command_log.unlink(missing_ok=True)
            result = self.run_launcher(path)
            relative_path = path.relative_to(REPOSITORY_ROOT)
            with self.subTest(path=relative_path):
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.command_log.read_text(encoding="utf-8")
                self.assertIn(activate_token, commands)
                self.assertIn("module <load> <conda>", commands)

                if "experiments/spectra/JAX/GPU" in relative_path.as_posix():
                    self.assertIn("module <load> <gpu>", commands)
                else:
                    self.assertIn("module <load> <cpu>", commands)

    def test_run_all_launchers_preflight_and_submit_six_stubbed_jobs(self):
        launchers = sorted(EXPERIMENT_ROOT.rglob("Run_All.sh"))

        self.assertEqual(len(launchers), 4)
        for path in launchers:
            self.command_log.unlink(missing_ok=True)
            result = self.run_launcher(path)
            with self.subTest(path=path.relative_to(REPOSITORY_ROOT)):
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.command_log.read_text(encoding="utf-8")
                self.assertEqual(commands.count("sbatch "), 6)

    def test_invalid_configuration_stops_before_module_or_conda(self):
        launcher = EXPERIMENT_ROOT / "spectra" / "NUMBA" / "Y1" / "single.sh"
        environment = self.clean_environment()
        environment["LIMBERCLOUD_RUNTIME_ROOT"] = ""

        result = subprocess.run(
            ["bash", "--noprofile", "--norc", str(launcher)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LIMBERCLOUD_RUNTIME_ROOT is required", result.stderr)
        self.assertFalse(self.command_log.exists())

    def test_copied_batch_script_resolves_repo_from_submit_directory(self):
        launcher = EXPERIMENT_ROOT / "spectra" / "CCL" / "Y1" / "single.sh"
        copied = self.root / "slurm_script"
        copied.write_text(launcher.read_text(encoding="utf-8"), encoding="utf-8")
        copied.chmod(0o755)

        environment = self.clean_environment()
        environment["SLURM_SUBMIT_DIR"] = str(REPOSITORY_ROOT)

        self.command_log.unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", str(copied)],
            cwd=self.root,
            env=environment,
            check=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.command_log.read_text(encoding="utf-8")
        self.assertIn(f"conda <activate> <{self.venv_prefix}>", commands)
        self.assertNotIn("fatal: not a git repository", result.stderr)


if __name__ == "__main__":
    unittest.main()
