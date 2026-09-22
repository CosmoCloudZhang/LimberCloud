"""Dry-run every shell launcher inside a synthetic checkout.

These tests never touch the developer's real ``.venv``, Conda installation or
NERSC modules. A temporary project fixture supplies the checkout markers, a
stub ``.venv/bin/python`` and a runtime configuration, and stub commands record
their own argv. That is what makes the argument-forwarding assertions
meaningful: the recorded argv is the argv the driver would have received.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments"
GENERATOR_ROOT = REPOSITORY_ROOT / "scripts" / "generate_config"

# Directories copied into the synthetic checkout. Only launch machinery is
# needed; the science modules are never imported by a stubbed run.
FIXTURE_TREES = ("experiments", "scripts")

STUB_COMMAND = """#!/usr/bin/env bash
printf '%s' "${0##*/}" >> "${LIMBERCLOUD_SMOKE_LOG}"
for argument in "$@"; do
    printf ' <%s>' "${argument}" >> "${LIMBERCLOUD_SMOKE_LOG}"
done
printf '\\n' >> "${LIMBERCLOUD_SMOKE_LOG}"
"""

STUB_COMMANDS = ("conda", "mkdir", "module", "python", "sbatch", "srun")


def batch_launchers(root):
    """Return every ``#SBATCH`` launcher under one checkout root."""

    candidates = sorted((root / "experiments").rglob("*.sh")) + sorted(
        (root / "scripts" / "generate_config").glob("*.sh")
    )
    return [path for path in candidates if "#SBATCH" in path.read_text()]


class LauncherFixture(unittest.TestCase):
    """Build a throwaway checkout with stub commands and a stub interpreter."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        # A path containing a space is the interesting case for quoting.
        self.root = Path(self.temporary_directory.name).resolve() / "work space"
        self.root.mkdir()
        self.project = self.root / "LimberCloud"
        self.project.mkdir()

        for tree in FIXTURE_TREES:
            shutil.copytree(REPOSITORY_ROOT / tree, self.project / tree)
        # Checkout markers the loader looks for.
        (self.project / "src" / "limbercloud").mkdir(parents=True)
        (self.project / "src" / "limbercloud" / "__init__.py").write_text(
            "# synthetic checkout marker\n", encoding="utf-8"
        )
        (self.project / "logs").mkdir()

        # A stub interpreter inside a stub .venv prefix, so activate_venv.sh
        # validates a real prefix without needing this user's environment. It
        # answers a sys.prefix query so the installer's target check has
        # something to verify, and otherwise records its argv.
        self.venv_prefix = self.project / ".venv"
        (self.venv_prefix / "bin").mkdir(parents=True)
        interpreter = self.venv_prefix / "bin" / "python"
        interpreter.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ ${1:-} == "-c" && ${2:-} == *sys.prefix* ]]; then\n'
            f'    printf "%s\\n" "{self.venv_prefix}"\n'
            "    exit 0\n"
            "fi\n" + STUB_COMMAND.split("\n", 1)[1],
            encoding="utf-8",
        )
        interpreter.chmod(0o755)

        self.runtime_root = self.root / "runtime root"
        self.runtime_root.mkdir()
        self.onecovariance_root = self.root / "OneCovariance"
        self.onecovariance_root.mkdir()
        (self.onecovariance_root / "covariance.py").write_text(
            "# smoke-test fixture\n", encoding="utf-8"
        )
        (self.project / ".env").write_text(
            f'LIMBERCLOUD_RUNTIME_ROOT="{self.runtime_root}"\n'
            f'LIMBERCLOUD_ONECOVARIANCE_ROOT="{self.onecovariance_root}"\n',
            encoding="utf-8",
        )

        self.command_log = self.root / "commands.log"
        self.stub_bin = self.root / "bin"
        self.stub_bin.mkdir()
        for command in STUB_COMMANDS:
            path = self.stub_bin / command
            path.write_text(STUB_COMMAND, encoding="utf-8")
            path.chmod(0o755)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def clean_environment(self, **updates):
        environment = {
            "HOME": str(self.root),
            "LANG": "C",
            "LIMBERCLOUD_SMOKE_LOG": str(self.command_log),
            "PATH": f"{self.stub_bin}:/usr/bin:/bin",
            "SLURM_CPUS_PER_TASK": "2",
        }
        environment.update(updates)
        return environment

    def run_launcher(self, path, arguments=(), cwd=None, **environment_updates):
        self.command_log.unlink(missing_ok=True)
        return subprocess.run(
            ["bash", "--noprofile", "--norc", str(path), *arguments],
            cwd=str(cwd if cwd is not None else self.project),
            env=self.clean_environment(**environment_updates),
            check=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

    def recorded_commands(self):
        return self.command_log.read_text(encoding="utf-8")


class LauncherSmokeTests(LauncherFixture):
    def test_all_batch_launchers_reach_only_stubbed_commands(self):
        launchers = batch_launchers(self.project)

        self.assertEqual(len(launchers), 34)
        activate_token = f"conda <activate> <{self.venv_prefix}>"
        for path in launchers:
            result = self.run_launcher(path)
            relative_path = path.relative_to(self.project)
            with self.subTest(path=relative_path):
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.recorded_commands()
                self.assertIn(activate_token, commands)
                self.assertIn("module <load> <conda>", commands)

                if "experiments/spectra/JAX/GPU" in relative_path.as_posix():
                    self.assertIn("module <load> <gpu>", commands)
                else:
                    self.assertIn("module <load> <cpu>", commands)

    def test_spectra_launchers_forward_arguments_with_quoting_intact(self):
        table = self.runtime_root / "sample table"
        arguments = [
            "--sample-count=3",
            "--sample-table",
            str(table),
        ]
        launchers = [
            path
            for path in batch_launchers(self.project)
            if path.parent.name in {"Y1", "Y10"}
            and "spectra" in path.relative_to(self.project).as_posix()
        ]

        self.assertEqual(len(launchers), 24)
        for path in launchers:
            result = self.run_launcher(path, arguments)
            with self.subTest(path=path.relative_to(self.project)):
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.recorded_commands()
                self.assertIn("<--sample-count=3>", commands)
                self.assertIn(f"<--sample-table> <{table}>", commands)
                # The lower-case driver name must survive on Bash 3.2, where
                # ${LABEL,,} is a syntax error.
                self.assertRegex(commands, r"<[^>]*/(single|double|triple)\.py>")

    def test_benchmark_and_generator_wrappers_forward_arguments(self):
        wrappers = sorted(
            (self.project / "experiments" / "benchmarks").rglob("figure.sh")
        )
        wrappers += sorted((self.project / "scripts" / "generate_config").glob("*.sh"))

        self.assertEqual(len(wrappers), 8)
        for path in wrappers:
            arguments = (
                ["--folder", "pilot run"]
                if path.name == "figure.sh"
                else ["--eta-ia", "0.0"]
            )
            result = self.run_launcher(path, arguments)
            with self.subTest(path=path.relative_to(self.project)):
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.recorded_commands()
                self.assertIn(f"<{arguments[0]}> <{arguments[1]}>", commands)

    def test_run_all_launchers_forward_the_same_selection_to_six_jobs(self):
        launchers = sorted((self.project / "experiments").rglob("Run_All.sh"))

        self.assertEqual(len(launchers), 4)
        for path in launchers:
            result = self.run_launcher(
                path, ["--sample-count=5", "--sample-table", "pilot run"]
            )
            with self.subTest(path=path.relative_to(self.project)):
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.recorded_commands()
                self.assertEqual(commands.count("sbatch "), 6)
                self.assertEqual(commands.count("<--sample-count=5>"), 6)
                self.assertEqual(commands.count("<--sample-table> <pilot run>"), 6)

    def test_invalid_configuration_stops_before_module_or_conda(self):
        launcher = (
            self.project / "experiments" / "spectra" / "NUMBA" / "Y1" / "single.sh"
        )
        result = self.run_launcher(launcher, LIMBERCLOUD_RUNTIME_ROOT="")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LIMBERCLOUD_RUNTIME_ROOT is required", result.stderr)
        self.assertFalse(self.command_log.exists())

    def test_copied_batch_script_resolves_repo_from_submit_directory(self):
        launcher = self.project / "experiments" / "spectra" / "CCL" / "Y1" / "single.sh"
        copied = self.root / "slurm_script"
        copied.write_text(launcher.read_text(encoding="utf-8"), encoding="utf-8")
        copied.chmod(0o755)

        result = self.run_launcher(
            copied,
            cwd=self.root,
            SLURM_SUBMIT_DIR=str(self.project),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.recorded_commands()
        self.assertIn(f"conda <activate> <{self.venv_prefix}>", commands)
        self.assertNotIn("fatal: not a git repository", result.stderr)

    def test_missing_real_environment_still_reports_a_clear_error(self):
        shutil.rmtree(self.venv_prefix)
        launcher = (
            self.project / "experiments" / "spectra" / "NUMBA" / "Y1" / "single.sh"
        )
        result = self.run_launcher(launcher)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing .venv link", result.stderr)


class InstallerGuardTests(LauncherFixture):
    """The installer mutates an environment, so it must check its target first."""

    def run_installer(self, **environment_updates):
        installer = self.project / "scripts" / "nersc" / "install_mpi_h5py.sh"
        environment = {
            "CONDA_PREFIX": str(self.venv_prefix),
            "CONDA_DEFAULT_ENV": "limbercloud",
            "HDF5_DIR": str(self.root / "hdf5"),
            "LIMBERCLOUD_INSTALL_DRY_RUN": "1",
            # The activated prefix's own interpreter must come first, exactly as
            # it does after ``conda activate``.
            "PATH": f"{self.venv_prefix / 'bin'}:{self.stub_bin}:/usr/bin:/bin",
            "PROJECT_ROOT": str(self.project),
        }
        environment.update(environment_updates)
        self.command_log.unlink(missing_ok=True)
        # ``cc`` must look available without building anything.
        compiler = self.stub_bin / "cc"
        compiler.write_text(STUB_COMMAND, encoding="utf-8")
        compiler.chmod(0o755)
        return subprocess.run(
            ["bash", "--noprofile", "--norc", str(installer)],
            cwd=str(self.project),
            env=self.clean_environment(**environment),
            check=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

    def test_accepts_only_the_selected_prefix(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Dry run", result.stdout)

    def test_rejects_base_cosmoconda_and_unrelated_prefixes(self):
        other = self.root / "other-env"
        (other / "bin").mkdir(parents=True)
        cases = {
            "base": {"CONDA_DEFAULT_ENV": "base"},
            "CosmoConda": {"CONDA_DEFAULT_ENV": "CosmoConda"},
            "unrelated": {"CONDA_PREFIX": str(other)},
        }
        for name, updates in cases.items():
            with self.subTest(prefix=name):
                result = self.run_installer(**updates)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Dry run", result.stdout)

    def test_rejects_an_interpreter_outside_the_target_prefix(self):
        foreign_bin = self.root / "foreign" / "bin"
        foreign_bin.mkdir(parents=True)
        foreign = foreign_bin / "python"
        foreign.write_text(
            "#!/usr/bin/env bash\nprintf '/some/other/prefix\\n'\n", encoding="utf-8"
        )
        foreign.chmod(0o755)
        result = self.run_installer(PATH=f"{foreign_bin}:{self.stub_bin}:/usr/bin:/bin")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sys.prefix", result.stderr)


class OneCovarianceStartupTests(unittest.TestCase):
    """The external checkout must import in the interpreter that will run it."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_check(self, script):
        # The checker must run in the selected interpreter. Make's PYTHON
        # variable is not exported, and bare python3 on this host is 3.6.
        return subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "nersc" / "check_onecovariance.py"),
                str(script),
            ],
            check=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

    def test_reports_missing_dependencies_rather_than_the_directory(self):
        script = self.root / "covariance.py"
        script.write_text(
            "import os\nimport definitely_not_installed_package\n", encoding="utf-8"
        )
        result = self.run_check(script)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("definitely_not_installed_package", result.stderr)

    def test_accepts_a_resolvable_executable(self):
        script = self.root / "covariance.py"
        script.write_text("import os\nimport sys\n", encoding="utf-8")
        result = self.run_check(script)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dependency check passed", result.stdout)

    def test_missing_executable_is_an_error(self):
        result = self.run_check(self.root / "absent.py")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
