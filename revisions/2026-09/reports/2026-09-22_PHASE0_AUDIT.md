# Phase 0 / 0A review against current HEAD

Audit date: 2026-09-22. Read-only review of baseline `c0cf5fae8d9d48b51710f942f2cc7ad24779e4fc`, environment implementation `d664420db0c2991fe3365d73a3b72b27f838a9eb`, report update `cb3e567`, and current integrated source `13a3c2deb3e72bd22cfad5f4eb1a6191899a47d6`. No production source, environment, manuscript, Git index, or runtime data was changed by this audit. The local `manuscript/` repository and existing untracked endpoint review were left intact.

## Outcome

The environment stage delivered useful real work: a separate environment recipe, preserved prior environment (as remotely reported), a non-executing fixed `.env` loader, `.venv`-selected activation for batch jobs, removed obsolete `--path` arguments, explicit sample controls, and recorded allocated dependency smokes. It is not a complete current acceptance gate. The next implementation prompt should begin with the bounded repairs below, then run real end-to-end tiny execution. Do not recreate the already-tested environment merely to satisfy a rewritten phase label.

The largest current workflow problems are a deleted manuscript gitlink, no-op default launches that can overwrite legacy timing files, ignored shell arguments, a fiducial-only flag that does not actually evaluate the fiducial in the 24 drivers, and local tests that are not self-contained. Some belong to the subsequent partial science commit; they should be handled as current-state cleanup rather than attributed incorrectly to phase 0.

## Findings requiring concrete repair

### P1 — Restore the manuscript gitlink deleted by the later science commit

Evidence: `git ls-tree c0cf5fa manuscript` and `git ls-tree d664420 manuscript` both return mode `160000`, commit `90d12f4f3e574a67c25944d27d7ded553e09402b`. `git ls-tree HEAD manuscript` returns nothing. `git show --stat 13a3c2d -- manuscript` records the deletion. `.gitmodules:1-3` still declares the submodule. Current local `git status --short` shows `?? manuscript/`, while `manuscript/.git` remains a Git file. The C01 report's opening lines describe the absent/deinitialized path as left untouched, but the final commit did remove the tracked gitlink.

Impact: future clones no longer know the paper pin; `git submodule update ... manuscript` and the documented parent/paper workflow cannot function reliably, and a broad add could ingest the local paper as ordinary files.

Fix: restore the parent index/tree's exact mode-160000 reference at the recorded paper commit. Preserve the initialized local paper worktree and `.gitmodules`. On NERSC do not initialize the paper to repair its parent reference. This is a bounded code-repository repair, not manuscript editing.

Acceptance: both `git ls-tree HEAD manuscript` and the staged diff show precisely the intended gitlink; a temporary paper-uninitialized code checkout still installs/builds/runs ordinary checks. No paper files become ordinary parent entries.

### P1 — A safe zero-sample default can erase existing timing artifacts

Evidence: `experiments/spectra/CCL/Y1/single.py:83-105` resolves omitted count or `--fiducial-only` to zero iterations. Lines 135-138 nevertheless unconditionally call `numpy.savetxt` on `Time_Single_<allocation>*.txt`. `run_id` is optional (line 23, output path line 52). The other 23 runners use the same timing pattern. The shell at `experiments/spectra/CCL/Y1/single.sh:54` passes no sample controls, so ordinary submission reaches this zero-row behavior. `Run_All.sh:24-27` repeats it for six jobs.

Impact: a seemingly harmless smoke/default launch can truncate existing legacy timing evidence into empty files, despite doing no science. A positive but tiny count can likewise overwrite legacy products without a distinct run identity.

Fix: validate effective work and output identity before reading large inputs/creating outputs. A no-op selection should exit clearly without writes (or be rejected), and all new runs should require/use an isolated run ID, preserving old products read-only. The shared execution work can replace the duplicated path as long as this protection exists immediately.

Acceptance: seed a synthetic existing timing file; invoking omitted controls and a rejected/inconsistent selection leaves its contents and metadata unchanged. A tiny accepted run writes only its own namespace.

### P2 — Batch launchers ignore user-supplied execution controls

Evidence: `experiments/spectra/CCL/Y1/single.sh:54` has a fixed Python invocation with no `"$@"` or equivalent parser; the same applies to all 24 spectrum launchers. `experiments/spectra/CCL/Run_All.sh:26` also omits argument propagation. The Python CLI now advertises `--sample-count`, `--sample-table`, `--fiducial-only`, `--include-fiducial`, `--run-id`, and other controls, but passing them after the `.sh` path does not reach Python. `documents/nersc.md:72-79` shows submission followed by the instruction to explicitly choose sample count, without an effective way to do so through the shown launcher.

Fix: implement one validated forwarding/launch path and use it consistently. Pass explicit bounded pilot/campaign selections; retain one task and `srun -n 1`. Validate unknown/inconsistent flags before expensive allocation work where feasible. Ensure Run_All forwards the same coherent run/sample identity into every child configuration.

Acceptance: stub srun/sbatch and assert the exact requested flags reach each driver; rejection is tested. A tiny allocated real driver reports the requested sample IDs. Do not use the old wrapper as proof that an intended count ran.

### P2 — `--fiducial-only` currently means no evaluation, and some exposed flags are ignored

Evidence: `src/limbercloud/experiments/sample_controls.py:17-47,92-105` resolves fiducial-only to zero sampled rows. That is correct for the count, but `validation/samples.py:328-337` rejects include-fiducial and returns an empty list for zero. `CCL/Y1/single.py:105` loops over sampled count only. Therefore no fiducial is computed by these drivers. The helper adds `--run-config` and `--mode` at `sample_controls.py:151-171`, yet `single.py:160-171` forwards neither to `main`; they silently do nothing. The C01 report explicitly acknowledges that drivers still lack a fiducial execution path.

Fix: complete the real shared runner with distinct fiducial warmup/evaluation and sampled timing counts. Until an argument is implemented, reject it rather than silently accepting it. Enforce the run configuration in real execution, not solely helper tests.

Acceptance: a tiny instrumented evaluator observes ID 0 once for fiducial-only; ID 0 plus IDs 1..N for an explicitly requested mixed run; sampled timing includes only IDs 1..N. Tests must exercise actual driver dispatch, not just `evaluation_sample_ids`.

### P2 — Local environment/launcher tests fail and depend on the real `.venv`

Executed safely on local macOS with stdlib tooling:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -p test_environment_contracts.py -v`: 12 tests, 4 failures.
- `... -p test_launcher_smoke.py -v`: 4 test methods, 35 failure instances (34 batch subtests and copied-script test) because checkout `.venv` is absent. Invalid-config and Run_All stubs passed.

Concrete causes:

1. `tests/test_environment_contracts.py:20,67-69,187-188` compares `/var/folders/...` temporary path spellings to the loader's physical `/private/var/folders/...` path. Resolve expected paths before comparing; this is a fixture bug, not incorrect loader root selection.
2. `scripts/nersc/modules/cpu.sh:5-8` and `gpu.sh:5-8` evaluate `BASH_SOURCE[0]` inside command substitution. Under `/bin/bash` 3.2 when sourced, this loses the expected path. Direct reproduction with a harmless module function printed `load cpu`, then tried repository-root `common.sh` and failed. The test captures only the first module load. Capture the source path before command substitution. These are NERSC-targeted profiles, but mocked local tests need portability or explicit documented host requirements.
3. `tests/test_launcher_smoke.py:38,54-62` uses the real repository and real `.venv`. It stubs python/conda on PATH but `activate_venv.sh:14-30` first checks a real prefix and executable. Build a temporary project fixture with a fake `.venv/bin/python`, copy required helpers and scripts, and never depend on or alter user environment selection.
4. All spectrum launchers use `${LABEL,,}` (e.g. CCL single.sh:54), unsupported in macOS Bash 3.2. This will surface after the fixture issue. Replace with an already-known lowercase token or portable normalization, or explicitly execute a declared supported newer Bash in NERSC-only smoke tests. The portable loader itself passed its value/non-execution checks.

Acceptance: the isolated suites pass with absent real `.venv`, under declared supported Bash versions, from an arbitrary working directory and a path containing spaces, including copied Slurm scripts and nested-paper fixtures.

### P2 — Kernel acceptance is only an import-path probe; selected-prefix activation is incomplete

Evidence: `scripts/jupyter/launch_kernel.sh:9-12,20-32` runs `.venv/bin/python` after config loading and setting CFS file locking, but does not source site modules or Conda activation hooks. Batch launchers do both before Python. `--probe` at lines 27-29 imports only `limbercloud` and checks a runtime-root variable. `C00_ENV_MAINTENANCE_STAGE_REPORT.md:140` explicitly says the full notebook session beyond probe was not exercised.

Risk: an editor/Jupyter server without the preloaded GNU/Cray/HDF5 stack or Conda hooks can use the correct interpreter but lack required shared-library state. The remote report's allocated command smokes do not establish this kernel path.

Fix: give kernel startup a documented host-aware bootstrap consistent with the selected prefix and supported CPU/GPU profile, or require and verify the equivalent inherited environment. Portable macOS must not load NERSC modules. Keep the login-safe probe lightweight and use a separate real kernel smoke on an allocation for scientific/HDF5/device imports.

Acceptance: a fresh selected LimberCloud kernel from the actual supported editor/Jupyter launch path reports the same interpreter, package paths, modules/hooks, and backend identity as a batch job; tiny allocated HDF5 roundtrip and backend operation succeed. Do not import potentially MPI-initializing HDF5 on login just to broaden the probe.

### P2 — External OneCovariance checkout does not provide a separate Python environment

Evidence: `experiments/covariance/Y1/matrix.sh:33-35` activates the selected LimberCloud prefix, then lines 54-55 invoke both preparation and external `covariance.py` with that same `python`. `C00_ENV_MAINTENANCE_STAGE_REPORT.md:38` and `documents/environment.md:69-71` say OneCovariance requirements stay outside the minimal environment because its checkout is external. The report lists requirements/build concerns but no end-to-end OneCovariance startup check in the new prefix.

Risk: checkout location does not isolate imports/native libraries. Missing upstream runtime/build dependencies can fail the covariance phase even though CCL/CAMB/MPI/JAX probes passed. This review does not claim a particular dependency is actually missing remotely; it identifies an unclosed compatibility gate.

Fix: reconcile the installed upstream revision's runtime/native extension dependencies with the exact interpreter that matrix.sh invokes. Under the existing single-prefix contract, make that prefix sufficient; otherwise record an explicit reviewed separate upstream execution environment instead of implying directory location solves it.

Acceptance: allocated minimal upstream import/startup plus a bounded covariance sentinel at pinned revision, with producing interpreter/version evidence. Do not run a full production covariance to discover basic import failures.

### P2 — MPI/HDF5 installer does not enforce its advertised target or replacement guarantees

Evidence: `scripts/nersc/install_mpi_h5py.sh:27-35` checks only that CONDA_PREFIX exists and is not named CosmoConda; it permits base or any unrelated environment and does not compare the invoked Python's sys.prefix against the target. It mutates pip/NumPy and force-removes HDF5 at lines 42-48. `mpi4py` installation at lines 56-58 lacks force-reinstall/version pin: an already-satisfied generic wheel can remain untouched despite the source-build claim. HDF5 and mpi4py installation versions are open-ended; remote-tested h5py 3.16.0 also differs from portable recipe's <3.16 bound (not automatically wrong, but it must be explicit).

Fix: require and validate the intended dedicated prefix/interpreter (refuse base and unrelated prefixes); verify the GNU/Cray stack before mutation; deterministically rebuild when provenance differs, or fail clearly rather than accepting an unknown existing mpi4py wheel. Pin or record reproducible source-build versions and extension linkage. No reason to rebuild an already verified correct environment merely for this audit.

Acceptance: stubbed tests reject base/CosmoConda/mismatched Python before any package mutation and exercise existing-generic-mpi4py handling; actual rebuild, only if required, is checked on an allocation and preserves the old working environment.

### P2 — `make check` is no longer uniformly login-safe with MPI-enabled HDF5

Evidence: `Makefile:3-4` discovers every test. The added `tests/test_science_artifacts.py:339-368` executes real HDF5 checkpoint operations and imports h5py. The accepted NERSC prefix uses MPI-enabled h5py according to the C00 report. `documents/nersc.md:92-96` still describes `make check` first as lightweight and puts allocated HDF5 checks later.

Fix: separate clearly named fast/static tests from HDF5/runtime tests or run the entire applicable suite in a supported allocation when using the MPI HDF5 build. Preserve full test coverage; do not skip runtime verification silently. Plain local serial-h5py runs can remain ordinary local tests.

Acceptance: reports name commands, host/allocation, and suite selection explicitly. No new test suite accidentally initializes MPI on a login host.

## Smaller maintenance items

- Scripts retain duplicate project-root walkers. A copied batch script works only when SLURM_SUBMIT_DIR is the project root; submitting from a nested directory is not walked upward from the submit dir (`single.sh:19-30`). Either document/validate root-only submission or support walking from both submit dir and script dir in one helper. Test the behavior.
- Durable logs remain unresolved: `Run_All.sh:22` creates PSCRATCH checkout logs and Slurm outputs use `logs/%x_%j.out`, while README describes runtime-root logs. Preserve acceptance/provenance logs on CFS or copy them into the accepted report bundle; PSCRATCH must not be their sole durable home.
- `pyproject.toml:35-36` declares `py.typed`, but no such source file was found. Add a genuine typing marker only if intended, otherwise remove the package-data entry. This is not a blocker for scientific correctness.
- The project-wide Ruff configuration enables import sorting but the report only claims touched surfaces passed. Run the documented full `make check` equivalent in the correct environment; report legacy findings honestly and keep formatting changes bounded. Do not infer full compliance from a subset.
- README and environment docs contain old stage numbers/claims after current partial stage 1 work. Update them only to actual behavior after repairs. Historical reports should remain historical, with a follow-up explaining superseded status and the restored gitlink.

## Evidence achieved in this local audit

- Notebook static validator: **40 notebooks passed**. It parses JSON/Python AST; no notebook cells were executed.
- Python AST parsing: **66 files passed** across src, experiments, scripts.
- `/bin/bash -n`: **48 shell files passed**. Syntax parsing does not establish Bash 3.2 runtime expansion compatibility.
- Environment test results and launcher failures described above were reproduced with command/module stubs. No science imports, Conda installation, MPI initialization, jobs, or data production were performed.
- The reports' Perlmutter dependency results are remotely reported evidence, not locally reproduced: two-rank MPI job 58590099; Numba/JAX CPU job 58590111; JAX GPU job 58590157; reported CFS HDF5 roundtrip and CCL/CAMB tiny distance. Retain these as prior evidence and request exact retained log/artifact identities if needed for final acceptance.
- No full scientific driver main() or fiducial correctness result is established by phase0/0A reports. The C01 report explicitly says none was executed.

## Proposed next prompt acceptance boundary

Start with existing environment inventory reconciliation, restore the gitlink, repair isolated portable tests and launcher argument/no-op behavior, close the selected-prefix kernel/upstream dependency gates, and retain prior accepted allocated evidence where still applicable. Complete real shared fiducial/sample execution under the settled science contract before any expensive pilot. Return a bounded diff, current code/paper pin, commands and execution hosts, tiny-run observed sample IDs, produced artifact identities, failures, and remaining gate. Automatic commit/push may preserve checkpoints, but it does not by itself accept the stage.
