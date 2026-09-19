# C00 / Prompt 0A — Environment and maintenance stage report

**Stage date (UTC):** 2026-09-19  
**Host:** Perlmutter (`login` + interactive CPU/GPU allocations)  
**Checkout:** `/pscratch/sd/y/yhzhang/LimberCloud`  
**Baseline inventory:** `revisions/2026-09/reports/C00_REMOTE_HANDOFF_INVENTORY.md`  
**Authorization:** Prompt 0A — environment/maintenance only. No 1,001-row campaign. No manuscript initialization.

---

## 1. Executive summary

Dedicated `limbercloud` Conda environment created and validated alongside preserved CosmoConda. Configuration loader renamed to `scripts/load_config.sh` with `.venv`-only interpreter selection. Spectra runners gained `--sample-count` / `--fiducial-only` (default zero samples). Unused `--path` plumbing removed. Docs updated for the paper-submodule workflow. Bounded software checks and tiny allocated MPI/HDF5/CPU/GPU smokes passed.

| Item | Value |
| --- | --- |
| Paper pin (parent gitlink) | `90d12f4f3e574a67c25944d27d7ded553e09402b` |
| Pre-stage code commit | `c0cf5fae8d9d48b51710f942f2cc7ad24779e4fc` |
| `.venv` target after acceptance | `/global/homes/y/yhzhang/.conda/envs/limbercloud` |
| CosmoConda | Preserved at `/global/homes/y/yhzhang/.conda/envs/CosmoConda` |
| OneCovariance | `/global/homes/y/yhzhang/opt/OneCovariance` @ `311c2cfbe9d584d9d29abd3d8edaeea673470b7f` |

---

## 2. Environment recipes and installation

### Recipes

| File | Role |
| --- | --- |
| `environment.yml` | Portable CPU: `limbercloud`, CPU JAX, serial `h5py`, conda-forge `mpi4py`, explicit CAMB; **no** CosmoSIS; **no** mandatory CUDA JAX |
| `environment.nersc.yml` | NERSC CUDA JAX variant; scientific pins aligned; mpi4py/h5py deferred to site build |
| `scripts/nersc/create_environment.sh` | Creates named/prefix env; refuses CosmoConda; leaves existing `.venv` unchanged; `--nersc` selects NERSC YAML |
| `scripts/nersc/install_mpi_h5py.sh` | Source-builds mpi4py (`cc -shared`) and MPI h5py against `cray-hdf5-parallel`; removes conda serial h5py/hdf5; prepends `/usr/lib64/pkgconfig` for `cray-xpmem` |

### OneCovariance audit (`311c2cf`)

`conda_env.yaml` / install docs require Python ≤3.12.9 in their example env, plus `gfortran`, `gsl`, `pybind11`. LimberCloud’s minimal recipe does **not** install OneCovariance into `limbercloud`; covariance jobs continue to invoke the external checkout via `LIMBERCLOUD_ONECOVARIANCE_ROOT`.

### Selected interpreter

After validation, `.venv` → `limbercloud`. CosmoConda remains available for comparison.

---

## 3. Exact tested versions / modules / interpreters

### Python packages (`limbercloud`)

| Package | Version |
| --- | --- |
| Python | 3.12.14 |
| numpy | 2.2.6 |
| scipy | 1.13.1 |
| numba | 0.63.1 |
| astropy | 7.2.0 |
| matplotlib | 3.10.9 |
| pyccl | 3.2.1 |
| camb | 1.6.5 |
| jax | 0.9.2 (+ CUDA 12 plugins on NERSC) |
| h5py | 3.16.0 (**mpi True**; linked to Cray `libhdf5_parallel_gnu` 1.14.3) |
| mpi4py | 4.1.2 (source-built against Cray MPICH) |
| ipykernel | 7.1.0 |
| ruff | 0.16.7 |
| cosmosis | **absent** (excluded) |

### Modules (CPU profile)

`cpu/1.0`, `conda/Miniforge3-25.11.0-1`, `PrgEnv-gnu/8.7.0`, `gcc-native/14`, `cray-mpich/9.1.0`, `cray-hdf5-parallel/1.14.3.9`, `craype/2.7.36`

### HDF5 locking (CFS)

`scripts/nersc/modules/common.sh` exports `HDF5_USE_FILE_LOCKING=FALSE`. Verified on CFS runtime root with a serial `h5py.File` write/reopen round-trip. Disabled locking still requires exclusive writer ownership; current science path remains one serial writer per artifact (no `driver='mpio'`).

---

## 4. Configuration / path cleanup

- Added portable `scripts/load_config.sh` (Bash 3.2-compatible; no namerefs/associative arrays).
- Removed `scripts/nersc/load_environment.sh`.
- Removed `LIMBERCLOUD_CONDA_ENV`, `LIMBERCLOUD_ENV_FILE`, `LIMBERCLOUD_REPO_ROOT`, and legacy CosmoENV/OneCovariance aliases (obsolete keys in `.env` are errors).
- Kept `LIMBERCLOUD_RUNTIME_ROOT`, conditional `LIMBERCLOUD_ONECOVARIANCE_ROOT`, optional `LIMBERCLOUD_TEXLIVE_BIN`.
- `PROJECT_ROOT` discovered via `SLURM_SUBMIT_DIR` markers or directory walk (`scripts/load_config.sh` + `src/limbercloud`); nested manuscript Git fixtures covered by tests.
- Batch jobs: `load_config.sh` → module profile → `scripts/nersc/activate_venv.sh`.
- Jupyter: `scripts/jupyter/launch_kernel.sh` / `register_kernel.sh` (kernelspec name `limbercloud`, display **LimberCloud**).

---

## 5. Sample controls and path cleanup

- Shared helpers: `src/limbercloud/experiments/sample_controls.py`.
- All 24 spectra runners: `--sample-count` (non-fiducial rows; default **0**), `--fiducial-only` (implies 0; rejects nonzero sample-count).
- Campaign must pass `--sample-count=1000` explicitly (not launched).
- `--number` remains host CPU allocation / filename label.
- Removed unused `path` / `--path` from 24 spectra runners + 2 benchmark readers and all shell callers.

---

## 6. Validation evidence

### Software / contract (login-safe)

- `python -m unittest discover -s tests` → **34 OK**
- `bash -n` on experiment/script shells → OK
- `ruff check` on spectra + new experiment helpers → OK
- sdist build excludes `manuscript/` and `revisions/` → OK
- Kernel `--probe` after `.venv` switch → OK
- `scripts/nersc/diagnose_environment.sh` → OK

### Allocated smokes (not science campaigns)

| Test | Result |
| --- | --- |
| `srun -n 2` mpi4py | Cray MPICH 9.1.0.794; ranks 0/1; allreduce sum=1 (job `58590099`) |
| CFS h5py round-trip | OK under `HDF5_USE_FILE_LOCKING=FALSE`; `h5py.get_config().mpi == True` |
| CCL + CAMB tiny distance | OK |
| Numba JIT + JAX CPU (`JAX_PLATFORMS=cpu`) | OK (job `58590111`) |
| JAX GPU device + sum | `CudaDevice(id=0)`, backend `gpu` (job `58590157`) |

Two-rank MPI test validates the dependency only; it does not change the one-task science benchmark.

---

## 7. Documentation and style

- Updated `README.md`, `documents/{environment,nersc,manuscript-workflow}.md` for `.venv`, `load_config.sh`, sample controls, NERSC `git pull --ff-only --no-recurse-submodules`, and direct paper-repository workflow (no parent subtree publishing).
- `.vscode/settings.json`: indentation guides + verified scientific `cSpell.userWords` (authorized settings scope).
- Docstrings on touched runners moved toward Google Args/Returns layout; JAX import-time device setup order preserved (`jax` after logging setup / backend import pattern retained with `# noqa: E402` where required).

---

## 8. Pending checks (either platform)

| Item | Owner / note |
| --- | --- |
| macOS portable env create + `load_config.sh` under `/bin/bash` 3.2 | **Local review** — recipe and loader written; not executed on a Mac in this stage |
| macOS editor settings / spellcheck dictionary | **Local review** — tracked `.vscode` updated; do not assign simultaneous source edits to local owner for the same files |
| Cross-node MPI (2+ nodes) | Not claimed; only single-node two-rank smoke |
| Parallel h5py `driver='mpio'` collective write | Not run; science remains single-writer |
| Login-hosted JupyterLab full notebook session beyond `--probe` | Kernel registered; interactive lab session not exercised here |
| Full `make check` notebook JSON validator | Shell/unit/ruff subset run; run full `make check` if notebook validator dependencies differ |
| Broad docstring/Ruff pass on untouched historical modules | Bounded to this stage’s touched surfaces |

---

## 9. Commit and paper pin

Recorded at commit time in the section below (filled after `git commit`).

- **Code commit:** _(see git log after stage commit)_
- **Paper pin:** `90d12f4f3e574a67c25944d27d7ded553e09402b` (`git ls-tree HEAD manuscript`)
- **Manuscript:** uninitialized/absent on NERSC; `.gitmodules` intact; no `git -C manuscript`
