# C00 — Remote NERSC handoff inventory

**Inventory date (UTC):** 2026-09-19  
**Host:** `login20` (Perlmutter login)  
**User:** `yhzhang`  
**Authorization:** Prompt 0 / C00 only — inventory and this report. No environment replacement, no package installs, no SLURM submissions, no scientific production drivers, no manuscript initialization, no source implementation.

**Plan baseline references:** `revisions/2026-09/CODE_REVISION_PLAN.md` (C00/C13/C15/C16), `MANUSCRIPT_REVISION_PLAN.md`, `COAUTHOR_COMMENT_INVENTORY.md`, `supporting/limber_environment_followup.md`, `CURSOR_IMPLEMENTATION_PROMPTS.md`, `documents/{nersc,environment,runtime-tree,manuscript-workflow}.md`.

---

## 1. Executive summary / authorization boundary

This checkout is ready for a **Prompt 0A environment/maintenance stage**, not for scientific smoke jobs.

| Item | Actual value |
| --- | --- |
| Code commit | `c0cf5fae8d9d48b51710f942f2cc7ad24779e4fc` (`main`, matches `origin/main`) |
| Working tree | Dirty only by **absent** `manuscript/` working directory (`D manuscript`); tracked gitlink and `.gitmodules` intact |
| Recorded paper pin | `90d12f4f3e574a67c25944d27d7ded553e09402b` via `git ls-tree HEAD manuscript` |
| PSCRATCH checkout | `/pscratch/sd/y/yhzhang/LimberCloud` |
| CFS runtime root | `/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/LimberCloud` |
| Working Python | CosmoConda → `.venv` symlink; Python 3.12.12 |
| OneCovariance | `/global/homes/y/yhzhang/opt/OneCovariance` @ `311c2cfbe9d584d9d29abd3d8edaeea673470b7f` |

**Critical readiness constraints (do not bypass):**

1. Existing spectra runners hard-code `count2 = 1000` sampled evaluations; `--number` is a **CPU-allocation filename/thread-budget label**, not a sample limit. There is **no** `--sample-count` / `--fiducial-only` yet.
2. CosmoConda’s `mpi4py` links **conda-forge MPICH 4.3.2**, while job module profiles load **Cray MPICH 9.1.0**. MPI runtime compatibility is **unverified** (not tested on this login host; NERSC docs require allocated `srun` checks).
3. CosmoConda’s `h5py` is the **`nompi`** conda-forge build against conda `hdf5 1.14.6`; job profiles also load `cray-hdf5-parallel/1.14.3.9`. Parallel HDF5/`h5py.get_config().mpi` capability is **unverified** and package metadata indicates a serial build.
4. Preserve CosmoConda; create/validate a separate `limbercloud` environment only in Prompt 0A.

---

## 2. Repo state + paper pin (gitlink only)

### Git identity

```text
commit:  c0cf5fae8d9d48b51710f942f2cc7ad24779e4fc
branch:  main...origin/main (up to date)
remote:  origin  git@github.com:CosmoCloudZhang/LimberCloud.git
tip msg: Finalize local manuscript and NERSC implementation plans
```

Recent history: `c0cf5fa` ← `7652a9f` ← `b16dcdc` (paper-submodule conversion).

Historical audit commit cited in plans (`0876bf4…`) is **not** this HEAD; reconcile scientific work against `c0cf5fa` going forward.

### Dirty / untracked

- **Dirty:** `manuscript` path recorded as deleted in the working tree (`git status`: `D manuscript`). Expected for an intentionally deinitialized NERSC paper checkout.
- **No other modified/untracked files** observed at inventory time (before writing this report).
- **Do not** `git add`/`restore` manuscript as part of inventory; keep the tracked gitlink.

### Paper pin (parent only; manuscript not initialized)

```text
$ git ls-tree HEAD manuscript
160000 commit 90d12f4f3e574a67c25944d27d7ded553e09402b	manuscript

$ cat .gitmodules
[submodule "manuscript"]
	path = manuscript
	url = ../LimberCloudPaper.git
```

`git submodule status manuscript` shows a leading `-` (uninitialized). Directory `/pscratch/sd/y/yhzhang/LimberCloud/manuscript` is **absent**. No `git -C manuscript` was run.

### Local instructions present on NERSC

| Path | Role |
| --- | --- |
| `README.md` | Install, `.env`, kernels, runtime tree pointers |
| `documents/nersc.md` | Perlmutter workflow, Prompt 0 entry, module/job notes |
| `documents/environment.md` | CosmoConda reuse vs create; `.venv`; kernels |
| `documents/runtime-tree.md` | External `data/config/results/plots/logs` layout |
| `documents/manuscript-workflow.md` | Local paper vs NERSC absent paper; CFS figure handoff |
| `revisions/2026-09/*` | Plans, prompts, environment follow-up |
| No `AGENTS.md` | Not present in this checkout |

---

## 3. Path map (PSCRATCH, CFS, OneCovariance)

| Role | Absolute path |
| --- | --- |
| Code checkout (PSCRATCH) | `/pscratch/sd/y/yhzhang/LimberCloud` |
| Checkout-local logs (SLURM `-o logs/…`) | `/pscratch/sd/y/yhzhang/LimberCloud/logs` (empty dir present) |
| Runtime / durable data+results (CFS) | `/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/LimberCloud` |
| Config | `…/LimberCloud/config/` |
| Survey inputs | `…/LimberCloud/data/{Y1,Y10}/` |
| Spectra/covariance/validation results | `…/LimberCloud/results/{spectra,covariance,validation}/` |
| Plots | `…/LimberCloud/plots/{kernel,spectra,error,benchmark,power}/` |
| Runtime `logs/` under CFS | **Missing** (canonical tree docs expect it; jobs currently log under checkout `logs/`) |
| Legacy/extra CFS trees | `…/LimberCloud/{JAX,PYTHON}/` (outside documented canonical layout; treat as historical) |
| OneCovariance checkout | `/global/homes/y/yhzhang/opt/OneCovariance` |
| Shared TeX Live `pdflatex` | `/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/texlive/2026/bin/x86_64-linux` |
| Suggested CFS publication-product handoff | `/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/LimberCloud/handoff/` *(does not exist yet; create in a later accepted stage)* |

`.env` (ignored; path values only, no secrets observed):

```dotenv
LIMBERCLOUD_CONDA_ENV=CosmoConda
LIMBERCLOUD_ONECOVARIANCE_ROOT=/global/homes/y/yhzhang/opt/OneCovariance
LIMBERCLOUD_RUNTIME_ROOT=/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/LimberCloud
LIMBERCLOUD_TEXLIVE_BIN=/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/texlive/2026/bin/x86_64-linux
```

No exported `LIMBERCLOUD_*` in the inventory login shell; launchers load `.env` via `scripts/nersc/load_environment.sh`.

### OneCovariance version

| Field | Value |
| --- | --- |
| Path | `/global/homes/y/yhzhang/opt/OneCovariance` |
| Git HEAD | `311c2cfbe9d584d9d29abd3d8edaeea673470b7f` |
| Branch | `main...origin/main` |
| Remote | `git@github.com:rreischke/OneCovariance.git` |
| Tip subject | `Ich steh' auf diese ganze Scheiße: fix indexing issue SMF NG term` |
| Entry point | `covariance.py` present |
| Working tree | Clean relative to that commit at inventory time |

---

## 4. Config / provenance snapshots

### Configuration JSON (CFS)

| File | SHA-256 | Schema summary |
| --- | --- | --- |
| `cosmology.json` | `6df3fc194d1d18120994ed3be820a3b57eed4497af29422f7c2cd4a1c2ca2ba6` | Scalar ΛCDM-like keys (`H=0.6736`, `W0=-1`, `WA=0`, `OMEGA_K=0`, `NS`, `AS`, densities, …) |
| `survey.json` | `308439d29d2befb83d3caf1898d52a9a0cc580f21f79fabb2ad6e11396dc2b2c` | Y1/Y10 `AREA=18000`, shared `FRACTION` |
| `number_density.json` | `bde5b96fafb5fbac236e6c3fdc201bdd63b28b64bd2d615cf71a070505087bc9` | Y1: 5 lens + 5 source; Y10: 10 lens + 5 source |
| `magnification_bias.json` | `1b94ccca8e23cd9339427ac3e7eece735366696e47c0af65213d43fbca745f0f` | Y1 len=5; Y10 len=10 (values are stored slopes/response — scientific naming still open per C01) |
| `galaxy_bias.json` | `4ac5d2780b3d8aa0256489d8409f8afaceef9135618ef45819a9db4f7b1fb114` | Y1/Y10 length-351 grids |
| `intrinsic_alignment.json` | `40d78073bb62d5eca5583ea0e00996253bdc4dd3e3e07c49a86c90e44c2bd93e` | Key `A` length-351 only — **no eta/pivot metadata stored** |

Generator note (code, not config metadata): `scripts/generate_config/intrinsic_alignment.py` uses `eta_pivot = 0.5`, conflicting with draft manuscript `eta_IA=0.0` (open C01 decision).

### Survey n(z) arrays (all <2 MB; hashed)

| Relative path | SHA-256 | Structure |
| --- | --- | --- |
| `data/Y1/lsst_lens_bins.npy` | `d1569ac6994a849a3c347426166f6a2e04d51928e780400eec7e85f7c82b6da8` | dict: `redshift_range`(500,), `bins` dict len=5 |
| `data/Y1/lsst_source_bins.npy` | `042bc3487911a521ebd5f4d91267feb2cea8089f9aaa5ad6827de069da70d842` | `bins` len=5 |
| `data/Y1/lsst_lens_sample_nz.npy` | `7bd851ab479189c853bf311dfe0f4e14796bb7b40295a572db6f64f7447a5cf8` | `redshift`/`distribution` (500,) float64 |
| `data/Y1/lsst_source_sample_nz.npy` | `b65b5212c0d5a2ec175c1123db90bab156748078927b841c7e5340e7cb9daf6b` | same |
| `data/Y10/lsst_lens_bins.npy` | `af784b43907e3586a52a6d964e1978502de0e25b22115e3330a171587f612bba` | `bins` len=10 |
| `data/Y10/lsst_source_bins.npy` | `39507626c862e617e6688600752eb8184f548f279b3868e654855b72b4776480` | `bins` len=5 |
| `data/Y10/lsst_lens_sample_nz.npy` | `6cbf8b055175eb45ec77471654be67096ac31356c91abe82b2275edabc6ee61e` | (500,) |
| `data/Y10/lsst_source_sample_nz.npy` | `dacede8d0191e496cdc9cc43921d6d2d5032a5e5d57304f41209eaca9055856a` | (500,) |

### Existing covariance products (identity only; large files not dumped)

| Product | Notes |
| --- | --- |
| `results/covariance/Y1/CONFIG.ini` | SHA-256 `64149bb8c81da0b96d71481bffd7f10a1161298e9c97946869961053ef94410e`; `gauss/nongauss/ssc=True`; `ell_bins_*=20` for clustering/lensing outputs; `num_cores=256`; areas 18000 deg² |
| `results/covariance/Y10/CONFIG.ini` | SHA-256 `ba9d117f272068abce589223c82339cab727d82852518229aad7a14f62fdd021`; same term flags; 10-lens n_eff |
| `MATRIX.ascii` Y1/Y10 | ~1112 / ~2412 lines including headers → consistent with ~1100 / ~2400 data-vector sizes quoted in the plan |
| `Cell_*.ascii` | Header `ell tomo_i tomo_j Cell_*`; Y1 `Cell_gg` has 2525 data rows ≈ 101×25 (raw multipole table, not 20-band output) |
| Companion files | `LIST.ascii`, `MATRIX_{gauss,nongauss,SSC}.ascii`, `ALIGNMENT/MAGNIFICATION/GALAXY/LENS/SOURCE.ascii`, `VALUE.ini`, `GAUSSIAN.ini` (Y1) |

These are **pre-revision** products; C05/C06 ordering defects remain plan findings until regenerated under corrected serialization.

### Spectra timing artifacts

Under `results/spectra/{CCL,NUMBA,JAX/{CPU,GPU}}/{Y1,Y10}/`: timing TXT files named `Time_{Single,Double,Triple}_128*.txt` (allocation token **128**). Count of spectra result files ≈ 78 TXT artifacts; no HDF5 spectra archive yet (planned).

---

## 5. Environment inventory (Conda / `.venv` / kernels / modules / scheduler)

### Interpreter selection

| Layer | Actual |
| --- | --- |
| `.venv` | Symlink → `/global/homes/y/yhzhang/.conda/envs/CosmoConda` (created 2026-08-23) |
| Python | 3.12.12 (`/.venv/bin/python` → `python3.12`) |
| Editable install | `limbercloud 0.1.0` editable at `/pscratch/sd/y/yhzhang/LimberCloud` |
| `environment.yml` name | Still `CosmoConda`; lists scientific baseline **without** mpi4py/h5py/CAMB explicit pins (CAMB/mpi4py/h5py are present in the live env anyway) |

### Jupyter kernels (user registry)

| Kernelspec | argv / role |
| --- | --- |
| `limbercloud-cosmoconda` display **LimberCloud** | `/pscratch/.../scripts/jupyter/launch_kernel.sh` → loads `.env` then `.venv/bin/python -m ipykernel_launcher` |
| `cosmoconda` | Direct `/global/homes/.../CosmoConda/bin/python` (**does not** load checkout `.env`) |
| `baseconda`, `railconda` | Unrelated collaboration kernels |

Docs prefer selecting **LimberCloud** after `scripts/jupyter/register_kernel.sh`.

### Key Python packages (metadata / `pip show`; no MPI/h5py capability imports)

| Package | Version | Install / build notes |
| --- | --- | --- |
| numpy | 2.2.6 | conda-forge `py312h72c5963_0` |
| scipy | 1.13.1 | pip/site-packages |
| numba | 0.63.1 | |
| astropy | 7.2.0 | |
| matplotlib | 3.10.8 | |
| pyccl | 3.2.1 | conda-forge `py312h29fb263_1` |
| camb | 1.6.5 | conda-forge `py312hbd90422_0` (present; not listed in `environment.yml`) |
| jax / jaxlib | 0.9.1 | + `jax-cuda12-pjrt/plugin` 0.9.1 and NVIDIA CUDA **12** wheels |
| mpi4py | 4.1.1 | conda-forge `py312hd0af0b3_102`; depends `mpich >=3.4` |
| mpich (conda) | 4.3.2 | `h23078de_105` — **not** Cray MPICH |
| h5py | 3.15.1 | conda-forge **`nompi_py312ha4f8f14_101`** |
| hdf5 (conda) | 1.14.6 | **`nompi_h1b119a7_105`** |
| ipykernel | 7.1.0 | |
| ruff | 0.16.3 | |
| cosmosis | 3.25 | Present in CosmoConda; **exclude from dedicated limbercloud env** per plan |

### Activation / module profiles (job scripts)

`scripts/nersc/modules/cpu.sh` → `module load cpu` + `common.sh`  
`scripts/nersc/modules/gpu.sh` → `module load gpu` + `common.sh`  

`common.sh` loads: `conda`, `cray-mpich`, `PrgEnv-gnu`, `cray-hdf5-parallel`.

**After sourcing `cpu.sh` in a subshell (inventory), loaded stack included:**

- `cpu/1.0`, `conda/Miniforge3-25.11.0-1`
- `PrgEnv-gnu/8.7.0`, `gcc-native/14`, `craype/2.7.36`
- `cray-mpich/9.1.0`, `cray-hdf5-parallel/1.14.3.9` (`HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.14.3.9/gnu/12.3`)
- Wrapper `cc` → gcc-14.3.0

Login shell at inventory time also had `gpu/1.0` + `cudatoolkit/13.2` loaded before the CPU profile replaced `gpu` with `cpu`.

### Scheduler resources (current launchers)

| Job class | Account | Constraint | CPUs/task | GPUs | Time | Nodes/tasks |
| --- | --- | --- | --- | --- | --- | --- |
| Spectra CCL/NUMBA/JAX-CPU | `m1727` | `cpu` | 128 | — | 04:00:00 | 1 / 1 |
| Spectra JAX-GPU | `m1727` | `gpu` | 128 | 1 (`--gpus-per-node=1`) | 04:00:00 | 1 / 1 |
| Covariance matrix | `m1727` | `cpu` | **256** | — | 04:00:00 | 1 / 1 |
| Benchmarks | `m1727` | (see scripts) | (see scripts) | — | 04:00:00 | 1 / 1 |

All spectra jobs: `srun -n 1 -c $SLURM_CPUS_PER_TASK` (GPU adds `-G 1`). Thread env exports set `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/etc. from `SLURM_CPUS_PER_TASK`. Covariance sets `HDF5_USE_FILE_LOCKING=FALSE` (CFS-relevant).

### Login-host GPU/driver metadata (not a compute validation)

- `nvidia-smi`: Driver 580.178.04, CUDA Version 13.2, A100-PCIE-40GB present on this login node.
- JAX stack ships **CUDA 12** plugins while the site module advertises toolkit **13.2** — treat GPU JAX as **needs allocated verification**, not proven by this inventory.

---

## 6. MPI/HDF5 candidate stack + unverified capabilities

### Official NERSC guidance (rechecked 2026-09-19 via docs.nersc.gov Parallel Python)

Source: [NERSC Parallel Python](https://docs.nersc.gov/development/languages/python/parallel-python/)

Documented recommendations relevant to Prompt 0A:

- Build **mpi4py from source** with Cray wrapper: `MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py` under `PrgEnv-gnu` + `cray-mpich` (+ conda).
- Alternatives: clone `nersc-mpi4py`, or conda `mpich` + `cray-mpich-abi` with compatibility checks.
- Build **parallel h5py** with `module load cray-hdf5-parallel`, then `HDF5_MPI=ON CC=cc pip install … --no-binary=h5py --no-build-isolation --no-deps h5py` after mpi4py.
- **mpi4py does not work on login nodes**; inspect `MPI.Get_library_version()` under `srun` on compute nodes.
- For HDF5 writes outside `$SCRATCH` (includes CFS): `export HDF5_USE_FILE_LOCKING=FALSE`.
- Parallel h5py smoketests use multi-rank `srun` and `driver='mpio'` — separate from LimberCloud’s planned single-writer archive.

### What CosmoConda actually contains (linkage / package metadata only)

| Component | Observation | Runtime claim |
| --- | --- | --- |
| `mpi4py` `.so` | `NEEDED libmpi.so.12` → resolves to **conda** `…/CosmoConda/lib/libmpi.so.12` (MPICH 4.3.2 + libfabric/ucx from conda) | **Unverified** under `srun`/Cray MPICH |
| Job modules | Load **Cray MPICH 9.1.0** | Distinct from conda MPICH — mismatch risk |
| `h5py` build string | `nompi_…` | Package identity is serial; **`h5py.get_config().mpi` not queried** (import deferred) |
| `h5py` linkage | `defs*.so` → conda `libhdf5.so.310` / `libhdf5_hl.so.310` (1.14.6 nompi) | Does **not** link Cray parallel HDF5 1.14.3.9 |
| Module HDF5 | `cray-hdf5-parallel/1.14.3.9` present when CPU profile sourced | Independent of Python extension linkage |

**Explicitly not done here (per Prompt 0):** no `import mpi4py.MPI`, no `MPI.Get_library_version()`, no `import h5py` on the login host.

**Candidate stack for a new `limbercloud` env (Prompt 0A):** GNU/`cc` + `cray-mpich/9.1.0` + `cray-hdf5-parallel/1.14.3.9` + source-built mpi4py (± optional MPI-enabled h5py). Do **not** install a generic MPI stack over NERSC’s configured stack. Keep CosmoConda untouched until the candidate passes allocated checks.

---

## 7. Env-var / root-discovery / alias caller audit

### Intended model (from plans; not yet implemented)

| Concept | Intent |
| --- | --- |
| `PROJECT_ROOT` | Auto-discovered code checkout |
| `.venv` | Sole selected Python prefix |
| `RUNTIME_ROOT` / `LIMBERCLOUD_RUNTIME_ROOT` | External CFS data/results |
| Remove after migration | `LIMBERCLOUD_CONDA_ENV`, `LIMBERCLOUD_ENV_FILE`, `LIMBERCLOUD_REPO_ROOT`, legacy `CosmoENV` / `ONECOVARIANCE_SCRIPT` / `ONE_COVARIANCE_ROOT` |

### Actual callers of migration-sensitive variables

| Variable | External/runtime callers | Notes |
| --- | --- | --- |
| `LIMBERCLOUD_CONDA_ENV` | **All** experiment/config shell launchers (`conda activate "${LIMBERCLOUD_CONDA_ENV}"`, ~34 scripts) + loader + docs/tests | Set in `.env` to `CosmoConda`; **must migrate with all callers** |
| `LIMBERCLOUD_ENV_FILE` | `load_environment.sh` + unit tests (`test_environment_contracts.py`, `test_launcher_smoke.py`); **not** set in this checkout’s `.env` | Optional dotenv override only |
| `LIMBERCLOUD_REPO_ROOT` | Every launcher’s `REPO_ROOT=…` discovery; `Run_All.sh` **exports** it; tests assert export | Optional override; heavily used as discovery glue |
| `LIMBERCLOUD_RUNTIME_ROOT` | Required; set in `.env`; all jobs | **Keep** |
| `LIMBERCLOUD_ONECOVARIANCE_ROOT` | Covariance `matrix.sh` + loader validation | **Keep** for covariance |
| `LIMBERCLOUD_TEXLIVE_BIN` | Optional; set in `.env` | Keep if plotting needs it |
| Legacy aliases | Accepted by loader; **not** present in this `.env` | Safe to remove after caller/test updates |

### Root discovery behavior

Pattern used by launchers:

```bash
REPO_ROOT="${LIMBERCLOUD_REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
```

Implications:

- **Copied SLURM scripts:** `SLURM_SUBMIT_DIR` preferred — good when submit cwd is the LimberCloud checkout.
- **Nested paper repo (local):** `git rev-parse --show-toplevel` from inside an initialized `manuscript/` would resolve to **LimberCloudPaper**, not LimberCloud — a known C13/C15 risk. `load_environment.sh` itself derives the code root from its script path (`scripts/nersc/../..`), which is safer, but launchers still gate on git toplevel first.
- **NERSC today:** `manuscript/` absent, so discovery from script paths / submit dir correctly finds `/pscratch/sd/y/yhzhang/LimberCloud`. No need to initialize the paper to test discovery; use synthetic nested-git fixtures later.

Kernel launcher (`scripts/jupyter/launch_kernel.sh`) uses path-relative `REPOSITORY_ROOT` + `.venv/bin/python` (aligned with the intended `.venv` model), while batch jobs still `conda activate` by name — **selector disagreement** called out in the environment follow-up.

---

## 8. Runner `--number` vs sample controls

### Confirmed semantics

In spectra Python entry points (example `experiments/spectra/NUMBA/Y1/triple.py`):

- Hard-coded campaign counters: `count1=100`, `count2=1000`, ten cumulative checkpoints via `count_list`.
- CLI `--number`: documented as “number of cores for parallel computation”; used in **output filenames** (`Time_{label}_{number}.txt`, …).
- Shell passes `--number="${SLURM_CPUS_PER_TASK}"` (128 for spectra).
- Thread pools are set from **shell exports** of `SLURM_CPUS_PER_TASK`, not by interpreting `--number` as a sample limit inside the Python loop inspected here.

Therefore: **`--number` = host CPU allocation label** (and related thread budget via the shell), **not** a sample-count control. **`Single`/`Double`/`Triple`** select probe configurations (EE / TE+TT / EE+TE+TT), not tiny runs.

### Where to add controls (before any computational smoke)

Add and test in the shared experiment CLI / runners (Prompt 0A minimum before any science job; Prompt 1 for full campaign contract):

- `--sample-count` — number of **non-fiducial** sampled rows  
- `--fiducial-only` — implies zero sampled rows; reject inconsistent combinations  
- Wire through all 24 spectra entry points (and future NUMERIC) **or** a shared evaluator wrapper they call  
- Keep `--number` as allocation/filename label  

Until those exist, **do not** submit existing `*.sh` spectra jobs as “smoke tests.”

---

## 9. Mismatches + lightweight checks + allocation tests

### Mismatches / risks

1. **Conda MPICH mpi4py vs Cray MPICH modules** — primary environment defect for future MPI use.  
2. **Serial conda h5py/hdf5 vs loaded cray-hdf5-parallel** — Python does not use the module HDF5; parallel h5py not installed.  
3. **`environment.yml` vs live CosmoConda** — YAML omits mpi4py/h5py/CAMB; live env is a larger collaboration stack (includes CosmoSIS).  
4. **Kernel vs batch interpreter selection** — `.venv`/kernel launcher vs `LIMBERCLOUD_CONDA_ENV` activation.  
5. **JAX CUDA 12 wheels vs site CUDA 13.2 module** — needs GPU-node verification.  
6. **IA eta 0.5 generator vs manuscript 0.0**; magnification slope vs `q=5s−2` naming — scientific, not env.  
7. **Covariance CONFIG `ell_bins=100` vs output `ell_bins_*=20`** and Cell tables with ~101 raw ell samples — estimator/contract clarity for C03/C05.  
8. **CFS runtime `logs/` missing**; SLURM logs go to PSCRATCH checkout `logs/`.  
9. **Working tree dirty** solely due to absent manuscript path — expected; do not “fix” by initializing the paper.

### Lightweight checks allowed without MPI init on login

Safe now / in Prompt 0A pre-allocation:

- `git` status, `ls-tree` paper pin, path existence, `.env` key presence  
- `readlink .venv`, `python -V`, `pip show` / conda-meta / `ldd` on extensions  
- Syntax/contract tests that **do not** import `mpi4py.MPI` or MPI-enabled h5py (existing `tests/test_environment_contracts.py`, `test_experiment_contracts.py`, launcher smoke with mocked dotenv)  
- Kernel `--probe` path only if it stays free of MPI-initializing imports  
- Config JSON schema/hash checks; small npy header inspection  

### Allocation-based tests required next (Prompt 0A / inventory follow-up)

On an interactive or batch **CPU** allocation (example pattern from NERSC docs; exact script to be written in 0A):

1. Activate candidate (or carefully probe CosmoConda) under `scripts/nersc/modules/cpu.sh`.  
2. `srun -n 2 … python -c 'from mpi4py import MPI; print(MPI.Get_library_version()); …'` — record library string; expect Cray MPICH only after a proper rebuild.  
3. Import h5py **after** setting `HDF5_USE_FILE_LOCKING=FALSE` on a CFS path; record `h5py.version.info` and `h5py.get_config().mpi` without claiming parallel support unless true.  
4. Serial HDF5 round-trip on CFS under the runtime root.  
5. Tiny CCL+CAMB, Numba, and JAX CPU device checks with explicit thread settings.  
6. Separate **GPU** allocation: JAX device visibility/synchronization under `gpu.sh`.  
7. Optional: two-node MPI only if claiming cross-node support.  
8. Notebook kernel start on the actual Jupyter/compute host with LimberCloud kernelspec.

Do **not** run 1,000-iteration legacy drivers for these checks.

---

## 10. Bounded next-stage proposal + unresolved inputs

### Propose: Prompt 0A only (C13 / C15 / C16 maintenance)

**Owner:** Remote Cursor on this NERSC checkout  
**Inputs frozen for the stage:**

| Input | Value |
| --- | --- |
| Code commit at inventory | `c0cf5fae8d9d48b51710f942f2cc7ad24779e4fc` |
| Paper pin (parent gitlink) | `90d12f4f3e574a67c25944d27d7ded553e09402b` |
| Checkout | `/pscratch/sd/y/yhzhang/LimberCloud` |
| Runtime | `/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/LimberCloud` |
| OneCovariance | `/global/homes/y/yhzhang/opt/OneCovariance` @ `311c2cf` |
| Preserve | CosmoConda + current `.venv` target until candidate accepted |
| Manuscript | Remain uninitialized/absent; no `git -C manuscript` |

**In-scope work (summary):**

1. Create separate `limbercloud` env (CPU recipe + NERSC CUDA variant); NERSC source builds for mpi4py (± parallel h5py per NERSC recipe); exclude CosmoSIS; include CAMB, h5py, mpi4py.  
2. Unify interpreter selection on `.venv`; rename `load_environment.sh` → `load_config.sh`; migrate/remove `LIMBERCLOUD_CONDA_ENV` / `LIMBERCLOUD_ENV_FILE` / `LIMBERCLOUD_REPO_ROOT` with **all** shell/test/doc callers.  
3. Root-discovery hardening against nested paper git (fixtures; no NERSC paper init).  
4. C15 path cleanup (unused `--path`), naming toward `PROJECT_ROOT`, style/docstring/Ruff as specified.  
5. Implement and unit-test `--sample-count` / `--fiducial-only` **before** any science `sbatch`.  
6. Record allocated MPI/HDF5/CPU/GPU smoke evidence in a new report under `revisions/2026-09/reports/`.  
7. C16 docs already largely present; refresh only as helpers change; keep package excludes for paper/revision assets.

**Out of scope for 0A:** production 1,001-row campaign, covariance regeneration, environment replacement of CosmoConda, manuscript edits, initializing `manuscript/`.

### Unresolved inputs (block dependent science, not 0A start)

| ID | Question | Needed for |
| --- | --- | --- |
| S1 | IA law: eta 0.0 (draft) vs 0.5 (generator); store parameters in JSON | C01 spectra/covariance regeneration |
| S2 | Magnification field = slope `s` vs response `q` | C01/C02 |
| S3 | Sampling range ±5% vs ±10%; fixed `wa`/curvature policy | C08 shared table |
| S4 | Whether NG+SSC in existing CONFIG match intended paper claims | C07 / manuscript labels |
| S5 | Accept serial-only h5py for campaign vs invest in MPI h5py (still single-writer) | C13 acceptance criteria |
| S6 | Account/QOS/reservation for interactive MPI smokes (`m1727` assumed from launchers) | Prompt 0A allocation tests |
| S7 | Exact CFS handoff directory name/permissions under LimberCloud | Later publication export |

### Immediate commands for the next agent (read-only confirmation)

```bash
cd /pscratch/sd/y/yhzhang/LimberCloud
git rev-parse HEAD
git ls-tree HEAD manuscript
git status --short
readlink -f .venv
# Then implement Prompt 0A; do not sbatch spectra runners yet.
```

---

## Appendix — Inventory method limits

- No SLURM jobs submitted; no packages installed; CosmoConda not modified.  
- No manuscript submodule init; `.gitmodules` and gitlink untouched by this stage (working tree still shows absent `manuscript/` path).  
- MPI/HDF5 **capabilities** marked unverified wherever they would require login-host MPI initialization or speculative ABI claims.  
- Large matrices/lists were not copied into Git; only hashes, headers, shapes, and key CONFIG fields were recorded.
