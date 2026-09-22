# Perlmutter workflow

## Development ownership and Git updates

Implement code, scripts, notebooks, environment helpers, and tests on NERSC.
Edit and compile the manuscript locally. The remote `manuscript/` submodule
stays uninitialized, empty, or absent; retain its Git reference and
`.gitmodules` entry. Set these defaults once in the NERSC checkout:

```bash
git config --local submodule.recurse false
git config --local fetch.recurseSubmodules false
```

These settings affect only that checkout. The tracked `.gitmodules` also
disables automatic manuscript fetching; explicit recursive commands can
override defaults. Routine remote updates use:

```bash
git pull --ff-only --no-recurse-submodules
git ls-tree HEAD manuscript
```

Read the recorded paper commit with `git ls-tree`; do not initialize the paper
or inspect it with `git -C manuscript` when its directory is empty. Export
validated figures and tables to a CFS bundle with checksums and provenance for
local integration, rather than writing into a remote manuscript directory.
See [manuscript-workflow.md](manuscript-workflow.md) for the complete handoff.

Follow the revised remote Phases 1–5A in the
[revision prompts](../revisions/2026-09/CURSOR_IMPLEMENTATION_PROMPTS.md) on
NERSC after reviewing the current implementation audit. Phase 5B is for local manuscript work; the completed environment is retained with targeted repairs.

## Checkout and environment

Preserve CosmoConda until the dedicated `limbercloud` environment and notebook
launches pass. Create the NERSC variant with:

```bash
module load conda
scripts/nersc/create_environment.sh --name limbercloud --nersc
conda activate limbercloud
source scripts/nersc/modules/cpu.sh
scripts/nersc/install_mpi_h5py.sh
scripts/nersc/diagnose_environment.sh
```

Point `.venv` at the accepted prefix only after inspection. See
[environment.md](environment.md).

## Per-checkout configuration

```bash
cp .env.example .env
```

```dotenv
LIMBERCLOUD_RUNTIME_ROOT=/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/LimberCloud
LIMBERCLOUD_ONECOVARIANCE_ROOT=/global/homes/y/yhzhang/opt/OneCovariance
LIMBERCLOUD_TEXLIVE_BIN=/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/texlive/2026/bin/x86_64-linux
```

`scripts/load_config.sh` reads the fixed `${PROJECT_ROOT}/.env` without
executing it. Exported canonical variables take precedence. Python selection
uses `.venv` via `scripts/nersc/activate_venv.sh`. Do not set
`LIMBERCLOUD_CONDA_ENV`, `LIMBERCLOUD_ENV_FILE`, or `LIMBERCLOUD_REPO_ROOT`.

## Modules and job submission

- CPU, configuration, covariance, and plotting jobs load
  `scripts/nersc/modules/cpu.sh`.
- JAX GPU jobs load `scripts/nersc/modules/gpu.sh`.
- Both profiles load Conda, Cray MPI, GNU, and parallel HDF5, and set
  `HDF5_USE_FILE_LOCKING=FALSE` for CFS. Disabled locking is not a multi-writer
  solution; require exclusive ownership per artifact namespace.

Batch scripts resolve `PROJECT_ROOT` from `SLURM_SUBMIT_DIR` or by walking from
the script path, looking for `scripts/load_config.sh` and `src/limbercloud`
(never the nested paper Git root).

From the checkout root, load the configuration into the submitting shell so
the table path expands correctly. These examples assume an existing canonical
table under `results/spectra/inputs/pilot/`; the second example requires
sample IDs 0 through 10. The first launch uses the default fiducial-only
selection:

```bash
source scripts/load_config.sh
mkdir -p logs
sbatch --chdir="${PWD}" experiments/spectra/NUMBA/Y1/single.sh \
    --sample-table "${LIMBERCLOUD_RUNTIME_ROOT}/results/spectra/inputs/pilot"
```

Every launcher forwards its own arguments to the driver, with quoting intact, so
a submitted selection reaches Python:

```bash
sbatch --chdir="${PWD}" experiments/spectra/NUMBA/Y1/single.sh \
    --sample-count=10 \
    --sample-table "${LIMBERCLOUD_RUNTIME_ROOT}/results/spectra/inputs/pilot"
```

`Run_All.sh` passes the same selection into all six child jobs.

Every spectra run requires `--sample-table` and evaluates its fiducial, sample 0.
`--sample-count` adds that many further cosmologies, and the default `0` is
the fiducial alone. Timing files are written in the family/survey directory
using separate `Time_*_Fiducial.txt` and `Time_*_Cosmology.txt` products. A
fiducial-only rerun preserves sampled files; a sampled run replaces both
populations. Sampled files store actual checkpoint counts and cumulative
seconds, so small pilots use the same interface. Older `*_128*` names do not match
the new names, so they are left in place. Do not submit the 1,000-sample
campaign unless you pass `--sample-count=1000` explicitly. CPU count stays in
`#SBATCH --cpus-per-task` and the thread environment; it is not a driver flag
or a filename token.

## Configuration generation order

1. `cosmology.sh`
2. `survey.sh`
3. `number_density.sh`
4. `magnification_bias.sh`
5. `galaxy_bias.sh`
6. `intrinsic_alignment.sh`

The generator wrappers also forward their arguments, so `intrinsic_alignment.sh
--eta-ia 0.0` reaches Python. Omitting `--eta-ia` already adopts the campaign
value 0.0; any other value is written as an explicitly diagnostic array that the
accepted-campaign readers refuse.

## Validation sequence

1. Login-safe path, configuration, syntax, and contract checks (`make check`).
   This runs `make test-fast`, which imports NumPy and SciPy only. `make`
   uses `.venv/bin/python3` when that link exists, so it does not pick up
   `/usr/bin/python3`. Override with `make PYTHON=...` only for a deliberate
   other prefix.
2. Dedicated `limbercloud` imports and kernel `--probe` without initializing MPI
   on login nodes.
3. Allocated checks, because `make test-science` and `make check-all` import
   h5py. With the MPI-linked h5py build in this environment they are not
   login-safe. Run them together with the tiny CCL+CAMB, Numba, JAX CPU and
   serial HDF5 round-trip checks under CFS with
   `HDF5_USE_FILE_LOCKING=FALSE`, plus a two-rank `mpi4py` compatibility `srun`
   (dependency only; not a science benchmark).
4. Allocated kernel check: `scripts/jupyter/launch_kernel.sh --science` verifies
   that the editor's kernel inherits the same modules, Conda hooks and
   interpreter as a batch job, performs an HDF5 round trip and evaluates a CCL
   background quantity. The login-safe `--probe` reports identity only.
5. Separate GPU allocation for JAX device visibility.
6. Bounded science pilots only after sample controls and scientific stages.

All jobs use the canonical runtime tree in [runtime-tree.md](runtime-tree.md).
