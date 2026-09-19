# Perlmutter workflow

## Development ownership and Git updates

Implement code, scripts, notebooks, environment helpers, and tests on NERSC.
Edit and compile the manuscript locally. The remote `manuscript/` submodule
stays uninitialized, empty, or absent; retain its Git reference and
`.gitmodules` entry. Routine remote updates use:

```bash
git pull --ff-only --no-recurse-submodules
git ls-tree HEAD manuscript
```

Read the recorded paper commit with `git ls-tree`; do not initialize the paper
or inspect it with `git -C manuscript` when its directory is empty. Export
validated figures and tables to a CFS bundle with checksums and provenance for
local integration, rather than writing into a remote manuscript directory.
See [manuscript-workflow.md](manuscript-workflow.md) for the complete handoff.

Follow Prompts 0A–3 in the
[revision prompts](../revisions/2026-09/CURSOR_IMPLEMENTATION_PROMPTS.md) on
NERSC after the inventory report. Prompt 4 is for local manuscript work.

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

```bash
mkdir -p logs
sbatch --chdir="${PWD}" experiments/spectra/NUMBA/Y1/single.sh
```

Spectra runners default to `--sample-count=0`. Do not submit the 1,001-row
campaign unless you pass `--sample-count=1000` explicitly (plus fiducial when
that path exists). `--number` is the CPU allocation label.

## Configuration generation order

1. `cosmology.sh`
2. `survey.sh`
3. `number_density.sh`
4. `magnification_bias.sh`
5. `galaxy_bias.sh`
6. `intrinsic_alignment.sh`

## Validation sequence

1. Lightweight path, configuration, syntax, and contract checks (`make check`).
2. Dedicated `limbercloud` imports and kernel `--probe` without initializing MPI
   on login nodes.
3. Tiny allocated CPU checks: CCL+CAMB, Numba, JAX CPU, serial HDF5 round-trip
   under CFS with `HDF5_USE_FILE_LOCKING=FALSE`, and a two-rank `mpi4py`
   compatibility `srun` (dependency only; not a science benchmark).
4. Separate GPU allocation for JAX device visibility.
5. Bounded science pilots only after sample controls and scientific stages.

All jobs use the canonical runtime tree in [runtime-tree.md](runtime-tree.md).
