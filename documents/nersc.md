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

Start implementation with Prompt 0 in the
[revision prompts](../revisions/2026-09/CURSOR_IMPLEMENTATION_PROMPTS.md), then
follow Prompts 0A–3 on NERSC. Prompt 4 is for local manuscript work. The setup
below describes current helpers; the planned environment and experiment
changes remain pending implementation and validation.

## Checkout and environment

The collaboration environment is named `CosmoConda`. An existing validated
environment may also contain CosmoSIS, parallel HDF5, MPI builds, and other
project software; keep that environment and install only LimberCloud:

```bash
module load conda
conda activate CosmoConda
python -m pip install --no-deps -e .
```

Do not run the environment-creation helper against an existing environment.
`environment.yml` and `scripts/nersc/create_environment.sh` are an opt-in path
for new standalone installations, not an update mechanism. See
[environment.md](environment.md) for the two workflows and `.venv` setup.

## Per-checkout configuration

Copy the public template and edit the ignored file:

```bash
cp .env.example .env
```

A typical configuration is:

```dotenv
LIMBERCLOUD_RUNTIME_ROOT=/path/to/external/LimberCloud
# LIMBERCLOUD_CONDA_ENV=/full/path/to/CosmoConda
# LIMBERCLOUD_ONECOVARIANCE_ROOT=/path/to/OneCovariance
# LIMBERCLOUD_TEXLIVE_BIN=/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/texlive/2026/bin/x86_64-linux
```

`LIMBERCLOUD_CONDA_ENV` defaults to the name `CosmoConda`, so it is needed only
for a custom name or full prefix. `LIMBERCLOUD_ONECOVARIANCE_ROOT` is required
only by the two covariance launchers and must contain `covariance.py`.
`LIMBERCLOUD_TEXLIVE_BIN` is optional and may point to the directory containing
`pdflatex` when it is not already on `PATH` (for example the shared CosmoCloud
TeX Live install on CFS).

The launchers parse `.env` without executing it. An already exported canonical
variable takes precedence, and `LIMBERCLOUD_ENV_FILE` may select another dotenv
file. `LIMBERCLOUD_REPO_ROOT` remains an optional advanced override; scripts
otherwise derive the checkout root from Git.

The old `CosmoENV`, `ONECOVARIANCE_SCRIPT`, and `ONE_COVARIANCE_ROOT` names are
temporary migration aliases. Do not add them to new configuration.

## Modules and job submission

Every batch script uses a centralized module profile:

- CPU, configuration, covariance, and plotting jobs load
  `scripts/nersc/modules/cpu.sh`.
- JAX GPU jobs load `scripts/nersc/modules/gpu.sh`.
- Both profiles load the shared Conda, Cray MPI, GNU programming environment,
  and parallel-HDF5 modules.

The scripts no longer source `~/.bashrc`. They load `.env`, select the module
profile, and activate `LIMBERCLOUD_CONDA_ENV` directly. GPU jobs still request
GPU nodes and devices through their `#SBATCH` directives; loading `gpu` alone
does not allocate hardware.

Create the checkout-local log directory before direct submissions:

```bash
mkdir -p logs
sbatch --chdir="${PWD}" experiments/spectra/NUMBA/Y1/single.sh
```

The four `Run_All.sh` launchers create `logs/`, validate `.env` before the first
submission, and use the repository as the Slurm working directory.

## Configuration generation order

Run the generators in this order:

1. `cosmology.sh`
2. `survey.sh`
3. `number_density.sh`
4. `magnification_bias.sh`
5. `galaxy_bias.sh`
6. `intrinsic_alignment.sh`

The final two depend on the generated cosmology configuration.

## Validation sequence

Follow C00, C13, and C14 in the
[code plan](../revisions/2026-09/CODE_REVISION_PLAN.md): inventory the established
environment first and build a separate candidate without replacing it.

1. Run lightweight path, configuration, syntax, and contract checks first.
   Keep MPI/HDF5 imports and scientific computations in the appropriate
   allocated context; importing an MPI-enabled library can initialize MPI.
2. Confirm the selected Python imports this checkout. Validate the candidate
   environment's CPU/GPU, MPI, and HDF5 compatibility before adopting it.
3. Implement bounded-run controls before submitting scientific smoke tests.
   Current runners use 1,000 iterations: `Single` means EE and `--number`
   specifies CPU allocation. Neither makes an existing job a tiny run.
4. Run the planned tiny CCL, Numba, JAX CPU/GPU, and NUMERIC checks in allocated
   jobs. Confirm JAX devices and asynchronous timing synchronization. The new
   sample controls and HDF5 outputs are planned interfaces, not existing ones.
5. Validate covariance generation and vector/ell ordering in a bounded case
   before the full matrix. Check storage, resume, and timing contracts as well
   as numerical agreement.
6. Validate notebook structure and saved-data loading. Select the registered
   **LimberCloud** kernel for interactive execution. Publication plotting may
   need TeX on NERSC, but manuscript editing and compilation remain local.
7. Confirm code checks work with `manuscript/` both absent and empty. Complete
   the plan's scientific gates before any 1,001-sample production campaign.

All jobs use the canonical runtime tree documented in
[runtime-tree.md](runtime-tree.md). Verify that tree before submitting jobs.
