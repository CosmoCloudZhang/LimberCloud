# Perlmutter workflow

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
```

`LIMBERCLOUD_CONDA_ENV` defaults to the name `CosmoConda`, so it is needed only
for a custom name or full prefix. `LIMBERCLOUD_ONECOVARIANCE_ROOT` is required
only by the two covariance launchers and must contain `covariance.py`.
`LIMBERCLOUD_TEXLIVE_BIN` is optional and may point to the directory containing
`pdflatex` when it is not already on `PATH`.

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

The four `run_all.sh` launchers create `logs/`, validate `.env` before the first
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

Validation must not recreate or update an established `CosmoConda`.

1. Confirm `.venv/bin/python` imports this checkout and the required scientific
   packages.
2. Run `make check` on a login node. This is read-only apart from ordinary test
   caches ignored by Git.
3. Run the dotenv-loader tests, which use temporary files and clean subprocess
   environments rather than the real `.env`.
4. Submit one CCL Y1 Single smoke job.
5. Submit one Numba Y1 Single smoke job.
6. Submit one JAX CPU Y1 Single smoke job.
7. Submit one JAX GPU Y1 Single smoke job and confirm `jax.devices()` reports a
   GPU.
8. Submit one covariance smoke job after setting
   `LIMBERCLOUD_ONECOVARIANCE_ROOT`.
9. Confirm generated filenames follow the `Time_Single_*` contract before
   submitting the full experiment matrix.
10. Validate all notebooks and select the `.venv`/`CosmoConda` kernel for
    interactive execution.

All jobs use the canonical runtime tree documented in
[runtime-tree.md](runtime-tree.md). Verify that tree before submitting jobs.
