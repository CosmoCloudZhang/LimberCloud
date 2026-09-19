# LimberCloud

LimberCloud is an analytic framework for fast, scalable computation of angular
power spectra for weak gravitational lensing and large-scale structure.

The [September revision package](revisions/2026-09/README.md) contains the
author-comment inventory, finalized code and manuscript plans, and staged
implementation prompts. Code, scripts, and notebook implementation take place
on NERSC; manuscript editing and publication-figure integration take place
locally.

## Repository structure

```text
src/limbercloud/       Reusable Numba/JAX projection code and runtime paths
experiments/           CCL, Numba, JAX, covariance, and benchmark entry points
scripts/               Configuration generators, validators, and NERSC helpers
notebooks/             Derivation, spectrum, error, kernel, and power notebooks
manuscript/            Optional LimberCloudPaper submodule, edited locally
tests/                 Fast path, experiment-contract, and backend checks
documents/             Runtime, NERSC, and manuscript workflows
revisions/             Author feedback, revision plans, and agent prompts
```

## Installation

Prefer a dedicated `limbercloud` environment. Preserve an existing
`CosmoConda` until that environment and notebook launches pass.

```bash
# Portable CPU (no mandatory CUDA JAX)
scripts/nersc/create_environment.sh --name limbercloud

# NERSC CUDA variant + site MPI/HDF5 builds
scripts/nersc/create_environment.sh --name limbercloud --nersc
conda activate limbercloud
source scripts/nersc/modules/cpu.sh
scripts/nersc/install_mpi_h5py.sh
```

Temporary CosmoConda reuse (do not recreate it for LimberCloud alone):

```bash
module load conda
conda activate CosmoConda
python -m pip install --no-deps -e .
```

The ignored `.venv` link is the sole checkout-local interpreter selector. See
[documents/environment.md](documents/environment.md).

## Environment variables and runtime data

`LIMBERCLOUD_RUNTIME_ROOT` must point to the external directory that contains
the canonical `data/`, `config/`, `results/`, `plots/`, and `logs/` tree.
`PROJECT_ROOT` is discovered automatically. Python comes from `.venv`.

| Variable | Requirement | Purpose |
| --- | --- | --- |
| `LIMBERCLOUD_RUNTIME_ROOT` | Required | External data, configuration, result, plot, and log root |
| `LIMBERCLOUD_ONECOVARIANCE_ROOT` | Covariance jobs only | OneCovariance checkout containing `covariance.py` |
| `LIMBERCLOUD_TEXLIVE_BIN` | Optional | Directory containing `pdflatex` for plotting |

```bash
cp .env.example .env
```

```dotenv
LIMBERCLOUD_RUNTIME_ROOT=/path/to/external/LimberCloud
# LIMBERCLOUD_ONECOVARIANCE_ROOT=/path/to/OneCovariance
# LIMBERCLOUD_TEXLIVE_BIN=/path/to/texlive/bin
```

Jobs and notebooks load `${PROJECT_ROOT}/.env` through `scripts/load_config.sh`
without executing it. Exported values take precedence. Do not set
`LIMBERCLOUD_CONDA_ENV`, `LIMBERCLOUD_ENV_FILE`, or `LIMBERCLOUD_REPO_ROOT`.

### Cursor or VS Code notebooks

Register the **LimberCloud** kernel once:

```bash
scripts/jupyter/register_kernel.sh
scripts/jupyter/launch_kernel.sh --probe
scripts/nersc/diagnose_environment.sh
```

## Verification

```bash
make check
```

Spectra runners default to `--sample-count=0`. Pass `--fiducial-only` or an
explicit `--sample-count` (campaign: `1000`) before science jobs. `--number` is
the host CPU allocation label, not a sample limit.

## Experiments

Backend matrix under `experiments/spectra/`: CCL, Numba CPU, JAX CPU, and JAX
GPU for Y1/Y10 × Single/Double/Triple. `Single`/`Double`/`Triple` select probe
configurations (EE / TE+TT / EE+TE+TT), not tiny runs.

## Notebooks and manuscript

The manuscript is the separate `LimberCloudPaper` repository, checked out
locally through the optional `manuscript/` submodule. On NERSC, leave it
uninitialized or absent and update the parent with:

```bash
git pull --ff-only --no-recurse-submodules
git ls-tree HEAD manuscript
```

Do not initialize the paper for code work. Package installs and ordinary checks
must succeed with `manuscript/` absent. Overleaf synchronization, if used,
targets the paper repository directly—not a parent-repository subtree.

See [documents/manuscript-workflow.md](documents/manuscript-workflow.md),
[documents/nersc.md](documents/nersc.md), and
[documents/runtime-tree.md](documents/runtime-tree.md).
