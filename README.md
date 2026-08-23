# LimberCloud

LimberCloud is an analytic framework for fast, scalable computation of angular
power spectra for weak gravitational lensing and large-scale structure.

## Repository structure

```text
src/limbercloud/       Reusable Numba/JAX projection code and runtime paths
experiments/           CCL, Numba, JAX, covariance, and benchmark entry points
scripts/               Configuration generators, validators, and NERSC helpers
notebooks/             Derivation, spectrum, error, kernel, and power notebooks
manuscript/            Journal manuscript and tracked publication figures
tests/                 Fast path, experiment-contract, and backend checks
docs/                  Runtime, NERSC, and manuscript workflows
```

## Installation

LimberCloud's NERSC environment is named `CosmoConda`. If you already have a
validated `CosmoConda` with CCL, JAX, Numba, MPI, parallel HDF5, CosmoSIS, or
other collaboration software, keep it: this repository does not require it to
be recreated. Activate it and install only this checkout:

```bash
module load conda
conda activate CosmoConda
python -m pip install --no-deps -e .
```

For a new standalone installation, `environment.yml` describes the minimum
project environment baseline. The opt-in setup helper refuses to overwrite an
existing Conda environment or checkout-local `.venv`:

```bash
scripts/nersc/create_environment.sh --name CosmoConda
```

The ignored `.venv` entry in the checkout may be the environment itself or a
per-user symlink to it. The tracked VS Code configuration uses that stable local
name without committing anyone's absolute Conda path. See
[docs/environment.md](docs/environment.md) for reuse, new-installation, GPU, and
editor setup details.

## Environment variables and runtime data

The Git checkout does not contain the LSST input arrays or production results.
`LIMBERCLOUD_RUNTIME_ROOT` must point to the external directory that contains
the canonical `data/`, `config/`, `results/`, `plots/`, and `logs/` tree.

The project recognizes the following environment variables:

| Variable | Requirement | Purpose |
| --- | --- | --- |
| `LIMBERCLOUD_RUNTIME_ROOT` | Required | External data, configuration, result, plot, and log root |
| `LIMBERCLOUD_CONDA_ENV` | Optional; defaults to `CosmoConda` | Conda environment name or full prefix |
| `LIMBERCLOUD_ONECOVARIANCE_ROOT` | Covariance jobs only | OneCovariance checkout containing `covariance.py` |
| `LIMBERCLOUD_TEXLIVE_BIN` | Optional | Directory containing `pdflatex` for plotting |

### Shell and Slurm jobs

Create a private `.env` from the tracked template and edit the machine-specific
paths:

```bash
cp .env.example .env
```

The usual configuration is:

```dotenv
LIMBERCLOUD_RUNTIME_ROOT=/path/to/external/LimberCloud
# LIMBERCLOUD_CONDA_ENV=/full/path/to/CosmoConda
# LIMBERCLOUD_ONECOVARIANCE_ROOT=/path/to/OneCovariance
```

Omit `LIMBERCLOUD_CONDA_ENV` when `conda activate CosmoConda` works. Exported
canonical variables take precedence over `.env`, and `LIMBERCLOUD_ENV_FILE`
may select a different dotenv file. NERSC launchers load and validate this
configuration before activating Conda. The deprecated `CosmoENV`,
`ONECOVARIANCE_SCRIPT`, and `ONE_COVARIANCE_ROOT` names are accepted only as
temporary migration aliases and should not be added to new `.env` files.

### Cursor or VS Code notebooks

When a notebook is opened directly in Cursor or VS Code, its Python kernel does
not inherit variables exported later in an integrated terminal. Instead, create
the repository-root `.env` described above. For a Cursor Remote SSH session,
create it in the remote NERSC checkout rather than in the local checkout.

Do not include the shell keyword `export` in `.env`. The tracked
`.vscode/settings.json` uses `.venv` as the project interpreter and injects this
file into Python tools and new integrated terminals. `.env` and `.venv` are
ignored by Git. Select `.venv`/`CosmoConda` once in both **Python: Select
Interpreter** and the notebook kernel picker, then reload the editor window and
restart notebook kernels. Verify with:

```python
import os
import sys
import limbercloud

print(sys.executable)
print(limbercloud.__file__)
print(os.environ.get("LIMBERCLOUD_RUNTIME_ROOT"))
```

All inputs and outputs use one canonical runtime tree. See
[docs/runtime-tree.md](docs/runtime-tree.md) for its directory, configuration,
and timing-file contracts, and [docs/nersc.md](docs/nersc.md) for the Perlmutter
workflow.

## Verification

Run the lightweight local checks with:

```bash
make check
```

This checks the package and experiment contracts, cross-backend projection
values when the optional scientific dependencies are available, Python and
notebook lint, shell syntax, and notebook JSON/path setup. Full CCL/JAX/GPU and
covariance validation remains a Perlmutter workflow.

## Experiments

The complete backend matrix is retained under `experiments/spectra/`:

- CCL: Y1/Y10 × Single/Double/Triple
- Numba CPU: Y1/Y10 × Single/Double/Triple
- JAX CPU: Y1/Y10 × Single/Double/Triple
- JAX GPU: Y1/Y10 × Single/Double/Triple

Slurm launchers require `LIMBERCLOUD_RUNTIME_ROOT` and derive the Git checkout
path automatically. See [docs/nersc.md](docs/nersc.md) before running the
production matrix.

## Notebooks and manuscript

Notebooks read the runtime root from the environment and use the installed
package for path resolution. Their stored outputs have been preserved, but the
scientific notebooks should be re-executed on Perlmutter after the canonical
runtime tree has been populated.

The manuscript lives under `manuscript/`. Publication figure PDFs are tracked;
LaTeX auxiliary files and `main.pdf` are ignored. See
[docs/manuscript-workflow.md](docs/manuscript-workflow.md) for the local Git to
Overleaf workflow.
