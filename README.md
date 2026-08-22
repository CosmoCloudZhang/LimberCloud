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

Create or activate an environment containing the required scientific libraries,
then install this checkout in editable mode:

```bash
python3 -m pip install -e .
```

For a development environment managed with pip:

```bash
python3 -m pip install -e '.[science,dev]'
```

NERSC environments should continue to use the collaboration's validated CCL,
JAX, Numba, and MPI-compatible dependency versions rather than rebuilding those
packages unnecessarily.

## Environment variables and runtime data

The Git checkout does not contain the LSST input arrays or production results.
`LIMBERCLOUD_RUNTIME_ROOT` must point to the external directory that contains
the canonical `data/`, `config/`, `results/`, `plots/`, and `logs/` tree.

The project recognizes the following environment variables:

| Variable | Requirement | Purpose |
| --- | --- | --- |
| `LIMBERCLOUD_RUNTIME_ROOT` | Required | External data, configuration, result, plot, and log root |
| `CosmoENV` | Required by NERSC launchers | Name or path of the collaboration's validated Conda environment |
| `ONE_COVARIANCE_ROOT` | Required for covariance jobs | Path to OneCovariance's `covariance.py` |

### Shell and Slurm jobs

Export the required values in the shell from which scripts or Slurm jobs are
launched:

```bash
export LIMBERCLOUD_RUNTIME_ROOT=/path/to/external/LimberCloud
export CosmoENV=/path/to/or/name/of/validated/conda/environment
```

For covariance jobs, also set:

```bash
export ONE_COVARIANCE_ROOT=/path/to/OneCovariance/covariance.py
```

Confirm that the required values are visible before launching a job:

```bash
printf '%s\n' "${LIMBERCLOUD_RUNTIME_ROOT}"
printf '%s\n' "${CosmoENV}"
```

### Cursor or VS Code notebooks

When a notebook is opened directly in Cursor or VS Code, its Python kernel does
not inherit variables exported later in an integrated terminal. Instead, create
a `.env` file in the repository root. For a Cursor Remote SSH session, create
the file in the remote NERSC checkout rather than in the local checkout:

```dotenv
LIMBERCLOUD_RUNTIME_ROOT=/path/to/external/LimberCloud
```

Do not include the shell keyword `export` in `.env`. The tracked
`.vscode/settings.json` points the Python and Jupyter extensions to this file,
and `.env` is ignored by Git so each checkout can use its own runtime path.
After creating or changing the file, reload the editor window and restart the
notebook kernel. Verify the kernel environment with:

```python
import os

print(os.environ.get("LIMBERCLOUD_RUNTIME_ROOT"))
```

The shell launchers do not source `.env`; continue to use `export` in the shell
that submits Slurm jobs. All inputs and outputs use one canonical runtime tree.
See [docs/runtime-tree.md](docs/runtime-tree.md) for its directory,
configuration, and timing-file contracts, and [docs/nersc.md](docs/nersc.md)
for the Perlmutter workflow.

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
