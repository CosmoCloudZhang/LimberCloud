# LimberCloud

LimberCloud is an analytic framework for fast, scalable computation of angular
power spectra for weak gravitational lensing and large-scale structure.

## Repository structure

```text
src/limbercloud/       Reusable Numba/JAX projection code and runtime paths
experiments/           CCL, Numba, JAX, covariance, and benchmark entry points
scripts/               Configuration generators and migration utilities
notebooks/             Derivation, spectrum, error, kernel, and power notebooks
manuscript/            Journal manuscript and tracked publication figures
tests/                 Fast path, experiment-contract, and backend checks
docs/                  Runtime, NERSC, and manuscript workflows
```

General source directories and filenames use lowercase names. Scientific labels
such as `CCL`, `JAX`, `CPU`, `GPU`, `Y1`, `Y10`, `NN`, and `SS` remain uppercase.
Experiment configurations are displayed as `Single`, `Double`, and `Triple`,
while their source files are `single.py`, `double.py`, and `triple.py`.

Jupyter notebooks use `Title_Case_With_Underscores` for readability while
preserving scientific labels, for example `EE_Error_Analysis.ipynb` and
`Coefficient_B01_Validation.ipynb`. Their descriptive directories remain
lowercase, such as `error_analysis/` and `matter_power/`.

## Installation

Create or activate an environment containing the required scientific libraries,
then install this checkout in editable mode:

```bash
python3 -m pip install -e .
```

For a development environment managed with pip:

```bash
python3 -m pip install -e '.[science]'
```

NERSC environments should continue to use the collaboration's validated CCL,
JAX, Numba, and MPI-compatible dependency versions rather than rebuilding those
packages unnecessarily.

## Runtime data

The Git checkout does not contain the LSST input arrays or production results.
Set the external root before running scripts or notebooks:

```bash
export LIMBERCLOUD_RUNTIME_ROOT=/path/to/external/LimberCloud
```

The current transition default reads and writes the historical uppercase layout:

```bash
export LIMBERCLOUD_LAYOUT=legacy
```

After verified migration, use the canonical lowercase layout:

```bash
export LIMBERCLOUD_LAYOUT=canonical
```

See [docs/runtime-layout.md](docs/runtime-layout.md) for the complete mapping and
the non-destructive migration utility.

## Verification

Run the lightweight local checks with:

```bash
make check
```

This checks the package and experiment contracts, cross-backend projection
values when the optional scientific dependencies are available, shell syntax,
and notebook JSON/path setup. Full CCL/JAX/GPU and covariance validation remains
a Perlmutter workflow.

## Experiments

The complete backend matrix is retained under `experiments/spectra/`:

- CCL: Y1/Y10 × Single/Double/Triple
- Numba CPU: Y1/Y10 × Single/Double/Triple
- JAX CPU: Y1/Y10 × Single/Double/Triple
- JAX GPU: Y1/Y10 × Single/Double/Triple

Slurm launchers require `LIMBERCLOUD_RUNTIME_ROOT`; they derive the Git checkout
path automatically and default to the legacy runtime layout. See
[docs/nersc.md](docs/nersc.md) before switching production jobs to the canonical
layout.

## Notebooks and manuscript

Notebooks read the runtime root from the environment and use the installed
package for path resolution. Their stored outputs have been preserved, but the
scientific notebooks should be re-executed on Perlmutter after the canonical
input migration.

The manuscript lives under `manuscript/`. Publication figure PDFs are tracked;
LaTeX auxiliary files and `main.pdf` are ignored. See
[docs/manuscript-workflow.md](docs/manuscript-workflow.md) for the local Git to
Overleaf workflow.
