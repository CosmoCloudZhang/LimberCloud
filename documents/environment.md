# Python environment and editor setup

LimberCloud uses one selected interpreter per machine, referenced by the
checkout-local `.venv` link. Create or reuse an environment once, point `.venv`
at it, and use the same prefix for scripts, the editor, and Jupyter.

Local work covers planning, review, and the manuscript. The NERSC paper
submodule remains uninitialized or absent. See
[manuscript-workflow.md](manuscript-workflow.md).

## Layers

1. `environment.yml` — portable CPU recipe (`limbercloud`; CPU JAX, serial
   `h5py`, conda-forge `mpi4py`). No CosmoSIS. No mandatory CUDA JAX.
2. `environment.nersc.yml` — NERSC CUDA JAX variant with matching scientific
   pins. Build `mpi4py` / parallel `h5py` afterward with
   `scripts/nersc/install_mpi_h5py.sh` against Cray MPICH and
   `cray-hdf5-parallel` (do not install a generic MPI stack over NERSC's).
3. NERSC module profiles under `scripts/nersc/modules/` — Conda, GNU, Cray
   MPICH, parallel HDF5. They also set `HDF5_USE_FILE_LOCKING=FALSE` for CFS
   writes; disabled locking still requires exclusive writer ownership.
4. Private `${PROJECT_ROOT}/.env` — external path configuration only.

## Preserve CosmoConda while validating limbercloud

A working collaboration `CosmoConda` may contain CosmoSIS and other software.
Keep it until the dedicated `limbercloud` environment and notebook launches
pass. To reuse CosmoConda temporarily:

```bash
module load conda
conda activate CosmoConda
python -m pip install --no-deps -e .
test ! -e .venv || readlink -f .venv
# Only if .venv is absent:
# ln -s "${CONDA_PREFIX}" .venv
```

Never replace an existing `.venv` until its target has been inspected.

## Create the dedicated limbercloud environment

Portable / laptop CPU:

```bash
scripts/nersc/create_environment.sh --name limbercloud
```

NERSC CUDA variant (leaves MPI/HDF5 Python builds to the site installer):

```bash
module load conda
scripts/nersc/create_environment.sh --name limbercloud --nersc
conda activate limbercloud
source scripts/nersc/modules/cpu.sh
scripts/nersc/install_mpi_h5py.sh
```

The helper refuses to modify CosmoConda or an existing `.venv`. After
validation, point `.venv` at the accepted prefix if it still targets
CosmoConda:

```bash
readlink -f .venv
rm .venv   # only after inspection
ln -s "${CONDA_PREFIX}" .venv
```

OneCovariance (`@311c2cf` on NERSC) needs a Python build with `gfortran`,
`gsl`, and `pybind11` for its own install; those stay outside the minimal
LimberCloud recipe unless you install OneCovariance into the same prefix.

## Select, check, and register

```bash
# Configuration (fixed .env; exported values win)
cp .env.example .env   # first time
source scripts/load_config.sh

# Interpreter diagnostics
scripts/nersc/diagnose_environment.sh

# Notebook kernel (display name LimberCloud)
scripts/jupyter/register_kernel.sh
scripts/jupyter/launch_kernel.sh --probe
```

## Local configuration

| Key | Required | Meaning |
| --- | --- | --- |
| `LIMBERCLOUD_RUNTIME_ROOT` | Yes | External runtime data and results tree |
| `LIMBERCLOUD_ONECOVARIANCE_ROOT` | Covariance only | Checkout containing `covariance.py` |
| `LIMBERCLOUD_TEXLIVE_BIN` | No | Directory containing `pdflatex` |

`PROJECT_ROOT` is discovered automatically. Python comes from `.venv` only.
Do not set `LIMBERCLOUD_CONDA_ENV`, `LIMBERCLOUD_ENV_FILE`, or
`LIMBERCLOUD_REPO_ROOT`.

Batch jobs source `scripts/load_config.sh`, the CPU/GPU module profile, and
`scripts/nersc/activate_venv.sh`.

## VS Code and notebooks

Tracked `.vscode/settings.json` selects `.venv`, enables indentation guides,
and injects `.env`. Prefer the **LimberCloud** Jupyter kernel
(`limbercloud`), which loads configuration through
`scripts/jupyter/launch_kernel.sh`.

```python
import os
import sys
import limbercloud

print(sys.executable)
print(limbercloud.__file__)
print(os.environ["LIMBERCLOUD_RUNTIME_ROOT"])
```

## Sample-count safety

Spectra runners default to `--sample-count=0`. Pass `--fiducial-only` or an
explicit `--sample-count` (campaign: `1000`) before any science launch.
`--number` remains the host CPU allocation label, not a sample limit.
