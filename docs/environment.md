# Python environment and editor setup

LimberCloud uses three separate configuration layers:

1. `environment.yml` describes a minimum standalone `CosmoConda` for new
   installations.
2. NERSC module profiles under `scripts/nersc/modules/` select system-provided
   CPU, GPU, MPI, HDF5, and Conda support for jobs.
3. A private repository-root `.env` records per-checkout paths and overrides.

The YAML does not replace NERSC modules, and `.env` does not install packages.

## Reuse an existing CosmoConda

A working collaboration environment may contain more than LimberCloud needs,
including parallel HDF5, CosmoSIS, and locally validated MPI builds. Do not
recreate or update it merely because this repository provides
`environment.yml`. Install only the checkout without dependency resolution:

```bash
module load conda
conda activate CosmoConda
python -m pip install --no-deps -e .
```

VS Code uses the ignored checkout-local `.venv` name. If it does not exist,
link it to the already active environment:

```bash
test ! -e .venv
ln -s "${CONDA_PREFIX}" .venv
```

Never replace an existing `.venv` until its target has been inspected with
`readlink .venv`.

## Create a new minimum environment

New users may invoke the opt-in helper with a new environment name or absolute
prefix:

```bash
scripts/nersc/create_environment.sh --name CosmoConda
# Or: scripts/nersc/create_environment.sh --prefix /absolute/new/prefix
```

The helper stops if the requested Conda environment already exists. It creates
the environment from `environment.yml`, installs this checkout in editable mode
with `--no-deps`, and creates the local `.venv` link only when that path is
absent. If `.venv` already exists, the helper reports it and leaves it unchanged.
It is not an upgrade command for an established collaboration environment.

`environment.yml` is a curated minimum rather than a byte-for-byte export of a
developer environment. Raw Conda exports contain unrelated packages and
platform build strings, while the NERSC modules remain outside Conda.

The minimum manifest intentionally does not install `mpi4py` or `h5py`.
Parallel Python bindings must be built or cloned against NERSC's Cray MPICH and
parallel HDF5 stack; ordinary Conda builds do not establish that integration.
Keep a working build in an existing `CosmoConda`. New users who need it should
follow the [NERSC parallel Python guide](https://docs.nersc.gov/development/languages/python/parallel-python/)
and validate it on compute nodes before production use.

## Local configuration

Create an ignored `.env` from the public template:

```bash
cp .env.example .env
```

The canonical keys are:

| Key | Required | Meaning |
| --- | --- | --- |
| `LIMBERCLOUD_RUNTIME_ROOT` | Yes | External runtime data and results tree |
| `LIMBERCLOUD_CONDA_ENV` | No | Conda name or prefix; defaults to `CosmoConda` |
| `LIMBERCLOUD_ONECOVARIANCE_ROOT` | Covariance only | Checkout containing `covariance.py` |
| `LIMBERCLOUD_TEXLIVE_BIN` | No | Directory containing `pdflatex` |

An already exported canonical key overrides the value in `.env`. Set
`LIMBERCLOUD_ENV_FILE` to use a different dotenv file. The parser accepts plain
`KEY=value` assignments and quoted values but does not execute shell code.

`CosmoENV`, `ONECOVARIANCE_SCRIPT`, and `ONE_COVARIANCE_ROOT` are deprecated
migration inputs. New configuration must use the canonical names above.

## CPU and GPU separation

CPU jobs source `scripts/nersc/modules/cpu.sh`; JAX GPU jobs source
`scripts/nersc/modules/gpu.sh`. Both source the common Conda, Cray MPI, GNU
programming-environment, and parallel-HDF5 profile. Loading a GPU module does
not allocate a GPU: GPU launchers also retain their Slurm GPU constraint and
GPU resource request.

The GPU-capable JAX packages in `CosmoConda` and the selected NERSC CUDA module
must remain a validated pair. The setup in an existing collaboration
environment is left untouched.

## VS Code and notebooks

The tracked `.vscode/settings.json` selects `${workspaceFolder}/.venv`, adds
`src/` for static analysis, and injects `.env` into new integrated terminals.
After creating or changing `.venv`:

1. Run **Python Environments: Refresh All Environment Managers**.
2. Run **Python: Select Interpreter** and choose `.venv`/`CosmoConda`.
3. In a notebook, use **Select Kernel** and choose the same environment.
4. Reload the window and restart existing notebook kernels.

The project does not rely on notebook metadata to force a machine-specific
kernelspec. VS Code remembers each user's kernel selection.

Verify both Python files and notebooks with:

```python
import os
import sys
import limbercloud

print(sys.executable)
print(limbercloud.__file__)
print(os.environ["LIMBERCLOUD_RUNTIME_ROOT"])
```

The executable should resolve through `.venv`, and `limbercloud.__file__`
should resolve under this checkout's `src/limbercloud/` directory.
