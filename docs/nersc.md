# Perlmutter workflow

## Checkout and environment

Clone or update the repository on Perlmutter, activate the scientific Conda
environment, and install the checkout in editable mode:

```bash
python3 -m pip install -e '.[dev]'
```

Define the external runtime root before submitting jobs:

```bash
export LIMBERCLOUD_RUNTIME_ROOT=/path/to/external/LimberCloud
export ONECOVARIANCE_SCRIPT=/path/to/OneCovariance/covariance.py
```

`LIMBERCLOUD_REPO_ROOT` is optional. Slurm scripts derive the repository root
from Git when it is not set. `LIMBERCLOUD_TEXLIVE_BIN` is optional and should
point to the directory containing `pdflatex` when the system TeX installation is
not on `PATH`.

Create the checkout-local log directory before direct `sbatch` submissions:

```bash
mkdir -p logs
```

The `run_all.sh` launchers create it automatically and submit with the repository
as the Slurm working directory.

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

1. Run `make check` locally or on a login node with the relevant dependencies.
2. Submit one CCL Y1 Single smoke job.
3. Submit one Numba Y1 Single smoke job.
4. Submit one JAX CPU Y1 Single smoke job.
5. Submit one JAX GPU Y1 Single smoke job.
6. Confirm the generated filenames follow the `Time_Single_*` contract.
7. Submit the full experiment matrix using the backend `run_all.sh` files.
8. Run covariance and notebook validation.

All jobs use the canonical runtime tree documented in
[runtime-tree.md](runtime-tree.md). Verify that tree before submitting jobs.
