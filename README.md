# LimberCloud

LimberCloud is an analytic framework for fast, scalable computation of angular
power spectra for weak gravitational lensing and large-scale structure.

The [September revision package](revisions/2026-09/README.md) contains the
author-comment inventory, the updated 22 September code/manuscript completion
plans, the Phase 0/1 implementation audit, and detailed new Phase 1–5 prompts. Code, scripts, and notebook implementation take place
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

`--probe` reports the interpreter and checkout identity only and stays
login-safe. `--science` additionally performs an HDF5 round trip and a small
CCL background evaluation, so run it under an allocation on a supported host.

## Verification

```bash
make check       # login-safe: lint, fast tests, shell syntax, notebook parsing
make check-all   # adds the h5py and compiled-backend suites; run on an allocation
```

`make` uses `.venv/bin/python3` when that link exists, so it does not pick up
the system Python. Override with `make PYTHON=...` only for a deliberate other
prefix.

`make test-fast` imports NumPy and SciPy only. `make test-science` imports h5py
and the Numba and JAX backends; with an MPI-linked h5py build it is not
login-safe.

Spectra runners always evaluate the fiducial, sample 0 of `--sample-table`.
`--sample-count` is how many further cosmologies to evaluate; the default `0`
is the fiducial alone. Products are written in the family/survey directory,
for example `results/spectra/NUMBA/Y1/Time_Triple_Fiducial.txt` and
`Time_Triple_Cosmology.txt`. A fiducial-only rerun replaces only `Fiducial`
files; a sampled run replaces both populations. Sampled files contain actual
checkpoint counts and cumulative seconds, excluding sample 0. Older unqualified
and `*_128*` names are left in place and are not selected by the new reader.
Host CPU count stays in the Slurm allocation and thread environment; it is
not a filename token or a sample limit.
Launchers forward their arguments to the driver, so submitted selections reach
Python:

```bash
sbatch experiments/spectra/NUMBA/Y1/triple.sh \
    --sample-count=10 --sample-table "$LIMBERCLOUD_RUNTIME_ROOT/results/spectra/inputs/pilot"
```

The benchmark figure reads `Cosmology` timing products from those same directories
and uses their recorded counts. It does not take a run ID. The legend label is
`CCL`; in stage panels it repeats the CCL end-to-end reference, as stated in the
figure footnote. Counts need not be multiples of 100: a small pilot gets one
checkpoint, and larger runs record each 100 plus the final requested count.

## Experiments

Backend matrix under `experiments/spectra/`: CCL, Numba CPU, JAX CPU, and JAX
GPU for Y1/Y10 × Single/Double/Triple. `Single`/`Double`/`Triple` select probe
configurations (EE / TE+TT / EE+TE+TT), not tiny runs. Every family evaluates
the same 21 multipoles, `numpy.geomspace(20, 2000, 21)`; the radial
interpolation order belongs to the NUMERIC family alone.

## Notebooks and manuscript

Each original Mathematica coefficient derivation in `notebooks/derivation/`
has a readable same-basename Jupyter edition. The corrected mathematical text
uses `p = 1 - P1/P2` for normalized power; preserved Wolfram source cells and
saved outputs are identified separately from executable Python equivalents.
The conversion inventory and notation notes live beside these notebooks.
The four additional NS/SS boundary derivations use the same layout. Reading
the derivations requires no Mathematica installation; symbolic recomputation
uses SymPy in the notebook environment.

The manuscript is the separate `LimberCloudPaper` repository, checked out
locally through the optional `manuscript/` submodule. On NERSC, leave it
uninitialized or absent. Automatic manuscript fetching is disabled in
`.gitmodules`; explicit local paper updates remain available. On NERSC only,
set these checkout-local defaults once, then update the parent with:

```bash
git config --local submodule.recurse false
git config --local fetch.recurseSubmodules false
git pull --ff-only --no-recurse-submodules
git ls-tree HEAD manuscript
```

Do not initialize the paper for code work. Package installs and ordinary checks
must succeed with `manuscript/` absent. Overleaf synchronization, if used,
targets the paper repository directly—not a parent-repository subtree.

See [documents/manuscript-workflow.md](documents/manuscript-workflow.md),
[documents/nersc.md](documents/nersc.md), and
[documents/runtime-tree.md](documents/runtime-tree.md).
