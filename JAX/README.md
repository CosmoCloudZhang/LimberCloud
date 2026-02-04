# JAX-Accelerated Projections and Analysis Notebooks

This folder contains JAX-based projection kernels and analysis notebooks used to compute and validate angular power spectra and related error budgets for weak-lensing and galaxy-clustering observables. The code emphasizes fast, vectorized evaluation of line-of-sight integrals and includes CPU/GPU benchmarking workflows.

## Contents

### Projection kernels (`PROJECTION/`)

- `NN.py`, `NS.py`, `SN.py`, `SS.py` implement JAX-jitted kernels for line-of-sight projections of matter power spectra on comoving distance grids.
- Each module defines:
  - Closed-form kernel elements for piecewise integration (multiple `element*` functions).
  - A `coefficient(...)` routine that assembles tomographic coefficient tensors using `jax.vmap` and `jax.lax.fori_loop`.
  - A `function(...)` interface that contracts kernel coefficients with tomographic basis functions via `einsum`.

These kernels provide the core building blocks for number-counts and shear projections and are designed for efficient batching over multipoles and redshift bins.

### Angular power spectra (`CELL/`)

Notebooks under `CELL/Y1` and `CELL/Y10` compute tomographic angular power spectra for:

- `EE.ipynb`: shear–shear (`κ–κ`) spectra
- `TE.ipynb`: galaxy–shear (`g–κ`) spectra
- `TT.ipynb`: galaxy–galaxy (`g–g`) spectra

They load fiducial inputs from `DATA/` and `INFO/`, build redshift and comoving-distance grids, evaluate matter power spectra with `pyccl`, and construct tomographic `C_ell` terms (including intrinsic-alignment contributions where applicable).

### Error propagation (`ERROR/`)

Notebooks under `ERROR/Y1` and `ERROR/Y10` read precomputed spectra and covariance inputs to evaluate error propagation and diagnostic comparisons (e.g., CCL-based versus data-driven spectra). These notebooks work on the same tomographic grids and multipole ranges used for the power-spectrum calculations.

### CPU/GPU benchmarks (`CPU/`, `GPU/`)

Benchmark notebooks under `CPU/Y1`, `CPU/Y10`, `GPU/Y1`, and `GPU/Y10` use JAX implementations (`SSjax`, `NSjax`, `NNjax`) to evaluate projection kernels and compare performance across devices:

- `SINGLE.ipynb`, `DOUBLE.ipynb`, `TRIPLE.ipynb` represent different workload configurations.
- The CPU notebooks set JAX environment flags for multi-threading, while GPU notebooks query CUDA devices and run the same kernels on accelerators.

## Dependencies

- `jax`, `jaxlib`
- `pyccl`
- `numpy`, `scipy`
- `h5py` (for calibration inputs in benchmark notebooks)
- `matplotlib` (plotting in notebooks)

## Usage Notes

These notebooks are research workflows that assume existing datasets under `DATA/`, `INFO/`, and external calibration folders (see paths defined in the notebook cells). Update the paths to match your environment before executing.
