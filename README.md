# LimberCloud: Tensorised Angular Power Spectra

## Overview

**LimberCloud** is a tensorised analytic framework for **fast, scalable computation of angular power spectra** and their **covariance matrices** in weak lensing and large-scale structure analyses. The project:

- Derives and validates analytic expressions and coefficient tables in `MATH/`.
- Implements a reference pipeline in `PYTHON/` to compute kernels, power spectra, and C_ell, and to quantify absolute/relative errors.
- Generates covariance inputs in `COVARIANCE/` using fiducial cosmology and astrophysical parameters from `INFO/`.
- Provides a JAX implementation in `JAX/` to validate parity with the Python baseline and benchmark CPU/GPU performance.

The repository combines scripts and notebooks that document the derivations, implement the numerical pipeline, and validate precision across implementations.

## Repository Layout

- `MATH/`: Analytic derivations and coefficient generation for tensorised expressions.
- `INFO/`: Fiducial cosmology, survey, and astrophysical parameter definitions.
- `COVARIANCE/`: Precomputed covariance inputs for different survey configurations (e.g., `Y1`, `Y10`).
- `PYTHON/`: Reference implementation for kernels, power spectra, C_ell, and error analysis.
- `JAX/`: JAX implementation used for parity checks and performance benchmarking.

## Scientific Workflow

1. **Derivation and coefficient generation** (`MATH/`).
2. **Parameter definition** (`INFO/`).
3. **Covariance inputs** (`COVARIANCE/`).
4. **Reference computations and validation** (`PYTHON/`).
5. **Accelerated implementation and benchmarking** (`JAX/`).

## Getting Started

Most scripts accept a base folder via `--folder`. For example:

```bash
python INFO/COSMOLOGY.py --folder /path/to/LimberCloud
python COVARIANCE/Y1/DATA.py --folder /path/to/LimberCloud
```

Notebooks in `PYTHON/` and `JAX/` are organized by task and survey configuration (`Y1`, `Y10`).

## Dependencies

- Python 3.x
- Jupyter (for notebooks)
- `pyccl`, `numpy`, `scipy`, `astropy`
<<<<<<< HEAD
- JAX
