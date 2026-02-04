# Angular Power Spectra Inputs for Covariance Analysis

## Overview

This module prepares the redshift-dependent inputs and angular power spectra needed for covariance analyses of weak-lensing and galaxy-clustering observables. It constructs normalized source and lens distributions, alignment, galaxy-bias, and magnification-bias tables, and computes the tomographic angular power spectra using **pyccl**.

The implementation lives in `COVARIANCE/Y1/DATA.py` and `COVARIANCE/Y10/DATA.py`, which are identical in logic and are typically run with different configuration tags.

## Scientific Scope

- **Source (shear) and lens (number counts) redshift distributions** are interpolated onto a fixed grid and normalized.
- **Intrinsic alignment, galaxy bias, and magnification bias** are read from `INFO/*.json` and tabulated versus redshift.
- **Angular power spectra** are computed for:
  - **κ–κ** (cosmic shear, `Cell_kappakappa`)
  - **g–κ** (galaxy–shear, `Cell_gkappa`)
  - **g–g** (galaxy clustering, `Cell_gg`)

The spectra are evaluated on a logarithmic multipole grid and saved in a flattened tomographic format suitable for downstream covariance assembly.

## Inputs

The scripts expect the following structure under the base `folder`:

- `DATA/<tag>/lsst_source_bins.npy`
- `DATA/<tag>/lsst_lens_bins.npy`
- `INFO/COSMOLOGY.json`
- `INFO/ALIGNMENT.json`
- `INFO/GALAXY.json`
- `INFO/MAGNIFICATION.json`

## Outputs

Files are written to `COVARIANCE/<tag>/`:

- `SOURCE.ascii`: normalized source n(z) for each shear bin
- `LENS.ascii`: normalized lens n(z) for each clustering bin
- `ALIGNMENT.ascii`: intrinsic-alignment amplitude A(z) per source bin
- `GALAXY.ascii`: galaxy-bias b(z) per lens bin
- `MAGNIFICATION.ascii`: magnification-bias m(z) per lens bin
- `Cell_kappakappa.ascii`: shear–shear angular power spectra
- `Cell_gkappa.ascii`: galaxy–shear angular power spectra
- `Cell_gg.ascii`: galaxy–galaxy angular power spectra

Each `Cell_*.ascii` file stores `ell`, `tomo_i`, `tomo_j`, and the corresponding spectrum value.

## Numerical Configuration

- Redshift grids: 351 points per distribution (`grid_size = 350`)
- Multipoles: `ell` from 20 to 2000 (`geomspace`, 101 points)
- Power spectrum: `pyccl` with Halofit (`mead2020_feedback`) and CAMB transfer function
- Tracers: `WeakLensingTracer` and `NumberCountsTracer` with IA, bias, and magnification terms

## Usage

```bash
python /path/to/COVARIANCE/Y1/DATA.py --tag <config_tag> --folder <base_folder>
```

Use `Y10/DATA.py` analogously if your configuration uses that subfolder.
