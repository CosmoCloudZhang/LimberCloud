# Covariance Inputs for Angular Power Spectra

## Scientific Scope

This directory builds the inputs needed by covariance estimators for tomographic
angular power spectra in weak-lensing and large-scale-structure analyses. The
core script (`Y1/DATA.py` and `Y10/DATA.py`) converts survey-specific inputs into
redshift-distribution tables, tracer bias tables, and the baseline angular power
spectra \(C_\ell\) for three observable pairs:

- shear–shear (kappa–kappa)
- galaxy–shear (g–kappa)
- galaxy–galaxy (g–g)

All power spectra are computed with `pyccl` in a fiducial cosmology and written
as ASCII tables for downstream covariance calculations (e.g., OneCovariance).

## Data Products

For a given `tag`, the pipeline writes to `COVARIANCE/<tag>/`:

- `SOURCE.ascii`: normalized source redshift distributions \(n_i(z)\)
- `LENS.ascii`: normalized lens redshift distributions \(n_i(z)\)
- `ALIGNMENT.ascii`: intrinsic-alignment amplitude \(A_i(z)\)
- `GALAXY.ascii`: galaxy bias \(b_i(z)\)
- `MAGNIFICATION.ascii`: magnification bias \(m_i(z)\)
- `Cell_kappakappa.ascii`: shear–shear \(C_\ell\)
- `Cell_gkappa.ascii`: galaxy–shear \(C_\ell\)
- `Cell_gg.ascii`: galaxy–galaxy \(C_\ell\)

## Method Summary

- Loads source/lens tomographic bins from `DATA/<tag>/` and interpolates them
  onto a 351-point redshift grid (uniform in \(z\)), then renormalizes each bin.
- Loads alignment, galaxy-bias, and magnification-bias models from `INFO/`.
- Builds a `pyccl.Cosmology` instance and computes \(C_\ell\) on a geometric
  multipole grid (\(\ell=20\)–2000, 101 points) using the Limber approximation
  with spline integration.

## Usage

Run the generator for a specific tag:

```bash
python /path/to/COVARIANCE/<tag>/DATA.py --tag <tag> --folder <base_folder>
```

`<base_folder>` must contain `DATA/`, `INFO/`, and `COVARIANCE/`.

## Notes

- `Y1/DATA.sh` and `Y10/DATA.sh` are Slurm job scripts that run the generator
  and then call OneCovariance with a tag-specific configuration file.
