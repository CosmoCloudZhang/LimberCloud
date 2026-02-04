# LimberCloud Fiducial Configuration Inputs

This folder defines the fiducial inputs used throughout the LimberCloud pipeline. Each script writes a self-contained JSON file under `INFO/` that encodes baseline cosmology, redshift-bin densities, and bias models for weak-lensing and galaxy-clustering analyses.

## Scope and Outputs

The scripts generate the following JSON products:

- `COSMOLOGY.json`: fiducial cosmological parameters (e.g., `H`, `Omega_*`, `A_s`, `n_s`, `w0`, `wa`).
- `ALIGNMENT.json`: intrinsic-alignment amplitude as a function of redshift, computed with `pyccl` growth and density.
- `DENSITY.json`: number-density normalizations for lens/source bins in `Y1` and `Y10`.
- `GALAXY.json`: linear galaxy-bias model per redshift, derived from growth factors.
- `MAGNIFICATION.json`: magnification-bias coefficients per lens bin for `Y1` and `Y10`.
- `SURVEY.json`: survey area and sky fraction for each tag.

All files are written under `INFO/` and are intended to be consumed by downstream modules (e.g., power spectrum and covariance builders).

## Script Details

- `COSMOLOGY.py`: defines the fiducial cosmology and writes `COSMOLOGY.json`.
- `ALIGNMENT.py`: loads `COSMOLOGY.json`, computes alignment amplitude on a redshift grid, and writes `ALIGNMENT.json`.
- `GALAXY.py`: loads `COSMOLOGY.json`, computes growth-factor–scaled galaxy bias for `Y1` and `Y10`, and writes `GALAXY.json`.
- `DENSITY.py`: stores lens/source number-density normalizations for the configured tomographic bins in `DENSITY.json`.
- `MAGNIFICATION.py`: stores fiducial magnification-bias coefficients in `MAGNIFICATION.json`.
- `SURVEY.py`: sets survey areas and computes sky fractions in `SURVEY.json`.

## Usage

Each script is a CLI that accepts `--folder` as the dataset base path:

```bash
python /path/to/INFO/COSMOLOGY.py --folder <base_folder>
```

Shell wrappers (`*.sh`) mirror these commands for batch execution on SLURM-enabled systems.
