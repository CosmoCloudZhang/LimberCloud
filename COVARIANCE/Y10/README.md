# Covariance Matrix for Angular Power Spectra

## Overview

This code calculates and stores the covariance matrix for the angular power spectra, focusing on weak lensing and matter power spectra in cosmology. Using the **LSST (Large Synoptic Survey Telescope)** data for sources, lenses, galaxy biases, alignment biases, and magnification, the code produces a covariance matrix for weak lensing and matter distributions across various redshift bins.

The code also integrates cosmological models using **pyccl** (Python interface for the **Camb** cosmology library). The results are stored in ASCII files that contain redshift values and their corresponding power spectra. The core function of the code computes the angular power spectra for a given configuration and generates files for further analysis.

## Dependencies

- `pyccl` - Cosmological calculations
- `numpy` - Numerical operations
- `scipy` - For integration
- `astropy` - To handle and store data in tables
- `json` - Read configuration files

## File Structure

The file is designed to process the following data in the given folder structure:

- **DATA**: Contains the source and lens data (`lsst_source_bins.npy`, `lsst_lens_bins.npy`)
- **INFO**: Includes cosmology, alignment, galaxy, and magnification data (`COSMOLOGY.json`, `ALIGNMENT.json`, `GALAXY.json`, `MAGNIFICATION.json`)
- **COVARIANCE**: The output directory where the resulting ASCII files will be saved.

The code calculates the angular power spectra for three primary components:
1. **Cell EE** - Weak lensing and source bin correlation
2. **Cell TE** - Galaxy lens and source bin correlation
3. **Cell TT** - Galaxy lens and lens bin correlation

### Steps in the Code:
1. **Redshift Grid Definition**: Creates a grid for source and lens redshift values.
2. **Integration**: Calculates power spectrum integrals using numerical methods.
3. **File Generation**: Writes tables to `.ascii` files for each of the covariance components (EE, TE, TT).
4. **Cosmological Model Setup**: Configures the cosmological model using parameters from the `COSMOLOGY.json` file.

## Running the Code

To run the code, use the following command:

```bash
python covariance.py --tag <tag> --folder <path_to_data>
