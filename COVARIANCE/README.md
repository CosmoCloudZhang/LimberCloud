# Covariance Matrix Calculation for Angular Power Spectra

## Overview

This program provides a fully analytic framework for the fast and scalable computation of angular power spectra in cosmology, focusing on **weak gravitational lensing** and **large-scale structure** (LSS). It calculates the covariance matrix necessary for angular power spectra by modeling various cosmological phenomena such as source redshift bins, lens redshift bins, alignment, galaxy biases, and magnification effects. The framework leverages the **pyccl** library to handle the cosmological calculations efficiently.

## Purpose

This code is designed to:
- Calculate and store covariance matrix data for angular power spectra.
- Use weak lensing and large-scale structure data to compute cross-correlations between different tomographic bins.
- Generate ASCII files for covariance matrix calculations, including redshift, alignment, galaxy, and magnification information.

## Directory Structure

- **DATA/**: Contains the source and lens data used for power spectrum calculations.

## Code Description

The main function of this framework is `main(tag, folder)`:
1. **Inputs**:
    - `tag`: The configuration tag used to locate specific datasets.
    - `folder`: The base folder that contains the `DATA` and `INFO` directories.
  
2. **Outputs**:
    - It generates covariance matrix data stored in the `COVARIANCE` directory. The output includes the source, lens, alignment, galaxy, and magnification data for each redshift bin.

3. **Workflow**:
    - **Data Loading**: The program loads data files (source bins, lens bins, cosmology parameters) and processes them for power spectrum calculations.
    - **Power Spectrum Calculation**: It computes the weak lensing angular power spectrum using the pyccl library, and handles various biases and redshift-dependent effects.
    - **Covariance Matrix**: The program computes and stores the covariance matrix for angular power spectra, saved as ASCII files for further analysis.

## Usage

To run the program, execute the following command in your terminal:

```bash
python <path_to_script> --tag <config_tag> --folder <base_folder>
