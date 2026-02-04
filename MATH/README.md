# LimberCloud-Cosmology Calculations and Angular Power Spectra Coefficients

## Overview

This folder contains Python scripts designed to perform **cosmological calculations** and compute **angular power spectra** coefficients for weak gravitational lensing and large-scale structure analysis. The scripts are used for calculating and storing fiducial values of cosmological parameters (such as `Hubble constant`, `dark energy parameters`, `matter density`, etc.), redshift distributions, and various biases that affect the angular power spectra computations.

## Folder Structure

1. **MATH/**:
   - Contains scripts and notebooks for the **mathematical computations** related to angular power spectra, including the calculation of integrals, coefficients, and power spectra.
   - **NN/**: Directory containing notebooks and Python scripts related to numerical methods and coefficient calculations for angular power spectra.
   - **NS/**: Directory containing scripts for non-standard cosmological parameter sets, including alternative models or configurations.
   - **SS/**: Directory for supplementary computations that focus on specific studies and models related to cosmological data analysis.

### **NN/**:
- **Coefficients-B1.ipynb**: Jupyter notebook for computing coefficients for angular power spectra based on cosmological models.
- **Coefficients-B2.ipynb**: Jupyter notebook for another set of coefficient calculations, tailored to a different angular power spectrum scenario.
- **Coefficients-B3.ipynb**: Further calculations for a different set of coefficients used for angular power spectrum analysis.

### **NS/**:
This directory houses configurations for non-standard models and special cases. These scripts may involve alternative assumptions, configurations, or new techniques not covered in the `NN` folder. Specific details will be addressed within each script.

### **SS/**:
Contains supplementary scripts designed for specialized analysis or data post-processing. The code here includes additional steps or refinements needed to handle specific datasets or scenarios within the cosmological framework.

## Purpose

The scripts in this folder perform the following operations:
- **Computation of Coefficients**: The notebooks and scripts calculate coefficients for angular power spectra related to weak lensing and large-scale structure.
- **Integration**: The scripts handle the numerical integration required for the calculation of angular power spectra.
- **Power Spectrum Calculation**: The framework computes the power spectrum by integrating over the redshift bins and cosmological parameters.
- **Cosmological Parameters**: The files read cosmological parameters (e.g., from `COSMOLOGY.json`), and use them in the calculations.

### Key Operations:
- **Cosmological Parameter Initialization**: The code reads from configuration files like `COSMOLOGY.json` to initialize parameters such as Hubble constant, dark energy density, and matter density.
- **Redshift Grid**: Redshift grids are generated, and corresponding scale factors and comoving distances are computed.
- **Coefficient Calculation**: The core calculations involve determining coefficients that relate to the angular power spectrum, including integral evaluations and adjustments based on redshift and cosmological parameters.

## Key Functions

### 1. **integral_I1(chi1, chi2, power1, power2, redshift1, redshift2)**:
   This function computes the integral required for the covariance matrix calculation. The formula is dependent on the comoving distances (`chi1`, `chi2`) and the power spectra (`power1`, `power2`).

### 2. **coefficient_J1(chi1, chi2, power1, power2, redshift1, redshift2)**:
   This function computes the coefficient for the angular power spectrum using the provided cosmological parameters. It is used to calculate the relationship between comoving distance and power spectra for weak lensing analysis.

## Usage

To run the scripts and generate the desired outputs, use the following command:

```bash
python <script_name.py> --folder <path_to_base_folder>
