# LimberCloud: A Framework for Cosmological Power Spectra Calculations

## Overview

**LimberCloud** is a fully analytic framework designed for the **fast and scalable computation** of **angular power spectra** in cosmology. The framework utilizes **weak gravitational lensing** and **large-scale structure** (LSS) data to calculate covariance matrices for angular power spectra. The code performs various mathematical and statistical analyses, including power spectrum generation, kernel calculations, error modeling, and projections that are essential for weak lensing surveys and cosmological simulations.

This repository consists of Python scripts, Jupyter notebooks, and configuration files that work together to compute, store, and analyze the fiducial cosmological parameters required for power spectra analysis.

## Folder Structure

1. **`PYTHON/`**:
   - This folder contains the core computational scripts and Jupyter notebooks used for various mathematical calculations and projections.
   - **`POWER/`**: Contains notebooks related to **power spectrum** computations.
   - **`CELL/`**, **`ERROR/`**, **`KERNEL/`**, **`PROJECTION/`**: These folders contain scripts related to **error modeling**, **cosmological kernels**, and **projection techniques** used for weak lensing and large-scale structure analysis.
   - **`README.md`**: Provides an overview of the **PYTHON** folder.

2. **`COVARIANCE/`**:
   - Contains subdirectories **`Y1`** and **`Y10`**, which store the computed **covariance matrices** for different datasets and configurations.
   - The matrices are stored in **ASCII** format, ready for analysis.
   - **`README.md`**: Describes the covariance matrices for angular power spectra.

3. **`MATH/`**:
   - Includes scripts and notebooks for **mathematical operations** such as integration and coefficient calculations for the angular power spectra.
   - **`NN/`**, **`NS/`**, **`SS/`**: These subdirectories contain notebooks for **numerical integration** and **coefficient calculation**.
   - **`README.md`**: Overview of the mathematical operations in the **MATH** folder.

4. **`INFO/`**:
   - Contains **configuration** and **parameter files** necessary for the cosmological calculations.
   - Files include scripts like **`COSMOLOGY.py`**, **`ALIGNMENT.py`**, **`MAGNIFICATION.py`**, and more, which define and store cosmological parameters, biases, and other relevant data.
   - **`README.md`**: Provides an explanation of the input parameter files for cosmological analysis.

## Purpose

The **LimberCloud** framework is used for:
- **Cosmological parameter modeling**: Including Hubble constant, dark energy, and dark matter densities.
- **Angular power spectra calculation**: Using weak gravitational lensing and large-scale structure data.
- **Covariance matrix generation**: To assess the correlation between different redshift bins and other cosmological effects.
- **Error modeling**: To quantify uncertainties in the angular power spectrum calculations.
- **Projection techniques**: Used to transform and model cosmological data from one form to another.

## Usage

### Running the Code

Each folder and script in the repository can be used independently for specific tasks such as cosmological parameter calculations, power spectrum modeling, or error assessments.

1. **Running a Python Script**:
   To run any of the Python scripts, use the following command:

   ```bash
   python <script_name.py> --folder <path_to_base_folder>
