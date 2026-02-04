# LimberCloud-Configuration and Fiducial Values for Cosmological Parameters

This folder contains various Python scripts that store the fiducial values for key cosmological parameters used in the computation of angular power spectra in weak gravitational lensing and large-scale structure. These scripts are part of a larger framework for cosmological simulations that involve different models and parameters for density, galaxy biases, alignment, and more.

## Purpose

The scripts in this folder compute and store the following information:
- **ALIGNMENT.py**: Fiducial values of intrinsic alignment parameters.
- **COSMOLOGY.py**: Fiducial values for cosmological parameters, including dark energy, dark matter, and other key cosmological constants.
- **DENSITY.py**: Fiducial values of density configuration for different redshift bins.
- **GALAXY.py**: Fiducial values for galaxy bias, which accounts for the difference between galaxy distributions and matter distributions.
- **MAGNIFICATION.py**: Fiducial values for magnification bias.
- **SURVEY.py**: Fiducial values for survey area and fraction, used for survey configuration.

These values are stored in **JSON** format and are used throughout the cosmological simulations for consistency in input parameters.

## Scripts Overview

1. **ALIGNMENT.py**
   - Stores intrinsic alignment parameters for weak lensing and large-scale structure.
   - It loads cosmology parameters, computes alignment information, and stores it in the `INFO` directory.
   
2. **COSMOLOGY.py**
   - Contains fiducial cosmology parameters such as `Hubble constant (H)`, `dark energy density (Omega_DE)`, `dark matter density (Omega_CDM)`, etc.
   - These parameters are used in cosmological calculations like power spectrum generation and other simulation processes.

3. **DENSITY.py**
   - Stores the fiducial density configuration for different redshift bins (`Y1`, `Y10`), including source and lens densities.
   - This is key to modeling the matter distribution and how it affects gravitational lensing.

4. **GALAXY.py**
   - Computes galaxy bias, which is used to model how galaxies trace the underlying matter distribution.
   - The script computes this for different redshift bins and saves the information for later use.

5. **MAGNIFICATION.py**
   - Stores the fiducial values for magnification bias, a crucial aspect of weak lensing that affects galaxy counts and their distribution.

6. **SURVEY.py**
   - Computes fiducial values related to survey areas and their corresponding sky fractions, which are important for simulating the observed universe from different survey configurations.

## Shell Scripts for Execution

In addition to Python scripts, shell scripts are provided for job submission on the cluster to calculate these parameters efficiently using the SLURM scheduler. Each shell script handles the execution of the corresponding Python script in parallel with the necessary computational resources:

1. **ALIGNMENT.sh**
   - Submits a job to the cluster to run `ALIGNMENT.py` and calculate the fiducial values for alignment parameters.
   
   Example command inside the script:
   ```bash
   python -u "${BASE_PATH}INFO/ALIGNMENT.py" --folder=$BASE_FOLDER
