# LimberCloud-Python Code for Cosmological Analysis

This folder contains Python scripts and Jupyter notebooks used for cosmological analysis, particularly focused on the computation and modeling of angular power spectra, weak gravitational lensing, and large-scale structure. The files here calculate key parameters such as kernel functions, power spectra, error models, and projections that are vital for cosmological surveys.

## Folder Structure

1. **POWER**:
   - **`POWER.ipynb`**: A Jupyter notebook dedicated to the computation of power spectra. It includes steps for the calculation and visualization of angular power spectra.
   - **`EE.ipynb`**, **`TE.ipynb`**, **`TT.ipynb`**: These notebooks are dedicated to the specific angular power spectra for different modes:
     - **EE**: E-mode angular power spectrum, related to the polarization of the cosmic microwave background.
     - **TE**: Temperature-Emissivity cross-correlation angular power spectrum.
     - **TT**: Temperature-temperature angular power spectrum.

2. **CELL**:
   - **`Y1`**, **`Y10`**: These directories contain specific configurations for different datasets (e.g., Y1 and Y10 refer to different redshift bin configurations or survey datasets).
   - **`EE.ipynb`**, **`TE.ipynb`**, **`TT.ipynb`**: Corresponding notebooks for the computation of **EE**, **TE**, and **TT** power spectra for the Y1 and Y10 datasets.

3. **ERROR**:
   - **`Y1`**, **`Y10`**: These directories contain error modeling information for the corresponding datasets. They are used for evaluating the uncertainties and biases in the cosmological model calculations.
   - Error models for different angular power spectra computations are used to quantify how uncertainties in cosmological parameters affect the final results.

4. **KERNEL**:
   - **`Y1`**, **`Y10`**: Directories related to the kernel functions used in weak lensing and power spectrum analysis for the different datasets. These files help compute the kernels necessary for transforming different data fields.

5. **PROJECTION**:
   - **`NN.py`**, **`NS.py`**, **`SN.py`**, **`SS.py`**, **`__init__.py`**: These Python files are used for different projection techniques used in weak lensing and large-scale structure analysis. They likely contain the projection models and the calculations needed for transforming the data from one form to another.

## Purpose

The code in this folder is part of an analytical framework designed for the following purposes:
- **Computation of angular power spectra**: Using weak gravitational lensing and large-scale structure data.
- **Error modeling**: To evaluate the uncertainties in the cosmological calculations.
- **Projection techniques**: To model various transformations and projections required in cosmology.

## Dependencies

- **Python 3.x**: Ensure you are using Python 3.x for compatibility with the scripts.
- **Jupyter**: For running the `POWER.ipynb` notebook.
- **pyccl**: Cosmology library for power spectrum calculations.
- **numpy**: For numerical operations.
- **scipy**: For scientific computations, including integration and optimization.
- **astropy**: For handling cosmological units and data structures.

### Installation of dependencies:

```bash
pip install pyccl numpy scipy astropy
