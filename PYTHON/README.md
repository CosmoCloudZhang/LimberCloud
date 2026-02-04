# LimberCloud Python Analysis Suite

This folder contains the Python notebooks and modules used to compute and validate angular power spectra, kernels, and error models for weak-lensing and galaxy-clustering analyses. The notebooks provide interactive workflows for Y1/Y10 configurations, while the `PROJECTION` package supplies high-performance projection operators implemented with `numba`.

## Contents and Scientific Role

- `POWER/POWER.ipynb`: interactive computation and visualization of power spectra.
- `KERNEL/Y1|Y10/KAPPA.ipynb`, `KERNEL/Y1|Y10/PHI.ipynb`: kernel construction for lensing and number-count observables.
- `CELL/Y1|Y10/EE.ipynb`, `CELL/Y1|Y10/TE.ipynb`, `CELL/Y1|Y10/TT.ipynb`: tomographic angular power spectra notebooks for the Y1 and Y10 setups.
- `ERROR/Y1|Y10/EE.ipynb`, `ERROR/Y1|Y10/TE.ipynb`, `ERROR/Y1|Y10/TT.ipynb`: uncertainty and error modeling for the corresponding spectra.
- `CPU/Y1|Y10/SINGLE.ipynb`, `CPU/Y1|Y10/DOUBLE.ipynb`, `CPU/Y1|Y10/TRIPLE.ipynb`: performance and scaling studies for different numerical integration strategies.
- `PROJECTION/`: optimized projection kernels for angular spectra calculations.
  - `SS.py`, `SN.py`, `NS.py`, `NN.py`: `numba`-accelerated coefficient and spectrum builders for different field combinations (shear–shear, shear–number, number–shear, number–number).

## Dependencies

- Python 3.x
- Jupyter (for notebooks)
- `numpy`, `scipy`, `numba`
- `pyccl` (cosmology and power spectrum backend)
- `astropy` (tables/units used in some workflows)

### Installation

```bash
pip install pyccl numpy scipy numba astropy
