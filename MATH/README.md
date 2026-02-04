# LimberCloud Analytical Coefficient Library

This folder contains symbolic and semi-analytic coefficient tables used in the line-of-sight projection kernels for angular power spectra. The coefficients are stored as plain-text expressions (`Coefficients-B*.txt`) and are mirrored in notebooks (`.ipynb`, `.nb`) for derivations and validation.

The coefficients are organized by kernel type:

- **NN**: number-counts × number-counts
- **NS**: number-counts × shear
- **SS**: shear × shear

These expressions appear in the JAX projection kernels and related numerical pipelines, where they are evaluated on comoving-distance and multipole grids.

## Contents by Subfolder

### `NN/`

- Coefficient sets `B1`–`B3` in `Coefficients-B*.txt`
- Jupyter notebooks (`.ipynb`) and Mathematica notebooks (`.nb`) with derivations
- `MATH.py`: formats `Coefficients-B1.txt` into NumPy-ready syntax (e.g., `Log` → `numpy.log`, `^` → `**`)

### `NS/`

- Coefficient sets `B1`–`B8` in `Coefficients-B*.txt`
- Derivation notebooks (`.ipynb`, `.nb`)
- `MATH.py`: same formatting utility for the `NS` coefficient files

### `SS/`

- Coefficient sets `B1`–`B10` in `Coefficients-B*.txt`
- Derivation notebooks (`.ipynb`, `.nb`)
- `MATH.py`: same formatting utility for the `SS` coefficient files

## File Conventions

The `Coefficients-B*.txt` files encode analytic expressions for integral kernels (e.g., `I1`, `J1`) in terms of compact variables such as `a`, `p`, and `z`. These expressions are intended to be imported and evaluated by downstream code rather than executed as standalone scripts.

## Usage Notes

If you update the symbolic expressions, run the corresponding `MATH.py` to normalize the syntax before integrating the coefficients into numerical code paths.
