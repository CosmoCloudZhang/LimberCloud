# LimberCloud repository instructions

- Preserve the complete CCL, Numba, and JAX experiment matrix.
- Preserve `CPU`, `GPU`, `Y1`, `Y10`, `NN`, `NS`, `SN`, `SS`, `EE`, `TE`, and `TT` as scientific labels.
- Use `Single`, `Double`, and `Triple` as user-facing configuration values and lowercase names for source files.
- Use `Title_Case_With_Underscores` for Jupyter notebook filenames while preserving scientific labels, for example `EE_Error_Analysis.ipynb` and `Coefficient_B01_Validation.ipynb`.
- Keep descriptive notebook directories lowercase; retain uppercase scientific subdirectories such as `NN`, `SS`, `Y1`, and `Y10`.
- Treat the legacy NERSC `DATA`, `INFO`, `PLOT`, `COVARIANCE`, `PYTHON`, `JAX`, and `LOG` paths as compatibility contracts.
- Use `limbercloud.io.ProjectPaths` for runtime paths. Do not add new hard-coded `/pscratch` or `/global/cfs` paths to Python code or notebooks.
- Do not rename JSON keys during path, filename, or package refactors.
- Do not reformat analytic projection formulas in the same change as logic or path changes.
- Keep file moves, formatting changes, and scientific changes separate where practical.
- Preserve existing CLI arguments unless a compatibility alias is provided.
- Run `make test`, Python syntax checks, and `bash -n` after relevant changes.
- Validate CPU and GPU production behavior on Perlmutter before changing the default runtime layout.
- Compile and visually inspect the manuscript after LaTeX or figure changes.
- Track publication figure PDFs under `manuscript/figures`; do not track `manuscript/main.pdf` or LaTeX auxiliary files.
