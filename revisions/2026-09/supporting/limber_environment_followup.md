# Environment, MPI and HDF5 follow-up

Decision record checked on 2026-09-19. This supplements the final code plan and remote prompts; it does not report an installed or tested replacement environment. No NERSC environment, module profile or production dataset was changed during this review.

## Agreed scope

- Use one project environment per machine for scripts, editor and Jupyter. Select it through the checkout-local `.venv` link; a Conda environment remains the underlying installation. Keep the working CosmoConda intact while creating and validating a separate `limbercloud` environment.
- Include `mpi4py` for future task-based work and `h5py` for the spectra archive. Exclude CosmoSIS. Installing MPI support does not introduce multiple cosmology tasks into this paper's benchmark: retain one SLURM task and sequential samples, with backend-internal threading/JAX execution.
- Prefer a validated MPI-enabled h5py build on NERSC when compatible with the common environment. Ordinary serial h5py is sufficient for the current single-writer archive and the portable local environment. Parallel h5py capability is not a benchmark acceptance requirement.
- Make the spectra outputs HDF5 (`.h5`), with sample IDs, cosmologies, axes, spectra, timing stages and provenance described by the main plan. The small immutable shared `Cosmologies.npz` input table may remain NPZ. Update planned spectra readers, filenames and manifests together.

## What the repository currently establishes

`environment.yml` names CosmoConda and contains Python, pip, setuptools, NumPy, SciPy, Numba, Astropy, Matplotlib, PyCCL, ipykernel and Ruff. Its unconditional `jax[cuda12]` entry is a NERSC-oriented choice, not a portable laptop recipe. It currently has neither mpi4py nor h5py nor CosmoSIS.

The calculations select CCL's `boltzmann_camb`; declare CAMB explicitly even if a package manager currently supplies it transitively. The Python derivation notebooks numerically validate formulas; the symbolic originals are Mathematica notebooks, so SymPy is not implied. Current plotting notebooks use external LaTeX. Audit the actual OneCovariance revision and its dependencies remotely before asserting complete manuscript reproducibility.

`scripts/jupyter/launch_kernel.sh` runs `.venv/bin/python`, whereas NERSC launchers select `LIMBERCLOUD_CONDA_ENV`. Those selectors can disagree. `LIMBERCLOUD_ENV_FILE` instead chooses a dotenv configuration file; it does not select Python. The loader contains Bash namerefs, which need replacing or a clearly tested portable alternative for the default macOS shell environment. Existing module profiles already load Conda, GNU, Cray MPICH and parallel HDF5; loading these modules does not establish which libraries a Python extension actually uses.

## Configuration and interpreter migration

1. Standardize the checkout variable as `PROJECT_ROOT`, preserving distinct `RUNTIME_ROOT` semantics. Derive the code root from its scripts; retain copied-SLURM-script resolution through `SLURM_SUBMIT_DIR`. Do not discover the nested manuscript repository by accident.
2. Read the fixed `${PROJECT_ROOT}/.env` through the existing non-executing configuration semantics. Rename the helper to `load_config.sh` and migrate all tracked callers/tests/docs together. Preserve deliberate environment-over-dotenv precedence and reject malformed configuration.
3. Remove `LIMBERCLOUD_ENV_FILE`, `LIMBERCLOUD_CONDA_ENV`, the redundant repository-root override and confirmed obsolete aliases after inventorying remote launch usage. Replace separate environment selection with the resolved `.venv` target; activate that Conda prefix where activation hooks are needed and invoke its Python explicitly. Do not blindly replace a user's existing file, directory or link.
4. Keep `LIMBERCLOUD_RUNTIME_ROOT` as the necessary external location. Keep OneCovariance and TeX overrides only for the workflows that need them; optional `.env.example` entries should be commented. MPI/HDF5 compiler flags are installer details, not additional everyday LimberCloud configuration variables.
5. Document a short existing-environment path, a separate creation path, one kernel-registration command and a diagnostic command showing the executable, prefix, LimberCloud import path and versions. An installed environment alone does not diagnose the user's earlier notebook failure.
6. NERSC notebook startup must receive the same required compiled-library/module setup as the selected environment. Test kernel startup on its actual execution host; a valid kernelspec path alone is insufficient. Keep portable local startup independent of NERSC commands.

## Recipes and package declarations

Use a shared dependency specification with a portable CPU variant and a NERSC CUDA variant, then resolve exact platform-specific records after validation. Include the scientific packages above, explicit CAMB, mpi4py and h5py. JupyterLab remains an optional interface; ipykernel supplies notebook execution. Check package metadata against actual imports and supported Python versions; keep MPI-specific imports optional for non-MPI library use, even though the complete research environment includes mpi4py.

The portable MPI installation must supply a compatible MPI runtime. On NERSC, leave mpi4py and the selected h5py build to the site-specific installation step rather than allowing a generic environment solve to select an unrelated MPI stack. Retain the same scientific versions across CPU/GPU profiles where possible, with explicit JAX device selection at runtime. Record build strings, channels, pip artifacts, compiler/module versions and CUDA/driver/JAX devices; a ranged YAML or `pip freeze` alone cannot recreate the site libraries.

## NERSC-specific installation decision

NERSC recommends building mpi4py against its GNU/Cray MPICH stack using the Cray `cc` wrapper; ordinary generic installations can select an incompatible MPI runtime. MPI programs run through `srun` on allocated compute nodes. Its supported alternatives include cloning `nersc-mpi4py`/`nersc-h5py`, or an external-MPICH Conda package with `cray-mpich-abi`; the ABI alternative requires compatibility checks. Prefer the source-build route for this project's new custom environment. See [NERSC Parallel Python](https://docs.nersc.gov/development/languages/python/parallel-python/).

For an already activated **new** environment, with `PrgEnv-gnu`, `cray-mpich`, Conda and `cray-hdf5-parallel` available, adapt these installation commands to the validated version pins:

```bash
MPI4PY_BUILD_MPICC="cc -shared" python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py
HDF5_MPI=ON CC=cc python -m pip install -v --no-cache-dir --no-binary=h5py --no-build-isolation --no-deps h5py
```

The NERSC recipe uses the short `MPICC` alias and includes `--force-reinstall`; use that option only for a deliberate rebuild in the new candidate environment, never as an automatic repair to the working CosmoConda. The full `MPI4PY_BUILD_MPICC` name is upstream's preferred automation spelling. Disabling pip's cache avoids reusing an extension built against another MPI installation. See [mpi4py installation](https://mpi4py.readthedocs.io/en/stable/install.html).

Before the h5py build, install the selected release's build prerequisites, including NumPy, Cython, Python `pkgconfig`, setuptools and any required build tools; `--no-build-isolation --no-deps` will not supply them. Confirm HDF5 discovery targets the module's parallel library, specifying `HDF5_DIR` only if needed. Validate the Python/NumPy/mpi4py/HDF5 combination and rebuild when its required binary interface changes. See [h5py installation](https://docs.h5py.org/en/stable/build.html) and [NERSC HDF5 modules](https://docs.nersc.gov/development/libraries/hdf5/).

## Storage behavior and the CFS boundary

For the current campaign, use ordinary h5py file access with exactly one writer per run. Do not select the `mpio` driver or add explicit MPI initialization just because these libraries are installed. Check any import-time MPI dependencies of the selected parallel build in the real notebook context. Parallel access to one output file is separate future work: it requires MPI-enabled HDF5 and h5py, coordinated ranks and collective metadata operations. See [h5py parallel HDF5](https://docs.h5py.org/en/stable/mpi.html).

NERSC currently instructs disabling HDF5 file locking outside `$SCRATCH` using `HDF5_USE_FILE_LOCKING=FALSE`. Scope this to the validated NERSC/CFS launch profile before importing h5py, record it in provenance and verify behavior on the actual CFS destination. See [NERSC's file-locking guidance](https://docs.nersc.gov/development/languages/python/parallel-python/#parallel-io-with-h5py).

Our implementation must separately enforce exclusive run ownership, reject simultaneous writers/resumers, and keep readers on completed immutable checkpoints. Disabling HDF5 locks does not make concurrent access safe. Write bounded checkpoints outside compute timers, close/validate them, atomically publish on the destination filesystem, and publish the manifest last. An append plus flush is not a transaction and must not be described as one. CFS remains the durable destination; any PSCRATCH staging needs an explicit validated transfer/completion step.

## Remote evidence required before adopting the environment

1. Capture both Git revisions and dirty state; existing Conda package/build records; `.venv` target; kernel interpreter; current modules; compiler paths; actual CFS input/output locations; and OneCovariance revision/configuration. Avoid including secrets in reports.
2. Create the separate candidate environment and record installation choices. Confirm the editor, kernel launcher, command-line scripts and SLURM launcher resolve its same Python prefix. Verify package imports and a tiny CAMB-backed CCL calculation, then representative Numba and JAX calculations.
3. In a small allocation, run an MPI rank/count and reduction test with `srun`; record `MPI.Get_library_version()`. Add a minimal cross-node check before claiming cross-node MPI support. These are installation checks, not changes to benchmark task counts.
4. Record `h5py.version.info` and `h5py.get_config().mpi`; inspect loaded MPI/HDF5 libraries, not just package names. Check serial HDF5 round trips, metadata/dtype preservation and checkpoint/restart behavior on the actual CFS path and with the notebook reader.
5. If the parallel h5py variant is adopted, run a separate tiny collective-file test under `srun` and verify every rank's values after closing. If only serial capability is validated, state that explicitly and retain the single-writer workflow.
6. Run tiny allocated CPU and GPU tests with explicit device/thread settings, synchronization and recorded hardware; test the actual NERSC notebook startup context separately. Preserve previous environments and results until these checks pass.
7. Save a concise remote validation report with commands, outcomes, package/platform records, external module versions and unresolved limitations. Only then choose exact lock records and point `.venv` to the accepted environment. Do not claim the local review has performed these tests.

All online sources above were checked on 2026-09-19. Recheck site instructions when implementing because NERSC's module and package versions can change.
