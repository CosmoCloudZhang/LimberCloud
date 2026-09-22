# Phase 1 — Foundation and endpoint corrections

Implementation report for `revisions/2026-09/prompts/PHASE_1_FOUNDATION_CORRECTIONS.md`.
Written after the source changes it describes; this report is not part of the
compute-source manifest and does not hash itself.

## Current contract and acceptance addendum — 22 September 2026

The [working-tree contract review](2026-09-22_PHASE1_CONTRACT_REVIEW.md) supersedes
the current-state and acceptance claims in sections 1–10 below. Those sections
are retained as the historical implementation record: their file list, test
counts, source fingerprint and removed-test results describe an earlier tree.

The current contract is:

- Names are `Time_Triple.txt`, `Time_Triple_COSMOLOGY.txt`,
  `Spectra_Single_EE.h5` and `Manifest_Triple.json`. Only NUMERIC adds an
  order, for example `Time_Triple_LINEAR_COSMOLOGY.txt`. Older `*_128*`
  products remain separate; there is no allocation token or migration.
- Spectra drivers accept `--tag`, `--label`, `--folder`, `--sample-count`
  and `--sample-table`. CPU resources remain in Slurm and the thread
  environment. There is no `--number`, `--fiducial-only`,
  `--include-fiducial`, `--run-id`, `--run-config`, `--mode` or `--resume`.
- Each timing launch evaluates table sample 0 followed by N additional rows.
  N defaults to zero; the table is required even then. The family/survey
  timing files are replaced on rerun (JAX includes its device directory).
  For N=0 the files contain fiducial timing; for N>0 they contain cumulative
  sampled timing excluding the fiducial. These meanings need explicit metadata
  in Phase 2. The sample generator retains `--run-id` for
  `results/spectra/inputs/<run_id>/`.
- The performance scripts read CCL, Numba and JAX only, expose neither
  `--number` nor `--interpolation`, and save `benchmark_{label}.pdf`.
  The source label `CCL (total)` is currently hidden by absent stage legends;
  actual count handling and visible qualification need repair.
- The dedicated endpoint test, diagnostic script and full-assembly oracle
  helpers are absent. The retained reference functions are the NUMERIC
  contract, hats/power reconstruction, `nn_interval_quadrature`, terminal
  source integrals and `lensing_efficiency`. Historical results below are
  not executable regression coverage of the current tree.

**Phase 1 is not fully accepted.** The current review reproduced cancellation
in the terminal source formulas on narrow intervals and the wrong observer
power law for a one-interval grid. The ordinary NS/SN/SS zero-power limitation
already listed below also remains. Nuisance hashes are checked for presence
but not bound to sample 0 and the selected solver. Thus the earlier statement
that only external evidence remains, and checklist R09's former pass, are
superseded. R05/R08/R09 now distinguish implementation from current acceptance.
The 48-point terminal rule is a numerical approximation whose accuracy and
cost must be measured; the assertion of stability for any spacing is withdrawn.

The current review ran `make check`: 72 fast tests, lint, shell syntax and
40 notebook parse checks passed. Its small compiled Numba diagnostic is
described separately in the linked review; it is not an allocated science or
GPU acceptance run. Kernel/HDF5, real OneCovariance, GPU and macOS portability
evidence remains outstanding. No science implementation or runtime product
was changed by this review.

The later author-approved [expanded Phase 1 plan](../prompts/PHASE_1_FOUNDATION_CORRECTIONS.md)
now owns the bounded closeout, including 21 faithful Jupyter derivation editions,
four new boundary cases, independent review before terminal-module removal,
provenance/component checks and separate Fiducial/Cosmology timing products.
Its planned benchmark legend is `CCL`, with the repeated end-to-end reference
explained in the caption. The filenames and source behavior described above
remain the reviewed implementation state, not evidence that these planned
changes have run. Phase 2 consumes the accepted foundations for the shared
evaluator, NUMERIC and HDF5 transactions. It retains the simplified timing CLI;
resolved configuration, diagnostics and transaction recovery belong to internal
APIs/recovery tooling. The old section 10 instruction to restore removed driver
flags is superseded. The established notebook n=100 numerical integration is
distinct from the temporary terminal-source formulas; the latter's cancellation
does not demonstrate a failure of the former.

## Historical implementation record

## 1. Baseline and repository boundary

| Item | Value |
|---|---|
| Branch | `main`, tracking `origin/main` |
| Entry HEAD | `72bb8af8c486a579e08405e9b68308c7265147ef` ("Update revision plans and restore manuscript submodule") |
| Exit HEAD | `72bb8af8c486a579e08405e9b68308c7265147ef` — **no commit was made** |
| Entry working tree | clean (`git status --porcelain` empty) |
| Exit working tree | 99 modified tracked files, 12 untracked source/test files plus this report, `+3456 / -1007` lines |
| Code root | `/pscratch/sd/y/yhzhang/LimberCloud` |
| Runtime root | `/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/LimberCloud` |
| Interpreter | `.venv -> /global/homes/y/yhzhang/.conda/envs/limbercloud`, CPython 3.12.14 |
| Packages | numpy 2.2.6, scipy 1.13.1, pyccl 3.2.1, camb 1.6.5, numba 0.63.1, jax/jaxlib 0.9.2, h5py 3.16.0 (MPI-enabled) |
| Plan read | `CODE_REVISION_PLAN.md`, `COMPLETION_CHECKLIST.md`, `reports/2026-09-22_{IMPLEMENTATION_REVIEW,PHASE0_AUDIT,PHASE1_AUDIT,COVARIANCE_AUDIT}.md`, `MANUSCRIPT_REVISION_PLAN.md`, `reports/C0{0,1}_*`, `supporting/limber_numeric_endpoint_review.md`, `supporting/AUTHOR_DECISIONS_2026-09-22.md` |

### Paper gitlink

**Already repaired before this phase; nothing was changed here.** The deletion
recorded in the Phase-0 audit belonged to `13a3c2d`; commit `72bb8af` restored
it. Both the tree and the index now carry the intended pin:

```text
git ls-tree HEAD manuscript   -> 160000 commit 90d12f4f3e574a67c25944d27d7ded553e09402b	manuscript
git ls-files --stage manuscript -> 160000 90d12f4f3e574a67c25944d27d7ded553e09402b 0	manuscript
```

`.gitmodules` still declares `path = manuscript`, `url = ../LimberCloudPaper.git`.
The local `manuscript/` directory is empty and uninitialised; nothing under it
was staged, `git -C manuscript` was never run, and the remote paper was not
initialised. Code-only checks (`make check`, `make check-all`) pass with the
paper absent.

### Review packet

No commit separates the stages, so entry and exit share HEAD. The exit source
state is the working tree at `72bb8af` plus the changes listed in section 8.
The deterministic compute-source manifest covers 62 `.py` files under
`src/limbercloud/` and `experiments/spectra/`, tracked and untracked alike, and
fingerprints to `81cb15ee98836cd12bd205e49469be3cbe18f8fa6db9b2ec3b572d485d24b282`.
Reports, documentation, notebooks, `.env`, environment binaries and runtime
arrays are outside that scope by construction, so this report cannot change it.
A later comment-only alignment of the JAX NS/SN terminal-row/column annotations
changed the digest relative to an earlier working-tree snapshot; the science is
the same.

## 2. Phase-0 execution and portability repairs

| Gap | Repair | Evidence |
|---|---|---|
| Launchers ignored user arguments | All 24 spectra launchers, 4 `Run_All.sh`, 2 benchmark `figure.sh`, 6 configuration generators and 2 covariance `matrix.sh` now end their Python invocation with `"$@"` | `tests/test_launcher_smoke.py::test_spectra_launchers_forward_arguments_with_quoting_intact` asserts the recorded argv of a stub `srun`/`python` for all 24 drivers, using a checkout under `.../work space/` and `--run-id "pilot run"` |
| `Run_All` chains dropped the selection | `sbatch ... "$@"` | `test_run_all_launchers_forward_the_same_selection_to_six_jobs` requires 6 `sbatch` calls each carrying `--sample-count=5` and `--run-id "pilot run"` |
| `${LABEL,,}` is Bash 4 only | Each launcher now sets an explicit `SCRIPT="single"` token beside `LABEL="Single"` | The same test asserts the resolved `.../single.py` path; `make shell` parses all 48 scripts |
| `BASH_SOURCE[0]` lost under Bash 3.2 | `scripts/nersc/modules/{cpu,gpu}.sh` and `scripts/jupyter/launch_kernel.sh` capture the source path into a variable **before** any command substitution | `tests/test_environment_contracts.py::ModuleProfileTests` asserts the exact module order `cpu/gpu, conda, cray-mpich, PrgEnv-gnu, cray-hdf5-parallel` |
| Path fixtures compared `/var` with `/private/var` | `tests/test_environment_contracts.py` resolves the temporary root in `setUp` before comparing with the loader's `pwd -P` output | Comment records why |
| Launcher tests needed the real `.venv` | `tests/test_launcher_smoke.py` builds a synthetic checkout: copied `experiments/` and `scripts/`, a `src/limbercloud/` marker, a stub `.venv/bin/python`, a stub `.env`, stub `conda/module/python/srun/sbatch/mkdir/cc` | 13 tests pass with the user's environment untouched; `test_missing_real_environment_still_reports_a_clear_error` deletes the stub prefix and requires the real message `missing .venv link` |
| No-op default could truncate timing files | `require_effective_work` raises `NoWorkRequested` for a zero selection and every driver returns before opening data or creating output | `RunGuardTests::test_a_default_launch_requests_no_work_and_writes_nothing`, and `test_existing_timing_products_are_never_overwritten` seeds `Time_Single_128.txt` and asserts byte-identical content after a refused rerun |
| `--fiducial-only` silently did nothing | Refused with an explicit message naming the shared evaluator as the owner | `test_fiducial_requests_are_refused_rather_than_reported_complete` |
| `--run-config` / `--mode` accepted and ignored | `require_supported_controls` rejects `--run-config`, `--mode validation` and `--resume` | `test_unimplemented_controls_fail_instead_of_being_ignored` |
| New runs could write into historical roots | A positive `--sample-count` requires `--sample-table` and `--run-id`; `prepare_result_directory` refuses a directory that already holds `Time_*.txt` | `test_a_real_run_needs_a_table_and_an_isolated_run_id` |
| Kernel lacked modules and Conda hooks | `launch_kernel.sh` sources the CPU or GPU module profile and `activate_venv.sh` on `NERSC_HOST`, selected by `LIMBERCLOUD_KERNEL_PROFILE`; portable macOS behaviour is unchanged. `--probe` stays identity-only and login-safe; the new `--science` performs a real HDF5 round trip under the runtime root and a CCL background evaluation | Code and `documents/nersc.md`. **`--science` has not been executed on an allocation**; see section 7 |
| OneCovariance ran in an unverified interpreter | `scripts/nersc/check_onecovariance.py` parses the external `covariance.py` with `ast` and resolves every top-level import in the interpreter that will run it; both `matrix.sh` files call it before the preparation step and honour `LIMBERCLOUD_ONECOVARIANCE_PYTHON` | `OneCovarianceStartupTests` (3 tests) covers a resolvable script, a missing dependency and an absent executable. **The real OneCovariance checkout was not started**; see section 7 |
| Installer accepted base/unrelated prefixes | `install_mpi_h5py.sh` resolves the `.venv` target, compares `CONDA_PREFIX` and the interpreter's own `sys.prefix` against it, refuses `base` and `CosmoConda`, and force-reinstalls `mpi4py` from source so an existing generic wheel is never accepted | `InstallerGuardTests` (3 tests) with `LIMBERCLOUD_INSTALL_DRY_RUN=1`; **no package was rebuilt and the validated environment is unchanged** |
| `make check` no longer login-safe | Split into `test-fast` (NumPy/SciPy only) and `test-science` (h5py and compiled backends); `check` runs the fast gate, `check-all` runs both | `Makefile`, `README.md`, `documents/nersc.md` |
| Bare `python3` in `make` selected `/usr/bin/python3` 3.6 | `Makefile` uses `.venv/bin/python3` when that link exists; documented in `README.md` and `documents/nersc.md`. Override with `make PYTHON=...` only for a deliberate other prefix | `make check` now runs the selected 3.12 interpreter without activating Conda first. OneCovariance unit tests call `sys.executable` rather than Make's unexported `PYTHON` variable, which previously spawned 3.6 and failed on `from __future__ import annotations` |
| `py.typed` declared but absent | Removed the package-data entry rather than adding a marker that was never intended | `pyproject.toml` |

## 3. eta_IA and fiducial provenance

`eta_IA = 0.0` is adopted as the LSST DESC SRD fiducial. The omitted-argument
branch that produced 0.5 and recorded the choice as unresolved is gone.

- `scripts/generate_config/intrinsic_alignment.py` defaults to 0.0 and records
  `eta_decision="adopted"`. Any other `--eta-ia` value is written as
  `"diagnostic"` and refused by accepted-campaign readers.
- `validation/contract.py` replaces `EtaIADecision`'s unresolved machinery with
  `adopted()` / `diagnostic()`, and adds `require_accepted_eta` for loaded
  artifacts. `A_IA = 0.5`, `z_pivot = 0.5`, `C1 = 5e-14/h**2`, the comoving
  present-day matter density and `D(0) = 1` are unchanged and are separate named
  constants, so none of them moved with the slope.
- `validation/nuisance.py` is the single validating loader. All 24 drivers use
  it instead of reading `alignment_info['A']` directly.

### Regenerated artifacts and independent validation

The three nuisance tables under the runtime `config/` were regenerated. The
previous files were preserved first as `*.historical-2026-09-22.json` and keep
their own identity; they carried **no** eta metadata at all, so the new readers
refuse them.

| Check | Result |
|---|---|
| Generated metadata | `eta_pivot = 0.0`, `eta_decision = adopted`, `a_pivot = 0.5`, `z_pivot = 0.5`, `C1 = 5e-14/h**2`, redshift axis `linspace(0, 3.5, 351)`, effective law, fixed-tabulated policy, package versions |
| Independent law | An independently constructed `-C1 rho_m(0)/D(z) * A_IA * ((1+z)/(1+z_pivot))**0` agrees to a maximum relative difference of **0.0** |
| `D(0)` | `1.0` |
| Sign | Every sample negative (signed NLA retained) |
| Historical ratio | `A_old / A_new` matches `((1+z)/1.5)**0.5` to `2.2e-16`, confirming the old array really was `eta = 0.5` |
| Size of the change | up to **73%** at `z = 3.5`. Updating metadata alone would not have reproduced it |

### Effective fiducial constructor

`validation/cosmology.py` centralises the effective primary/derived cosmology
and solver specification: `transfer_function='boltzmann_camb'`,
`matter_power_spectrum='halofit'`, `mass_split='single'`, CAMB `kmax=100`,
`lmax=5000`, `mead2020_feedback`, `HMCode_logT_AGN=7.8`, plus explicit
radiation, temperature and neutrino conventions. `OMEGA_GAMMA` became a fixed
nonzero primary parameter of the campaign table, so the photon density is
passed explicitly for every sample and is not sampled; that is the declared
radiation convention, recorded in the table manifest.

The audited divergence is closed rather than documented: `validation/samples.
ccl_cosmology_kwargs` now delegates to the same constructor the generators use,
so `Omega_g` is supplied and `kmax` is 100 on both sides. Measured at sample 0:

| Quantity | Generator vs evaluator |
|---|---|
| `D(z)` at z = 0, 0.5, 1, 2, 3.5 | max abs diff `0.0` |
| `H(z)/H0` | max abs diff `0.0` |
| `chi(z)` | max abs diff `0.0` Mpc |
| `rho_m(0)` comoving | abs diff `0.0` |

Fingerprints: solver `b1ff9063973247fb…`, fiducial input hash
`e295e7a77b24ec97…`, generating model `2d51bf72d2bb286f…`. The IA and
galaxy-bias tables carry the same generating-model fingerprint, which is how a
future divergence would be caught.

Galaxy bias gained its redshift axis, amplitudes, policy, input hash, model and
solver fingerprints and package versions. Magnification bias declares
`_quantity = magnification_slope_s` and `_response = q = 5s - 2`. CCL still
receives the slope `s` (`ccl_magnification_bias`) and the analytical weighting
still receives `q = 5s - 2` (`analytical_magnification_response`), tested for
`s = 0.4` giving exactly zero and for nonuniform slopes weighting each lens bin
separately. The nuisance tables remain fixed at the fiducial cosmology while the
background, power and geometric prefactors follow the active sample.

## 4. Radial interpolation restricted to NUMERIC

`validation/method.py` introduces one validated `MethodIdentity(family, device,
interpolation)`. CCL and NUMBA are CPU; JAX must choose CPU or GPU; NUMERIC
requires `linear`/`quadratic`/`cubic` and is CPU. Every other combination
raises `MethodError`.

Repaired together:

- `ProjectPaths.spectrum_results` resolves through `MethodIdentity`.
- `io/artifacts.py`: `spectra_basename`, `timing_basename`,
  `samples_timing_basename` and `manifest_basename` take a `family` keyword and
  accept an order token only for NUMERIC. Non-NUMERIC basenames are byte-for-byte
  unchanged, including stage suffixes and allocation tokens
  (`Time_Triple_128_COSMOLOGY.txt`, `Spectra_Single_4_EE.h5`,
  `Time_Double_64_SAMPLES.h5`, `Manifest_Triple_128.json`).
- `ArtifactIdentity.__post_init__` validates the family/device/order
  combination, the survey and the configuration, and rejects an empty sample
  table hash or estimator fingerprint.
- Both benchmark readers take a `family` argument per call and pass the order
  only to NUMERIC paths and names. `--interpolation` now adds an optional
  NUMERIC curve instead of renaming CCL, NUMBA and JAX files. The figure
  filename gained the run identity so two runs cannot overwrite each other, and
  the repeated CCL total in the stage panels is labelled `CCL (total)`.

No CCL, NUMBA or JAX driver gained an interpolation loop; the drivers contain no
`--interpolation` string at all, asserted by
`test_known_driver_defects_are_wired_to_the_contract`. Actual NUMERIC
integration remains Phase 2; only the module and family architecture is fixed.

Tests: `tests/test_angular_contract.py::MethodIdentityTests`,
`tests/test_project_paths.py::test_a_radial_order_reaches_numeric_paths_only`,
`tests/test_science_artifacts.py::ArtifactNamingTests`.

## 5. The exact angular contract

`validation/estimator.py` now defines one current contract.

```text
ell   = numpy.geomspace(20.0, 2000.0, 21, dtype=float64)
x     = numpy.log(ell)
sp    = scipy.interpolate.CubicSpline(x, ell * cl, axis=-1, bc_type="natural")
band[b] = sp.integrate(x[b], x[b+1]) / (ell[b+1] - ell[b])
```

`AngularContract` carries the 21 raw nodes, the 21 edges, the 20 display
centres, the transform, the natural boundary condition, the `delta_ell`
normalization, dtype, axis order and a versioned fingerprint. It rejects
nonfinite, nonpositive or nonincreasing multipoles, shape and axis mismatches
and nonfinite spectra; signed and exactly zero spectra pass unchanged.
`bandpower_operator_matrix` returns the exact 20×21 weights and reproduces
direct integration for signed vectors and batches.

| Identity | Digest |
|---|---|
| Operator version | `limbercloud.angular.natural-spline.v1` |
| Angular contract | `e8c9cad492f5233ff39aa51428d22ab2ec5f60de61d2db38c1f3046b91c7cf1e` |
| Raw 21-node estimator | `a49e81ea001f7c16135226f076c88de40d192824f5c9ea7c6f1be98b61818565` |
| Band 20 estimator | `2182a0fd352dec0ec22d9b0c16facff5537e1478cfce332189a6682bbf04d5ab` |

Every fingerprint is computed from the actual float64 coordinate bytes plus the
boundary condition and operator version, so the historical not-a-knot helper
cannot present itself as compatible. The raw and band identities differ, so the
two datasets stay distinct. The 20 geometric centres and the 101-point grid
remain available as explicitly legacy readers
(`historical_ccl_estimator`, `legacy_covariance_input_ell`) and are not the
current default.

CCL's six runners now evaluate the same 21 nodes as the analytical methods:
`canonical_ell_nodes()` replaces `numpy.sqrt(ell_grid[1:] * ell_grid[:-1])`, and
the result arrays are sized `ell_size + 1`. These drivers still publish only
timing text; Phase 2 applies the operator inside the shared evaluator for all
methods.

Measured test evidence:

| Check | Result |
|---|---|
| Notebook formula reproduced term by term | exact (`atol = 0`) |
| Natural distinguished from not-a-knot on a curved signed fixture | relative difference **9.2e-2**, far above the 1e-3 threshold asserted |
| Exact fixture, `ell*C` linear in `log ell` | matches the elementary closed form to `1.7e-16` |
| Independent integration of the same piecewise polynomial coefficients | agrees to `1e-11` |
| Linearity and batch consistency with the 20×21 matrix | `1e-12` |
| Constant `C` sampled at the nodes | worst band error **1.8e-3**, measured rather than renormalised away |

The old test that attached a 21-node fingerprint to three actual `ell` values is
gone: the checkpoint fixture now declares a three-node estimator, and
`ArtifactIdentity` compares the declared coordinates with the stored `ell`
dataset on write and on read (`test_a_forged_estimator_fingerprint_is_caught_
against_the_stored_axis`).

Covariance is untouched. Changing a 101 constant to 21 would not make it
scientifically matched; the physical weighting is the Phase-3 adapter's work.

## 6. NN, NS, SN and SS endpoints

Historical results below predate the current review. The claims of universal
spacing stability and complete observer coverage are superseded by the
reproduced counterexamples in the current-state addendum.

### Sources read

`notebooks/derivation/NN/Coefficient_B0{1,2,3}.nb` and their `.txt` exports,
the NN/NS/SS Python validation notebooks and both analytical backend modules
were read as text. **Mathematica itself was not run.** The observer result
`1/12, 1/12, 1/4` for `P = P1 (chi/chi1)^3` is retained; the withdrawn `1/2`
change was not applied.

### NN

The final-interval `element3` is now stored into `B[N, N]` on every interval in
both backends. The guard `n + 1 < grid_size` in
`numba_backend/nn.py` and the `jnp.where(...)` zero in `jax_backend/nn.py` are
removed. `NN_FINAL_DIAGONAL_POLICY` changed from
`current_implementation_omits_final_diagonal` to
`full_basis_including_final_diagonal`.

Mandatory regression, `chi = [0, 1, 2]`, `P = [0, 1, 1]`, both density vectors
`[0, 0, 1]`, overall factor 1:

| Backend | Value |
|---|---|
| Numba | `0.11370563888010943` |
| JAX | agrees to `1e-13` |
| Exact `3/2 - 2 log 2` | `0.1137056388801094` |

The two other published weightings are reproduced as well: `[0, 1, 1]` gives
`0.75` and `[0, 1, 0]` gives `0.47741127776021886` — the latter unchanged, since
it carries zero terminal density. A separate test zeroes `B[N, N]` and confirms
that a distribution vanishing at the last node is unaffected.

### The terminal lensing cases

The complete-interval terminal expression used by NS/SN element 7 and 8 and by
SS element 4 and 10 assumes the whole node-`N` source hat lies **above** the
evaluation point, which makes the source integral linear in `x`. That holds only
while `x <= chi[N-1]`. On the final interval `[a, b]` the evaluation point sits
inside the source support, so the lower limit is `x` itself:

```text
H_N(x) = integral_x^b [(u-a)/(b-a)] [(u-x)/u] du
       = [(b-x)^2/2 - a*(b-x) + a*x*log(b/x)] / (b-a)
F_N(x) = integral_x^b [(b-u)/(b-a)] [(u-x)/u] du
       = [(b^2-x^2)/2 - b*x*log(b/x)] / (b-a)
```

`F_N` is the matching partial integral of the node `N-1` hat, whose falling half
also lies on `[a, b]`. Both were verified against direct quadrature and against
continuity with the below-interval branch at `x = a`.

Repairs, derived case by case rather than by removing guards:

| Entry | Previous behaviour | Now |
|---|---|---|
| NS `B[N, N-1]`, SN `B[N-1, N]` | skipped by `n + 1 < grid_size` | element 2 stored on every interval; it needs no `chi[n+2]`, and its existing formula is the correct partial-hat case |
| NS `B[N-1, N]`, SN `B[N, N-1]` | element 7 evaluated with the below-interval branch | new terminal kernel, falling density hat against `H_N` |
| NS `B[N, N]`, SN `B[N, N]` | skipped by `n + 1 < grid_size` | new terminal kernel, rising density hat against `H_N` |
| SS `B[N-1, N]`, `B[N, N-1]` | element 4 with the below-interval branch | new terminal kernel, `F_N * H_N` |
| SS `B[N, N]` | element 10 with the below-interval branch | new terminal kernel, `H_N * H_N` |
| NS `B[N-1, N-1]`, SS `B[N-1, N-1]` | element 1 | unchanged; element 1 is already the partial-hat case and matches the oracle |
| Guards protecting `chi[n+2]` (NS 5/6, SS 2/5/6/7) | present | unchanged; they protect an unavailable node, not a missing case |

The ordinary 3 / 8 / 10 element catalogue is unchanged. Two terminal cases were
added for NS and SN and two for SS, giving 3 / 8 + 2 / 10 + 2.

The terminal kernels live in `projection/{numba,jax}_backend/terminal.py`. They
evaluate the derived integrands with a fixed 48-point Gauss–Legendre rule on the
single final interval. Closed forms exist — they were derived symbolically —
but they carry `(b-a)^-4` and `(b-a)^-5` prefactors that cancel against the
logarithm and lose all significance on a narrow terminal interval. The
integrands are analytic on `[a, b]` for `a > 0`, so the fixed rule converges
geometrically and reproduces adaptive quadrature to machine precision while
staying stable for any node spacing. The cost is one fixed rule per coefficient
build, not per node pair. This choice is a deliberate departure from the
closed-form style of the ordinary elements and is flagged for review.

### Independent oracle

`validation/reference.py` gained an oracle that reconstructs the hats, the
declared power law (cubic on an observer interval, linear afterwards), the
linearised `1+z` and the lensing source integrals from their definitions, and
never calls the closed forms under test:

- `hat`, `declared_power`, `one_plus_z_linear`
- `nn_coefficient_oracle`, `nn_spectrum_oracle` — full NN assembly by quadrature
- `terminal_source_integral`, `full_source_integral`, `source_integral_quadrature`
- `lensing_efficiency` (closed form) and `lensing_efficiency_oracle` (quadrature),
  which agree to `1e-11`; the closed form removes the nested-quadrature noise
  that otherwise limits comparisons to `2e-8`

`tests/test_projection_endpoints.py` runs the compiled Numba and JAX kernels
against it on a regular interior grid, an observer-plus-narrow-interval grid, a
one-interval grid and a signed-power grid. Agreement between the two ports alone
is never accepted as evidence.

| Check | Result |
|---|---|
| NN vs oracle, all grids, both backends | within `rtol = 1e-8`, `atol = 1e-11 * max` |
| NS, SN, SS vs oracle including every terminal entry | max relative `2.7e-12` (NS/SN), `2.0e-13` (SS) |
| Defects the tests catch | NS terminal source entry was **3.1%** wrong; NS terminal density row was missing entirely; SS terminal diagonal was **0.6%** wrong |
| NN symmetry | exact (`atol = 0`) |
| NN Gram positive semidefiniteness for nonnegative power | smallest eigenvalue `>= -1e-12 * max`; the previous terminal 2×2 minor had determinant `-6.3e-3` |
| NS vs SN transpose under swapped legs | exact (`atol = 0`) |
| SS symmetry and finiteness on a `1e-7`-wide terminal interval | symmetric to `1e-12`, all finite |
| Terminal kernels with zero power | exactly zero, not `0/0` |

Tolerances follow the oracle's quadrature request (`epsabs = 1e-12`,
`epsrel = 1e-10`) and the cancellation scale of the integrands; they sit above
the oracle's own error and six orders of magnitude below the defects above.

### Two further defects the oracle exposed

1. **`jax_backend/{ns,sn}.py` mixed two leg conventions.** Elements 1, 3 and 4
   used the NS row convention while elements 2, 5, 7 and 8 used the SN one, so
   neither file equalled its Numba counterpart nor that counterpart's transpose.
   This is present at `13a3c2d` and at HEAD; it is not introduced here. Both
   files were realigned entry for entry with their Numba versions. Both backends
   now agree to `1e-13` and NS is the exact transpose of SN. A follow-up pass
   also swapped the remaining “row”/“column” comments so NS names the terminal
   density **row** and SN names the terminal density **column**.
2. **`jax_backend/nn.py::spectra` could not run.** It passed
   `dtype=jnp.float64` to `jnp.einsum`, which JAX 0.9.2 does not accept, so any
   call raised `TypeError`. Removed; float64 comes from `jax_enable_x64`.

### Measured scientific effect

`revisions/2026-09/endpoint_diagnostic.py` evaluates the EE spectrum once at the
fiducial cosmology with the real survey distributions and rebuilds it with the
previous terminal behaviour restored. It writes no artifact.

| Quantity | Y1 | Y10 |
|---|---|---|
| Terminal source weight `psi(z=3.5)` | `4.68e-3` (`1.1e-3` of peak) | `4.18e-2` (`1.2e-2` of peak) |
| Terminal lens weight `psi(z=3.5)` | `0.0` | `0.0` |
| EE raw-node max fractional change | `9.0e-11` | `4.6e-9` |
| EE bandpower max fractional change | `9.1e-11` | `4.6e-9` |
| TT `GG` max fractional change from the NN final diagonal | `0.0` | — |

The honest reading: the corrections are structurally required — they restore the
Gram-matrix property, the terminal basis support and the NS/SN transpose — but
with the current SRD tabulations interpolated onto `linspace(0, 3.5, 351)` the
lens distributions vanish exactly at the last node and the source distributions
are three orders of magnitude below their peak there, so the change to current
Y1 and Y10 spectra is far below any analysis tolerance. It would not be for a
distribution with real support at the grid edge. This is a bounded login-node
diagnostic at one cosmology, not a campaign result.

## 7. Contracts frozen for Phase 2

`validation/run_identity.py` defines the three layers Phase 2 enforces.

- **`SharedScience`** — survey, configuration and probe list, sample-table hash,
  solver fingerprint, per-table nuisance model fingerprints, radial grid
  description, the angular operator, `eta_IA`, the nuisance policy, the endpoint
  policy and the pair orientation. It validates rather than stores: a nonzero
  `eta_IA`, a `per_sample` nuisance policy, the superseded endpoint policy, an
  empty hash, an unknown survey or configuration, or a radial grid missing
  `node_count`/`minimum`/`maximum` all raise.
- **`ProducerWorkload`** — the shared science plus the validated
  family/device/order, the immutable requested sample IDs, the quadrature policy,
  the declared timing boundary, the compute-source manifest fingerprint and the
  numerical dependency signature. `may_resume` requires the full fingerprint;
  `is_comparable_with` requires only the shared science, so different methods
  keep different producer identities while comparing compatible science. A
  changed NumPy version blocks a resume even when the source digest is unchanged.
- **`ExecutionRecord`** — append-only: producer fingerprint, HEAD as provenance,
  scheduler job, host, PID, start time, output paths, checksums and status. None
  of these appear in the workload dictionary, so the fingerprint is acyclic and a
  report or documentation edit cannot break a resume.

`compute_source_manifest` hashes every `.py` under the declared roots, staged,
unstaged and untracked alike, from a plain checkout. It excludes
`revisions/` and `documents/`, tested explicitly.

Related constructor hardening in this phase: `ArtifactIdentity` validates the
method combination and the estimator; `_require_lock` now requires that the held
lock own the exact namespace being written, so a lock on A no longer authorises
a write into B; `assemble_probe` refuses to broadcast components of different
shapes.

`tests/test_run_identity.py` adds 14 tests covering method restrictions,
malformed identities and coordinates, changed-configuration rejection, resume
versus comparison, and manifest scope. Phase 2 still owns sample transactions
and consolidation.

## 8. Evidence, commands and file map

All commands ran on the Perlmutter **login** node `perlmutter` with
`PYTHONPATH=src` and the `.venv` interpreter.

| Command | Result |
|---|---|
| `make lint` via `.venv/bin/python3 -m ruff check .` | All checks passed |
| `make check` (`lint`, `test-fast`, `shell`, `notebooks`) | 73 tests OK; shell `bash -n` on every `experiments/` and `scripts/` launcher; 40 notebooks validated |
| `JAX_PLATFORMS=cpu make test-science` | 58 tests OK in 40.5 s on this login node. h5py 3.16.0 imported with `mpi=True` without initialising a communicator. The documented policy remains: run `test-science` / `check-all` on an allocation, because an MPI-linked build is not guaranteed login-safe |
| Combined fast + science | 131 tests OK |
| Nuisance regeneration and independent validation | section 3 |
| Background comparison at sample 0 | section 3 |
| EE endpoint diagnostic, Y1 and Y10 | section 6 |

New files: `src/limbercloud/validation/{method,cosmology,nuisance,run_identity}.py`,
`src/limbercloud/experiments/run_guards.py`,
`src/limbercloud/projection/{numba,jax}_backend/terminal.py`,
`scripts/nersc/check_onecovariance.py`,
`tests/test_{angular_contract,projection_endpoints,run_identity}.py`,
`revisions/2026-09/endpoint_diagnostic.py`.

Substantially rewritten: `validation/{estimator,contract,reference,evaluate}.py`,
`io/artifacts.py`, `tests/test_{science_artifacts,launcher_smoke}.py`, both
`experiments/benchmarks/*/benchmark.py`.

Mechanically updated: 24 spectra drivers, 38 shell launchers and wrappers.

### Input invalidations

- Any spectrum, covariance or timing product generated with `eta_IA = 0.5` is
  superseded. The IA array changed by up to 73%.
- Any product built with CAMB `kmax = 50` or a derived `Omega_g` is superseded.
- Any coefficient tensor or spectrum built before the endpoint repairs is
  superseded, although section 6 shows the numerical effect on current Y1/Y10 EE
  is `<= 5e-9`.
- Any CCL product evaluated at the 20 geometric centres is legacy-only.
- No accepted campaign artifact existed, so nothing published was invalidated.

## 9. Unresolved items, separated from code completion

**Code complete, external evidence outstanding.** These need an allocation and
were not run:

1. `scripts/jupyter/launch_kernel.sh --science` has never been executed. The
   claim that a real editor kernel inherits the batch modules and Conda hooks is
   **not** established; only the code path and the login-safe `--probe` are.
2. OneCovariance was not started. `check_onecovariance.py` is tested against
   synthetic scripts; the real checkout's dependencies in the selected
   interpreter remain unverified.
3. `install_mpi_h5py.sh` was exercised only with `LIMBERCLOUD_INSTALL_DRY_RUN=1`.
   No package was rebuilt; the validated environment is deliberately unchanged.
4. Local macOS Bash 3.2 portability was not executed here. The NERSC Bash path
   passes; the Bash 3.2 repairs are source-level and need a handoff to the local
   owner for execution, without simultaneous code edits.
5. JAX GPU was not exercised. All JAX results above are CPU
   (`JAX_PLATFORMS=cpu`).

**Code incomplete, carried forward.**

6. The ordinary NS/SN/SS elements still form `p = 1 - power1/power2` and return
   nonfinite values when the right-node power is exactly zero. The new terminal
   kernels are zero-safe. A zero-safe rewrite of the ordinary catalogue belongs
   with the sampled Limber power setup (C02) and was not attempted here.
7. The terminal integrals use a fixed Gauss–Legendre rule rather than closed
   forms. See section 6 for the numerical-stability reason; this is the main
   design decision in this phase that warrants review.
8. Derivation annotations in the Mathematica notebooks still carry interior-only
   labels (`i = j = n+1 < N`). The Python side and this report record the full
   terminal coverage; the `.nb` index annotations were not edited.
9. The 24 drivers remain copied timing loops. The shared evaluator, the NUMERIC
   family, the fiducial execution path, HDF5 spectrum writing and the 21→20
   operator inside execution are Phase 2.

## 10. Phase 2 entry conditions (historical, superseded)

Use the current Phase 2 prompt and the addendum above. The flags proposed in
item 3 below were subsequently removed by the author and must not be restored.

1. Build the one-cosmology evaluator against `SharedScience` /
   `ProducerWorkload`, dispatching CCL, NUMBA, JAX and NUMERIC and returning raw
   21-node samples and 20 bandpowers from `AngularContract`.
2. Extract `projection/numeric_backend/` and add the six
   `experiments/spectra/NUMERIC/{Y1,Y10}/{single,double,triple}.py` wrappers with
   launchers. `MethodIdentity` and `ProjectPaths` already accept them.
3. Replace the 24 copied loops with thin wrappers, and remove the temporary
   rejections of `--fiducial-only`, `--include-fiducial`, `--run-config` and
   `--mode validation` at the same time as the evaluator makes them operational.
4. Store raw 21 and band 20 as distinct datasets, write per-sample timings once
   per workload, and enforce the run identity on construction and on read.
5. Close the outstanding external evidence in section 9 before any pilot gate,
   and the zero-safe power rewrite before production.

Production remains blocked. Nothing in this phase ran a campaign, a pilot, a
benchmark or a GPU job, and no accepted scientific artifact was produced.
