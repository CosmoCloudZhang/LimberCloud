# Ensemble follow-up: current source and interpolation contract

**Date:** 19 September 2026. **Inspected revision:** [`0876bf4a50e869be1289a3eecb46931e7c8eb534`](https://github.com/CosmoCloudZhang/LimberCloud/tree/0876bf4a50e869be1289a3eecb46931e7c8eb534). This is a read-only source audit in a temporary checkout, supporting the updated code and manuscript plans. No production source was changed, no spectra were generated, and no Perlmutter jobs or installed OneCovariance version were inspected. Notebook cell references below are **zero-based indices in the notebook JSON**, counting all cells.

## 1. Correction to the earlier observer-interval diagnosis

The September 15 plan and historical supporting audit treated the first interval as having linear power with a zero observer endpoint. That assumption was incomplete. The current [matter-power notebook][power-notebook], cell 4, explicitly uses

\[
P(\chi)=P_1(\chi/\chi_1)^3,\qquad 0\leq\chi<\chi_1,
\]

and linear interpolation on subsequent intervals. All **21** coefficient-validation notebooks under `notebooks/derivation/{NN,NS,SS}` likewise define a `formula_0` with cubic power and select it when `chi1 == 0` (cell 7). See, for example, [NN B03][nn-b03] and [SS B01][ss-b01]. This is an intended first-interval policy, not an inference from one isolated plot.

For the NN basis, write \(x=\chi/\chi_1\), with falling and rising hats \(1-x\) and \(x\). Removing the common factor \(P_1/\chi_1\), the three observer-interval integrals are

\[
\int_0^1 x(1-x)^2\,dx=\frac1{12},\quad
\int_0^1 x^2(1-x)\,dx=\frac1{12},\quad
\int_0^1 x^3\,dx=\frac14.
\]

These were independently checked with exact rational arithmetic and match the observer branches in [Numba NN][nn-numba] and [JAX NN][nn-jax]. **The earlier recommendation to change NN `element3` from `1/4` to `1/2` is withdrawn.** The latter value describes a different, linear-power first-interval model. It must not be substituted into the implemented cubic policy.

Required follow-up is to document and test the same observer policy across the analytical implementation, numerical implementation checks, power figures, and manuscript. [Section 4][section4] currently describes linear interpolation without this exception. A first-interval cubic power law is distinct from the global third-order spline option called `cubic` in the numerical notebook comparisons.

This correction does **not** establish that all coefficient formulas and endpoint assembly are correct. The separate omission of the final NN diagonal remains visible: Numba only adds the rising-hat diagonal when `n + 1 < grid_size`, and JAX has the same condition. Validate the full final-interval treatment with nonzero endpoint values, or enforce and document a verified zero-endpoint restriction. Mixed NS support still includes a first subdiagonal. The related final-interval lensing cases remain subject to the independent tests in the code plan.

## 2. What the existing numerical comparisons actually vary

All six Y1/Y10 EE/TE/TT spectrum notebooks have identical numerical helper definitions in cell 4; source hashes were compared. [Y1 TE][te-notebook] provides a representative complete example.

| Quantity | Current implementation | Implication for metadata and interpretation |
|---|---|---|
| Distribution preparation | `numpy.interp` maps input redshift distributions onto 351 uniform redshift nodes from 0 to 3.5; each distribution is normalized by trapezoidal integration | Record both original data identity and evaluation grid; this preprocessing remains linear for all numerical orders |
| Radial distribution | `interp1d(chi_grid, phi_m_grid, kind=type)` | The order varies the reconstructed radial distribution in comoving distance, not directly the original redshift-space function |
| Lensing scale factor | `interp1d(chi_grid, 1/(1+redshift_grid), kind=type)` followed by division by the interpolated value | Numerical `slinear` interpolates **a**, whereas the analytical derivation interpolates **1+z**; these are different finite-grid approximations |
| Effective power | `interp1d(chi_grid, power_grid, kind=type)`; callers pass `POWER_GRID * AMPLITUDE_<component>` | Where bias or alignment depends on redshift, the interpolated quantity is the product, not independently interpolated factors |
| Interpolation orders | SciPy `slinear`, `quadratic`, `cubic` | Display/file labels may use LINEAR/QUADRATIC/CUBIC, but preserve the actual SciPy `kind` in metadata |
| Radial quadrature | `fixed_quad(..., n=100)` for the outer projection and each inner lensing integral | Order comparison alone does not certify quadrature convergence; a knot-aware or otherwise independently converged check is needed |
| First power interval | Numerical helpers apply their selected global interpolant; they do not special-case the cubic observer interval | NUMERIC–linear as currently written is not an exact same-representation check of the analytical coefficients |
| Angular postprocessing | Natural `CubicSpline(log(ELL_GRID), ELL_GRID*C)` integrated over log ell and divided by linear bin width | This is a band average and is separate from radial interpolation order |

The comparison therefore tests the **combined interpolation recipe**. It does not separately establish that changing only the density kernel, only the scale factor, or only the power interpolation causes a given residual. Preserve this useful three-order comparison, and describe that scope explicitly. A limited diagnostic that varies one ingredient at a time can clarify an unexpected difference; it need not become a full optimization campaign.

For the analytical implementation check, define a separate numerical reference that reconstructs the **same effective power, cubic observer interval, radial hats, and linear 1+z factor** as the intended analytical model, then integrates it independently. For approximation accuracy, compare the standard numerical orders and CCL under matched physical inputs. Do not silently change the old reference recipe and treat resulting residuals as the same historical measurement.

## 3. Current notebook outputs and plotting behaviour

The spectrum notebooks save flattened, unlabeled arrays under `ProjectPaths.validation_results(survey)`, which resolves to `results/validation/spectra/{Y1|Y10}`. See [path helpers][paths]. Each probe writes:

```text
C_CCL_<EE|TE|TT>.txt     CCL
C_DATA_<EE|TE|TT>.txt    LimberCloud (Numba notebook implementation)
C_DATA1_<EE|TE|TT>.txt   NUMERIC, SciPy slinear
C_DATA2_<EE|TE|TT>.txt   NUMERIC, SciPy quadratic
C_DATA3_<EE|TE|TT>.txt   NUMERIC, SciPy cubic
```

The write cells are 9 for EE/TT and 10 for TE. These files have no sample ID, cosmology table, run identity, explicit axes, interpolation contract, or completion metadata. They are historical fiducial notebook products, not saved benchmark ensembles. The experiment-output naming extension in the code plan should replace this ambiguity while preserving an explicitly labelled compatibility reader if needed.

CCL is evaluated at 20 geometric bin centres, while analytical and numerical arrays start at 21 bin edges and are then averaged over bands. This mismatch remains in every spectrum notebook. Resolve the common angular estimator before attributing residuals to interpolation.

All six error notebooks use the existing absolute fractional residual `abs(C_method/C_CCL - 1)` and logarithmic y axes. This agrees with the user's preferred visual format. Keep the main absolute/log format; preserve signed spectra and residuals in the saved data for covariance-weighted statistics. Compute ensemble quantiles **after taking absolute residuals per matched cosmology**, rather than taking the absolute value of signed quantiles. Exact zeros cannot appear on a log axis; mask or explicitly mark them only in rendering without adding a floor to statistical calculations.

The current zero-denominator fallback gives `abs(0 - 1) = 1`, an artificial 100% error when CCL is exactly zero. Replace this with an explicit invalid/near-zero mask and retain an absolute covariance-scaled residual where meaningful. Error-notebook cell 4 still loads the covariance as float32 and uses the historical column-wise triangular index for TT/EE; TE uses a lens-major rectangular index. The earlier verified OneCovariance interface mapping therefore remains relevant, subject to checking the version actually installed on Perlmutter. Oversized figures also remain: most pair panels allocate five inches per bin along both axes, independently of the expensive computation.

## 4. Other prerequisite findings still present in this revision

| Finding | Current source evidence | Required treatment |
|---|---|---|
| IA law differs from manuscript | [IA generator][ia], lines 55–65: eta=0.5 and only the resulting `A` array saved; manuscript Section 3 states eta=0 | Reconcile the intended law and save its parameters and redshift axis; do not infer artifact provenance from filenames |
| CCL count-tracer magnification conversion | [CCL Triple][ccl-triple], line 73 transforms values with `5*s-2`, then line 140 supplies the result as `mag_bias` | Distinguish slope s from response q; notebook/covariance CCL paths instead pass stored slope values |
| MS response omitted | [Numba Triple][numba-triple], lines 290–301: MS uses bare lens phi, MI uses magnification-weighted lens phi | Assemble both magnification terms consistently; this pattern also appears in the JAX variants |
| Sampled cosmology and prefactors mismatch | [Numba Triple][numba-triple], amplitudes 108–123 precede the sampled-cosmology loop; radial phi 167–168 still uses fiducial H | Compute conversion and lensing factors from the active cosmology, and declare how nuisance functions behave under cosmology variation |
| Covariance grid overwritten | [Y1 covariance writer][covariance], source grid 45, lens grid 67, then source tracer 151 uses the latter variable | Use explicit source/lens grids and validated resampling |
| Covariance table order and precision | [Y1 covariance writer][covariance], float32 arrays and ell-only `argsort` at 160, 183, 206; Y10 has the same construction | Apply the previously verified complete `(ell,i,j)` serialization contract and float64 precision; recheck upstream reader on Perlmutter |

These source checks do not reproduce the earlier upstream sentinel experiments or establish the physical validity of any current covariance product. Existing correct-looking figures are not evidence that these configuration and interface mismatches have been resolved.

## 5. Consequences for the agreed scope

- Use a named fiducial plus 1,000 sampled cosmologies. The same seed can produce matching draws only when RNG algorithm, draw order, parameter order, ranges, and random-call history agree. Save one common parameter table and its identity; every method and workload should consume it. Current CCL Triple uses ±10% multiplicative bounds, whereas Numba Triple uses ±5%, so changing the seed alone is insufficient.
- Preserve sequential accumulated timing for the sampled cosmologies within one executing task, using the existing internal Numba/JAX parallelism and measured resource controls. No task-based cosmology parallelization is required by this follow-up. Keep saved output and diagnostics outside the relevant compute intervals, and declare whether the separate fiducial has warmed compiled code before the timing sequence.
- Use NUMERIC as the method family and interpolation order as an explicit setting and filename/path discriminator. Metadata must retain the complete interpolation and quadrature contract above, not only the word LINEAR or CUBIC.
- Keep the current absolute log-scale residual layout, adding a restrained pointwise 16th–84th percentile band across the 1,000 sampled cosmologies where legible. The separately identified fiducial is not an additional random draw in those percentiles. Report sample failures/masks rather than silently dropping them.
- Grid refinement and more ell evaluation nodes have computational cost. The planned work should validate the adopted setup and reference accuracy with limited checks. It should not grow into a comprehensive redshift-grid or ell-binning optimization study; those remain future work. Distinguish evaluation nodes from analysis bins and match the angular estimator across methods.
- Piecewise-linear density interpolation supplies compact local basis functions and tractable coefficient formulas. It is **not mathematically the only interpolation linear in nodal values**: a fixed linear-in-data spline basis can also permit factorization, with different coefficient structure. Avoid an exclusive claim that only linear interpolation permits the tensor form.
- An arbitrary physical matter-power model can supply node values, but the present closed-form coefficient formulas assume a particular between-node representation. Alternative power laws or higher-order models require newly derived or numerically computed coefficients and validation; they are not an existing drop-in capability of these formulas. Do not call cubic interpolation universally stable or guaranteed more accurate; it is a useful conventional higher-order comparator whose behaviour must be checked.

[power-notebook]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/notebooks/power/Matter_Power_Interpolation.ipynb
[nn-b03]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/notebooks/derivation/NN/Coefficient_B03_Validation.ipynb
[ss-b01]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/notebooks/derivation/SS/Coefficient_B01_Validation.ipynb
[nn-numba]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/src/limbercloud/projection/numba_backend/nn.py#L5-L68
[nn-jax]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/src/limbercloud/projection/jax_backend/nn.py#L5-L76
[section4]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/manuscript/sections/section4.tex#L1-L11
[te-notebook]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/notebooks/spectra/Y1/TE_Spectrum_Validation.ipynb
[paths]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/src/limbercloud/io/project_paths.py#L101-L105
[ia]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/scripts/generate_config/intrinsic_alignment.py#L48-L69
[ccl-triple]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/CCL/Y1/triple.py#L70-L140
[numba-triple]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/NUMBA/Y1/triple.py#L108-L301
[covariance]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/covariance/Y1/matrix.py#L37-L211

## 6. Current experiment and timing contract

The source audit covered all **24 Python runners**, their **24 SLURM launchers**, four `Run_All.sh` helpers, both benchmark readers, and the relevant projection/path helpers. No `AGENTS.md` was present in the inspected checkout. The matrix is four implementations (CCL, NUMBA, JAX CPU, JAX GPU), two surveys, and three workloads:

| Canonical label | Computed probes | Representative evidence |
|---|---|---|
| `Single` | EE | [CCL Single][exp-ccl-single] |
| `Double` | TE and TT | [CCL Double][exp-ccl-double] |
| `Triple` | EE, TE, and TT | [CCL Triple projections][exp-ccl-projection] |

`Configuration.parse` accepts case-insensitive input but writes title-case labels. Within each family/workload, Y1 and Y10 Python sources are identical except for one explicit NumPy conversion in CCL Single; survey data differ through the tag. JAX CPU and GPU Python sources differ only in the device output directory. [Canonical labels][exp-labels]

### Sampling and accumulation

- Each runner performs 1,000 sequential draws and saves ten cumulative times at counts 100, 200, ..., 1,000. There is no separately computed fiducial or RNG seed in these runners. [CCL count and draw loop][exp-ccl-loop]
- CCL, JAX CPU/GPU, and NUMBA Single use multiplicative 0.9–1.1 bounds; NUMBA Double/Triple use 0.95–1.05. The shared table in the updated plan must remove this discrepancy. [NUMBA Single draws][exp-numba-single], [NUMBA Triple draws][exp-numba-loop]
- The fiducial configuration has `W0=-1`, `WA=0`, and `OMEGA_K=0`. A new generator must order negative-parameter bounds correctly and record that multiplying a zero fiducial leaves that parameter fixed, unless explicitly adopting nonzero additive bounds. [Fiducial values][exp-fiducial]
- The new named fiducial is additional to the 1,000 sampled rows. If computed first, it can warm JAX/Numba: report that timing policy explicitly rather than comparing warm sampled times with historical first-call-inclusive totals.

### Existing paths and proposed extension

The external runtime root is supplied by `LIMBERCLOUD_RUNTIME_ROOT`; it is distinct from the repository checkout. Current method/survey directories are defined centrally, and only CCL, NUMBA, and JAX are currently accepted. [Spectrum path helper][exp-paths]

```text
results/spectra/CCL/{Y1|Y10}/
results/spectra/NUMBA/{Y1|Y10}/
results/spectra/JAX/{CPU|GPU}/{Y1|Y10}/

Time_{Single|Double|Triple}_{NUMBER}.txt
Time_{Single|Double|Triple}_{NUMBER}_COSMOLOGY.txt
Time_{Single|Double|Triple}_{NUMBER}_CELL.txt          # CCL
Time_{Single|Double|Triple}_{NUMBER}_COEFFICIENT.txt   # NUMBA/JAX
Time_{Single|Double|Triple}_{NUMBER}_PROJECTION.txt    # NUMBA/JAX
```

`NUMBER` currently means allocated host CPUs, including for JAX GPU; all launchers default to 128. GPU count is separately one. Preserve that meaning and save explicit CPU/GPU metadata. These files contain timings only; final spectra are discarded. [CCL save statements][exp-ccl-save], [NUMBA save statements][exp-numba-save]

Retain these timing basenames for existing methods, with a new explicit `run_id` directory and corresponding reader support. Add saved probe bundles such as `Spectra_Triple_128_EE.npz`, with all 1,001 identified rows, actual parameters, axes, completion state, and matching metadata. For the new family, the agreed naming direction is:

```text
results/spectra/NUMERIC/LINEAR/Y1/{run_id}/
  Time_Triple_128_LINEAR.txt
  Time_Triple_128_LINEAR_COSMOLOGY.txt
  Time_Triple_128_LINEAR_CELL.txt
  Spectra_Triple_128_LINEAR_EE.npz
```

Use equivalent QUADRATIC/CUBIC names and TE/TT bundles when selected. The existing readers use exact filenames at the old method/survey root and hardcode the ten evaluation counts. Adding `run_id`, NUMERIC order, and metadata checks therefore requires deliberate reader changes; this is not a drop-in directory change. Legacy loading must be explicit and must not silently mix historical timings into a new paired run. [Benchmark lookup and counts][exp-benchmark]

### One task with internal parallelism

All launchers request one task per node and use `srun -n 1`. The existing `Run_All.sh` submits six survey/workload jobs per family; it does not divide a cosmology sequence across tasks. [NUMBA launcher][exp-numba-shell], [submission helper][exp-run-all]

NUMBA sets its OpenMP threading layer and thread count from the allocation. SS/SN/NS coefficient kernels have explicit `parallel=True` and inner `prange`; NN has `njit` without explicit parallel mode. Tensor projection uses NumPy `einsum`. [SS coefficient loop][exp-numba-ss], [SN coefficient loop][exp-numba-sn], [NS coefficient loop][exp-numba-ns], [NN loop][exp-numba-nn], [tensor contraction][exp-numba-tensor]

JAX CPU configures XLA threading; JAX GPU selects CUDA and one GPU. Its coefficient kernels use `jit`, `vmap`, and `lax.fori_loop`, with jitted tensor contraction. Coefficients and projected components are already synchronized before the corresponding timers stop. Preserve these barriers and the sequential outer loop. [CPU launcher][exp-jax-cpu], [GPU launcher][exp-jax-gpu], [JAX kernel][exp-jax-kernel], [timing barriers][exp-jax-barriers]

### Output and timing corrections specific to these runners

1. **Experiment outputs currently have different ell coordinates:** CCL evaluates 20 geometric bin centres, while NUMBA/JAX project at 21 edge nodes. Unlike the notebook path described above, these runners do not subsequently form band averages. Match the selected angular estimator before comparing saved arrays. [CCL ell definition][exp-ccl-ell], [NUMBA ell definition][exp-numba-ell]
2. **JAX does not assemble final probe totals:** it computes and synchronizes separate SS/SI/IS/II and other components; NUMBA adds components into EE/TE/TT. Include consistent total assembly and its compute cost when adding spectrum outputs. [JAX components][exp-jax-components], [NUMBA total assembly][exp-numba-totals]
3. **Stage labels do not imply equivalent work:** CCL's COSMOLOGY timer ends after object construction and parameter setup, leaving lazy background/power evaluation inside CELL; LimberCloud evaluates distance and matter power before its COSMOLOGY timer ends. The benchmark reader deliberately repeats CCL total as a reference in all stage panels. State that interpretation in labels/captions and use matched total work for cross-method speed comparisons. [CCL stage boundary][exp-ccl-loop], [LimberCloud stage boundary][exp-lc-cosmology], [benchmark stage panels][exp-stage-panels]

[exp-ccl-single]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/CCL/Y1/single.py#L112-L118
[exp-ccl-double]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/CCL/Y1/double.py#L132-L144
[exp-ccl-projection]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/CCL/Y1/triple.py#L132-L148
[exp-labels]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/src/limbercloud/config/experiments.py#L8-L31
[exp-ccl-loop]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/CCL/Y1/triple.py#L91-L130
[exp-numba-single]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/NUMBA/Y1/single.py#L104-L115
[exp-numba-loop]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/NUMBA/Y1/triple.py#L125-L155
[exp-fiducial]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/scripts/generate_config/cosmology.py#L25-L47
[exp-paths]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/src/limbercloud/io/project_paths.py#L72-L93
[exp-ccl-save]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/CCL/Y1/triple.py#L153-L163
[exp-numba-save]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/NUMBA/Y1/triple.py#L352-L364
[exp-benchmark]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/benchmarks/Y1/benchmark.py#L12-L54
[exp-numba-shell]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/NUMBA/Y1/triple.sh#L9-L42
[exp-run-all]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/NUMBA/Run_All.sh#L10-L14
[exp-numba-ss]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/src/limbercloud/projection/numba_backend/ss.py#L268-L327
[exp-numba-sn]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/src/limbercloud/projection/numba_backend/sn.py#L147-L190
[exp-numba-ns]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/src/limbercloud/projection/numba_backend/ns.py#L147-L190
[exp-numba-nn]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/src/limbercloud/projection/numba_backend/nn.py#L48-L69
[exp-numba-tensor]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/src/limbercloud/projection/numba_backend/tensor.py#L5-L6
[exp-jax-cpu]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/JAX/CPU/Y1/triple.sh#L32-L43
[exp-jax-gpu]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/JAX/GPU/Y1/triple.sh#L7-L42
[exp-jax-kernel]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/src/limbercloud/projection/jax_backend/ss.py#L290-L358
[exp-jax-barriers]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/JAX/CPU/Y1/triple.py#L262-L277
[exp-ccl-ell]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/CCL/Y1/triple.py#L84-L89
[exp-numba-ell]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/NUMBA/Y1/triple.py#L85-L105
[exp-jax-components]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/JAX/CPU/Y1/triple.py#L280-L382
[exp-numba-totals]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/NUMBA/Y1/triple.py#L256-L350
[exp-lc-cosmology]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/spectra/NUMBA/Y1/triple.py#L164-L179
[exp-stage-panels]: https://github.com/CosmoCloudZhang/LimberCloud/blob/0876bf4a50e869be1289a3eecb46931e7c8eb534/experiments/benchmarks/Y1/benchmark.py#L110-L120
