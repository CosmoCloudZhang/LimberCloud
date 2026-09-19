# Code revision plan: covariance correctness and reproducible validation on Perlmutter

**Original audit:** 15 September 2026. **Plan updated:** 19 September 2026 following the user's scope clarification and a fresh GitHub script/notebook review. **Status:** proposed implementation; no production code changes or scientific jobs performed for this plan.

**Current review baseline:** GitHub main [`0876bf4a50e869be1289a3eecb46931e7c8eb534`](https://github.com/CosmoCloudZhang/LimberCloud/tree/0876bf4a50e869be1289a3eecb46931e7c8eb534), retrieved into an isolated temporary clone on 19 September. Inspect and reconcile the actual Perlmutter checkout before implementation; its current revision and runtime products have not been inspected in this update. Existing detailed citations to `7d29b2f` and the historical local `57c0731` state describe the 15 September audit, not the current remote checkout. New findings below and the [ensemble follow-up](supporting/limber_ensemble_followup.md) supersede conflicting historical findings, especially the NN observer-interval interpretation. References to inspected OneCovariance remain separately pinned.

## 0. Agreed scope and current-script reconciliation

Implement one explicitly labelled fiducial cosmology plus **1,000 sampled cosmologies**, with the same saved cosmology table across Y1/Y10, Single/Double/Triple, CCL, NUMBA, JAX CPU/GPU and NUMERIC linear/quadratic/cubic. The fiducial is sample `000000`; sampled IDs are `000001` through `001000`. Ensemble quantiles use the 1,000 sampled rows only. This is a planned campaign, not an existing result.

Preserve the paper's **one-task, sequential cumulative-time benchmark**. Current launchers use one task and `srun -n 1`; each process evaluates cosmologies sequentially. Numba and JAX provide internal compiled/threaded/vectorized computation. Do not add job arrays over cosmologies, multiprocessing pools, MPI, multi-device distribution or simultaneous methods within a benchmark process. Task-level production parallelism and scaling studies are future work. Existing `Run_All.sh` submits independent benchmark configurations; it does not parallelize the cosmology loop. Independent jobs must receive uncontended declared resources for timing.

The current source establishes:

| Contract | Current implementation at `0876bf4` | Required revision |
|---|---|---|
| Sampling | 24 Python runners, no seed; 1,000 draws per runner | One canonical seeded generator and persisted parameter table, consumed by all runners |
| Ranges | Numba Double/Triple use ±5%; CCL, JAX and Numba Single use ±10% | One declared common range; proposed paper default is the existing ±10% nonzero-parameter range, with supported/fixed parameters explicit |
| Timings | Ten cumulative records at 100, 200, ..., 1,000 sampled evaluations; no spectra saved | Retain those x values and cumulative computation totals; save each evaluation's spectra and stage durations |
| CPU count | `--number` is passed from `SLURM_CPUS_PER_TASK`, including JAX/GPU | Preserve filename token as allocation label; separately record actual physical cores, thread pools and GPU count |
| Parallelism | One process; Numba NN is serial `njit`, NS/SN/SS use `parallel=True`/`prange`; JAX uses JIT/vectorization | Preserve existing implementation; do not claim all stages scale with Numba thread count |
| Configuration labels | Single=EE, Double=TE+TT, Triple=EE+TE+TT | Preserve this mapping; the earlier plan's Double=EE+TE statement was incorrect |
| NUMERIC | Notebook-only `slinear`, `quadratic`, `cubic` interpolants plus numerical quadrature | Extract one NUMERIC family with `--interpolation linear|quadratic|cubic`; map linear to the legacy `slinear` setting |
| Primary error figures | Absolute fractional error on logarithmic y axes | Retain absolute/log presentation and pair layout; add optional magnitude-quantile bands |
| Numerical scope | Chosen radial grid and ell binning | Document choices and run focused correctness/reference checks; defer comprehensive grid optimization and alternative power/ell schemes |

See C08–C12 for precise sample, file, restart, statistical and timing contracts. [Cursor execution prompts](CURSOR_IMPLEMENTATION_PROMPTS.md) split implementation into reviewable stages. The user's clarification supersedes earlier recommendations to introduce task-based ensemble parallelism, make signed errors the main plot, or require an optimal-grid study.

## 1. Recommended strategy and priorities

Move expensive scientific validation into deterministic Slurm scripts that retain one sequential cosmology loop per method/configuration. Let notebooks load checked results and make figures. Share the scientific evaluator with existing experiment entry points, while keeping validation and timing as explicitly different execution modes; a timing run may save reusable spectra outside its timers.

The covariance problem has two demonstrated ordering defects: spectra are serialized in an order that the inspected OneCovariance reader misinterprets, and TT/EE plotting notebooks assign covariance diagonals to the wrong pairs. Fix these through a small interface layer. Additional mismatches in magnification, IA configuration, cosmology prefactors and angular estimators must be resolved before treating newly shared script outputs as scientifically interchangeable.

The existing benchmark runs cannot simply supply the notebook plots: they sample random cosmologies, save timing totals rather than spectra, and omit the notebooks' independent numerical references. Moving those computations into one giant benchmark run would preserve the expensive restart problem. The useful shared unit is one evaluation with explicit inputs and outputs.

### Priority and manuscript dependencies

| ID | Priority | Deliverable | Manuscript dependency |
|---|---|---|---|
| C01 | prerequisite | One declared physics/configuration contract | Fiducial setup, IA equation, magnification definition and reproducibility |
| C02 | prerequisite | Correct component assembly and active-cosmology factors | All component comparisons; any random-cosmology accuracy claim |
| C03 | prerequisite | Matched multipoles and angular estimator | Accuracy figures, residuals and covariance comparisons |
| C04 | prerequisite | Independent endpoint/formula and quadrature validation | Exactness of interpolated integral; derivation appendix |
| C05 | urgent | Correct OneCovariance input serialization | All regenerated covariance products |
| C06 | urgent | Explicit covariance output index map | Correct uncertainty curves in every TT/EE panel |
| C07 | urgent | Reproducible covariance configuration and physical validation | Survey-error and inference-impact statements |
| C08 | high | Shared evaluator and batch validation runner | Reproducible Y1/Y10 validation evidence |
| C09 | high | Numeric artifacts with axes, hashes and completion status | Figure and table provenance |
| C10 | high | Plot-only notebooks and bounded memory | Reliable reruns and readable figure exports |
| C11 | high | Explicit analysis selections and residual summaries | Niko's Y10/eligible-pair comments; meaningful accuracy claims |
| C12 | after correctness | Matched timing campaign and stage definitions | Abstract speed-up, benchmark figure/table and conclusions |
| C13 | operational | Preserve current environment; profile resources | Reproducible Perlmutter execution |
| C14 | release gate | Focused tests and staged acceptance | Evidence required before replacing draft claims/figures |

Recommended patch sequence: **C01–C04 scientific definitions → C05–C07 covariance repair → C08–C10 production/plot split → C11 validated results → C12 timing**. C05/C06 unit fixes can be developed in parallel with C01–C04, but corrected final covariance requires physically matched input spectra. C13/C14 apply throughout. None of this requires rewriting the tensor framework, introducing MPI, or building an emulator.

The companion [COAUTHOR_COMMENT_INVENTORY.md](COAUTHOR_COMMENT_INVENTORY.md) and [MANUSCRIPT_REVISION_PLAN.md](MANUSCRIPT_REVISION_PLAN.md) organize author comments and prose changes. This document supplies the implementation and evidence dependencies; it does not repeat each editorial comment.

## 2. Scientific correctness before output reuse

### C01 — Declare one configuration contract

**Findings.** The latest [IA generator][ia] sets `eta_pivot=0.5`; the current [manuscript §3.2][setup] says `eta_IA=0.0`. The older local generator used zero. The generated JSON stores the resulting array without the law's parameters or its redshift axis, so identical filenames can represent different fiducials. Do not infer which generator produced the Perlmutter artifacts from their names.

The generator's density convention is otherwise explicit: `rho_x(a=1, species='matter', is_comoving=True)` and `C1=5e-14/h**2`. Preserve the interpretation `rho_m,0 = Omega_m,0 rho_crit,0`, the signed IA coefficient, and `D(0)=1`; do not add a physical-density `(1+z)^3` factor. CCL calls with `use_A_ia=False` expect the already normalized raw coefficient. [Official CCL tracer documentation][ccl-api] supports this distinction.

The [magnification generator][magnification] saves the values quoted in Eq. 3.3. The [CCL Triple runner][ccl-triple] computes `5*values-2` and then supplies that transformed quantity as `NumberCountsTracer.mag_bias`. CCL expects the number-count slope **s**, while the analytical count response is **q=5s−2**. The covariance exporter supplies the untransformed stored values to CCL. These paths are inconsistent even before table ordering is considered. The numerical interpretation of the stored values must be settled using their survey-source provenance; the current code pattern strongly suggests slopes, but the field is ambiguously named.

**Minimal edits.**

1. Store named `magnification_slope_s` and derived `magnification_response_q`; pass `s` to CCL and use `q` in the analytical magnification-weighted distribution. Derive once, not at multiple API boundaries.
2. Record IA amplitude, pivot, eta, density convention, normalization and redshift grid. Choose the intended fiducial law explicitly. If the paper's stated baseline is retained, restore eta zero in the generator; if eta 0.5 is intended, change the draft and regenerate affected results. Neither choice is established by this audit.
3. Store the bias model and its redshift grid. Distinguish keeping a fixed tabulated nuisance function during parameter sampling from recomputing a cosmology-dependent `1/D(z)` law.
4. Preserve separate source/lens grids. [Covariance preparation][cov-y1] samples source distributions on their support, overwrites `z_grid` with lens support, then builds source tracers using the overwritten grid. It becomes wrong if those supports differ. Use explicit grid names and interpolation or assert the common validation grid; matching array lengths is insufficient.
5. Record survey area, exact lens/source selection, effective number densities and shape-noise convention. [Survey][survey] uses 18,000 deg² for both Y1 and Y10, and [density configuration][density] uses the listed sparse lens samples. These are explicit settings, not proof of an unmodified SRD scenario.

**Tests and invalidation.** For `s=0.4`, every magnification contribution must vanish (`q=0`). Use non-unit, bin-dependent slopes to detect double conversion. Check IA arrays against the declared law and redshift axis, including eta zero and nonzero. Changing any of these definitions invalidates dependent spectra, covariance, errors and timings as evidence for the new configuration; preserve old products with their original identity.

### C02 — Correct component assembly and sampled-cosmology factors

The [Numba Triple runner][numba-triple] applies the magnification response to MI but omits it from MS (lines 290–301). JAX shares the omission; Double/Y10 variants follow the same assembly pattern. The TE validation notebook includes the response in both terms. `amplitude_ms` contains the lensing prefactor squared and does not absorb the missing bin response. A shared evaluator must assemble MS and MI with the same magnification-weighted lens distribution.

The same timed runners calculate lensing amplitudes using fiducial `Omega_m,H` before constructing sampled cosmologies; their radial-density conversion combines sampled `E(z)` with fiducial H. This does not define the same physical spectrum as a CCL calculation using the sampled cosmology. Recompute distance conversion and lensing prefactors from the active cosmology. Explicitly choose whether IA/galaxy-bias laws remain fixed functions or are regenerated, and apply that choice to both backends. These findings are visible in [Numba Triple][numba-triple] and [JAX GPU Triple][jax-triple].

Begin with deterministic fiducial validation. Then generate one seeded parameter table for all backends/configurations. Fresh inspection clarifies the range discrepancy: Numba Double/Triple use ±5%, while Numba Single and all CCL/JAX configurations use ±10%. Equal seeds do not repair unequal bounds or changed RNG call order. Multiplicative perturbation of zero `wa`/curvature leaves those parameters zero; keep these fixed and label that scope unless an explicitly supported extension is chosen. Do not introduce curvature into this spatially flat implementation through an additive sampling range. C08 specifies the shared table and restart contract.

Tests should isolate physical components: turn IA and magnification off independently; use distinct bin responses; test reverse mixed-pair transposition; verify active-cosmology amplitude/distance changes; compare total component sums with CCL at the same multipoles. Keep the common linear-bias/NLA validation scope explicit. Bin-dependent scale-dependent bias requires additional tensors or an independently validated operator decomposition; it is not a free extension of the measured runtime.

### C03 — Use the same angular estimator

The [spectrum notebooks][spectra-notebooks] evaluate CCL at geometric bin centres. Their analytical branch integrates a spline of `ell*C_ell` over `log(ell)` and divides by bin width, yielding a uniform-`dell` band average. Comparing those arrays mixes interpolation error with an estimator difference. Timing scripts also differ: CCL uses 20 centres while Numba/JAX use 21 edges.

Use two clearly named outputs where needed:

- **Sampled spectra:** every method evaluated at exactly the same `ell` nodes. Use this for the cleanest numerical-method comparison and as input to a table interface expecting sampled `C_ell`.
- **Bandpowers:** apply the same explicit, normalized window to CCL and LimberCloud samples. Store window definition, edges and effective centres. Use this for a covariance-normalized bandpower comparison.

The inspected [OneCovariance multipole builder][onecov-ell] integer-casts logarithmic edges before calculating centres. Notebook floating edges differ by up to **2.235%** for 20 bins over 20–2000; the first centres are 22.440369 versus 22.360680. Verify the Perlmutter version and save its actual edges/centres. Relabelling a centre evaluation as a band average is not a correction. The existing 101 covariance-input samples and 20 output bands serve different purposes; one cannot replace the former with 20 band averages without changing the contract.

Tests: constant and power-law spectra under the selected window; identical nodes and output shape across backends; rejection of unequal window/edge hashes; comparison against the actual upstream binned estimator. Any convergence table must state whether it concerns sampled spectra or bandpowers.

### C04 — Validate endpoint formulas and the independent reference

**Correction to the 15 September audit:** do not replace the NN observer-interval `1/4` coefficient by `1/2` on the basis of the old plan. The current matter-power notebook explicitly uses `P(chi)=P1*(chi/chi1)^3` on the first interval, then linear interpolation on later intervals. For this cubic observer policy the three NN hat-product integrals are `(P1/chi1)*(1/12, 1/12, 1/4)`, matching the special branches in both backends. The earlier defect claim assumed linear P on that interval and therefore did not establish a bug in the intended model. Document and independently test this policy across NN/NS/SS and numerical references; the notebook alone does not prove every special branch is correct. NN coefficient assembly still excludes the final endpoint diagonal; check that separately against the declared far-boundary convention and nonzero-endpoint fixtures. [Fresh source evidence and derivation](supporting/limber_ensemble_followup.md).

The manuscript's final-interval lensing-basis cases also need a truncated endpoint treatment or an explicit validated zero-endpoint condition. Mixed NS coefficient support includes a first subdiagonal ([NS implementation][ns-numba]); an assertion that its entire lower triangle vanishes is too strong. Correct formula/support prose alongside any implementation change.

Use independent scalar quadrature tests for all 21 NN/NS/SS structural cases, including first/final intervals, nonzero far-endpoint densities, signed/zero amplitudes and narrow intervals. With the declared cubic observer policy, the NN integrand is O(chi) for bounded endpoint densities and remains integrable even when those densities are nonzero. Include nonzero observer-hat fixtures; do not impose an unnecessary zero-density condition carried over from the old linear-power assumption. Check regularity separately for other families or alternate representations and reject genuinely divergent fixtures. The historical audit independently checked the three interior NN expressions on one interval to approximately `6e-11` relative agreement; that is a limited check, not a whole-package validation. Existing [projection tests][projection-tests] compare only representative `element1` values across two ports, which can share a mistake.

Extract the notebooks' three numerical-reference methods intact first. Their repeated 100-point nested `fixed_quad` calculations are expensive, and increasing interpolant order alone does not establish quadrature convergence. Preserve a historical reproduction mode, then establish sufficient integration accuracy on focused fixtures, splitting at knots or using another independently converged scheme as needed. Compare the same piecewise representation, including its observer interval, when testing analytic integration. Separately compare differing interpolants to assess the adequacy of the chosen representation. Cubic is a useful comparator, not an exact reference or a guarantee against overshoot. Document changes to references separately from changes to LimberCloud.

Do not make a comprehensive search for optimal Delta z, ell grids or power interpolation a completion requirement. Keep the configured Delta z=0.01 baseline and declared ell setup, with only limited refinement/tolerance checks needed to substantiate accuracy at that setting. Linear distribution interpolation is chosen for local support, sparse structure and simple analytic coefficients; factorization requires a basis linear in distribution values and is not unique to first-degree interpolation. Alternative fixed spline bases can preserve that algebra but need different coefficients. An arbitrary P(k,z) can define tensor integrals, but existing closed-form coefficients assume the present interval representation (including the special first interval). General power-law alternatives remain future derivation and validation work, not a drop-in parameter switch.

## 3. Minimal OneCovariance correction

### C05 — Serialize all input pairs in the required order

Both [Y1][cov-y1] and [Y10][cov-y10] writers use `numpy.argsort(ell.flatten())` at lines 159, 182 and 205. With repeated multipoles, default sorting does not define tied bin-pair order. The inspected [upstream reader][onecov-input] takes every `N_i*N_j`th ell and directly reshapes the value column into `(N_ell,N_i,N_j)`; it does not reconstruct values from the supplied labels.

Required order is **ell outermost, bin i next, bin j innermost**:

```text
ell_0  1  1  C_11(ell_0)
ell_0  1  2  C_12(ell_0)
...
ell_0  N_i  N_j  C_NiNj(ell_0)
ell_1  1  1  C_11(ell_1)
```

Keep the complete Cartesian pair set, including both EE/TT triangles. Input tables and the unique output data vector have different conventions. For TE, i denotes the clustering/lens bin and j the shear/source bin. The official [OneCovariance input documentation][onecov-api] and example confirm the full-combination requirement.

The smallest safe correction to each existing writer is:

```python
order = numpy.lexsort((index2.ravel(), index1.ravel(), ell.ravel()))
table_cell['ell'] = ell.ravel()[order]
table_cell['tomo_i'] = index1.ravel()[order]
table_cell['tomo_j'] = index2.ravel()[order]
table_cell[value_column] = value.ravel()[order]
```

The last key is primary. Prefer one small tested serializer shared by the two survey wrappers. Another valid implementation transposes values from `(i,j,ell)` to `(ell,i,j)` before flattening and constructs matching labels directly. Use float64 ell and spectra, integer labels, and full output precision.

**Executed test:** the exact installed upstream reader was extracted through Python AST and called on a temporary 2-lens × 3-source × 4-ell sentinel table. Existing sorting decoded **19/24 entries incorrectly**; lexsort decoded **24/24 correctly**. This demonstrates the interface failure without running CCL or a covariance calculation. The actual Perlmutter reader must pass the same test before production.

Validate monotonic unique support, contiguous bin labels, no duplicate/missing `(ell,i,j)` tuples, complete rectangular coverage, finite values and EE/TT symmetry. Preserve signed cross spectra. Reject corrupt inputs before expensive calculation.

### C06 — Decode the output using explicit labels

Current [error notebooks][error-notebooks] use `INDEX=j*(j+1)//2+i` for TT/EE. That enumerates pairs column by column. The inspected [OneCovariance output builder][onecov-output] uses row-wise upper triangles. For three bins:

```text
upstream: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
old map:  (0,0), (0,1), (1,1), (0,2), (1,2), (2,2)
```

Use an explicit pair list, then label every variance:

```python
pairs = [(i, j) for i in range(n_bins) for j in range(i, n_bins)]
for index, (i, j) in enumerate(pairs):
    sigma[i, j, :] = sigma_by_pair[index, :]
    sigma[j, i, :] = sigma_by_pair[index, :]
```

Default blocks are TT unique pairs, TE full lens-major pairs, EE unique pairs; ell varies fastest within a pair for the single sample dimension used here. TE's `i*N_source+j` agrees with that default. Obtain the final map from verified output/config metadata, especially if custom probe/pair selections are enabled. A three-bin test is essential because two bins cannot reveal this triangular mismatch.

Load float64 covariance once in a postprocessor and cache a small labelled diagonal product for plots. Check full matrix dimensions before slicing. With all configured pairs and 20 common bands, expected sizes are 1100 for Y1 (5 lenses, 5 sources) and 2400 for Y10 (10 lenses, 5 sources). These dimensions are conditional on those selections, not universal constants.

**Executed test:** the exact installed matrix-builder method produced the stated row-wise ordering for a sentinel tensor. Separately, a simultaneous row/column permutation `P C P^T` preserves eigenvalues: output relabelling can repair panel assignments but cannot turn an indefinite physical matrix positive definite. C05 requires covariance regeneration; C06 requires correct downstream mapping.

### C07 — Make configuration and covariance validation explicit

Current [covariance launchers][cov-launcher] require `${RUNTIME_ROOT}/results/covariance/${TAG}/CONFIG.ini`, but the repository has no INI renderer/template. Their existing upstream-script preflight does not establish that the configuration points to the new tables or agrees with the survey, cosmology, binning and noise.

Capture the actual Perlmutter INI, then add a small versioned template/renderer or checked configuration builder. Resolve paths through `ProjectPaths`; record upstream commit, effective INI and input hashes. Emit both labelled-list and matrix formats for cross-checking. Use a fresh run directory and publish the manifest only after validation. Files named `ALIGNMENT.ascii` or `MAGNIFICATION.ascii` being written does not prove OneCovariance reads them.

Start with a clearly labelled **Gaussian-only diagnostic**, generated from the matched full observed-field spectra and correct noise. Then separately assess requested connected non-Gaussian and super-sample covariance (SSC). External `C_ell` tables establish Gaussian input products; they do not automatically establish a matching internal higher-order model. The inspected upstream `onecov` source has no magnification API references, while its internal weights/responses generate NG/SSC. Verify those physical terms before promoting a full-covariance inference claim. Do not silently remove intended contributions merely to obtain a successful matrix.

There is already a paper inconsistency: [§3][setup] calls reference curves Gaussian at line 56 and describes Gaussian+NG+SSC at line 64; the notebooks load total `MATRIX.ascii`. Identify the actual plotted product and correct its label and conclusions.

#### What the local VAECloud files establish

Bounded discovery found an artifact tree under `/Users/s2227120/VAECloud`, but no associated source scripts. Every local spectrum table violates lexicographic order:

| Local table | Rows out of required order |
|---|---:|
| Y1: each of gg, gkappa, kappakappa | 2410 / 2525 |
| Y10: gg | 9992 / 10100 |
| Y10: gkappa | 4940 / 5050 |
| Y10: kappakappa | 2410 / 2525 |

Both local matrices are finite, symmetric and have positive diagonals. Float64 tests nevertheless give:

| Local file | Shape | Minimum correlation-matrix eigenvalue | Cholesky |
|---|---:|---:|---|
| [Y1 MATRIX.ascii](/Users/s2227120/VAECloud/CELL/Y1/COVARIANCE/MATRIX.ascii) | 1100 × 1100 | −28.7663205854 | fails |
| [Y10 MATRIX.ascii](/Users/s2227120/VAECloud/CELL/Y10/COVARIANCE/MATRIX.ascii) | 2400 × 2400 | −401.0202129190 | fails |

These are measurements of the local files, **not the current LimberCloud covariance or the user's previously corrected Perlmutter run**. Locate the exact successful VAECloud revision/config/output before adopting it as a golden fixture. Large negative values here cannot be explained as a tiny roundoff issue. Diagonal jitter, eigenvalue clipping or a nearest-positive-definite projection would hide an interface/model problem rather than establish correct usage.

#### Acceptance checks

1. Round-trip all input tables through the actual installed reader and compare with source arrays.
2. Check the observed-field spectral matrix, including the adopted noise, for symmetry and positive semidefiniteness within a scale-aware tolerance.
3. Compare an independent Gaussian calculation for TT, TE, EE and cross-probe blocks. At individual modes, use `(S_ac*S_bd + S_ad*S_bc)/[(2ell+1)f_sky]`, with `S=C+N`; then apply the actual estimator windows. An unspecified `Delta ell` approximation is inadequate as an exact bandpower test.
4. Check finite square shape, measured relative asymmetry, positive diagonal, labels/dimensions, component sums, correlation bounds, Cholesky, eigenvalues, conditioning and a deterministic solve residual. Record tolerances and original asymmetry before any diagnostic symmetrization.
5. Apply the same science-vector mask to residuals and both matrix axes. Validate that selected principal submatrix. Avoid duplicating symmetric observables, which introduces artificial singularity.
6. Add NG and SSC separately after reviewing model consistency and convergence. Individual components can be semidefinite; the final covariance used in a Gaussian likelihood must support a stable solve on the chosen independent data vector.
7. Replace covariance-dependent figures/claims only after these gates pass. Positive definiteness is necessary for that likelihood, but insufficient to certify ordering, selection or physics.

## 4. Production scripts and plotting notebooks

### C08 — Share the evaluator and generate one cosmology table

Add `src/limbercloud/validation/evaluate.py` with explicit inputs for survey/configuration, active cosmology, nuisance conventions, redshift grid, ell nodes, method/backend and requested components. Return final EE/TE/TT arrays and requested diagnostics. Keep plotting and file I/O outside the evaluator. Add `reference.py` for NUMERIC interpolation/quadrature and a separate, identical-integrand quadrature oracle for analytical coefficient tests. Preserve Single=EE, Double=TE+TT, Triple=EE+TE+TT. Timing each configuration means actually executing its requested workload, not slicing cached Triple outputs.

Implement one small seeded sampling helper, for example `experiments/spectra/generate_samples.py`. Generate a table once at `results/spectra/inputs/<run_id>/Cosmologies.npz` with `Manifest.json`, then require every runner to load it. Record the seed, NumPy/bit-generator versions, parameter order, explicit sorted bounds, actual values, sample IDs and content hash. Same seed alone is sufficient only with identical RNG, draws, ordering and ranges; persisting the table prevents these dependencies from drifting. Use `default_rng` with an explicitly recorded bit generator. Sort multiplicative bounds for negative fiducials such as w0=-1, and explicitly keep zero wa and curvature fixed under this study. Do not change this flat-cosmology scope incidentally. A proposed common domain is ±10% for the currently sampled nonzero parameters; finalize and document the supported parameter list/bounds before the campaign. Fiducial ID 0 is separate and does not consume a random draw. On restart, load the table rather than advancing an RNG to infer the missing rows.

Make the existing `experiments/spectra/{CCL,NUMBA,JAX/...}/{Y1,Y10}/{single,double,triple}.py` entries thin users of the evaluator and sample runner. Add a NUMERIC family under `experiments/spectra/NUMERIC/` with `--interpolation linear|quadratic|cubic` (linear maps to legacy SciPy `slinear`). Mirror survey/configuration wrappers only where the current launcher convention needs them; do not duplicate scientific kernels for each order. Add proposed shared options `--run-config`, `--sample-table`, `--run-id`, `--mode validation|benchmark`, `--include-fiducial`, `--sample-count`, and `--resume`. Preserve existing `--tag` (survey), `--path`, `--label`, `--folder` and `--number` meanings in compatibility wrappers; a new shared CLI may use clearer aliases such as `--survey`. A small validation command defaults to one fiducial; the explicit paper campaign requests the fiducial plus all 1,000 table rows for every method and interpolation order.

The versioned run configuration declares physics/input identities, radial grid, analysis ell nodes/windows and a separate sampled ell grid for covariance input. The current 101-point raw grid over 20–2000 is a compatibility default to check, not a universal OneCovariance requirement. Validation and covariance preparation share this contract. Pilot CCL+Numba first, then JAX CPU/GPU and NUMERIC. Production remains one sequential cosmology loop per job; there is no new task-based cosmology parallelism. Reuse CCL tracers/backgrounds within an evaluation, retaining identical outputs and transparently remeasuring timings after any change.

Preserve the notebook references' original behaviour in a labelled reproduction mode before changing their estimator/tolerances. Check the numerical quadrature on a small representative set, then run all three accepted interpolation orders over the same 1,001 rows. A cheap pilot is a development gate, not a substitute for the requested ensemble. Resume completed samples without redoing them; save failures explicitly and never replace a failed cosmology by another random draw.

### C09 — Extend existing timing names with explicit spectra products

Current timing products live under `results/spectra/CCL/<survey>`, `NUMBA/<survey>`, `JAX/CPU/<survey>` and `JAX/GPU/<survey>`. Preserve these family/device distinctions and title-case configuration labels. Add a run-ID directory so new physics/sampling generations cannot overwrite old results. The `128` token below continues to mean the `--number` CPU-allocation label, even on a GPU job; GPU identity/count belongs in metadata. Examples for Triple/Y1:

```text
results/spectra/CCL/Y1/<run_id>/
  Time_Triple_128.txt
  Time_Triple_128_COSMOLOGY.txt
  Time_Triple_128_CELL.txt
  Spectra_Triple_128_EE.npz
  Spectra_Triple_128_TE.npz
  Spectra_Triple_128_TT.npz
  Manifest_Triple_128.json

results/spectra/NUMBA/Y1/<run_id>/
  Time_Triple_128.txt
  Time_Triple_128_COSMOLOGY.txt
  Time_Triple_128_COEFFICIENT.txt
  Time_Triple_128_PROJECTION.txt
  Spectra_Triple_128_EE.npz
  ...

results/spectra/JAX/{CPU,GPU}/Y1/<run_id>/
  Time_Triple_128.txt
  Spectra_Triple_128_EE.npz
  ...

results/spectra/NUMERIC/LINEAR/Y1/<run_id>/
  Time_Triple_128_LINEAR.txt
  Time_Triple_128_LINEAR_COSMOLOGY.txt
  Time_Triple_128_LINEAR_CELL.txt
  Spectra_Triple_128_LINEAR_EE.npz
  Spectra_Triple_128_LINEAR_TE.npz
  Spectra_Triple_128_LINEAR_TT.npz
  Manifest_Triple_128_LINEAR.json
```

Use QUADRATIC/CUBIC in both directory and basename for the other NUMERIC settings. The redundancy makes detached files identifiable. Use the same pattern for Single/Double, other resource labels and Y10. One path/name helper owns these rules. `ProjectPaths.spectrum_results` currently accepts only CCL/NUMBA/JAX and must be extended. Both `experiments/benchmarks/{Y1,Y10}/benchmark.py` use exact old filenames/root paths and hardcoded cumulative counts; explicitly add run-ID, metadata and NUMERIC support. Keeping the old basename alone does not make a new subdirectory layout compatible. Offer explicit legacy-reading mode for historical plots, without mixing old timings with the new matched ensemble.

Each consolidated spectra file contains numeric `sample_id`, `is_fiducial`, cosmology values/parameter labels, ell coordinates, bin-pair labels and `cl` with axes `(sample, ell, pair)`; pair orientation and symmetric-pair policy are declared. The full campaign has 1,001 rows, fiducial first. Store bandpowers separately within a clearly named array with window/edge metadata when required. Save per-sample stage timings in a numeric companion file with stage names. NPZ plus JSON is sufficient initially; use NPY with memory mapping only if the measured product size warrants it. No pickle/object arrays or new database are needed.

Minimum manifest fields:

| Group | Required content |
|---|---|
| Identity | schema version, run ID, survey, family/backend/device, interpolation, configuration, source revision and dirty-diff identity |
| Inputs | shared sample-table hash, actual parameters, configuration/data hashes, IA/bias choices, named s/q, survey/noise settings |
| Coordinates | radial axes, ell nodes/edges/window, observable/bin labels/orientation, explicit vector map |
| Numerics | dtype, interpolation variable and endpoint policy, quadrature/tolerances, tensor/chunk settings, reference-check status, component definitions |
| Products | files/checksums, sampled spectra and requested bandpowers/components, completed/failed sample IDs, per-sample times and validity masks |
| Execution | package versions, CPU allocation/affinity/thread pools, device/count, warm-up policy, job/segment IDs, wall time and peak memory |

Checkpoint each completed sample (for example in a configuration/resource-specific `checkpoints/` namespace), atomically writing its arrays then its completion record. Release the sample's large intermediates. Finalize the compact ensemble files after all expected IDs validate; write the completed run manifest last. Reject duplicate/missing IDs or dependency mismatches. A failed run remains visibly partial; a deliberately analysed partial ensemble must report its actual matched count and failures. Never silently drop failures or change the advertised 1,000-sample denominator.

The existing `results/validation/spectra/<survey>/<run_id>` parent can hold reference diagnostics, index manifests and derived summaries pointing to these base experiment products. Avoid duplicate competing master spectra. Covariance artifacts refer to input spectra; error summaries refer to both. Base spectra do not require a covariance hash, avoiding a circular dependency. Optional `C_CCL_*`/`C_DATA*` text exports represent only a labelled fiducial compatibility view. Plot loaders must report missing/stale artifacts rather than launching calculations.

### C10 — Bound both compute and figure memory

One float64 coefficient tensor with 351×351×21 entries is **19.74 MiB**. Twelve components total approximately **236.87 MiB** before temporary allocations, cosmology objects and backend overhead. That lower bound alone does not explain a full-node OOM. Memory scales quadratically with radial nodes: one tensor is 314.47 MiB at 1401×1401×21, and 3009.97 MiB at 1401×1401×201. [Coefficient construction][ss-numba] also contains nested work beyond allocation size.

For validation, prepare, project and release components sequentially, retaining small projected arrays. Process bounded ell chunks if profiling shows a need; 4 or 8 ell values are tuning candidates, not measured optima. JAX chunks should use stable shapes/padding to avoid repeated compilation. Save coefficients only for explicit diagnostics. Record chunking in the artifact and benchmark contract.

Plotting is a separate credible memory burden. Existing 5-inch-per-panel, 512-dpi rasterized figures imply approximate full RGBA canvases of **625 MiB** for a 25×25-inch Y1 figure, **1.22 GiB** for Y10 TE and **2.44 GiB** for Y10 TT. Renderer buffers/copies add to these calculated sizes; no current job's peak RSS was measured. The [spectrum/error notebooks][spectra-notebooks] also retain figures without explicit closure.

Make the six spectrum and six error notebooks plot-only. Select spectra by `is_fiducial`/sample ID 0, not whichever file/row happens to be first; the consolidated storage order is fiducial first for convenience. Load compact summaries for ensemble bands and tables, with optional saved spectra for diagnostics. Use compact publication dimensions, ordinary screen DPI, vector line artists where appropriate, and paginated all-pair diagnostics. Close each figure after saving/display. Preserve useful explanatory markdown and independent diagnostic plots. Do not imply that moving calculation out of notebooks alone fixes oversized rendering.

Kernel and matter-power notebooks can later consume cached background/figure inputs. Derivation notebooks remain independent educational/checking tools and should not become required production steps.

### C11 — Preserve absolute error plots and quantify the ensemble

Add `experiments/validation/summarize.py`. Match samples by ID and sample-table hash, then verify identical physics, ell/window and pair labels before subtracting. Use `delta C = C_method - C_CCL` at the same cosmology. Report per probe/survey/sample the median, 95th percentile and maximum of `abs(delta C/C_CCL)` over explicitly valid data-vector entries, plus counts and worst pair/ell. Keep these within-vector summaries distinct from quantiles across the 1,000 cosmologies. Include `abs(delta C)` and `abs(delta C)/sigma` with a validated covariance. Retain signed differences in diagnostic products.

Keep the current main pair-panel format and absolute errors on logarithmic y axes. Show the explicit fiducial curve; where readable add an ensemble median and a light pointwise 16th–84th percentile band for LimberCloud versus CCL. Calculate magnitudes for each cosmology **before** taking quantiles; exclude fiducial ID 0. Label bands as variation under the declared cosmology sampling, not observational errors, posterior intervals or an envelope containing 68% of whole curves. Keep detailed NUMERIC-order bands in supplementary panels if the main layout becomes crowded. Store pointwise valid counts and tail summaries; do not imply a narrow central band excludes rare failures.

Existing zero-denominator fallbacks fabricate a 100% fractional error or a sigma ratio of one. Replace them with explicit undefined/near-zero masks, using a declared scale-aware small-signal criterion. A signed cross spectrum is valid; only the ratio loses meaning near its zero. An exact residual zero remains zero in arrays and statistics, but is masked or shown below a labelled plotting floor on a log axis. Do not clip raw errors to that floor or modify summary quantiles. Apply the same display convention when a percentile bound is zero. Absolute values support the requested dynamic range but do not cure unstable integration; retain signed diagnostics to investigate it.

Implement lens-source eligibility and scale cuts as named, reproducible masks with their definition and source. A concrete candidate matching Niko's thresholds is the [pinned binny selection example][binny-selection]: require the source-distribution peak to lie above the lens-distribution peak and a normalized minimum-overlap fraction no larger than 0.10 for Y1 or 0.25 for Y10. For individually unit-normalized distributions, use `overlap = integral min(p_lens(z), p_source(z)) dz`; the documented general metric divides the unnormalized minimum integral by the product of the two normalization integrals. Normalize both distributions on a declared common grid before evaluating the statistic, and record the peak/tie convention and numerical integration rule.

Label this explicitly as the adopted Niko/binny prescription. The thresholds and exact metric were verified in that implementation example, not independently established here as the unique SRD definition. The actual sampled Y1/Y10 arrays are still required to enumerate the accepted pair list and compare it with the intended analysis. Preserve that list in the artifact and figure legend/caption metadata.

Keep all field spectra required for covariance construction, and apply the likelihood selection only to the final vector/matrix. Specify `k` units and the effective-redshift/window prescription for `ell_max`; using the upper edge of a broad bin does not enforce a sharp k cut throughout its support.

For each matched cosmology s compute `D_s = delta C_s^T Sigma_fid^{-1} delta C_s` by a Cholesky solve with the exact selected/binned vector. Use one validated fiducial covariance per survey, a fixed selection and one reused factorization for the main campaign. This provides a common survey-error scale across cosmologies; it does not assume that physical covariance is cosmology independent. A few representative covariance recalculations can be added if needed to assess a sensitive conclusion; 1,000 covariance jobs are out of scope. Do not construct the joint statistic by adding per-probe values when cross-probe covariance is present.

Report the full D as primary and optionally D/N_data as discrepancy per selected element. If calling the latter reduced chi-square, define it explicitly: these are deterministic predictions at the same parameters, no parameters are fitted, and ideal agreement is zero rather than one. D equals a chi-square difference for a noiseless CCL reference, not a general observed-data likelihood shift. For actual residual r=d-C_CCL, the shift is `D - 2 r^T Sigma^{-1} delta C`. The 1,000 cosmologies are sample points in the test domain, not independent observed surveys; do not add their D values into one survey statistic or call a small D/N_data sufficient when full D matters.

Start with a manuscript table containing survey/method, N_data, successful/attempted matched count, fiducial D, ensemble median D and 16th–84th range, with 95th percentile and maximum/sample ID in the main or supporting table. The 68% range is empirical variation under the sampling design, not uncertainty on a median or a posterior credible interval. Decide on an extra figure after seeing the distribution: table plus bands for small variation, an ECDF for tails, parameter-versus-D scatter for trends, or a violin only if its smoothed distribution adds useful information. Produce all Y1/Y10 numbers from accepted artifacts; no new maxima or D distributions exist yet.

## 5. Timing and Perlmutter execution

### C12 — Rerun timings with matched boundaries

Current [benchmark plotting][benchmark] repeats total CCL time in panels labelled cosmology, coefficient and projection. CCL's cosmology timer stops after object construction, so lazy setup can be charged later; LimberCloud explicitly evaluates background/power before ending that stage. Numba sums projected EE components inside its timed stage, while JAX synchronizes separate component arrays without the same final sum. There is no separate explicit warm-up, and cumulative timing files mix first-call effects with subsequent work. JAX already calls `block_until_ready()`; preserve those barriers. [CCL][ccl-single], [Numba][numba-single] and [JAX][jax-single] show these boundaries.

Retain one SLURM task and the sequential cosmology loop. Record separately:

1. Cold initialization/compilation and any transfer/setup costs.
2. Warm coefficient construction plus projection, including the same component sum and declared output estimator.
3. Complete end-to-end evaluation including matched cosmology preparation and the same outputs.

A projection-only measurement may be retained with explicit precomputed-coefficient/device-residency assumptions. Use `time.perf_counter`, synchronize measured JAX work, and write data outside timers. Include the same final EE/TE/TT component assembly and angular estimator in all full-prediction timings. State where required host/device transfer is charged; device-only contraction may be a separately labelled quantity. If measuring isolated CCL projection, explicitly precompute required background/power with APIs verified in the installed version. Do not pretend CCL lazy preparation and LimberCloud explicit preparation yield equivalent stage labels without resolving their boundaries. If CCL total remains a reference line in all benchmark panels, label it as total in each panel.

Computing the fiducial first warms compiled/cache paths. Exclude it and any dedicated warm-up from the sampled cumulative count and publish the new curves as warm sampled evaluations. Keep cold startup/fiducial/compilation costs separately. Record each sample's nonoverlapping compute-stage durations; derive the ten cumulative totals at 100, 200, ..., 1,000 by summing the sampled evaluations. Do not rerun prefixes to obtain those points. Serialization/checkpoint I/O is outside compute timers; separately record total job wall time including I/O. Check total-stage accounting without double counting nested timers. In addition to the accumulated curves, provide per-evaluation medians/spread across the declared sample table, noting this combines cosmology-dependent work with timing variation.

Science production can resume sequentially. Record restart segment IDs and repeat warm-up outside each segment's timers. A sum of resumed compute durations is labelled segmented accumulation, not uninterrupted elapsed-time evidence. Final claims about one continuous benchmark sequence should use an uninterrupted accepted run. Use the same shared samples and workload for all four backends and three NUMERIC orders; no task-based throughput campaign is requested. Existing configuration jobs may be submitted independently with declared, uncontended resources.

Preserve the C09 timing basenames and explicit legacy-read mode; update readers for new directories, methods and metadata. Never mix generations after physics/grid/hardware/threading changes. A timing-mode run can also produce the reusable spectra, so the final campaign need not calculate the same 1,001 spectra twice merely to separate plotting from timing. Tiny validation/reference pilots remain independent of the full campaign. Use new evidence to resolve the abstract's approximately 3 ms claim and speed-up denominator.

### C13 — Preserve the current environment fixes

Current GitHub already centralizes activation and replaces ambiguous environment names. Keep [the shared loader][environment-loader], `modules/cpu.sh` or `modules/gpu.sh`, and their common module stack. Canonical keys are:

| Key | Meaning |
|---|---|
| `LIMBERCLOUD_RUNTIME_ROOT` | External data/config/results/plots/log root |
| `LIMBERCLOUD_CONDA_ENV` | Existing environment name/prefix; default `CosmoConda` |
| `LIMBERCLOUD_ONECOVARIANCE_ROOT` | Checkout directory containing `covariance.py` |
| `LIMBERCLOUD_REPO_ROOT` | Optional checkout-location override |
| `LIMBERCLOUD_TEXLIVE_BIN` | Optional TeX executable directory |

The loader supports legacy aliases; do not make them canonical again. Existing wrappers use `set -eo pipefail`, load the selected CPU/GPU profile, and no longer source `.bashrc`. Preserve the project `.venv`/kernelspec startup arrangement, which loads project configuration; an ordinary global kernel does not necessarily do so. `environment.yml` is a candidate dependency specification, not a solved lock or evidence of installed versions. Reuse the working collaboration stack; do not recreate it merely to split computation from plotting. Correct only minor stale documentation such as lowercase `run_all.sh` where current Linux filenames are `Run_All.sh`.

Perlmutter CPU nodes have 128 physical cores and 512 GB RAM; GPU nodes have 64 physical cores, 256 GB host RAM and four A100 GPUs. Slurm CPU counts refer to logical CPUs; distinguish those from physical threads. [Official architecture][nersc-architecture] and [affinity guidance][nersc-affinity] should inform the allocation, then verify actual affinity on the node.

Preserve current launch allocations as the starting benchmark configuration and verify their actual affinity/thread usage. Under SMT=2, allocating 128 logical CPUs is not proof of 128 distinct physical cores. If binding or oversubscription needs correction, document it as a new timing generation; do not turn this revision into a full thread-count optimisation study. Use one dominant internal thread budget with verified binding. Avoid multiple Python processes each inheriting every CPU. Covariance launchers currently request 256 CPUs; the historically inspected VAE INIs use 128 workers, so check the actual OneCovariance worker/thread arrangement separately from the one-task spectrum benchmark.

Numba's projection helper uses NumPy `einsum`; increasing Numba threads does not automatically parallelize the whole stage. Treat contraction optimization as a separately measured change. Use one GPU per current JAX process, assert the selected device, and log GPU memory. Allocating four GPUs alone does not distribute this implementation.

## 6. File-by-file implementation packages

New filenames below are proposals. Keep each package reviewable and retain compatible wrappers.

| Package and files | Proposed changes | Completion check |
|---|---|---|
| C01–C02: `scripts/generate_config/{intrinsic_alignment,magnification_bias,galaxy_bias}.py`; shared evaluator | Named physics, axes and active-cosmology rules; correct s/q and MS assembly | Component-isolation and configuration tests; regenerate dependent products |
| C03–C04: `src/limbercloud/projection/{numba_backend,jax_backend}/{nn,ns,ss}.py`; new reference/window helpers | Only independently demonstrated boundary/formula corrections; common estimator | All structural/endpoint tests and reference convergence pass |
| C05–C06: new `src/limbercloud/io/covariance.py`; both `experiments/covariance/*/matrix.py` | Serialize complete tables; explicit pair maps; float64 and grid assertions | Exact installed reader/output contract round-trip |
| C07: proposed `experiments/covariance/{prepare,validate}.py` and INI template; both `matrix.sh` | Checked configuration, fresh output identity, post-run validation | Gaussian oracle, label/list agreement and PD/solve report |
| C08: new `src/limbercloud/validation/{evaluate,reference}.py`; shared sampler; `experiments/spectra/NUMERIC`; small validation entry | Shared science; fiducial + saved 1,000-row ensemble; one NUMERIC family with three settings | Matched samples/outputs, focused reference checks and sequential restart |
| C09: new `src/limbercloud/io/validation.py`; `project_paths.py` | Timing/spectra naming, sample axes, fingerprints, checkpoints and completion checks | Missing/stale/mismatched products fail; no accidental mixing of generations |
| C10: all `notebooks/spectra/*/*` and `notebooks/error/*/*` | Load compact accepted artifacts; plot at bounded size; close figures | Fresh-kernel plotting without importing CCL/CAMB/JAX/Numba |
| C11: new `experiments/validation/summarize.py` | Absolute-error bands, masks, per-cosmology D and optional D/N_data, Y1/Y10 tables | Counts/quantiles and statistics trace to selected arrays/covariance |
| C12: 24 existing `experiments/spectra` Python entries plus NUMERIC; both benchmark plotters | Shared physics, saved outputs, sequential timing and metadata | Sampled-only accumulation at 100…1,000; clear cold/warm/stage/full timings |
| C13: proposed validation Slurm wrapper; existing launchers/docs | Reuse current environment helpers; dependency/resource logging; no oversubscription | Allocated smoke test and measured RSS/affinity |
| C14: focused tests, `scripts/validate_notebooks.py`, figure manifest/docs | Contract/science tests and provenance checks | Local checks plus explicitly separate Perlmutter science gates |

## 7. C14 — Verification gates and command templates

### Gate A: fast deterministic checks

Add meaningful small tests for the observed failures: 2×3 input-table sentinel; three-bin output triangle; s/q and zero magnification; MS/MI factors; active cosmology; named grids and IA law; first/final tensor intervals with the correct cubic observer policy; common window; shared sampling across all configurations and restart; invalid artifact rejection; nonzero/signed/near-zero spectra. Verify quantiles after absolute values, fiducial exclusion, known covariance quadratic forms and cumulative-stage accounting outside file I/O. Preserve the current environment/launcher smoke tests. `make check` validates code/notebook contracts but does not establish scientific accuracy or runtime success.

### Gate B: allocated Y1/Y10 science

Run a mathematically valid tiny setup first, then one full Y1 fiducial CCL+Numba comparison. Measure stage time and peak RSS. Establish reference tolerance and perform bounded resolution sensitivity checks; repeat Y10 and representative nonfiducial samples with every backend/order. This is not an optimal-grid search. Parse covariance tables with the installed upstream reader, run Gaussian-only covariance, validate C07, then assess additional components. Preserve a separate run identity for every physics/reference change.

### Gate C: publication products

Generate summaries, masks and all figures from completed products in a clean plotting kernel. Review near-zero panels and selected pairs, precision, axes and legibility. Record figure-to-run/config/covariance links in `manuscript/figures/manifest.toml`. Update manuscript numerical claims only after the resulting summaries support them.

### Gate D: performance evidence

Run the physically matched one-task timing campaign after pilot science acceptance, as an explicit timing mode with spectra saved outside timers. Reuse its accepted spectra for the final 1,000-sample accuracy summaries; keep pilot/reference diagnostics separate. Record cold and warm conditions, package/device versions, thread allocation and output contract. Then revise the speed-up table/abstract and ensemble accuracy claims.

### Perlmutter templates — proposed commands only

These commands were **not executed**. New validation/prepare/validate CLIs below are proposed interfaces to implement with the file packages above. Run expensive work only inside an appropriate allocation; use the actual approved project account, paths and resource layout.

Existing preparation/preflight pattern:

```bash
cd /path/to/reconciled/LimberCloud
source scripts/nersc/load_environment.sh
source scripts/nersc/modules/cpu.sh
conda activate "${LIMBERCLOUD_CONDA_ENV}"
limbercloud_require_onecovariance
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
git rev-parse HEAD
git -C "${LIMBERCLOUD_ONECOVARIANCE_ROOT}" rev-parse HEAD
python -c 'import sys; print(sys.executable)'
module list
make check
```

The job wrapper should record resolved numerical-package versions, config/data hashes and actual thread/device state as well. If editable installation is necessary, use the project's existing procedure and `python -m pip install --no-deps -e .` to preserve the established dependency stack.

Proposed deterministic production and covariance steps, inside an `sbatch` script or a verified compute-node shell with the selected binding. Obtaining an `salloc` allocation alone does not prove a command runs on a compute node; use an appropriate `srun` job step when launching from that shell. Set the Numba/OpenMP thread budget for the spectrum step and reset worker/thread pools for the covariance step, rather than inheriting unrestricted threading across the sequence. The run-config file below is a proposed configuration to create and validate under C01/C03/C08; it declares both analysis and raw covariance ell grids:

```bash
python experiments/validation/spectra.py \
    --run-config "${LIMBERCLOUD_RUNTIME_ROOT}/config/validation_revision_y1.json" \
    --survey Y1 --backend NUMBA --configuration Triple \
    --reference-methods ccl \
    --run-id revision_y1_fiducial --resume

python experiments/covariance/prepare.py \
    --run-config "${LIMBERCLOUD_RUNTIME_ROOT}/config/validation_revision_y1.json" \
    --survey Y1 --validation-run revision_y1_fiducial \
    --covariance-mode gaussian --run-id revision_y1_gaussian

python "${LIMBERCLOUD_ONECOVARIANCE_ROOT}/covariance.py" \
    "${LIMBERCLOUD_RUNTIME_ROOT}/results/covariance/Y1/revision_y1_gaussian/CONFIG.ini"

python experiments/covariance/validate.py \
    --survey Y1 --run-id revision_y1_gaussian

python experiments/validation/summarize.py \
    --survey Y1 --run-id revision_y1_fiducial \
    --covariance-run revision_y1_gaussian
```

These commands illustrate the small fiducial/covariance pilot. After acceptance, run the explicit shared-table campaign for CCL, NUMBA, JAX CPU/GPU and NUMERIC LINEAR/QUADRATIC/CUBIC, each with the fiducial plus 1,000 sampled rows and one task. Plotting refreshes only read those products. Implement `prepare.py` so it refuses an artifact lacking the raw sampled spectra declared by the shared run configuration or whose physical/estimator contract differs; check monotonic ell support, sufficient range and adequate sampling resolution, with 101 points only the current compatibility default. It must not silently treat 20 bandpowers as raw inputs. Repeat for Y10 after the pilot gates pass. The Slurm wrappers can link preparation/covariance/summary steps with `afterok` dependencies or execute them sequentially with fail-fast semantics. These dependencies do not parallelize cosmologies. A failed validation must prevent downstream publication of a completed manifest.

Retrieve diagnostic accounting for actual job IDs:

```bash
: "${JOB_ID:?Set JOB_ID to the actual Slurm job ID}"
sacct -j "${JOB_ID}" --format=JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS
```

Check which job step owns the measured RSS and consult its logs before concluding that a failure is host OOM, GPU OOM, timeout or environment-related. Compare plotting and compute stages separately. Follow [NERSC job guidance][nersc-jobs] when finalizing allocation/binding options.

## 8. Audit coverage and evidence limits

The review covered all seven manuscript sections and main file/figure mapping; 24 experiment entry points and their launchers; both covariance producers; six spectrum and six error notebooks; configuration generators; core NN/NS/SS/projection implementations; runtime paths, environment/kernel helpers and relevant tests. Additional kernel, power and derivation notebooks were reviewed for role and dependencies. Current-main directory renames were reconciled; no recommendation restores obsolete environment setup.

The **15 September audit** executed source comparisons, input/output sentinels, historical local VAE table ordering and float64 covariance eigenvalue/Cholesky checks, the quoted limited NN scalar quadrature check, and analytical memory/ell-centre calculations. Its installed OneCovariance source was at `91e139622eba568d7f742a2d12f6b66a106e6e68`; its source was unmodified, while three existing output spectrum files were dirty and left untouched. Those historical source trees/artifacts are not bundled in this review folder. The **19 September update** reread the fresh GitHub scripts/notebooks/core sources, reconciled contracts and revised planning documents; it did not repeat production scientific calculations.

The current Perlmutter OneCovariance revision, effective LimberCloud INI, exact active environment, source data, scientific arrays, job logs, timing distributions and measured RSS were not inspected here. No fresh scientific spectra, correct production covariance, Y10 residual maxima or replacement speed-up measurements are claimed. The successful historical VAECloud correction may be a different remote revision/product and remains a retrieval task. This plan creates a route to those results; it does not mark their validation complete.

<!-- Source links are pinned to the audited code revision. -->
[ia]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/scripts/generate_config/intrinsic_alignment.py#L50-L68
[magnification]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/scripts/generate_config/magnification_bias.py#L27-L51
[setup]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section3.tex#L7-L64
[ccl-triple]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/experiments/spectra/CCL/Y1/triple.py#L69-L147
[numba-triple]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/experiments/spectra/NUMBA/Y1/triple.py#L108-L301
[jax-triple]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/experiments/spectra/JAX/GPU/Y1/triple.py#L115-L321
[cov-y1]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/experiments/covariance/Y1/matrix.py#L37-L211
[cov-y10]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/experiments/covariance/Y10/matrix.py#L37-L211
[cov-launcher]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/experiments/covariance/Y1/matrix.sh#L14-L40
[survey]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/scripts/generate_config/survey.py#L27-L43
[density]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/scripts/generate_config/number_density.py#L26-L68
[spectra-notebooks]: https://github.com/CosmoCloudZhang/LimberCloud/tree/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/notebooks/spectra
[error-notebooks]: https://github.com/CosmoCloudZhang/LimberCloud/tree/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/notebooks/error
[nn-numba]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/src/limbercloud/projection/numba_backend/nn.py#L37-L70
[nn-jax]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/src/limbercloud/projection/jax_backend/nn.py#L36-L74
[ns-numba]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/src/limbercloud/projection/numba_backend/ns.py#L148-L166
[ss-numba]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/src/limbercloud/projection/numba_backend/ss.py#L269-L316
[projection-tests]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/tests/test_projection_consistency.py
[paths]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/src/limbercloud/io/project_paths.py#L95-L111
[benchmark]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/experiments/benchmarks/Y1/benchmark.py#L99-L133
[ccl-single]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/experiments/spectra/CCL/Y1/single.py#L61-L131
[numba-single]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/experiments/spectra/NUMBA/Y1/single.py#L67-L211
[jax-single]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/experiments/spectra/JAX/GPU/Y1/single.py#L180-L228
[environment-loader]: https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/scripts/nersc/load_environment.sh
[onecov-input]: https://github.com/rreischke/OneCovariance/blob/91e139622eba568d7f742a2d12f6b66a106e6e68/onecov/cov_input.py#L6053-L6079
[onecov-output]: https://github.com/rreischke/OneCovariance/blob/91e139622eba568d7f742a2d12f6b66a106e6e68/onecov/cov_output.py#L4779-L4855
[onecov-ell]: https://github.com/rreischke/OneCovariance/blob/91e139622eba568d7f742a2d12f6b66a106e6e68/onecov/cov_ell_space.py#L425-L460
[onecov-api]: https://onecovariance.readthedocs.io/en/latest/api.html
[ccl-api]: https://ccl.readthedocs.io/en/latest/api/pyccl.tracers.html
[nersc-architecture]: https://docs.nersc.gov/systems/perlmutter/architecture/
[nersc-affinity]: https://docs.nersc.gov/jobs/affinity/
[nersc-jobs]: https://docs.nersc.gov/systems/perlmutter/running-jobs/

[binny-selection]: https://github.com/binny-org/binny/blob/eb913fe99675019d8f410ef40d1b5aed4b560291/docs/examples/tomography/selections.rst#L241-L321
