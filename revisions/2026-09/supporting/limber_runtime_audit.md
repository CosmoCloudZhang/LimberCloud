> 22 September execution routing: use the current [code plan](../CODE_REVISION_PLAN.md) and [new prompt index](../CURSOR_IMPLEMENTATION_PROMPTS.md). This source audit retains its original historical findings and uncertainty; current author decisions supersede conflicting provisional choices.

# Runtime and notebook audit for the LimberCloud revision plan

> **Status — 2026-09-19:** This is historical audit evidence, not the current execution plan. The [code plan](../CODE_REVISION_PLAN.md), [manuscript plan](../MANUSCRIPT_REVISION_PLAN.md), and [implementation prompts](../CURSOR_IMPLEMENTATION_PROMPTS.md) take precedence over its provisional recommendations. Code and scientific scripts are edited and validated on NERSC; manuscript sources, accepted publication figures, and the paper manifest are edited locally. The manuscript submodule may remain uninitialized and absent on NERSC; scientific jobs and ordinary checks must not require it. Old environment selectors, `.npz` spectrum proposals, and the staging sequence below are superseded by the dedicated environment selected through `.venv`, the final HDF5 spectra contract, and the staged prompts. Include `mpi4py` and `h5py` in the validated environment while keeping task-based benchmark parallelism outside scope. Historical findings and source references below are retained without rewriting the original evidence.

> **Historical note, superseded where indicated (19 September):** use the updated [code plan](../CODE_REVISION_PLAN.md) and [fresh audit](limber_ensemble_followup.md). The workload description below was inaccurate: Single=EE, Double=TE+TT, Triple=EE+TE+TT. New production will use one fiducial plus 1,000 shared sampled cosmologies and save spectra while preserving sequential one-task timing and existing internal parallelism. Task-based cosmology parallelism and comprehensive grid optimisation are outside this revision.

Audit date: 2026-09-15. Baseline: current GitHub `7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c`, inspected at `/private/tmp/limber_review/github-main`, after initial comparison against local `57c0731`. Read-only source audit; no production implementation edits, jobs, or scientific reruns were performed. `LIMBERCLOUD_RUNTIME_ROOT` is not set in the local process. Therefore no current Perlmutter result arrays, measured peak memory, timings, or environment versions were verified. Notebook cell references below are **zero-based**.

## Recommendation

Use one shared, deterministic scientific evaluation implementation for the scripts and validation. Move expensive validation production into Slurm jobs and make the six spectrum notebooks and six error notebooks load checked products and draw figures. Keep timing as a separate execution mode, even if it uses the same evaluator and appears in the same submission workflow. The existing timing files cannot supply the plots: the timing runners discard spectra and use randomly changing cosmologies, whereas the notebooks compute a fixed fiducial cosmology and three extra numerical-reference methods.

The smallest useful change is a new validation runner plus a small shared evaluator/artifact layer, then thin notebook readers. A rewrite of every coefficient formula, MPI orchestration framework, or data format is unnecessary. Correct the scientific mismatches below before treating the script results as interchangeable with the notebook results.

## 0. Baseline update and migration mapping

The current GitHub environment/launcher cleanup is already implemented. It should be adopted before making the next scientific changes, after preserving/reviewing local work. This audit does not request reimplementing that cleanup.

| Older local path/contract | GitHub 7d29b2f target |
| --- | --- |
| `notebooks/error_analysis/` | `notebooks/error/` |
| `notebooks/kernels/` | `notebooks/kernel/` |
| `notebooks/matter_power/` | `notebooks/power/` |
| `notebooks/derivations/` | `notebooks/derivation/` |
| `experiments/spectra/*/run_all.sh` | `Run_All.sh` with exact capitals on Linux |
| `CosmoENV` | `LIMBERCLOUD_CONDA_ENV`, default `CosmoConda` |
| Ambiguous `ONE_COVARIANCE_ROOT` / script alias | `LIMBERCLOUD_ONECOVARIANCE_ROOT`, the directory containing `covariance.py` |
| Shell-only exports / later terminal export to notebooks | Common dotenv loader; project kernel loads dotenv at startup |
| Direct `.bashrc` sourcing/module duplication | `scripts/nersc/modules/{cpu,gpu,common}.sh` |

New `environment.yml` specifies a standalone candidate environment with Python 3.12, NumPy 2.2, SciPy 1.13, Numba 0.63, Astropy 7.2, Matplotlib 3.10, pyccl 3.2, ipykernel 7.1, and JAX CUDA12 0.9.x ranges. These are **manifest specifications, not verification of the active Perlmutter environment or a reproducible solved lock**. The creation helper is opt-in for new names/prefixes and refuses to modify an existing environment; it deliberately omits generic mpi4py/h5py builds. Reuse validated collaboration builds and record actual resolved versions. New GPU dependencies do not imply GPU allocation, and the project plotting kernel launcher does not itself load the CPU/GPU module profiles.

Notebook source comparison against the new GitHub checkout confirms the scientific findings below remain. Spectra notebooks changed `LABEL='CELL'` to `SPECTRA`, adopted explicit `pyccl.background.h_over_h0(cosmo=...)`, and wrapped CCL output in float64; kernel notebooks also repaired legend variables. Error notebook code and the power notebook are unchanged after directory migration. Core projection sources are unchanged. The only timing Python edit is the float64 wrapper in CCL/Y1/Single. Existing benchmark runners still call `cosmology.h_over_h0`; the extracted shared helper should use one API verified with the actual installed pyccl version.

Minor documentation corrections remain: `docs/nersc.md:72` says lowercase `run_all.sh` although the files/tests use `Run_All.sh`; `docs/environment.md:119` says the display name is `LimberCloud (.venv / CosmoConda)`, while `scripts/jupyter/register_kernel.sh:6` sets it to `LimberCloud`. These should be corrected without changing the functioning environment machinery.

## 1. Verified architecture and execution contracts

- `src/limbercloud/io/project_paths.py` defines external `data`, `config`, `results`, `plots`, and `logs`, including `validation_results(survey)` at lines 107–111. Its current docstring says validation spectra are written by notebooks; the path itself can stay unchanged after moving writers into scripts.
- `experiments/spectra/` preserves 24 entry points: CCL, NUMBA, JAX/CPU, JAX/GPU × Y1/Y10 × Single/Double/Triple. **Corrected 19 September:** Single contains EE; Double contains TE+TT; Triple contains all three. In the older local checkout the Y1/Y10 Python files were exactly equal after normalizing the year tag. In the 15 September GitHub snapshot, CCL/Y1/Single alone additionally wraps the CCL result in a float64 NumPy array; the other pairs retain equivalent survey-generic source. JAX CPU/GPU differ in device selection and shared backend usage, not the survey physics.
- The default grid is 351 redshift nodes from 0 to 3.5 and 21 logarithmic multipole nodes from 20 to 2000. The CCL timing scripts evaluate 20 geometric bin centers; LimberCloud timing scripts evaluate 21 nodes. See `experiments/spectra/CCL/Y1/single.py:39–69` and `experiments/spectra/NUMBA/Y1/single.py:40–75`.
- Each timing run performs 1,000 cosmology draws and writes cumulative timings after 100, 200, …, 1,000 evaluations. No scientific spectrum arrays are saved. NUMBA `single.py:84–101,199–211`; CCL `single.py:71–86,121–131`; JAX/GPU `single.py:216–228`.
- Configurations are generated by `scripts/generate_config/*.py`, and six corresponding NERSC launchers exist. Cosmology must precede galaxy-bias/IA generation. Galaxy bias is a common redshift function `factor[tag]/growth_factor(z)`, rather than a separate function per tomographic bin. IA is a redshift array computed once at the fiducial cosmology. Magnification slope is per lens bin.
- `tests/test_experiment_contracts.py` checks matrix presence, label aliases and current environment-launcher contracts; `tests/test_environment_contracts.py` and `tests/test_launcher_smoke.py` additionally exercise dotenv/aliases and dry-run launchers with stubs; `tests/test_project_paths.py` checks paths. `tests/test_projection_consistency.py` compares a few element1 values across Numba/JAX; it does not validate full spectra, ell estimators, random cosmologies, component assembly, covariance ordering, or script/notebook equivalence.
- `scripts/validate_notebooks.py` is a lightweight notebook source/path validator. Passing `make check` does not establish scientific rerun equivalence.

## 2. Problems to resolve before sharing outputs

### R1 — The validation estimator differs between CCL and LimberCloud

**Verified, high priority.** In `notebooks/spectra/Y1/EE_Spectrum_Validation.ipynb`, cell 7 evaluates CCL at `ELL_DATA = sqrt(edge_lo*edge_hi)`. Cell 9 interpolates `ell*C(ell)` against `log(ell)`, integrates that spline over each log bin, and divides by `edge_hi-edge_lo`. Algebraically the latter is a uniform-`dell` band average, not a value at the geometric bin center. The same pattern appears in TE/TT and both survey years.

The observed residual consequently mixes radial/interpolation errors with an angular estimator difference. A covariance of a particular bandpower estimator cannot silently be attached to another one.

**Minimal edit:** explicitly choose and record one estimator. For a first clean method comparison, compute both methods at the exact same ell evaluation points. For covariance-normalized bandpower results, apply the same explicit weight/window to both the CCL reference and LimberCloud spectra. Store both raw sampled `C_ell` and separately named bandpowers if both are needed. Supply raw sampled `C_ell` to a covariance interface expecting spectra, not band averages relabelled as samples. Decide the actual OneCovariance weighting using its installed version; do not assume the existing uniform-dell average is its estimator.

**Validation:** constant and known power-law spectra under the adopted window; equal node arrays across backends; estimator label and edge/window hash agreement before comparing residuals or covariance.

### R2 — Missing magnification factor in timed MS contributions

**Verified, high priority if sharing evaluators.** `experiments/spectra/NUMBA/Y1/triple.py:290–295` passes unweighted `lens_phi_grid` for MS; its MI term at lines 297–301 applies `magnification_bias[:, None]`. JAX/CPU Triple has the same MS omission at lines 310–315. The Double and Y10 counterparts inherit the corresponding pattern. In contrast, TE validation notebook cell 9 applies the magnification factor to **both** MS and MI, as do its numerical-reference components in cell 8.

`amplitude_ms` contains only the lensing amplitude squared; the missing factor is not already absorbed there. Current timing results therefore cannot automatically become scientific validation spectra.

**Minimal edit:** the shared component assembly must weight the magnification distribution consistently. Add a component-level check using distinct, non-unit magnification factors, including a zero factor. Verify the MS and MI components vanish when their lens magnification factor is zero; retain galaxy–shear/galaxy–IA contributions.

### R3 — Random cosmology runs combine sampled backgrounds with fiducial factors

**Verified discrepancy; its interpretation depends on intended nuisance model.** NUMBA `single.py:78–82` computes the lensing amplitude using fiducial Omega_m and H, before drawing a new cosmology at lines 102–115. Line 127 uses sampled `cosmology.h_over_h0` with fiducial H. Triple similarly computes amplitudes at lines 108–123 and later samples cosmology at lines 143–155. JAX shares this structure.

The background-to-radial-distribution conversion must use the same cosmology as the distance calculation. The lensing amplitude must use the sampled cosmology if these evaluations are intended to represent valid spectra at that cosmology. IA and galaxy-bias redshift arrays are loaded from fiducial config and kept fixed in **both** implementations; that can represent a deliberate fixed-function benchmark, but it must be distinguished from regenerating an assumed `1/D(z)` model at each draw.

**Minimal edit:** keep fiducial validation deterministic. Before enabling random-cosmology validation, derive H0/Omega_m prefactors from the active cosmology; declare whether nuisance functions are held fixed or regenerated and apply that rule to CCL and LimberCloud. Generate one seeded parameter table and use it for all backends. Current independent `numpy.random.uniform` calls do not produce a matched sample; Single uses ±10% and Triple ±5%. Multiplicative sampling of fiducial zero parameters (wa/Omega_k) leaves them zero, so do not claim coverage of evolving dark energy/curvature from those draws.

### R4 — Current timing stage names do not describe matched boundaries

**Verified.** CCL `single.py:87–110` stops the cosmology timer immediately after constructing a `pyccl.Cosmology` and setting GSL parameters. Tracer creation and `angular_cl`, where lazy background/power preparation may occur, are under CELL at lines 112–119. LimberCloud explicitly evaluates distances and non-linear power before ending COSMOLOGY at `NUMBA single.py:124–138`. The two COSMOLOGY files therefore describe different amounts of work.

`experiments/benchmarks/Y1/benchmark.py:111,115,119` repeats the **CCL total** curve in panels labelled cosmology, coefficient, and projection. The same applies to Y10. This is acceptable only as an explicitly labelled reference total; as written it is easy to misread as a stage-to-stage comparison.

Numba sums four projected components in its timed projection stage (`single.py:167–197`); JAX stores four separate arrays and synchronizes them (`JAX/GPU single.py:180–214`) but does not sum them into a complete EE spectrum there. No script performs the notebook's final bandpower averaging during this timing. Current timing arrays include compilation/cold effects in the first cumulative segment; no explicit warm-up is present. JAX **already** calls `block_until_ready()` on coefficients and projected outputs, so preserve those barriers rather than suggesting they are missing.

**Minimal edit:** report three explicit measurements: (1) cold/compile setup separately; (2) warm coefficient construction plus projection, with component sum and output estimator stated; (3) complete end-to-end evaluation including cosmology and the same output contract. If projection-only is retained, label precomputed coefficient/device residency assumptions. Use `time.perf_counter`; synchronize each measured JAX result. Precompute CCL background/power using verified APIs of the installed CCL version when reporting a separate projection stage. Save counts and per-evaluation stage durations in a sidecar, retain the existing `Time_Single_128*.txt` exports for compatibility, and write new benchmark generations to a distinct run directory. Do not overwrite historical results while claiming unchanged comparability.

### R5 — Flat text products lack sufficient self-description

Spectrum notebooks save `C_CCL_{EE,TE,TT}.txt`, `C_DATA_*`, and `C_DATA1/2/3_*` flattened with default C ordering. Error notebooks reconstruct shape by re-reading survey input files and hard-coded `ELL_SIZE=20`; they do not read ell axes, shape declarations, configuration digests, software versions, or completion status.

**Minimal edit:** one `.npz` per completed survey/observable/run, or `.npy` arrays plus JSON if memory mapping is needed. Numeric `.npz` payloads avoid new dependencies and pickle. Include an optional compatibility exporter for existing text consumers during migration. A notebook must reject stale or incomplete artifacts rather than recompute silently.

Required fields: schema version, survey, backend/device, fiducial/sample ID, git revision and dirty-diff identity if applicable, input/config hashes, effective cosmology and nuisance conventions, package versions, redshift/distance nodes, ell nodes, bin edges/centers/window, bin IDs and orientation, shapes/dtypes, component definitions, full spectra and references, quadrature settings/convergence status, eligible-pair and scale masks, and completion status. Write to a temporary filename in the same directory, then replace atomically after verification. Save the manifest last. Resume only if dependency hashes match.

### R6 — Error notebooks assume covariance vector order and degrade precision

EE error notebook cell 4 assumes block order TT→TE→EE and pair-major, ell-fastest packing. It uses the triangular index `j*(j+1)//2+i` for i≤j, which enumerates `(0,0),(0,1),(1,1),(0,2),…` rather than row-major upper triangle `(0,0),(0,1),(0,2),…,(1,1),…`. TE assumes `lens_index*source_count+source_index`. The correct mapping must come from a verified covariance export contract; positive definiteness alone cannot detect a simultaneous row/column permutation.

Every error notebook reads the entire `MATRIX.ascii` as float32, though it only plots its diagonal. Use a verified float64 covariance reader and canonical data-vector index map shared with the covariance producer; cache the small labelled variance/sigma array for notebooks. Keep full covariance in float64 for eigenvalue/Cholesky and likelihood checks.

Residual calculations set division output to zero where CCL is zero, then subtract one, producing an artificial 100% residual. Covariance ratios use a default one. Replace these placeholders with an explicit undefined/near-zero mask and report absolute and sigma-normalized residuals there. If a joint residual significance is reported, solve using the full covariance and the exact same selected data-vector order.

### R7 — The source comparison itself needs a converged numerical reference

EE spectrum notebook cell 4 implements nested 100-point `fixed_quad` calls and cell 7 repeats them for every ordered bin pair, every ell, and three interpolation orders. These are valuable independent references, but higher interpolation order does not establish quadrature convergence. Linear interpolants have knot structure; a single global Gaussian rule can miss small features.

Move these routines intact first, keeping a legacy-reproduction mode for diagnosis. Then certify the reference by increasing integration accuracy and splitting at interpolation knots or using another converged scheme. Record any differences as changes to the reference, separately from LimberCloud changes. Do not replace the numerical references with the analytic backend itself, which would remove the independent check.

## 3. Why current notebooks can be slow or memory-heavy

### 3.1 Slow computation

For each of six spectrum notebooks, CAMB/CCL setup is repeated. Then three sets of nested quadrature repeat shear-kernel integrals for each component/pair/ell; auto spectra compute symmetric ordered pairs twice. The final Numba component calls each build a dense coefficient tensor. Rerunning plotting cells from a clean kernel therefore triggers expensive setup and loops.

**First improvements:** save final products after each observable/reference method; move these computations into jobs; reuse a cosmology and distributions within one survey evaluation; build CCL tracers once per bin per cosmology instead of twice per pair; use unique auto pairs and mirror only after symmetry is validated. Cache reusable quadrature kernels for identical inputs if profiling shows benefit. Split independent observable/reference-method jobs if restart granularity matters; do not add unrestricted pair multiprocessing on top of 128-thread native libraries.

### 3.2 Coefficient and scratch memory

The coefficient allocation is `(N_z, N_z, N_ell)` float64 in all backends (`numba_backend/ss.py:269–272`, `jax_backend/ss.py:291–294`). One default 351×351×21 tensor is **19.74 MiB**. Holding 12 Triple components is about **236.87 MiB** before allocator/workspace/CCL overhead. This alone does not explain an OOM on a full Perlmutter node.

Scaling is quadratic in radial nodes and linear in multipoles. One tensor becomes 78.73 MiB at 701×701×21, 314.47 MiB at 1401×1401×21, and 3,009.97 MiB at 1401×1401×201. JAX construction also has compiler, device, and temporary-array memory beyond this lower bound. Numba SS contains an n/i/j loop (`ss.py:311–316`), so more grid points increase compute more steeply than the tensor memory alone suggests.

**Minimal policy:** process one cosmology and bounded ell chunk at a time; build/project/release components sequentially for validation and retain the small projected outputs. Save coefficients only for an explicit diagnostic need. A 4- or 8-ell chunk is a tuning candidate, not a guaranteed optimum. For JAX use fixed chunk shapes/padding and masks to avoid recompilation on every remainder size. Keep production benchmarks on their explicitly declared unchunked/chunked mode and rerun after a change. No new sparse tensor algorithm is needed for the initial migration.

### 3.3 Plot rendering is a credible major memory source

Spectrum/error figures allocate grids with 5 inches per bin and export rasterized artists at 512 dpi. For configured Y1 5×5 grids, the 25×25-inch full RGBA canvas alone is about **625 MiB**. Y10 TE 25×50 inches is about **1.22 GiB**, and Y10 TT 50×50 inches about **2.44 GiB**. Renderer buffers and copies can add to this. Source reference: spectrum EE cell 10; TE cell 11; error notebooks cell 6. `pyplot.close` is absent in these notebook figures. These are calculated canvas estimates, **not measured peak RSS or a confirmed cause of the reported crashes**.

Keep sparse line plots vector-based where practical, use a compact paper-sized main figure, put all-pair diagnostics in paginated/smaller figures, use an ordinary screen DPI for notebook display, and close each figure after saving/display. Simply moving spectra calculation out of the notebook will not fix a multi-gigabyte plotting canvas.

## 4. Proposed small file-level change set

Names of new files are proposals.

| Location | Proposed edit | Acceptance condition |
| --- | --- | --- |
| `src/limbercloud/validation/evaluate.py` (new small module) | Shared fiducial/config loading, CCL tracer construction, component assembly, explicit ell estimator; functions return arrays rather than writing plots | Fixed fiducial script equals preserved notebook reference once estimator is matched; MS factor and active-cosmology prefactors tested |
| `src/limbercloud/validation/reference.py` (new) | Extract independent slinear/quadratic/cubic numerical integrals with recorded accuracy parameters | Legacy reproduction first, then convergence table and documented reference changes |
| `src/limbercloud/io/validation.py` (new) | Numeric artifact writer/reader with schema, fingerprints, axes, component/pair labels and completion checks | Deliberately mismatched survey/grid/config fails; round-trip preserves arrays and dtype |
| `experiments/validation/spectra.py` (new) | CLI `--survey`, `--backend`, `--configuration`, `--reference-methods`, `--ell-chunk-size`, `--run-id`, `--resume`; deterministic validation defaults | Can resume completed compatible products without rerunning CAMB/reference loops; no partial file accepted |
| `experiments/validation/spectra.sh` (new) | Perlmutter CPU batch wrapper using existing `load_environment.sh`, CPU module profile, `LIMBERCLOUD_CONDA_ENV`, and runtime path conventions | A small allocated smoke run logs interpreter, versions, affinity, thread counts, timing and peak RSS |
| `experiments/validation/summarize.py` (new) | Produce Y1/Y10 residual table and masks from matched products; optional covariance normalization only when verified covariance exists | Explicit max/median/95th-percentile summaries for all defined points and selected science vector, with count, worst pair/ell and excluded/near-zero count |
| `notebooks/spectra/{Y1,Y10}/{EE,TE,TT}_Spectrum_Validation.ipynb` | Replace production cells with artifact reads and light calculations; retain explanatory markdown and figures | Clean kernel plots from artifacts without importing CAMB, CCL, Numba or JAX |
| `notebooks/error/{Y1,Y10}/{EE,TE,TT}_Error_Analysis.ipynb` | Load labelled covariance diagonal and residuals; no hand-coded offsets; preserve undefined mask | Random-permutation fixture mapped by labels gives identical physical panels |
| `experiments/spectra/{CCL,NUMBA,JAX/CPU,JAX/GPU}/{Y1,Y10}/*.py` | After shared evaluator is verified, call common physics; preserve CLI compatibility and add explicit timing-mode config/metadata/warm-up | Same matched spectra across backends, clear cold/warm/stage boundaries, outputs saved outside timers |
| `experiments/benchmarks/{Y1,Y10}/benchmark.py` | Read stage metadata; stop presenting CCL total as if it were each CCL stage; plot certified same-generation timing products | Headline timing can be traced to precise statistic/stages/hardware/config/sample count |
| Existing `*.sh` launchers | Preserve existing fail-fast activation and dotenv preflight; distinguish logical CPUs from physical threads and record resource provenance | Effective CPU affinity/thread pools agree with requested layout; no accidental backend fallback |
| `src/limbercloud/io/project_paths.py`, `README.md`, `docs/{runtime-tree,nersc}.md` | Document script-produced validation artifacts and stage dependencies; retain canonical paths | One explicit run→validate→covariance→summarize→plot workflow |
| Existing tests plus focused new scientific fixtures | Extend beyond file presence/element1 to component assembly, estimators, schemas and index mapping | Tiny fast tests reject the actual observed failure modes |

Avoid extracting a generalized workflow framework in the first pass. Existing entry-point files can remain thin compatibility wrappers. The shared evaluator can be adopted by Numba validation first, then timing scripts in a second isolated patch.

## 5. Perlmutter-specific execution plan

Official guidance verified on 2026-09-15: Perlmutter CPU nodes have 128 physical cores and 512 GB RAM; GPU nodes have 64 physical cores, 256 GB host RAM, and four A100 GPUs. Slurm `--cpus-per-task` counts logical CPUs; NERSC recommends ordinarily using one OpenMP thread per physical core. Thus current CPU wrappers requesting `-c 128` and setting every pool to 128 should not be described as a verified 128-physical-core run. On GPU wrappers, the same setting can expose 128 software threads for 64 physical cores. Confirm actual affinity on the allocated node. [Architecture](https://docs.nersc.gov/systems/perlmutter/architecture/), [example jobs](https://docs.nersc.gov/jobs/examples/), [affinity](https://docs.nersc.gov/jobs/affinity/).

Use the current centralized `scripts/nersc/load_environment.sh`, `scripts/nersc/modules/cpu.sh` or `gpu.sh`, and `conda activate "${LIMBERCLOUD_CONDA_ENV}"` (defaults to `CosmoConda`). Existing wrappers already use `set -eo pipefail` and no longer source `.bashrc`; preserve this. The dotenv parser loads canonical keys without executing shell code, and exported canonical values take precedence. Preserve the existing validated CCL/CAMB/JAX/CUDA stack and install the checkout only with `python -m pip install --no-deps -e .`. Log interpreter, package versions, module list, git revision, SLURM job ID, actual device and thread-affinity settings. For notebooks, preserve `.venv` and the project kernelspec: `scripts/jupyter/launch_kernel.sh` loads the checkout dotenv on startup. The ordinary global CosmoConda kernel does not do that. Do not recreate an established environment merely to move computation out of notebooks.

For a **candidate full CPU node** setup under the documented SMT=2 allocation convention, one task with 256 logical CPUs and 128 OpenMP/Numba threads corresponds to all 128 physical cores. For a smaller initial validation job, a candidate is one task, 32 logical CPUs and 16 OpenMP/Numba threads; benchmark smaller/larger layouts before production. Alternatively use an explicitly tested no-SMT allocation convention. Do not hard-code `NUMBA_NUM_THREADS=SLURM_CPUS_PER_TASK` while calling the latter a physical-core count. Keep `OMP_PLACES=cores`, test core binding, and record what was actually used. These are proposed configurations, not measured best settings. [Running jobs](https://docs.nersc.gov/systems/perlmutter/running-jobs/).

Set a single intended dominant thread pool for each workload. For a Numba reference/validation worker, start with bounded Numba/OpenMP threads and BLAS/NumExpr pools at one, then measure; CCL/CAMB may benefit from their own OpenMP threads. Do not introduce multiple Python processes each inheriting all node threads. Current projection `numba_backend/tensor.py:5–6` is a NumPy `einsum` call with default contraction optimization, so assigning more Numba threads does not in itself parallelize the entire projection stage. Test `einsum` optimization separately only after baseline correctness and record any timing change.

Keep one GPU per JAX process for the current single-device implementation; merely allocating four GPUs will not use them. The script already requests one GPU and checks JAX CUDA platform by configuration; add a runtime assertion/log. Do not change GPU resource settings and use older timing products as evidence of current performance.

Recommended sequence:

1. On Perlmutter, inspect the known checkout, active environment, exact runtime inputs and previous failed job logs. No recursive scans of shared roots. Read `sacct`/MaxRSS where available to distinguish plotting OOM from compute, device memory, time limit, or environment failure.
2. Run fast tests and a tiny allocated fiducial smoke test with two radial intervals or another mathematically valid small grid, a few multipoles, and two bins. Validate all components and output schema.
3. Run one Y1 full fiducial validation with CCL and Numba; preserve legacy and matched-estimator products separately. Profile elapsed time, peak RSS, tracer/reference/coeff/projection stages.
4. Run Y10, then selected grid/quadrature convergence cases. Save each complete product, with bounded concurrency if independent jobs are used.
5. Run the corrected OneCovariance path only after raw-spectrum/input/output ordering and estimator contracts pass; use successful validation dependencies rather than assuming timing jobs generated its inputs.
6. Produce residual summaries and render all notebooks from completed artifacts with a fresh plotting kernel. Check raster-memory reduction and readability.
7. Run corrected, explicitly defined timing experiments as a separate evidence campaign. A `Run_All.sh` may offer validation and timing actions, but validation must not first require 1,000 random cosmology evaluations or every CPU/GPU configuration.

Whether these jobs are separate Slurm submissions or sequential steps in one approved allocation is an operational choice. Sharing code/configuration is essential; sharing warmed process caches between numerical validation and a purported cold benchmark changes the benchmark meaning and must be avoided or explicitly recorded.

## 6. Additional notebooks and scope

- `notebooks/kernel/*/Redshift_Distribution_Psi.ipynb`: loads distributions and plots; already relatively light. Keep interactive unless schema consistency requires a small reader update.
- `Radial_Distribution_Phi.ipynb` and `Lensing_Kernel_Kappa.ipynb`: instantiate CCL and compare direct kernels/interpolants. Useful secondary migration after the spectrum/error bottleneck is fixed; shared prepared backgrounds/distributions can serve these figures. Do not conflate them with expensive random benchmark sampling.
- `notebooks/power/Matter_Power_Interpolation.ipynb`: uses 351 radial samples, 501 dense samples and four ell values, evaluating non-linear power and drawing multiple figures. Cache these small figure inputs in a script when the shared cosmology preparation is available. Close figures and retain the independent interpolation checks.
- `notebooks/derivation/`: educational/algebraic validation scope; keep independent. Parent audit can assess formula correctness. They should not become an implicit production dependency.

## 7. Evidence gates for claims

- **Science gate:** shared fiducial component spectra, matched ell estimator, correct active-cosmology prefactors and magnification factors; reference convergence and finite arrays.
- **Artifact gate:** exact config/hash/shape/axis agreement, explicit completed status, deterministic rerun and restart behavior.
- **Covariance gate:** verified raw-table and data-vector mapping, float64 symmetry/diagonal/eigenvalue/Cholesky checks, correct survey/noise/selection model. Positive definiteness is necessary in the intended independent noisy vector but not sufficient to certify ordering or physics.
- **Validation gate:** Y1 and Y10 summaries for selected and full diagnostic pairs, with near-zero masking and full covariance significance if claimed. Raw absent artifacts cannot support new numerical residual values in the paper.
- **Runtime gate:** measured peak memory and clear workload-specific resource settings; notebooks rerun without production compute; corrected timings with hardware/stage/cold-vs-warm provenance.

Memory consulted only for continuity of known NERSC/runtime conventions: `MEMORY.md:882–887`, with all current source facts rechecked. The older local checkout had conflicting `ONE_COVARIANCE_ROOT`/`ONECOVARIANCE_SCRIPT` documentation; **GitHub 7d29b2f has resolved this** with canonical `LIMBERCLOUD_ONECOVARIANCE_ROOT` meaning the checkout directory containing `covariance.py`, and deprecated aliases normalized by the shared loader. Do not reintroduce old variables in the new runner.
