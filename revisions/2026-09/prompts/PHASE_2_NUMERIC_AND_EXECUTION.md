# Phase 2 — NUMERIC, the shared evaluator, and saved sequential execution

Copy the prompt below into the remote Cursor implementation task after Phase 1 is accepted.

---

Implement Phase 2 in the remote LimberCloud code checkout. This phase turns the scientific contracts and helper modules into one working evaluation, checkpoint, and timing pipeline. Complete the implementation and bounded tests described here, then stop at the Phase 2 handoff. Do not advance to covariance, allocated pilot campaigns, or production automatically. Do not commit or push unless the user separately authorizes it.

## Entry gate and ownership

Read `revisions/2026-09/CODE_REVISION_PLAN.md`, the current Phase 1 prompt under `prompts/`, its report under `reports/`, `reports/C01_SCIENCE_ARTIFACT_STAGE_REPORT.md`, and the current code before editing; shortened paths here are relative to `revisions/2026-09/`. Record the starting commit, dirty files, environment, and paper pin from `git ls-tree HEAD manuscript`. Keep `manuscript/` absent/deinitialized: do not initialize it, run `git -C manuscript`, or edit manuscript sources. Preserve unrelated work. Remote Cursor owns runtime implementation; local Codex owns the revision plans and manuscript handoff.

Require accepted Phase 1 evidence for eta_IA=0.0, the corrected NN assembly with all required endpoint terms, retained observer factors 1/12, 1/12, 1/4, the shared 21-node natural-spline angular estimator, and NUMERIC-only radial interpolation identity. These are author decisions; do not ask again whether to adopt them. If their implementation or evidence is absent, report the exact missing entry condition and continue independent Phase 2 work that cannot invalidate it. Do not publish comparable scientific products until the entry conditions hold.

Inspect helpers rather than assuming their names establish functionality. At the audited `13a3c2d`, `validation/evaluate.py` only sums supplied arrays; drivers ignore `--run-config`/`--mode`, refuse resume, do not evaluate the fiducial, and write no spectra. The artifact helpers are not an integrated runner.

Preserve the master plan's entry/exit source manifests and reconstructable snapshots, returning the incremental phase diff even if HEAD remains unchanged from Phase 1. Capture relevant staged/unstaged/untracked source without secrets/runtime data/paper contents; record reports separately from the producing snapshot.

## 1. Implement one actual sample evaluator

Make `src/limbercloud/validation/evaluate.py` the shared, explicit evaluation interface, with narrowly scoped backend adapters where useful. Inputs must identify survey, configuration, saved sample ID and cosmology row, physical/nuisance data, radial grid, canonical ell nodes, method/device, and NUMERIC options when applicable. Return final requested probe spectra, coordinates/pairs, 20 bandpowers, requested diagnostics, and defined stage durations. Keep plotting and file I/O outside this evaluator.

Prepare the active CCL cosmology and common physical inputs once per sample. Use that sample's h and Omega_m for radial conversion and lensing amplitude. Preserve fixed fiducial IA and galaxy-bias tables across sampled cosmologies; validate their metadata, eta=0, grids, and hashes before use. CCL receives magnification slope s; analytical/numeric magnification terms receive q=5s-2 in the correct lens distribution. Reuse backgrounds/tracers within the requested evaluation where valid. Preserve all component signs, factors and pair orientation.

Use the Phase 1 effective cosmology/solver specification and nuisance-reference provenance. Centralize analytical/NUMERIC power-grid setup with chi in Mpc, k=(ell+1/2)/chi in Mpc^-1 and P in Mpc^3. Evaluate the provider only at positive finite chi; define the observer ordinate from the declared limit instead of the current nan_to_num(infinite k) path. Record actual positive-node k coverage, solver limits and extrapolation; never clamp k or mask failed powers to zero. Test with a fake provider that rejects observer infinity/max-float, and obtain bounded power-support sensitivity evidence. Direct CCL angular evaluation retains its declared reference settings; shared grid construction must not silently change its internal integration policy.

Execute Single=EE, Double=TE+TT, Triple=EE+TE+TT as their actual workloads. Do not run Triple and slice it to claim Single/Double timing. Assemble JAX components into final probes and synchronize their completion. Require exact component shapes and float64; do not permit accidental broadcasting. Map internal `(bin_i, bin_j, ell)` tensors to a documented common `(ell, pair)` representation. Declare symmetric-pair storage and TE lens/source orientation explicitly.

Every method evaluates the same `geomspace(20, 2000, 21)` nodes. Apply the Phase 1 operator to every final probe: natural cubic spline of ell*C_ell versus log(ell), interval integration, division by linear bin width. Return 21 raw samples and 20 bandpowers. The 20 bandpowers are the residual/covariance vector. Do not restore geometric-centre CCL evaluation or the separate 101-point covariance grid.

## 2. Extract the NUMERIC method faithfully

Create `src/limbercloud/projection/numeric_backend/` with one canonical shared implementation, for example `quadrature.py` plus public exports. Extract the numerical functions from zero-based cell 4 of all six `notebooks/spectra/{Y1,Y10}/{EE,TE,TT}_Spectrum_Validation.ipynb` notebooks. Their duplicate definitions should become one maintained implementation.

The radial order interpolates phi(chi), a(chi)=1/(1+z), and component-effective P_ell(chi), including amplitude factors multiplied into power at the nodes before interpolation. Preserve the original lensing/density windows and SS/SN/NS/NN integration. Map public `linear` to SciPy `slinear`; support quadratic and cubic. First reproduce the notebook routines with their inner/outer `fixed_quad(..., n=100)` settings on bounded fixtures. Only then introduce justified quadrature/vectorization improvements, with separate versioned numerical identities and convergence evidence. Cubic is a comparator, not guaranteed truth.

Keep the independent matched-integrand analytical oracle in validation/tests. NUMERIC linear interpolates a rather than 1+z and does not inherit the analytical cubic observer power law; do not silently change it into that oracle. The later angular natural spline is independent of this radial interpolation setting.

Preserve the radial interp1d construction/boundary semantics from the notebook; radial cubic is not the angular natural spline. Validate required node counts, strict radial ordering, support/extrapolation behavior and positive interpolated scale factor. Reject an unsupported order/grid rather than downgrading it. Preserve signed effective component power and record radial boundary settings separately from the angular operator. A curved radial regression must catch accidental substitution of a natural spline.

Validate observer asymptotics for every numerical component before accepting convergence. For example P_eff=chi and both density weights constant at zero give the divergent NN integral of 1/chi, despite finite output from interior Gauss nodes. Distinguish removable endpoint limits from non-integrable inputs; test an integrable zero-at-observer density and a genuinely divergent fixture, and verify the actual survey inputs meet the necessary behavior. Reject divergence with an explicit diagnostic, never an epsilon cutoff, endpoint clamp or silent analytical cubic-power substitution. Unrestricted far-endpoint coefficients remain supported. For production, split quadrature at interpolation knots and moving source limits and demonstrate refinement at inner and outer levels; keep unsplit n=100 runs as labelled notebook reproduction, not automatic convergence evidence.

## 3. Use thin entries and operational arguments

Convert the 24 existing entries under `experiments/spectra/{CCL,NUMBA,JAX/CPU,JAX/GPU}/{Y1,Y10}/` into thin wrappers. Add six wrappers at `experiments/spectra/NUMERIC/{Y1,Y10}/{single,double,triple}.py`, with corresponding launcher integration. There are 30 wrappers and 42 method/order-survey-configuration workloads: four existing methods plus three NUMERIC orders, multiplied by two surveys and three configurations. Do not duplicate science kernels by wrapper or order.

Use a common runner under `src/limbercloud/experiments/`. Each wrapper fixes its method, device, survey and configuration; validate retained `--tag`/`--label` aliases against that identity. Preserve the runtime root and allocation-label meanings. Only NUMERIC accepts `--interpolation linear|quadratic|cubic`. Validate family/order/device combinations in CLI, configuration, artifacts and filename helpers; a NUMERIC token must never appear in CCL/NUMBA/JAX names.

Wire `--run-config`, `--sample-table`, `--run-id`, `--mode`, `--fiducial-only`, `--include-fiducial`, `--sample-count`, and `--resume` into real behavior. Reject unsupported arguments instead of ignoring them. Only an explicitly invoked validation entry or explicitly selected validation mode (through CLI or run config) may default to exactly sample 0. Ordinary workload launchers with no selection fail clearly without writing outputs. `--fiducial-only` evaluates and saves 0, while `--sample-count` counts nonfiducial rows. Honor `--include-fiducial`; reject contradictory or empty selections before writing outputs. Provide a dry-run showing IDs, workloads and paths. A no-work request must not overwrite timing files with empty arrays.

Deliver a versioned run-config schema/public example and persist its fully resolved form. Include physics/nuisance input references and hashes, table/sample selection, survey/configuration/method/device/order, radial/ell/estimator policies, numerical tolerances/quadrature, timing mode, allocation label and run namespace. Store private paths only in external resolved configs. Defaults < config < explicitly supplied CLI for permitted values; argparse defaults must not overwrite supplied config values. Wrapper-fixed identity and frozen resume contracts cannot be overridden. Reject unknown fields, incompatible aliases and contradictory selections before computation/output. A config-only selection is explicit, not a no-work default. Define --mode validation|benchmark: identical inputs yield identical science, axes and artifact schema; both modes save the mandatory per-sample timing record and spectra; validation may add declared diagnostics outside sampled compute timers, while benchmark adds the agreed comparison policy and timing summaries. Freeze/justify numerical tolerances before production comparisons. Test config-only runs, allowed CLI overrides, forbidden identity/resume changes and mode equality of spectra.

Update shell launchers and `Run_All.sh` argument propagation together. Python JAX adapters must verify actual requested CPU/GPU placement and x64 so direct invocation cannot mislabel hardware. Keep production one sequential cosmology loop per job, with existing internal parallelism. No MPI science campaign or task-based cosmology parallelism is requested.

## 4. Freeze the table and implement immutable artifacts

Harden `validation/samples.py`, `io/artifacts.py`, and `io/project_paths.py`. Generate a campaign table once and select pilot/subset rows from that persisted table. Loading/resume must never redraw parameters. Preserve sample 0 without RNG consumption, sorted negative-w0 bounds, and fixed zero WA/curvature. Store actual generation bounds/half-width and complete semantic identity; reject malformed IDs, fiducial flags, parameter order/shapes, nonfinite values, and out-of-contract fixed parameters. Repeated publication may be idempotent for identical content; different content requires a new namespace and must not overwrite an accepted table.

Keep required family paths and basenames:

```text
results/spectra/NUMBA/Y1/<run_id>/
  Spectra_Triple_128_EE.h5
  Time_Triple_128.txt
  Time_Triple_128_SAMPLES.h5
  Manifest_Triple_128.json
results/spectra/NUMERIC/LINEAR/Y1/<run_id>/
  Spectra_Triple_128_LINEAR_EE.h5
  Time_Triple_128_LINEAR.txt
  Time_Triple_128_LINEAR_SAMPLES.h5
  Manifest_Triple_128_LINEAR.json
```

Use the corresponding JAX/CPU and JAX/GPU roots, Y10, configurations, probes, and NUMERIC orders. Inside these roots, isolate checkpoints and writer ownership by configuration and allocation, for example `checkpoints/Triple_128/`; sample+probe alone is insufficient because Single/Triple EE would collide. One writer owns each concrete artifact namespace. A held lock must belong to the exact normalized namespace being written. A valid identity-mismatched shard is a conflict, not corrupt data eligible for overwrite.

Require a full immutable run-contract fingerprint: deterministic compute-source manifest and numerical dependency/build signature, sample table, physical and nuisance input hashes, grids, eta/endpoint policies, method/device/order, dtype, quadrature, estimator, and pair convention. Record environment/hardware and timing boundary metadata. A changed contract cannot resume old checkpoints silently.

Make identity construction acyclic and stable: shared comparison contract, immutable producer/workload contract with exact requested IDs/probes, and append-only execution records are distinct. Hash a deterministic relevant compute-source manifest including staged/unstaged/untracked code; record HEAD/dirty state separately as provenance. Reports/docs, output paths/checksums/status, scheduler job IDs, restart records and downstream covariance/selection references do not enter the spectrum compute fingerprint. Covariance depends on spectra, not vice versa. A report-only edit or relocation of identical inputs cannot invalidate compute compatibility. Resume requires the exact producing workload contract; comparing methods validates shared science while retaining different producer identities. The numerical signature records the versions/builds actually used by that method, including CCL/CAMB, NumPy/SciPy, Numba or JAX/jaxlib/XLA and relevant BLAS/CUDA/compiler/precision settings. Changed numerical dependencies or settings reject silent resume; host/job IDs and timestamps remain segment provenance. Test changed computation/dependency rejection, report-only edit compatibility, equivalent-path relocation and mismatched requested coverage.

Implement a completed sample transaction covering every probe requested by its configuration plus one timing record. Individual probe shards do not make a Triple sample complete. On interruption, preserve completed sample transactions; document treatment of incomplete/orphan shards and avoid counting partial timings twice. Record failures with original sample IDs and errors; never draw replacements. Validate HDF5 structure, actual coordinates versus estimator identity, both pair_i and pair_j, cosmology versus table, ID/fiducial consistency, finite spectra, and finite nonnegative durations. Corrupt readable HDF5 must produce a controlled artifact error, not escape discovery through an unhandled missing-key exception.

Store explicit sampled-node and bandpower datasets: raw `(sample, 21, pair)` and derived `(sample, 20, pair)`, float64, with their own coordinates/edges and estimator metadata. Select fiducial by ID/flag, not position. Do not retain coefficient tensors. Stream consolidation one bounded shard at a time into chunked output rather than stacking the ensemble in memory. Use standard lossless compression and no undeclared reader plugin.

Write temporary products, close/reopen/validate them, and publish by destination-filesystem rename. For cross-filesystem staging, copy to a destination temporary file and verify checksums before rename. Preserve accepted generations on failed replacement; define durability separately from rename visibility. Publish the manifest last only after validating all required products, probe/sample membership, attempted/completed/failed sets, schema and checksums. Reject empty manifests. Partial diagnostic exports have a separate incomplete/diagnostic status and reader mode. They never publish a completed-run manifest or satisfy scientific/campaign acceptance; completion is checked against immutable requested IDs/probes, not the subset that happens to exist.

## 5. Measure and read actual completed computation

Use `perf_counter`. Separate explicit pre-sample startup, compilation, fiducial and warmup from sampled timings. Recompilation or lazy preparation occurring inside a timed sampled call remains charged and diagnosed; never retrospectively subtract it. Retain JAX synchronization; define transfers and include final component assembly and the shared angular estimator in the common completed-computation boundary. Describe CCL lazy initialization accurately instead of asserting stage equivalence from matching names.

Keep prepared cosmology/power and analytical basis state reusable through explicit interfaces with validated identities. Phase 3 will exercise supplied-state coefficient+contraction and changed-distribution fixed-basis benchmarks. A changed cosmology, grid, effective-power nuisance law or endpoint/operator policy must invalidate the relevant cache. A distribution update may reuse only genuinely distribution-independent tensors; rebuild the required distribution weights and affected CCL tracers. Do not cache completed outputs by shape or relabel a full prediction as a contraction.

Store nonoverlapping stage durations once per sample/configuration in `Time_*_SAMPLES.h5`, including sample IDs and restart segment IDs. Derive cumulative 100,200,...,1000 sampled timings by summation; fiducial and warmup do not count. Put all checkpointing/consolidation/output I/O outside compute timers, and record job wall time and I/O separately. Resume scientific work sequentially with warmup outside each segment's timings; label segmented timing. An uninterrupted benchmark claim requires an uninterrupted accepted run.

Update both benchmark readers to load accepted manifests, actual sample counts and timing records, and all NUMERIC orders. Remove interpolation leakage into existing-family filenames. Historical products require an explicit legacy path and must not mix with new accepted data. If CCL total remains a baseline in stage panels, label it as total. Include run identity in plot outputs to avoid overwriting unrelated runs.

## Verification, exit gate and report

Run meaningful bounded tests, including actual parser-to-runner execution with injected small backends, and a tiny real science evaluation where the environment supports it. Do not substitute source-string checks or successful `--help` for execution. Cover all wrapper identities and 42 workload selections without launching the campaign. Demonstrate sample 0 and sampled rows are evaluated/stored; shapes/pairs and 21->20 coordinates agree; natural-spline reproduction uses a curved fixture that distinguishes boundary conditions; NUMERIC reproduces notebook references and passes bounded convergence checks.

Exercise interrupted sample transactions, restart skipping, malformed shards, wrong-directory locks, concurrent ownership, configuration collisions, changed science identities, table overwrite refusal, missing probes/empty manifests, and bounded consolidation. A fake clock/delayed writer should prove I/O exclusion and correct cumulative counts; verify JAX completion precedes the timer stop. Check raw21/band20 round trips and benchmark fixtures across methods/orders. Record unavailable GPU or site-specific evidence explicitly for Phase 3; do not manufacture it or run workloads on login nodes.

Write `revisions/2026-09/reports/PHASE_2_NUMERIC_AND_EXECUTION_REPORT.md` with starting/ending code identity, dirty patch identity when uncommitted, paper pin, files/functional changes, commands and results, exact tested runtime, tiny evaluated IDs/configurations, schemas and external artifact locations, NUMERIC reproduction/convergence evidence, timing semantics, restart/failure evidence, unresolved implementation limitations, and the precise Phase 3 entry gate. No full 1,001-row campaign, covariance acceptance, manuscript edits, automatic commit/push, or automatic phase advance belongs to this phase.
