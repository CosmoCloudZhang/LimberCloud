# Phase-1 implementation audit for new editing plans

Audited local HEAD `13a3c2d` on 2026-09-22. Read-only inspection of repository code, tests, C01 report, and `/Users/CosmoCloudZhang/.codex/attachments/7a93f6ad-2086-4cd2-9c36-cd1fa25f4361/Pasted text.txt`. No repository edits, scientific execution, or network access. Paths below are relative to `/Users/CosmoCloudZhang/Project/LimberCloud/`. System Python has no NumPy/SciPy/h5py, so findings are source-grounded rather than newly executed numerical tests.

## Decisions already fixed by author

- Only NUMERIC has a selectable radial interpolation order: linear -> legacy `slinear`, quadratic, cubic. CCL/NUMBA/JAX CPU/GPU do not gain that option or corresponding name tokens.
- NUMERIC gets `experiments/spectra/NUMERIC/{Y1,Y10}/{single,double,triple}.py` and one shared implementation under `src/limbercloud/projection/numeric_backend/`.
- Current fiducial IA slope is eta_IA=0.0 everywhere; historical 0.5 is provenance only.
- All methods evaluate the same 21 geomspace ell nodes from 20 to 2000. A shared natural cubic spline of ell*C_ell versus log(ell), integrated over each interval and divided by linear bin width, produces 20 bandpowers. Those 20 are the residual/covariance vector. Store 21 raw samples for reproduction. Remove the 101-point shared covariance-grid contract.
- Single=EE, Double=TE+TT, Triple=EE+TE+TT. No triple-cache slicing for timing.
- Endpoint mathematics is handled separately by root; retain observer 1/12,1/12,1/4.

## What phase 1 actually implements

Useful foundations exist: sorted multiplicative bounds, seeded persisted sample rows, active-cosmology h/Omega_m prefactors, distinct CCL slope s versus analytical q=5s-2, corrected MS distribution weighting, final JAX component assembly and synchronization, NN interval oracle, and temporary-file/checksum publication primitives.

The C01 report is candid that no scientific driver main() ran and no integrated spectra writer exists (`revisions/2026-09/reports/C01_SCIENCE_ARTIFACT_STAGE_REPORT.md:9,34-55`). Its 52 passing tests establish helper behavior and selected source-string wiring; they do not establish an end-to-end evaluator, accepted science, or restartable campaign.

## Priority repairs and acceptance tests

### 1. Replace the nominal evaluator with real single-sample execution

`src/limbercloud/validation/evaluate.py:19-52` receives already computed components and only sums them. It accepts no cosmology, survey input, grids, backend/device, nuisance data, or NUMERIC settings. The eta decision is discarded at line 47, despite documentation saying it is stored. The optional estimator fingerprint is compared only to another supplied fingerprint: array coordinates/shape are not checked. `assembly.py:159-161` allows NumPy broadcasting between incompatible component shapes.

Repair: implement one actual evaluator that receives a validated request and a saved cosmology row/ID, prepares the active background/distributions/power, dispatches CCL/NUMBA/JAX/NUMERIC, assembles only requested probes, applies the shared angular operator, and returns raw 21-node samples, 20 bandpowers, coordinate/pair metadata, diagnostics and stage durations. I/O remains outside. Keep the current assembly helper as a helper, not proof of execution.

Tests: injected backend spy confirms sample IDs and exact workloads; shifted cosmology changes active prefactors and radial conversion; s=0.4 zeros all magnification contributions while preserving galaxy terms; component shape mismatches rejected; all methods return exactly the same coordinates/pairs/estimator. Tiny real CCL+Numba execution must precede the pilot gate; JAX/GPU evidence must be real allocated device execution.

### 2. Make CLI options operational and safe

All 24 drivers still run their own copied loops. Representative `experiments/spectra/NUMBA/Y1/single.py:26,42-46,108-126,226-230,248-263` rejects resume, reads no fiducial, discards spectra after timing, and writes only TXT. `sample_controls.py:146-170` exposes --run-config and --mode; representative call at single.py:252-263 forwards neither, so both flags are silently ignored. --mode defaults to benchmark. `samples.py:303-344` rejects --include-fiducial and returns [] for --fiducial-only. `sample_controls.py:101-104` describes no sampled rows, while the user reasonably expects a real fiducial evaluation. Zero-work launches still load science files and overwrite timing TXT with empty arrays.

Repair: thin wrappers fix survey/configuration/backend/device and call one runner; validate any --tag/--label aliases against wrapper identity. Wire every exposed flag or reject/remove it until supported. Validation default executes exactly sample 0; campaign explicitly requests 0+1..1000; --fiducial-only genuinely evaluates/stores 0; resume skips validated IDs. Fail selections before expensive imports/data loads/output writes where possible. A dry-run enumerates requested IDs, workload and output paths. Current shell launchers do not forward sample/run flags (`experiments/spectra/JAX/GPU/Y1/triple.sh:57`); update all launchers and Run_All propagation together.

Tests: parser-to-runner integration, not --help alone; IDs 0 and 0/1/2 actually reach backend; inconsistent counts rejected; no silent ignored flags; no-op does not erase results; wrapper label mismatch rejected; shell argument passthrough verified.

### 3. Fix angular estimator implementation, not only metadata

`validation/estimator.py:117` uses CubicSpline's default not-a-knot boundary condition. Notebook recipe explicitly has bc_type='natural' (`notebooks/spectra/Y1/EE_Spectrum_Validation.ipynb:338-341`, zero-based cell 9). The shared helper therefore does not reproduce the stated notebook operator. Existing test `tests/test_science_artifacts.py:241-247` uses a function linear in log ell, where spline boundary choices coincide, so it cannot catch this.

`estimator.py:20,68-85` still defines the 101-point covariance grid. CCL Triple still evaluates 20 geometric centres (`experiments/spectra/CCL/Y1/triple.py:99-104,142-158`). NUMBA/JAX still evaluate 21 edges and never invoke bandpower conversion. `EllEstimator.__post_init__:140-148` checks only name/nonempty length; fingerprints omit spline boundary condition/version and window recipe.

Repair: canonical node and bandpower constructors, explicitly natural cubic spline, positive finite increasing ell validation, finite shape-checked spectra, 21->20 operator on every family, and complete estimator identity (nodes, edges, transform, natural boundary, normalization, axis/pair convention). Historical centres and 101 grids only in explicitly labelled legacy readers if retained. Covariance adapter must consume this exact bandpower vector/window contract; do not merely relabel 101 points as bandpowers.

Tests: curved non-polynomial fixture distinguishes natural from not-a-knot; direct reproduction of notebook formula; multi-axis spectra; identical CCL and tensor coordinates; invalid ell and wrong sizes rejected; persisted fingerprint recomputed from actual coordinates/recipe.

### 4. NUMERIC implementation and interpolation leakage

`validation/reference.py:18-49` currently only describes interpolation; it contains no notebook NUMERIC integration routines. All six spectra notebooks have identical numeric definitions in zero-based cell 4. They interpolate phi(chi), a(chi), and component-effective power (P multiplied by IA/galaxy/lensing amplitude before interpolation), and use fixed_quad n=100 for inner lensing and outer integrals. Exact references: Y1 EE notebook lines 103,112,117,122,127,258-261. Numeric linear is an approximation comparison, not the oracle for analytical linear 1+z plus cubic observer power.

`ProjectPaths.spectrum_results:100-104` correctly requires NUMERIC order and rejects it elsewhere. In contrast all four basename helpers in `io/artifacts.py:508-585` accept any interpolation token without a family. `ArtifactIdentity:180-217` has no validation. Both benchmark readers forward interpolation to every CCL/NUMBA/JAX filename (`experiments/benchmarks/Y1/benchmark.py:102-120`) and never load a NUMERIC folder/curve at all; Y10 reader is identical.

Repair: one numeric_backend implementation extracted/reproduced before numerical improvements; keep independent analytical matched-integrand oracle under validation/tests. Only NUMERIC wrapper/parser and family-aware identity/naming code permit radial order. Use one order per run, not duplicated kernels. Benchmark reader selects each NUMERIC order only for NUMERIC and leaves other filenames unchanged.

Tests: notebook reproduction for all 3 orders/probes/years on bounded inputs; clear reproduction versus improved quadrature identity; convergence checks; forbidden non-NUMERIC order in CLI/identity/name functions; NUMERIC missing/invalid order rejected; benchmark reads actual fixtures for all families/orders.

### 5. Adopt eta=0 and reject stale nuisance artifacts

`contract.py:61-65,83-149` retains unresolved 0.5/0.0 machinery. Generator `scripts/generate_config/intrinsic_alignment.py:54-81,100` defaults to 0.5 and writes unresolved metadata. Drivers read only alignment_info['A'] (`NUMBA/Y1/single.py:75-77`) without validating eta, radial coordinates, nuisance cosmology policy, or config hash.

Repair: adopted current campaign constant 0.0, regenerate external fiducial IA artifact with explicit metadata and checksum, and validate loaded artifact eta/grid/policy before evaluation. Freeze tabulated fiducial IA and galaxy bias across sampled cosmologies as already specified. Historical discrepancy stays only in historical audit text. Do not silently use existing 0.5 JSON because source default changed.

Tests: default generator writes eta=0 resolved/adopted metadata; stale 0.5 or unlabelled nuisance tables fail new-campaign loading; grid mismatch fails; shared evaluator and every family use the same nuisance hash.

### 6. Fix artifact namespace collisions and immutable identity

`ProjectPaths.spectrum_results:72-115` yields family/device/survey/run_id; configuration/allocation are only in final filenames. `checkpoint_name:259-270` is sample ID + probe only. Thus Single and Triple EE checkpoints collide if they share the declared run_id. `write_sample_checkpoint:408-416` treats identity mismatch as invalid and overwrites that path. A changed configuration can therefore destroy another configuration's completed shard. Locks serialize this but do not resolve namespace ownership.

`ArtifactIdentity:195-217` covers only run/survey/family/configuration/table/estimator/device/order. It omits radial grid, input/nuisance hashes, eta/endpoint/science policy, source code/dirty patch, dtype, quadrature and environment. A resumed run can combine scientifically different code/configs without identity rejection. `_require_lock:376-378` checks only any held lock, not that its directory equals the write namespace; a lock on A permits writes into B.

Repair: explicit configuration/allocation-aware checkpoint namespace or distinct immutable run IDs with a campaign grouping identity. Resolve/canonicalize lock path and require the lock owns the exact namespace. Validate identity enums/family-order/device combinations and store a canonical full run-contract fingerprint. A valid mismatched shard is a conflict, never corruption eligible for overwrite. Quarantine invalid shards explicitly; record replacement history. Store code commit/dirty diff, input hashes and numerical policy before any sample executes.

Tests: Single/Triple and allocation collisions prevented; wrong-directory lock rejected; changed nuisance/grid/endpoint/code identity blocks resume; concurrent writers and stale recovery tested; mismatched valid shard remains unchanged.

### 7. Strengthen HDF5 validation, consolidation and manifest publication

`validate_checkpoint_file:340-372` trusts sample/probe/estimator attributes, does not reconcile duplicate sample_id/is_fiducial datasets, validate ID=0 iff fiducial, match cosmology to the canonical table, enforce finite cl/nonnegative timings, compare actual ell to fingerprint, or validate pair semantics. Missing required HDF5 keys raise KeyError rather than ArtifactError; `completed_sample_ids:483-487` catches only ArtifactError, so a structurally damaged readable HDF5 can abort discovery. The test fixture itself binds a 21-node estimator fingerprint to 3 actual nodes (`tests/test_science_artifacts.py:95,102-118`) and is accepted.

`consolidate_probe:618-637` reads all shards and stacks the entire ensemble in memory, checks pair_i but not pair_j, and initially does not verify requested file name/probe/sample against internal record. `_validate_consolidated:679-694` checks little beyond identity, IDs and axis0. It writes only raw cl; sampled/ group has ell only (lines 665-667); 20-bandpower storage is absent. Per-sample timing HDF5 has a naming helper but no writer; stage timings remain duplicated per probe in sample shards and disappear during consolidation.

`write_manifest:728-746` hashes arbitrary supplied files, does not validate required product schema/content or completeness per probe, records status complete even if products empty, and accepts arbitrary completed/failed lists. read_completed_manifest:767-779 accepts an empty product map. Missing required provenance can be hidden in optional extra. Consolidated status remains validated-not-published; define a consistent file/manifest status contract. `publish_file:80-98` uses atomic replace appropriately for rename visibility, but permits overwriting immutable destinations; multi-file publication must keep existing completed generations coherent. If durable crash survival is promised, define flush/fsync guarantees explicitly rather than equating atomic rename with durability.

Repair: strict readers verify required schema, axes/pairs, ID/fiducial/cosmology equality, finite payloads, numerical identity and coordinate fingerprint. Stream one bounded shard at a time into preallocated chunked output. Store explicit raw_nodes and bandpowers groups with axes and identities; designate bandpowers as downstream vector. Write sample timing once per configuration, linked through the manifest. Validate required probes/sample membership, attempted/completed/failed sets and all products before publishing an immutable manifest last. Distinguish publication complete from campaign scientific acceptance. Resume must handle a partially completed Triple sample without double-counting timings.

Tests: missing dataset, wrong pair_j, forged estimator, inconsistent ID/fiducial/cosmology, NaN cl, negative timing, empty manifest, missing Triple probe, incorrect completion list all fail; injected failure at every publication phase leaves prior accepted generation readable and no false completed sample; restart skips valid work and repeats only defined incomplete work; bounded-memory consolidation; raw21/band20 roundtrip exactly reproduces estimator.

### 8. Make shared cosmology tables immutable and self-validating

`save_cosmology_table:369-415` replaces Cosmologies.npz and Manifest.json in an existing directory without namespace ownership or immutability. Re-running generate_samples on the same run_id can silently change every sample. It serializes half_width as global 0.10 (line399) even when generate_cosmology_table(...half_width=...) used another value. load_cosmology_table:436-465 checks checksums/hash but not schema, unique IDs, shapes, finite values, exact parameter order, fixed parameters, bounds or fiducial flags; content_hash omits is_fiducial/bounds/RNG metadata. Frozen dataclass still contains mutable arrays. Parameter-major draws mean changing generated sampled_count changes later columns' sample prefixes; always use the persisted campaign table for pilots/subsets.

Repair: one immutable campaign table created once, idempotent same-content publication or explicit new namespace; store actual generation parameters and complete semantic hash; strict validation; no regeneration during resume. Fiducial model constructor consistency must include the same CCL settings as nuisance generators or explicitly record their fixed-fiducial distinction (generator uses Omega_g/kmax100; sampled constructor uses default Omega_g/kmax50).

Tests: second different table cannot overwrite accepted run; nondefault half-width roundtrip metadata correct; malformed IDs/flags/parameters rejected; pilot selects IDs from the same full table; fiducial consumes no RNG draw; restart never invokes RNG.

### 9. Timing and benchmark readers need operational repair

Drivers use time.time and include first JIT/CCL lazy initialization in first sampled durations (`NUMBA/Y1/single.py:127-184`; `JAX/GPU/Y1/triple.py:177-306`). JAX correctly synchronizes coefficients/components/final sums at lines290-303,395-415, but has no warmup/cold split, host-transfer boundary or bandpower calculation. Actual device selection is shell-only (GPU triple.sh:39); direct GPU Python entry does not assert hardware and could label CPU execution GPU. CCL creation is timed separately from lazy physics/tracers under CELL (`CCL/Y1/triple.py:129-161`), so named stages are not automatically equivalent. TXT carries no sample counts, sample IDs, segment identities or provenance.

Benchmark reader hardcodes 100..1000 (`Y1/benchmark.py:77-81`), uses no manifest, and accepts legacy root implicitly. It plots total CCL time again in every stage panel (`138,142,146`) without a distinct baseline label. NUMERIC is never loaded. Output figure filename has no run ID and can overwrite different runs (`154`).

Repair: perf_counter; explicit cold initialization/fiducial/warmup outside sampled timers; immutable per-sample nonoverlapping stage record once per configuration; final assembly + agreed transfer + 21->20 estimator inside declared compute endpoint; all I/O/checkpoint/consolidation outside compute timers, with total wall+I/O separate. Resume science by ID and timing by segment with rewarmup; continuous benchmark claims require uninterrupted accepted run. Python backend asserts actual JAX device/x64. Readers require manifests for new products, derive counts from accepted sample timing data, validate common sample table/workload/estimator/hardware boundary, and explicitly select historical mode. Label CCL total baseline honestly if retained in stage panels.

Tests: fake clock/delayed writer demonstrates checkpoint I/O excluded; JAX completion awaited before clock stop; fiducial/cold excluded from cumulative N; 100..1000 equals sums of exactly those sampled durations; restart segmented label and no duplicated duration; small-count plots correct; wrong family order/path, missing manifest and mixed table/estimator rejected.

## Suggested ownership across the new five phases

1. Adopt scientific constants/contracts and repair independent kernel/estimator evidence (root supplies endpoint findings). Deliver concrete science tests, regenerate/validate nuisance identity, freeze the canonical sample-table contract.
2. Implement actual common evaluator, standalone NUMERIC extraction, thin wrappers and precise CLI behavior. Demonstrate a real tiny end-to-end fiducial plus sampled row through every available backend, no full campaign.
3. Complete immutable artifacts/restart/timing and reader integration. Failure injection plus real allocated pilot demonstrates persisted raw21/band20, reliable resume, bounded memory and credible cost.
4. Build/validate covariance and inference products strictly on accepted 20-bandpower vectors and identical ordering/windows; validate integration adapter rather than inheriting 101-point assumptions. Then run explicitly authorized accepted sequential campaign and record failures/counts/continuous versus segmented timing.
5. Produce accepted ensemble summaries/figures/tables and reproducible handoff; plot-only notebooks read accepted artifacts. Root may partition covariance/production/manuscript differently, but no later phase should claim a helper-only phase1 is an operational campaign.

This audit is meant to support detailed editing prompts and acceptance gates, not to reopen fixed author decisions or start production edits.
