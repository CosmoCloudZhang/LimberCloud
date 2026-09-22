# Completion plan: corrected science and reproducible project release

**Revised 22 September 2026 against `13a3c2d` and the author's latest instructions.** This is the authoritative plan for new Phases 1–5. It supersedes the 19 September sequence and conflicting provisional choices in historical audits/reports. Reports remain records of what was attempted, not acceptance of unfinished work. Remote Cursor owns implementation and NERSC execution; local Codex owns plans/review and the separate paper repository.

Read the [current audit](reports/2026-09-22_IMPLEMENTATION_REVIEW.md), [prompt index](CURSOR_IMPLEMENTATION_PROMPTS.md), [completion checklist](COMPLETION_CHECKLIST.md), [manuscript plan](MANUSCRIPT_REVISION_PLAN.md), and [102-comment inventory](COAUTHOR_COMMENT_INVENTORY.md). The [old code plan](supporting/CODE_REVISION_PLAN_2026-09-19_ARCHIVE.md) preserves detailed historical evidence; its provisional science/sequence instructions no longer govern. [Author-supplied comments](supporting/AUTHOR_DECISIONS_2026-09-22.md) are preserved separately.

## 0. Audited state and new sequence

| Revision | What exists | Remaining work |
|---|---|---|
| `c0cf5fa` | 19 September plans; paper gitlink `90d12f4f3e574a67c25944d27d7ded553e09402b` | Historical aggregate-review baseline |
| `d664420`, `cb3e567` | Dedicated remote environment, CPU/NERSC recipes, config-loader migration, unused-argument removal, sample-control scaffolding; allocated smoke results reported | Launcher forwarding, portable tests, real kernel setup and other phase-0 follow-ups |
| `13a3c2d` | Sample helpers, physics naming/selected runner fixes, scalar NN oracle, estimator identities, checkpoint utilities | True evaluator/NUMERIC execution, fiducial evaluation, HDF5 runner wiring, common ell, eta decision, endpoints, complete identities/transactions |
| `13a3c2d` parent tree | `.gitmodules` remains; manuscript gitlink deleted | Restore exact gitlink without touching local paper files |

Retain useful work and the validated environment. A helper unit test does not prove drivers use the helper, a `--help` check does not prove a fiducial runs, and an environment smoke is not scientific acceptance.

| New phase | Scope | Exit evidence |
|---|---|---|
| **1: Foundation corrections** | Phase-0 follow-ups/gitlink, eta=0, NUMERIC-only interface, natural 21→20 estimator, complete NN/NS/SN/SS endpoints, typed scientific identities | Regression tests, coherent current contracts, real selected-environment checks |
| **2: NUMERIC and execution** | Direct numerical backend, actual shared evaluator/adapters, 30 thin wrappers, sample/fiducial loop, transactional HDF5, fair timing/readers | Bounded end-to-end execution, notebook reproduction, restart/failure/identity checks |
| **3: Covariance and pilots** | Matching covariance windows/physics/labels, selected vectors, allocated CPU/GPU/NUMERIC pilots and cost estimates | Accepted covariance and pilot evidence; launch matrix for all 42 workloads |
| **4: Campaign and analysis** | 42×1,001 evaluations, summaries/D, continuous timing evidence, lightweight plots and export | Matched counts/identities, numerical evidence and complete checked CFS bundle |
| **5A: Release and handoff** | Reproducibility/package/docs/notebook checks, evidence/claim audit, final bounded repairs | Requirement-by-requirement completion, reproducible commands, verified handoff |
| **5B: Local paper completion** | Editorial work may start earlier; final accepted figures/claims, 102-comment ledger, compile/page review | Reviewed manuscript and provenance; publication remains separately requested |

Old Prompt 0/0A are completed attempts requiring targeted follow-up; old Prompt 1 is partial implementation. Do not execute old Prompts 2–4 alongside this sequence. Each new prompt authorizes routine edits/tests and its stated allocated work when invoked; stop at its report, not automatically at the next phase. Production begins only in explicitly invoked Phase 4 after its required Phase 3 readiness gates pass. Missing allocations block their evidence, not unrelated work.

Return each phase's starting commit, complete diff/file map, tested commands/environment, code/dirty identity, paper pin, artifacts, unresolved items and next gate. Do not automatically commit/push/merge/reset or rewrite history. If Git publishing is separately requested, stage only reviewed paths and record start/end commits. Production must identify a producing commit or an explicitly captured, hashed scientific patch over it; a dirty boolean alone is insufficient. No routine reapproval of eta, NUMERIC or endpoint decisions is needed.

Consecutive uncommitted phases may share HEAD. At each phase entry and exit, preserve a source-file manifest and a reconstructable snapshot of the relevant tracked, staged, unstaged and untracked code/configuration/notebooks. Provide the incremental entry-to-exit phase diff as well as the aggregate diff from HEAD. Include deletions, executable modes and the separately recorded gitlink/index state; exclude secrets, `.env`, environment binaries, runtime arrays and paper contents. Do not use a blanket add/reset/stash to create this review packet. Later report edits are recorded separately from the source snapshot, so the review record does not hash itself.

## 1. Fixed author decisions

1. **Radial interpolation belongs only to NUMERIC.** CCL, NUMBA, JAX CPU/GPU have no NUMERIC-order CLI setting and are not rerun at three orders. Enforce this in parsing, dispatch, identities, paths, filenames, readers and manifests.
2. **Separate NUMERIC family.** Add `experiments/spectra/NUMERIC/{Y1,Y10}/{single,double,triple}.py` and launchers, using one implementation in `src/limbercloud/projection/numeric_backend/`. Extract direct numerical integration from spectra notebooks. Never disguise analytical tensors as NUMERIC.
3. **Current eta_IA=0.0 throughout.** Preserve the distinct `A_IA=0.5` and `z_pivot=0.5`. Regenerate IA and dependent products; do not relabel eta=0.5 files. Historical records remain historical. The adopted value is the requested SRD-based fiducial; paper attribution must cite the actual source without reopening the value.
4. **Include final NN diagonal; retain observer 1/4.** Complete related terminal NS/SN/SS basis cases and derivation annotations. No endpoint clamping to hide missing terms.
5. **Exactly 21 common multipoles and 20 bandpowers.** Every method evaluates float64 `geomspace(20,2000,21)`. Use the notebook's **natural cubic spline** of `ell*C_ell` in `log(ell)`, integrate and divide by linear bin width. The 20 bandpowers form the residual and covariance vector; retain the 21 raw samples only for reconstruction/diagnostics. Centres are display coordinates. Old 20-centre CCL and separate 101-point shared covariance grids are legacy-only.
6. **One matched campaign.** Fiducial ID 0 plus IDs 1–1000 from one persisted seeded table across Y1/Y10 and Single=EE, Double=TE+TT, Triple=EE+TE+TT. Seven method/device/order choices × 2 surveys × 3 configurations = **42 workloads**, each 1,001 evaluations. There are 30 wrappers: 24 existing plus 6 NUMERIC, not 42 science implementations.
7. **Sequential timing.** One process/Slurm task per workload, sequential cosmologies, existing internal Numba/JAX parallelism. No MPI science/multiprocessing/sample arrays. Fiducial/warm-up and I/O are excluded from the 1,000-sample compute totals.
8. **Ownership/scope.** Remote code/scripts/notebooks/environment/tests; local TeX/figure integration/ledger/PDF review. No paper initialization on NERSC. CFS retains accepted outputs; PSCRATCH is staging. No emulator, global grid optimization, full posterior or MPI scaling campaign, unsolicited messages, or Overleaf publication.

## 2. Detailed requirements: stable C00–C16 identifiers

### C00 — Reconcile state and repair the paper reference

Record actual branch/HEAD/status, plan hash, inputs/runtime/config, `.venv` target/modules/device and OneCovariance commit. Reconcile newer remote work without resetting to 13a3c2d. Reports locate code at `/pscratch/sd/y/yhzhang/LimberCloud`, CFS runtime at `/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/LimberCloud`, OneCovariance at `/global/homes/y/yhzhang/opt/OneCovariance` (reported 311c2cf...), while older upstream audits used 91e1396...; verify actual versions/paths.

Restore the mode 160000 manuscript gitlink at `90d12f4f3e574a67c25944d27d7ded553e09402b` from cb3e567/d664420 unless a subsequent deliberate local-owner update is established. Retain `.gitmodules`. Leave remote paper absent; preserve local nested repository/files/changes. Test the parent index/tree, not directory existence. Until an authorized commit records the repair, report both the correct staged gitlink and the still-unrepaired historical HEAD. Do not stage local paper contents into the parent or use `git -C manuscript` as a remote pin check.

### C01 — Resolved physics and IA

Default/current eta=0; remove unresolved-state execution/default 0.5. Accepted campaign rejects missing/nonzero/unresolved eta metadata. Preserve signed NLA, `C1=5e-14/h**2`, comoving rho_m, 0, D(0)=1, A_IA=0.5 and z_pivot=0.5; no physical-density `(1+z)^3` factor or wrong CCL `use_A_ia=False` normalization. Regenerate IA with its redshift axis/law/input hashes. Generic alternate-eta diagnostic fixtures may exist outside current acceptance.

IA/galaxy-bias nuisance functions remain fixed tables generated at the fiducial cosmology, as the current report specifies. Background/power/geometric prefactors vary with each sampled cosmology. Record both identities. Name magnification slope s and response q=5s−2; CCL receives s, weighted distributions receive q once. Preserve separate source/lens grids and validate support/resampling/normalization. Record area, actual distributions, densities and shape-noise convention. Changing physics invalidates dependent spectra/covariance/timing evidence under old identities.

Audit the effective fiducial constructors, not only their primary parameter JSON: current nuisance generators explicitly pass Omega_g and CAMB kmax=100, while the sampled constructor omits Omega_g and uses kmax=50. Record effective radiation/neutrino/temperature conventions, transfer/nonlinear solver settings, CCL/CAMB versions and derived D(z), H(z), chi(z), rho_m(0). Establish the intended common fiducial model and consistent primary-to-derived mapping, or document and validate a deliberately distinct fixed nuisance-reference model. Do not silently remove a difference or preserve it solely because both tables are hashed. IA and galaxy-bias tables both require this provenance and current consumer validation.

### C02 — Components and sampled cosmology

For every sample derive radial H(z)/c, distances, lensing amplitudes and power from that sample. Verify fixes feed all backends and NUMERIC. EE=SS+SI+IS+II; TE=MS+MI+GS+GI; TT=MM+MG+GM+GG, with explicit leg orientation/spin factors. Both MS and MI use q-weighted lens density. Test s=0.4, bin-distinct q, IA/magnification off, mixed transposes, h/Omega_m changes and CCL component totals. Preserve flat/linear-bias/NLA scope. Zero/signed effective component power is valid; formulas dividing by right-node power need zero-safe algebra/limits.

Centralize sampled Limber power setup for analytical/NUMERIC methods: chi in Mpc, k=(ell+1/2)/chi in Mpc^-1 and P in Mpc^3. Call the power provider only at positive finite chi. Set and test the observer ordinate from the declared limiting convention, never by replacing infinite k with max-float or failed powers with zero. Record solver support, actual required positive-node k range and any extrapolation; test bounded sensitivity and never silently clamp k. Shared helpers must retain sample-dependent units and cosmology.

### C03 — One natural-spline angular operator

Implement once in `validation/estimator.py` or one equivalent shared module:

```python
ell = numpy.geomspace(20.0, 2000.0, 21, dtype=numpy.float64)
x = numpy.log(ell)
spline = scipy.interpolate.CubicSpline(x, ell * cl_nodes,
                                      axis=-1, bc_type="natural")
band[b] = spline.integrate(x[b], x[b+1]) / (ell[b+1] - ell[b])
```

This averages reconstructed `C_tilde(ell)=spline(log ell)/ell` uniformly in dell. The 13a3c2d helper uses default not-a-knot, unlike the notebook natural condition. Reject nonfinite/nonpositive/nonincreasing ell, shape/axis mismatches, nonfinite spectra and requests outside support; zero/signed spectra remain valid. Display centres are derived metadata. Persist nodes/edges, transformed variables, spline degree/boundary condition, deltaell normalization, units/axes and operator/schema version in a deterministic hash. Optional 20×21 weights must match direct integration for signed vectors/batches; distinguish in-memory and on-disk axis order.

Test notebook reproduction, natural versus not-a-knot distinction, linearity, exact fixtures where ell*C is constant/linear in logell, and independent integration of the same piecewise polynomial. Constant C or continuum power laws are representation-accuracy checks, not necessarily exact spline identities; do not renormalize the specified operator to force them. Apply the same operator to every method, including CCL and every NUMERIC order.

Only the 20 bandpowers feed residuals, sigma-normalization, selection and D. The 21 raw samples are retained for reconstruction, not as a second science comparison vector. Legacy centres/101-node products require explicit read-only compatibility mode and cannot mix with accepted current products.

### C04 — Complete terminal bases and independent references

Keep cubic observer power P=P1*(chi/chi1)^3 and linear ordinary intervals. Read NN Mathematica originals, exported formulas, all 21 structural-case validation notebooks and both backends. Textual `.nb` expressions/cached outputs may be read without claiming a Mathematica rerun. Update index annotations with their verification history.

For N intervals and N+1 nodes, NN stores all four placements on every interval including final element3. B02/B03 interior-only labels also need terminal coverage. Density is zero outside the domain; terminal nodal density is unrestricted. Excluding a single integration endpoint does not remove its hat over the last interval.

Check NS terminal density row, SN terminal column, and NS/SN/SS terminal lensing support. With final interval [a,b], when a<=x<=b:

```text
H_N(x) = integral_x^b [(u-a)/(b-a)] [(u-x)/u] du
       = [(b-x)^2/2 - a*(b-x) + a*x*log(b/x)]/(b-a).
```

The full-interval expression valid below a cannot be used unchanged inside [a,b]. Some guards protect unavailable chi[n+2], so derive separate terminal cases, not blanket guard removal. Preserve symmetry/transpose and regular limits; clarify whether the ordinary 3/8/10 catalog needs extra boundary branches.

The independent oracle reconstructs hats, source integrals, linear 1+z and the declared power, without calling the closed forms under test. Cover every structural case plus first/final/one-interval, nonzero observer/terminal density, zero/signed power, irregular/narrow intervals and full contractions. Use absolute+relative tolerances and refinement evidence. Check NN Gram-matrix symmetry/PSD for nonnegative effective power and NS/SN transposes; no eigenvalue repair. Regression: chi=[0,1,2], P=[0,1,1], both phi=[0,0,1] gives `3/2−2*log(2)=0.1137056388801094`, not zero. Zero-terminal cases remain unchanged. Execute compiled Numba/JAX tests and bounded real Y1/Y10 before/after checks; earlier scalar checks alone are not backend acceptance.

NUMERIC orders are separate approximation comparisons: they interpolate phi, a and component-effective power. NUMERIC-linear does not automatically match linear 1+z/cubic-observer analytical integrands. Retain a separately named matched-integrand oracle. No global optimal-grid/power-law derivation study is required.

### C05 — Covariance input serialization

Reuse the accepted fiducial CCL full observed-field spectra at the 21 raw nodes, with the current physics. No independent 101-node backend calculation. Point-spectrum interfaces receive raw/reconstructed C_ell, not bandpowers relabelled as samples. Preserve all fields/pairs required for covariance before inference selection; explicit source/lens grids, float64 values, integer IDs, complete `(ell,bin_i,bin_j)` order and full Cartesian pairs when upstream requires them. TE is lens-first/source-second.

Verify the actual installed reader with non-square/multi-bin sentinel round trips; never sort only repeated ell values. Reject duplicate/missing tuples, nonfinite values, ambiguous axes and incorrect EE/TT symmetry. Record input and source identities.

### C06 — Covariance windows and labels

The accepted continuum windows are top hats `W_b=1/deltaell_b` on the exact floating edges. All Gaussian/NG/SSC components must use these windows. Standard mode/ell weighting, integer-rounded boundaries or centre covariance is not automatically compatible; relabelling coordinates does not repair it. Inspect actual upstream mean/noise/component binning and implement a verified adapter or equivalent project calculation. A validated project Gaussian route is permitted, labelled Gaussian; it cannot silently replace a required full-covariance claim.

Reconstruct the fiducial signal from the 21 raw samples with the same natural spline. Internal integration may refine that function for window/covariance quadrature, but it is not a new shared backend grid or comparison vector. Do not treat 21 theoretical samples as independent observed modes. B*Sigma*B^T is valid only when Sigma is the physically defined covariance of B's input random variables with correct windows/mode normalization, not merely because the mean has a 20×21 interpolation matrix.

For field pairs (a,b), (c,d), independently evaluate Gaussian band covariance from

```text
integral W_i(ell)*W_j(ell)
       * [S_ac(ell)*S_bd(ell) + S_ad(ell)*S_bc(ell)]
       / [(2*ell+1)*f_sky] d ell,   S=C+N.
```

State the continuum/full-sky or flat-sky approximation and normalization; do not silently substitute integer-mode windows or an unexplained deltaell. Disjoint top hats have zero Gaussian cross-band covariance under this diagonal-mode approximation. Integrate both legs of connected kernels against the same windows. Validate noise independently, including pure-noise cases.

Use one explicit adapter from stored `(sample,band,pair)` arrays to vectors: select a sample, transpose each probe to `(pair,band)`, flatten with band fastest, and concatenate probes in the declared TT/TE/EE order. Verify multidimensional sentinels against covariance labels; direct flattening of stored band-major data is incorrect. Use explicit output labels `(probe,bin_i,bin_j,band_id)`: row-wise unique EE/TT triangles, full lens-major TE, saved actual order. Test >=3 same-field bins and non-square TE. Conditional full counts are 1100 for 5 lens/5 source Y1 and 2400 for 10 lens/5 source Y10 at 20 bands; selected counts are computed. Never infer pair mapping solely from unchecked triangular arithmetic.

### C07 — Physical covariance and fixed selections

Version effective INI/adapter, upstream commit, command, physical/noise/window hashes and inputs. Prove which files are consumed. Start Gaussian observed-field validation for all probes and cross-probe blocks; check source/lens density units/shape noise. Check observed-field C+N symmetry/PSD at integration nodes. Diagnose significant interpolation/model violations; do not clip them away.

Validate Gaussian, NG and SSC separately: external C_ell tables do not prove matched internal IA/magnification responses. Establish physical equivalence before full-covariance claims; otherwise finish independent work and isolate the affected claim/decision. No silent component deletion. Every component needs labels/shape/finiteness, measured asymmetry, physical-model/window agreement and convergence; check their recorded sum. Positive diagonals/correlation bounds and covariance definiteness apply to the adopted complete covariance and Gaussian reference, with selected-vector factorization and solve residuals where invertibility is required. A connected NG correction need not itself be positive semidefinite; retain signed contributions and do not reject or repair one solely for negative eigenvalues. Apply any separate SSC PSD expectation only when justified by its response-covariance model. No jitter/nearest-PD repair to conceal defects.

For independently sampled disjoint tomographic catalogues, document n_sr=n_arcmin^-2*(180*60/pi)^2, N_TiTj=delta_ij/n_lens_i and N_EiEj=delta_ij*sigma_e,component_i^2/n_source_i. The paper's per-component sigma_e=0.26 has no additional factor 1/2. N_TE=0 requires the declared absence of correlated count/shape noise. Preserve an explicitly different overlap/noise model if supplied. Test conversion and amplitudes from actual survey metadata separately from the Gaussian integration oracle; keep physical white noise out of the noise-free signal spline.

Freeze one selection per survey at the fiducial and save it. Niko/binny GGL rule: source peak above lens peak, normalized minimum-overlap integral<=0.10 (Y1) / 0.25 (Y10), common z grid, declared ties/quadrature, actual pair list. Keep all-pair diagnostics separate and all fields before covariance. The main TT autos and EE unique-pair policy must be explicit in the selected-vector config. Read existing accepted scale cuts and provenance; state k/distance units, representative z and whole-band inclusion rule. The draft's approximate ell numbers are not validated config. If a genuine missing k-cut choice cannot be resolved from author/source inputs, isolate it before production acceptance; do not invent it or choose cuts to reduce residuals.

### C08 — Actual evaluator, numerical backend and CLI

Replace the current assembly-only facade with real one-cosmology execution, retaining the small assembler internally. Inputs explicitly identify physics, survey/configuration, sample row, backend/device or NUMERIC order, grids and requested mode. Return requested EE/TE/TT raw 21 and band 20, coordinates/pairs, timings/diagnostics; no file/plot I/O. Lazy-load required backends; help/readers must not initialize all scientific runtimes/MPI. CCL/NUMERIC need not fabricate coefficient tensors.

Extract one NUMERIC implementation: radial order controls phi(chi), a(chi) and power×component amplitude. Map linear to SciPy slinear. Record separate initial z resampling. Preserve labelled legacy notebook reproduction with nested/outer 100-point fixed_quad, then establish converged quadrature at knots/boundaries under a new recorded numerical identity. Cubic is a comparator, not truth. Share physical inputs/component assembly, not analytical integration routines.

NUMERIC must validate observer integrability from the actual interpolants. With P_eff proportional to chi and both density weights nonzero at zero, NN behaves as 1/chi and diverges; finite Gauss-node output is not evidence of convergence. Require an integrable input/support convention or reject the evaluation with its identity and diagnostic. Do not hide divergence with epsilon cutoffs, endpoint clamping or the analytical cubic observer law. Split production quadrature at interpolation knots and moving source limits, and test finite and divergent cases separately. Notebook reproduction alone does not certify production quadrature.

Preserve the notebook radial interp1d construction and boundary behavior. The angular natural-spline condition does not apply to radial cubic interpolation. Validate minimum node counts, strict ordering, supported domains and positive interpolated a; do not silently downgrade order or clip signed effective powers. Fingerprint radial interpolation/extrapolation boundaries and quadrature separately from the angular operator, with a curved radial regression distinguishing the two spline constructions.

One immutable canonical Cosmologies.npz plus manifest lives under `results/spectra/inputs/<campaign_id>/`. Hash names/order/values/IDs/fiducial flags/bounds/seed/RNG metadata; adopt current ±10% nonzero-primary bounds, sorted for negative w0, fixed zero WA/curvature. Respect the flat/neutrino cosmology contract; log failures, never redraw. Explicit seed is recorded campaign config, not hidden state. All 42 workloads load the same table.

No-argument experiment execution is harmless and writes no empty timing artifacts. A dedicated validation entry may explicitly default to one fiducial. `--fiducial-only` executes ID 0; `--sample-count N` selects N nonfiducial IDs; `--include-fiducial` adds 0; campaign explicitly requests 1000+fiducial. Conflicts fail before expensive work. All Run_All/sbatch/shell chains forward argv intact, including spaces. Keep --number as host allocation label, meaningful tag/label/folder/run/mode/resume semantics, and obsolete --path removed. Test argv reaching the final Python process. Single/Double compute their own workloads.

Provide one versioned run-config schema and public template, with a persisted fully resolved configuration. Defaults precede config values, then explicitly supplied CLI overrides for permitted fields; wrapper-fixed identity and frozen resume contracts cannot be overridden. Config-only sample selection counts as an explicit request. Define explicitly configured or CLI-selected validation and benchmark modes to produce identical physics/coordinates/storage and mandatory sample timing records for identical inputs, differing only in declared diagnostics/comparison policy; extra validation work is outside benchmark timers. Unknown fields, contradictory selections and unsupported order/device settings fail before output creation. Freeze numerical tolerances and their justification before judging production results.

### C09 — Complete identity, transactions and HDF5

Preserve roots and title-case tokens, e.g.:

```text
results/spectra/NUMBA/Y1/<run_id>/Time_Triple_128.txt
results/spectra/NUMBA/Y1/<run_id>/Spectra_Triple_128_EE.h5
results/spectra/JAX/GPU/Y1/<run_id>/Spectra_Triple_128_TE.h5
results/spectra/NUMERIC/LINEAR/Y1/<run_id>/Time_Triple_128_LINEAR.txt
results/spectra/NUMERIC/LINEAR/Y1/<run_id>/Spectra_Triple_128_LINEAR_EE.h5
Time_<configuration>_<allocation>[_<order>]_SAMPLES.h5
```

Retain meaningful existing stage suffixes and document NUMERIC stages. Order tokens are required only for NUMERIC. Benchmark readers include the numerical family and derive stages/counts from accepted artifacts. Legacy readers are explicit; new runs never overwrite historical roots.

Full identity includes schema/algorithm, producing code/patch, physical/input hashes including eta/nuisance/source-lens grids, sample table, radial grid, angular operator, family/device/order, configuration, allocation/timing policy and quadrature. Covariance/selection are derived identities, not prerequisites for base spectra. Validate combinations on construction/read; reject old incomplete schemas or migrate with proof.

Separate the shared comparison contract (table/physics/grids/angular operator/pairs), exact producer/workload contract (method, numerical policy, source snapshot and immutable requested IDs/probes), and append-only execution records. Resume checks the exact workload contract; cross-method comparisons check compatible shared science, retaining each producer's different method/source identity. Hash a deterministic compute-source manifest including relevant staged/unstaged/untracked files; record HEAD as provenance without letting report-only or documentation-only changes break resume. Exclude output checksums/status, job IDs, output locations and downstream covariance/selection references from the compute fingerprint. These records refer to it, not vice versa. Canonical serialization and a declared source-file scope make the hash reproducible and acyclic. Include a method-specific numerical dependency/build signature in producer compatibility (used CCL/CAMB/NumPy/SciPy/Numba/JAX+jaxlib/XLA versions and relevant BLAS/CUDA/compiler/precision settings); changed numerical runtimes cannot resume silently under an unchanged source digest.

Float64 datasets: raw cl `(sample,21,pair)` and bandpowers `(sample,20,pair)`, exact ell/edges/band coordinates, IDs/is_fiducial, parameter names/rows, pair orientation, completion/failures. Fiducial first is convenience, ID is authoritative. Timings occur once per sample/workload, not multiplied per probe. Chunk bounded reads and declare lossless compression. Do not archive all coefficient tensors.

One writer owns each actual workload namespace. Avoid current collisions between Single/Triple or allocations sharing roots: configuration/allocation-specific lock/checkpoint/manifest subpaths or separate workload run IDs under a campaign index. A sample completes only after every requested probe and raw/band representation plus its timing record are validated and a sample commit marker publishes last. A per-probe shard is not a completed Triple sample.

Write temporary shards, close/reopen, validate values/identity/axes/dtype/finiteness/fiducial/canonical cosmology/stages, checksum and rename on the destination filesystem. Cross-filesystem staging copies to a temporary CFS file and verifies before rename. Preserve valid shards and failure history. Resume only identical contracts; do not overwrite a valid identity-mismatched file as if corrupt or silently skip corrupt products.

Locks verify their exact namespace and owner token. Record scheduler job/host/process/start identity and provide race-safe stale-job recovery across nodes; no live-lock theft or reliance on reused PIDs. Consolidation is bounded-memory and validates exact immutable expected ID×probe coverage, both pair axes and rows before publishing the run manifest last. Empty/subset/caller-invented completion is rejected. Partial diagnostic outputs have an explicit incomplete status and reader mode; they never satisfy the accepted completed-run contract. Readers validate semantics as well as checksums. Retain attempted/completed/failed/matched counts, segment IDs and warm-up separation.

### C10 — Lightweight plots and memory

Six spectra and six error notebooks read accepted band 20/fiducial slices and compact summaries without CCL/CAMB/Numba/JAX evaluation imports. Preserve educational markdown/layouts and absolute/log residuals; bound canvas/DPI, paginate all-pair diagnostics, close files/figures and measure RSS. Do not load the ensemble for a fiducial plot. Kernel/power/derivation notebooks retain focused checks using current inputs. Project/release components or chunk ell with stable JAX shapes only as profiling requires; do not introduce unrelated memory redesign.

### C11 — Residuals and fixed-covariance discrepancy

Join by canonical IDs and full physics/operator/pair identities. Use each sample's CCL bandpowers denominator. Save signed and absolute differences, masked/countable near-zero fractional ratios and covariance-scaled residuals. Signed cross spectra are valid. No fabricated 100% fallback. Raw exact zeros remain zero; log display never changes statistics.

Exclude ID 0 from 1,000-sample quantiles. Compute magnitudes before pointwise median/16th–84th bands; label variation over sampled cosmologies, not posterior/observational errors or whole-curve coverage. Save elementwise valid counts, distinct within-vector versus across-sample summaries, tails/worst IDs. Never conceal unmatched/failed samples.

Factor one accepted fiducial covariance per survey/selection once. Compute each `D_s=delta_b^T Sigma_fid^-1 delta_b` by stable solve; full D primary, optional D/N_data secondary. Include cross-probe covariance; do not sum per-probe D or 1,000 cosmologies as independent surveys. Ideal deterministic agreement is 0, not 1. For actual data residual r, chi-square shift is `D−2*r^T Sigma^-1 delta_b`; D alone is not a general likelihood shift. Tables include N_data, attempted/completed/matched counts, fiducial D, median/16th–84th/95th/max and worst IDs for Y1/Y10/methods/orders. Triple provides joint D; Single/Double are declared subvector checks. Verify overlapping configuration spectra agree. Choose extra distribution figures from measured data, not conjecture; report unacceptable discrepancies candidly.

### C12 — Sequential benchmark and campaign

Freeze an immutable execution snapshot including any captured source patch/untracked code so queued/running jobs cannot observe later checkout edits. Give the shared campaign index a single writer or atomic aggregation of immutable workload records. Run 42 accepted workloads with uncontended declared resources, one sequential task each. Full compute timing includes identical final component assembly and angular estimator outputs, with declared host/device transfer/materialization and synchronized JAX. Separate explicit pre-sample cold/fiducial/compile/warm-up costs. Recompilation or lazy work occurring inside a timed sample remains charged and diagnosed; never subtract it retrospectively. Use nonoverlapping per-sample perf_counter stages; do not equate CCL lazy preparation or NUMERIC integration with analytical coefficient stages merely by name.

Derive cumulative 100,200,...,1000 totals from sampled durations only; no rerunning prefixes or counting fiducial/warm-up/checkpoint I/O. Wall time including I/O is separate. Label repeated CCL-total reference lines correctly. Record actual logical/physical cores/affinity/thread pools/BLAS/OpenMP, precision, GPU model/count/memory and environment. Allocation 128 is not proof of 128 physical cores or GPU count.

Resume science with segment provenance/warm-up. Segmented sums are labelled; continuous benchmark claims require uninterrupted accepted runs. Reuse spectra when possible. Phase 3 assesses NUMERIC cost and wall limits before launch; infeasible requirements are reported, not quietly reduced from 1,000 rows.

Separately pilot in Phase 3 and accept in Phase 4 a bounded Y1/Y10 Triple fixed-cosmology benchmark for CCL/NUMBA/JAX CPU/GPU. Use frozen changed-distribution fixtures and repetitions, supplied cosmology/power for coefficient+contraction, and fixed-basis contraction. Compare complete distribution updates, including weight/tracer preparation and the final 20-band outputs, against efficient CCL cosmology/power reuse with affected tracers rebuilt. Report isolated contraction separately, test cache invalidation and uncached agreement, and preserve individual timing/fixture identities. This satisfies manuscript MA43 without expanding the 42×1,001 matrix. At snapshot startup verify actual imported package paths, not merely cwd, so shared editable installs cannot select the live checkout.

### C13 — Preserve environment and close execution gaps

Keep working limbercloud/CosmoConda and reported allocated smokes. `.venv` is sole interpreter selector, fixed .env is nonexecuting external config, PROJECT_ROOT is discovered, RUNTIME_ROOT external. Retain portable CPU/NERSC CUDA recipes, explicit CAMB/h5py/mpi4py, no CosmoSIS; record exact solved packages/modules. Installers must validate the intended prefix and never mutate base/unrelated environments or accept an unintended generic MPI wheel as a site build.

Fix Bash 3.2 module-path discovery and symlink-normalized test expectations. Test in isolated synthetic checkouts/.venv stubs; do not require the user's actual link for launcher tests. Forward science flags and forbid no-op output writes. Verify real kernels inherit required modules/Conda hooks and selected interpreter/checkout. A base limbercloud import probe is insufficient: test a supported-host scientific operation and actual HDF5 read. MPI-linked h5py cannot automatically be called login-safe; use supported allocated kernels/tests or a validated compatible serial build in the same environment. Preserve separate allocated MPI capability checks and CFS locking setup before import. Verify external OneCovariance's actual executable/dependencies, not merely its directory. Retain logs on CFS. Recheck official NERSC guidance when changing site builds.

### C14 — Acceptance ledger

Use [COMPLETION_CHECKLIST.md](COMPLETION_CHECKLIST.md). Every row links implementation, command/test, environment, result, artifact identity and limitation. Skipped science tests/missing allocations are not passes. Run selected-environment make check, shell/notebook checks, focused science tests and clean code-only builds. Before spectra production require matched physics/operator/endpoints, working samples/transactions and accepted pilots; covariance/selection gates additionally govern their dependent analyses. Before claims require complete matched results, covariance/selection, truthful timings and verified exports. Continue independent work around a blocked dependency, but never mark that dependency complete.

Record readiness separately for spectra production, selected Gaussian analysis, the installed OneCovariance adapter, and full NG/SSC analysis. Validated physics/execution/pilots can make spectra production ready while covariance work remains pending. Selected Gaussian analysis additionally needs accepted Gaussian windows/noise/vector/selection; it may use the permitted project backend while the upstream-specific gate stays open. Full-covariance claims need their actual accepted components. Phase 4 must honor these named dependencies under its explicit launch request; do not turn an unrelated failed gate into a blanket halt or call the overall project complete through a fallback.

### C15 — Bounded maintenance

Retain successful argument/import cleanup, preserve JAX initialization/precision, agreed docstrings (summary below opening delimiter; shapes/units in Args/Returns), whitespace and indentation guides. Reconcile scientific spelling settings with requested scope; no unrelated personal settings. Fix concrete touched-file lint/test/docs defects; avoid a broad refactor. Document simple separate commands for dry-run/validation, fiducial, pilot, workload, resume, summary and plotting.

### C16 — Reproducibility and paper handoff

Build/install with absent/empty paper; exclude paper/revision/runtime assets from distributions. Preserve repaired gitlink/.gitmodules and author notebook edits. Update README/documents to actual commands/new phases; no false current capabilities or obsolete subtree publishing.

Build from a clean materialization of the accepted source snapshot, including reviewed uncommitted changes; clean HEAD alone may still contain the old implementation. Record installed-package provenance. Final rendering covers the six spectra and six error notebooks plus both benchmark scripts under experiments/benchmarks/{Y1,Y10}/, using their actual launch paths.

Export accepted publication figures, compact tables/summaries/evidence matrix and checksummed manifest to a labelled CFS handoff. Record producer commands/source commit or captured patch, environment/input/run/sample/operator/covariance/selection identities, counts, acceptance and figure-to-claim map. Full arrays stay on CFS; remote producers never require manuscript/. Verify export bytes and provide transfer instructions.

Local Phase 5B verifies transferred checksums, integrates figures/claims/ledger/provenance, compiles and reviews all pages. Editorial work may precede numerical acceptance. Missing evidence blocks only dependent claims. If Git publication is requested, publish paper then parent pointer; no automatic Overleaf or coauthor messages. Completion means reproducible scientific results and accurate documentation, not merely exit-zero jobs.

## 3. Evidence limits

This planning update audits code at 13a3c2d, local safe checks and historical remote reports; it does not claim a new Perlmutter run, environment change, source repair or TeX edit. Detailed new findings live in reports/2026-09-22_*AUDIT.md. Historical covariance/runtime/environment/ensemble/marginalisation/manuscript investigations retain their original provenance under supporting/. The current plan resolves conflicts; historical scientific measurements are not upgraded into current accepted results.
