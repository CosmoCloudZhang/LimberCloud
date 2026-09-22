# Completion and evidence checklist

Updated 22 September 2026. This is an acceptance ledger template, not evidence that the implementation passes. Populate each row with actual files, commands, environment, source/artifact identities and report links. Allowed statuses: pending, implemented-unverified, passed, failed, blocked-external. Skipped checks are unverified. A report may be complete while its scientific gate remains blocked.

| ID | Phase | Requirement | Required evidence | Initial status |
|---|---|---|---|---|
| R01 | 1 | Restore/preserve paper gitlink and .gitmodules | Correct 160000 pin in reviewed index and, after authorized commit, HEAD; no paper content changes | passed (index and HEAD already pin `90d12f4`; no paper content touched; commit of this working tree is not authorized here) |
| R02 | 1 | Portable loader/module discovery and isolated launcher tests | Bash 3.2/NERSC tests, canonical paths, no real .venv dependency in fixtures | implemented-unverified (NERSC Bash and isolated fixtures passed; macOS `/bin/bash` 3.2 was not executed) |
| R03 | 1–2 | Actual shell forwarding and fiducial-first execution | Captured supported argv; table required; N=0 runs sample 0, N>0 runs 0..N; Fiducial/Cosmology replacement rules verified | implemented-unverified (forwarding passes; current loops include fiducial; actual integrated execution evidence remains Phase 2) |
| R04 | 1 | Real kernel/OneCovariance/environment compatibility | Actual interpreter/module/HDF5 operation under supported host; prefix guards; resolved builds | implemented-unverified (prefix/installer/OneCovariance unit tests passed; kernel `--science` and the real OneCovariance checkout were not started) |
| R05 | 1 closeout | eta_IA=0 and bound nuisance provenance | Law/grid/content-hash tests; generating fiducial and solver matched to table sample 0; reject stale arrays | failed (historical regeneration evidence retained; current loaders only require nonempty hashes and do not bind them to the selected table/solver; see current review) |
| R06 | 1 | NUMERIC-only order identity everywhere | CLI/path/filename/benchmark/artifact invalid-combination tests | passed |
| R07 | 1 | Canonical 21→20 natural operator | Notebook reproduction, identity/axis checks, boundary-condition-sensitive tests | passed |
| R08 | 1 closeout | NN final diagonal and preserved observer 1/4 | Independent/compiled-backend first/final/full-contraction tests retained in current suite | implemented-unverified (repair present and historical results reported; compiled assembled regression was removed and must be covered in existing science tests) |
| R09 | 1 closeout | NS/SN/SS complete boundary coefficients and stable special cases | NS B09/B10 and SS B11/B12 derivations/exports; corrected Y1/Y10 lensing-kernel branch; compact independent compiled ordinary/final/narrow/one-interval/transpose/zero-power checks after files exist; separate-agent review and terminal.py removal gate | failed (narrow source cancellation and wrong one-interval observer power reproduced; ordinary zero-right-power gap retained) |
| R10 | 1–2 | Matched physical assembly, disabled components and sampled factors | Small activity helper before coefficients; exact zero IA/q and bin-specific legs; shaped zeros/active shared tensors; s/q, MS/MI, current cosmology, grids, zero-right-power linearity branch; finite power calls/units/support/observer limits | pending |
| R11 | 2 | One immutable seeded table | IDs 0..1000, bounds/order/hash validation, overwrite/refusal/resume tests | pending |
| R12 | 2 | True evaluator and 30 thin wrappers | Real sample execution, requested workloads only, device/shape checks; resolved execution specification and internal diagnostic/timed-policy equality tests | pending |
| R13 | 2 | NUMERIC backend | Reproduce existing nested/outer n=100 method; vary inner/outer order separately; retain n=100 if accurate, split knots/increase order only if needed; radial boundaries, valid node counts and actual observer integrability | pending |
| R14 | 2 | Workload namespaces/locks/sample transactions | Cross-configuration collision, wrong lock, interruption/new-node recovery tests | pending |
| R15 | 2 | Strict raw 21/band 20 HDF5 and manifests | Separate shared-science/producer/execution identities, stable source scope, full coverage, corrupt/missing probes, bounded consolidation | pending |
| R16 | 1 text contract; 2 execution | Separate Fiducial/Cosmology timing and honest readers | Two-column sampled counts; N=0 preserves prior Cosmology; tiny/single/nonmultiple/1000 checkpoints; table/count compatibility; synchronized sample timing and I/O exclusion; legend `CCL` with end-to-end stage-reference caption | pending (current files/readers predate the accepted split; revised plan is not implementation evidence) |
| R17 | 3 | Actual upstream covariance input round-trip | Float64 ell-major full pairs; non-square sentinel at installed commit | pending |
| R18 | 3 | Correct covariance windows and noise | Floating uniform-dell window/mean test; independent Gaussian integral plus actual density/dispersion/unit/noise tests | pending |
| R19 | 3 | Physically consistent covariance components | Gaussian/NG/SSC model identities; signed connected-correction checks; complete-covariance factorization/solve/conditioning | pending |
| R20 | 3 | Labels and selected vector | Pair-band sentinel, source-selection list, exact masks/scale units/window | pending |
| R21 | 3 | Allocated 42-workload pilot coverage | Actual CPU/GPU/NUMERIC results, IDs, resolution, memory/time/cost | pending |
| R22 | 4 | 42 complete workloads × 1,001 samples | Immutable expected coverage, requested/attempted/completed/failed/matched ledger | pending |
| R23 | 4 | Residuals, quantiles and D | band 20 identity, zero/near-zero/count tests, covariance solve, tail tables | pending |
| R24 | 3–4 | Fair timing campaign and fixed-cosmology updates | Sampled 100..1000 sums, resources, cold/I/O/segments, continuous evidence; separate supplied-state/contraction/update fixtures and efficient CCL comparison | pending |
| R25 | 4 | Accepted plot-only spectrum/error notebooks | Fresh-kernel render from slices/summaries, no evaluator calls, bounded RSS | pending |
| R26 | 4–5A | Complete CFS handoff | Verified files/checksums/producer identities/claim map; large arrays retained | pending |
| R27 | 5A | Reproducibility/package/docs | Build/install actual reviewed snapshot, absent/empty paper, 12 plot notebooks and 2 benchmark scripts, actual command examples | pending |
| R28 | 5B | 102-comment paper revision | Ledger/source crosswalk, accepted values/figures, consistent claims | pending |
| R29 | 5B | Local paper verification and project walkthrough | Verified bundle, compiled/page-reviewed PDF, final architecture explanation | pending |
| R30 | 1 closeout | Faithful Jupyter editions of all 21 Mathematica derivations | Original .nb retained; source hashes/content-cell map; same scientific content/order/equations/assumptions/cached outputs; readable without Mathematica; notation-impact register traced through txt/validation/backends | pending (conversion and executable-equivalence validation not yet performed) |
| R31 | 1 closeout | Independent mathematical review before terminal-module removal | Separate agent reviews completed old/new case map, expressions/exports, observer/narrow/zero limits and actual dispatch/evidence; resolves findings; remove both terminal.py only after replacement, compiled tests and import checks pass | pending (planning review alone cannot close completed-code acceptance) |

The [current working-tree review](reports/2026-09-22_PHASE1_CONTRACT_REVIEW.md)
and the [Phase 1 report addendum](reports/PHASE_1_FOUNDATION_CORRECTIONS_REPORT.md)
supersede the earlier Phase 1 pass labels. The existing phase sequence remains;
the expanded Phase 1 prompt now owns foundation closeout and Phase 2 consumes
its accepted results. The earlier reported `make check` passed 72 fast tests,
lint, shell syntax and 40 notebook parse checks; those historical results
do not certify the planned notebook editions, compiled endpoint repairs,
allocated science or GPU behavior. Plan edits do not change acceptance statuses.

For each row record a compact evidence block: requirement ID; status; implementation paths; test/producer command; selected interpreter/modules; result/tolerances; producing commit or base+captured patch; input/run/sample/operator/covariance/selection hashes where applicable; output location/checksum; unresolved dependency and responsible next action.

Every phase also returns entry/exit source manifests and a reconstructable incremental diff, including relevant uncommitted changes. Record named readiness separately: spectra_production_ready, selected_gaussian_analysis_ready, onecovariance_adapter_verified and full_covariance_analysis_ready. A project-Gaussian route can pass its own analysis gate while R17/upstream or full-component evidence remains pending. Spectra production requires its physics/execution/pilot rows; selected statistics additionally require accepted covariance and selection. Never mark an outstanding row passed through this dependency separation.

Phase acceptance requires its necessary rows to pass, including evidence dependencies inherited from earlier phases. A failed or missing physical covariance component may leave only its dependent claim blocked; it cannot be silently called a completed full-covariance result. An incomplete campaign or missing continuous timing evidence cannot be replaced by success on a smaller subset. All independent implementation/documentation work should continue while such limits are accurately recorded.

External publication is not a completion requirement of this implementation/editing plan. Git pushes/merges, Overleaf synchronization, coauthor messages and paper submission require their separate explicit request. Preserve the review boundary and actual scientific producing revision even after later documentation or paper commits.
