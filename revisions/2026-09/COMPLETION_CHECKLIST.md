# Completion and evidence checklist

Updated 22 September 2026. This is an acceptance ledger template, not evidence that the implementation passes. Populate each row with actual files, commands, environment, source/artifact identities and report links. Allowed statuses: pending, implemented-unverified, passed, failed, blocked-external. Skipped checks are unverified. A report may be complete while its scientific gate remains blocked.

| ID | Phase | Requirement | Required evidence | Initial status |
|---|---|---|---|---|
| R01 | 1 | Restore/preserve paper gitlink and .gitmodules | Correct 160000 pin in reviewed index and, after authorized commit, HEAD; no paper content changes | pending |
| R02 | 1 | Portable loader/module discovery and isolated launcher tests | Bash 3.2/NERSC tests, canonical paths, no real .venv dependency in fixtures | pending |
| R03 | 1–2 | Actual shell forwarding and harmless default execution | Captured final argv; no empty output overwrite; real fiducial 0 operation | pending |
| R04 | 1 | Real kernel/OneCovariance/environment compatibility | Actual interpreter/module/HDF5 operation under supported host; prefix guards; resolved builds | pending |
| R05 | 1 | eta_IA=0 and regenerated inputs | Law/grid/hash tests; IA and galaxy-bias effective-fiducial/solver metadata; current products reject historical 0.5; preserve amplitude/pivot | pending |
| R06 | 1 | NUMERIC-only order identity everywhere | CLI/path/filename/benchmark/artifact invalid-combination tests | pending |
| R07 | 1 | Canonical 21→20 natural operator | Notebook reproduction, identity/axis checks, boundary-condition-sensitive tests | pending |
| R08 | 1 | NN final diagonal and preserved observer 1/4 | Independent/compiled-backend first/final/full-contraction tests | pending |
| R09 | 1 | NS/SN/SS terminal support and stable special cases | All ordinary/terminal oracle cases, transpose/symmetry/zero-power evidence | pending |
| R10 | 1–2 | Matched physical assembly and sampled factors | s/q, MS/MI, IA, active cosmology, separate grids, component isolation; finite positive-chi power calls, units/support/observer limits | pending |
| R11 | 2 | One immutable seeded table | IDs 0..1000, bounds/order/hash validation, overwrite/refusal/resume tests | pending |
| R12 | 2 | True evaluator and 30 thin wrappers | Real sample execution, requested workloads only, device/shape checks; versioned config/precedence and mode-equality tests | pending |
| R13 | 2 | NUMERIC backend | Notebook reproduction, radial boundaries, valid node counts, observer integrability and knot-split convergence | pending |
| R14 | 2 | Workload namespaces/locks/sample transactions | Cross-configuration collision, wrong lock, interruption/new-node recovery tests | pending |
| R15 | 2 | Strict raw 21/band 20 HDF5 and manifests | Separate shared-science/producer/execution identities, stable source scope, full coverage, corrupt/missing probes, bounded consolidation | pending |
| R16 | 2 | Honest per-sample timing/readers | I/O exclusion, one record/sample, counts, synchronization, legacy separation | pending |
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

For each row record a compact evidence block: requirement ID; status; implementation paths; test/producer command; selected interpreter/modules; result/tolerances; producing commit or base+captured patch; input/run/sample/operator/covariance/selection hashes where applicable; output location/checksum; unresolved dependency and responsible next action.

Every phase also returns entry/exit source manifests and a reconstructable incremental diff, including relevant uncommitted changes. Record named readiness separately: spectra_production_ready, selected_gaussian_analysis_ready, onecovariance_adapter_verified and full_covariance_analysis_ready. A project-Gaussian route can pass its own analysis gate while R17/upstream or full-component evidence remains pending. Spectra production requires its physics/execution/pilot rows; selected statistics additionally require accepted covariance and selection. Never mark an outstanding row passed through this dependency separation.

Phase acceptance requires its necessary rows to pass, including evidence dependencies inherited from earlier phases. A failed or missing physical covariance component may leave only its dependent claim blocked; it cannot be silently called a completed full-covariance result. An incomplete campaign or missing continuous timing evidence cannot be replaced by success on a smaller subset. All independent implementation/documentation work should continue while such limits are accurately recorded.

External publication is not a completion requirement of this implementation/editing plan. Git pushes/merges, Overleaf synchronization, coauthor messages and paper submission require their separate explicit request. Preserve the review boundary and actual scientific producing revision even after later documentation or paper commits.
