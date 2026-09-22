# Updated implementation review and disposition

Reviewed on 22 September 2026 against local parent HEAD 13a3c2d. The review covered source, Git history and configuration; all experiment families and launcher patterns; shared contracts, artifacts and tests; derivation, spectra, error, kernel and power notebooks; covariance paths; documentation; and the existing manuscript and revision requirements. Three parallel reviews examined environment and maintenance, science and artifacts, and covariance and release. This was a planning and review task: production source, environment, gitlink and TeX were not changed.

## Phase 0: retain the foundation, close important gaps

The dedicated environment and loader cleanup are useful, and the remote report records successful tiny allocated CPU/GPU/MPI/HDF5 checks. They do not establish all current execution paths. The new plan repairs issues in place rather than rebuilding a working environment.

| Finding | Assessment | New owner/gate |
|---|---|---|
| All 24 spectra shell launchers omit forwarded user argv | Actual sample/run flags can be ignored; zero-default jobs may produce no science | Phase 1 forwarding, Phase 2 end-to-end tests |
| Default zero-sample driver writes empty legacy timing files | Existing output can be replaced by a no-op | Phase 1 prevent writes; Phase 2 immutable new-run outputs |
| Kernel launch lacks proven required module/activation state | Base-package probe does not prove actual HDF5/scientific notebook behavior | Phase 1 supported-host kernel acceptance |
| Bash 3.2 source-path/realpath fixtures and real .venv-dependent tests | Clean local checkout is not covered by reported remote success | Phase 1 portability/test isolation |
| External OneCovariance source path does not establish interpreter dependencies | Actual covariance invocation needs a tested runtime | Phase 1/3 |
| MPI installer prefix/wheel assumptions and HDF5 login-safety wording | Setup/verification must respect selected prefix and site runtime | Phase 1 |
| Manuscript gitlink deleted in 13a3c2d | This is a Phase 1 regression, not evidence that the original submodule design was wrong | Phase 1 restore exact reference; preserve local paper |

Safe local checks in the detailed environment audit validated 40 notebook structures, parsed 66 Python files and syntax-checked 48 shell scripts. The environment suite ran 12 tests with 4 failures (two equivalent-path fixture comparisons and two module-source-path failures); launcher tests failed because fixtures required a real root .venv. These are not reported as passes. The remote reports describe historical suites of 34 and 52 passing tests in their recorded environments; those results do not erase the locally reproduced failures.

## Phase 1: scaffolding is present; execution and science acceptance remain

| Finding | Consequence | Required repair |
|---|---|---|
| eta 0.5 remains default/unresolved | Current author decision not implemented | Adopt 0, regenerate arrays, validate all consumer metadata |
| Shared angular helper uses default not-a-knot | It does not reproduce notebook natural spline | One versioned natural 21→20 operator for all methods |
| CCL 20 centres versus analytical 21 edges | Arrays are not the same estimator | Common 21 evaluations, band 20 comparisons/covariance |
| NN terminal diagonal omitted; NS/SN terminal density restrictions | Full nodal basis incomplete | Complete independently derived terminal cases; keep observer 1/4 |
| NUMERIC exists as contract text, not backend/experiments | No actual numerical campaign path | Dedicated numeric_backend and six thin wrappers |
| Benchmark/filename interpolation token leaks | Other families can acquire NUMERIC names | Typed method restriction through every layer |
| evaluate_configuration only sums supplied arrays | It is not a sample/backend evaluator | Real adapters and common execution loop |
| Mode/run-config ignored; fiducial-only evaluates nothing | Help text overstates operational behavior | Real parser-to-evaluator-to-writer tests |
| Checkpoints collide across Single/Triple; identity mismatch can overwrite | Scientific data/ownership can be corrupted | Workload namespaces, strict conflicts, sample transactions |
| Incomplete metadata/coordinate/pair/table/manifest validation | Hash/checksum alone does not establish scientific consistency | Full immutable contract and semantic readers |
| Consolidation stacks all shards; stages/probe completion incomplete | Memory/restart/count semantics do not meet campaign design | Streamed consolidation, one timing/sample, full probe coverage |

## Covariance needs estimator and physical verification

Current covariance wrappers still have historical 101 nodes, float32 tables, ell-only sorting and overwritten source-grid risk; error notebooks retain wrong same-field triangular indexing and false-zero fallbacks. At the historical upstream pin inspected in this audit, Gaussian integer-bin weighting and NG/SSC annulus weighting differ from the author's floating-edge uniform-dell estimator. Reinspect the installed 311c2cf... or actual later commit. The plan requires an explicit window adapter, independent observed-field/noise Gaussian tests and truthful component acceptance; changing grid length/labels alone does not resolve this.

The raw 21 theory samples are not independent observed modes. Internal covariance quadrature may refine the canonical reconstruction without introducing a second shared scientific evaluation grid. The 20 bandpowers remain the data vector. Selected-pair/scale rules, covariance physics and resource feasibility require actual inputs and allocated measurements; none is fabricated by this planning task.

## Decisions and deliverables

The author's new eta 0, NUMERIC-only orders, common 21-to-20 angular contract and NN inclusion are fixed in the plan/prompts. Natural boundary conditions and related terminal cases follow source inspection. New phases 1–5A are remote; 5B remains local. The prompts include concrete files, invariants, tests, reports, exit gates, no automatic commit/push, and the final publication bundle/ledger/PDF workflow.

The final cross-check also connects the manuscript's reusable-basis timing claim to a bounded fixed-cosmology/distribution-update benchmark in Phases 3–4. This is separate from the 42 cosmology-varying workloads and compares declared cached-state operations with an efficient CCL baseline. Production snapshots must verify actual package import paths so queued jobs cannot silently use a subsequently edited checkout.

Detailed evidence with source lines and specific repair tests: [Phase 0 audit](2026-09-22_PHASE0_AUDIT.md), [Phase 1 audit](2026-09-22_PHASE1_AUDIT.md), [covariance/release audit](2026-09-22_COVARIANCE_AUDIT.md). Current primary source citations and upstream pin limitations are in the covariance audit. Historical NERSC reports remain unchanged. Original feedback/comment attribution remains unchanged. No phase is marked scientifically accepted merely because the new prompt exists.

## Planning-package verification

The initial documentation checks validated 55 local links across 12 active plans/indexes/prompts, balanced fenced blocks, all 102 original attributed inventory rows against HEAD, unchanged hashes for the five original feedback files and comments.json, and verbatim preservation of the new author attachment. `git diff --check` passed. Changes in this task are confined to revision documentation, the root README and NERSC documentation; the pre-existing local manuscript checkout remains untouched. The package checksum manifest is regenerated after the final edits. These checks establish document consistency and source preservation, not scientific implementation acceptance.

The subsequent [second plan review](2026-09-22_SECOND_PLAN_REVIEW.md) records additional corrections incorporated into the active documents, followed by renewed link/source-preservation/checksum verification. The scientific implementation and external evidence remain future execution work.
