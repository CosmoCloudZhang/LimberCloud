# Detailed execution prompts: revised Phases 1–5

Updated 22 September 2026 after review of `13a3c2d`, the Phase 0/1 reports and the author's latest comments. Use this sequence instead of the old Prompt 0/0A/1/2/3/4 file, preserved as a [historical archive](supporting/CURSOR_PROMPTS_2026-09-19_ARCHIVE.md). Do not send all prompts as one automatic campaign instruction.

Each linked file contains a complete copy-paste prompt with required inputs, concrete changes, tests, evidence and an exit report. The [code plan](CODE_REVISION_PLAN.md) defines the scientific contract; the [audit](reports/2026-09-22_IMPLEMENTATION_REVIEW.md) explains current gaps; the [checklist](COMPLETION_CHECKLIST.md) tracks completion. Source paths in prompts refer to the repository root; plan/report names are under `revisions/2026-09/` and `revisions/2026-09/reports/` respectively.

The [second review](reports/2026-09-22_SECOND_PLAN_REVIEW.md) is incorporated: NUMERIC observer/radial checks, safe power sampling, effective fiducial provenance, precise covariance/noise tests, config and resume semantics, and phase-specific review packets. The fixed author decisions and 42-workload campaign are unchanged.

| Order | Copy-paste prompt | Where | What must be accepted next |
|---|---|---|---|
| 1 | [Foundation corrections](prompts/PHASE_1_FOUNDATION_CORRECTIONS.md) | Remote Cursor/NERSC | Phase 0 follow-ups, gitlink, eta 0, NUMERIC interface, natural 21→20 operator, all terminal bases |
| 2 | [NUMERIC and saved execution](prompts/PHASE_2_NUMERIC_AND_EXECUTION.md) | Remote Cursor/NERSC | Actual numerical backend/evaluator, 30 wrappers, sample 0 and sampled runs, transactional artifacts, timing |
| 3 | [Covariance and allocated pilots](prompts/PHASE_3_COVARIANCE_AND_PILOTS.md) | Remote Cursor/NERSC | Physical/window/pair/mask covariance checks, 42-workload tiny pilots, bounded distribution-update benchmark and costed production manifest |
| 4 | [Production and final analysis](prompts/PHASE_4_CAMPAIGN_AND_ANALYSIS.md) | Remote Cursor/NERSC | Explicitly invoked 42-workload campaign, 1001 rows each, distribution-update evidence, summaries/timing/figures/export |
| 5A | [Release verification and handoff](prompts/PHASE_5_RELEASE_AND_HANDOFF.md) | Remote Cursor/NERSC | Reproduction, docs/package/notebook/evidence checks and verified CFS bundle |
| 5B | [Local manuscript completion](prompts/PHASE_5B_LOCAL_MANUSCRIPT.md) | Local Codex | 102-comment ledger, verified figures/claims, compiled/page-reviewed paper and final project walkthrough |

Phase 1 includes targeted repairs to the completed Phase 0 work; it does not recreate the working environment. Phase 2 completes the partial old Prompt 1 foundation instead of assuming helper modules establish end-to-end capability. Phase 3 does not launch production. Phase 4 authorizes production only when that prompt is explicitly invoked after accepted pilots. Phase 5B editorial work may proceed earlier; numerical claims wait for accepted exports.

## Fixed decisions the prompts must not reopen

- eta_IA=0.0 throughout current products; amplitude 0.5 and pivot 0.5 are distinct.
- Radial linear/quadratic/cubic only under NUMERIC; no order token or loop for CCL/NUMBA/JAX.
- Common 21 geomspace nodes 20–2000, **natural** cubic spline of ell*C versus logell, 20 uniform-dell bandpowers for residual/covariance vectors. Raw 21 is reconstruction data; centres display only; no separate 101 shared science grid.
- Include final NN diagonal, preserve cubic observer 1/4, complete independent NS/SN/SS terminal checks.
- One shared fiducial+1000 table, 42 workloads, sequential one-task execution, CFS results and local paper ownership.

## How a stage ends

Return a reviewable diff and report before advancing. Include start revision, index/HEAD paper reference, full dirty-patch/source identity, changed files and behavior, exact commands/environment/results, input invalidations, output locations/checksums, attempted/completed/matched counts, unresolved items and next gate. No automatic commit, push, merge, history rewrite, Overleaf sync or coauthor communication. If a commit/push is later requested, use the recorded before/after revisions for a clear phase comparison.

If successive stages remain uncommitted, HEAD is not a phase boundary. Preserve reconstructable entry/exit source snapshots and manifests and provide an incremental phase diff alongside the aggregate HEAD diff. Keep reports, secrets and runtime outputs outside the compute-source fingerprint. The release build must materialize the accepted source snapshot, not merely old clean HEAD.

A report with a missing scientific gate is a useful handoff, not acceptance of that gate. Continue independent authorized work while isolating unavailable external evidence. No routine reapproval is needed for the fixed author decisions or bounded edits/tests in an invoked prompt. Do not invent a missing genuine scale-cut or NG/SSC physics choice; state the exact source/evidence needed for the dependent claim.

Phase 3 reports separate spectra-production, selected-Gaussian-analysis, installed-upstream-adapter and full-covariance readiness. An explicitly invoked Phase 4 follows the gates required by each operation: valid spectra can proceed while an unrelated covariance claim remains pending. Preserve overall partial status and do not treat a project-Gaussian fallback as accepted OneCovariance/NG/SSC evidence.

The remote manuscript remains absent/deinitialized; retain its gitlink and `.gitmodules`. After the accidental deletion is repaired, an uncommitted repair is visible in the index before HEAD. Record both until an authorized commit captures it. Never stage the absent directory as deletion again or replace the local paper checkout. Runtime arrays stay outside Git, and remote publication exports go to CFS rather than manuscript/.

Final acceptance means all necessary [checklist](COMPLETION_CHECKLIST.md) rows have actual evidence and the local paper accurately reflects it. It cannot guarantee favorable numerical results; discrepancies, failed workloads and unsupported physical claims remain visible rather than being hidden to declare completion.
