# September revision package — current completion plan

**Updated 22 September 2026.** The new plan audits Cursor's environment and partial science/artifact implementation at `13a3c2d`, adopts the author's latest decisions, and provides detailed new Phases 1–5 through scientific execution, release and local paper completion. It is an implementation plan, not a claim that the campaign or repairs have already run.

## Start here

1. [Current implementation review](reports/2026-09-22_IMPLEMENTATION_REVIEW.md): what exists, what failed review, and remaining work.
2. [Code completion plan](CODE_REVISION_PLAN.md): fixed decisions, phase map and detailed C00–C16 requirements.
3. [Cursor prompt index](CURSOR_IMPLEMENTATION_PROMPTS.md): five remote prompts plus the separate local paper prompt.
4. [Completion checklist](COMPLETION_CHECKLIST.md): 29 requirements and their actual evidence/status.
5. [Manuscript plan](MANUSCRIPT_REVISION_PLAN.md) and [102-comment inventory](COAUTHOR_COMMENT_INVENTORY.md): detailed paper changes and preserved original attribution.

The [second review and incorporated improvements](reports/2026-09-22_SECOND_PLAN_REVIEW.md) adds precise observer/power/noise checks, effective fiducial provenance, configuration and resume semantics, named readiness gates and reviewable diffs between uncommitted phases. It preserves the agreed science and campaign scope.

## What changed in this version

- Phase 0 is retained with concrete repairs for argument forwarding, harmless defaults, portable tests, real kernel/HDF5 startup and environment checks. The latest commit's removed manuscript gitlink must be restored.
- eta_IA=0.0 is fixed, with regenerated current inputs; A_IA and pivot remain 0.5.
- NUMERIC gets a separate reusable implementation and experiment family. Its radial-order option cannot leak into other methods, filenames or readers.
- NN final diagonal is included, cubic observer 1/4 retained, and related NS/SN/SS terminal cases validated.
- All methods use 21 common raw nodes and 20 bandpowers from the actual notebook **natural** angular spline. Covariance windows must match; old centre/101-grid products are explicitly legacy.
- Execution must really evaluate the fiducial and samples, save spectra, enforce complete identities/transactions and preserve fair timing. Tests of unattached helpers are insufficient.
- Completion covers all 42 matched workloads, a separate bounded fixed-cosmology distribution-update benchmark, summaries/figures, checked CFS handoff, reproducibility and the separate local paper review.

## Execution and ownership

Remote Cursor runs Phase 1 corrections, Phase 2 NUMERIC/execution, Phase 3 covariance/pilots, Phase 4 explicitly invoked production, and Phase 5A release/handoff. Local Codex owns review and Phase 5B paper editing, response ledger, publication-figure integration and PDF compilation/inspection. Editorial work can proceed in parallel; accepted numerical claims wait for the remote evidence.

Stop at each phase's report and review its complete diff. Prompts do not automatically commit/push or advance. Committed work can still be reviewed by comparing stage-start/end revisions; no history reset is needed. Keep `manuscript/` deinitialized on NERSC while retaining the parent gitlink. Preserve the existing local paper repository and author changes. The scientific producing revision/patch remains recorded even when later docs/paper commits advance the project.

## Evidence and historical material

- Detailed new audits: [Phase 0](reports/2026-09-22_PHASE0_AUDIT.md), [Phase 1](reports/2026-09-22_PHASE1_AUDIT.md), [covariance/release](reports/2026-09-22_COVARIANCE_AUDIT.md).
- Existing remote reports remain unchanged: [inventory](reports/C00_REMOTE_HANDOFF_INVENTORY.md), [environment/maintenance](reports/C00_ENV_MAINTENANCE_STAGE_REPORT.md), [partial science/artifacts](reports/C01_SCIENCE_ARTIFACT_STAGE_REPORT.md).
- [Author's 22 September attachment](supporting/AUTHOR_DECISIONS_2026-09-22.md) is preserved verbatim with its later disposition explained.
- [Endpoint derivation review](supporting/limber_numeric_endpoint_review.md) records the earlier scalar evidence and its limits.
- [19 September code plan](supporting/CODE_REVISION_PLAN_2026-09-19_ARCHIVE.md) and [old prompts](supporting/CURSOR_PROMPTS_2026-09-19_ARCHIVE.md) are archived, not executable current instructions. Older supporting audits retain their pinned provenance.
- Original feedback remains in `feedback/`, with the 102-record extraction in [comments.json](supporting/comments.json). No original attributed comment is rewritten by the new author choices.

`SHA256SUMS.txt` covers the current revision package, excluding itself. Verify from this directory with `shasum -a 256 -c SHA256SUMS.txt`. Large runtime arrays are never included. The actual phase reports, acceptance artifacts and paper result remain required; writing this plan does not manufacture them.
