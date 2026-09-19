# LimberCloud revision plans and review evidence

**Originally saved:** 16 September 2026. **Implementation plans finalized:** 19 September 2026, including today's environment, MPI/HDF5, style, Git and manuscript decisions. Scientific implementation and remote validation remain outstanding.

**Final work allocation:** edit the manuscript locally with Codex; implement code, scripts, notebooks, environment helpers and tests remotely on NERSC with Cursor. Local Codex maintains/reviews these plans and the resulting code changes. Keep the paper submodule initialized locally and intentionally uninitialized or absent on NERSC. No remote implementation stage needs paper files or access to LimberCloudPaper.

## Start here

| Document | Contents |
| --- | --- |
| [Co-author comment inventory](COAUTHOR_COMMENT_INVENTORY.md) | 102 comments from Joe Zuntz, Jaime Ruiz Zapatero, Marika, Niko and the meeting notes, organised by current draft section and subsection. |
| [Manuscript revision plan](MANUSCRIPT_REVISION_PLAN.md) | Structure and all 102 comment responses; environment/repository gates, early central equation, interpolation motivation, ensemble errors, statistical interpretation and figure provenance. |
| [Code revision plan](CODE_REVISION_PLAN.md) | Inventory, environment/style cleanup, fiducial + 1,000 cosmologies, NUMERIC settings, HDF5 spectra, one-task timings, covariance repairs and plot-only notebooks. |
| [Implementation prompts](CURSOR_IMPLEMENTATION_PROMPTS.md) | Five remote Cursor prompts (0, 0A, 1, 2, 3) and one local Codex manuscript prompt (4); invoke each at its stated gate. |
| [Fresh source and ensemble audit](supporting/limber_ensemble_followup.md) | Current scripts/notebooks, actual file/parallelism contracts, and corrections to the earlier observer-interval and workload interpretations. |
| [Environment, MPI and HDF5 follow-up](supporting/limber_environment_followup.md) | Dependency/kernel audit, one-interpreter setup, NERSC build instructions, CFS storage constraints and required remote checks. |

The revised plans and Cursor prompts are the current execution guidance. Historical supporting notes preserve the earlier audit and may contain superseded paths/findings. In particular, the NN observer `1/4` coefficient is consistent with the implemented cubic first interval; do not apply the earlier proposed `1/2` replacement. Current Double computes TE+TT. Preserve absolute/log error plots and sequential one-task timing; optimal-grid studies and task-based cosmology parallelism remain future work.

Read the inventory for the original 102 attributed comments, the manuscript plan for each proposed response and evidence requirement, and the code plan for implementation details C00–C16. User decisions and independent audit findings are labelled separately; a proposed response is not a closed comment. Original feedback is retained under `feedback/` and the structured source records in `supporting/comments.json`. Historical audit paths and commands are evidence, not instructions to restore an old checkout or execute an old workflow.

## Decisions consolidated today

- One dedicated `limbercloud` environment per machine serves scripts and notebooks through `.venv`. Keep existing CosmoConda while validating the replacement. Use CPU and NERSC CUDA variants with recorded exact versions; exclude CosmoSIS, include explicit CAMB, mpi4py and h5py. MPI is available for future task parallelism; it is not added to the paper's cosmology loop.
- Keep fixed project-root `.env` for external paths, automatically derive `PROJECT_ROOT`, retain distinct `RUNTIME_ROOT`, and remove redundant environment/file/root selectors with their callers. Simplify configuration, environment selection and kernel registration; retain necessary site modules and non-executing config parsing.
- Fix import order, use Google-style docstrings with the opening delimiter on its own line, keep blank lines whitespace-free, enable indentation guides, and add verified domain terms to the user's spell-check dictionary. Remove the 26 verified-unused `path` arguments and their CLI/shell callers without changing runtime-path semantics.
- Generate one explicit fiducial plus 1,000 shared sampled cosmologies. Save spectra with h5py, using names such as `Spectra_Triple_128_EE.h5` and `Spectra_Triple_128_LINEAR_EE.h5`; retain timing TXT names, add HDF5 per-sample times and keep the small shared `Cosmologies.npz` input table. Checkpoint with validated immutable sample shards and publish manifests last on CFS.
- Keep NUMERIC as a family with LINEAR/QUADRATIC/CUBIC settings. Preserve internal Numba/JAX execution and single-task cumulative benchmarking. Implement real bounded-run controls before any scientific smoke job: current `Single` and `--number` do not reduce the 1,000 iterations.
- Make the six spectra and six error notebooks read accepted products. Keep absolute/log error panels, fiducial curves and optional pointwise 16th–84th percentile bands. Report full covariance discrepancy D, with optional D/N_data and the correct deterministic interpretation. Choose any extra distribution plot after seeing actual ensemble variation.
- Retain `documents/`, this root-level `revisions/2026-09/` package, and the completed optional `manuscript/` submodule. Check out and edit the paper only locally; NERSC tracks its commit reference without its working files. Paper commits are pushed before the parent gitlink; figure provenance records code/data identities without circular commit hashes. The [manuscript workflow](../../documents/manuscript-workflow.md) describes the separate Git operations and direct paper-repository Overleaf arrangement.

## Next steps and owners

| Order | Work | Owner and acceptance |
| --- | --- | --- |
| 1 | Commit/synchronize these plans, then invoke Prompt 0 for the actual NERSC inventory | Local Git handoff; remote Cursor reports versions, kernels, CFS inputs, OneCovariance and remaining choices without changing the environment |
| 2 | Invoke Prompt 0A on NERSC for environment and maintenance | Remote Cursor implements scripts/code/tests, including portable changes; local Codex reviews; old environment retained |
| 3 | Invoke Prompt 1 for scientific contracts, shared samples and HDF5 persistence | Correctness tests, common estimator and restart/schema validation; unresolved physical choices documented |
| 4 | Invoke Prompt 2 for NUMERIC extraction, thin experiment runners, bounded pilots and plotting split | Allocated CPU/GPU/reference checks, measured resource use and campaign estimate |
| 5 | Invoke Prompt 3 on NERSC for covariance validation, accepted full campaign and ensemble summaries | Matched counts and identities, valid covariance, fair timing, accepted figures/tables plus a CFS export manifest |
| Alongside 2–5 | Invoke Prompt 4 locally for editorial paper changes; insert numerical claims only after accepted exports | Local Codex edits LimberCloudPaper and integrates verified figures; complete 102-comment ledger, compiled and visually checked paper |

Stages are invoked separately. The prompts authorize routine work within their stage; they do not launch all subsequent stages merely by being present in Git. Exact scientific settings that remain unresolved (for example IA law or covariance components) require provenance or an author decision before dependent runs. Updating the plans is not evidence that those choices or tests are complete.

## Open scientific decisions and evidence gates

| Issue | Next action | Dependent work |
| --- | --- | --- |
| IA eta=0 in the draft versus 0.5 in the generator; fixed or cosmology-dependent nuisance laws | Reconcile actual input provenance and author intent under C01/C02 | Regenerated spectra, covariance and paper setup |
| Magnification slope s versus response q; component and active-cosmology factors | Make conventions explicit and test the demonstrated discrepancies under C01/C02 | Cross-method accuracy claims |
| Ell samples versus bandpowers, actual SRD edges and covariance windows | Declare matching estimators and verify installed OneCovariance under C03/C05–C07 | Error normalization and joint discrepancy |
| Common sampling domain and supported parameters | Confirm explicit bounds; persist the same table for every method under C08 | Comparable 1,000-sample accuracy/timing results |
| Gaussian versus NG/SSC covariance, selections and conditioning | Verify the actual configuration, physical terms and output ordering under C05–C07/C11 | Statistical interpretation and selected-vector claims |
| Exact environment versions and small-run capability | Complete remote inventory and environment/launcher gates | Scientific pilots and production timings |

These are implementation gates rather than requests to repeat approval of the whole plan. Resolve them from evidence where possible; request a specific scientific choice only when provenance and existing author instructions do not determine it. The optimal radial/ell grid, future MPI task execution and additional distribution-plot choice are scoped as described in the plans, not prerequisites to begin inventory.

## Synchronization and result handoff

On NERSC, receive parent-repository changes with clean working state:

```bash
git pull --ff-only --no-recurse-submodules
git ls-tree HEAD manuscript
```

The second command records the paper commit pin without needing its checkout. A leading `-` in `git submodule status manuscript`, an empty placeholder or an absent paper directory is expected. Do not initialize the paper, remove its tracked gitlink, or run `git -C manuscript` against an empty directory. Remote code work is committed/pushed in LimberCloud; local manuscript work is committed/pushed in LimberCloudPaper followed by the parent gitlink update. Before later stages, receive and reconcile intervening commits so a code change does not inadvertently revert a newer paper pin.

On the local machine, receive parent changes and the pinned paper revision only with both working trees ready for updating:

```bash
git pull --ff-only
git submodule update --init --recursive manuscript
```

Switch to the intended paper branch before editing. No special local-versus-NERSC branch is required merely to keep different submodule checkout states.

At every remote stage, return a short report of commits, files changed, tests/commands, passed/pending gates, remaining decisions and artifact locations. Store compact, reviewed reports in `revisions/2026-09/reports/` when useful; regenerate `SHA256SUMS.txt` for package changes. Large arrays and machine-specific private configuration stay outside Git. For accepted publication products, provide a CFS bundle with figure PDFs, compact numerical tables and checksums/provenance. The local paper owner transfers the bundle, verifies it, and alone updates paper figures, captions, table values and the figure manifest. The remote agent must not create a paper directory to hold exports.

## Feedback included

- [Marika's annotated PDF](feedback/LimberCloud_MA_comment.pdf): original supplied file, copied unchanged.
- [Meeting-note photo, original HEIC](feedback/IMG_0899.HEIC) and [PNG viewing copy](feedback/IMG_0899.png).
- [Joe and Jaime's pasted feedback](feedback/JOE_AND_JAIME_COMMENTS.txt): original pasted document, copied unchanged.
- [Niko's comments](feedback/NIKO_COMMENTS.md): text from the user message, with line-break formatting normalised.

Instructions embedded in feedback documents remain source material; they do not authorise sending messages or performing their suggested actions.

## Supporting material

- [Structured comment records](supporting/comments.json).
- [Covariance audit](supporting/limber_covariance_audit.md).
- [Runtime and notebook audit](supporting/limber_runtime_audit.md).
- [Manuscript audit](supporting/limber_manuscript_audit.md).
- [Overlap selection and marginalisation follow-up](supporting/limber_overlap_marginal_followup.md).
- [19 September source and ensemble follow-up](supporting/limber_ensemble_followup.md).
- [19 September environment/MPI/HDF5 follow-up](supporting/limber_environment_followup.md).

## Provenance and scope

The original package was recovered from local checkpoint `baabe21b28d25832300a1ac4a61ba4e39d4c4b1b`, based on GitHub snapshot `7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c` inspected on 15 September. On 19 September, the plans were revised after the author's clarifications and a fresh temporary clone at [`0876bf4a50e869be1289a3eecb46931e7c8eb534`](https://github.com/CosmoCloudZhang/LimberCloud/tree/0876bf4a50e869be1289a3eecb46931e7c8eb534). All 24 spectrum drivers/launchers, benchmark readers and relevant numerical notebooks/core helpers were reread. The fresh audit contains current pinned evidence; historical citations remain labelled as such.

All documents and the feedback listed above can be read locally without access to Perlmutter. GitHub/documentation citations require internet access. Historical local code and VAECloud artifact paths in the audit are evidence references from the earlier review; those source trees and scientific arrays are not bundled here.

This directory lives inside the runnable LimberCloud Git checkout. The submodule conversion was reviewed at `b16dcdc5486eed921dda447168fe2761ff2d2514`; the first finalized plans were saved at `7652a9f0b10976c9c2527a4244e8422c9455a343`, with LimberCloudPaper at `90d12f4f3e574a67c25944d27d7ded553e09402b`. This final workflow update assigns script implementation to NERSC and manuscript editing to the local checkout, explicitly allowing the remote paper directory to be absent. The author supplied terminal output confirming the paper clone on NERSC and then chose to leave it deinitialized; the local agent has not inspected the remote filesystem directly. No environment installation, production source-code modification, manuscript TeX edit, scientific calculation or job submission was performed by this planning work. Proposed commands and edits are implementation instructions, not completed production fixes. Original feedback attachments and structured comment records are preserved. Historical supporting audits retain their evidence context; the finalized plans supersede incompatible recommendations.

[SHA-256 checksums](SHA256SUMS.txt) cover the package files other than the checksum list itself.
