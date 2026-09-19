# LimberCloud revision plans and review evidence

**Originally saved:** 16 September 2026. **Implementation plans finalized:** 19 September 2026, including today's environment, MPI/HDF5, style, Git and manuscript decisions. Scientific implementation and remote validation remain outstanding.

## Start here

| Document | Contents |
| --- | --- |
| [Co-author comment inventory](COAUTHOR_COMMENT_INVENTORY.md) | 102 comments from Joe Zuntz, Jaime Ruiz Zapatero, Marika, Niko and the meeting notes, organised by current draft section and subsection. |
| [Manuscript revision plan](MANUSCRIPT_REVISION_PLAN.md) | Structure and all 102 comment responses; environment/repository gates, early central equation, interpolation motivation, ensemble errors, statistical interpretation and figure provenance. |
| [Code revision plan](CODE_REVISION_PLAN.md) | Inventory, environment/style cleanup, fiducial + 1,000 cosmologies, NUMERIC settings, HDF5 spectra, one-task timings, covariance repairs and plot-only notebooks. |
| [Cursor implementation prompts](CURSOR_IMPLEMENTATION_PROMPTS.md) | Six prompts: inventory, environment/maintenance, scientific foundation, pilots, final campaign, and manuscript work. |
| [Fresh source and ensemble audit](supporting/limber_ensemble_followup.md) | Current scripts/notebooks, actual file/parallelism contracts, and corrections to the earlier observer-interval and workload interpretations. |
| [Environment, MPI and HDF5 follow-up](supporting/limber_environment_followup.md) | Dependency/kernel audit, one-interpreter setup, NERSC build instructions, CFS storage constraints and required remote checks. |

The revised plans and Cursor prompts are the current execution guidance. Historical supporting notes preserve the earlier audit and may contain superseded paths/findings. In particular, the NN observer `1/4` coefficient is consistent with the implemented cubic first interval; do not apply the earlier proposed `1/2` replacement. Current Double computes TE+TT. Preserve absolute/log error plots and sequential one-task timing; optimal-grid studies and task-based cosmology parallelism remain future work.

## Decisions consolidated today

- One dedicated `limbercloud` environment per machine serves scripts and notebooks through `.venv`. Keep existing CosmoConda while validating the replacement. Use CPU and NERSC CUDA variants with recorded exact versions; exclude CosmoSIS, include explicit CAMB, mpi4py and h5py. MPI is available for future task parallelism; it is not added to the paper's cosmology loop.
- Keep fixed project-root `.env` for external paths, automatically derive `PROJECT_ROOT`, retain distinct `RUNTIME_ROOT`, and remove redundant environment/file/root selectors with their callers. Simplify configuration, environment selection and kernel registration; retain necessary site modules and non-executing config parsing.
- Fix import order, use Google-style docstrings with the opening delimiter on its own line, keep blank lines whitespace-free, enable indentation guides, and add verified domain terms to the user's spell-check dictionary. Remove the 26 verified-unused `path` arguments and their CLI/shell callers without changing runtime-path semantics.
- Generate one explicit fiducial plus 1,000 shared sampled cosmologies. Save spectra with h5py, using names such as `Spectra_Triple_128_EE.h5` and `Spectra_Triple_128_LINEAR_EE.h5`; retain timing TXT names, add HDF5 per-sample times and keep the small shared `Cosmologies.npz` input table. Checkpoint with validated immutable sample shards and publish manifests last on CFS.
- Keep NUMERIC as a family with LINEAR/QUADRATIC/CUBIC settings. Preserve internal Numba/JAX execution and single-task cumulative benchmarking. Implement real bounded-run controls before any scientific smoke job: current `Single` and `--number` do not reduce the 1,000 iterations.
- Make the six spectra and six error notebooks read accepted products. Keep absolute/log error panels, fiducial curves and optional pointwise 16th–84th percentile bands. Report full covariance discrepancy D, with optional D/N_data and the correct deterministic interpretation. Choose any extra distribution plot after seeing actual ensemble variation.
- Retain `documents/`, this root-level `revisions/2026-09/` package, and the completed optional `manuscript/` submodule. Paper commits are pushed before the parent gitlink; figure provenance records code/data identities without circular commit hashes. Update old subtree/Overleaf instructions during implementation.

## Next steps and owners

| Order | Work | Owner and acceptance |
| --- | --- | --- |
| 1 | Commit/synchronize these plans, then invoke Prompt 0 for the actual NERSC inventory | Local Git handoff; remote Cursor reports versions, kernels, CFS inputs, OneCovariance and remaining choices without changing the environment |
| 2 | Invoke Prompt 0A for environment and maintenance | Remote environment/launcher validation; local portable work assigned by file group; one owner per file, old environment retained |
| 3 | Invoke Prompt 1 for scientific contracts, shared samples and HDF5 persistence | Correctness tests, common estimator and restart/schema validation; unresolved physical choices documented |
| 4 | Invoke Prompt 2 for NUMERIC extraction, thin experiment runners, bounded pilots and plotting split | Allocated CPU/GPU/reference checks, measured resource use and campaign estimate |
| 5 | Invoke Prompt 3 for covariance validation, accepted full campaign and ensemble summaries | Matched counts and identities, valid covariance, fair timing, evidence-linked figures/tables |
| Alongside 2–5 | Invoke Prompt 4 for editorial paper changes; insert numerical claims only after accepted results | Local manuscript work in LimberCloudPaper; complete 102-comment ledger, compiled and visually checked paper |

Stages are invoked separately. The prompts authorize routine work within their stage; they do not launch all subsequent stages merely by being present in Git. Exact scientific settings that remain unresolved (for example IA law or covariance components) require provenance or an author decision before dependent runs. Updating the plans is not evidence that those choices or tests are complete.

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

This directory now lives inside the runnable LimberCloud Git checkout, reviewed locally at `b16dcdc5486eed921dda447168fe2761ff2d2514`, with LimberCloudPaper at `90d12f4f3e574a67c25944d27d7ded553e09402b`. The final pass reviewed local source/environment setup and official NERSC/mpi4py/h5py documentation. No connection to Perlmutter, environment installation, production source-code modification, manuscript TeX edit, scientific calculation or job submission was performed. Proposed commands and edits are implementation instructions, not completed production fixes. Original feedback attachments and structured comment records are preserved. Historical supporting audits retain their original evidence context; the finalized plans supersede their incompatible recommendations.

[SHA-256 checksums](SHA256SUMS.txt) cover the package files other than the checksum list itself.
