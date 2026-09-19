# LimberCloud local review package

**Originally saved:** 16 September 2026. **Plans and current-source review updated:** 19 September 2026.

## Start here

| Document | Contents |
| --- | --- |
| [Co-author comment inventory](COAUTHOR_COMMENT_INVENTORY.md) | 102 comments from Joe Zuntz, Jaime Ruiz Zapatero, Marika, Niko and the meeting notes, organised by current draft section and subsection. |
| [Manuscript revision plan](MANUSCRIPT_REVISION_PLAN.md) | Updated structure and all 102 comment responses, including the early central equation, interpolation motivation, ensemble error presentation and statistical interpretation. |
| [Code revision plan](CODE_REVISION_PLAN.md) | Shared fiducial + 1,000 cosmologies, NUMERIC settings, spectra/timing file names, one-task benchmarks, covariance repairs and plot-only notebooks. |
| [Cursor implementation prompts](CURSOR_IMPLEMENTATION_PROMPTS.md) | Four staged prompts for the Perlmutter code agent and manuscript agent, with acceptance gates and evidence requirements. |
| [Fresh source and ensemble audit](supporting/limber_ensemble_followup.md) | Current scripts/notebooks, actual file/parallelism contracts, and corrections to the earlier observer-interval and workload interpretations. |

The revised plans and Cursor prompts are the current execution guidance. Historical supporting notes preserve the earlier audit and may contain superseded paths/findings. In particular, the NN observer `1/4` coefficient is consistent with the implemented cubic first interval; do not apply the earlier proposed `1/2` replacement. Current Double computes TE+TT. Preserve absolute/log error plots and sequential one-task timing; optimal-grid studies and task-based cosmology parallelism remain future work.

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

## Provenance and scope

The original package was recovered from local checkpoint `baabe21b28d25832300a1ac4a61ba4e39d4c4b1b`, based on GitHub snapshot `7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c` inspected on 15 September. On 19 September, the plans were revised after the author's clarifications and a fresh temporary clone at [`0876bf4a50e869be1289a3eecb46931e7c8eb534`](https://github.com/CosmoCloudZhang/LimberCloud/tree/0876bf4a50e869be1289a3eecb46931e7c8eb534). All 24 spectrum drivers/launchers, benchmark readers and relevant numerical notebooks/core helpers were reread. The fresh audit contains current pinned evidence; historical citations remain labelled as such.

All documents and the feedback listed above can be read locally without access to Perlmutter. GitHub/documentation citations require internet access. Historical local code and VAECloud artifact paths in the audit are evidence references from the earlier review; those source trees and scientific arrays are not bundled here.

This directory contains planning material, not the runnable Git checkout. GitHub was read for this update; no connection to Perlmutter, production source-code modification, manuscript TeX edit, scientific calculation or job submission was performed. Proposed commands and edits are implementation instructions, not completed production fixes. Original feedback attachments and structured comment records are preserved.

[SHA-256 checksums](SHA256SUMS.txt) cover the package files other than the checksum list itself.
