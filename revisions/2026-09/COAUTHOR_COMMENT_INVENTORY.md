# Co-author comment inventory in draft order

**22 September author decisions:** eta_IA=0.0 is adopted; radial interpolation orders belong only to NUMERIC; all methods evaluate the same 21 geometric edges 20–2000 and use 20 natural-spline uniform-dell bandpowers for residual/covariance vectors; include NN final diagonal and validate all terminal bases while retaining observer 1/4. These later author instructions supersede provisional scientific choices without changing any of the 102 attributed source records below. Use new remote Phases 1–5A and local [Phase 5B](prompts/PHASE_5B_LOCAL_MANUSCRIPT.md). The updated audit also records the current gitlink regression and incomplete execution/artifact paths.


**Source review date:** 15 September 2026. **Workflow finalized:** 19 September 2026. **Status:** comments synthesised; proposed responses are in [MANUSCRIPT_REVISION_PLAN.md](MANUSCRIPT_REVISION_PLAN.md), and implementation dependencies are in [CODE_REVISION_PLAN.md](CODE_REVISION_PLAN.md). No manuscript, scientific code, figures or production results were changed in this review.

**19 September author clarification:** all 102 source records below retain their original attribution. Updated plans now use one fiducial plus 1,000 matched cosmologies, NUMERIC interpolation settings, saved spectra, sequential one-task timing and absolute/log error plots. Interpolation comparisons test the adequacy of the adopted representation; comprehensive redshift/ell/power optimisation is future work. The early central equation and GitHub link remain requested. These are later author instructions, not additional claims attributed to the coauthors. Local Codex owns manuscript editing, figure integration, comment responses and PDF review; remote Cursor owns code/scripts/notebooks/environment implementation, NERSC execution and evidence export. The remote manuscript checkout is intentionally uninitialized or absent. Read the [fresh source audit](supporting/limber_ensemble_followup.md) and [Cursor prompts](CURSOR_IMPLEMENTATION_PROMPTS.md) with the updated plans; the source table below records the historical 15 September audit.

## 1. Sources and version control

| Source | What was reviewed | Attribution and limits |
| --- | --- | --- |
| [15 September GitHub snapshot](https://github.com/CosmoCloudZhang/LimberCloud/tree/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c) | Repository retrieved read-only into a temporary checkout; implementation, notebooks, configuration, launchers, tests, documentation, all seven TeX sections and main document inspected across the original audit | Historical evidence baseline; the revised plans also incorporate the 19 September review at 0876bf4. |
| Historical local checkout, 15 September | `57c0731cf87562d71a308191deccdba0245cc099`, initially clean `main` | Left on its original commit during that audit; this is not the present checkout. All seven scientific section files and core projection files match the newer GitHub snapshot; environment infrastructure and notebook paths differ. The newer main document also changes the author block; this review does not change authorship. |
| [Pasted feedback document](feedback/JOE_AND_JAIME_COMMENTS.txt) | All actual JZ and JRZ comments | JZ = Joe Zuntz; JRZ = Jaime Ruiz Zapatero, identified in the supplied document and Jaime Ruiz-Zapatero in the manuscript. The user's “Jaimie” is treated as referring to this author. |
| [Marika's marked PDF](feedback/LimberCloud_MA_comment.pdf) | All 37 pages visually inspected, including margins and references | Handwriting is flattened into the page, with no extractable annotation objects. MA entries are faithful summaries, not claimed verbatim transcriptions. Physical PDF page numbers include the title page; printed paper pages generally equal PDF page minus one. A dark-ink note and unlabelled underlines are identified cautiously below. |
| [Meeting notes photo](feedback/IMG_0899.HEIC) | All six numbered notes and the starred covariance note; heading dated 2026-08-21, meeting with Marika | MT entries are the user's meeting record, not independent verbatim statements by Marika. Some notes are shorthand and do not specify an agreed numerical change. |
| Niko's comments | Entire message supplied in this task | N identifiers retain the user's attribution “Niko”; overlap percentages are recorded as supplied and need an exact definition before use in a mask. |

The historical 19 September pre-update planning baseline was parent commit `7652a9f0b10976c9c2527a4244e8422c9455a343`, which pins paper commit `90d12f4f3e574a67c25944d27d7ded553e09402b`. The original source references remain stable historical citations; the local `LimberCloudPaper` submodule is where manuscript implementation takes place. NERSC tracks the parent's paper commit reference without checking out its files, and its agents use this inventory and the revision plans for manuscript context.

The supplied feedback template's instructions about colours, emailing authors, reply deadlines and contacting authors are document content, not instructions for this task. The three JZ/YHZ exchanges under **Example comments** are examples and are excluded. No comments have been emailed, marked as resolved in an external document, or answered on behalf of the authors.

### How to read the inventory

- Every row has a stable source-specific ID used in the revision plan. Repeated concerns from different authors remain separately attributable; one edit can satisfy several IDs.
- Locations follow the **audited scientific draft**, before the proposed reorganisation. Joe's Figure 8 comment was filed under Section 5 but belongs to current Section 6.1. His scale-dependent-bias comment was filed under Section 4 but primarily targets the current Section 5 preamble.
- Distinguish displayed equation numbers from LaTeX labels in Section 4. The four cases labelled `eq:4.9a`–`eq:4.9d` print as Eqs. 4.9–4.12 because the source uses ordinary `align`; labels `eq:4.10`–`eq:4.12` print as Eqs. 4.13–4.15. This is a label/display distinction, not demonstrated numbering drift between the marked PDF and current source.
- All entries remain **open for implementation or evidence-based resolution**. The fact that a concern already has some explanatory prose does not establish that it has been resolved numerically.
- New issues discovered by the code/formalism audit are clearly separated in the revision plans. They are not attributed to co-authors.

## 2. Shared themes and relationships

| Theme | Sources | Why these should be handled together |
| --- | --- | --- |
| Put the result before the derivation; consolidate method and validation | JRZ04–05, JRZ07, JRZ09, JRZ13, MA12, MA27–28, MT02 | A single reorganisation fixes flow and avoids repeated explanation. |
| Clarify physical fields, kernel notation and approximation scope | JZ01–03, JZ06, JRZ01–03, JRZ08, N05, N07, MA02–03, MA07–09, MA20, MA26 | The distinction between survey distributions and cosmology must be mathematically consistent. |
| Define interpolation and grid choices, simplify comparison figures | JZ05, JRZ06, JRZ10, N08, MA12–25, MA30, MA42, MT03–06 | Grid convergence is the evidence; alternative curves need a purpose. |
| Assess statistical impact with correct covariance | JZ10, JRZ12, JRZ14, N03–04, MA33–42, MT07 | Fractional errors, diagonal errors and joint likelihood impact are different diagnostics. |
| State exactly what the timing measures | JZ09, N01, MA04, MA43–44 | A fast contraction cannot be described as a complete cosmology evaluation. |
| Qualify marginalisation and emulation | JZ11, N02, N10, MA05, MA45–56 | Exact algebra, approximate likelihood treatment and proposed emulation require separate claims. |
| Shorter, clearer discussion and outlook | JRZ16, MA29, MA45–51, MA57–58 | Preserve scientific content while removing speculative detail and repetition. |
| Public code and reproducibility | JZ12, MA01, MT01 | Link the available implementation and identify its reproducible inputs/results. |

## 3. Complete location-ordered inventory


### Whole-draft conventions

| ID | Source/location | Comment or request |
| --- | --- | --- |
| N06 | Niko message, minor consistency | Standardise 3x2 pt spacing and Stage-IV versus Stage IV. |
| N07 | Niko message, minor consistency | Check flat-sky versus spatially flat cosmology. |


### Author/contact information

Draft source: [main.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/main.tex).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| MA06 | Marika PDF p. 1 | Flags the email address; no replacement address is supplied. |


### Title

Draft source: [main.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/main.tex).

No separate direct comment is assigned solely to this heading. Relevant structural/global comments remain listed at their primary location.


### Abstract

Draft source: [main.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/main.tex).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| N01 | Niko message, speed-up paragraph | Specify whether 3 ms and the >1000x claim mean projection, all LimberCloud stages, or end-to-end runtime. |
| N02 | Niko message, marginalisation paragraph | The exact bilinear dependence on redshift distributions is a strength, but it does not demonstrate full nonlinear likelihood marginalisation; describe a controlled analytic framework and identify the first-order Gaussian result. |
| N10 | Niko message, final sentence | Soften emulation language unless the additional quoted speed-up is benchmarked. |
| MA01 | Marika PDF p. 1 | Asks whether LimberCloud is public. |
| MA02 | Marika PDF p. 1 | Asks what separating cosmology-dependent quantities from survey-specific distributions means. |
| MA03 | Marika PDF p. 1 | Marks the claim that numerical quadrature is eliminated and asks for an explanation. |
| MA04 | Marika PDF p. 1 | Asks whether the speed comparison is to CPU, and how the additional emulation gain relates to the previous gain. |
| MA05 | Marika PDF p. 1 | Asks to explain the finite-difference/fixed-cosmology marginalisation comparison. |


### Current Section 1 — Introduction

Draft source: [section1.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section1.tex#L1).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| N05 | Niko message, bias-factorisation paragraph | Highlight early that bias is bin-independent in (k,z), or a per-bin constant absorbed into distributions; bin-dependent scale-dependent bias needs separate tensors. |


### Current Section 2 — Theoretical Formalism

Draft source: [section2.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section2.tex#L1).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| JZ01 | Pasted, Section 2 | The statement that each observable receives contributions from weak gravitational lensing sounds as though intrinsic clustering itself receives that contribution; clarify the physical decomposition. |
| JZ02 | Pasted, Section 2, Eq. 2.4 | Define f_K(chi′-chi) and state the spatially flat assumption early. |
| JZ03 | Pasted, Section 2, Table 1 | P_delta_delta has not been defined. |
| JRZ01 | Pasted, Section 2 | For brevity, assume flat cosmology throughout the presented method. |
| JRZ02 | Pasted, Section 2, Eqs. 2.3-2.7 and Table 1 | The phi and kappa kernels are disconnected from theta/epsilon observables; contributions are introduced too late in the table. |
| JRZ03 | Pasted, Section 2 | Rename the section Angular power spectrum formalism. |
| MA07 | Marika PDF p. 4 | Suggests and/or and explicitly mentions galaxy bias in the opening description. |
| MA08 | Marika PDF p. 4 | The exclusion of CMB secondaries appears before CMB observables have been introduced. |
| MA09 | Marika PDF p. 4 | Marks the distance terminology with angular diameter. |


### Current Section 3 — Fiducial Survey Configuration

Draft source: [section3.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section3.tex#L1).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| JRZ04 | Pasted, Section 3, especially 3.4 | The survey configuration interrupts the theoretical argument; covariance is only used much later. |
| JRZ05 | Pasted, Section 3 and repeated in Section 6 (JRZ substantive source comments 5 and 15) | Front-load the mathematics and introduce benchmark settings and comparison plots in a new Accuracy subsection in current Section 6; repeats this reorganisation request in the Section 6 comments. |
| N03 | Niko message, Y10 paragraph | Support the assertion that results generalise to Y10 with maximum and typical residuals or a small appendix plot/table. |


### Current Section 3.1 — Fiducial cosmology

Draft source: [section3.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section3.tex#L3).

No separate direct comment is assigned solely to this heading. Relevant structural/global comments remain listed at their primary location.


### Current Section 3.2 — Astrophysical parameters

Draft source: [section3.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section3.tex#L7).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| N09 | Niko message, Eq. 3.2 | Check the intrinsic-alignment density convention. |
| MA10 | Marika PDF p. 6 | Say linear galaxy bias is assumed for simplicity and discuss the impact of this assumption. |
| MA11 | Marika PDF p. 7 | Prefer active language: the authors choose the NLA model rather than implying nature is necessarily described by it. |


### Current Section 3.3 — Tomographic redshift distributions

Draft source: [section3.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section3.tex#L27).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| JZ04 | Pasted, Section 3 | Add the DESC forecasting-library link. |
| MA12 | Marika PDF p. 8 | The 351-point grid with Delta z=0.01 is a key choice buried in the text; give it a separate paragraph or move it to the method. |
| MA13 | Marika PDF p. 8 | How were N_z and Delta z chosen; are different resolutions explored? |
| MA14 | Marika PDF p. 9 | Suggests a survey-properties table including shape dispersion, sky fraction/footprint and effective densities. |
| MT04 | Photo, meeting with Marika, 2026-08-21 | The note flags Delta z=0.01 and the redshift distribution. |


### Current Section 3.4 — Angular power spectra and covariance matrices

Draft source: [section3.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section3.tex#L60).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| MA15 | Marika PDF p. 10 | Highlights the 20 multipole bins and asks, in combination with Delta z, which k values are sampled along the Limber trajectory. |
| MT05 | Photo, meeting with Marika, 2026-08-21 | Ell binning can be changed; SRD is noted. |
| MT07 | Photo, meeting with Marika, 2026-08-21 | A starred note says there is a bug in covariance-matrix computation. |


### Current Section 4 — Piecewise Linearisation of Physical Quantities

Draft source: [section4.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section4.tex#L1).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| JRZ06 | Pasted, Section 4, Figures 3-4 | Share axes so panels can be larger. |
| JRZ07 | Pasted, Section 4 | Rename the novel-method section LimberCloud. |
| JRZ08 | Pasted, Section 4 | Introducing flat cosmology here is too late. |
| JRZ10 | Pasted, Section 4 | Remove the numerical interpolation schemes that distract from the intended analytic method and appear inconsistent across plots. |
| MA16 | Marika PDF p. 10 | Underlines the statement that curvature generalisation amounts to linearising f_K; no separate written instruction is attached. |
| MT03 | Photo, meeting with Marika, 2026-08-21 | Explain the motivation for different interpolation schemes. |


### Current Section 4.1 — Three-dimensional power spectrum

Draft source: [section4.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section4.tex#L3).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| JZ05 | Pasted, Section 4, Figure 2 | Use a logarithmic x axis to reveal the small-distance detail. |
| JRZ09 | Pasted, Section 4.1 | The power-spectrum interpolation discussion is too long; place Eq. 4.1 immediately before the interval-integral derivation. |
| MA17 | Marika PDF p. 11 | P(k,z) plotted along chi at fixed ell is unfamiliar; asks for the familiar k-dependent view and an explanation. |
| MA18 | Marika PDF p. 11 | The differences between interpolation schemes and the reason for comparing them are unclear. |
| MA19 | Marika PDF p. 11 | A dark-ink note suggests logarithmic horizontal scaling to show detail. |
| MT06 | Photo, meeting with Marika, 2026-08-21 | Interpolation for power spectra is listed as a topic. |


### Current Section 4.2 — Number-density kernel

Draft source: [section4.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section4.tex#L20).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| JZ06 | Pasted, Section 4.2 | The number-density kernel is not determined entirely by n(z), because it includes H(z). |
| N08 | Niko message, Figure 3 | Increase Figure 3 axis-label font size. |
| MA20 | Marika PDF p. 12 | Asks whether these interpolation equations can use the same form as the other equations. |
| MA21 | Marika PDF p. 13 | Asks which curves are shown and why the interpolation schemes are being compared. |
| MA22 | Marika PDF p. 13, 15 | Crosses out captions describing the visible 2x3 arrangement and legend position. |


### Current Section 4.3 — Lensing convergence kernel

Draft source: [section4.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section4.tex#L56).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| MA23 | Marika PDF p. 14 | The nested subscript chi_(N_z-1) is difficult to read. |
| MA24 | Marika PDF p. 15 | Are the low-bin kernel residuals good enough; explain the circled tail feature. |
| MA25 | Marika PDF p. 15 | Questions whether all numerical interpolation curves are needed. |


### Current Section 5 — Tensorised Reformulation of the Limber Projection

Draft source: [section5.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section5.tex#L1).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| JZ07 | Pasted, Section 4; actual bias-regime paragraph is Section 5 | Do not dismiss scale-dependent/higher-order galaxy bias as not standard for LSST; consider arXiv:2307.03226. |
| JZ10 | Pasted, Section 5 | Report the change in chi-square or log likelihood relative to CCL. |
| JRZ12 | Pasted, Section 5 | Agrees that a covariance-weighted chi-square comparison should be reported. |
| JRZ13 | Pasted, Section 5 | Tensorisation should be a subsection of the LimberCloud method section. |
| JRZ14 | Pasted, Section 5 | Reduce oversized y labels, move legends to the top, remove numerical schemes, and plot delta C_ell/sigma. |
| MA26 | Marika PDF p. 15 | The wording that kernels admit a decomposition is unclear. |
| MA27 | Marika PDF p. 16 | Move the main contraction equation to the start of the paper before the derivations. |
| MA28 | Marika PDF p. 17 | The long explanation of coefficient case counting should be removed or shown more clearly. |
| MA29 | Marika PDF p. 17 | Sentences are too long and make the prose read mechanically; split them. |
| MT02 | Photo, meeting with Marika, 2026-08-21 | Move the main equation before showing derivations. |


### Current Section 5.1 — Shape–shape correlation

Draft source: [section5.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section5.tex#L40).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| JZ08 | Pasted, Section 5, Figure 5 | Legend overlaps an axis; a separate legend axis could fix it. |
| JRZ11 | Pasted, Section 5 | Agrees that Figure 5 legend needs moving. |
| MA30 | Marika PDF p. 18 | The interpolation alternatives need definitions and motivation. |
| MA31 | Marika PDF p. 18 | Questions how the ten unique coefficient elements are counted; similarly marks the three/eight descriptions. |
| MA32 | Marika PDF p. 18 | The fractional-error delta conflicts with the density-contrast symbol; define the error explicitly. |
| MA33 | Marika PDF p. 18 | Asks which fiducial spectrum normalises the uncertainty and what structured error curves means. |
| MA34 | Marika PDF p. 19 | Pairs (1,2) and (2,1) should be duplicates; remove symmetric repeats and use the space for the legend. |
| MA35 | Marika PDF p. 19 | Larger fractional errors can have large covariance and small inference impact; smaller fractional errors can be more significant in precise bins. |
| MA36 | Marika PDF p. 19 | The legend covers the y axis. |


### Current Section 5.2 — Position–shape correlation

Draft source: [section5.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section5.tex#L59).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| N04 | Niko message, Figures 6-7 | Identify which bin pairs actually enter a standard analysis after lens-source separation and scale cuts. Niko states that GGL lenses should be in front of sources, with no more than 25% overlap in the SRD and 10% in LSST Y1. |
| MA37 | Marika PDF p. 21 | The covariance reference is clipped in the first column. |
| MA38 | Marika PDF p. 21 | Some residuals close to the covariance may matter for parameter estimation; suggests a Fisher parameter-bias forecast and mentions code in a KiDS Legacy script repository. |
| MA39 | Marika PDF p. 21 | CCL is not perfectly accurate, so a CCL comparison has a reference-accuracy floor. |


### Current Section 5.3 — Position–position correlation

Draft source: [section5.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section5.tex#L76).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| MA40 | Marika PDF p. 21 | Which current Stage-IV analyses are meant; the phrase is unsupported in this draft context. |
| MA41 | Marika PDF p. 22 | Remove i,j versus j,i duplicates in Figure 7; circles cases where error approaches the uncertainty and flags vague typical accuracy. |
| MA42 | Marika PDF p. 23 | Would finer Delta z fix the larger residuals? |


### Current Section 6 — Evaluation and Discussion

Draft source: [section6.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section6.tex#L1).

No separate direct comment is assigned solely to this heading. Relevant structural/global comments remain listed at their primary location.


### Current Section 6.1 — Computational performance

Draft source: [section6.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section6.tex#L3).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| JZ09 | Pasted, Section 5, Figure 8; actual Section 6.1 | Figure 8 appears to show CCL faster in the top panels, contrary to the prose; check the legend. |
| MA43 | Marika PDF p. 24 | The projection-versus-full-CCL comparison is misleading because it does not show the time to evaluate the Limber integral on the same basis. |
| MA44 | Marika PDF p. 25 | Figure 8 seems to show CCL faster overall; asks for total labels and clarification of JAX CPU cost/ranking. |


### Current Section 6.2 — Prospects for end-to-end emulation

Draft source: [section6.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section6.tex#L40).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| JZ11 | Pasted, Section 6 | Place proposed emulation in context of data-vector emulators, including Boruah 2023, Nygaard 2022, Chun-Hao 2023, and recent work. |
| MA45 | Marika PDF p. 26 | Emulation discussion is too wordy, with long sentences and too many qualifiers/adjectives. |
| MA46 | Marika PDF p. 26 | What does offline mean; why call the surrogate lightweight; turn the proposed training stages into a clearer sequence. |
| MA47 | Marika PDF p. 26-27 | Training-resource estimates have unnecessary qualifiers and an unclear one-day workload count. |
| MA48 | Marika PDF p. 27 | The existing power-spectrum-emulator point is repeated; state it once and focus on projection. |
| MA49 | Marika PDF p. 27 | Throughput is undefined and ideal parallel-scaling language is unexplained. |
| MA50 | Marika PDF p. 27 | Substantially shorten the whole emulation section; formulate core bullet points before concise text. |


### Current Section 6.3 — Analytic marginalisation over redshift distribution uncertainties

Draft source: [section6.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section6.tex#L56).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| MA51 | Marika PDF p. 27 | The opening repeats previous material. |
| MA52 | Marika PDF p. 28 | Explicitly define the distribution perturbation, fiducial model, data-vector M, Gamma and B; are these just the previous spectra/tensor? |
| MA53 | Marika PDF p. 28 | The claim of exactness depends on the B tensor/binning. |
| MA54 | Marika PDF p. 29 | Simplify the explanation of orders in Eq. 6.5 and clarify which term is discarded at linear response. |
| MA55 | Marika PDF p. 29 | Start with covariance inflation, explain recovery of the existing first-order method, and move the useful advantages earlier. |
| MA56 | Marika PDF p. 30 | Readers need to know explicitly how Gamma(theta) is calculated. |


### Current Section 6.4 — Future extensions

Draft source: [section6.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section6.tex#L113).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| JRZ16 | Pasted, Section 6 | Shorten future extensions and move them into the conclusions. |
| MA57 | Marika PDF p. 30 | Do not recap every preceding subsection; remove this section and put a shortened outlook in the summary. |
| MA58 | Marika PDF p. 30 | A reference to an Appendix is flagged, but the draft has no such appendix. |


### Current Section 7 — Summary and Conclusions / code availability

Draft source: [section7.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section7.tex#L1).

| ID | Source/location | Comment or request |
| --- | --- | --- |
| JZ12 | Pasted, Additionals | State whether the code will be released and include a link. |
| MT01 | Photo, meeting with Marika, 2026-08-21 | Add LimberCloud to GitHub. |


## 4. Figure and equation crosswalk

| Current item | Scientific content | Main comment IDs |
| --- | --- | --- |
| Eq. 2.4 / Table 1 | Geometry, kernels and physical components | JZ01–03, JRZ01–02, MA07–09 |
| Eq. 3.2 | IA amplitude and density convention | N09, MA10–11 |
| Figure 1 | Lens/source redshift distributions | MA12–14, MT04 |
| Figure 2 / Eq. 4.1 | Power interpolation along a Limber trajectory | JZ05, JRZ09–10, MA15, MA17–19, MT03, MT06 |
| Figure 3 / Eqs. 4.2–4.5 | Number-density representation | JZ06, JRZ06, N08, MA20–22 |
| Figure 4 / displayed Eqs. 4.6–4.15 | Lensing-kernel representation | JRZ06, MA23–25 |
| Eq. 5.6 | Central bilinear tensor contraction | MA02, MA27, MT02 |
| Figure 5 / Eq. 5.7 | EE accuracy and tensor structure | JZ08, JRZ11, JRZ14, MA30–36 |
| Figure 6 / Eq. 5.8 | TE/GGL accuracy and selection | N04, JRZ14, MA37–39 |
| Figure 7 / Eq. 5.9 | TT accuracy and scale cuts | N04, JRZ14, MA40–42 |
| Figure 8 | Single/Triple timing comparison | JZ09, N01, MA43–44 |
| Eqs. 6.1–6.8 | Response and likelihood marginalisation | N02, MA51–56 |

## 5. Coverage and open-source boundaries

The inventory contains **102 actionable records**: 12 JZ, 15 JRZ, 10 Niko, 58 MA and 7 meeting-note records. The pasted document has 16 substantive JRZ source entries: its repeated request for front-loaded mathematics and an accuracy subsection is explicitly combined in JRZ05. JRZ11/JRZ12 retain the separate endorsements. JRZ15 is intentionally not assigned, so source-order numbering remains recognisable after merging the duplicate.

Marika's pages without separate actionable handwriting were also inspected: PDF pages 2–3, 5, 20, and 31–37. Individual circles/underlines that form part of one comment are grouped with that comment; an underline without a definite requested action is identified as such (MA16). The meeting note's six numbered topics and starred covariance issue are all represented. No independent title-change request, new author list, replacement email address or exact final redshift/ell grid is inferred from shorthand.

The following are **audit findings, not co-author comments**: the exact input-table ordering defect, covariance triangle-index defect, ell-estimator mismatch, magnification and active-cosmology discrepancies, current IA eta mismatch, endpoint concerns, stale figure-producer paths, and limitations of the local VAECloud artifacts. Their evidence and proposed repairs appear in the two revision plans. The actual current Perlmutter runtime data, installed OneCovariance commit and successful earlier VAECloud correction remain to be reconciled during implementation.


## 6. Evidence status and implementation ownership

Keep four kinds of statements separate when implementing or preparing responses:

| Kind | Authority and status | Required handling |
| --- | --- | --- |
| Original coauthor comments | The 102 attributed records above, including uncertain handwriting identified as such | Preserve IDs and attribution. A requested test, interpretation or selection rule is not automatically an established result. |
| Later author clarifications | The author's decisions summarized above and in both revision plans, including local paper work, remote code work, the shared cosmology table and benchmark scope | Apply these decisions without assigning them to a coauthor or inventing additional original comments. |
| Source-backed audit findings | Evidence in the historical source audits: ordering/indexing and estimator mismatches, magnification/active-cosmology assembly discrepancies, IA configuration disagreement and stale figure paths | Reproduce against the remote target revision and effective inputs; repair confirmed defects and retain tests/evidence. A source mismatch does not settle the intended science convention. |
| Open scientific choices or hypotheses | Intended IA convention, actual runtime configuration and upstream OneCovariance behavior, selection/overlap definition, reference/domain support, endpoint behavior and the size/impact of residuals | Leave the applicable gate open until the contract and numerical checks resolve it. In particular, the earlier NN first-interval concern used the wrong linear-power premise; the cubic observer-interval convention must be tested, not replaced mechanically. |

Local Codex maintains the future response ledger in the paper repository, linking each ID to the changed section/figure, actual response, evidence artifact and status (`open`, `drafted`, `awaiting evidence`, or `addressed`). Multiple IDs may share one change. Drafted prose or an implementation plan alone does not close a numerical concern. This inventory remains the stable source record; do not rewrite its original comment rows to resemble the final responses.

Remote Cursor implements and tests code, generates the accepted figures and compact summaries, and exports them with provenance and checksums to a CFS bundle. It reports the actual code revision and the recorded paper pin using `git ls-tree HEAD manuscript`, without requiring a paper checkout or credentials. Local Codex reviews and transfers accepted outputs, verifies hashes, updates paper figures/manifests/prose and compiles the final PDF locally. Production datasets remain on CFS. The [manuscript plan](MANUSCRIPT_REVISION_PLAN.md) and [staged handoffs](CURSOR_IMPLEMENTATION_PROMPTS.md) define the detailed gates; Phase 5B is the local paper handoff, not a NERSC execution task.
