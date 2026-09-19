# Manuscript revision strategy and individual proposed edits

**Updated 19 September 2026 — planning document, not an implemented manuscript revision.** Read alongside [COAUTHOR_COMMENT_INVENTORY.md](COAUTHOR_COMMENT_INVENTORY.md) and [CODE_REVISION_PLAN.md](CODE_REVISION_PLAN.md). This update incorporates the author's decisions on a fiducial-plus-ensemble experiment, serial benchmark execution, absolute error plots, and the scope of interpolation/grid studies. Comment IDs below refer to the inventory. Existing line-linked scientific source locations refer to GitHub commit [`7d29b2f`](https://github.com/CosmoCloudZhang/LimberCloud/tree/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c), the 15 September audit baseline; historical local checkout references in supporting audits do not describe this review-package directory.

The current implementation was reread from a fresh GitHub checkout at [`0876bf4`](https://github.com/CosmoCloudZhang/LimberCloud/tree/0876bf4a50e869be1289a3eecb46931e7c8eb534) on 19 September. It still has 24 benchmark drivers running 1,000 serial, unseeded cosmology evaluations and saving cumulative timings. Numba Double/Triple use a narrower sampling range than the other drivers. The new ensemble and saved-spectrum products below are proposed changes, not existing results. Perlmutter's checkout, environment and external arrays still require reconciliation before implementation.

## 1. Recommended high-level revision

**Make this a methods paper with a clearly demonstrated numerical result:** explain the observable and the reusable tensor, derive it compactly, then establish its accuracy and timing under a reproducible validation contract. Retain analytic marginalisation and emulation as carefully scoped implications. A complete emulator or full posterior-marginalisation demonstration is not necessary to complete this revision if the claims are limited accordingly.

Accept the shared structural recommendation from Jaime and Marika. The present sequence interrupts the derivation with survey details, asks readers to interpret several interpolation schemes before understanding the main result, and mixes accuracy with method derivation. Moving paragraphs without changing this logic would leave their central concern unresolved.

### Proposed final outline

| New location | Content and purpose | Material moved from current draft |
| --- | --- | --- |
| **1 Introduction** | Observed galaxy shapes/positions, repeated projection cost, relevant prior methods, short statement of the contribution and assumptions. Show the central contraction with defined symbols near the end; opening Section 2 is an acceptable alternative if the flow is clearer. Add the repository link. | Current 1; a short interpretation of Eq. 5.6; early bias/flatness qualifications |
| **2 Angular power spectrum formalism** | Define observed fields, component spectra, shared kernels, spin factors and adopted geometry together. Keep Table 1 after its symbols are explained. | Current 2, shortened |
| **3 The LimberCloud method** | Start with the result and the distinction between precomputation and redshift-distribution updates. | Merge current 4 and 5 |
| 3.1 Grid representation and interpolation | Define grid nodes, intervals, p(z), phi(chi), the Limber trajectory, linear input approximations and boundary treatment. | Current 3.3 numerical grid; compact 4.1–4.2 |
| 3.2 Kernel representation | Density basis and integrated lensing basis, with a short derivation and clear dependence on the input distribution. | Current 4.2–4.3 |
| 3.3 Tensorised projection | Derive the main contraction once; give one concise table of structural formula families and describe observable assembly. | Current 5 preamble and equation portions of 5.1–5.3 |
| 3.4 Parameter dependence and implementation | Bias-factorisation regimes, what is cached/recomputed, backend overview and resource scaling. | Current 5 bias paragraph; necessary implementation material from 6.1–6.2 |
| **4 Validation and computational performance** | Collect the test configuration, accuracy and performance evidence. | Current 3; empirical parts of 4–5; current 6.1 |
| 4.1 Fiducial configuration and reference calculation | Y1/Y10 table, one named fiducial plus 1,000 shared sampled cosmologies, parameter ranges/seed/table identity, IA/magnification conventions, ell estimator, masks, covariance and reference tolerances. | Current 3.1–3.4 |
| 4.2 Accuracy and sensitivity | Matched CCL comparisons; absolute fractional and covariance-normalised errors on log axes; optional ensemble percentile bands; per-cosmology joint discrepancy; bounded reference/resolution checks; concise Y10 summary. | Current 5 accuracy paragraphs and largely preserved Figures 5–7 layout; new numerical summaries |
| 4.3 Computational performance | Fair serial accumulated timing and reusable-basis workloads, internal threading/device execution, cold/warm definitions, stage table, corrected Figure 8. | Current 6.1 |
| **5 Implications for inference** | Separate demonstrated algebra from numerical or inference work still to do. | Current 6.2–6.3 |
| 5.1 Repeated redshift-distribution updates | Explain the measured reusable-basis operation and fast/slow parameter use. Avoid claiming measured sampler acceleration. | Current 6.1 final discussion |
| 5.2 Controlled analytic marginalisation | Explicit response matrix; linear-Gaussian result; exact quadratic model response; higher-order limitations. | Current 6.3, shortened/reordered |
| 5.3 Emulation prospects | Contrast target choices and remaining accuracy/compression requirements in about three compact paragraphs. | Current 6.2, substantially shortened |
| **6 Summary and outlook** | Verified contribution, validated scope, practical limits, one short paragraph on extensions. | Current 7 and a much shorter 6.4 |
| Appendix A | Coefficient cases, endpoint conventions, 3/8/10 formula-family accounting and links to executable derivations. | Long current 4.3/5 details |
| Appendix B | Linear/quadratic/cubic numerical comparisons that test the adequacy of the adopted representation; selected kernel diagnostics and bounded resolution checks. | Current Figures 2–4, simplified where useful |
| Appendix C | Y10 summary and full-pair diagnostics, selection lists and optional additional timing details. | Existing Y10 products after validation; diagnostic-only panels |

This outline implements Jaime's requested **Accuracy** subsection even though renumbering places it in new Section 4 rather than old Section 6. Preserve existing LaTeX labels through an initial move-only pass and record a current-to-new crosswalk; update displayed numbering and prose references in a later pass. The exact number of appendices can be reduced if material fits naturally together.

### Editorial choices that reconcile the comments

1. **Keep the core mathematics; shorten its presentation.** Marika wants the main equation first; Jaime wants less interpolation exposition. Show the endpoint equation early, keep a short reproducible derivation, and move long cases to an appendix.
2. **Give the numerical alternatives a clear purpose and a compact presentation.** The author's motivation is to test whether the adopted linear representations of distribution/kernel inputs and sampled power are accurate enough. Retain numerical linear/quadratic/cubic comparisons with the interpolated object, coordinate, grid and quadrature specified. Cubic is a useful comparator, not guaranteed truth. Keep the main LimberCloud–CCL figures readable and put detailed alternative-scheme comparisons in the appendix or diagnostic notebooks; their removal from main panels does not remove this validation requirement.
3. **Use the full diagnostic set and an explicit science selection.** Keep difficult pairs visible in an appendix; mark the chosen science mask in main results. Selection must follow a documented analysis rule, not the pattern of favourable residuals.
4. **Keep absolute error plots and add ensemble context.** Use absolute fractional error and absolute delta C/sigma on log y axes, preserving the established layout. Fiducial curves and pointwise ensemble bands answer local questions; the full covariance quadratic form checks accumulation and correlations. Signed diagnostics remain available for investigating coherent errors and zero crossings.
5. **Keep the inference discussion proportional to its evidence.** Exact quadratic dependence in density coefficients is useful. The demonstrated Gaussian marginal is conditional on linearising the prediction and using a Gaussian prior; full nonlinear marginalisation and emulation remain future work.
6. **Revise the abstract and conclusions last.** The present draft contains mutually incompatible all-bin accuracy and runtime claims. New wording must follow the validated products, rather than preserve a preferred headline.

## 2. Order of work and evidence required

Editorial reorganisation can proceed in parallel with code validation. Numerical conclusions and statistical interpretations must wait for the following dependencies.

| Gate | Work | Completion evidence |
| --- | --- | --- |
| E1 — Model and estimator contract | Fix the intended survey inputs, IA law, slope/response convention, active-cosmology dependence, grid, ell points/windows and analysis mask. | Versioned effective configuration and labelled data-vector definition; agreement between code and declared fiducial model |
| E2 — Spectrum correctness | Repair confirmed component/serialization defects; check independent interval integrals and endpoint behaviour; compare CCL/Numba/JAX and NUMERIC interpolation settings on the same input and estimator. | Component-level tests, representative nonfiducial comparisons, reference tolerance checks and bounded grid sensitivity; no unexplained cancellations used as validation |
| E3 — Covariance correctness | Correct input ell/bin order and output pair mapping; reproduce actual upstream reader and binner; check noise and model components. | Round-trip labels/values, independent Gaussian checks, float64 symmetry/conditioning/Cholesky, saved component identity and matching window |
| E4 — Accuracy evidence | Run one explicit fiducial and 1,000 matched sampled cosmologies per adopted survey configuration; use scripts to save spectra and notebooks to analyse them. Apply one saved mask to plots, residuals and covariance. | Per-sample spectra/status/identities, counts, pointwise percentiles, worst pair/band, undefined-ratio counts and a per-cosmology joint-discrepancy table; fiducial excluded from ensemble quantiles |
| E5 — Timing evidence | Run corrected physics with matched work in a serial cosmology loop within one task, retaining internal Numba/JAX parallel execution, common sample table, declared cold/warm boundaries and actual device synchronization. | Accumulated and stage/per-evaluation timing data, resource/thread settings and hardware/environment metadata; rerendered benchmark figure; serialization outside compute timers |
| E6 — Paper and figure revision | Move sections; implement individual edits below; regenerate affected figures from accepted artifacts; reconcile bibliography and availability. | Source diff, complete comment-to-change mapping, traceable figure manifest |
| E7 — Final review | Compile the whole manuscript and inspect every page, equation, figure and reference. | No unresolved references, clipping, legend overlap or unexplained claims; each closed comment cites a concrete change and evidence |

Do not enlarge statistical error bars, omit difficult pairs without a rule, or repair matrices with jitter to reach an accuracy target. If a verified selected-vector discrepancy remains significant, refine the numerical representation or limit the supported claim. No fresh Y1/Y10 residual, end-to-end speed-up or corrected production covariance is claimed by this planning review.

The current revision does not require task-based parallelism, a search for an optimal redshift or ell grid, a new power-interpolation family, or 1,000 covariance recomputations. Preserve the necessary correctness, reference-tolerance and small resolution-sensitivity checks. Broader optimisation and deployment in parallel inference pipelines belong in future work.

## 3. Individual proposed edits in current draft order

Each ID receives a specific disposition below. “Editorial” means the edit can be drafted immediately; other entries require the named source check or numerical gate before being described as complete. Endorsements retain their own row for attribution but share implementation with the original request.


### Whole-draft conventions

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| N06 | Use a single LaTeX macro for 3\times2\,\mathrm{pt}; choose Stage-IV consistently as an adjective and apply the house style globally. | Editorial |
| N07 | Replace flat-sky where f_K=chi follows from zero spatial curvature; retain true flat-sky terminology only for angular approximations and distinguish it from Limber. | Editorial |

Apply these conventions to the abstract, captions, table headers, labels and conclusion as well as prose. Keep scientific meaning ahead of mechanical replacement: a real angular flat-sky approximation must retain its correct name. Shorten long sentences and remove repeated promotional language while retaining necessary technical terms.


### Author/contact information

Draft source: [main.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/main.tex).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| MA06 | Check the desired current corresponding-author address during final front-matter review. Do not infer a replacement from the mark. | Author information |


### Title

Draft source: [main.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/main.tex).

No separate direct comment is assigned solely to this heading; implement the linked structural/global changes and the contextual checks below where applicable.


### Abstract

Draft source: [main.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/main.tex).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| N01 | Use separately measured labels for cosmology, coefficient, contraction and full prediction. The existing text identifies 3 ms as contraction only; remove the headline ratio until matched benchmark outputs support it. | Timing validation |
| N02 | Say exact quadratic response for the discretised density coefficients; Gaussian covariance inflation follows after linearising that response. Explicitly defer nonlinear likelihood validation. Apply throughout Sections 1, 6.3, 6.4 and 7. | Formalism audit |
| N10 | Remove unmeasured additional 3-4 orders of magnitude; retain a short future tensor-emulation opportunity and a clear distinction between incremental and total speed-up. | Editorial |
| MA01 | Add a footnote linking https://github.com/CosmoCloudZhang/LimberCloud at first use and a code-availability statement with the eventual results commit/release; avoid promising a release status not verified. | Editorial |
| MA02 | Explain in physical terms that redshift-distribution changes reuse a precomputed tensor; give the short contraction equation early and define the survey vector correctly. | Editorial |
| MA03 | Restrict the statement to the radial Limber integrals of the adopted piecewise representation; cosmology calculations and any ell-band integration remain distinct numerical stages. | Editorial |
| MA04 | Name backend/hardware/workload and timing boundary for each comparison; remove speculative cumulative multiplication of speed-up factors. | Timing validation |
| MA05 | Explain analytic response coefficients and their cosmology dependence in the discussion; avoid claiming all previous work must use a fixed cosmology or numerical derivatives. | Literature and formalism verification |

#### Claim boundaries for the rewritten abstract

Use four claims only: (1) analytic integration of the adopted piecewise radial representation; (2) a tensor contraction separating sampled redshift distributions from cosmological/model coefficients under stated bias assumptions; (3) the measured accuracy and runtime for an explicitly named configuration; (4) a controlled marginalisation framework and future emulation opportunity. The current Section 6.1 identifies roughly 3 ms with contraction only. Its approximate two-fold total-runtime statement and the introduction's greater-than-ten-fold statement cannot both be retained without reconciling the underlying evidence.

Provisional wording that can be used before numerical values are settled: *LimberCloud evaluates the radial Limber integrals of a piecewise representation analytically and expresses the spectra as bilinear contractions of sampled redshift distributions. We validate the implementation against a matched CCL calculation for specified LSST-like configurations. The tensor also supplies analytic redshift-distribution responses for a controlled marginalisation framework.* Add numerical accuracy and runtime sentences only from E4/E5; specify the selected pair set and whether the time is a warm contraction, coefficient plus contraction, or full prediction. Avoid an unqualified all-pairs sub-percent claim: current Section 5.2 itself reports approximately 10% fractional residuals for some diagnostic GGL pairs.


### Current Section 1 — Introduction

Draft source: [section1.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section1.tex#L1).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| N05 | Add a short assumption statement in the introduction and a three-regime table in the method, with computational consequences. | Formalism audit |

#### Early explanation of the reusable computation

After introducing the observed spectra, show the schematic equation C_ell^(ab)=p_a^T Bhat_ell(theta) p_b near the end of Section 1, or at the opening of Section 2 if that reads more naturally. This is the early result that motivates the derivation, not a derivation inserted into the introduction. Define p as normalized redshift-density samples on a fixed redshift grid, theta as cosmological plus relevant astrophysical parameters, and Bhat as the model-dependent tensor. At fixed theta, a change of p changes only the final contraction. This does not imply every nuisance parameter is cheap: a parameter altering P_uv or a non-factorable bias changes the tensor. It also does not mean covariance, survey area or every instrumental systematic is represented by p. Refer back at the start of the method and derive the result only once. Update the roadmap after section movement.

Limit claims about competing methods. Distinguish three-dimensional power emulation, projected data-vector emulation, fast/slow caching and this reusable distribution representation. Do not claim all other methods lack caching or parallelism.


### Current Section 2 — Theoretical Formalism

Draft source: [section2.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section2.tex#L1).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| JZ01 | Define observed shape and position fields first; explain shear/IA versus intrinsic clustering/magnification, then expand their correlations. Distinguish intrinsic clustering from observed number-count correlations, which can contain magnification terms; do not say every individual term receives every contribution. | Editorial |
| JZ02 | Adopt f_K(chi)=chi in the main derivation; explain it is transverse comoving distance. If the general expression remains, define both arguments and the curvature sign convention. | Editorial |
| JZ03 | Define the matter-density-contrast power spectrum, units, nonlinear prescription, and relation to each tracer spectrum before Table 1. | Editorial |
| JRZ01 | Use spatial flatness in the main formalism and a short scope sentence; move curvature generalisation to the outlook. | Editorial |
| JRZ02 | Add an explicit map: observed epsilon=gamma+I and theta=g+mu; gamma/mu use the lensing kernel, g/I the number-density kernel, with spin/bias factors specified. Introduce this before the expanded spectra. | Editorial |
| JRZ03 | Adopt that title and keep this section limited to established notation and assumptions. | Editorial |
| MA07 | Define which observable receives which physical contribution and name galaxy bias as an adopted tracer model. | Editorial |
| MA08 | Keep one short scope sentence for the three galaxy observables; move the CMB extension discussion to the outlook. | Editorial |
| MA09 | Clarify that f_K is transverse comoving distance and D_A=f_K/(1+z) is physical angular-diameter distance; do not substitute D_A into the comoving Limber equation. | Formalism audit |

#### Formalism and scope to preserve

Start with the adopted observed-field decomposition epsilon=gamma+I and theta=g+mu. Define the resulting EE, TE and TT component sums and explain that magnification is a lensing contribution to observed counts; ordinary galaxy clustering and magnification must not be conflated. Map number-density and lensing kernels to each physical component before Table 1.

Use spatial flatness to set f_K(chi)=chi. It is independent of the angular flat-sky approximation and the Limber approximation. The comoving transverse distance is f_K, while physical angular-diameter distance is f_K/(1+z). If the curved formula is retained anywhere, check its sign: the convention with positive K corresponding to a closed sine geometry has K=-Omega_K H0^2/c^2.

Define P_delta_delta, its matter-field convention and units, particularly with massive neutrinos. Keep spin prefactors and both IA cross terms consistent with the actual tracer implementation. State that the two-kernel multiplicative-bias presentation covers the adopted linear/NLA model. More general nonlinear/stochastic bias or IA may require independent operator cross-spectra; termwise tensorisation may still apply but its runtime and accuracy are not established by this fiducial test. A redshift-dependent magnification response weights the selected galaxies inside the lensing integral; only a bin-constant response can automatically be moved outside it.


### Current Section 3 — Fiducial Survey Configuration

Draft source: [section3.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section3.tex#L1).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| JRZ04 | Move the full fiducial configuration directly ahead of the accuracy results in a unified validation section. Retain only grid definitions needed for the method earlier. | Editorial |
| JRZ05 | Use the proposed new order: formalism, LimberCloud method, validation setup/accuracy, performance, concise implications, conclusions. Group repeated requests in both source sections under this ID. | Editorial |
| N03 | Generate a matched Y1/Y10 residual summary with bin counts, median/95th/max, selected/all pair distinction, worst-case labels, and estimator metadata. Do not reuse unverified figure values as new measurements. | Y10 numerical validation |

#### What the survey names establish

Describe the configuration as LSST-like or a specified forecast configuration when it deliberately differs from the SRD. Current survey generators use an 18,000 deg^2 footprint for both Y1 and Y10, and the draft motivates lens densities much smaller than the SRD gold sample. These choices directly affect the statistical error scale. Record the actual sample definition, source/lens counts, densities and footprint; do not use the sentence about sample realism as evidence that any numerical error is negligible. Run an SRD-like density/footprint sensitivity case if a Stage-IV inference claim depends on this choice.

The current draft's assertion that Y10 was verified must be supported by a reproducible table, even though Y10 PDF exports already exist. Suggested columns per EE/TE/TT: number of selected entries, median absolute fractional residual over a declared nonzero-reference subset, 95th percentile, maximum with pair/ell, maximum absolute delta C/sigma, and per-probe/joint discrepancy. Report all-pair diagnostic extrema separately. Define typical as a statistic rather than a visual impression. No values for this new table have been measured in this review.


### Current Section 3.1 — Fiducial cosmology

Draft source: [section3.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section3.tex#L3).

No separate direct comment is assigned solely to this heading; implement the linked structural/global changes and the contextual checks below where applicable.

No isolated source comment names this subsection. Its cosmology and transfer/nonlinear-power settings must nevertheless be moved into new 4.1 and checked against the effective run configuration under E1. Record sampled versus fixed parameters, neutrino convention, CAMB/HMCode options, k support and package versions. The covariance writer currently uses a different CAMB kmax from several spectrum paths; establish sufficient support and convergence rather than silently equating the configurations.

#### Fiducial and matched cosmology ensemble

Use one explicitly named fiducial cosmology plus 1,000 sampled cosmologies, shared across CCL, LimberCloud–Numba, LimberCloud–JAX CPU/GPU and NUMERIC with linear/quadratic/cubic settings. Share cosmological parameters across Y1/Y10 and Single/Double/Triple configurations when the same parameter domain is intended; survey/tracer settings remain explicit configuration inputs. Select the fiducial by ID for spectrum visualisation rather than assuming an arbitrary first random draw is fiducial.

A fixed seed supports reproducible draws, but equal seeds do not guarantee matching cosmologies when generators, draw order or parameter ranges differ. Generate one canonical table, save actual parameter values and stable IDs, and record seed, generator, ranges, parameter order and table hash in the run metadata. The current ±5% Numba Double/Triple versus ±10% other-driver ranges must be reconciled explicitly. Do not choose the scientific parameter domain merely from whichever script is edited first. Use the saved table as the matching contract for all methods and for restart; sampled IDs, not file position or completion order, identify a comparison.

The fiducial is an additional validation case and is excluded from the 1,000-sample timing curve and ensemble percentiles. Warm-up/compilation is recorded separately. State which parameters were varied and held fixed, how the range was chosen, and that the ensemble tests only that declared domain. Retain failed samples and failure reasons, report complete matched counts, and assess failure patterns before claiming stable accuracy across the domain. Do not silently resample or omit hard cosmologies.


### Current Section 3.2 — Astrophysical parameters

Draft source: [section3.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section3.tex#L7).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| N09 | Define rho_m,0=Omega_m rho_crit,0 and C1 units; the generator already uses comoving density at a=1. Verify agreement with the CCL raw IA coefficient. Separately resolve latest GitHub eta=0.5 versus draft eta=0, and regenerate matching configuration/results. | IA numerical validation |
| MA10 | State this as the validated fiducial model, explain numerical versus astrophysical accuracy, and connect to the bin-dependent bias cost table. | Editorial |
| MA11 | Write We adopt the NLA model for the fiducial tests and apply the same choice-based language to other modelling assumptions. | Editorial |

#### IA and magnification corrections require one declared convention

For Eq. 3.2 use rho_m,0=Omega_m,0 rho_crit,0, D(0)=1 and the stated C1 units. The generator already evaluates rho_x at a=1 with is_comoving=True and C1=5e-14/h^2. This is consistent with a comoving mean-density convention; do not add an extra (1+z)^3. CCL receives a signed, already normalized IA coefficient when use_A_ia=False.

A separate current-version issue needs resolution: GitHub 7d29b2f changes eta_pivot from 0 to 0.5, while the manuscript still states eta_IA=0. Stored JSON and old figures do not automatically acquire the new law. Determine the intended fiducial eta from the adopted scientific setup, then regenerate affected products and update the table together. This review does not choose between the values.

For magnification distinguish the count slope s from the response q=5s-2. Eq. 3.3 currently labels the stored slopes (0.659,...,0.994) as a response. The covariance writer passes slopes to CCL, while benchmark and validation paths require an audit of where q is computed; passing q into a CCL argument expecting s transforms it twice. Use the same physical field in the reference and LimberCloud before recalculating residuals. [PyCCL tracer API](https://ccl.readthedocs.io/en/latest/api/pyccl.tracers.html).


### Current Section 3.3 — Tomographic redshift distributions

Draft source: [section3.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section3.tex#L27).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| JZ04 | Cite the actual DESC SRD and link the forecasting/analysis_choices repository and binny separately; identify the version used to generate the inputs. | Source provenance |
| MA12 | Introduce grid nodes versus intervals in the method and list the numerical values in the validation setup, distinct from tomographic bin counts. | Editorial |
| MA13 | Explain Delta z=0.01 as the adopted setting, not a required or optimised value. Use a small fixed-physics resolution-sensitivity check to support the numerical claim; defer a comprehensive search for the best grid. | Bounded sensitivity validation |
| MA14 | Create a compact Y1/Y10 settings table including per-component sigma_e, area, f_sky, per-bin densities, distribution source and deviations from the SRD. | Configuration provenance |
| MT04 | State that coarser/finer redshift grids are possible and change coefficient cost/accuracy; comprehensive grid optimisation is future work. Retain targeted sensitivity checks needed to interpret the adopted setting. | Bounded sensitivity validation |

#### Grid, normalisation and source provenance

Separate tomographic bins from numerical redshift intervals: 351 nodes correspond to 350 intervals for Delta z=0.01 over 0<=z<=3.5. Describe resampling, normalisation weights and endpoint handling. The scientific distributions and numerical grid are separate choices. A finer numerical grid should reconstruct the same intended distributions before attributing a convergence trend to the projection.

Delta z=0.01 is a practical choice in this demonstration, not a mathematical requirement or an established optimum. Finer grids can reduce representation error while increasing coefficient-tensor construction and storage costs; they cannot fix mismatched physics, estimator or endpoint conventions. A small comparison at selected coarser/finer settings and representative difficult pairs is sufficient to assess sensitivity for this revision. Do not require a broad redshift-resolution campaign or promise monotonic accuracy for every sample. Defer adaptive grids and optimisation of accuracy versus cost to future work.

For the settings table include Y1/Y10 lens and source bin counts, z support/grid spacing, per-bin n_eff, per-component shape dispersion, sky area/f_sky, IA law, magnification slopes, galaxy-bias model and the source/version of each. Preserve units and define shape noise consistently with the covariance software. Cite the [DESC SRD](https://arxiv.org/abs/1809.01669); the 2009 Science Book currently cited as SRD is a different document. Link the actual distribution-generation package and analysis choices without asserting that an unavailable forecasting repository has been inspected.


### Current Section 3.4 — Angular power spectra and covariance matrices

Draft source: [section3.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section3.tex#L60).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| MA15 | Define ell edges/centres/window, k_n(ell)=(ell+1/2)/chi_n, units and valid k range; give chi=0 and extrapolation handling. | Estimator and range validation |
| MT05 | State that the present ell binning follows the intended LSST DESC SRD setup, verify its exact source/version and edges, and clarify that other ell sampling/binnings are possible. Defer alternative-strategy optimisation while matching estimators now. | Estimator and provenance validation |
| MT07 | Implement the covariance adapter/input-order/output-index checks in the code plan before interpreting statistical errors. | Covariance validation |

#### Use one observable estimator throughout

Define whether a data element is a sampled C_ell at one multipole or a bandpower sum/integral with specified weights. The current CCL notebook values at geometric centres and the LimberCloud uniform-dell averages are different estimators. In the inspected OneCovariance revision, integer-rounded ell edges also differ from the notebook's floating edges. First establish pointwise agreement on identical ell values, then apply the same verified band window to both spectra and the covariance. Keep the 101 raw input ell samples distinct from the 20 analysis bands.

The number and placement of evaluated multipoles and the number of analysis bands are configurable choices. More evaluated multipoles generally require more projection work; adding bands need not increase this work if they reuse the same raw spectra. Smooth angular spectra may support sparse evaluation with subsequent interpolation, but that interpolation must be checked at the required precision. This paper keeps its declared SRD-motivated choice and does not optimise alternative ell strategies. Verify the precise SRD prescription and document any deviation instead of inferring it solely from the meeting note.

Choose and label the covariance component. Current error notebooks load MATRIX.ascii, but surrounding prose alternates between Gaussian and Gaussian+connected non-Gaussian+SSC. Start with a validated Gaussian observed-field covariance if this is the intended transparent numerical benchmark; preserve or restore other components only with their actual model assumptions documented. Supplying full observed C_ell does not by itself establish consistent magnification terms in internally calculated NG/SSC.

The input ordering repair must precede covariance regeneration; the output pair-index repair must precede plotting or likelihood use. A simultaneous row/column permutation preserves eigenvalues and cannot repair an indefinite already assembled matrix. Current local VAECloud artifacts are not the successful corrected reference recalled by the user: both fail Cholesky. Retrieve the exact successful configuration/revision before treating it as a regression fixture.


### Current Section 4 — Piecewise Linearisation of Physical Quantities

Draft source: [section4.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section4.tex#L1).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| JRZ06 | Share chi and appropriate ratio limits, suppress repeated ticks, and use sensible per-panel signal limits when amplitudes differ; enlarge fonts at manuscript print size. | Figure regeneration |
| JRZ07 | Merge current Sections 4 and 5 into a section entitled The LimberCloud method, led by the central contraction. | Editorial |
| JRZ08 | State spatial flatness in the established formalism, then refer back without reintroducing it in every derivation. | Editorial |
| JRZ10 | Simplify the main LimberCloud–CCL figures while retaining a focused NUMERIC linear/quadratic/cubic appendix comparison testing the adequacy of the adopted representation. Define each interpolated object/coordinate and avoid unnecessary overlapping curves. | Figure and numerical validation |
| MA16 | Treat this as a flagged claim for scrutiny; shorten to an unimplemented extension rather than promising an unchanged analytic structure. | Editorial |
| MT03 | Explain the author's motivation explicitly: alternative interpolants test whether linear distribution/kernel-input and power representations are adequate. Treat cubic as a comparator, not a proven exact or universally stable result. | Diagnostic validation |

#### Keep diagnostic scope modest

The aim is to explain why the chosen representation supports closed-form radial integration and whether its error is adequate for the demonstrated use. Use NUMERIC as the method family and linear/quadratic/cubic as interpolation settings in captions and tables. Identify the exact function interpolated in each diagnostic, its coordinate and nodes, support/extrapolation and numerical quadrature tolerance. Numerical quadrature of the same representation, including its special first interval, tests the analytic implementation; changing interpolation order additionally tests representation sensitivity. Keep those conclusions separate, and vary one interpolated object at a time in targeted diagnostics when attribution matters. The end-to-end comparator can use its documented combined settings. In the inspected notebooks, the order setting changes radial phi, a(chi), and effective power-times-amplitude together; the analytic path instead interpolates 1+z directly. Consequently the existing NUMERIC–linear curve is not automatically an independent integration of exactly the same analytic integrand.

Do not describe the final lensing kernel itself as piecewise linear merely because its input distribution is linearly interpolated: integrating the basis produces polynomial and logarithmic terms. Avoid the general assertion that higher-order interpolation necessarily breaks analytic integrability; the conclusion depends on the variable and functional family. Cubic interpolation is a useful established comparison, but can overshoot or be sensitive to boundaries; reference-tolerance and modest refinement checks are still required. Main error panels can remain focused on CCL and LimberCloud, with the motivated alternative-scheme comparisons in an appendix and the saved analysis products.


### Current Section 4.1 — Three-dimensional power spectrum

Draft source: [section4.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section4.tex#L3).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| JZ05 | If retaining the P(k,z) diagnostic, use positive-distance log chi or a low-chi inset and explicitly exclude chi=0 from log display. Keep the operational Limber-trajectory plot; optionally add a familiar P(k) reference. | Figure regeneration |
| JRZ09 | Condense to a short explanation of linear interpolation along k=(ell+1/2)/chi, its endpoint samples and accuracy controls. Keep the equation in the method; move the diagnostic to validation/appendix. | Editorial |
| MA17 | Explain that the interpolated function follows a Limber trajectory in (k,z), so the chi plot tests the actual algorithm. Add a small explanatory k-z diagram or appendix P(k) comparison only if useful. | Editorial |
| MA18 | Define what is interpolated and held fixed, then use the linear/quadratic/cubic numerical comparison to test the adequacy of the adopted power representation. Put detailed curves in a focused appendix diagnostic. | Diagnostic validation |
| MA19 | Implement the same positive-chi log view or inset as JZ05. The identity of the dark-ink writer is not established independently; retained under the supplied PDF source. | Figure regeneration |
| MT06 | Clarify the interpolated function along the Limber trajectory and its special first-interval treatment. Describe alternative power representations, including local power laws, as future work requiring fresh derivation and validation. | Formula and bounded sensitivity validation |

#### Explain the sampled power function

The interpolation is along F_ell(chi)=P_uv((ell+1/2)/chi,z(chi)), not just P(k) at a fixed redshift. Specify endpoint samples F_ell(chi_n), chi-grid construction and finite k-domain handling. The notation should distinguish the interpolant from the exact input function, using an approximation sign or a separately defined interpolated symbol. At chi=0 the nominal Limber k diverges; document the implemented limiting/support convention and test it independently. A claimed negligible low-chi contribution requires a weighted projected-error check, not only a small kernel plotted by eye.

Separate the representation of distribution values from the power-spectrum representation. The current ordinary-interval coefficients assume power linear in chi along the Limber trajectory. The power notebook uses the special observer interval F_ell(chi)=F_ell(chi_1)(chi/chi_1)^3, not a linear segment from zero. Make this convention explicit in the paper and check its consistent use in all coefficient families; it does not establish that the physical power function is globally almost linear. A different cosmology/power provider can supply node values, but using an arbitrary continuous P function inside the radial integral does not make the current closed-form formulae valid for that function. Local power-law interpolation may be useful, but needs its own coefficient derivation or numerical integration, endpoint/sign handling and accuracy/cost comparison. Defer that development and optimal-representation search to future work.


### Current Section 4.2 — Number-density kernel

Draft source: [section4.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section4.tex#L20).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| JZ06 | Write phi_i=H(z_i;theta)p_i/c; reserve survey-only language for p_i and absorb the H/c factors into a transformed tensor when making the strict separation claim. | Formalism audit |
| N08 | Regenerate the panel layout at final manuscript size and verify readability, including ticks and ratio labels. | Figure regeneration |
| MA20 | Adopt a consistent basis-function notation and alignment for power, density and lensing kernels; define indices before displaying them. | Editorial |
| MA21 | Name the tested function and reference in the caption; restrict the main plot to CCL and LimberCloud, with consistent colour semantics. | Figure regeneration |
| MA22 | Replace layout narration with what is being tested, the reference, residual definition, denominator treatment and scientific takeaway. | Editorial |

#### Make survey-only dependence exact in the notation

The existing phi_i=H(z_i;theta)p_i/c is cosmology dependent. On a fixed redshift grid define D_ii(theta)=H(z_i;theta)/c, phi_a=D p_a, and Bhat_uv=D^T B_uv D. Then C_uv^(ab)=p_a^T Bhat_uv p_b. Absorbing D into each kernel basis is equivalent and may be clearer in the derivation. This is a diagonal rescaling of the existing algebra, not a new projection method. For different grids use the corresponding left/right D and basis mapping explicitly.

Reserve exact for bilinearity of the fixed discretised model. The original continuum Limber integral still has finite-grid, interpolation, support and numerical-input errors. Use the p-based tensor consistently as the proposed emulator target and in the marginalisation response.

Explain the choice of a piecewise-linear density basis as a simple representation linear in the node values with tractable radial integrals. Do not claim it is the only representation that permits factoring out those values: a fixed spline or other basis whose coefficients depend linearly on the node values can also preserve bilinearity. Shape-preserving or adaptive interpolants whose weights depend nonlinearly on the data require separate treatment. The present closed-form coefficients, support structure and runtime evidence belong to the adopted basis; extending them is beyond this revision.


### Current Section 4.3 — Lensing convergence kernel

Draft source: [section4.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section4.tex#L56).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| MA23 | Use clearly grouped node indices and consistent N_grid/N_interval notation; inspect equation typography in the final PDF. | Editorial |
| MA24 | Show where the kernel carries significant weight, identify near-zero denominator effects and boundary artefacts, and quantify their effect on the projected spectra. | Convergence and estimator validation |
| MA25 | Apply JRZ10 consistently to Figures 2-7; retain the focused interpolation-adequacy comparison in the appendix and remove redundant curves from main panels. | Editorial |

#### Endpoint and formula checks discovered in this audit

The expression at source label `eq:4.9b` (displayed Eq. 4.10) contains n+2 and needs a special case on the final interval n=N_z-1; the generic last-node expression also assumes a full supporting interval. The text saying only i=n has a partial interval is false: i=n+1 contains one too. Either derive the truncated endpoint expressions or impose and validate an explicit zero-endpoint distribution convention. Do not hide this behind notation cleanup.

Independent ordinary-interval checks and agreement between Numba/JAX ports are useful but insufficient for endpoints. The 15 September audit interpreted the first NN interval using linear power; the fresh review found an explicit cubic observer-interval convention in the power notebook. The NN values 1/12, 1/12 and 1/4 agree with that cubic convention, so do not change 1/4 to 1/2 based on the old premise. Reconcile this convention with every coefficient family and the derivation text, and independently check the separately identified omitted final diagonal. Twenty-one derivation notebooks already exist: preserve and replay them, adding boundary and signed-amplitude cases rather than claiming there is no previous derivation evidence. Observer-limit tests must implement the actual power/support convention and verify integral finiteness; reject divergent fixtures and test nonzero far-endpoint density separately.


### Current Section 5 — Tensorised Reformulation of the Limber Projection

Draft source: [section5.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section5.tex#L1).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| JZ07 | Remove the forecast about model standardness; describe the validated fiducial linear-bias case and the bin-dependent scale-dependent case separately. Cite the nonlinear-bias challenge and state its extra basis cost. | Source provenance |
| JZ10 | Compute the joint covariance-weighted residual on an explicitly selected, identically ordered and identically binned data vector; explain when it equals a likelihood change. | Covariance and estimator validation |
| JRZ12 | Implement JZ10, retaining the full cross-observable covariance for the joint statistic. | Covariance and estimator validation |
| JRZ13 | Merge current Sections 4 and 5; keep a readable derivation and move lengthy coefficient cases to an appendix. | Editorial |
| JRZ14 | Preserve the common layout and show absolute fractional residuals and absolute delta C/sqrt(Cov_ii) on log y axes, with units and retained/excluded bins explicit. Use signed views only as supplementary diagnostics. | Covariance and figure validation |
| MA26 | Say each kernel is a weighted sum of the sampled distribution values, with weights determined by cosmology and the grid. | Editorial |
| MA27 | Show the equation and physical interpretation at the end of the introduction, or opening of Section 2; refer back at the start of the method and derive it only once there. | Editorial |
| MA28 | Use a short structural table for density-density, density-lensing and lensing-lensing kernels; put detailed 3/8/10 formula families and boundary cases in an appendix. | Editorial |
| MA29 | Rewrite the bias-regime paragraph into three short cases with assumptions and costs, and remove repeated assertions of novelty. | Editorial |
| MT02 | Implement MA27 and the new method-first structure. | Editorial |

#### Formula families, reusable biases and validation flow

Explain 3, 8 and 10 as structural analytic formula families for density-density, density-lensing and lensing-lensing contributions. They are not the total number of populated tensor entries. Include a small kernel-type/support table and a reproducible appendix case map. The mixed tensor has a first subdiagonal in the inspected orientation, so the discussion must not describe the entire lower triangle as zero. Reverse mixed orientations are transposes; symmetry needs the physical legs to match.

Give three bias regimes: (1) shared functions of k,z reuse a common tensor per physical contribution; (2) per-bin constants can weight distribution vectors, preserving reuse; (3) independently bin-dependent functions of k,z require separate tensors or a separately validated expansion into shared operator spectra. Bin-independent NLA here means the adopted common amplitude/redshift law, not a universal property of IA modelling.

Move the empirical content of current 5.1-5.3 into new Accuracy. Explain component assembly in the method once; place all reference definitions, statistics and masks next to the comparisons.


### Current Section 5.1 — Shape–shape correlation

Draft source: [section5.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section5.tex#L40).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| JZ08 | Use one triangle for symmetric shear pairs and a dedicated top/empty-panel legend; verify the rendered page. | Figure regeneration |
| JRZ11 | Implement JZ08 with a shared external legend and check every accuracy panel. | Figure regeneration |
| MA30 | Use the same decision as JRZ10/MA18; no unexplained comparator remains in the main accuracy figures. | Editorial |
| MA31 | Distinguish formula families from nonzero entries in a tensor; explain the case partitions and symmetries with a small table or appendix example. | Formalism audit |
| MA32 | Use r_C or epsilon_C for signed relative residual, define its absolute version separately, and reserve delta for density contrast. | Editorial |
| MA33 | State CCL as the reference, use sqrt(Cov_ii)/abs(C_i) for fractional uncertainty, and replace structured with a description of the observed scale dependence after revalidation. | Covariance and estimator validation |
| MA34 | Use unique upper-triangle EE panels and a dedicated legend; verify symmetry numerically before removing duplicates. | Spectrum and figure validation |
| MA35 | Explain both directions of this comparison; use covariance-normalised residuals and the joint quadratic statistic, rather than declaring relevance from fractional error alone. | Covariance validation |
| MA36 | Apply the shared top/empty-panel legend design and render-check all labels. | Figure regeneration |

#### Statistical interpretation and plot definition

For each matched cosmology s, define delta C_(s,A)=C_(s,A)^LC-C_(s,A)^CCL, signed fractional residual r_(s,A)=delta C_(s,A)/C_(s,A)^CCL only where the reference is meaningfully nonzero, and local normalised residual u_(s,A)=delta C_(s,A)/sqrt(Sigma_AA). The main plots use abs(r) and abs(u) on log y axes; this preserves readable dynamic range through changes of sign. A crossing through zero remains a real zero and is not numerically repaired by taking an absolute value. If a fractional uncertainty is shown, it is sqrt(Sigma_AA)/abs(C_(s,A)^CCL), not the diagonal variance divided by C. The existing label Covariance is therefore too imprecise. Undefined ratios must be masked/reported; do not fill zeros with artificial 0%, 100% or unity values.

Use unique EE pairs for the science vector, one consistent legend and readable final-size axes. Retain the established main error layout and a prominent named-fiducial curve. After inspecting the ensemble, add a restrained median and pointwise 16th–84th percentile band for LimberCloud versus CCL if readable. Calculate abs(r) or abs(u) separately for every cosmology before taking quantiles: abs(median(r)) is not median(abs(r)). The denominator for each fractional error is that cosmology's own CCL spectrum. Exclude the separate fiducial from ensemble quantiles and record the valid sample count at each plotted point.

Label the shading as variation across the sampled cosmologies, not observational uncertainty, a posterior interval or a simultaneous envelope containing 68% of whole curves. Compute summaries from raw values. On log axes mask exact zeros or mark them below a clearly labelled display floor, retain raw zeros in the products, and do not use a positive plotting floor to alter statistics. A percentile band touching zero requires the same display treatment. Signed residuals remain available in diagnostic notebooks for investigating coherent errors and zero crossings. Smaller fractional errors in highly precise bins can matter more than large ratios of nearly vanishing signals.


### Current Section 5.2 — Position–shape correlation

Draft source: [section5.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section5.tex#L59).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| N04 | Implement a labelled Niko/binny selection: unit-normalised minimum-overlap integral <=0.10 for Y1 and <=0.25 for Y10, with source peak above lens peak, as verified in the pinned binny example. Save the exact pair/scale mask and mark exclusions. Do not imply this establishes every SRD implementation or infer surviving pairs without the actual distributions. | Selection definition and covariance validation |
| MA37 | Use a covariance-normalised residual view or limits that contain the reference curves, and make any differing axis limits explicit. | Figure regeneration |
| MA38 | First compute the joint discrepancy; then, if non-negligible or needed for the claim, compute a local Fisher bias with the same mask/covariance and validated parameter derivatives. Do not claim a posterior test from this approximation. | Covariance and derivative validation |
| MA39 | Describe agreement with a matched CCL calculation; perform tolerance/reference convergence and do not identify CCL with exact physics. | Reference convergence |

#### Pair selection and inference-impact measurement

Use an explicit, versioned selection rule. A verified binny example implements foreground lenses using source peak redshift greater than lens peak redshift and overlap thresholds 0.10 for Y1 and 0.25 for Y10. For individually normalised p(z), its minimum-overlap measure reduces to integral min[p_lens(z),p_source(z)] dz. This is an operational Niko/binny choice, not proof that every SRD implementation uses this exact metric. Record the quadrature, peak definition and strict/non-strict boundary comparison, then enumerate the surviving pairs from the actual input arrays. [Pinned binny example](https://github.com/binny-org/binny/blob/eb913fe99675019d8f410ef40d1b5aed4b560291/docs/examples/tomography/selections.rst#L241).

Keep every field cross-spectrum needed to construct covariance; apply the science mask afterwards to the data vector and both matrix axes. Keep the full selected lens-by-source rectangle; do not impose a<=b as for same-population spectra. The reciprocal source-by-lens orientation need not be duplicated.

For the selected vector compute D_model=delta C^T Sigma^{-1}delta C by a Cholesky solve. This is a model-discrepancy statistic. It equals the chi-square difference for a noiseless CCL reference at the same parameters and covariance. For real data with r=d-C_CCL, the actual difference is D_model-2 r^T Sigma^{-1}delta C, so do not equate D_model with an arbitrary observed-data log-likelihood shift. Per-probe quadratic forms generally do not sum to the joint value because cross-probe covariance matters.

Compute D_s independently for every matched cosmology, using one validated fiducial covariance per survey and a fixed labelled selection as the primary comparison. This deliberately measures discrepancies across the domain against one common survey error scale; it does not assert that the physical covariance is independent of cosmology. Reuse its factorization after the full covariance correctness gate. Covariance recomputation at a few representative cosmologies can assess sensitivity if the result or claim warrants it; recalculating all 1,000 covariances is not required for this revision.

Report full D_s as the primary quantity. D_s/N_data can additionally be reported as discrepancy per selected data-vector element. If the paper retains the phrase reduced chi-square, explicitly define this normalization and avoid a fitted degrees-of-freedom claim: no parameters are fitted in this same-parameter deterministic comparison, and its ideal value is zero, not the noisy-data goodness-of-fit expectation near one. Do not average away a substantial full-vector discrepancy, treat the 1,000 cosmologies as independent observed surveys, or interpret their summed D as a survey statistic.

Start the manuscript ensemble summary with a compact table: survey/method, N_data, successful/attempted matched counts, fiducial D, sampled median D, 16th–84th percentile range, 95th percentile and maximum with sample ID. If too wide, put tail/failure details in the supporting table and retain a direct link. The 68% interval describes empirical variation under the declared sampling design, not uncertainty on the median or a posterior credible interval. Decide on an additional distribution plot only after inspecting the data: a table and pointwise bands may suffice for small variation; an empirical cumulative distribution highlights tails; parameter-versus-D scatter plots diagnose systematic dependence. Use violins only if comparing distributions adds information and the density smoothing is appropriate.

If parameter-bias claims remain necessary, use a local Fisher calculation with validated derivatives and the same masks/covariance; label it a local approximation. Marika's mentioned KiDS code is a possible implementation reference, not an accessed/verified dependency in this review. A full MCMC campaign is conditional on the size of the discrepancy and intended claim.


### Current Section 5.3 — Position–position correlation

Draft source: [section5.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section5.tex#L76).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| MA40 | Name the actual published analysis or forecast and date/version, or say planned Stage-IV configurations; remove generic claims unsupported by a citation. | Literature verification |
| MA41 | Plot unique TT pairs, distinguish selected autos from diagnostics, give a precise summary and worst cases, and evaluate their joint impact. | Covariance and estimator validation |
| MA42 | Include the affected pairs in bounded resolution-sensitivity checks and separate grid error, ell-window mismatch, cancellations, endpoint effects and CCL tolerance before assigning a cause. A global search for optimal Delta z is deferred. | Bounded sensitivity validation |

#### Scale cuts and error budgets

List the TT pairs actually selected by the adopted analysis and mark excluded scales. Autos-only is a documented analysis choice, not a universal rule for every 3x2pt study. k_max must have stated units. A conversion ell_max=k_max chi(z_upper)-1/2 does not guarantee k<k_max everywhere in a broad redshift kernel; document the chosen effective-redshift or window-based prescription and test sensitivity.

Do not dismiss numerical error because photometric-redshift or nonlinear-bias uncertainty is large. Its acceptability follows from the validated numerical budget on the selected vector. Compare affected pairs at a small number of Delta z values, fixing the ell estimator and reference settings, to distinguish discretisation error from estimator differences, endpoint behaviour and small-signal cancellations. This is a targeted sensitivity check, not an optimal-grid study. Both codes using Limber establish agreement within Limber; this does not validate the physical Limber approximation at the lowest multipoles.


### Current Section 6 — Evaluation and Discussion

Draft source: [section6.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section6.tex#L1).

No separate direct comment is assigned solely to this heading; implement the linked structural/global changes and the contextual checks below where applicable.

No additional unique comment is assigned only to this parent heading. Rebuild its introductory transition after moving accuracy into new Section 4 and splitting measured performance from inference implications. Do not state that sub-percent accuracy has already been established until the new quantitative table exists.


### Current Section 6.1 — Computational performance

Draft source: [section6.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section6.tex#L3).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| JZ09 | Reconcile plotted arrays and claims. Show comparable end-to-end times, label any CCL-total reference as total, and remove it from stage panels if it invites a false stage comparison. | Timing validation |
| MA43 | Benchmark matched end-to-end and fixed-cosmology nuisance-update workloads; report isolated contraction time explicitly, without promoting it to a full prediction time. | Timing validation |
| MA44 | Regenerate from recorded stage arrays with correct labels; report hardware, compilation, synchronisation, repeated evaluation statistics, and explain measured ranking only when supported. | Timing validation |

#### A defensible timing table

Report at least: cold initialisation/JIT; warm full prediction (cosmology+coefficient+contraction+required transfers); warm coefficient plus contraction with a supplied cosmology/power state; and fixed-basis contraction for a changed redshift distribution. State how many ell outputs and bin pairs are produced, output precision and hardware. A complete likelihood also includes its covariance/data operations, so a spectrum timer should not automatically be labelled likelihood time.

Use a common deterministic cosmology sample list and identical tracer physics. CCL's lazy CAMB/power work can be charged to a later call; its constructor time is not necessarily its full cosmology cost. The CCL scripts already save COSMOLOGY and CELL stage files, but these boundaries are not automatically comparable with LimberCloud's. The benchmark plotting code repeats CCL TOTAL in every stage panel. Remove that repetition or make the reference unmistakable; never interpret it as a measured CCL coefficient/cosmology stage.

Retain one execution task with a serial loop over the 1,000 sampled cosmologies for each benchmark configuration. Numba's internal threading and JAX's compiled CPU/GPU execution remain part of the measured implementation; record allocated versus effective CPU/thread/device settings. The accumulated-time checkpoints remain at 100, 200, ..., 1,000 sampled evaluations, excluding the additional fiducial and warm-up. The current workload mapping is Single=EE, Double=TE+TT, and Triple=EE+TE+TT; preserve and explain it, with identical requested outputs across implementations. Saving spectra is outside compute-stage timers, with end-to-end wall time and serialization overhead available separately. Do not introduce task arrays, multiprocessing across cosmologies, multi-GPU throughput or ideal scaling as part of the paper's benchmark evidence. Those are future analysis/deployment options.

Figure notebooks consume the accepted saved spectra/timings and metadata; they do not rerun the 1,001-case calculations. The saved-spectrum name and schema should follow the code plan, including explicit NUMERIC interpolation identity. Distinguish NUMERIC integration timing from isolated analytic contraction timing and avoid promising a production run duration before pilot measurements.

Current JAX calls already synchronize results; preserve those barriers and separate warm-up explicitly. Correct sampled-cosmology prefactors, magnification assembly and ell estimators before reusing any runtime result for a physical comparison. Include an efficient CCL fixed-cosmology/tracer-caching comparison for nuisance updates. Do not compare an isolated GPU contraction to the full CPU pipeline and describe the ratio as an end-to-end algorithm speed-up.


### Current Section 6.2 — Prospects for end-to-end emulation

Draft source: [section6.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section6.tex#L40).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| JZ11 | Add a concise comparison of matter-power, survey-specific data-vector, and proposed tensor emulators, including the verified Boruah/Nygaard references and relevant newer work. Chun-Hao To is first author of LINNA: Likelihood Inference Neural Network Accelerator, To et al. (2023), JCAP 2023(01),016, https://arxiv.org/abs/2203.05583 (DOI https://doi.org/10.1088/1475-7516/2023/01/016). This is the likely intended mapping of Joe's shorthand Chun-Hao et al. 2023; the identification is an inference, while the paper metadata are verified. Check and use exact bibliography records before insertion. | Literature verification |
| MA45 | Reduce it to opportunity, comparison with existing emulators, unresolved accuracy/storage cost, and future validation. Outline these points before writing short prose. | Editorial |
| MA46 | Define training before inference and avoid an unmeasured model-size claim; present two short steps only if the training proposal is retained. | Editorial |
| MA47 | Remove numerical training-time forecasts from the main result. If retained in an appendix, identify model, 10^6 samples, tensor output dimension, storage, hardware, parallel efficiency and assumptions; label as estimates. | Timing and storage estimate |
| MA48 | Explain once which part a P(k) emulator replaces; then state the distinct benefit and cost of the proposed tensor target. | Editorial |
| MA49 | Define throughput as completed predictions per second and remove claims of linear GPU scaling unless measured; distinguish batching from latency. | Timing validation |
| MA50 | Keep approximately three compact paragraphs, with detailed resource speculation removed or explicitly optional. | Editorial |

#### Literature and feasible scope

Use a compact target comparison: P(k,z) emulators accelerate the three-dimensional input; direct data-vector emulators accelerate a particular observable/configuration; a proposed Bhat emulator could retain flexibility under distribution updates but has a much larger output target. Verified starting points are [Boruah et al.](https://arxiv.org/abs/2203.06124), [Nygaard et al., CONNECT](https://arxiv.org/abs/2205.15726), and [To et al., LINNA](https://arxiv.org/abs/2203.05583). The last is the likely interpretation of Joe's Chun-Hao shorthand; CONNECT emulates CLASS outputs and should not be called a demonstrated LSST 3x2pt emulator. Perform a focused final search for relevant more recent work before claiming the literature is complete.

Remove guarantees about one million training points covering 10-50 parameters, lightweight networks, one-day training, ideal multi-GPU scaling and additional 3-4 orders of speed-up. These are research hypotheses. A million output values for a million samples already implies roughly 8 TB in float64 before metadata/replication; compression and accuracy must be demonstrated. Even zero-cost tensor emulation leaves the contraction cost, so the projected gain must not be multiplied by another arbitrary gain. Keep only the proposed training/compression/error-validation tasks and the specific survey-flexibility benefit.


### Current Section 6.3 — Analytic marginalisation over redshift distribution uncertainties

Draft source: [section6.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section6.tex#L56).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| MA51 | Begin with the quadratic density response and the useful analytic derivative, avoiding another recap of the emulation motivation. | Editorial |
| MA52 | Define a flattened nuisance-mode vector, data index, lens/source matrices and tensor for each element. Write Gamma as an explicit derivative/contraction and document dimensions. | Editorial |
| MA53 | State exact algebra for fixed discretised representation and linear density coefficients; separate radial interpolation error from nuisance-response truncation. | Formalism audit |
| MA54 | Say that dropping the quadratic term in the prediction makes the likelihood Gaussian in the nuisance coefficients; a quadratic term in the likelihood still remains. Move the quartic expansion to an appendix. | Editorial |
| MA55 | Present Sigma_eff=Sigma+Gamma Pi Gamma^T immediately after defining Gamma, retain the parameter-dependent log determinant, then discuss exact quadratic response and limitations. | Formalism audit |
| MA56 | Provide component-level Gamma for each data-vector element, including both legs and cross-bin nuisance correlations, with a small worked derivation. | Formalism audit |

#### Precise marginalisation formulation to insert

Use separate matrices for each population: C_uv=Phi_u^T B_uv Phi_v. This matters for Y10's rectangular lens-source combination; one common Phi does not describe different lens/source grids or populations. Prefer the strictly survey-only p/Bhat convention introduced in the method. Let x contain all independent nuisance-mode amplitudes, including correlations between bins/populations, and write p_a=bar p_a+F_a x. Here a and b each denote a population and bin, not a bin number alone. If x has m components and the relevant population has G grid nodes, F_a has shape G-by-m and Gamma has shape N_data-by-m. Normalisation-preserving modes satisfy w^T F_a=0 for the adopted redshift quadrature weights.

For each independent data element A=(u,v,a,b,ell or band),

    M_A(x)=bar M_A + sum_mu Gamma_A,mu x_mu + x^T Q_A x,
    Gamma_A,mu = sum_ij Bhat_A,ij [F_a,i,mu bar p_b,j + bar p_a,i F_b,j,mu],
    Q_A = sym(F_a^T Bhat_A F_b).

Define sym(U)=(U+U^T)/2. Sum component/window contributions where A is an observed bandpower. The response is explicit and cosmology dependent. Exact quadratic dependence applies to this fixed-grid linear density/mode parameterisation. A shift p(z-delta z), a positivity transform or a nonlinear renormalisation is not exactly linear in x and cannot inherit the polynomial claim unchanged.

After dropping x^T Q_A x from the **prediction**, independent additive Gaussian data noise with covariance Sigma independent of x and a zero-mean Gaussian prior Pi give

    Sigma_eff(theta)=Sigma+Gamma(theta) Pi Gamma(theta)^T,
    -2 log L_marg = Delta^T Sigma_eff^{-1} Delta + log det Sigma_eff + constant,
    Delta = data - bar M(theta).

State the noise/prior assumptions and retain the parameter-dependent log determinant. If transforming a prior from p-space to phi-space, transform its covariance with the same cosmology-dependent D; holding that transformed prior fixed would change the statistical model. Working in p-space keeps the adopted distribution prior explicit. Normalisation under the chosen z quadrature does not automatically imply exact normalisation of the separately interpolated chi-space density at finite resolution; quantify that representation error, and avoid an additional nonlinear per-proposal renormalisation while claiming an exact quadratic response. This calculation still keeps the quadratic-in-x term generated by the linear prediction in the likelihood exponent. It is not obtained by discarding every term of second order in that exponent.

Including Q generally makes the exponent quartic and the marginal distribution non-Gaussian. Isserlis' theorem provides Gaussian polynomial moments, not a finite closed-form integral of an arbitrary quartic exponential. Optional exact prior-moment diagnostics include mean shift tr(Q_A Pi) and additional covariance 2 tr(Q_A Pi Q_B Pi); matching these moments does not establish an exact Gaussian likelihood. A controlled perturbative marginal requires an error assessment and the appropriate completed-square Gaussian weighting. Positivity constraints can invalidate an unbounded Gaussian approximation; choose a regime/mode prior where that approximation is justified or state the limitation.

This resolves Niko's concern without losing the contribution: the framework gives the linear response directly and retains the full quadratic response for future accuracy checks. It does not establish that all nonlinear marginalisation is already solved or validated.


### Current Section 6.4 — Future extensions

Draft source: [section6.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section6.tex#L113).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| JRZ16 | Replace the long speculative section with one short concluding outlook paragraph; retain only defensible possible extensions. | Editorial |
| MA57 | Follow JRZ16; keep a single short outlook with clearly unimplemented extensions. | Editorial |
| MA58 | Either add a genuinely needed coefficient appendix and use a label-based cross-reference, or remove the dangling claim; do not cite an absent derivation. | Editorial |

#### Outlook boundaries

Move a short version to the final section. Optimising redshift resolution, adaptive/sparse ell evaluation with interpolation, alternative power representations such as local power laws, and task-based execution in real analyses are future work. State the accuracy/cost motivation without promising an optimal choice or measured scaling. Curvature, CMB lensing and beyond-Limber generalisations are also possible research directions, not tested features of this implementation. Curved f_K(chi'-chi) involves two distances; a delta-function CMB source and its high-redshift support do not automatically fit the current smooth source grid ending at z=3.5; beyond-Limber work introduces unequal-time/transfer structure and additional integrations. Do not promise identical formulae, preserved speed or a modest cost without derivation. Remove the unsupported Appendix reference unless the new appendix actually supplies the cited material. Keep this outlook short rather than expanding every future direction into its own programme.


### Current Section 7 — Summary and Conclusions / code availability

Draft source: [section7.tex](https://github.com/CosmoCloudZhang/LimberCloud/blob/7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c/manuscript/sections/section7.tex#L1).

| ID | Proposed edit / disposition | Required evidence |
| --- | --- | --- |
| JZ12 | Add a code-availability statement with the public GitHub URL and the tested commit/release; distinguish public source from external runtime data and unimplemented extensions. | Release provenance |
| MT01 | Public repository already exists; add its link and versioned reproducibility statement. | Release provenance |

#### Final summary and availability

Summarise the contribution, measured selected-vector accuracy, and explicitly scoped runtime in three short paragraphs. State the supported geometry and bias model, then the controlled marginalisation result and brief outlook. Use the same numerical claims and definitions as the abstract. No new speculation should appear here.

Include [the public LimberCloud repository](https://github.com/CosmoCloudZhang/LimberCloud), the actual commit/release used for the results, access instructions for example inputs/configuration and the shared sample table, and the distinction between tracked code/publication figures and external runtime data. The review commit is not automatically the future results commit. Check acknowledgments against the hardware actually used for the reported timings; the existing draft credits Google Cloud while the current run plan targets Perlmutter. Preserve collaborator funding and authorship unless separately instructed.


## 4. Figure production plan

| Figure group | Proposed final treatment | Required input and verification |
| --- | --- | --- |
| Distribution/settings material (current Figure 1) | Keep one compact named-fiducial plot and a Y1/Y10 configuration table | Actual distribution/config hashes; normalisation, support and bin labels; explicit fiducial ID |
| Power and kernel diagnostics (current Figures 2–4) | Move to Appendix B unless a compact example materially explains the method; common axes, positive-chi log/inset where useful; clearly motivated NUMERIC linear/quadratic/cubic comparisons | Stored figure inputs, interpolation object/coordinate and endpoint identities; bounded reference/grid checks; near-zero ratio mask |
| EE/TE/TT accuracy (current Figures 5–7) | Preserve the main layout and absolute fractional errors/absolute delta C/sigma on log y axes; fiducial curve with optional pointwise 16th–84th ensemble band; unique EE/TT triangle; TE remains rectangular; annotate excluded ranges and keep all-pair appendix | Corrected spectrum estimator, certified fiducial covariance, explicit mask and matching CCL denominator per sample; quantiles computed after taking magnitudes; zero/undefined handling and valid counts |
| Y10 | Small main or appendix summary table plus selected/all-pair supplementary panels | Fresh numerical summary with counts, quantiles, maxima and worst-case identifiers |
| Ensemble joint discrepancy (new table; optional figure) | Report fiducial plus sampled median/68% range and tail diagnostics for D; optional D/N_data. Choose an ECDF, parameter scatter or violin only when the observed spread warrants it | Per-sample full-vector quadratic statistics, sample-table identity/domain, selected counts and covariance identity; empirical-interval wording |
| Performance (current Figure 8) | Comparable serial accumulated full-prediction panel plus correctly labelled LimberCloud stage costs; optional fixed-cosmology comparison | Shared 1,000-sample workload excluding fiducial/warm-up, resource/thread/device metadata, consistent stage boundaries and serialization outside compute timer |

Regenerate through the declared producers and copy only validated publication outputs into `manuscript/figures/`. The live figure manifest still names old notebook directories despite the recent rename; repair it as part of the code plan. Record artifact IDs/hashes, effective configuration, selection and software revision in provenance metadata. Inspect at final page size: typography, clipping, legend placement, shared axes, colour consistency, displayed sample and covariance component. A successful TeX build alone is insufficient.

Keep notebook work restricted to loading selected saved arrays, computing summaries and rendering. Bound canvas size at the intended manuscript dimensions; moving computation to scripts alone does not remove the memory cost of oversized plots. Save raw summary tables separately from display transforms so log floors, omitted markers and drawing order cannot change reported statistics.

## 5. Focused additional corrections found by the audit

These are not attributed to co-authors. Several are necessary to answer their comments honestly.

| Finding | Location | Minimal resolution |
| --- | --- | --- |
| Strict survey-only separation currently uses cosmology-dependent phi | Current 1, 4.2, 5, 6.3, 7 | Use p(z) and Bhat=D^T B D consistently, or explicitly qualify the existing phi formulation |
| Current scientific configuration differs from generated/benchmarked quantities | Current 3.1–3.2 and experiment scripts | Set effective model contract; fix active-cosmology prefactors/magnification assembly; resolve eta=0 versus 0.5; regenerate dependent evidence |
| First/final-interval assumptions and formula descriptions need independent checking | Current 4.1, 4.3–5; analytic backends | Document cubic power in the observer interval, correcting the earlier linear-premise NN diagnosis; replay derivations with the actual piecewise convention and test final endpoints independently |
| Error figures use questionable covariance indexing, labels and a different ell estimator from CCL/OneCovariance | Current 3.4 and 5.1–5.3 | Correct producer/consumer contracts and regenerate before statistical interpretation |
| All-pair accuracy and runtime summaries contradict details | Abstract, 1, 5.2, 6.1, 7 | Replace from one authoritative accuracy/timing summary after validation |
| Exact algebra is conflated with a fully solved nonlinear nuisance integral | Current 6.3–6.4 | Use the explicit linear-response marginal plus precise quadratic-response limitations |
| SRD cited as the 2009 Science Book; incomplete/wrong bibliographic fields | main.tex bibliography | Cite actual SRD; verify relevant article metadata, including SciPy's journal as Nature Methods and missing book titles |
| Figure manifest and some documentation point at old directory names | figures/manifest.toml and runtime docs | Verify every producer against the target commit and record regenerated artifact provenance |
| Broad future-extension and training-feasibility claims exceed tested scope | Current 6.2, 6.4 and 7 | Shorten to conditional research directions; remove implied implemented features and unmeasured speed forecasts; defer optimal z/ell strategy, new power interpolation and task parallelism |

## 6. Completion checklist for the eventual revision

1. Maintain an implementation ledger: comment ID, changed source/figure, response text, evidence artifact and unresolved question if any. Multiple IDs may link to one change.
2. Execute E1–E5 using the staged code plan; keep failed/unmatched historical artifacts identifiable instead of overwriting their provenance.
3. Make the structural move separately from scientific edits where practical. Preserve labels first; avoid mechanical line-number-based edits to the older marked PDF.
4. Implement every row in Section 3 or record a reasoned alternative that answers the underlying concern. A source request is not automatically permission to change scientific assumptions.
5. Recompute all numerical prose from accepted summaries; make abstract, introduction, captions, discussion and conclusion agree.
6. Check definitions, units, summation ranges, endpoint assumptions, transpose/symmetry conventions and all equation/figure cross-references.
7. Verify bibliography, code/data availability, actual benchmark hardware acknowledgments and collaborator-supplied front matter.
8. Build a fresh manuscript and visually inspect the complete PDF. Only then mark comments addressed and prepare concise co-author responses with the actual revised text.

## 7. Manuscript-agent handoff

Use the manuscript prompt in [CURSOR_IMPLEMENTATION_PROMPTS.md](CURSOR_IMPLEMENTATION_PROMPTS.md) with this plan and the code plan available in the execution checkout. First reconcile the target branch/commit and preserve in-progress author changes. Make the structural/notation/source-link edits and comment ledger before accepted numerical products arrive; keep numerical claims explicitly pending rather than importing old results into the new ensemble narrative. Use the named fiducial, shared sample-table ID, validated covariance/mask and saved per-sample summaries supplied by the code agent for final tables and figures. Apply this update's dispositions when they supersede the 15 September audit or a tentative earlier recommendation.

After E1–E5 products pass their gates, regenerate figures, choose the optional ensemble-distribution visualization from the measured spread, update numerical prose together, verify bibliography and availability, compile and visually inspect every page. Return a source diff, comment-response ledger, figure/summary provenance and a clear list of unresolved scientific choices. Do not claim production runs, optimized grids, arbitrary-function analytic integration or noisy-data reduced-chi-square acceptance that the evidence does not establish.

**Review scope:** the original audit covered scientific TeX sections, main document/bibliography, figure manifest and source feedback, plus code/configuration/notebook/runtime contracts. The 19 September update reread the GitHub implementation and revised these plans; it did not edit source/manuscript files, access Perlmutter runtime products, submit jobs, measure new ensemble accuracy or perform an emulator/posterior study. Historical primary references remain evidence citations; any time-sensitive publication/release assertions must be rechecked at manuscript execution.
