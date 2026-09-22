> 22 September execution routing: use the current [code plan](../CODE_REVISION_PLAN.md) and [new prompt index](../CURSOR_IMPLEMENTATION_PROMPTS.md). This source audit retains its original historical findings and uncertainty; current author decisions supersede conflicting provisional choices.

# Manuscript and formalism audit for the comment synthesis

> **Status — 2026-09-19:** This is historical audit evidence, not the current execution plan. The [code plan](../CODE_REVISION_PLAN.md), [manuscript plan](../MANUSCRIPT_REVISION_PLAN.md), and [implementation prompts](../CURSOR_IMPLEMENTATION_PROMPTS.md) take precedence over its provisional recommendations. Code and scientific scripts are edited and validated on NERSC; manuscript sources, accepted publication figures, and the paper manifest are edited locally. The manuscript submodule may remain uninitialized and absent on NERSC; scientific jobs and ordinary checks must not require it. Manuscript paths below identify evidence from the historical code-repository snapshot, not a requirement to initialize or edit the paper remotely. The NN observer correction immediately below remains essential: retain `1/4` for the cubic first interval, not `1/2`. Original findings, author-comment interpretations, and their later recorded corrections are preserved.

> **Historical note, corrected 19 September:** the [fresh source audit](limber_ensemble_followup.md) establishes a cubic observer-interval power policy, under which the NN `1/4` factor is correct. The linear-first-interval diagnosis below is withdrawn; do not replace it by `1/2`. The author's later instructions retain absolute/log main error plots and motivated NUMERIC order comparisons, with optimal-grid and task-parallel studies deferred. Follow the updated [manuscript plan](../MANUSCRIPT_REVISION_PLAN.md) and [code plan](../CODE_REVISION_PLAN.md) where they supersede this preliminary note.

Read-only evidence audit, 2026-09-15. This is an agent working note for the final three requested Markdown documents. No manuscript or scientific implementation has been edited. The comments in the pasted document are evidence, not instructions to send email, alter comment colours, or contact coauthors. The three JZ/YHZ exchanges under **Example comments** are template examples and must not enter the real comment inventory. The substantive list identifies JRZ as **Jaime Ruiz Zapatero** (the user wrote Jaimie).

## 0. Snapshot reconciliation: local checkout versus current GitHub main

The parent retrieved GitHub main at **7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c** in `/private/tmp/limber_review/github-main`; the user's working checkout remains at local57c0731. All seven section TeX files and all core projection implementations are byte-identical. This note's source lines refer to the user checkout unless explicitly labelled GitHub main.

The three relevant source deltas were checked directly:

- GitHub `scripts/generate_config/intrinsic_alignment.py:57` changes **eta_pivot from0.0 to0.5**. It does not change the mean-density convention. Current manuscript §3.2 still says eta_IA=0.0, so the newest generator introduces an additional declared-fiducial mismatch. Resolve intended IA law and regenerate affected artifacts; do not silently choose old or new values.
- GitHub `manuscript/main.tex` removes Nora Elisa Chisari from author block and renumbers affiliations. Title/abstract/scientific text is unchanged. Main.tex science line numbers after this block are two lower on GitHub. Treat authorship as user/coauthor state; do not restore or change it as part of scientific planning.
- GitHub `experiments/spectra/CCL/Y1/single.py` wraps angular_cl output in numpy.array(dtype=float64); no model/timing contract defect discussed below is resolved.

GitHub renamed notebook directories `kernels→kernel`, `error_analysis→error`, `derivations→derivation`, and `matter_power→power` (verify individual final filenames when implementing). **The figure manifest still references old paths**, hence several producer paths are now stale. Final coding plan must repair the manifest and documentation path references against the target snapshot while preserving the user's current checkout. No scientific benchmark or covariance was re-run in this audit.

## 1. What was inspected

All seven manuscript section TeX files, main.tex including title/abstract/bibliography, figure manifest, relevant experiment scripts, fiducial cosmology/galaxy-bias/IA/magnification/survey generators, projection tensor and analytic element implementations, and existing projection tests. Main benchmark/accuracy data reside outside this checkout; PDF exports alone are not newly verified numerical evidence. Existing public Y10 figure exports are present, but no current fresh Y10 maxima or likelihood residual can be claimed from them. The root agent is inspecting Marika's annotated PDF/image and the full notebook/runtime/covariance context separately.

## 2. Current location map

Current top-level titles are declared in `manuscript/main.tex:44–63`.

| Current location | Subsection title / source | Equations and figures |
|---|---|---|
| Title / abstract | main.tex:14 / main.tex:38 | Unqualified sub-percent, 3 ms, additional emulation speed-up, analytic marginalisation claims |
| §1 Introduction | sections/section1.tex:1–11 | Last substantive paragraph gives bilinear headline and >order-of-magnitude speed-up; final paragraph roadmap is stale |
| §2 Theoretical Formalism | sections/section2.tex:1–140, no subsections | Eqs2.1–2.7; Table1 at59–140 |
| §3 Fiducial Survey Configuration | section3.tex:1 | Y10 assertion without shown numbers |
| §3.1 Fiducial cosmology | section3.tex:3–5 | Flat LambdaCDM, CAMB/HMCode |
| §3.2 Astrophysical parameters | section3.tex:7–25 | Eqs3.1 galaxy bias,3.2 IA,3.3 magnification |
| §3.3 Tomographic redshift distributions | section3.tex:27–58 | Eqs3.4–3.6; Figure1 PSI_Y1.pdf at37–42 |
| §3.4 Angular power spectra and covariance matrices | section3.tex:60–64 | 20 analysis bands; 101 covariance input multipoles |
| §4 Piecewise Linearisation of Physical Quantities | section4.tex:1 | Flat geometry introduced late |
| §4.1 Three-dimensional power spectrum | section4.tex:3–18 | Eq4.1; Figure2 POWER.pdf |
| §4.2 Number-density kernel | section4.tex:20–54 | Eqs4.2–4.5; Figure3 PHI_Y1.pdf |
| §4.3 Lensing convergence kernel | section4.tex:56–103 | Eqs4.6–4.12; Figure4 KAPPA_Y1.pdf |
| §5 Tensorised Reformulation of the Limber Projection | section5.tex:1–38 | Eqs5.1–5.6; 21 formulas; bias regimes at36 |
| §5.1 Shape–shape correlation | section5.tex:40–57 | Eq5.7; Figure5 ERROR_EE_Y1.pdf |
| §5.2 Position–shape correlation | section5.tex:59–74 | Eq5.8; Figure6 ERROR_TE_Y1.pdf |
| §5.3 Position–position correlation | section5.tex:76–94 | Eq5.9; Figure7 ERROR_TT_Y1.pdf |
| §6 Evaluation and Discussion | section6.tex:1 | States accuracy already established |
| §6.1 Computational performance | section6.tex:3–38 | Figure8, benchmark_Single_Y1.pdf and benchmark_Triple_Y1.pdf at13–29 |
| §6.2 Prospects for end-to-end emulation | section6.tex:40–54 | Unvalidated 1e6 training set, one-day/linear-scaling/speed-up forecasts |
| §6.3 Analytic marginalisation over redshift distribution uncertainties | section6.tex:56–111 | Eqs6.1–6.8; exact quadratic data-vector expansion vs approximate likelihood marginalisation |
| §6.4 Future extensions | section6.tex:113–121 | Curvature, CMB lensing, RSD, beyond Limber |
| §7 Summary and Conclusions | section7.tex:1–9 | Repeats speed/accuracy/training feasibility claims |
| Acknowledgments / bibliography | main.tex:65–275 | Code availability absent; references include incorrect SRD substitution and some incomplete entries |

Figures are mapped to producers by `manuscript/figures/manifest.toml`: POWER→matter-power notebook; PSI/PHI/KAPPA→kernel notebooks; ERROR_EE/TE/TT→error-analysis notebooks; benchmark figures→`experiments/benchmarks/Y1/benchmark.py`. Equivalent Y10 exports/producers are listed in the same manifest. Figure8 comments filed under old §5 actually belong to **current §6.1**; scale-dependent-bias comment filed under §4.2 is addressed mainly in **current §5 preamble**, with implications for §2, §3.2, §6.2, and the Introduction.

## 3. Recommended high-level structure

Accept Jaime's central recommendation: deliver the mathematical result before the survey configuration and move empirical validation into one coherent evaluation section. This removes three repeated explanations of the same tensorisation and avoids forcing readers to interpret several interpolation schemes before they know the intended computational model.

Recommended final outline:

1. **Introduction.** Physical observables and repeated-projection cost; related projection/emulation methods; the precise new contribution and its assumptions; concise roadmap.
2. **Angular power spectrum formalism.** Define observed fields, physical components, spin factors and shared kernels together. State spatial flatness, Limber, NLA/linear deterministic galaxy-bias validation scope, no RSD. Retain Table1 after explaining its symbols. Do not market two kernels as universally sufficient for all nonlinear bias/IA physics.
3. **LimberCloud.**
   - 3.1 Grid representation and approximation: separate interval index/grid spacing from tomographic bins; distinguish p(z) and phi(chi); explain linear interpolation of P along k=(ell+1/2)/chi and z(chi). Keep compact Eq4.1; no standalone lengthy power-only section.
   - 3.2 Kernel basis: short density hat-function and lensing-integral derivation, ending in W=sum r p. Put expanded Eq4.9 cases and the catalog of analytic coefficients in Appendix A.
   - 3.3 Tensorised projection: derive boxed bilinear contraction early; explain 3/8/10 structural classes without implying only 3/8/10 nonzero entries.
   - 3.4 Parameter dependence, bias regimes and implementation: what is reusable, when to recompute coefficients, backend architecture, complexity/memory, boundary conventions.
4. **Validation and computational performance.**
   - 4.1 Reference cosmology, survey samples and numerical settings (old §3); include Y1/Y10 definitions, selected bin pairs, band windows/cuts and covariance provenance.
   - 4.2 Accuracy and convergence: LimberCloud–CCL, eligible pairs, Delta C/sigma, joint Delta chi-squared, convergence and Y10 table. Detailed raw all-pairs panels and kernel diagnostics in appendix/supplement.
   - 4.3 Performance: fair cold/warm end-to-end, non-cosmology, fixed-basis reprojection benchmarks; stage timing table and corrected Figure8.
5. **Implications for inference.**
   - 5.1 Fast/slow parameter updates (measured contraction capability, sampler performance not measured).
   - 5.2 Controlled analytic marginalisation framework (known polynomial response, linear-Gaussian result, validation deferred). Put quartic-exponent expansion in appendix if it crowds the main argument.
   - 5.3 Emulation prospects and comparison to direct-data-vector emulators (short, no unbenchmarked training-time guarantees).
6. **Summary and outlook.** Three verified results, two limitations, short paragraph on extensions. Move/shorten old §6.4 here as Jaime requests; avoid a second long future-work review.

Appendix A: kernel coefficients, endpoint conventions, closed-form integral catalog or precise linked executable derivation.
Appendix B: interpolation/kernel diagnostics (old Figures2–4, possibly simplified) and convergence checks.
Appendix C: full all-bin Y1/Y10 residuals plus covariance/data-vector ordering and additional benchmark details.

A smaller-change alternative is retaining seven numbered sections but merging current §§4–5 under LimberCloud and moving current §3 immediately before accuracy in §6. The six-section version above is cleaner. Whichever is chosen, maintain a current→new equation/figure/comment crosswalk before changing labels. Do not delete derivation notebooks or alternate numerical artifacts just because their curves move out of the main paper.

## 4. Substantive JZ and JRZ comment inventory with proposed responses

These are all real pasted-document requests, retaining the two explicit endorsements as independent author support rather than duplicating work. No substantive pasted comments occur under Title, Abstract, §1 or §7; comments from Niko/Marika still belong there when location-mapped.

### Current §2

- **JZ: misleading statement that each observable receives shear/lensing and clustering contributions** (section2.tex:3). Replace by direct field definitions epsilon=gamma+I and theta=g+mu, then derive each observable's terms. Explain that magnification is a genuine lensing contribution to observed counts, whereas ordinary shear is not a component of the count field. This also addresses JRZ's disconnect below.
- **JZ: define f_K(chi'-chi) and state flat cosmology at Eq2.4** (section2.tex:9–35). Best combined resolution with JRZ: adopt spatially flat geometry from the outset, f_K=chi, explain source–lens separation chi'-chi. If curved definition retained in appendix, correct sign convention K=-Omega_K H0^2/c^2 when K>0 denotes closed geometry; current sign at19 conflicts with standard Omega_K convention.
- **JRZ: assume flat cosmology for brevity**. Accept main-text scope; preserve only brief extension statement. No need to expand curvature derivation in this paper.
- **JZ: define P_delta_delta before Table1** (section2.tex:21,55–57,77). Define delta as matter overdensity, P as its nonlinear 3D power spectrum, units Mpc^3 for k in Mpc^-1, and specify total-matter versus cold+baryon convention consistently with CCL and massive neutrinos.
- **JRZ: W_phi/W_kappa disconnected from theta/epsilon spectra; physical contributions only decoded in table** (section2.tex:23–57). Lead with observed fields and kernels, state gamma uses W_kappa and I/g use W_phi within adopted models; specify spin conversion in A_uv, then show the three observed sums and table. Avoid introducing the same glyph phi as observed clustering field.
- **JRZ: rename section Angular power spectrum formalism**. Accept exactly or sentence-case house equivalent.

### Current §3

- **JZ: link DESC forecasting library** (section3.tex:29). Cite actual versioned [LSSTDESC/forecasting](https://github.com/LSSTDESC/forecasting) source and the actual distribution generator/binny used; pin commit/tag when implementing, plus provenance of saved bins. The SRD itself is [arXiv:1809.01669](https://arxiv.org/abs/1809.01669), not the 2009 Science Book currently referenced as SRD. Root should verify current repo link and overlap implementation.
- **JRZ: survey configuration interrupts theory, especially covariance section** (section3.tex:1–64). Move after unified LimberCloud formalism; put n_eff, f_sky, windows and covariance next to the statistical validation that uses them.
- **JRZ: front-load maths; move CCL comparisons to evaluation Accuracy subsection**. Implement globally as outline above; preserve all essential validation while reducing repeated diagnostic plots in the main text.

### Current §4

- **JRZ: Figures3/4 shared axes, larger panels**. Shared x-range and shared axis labels, align main/residual panels, remove redundant per-panel axes; avoid forcing identical amplitude y-limits if it hides actual differences. Common residual limits wherever meaningful; label any local exception. Font size judged in final PDF print dimensions.
- **JZ: Figure2 logarithmic x-axis**. Apply chi>0 logarithmic x-axis or inset resolving chi<200 Mpc; mask chi=0 explicitly. If moved to appendix, still make readable and preserve the explanation of negligible kernel-weighted contribution only after a numerical weighted check.
- **JZ: W_phi not entirely survey-determined because H(z) enters** (section4.tex:22–28). Correct definition and tensor bookkeeping as §6.1 below; this is central to factorisation, not a cosmetic wording fix.
- **JZ: nonlinear/scale-dependent bias not safely dismissed as nonstandard for LSST** (currently section5.tex:36). Remove that language; cite [Nicola et al. 2024 / arXiv:2307.03226](https://arxiv.org/abs/2307.03226), already represented in bibliography by 2024JCAP...02..015N. Separate the simple validation model from expected analyses; avoid extrapolating measured runtime to bin-dependent nonlinear models.
- **JRZ: rename method section LimberCloud**. Accept by merging §§4–5.
- **JRZ: move flat-cosmology declaration to §2**. Resolve with JZ/JRZ flatness comments once; update all later wording.
- **JRZ: §4.1 overly complicated; fold Eq4.1 into tensorisation**. Accept compact representation in new §3.1; move power-only validation plot to appendix.
- **JRZ: remove distracting Numeric schemes**. Main science plots should compare LimberCloud to the precisely defined CCL reference, with grid refinement if possible. Preserve higher-order/different-coordinate interpolation diagnostics only in appendix if they diagnose an error mechanism. State exactly what differs between schemes. Do not claim all higher-order interpolation necessarily breaks analytic integrability: polynomial-in-chi interpolation can remain analytically integrable with the same rational/log families, with more terms.

### Current §5 (plus Figure8 now in §6.1)

- **JZ: Figure5 legend overlaps axis**; **JRZ explicitly seconds**. Dedicated figure-level legend above panels or separate legend row; joint figure export QA for all ERROR panels.
- **JZ: Figure8 apparent CCL timing/legend inconsistency** (section6.tex:11,28,32; experiments/benchmarks/Y1/benchmark.py:118–128). Plot code explicitly repeats total CCL time in each stage panel. Explain/reference it as CCL end-to-end, or remove repeated curve and instead plot a distinct baseline in its own panel. Text at32 incorrectly treats this repeated total as an independently measured cosmology-stage curve. Fresh timing table must eliminate that contradiction.
- **JZ: report Delta chi-squared/log likelihood**; **JRZ explicitly agrees**. After covariance validation and exact vector ordering, compute r=C_LimberCloud-C_CCL on identical bands/mask, solve L y=r for Cholesky Sigma=LL^T, report Delta chi-squared=y^T y for noiseless CCL reference. Report per-probe and joint values using full cross-probe covariance; diagonal-only normalized errors cannot establish this result. State model-error distance, not an observed noisy-data likelihood shift. Keep failures/missing raw artifacts as pending work rather than guessed values.
- **JRZ: §5 should be subsection of LimberCloud**. Accept; route equations into new method section, empirical paragraphs into Accuracy.
- **JRZ: oversized y labels, top legend, remove Numeric curves, Delta C/sigma**. Original provisional response proposed signed main residuals. **Superseded 19 September:** preserve absolute fractional and absolute covariance-scaled errors on log axes, with explicit zero handling and signed diagnostic data. Retain NUMERIC order comparisons for the stated interpolation-adequacy purpose, moving detailed curves to supplementary material when needed for legibility. Joint discrepancy remains separately required for statistical-impact claims.

### Current §6

- **JZ: contextualise end-to-end emulation with direct-data-vector emulators**. Add concise comparison of target, survey/redshift flexibility, physical assumptions, training region, and demonstrated validation. Verified relevant starting points: [Boruah et al. arXiv:2203.06124](https://arxiv.org/abs/2203.06124) (publication year may be2023) and [Nygaard et al. CONNECT arXiv:2205.15726](https://arxiv.org/abs/2205.15726). CONNECT emulates CLASS observables; do not describe it as an LSST3x2pt emulator. The reference described by first name as Chun-Hao et al.2023 is very likely **To et al.2023, LINNA: Likelihood Inference Neural Network Accelerator**, JCAP01(2023)016, [arXiv:2203.05583](https://arxiv.org/abs/2203.05583): authorship/title and method were verified from the primary arXiv record. Label the mapping from Joe's shorthand as an inference; cite with the surname To. LINNA automatically generates training sets and neural surrogates for posterior inference and validates multi-probe data-vector analyses. Existing related works should not all be described as monolithic or incapable of parameter caching/parallelism.
- **JRZ: accuracy subsection after front-loaded maths**. Same restructuring request as §3; keep a single master change with cross-references.
- **JRZ: shorten §6.4 and move to §7**. Accept; retain one short outlook paragraph with genuine unimplemented dependencies, not detailed claims that curved/beyond-Limber/CMB cases are already accommodated.

### Additional (whole paper)

- **JZ: release code and give link**. Add code/data-availability paragraph linking the verified repository and reference commit/tag, reproducible example/config files and environment; describe actual release status. Do not claim PyPI/archive/DOI or release policy that is absent. User authorized planning docs, not public publication.

## 5. Niko inventory with exact implications

1. **Speed headline ambiguous**: main.tex:38; section1.tex:9; section6.tex:7–38; section7.tex:5. Distinguish (a) cold total including setup/JIT/device transfer, (b) warm full cosmology+coefficient+projection, (c) coefficient+projection with cosmology cached, (d) fixed-basis distribution update projection only. Quoted3ms is projection, not full likelihood. Existing draft gives ~2x end-to-end and >1000x relative to full CCL for projection; these are different denominators. For fast/slow comparison, separately benchmark CCL with the cosmology cached and tracers reused where possible. Remove the intro's >10x total wording until reproduced. Timing scripts return twelve component contractions rather than assemble a full likelihood, so call the measured operation spectra/projection and state omissions.
2. **Qualify analytic marginalisation**: main.tex:38; section6.tex:58,70,91–111,115; section7.tex:7. Exact additive-density quadratic data vector; exact Gaussian integral only after dropping quadratic model term. Keep log determinant of cosmology-dependent effective covariance. Replace §6.4's exact analytic marginalisation claim. State validation pending; do not claim accuracy of neglected terms from small priors without a metric.
3. **Show Y10 validation**: section3.tex:1; unused Y10 PDFs in figures/manifest.toml. Add small table per probe with eligibility rules, finite-signal typical/max fractional residual, max absolute normalized residual, joint Delta chi-squared and sample counts after validating code/covariance. Define typical, e.g. median absolute residual over selected entries; do not average undefined fractional ratios across zeros. If raw evidence cannot reproduce the claim, change sentence to scope/future validation, not a numerical estimate.
4. **Mark standard pairs and cuts, Figures6/7**: section5.tex:67,84. Preserve Niko's distinction: SRD overlap limit25%, Y1 limit10%, with lens in front. Verify the actual mathematical overlap definition and source version before implementation; thresholds alone do not specify a selection algorithm. Build one pair+ell mask used for summary tables, plots and Delta chi-squared. Label excluded pairs/ell ranges, include raw all-pairs diagnostic appendix. TT autos-only is an analysis choice, not a universal3x2pt law. Do not make current plot residuals disappear by asserting all bad pairs are excluded without applying the rule.
5. **Bias factorisation assumptions earlier**: section2.tex:23,57; section5.tex:36; section1.tex:9. State shared bin-independent P_uv(k,z) or constant bin amplitudes. Bins with different scale-dependent biases require separate tensors or a separately validated expansion in shared operator spectra; the 21 integral types persist but cost and symmetry/reuse change. NLA is bin-independent only under this paper's common-A/common-law assumption.
6. **Style3x2pt/StageIV/flatness**: adopt one mathematical macro for3x2pt and one house style Stage-IV. Replace incorrect flat-sky wording section4.tex:58, section5.tex:7, section6.tex:117 with spatially flat; retain true angular/full-sky spin distinctions. No angular flat-sky assumption is needed merely to set f_K=chi.
7. **Figure3 labels**: enlarge final rendered axes/ticks while implementing JRZ shared-layout request, including source/lens sample identification and units.
8. **IA density Eq3.2**: explicitly rho_m,0=Omega_m,0 rho_crit,0 in comoving M_sun Mpc^-3, C1 carries h^-2 and D(0)=1. Fiducial generator uses `rho_x(a=1,species='matter',is_comoving=True)` and C1=5e-14/h², compatible with this convention. Do not insert physical rho(z) or extra(1+z)^3. Scripts currently store this law on the fiducial cosmology and reuse it during random cosmology timings; this is a separate runtime consistency issue. PyCCL use_A_ia=False means the provided input is already the raw normalized signed IA coefficient.
9. **Soften abstract emulation forecast**: delete additional three-to-four-orders claim. It is not benchmarked and is phrased as an additional factor on top of projection gain although projection remains part of emulated runtime. Recast as a future route to remove the cosmology/coefficient cost with surrogate accuracy and target compression still to demonstrate.

## 6. Additional scientifically important audit findings (do not attribute to coauthors)

### 6.1 Survey-only vectors and exactness

Current Eq4.2 defines phi_i=H_i/c p_i, so phi_i depends on cosmology. Current boxed Eq5.6 is still algebraically correct conditional on cosmology, but prose that coefficients contain all cosmology and phi is wholly survey-only is inaccurate. Minimal correction is define D_ii=H(z_i)/c, phi=D p and Bhat=D B D. Then C=p_a^T Bhat p_b with genuinely survey-only p(z) vectors on the fixed z grid. Equivalently absorb H_i/c directly into each kernel basis coefficient. This is a cheap diagonal rescaling, not a new integral method. Document the convention once and carry it consistently into marginalisation and emulator target.

Use approximate signs/explicit definitions for interpolation Eqs4.1,4.3,4.10 and ensuing continuum equality. Bilinearity of the **discretised, approximated forward model** is exact; original continuous Limber spectrum differs by finite grid, interpolation, finite support and endpoint error. Piecewise-linear interpolation applies to selected input functions, not to the final lensing kernel: s_kappa includes logs/quadratics, and r_kappa adds chi(1+z). Section5.tex:32 incorrectly says all kernel coefficients are at most linear.

### 6.2 Boundary equations and code

Eq4.9b uses n+2, hence is invalid on final interval n=N_Z-1; Eq4.9d assumes the endpoint basis support is wholly above chi, likewise fails when chi lies inside final interval. Provide special truncated rising-hat integral for i=N_Z,n=N_Z-1 or impose/validate zero endpoint density explicitly. The sentence at section4.tex:79 that only i=n integrates a partial interval is false: i=n+1 also has a partial interval, as chi²/log(chi) in Eq4.9b already show.

A read-only independent scalar midpoint quadrature on [0.7,1.3] with P endpoints[0.8,1.0] verified NN element1–3 to~6e-11 relative; this is a local analytic check, not a whole-code validation. **Observer diagnosis corrected 19 September:** the original claim assumed linear first-interval power, which would yield P1/(2 chi1) for the rising-hat term. The intended model instead has cubic power and correctly yields P1/(4 chi1); the power notebook and all21 derivation notebooks verify that policy. Do not apply the originally proposed factor change. Separately, NN coefficient assembly excludes final B[N_Z,N_Z] (`numba_backend/nn.py:66–68`; `jax_backend/nn.py:70–71`), requiring validation against the declared far-endpoint convention. A small independent quadrature suite should cover all21 structural cases, first/final intervals, signed amplitudes and narrow intervals, rather than only equality between ports. Current representative-element consistency tests do not establish every formula or assembled boundary is validated.

Mixed density×lensing summed tensor has a first subdiagonal: e.g. `numba_backend/ns.py:159–162` fills B[n+1,n]. Thus section6.tex:54 claiming its entire lower-triangular half is zero is too strong. State upper-Hessenberg support for the indicated orientation (transpose for reverse orientation), after endpoint validation. Number-density×number-density is tridiagonal. Symmetry of B for a fixed physical pair requires its two legs to be the same kernel/physics; reverse mixed pairs are transposes, not each symmetric.

### 6.3 Bias and magnification definitions

`section2.tex:23` calls the two-kernel/multiplicative-bias framework fully general. Deterministic b_g b_g P_mm, b_I b_I P_mm describes adopted linear/NLA template; nonlinear/stochastic galaxy bias or more general IA has independent operator cross-spectra, not necessarily a single multiplicative bias. General bilinearity may be preserved termwise, but universal representation/accuracy/cost has not been demonstrated.

Magnification coefficient b_mu(z) is not an arbitrary foreground-redshift multiplicative bias: the number-count slope belongs to source galaxy selection inside the lensing integral. For constant-per-bin q=5s-2 it factors out, as intended in validation. If response varies over source redshift it must weight the distribution inside that kernel; it should not simply multiply P(k,z) evaluated at foreground chi. State bin-constant assumption and distinct slope s vs response q.

Current Eq3.3 reports[0.659,...,0.994] as b_mu, exactly the stored values in scripts/generate_config/magnification_bias.py. Experiment scripts convert these to q=5s-2. CCL then passes q as `mag_bias`, although official CCL requires s. Verified primary API documentation: https://ccl.readthedocs.io/en/latest/api/pyccl.tracers.html . Root agent should handle resulting code plan and artifact invalidation scope.

### 6.4 Benchmark contracts invalidate simple headline rewriting

`experiments/spectra/JAX/GPU/Y1/triple.py:93–113` evaluates21 ell edges; CCL triple.py:84–89 evaluates20 geometric centers. No common seeded cosmology table is used. JAX's lensing amplitude is frozen at fiducial cosmology115–131 despite random cosmology153–162, and density conversion176–177 uses sampled E(z) times fiducial h. Bias/IA arrays are also frozen fiducial. JAX's ms term310–315 omits magnification response. These issues require fresh physically matched accuracy/timing runs, not only a clearer legend.

CCL script already saves `_COSMOLOGY` and `_CELL` timings, contrary to draft assertion no separation exists. However cosmology constructor timing may omit lazy CAMB/P(k) work triggered later: explicitly precompute identical distance/power requirements before stage timers. Optimize a reasonable CCL reference by caching tracer objects per sample/bin and using symmetry where applicable, then report both the historical reference and fair reference if useful. Do not imply computational libraries intrinsically cannot parallelize repeated calls.

### 6.5 Covariance and inference strength

Section3.tex:54 asserts deliberately sparse lens sample is essential/realistic and claims denser samples inevitably have much worse photo-z; without a measured sample definition this is not an acceptable reason to enlarge error bars. State exact selected sample provenance and test SRD-like density baseline separately. Both Y1/Y10 have18,000deg² in scripts/generate_config/survey.py; determine whether this intentional common-footprint test should be called Y1-like rather than unmodified SRD Y1. Covariance positivity alone does not establish correct ordering/physics; regenerate and validate before statements below survey errors.

Section5.tex:84 dismisses residuals using photo-z/bias dominance and a claimed standard cut at upper-bin redshift. This cannot replace joint residual testing. An ell cutoff at the upper edge does not guarantee k<kmax across a broad distribution: document exact adopted effective-redshift/window prescription and units. Report error sensitivity for specific cuts and avoid statements that any future nonlinear-bias model will make numerical error irrelevant. Both LimberCloud and CCL using Limber measure interpolation error relative to Limber, not Limber's physical adequacy at low ell.

### 6.6 Marginalisation notation and scope

Eq6.1 needs separate Phi_L and Phi_S for position–shape and Y10 rectangular10×5 combinations; common Phi is only appropriate for common-population legs. Write C_uv=Phi_u^T B_uv Phi_v, then stack all independent nuisance components into a vector x and give dimensions. Eq6.3 currently reuses matrix Xi as a vector without specifying flattening.

Exact quadratic dependence applies to additive redshift-density components (or linear modes) with fixed normalization and bias, not arbitrary shifts p(z-delta z), broadening, selection changes, positive-log-density coordinates, or nonlinear renormalization. Use a zero-integral mode basis to preserve normalization; a Gaussian prior on all raw bins is singular if normalization is enforced, or otherwise permits unphysical total density. Positive densities also constrain the domain, so unbounded Gaussian integration is an approximation if positivity constraints matter.

For model element A written M_A=Mbar_A+Gamma_A x+x^T Q_A x with symmetric Q_A and zero-mean Gaussian prior Pi, exact prior moments are mean shift tr(Q_A Pi) and extra covariance2 tr(Q_A Pi Q_B Pi). These moments are optional appendix diagnostics and **do not** make the marginalized likelihood Gaussian/exact. Integrating exp(quartic polynomial) is not reduced to a finite Isserlis operation. A perturbation expansion around the linear-Gaussian marginal uses the completed-square conditional Gaussian, not just moments of the unweighted prior; convergence/error control must be demonstrated. Avoid claiming guaranteed small corrections merely from small marginal variances.

### 6.7 Unsupported future reach

CMB source delta at last scattering is not directly represented by the current smooth-grid source basis truncated at z=3.5; it needs explicit source-plane/boundary treatment and extended support for CMB autos. Curved f_K(chi'-chi) is a two-distance factor, so linearly approximating f_K(chi)/chi alone is insufficient proof of the claimed integral family. Beyond-Limber projection requires unequal-time P(k,z,z')/transfer functions and additional k/radial integrals; bilinear survey dependence can remain but does not establish current computational cost or integral expressions. Remove claims these extensions need no formula change or only modest cost until derived.

Emulation estimates in section6.tex:46–54 are speculative:10^6 samples do not imply adequate coverage of10–50 dimensions, compressibility and target precision untested, storage/output bandwidth material. A million entries×million samples is10^12 numbers (~8TB float64) before replication; current21ell×351² tensors already have~2.59million elements per structural matrix before symmetry/zero removal. The0.5s coefficient cost×1e6 gives5.8days; multiplied25 gives145days, not~100 except coarse rounding. Parallel processes do not multiply GPU throughput without resource/occupancy evidence. Retain only conditional research tasks and a clear target comparison with direct data-vector emulation.

### 6.8 Bibliography / reproducibility

DESC SRD citation should be1809.01669; currently2009arXiv0912.0201L at main.tex:125 is the LSST Science Book. First Weinberg/Peebles entries at79–81 omit titles; SciPy citation at269 names Nature Medicine where its DOI is for Nature Methods. Resolve exact publication metadata as a focused final bibliography pass. section6.tex:117 refers to methods used in the Appendix, but main.tex includes no appendix. Add the promised derivations or remove dangling prose. A figure manifest currently lists producers, not runtime hashes/environment/numerical selection; extend provenance on regeneration.

## 7. Dependencies and acceptance gates for editing strategy

1. Freeze current manuscript/repo/attachments and map all comments without editing scientific assets.
2. Set data-model contracts: survey definitions, p(z) vs phi(chi), bias/slope/IA conventions, ell bands and bin ordering/masks.
3. Apply narrowly scoped correctness repairs and independent analytic/unit checks; confirm fiducial and several nonfiducial spectra component-by-component against CCL.
4. Repair/validate covariance ordering, physics and positive definiteness; calculate normalized and joint residuals with one shared mask.
5. Re-run only necessary Y1/Y10 science/benchmark products under controlled Perlmutter scripts; ensure JIT/device synchronization and matched settings.
6. Perform high-level reorganisation and now-evidence-supported numerical edits; revise abstract and conclusions last.
7. Re-export every affected figure at final sizes, compile from clean staged sources, inspect labels/legends/ratios, validate cross-references and bibliography, then produce a comment→change→evidence completion table. A proposed change stays pending until this evidence exists.

## 8. Memory use

A quick registry lookup found general LimberCloud audit/NERSC history, but all substantive claims above were rechecked from current source or primary API/paper links. Registry pointers used: MEMORY.md:620–648 (repository/manuscript audit context), MEMORY.md:827–870 (workflow context). Parent should cite only exact relevant ranges actually used in its final reply and incorporate its own memory use once; no scientific numerical result above is inherited from memory.
