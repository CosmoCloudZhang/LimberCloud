# Followup: overlap selection and marginalisation algebra

> **Status — 2026-09-19:** This is supporting evidence, not the current execution plan. The [code plan](../CODE_REVISION_PLAN.md), [manuscript plan](../MANUSCRIPT_REVISION_PLAN.md), and [implementation prompts](../CURSOR_IMPLEMENTATION_PROMPTS.md) define the accepted scope and sequence. Code and scientific scripts are edited and validated on NERSC; manuscript sources, accepted publication figures, and the paper manifest are edited locally. The manuscript submodule may remain uninitialized and absent on NERSC; scientific jobs and ordinary checks must not require it. The pinned overlap definition and algebra below remain evidence for the final plans; they do not constitute completed numerical validation or instructions to start an additional inference study. The original body is preserved.

## 1. Verified overlap sources

### Primary binny implementation: concrete match to Niko

Inspected public `binny` current HEAD **eb913fe99675019d8f410ef40d1b5aed4b560291** in `/private/tmp/limber_review/binny`.

- [Pinned selection examples](https://github.com/binny-org/binny/blob/eb913fe99675019d8f410ef40d1b5aed4b560291/docs/examples/tomography/selections.rst#L241), lines241–321: LSST Y1 uses maximum overlap0.10; Y10 uses0.25. Both require **source-bin peak redshift strictly greater than lens-bin peak redshift**. This is peak ordering, not mean ordering, median ordering, interval-edge ordering, or the probability that a random source lies behind a random lens. The overlap comparison is **≤**, not strict<.
- [Pinned metric](https://github.com/binny-org/binny/blob/eb913fe99675019d8f410ef40d1b5aed4b560291/src/binny/correlations/metrics.py#L136), lines136–195: `metric_min_overlap_fraction` computes

  O(a,b) = integral min[n_L^a(z),n_S^b(z)] dz / [(integral n_L^a dz)(integral n_S^b dz)].

  The docs call it fractional overlap, but this expression is scale-sensitive for unnormalised curves. **For individually unit-normalised p(z), as in LimberCloud, it is simply integral min[p_L,p_S] dz.** Use normalized p(z), not number-density amplitudes, or raw phi(chi) evaluated on a z grid. `metric_overlap_coefficient` instead divides by min(norm_L,norm_S); the two agree for unit-normalized distributions.
- [Comparator implementation](https://github.com/binny-org/binny/blob/eb913fe99675019d8f410ef40d1b5aed4b560291/src/binny/correlations/filters.py#L73), lines73–120: `score_relation` with pos_a0,pos_b1,relation'gt' compares source score > lens score. The source example is not accidentally reversed.
- [LSST preset](https://github.com/binny-org/binny/blob/eb913fe99675019d8f410ef40d1b5aed4b560291/src/binny/surveys/configs/lsst_survey_specs.yaml): separately normalizes every bin. Thresholds reside in selection examples, not in preset data. The current preset's areas14000/18000deg² are not identical to original SRD v1.0.2 effective footprints.

Recommended plan wording: **Apply and record the Niko/binny lens-source selection: source peak above lens peak and the normalized minimum-overlap integral ≤0.10(Y1) or≤0.25(Y10). Compute and publish the resulting pair list for the exact saved distributions and grid used in this paper.** Cite the pinned binny source. It is now sufficiently specified to propose this concrete algorithm without asking the user what 'overlap' means. Still label the mask as this analysis's adopted Niko/binny choice, rather than asserting it is the only standard3x2pt selection or that it exactly reproduces an uninspected forecasting implementation.

### Official SRD and alternative primary analysis

[DESC SRD v1.0.2](https://arxiv.org/pdf/1809.01669), AppendixD2.1, printedpp53–54(PDFpages58–59), gives five source bins, five/ten lens bins, TT autos, and scale cuts. I did **not** locate the25%/10% overlap fractions or a formula there after searching the PDF text and reading this section. Thus SRD alone does not substantiate the precise numerical overlap rule. AppendixC1 gives effective footprints12300/14300deg², so current LimberCloud18000deg² should be presented as a declared variation. These points are distinct from the selected overlap rule.

`https://github.com/LSSTDESC/forecasting` and its claimed `analysis_choices` path were unavailable through web and unauthenticated git inspection (`could not read Username`); cannot distinguish private/missing/moved from that evidence. Do not mark its current source or link verified.

A useful caution against universality: [SwiftC_l paper](https://arxiv.org/abs/2505.22718), published JCAP01(2026)043, §4.3.1 (published PDFp12; accessible primary text at https://inspirehep.net/files/36b55c03a72ee7d8d9c19fdf6727d637) explicitly uses **Y10** distributions with maximum10% overlap, showing that analyses can choose different thresholds. It also says its broad ell range is for numerical demonstration. This does not contradict Niko's requested25%Y10 implementation; it means name the adopted rule.

## 2. Independent algebra verification of p(z) transformation

Let G=N_Z+1. For common z-grid points z_i, define

D(theta) = diag[H(z_i;theta)/c], phi_X^a = D p_X^a,

where X is the **population** L(lens) or S(source), and p_X^a is the unit-normalized redshift-space density sampled on the fixed z grid. The current chi-space tensor B_chi satisfies

C_alpha = (phi_X^a)^T B_chi,alpha phi_Y^b.

Therefore

K_alpha(theta) = D(theta)^T B_chi,alpha(theta) D(theta),
C_alpha = (p_X^a)^T K_alpha p_Y^b.

D is diagonal, so D^T=D. For different leg grids use D_X^T B D_Y. This transformation is exact algebra for the same discretized model and adds no quadrature approximation. It does not assert exactness relative to the continuum Limber integral. It moves the cosmology-dependent Jacobian out of the survey vectors and into the reusable basis. No extra grid spacing belongs in D when p_i are density values; the hat-function integrations already carry interval lengths. If instead variables are bin masses, a separately defined weight conversion is needed.

Do not use a single same-population Phi on both sides for GGL: with P_L∈R^(G×N_L), P_S∈R^(G×N_S), C_LS=P_L^T K_LS P_S. Y10 GGL is10×5, not square. Each physical contribution may have its own K; constants such as bin-dependent magnification responses can be folded into K_alpha or response vectors consistently. Add components before selecting/stacking the observed theory vector.

## 3. Explicit Gamma with an unambiguous flattened nuisance vector

Define column-major stacking within each population:

x = [vec_F(delta P_L); vec_F(delta P_S)],

so a bin's G grid values are consecutive. Write a data-vector element alpha=(X,Y,a,b,ell), with X,Y∈{L,S}, as

M_alpha = (p_X^a)^T K_alpha p_Y^b.

For a nuisance coordinate (T,c,k), T∈{L,S}, c its bin, k its grid point, the exact first derivative at the fiducial distributions is

Gamma_{alpha,(T,c,k)} = 1[T=X] 1[c=a] (K_alpha pbar_Y^b)_k
                          + 1[T=Y] 1[c=b] (K_alpha^T pbar_X^a)_k.

This works for unequal sample sizes, cross-bin correlations, and nonsymmetric mixed tensors. In an auto-spectrum where X=Y and a=b, **both terms contribute**; if K is symmetric this is2K pbar. Do not silently lose the factor2. For a sum over physical components, sum their Gamma rows with the same constants/spin factors as the data model. Apply any linear bandpower window and data mask to the data, K/Gamma and covariance consistently.

The exact quadratic remainder for this element is

q_alpha(x) = (delta p_X^a)^T K_alpha(delta p_Y^b).

To express it as x^T Q_alpha x with Q_alpha symmetric, insert K/2 and K^T/2 into the two off-diagonal blocks for distinct nuisance groups; for the same group use(K+K^T)/2. This avoids counting the cross term twice.

### Normalization, priors and marginalisation scope

- Use a linear mode map x=E eta satisfying zero-integral perturbations for each population/bin, w_z^T E_bin=0 for the chosen redshift integration convention. Then Gamma_eta=Gamma_x E and Q_eta=E^T Q_x E. Fixed E preserves exact quadratic dependence.
- A prior Pi_p on redshift-density components transforms to Pi_phi=T(theta) Pi_p T(theta)^T, where T is block-diagonal with one D per bin. Holding a chi-space prior fixed while changing cosmology would represent a different prior. Working consistently in p-space avoids this issue.
- With Gaussian eta prior Pi and Gaussian data covariance Sigma independent of eta, omitting q gives Sigma_eff(theta)=Sigma(theta)+Gamma_eta(theta) Pi Gamma_eta(theta)^T. Include logdet Sigma_eff plus the quadratic residual term. This is exact for the linearized model; the full quadratic model produces a non-Gaussian mixture and generally a quartic likelihood exponent.
- Normalization constraint in p(z) is exact under its declared quadrature weights; the interpolated chi-space density's integral can differ at finite grid resolution. Do not enforce an extra nonlinear per-proposal renormalization without changing the claimed quadratic model. Positivity, nonlinear shifts/broadening and moving grids also need separate treatment.
