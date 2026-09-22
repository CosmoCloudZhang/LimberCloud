> Historical review of c0cf5fa. The repository has since advanced to13a3c2d; use the22 September implementation audit and current CODE_REVISION_PLAN.md for execution. Its derivation/scalar evidence remains applicable, but its statement about the then-advertised remote revision is historical.

# NUMERIC interface and terminal-node review

Date: 22 September 2026. Status: review and recommendations, not implemented changes or accepted production results. Inspected local code: `c0cf5fae8d9d48b51710f942f2cc7ad24779e4fc`. Read-only GitHub checks found the same commit at origin/main and only main among advertised branches. The Cursor implementation quoted in the conversation was therefore not available as a newer revision in this repository. No production source, environment, manuscript, or existing plan was modified by this review.

## NUMERIC is a separate method family

The 24 inspected CCL/NUMBA/JAX experiment entry points do not expose a radial interpolation-order argument. The six spectra notebooks instead contain identical direct numerical integration helpers in zero-based cell 4 (counting all notebook cells). The existing plan already calls for extracting these into a NUMERIC family.

Recommended structure:

```text
experiments/spectra/NUMERIC/
  Y1/{single,double,triple}.py
  Y10/{single,double,triple}.py
  # corresponding launchers and Run_All.sh

src/limbercloud/projection/numeric_backend/
  __init__.py
  quadrature.py
```

These are proposed paths. Six thin wrappers should call one shared numerical implementation. Preserve Single=EE, Double=TE+TT, Triple=EE+TE+TT. Do not copy quadrature code across survey, configuration, or interpolation order. A single module could serve initially if a package adds no value.

Only NUMERIC should accept `--interpolation linear|quadratic|cubic`; map `linear` to the notebook's SciPy `slinear` setting. Other methods should reject the option, and shared dispatch should reject incompatible combinations. Their metadata should record the NUMERIC order as not applicable, with their own fixed numerical conventions recorded separately. JAX CPU/GPU remain device choices for the same analytic implementation.

The notebook order controls three radial interpolants: density phi(chi), scale factor a(chi), and component effective power (power times the component amplitude before interpolation). The numerical helpers integrate density/lensing windows directly using SciPy quadrature, rather than assembling the analytic coefficient tensors. The existing nested and outer quadrature use 100-point fixed rules; extract a labelled reproduction mode first and establish convergence on bounded cases before treating the result as a reference.

Keep common cosmology samples, physical inputs, component assembly, multipole definitions, output schema, and execution infrastructure shared. A common spectra result contract does not require every implementation to produce coefficient tensors. Timing stages should describe the work actually performed.

NUMERIC linear is not automatically an independent integration of the exact NUMBA/JAX integrand: the former interpolates a, whereas the latter linearizes 1+z, and the analytical first power interval has its own cubic prescription. Preserve a separate matched-integrand quadrature oracle for formula and assembly checks. The cubic spline used later for angular band averaging is another operation and must not be controlled by this radial interpolation flag.

Evidence: `notebooks/spectra/Y1/EE_Spectrum_Validation.ipynb`, cell 4 and component calls in cell 7; `revisions/2026-09/CODE_REVISION_PLAN.md`, C08/C09; `src/limbercloud/io/project_paths.py:74` currently only recognizes CCL, NUMBA, and JAX and will need the planned NUMERIC extension.

## Retain the observer coefficient 1/4

The Mathematica files are textual Wolfram Notebook expressions. Their TeXAssistant `input` fields, integration cells, and saved outputs can be read without a Mathematica runtime. This review read the three NN notebooks, corresponding exported coefficient text, the validation integrands, and the power reconstruction notebook. It did not rerun Mathematica.

`notebooks/derivation/NN/Coefficient_B03.nb` defines J3 as the integral of t^3 from zero to one and stores the result 1/4. B01/B02 give 1/12 each. The exported B03 text also has J3=1/4. The power notebook (cell 4) and B03 validation notebook (cell 7) explicitly use cubic power on the observer interval.

With x=chi/chi_1, removing the factor P_1/chi_1 gives:

```text
falling/falling: integral x (1-x)^2 dx = 1/12
falling/rising:  integral x^2 (1-x) dx = 1/12
rising/rising:   integral x^3 dx       = 1/4
```

There is no basis for changing 1/4 to 1/2 while retaining this power model. This local cubic power prescription is distinct from NUMERIC's global cubic spline option.

## Include the final NN diagonal under the stated basis contract

The manuscript explicitly uses nodes 0,...,N, includes both hats on every interval, and sets density to zero only outside the grid (`manuscript/sections/section4.tex:26-41`). Section 5 sums both node indices through N. For final interval [a,b]=[chi_(N-1),chi_N], the right endpoint basis h_N=(chi-a)/(b-a) is supported throughout that interval.

Its diagonal contribution is

```text
B_NN(ell) = integral_a^b P_ell(chi) / chi^2 * [(chi-a)/(b-a)]^2 dchi.
```

This is exactly the existing ordinary-interval element3 integrand. No new NN closed form is required. Numba omits its storage at `src/limbercloud/projection/numba_backend/nn.py:66-68`; JAX replaces it by zero at `src/limbercloud/projection/jax_backend/nn.py:70-71`, while both retain terminal off-diagonal terms.

The Mathematica B03 labels already say i=j=n+1<N, and B02 labels also exclude the terminal right node. Consequently this is an uncovered terminal basis case/inconsistent endpoint specification across derivation, manuscript, and assembly, rather than an established isolated Python transcription mistake. Extend the documented NN index range to the final node and test it when correcting the assembly.

Excluding the single integration point chi_N has no effect on a continuous integral. Removing the endpoint basis coefficient affects the whole last interval. Similarly, a lensing efficiency kernel vanishing at chi_N does not imply the underlying source-density value phi_N vanishes or that its basis contributes nothing at smaller chi.

For an NN component with overall factor A, adding the missing diagonal changes its contracted spectrum by A*phi_a,N*phi_b,N*B_NN. This vanishes when either participating endpoint weight is exactly zero. It is not a universal zero: runners include z=3.5, interpolate and normalize all nodes, and never enforce that condition (`experiments/spectra/NUMBA/Y1/triple.py:40-64,167-168`). The runtime distribution arrays were unavailable, so this review establishes neither their actual endpoint values nor the numerical size of the effect in the survey products.

Setting density to zero at the final node would change its interpolant across the final interval and may alter normalization. If a tapered distribution is scientifically intended, declare it in the common input preparation and use it consistently across methods; it is unnecessary as a restriction on the general coefficient operator.

## Independent scalar verification

The local system Python has no NumPy/SciPy/Numba/JAX environment. A temporary, standard-library-only check extracted the actual Numba element function bodies via AST, removed decorators, and evaluated their scalar formulas using math.log and a scalar substitute for full_like. This checks formula values; it does not execute either compiled backend. A small scalar assembly mirrored the current source indexing, and an independent composite-Simpson integration evaluated the piecewise integrands.

Checks completed:

- Observer factors verified by exact rational integration and source formulas: 1/12, 1/12, 1/4.
- All three NN elements on three ordinary intervals checked against direct quadrature; largest absolute difference was below 2e-15 for these cases.
- Synthetic grid chi=[0,1,2], power nodes=[0,1,1], cubic first interval and constant power on the last interval:

| Nodal weights on both legs | Assembly with current guard | Assembly including final diagonal | Direct quadrature |
|---|---:|---:|---:|
| [0,0,1] | 0 | 0.11370563888010943 | 0.11370563888010939 |
| [0,1,1] | 0.6362943611198904 | 0.7499999999999998 | 0.75 |
| [0,1,0] | 0.47741127776021886 | 0.47741127776021886 | 0.47741127776021886 |

For the endpoint-only case the exact value is 3/2-2*log(2). These are dimensionless synthetic checks, not scientific acceptance runs or estimates of the observed survey error.

For nonnegative effective power the complete NN matrix is a Gram matrix and should be positive semidefinite. A zero terminal diagonal with a nonzero terminal off-diagonal violates that property; the synthetic current terminal 2x2 principal minor has determinant -0.00631095854446908. This is an additional structural check, not grounds for clipping eigenvalues.

Temporary reproducibility files: `/private/tmp/limbercloud-endpoint-review-20260922/check_nn.py` and `results.json` in the same directory. These temporary files may not persist across system cleanup.

## Related terminal cases must be checked together

NN is not the only place with terminal restrictions. NS right-density contributions have n+1<N guards (`numba_backend/ns.py:160-162,186-188`), leaving the density-index N row zero; SN has the corresponding missing column. The Mathematica NS labels also impose interior-only right-density cases. SS stores its terminal diagonal (`numba_backend/ss.py:324-326`), which does not by itself prove the lensing terminal formula is valid on every interval.

The complete-interval terminal lensing formula in manuscript section 4, equation 4.9d, applies when the evaluation point is below the last interval. Within [a,b] the lower source-integration limit is the evaluation point x, requiring the truncated rising-hat integral:

```text
H_N(x) = integral_x^b [(u-a)/(b-a)] * [(u-x)/u] du
       = [(b-x)^2/2 - a*(b-x) + a*x*log(b/x)] / (b-a).
```

Thus final-interval NS/SN/SS contributions need matched-integrand checks rather than indiscriminately removing every guard. Some guards protect accesses to an unavailable next node, chi_(n+2), and require distinct terminal formulas.

Recommended remote acceptance checks: basis-vector inputs concentrated at the last and penultimate nodes; zero and nonzero endpoint cases; first/final intervals independently; NN symmetry and nonnegative-power positivity; NS/SN transpose consistency and correct terminal support; quadrature of full NN/NS/SN/SS contractions; then small real Y1/Y10 comparisons measuring the change. Retain the cubic observer convention. Update derivation annotations and manuscript endpoint cases alongside the implementation. No full campaign should be used to substitute for these bounded checks.
