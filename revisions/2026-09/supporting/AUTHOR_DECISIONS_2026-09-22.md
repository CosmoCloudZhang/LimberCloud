# Author-supplied comments, 22 September 2026

The text below is preserved verbatim from the supplied attachment. Its last paragraph describes the state when it was written, not this documentation update. The author subsequently requested incorporating these changes into the completion plan. The plan adopts eta=0, NUMERIC-only radial orders, canonical21-to20 bandpowers, and inclusion of the final NN diagonal. The new plan adds evidence-backed NS/SN/SS terminal validation and pins the notebook natural-spline boundary condition. These are not new coauthor comments.

---

Interpolation orders belong only to NUMERIC. The plan already says that, and the 24 CCL, NUMBA, and JAX drivers were not given linear, quadratic, or cubic loops. The mistake is narrower: the benchmark reader and the shared filename helpers still accept an interpolation token for every family, so a NUMERIC order can be written into CCL, NUMBA, and JAX names. That wiring should be removed. Radial `linear` / `quadratic` / `cubic` stays a NUMERIC-only comparison of how `phi`, `a(chi)`, and `P` are reconstructed. It is a different operation from the angular cubic spline that turns sampled `C_ell` into bandpowers.

`eta_IA` will be `0.0` everywhere, as the LSST DESC SRD value. The generator currently still writes `0.5` when `--eta-ia` is omitted and records the choice as unresolved. That fork goes away.

## The NN `1/4` factor and the missing last diagonal

These are two different pieces of the same NN matrix, and only the second one is a code defect.

On the first radial interval the power is `P(chi) = P1 * (chi / chi1)^3`. The NN window is built from a falling hat and a rising hat. The three integrals of `P * hat_i * hat_j / chi^2` on that interval are exactly `1/12`, `1/12`, and `1/4`, in units of `P1 / chi1`. Both the Numba and JAX `element3` special cases return `1/4` when `chi1 = 0`. That matches the derivation. The older suggestion to replace `1/4` by `1/2` assumed linear power on that interval, which is a different model. I will not make that change.

The other issue is how those interval contributions are placed into the matrix. For an ordinary interval between node `n` and node `n+1`:

- `element1` is the left-node diagonal, `coefficients[n, n]`
- `element2` is the off-diagonal between `n` and `n+1`
- `element3` is the right-node diagonal, `coefficients[n+1, n+1]`

The loop visits every interval, including the last one. On that last interval it still adds `element1` and `element2`, then skips `element3` because of this guard:

```66:68:src/limbercloud/projection/numba_backend/nn.py
        if n + 1 < grid_size:
            element = element3(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1])
            coefficients[n + 1, n + 1, :] += element
```

JAX has the same condition and writes a zero instead. The closed form of `element3` is not wrong; the last call is never stored. An independent quadrature of that same last-interval integrand is nonzero whenever the power and the rising hat are nonzero there. The SS kernel does store its own far-endpoint diagonal, so NN and SS are not following the same endpoint rule.

If the lensing or density weight is exactly zero at the final radial node, this missing entry does not change the spectrum, because it is multiplied by that zero weight. The redshift grid runs to `z = 3.5`, and the SRD distributions are not defined to vanish there, so the omitted term is a real missing contribution. I would treat this as a code-assembly bug and include the final NN diagonal in both backends. I will not do that until you confirm it, because it changes the spectra.

## Ell nodes and bandpowers

Every backend will evaluate `C_ell` on the same 21 geomspace edges from 20 to 2000: CCL, NUMBA, JAX CPU, JAX GPU, and all three NUMERIC orders. The current CCL drivers still use the 20 geometric centres `sqrt(edge_i * edge_{i+1})`. Those centres will stop being the CCL evaluation grid.

From those 21 samples, one shared operator builds the bandpowers. It is the notebook recipe: a cubic spline of `ell * C_ell` against `log(ell)`, integrated across each bin and divided by the linear bin width. That equals

`(1 / (ell_{i+1} - ell_i)) * ∫ C(ell) d ell`

and it produces 20 bandpowers. Residual-bias comparisons use these 20 bandpowers. Covariance uses the same 20 bandpowers. The 21-edge `C_ell` arrays stay in the file so the spline can be reproduced; they are not a second comparison vector and not the covariance data vector. The separate 101-point covariance grid goes out of the shared contract.

This angular spline is applied after every method, including NUMERIC. It does not replace NUMERIC’s radial interpolant.

## Edits I would make after your check

Code, still uncommitted:

- `scripts/generate_config/intrinsic_alignment.py`: default `eta_pivot = 0.0`, and record that value as the SRD choice.
- `src/limbercloud/validation/contract.py` and the tests that still expect an unresolved `0.5` versus `0.0` decision.
- `experiments/benchmarks/Y1/benchmark.py` and `Y10/benchmark.py`: pass `--interpolation` only when reading NUMERIC. CCL, NUMBA, and JAX filenames stay `Time_<configuration>_<allocation><suffix>.txt`.
- `src/limbercloud/io/artifacts.py`: an interpolation token is valid only for NUMERIC basenames.
- `src/limbercloud/validation/estimator.py`: the shared estimator is 21 edges, then the cubic-spline bandpowers. Geometric centres and the 101-point grid remain available only as readers of old products.
- The 24 spectra drivers: CCL’s `ell` array becomes those 21 edges. No driver gains an interpolation loop.
- Numba and JAX `NN.coefficient`: add the last-interval `element3` onto `coefficients[-1, -1]`, with a test that the stored diagonal matches the independent quadrature. Only if you confirm that fix.

Plan text, so later prompts do not reopen these choices:

- `CODE_REVISION_PLAN.md` C01: `eta_IA = 0.0` is the adopted SRD fiducial. The generator value `0.5` is historical and is not used.
- C03 and C08: one evaluation grid of 21 edges for every family; bandpowers from the cubic spline above are the residual and covariance vectors. Delete the 101-point covariance-input grid and the instruction to compare CCL at 20 centres.
- C04: keep the `1/12`, `1/12`, `1/4` observer result. State that the NN final-diagonal omission is an assembly bug, pending your confirmation to include the term.
- `CURSOR_IMPLEMENTATION_PROMPTS.md` Prompt 1 and Prompt 2: interpolation orders apply only to `experiments/spectra/NUMERIC/`. The cosmology table is shared with CCL, NUMBA, and JAX, but those families are not rerun at three orders.
- `MANUSCRIPT_REVISION_PLAN.md`: the `eta_IA` discrepancy is closed at `0.0`.

I have not edited any of this yet.
