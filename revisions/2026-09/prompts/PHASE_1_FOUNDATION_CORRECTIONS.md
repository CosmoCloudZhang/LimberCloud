# Phase 1 — Close the foundation and endpoint corrections

**Current Phase 1 implementation and closeout plan, revised 22 September 2026.** Start from the existing repairs and verify them; do not replay completed work or restore superseded contracts. This prompt incorporates the author's post–Phase 1 decisions: faithful Jupyter editions of all 21 Mathematica derivations, four additional boundary cases, independent mathematical review before removing the temporary terminal modules, small provenance/component checks, and separate `Fiducial`/`Cosmology` timings. The [working-tree review](../reports/2026-09-22_PHASE1_CONTRACT_REVIEW.md) records defects, not acceptance of the work below.

Use this complete prompt for the next bounded Phase 1 implementation pass. Phase 1 owns these foundation corrections; [Phase 2](PHASE_2_NUMERIC_AND_EXECUTION.md) reuses accepted results for evaluator/NUMERIC/storage work. Independent Phase 2 scaffolding may proceed, but scientific acceptance depends on this closeout. Editing this plan does not execute its notebook conversion, kernel changes or tests. Invoking the implementation prompt authorizes its bounded work, not the full campaign.

Retain the simplified launch contract: no host-allocation filename token or `--number`; performance plots contain CCL, Numba and JAX only and label the CCL curve `CCL`. The deleted `tests/test_projection_endpoints.py`, `revisions/2026-09/endpoint_diagnostic.py`, and full-assembly `*_oracle` helpers stay absent. Compact tests belong in the existing science suite after the derivation files are generated.

---

Implement revised Phase 1 in the actual NERSC LimberCloud checkout. Read `revisions/2026-09/CODE_REVISION_PLAN.md`, `COMPLETION_CHECKLIST.md`, `reports/2026-09-22_IMPLEMENTATION_REVIEW.md`, the three detailed `2026-09-22_*AUDIT.md` reports, `MANUSCRIPT_REVISION_PLAN.md`, the existing C00/C01 reports and `supporting/limber_numeric_endpoint_review.md`. These paths are relative to the revision directory unless fully qualified. The 22 September author decisions supersede unresolved choices and old prompts. Inspect the implementation, not only the reports. Preserve unrelated changes and the existing environment.

Deliver the concrete repairs below with focused tests and a complete report. Routine fixes/tests in scope are authorized by this prompt; do not ask again whether eta should be zero or the NN final diagonal included. Stop at the Phase 1 handoff, with a reviewable diff. Do not automatically commit, push, merge, reset, execute later phases, run production, or edit/initialize the paper.

## 1. Establish the baseline and repair the repository boundary

Record `git status`, branch, HEAD, current plan hash, interpreter/.venv target, runtime roots, modules and `git ls-tree HEAD manuscript`. Compare the actual checkout with reviewed commit `13a3c2d`, retaining newer deliberate work. The local audit found that commit deleted the manuscript gitlink, although `.gitmodules` still names it. The current review records the correct pin in both HEAD and index; verify and preserve that repair. Historical reports do not override the actual tree.

Follow the master plan's phase-review packet contract: preserve reconstructable entry/exit source snapshots and manifests, including staged/unstaged/untracked changes, then return a phase-only diff and aggregate HEAD diff. Do not assume a new commit separates stages or capture secrets/runtime data/paper contents. Keep source identity separate from reports written afterward.

If absent or unintentionally changed, restore the parent index/tree's mode-160000 entry for `manuscript` at `90d12f4f3e574a67c25944d27d7ded553e09402b` from the verified pre-deletion parent revision, unless a later intentional local-owner paper update is evidenced. Restore only the gitlink, not the paper contents. An index-only `git update-index --add --cacheinfo 160000,<verified-pin>,manuscript` is a possible implementation after checking the existing index; inspect `git ls-files --stage manuscript` and the staged diff afterward. An index repair does not change `git ls-tree HEAD manuscript` until a separately authorized commit; report both and never stage the absent working directory as another deletion. Do not initialize the remote paper or run `git -C manuscript`. Do not stage local nested paper files as ordinary files. Verify `.gitmodules` and gitlink together, code-only builds and absence of unintended paper changes. Keep this recovery isolated in the report/diff.

## 2. Finish Phase-0 execution and portability repairs

Retain the dedicated environment, portable CPU/NERSC recipes, successful loader migration, removed --path and allocated MPI/HDF5/device evidence. Do not rebuild environments just to repeat a report. Verify the following retained requirements and fix only remaining gaps:

- All 24 scientific shell launchers and Run_All chains must forward supported arguments into Python with quoting intact. Verify `--sample-count`, `--sample-table`, `--tag`, `--label` and `--folder` via isolated stub Python/srun/sbatch fixtures, including paths with spaces. `generate_samples.py` alone retains `--run-id` for its input-table directory. Check generator/benchmark wrappers where documented options similarly stop at the shell boundary.
- Require `--sample-table` before evaluation/output, even for default `--sample-count 0`. Every timing launch evaluates sample 0, then samples 1..N. There is no `--fiducial-only`, `--include-fiducial`, `--run-id`, `--run-config`, `--mode` or `--resume` on spectra timing drivers. Timings remain in the existing family/survey directory (JAX adds device), using section 7's population names; reruns replace only populations evaluated. Do not create timing run-ID directories or empty sampled products for N=0. Older `*_128*` and superseded unqualified timing products remain untouched and are not selected by the new reader. Core counts stay in Slurm headers and thread variables, not Python flags or filenames.
- Fix source-file discovery in `scripts/nersc/modules/{cpu,gpu}.sh` and shared helpers for Bash 3.2. Capture `BASH_SOURCE[0]` in the appropriate caller scope before substitutions where necessary; do not rely on a substitution preserving the source stack. Replace remaining Bash 4-only case conversion such as `${LABEL,,}` in supported portable launch paths. Verify both macOS /bin/bash and NERSC Bash. Normalize path fixtures consistently for /var versus /private/var rather than assuming textual equality of equivalent paths.
- Make launcher tests self-contained with synthetic checkout markers, `.venv` stubs and runtime config. They must not require this user's real .venv. Keep missing/broken real environment errors clear in actual entry points. Preserve nonexecuting dotenv parsing, exported-value precedence, copied-Slurm root discovery and nested-paper-root behavior.
- Correct the real kernel startup path: the selected `.venv` interpreter must receive required Conda activation hooks and NERSC modules on supported hosts, while portable macOS behavior stays valid. A probe that imports only base limbercloud is not a scientific/HDF5 kernel test. Test actual kernel interpreter/check-out identity and a tiny HDF5 read/scientific operation under a supported allocation when the site-linked build requires it. Do not import MPI-initializing libraries on login hosts to manufacture capability evidence.
- Verify external OneCovariance's executable and dependency environment. Current launcher behavior may use `.venv/bin/python` against the external checkout; a separate source directory is not a separately selected interpreter. Install/validate required compatible dependencies in the selected research environment or document a deliberate explicit OneCovariance execution environment; do not accidentally run it in an unverified interpreter.
- Protect `install_mpi_h5py.sh` against base, CosmoConda, unrelated prefixes and accidental generic preinstalled MPI wheels. Installation is an explicit setup operation, never ordinary launch. If no rebuild is needed, verify code guards with fixtures and retain the validated installed environment. Do not source-build replacements merely for this audit. Validate version/linkage and communication only in the supported environment/allocation when changed.
- Separate truly lightweight checks from MPI-linked h5py/science tests in docs and commands. Full make check is not automatically login-safe if it imports site-linked HDF5. Preserve logs on durable runtime storage and record exact package/module/build identities.

Run the relevant existing tests after fixing these causes, not by weakening assertions to accept the defect. Fix bounded docstring/import/documentation issues as touched, preserving initialization order and the agreed docstring layout. Do not turn this phase into an unrelated reformat.

## 3. Close eta_IA and current configuration provenance

Retain eta_IA=0.0 in `scripts/generate_config/intrinsic_alignment.py`, its CLI/defaults, generated JSON, `validation/contract.py` and current consumers. Verify that no omitted-argument path reinstates the withdrawn unresolved/0.5 behavior. Do not delete historical report facts; do not leave current products accepted with unresolved/missing metadata.

Preserve A_IA=0.5, z_pivot=0.5, signed IA, C1=5e-14/h**2, comoving present-day matter density and D(0)=1. Do not globally replace every 0.5. Generate new IA arrays with redshift axis, effective law, fiducial input hash, eta 0 and fixed-tabulated nuisance policy. Validate them against an independent law calculation. Retain old eta 0.5 products under their identities; current readers reject them for accepted comparisons. Updating metadata alone is insufficient to regenerate the array.

IA/galaxy-bias functions stay fixed fiducial tables across cosmology samples. Recheck active h/Omega_m/background factors and the current s/q/MS fixes across all 24 runners so later adapters inherit one correct contract. CCL receives slope s, magnification density weights q=5s−2. Test s=0.4 and nonuniform slopes, both MS/MI and source/lens grids. Current campaign eta choice is resolved; generic alternate-eta tests, if useful, are explicitly diagnostic.

Retain the shared effective-cosmology/solver specification introduced by the original Phase 1 repairs. The earlier audit found differing Omega_g/CAMB kmax settings; treat that as historical motivation, not a claim that the current centralization is absent. Verify actual generator/evaluator callers and accepted radiation/temperature/neutrino, transfer/nonlinear and version settings. Compare D(z), H(z), chi(z) and rho_m(0) at canonical sample 0 where evidence is still missing. Preserve existing IA/galaxy-bias axes and generating hashes, and add the small binding check below. Regenerate only affected stale tables using their declared model; a metadata rename is not regeneration.

### 3.1 Bind provenance with one small shared check

Reuse the current effective-cosmology/hash functions. Before evaluation/output, validate the table schema and selected solver fingerprint, then compare IA/galaxy-bias generating fiducial/model/solver identities against canonical sample 0. Retain law, eta, redshift-axis and finite-array checks. Fixed nuisance tables are not compared with each sampled cosmology. Report the mismatched field and generator needed for repair; nonempty hashes alone are not compatibility. Add compact mismatch fixtures without creating a new provenance framework or regenerating files by metadata relabelling.

## 4. Restrict radial interpolation to NUMERIC everywhere

Introduce/reuse one validated method identity containing family, JAX device where applicable, and NUMERIC order where applicable. CCL and NUMBA are CPU methods; JAX requires CPU/GPU; NUMERIC requires linear/quadratic/cubic and currently CPU quadrature. Reject unsupported family/device/order combinations. Check `ProjectPaths.spectrum_results`, `ArtifactIdentity`, timing/spectrum basename helpers and both benchmark readers. Preserve the existing method restrictions while changing timing names; verify paths, filenames and benchmark resolution stay consistent.

Only NUMERIC experiment parsing exposes `--interpolation`. The performance readers under `experiments/benchmarks/{Y1,Y10}/benchmark.py` load CCL/Numba/JAX only, accept neither `--interpolation` nor `--number`, and save `benchmark_{label}.pdf`. NUMERIC products are read by numerical-comparison notebooks and summaries. Preserve spectrum/manifest basenames without allocation tokens; update timing names to section 7. Test required NUMERIC order, missing/invalid order and forbidden order on every other family. Do not add order loops to CCL/NUMBA/JAX. Actual NUMERIC integration/execution remains Phase 2.

## 5. Implement the exact angular contract

Retain and verify the implemented 21-node/20-band contract in `validation/estimator.py`. Every method evaluates float64 geomspace 20..2000 with 21 nodes. Form bandpowers by a **natural** CubicSpline of ell*C against logell, integrate each log interval, divide by the corresponding linear deltaell. The source notebook specifies `bc_type='natural'`; SciPy's default not-a-knot is not equivalent.

Provide a current contract object/validator with exact raw nodes, 20 edges-intervals, display centres, axis order, natural boundary condition, transform/normalization, dtype and versioned fingerprint. Reject invalid coordinates/shapes/nonfinite values; signed and zero spectra are legitimate. Raw 21 and band 20 datasets remain distinct. Old centre and 101-grid helpers can remain explicitly legacy-only; they must not be the current execution-specification default. Centre coordinates do not select extra CCL evaluations.

Verify all six CCL runner arrays retain the same 21 nodes as analytical methods; this does not establish complete shared-artifact publication. Phase 2 applies the operator in the actual common evaluator to all methods. Fingerprints must be validated against actual arrays, not arbitrary claimed hashes. Include boundary condition and operator version so the old helper cannot pass as compatible.

Tests: reproduce notebook natural-spline output on a curved signed fixture, distinguish it from not-a-knot, verify transform/integral dimensions/linearity, test batches and wrong axes, and independently integrate the same piecewise polynomial. A useful exact fixture has ell*C constant or linear in logell. A constant C sampled at nodes is not necessarily preserved exactly after splining ell*C versus logell; test its interpolation error, not a different normalized operator. Retain rejection coverage for an identity that claims 21 nodes but supplies different coordinates.

Covariance will use these 20 bandpowers/windows in Phase 3. Do not simply change a 101 constant to 21 and call covariance scientifically matched; its physical weighting requires the next adapter stage.

## 6. Derivation editions, complete boundary coefficients and independent review

Complete this section in order: inspect original sources and build the case map; convert the existing derivations; derive/export the additional cases; implement stable coefficients and correct explanatory kernels; add compact validation; obtain independent mathematical review of the completed derivations and implementation; remove the temporary terminal modules only after the removal gate passes. Preserve all valid Phase 1 science repairs. No new test/diagnostic filename is required.

### 6.1 Explain the existing piecewise rule and locate its missing case

The inside/outside distinction is already the correct mathematical rule. On interval n, the ordinary rising-source case assumes `n + 1 == k < N` and includes the next, falling half of an interior hat. For `k == N`, no falling half beyond chi_N exists. The whole-support endpoint expression is valid below chi_{N-1}; when the evaluation point enters the final interval, source integration must instead begin at that point. No extension of the physical domain or signal beyond chi_N is being introduced.

Trace this distinction through the coefficient notebooks and `notebooks/kernel/{Y1,Y10}/Lensing_Kernel_Kappa.ipynb`. Their `r_n_k` inside-rising branch excludes k=N, while the separate k=N branch uses a whole-support expression throughout 0<chi<chi_N. Correct that branch explicitly for chi_{N-1}<chi<chi_N, retain the whole-support branch below the last interval, and preserve the chi=0 and chi=chi_N limits. This educational-kernel correction is separate from faithful conversion of the original derivation notebooks. Include a nonzero terminal basis fixture; a tiny/zero actual distribution tail can hide the issue in a survey plot.

The source inspection locates the missing branch precisely: in zero-based cell 4 of both kernel notebooks, `elif n + 1 == k < grid_size` excludes the final source node, so `n=N-1,k=N` reaches the later `elif k == grid_size` whole-support formula. The general moving-limit integral is already correct; the specialized branch coverage is incomplete. The new B cases complete that rule in the integrated coefficient catalog rather than introducing another physical lensing effect. Keep this explanation beside the case map so an author/reviewer can follow the correspondence from kernel to coefficient.

For final interval [chi1,chi2], define L(u)=(chi2-u)/(chi2-chi1), R(u)=(u-chi1)/(chi2-chi1), and, for chi inside it,

```text
H(chi) = integral_chi^chi2 R(u) * (u-chi)/u du
F(chi) = integral_chi^chi2 L(u) * (u-chi)/u du.
```

Below chi1, H instead integrates over the full [chi1,chi2] support. At chi1 the partial H must match the whole-support value (and its first derivative for chi1>0); at chi2 both partial source integrals vanish. The old full-support expression cannot represent H inside the last interval; removing the interior guard would also reference a nonexistent chi_{N+1}.

### 6.2 Convert all 21 existing Mathematica derivations faithfully

Read every `.nb` in `notebooks/derivation/NN` (B01–B03), `NS` (B01–B08), and `SS` (B01–B10), including text, displayed equations, executable inputs, assumptions and cached outputs. Preserve the originals. Add a same-basename `Coefficient_Bxx.ipynb` beside each `.nb`; retain the existing `_Validation.ipynb` and equation `.txt` files.

The Jupyter edition must preserve scientific content, cell order, case definitions, variable definitions, intermediate algebra, assumptions, final I/J expressions and saved results. It is a format conversion, not a shortened rederivation. Render mathematics as readable Markdown/LaTeX. Preserve original Wolfram input and cached output text with source-cell mapping (inline or a clearly linked appendix); Wolfram code belongs in labelled display cells, not executable Python cells. Preserve meaningful graphics if present. Original frontend layout metadata remains available in the unchanged `.nb`.

Keep the faithful transcription intact, including original symbol spellings. Put verified corrections and any corrected executable equivalent in explicitly marked adjacent notes/cells; this satisfies both exact-content preservation and correction of the notation. Do not silently substitute corrected algebra into a cell presented as original Mathematica output. The source inventory must count the actual content cells rather than the repeated notebook-outline cache. The 21 current notebooks contain 146 displayed formula cells, 42 executable Input cells and 42 saved Output cells; preserve those along with all surrounding explanatory content and grouping. The source review found no graphics or interactive objects. Recover displayed equations from their stored TeX source and check their rendered appearance against the original box structure.

Where Python/SymPy recomputation is provided, label it as an executable equivalent and keep it distinguishable from the transcribed original. Cached Mathematica output is historical output, never evidence of a fresh Python or Wolfram execution. A Python kernel and saved rendered equations must suffice for reading; no Mathematica installation is required. Symbolic tools are notebook/development dependencies, not spectra-runtime dependencies. Do not install/rebuild the production environment merely to convert notebook formats.

Use a small conversion inventory recording source path/hash and source-to-target content-cell coverage. Check that no derivation or output cell was silently omitted and visually inspect all converted notebooks. Pure symbolic editions must open without runtime data or CCL; make only the targeted notebook-validator adjustment needed to distinguish these from data-backed validation notebooks, whose runtime-path checks remain.

Preserve the author's headings and mathematical sequence. Correct verified notation errors transparently: show or retain the original spelling, label the correction, and distinguish a symbol rename from an altered equation. For each issue, trace `.nb` display/input/assumptions/output -> `.txt` -> validation function -> Numba/JAX coefficient. Specifically recheck the NS B07 p/y and NS B04 redshift y/z inconsistencies rather than assuming all are cosmetic. Record display-only errors, executable assumption mismatches and actual numerical changes separately. Do not rewrite an existing exported formula or tensor merely to fix typography; any algebraic correction requires its own derivation and numerical evidence.

The planning-stage source check identified the following notation cases (it is not a full 21-notebook symbolic rerun):

| Original | Confirmed discrepancy | Established impact and conversion action |
|---|---|---|
| NS B04 (`.nb:344`) | Displayed normalized redshift is named y; the executable expression/export use z | Display inconsistency; annotate y -> z without changing the coefficient algebra |
| NS B07 (`.nb:589,606`) | Normalized power is defined as p; I7 input/cached output use y, while assumptions name p; txt and both backends use defined p | Naming plus executable-assumption reproducibility issue; retain the original, annotate y -> p and rerun the corrected equivalent. No tensor-result defect was demonstrated from this rename |
| NS B08 (`.nb:281`) | Notebook consistently uses y for normalized power and x for the integration variable; txt and both backends use p | Consistent symbol renaming across representations, not an internal algebra defect; preserve original notation and state the mapping |
| SS B09 (`.nb:800`) | I9 integrand/output use p but the assumption list refers to y | Executable-assumption mismatch; correct the equivalent's assumption and establish equivalence separately. It does not by itself demonstrate an error in the exported tensor formula |
| NN B02 (`.nb:113,312-313`) | Mixed NN/phi-phi label; observer-transpose interval subscript 1 where the direct integral at line 161 uses 0 | Display-label/index corrections; keep them distinct from changes to coefficient algebra |
| NN B03 (`.nb:257`) | Extra closing TeX brace in the chi1 denominator | Rendering correction; preserve the original text in the source mapping |
| SS B06 (`.nb:672`) | Dimensional observer expression labels J7, while the following integral/input/export use J6 | Correct the cross-reference to J6 in an explicit note; do not substitute the J7 coefficient |

Static expression comparison found that exported I/J expressions for NS B04/B07/B08 and SS B06/B09 match both backends after normalizing NumPy/JAX names and shape wrappers. This is source-equivalence evidence, not an executed numerical or assembled-tensor regression. Re-evaluate the corrected NS B07 and SS B09 expressions with explicit symbol initialization/assumptions in a clean symbolic session; saved outputs do not establish independence from prior Wolfram symbol assignments. Keep boundary-domain corrections in a separate mathematical ledger rather than describing them as notation cleanup.

Audit all 21 sources during conversion; do not infer that these are the only possible discrepancies or that source inspection certifies numerical equivalence.

### 6.3 Add four boundary derivations in the same layout

For nodes 0..N and the final interval n=N-1, use this proposed catalog, confirmed by the independent case-map review before integration:

| New case | Final-interval placement | Existing expression's relationship |
|---|---|---|
| NS B09 | i=N-1, j=N | Terminal counterpart of B07 |
| NS B10 | i=N, j=N | Completes terminal case associated with B08 |
| SS B11 | i=N-1, j=N, plus transpose | Terminal counterpart of B04 |
| SS B12 | i=N, j=N | Terminal counterpart of B10 |

A separate planning-stage reviewer confirmed placement completeness under the finite-hat convention: on the final interval NS contains L*F (existing B01), R*F (existing B02), L*H (new B09) and R*H (new B10); SS contains F^2 (existing B01), F*H/H*F (new B11) and H^2 (new B12). Only source nodes N-1,N survive there. This establishes the structural case count, not symbolic-expression correctness, floating-point stability or compiled acceptance; the completed-work review in section 6.5 is still required.

Add a second mathematical correspondence check using the existing interior-source cases: NS B09/B10 must equal the b->0+ limits of NS B05/B06, and SS B11/B12 the b->0+ limits of SS B02/B05, where b=(chi[n+2]-chi[n+1])/chi[n+1]. The limit removes the descending half beyond the source node, leaving the terminal rising hat. Derive the limit symbolically; do not directly substitute b=0 into implementations containing 1/b. This check connects the new boundary expressions to the author's existing inside-support treatment, while the original moving-limit integrals remain the independent numerical reference.

For each add `Coefficient_Bxx.ipynb`, `Coefficient_Bxx.txt`, and `Coefficient_Bxx_Validation.ipynb`. The derivation layout matches the converted originals. SN uses the transposed NS result. Extend NS B02's valid terminal density-row coverage without inventing a duplicate coefficient. Preserve NN's final diagonal and the cubic-observer factors 1/12, 1/12, 1/4. Review the remaining structural cases for exhaustive coverage, including the SS falling/falling placement; the four-case list is not a substitute for that review.

Use the existing normalized notation consistently:

```text
t = chi/chi2
a = 1 - chi1/chi2
p = 1 - power1/power2
z = 1 - (1+redshift1)/(1+redshift2)
Q(t) = 1 - p*(1-t)/a
Z(t) = 1 - z*(1-t)/a
F_a(t) = (1-t^2)/(2*a) + t*log(t)/a
H_a(t) = (1-t)^2/(2*a) - (1-a)*(1-t)/a - (1-a)*t*log(t)/a
```

Thus F=chi2*F_a, H=chi2*H_a, and t is in [1-a,1]. State 0<a<1 for ordinary intervals and the real-parameter/domain assumptions explicitly; keep any more restrictive original assumptions visible in the faithful editions. Derive the moving-limit source functions first, then the four dimensionless ordinary integrals:

```text
NS I9  = integral H_a(t)*(1-t)/(a*t) * Q(t)*Z(t) dt
NS I10 = integral H_a(t)*(t-1+a)/(a*t) * Q(t)*Z(t) dt
SS I11 = integral F_a(t)*H_a(t) * Q(t)*Z(t)^2 dt
SS I12 = integral H_a(t)^2 * Q(t)*Z(t)^2 dt
```

Restore `chi2*power2*(1+redshift2)` for NS and `chi2^3*power2*(1+redshift2)^2` for SS. Explain the physical lensing amplitude and where it is folded into effective power so it is applied exactly once. Show intermediate expansion/integration, assumptions, readable final I/J expressions, and `.txt` exports in the author's existing `I9 = ...`, `J9 = ...` style. Do not run a hardcoded old formatter that overwrites a different coefficient file.

Derive observer J expressions separately with a=1, Q(t)=t^3 and Z(t)=1-z*(1-t). Taking a=1,p=1 in ordinary I retains linear power and is incorrect. Derive stable small-a evaluation forms/series and the continuous t*log(t) limit; choosing `log1p` alone does not prove cancellation is solved. Keep normalized p for derivation correspondence but provide zero-safe endpoint-power evaluation. New terminal expressions replace the corresponding final-interval uses, never double-count them; earlier interval contributions remain.

### 6.4 Small component and zero-power handling

Use one shared activity helper before affected coefficient construction/contraction. Exact identically zero IA removes SI, IS, II, MI and GI. Exact identically zero q=5s-2 removes MS, MI, MM, MG and GM. Both off leaves EE=SS, TE=GS, TT=GG. eta_IA=0 is not IA off; s=0 is not magnification off (q=0 means s=0.4). Preserve the adopted nonzero manuscript amplitudes. No new driver flags or small-amplitude threshold is needed.

Supply correctly shaped zeros for disabled components where the existing strict assembler requires named terms; preserve missing-component error checks. Do not skip a tensor still used by an active component. Bin-specific q=0 suppresses only affected legs/pairs: MS/MI lens rows, MG first lens leg, GM second lens leg, MM either leg. Apply equivalent physical choices in CCL.

Skipping whole components does not handle isolated right-endpoint zeros in active effective power. All ordinary-interval coefficients are linear in endpoint power, so a bounded special branch can use `C(P1,0)=P1*(C(1,1)-C(0,1))`; both endpoint powers zero give an exact zero. Validate this branch independently for cancellation before adopting it. Preserve nonzero signed/crossing powers, keep the observer cubic branch separate, and ensure denominators are safe before eager NumPy/JAX selection. Never add epsilon, clip signs, or replace provider failures/nonfinite inputs with zero. Do not claim gradient support merely from value-level tests.

### 6.5 Independent review, implementation and terminal-module removal

Use a separate agent, not the implementing agent, to review the completed derivations and case map before deleting either `projection/numba_backend/terminal.py` or `projection/jax_backend/terminal.py`. This review must read the original/converted notebooks, new equations/exports, actual dispatch/assembly and validation evidence. It must check every NN/NS/SN/SS placement on first/interior/final/one-interval grids, dimensional factors, moving limits, observer power, stable limits, zero-power handling, SS symmetry and NS/SN transpose. Record disagreements and resolutions. A planning-stage review of proposed integrands is not acceptance of future closed forms or compiled tensors.

Implement the accepted closed-form cases and stable limits in the existing `ns.py`, `sn.py` and `ss.py` coefficient layout in both backends. Evaluate terminal-only work only on the final interval. The current fixed 48-point production quadrature is temporary; outer quadrature does not fix cancellation in its source functions. Independent numerical integration remains a validation reference.

After the derivation files exist, extend the existing science tests (not a restored dedicated endpoint suite) with compact independent checks. Validation notebooks retain the author's Numerical integral / Coefficient / Case 1 chi1>0 / Case 2 chi1=0 organization, uppercase fixture variables and integral/coefficient/discrepancy display. Use the actual last interval for Case 1 and a one-interval observer grid for Case 2, then narrow/irregular, signed/zero-power and nonzero terminal-density fixtures. Add absolute error when the reference is zero. Compare production float64 results; extra high-precision calculations are diagnostics, not a substitute. Real-cosmology illustrations must use the current model/power conventions, not copy stale constructor/observer-query behavior.

Reconstruct original hats and moving-limit source integrals independently of the coefficient under test. Use compact full contractions to verify placement and transposition. Preserve the NN fixture chi=[0,1,2], P=[0,1,1], phi_a=phi_b=[0,0,1], factor=1 giving 3/2-2*log(2), and an unchanged zero-terminal-density fixture. Execute both compiled backends with justified absolute/relative tolerances and refinement/precision evidence; two ports agreeing is not an independent reference.

Remove both terminal.py modules only when (1) their needed functionality is absorbed into the validated coefficient cases, including observer/narrow/zero limits; (2) separate-agent review finds the case map self-consistent; (3) focused compiled comparisons pass; and (4) all imports/callers/constants are updated and no production fixed-quadrature dependency remains. Recheck imports and focused tests after removal. If any condition fails, retain the modules and report the specific unfinished item; deletion is an outcome of completion, not evidence of correctness.

The removal audit includes both backend `__init__.py` exports, NS/SN/SS call sites and terminal-rule identity fixtures such as `tests/test_run_identity.py`. Retain the small source-integral and hat references in `validation/reference.py` for independent validation; check their own cancellation and observer limits before treating them as high-accuracy references. Their retention does not require production terminal.py modules.

### 6.6 Preserve the established numerical notebook reference

All six `notebooks/spectra/{Y1,Y10}/{EE,TE,TT}_Spectrum_Validation.ipynb` notebooks contain the same numerical helper definitions in zero-based cell 4. They use `interp1d` for phi, scale factor a=1/(1+z), and component-effective power, with `slinear`, quadratic and cubic choices. The lensing source integral runs directly from the evaluation point to chi_max using `fixed_quad(..., n=100)`; the outer spectrum integration also uses n=100. This numerical path already handles partial terminal support and does not call terminal.py or its cancellation-prone source expressions.

Keep three distinct questions separate: convergence of the notebook quadrature, floating-point cancellation in the temporary analytic-source implementation, and differences between NUMERIC's interpolants and the analytical reconstruction. A larger Gauss order addresses the first, not the other two. The known terminal failures do not establish that the notebook numerical results are inaccurate. Conversely, cached comparison plots without an order-refinement study do not establish a universal error bound; do not report a new numerical failure or pass without execution evidence.

Phase 1 records this source contract; Phase 2 extracts it faithfully and performs a bounded refinement check, starting from n=100 and varying inner and outer orders separately (for example 100/200/400). Retain n=100 if it meets the declared tolerance for the accepted workload. Increase order or split at interpolation knots only when measured error requires it, with the changed numerical policy recorded. Preserve moving source limits and observer-integrability checks regardless of the chosen order. This is a small convergence check, not a new quadrature framework or global grid-optimization study.

## 7. Separate fiducial and sampled timings with minimal reader changes

Use this grammar, preserving existing uppercase stage tokens:

```text
Time_<configuration>[_<NUMERIC order>][_<stage>]_<Fiducial|Cosmology>.txt
Time_Triple_Fiducial.txt
Time_Triple_Cosmology.txt
Time_Triple_COSMOLOGY_Fiducial.txt
Time_Triple_COEFFICIENT_Cosmology.txt
Time_Triple_LINEAR_Cosmology.txt
```

Title-case `Cosmology` denotes samples 1..N; uppercase `COSMOLOGY` denotes the existing computation stage. Spectrum and manifest names remain `Spectra_Triple_EE.h5`, `Manifest_Triple.json`, etc.; no allocation token returns. Update shared path helpers, all current writers, readers, tests and command documentation together; Phase 2 applies the same contract to new NUMERIC wrappers.

Every launch evaluates sample 0 and saves its own total/stage durations. A fiducial record identifies sample ID 0 and duration_seconds; sampled files contain two columns `sample_count cumulative_seconds`. Reset accumulators after sample 0. Document any compilation/initialization included in the first evaluation; sampled totals exclude fiducial/warm-up and output I/O. No extra fiducial-only CLI is introduced.

Accept any nonnegative sample count. Checkpoints are 100,200,... up to N, with N appended if absent; 0<N<100 produces one checkpoint N, while N=0 produces no sampled file. Thus the manuscript's explicit 1000 request and existing two-sample pilots both work. Do not add a multiple-of-100 restriction or a separate pilot interface.

N=0 replaces only Fiducial products and leaves explicitly identified prior Cosmology products intact. N>0 replaces both populations' products in place. Add a small comment header with format/population, sample-table content hash, requested count and timing convention; reuse existing input/producer identity where available, without adding a latest-launch manager. Preserve old unqualified and `_128` files but never silently fall back to them. Multi-file atomic publication/recovery remains Phase 2 work.

The figure reads only Cosmology products and uses the saved count column, normalizes a one-row file, checks matching counts/table identities across methods/stages, and gives a clear missing-sampled-data message for fiducial-only inputs. The current dimension error comes from plotting ten invented x-values against one fiducial duration, not from the spectra dimensions. Use `CCL` as the legend label everywhere, as requested. A caption/documentation sentence explains that stage panels repeat the CCL end-to-end reference; do not relabel it `CCL (total)` or describe it as a measured CCL stage. Keep CCL/Numba/JAX only and `benchmark_{label}.pdf`.

Add small writer/reader checks for N=0, a tiny pilot, one checkpoint, a nonmultiple and N=1000, including preservation of sampled files on a fiducial rerun. This is a file-contract change, not a general benchmarking framework.

## 8. Freeze contracts for Phase 2 and return evidence

Define the complete run identity Phase 2 will enforce: physics/config/input hashes, eta/nuisance/endpoint policy, canonical sample table, radial/ell/angular operator, family/device/order, requested workload/pairs, quadrature and timing boundary, source/patch version. Current artifacts missing these cannot silently resume as new products. Make current constructors validate rather than store inconsistent fields. Test method restrictions, malformed IDs/coordinates and changed configuration rejection; Phase 2 finishes sample transactions/consolidation.

Use separate shared-science, producer/workload and execution-record identities as specified in C09. The deterministic compute-source manifest covers relevant staged/unstaged/untracked code, while later docs/reports, job IDs, output paths/checksums and covariance/selection references cannot feed back into the spectrum fingerprint. Preserve HEAD/index history as provenance. Different methods retain different producer identities while comparing compatible shared science.

Run relevant selected-environment tests, lint, shell/notebook checks and code-only package checks; classify unavailable/allocated checks honestly. Local portability tests may require a handoff to the local owner for execution, without assigning simultaneous code edits. Inspect the final diff and update current docs/contracts; preserve historical reports. Append a clearly dated closeout section to `revisions/2026-09/reports/PHASE_1_FOUNDATION_CORRECTIONS_REPORT.md`, preserving its historical execution record. Include baseline/diff/paper pin, repair matrix, the 21-notebook conversion inventory and notation-impact register, four new derivation/export links, separate-agent review and resolutions, terminal removal/import checks, timing/provenance contracts, exact tests/tolerances/runtime, input invalidations, artifacts, and precise Phase 2 entry conditions. List unresolved external evidence separately from code completion. Do not claim a real-kernel test, endpoint spectrum comparison, restored parent commit or production run that has not happened.
