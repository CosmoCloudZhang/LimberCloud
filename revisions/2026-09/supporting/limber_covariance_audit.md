# LimberCloud OneCovariance audit and minimal implementation plan

> **Status — 2026-09-19:** This is historical audit evidence, not the current execution plan. The [code plan](../CODE_REVISION_PLAN.md), [manuscript plan](../MANUSCRIPT_REVISION_PLAN.md), and [implementation prompts](../CURSOR_IMPLEMENTATION_PROMPTS.md) take precedence over its provisional recommendations. Code and scientific scripts are edited and validated on NERSC; manuscript sources, accepted publication figures, and the paper manifest are edited locally. The manuscript submodule may remain uninitialized and absent on NERSC; scientific jobs and ordinary checks must not require it. Source paths, environment contracts, and line numbers below describe the inspected historical snapshots. In particular, advice to preserve the old environment selectors does not override the later approved environment simplification, and earlier NPZ proposals do not override the final HDF5 spectra contract. The audit findings and original evidence below are retained for traceability.

Audit date: 2026-09-15. This is a read-only investigation and proposed implementation; no production code was changed and no Perlmutter jobs were submitted.

## 0. Live GitHub reconciliation (supersedes older environment/path details)

The parent task retrieved current GitHub main at `7d29b2f0a5e4f216bd4dd9b4cd36b9ffb6a4487c` into `/private/tmp/limber_review/github-main`. Checked the covariance code, renamed error notebooks, README and new environment loader against that checkout:

- Both covariance `matrix.py` files are unchanged, so COV-01 and the associated line numbers remain current.
- `notebooks/error_analysis/` is now `notebooks/error/`. All six notebooks have identical code-cell sources after the rename; COV-02 and COV-05–10 remain applicable. Use the new paths for implementation. Y10 TT raw JSON line numbers shifted: float32 load is line 131; triangular index line 139. Others retain float32 line 115 and index line 123.
- Covariance shell scripts now source `scripts/nersc/load_environment.sh` and `scripts/nersc/modules/cpu.sh`, activate `LIMBERCLOUD_CONDA_ENV`, call `limbercloud_require_onecovariance`, and derive `covariance.py` from `LIMBERCLOUD_ONECOVARIANCE_ROOT` (lines 18–20,33–34). They already use `set -eo pipefail`.
- README now correctly documents canonical `LIMBERCLOUD_CONDA_ENV` (default `CosmoConda`) and `LIMBERCLOUD_ONECOVARIANCE_ROOT`. The old `CosmoENV`, `ONECOVARIANCE_SCRIPT`, and `ONE_COVARIANCE_ROOT` names are compatibility aliases. Thus **COV-03 below is historical and already resolved upstream; do not propose another environment-variable migration.** Preserve the new shared loader/module architecture.
- The external `CONFIG.ini` prerequisite remains, now shell line 36; current main still has no configuration renderer or template. Preflight validates the upstream path, but additional INI/data identity and output checks remain necessary.
- Manuscript Gaussian-versus-total wording is unchanged.

All proposed code edits should target this current GitHub baseline after reconciling the user's local branch. Local artifacts and OneCovariance inspection are unchanged by this reconciliation, and the actual Perlmutter revision/config remain unverified.

## 1. Evidence and limits

Inspected both `experiments/covariance/{Y1,Y10}/matrix.py` and launchers; all six `notebooks/error_analysis/{Y1,Y10}/{EE,TE,TT}_Error_Analysis.ipynb`; configuration generators; `ProjectPaths`; tests; relevant manuscript covariance descriptions; local OneCovariance parser, output builder, multipole binning, bias/IA projection and local VAECloud artifacts.

- Current checkout: `/Users/s2227120/LimberCloud`.
- Local OneCovariance source: `/Users/s2227120/OneCovariance`, commit `91e139622eba568d7f742a2d12f6b66a106e6e68` (2025-10-06). Its three `output/Cell*.ascii` files were already modified; source files inspected here were unmodified. This is not evidence that Perlmutter uses this same revision.
- Local VAECloud is an artifact tree at `/Users/s2227120/VAECloud`; bounded discovery found covariance tables/configs/matrices, but no Python, notebook or shell source files. The previously corrected implementation mentioned by the user is not established by this local tree.
- `LIMBERCLOUD_RUNTIME_ROOT`, `CosmoENV` and `ONECOVARIANCE_SCRIPT` are unset in the current local environment. The current LimberCloud `CONFIG.ini`, tables and output covariance on Perlmutter remain uninspected. No claim below asserts that current remote matrices were measured.
- Official upstream documentation confirms the single-file input layout and full combination requirement: [API input format](https://onecovariance.readthedocs.io/en/latest/api.html#cov_input.FileInput._FileInput__read_in_Cell_files), [external-spectrum example](https://onecovariance.readthedocs.io/en/stable/guides/examples.html), and [output ordering](https://onecovariance.readthedocs.io/en/latest/guides/crash.html).

## 2. Confirmed defects

### COV-01 — Multipole-only sorting corrupts the upstream interpretation

Locations: both covariance `matrix.py` files, lines 159, 182, 205; allocations at 144–147, 168–171, 191–194. All three writers call `numpy.argsort(ell.flatten())` and apply that ordering to labels/values.

Input tensors have shape `(bin_i, bin_j, ell)`. The default `argsort` is unstable for repeated multipoles; it does not define the order of bin pairs tied at each ell. OneCovariance reads the first value column directly into `(N_ell, N_i, N_j)`, without reconstructing entries from the supplied bin labels: `/Users/s2227120/OneCovariance/onecov/cov_input.py:6053–6059,6077–6079`. Thus correctly labelled but scrambled rows still produce incorrectly assigned physical spectra.

Required single-file order:

```text
ell_0  1  1  C_11(ell_0)
ell_0  1  2  C_12(ell_0)
...
ell_0  N_i  N_j  C_NiNj(ell_0)
ell_1  1  1  C_11(ell_1)
...
```

The complete Cartesian pair set is required for these input files, including both triangles of EE and TT. Do not replace input tables with the unique pairs used in the output covariance. For TE, `i` is the clustering/lens bin and `j` the shear/source bin; the rectangular Y10 dimensions (10 by 5) are especially useful for testing this distinction.

Smallest correction, at each writer:

```python
order = numpy.lexsort((index2.ravel(), index1.ravel(), ell.ravel()))
```

The last lexsort key is primary. An equally valid explicit serializer writes ell-major arrays using `numpy.moveaxis(value, -1, 0).reshape(-1)`, with matching `repeat(ell)` and tiled row-major pair labels. Prefer one tested lightweight helper reused by both surveys, but do not redesign the numerical projection machinery to fix serialization.

Executed an exact-reader contract test by extracting the installed upstream method with Python AST and invoking it on a temporary 2-lens × 3-source × 4-ell sentinel table. Existing sort decoded 19/24 entries incorrectly; lexsort decoded all entries correctly. This verifies the mechanism independently of CCL or a covariance run.

### COV-02 — TT/EE covariance diagonals are assigned to the wrong bin pairs

Locations: all four TT/EE error notebooks, code cell 4 (zero-based), JSON lines 115–125. They use `INDEX = j * (j + 1) // 2 + i` inside `for i: for j >= i`.

That formula is for column-wise triangular order. Default upstream output uses row-wise upper-triangular order: `(0,0),(0,1),(0,2),...,(1,1),(1,2),...`. The output builder proves this at `onecov/cov_output.py:4779–4798`; `/Users/s2227120/VAECloud/CELL/Y1/COVARIANCE/MATRIX.ascii:1–11` also describes the block order. An extracted exact-method sentinel test reproduced this ordering.

For three bins, the notebook swaps the roles of `(0,2)` and `(1,1)`; larger cases misassign more uncertainties. Use enumeration of `[(i,j) for i in range(n) for j in range(i,n)]`, or validated `numpy.triu_indices(n)`. A closed-form correct index is `i*n - i*(i-1)//2 + (j-i)` for zero-based `i <= j`, but explicit pair metadata is clearer.

Default data-vector ordering is TT unique pairs, then TE full rectangular pairs, then EE unique pairs; spatial bins vary fastest within each pair (one sample dimension here). TE's `i*N_source+j` formula matches this default, but custom observable/pair selections can alter sizes and must be checked. A three-bin example is essential; a two-bin test cannot reveal the triangular error.

This is a plotting/consumer error independent of COV-01. Correcting labels changes which uncertainty appears in which panel, but cannot repair a physically indefinite covariance by itself.

### COV-03 — Historical environment mismatch, already resolved on current GitHub main

The older local checkout has `README.md:49,64` requiring `ONE_COVARIANCE_ROOT` while its shell scripts require `ONECOVARIANCE_SCRIPT`. Current main resolves this through canonical `LIMBERCLOUD_ONECOVARIANCE_ROOT` plus compatibility aliases and a shared loader. No new naming fix is needed; retain the upstream implementation described in section 0.

### COV-04 — Covariance configuration is an untracked external prerequisite

The only code references to `CONFIG.ini` are the two covariance launchers, current main line 36 (older local line 38). Neither `matrix.py` nor `scripts/generate_config` generates it, and no tracked INI template exists. The launcher can regenerate spectra and then run a stale configuration pointing at a different tree, survey, binning, cosmology or sample density.

Minimal remedy: capture and inspect the actual remote configuration; add a small versioned template or checked renderer with all paths supplied through `ProjectPaths`, plus a preflight that validates required files and hashes. Record the effective OneCovariance configuration and upstream commit in the output artifact. Do not copy VAECloud parameters blindly.

## 3. Additional scientific/data-contract issues requiring explicit checks

### COV-05 — Floating notebook bin centres do not match inspected upstream binning

Every error notebook constructs `geomspace(20,2000,21)` and geometric centres from floating edges (code cell 3; JSON lines 89–94). The inspected OneCovariance version integer-casts logarithmic boundaries and removes duplicates before finding geometric centres (`cov_ell_space.py:425–460`). For these settings the centre mismatch reaches 2.235%; the first notebook centre is 22.440369 versus upstream 22.360680.

Persist actual output edges/centres, and map reference spectra/residuals to those bands. Record whether the plotted mean is the band-averaged spectrum or the centre evaluation approximation. Do not merely relabel centre-evaluated residuals as band averages. Exact estimator/bandweight agreement is needed for a covariance-weighted residual statistic. Confirm the Perlmutter version, since bin construction can change.

### COV-06 — Source redshift arrays can be evaluated on a subsequently overwritten lens grid

In both `matrix.py`, source distributions are sampled on their own min/max grid (lines 43–50), then `z_grid` is overwritten with lens min/max (65–72). Later source tracers pair the earlier source distribution values with this overwritten lens grid (150–151,176). If source/lens supports differ, this is physically incorrect despite compatible array lengths. Whether current supplied data trigger this has not been verified.

The alignment/galaxy arrays are generated on a fixed 0…3.5 grid with 351 samples (`scripts/generate_config/{intrinsic_alignment,galaxy_bias}.py`), but JSON stores only values; covariance associates them with the current lens grid. Preserve distinct named grids and interpolate functions to the actual tracer grid, or use and assert the existing common 0…3.5 validation grid. Add explicit redshift-grid metadata to generated bias configs when implementing the artifact contract.

### COV-07 — Float32 and missing metadata weaken reproducibility

The writer stores ell and C_ell arrays as float32; all six notebooks load full matrices as float32. Keep indices integer and use float64 for input ell, spectra, covariance reading, validation and covariance-weighted statistics. Float32 alone is not proven to explain the present physical problems; it is an avoidable precision loss. If upstream ASCII output truncates precision, record its format and compare against in-memory/binary values before attributing a tiny eigenvalue to physics.

Store dimensions, explicit pair order, ell support and output bands, observable names, fiducial parameter/config hashes, input-data hashes, generator commit, OneCovariance revision and component labels. Reject stale or incompatible combinations rather than reshaping by assumed dimensions.

### COV-08 — Total versus Gaussian reference is inconsistent in prose

`manuscript/sections/section3.tex:56` calls the overlaid curves Gaussian; line 64 describes Gaussian + connected non-Gaussian + SSC. Notebooks load `MATRIX.ascii`, while local VAECloud INIs enable all three terms. The current remote LimberCloud INI is unknown. Decide the intended diagnostic explicitly and save/use a correspondingly named component. A Gaussian covariance is sufficient for a transparent first numerical-error reference if clearly labelled and generated from matched full observed-field spectra and noise. Do not silently remove intended NG/SSC contributions from an existing scientific claim.

### COV-09 — External full C_ell tables do not prove full NG/SSC model consistency

Current CCL input spectra include IA and magnification in observed shear/number-count tracers. Supplied full C_ell can supply the Gaussian observed-field products. The inspected upstream `onecov` source has no occurrences of `magnification` or `mag_bias`, while its connected trispectrum/SSC calculations use internal model/response/weight machinery. The exporter also writes `MAGNIFICATION.ascii` and `ALIGNMENT.ascii`, but a file being written is not evidence that the INI consumes it; the local VAE INI instead sets scalar IA parameters and has no magnification-file input.

Therefore an ordering repair and positive-definiteness check alone cannot demonstrate magnification-consistent NG/SSC. First validate Gaussian-only covariance against an independent formula. Then validate the exact components and assumptions of any requested NG/SSC run, or describe the scope as approximate and avoid using it to justify stronger inference claims. This may affect the paper's error-budget wording, not the Limber tensor algorithm.

### COV-10 — Selection and tiny-signal metrics

Keep complete field spectra at input so all required covariance contractions are available. Apply the standard science vector/pair/scale mask only to the output vector and both covariance axes. Record that mask in metadata. For near-zero cross spectra, fractional error and sigma/C can diverge; current `numpy.divide(..., out=ones, where=C!=0)` fills exact zeros with 1, which fabricates a residual. Use a validity mask/NaN plus absolute or covariance-normalized residuals, and report exclusions.

## 4. Local VAECloud precedent: measured state

The local files cannot serve as a corrected golden fixture:

| Artifact | Input rows differing from lexicographic order |
|---|---:|
| Y1 Cell_gg | 2410 / 2525 |
| Y1 Cell_gkappa | 2410 / 2525 |
| Y1 Cell_kappakappa | 2410 / 2525 |
| Y10 Cell_gg | 9992 / 10100 |
| Y10 Cell_gkappa | 4940 / 5050 |
| Y10 Cell_kappakappa | 2410 / 2525 |

Both `MATRIX.ascii` files are finite, exactly symmetric as read, and have positive diagonals. Nevertheless:

| Local matrix | Shape | Minimum eigenvalue of diagonally normalized correlation matrix | Cholesky |
|---|---:|---:|---|
| Y1 | 1100 × 1100 | -28.7663205854 | fails |
| Y10 | 2400 × 2400 | -401.0202129190 | fails |

These were calculated in float64 with `/opt/homebrew/anaconda3/bin/python`, NumPy and SciPy; normalized matrices were symmetrized only after measuring zero original asymmetry. The large negative values are not a borderline floating-point issue. They are evidence about these local files only. They do not refute the user's recollection of a separate successful corrected run. Locate the exact Perlmutter VAECloud correction/commit/config and its matching output manifest before adopting that result as a regression baseline.

A permutation of an already assembled covariance is `P C P^T`; it preserves eigenvalues and cannot turn an indefinite matrix positive definite. Misreading individual C_ell entries before covariance assembly can violate physical cross-spectrum consistency, and does require regenerating covariance after correcting the tables. Never propose diagonal jitter, eigenvalue clipping or nearest-SPD projection as the primary repair for these defects.

## 5. Minimal staged patch plan

### Stage A — Repair existing path, before any larger workflow change

1. Add a small `src/limbercloud/io/covariance.py` helper (proposed) for table serialization and explicit pair order; use it from both existing matrix writers. Alternatively a first tightly scoped commit can change the six argsort lines to lexsort plus tests.
2. Use float64 spectra/ell; validate monotonic ell, dimensions, finite values, contiguous 1-based labels, exactly one row for every `(ell,i,j)` tuple, and EE/TT symmetry within numerical tolerance. Retain all Cartesian pairs and signed cross-power values.
3. Fix the four TT/EE notebook mappings with the same explicit pair order. Keep TE orientation; stop loading matrices as float32. Validate full expected dimension before slicing.
4. Preserve the current shared environment loader and its upstream-path validation. Extend launch preflight for expected INI, resolved input/output paths and consistent tag. Add a covariance validation step after OneCovariance and stop on failure.
5. Rebuild covariance in a fresh versioned runtime output directory. Keep old matrices/figures as provenance until the new result is accepted.

### Stage B — Make the existing path reproducible

1. Capture a versioned INI template and render paths/survey parameters rather than relying on manually relocated absolute paths. Name Gaussian-only and full-component configurations distinctly.
2. Add `experiments/covariance/validate.py` (proposed) that produces a JSON validation report and compact NPZ with float64 covariance/diagonal, band metadata, pair maps, component identity and hashes.
3. Keep the 101-point covariance input ell support separate from the 20 output bands. Reuse independently generated reference CCL spectra only if ell support, physical tracer definitions, fiducial inputs and precision match exactly; 20 centre samples are not automatically a replacement for the 101 input samples.
4. Link covariance generation to the new batch-produced validation spectra artifact through a checked dependency. Plot notebooks load compact products. No CCL recomputation inside plotting.
5. Add explicit grid metadata and source/lens-grid checks. Add science-vector pair and scale masks without dropping the full input spectra needed for covariance products.

### Stage C — Acceptance gates on Perlmutter

1. Record current remote OneCovariance commit, effective INI, Python/NumPy/CCL/CAMB versions, node/thread settings, config/data hashes and actual VAE corrected reference identity.
2. Run pure serialization/ordering tests locally and under the Perlmutter environment; include non-square dimensions and multi-bin triangular mapping. The currently inspected `tests/test_experiment_contracts.py` only checks configuration aliases and experiment-file existence; it cannot catch these defects.
3. Generate all three tables once, parse them using the actual remote OneCovariance reader, and compare decoded entries to source arrays. Assert Gaussian signal blocks are symmetric; check the full observed-field spectral matrix (including appropriate noise) is positive semidefinite within a declared numerical tolerance.
4. Run Gaussian-only Y1 and Y10; compare a small independent Gaussian analytic covariance for a matched mode/band convention, survey area, per-bin density and single-component shape-noise definition. For modes before binning, the check is `Cov(C_ab,C_cd) = (S_ac*S_bd + S_ad*S_bc) / [(2ell+1) f_sky]`, with `S=C+N`; apply the actual estimator bandweights for binned comparison, not an unqualified delta-ell approximation. Test TT, TE, EE and cross-probe blocks.
5. Check finite square shape; relative asymmetry; positive diagonal; expected vector dimension (1100 for 5 lens/5 source/20 common bands; 2400 for 10 lens/5 source/20 common bands, full standard output); decoded row labels; labelled-list versus matrix consistency; component sum when components are enabled; correlation bounds; Cholesky; min/max correlation eigenvalues and conditioning; solve residual for a deterministic vector. Use float64. Large negative modes are an error, not a regularization request.
6. Validate the selected science-vector principal submatrix after applying the same mask to residuals and both covariance axes. A complete PD matrix has PD principal submatrices; duplicated symmetric observables can create artificial singularity and should not be added to the output vector.
7. Only after Gaussian passes, add NG and SSC separately with the physical model contract reviewed. Record their effects and convergence; individual contributions may require PSD-tolerance assessment rather than strict Cholesky, while a likelihood's final selected covariance must be invertible/PD for the adopted calculation.
8. Regenerate covariance-reference figures and associated quantitative residual summaries. State which component is plotted; review any conclusion that numerical residuals are negligible relative to survey uncertainty. Evaluate `delta^T C^{-1} delta` with a solve on the same selected/binned data vector if making a joint impact claim; diagonal error bars alone do not establish joint negligibility.

## 6. NERSC operational details

Preserve one CPU job per covariance survey and current-main shared Perlmutter environment/module conventions (`load_environment.sh`, `modules/cpu.sh`, `LIMBERCLOUD_CONDA_ENV`, `LIMBERCLOUD_ONECOVARIANCE_ROOT`). Do not restore the older hard-coded module setup or legacy canonical variable names. The existing launcher requests 256 CPUs and sets OMP threads to that count, whereas local VAECloud configs use `num_cores=128`; multiprocessing plus threaded BLAS/OpenMP can oversubscribe. Inspect actual parallelism and use one explicit worker/thread budget (workers times threads bounded by allocation); do not assume 256 is a safe value at every layer. Record MaxRSS and elapsed time with Slurm accounting during the pilot. Keep covariance jobs dependent on validated table generation and propagate nonzero exits. This audit did not run remote jobs or validate Perlmutter resource sufficiency.
