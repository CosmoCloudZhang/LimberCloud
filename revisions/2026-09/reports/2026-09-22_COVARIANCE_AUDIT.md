# Covariance, statistics, and release audit — 22 September 2026

Read-only review of LimberCloud `13a3c2deb3e72bd22cfad5f4eb1a6191899a47d6`, the attached author comments, current plans/notebooks/manuscript, and pinned upstream OneCovariance source. No production science runs were performed. The Phase 3 execution prompt is authored separately at `revisions/2026-09/prompts/PHASE_3_COVARIANCE_AND_PILOTS.md`.

## Findings still present at HEAD

Both `experiments/covariance/{Y1,Y10}/matrix.py` retain the old physical calculation. Phase 0 improved launch/environment plumbing; Phase 1 did not repair covariance. Important locations in Y1:

- Lines 41–76 generate source and lens samples on separate supports, then overwrite `z_grid`; later source tracers use the lens axis with source values.
- Lines 80–116 load historical alignment/galaxy/magnification arrays without shared phase-1 physical identity checks. Writing separate ASCII auxiliary files does not prove OneCovariance reads them.
- Lines 134–139 retain a separate 101-node grid.
- The three spectral table blocks allocate float32 values and ell arrays; `numpy.argsort(ell.flatten())` leaves the tied bin-pair order undefined.
- The wrappers invoke an externally located `CONFIG.ini` without a versioned renderer, input/output identity, pinned reader verification, or post-run physical validation.

The six error notebooks still load `MATRIX.ascii` as float32, reconstruct block identities by fixed offsets, and use the wrong TT/EE triangular formula `j*(j+1)//2+i`. With three bins, `(0,2)` and `(1,1)` are interchanged. Near-zero/exact-zero reference divisions fabricate errors of 1 and covariance ratios of 1. The notebook arrays are still text products with no enforced run/sample/window pairing. Main plots can be huge, e.g. Y10 TT 50-inch square at 512 dpi; moving compute out of notebooks does not by itself fix rendering memory.

The manuscript figure manifest retains old `notebooks/kernels`, `notebooks/matter_power`, and `notebooks/error_analysis` producer paths and has no scientific provenance fields. This must be repaired locally in Phase 5, following accepted remote export production in Phase 4. Current paper prose mixes Gaussian and total covariance and carries numerical claims not revalidated for corrected physics.

## Fixed angular contract and its consequence for covariance

The author fixes `e=geomspace(20,2000,21)` for ALL CCL, NUMBA, JAX CPU/GPU, NUMERIC-linear/quadratic/cubic evaluations. Store all 21 raw samples for reproducibility; comparison and covariance vectors contain 20 bandpowers. There is no extra 101-point shared evaluation grid.

Use the actual notebook angular recipe, including its natural boundary condition:

`s(x)=CubicSpline(log(e), e*C_nodes, bc_type='natural')`

`C_tilde(ell)=s(log(ell))/ell`

`B_b=integral_[log(e_b),log(e_(b+1))] s(x) dx / (e_(b+1)-e_b)`.

This is exactly the uniform-dell integral of the RECONSTRUCTED spectrum. It is not a theorem that the sparse reconstruction exactly equals every underlying continuous CCL spectrum. In particular, constant C does not become a polynomial in log ell and is not exactly preserved by this natural-spline recipe. Do not silently row-normalize a derived 20×21 operator to force constants to match; that changes the adopted estimator. Use exact tests with constant/linear transformed functions `ell*C(ell)` in log ell, and independent integration of the specified spline for general data. All estimator hashes must include boundary condition, coordinates, weights, width normalization, and version.

The physical uniform-dell window is `w_b(ell)=1/(e_(b+1)-e_b)` inside the bin and zero elsewhere. Means and covariance must use this window. Model evaluation at 21 nodes is only an efficient representation of the mean; those noninteger nodes are not 21 independent measured sky modes. Therefore `A diag(var_at_21_nodes) A.T` with an invented node variance is not an acceptable covariance construction. Likewise, passing 20 bandpowers into a reader expecting raw C(ell) is wrong.

For a declared continuum full-sky-fsky Gaussian approximation, define `S_ab(ell)=C_tilde_ab(ell)+N_ab` and

`Sigma_(ab,b),(cd,b') = integral w_b(ell) w_b'(ell) [S_ac*S_bd+S_ad*S_bc]/[(2*ell+1)*f_sky] d ell`.

For these nonoverlapping top hats the Gaussian ell blocks are diagonal. Pair and cross-probe covariance remains nontrivial. An upstream flat-sky `2*ell` formula or discrete integer sum is a different named approximation and must never be compared as though it were algebraically identical. Prefer a matched common continuum convention for production/oracle; compare prefactor-free upstream contractions if necessary. Survey geometry remains an approximation and must be labelled.

NG/SSC require `integral integral w_b(ell) K(ell,ell') w_b'(ell') d ell d ell'` with the SAME windows. Internal integration nodes and convergence refinement are permitted. They are implementation metadata in the covariance artifact, not a second evaluation coordinate for each spectra backend. Gaussian signal interpolation must use the canonical raw21 spline. No additional science spectrum evaluation grid is introduced.

## Fresh primary-source OneCovariance evidence

The GitHub connector successfully read full source at historical audit pin `91e139622eba568d7f742a2d12f6b66a106e6e68`. This pin is historical evidence, NOT proof of the installed NERSC revision.

1. [cov_ell_space.py](https://github.com/rreischke/OneCovariance/blob/91e139622eba568d7f742a2d12f6b66a106e6e68/onecov/cov_ell_space.py): lines 433–454 integer-cast geometric bin boundaries. `__bin_Gaussian` lines 2785–2797 and `__bin_cov_ell_gauss` lines 2816–2836 use uniform INTEGER-ell sums divided by numbers of integer multipoles. This differs from floating-edge continuum integration even though it is not ordinary mode-weighted binning.
2. In the SAME file, `__bin_cov_ell_nongauss` lines 2873–2894 and `__bin_non_Gaussian` lines 3086–3107 integrate `ell*ell'` and normalize by annulus areas `pi*(hi^2-lo^2)`. These are not the Gaussian routine's window and not the author's uniform-dell window. Consequently enabling Gaussian+NG+SSC with matching-looking ell labels does not establish a consistent total covariance.
3. The Gaussian splitter exposes a `calc_prefac` control documented at lines 4440–4445: false omits sky/mode prefactors. Installed behavior and return shapes must be checked before a local adapter consumes it; private method signatures are version-sensitive.
4. [cov_arbitrary_summary.py](https://github.com/rreischke/OneCovariance/blob/91e139622eba568d7f742a2d12f6b66a106e6e68/onecov/cov_arbitrary_summary.py) provides arbitrary filters, with means `(1/2pi) integral ell*C(ell)*W(ell) d ell` at lines 243–284. Therefore the candidate Fourier filter is `W=2*pi/(ell*deltaell)` inside each author bin. But this path also loads real-space filters (lines 158–159), computes real-space pair counts (174–177), and handles pure noise with real-space filters (717 onward). It cannot be accepted just by supplying Fourier top hats. Mean identity, all pure-noise/mixed terms, integration discontinuities, convergence, and full 3×2pt labels must be demonstrated.

Recommended execution: create a project-owned estimator adapter, preserve a deterministic independent Gaussian implementation, and inspect installed upstream contracts. Prefer explicit unbinned kernel projection through the author's windows, or verified arbitrary filters with all noise terms. Do not mutate an untracked external checkout silently; any necessary upstream change must be a recorded versioned patch/pin. Do not fall back to default binned upstream output after an adapter failure.

The upstream paper supports arbitrary-summary design but does not validate this project's adapter: [Reischke et al., OneCovariance](https://arxiv.org/abs/2410.06962). Only installed-source round-trip and physical tests can do that.

## Serialization, vector and physical validation requirements

Input tables contain every Cartesian TT and EE pair (both triangles) and every TE lens×source pair, ordered ell-major, then first bin, then second. Write float64 with enough ASCII digits. Test 2×3 rectangular sentinels with distinct per-ell/per-pair values; test three-bin triangles separately. Round-trip through the INSTALLED upstream reader. Assert finite values, coordinate identity, labels contiguous/1-based as expected, uniqueness, full coverage, signed cross spectra preserved, and EE/TT symmetry.

Output full vector: TT row-major unique upper triangle; TE row-major rectangular lens-first; EE row-major unique upper triangle; band varies fastest within each pair. Actual upstream labels must be mapped, not assumed. For unmasked Y1 5 lens/5 source and 20 bands, N=1100; for Y10 10 lens/5 source, N=2400. Single/Double/Triple scientific workload meanings remain EE / TE+TT / EE+TE+TT, irrespective of canonical storage order.

Keep all field spectra for covariance contractions. Then create a named fixed science mask: TT autos; EE unique pairs; adopted Niko/binny TE eligibility (source peak strictly after lens peak, normalized minimum-overlap <=0.10 Y1 or <=0.25 Y10); explicit deterministic scale cuts. Save exact accepted pair/band list, distribution/common-grid identity, peak-tie rule, overlap integration and thresholds. Compare with actual stored distributions, never invent a list from nominal bin labels. The scale-cut representative redshift and k units require a concrete adopted rule; the current manuscript .3 h/Mpc and quoted ell maxima are not self-consistent. A whole-bin upper-edge criterion preserves each admitted uniform band; a cut through a bin would define a different window. A cutoff based on one representative redshift is not a strict all-radial-support k restriction.

Noise: state source ellipticity dispersion is per component, convert arcmin^-2 densities to sr^-1 correctly, put shot/shape noise on appropriate same-field/same-bin diagonals, preserve signal cross terms and avoid adding observational noise to stored theory means. Galaxy/source populations and overlap assumptions must be explicit. eta_IA=0 leaves IA nonzero; it removes the adopted extra redshift exponent, not the IA signal.

Independent tests: signed PSD field-spectrum fixtures; TT/TE/EE/cross-probe Wick contractions; noise-only exact analytic continuous top-hat covariance `N-product*log((2hi+1)/(2lo+1))/(2*fsky*deltaell^2)`; independent nonconstant-signal quadrature; exact row-order/mask permutations; all-pair and selected-vector identities; window disagreement failure; absent/mismatched source inputs failure. For real Y1/Y10 use symmetry, positive diagonals, expected shapes, correlation bounds, eigenvalues/condition number, Cholesky and deterministic solve residual, recording tolerances and original asymmetry before any diagnostic symmetrization. No jitter, nearest-SPD or eigenvalue clipping to conceal defects.

## NG/SSC scope

The historical audit found full observed-field Gaussian inputs include IA/magnification, but external NG/SSC internal machinery had no established magnification response. A PSD covariance is not evidence for full physical consistency. Start with accepted Gaussian; test NG and SSC separately, including radial weights, bias, IA sign/law, magnification q versus s, cosmology, response/trispectrum model, shared windows, precision and convergence. If unsupported, record a scientific limitation and keep total-covariance claims blocked. Finish all Gaussian/uncertainty-independent work. Do not call a Gaussian diagnostic a complete forecast, or delete intended total-covariance scope quietly.

## Phases 3–5 completion recommendations

Phase 3: covariance implementation/round-trips/oracle; selected vector; allocated tiny CPU/GPU and all NUMERIC orders across both surveys and actual workload configurations; fiducial and representative shared sample science validation; measured RSS and stage/wall/I/O time; resource and total campaign estimate. Pilot exact workload manifest has 2 surveys ×3 configurations×7 method/order/device combinations=42 entries (CPU allocation choices can multiply this). Do not count CCL/NUMBA/JAX three times for interpolation. A sample count of two plus fiducial is sufficient for dispatch/IDs, not a substitute for high-resolution reference validation; use mathematically valid radial grids.

Phase 4: execute final shared1000+fiducial campaign only after accepted gates; sample completion manifest with expected/attempted/completed/matched counts; all42 workloads and resource settings; no subset mislabeled complete; interrupted/restarted timing labelled segmented. Error statistics use band20 and matching sample IDs; full D uses one accepted fixed fiducial covariance/survey and Cholesky, excludes fiducial from ensemble quantiles; never add per-probe D or all sample D as independent observed surveys. Save signed differences, valid masks/counts, exact-zero semantics, absolute residuals and covariance-scaled residuals; pointwise percentiles AFTER absolute values. Store D median/16–84/95/max and worst-sample IDs, Ndata, failed/eligible counts. Plot-only six spectra+six error notebooks, explicit accepted snapshot. Render at bounded size and inspect output, not just execute cells. Produce publication bundle checksums, run/config/sample/window/pair-mask/covariance hashes, software/hardware versions, source commit/dirty-patch identity, command and validation reports. Full arrays stay on CFS.

Phase 5: validate transfer manifest/checksums; local actual paper submodule only; refresh all stale producer paths and provenance; revised theory/endpoint/angular definitions, eta0, numerical claims supported by campaign; all102 comment ledger items; figure/table/caption/abstract/conclusion consistency; compile and visually inspect all pages; code/data availability; final release evidence ledger states what remains limited. Keep parent code and paper Git histories separate. No auto-commit/push or external publishing unless separately authorized. Model-discrepancy evidence must not claim arbitrary-data likelihood shifts, emulation gains, MPI scaling, exact quartic marginalisation or fully matched NG/SSC without its own evidence.
