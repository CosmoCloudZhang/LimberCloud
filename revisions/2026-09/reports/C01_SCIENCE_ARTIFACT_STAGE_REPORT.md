# C01–C09 / Prompt 1 — Science contract and artifact foundation

**Stage date:** 2026-09-22  
**Checkout:** `/pscratch/sd/y/yhzhang/LimberCloud`  
**HEAD:** `cb3e5670ecb51a04a68de9d9cfb8e362c9913c78` (uncommitted working tree on top of this commit)  
**Paper pin:** `90d12f4f3e574a67c25944d27d7ded553e09402b` (`git ls-tree HEAD manuscript`)  
**Historical science audit:** `0876bf4a50e869be1289a3eecb46931e7c8eb534`  
**Manuscript:** not initialized. No `git -C manuscript`. No manuscript file edits.  
**Campaign:** not submitted. No scientific driver `main()` was executed.

The pre-existing uncommitted notebook edit and the deinitialized `manuscript` gitlink were left as they were.

## Scientific choices recorded in code

| Choice | Decision in this stage |
| --- | --- |
| Sample campaign | Fiducial ID 0 plus sampled IDs 1–1000. ID 0 consumes no random draw. |
| Shared domain | ±10% multiplicative bounds on the current nonzero primary parameters. `WA` and `OMEGA_K` stay fixed only while they are zero. Nonzero values of those two are refused rather than given an invented additive range. |
| Negative `w0` | Bounds are sorted so the lower edge is less than the upper edge. |
| RNG | `numpy.random.default_rng`. Draws are parameter-major and vectorised, in `SAMPLED_PARAMETERS` order. Fixed parameters consume no draw. Restart loads `Cosmologies.npz` by sample ID. |
| Seed | Required on `experiments/spectra/generate_samples.py`. There is no hidden campaign seed. |
| Configuration | Single=`EE`, Double=`TE+TT`, Triple=`EE+TE+TT`. |
| Magnification | Stored generator values are slopes `s`. CCL `mag_bias` receives `s`. Analytical weights use `q=5s-2`. For `s=0.4`, `q=0` and the weighted lens row vanishes. |
| MS response | MS and MI both multiply the lens distribution by `q`. |
| Active cosmology | Lensing amplitude uses the sample's `Omega_m` and `h`. Radial weights use that same `h`, not the fiducial JSON value. |
| IA / galaxy bias | Both stay fixed tables generated at the fiducial cosmology. They are not regenerated with the sampled growth factor. |
| `eta_IA` | **Unresolved.** Generator provenance is 0.5. Manuscript provenance is 0.0. `EtaIADecision.resolved_value` raises until a caller passes an explicit value. The historical generator still writes the 0.5 array when `--eta-ia` is omitted, and the JSON status is `unresolved`. |
| Ell estimators | 21 LimberCloud geomspace edges, 20 CCL geometric centres, and the notebook uniform-`dell` band average are different estimators. Comparison requires a matching fingerprint. Raw sampled spectra and bandpowers are separate groups. |
| NN observer factor | Left at `1/12`, `1/12`, `1/4` for `P=P1*(chi/chi1)^3`. The withdrawn `1/2` replacement was not applied. |
| NN final diagonal | Current `NN.coefficient` still omits the rising-hat diagonal on the last interval. An independent quadrature of that interval is nonzero. The kernel was not changed: a zero-endpoint restriction versus including the term is still open. |
| NS support | The first subdiagonal is nonzero on a generic fixture. SS does write its final diagonal. |
| NUMERIC | `linear` maps to SciPy `slinear`. It interpolates `a(chi)` and does not special-case the cubic observer interval, so it is an approximation comparison. The matched-integrand oracle is separate. |

## What the timing drivers now do

The 24 spectra entries no longer call `numpy.random.uniform`. A positive `--sample-count` requires `--sample-table` and reads IDs `1..N`. `--fiducial-only` still means zero sampled rows. `--include-fiducial` is refused by these timing entries so sample 0 is not hidden inside the sampled timing loop. `--resume` is refused there as well.

HDF5 resume is implemented on the artifact writer: validated shards are skipped, corrupt or partial files are not completed samples, and failures are recorded instead of replaced by a new draw. One writer lock is required per namespace. Checkpoints are written to a temporary file, reopened, then renamed. A cross-filesystem publication copies onto the destination filesystem, checks the checksum, and only then renames. The manifest is published last. Readers reject an incomplete manifest or a checksum mismatch. Consolidated `cl` is float64 with axes `(sample, ell, pair)`. Coefficient tensors are not stored.

Benchmark readers keep the historical family/survey filenames when `--run-id` is omitted. `--legacy` makes that explicit and cannot be combined with `--run-id`. NUMERIC names use `LINEAR` / `QUADRATIC` / `CUBIC` in both the directory and the basename.

## Validation

`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` → **52 OK** (about 11 s).  
`ruff check` on the touched Python surfaces → **passed**.  
`NUMBA/Y1/single.py --help` and `generate_samples.py --help` exit 0 and expose `--sample-count` / `--fiducial-only`. Those processes do not enter `main()`.

Default sample selection is `0` sampled rows. `--fiducial-only` selects sample ID 0 in the shared ID list and zero sampled rows in the timing loop.

## Not done in this stage

- No 1,001-row table was written to CFS, and no production or pilot science job was submitted.
- Timing drivers still do not evaluate sample 0 and still do not write HDF5 spectra. The writer is covered by unit tests.
- CCL timing drivers still evaluate 20 geometric centres. Numba and JAX timing drivers still evaluate 21 edges. Those products are not saved as shared spectra. The next execution stage has to point every backend at one declared estimator before publishing comparable `cl` arrays.
- `eta_IA` and the NN far-endpoint policy remain explicit open choices for the local owner.
