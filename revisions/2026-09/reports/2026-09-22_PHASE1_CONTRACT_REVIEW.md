# Phase 1 contract and implementation review — 2026-09-22

## Verdict and reviewed state

The five revised launch/product choices are broadly sensible, but Phase 1 is not
fully closed. The remaining blockers are nuisance/table identity binding and
repeatable endpoint acceptance, including verified numerical defects in terminal
kernels. The ordinary angular natural-spline implementation and method/order
naming are internally consistent in the reviewed code.

- HEAD: `72bb8af8c486a579e08405e9b68308c7265147ef`; review includes the dirty working tree.
- Paper gitlink: `90d12f4f3e574a67c25944d27d7ded553e09402b`; no manuscript work.
- Compute scope: 62 Python files under `src/limbercloud` and `experiments/spectra`.
- Source fingerprint: `ff51b915f29cb6380ef08dace5c9d83b23a4a505fe0aea1b6b6e8638069a0f73`.
- Fingerprint obtained with `compute_source_manifest()` and `source_fingerprint()`;
  it excludes reports, plans, runtime data and the paper, and is a digest rather
  than a reconstructable source snapshot. The original Phase 1 report names an
  earlier source digest and must not certify this working tree unchanged.
- This review changes documentation/plans only; it makes no production or science
  source repairs, launches no campaign, and creates no accepted science artifact.

## Revised choices to preserve

1. Allocation-free names such as `Time_Triple.txt`, `Spectra_Single_EE.h5` and
   `Manifest_Triple.json` are reasonable. Only NUMERIC carries an order token.
   Historical `*_128*` files remain distinct; never silently mix their contents
   into current products.
2. Removing `--number` is reasonable: Slurm and thread environment variables own
   host allocation. Record actual resources with each execution so timings remain
   interpretable after filenames cease to encode allocation.
3. The performance figure may cover CCL, Numba and JAX only and save
   `benchmark_{label}.pdf`. NUMERIC remains a scientific comparison method.
   The repeated stage-panel CCL curve must visibly identify its total boundary.
4. Deleted endpoint-test/diagnostic filenames and full-assembly oracle helpers
   need not return. However, a historical report cannot replace executable
   regression coverage of currently changed endpoint code. Put compact independent
   fixtures in the retained science test modules; retain no unnecessary oracle API.
5. Every timing launch evaluates sample 0, followed by `--sample-count` extra rows;
   default 0 means the fiducial alone, and the saved sample table remains required.
   Reruns may replace canonical timing files in the family/survey directory.
   Do not restore the removed fiducial, run-ID, run-config, mode or resume flags.
   `generate_samples.py --run-id` still names the canonical input-table directory.

## Prioritized findings

### P1 — Terminal source integrals lose stability on narrow intervals

Both `projection/numba_backend/terminal.py:42–55` and
`projection/jax_backend/terminal.py:33–39` form small positive source integrals
from differences of larger polynomial/logarithmic terms. On `[1, 1+1e-5]`, at its
midpoint, the compiled Numba helper returned:

| Quantity | Current value | Independent value |
|---|---:|---:|
| H | `1.9914337512881217e-11` | `1.04165781259021e-11` |
| F | `-7.104319438459451e-12` | `2.0833177084804204e-12` |
| SS terminal diagonal | `3.6584501488052786e-27` | `2.6190066474076515e-27` |

The diagonal is 39.7% high. At width `1e-7`, the SS cross entry is
`-3.6309634331501030e-26` instead of positive `9.920633610311011e-38`.
This is cancellation in the source functions, not evidence of insufficient outer
Gauss order: independent positive source-integral quadrature at orders 16 and 32
agrees to better than `7e-16` relatively on these fixtures.

Use stable dimensionless/log1p/series expressions or an equivalent positive
integral representation. Test progressively narrow intervals, signs and assembled
coefficients. Only Numba was executed; JAX shares the vulnerable algebra, but
its numerical behavior and GPU execution remain untested here.

### P1 — A terminal interval that is also the observer uses the wrong power law

The terminal kernels interpolate node power linearly without handling the case
where the final interval is also the observer interval. For the single interval
`[0,1]`, powers `[0,1]`, and redshifts `[0,0]`, compiled Numba gives:

| Entry | Current | Declared cubic-observer reference |
|---|---:|---:|
| NS falling | `1/8` | `1/120` |
| NS rising | `1/24` | `1/120` |
| SS diagonal | `1/120` | `1/1120` |

The shared analytic contract retains `P=P1*(chi/chi1)^3` on the observer interval.
Apply that contract consistently when observer and terminal intervals coincide,
or explicitly reject an unsupported minimal grid. Add the one-interval fixture
alongside multi-interval tests; do not alter the accepted NN `1/4` factor.

### P1 — Nuisance and sample-table provenance is recorded without binding it

`validation/nuisance.py:76–95` checks that generating-model, fiducial-input and
solver fingerprints are present; it does not compare them with the selected
fiducial/table and current solver. Drivers such as
`experiments/spectra/NUMBA/Y1/triple.py:111–125` print the provenance and proceed.
A stale table with valid-looking metadata can therefore satisfy the structural
loader while representing a different fiducial nuisance model.

Separately, `validation/samples.py:404–455` returns the manifest's solver
fingerprint without validating it, and `timing_loop_rows()` discards the identity
when it returns row dictionaries. A lightweight temporary-table reproduction
changed only `solver_fingerprint` to `incompatible-solver` and `schema_version`
to `invalid-schema`: loading succeeded and the fiducial row was returned.

Before evaluation, bind the nuisance generating cosmology to saved sample 0 and
its solver, validate the table's schema/solver identity, and preserve fixed
fiducial nuisance arrays across nonfiducial samples. Keep actual array/input
checksums distinct from generating-cosmology provenance. Do not repair stale
arrays by relabelling metadata.

### P2 — Benchmark abscissae and displayed labels disagree with supported runs

Both `experiments/benchmarks/{Y1,Y10}/benchmark.py:67–70` hardcode 10 checkpoints
from 100 to 1000. Drivers permit other counts. A synthetic 50-sample fixture
actually represents `[1,6,11,17,22,28,33,39,44,50]`, but is plotted against
`[100,200,...,1000]`. A scalar default-run timing product instead fails because
its length does not match the ten x coordinates.

Persist/read actual sampled checkpoint IDs/counts, reject cross-method coverage
mismatches, and give fiducial-only input an explicit no-sampled-benchmark outcome.
A simple sidecar can close the immediate text-product gap; the later manifest
and per-sample timing products should become authoritative.

The stage rows use a `CCL (total)` label, but all three stage panels set
`show_legend=False` at lines 163, 183 and 198. The visible total-panel legend says
only CCL, so the new clarification does not appear in the saved figure. Add a
visible legend entry or annotation that identifies this repeated total curve.

### P2 — Endpoint acceptance is not reproducible in the retained tests

The deleted dedicated endpoint suite and diagnostic are no longer current-tree
evidence. Retained backend-consistency tests cannot establish correctness when
Numba and JAX share the same formula. Ordinary NS/SN/SS right-node-zero power
ratios also remain a known limitation acknowledged by the original Phase 1
report; repairing only terminal entries does not establish zero-safe assembly.

Retain compact independent fixtures for observer, terminal, narrow intervals,
zero/signed node powers, orientation and final contraction in existing science
tests. Mark R05 failed pending identity binding, R08 implemented but unverified
until retained compiled regression evidence exists, and R09 failed pending
stable/observer/zero-node coverage. Do not treat the old report link as a pass.

### Performance concern requiring measurement

JAX uses terminal quadrature inside loop bodies through `jnp.where`; Numba has
an explicit terminal branch. This is a source-level concern about extra terminal
work on ordinary intervals, not a measured performance regression. Inspect the
lowered JAX computation and measure before optimizing or assigning analytical
stage costs. Explain truthfully that terminal handling currently uses a
48-point quadrature rule rather than describing every coefficient as closed form.

## Work already assigned to later phases

Phase 2 must still integrate the real evaluator, final probes, raw21/band20
storage, device/precision checks and timed completed computation. Existing
artifact helpers are scaffolding: `io/artifacts.py:802–806` omits `pair_j` when
checking consolidated axes, readable malformed HDF5 can escape as missing-key
errors, and finite/semantic checks are incomplete. `read_completed_manifest()`
at line 967 accepts `{"status":"complete"}` with no products, as reproduced.
The current Phase 2 hardening requirements already cover these gaps; retain them.

Phase 2 identities need content hashes for physical/nuisance arrays, not only
fiducial model fingerprints. The current dependency signature records package
versions and hardcodes float64 enabled; actual build, precision, device and
thread evidence must be obtained from the executing method.

Covariance remains a Phase 3 gate. Both historical `matrix.py` files overwrite
the source grid with the lens grid, recompute 101-node float32 spectra, use
ell-only sorting and bypass accepted nuisance/model validation. Existing Phase 3
instructions explicitly replace these paths, verify the actual upstream reader,
and establish matched windows/noise/selection. Launcher repairs do not certify
the old covariance outputs.

## Subsequent-plan adjustments

- Close the bounded identity/endpoint/benchmark defects before certifying outputs
  from the shared evaluator. Independent Phase 2 implementation can continue.
- Reconcile Phase 2's contradictory interface paragraphs: keep the simple timing
  CLI; use an internal/API execution specification for configuration, diagnostics
  and strict HDF5 recovery. Do not reintroduce prohibited flags indirectly.
- Keep fresh timing reruns distinct from scientific checkpoint recovery. Define
  interruption, conflict and accepted-generation behavior concretely. Mutable
  canonical filenames require immutable external snapshots for accepted evidence.
- Preserve `benchmark_{label}.pdf`; record product/source identity in its companion
  provenance and snapshot accepted outputs rather than adding run-ID filenames.
- Keep Phase 3's 42 workload pilots and covariance gates. No allocation-token
  variants or extra covariance campaign are needed. Add measured terminal-cost
  evidence where it affects timing interpretation.
- Phase 4 freezes canonical input generations and accepted producer identities;
  repeated generation must not silently replace a campaign's table or evidence.
- Phase 5 must state the actual hybrid analytic/terminal-quadrature implementation
  and the boundaries of tests and accepted performance claims.

## Verification and limits

The root review ran `make check`: 72 fast tests passed, Ruff passed, shell parsing
passed and 40 notebooks parsed. `git diff --check` passed. These are code-level
checks; they do not establish scientific or allocated performance acceptance.

The peer terminal diagnostic is `/tmp/limbercloud_phase1_science_review.py`, with
output `/tmp/limbercloud_phase1_science_review.log`; these temporary files are not
durable regression artifacts. It executed actual compiled Numba kernels with one
thread on bounded synthetic inputs. The independent reference used 70-digit
Decimal source integrals and positive quadrature. The source/values above record
the resulting evidence while permanent regression integration remains pending.

No MPI, HDF5, CCL, CAMB, GPU or allocated science acceptance ran in this review.
The table/empty-manifest reproductions used temporary files and NumPy-only paths.
Benchmark findings used synthetic fixtures. No expensive production cosmology
or campaign result is claimed.
