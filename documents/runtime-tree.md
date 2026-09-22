# Runtime tree

LimberCloud keeps the Git checkout separate from its data and result tree.
Python code constructs external paths through `limbercloud.io.ProjectPaths`.

This page describes the current layout. The shared cosmology table and timing
drivers are available; every spectra launch requires a saved sample table.
The [code revision plan](../revisions/2026-09/CODE_REVISION_PLAN.md) defines
the pending Phase 2 NUMERIC implementation and integrated HDF5 spectrum,
checkpoint, and manifest transactions. Filename helpers alone do not make
those proposed products available. Durable datasets and
publication-export bundles stay on CFS; the local paper owner receives only
accepted figures, compact tables, and provenance as described in the
[manuscript workflow](manuscript-workflow.md).

Set the external root in the ignored repository `.env` before running an
experiment or notebook:

```dotenv
LIMBERCLOUD_RUNTIME_ROOT=/path/to/external/LimberCloud
```

An explicitly exported `LIMBERCLOUD_RUNTIME_ROOT` takes precedence over the
dotenv value. See [environment.md](environment.md) for the complete local
configuration contract.

The runtime tree has one canonical structure:

```text
data/
├── Y1/
└── Y10/
config/
├── cosmology.json
├── survey.json
├── number_density.json
├── galaxy_bias.json
├── magnification_bias.json
└── intrinsic_alignment.json
results/
├── spectra/
│   ├── inputs/<run_id>/
│   │   ├── Cosmologies.npz
│   │   └── Manifest.json
│   ├── CCL/{Y1,Y10}/
│   ├── NUMBA/{Y1,Y10}/
│   └── JAX/{CPU,GPU}/{Y1,Y10}/
├── covariance/{Y1,Y10}/
└── validation/spectra/{Y1,Y10}/
plots/
logs/
```

Scientific labels including `Y1`, `Y10`, `CCL`, `NUMBA`, `JAX`, `CPU`, and
`GPU` retain their uppercase spelling. General directory and configuration
filenames use lowercase spelling.

## Configuration files

The six configuration filenames are fixed by `ProjectPaths.config_file()`.
Their JSON keys and numerical values are independent of the filename cleanup
and must not be changed without a separate scientific review.

## Sample tables and timing files

`experiments/spectra/generate_samples.py` writes the canonical table under
`results/spectra/inputs/<run_id>/`. Its `--run-id` identifies that input
directory. Pass the directory through `--sample-table` on every spectra
launch, including a fiducial-only launch.

Every timing launch evaluates sample 0 first. `--sample-count N` then adds
sample IDs 1 through N; omitting it or setting it to zero evaluates only the
fiducial. `Fiducial` files record sample 0 separately, including any first-call
compilation/initialization. `Cosmology` files record cumulative samples 1..N,
excluding sample 0. Zero sampled rows produce only `Fiducial` files and preserve
previous `Cosmology` products under their original table identity.
The timing drivers do not accept `--run-id`, `--run-config`, `--mode`,
`--resume`, `--fiducial-only`, or `--include-fiducial`.

Experiment configurations use `Single`, `Double`, and `Triple` in generated
timing filenames. Examples are:

```text
Time_Single_Fiducial.txt
Time_Single_Cosmology.txt
Time_Single_COSMOLOGY_Fiducial.txt
Time_Single_COSMOLOGY_Cosmology.txt
Time_Single_PROJECTION_Cosmology.txt
Time_Single_COEFFICIENT_Cosmology.txt
Time_Single_CELL_Cosmology.txt
```

Not every backend writes every stage-specific file. Benchmark readers use the
same canonical naming contract as the experiment writers.

Title-case `Cosmology` identifies the sampled population; uppercase `COSMOLOGY`
identifies the computation stage. Each text product starts with a JSON comment
header recording the table hash, population, stage and timing convention.
The two numeric columns are sample ID 0 and seconds for `Fiducial`, or
sample count and cumulative seconds for `Cosmology`. Sampled checkpoints are
100, 200, ... followed by N if necessary; 0<N<100 has one checkpoint N.
Benchmark readers check table identity and counts and never treat a fiducial
duration as a sampled curve.

These files stay in the family/survey directory shown above, with a device
directory for JAX. A rerun replaces only the populations it evaluates. Older
unqualified and `*_128*` files are left in place and never used as a fallback. Host CPU
allocation belongs in the Slurm header and thread environment, not in a
filename or a `--number` driver argument.

Phase 2 spectrum products will use names such as `Spectra_Single_EE.h5` and
`Manifest_Triple.json` in these same roots. NUMERIC alone adds a radial-order
token, for example `Time_Triple_LINEAR_COSMOLOGY_Cosmology.txt`, under
`results/spectra/NUMERIC/LINEAR/Y1/`. The integrated publication and restart
contracts for those products remain pending.
