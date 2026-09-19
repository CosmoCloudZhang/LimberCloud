# Runtime tree

LimberCloud keeps the Git checkout separate from its data and result tree.
Python code constructs external paths through `limbercloud.io.ProjectPaths`.

This page describes the current layout. The
[code revision plan](../revisions/2026-09/CODE_REVISION_PLAN.md) defines the
pending shared cosmology table, NUMERIC interpolation variants, HDF5 spectra,
run identities, and checkpoint contracts. Implement and validate these on
NERSC before treating the proposed outputs as available. Durable datasets and
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

## Timing files

Experiment configurations use `Single`, `Double`, and `Triple` in generated
timing filenames. Examples are:

```text
Time_Single_1.txt
Time_Single_1_COSMOLOGY.txt
Time_Single_1_PROJECTION.txt
Time_Single_1_COEFFICIENT.txt
Time_Single_1_CELL.txt
```

Not every backend writes every stage-specific file. Benchmark readers use the
same canonical naming contract as the experiment writers.
