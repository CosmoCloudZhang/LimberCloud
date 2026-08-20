# Runtime tree

LimberCloud keeps the Git checkout separate from its data and result tree.
Python code constructs external paths through `limbercloud.io.ProjectPaths`.

Set the external root before running an experiment or notebook:

```bash
export LIMBERCLOUD_RUNTIME_ROOT=/path/to/external/LimberCloud
```

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
