# Runtime layout

LimberCloud separates the Git checkout from the external data and result tree.
Python code must construct external paths through `limbercloud.io.ProjectPaths`.

## Layout selection

The transition default is the existing Perlmutter layout:

```bash
export LIMBERCLOUD_LAYOUT=legacy
```

After the inputs are copied and verified, select the canonical layout:

```bash
export LIMBERCLOUD_LAYOUT=canonical
```

Writers never inspect both layouts and guess where to place output. The selected
layout is explicit, preventing a stale legacy file from being mixed with a new
canonical run.

## Directory mapping

| Legacy | Canonical |
| --- | --- |
| `DATA/` | `data/` |
| `INFO/` | `config/` |
| `PYTHON/CCL/` | `results/spectra/CCL/` |
| `PYTHON/NUMBA/` | `results/spectra/NUMBA/` |
| `JAX/CPU/` | `results/spectra/JAX/CPU/` |
| `JAX/GPU/` | `results/spectra/JAX/GPU/` |
| `COVARIANCE/` | `results/covariance/` |
| `PYTHON/CELL/` | `results/validation/spectra/` |
| `PLOT/` | `plots/` |
| `LOG/` | `logs/` |

Scientific labels including `Y1`, `Y10`, `CCL`, `JAX`, `CPU`, and `GPU` retain
their uppercase spelling.

## Configuration filename mapping

| Legacy | Canonical |
| --- | --- |
| `COSMOLOGY.json` | `cosmology.json` |
| `SURVEY.json` | `survey.json` |
| `DENSITY.json` | `number_density.json` |
| `GALAXY.json` | `galaxy_bias.json` |
| `MAGNIFICATION.json` | `magnification_bias.json` |
| `ALIGNMENT.json` | `intrinsic_alignment.json` |

The JSON keys and numerical values are intentionally unchanged.

## Safe input migration

Preview the migration on Perlmutter:

```bash
python3 scripts/migrate_runtime_layout.py "${LIMBERCLOUD_RUNTIME_ROOT}"
```

Perform verified copies:

```bash
python3 scripts/migrate_runtime_layout.py "${LIMBERCLOUD_RUNTIME_ROOT}" --execute
```

The utility copies only input data and configuration, verifies SHA-256 digests,
creates empty canonical output directories, never deletes legacy data, and
refuses to overwrite a different destination file. Historical derived results
remain readable through `LIMBERCLOUD_LAYOUT=legacy`.
