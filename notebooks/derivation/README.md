# Coefficient derivations

The Jupyter editions beside the original Mathematica notebooks make the full
derivations readable without Mathematica or external cosmology data. The original
`.nb` files are unchanged. Each edition follows the original scientific cell
order: dimensional ordinary/observer integrals, variable definitions, dimensional
prefactors, normalized I/J integrals, Wolfram inputs, and saved symbolic outputs.

The reading layer consistently uses **p for normalized power**,
`p = 1 - P1/P2`, while dimensional endpoint powers remain `P1` and `P2`.
Verified corrections are identified at the beginning of affected notebooks. An
appendix preserves every original content cell, including original symbol
spellings, assumptions, cached outputs, and cell metadata. Each reading cell links
to its original; group coordinates and source lines retain the correspondence.
Wolfram inputs are labelled text, never Python code. Saved Wolfram outputs are
transcriptions, not claims of a fresh Wolfram execution.

| Family | Existing coefficient editions | Formula cells | Input cells | Saved outputs |
| --- | --- | ---: | ---: | ---: |
| NN | [B01](NN/Coefficient_B01.ipynb), [B02](NN/Coefficient_B02.ipynb), [B03](NN/Coefficient_B03.ipynb) | 20 | 6 | 6 |
| NS | [B01](NS/Coefficient_B01.ipynb), [B02](NS/Coefficient_B02.ipynb), [B03](NS/Coefficient_B03.ipynb), [B04](NS/Coefficient_B04.ipynb), [B05](NS/Coefficient_B05.ipynb), [B06](NS/Coefficient_B06.ipynb), [B07](NS/Coefficient_B07.ipynb), [B08](NS/Coefficient_B08.ipynb) | 56 | 16 | 16 |
| SS | [B01](SS/Coefficient_B01.ipynb), [B02](SS/Coefficient_B02.ipynb), [B03](SS/Coefficient_B03.ipynb), [B04](SS/Coefficient_B04.ipynb), [B05](SS/Coefficient_B05.ipynb), [B06](SS/Coefficient_B06.ipynb), [B07](SS/Coefficient_B07.ipynb), [B08](SS/Coefficient_B08.ipynb), [B09](SS/Coefficient_B09.ipynb), [B10](SS/Coefficient_B10.ipynb) | 70 | 20 | 20 |
| **Total** | **21 notebooks** | **146** | **42** | **42** |

The 146 formula cells contain 239 stored TeX expressions. The source inventory
counts actual notebook content, not the repeated frontend outline cache. No
graphics or interactive objects occur in these 21 sources. Exact source and
target hashes, individual content-cell hashes, source lines/groups, and target
cell indices are recorded in [conversion_inventory.json](conversion_inventory.json).

Rebuild these 21 editions and their inventory with the selected project Python:

```bash
.venv/bin/python scripts/convert_derivation_notebooks.py
.venv/bin/python scripts/validate_notebooks.py
```

The converter is deliberately limited to the existing box types and source
catalog. Unsupported cells fail conversion rather than disappearing silently. It
does not rewrite the original `.nb`, equation `.txt`, or `_Validation.ipynb`
files, and does not generate the four new boundary derivations.

## Notation and impact ledger

| Source | Verified issue | Correction and impact |
| --- | --- | --- |
| NN B02 | Mixed NN/phi-phi display label; observer-transpose dimensional display uses interval 1 although the direct integral uses 0 | Corrected to phi-phi and interval 0 in the reading layer. The numerical coefficient is unchanged. |
| NN B03 | Extra closing brace in displayed observer denominator | Corrected TeX rendering; original source retained. No algebraic change. |
| NS B04 | Display defines normalized redshift as y, but integration/export/backend use z | Reading layer uses z. Display inconsistency only. |
| NS B07 | Defined power is p; I7 input/output use y while assumptions name p | Reading layer uses p consistently. This is an assumption-reproducibility issue as well as a rename. A fresh SymPy integration with explicit symbols agrees exactly with the unchanged I7 and J7 exports. |
| NS B08 | Notebook consistently calls normalized power y; export/backend call it p | Reading layer maps y to p. Original integration variable x remains x. Consistent renaming, not a formula defect. |
| SS B06 | Dimensional observer display refers to J7; its integral/input/export/backend use J6 | Reading layer corrects the cross-reference to J6. The J6 coefficient is unchanged. |
| SS B09 | I9 integrand/output use p, while an assumption list mentions y | Corrected assumption uses p. A fresh SymPy integration with explicit symbols agrees exactly with unchanged I9 and J9 exports. |

The two assumption-sensitive notebooks contain optional executable SymPy checks.
They initialize every symbol explicitly, state the ordinary interval domain
`0 < a < 1`, and separately integrate the observer's cubic power. They embed the
unchanged equation exports so execution needs no external runtime files. SymPy
and IPython are notebook dependencies only; reading the notebook needs neither.

On 22 September 2026, those checks were executed with SymPy 1.14.0: all four
ordinary/observer differences simplified to exactly zero. An additional symbolic
comparison of **all 42 cached outputs** with all 42 existing equation exports,
after the documented renaming, also gave exact zero differences. This establishes
export transcription consistency; it is not a fresh integration of every
original input or acceptance of every matrix-placement domain. The affected
exports and their existing Numba/JAX representations retain the same algebra.

All 230 original content-cell hashes and 21 unchanged source hashes were checked
against the generated appendices. The 239 displayed TeX expressions and 42
box-rendered cached outputs were compiled through LaTeX without mathematical
syntax errors. The original box nesting was retained in saved-output fractions,
powers, products, and logarithms; source TeX drives the displayed derivations.
No Wolfram kernel was run.

## Boundary-domain ledger

Notation corrections above are separate from boundary-case corrections. The
original headings remain historical statements, with these qualifications:

- NS B07's whole-support terminal-source expression requires the evaluation
  interval to end at or below the start of the final source support: `n+1 < N`.
  Its original `n < N` heading is too broad.
- NS B08's `n+1 < N` restriction correctly excludes the final density/source
  diagonal; it is not a complete expression for that final interval.
- NS B02's falling-source expression also applies to the final density row.
  Its original strict endpoint index restriction does not require new algebra.
- Final-interval source support must use a moving lower source limit. This is
  the same inside/outside rule as for interior hats, with no descending half
  beyond the last source node.

The additional catalog cases are NS B09/B10 (last-interval falling/rising density
times the terminal rising source) and SS B11/B12 (falling-times-rising source,
and rising-source square). They replace the inappropriate last-interval uses of
older expressions; they do not add a second copy of an existing contribution.
Their derivations and numerical validation are separate from this faithful
conversion inventory.
