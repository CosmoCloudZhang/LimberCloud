# Manuscript workflow

The Git repository is the source of truth for manuscript text and publication
figures. Edit and compile `manuscript/main.tex` locally, commit the reviewed
source, and then synchronize the `manuscript/` subtree with Overleaf.

Publication figures under `manuscript/figures/` are tracked. LaTeX auxiliary
files and `manuscript/main.pdf` are ignored. The runtime notebooks and benchmark
scripts write to the external `plots/` tree; only validated publication versions
should be copied into the manuscript.

Compile with:

```bash
make manuscript
```

After compilation, inspect the rendered PDF visually, including figure labels,
cropping, line wrapping, references, and page breaks. A successful LaTeX exit
code alone is not sufficient manuscript validation.
