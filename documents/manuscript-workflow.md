# Manuscript workflow

`LimberCloudPaper` is the source of truth for manuscript text and publication
figures. It is checked out locally through the parent repository's `manuscript/`
submodule. Local Codex handles manuscript edits, compilation, figure integration,
and review. Remote Cursor on NERSC handles code, scripts, notebooks,
environments, tests, and scientific runs. The [revision package](../revisions/2026-09/README.md)
defines the scientific gates and detailed responsibilities.

The parent tracks the paper through two entries: the URL/path in `.gitmodules`
and a mode-160000 `manuscript` reference to a paper commit. The shared
`fetchRecurseSubmodules = false` setting prevents automatic manuscript fetching
unless overridden; the explicit local update command below remains available.

## Receive changes and edit the paper locally

First commit or otherwise preserve your work in both repositories. From the
local parent checkout, receive the recorded versions:

```bash
git pull --ff-only
git submodule update --init --recursive manuscript
```

The update checks out the paper commit recorded by the parent and may leave
the paper in detached-HEAD state. Before editing, switch to its working branch:

```bash
git -C manuscript switch main
git -C manuscript pull --ff-only
```

This second step may advance the paper beyond the parent's recorded commit.
Review that difference before recording a new reference. Stage only the
intended paper source and publication assets inside `manuscript/`, inspect the
staged diff, commit them there, and push the paper repository first. Then, from
the parent checkout:

```bash
git add manuscript
git diff --cached --submodule=log
git commit -m "Update manuscript revision"
git push origin main
```

The parent records only the paper commit; `git add manuscript` does not commit
uncommitted files inside the paper. If either push encounters new upstream
commits, fetch and reconcile them before retrying. Avoid recording an older
paper commit when receiving independent code updates.

## Keep the NERSC checkout focused on code

Retain `.gitmodules` and the tracked `manuscript` Git reference in the parent
repository. The paper working directory on NERSC may be empty or absent; it
does not need paper-repository credentials. If the paper is currently
initialized, first inspect its own status and preserve any edits:

```bash
git -C manuscript status --short
git submodule deinit -- manuscript
```

Use the status command only for an initialized paper checkout. If deinit
reports local changes, preserve them before proceeding; do not force it.
After deinitialization, an empty placeholder can be removed with
`rmdir manuscript`. These are checkout-local operations and require no commit.
Cached paper Git history may remain under the parent's `.git/modules/`.

On NERSC only, set these checkout-local defaults once. They do not change the
Mac checkout or remove the parent repository's manuscript reference:

```bash
git config --local submodule.recurse false
git config --local fetch.recurseSubmodules false
```

For routine NERSC updates, run from the parent checkout:

```bash
git pull --ff-only --no-recurse-submodules
git ls-tree HEAD manuscript
```

The second command reports the recorded paper commit without opening the
paper repository. A leading `-` in `git submodule status manuscript` is expected
for an uninitialized submodule. Do not use `git -C manuscript` to inspect an
empty placeholder: Git can discover the parent repository and report its
status instead. Do not remove the tracked reference with `git rm`, delete its
`.gitmodules` entry, or initialize it as a code-setup step. Code checks must
work with both an absent and an empty `manuscript/` directory.

If a code commit accidentally deletes the gitlink, restore the reviewed paper
reference rather than adding paper files to the parent or creating another
submodule. An existing initialized local paper may already have been staged
correctly by `git add manuscript`, even if Git warned about an embedded
repository. Verify before committing:

```bash
git ls-files --stage manuscript
git diff --cached --submodule=log -- .gitmodules manuscript
```

The index must show exactly one mode-160000 `manuscript` entry at the intended
paper commit, with its matching `.gitmodules` URL. This staged restoration
appears in `git ls-tree HEAD manuscript` only after the parent commit records it.
Do not run `git rm --cached manuscript` merely to silence the warning; that
would remove the reference being restored.

## Transfer accepted figures and tables

NERSC scripts write publication exports under the external CFS runtime tree.
The export bundle contains accepted figures, compact summary tables,
checksums, generating commands, the code commit, and input/run/configuration
identities plus validation status. Remote agents supply proposed provenance
entries for the paper's figure manifest; the local paper owner makes the
actual manifest changes. Do not create a remote `manuscript/` export directory.

Transfer the bundle using the established NERSC file-transfer workflow and
verify its checksums locally. Copy accepted publication assets into
`manuscript/figures/` and update the paper's figure manifest and captions.
Large spectra and covariance arrays remain on CFS. Record the code commit
that generated each asset, then push the paper and update the parent reference;
this avoids requiring the two commits to refer to each other's future hashes.

Publication figures are tracked in the paper repository. LaTeX auxiliary files
and `manuscript/main.pdf` are ignored. Compile locally from the parent checkout:

```bash
make manuscript
```

After compilation, inspect the rendered PDF visually, including figure labels,
cropping, line wrapping, references, and page breaks. A successful LaTeX exit
code alone is not sufficient manuscript validation.

If Overleaf synchronization is used later, it should target the separate
`LimberCloudPaper` repository. The former parent-repository subtree workflow
is superseded by the submodule arrangement.
