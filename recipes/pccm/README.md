# pccm — a hand-written noarch recipe

`recipes/<name>/recipe.yaml` is normally generated from
`packages/<name>/package.yml` and never edited. This one is written by hand,
which the README's escape-hatch ladder allows only as a last resort with a
reason. The reason:

**pccm is a pure-python dependency, not a CUDA extension.** It is the code
generator that `cumm` and `spconv` import at build time (their `setup.py`
runs it to generate every translation unit) *and* at import time
(`cumm/__init__.py` and `spconv/build.py` both `import pccm`). It exists on
PyPI and on no conda channel — conda-forge ships `ccimport` (its sibling)
but not pccm — so a `cumm` or `spconv` `.conda` declaring the dependency it
really has is UNSAT until something provides it. This does.

Nothing in the shared template applies to it: no CUDA toolkit, no torch
flavour lock, no nvcc seat, no SASS census, no per-cell matrix (it is
`noarch: python`, one artifact for every cell). Squeezing it through
`package.yml` would mean teaching the loader, the matrix, the template and
both verifiers about a package kind that has exactly one member.

What it keeps from the rules that do apply:

- **Source pinned to a commit** — the `v0.4.16` tag dereferenced to
  `66b17a36…`, recorded in `extra.source_rev` like every other artifact.
- **No network in the build** — `pip install .` runs under
  `scripts/build_snippets/nonet.py` (pulled in as a second source so it is
  never a stale copy) with `--no-index`, so the only thing that can be
  installed is the checkout.
- **dist-info hygiene** — `RECORD` and `direct_url.json` are removed, as
  `build.sh` does.

## Building and publishing it

There is no matrix cell for it, so it is built by hand and published to the
`noarch` release, with its fragment committed under `meta/noarch/`:

    rattler-build build --recipe recipes/pccm/recipe.yaml \
        --output-dir out --test native -c conda-forge
    gh release upload noarch out/noarch/pccm-0.4.16-*.conda
    python tools/fragment.py out/noarch/pccm-0.4.16-*.conda noarch

`--test native` runs the recipe's import tests inside a fresh solve of the
built package; `tools/verify_conda.py` is CUDA-shaped (build string, SASS,
torch glob) and does not apply. The publish-side checks that do apply --
no `RECORD`, no `direct_url.json`, `depends` declared -- are made by hand
before the upload.

A rebuild bumps `CUW_BUILD_NUMBER` exactly as a matrix cell would; the
published asset is immutable.
