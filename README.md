# cuda-foundry

Compiles the CUDA extensions ComfyUI node packs depend on — `flash-attn`,
`natten`, `spconv`, `gsplat`, `torchvision`, `torchaudio` and ~40 more —
**once per cell**, and pours that one build into three publishing moulds:

| output | served at | who consumes it |
|---|---|---|
| **conda channel** — `.conda` packages, dependencies declared, shared libraries dynamically linked | `comfy-forge.github.io/cuda-foundry` | pixi / conda, comfy-env's solver |
| **PyPI index** — manylinux wheels, no `Requires-Dist` advertised | `comfy-forge.github.io/pypi-cuda-wheels` | direct-URL installs, which must not let a resolver chase dependencies |
| **PyPI index, `/deps/`** — the *same* wheels, advertising their PEP 658 `.metadata` sidecars | `comfy-forge.github.io/pypi-cuda-wheels/deps` | a plain `pip install` that needs dependencies resolved |

The existing `cuda-wheels` index is untouched and keeps serving: comfy-env
points at it today, and nothing here breaks that. `pypi-cuda-wheels` is this
repo's own publishing target, and a cutover — if it happens — is a one-line
change in comfy-env made deliberately, not a side effect of building.

The two PyPI trees are the same files. They differ only in whether the index
advertises `data-core-metadata`, which is what decides if pip fetches the
sidecar. Nothing is built twice and the trees cannot drift in content.

A cell is one `(package, torch, cuda, python, platform)` combination.

## Why one repo

`cuda-wheels` and `conda-cuda-packages` describe the same facts about the same
42 packages — source, revision, arch list, compile parallelism, dependencies —
and compile them with the same compiler against torch binaries that are
byte-identical (conda-torch's `pytorch` *is* the PyPI wheel, repacked). Keeping
two declarations of that meant two places to fix a patch, two chances to
mislabel a cell, and two answers to "what does this package depend on".

Here there is one `packages/<name>/package.yml` per package, and renderers
turn it into whatever a given output needs.

## What a package declares

Facts, not recipes. One template renders the build; per-package escape hatches
exist in this order, and a hand-written recipe is the last resort and needs a
README explaining why:

1. declarative fields in `package.yml`
2. `build_script_extra:` — a snippet appended to the shared build script
3. `patch_script` — a Python patch applied to the fetched source
4. a hand-written recipe, which the generator then skips

The earlier hand-written-recipe experiment is the argument for this: three
recipes, ~90% identical text, and they had already drifted —
`FLASH_ATTENTION_FORCE_BUILD` set in one and not the others, two divergent
Windows blocks. At 42 packages that is a maintenance fork waiting to happen.

## Compiled from source, enforced

No build here may ship a binary it did not compile. Not a convention — a
property of how the build runs. See `docs/ARCHITECTURE.md`; the short version
is a seccomp filter that removes network from the compile, per-package
declarations of the upstream flags that would otherwise fetch a prebuilt
wheel, a ledger of every translation unit compiled, and a canary that proves
the guarantee positively on every run.

## Layout

| path | purpose |
|---|---|
| `packages/<name>/package.yml` | the single source of truth for one package |
| `templates/` | the recipe and wheel-build templates |
| `scripts/` | loader, matrix generation, source fetch/patch, renderers |
| `tools/` | artifact verification, index and channel assembly |
| `docs/ARCHITECTURE.md` | the design, and the measurements behind it |
