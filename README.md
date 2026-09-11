# cuda-foundry

Compiles the CUDA extensions ComfyUI node packs depend on — `flash-attn`,
`natten`, `spconv`, `gsplat`, `torchvision`, `torchaudio` and ~40 more —
**once per cell**, and pours that one build into three publishing moulds:

| output | served at | who consumes it |
|---|---|---|
| **conda channel** — `.conda` packages, dependencies declared, shared libraries dynamically linked | `comfy-forge.github.io/cuda-foundry` | pixi / conda, comfy-env's solver |
| **PyPI index** — manylinux wheels with **no** `Requires-Dist` inside and no sidecar (release `<subdir>`) | `comfy-forge.github.io/pypi-cuda-wheels` | direct-URL installs, which must not let a resolver chase dependencies |
| **PyPI index, `/deps/`** — each wheel's *twin*: same filename, the curated `Requires-Dist` written into its METADATA, a PEP 658 `.metadata` sidecar byte-identical to it (release `<subdir>-deps`) | `comfy-forge.github.io/pypi-cuda-wheels/deps` | a plain `pip install` that needs dependencies resolved |

The existing `cuda-wheels` index is untouched and keeps serving: comfy-env
points at it today, and nothing here breaks that. `pypi-cuda-wheels` is this
repo's own publishing target, and a cutover — if it happens — is a one-line
change in comfy-env made deliberately, not a side effect of building.

The two PyPI trees serve **two files per wheel, repackaged from one compile**:
`make_wheel.py` writes the stripped wheel and then rewrites nothing but its
METADATA and RECORD to make the `/deps/` twin, and `verify_wheel.py` asserts
the two differ in nothing else. Two files rather than one file with a
dependency-bearing sidecar, because PEP 658 says the sidecar and the wheel's
METADATA *must be identical* and pip enforces it (measured: pip 26 refuses
every wheel whose sidecar declared what its METADATA did not; uv tolerated
it). Two release tags rather than two asset names, because pip takes a link's
filename from the URL's last path component, so the twin has to be reachable
under the canonical `*.whl` name. Nothing is compiled twice.

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

## One cell, above conda-forge, only where the cell is already pinned

This channel serves exactly one cell today: **py3.12 / CUDA 12.8 / torch 2.8**,
on linux-64 and win-64. Eleven of the names it carries are also on conda-forge
-- torchvision, torchaudio, torchao, torch-scatter, torchsparse, gsplat,
flash-attn, mmcv, pytorch3d, pyg-lib, detectron2 -- and conda's strict channel
priority is name-wide: if a higher-priority channel has ANY build of a name,
every build of that name on lower channels is invisible. So with this channel
listed above conda-forge, conda-forge's entire torchvision line is hidden, and
`python=3.11 torchvision` is UNSAT -- not "resolves to conda-forge's", UNSAT,
because the only torchvision the solver may see is py3.12/cu128/torch2.8.

That is the intended shape, and it is only correct in one place: **an
environment already pinned to the cell**, which comfy-env is. Put this channel
first there. Do not put it above conda-forge in a general-purpose environment,
and do not read a package's `carry: complete` as "conda-forge has no such
package" -- for the eleven names above that is false, and the comments that
said so are being removed. `carry` records that the owner accepted the
shadowing, nothing more. `docs/ARCHITECTURE.md` has the longer form.

## Packages

Generated by `scripts/generate_recipes.py` from `packages/*/package.yml`
(checked by `--check`, so it cannot drift). A **distribution restriction** is
one the upstream licence imposes; it is surfaced here and in the artifact's
`about.extra`, and whether such a package belongs on the channel at all is the
owner's decision, not the build's.

<!-- packages:begin -->
| package | version | license | PyPI purl | distribution restriction |
|---|---|---|---|---|
| `cc-torch` | 0.2 | BSD-3-Clause | none |  |
| `cubvh` | 0.1.2 | MIT AND LicenseRef-NVIDIA-Source-Code-License-instant-ngp AND MPL-2.0 | none |  |
| `cumesh` | 0.0.1 | MIT AND LicenseRef-NVIDIA-Source-Code-License-instant-ngp AND MPL-2.0 | none |  |
| `cumesh-vb` | 1.0 | MIT AND LicenseRef-NVIDIA-Source-Code-License-instant-ngp AND MPL-2.0 | none |  |
| `cumm` | 0.7.11 | Apache-2.0 | cumm |  |
| `custom-rasterizer-hy3d2` | 0.1 | LicenseRef-Tencent-Hunyuan-3D-2.1-Community-License | none | **Licence excludes the EU, UK and South Korea (LICENSE lines 3, 18, 24, 40)** |
| `detectron2` | 0.6 | Apache-2.0 | none |  |
| `diff-gaussian-rasterization` | 0.0.0 | LicenseRef-Gaussian-Splatting-License AND MIT | none |  |
| `diso` | 0.1.4 | CC-BY-NC-4.0 | diso |  |
| `dpvo-cuda` | 0.0.0 | MIT AND MPL-2.0 | none |  |
| `drtk` | 0.1.0 | MIT | none |  |
| `faithc-aot` | 1.5.0 | Apache-2.0 AND MIT | none |  |
| `flash-attn` | 2.8.3 | BSD-3-Clause | flash_attn |  |
| `flex-gemm` | 1.0.0 | MIT | none |  |
| `flex-gemm-ap` | 1.0.0 | MIT | none |  |
| `flex-gemm-vb` | 1.0.0 | MIT | none |  |
| `fused-ssim` | 0.0.0 | MIT | none |  |
| `gsplat` | 1.5.3 | Apache-2.0 AND MIT | gsplat |  |
| `gsplat-maskgaussian` | 1.5.3 | Apache-2.0 AND LicenseRef-Tencent-HY-World-2.0-Community-License AND MIT | none |  |
| `mmcv` | 1.7.2 | Apache-2.0 AND LicenseRef-NVIDIA-License | none |  |
| `natten` | 0.21.6 | MIT AND BSD-3-Clause | natten |  |
| `nunchaku` | 1.2.1 | Apache-2.0 AND BSD-3-Clause AND MIT | none |  |
| `nvdiffrast` | 0.4.0 | LicenseRef-NVIDIA-Source-Code-License-1-Way-Commercial | none |  |
| `nvdiffrec-render` | 0.0.1 | LicenseRef-NVIDIA-Source-Code-License-nvdiffrec | none |  |
| `o-voxel` | 0.0.1 | MIT AND MPL-2.0 | none |  |
| `o-voxel-vb` | 0.0.1 | MIT AND MPL-2.0 | none |  |
| `o-voxel-vb-ap` | 0.0.1 | MIT AND MPL-2.0 | none |  |
| `pointnet2-ops` | 3.0.0 | Unlicense | none |  |
| `pyg-lib` | 0.5.0 | MIT AND BSD-3-Clause AND Apache-2.0 | none |  |
| `pytorch3d` | 0.7.9 | BSD-3-Clause AND MIT | pytorch3d |  |
| `sageattention` | 2.2.0 | Apache-2.0 | sageattention |  |
| `sageattn3` | 1.0.0 | Apache-2.0 AND BSD-3-Clause | none |  |
| `simple-knn` | 1.0.0 | LicenseRef-Gaussian-Splatting-License | none |  |
| `spconv` | 2.3.8 | Apache-2.0 | spconv |  |
| `torch-cluster` | 1.6.3 | MIT | torch_cluster |  |
| `torch-generic-nms` | 0.1 | BSD-3-Clause | none |  |
| `torch-scatter` | 2.1.2 | MIT | torch_scatter |  |
| `torch-sparse` | 0.6.18 | MIT AND Apache-2.0 | torch_sparse |  |
| `torch-spline-conv` | 1.2.2 | MIT | torch_spline_conv |  |
| `torchao` | 0.13.0 | BSD-3-Clause | torchao |  |
| `torchaudio` | per torch pairing | BSD-2-Clause AND BSD-3-Clause AND Apache-2.0 | torchaudio |  |
| `torchsparse` | 2.0.0b0 | MIT | none |  |
| `torchvision` | per torch pairing | BSD-3-Clause | torchvision |  |
| `vox2seq` | 0.0.0 | MIT | none |  |
<!-- packages:end -->

## Layout

| path | purpose |
|---|---|
| `packages/<name>/package.yml` | the single source of truth for one package |
| `templates/` | the recipe template; every recipe, build script and verify op is rendered from it and `package.yml` |
| `scripts/` | loader (the package.yml schema, with every hard error explained), matrix generation, source fetch/patch, renderers |
| `tools/` | artifact verification, index and channel assembly |
| `docs/ARCHITECTURE.md` | the design, and the measurements behind it |
