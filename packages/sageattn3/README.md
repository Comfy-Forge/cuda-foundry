# sageattn3 — build notes

- **Source:** thu-ml/SageAttention v2.2.0 (`eb615cf6`), subdirectory
  `sageattention3_blackwell/`. Same repo and commit as `packages/sageattention`;
  a different distribution (`sageattn3` 1.0.0) with its own `setup.py`.
- **Modules:** two top-level extensions, `fp4attn_cuda` and `fp4quant_cuda`,
  one CUTLASS translation unit each, `sm_100a` / `sm_120a` only.
- **CUDA floor:** `setup.py` raises below CUDA 12.8; there is no cu12.4/12.6
  row and never can be.

## The subdirectory

`build_subdir: sageattention3_blackwell` points the build at the project's
own `setup.py`; the tarball is the whole repo at the pinned commit, so the
provenance recorded in the artifact is the repo's.

## CUTLASS is vendored at fetch time, pinned

Upstream's `setup.py` shells out to `git clone --depth 1 NVIDIA/cutlass` from
inside `build_ext` -- unpinned, whatever the default branch is that day, on
every build. The compile here has no network (L1), so that clone would fail;
and even where it could run it is the one input a pinned rev does not pin.
The patch clones CUTLASS **v4.2.1** (`f3fde583`, the last release before this
tag, i.e. the tree upstream was developing against) into `csrc/cutlass`
during the fetch, and pins the fallback clone line to the same tag so the
build can never reach for a different tree.

## Overrides

- **`arch_override.yml`** — Blackwell only, no PTX; see the file.

## The driver link

Both modules link `-lcuda` for `cuTensorMapEncodeTiled`; `cuda-driver-dev` on
linux-64 supplies the stub (with `-L$PREFIX/lib/stubs` via `build_env`), and
`cuda-cudart-dev_win-64` already ships `cuda.lib`. See
`packages/sageattention/README.md`.

## MSVC

Three header patches (kernel_traits.h, kernel_ws.h, launch.h) taken from
mengqin/SageAttention commit 8bb81e4 via the farm, each under
`#if defined(_MSC_VER)` in the source so the tarball is one tree for both
platforms. They work around MSVC's handling of dependent names in class
templates and of over-aligned by-value kernel parameters (C2719).

## Verification needs a Blackwell GPU

The verify op stops with a named message on any other part. It has not been
run on hardware from this repo yet.
