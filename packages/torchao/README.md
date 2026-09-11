# torchao — build notes

- **Source:** pytorch/ao v0.13.0 (`e318546f`), recursive (third_party/cutlass).
- **Why 0.13.0, not the farm's 0.18.0:** torchao pairs with torch, and
  0.13.0 is the release upstream built against torch 2.8.0. Every later
  release was measured on 2.8.0+cu128 and either does not import (0.18.0),
  has dropped the general CUDA kernels (0.16.0+), or refuses to load its own
  extensions through the (torchao, torch) table in `torchao/__init__.py`
  (0.14.0, 0.15.0). The full record is in `package.yml`.
- **Modules:** `_C` (abi3; fp6_llm, marlin, sparse_marlin,
  tensor_core_tiled_layout, activation24, rowwise_scaled_linear_cutlass
  s8s4/s4s4, built for the cell's arch list), `_C_cutlass_90a` (sm_90a only),
  `_C_cutlass_100a` (sm_100a only), and `torchao.prototype.mxfp8_cuda`
  (sm_100 + compute_120 PTX). The wheel is `cp39-abi3` because setup.py
  builds with `py_limited_api`; the conda package is still the py312 cell.

## Overrides

- **`arch_override.yml`** — Ampere and up: upstream's own wheels are built
  for 8.0;8.6 and the kernels are guarded `__CUDA_ARCH__ >= 800`. The SASS
  census reads the union over the four modules.

## Windows

Upstream ships no CUDA build for Windows, and the farm recorded its 0.18.0
Windows cells as pure-python wheels. Here the win-64 cell builds all four
CUDA modules (run 34601149667: `_C`, `_C_cutlass_90a`, `_C_cutlass_100a`,
`mxfp8_cuda`, SASS census identical to Linux's). Three things stood in the
way, each measured on its own run and fixed in `patches/torchao.py` at
build time or in the source with no platform branch in the patch:

- `setup.py` withholds the CUTLASS include directories on `not IS_WINDOWS`
  while its source filter drops only files with "cutlass" in the name;
  `activation24/*.cu` include `<cutlass/bfloat16.h>` and died with C1083
  (run 34586352085). Windows now gets the include directories and the MSVC
  host flags CUTLASS documents (`/Zc:__cplusplus /bigobj /Zc:preprocessor
  /permissive-`), forwarded through nvcc.
- `_C`, `_C_cutlass_90a` and `_C_cutlass_100a` are torch-ops libraries with
  no Python module, and distutils on Windows links every extension with
  `/EXPORT:PyInit_<name>` (LNK2001, run 34596571344). A stub source
  (`torchao/cuw_pyinit_stub.cpp`, empty off Windows) exports a PyInit that
  raises ImportError; the name comes from `TORCH_EXTENSION_NAME`.
- `mxfp8_quantize.cuh` uses `#warning`, which cl rejects even in a skipped
  group (C1021, run 34598640540); it is now `#pragma message`, which every
  host compiler here accepts.

## Patch

`patches/torchao.py`: stops `test/` shipping as a top-level package (it
collides with CPython's own stdlib `test`).
