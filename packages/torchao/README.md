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

The farm recorded (2026-08-23) that 0.18.0 produced a pure-python wheel on
every Windows cell, because its only CUDA sources are CUTLASS-based and
`setup.py` gates those on `not IS_WINDOWS`. 0.13.0 still has non-CUTLASS
CUDA sources that would reach nvcc on Windows, so the cell is dispatched
rather than assumed dead; `activation24/*.cu` include CUTLASS headers without
"cutlass" in their names and are the likely first failure. The outcome is
recorded in the batch report, not guessed here.

## Patch

`patches/torchao.py`: stops `test/` shipping as a top-level package (it
collides with CPython's own stdlib `test`).
