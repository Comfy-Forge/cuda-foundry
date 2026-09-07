# flash_attn — build notes
- **Source:** Dao-AILab/flash-attention (v2.8.3)
- **Quirks:** SM >= 8.0 (Ampere+). Patch inits only csrc/cutlass submodule (skips composable_kernel — ROCm only, breaks Windows paths). Bridges TORCH_CUDA_ARCH_LIST -> FLASH_ATTN_CUDA_ARCHS.
- **Arch list:** cu124-cu126 use default `"8.0 9.0"`. cu128+ use `"8.0 9.0 10.0 12.0"` (Blackwell).

**CUDA-version-specific behavior:** at the pinned tag **v2.8.3**, setup.py:180-191
emits plain `sm_80/90/100/120` cubins and nothing else -- no PTX, no family-specific
`100f`/`120f`, no SM 101/Thor. (The previous text here described upstream `main`, not
the tag we build; corrected 2026-08-25 after an NVIDIA review.) Upstream gained
forward-compat PTX in 7bdb426 / PR #1882 and the family-specific gencodes later; we
backport only the PTX line in patches/flash_attn.py. Bumping source_tag past 7bdb426
would inherit both and let most of the local gencode patch be deleted -- a version
policy call, not a build one.

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list_by_cuda`. 


## Overrides

- **`arch_override.yml`** — overrides the shared arch policy's lists per CUDA
  line (`arch_list_by_cuda`), ported verbatim from cuda-wheels. flash-attn
  supports SM >= 8.0 only, so the shared policy's 5.0/6.0/7.x rows are wrong
  for it.

## From-source

`package.yml` declares `force_source_build: {FLASH_ATTENTION_FORCE_BUILD:
"TRUE"}`. Upstream compares that env var against the literal string `"TRUE"`,
so `"1"` is silently permissive — see the comment in `package.yml` for the
full history, including the earlier experiment that published upstream's
prebuilt binary while believing it had compiled it.
