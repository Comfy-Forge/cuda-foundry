# nunchaku — build notes

- **Source:** nunchaku-ai/nunchaku v1.2.1 (`af49f249`), recursive (cutlass,
  json, mio, spdlog, Block-Sparse-Attention submodules).
- **Module:** one extension, `nunchaku._C`, ~25 CUDA translation units
  including eight Block-Sparse-Attention flash kernels.
- **Windows:** upstream's setup.py carries its own MSVC flag lists
  (`/std:c++20`, `/Zc:__cplusplus`, `/FS`, `/bigobj`); nothing platform-specific
  is patched.

## Overrides

- **`arch_override.yml`** — documents reality rather than steering it:
  `setup.py` ignores `TORCH_CUDA_ARCH_LIST` and derives its targets from
  `NUNCHAKU_INSTALL_MODE=ALL` (sm_75/80/86/89 plus sm_120a on CUDA >= 12.8,
  sm_121a on >= 13.0). The shared policy row asks for sm_70/90/100 that no
  build of this source produces. The PTX tail is `compute_89`, added by the
  patch on the highest plain target, since sm_120a is arch-conditional and
  cannot carry a portable one.

## Environment switches (`build_env`)

`NUNCHAKU_INSTALL_MODE=ALL` (the default FAST mode probes local GPUs and
asserts on a GPU-less runner) and `NUNCHAKU_BUILD_WHEELS=1` (upstream's
release switch; the default developer build adds `--generate-line-info`).

## Patch

`patches/nunchaku.py`: strips upstream's own `+cu12.8torch2.8` local version
tag (the publishing side adds `+cu128torch2.8`; two local tags is not
PEP 440), drops the unconditional debug flags `-g` / `-Og` / `-UNDEBUG` that
a release artifact should not carry, and adds the `compute_89` PTX tail.
