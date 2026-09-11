# spconv — build notes
- **Source:** traveller59/spconv (v2.3.8, commit 263d6b47).
- **Build:** pccm/ccimport, driven by upstream's `setup.py` with
  `SPCONV_DISABLE_JIT=1`. The generator imports the `cumm` installed in the
  host env — **our own cumm .conda**, from this channel, flavour-locked with
  a `cuda<NNN>_*` build glob — and emits several hundred `.cu` translation
  units that nvcc compiles through the seat wrapper; every listed gencode is
  baked into `core_cc` (this is the package whose SASS census is meaningful).
- **Torch-free** (`links_torch: false`): `core_cc` carries no DT_NEEDED on
  libtorch; torch is imported by `spconv.pytorch` only, at the consumer's
  choice.
- **Patch (`patches/spconv.py`):** bf16 sparse convolution (Simt fallback +
  Ampere TensorOp shuffle params + implicit-GEMM conv params) and the
  `torch.bfloat16` dtype mapping.
- **Unsharded** (farm owner order 2026-08-26): the seat wrapper's stub
  objects leave symbols undefined that spconv's link step needs, and the
  package fits in one job.

## Overrides

- **`arch_override.yml`** — overrides the shared arch policy's lists per CUDA
  line (`arch_list_by_cuda`), ported from cuda-wheels. See the file's own
  comment for the Blackwell regression it corrects and the ordering with
  cumm it depends on.
