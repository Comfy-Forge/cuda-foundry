# sageattention — build notes

- **Source:** thu-ml/SageAttention v2.2.0 (`eb615cf6`), the repo root.
  `sageattention3_blackwell/` in the same repo is a separate package here
  (`packages/sageattn3`).
- **Modules:** `_qattn_sm80` (Ampere INT8 QK / FP16 PV), `_qattn_sm89` (Ada
  FP8 PV), `_qattn_sm90` (Hopper, TMA, links the driver), `_fused`
  (quantisation). The Triton kernels in `sageattention/triton/` are pure
  Python and need the triton that ships with torch.

## Overrides

- **`arch_override.yml`** — narrows the shared arch policy to what
  `setup.py` can emit: `SUPPORTED_ARCHS = {8.0, 8.6, 8.9, 9.0, 12.0}` with a
  silent `continue` for everything else. The shared cu12.8 row's sm_70,
  sm_75 and sm_100 cannot be produced by this source, and the SASS census
  would fail every artifact for their absence. The farm kept those archs in
  its rows and waived them in the gate instead ("waive, do not narrow"); this
  repo has no waiver mechanism and a narrowed list that says what the
  artifact contains is the honest form available. Blackwell datacentre
  (sm_100) is an upstream gap: no kernel, no dispatch branch, and a clean
  `ValueError("Unsupported CUDA architecture")` from `core.py` on a B200.

## Per-module arch lists

Two of the four modules deliberately carry a subset of the cell's list:

- `_qattn_sm90` is compiled for `sm_90a` only. Its kernels use wgmma, and
  ptxas refuses `wgmma.mma_async` on any other target, so the global gencode
  list would fail the build outright.
- `_qattn_sm89` is compiled for sm_89 and up. Its FP8 QMMA path is gated on
  `__CUDA_ARCH__ >= 890` and compiles to `__brkpt()` below that: full-size
  cubins made of traps that nothing can ever dispatch to (core.py routes only
  `sm89` there). They were pure compile cost and wheel weight.

The publish gate's SASS census therefore reads the **union** over every
shipped module (tools/verify_conda.py, tools/verify_wheel.py); it used to
read the largest module alone, and `_qattn_sm89` is the largest.

## The driver link

`_qattn_sm90` calls `cuTensorMapEncodeTiled` and is linked with `-lcuda`.
torch's own CUDA libraries never carry `libcuda.so.1` in their `DT_NEEDED`
(measured on 2.8.0+cu128), so this extension does, and linking it needs the
driver stub: `cuda-driver-dev` on linux-64 (`host_deps_linux`) with
`-L$PREFIX/lib/stubs` on the link line (`build_env`), and nothing on win-64,
where `cuda.lib` ships in `cuda-cudart-dev_win-64`. The wheel never vendors
the driver and a driverless box fails at `import sageattention` rather than
degrading, which is upstream's behaviour too.
