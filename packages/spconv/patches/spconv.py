"""Patch spconv v2.3.8: bf16 sparse convolution.

Ported from cuda-wheels' packages/spconv/patches/spconv.py and re-anchored
against v2.3.8. Run by scripts/fetch_patched_sources.py with cwd set to the
unpacked source, BEFORE the tarball is sealed; every substitution asserts on
its own.

Adds bf16 GEMM shuffle params (Ampere TensorOp + Simt fallback) and bf16
implicit-GEMM conv params (3 FwdAndBwdInput + 2 BwdWeight) to spconv/core.py,
and the torch.bfloat16 -> tv.bfloat16 mapping to pytorch/cppcore.py. The
generated kernels are compiled AOT into core_cc like every other entry in
those lists, so a bf16 model no longer hits "no kernel for dtype". spconv's
gen_shuffle_params is gen_shuffle_params_v2 (an extra ds_for_sab argument
compared to cumm's), which is why the entries differ from cumm's.

NOT ported, with the reasons:
  * the CUMM_CUDA_VERSION rename in setup.py: the variable is not set here,
    so RELEASE_NAME is already `spconv` and the dep is already
    `cumm>=0.7.11, <0.8.0`; the arch list arrives as CUMM_CUDA_ARCH_LIST;
  * the installed-ccimport symlink fix: it edited a file in the BUILD
    environment (ccimport's loader.py), which fetch-time patching cannot
    reach, and it addressed the manylinux container's /opt/python symlink
    farm -- rattler-build's host prefix has no such indirection.
"""
import ast
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require  # noqa: E402

core_py = pathlib.Path("spconv/core.py")
content = core_py.read_text(encoding="utf-8")

# ── 1a. bf16 Simt fallbacks in SHUFFLE_SIMT_PARAMS ─────────────────────────
BF16_SIMT_FALLBACK = '''
    # cuda-foundry (packages/spconv/patches): bf16 Simt fallback kernels for
    # misaligned dimensions (TensorOp needs LDA aligned to 8 elements)
    *gen_shuffle_params((128, 128, 8), (32, 64, 8), ["bf16,bf16,bf16,f32,f32"],
                        "bf16,bf16,bf16,f32,f32", 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params((32, 64, 32), (32, 32, 8), ["bf16,bf16,bf16,f32,f32"],
                        "bf16,bf16,bf16,f32,f32", 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params((32, 32, 32), (32, 32, 8), ["bf16,bf16,bf16,f32,f32"],
                        "bf16,bf16,bf16,f32,f32", 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params((64, 128, 16), (32, 64, 8), ["bf16,bf16,bf16,f32,f32"],
                        "bf16,bf16,bf16,f32,f32", 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params((64, 64, 8), (32, 32, 8), ["bf16,bf16,bf16,f32,f32"],
                        "bf16,bf16,bf16,f32,f32", 2, kernel.GemmAlgo.Simt, None),
'''
simt = re.search(r"(SHUFFLE_SIMT_PARAMS\s*:.*?\n(?:.*\n)*?)(^\]\s*$)", content, re.MULTILINE)
require(simt is not None, "spconv: SHUFFLE_SIMT_PARAMS closing bracket not found")
content = content[:simt.start(2)] + BF16_SIMT_FALLBACK + content[simt.start(2):]

# ── 1b. SHUFFLE_AMPERE_PARAMS: the commented-out s8 block becomes bf16 ─────
BF16_SHUFFLE_PARAMS = '''SHUFFLE_AMPERE_PARAMS: List[GemmAlgoParams] = [
    # cuda-foundry (packages/spconv/patches): bf16 with f32 accumulator,
    # Ampere TensorOp (16, 8, 16)
    *gen_shuffle_params(
        (64, 64, 32),
        (32, 32, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 128, 32),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 128, 32),
        (64, 32, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 256, 32),
        (64, 64, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (256, 128, 32),
        (64, 64, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 64, 32),
        (64, 32, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (64, 128, 32),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
]'''
content, n = re.subn(r"^SHUFFLE_AMPERE_PARAMS\s*(?::\s*List\[GemmAlgoParams\]\s*)?=\s*\[.*?^\]",
                     BF16_SHUFFLE_PARAMS, content, count=1, flags=re.DOTALL | re.MULTILINE)
require(n == 1, "spconv: SHUFFLE_AMPERE_PARAMS block not found in spconv/core.py")

# ── 2. bf16 implicit-GEMM conv params ──────────────────────────────────────
BF16_CONV_PARAMS = '''
    # cuda-foundry (packages/spconv/patches): bf16 FwdAndBwdInput, Ampere
    # TensorOp (16, 8, 16)
    *gen_conv_params(ConvFwdAndBwdInput, (64, 64, 32), (32, 32, 32),
                     NDIM_DONT_CARE,
                     ConvIterAlgo.Optimized,
                     [2, 3, 4], ["bf16,bf16,bf16,f32,f32"],
                     NHWC, NHWC, NHWC,
                     GemmAlgo.Ampere,
                     TensorOp((16, 8, 16)),
                     mask_sparse=True,
                     increment_k_first=True,
                     access_per_vector=1),
    *gen_conv_params(ConvFwdAndBwdInput, (64, 128, 32), (32, 64, 32),
                     NDIM_DONT_CARE,
                     ConvIterAlgo.Optimized,
                     [2, 3, 4], ["bf16,bf16,bf16,f32,f32"],
                     NHWC, NHWC, NHWC,
                     GemmAlgo.Ampere,
                     TensorOp((16, 8, 16)),
                     mask_sparse=True,
                     increment_k_first=True,
                     access_per_vector=1),
    *gen_conv_params(ConvFwdAndBwdInput, (128, 64, 32), (64, 32, 32),
                     NDIM_DONT_CARE,
                     ConvIterAlgo.Optimized,
                     [2, 3, 4], ["bf16,bf16,bf16,f32,f32"],
                     NHWC, NHWC, NHWC,
                     GemmAlgo.Ampere,
                     TensorOp((16, 8, 16)),
                     mask_sparse=True,
                     increment_k_first=True,
                     access_per_vector=1),
    # bf16 BwdWeight, Ampere TensorOp (16, 8, 16)
    *gen_conv_params(ConvBwdWeight, (64, 64, 32), (32, 32, 32),
                     NDIM_DONT_CARE,
                     ConvIterAlgo.Optimized,
                     [2, 3, 4, 5], ["bf16,bf16,bf16,f32,f32"],
                     NHWC, NHWC, NHWC,
                     GemmAlgo.Ampere,
                     TensorOp((16, 8, 16)),
                     mask_sparse=True,
                     increment_k_first=True,
                     access_per_vector=1),
    *gen_conv_params(ConvBwdWeight, (64, 128, 32), (32, 64, 32),
                     NDIM_DONT_CARE,
                     ConvIterAlgo.Optimized,
                     [2, 3, 4, 5], ["bf16,bf16,bf16,f32,f32"],
                     NHWC, NHWC, NHWC,
                     GemmAlgo.Ampere,
                     TensorOp((16, 8, 16)),
                     mask_sparse=True,
                     increment_k_first=True,
                     access_per_vector=1),
'''
# The list's closing bracket is the line before the int8-debug extension
# block, which is the one place in the file that sequence occurs.
end_anchor = "]\n\nif not SPCONV_INT8_DEBUG:\n    IMPLGEMM_AMPERE_PARAMS.extend(["
require(content.count(end_anchor) == 1,
        "spconv: the end of IMPLGEMM_AMPERE_PARAMS was not found where v2.3.8 has it")
start = content.index("IMPLGEMM_AMPERE_PARAMS = [")
end = content.index(end_anchor)
require(start < end, "spconv: IMPLGEMM_AMPERE_PARAMS anchors are out of order")
content = content[:end] + BF16_CONV_PARAMS.rstrip("\n") + "\n" + content[end:]

core_py.write_text(content, encoding="utf-8")
final = core_py.read_text(encoding="utf-8")
ast.parse(final)
require(final.count("bf16,bf16,bf16,f32,f32") >= 5 + 7 * 2 + 5,
        "spconv: fewer bf16 entries on disk than were inserted")
print("spconv patch: bf16 Simt fallback (5), SHUFFLE_AMPERE (7), IMPLGEMM_AMPERE (3+2) -> spconv/core.py")

# ── 3. torch.bfloat16 -> tv.bfloat16 ───────────────────────────────────────
cppcore = pathlib.Path("spconv/pytorch/cppcore.py")
c = cppcore.read_text(encoding="utf-8")
old = "    torch.float16: tv.float16,\n"
require(c.count(old) == 1 and "torch.bfloat16" not in c,
        "spconv: the _TORCH_DTYPE_TO_TV float16 entry was not found once, or bfloat16 "
        "is already mapped")
cppcore.write_text(c.replace(old, old + "    torch.bfloat16: tv.bfloat16,\n", 1), encoding="utf-8")
require("torch.bfloat16: tv.bfloat16," in cppcore.read_text(encoding="utf-8"),
        "spconv: bf16 dtype mapping NOT on disk")
print("spconv patch: torch.bfloat16 -> tv.bfloat16 in pytorch/cppcore.py")
