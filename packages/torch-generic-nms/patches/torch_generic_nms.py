"""Patch torch_generic_nms: keep the cell's TORCH_CUDA_ARCH_LIST authoritative
on aarch64.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed -- so no `import torch`, and no platform
conditionals (one tarball serves every platform's build). The fix here is
inert off aarch64 and is safe to apply to the single shared tarball.

Upstream get_extensions() does, verbatim (identical to its sibling cc_torch,
same author):

    CC = os.environ.get("CC", None)
    if CC is not None:
        extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

That append is the whole bug on linux-aarch64. torch.utils.cpp_extension's
_get_cuda_arch_flags returns [] -- i.e. emits NO -gencode and lets nvcc fall
back to its own default arch -- the instant ANY user nvcc flag contains the
substring "arch" (measured; the same quirk fused-ssim's patch documents). On
linux-aarch64 CC is "aarch64-conda-linux-gnu-cc", whose "aarch64" CONTAINS
"arch", so torch silently drops the cell's TORCH_CUDA_ARCH_LIST and nvcc
compiles generic_nms.cu for its default architecture alone -- sm_75 under
CUDA 13 -- and the artifact fails verify_conda's SASS census ("want
['80','87','90','100','110','120'], got ['75']", run 34730199797).

On linux-64 CC is "x86_64-conda-linux-gnu-cc" (no "arch" substring), which is
why every linux-64 torch-generic-nms artifact built correctly.

The append is also redundant on EVERY platform: the conda build environment
already hands nvcc the pinned host compiler through NVCC_PREPEND_FLAGS
(-ccbin=$CXX, verified in the build env), so nothing is lost by dropping it.
Removing it is the whole fix -- torch then sees no "arch"-bearing flag and
compiles for the cell's TORCH_CUDA_ARCH_LIST.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require  # noqa: E402

MARKER = "# cuda-foundry: -ccbin={CC} append dropped"

OLD = (
    '    CC = os.environ.get("CC", None)\n'
    "    if CC is not None:\n"
    '        extra_compile_args["nvcc"].append("-ccbin={}".format(CC))\n'
)
NEW = (
    "    # cuda-foundry: -ccbin={CC} append dropped -- on linux-aarch64 CC is\n"
    '    # "aarch64-conda-linux-gnu-cc", whose "aarch64" contains the substring\n'
    '    # "arch", and torch\'s _get_cuda_arch_flags returns [] (no -gencode) as\n'
    '    # soon as any nvcc flag contains "arch", dropping the cell\'s\n'
    "    # TORCH_CUDA_ARCH_LIST so nvcc built sm_75 alone (fails the SASS census).\n"
    "    # The conda env already passes -ccbin=$CXX via NVCC_PREPEND_FLAGS, so\n"
    "    # this append was redundant on every platform. See fused-ssim's patch.\n"
)

setup_py = pathlib.Path("setup.py")
text = setup_py.read_text(encoding="utf-8")
if MARKER in text:
    print("torch_generic_nms patch: already applied")
    sys.exit(0)

require(
    text.count(OLD) == 1,
    "torch_generic_nms: the -ccbin={CC} append block matched "
    f"{text.count(OLD)} time(s) in setup.py, expected exactly 1 -- upstream "
    "changed at this pinned rev; re-read it and update the patch",
)
text = text.replace(OLD, MARKER + "\n" + NEW, 1)
setup_py.write_text(text, encoding="utf-8")

final = setup_py.read_text(encoding="utf-8")
require(
    '.append("-ccbin=' not in final,
    "torch_generic_nms: the -ccbin append still survives in setup.py on disk",
)
import ast  # noqa: E402

ast.parse(final)
print("torch_generic_nms patch: -ccbin={CC} append removed; "
      "TORCH_CUDA_ARCH_LIST authoritative")
