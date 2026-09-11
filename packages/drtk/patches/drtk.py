"""Patch DRTK: MSVC CRT/RTTI flags, half-operator macros, the C++ standard,
and setup.py's pkg_resources import.

Runs on the fetching machine with cwd = the checkout, before the tarball is
sealed; no platform conditionals (the MSVC edits are to the `win32` flag list
setup.py already selects by platform at build time), no `import torch`.

Ported from cuda-wheels, every substitution asserted on its own hit count:

1. Remove /GR- from the win32 cxx flags -- it disables RTTI, and PyTorch's
   headers need it (dynamic_cast / dynamic_pointer_cast -> C2280 without it).
2. /MT -> /MD. DRTK asks for the static CRT, nvcc compiles the .cu TUs
   against the dynamic one, and the mix is LNK2038 at link.
3. Delete both hardcoded C++-standard flags (linux cxx "-std=c++17" and
   nvcc_args.append("-std=c++20")) and let torch's cpp_extension pick the
   standard the installed torch needs. The c++20 carries an upstream comment
   pointing at pytorch/pytorch#122169, which is a torch-source build issue on
   torch 2.1/2.2 with CUDA 12.4 and says nothing about extensions; DRTK's own
   kernels use no C++20 (no designated initialisers, concepts, span, <=>,
   bit-field NSDMIs anywhere under src/ -- checked, not assumed). Compiling
   c10/ATen headers at C++20 against a libtorch built at C++17 (pytorch's
   CMAKE_CXX_STANDARD through 2.11) is one ODR-mismatched TU per extension
   for no gain.
4. Re-enable the half/bfloat16 operators. torch's cpp_extension puts
   -D__CUDA_NO_HALF_OPERATORS__ (and friends) on the nvcc line, which breaks
   the CUB headers DRTK includes (dispatch_histogram.cuh,
   agent_sub_warp_merge_sort.cuh). An #undef at the top of each .cu overrides
   the -D.

One addition over the farm's patch:

5. setup.py does `from pkg_resources import DistributionNotFound,
   get_distribution` -- only to ask whether pillow-simd is installed, to pick
   which pillow to declare. pkg_resources is gone from recent setuptools
   (the farm answered with `setuptools~=80.0`; this repo's torchvision pins
   `<82` for the same import). Pinning a build tool to keep a dead import
   alive is the wrong direction; the two lines are rewritten onto
   importlib.metadata, which has been in the standard library since 3.8 and
   answers the same question. The declaration it feeds is not even used --
   make_wheel.py replaces every Requires-Dist with package.yml's run_deps --
   but leaving setup.py unable to import would stop the build before that.

DRTK's arch handling is left alone and asserted: its hardcoded -gencode list
is guarded by `if not os.getenv("TORCH_CUDA_ARCH_LIST")`, and the recipe
always sets that variable, so torch's list is authoritative unpatched.
"""

import pathlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require, strip_std_flags  # noqa: E402

MARKER = "# patched-by: cuda-foundry drtk"
setup_file = Path("setup.py")
content = setup_file.read_text(encoding="utf-8")

if MARKER in content:
    print("drtk patch: setup.py already patched")
else:
    # 1. RTTI
    content, n_gr = re.subn(r'"/GR-",\s*|,\s*"/GR-"', "", content)
    require(n_gr == 1,
            f'expected exactly one "/GR-" element in drtk setup.py, found {n_gr} '
            f'-- upstream changed the win32 cxx flag list; without the removal '
            f'every Windows TU that touches a torch header dies with C2280')
    print("drtk patch: removed /GR- (RTTI required by PyTorch)")

    # 2. CRT
    n_mt = content.count('"/MT"')
    require(n_mt == 1,
            f'expected exactly one "/MT" element in drtk setup.py, found {n_mt} '
            f'-- upstream changed the win32 cxx flag list; the CRT would no '
            f'longer match nvcc\'s /MD and the link would fail with LNK2038')
    content = content.replace('"/MT"', '"/MD"')
    print("drtk patch: /MT -> /MD (CRT must match nvcc's /MD)")

    # 3. the C++ standard is torch's call
    content, n_std = strip_std_flags(content)
    require(n_std == 2,
            f"expected 2 hardcoded C++-standard flags in drtk setup.py (linux "
            f"cxx -std=c++17 and nvcc_args.append('-std=c++20')), found {n_std} "
            f"-- upstream changed; refusing to build against an unverified flag set")
    print(f"drtk patch: dropped {n_std} hardcoded C++-standard flag(s)")

    # 5. pkg_resources -> importlib.metadata
    old_import = "from pkg_resources import DistributionNotFound, get_distribution\n"
    require(content.count(old_import) == 1,
            "expected exactly one `from pkg_resources import DistributionNotFound, "
            "get_distribution` in drtk setup.py -- upstream changed; re-read it")
    content = content.replace(
        old_import,
        "from importlib.metadata import PackageNotFoundError as DistributionNotFound\n"
        "from importlib.metadata import distribution as get_distribution\n")
    require("pkg_resources" not in content,
            "setup.py still mentions pkg_resources after the rewrite")
    print("drtk patch: setup.py no longer imports pkg_resources")

    # The arch guard this package is trusted for.
    require('if not os.getenv("TORCH_CUDA_ARCH_LIST"):' in content,
            "drtk setup.py no longer guards its hardcoded -gencode list on "
            "TORCH_CUDA_ARCH_LIST being unset -- upstream changed; the cell's "
            "arch list would be silently discarded")

    setup_file.write_text(MARKER + "\n" + content, encoding="utf-8")

# 4. half/bfloat16 operators
UNDEF_BLOCK = (
    "// -- cuda-foundry patch: re-enable half/bfloat16 operators --\n"
    "#undef __CUDA_NO_HALF_OPERATORS__\n"
    "#undef __CUDA_NO_HALF2_OPERATORS__\n"
    "#undef __CUDA_NO_HALF_CONVERSIONS__\n"
    "#undef __CUDA_NO_BFLOAT16_CONVERSIONS__\n"
    "// -- end patch --\n\n"
)
cu_files = sorted(Path("src").rglob("*.cu"))
require(len(cu_files) > 0,
        "no src/**/*.cu found in the drtk tree -- the source layout changed; "
        "the half-operator #undef block would silently apply to nothing")
patched_cu = already = 0
for cu_file in cu_files:
    cu_content = cu_file.read_text(encoding="utf-8")
    if cu_content.startswith(UNDEF_BLOCK):
        already += 1
        continue
    require("__CUDA_NO_HALF_OPERATORS__" not in cu_content,
            f"{cu_file} already mentions __CUDA_NO_HALF_OPERATORS__ in a form "
            f"this patch did not write -- check whether upstream now handles "
            f"this itself")
    cu_file.write_text(UNDEF_BLOCK + cu_content, encoding="utf-8")
    patched_cu += 1
require(patched_cu + already == len(cu_files),
        f"only {patched_cu + already}/{len(cu_files)} .cu files carry the "
        f"half/bfloat16 #undef block")
print(f"drtk patch: {patched_cu} .cu file(s) got the half/bfloat16 #undef block "
      f"({already} already had it)")
print("drtk patch: done")
