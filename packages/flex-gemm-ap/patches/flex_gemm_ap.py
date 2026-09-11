"""Patch FlexGEMM-ap (PozzettiAndrea, 3432bc7) for the foundry build.

Ported from cuda-wheels (packages/flexgemm_ap/patches/flexgemm_ap.py) and
re-read against the pinned rev. The ap fork already spells its dependent
member-template calls with `.template` (the MSVC fix vanilla FlexGEMM needs),
so only two things remain, both fetch-time and cell-independent:

1. Hardcoded C++ standard: setup.py pins -std=c++17 / /std:c++17 in every
   flag list. torch's cpp_extension appends the standard the installed torch
   needs only when the caller gave none, so the pin is stripped.

2. triton Autotuner subclass: the fork's utils/autotuner.py forwards
   positional arguments to triton.runtime.Autotuner.__init__; the shared
   helper filters that by the installed triton's signature. A no-op where
   the fork has no such forward (it fails loud only on an unrecognised one).

See packages/flex-gemm/patches/flex_gemm.py for the reasoning behind each.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import fix_triton_autotuner_super_auto, require, strip_std_flags  # noqa: E402

PKG = "flex_gemm_ap"
require(pathlib.Path(f"{PKG}/kernels/cuda/ext.cpp").is_file(),
        f"{PKG}: the renamed package directory is not where the pinned rev keeps it")

setup_file = pathlib.Path("setup.py")
text = setup_file.read_text()
text, n_std = strip_std_flags(text)
require(n_std > 0, f"{PKG}: no hardcoded C++-standard flag in setup.py -- "
                   f"upstream changed; refusing to build against an unverified flag set")
setup_file.write_text(text)
require(not re.search(r"""['"](?:-Xcompiler=)?[-/]std[=:]c\+\+\d+['"]""", setup_file.read_text()),
        f"{PKG}: a C++-standard flag survived stripping")
print(f"{PKG} patch: dropped {n_std} hardcoded std flag(s); torch now selects it")

n_at = fix_triton_autotuner_super_auto(".")
print(f"{PKG} patch: triton Autotuner handling done ({n_at} file(s) rewritten)")
