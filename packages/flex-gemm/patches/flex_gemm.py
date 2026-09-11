"""Patch FlexGEMM (JeffreyXiang, 6dd94a8) for the foundry build.

Ported from cuda-wheels (packages/flexgemm/patches/flexgemm.py) and re-read
against the pinned rev. Runs with cwd = the cloned source, once, before the
tarball is sealed, so nothing here may depend on the cell or the host: the one
Windows fix is a source-level spelling that is valid on every compiler.

1. MSVC: dependent member-template calls need `.template`. hash.cu and
   sparse_neighbor_map.cu call `x.data_ptr<T>()` / `x.item<T>()` on objects
   whose type is deduced inside generic lambdas, which makes the name
   dependent; GCC/Clang accept the shorthand, MSVC rejects it ("type name is
   not allowed"). `.template data_ptr<T>()` is the conforming spelling and is
   well-formed everywhere, so it is applied unconditionally. The farm widened
   the match to ANY single type argument after `<int64_t>`/`<int32_t>` sites
   were missed -- what makes a call dependent is the object, not the
   argument. 41 sites at this rev.

2. Hardcoded C++ standard: setup.py pins -std=c++17 / /std:c++17 in every
   flag list. torch's cpp_extension appends the standard the installed torch
   needs only when the caller gave none, so the pin is stripped and torch
   decides (c++17 for torch 2.8; whatever the next torch asks for later).

3. triton Autotuner subclass: flex_gemm/utils/autotuner.py forwards 13
   positional arguments to triton.runtime.Autotuner.__init__, which older
   tritons (3.0/3.1, pinned by torch 2.4/2.5) do not accept. The helper
   filters the forward by the installed triton's signature at import time;
   a no-op on triton 3.4 (torch 2.8), a fix on the older lines.

NOT carried: the farm's pyproject.toml rewrite of the triton dependency into
platform-marked triton / triton-windows entries. Upstream's Requires-Dist
never reaches an artifact here -- tools/make_wheel.py strips it and writes
package.yml's run_deps into the sidecar -- so that edit would change nothing.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import fix_triton_autotuner_super_auto, require, strip_std_flags  # noqa: E402

PKG = "flex_gemm"
MSVC_FILES = [
    pathlib.Path(f"{PKG}/kernels/cuda/hash/hash.cu"),
    pathlib.Path(f"{PKG}/kernels/cuda/spconv/sparse_neighbor_map.cu"),
]
DEPENDENT_CALL = re.compile(r"(?<!template )\.(?!template )(data_ptr|item)<([A-Za-z_][\w:]*)>")

fixed = 0
for f in MSVC_FILES:
    require(f.is_file(), f"{PKG}: expected source {f} is missing")
    text = f.read_text()
    new, n = DEPENDENT_CALL.subn(r".template \1<\2>", text)
    if n:
        f.write_text(new)
        print(f"{PKG} patch: {f}: {n} dependent member call(s) got `.template`")
        fixed += n
require(fixed > 0 or all("template data_ptr" in f.read_text() for f in MSVC_FILES),
        f"{PKG}: no data_ptr<T>/item<T> calls found -- upstream changed; re-read")

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
