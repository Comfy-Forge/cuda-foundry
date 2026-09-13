"""Patch cumm v0.7.11: Blackwell in the arch table, bf16 GEMM params, three
runtime fixes, no exact-minor nvrtc-builtins link, the CUDA headers found
in $CONDA_PREFIX at run time (section 8) and the wheel's License field made
to match the Apache-2.0 LICENSE file (section 9).

Ported from cuda-wheels' packages/cumm/patches/cumm.py, re-checked against
v0.7.11 rather than assumed (that file was written for 0.8.2 and half of it
no-ops at this rev). Run by scripts/fetch_patched_sources.py with cwd set to
the unpacked source, BEFORE the tarball is sealed: no CUDA toolkit here, no
platform conditionals, and every substitution asserted on its own.

Ported:
  1. supported_arches += 10.0, 11.0, 12.0 (cumm/common.py). 0.7.11's table
     stops at 9.0 and get_cuda_arch_flags() raises "Unknown CUDA arch" on
     anything else, which is how the farm's spconv lost its Blackwell SASS
     for a while. 0.8.2 added exactly these tokens.
  2. bf16 GEMM params in cumm/gemm/main.py: SHUFFLE_AMPERE_PARAMS (nine
     TensorOp tile configs, f32 accumulate), five bf16 Simt fallbacks for
     unaligned channel counts, and the empty ampere_params list of
     GemmMainUnitTest.__init__ populated from it.
  3. tensorview_bind.py: zero_whole_storage_ gets its default Context arg,
     matching the C++ signature; spconv's algo.py calls it without one.
  4. constants.py: TENSORVIEW_INCLUDE_PATH is chosen by the presence of
     tensorview/core/all.h, not of an `include/` directory -- other packages
     (Eigen, embreex) create site-packages/include/ and cumm would pick it.
  5. cumm/nvrtc/__init__.py: an inline Itanium demangler ahead of the
     cu++filt subprocess, which does not exist on Windows.
  6. nvrtc/limits.h (new here): numeric_limits<float/double>::infinity /
     quiet_NaN / signaling_NaN are written with the GCC builtins
     __builtin_huge_valf, __builtin_nanf, __builtin_nansf and their double
     twins. NVRTC does not define them -- measured against libnvrtc 12.8.93
     and 12.9.86, under -std=c++14 and c++17 alike: `identifier
     "__builtin_huge_valf" is undefined` -- and limits.h is included by
     every NVRTC kernel through nvrtc_std.h, so cumm 0.7.11's whole run-time
     JIT path fails to compile on CUDA 12.x. libcudacxx's <cuda/std/limits>
     is available under NVRTC (the file already includes cuda/std/cfloat
     beside it), so the six calls become ::cuda::std::numeric_limits<T>::
     infinity()/quiet_NaN()/signaling_NaN(). With that, the verify op's
     NVRTC kernel compiles, loads and runs.

NOT ported -- the farm's std::abs/min/max "shims" for nvrtc_std.h. They were
guarded on "std::abs" being absent from nvrtc_std.h, which is the wrong file:
at 0.7.11 nvrtc/core.h already declares constexpr std::abs/min/max, so the
shim REDEFINES them and every NVRTC compile dies with "function template
std::abs has already been defined" (measured). Asserted absent instead.

NOT ported, with the reason:
  * populating third_party/cccl from $CUDA_HOME -- setup.py at 0.7.11 never
    reads that directory (no cccl reference anywhere), so the farm's copy of
    every CUDA header into the tarball was dead weight;
  * the CUMM_CUDA_VERSION rename in setup.py -- the variable is simply not
    set here, so RELEASE_NAME is already `cumm`; the arch list arrives as
    CUMM_CUDA_ARCH_LIST (package.yml build_env), which needs no version;
  * the dtype-header namespace rewrite and the common.py cudart fallback --
    their anchors do not exist at 0.7.11 (asserted below rather than
    silently skipped).

── 7. no exact-minor nvrtc-builtins link (new here) ───────────────────────
CummNVRTCLink adds, on Linux, `-Wl,--no-as-needed nvrtc-builtins` so that
core_cc records a DT_NEEDED on libnvrtc-builtins.so.12.8 -- an EXACT-MINOR
soname (libnvrtc.so.12 by contrast is major-only). That is poison for the
.conda: an env whose torch pulls cuda-nvrtc 12.9 (conda-torch's cu128
triton pins cuda-version 12.9, measured in a live solve) has
libnvrtc-builtins.so.12.9 and no .12.8, so the artifact would fail to load
beside the very torch it is meant for, and a `cuda-nvrtc 12.8.*` run dep
would make it UNSAT instead. libnvrtc needs no help finding its builtins in
a conda env -- it dlopens "libnvrtc-builtins.so.<major.minor>" through its
own $ORIGIN RPATH, and both files ship in one cuda-nvrtc package (measured:
a lone copy of libnvrtc.so.12 with builtins beside it compiles; without
them it fails with "failed to open libnvrtc-builtins.so.12.9"). The WHEEL
does need them carried, because auditwheel follows DT_NEEDED only; that is
package.yml's `wheel_vendor_extra`, which copies the builtins beside the
vendored libnvrtc and gives it an $ORIGIN RPATH. `add_libraries("nvrtc")`
is kept: libnvrtc.so.12 is the real dependency.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require  # noqa: E402


def sub_once(path: pathlib.Path, old: str, new: str, what: str) -> None:
    text = path.read_text(encoding="utf-8")
    n = text.count(old)
    require(n == 1, f"cumm: expected exactly one {what} anchor in {path}, found {n} "
                    f"-- upstream changed; re-check against v0.7.11")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    require(new in path.read_text(encoding="utf-8"), f"cumm: {what} NOT on disk")
    print(f"cumm patch: {what} -> {path}")


# ── 1. Blackwell / Thor / Jetson in the arch table ────────────────────────
# 0.7.11's table stops at 9.0 and get_cuda_arch_flags() raises "Unknown CUDA
# arch" on anything else. 0.8.2 added 10.0/11.0/12.0 (Blackwell + Thor).
# 8.7 (Jetson Orin) is added on top of that: it is absent from every x86 row
# but present in arch_policy_aarch64 for every cu line, so a linux-aarch64
# build -- which is governed by that policy, not by this package's x86
# arch_list_by_cuda -- passes "8.7" through cumm's table (and spconv reads
# the same table through cumm.get_cuda_arch_flags). nvcc has accepted sm_87
# since CUDA 11.4; it is inert on x86, where no row ever requests it.
sub_once(pathlib.Path("cumm/common.py"),
         "        '8.6', '8.9', '9.0'\n    ]",
         "        '8.6', '8.7', '8.9', '9.0', '10.0', '11.0', '12.0'\n    ]",
         "supported_arches table")

# ── 2. bf16 GEMM params ───────────────────────────────────────────────────
main_py = pathlib.Path("cumm/gemm/main.py")
content = main_py.read_text(encoding="utf-8")

AMPERE_PARAMS_BLOCK = '''
# cuda-foundry (packages/cumm/patches): bf16 Ampere GEMM kernels (sm_80+).
# TensorOp((16, 8, 16)) is the Ampere shape for 16-bit types; f32 accumulate
# because the hardware has no bf16 accumulation.
SHUFFLE_AMPERE_PARAMS: List[GemmAlgoParams] = [
    *gen_shuffle_params(
        (64, 64, 32),
        (32, 32, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 128, 32),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 128, 32),
        (64, 32, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (64, 64, 64),
        (32, 32, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (64, 128, 64),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 256, 32),
        (64, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (256, 128, 32),
        (64, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 64, 32),
        (64, 32, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (64, 128, 32),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
]

'''
turing = re.search(r"(SHUFFLE_TURING_PARAMS\s*:.*?\n(?:.*\n)*?^]\s*$)", content, re.MULTILINE)
require(turing is not None, "cumm: SHUFFLE_TURING_PARAMS not found in cumm/gemm/main.py")
require("SHUFFLE_AMPERE_PARAMS" not in content, "cumm: SHUFFLE_AMPERE_PARAMS already present?")
content = content[:turing.end()] + "\n" + AMPERE_PARAMS_BLOCK + content[turing.end():]

content, n = re.subn(r"(ampere_params\s*=\s*\[)\s*\n\s*(\])",
                     r"\1\n                    *SHUFFLE_AMPERE_PARAMS,\n                \2",
                     content, count=1)
require(n == 1, "cumm: the empty ampere_params list in GemmMainUnitTest.__init__ was not found")

BF16_SIMT_PARAMS = '''
    # cuda-foundry: bf16 Simt fallback kernels for misaligned dimensions
    *gen_shuffle_params(
        (128, 128, 8),
        (32, 64, 8), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (32, 64, 32),
        (32, 32, 8), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (32, 32, 32),
        (32, 32, 8), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (64, 128, 16),
        (32, 64, 8), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (64, 64, 8),
        (32, 32, 8), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Simt, None),
'''
simt = re.search(r"(SHUFFLE_SIMT_PARAMS\s*:.*?\n(?:.*\n)*?)(^\]\s*$)", content, re.MULTILINE)
require(simt is not None, "cumm: SHUFFLE_SIMT_PARAMS closing bracket not found -- the "
                          "artifact would ship without bf16 Simt fallback params")
content = content[:simt.start(2)] + BF16_SIMT_PARAMS + content[simt.start(2):]
main_py.write_text(content, encoding="utf-8")
final = main_py.read_text(encoding="utf-8")
require(final.count("SHUFFLE_AMPERE_PARAMS") == 2 and "bf16 Simt fallback" in final,
        "cumm: bf16 GEMM params NOT on disk as expected")
import ast  # noqa: E402
ast.parse(final)
print("cumm patch: bf16 GEMM params (9 Ampere TensorOp + 5 Simt) -> cumm/gemm/main.py")

# ── 3. zero_whole_storage_ default Context ────────────────────────────────
sub_once(pathlib.Path("cumm/tensorview_bind.py"),
         '.def("zero_whole_storage_", &tv::Tensor::zero_whole_storage_)',
         '.def("zero_whole_storage_", &tv::Tensor::zero_whole_storage_, py::arg("ctx") = tv::Context())',
         "zero_whole_storage_ binding")

# ── 4. include path chosen by the header, not the directory ──────────────
sub_once(pathlib.Path("cumm/constants.py"),
         """TENSORVIEW_INCLUDE_PATH = _TENSORVIEW_INCLUDE_PATHS[0]
if not TENSORVIEW_INCLUDE_PATH.exists():
    for p in _TENSORVIEW_INCLUDE_PATHS[1:]:
        if p.exists():
            TENSORVIEW_INCLUDE_PATH = p

assert TENSORVIEW_INCLUDE_PATH.exists()""",
         """_HEADER_SENTINEL = Path("tensorview") / "core" / "all.h"
TENSORVIEW_INCLUDE_PATH = None
for p in _TENSORVIEW_INCLUDE_PATHS:
    if (p / _HEADER_SENTINEL).exists():
        TENSORVIEW_INCLUDE_PATH = p
        break

assert TENSORVIEW_INCLUDE_PATH is not None and TENSORVIEW_INCLUDE_PATH.exists()""",
         "TENSORVIEW_INCLUDE_PATH resolution")

# ── 5 (not ported): nvrtc/core.h already provides std::abs/min/max ─────────
core_h = pathlib.Path("include/tensorview/core/nvrtc/core.h").read_text(encoding="utf-8")
for fn in ("abs", "min", "max"):
    require(f" {fn}(" in core_h and "namespace std" in core_h,
            f"cumm: nvrtc/core.h no longer defines std::{fn} -- the farm's shim "
            f"may be needed after all; re-check before adding it")
require("std::abs" not in pathlib.Path("include/tensorview/core/nvrtc_std.h").read_text(encoding="utf-8"),
        "cumm: nvrtc_std.h now defines std::abs itself?")
print("cumm patch: std::abs/min/max already in nvrtc/core.h; no shim added")

# ── 6. NVRTC has no GCC builtins: limits.h through libcudacxx ─────────────
limits_h = pathlib.Path("include/tensorview/core/nvrtc/limits.h")
sub_once(limits_h, "#include <cuda/std/cassert>\n",
         "#include <cuda/std/cassert>\n#include <cuda/std/limits>  // cuda-foundry: see below\n",
         "limits.h libcudacxx include")
for old, new in (
        ("__builtin_huge_valf()", "::cuda::std::numeric_limits<float>::infinity()"),
        ('__builtin_nanf("")', "::cuda::std::numeric_limits<float>::quiet_NaN()"),
        ('__builtin_nansf("")', "::cuda::std::numeric_limits<float>::signaling_NaN()"),
        ("__builtin_huge_val()", "::cuda::std::numeric_limits<double>::infinity()"),
        ('__builtin_nan("")', "::cuda::std::numeric_limits<double>::quiet_NaN()"),
        ('__builtin_nans("")', "::cuda::std::numeric_limits<double>::signaling_NaN()")):
    sub_once(limits_h, old, new, f"limits.h {old}")
require("__builtin_" not in limits_h.read_text(encoding="utf-8"),
        "cumm: limits.h still calls a GCC builtin NVRTC does not have")

# ── 5. inline demangler ahead of cu++filt ─────────────────────────────────
sub_once(pathlib.Path("cumm/nvrtc/__init__.py"),
         '''        res = subprocess.check_output(["cu++filt",
                                       name]).decode("utf-8").strip()
        return res''',
         '''        # cuda-foundry (packages/cumm/patches): NVRTC mangles with the Itanium
        # ABI on every platform, and cu++filt does not exist on Windows. The
        # common case, _ZN<len><name>...E, is decoded here; anything else
        # still goes to cu++filt, and a missing cu++filt returns the name.
        if name.startswith("_ZN") and name.endswith("E"):
            parts = []
            i = 3
            s = name[:-1]
            try:
                while i < len(s):
                    n = 0
                    while i < len(s) and s[i].isdigit():
                        n = n * 10 + int(s[i])
                        i += 1
                    if n > 0 and i + n <= len(s):
                        parts.append(s[i:i+n])
                        i += n
                    else:
                        break
                if parts and i == len(s):
                    return "::".join(parts)
            except Exception:
                pass
        try:
            res = subprocess.check_output(["cu++filt",
                                           name]).decode("utf-8").strip()
            return res
        except (FileNotFoundError, subprocess.CalledProcessError):
            return name''',
         "cu++filt demangler")

# ── 8. the toolkit headers from $CONDA_PREFIX first ───────────────────────
# _get_cuda_include_lib() is what every NVRTC compile at run time uses to find
# cuda_runtime.h and friends, and at 0.7.11 it knows two places: the prefix
# `which nvcc` (Get-Command on Windows) resolves into, and /usr/local/cuda
# (C:\Program Files\...\CUDA on Windows). A conda environment has the headers
# in $CONDA_PREFIX -- cuda-cudart-dev + cuda-cccl (run_deps) put them at
# targets/x86_64-linux/include + lib on linux-64 and Library/include +
# Library/lib on win-64 -- and need not have nvcc on PATH at all. Probe that
# FIRST, guarded on cuda.h AND the cudart link library actually being there.
# The guard is what keeps the BUILD honest: under rattler-build $CONDA_PREFIX
# is the host prefix, which carries no cuda.h (the toolkit lives in
# $BUILD_PREFIX), so the probe fails and upstream's nvcc lookup finds the
# cell's toolkit exactly as before.
sub_once(pathlib.Path("cumm/common.py"),
         """def _get_cuda_include_lib():
    global _CACHED_CUDA_INCLUDE_LIB
    if _CACHED_CUDA_INCLUDE_LIB is None:
        if compat.InWindows:
""",
         """def _cuw_conda_cuda_include_lib():
    \"\"\"cuda-foundry (packages/cumm/patches): ([include dirs], lib dir) from
    $CONDA_PREFIX when the CUDA headers and the cudart link library are both
    there (cuda-cudart-dev in the environment), else None.\"\"\"
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None
    prefix = Path(prefix)
    if compat.InWindows:
        candidates = [(prefix / "Library" / "include", prefix / "Library" / "lib", "cudart.lib")]
    else:
        candidates = [(t / "include", prefix / "lib", "libcudart.so")
                      for t in sorted((prefix / "targets").glob("*-linux"))]
    for include, lib, cudart in candidates:
        if (include / "cuda.h").exists() and (lib / cudart).exists():
            return ([include], lib)
    return None


def _get_cuda_include_lib():
    global _CACHED_CUDA_INCLUDE_LIB
    if _CACHED_CUDA_INCLUDE_LIB is None:
        _CACHED_CUDA_INCLUDE_LIB = _cuw_conda_cuda_include_lib()
        if _CACHED_CUDA_INCLUDE_LIB is not None:
            return _CACHED_CUDA_INCLUDE_LIB
        if compat.InWindows:
""",
         "$CONDA_PREFIX header lookup")
common_py = pathlib.Path("cumm/common.py").read_text(encoding="utf-8")
require("import os\n" in common_py and "from pathlib import Path" in common_py,
        "cumm: common.py no longer imports os / Path, which the conda lookup uses")
ast.parse(common_py)

# ── 9. the wheel's License field matches the LICENSE file ─────────────────
# setup.py says license='MIT'; the LICENSE file is the Apache-2.0 text and
# every source header says "Licensed under the Apache License, Version 2.0".
# The classifier is the one place MIT appears, so the wheel METADATA is what
# is corrected, to agree with the text it ships and with package.yml.
sub_once(pathlib.Path("setup.py"), "    license='MIT',", "    license='Apache-2.0',",
         "License classifier")

# ── 7. no exact-minor nvrtc-builtins link ─────────────────────────────────
sub_once(pathlib.Path("cumm/common.py"),
         '''        self.build_meta.add_libraries("nvrtc")
        if compat.InLinux:
            self.build_meta.add_ldflags("g++", "-Wl,--no-as-needed", "nvrtc-builtins")
            self.build_meta.add_ldflags("clang++", "-Wl,--no-as-needed", "nvrtc-builtins")
            self.build_meta.add_ldflags("nvcc", "-Wl,--no-as-needed", "nvrtc-builtins")
''',
         '''        self.build_meta.add_libraries("nvrtc")
        # cuda-foundry (packages/cumm/patches): the explicit nvrtc-builtins
        # link is removed. Its soname is exact-minor (libnvrtc-builtins.so.12.8)
        # and would bind the artifact to one CUDA minor; libnvrtc.so.12 finds
        # its builtins itself, and the wheel carries them via
        # wheel_vendor_extra.
''',
         "CummNVRTCLink builtins ldflags")
# The same link, spelled with a -l, in the class core_cc is actually built
# from (TensorViewBind); the first local build still recorded the DT_NEEDED
# because only common.py had been patched.
sub_once(pathlib.Path("cumm/tensorview_bind.py"),
         '''            if compat.InLinux:
                self.build_meta.add_ldflags("g++", "-Wl,--no-as-needed", "-lnvrtc-builtins")
                self.build_meta.add_ldflags("clang++", "-Wl,--no-as-needed", "-lnvrtc-builtins")
                self.build_meta.add_ldflags("nvcc", "-Wl,--no-as-needed", "-lnvrtc-builtins")
''',
         '''            # cuda-foundry (packages/cumm/patches): no explicit nvrtc-builtins
            # link -- exact-minor soname; see common.py CummNVRTCLink.
''',
         "TensorViewBind builtins ldflags")
for f in ("cumm/common.py", "cumm/tensorview_bind.py"):
    require("nvrtc-builtins\"" not in pathlib.Path(f).read_text(encoding="utf-8"),
            f"cumm: {f} still names nvrtc-builtins on a link line")

# ── not ported: assert the anchors really are absent at this rev ──────────
for hdr in ("half.h", "bfloat16.h", "tf32.h", "float8.h"):
    p = pathlib.Path("include/tensorview/gemm/dtypes") / hdr
    require(p.is_file() and "namespace CUDA_NAMESPACE_STD {" not in p.read_text(encoding="utf-8"),
            f"cumm: {hdr} now has the CUDA_NAMESPACE_STD block the farm rewrote -- "
            f"the dtype namespace patch must be ported after all")
require("can't find cudart include for nvrtc" not in pathlib.Path("cumm/common.py").read_text(encoding="utf-8"),
        "cumm: common.py now carries the cudart-include raise the farm shimmed -- port it")
require(not pathlib.Path("third_party").exists() and "cccl" not in pathlib.Path("setup.py").read_text(encoding="utf-8"),
        "cumm: setup.py now references third_party/cccl -- the header population must be ported")
print("cumm patch: not-ported sections confirmed absent at this rev")
