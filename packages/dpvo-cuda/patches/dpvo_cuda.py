"""Patch DPVO (princeton-vl/DPVO @ 859bbbf) into the dpvo_cuda package.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed, on a Linux host. One tarball feeds both
platforms, so every edit is unconditional; the Windows-motivated ones are
correct C++ on Linux too and are applied to both.

Ported from the farm's packages/dpvo_cuda/patches/dpvo_cuda.py and re-read
against the pinned rev. Every substitution asserts its own site count -- the
farm's version silently no-opped on at least one (`mutable_data_ptr<`, which
appears nowhere at 859bbbf) and could not tell.

  0. Eigen 3.4.0 at thirdparty/eigen-3.4.0, the path setup.py hardcodes.
     Upstream's README says to download it; the sandboxed build cannot, so
     it is vendored here as a sha256-pinned release tarball.
  1. Rename the distribution: name='dpvo' -> 'dpvo_cuda'. The Python package
     stays `dpvo` (find_packages), the extension modules stay top-level
     (cuda_corr, cuda_ba, lietorch_backends) -- same as the farm's wheels.
  2. `.type()` -> `.scalar_type()` in AT_DISPATCH_* (42 sites): torch 2.x
     deprecated Tensor::type() and the dispatch macros want a ScalarType.
  3. `<long,` -> `<int64_t,` in packed_accessor32 template arguments (50
     sites) and `.item<long>()` -> `.item<int64_t>()` (4 sites). On Windows
     `long` is 32-bit: the accessor would then read int64 index tensors as
     pairs of int32 -- wrong results, not merely a link error -- and torch's
     DLL exports no `long` instantiations, which is how the farm found it.
  4. Two C99 compound literals `Jj = (float[6]){...}` in ba_cuda.cu, which
     MSVC rejects, rewritten as element assignments.
  5. Delete a dead `atomicAdd(&r_total[0], ...)` on a double: r_total's only
     reader is a commented-out std::cout, and atomicAdd(double*) does not
     exist below sm_60 -- keeping it would cost the policy's Maxwell rows.
  6. Eigen 3.4.0's arg_default_impl reaches `arg` through EIGEN_USING_STD,
     which on the nvcc device pass expands to `using ::arg;` -- and MSVC has
     no global ::arg. Take the branch HIP already takes (`using std::arg;`).
     Surfaces only under C++20 (torch >= 2.12 on Windows), inert at 2.8;
     applied anyway so one Eigen tree is right for every cell.

setup.py emits no arch flag and reads no GPU: the cell's TORCH_CUDA_ARCH_LIST
is authoritative. Asserted at the end.
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import EIGEN_3_4_0, require, vendor_eigen  # noqa: E402


def sub_count(path, old, new, what, expect):
    """Replace every `old` in `path`; require exactly `expect` sites (or an
    already-applied file with zero left and `new` present)."""
    p = pathlib.Path(path)
    t = p.read_text()
    n = t.count(old)
    if n == 0:
        require(expect == 0 or new in t,
                f"dpvo_cuda: {what}: {old!r} not found in {path} and the "
                f"replacement is not there either -- upstream changed")
        print(f"dpvo_cuda patch: {what}: {path}: already applied")
        return 0
    require(n == expect, f"dpvo_cuda: {what}: expected {expect} site(s) of "
                         f"{old!r} in {path}, found {n} -- upstream changed")
    p.write_text(t.replace(old, new))
    print(f"dpvo_cuda patch: {what}: {path}: {n} site(s)")
    return n


# ── 0. Eigen ──────────────────────────────────────────────────────────────
eigen_dir = pathlib.Path("thirdparty/eigen-3.4.0")
vendor_eigen(eigen_dir, EIGEN_3_4_0)
setup_text = pathlib.Path("setup.py").read_text()
require("thirdparty/eigen-3.4.0" in setup_text,
        "dpvo_cuda: setup.py no longer points include_dirs at "
        "thirdparty/eigen-3.4.0 -- upstream changed; move the vendored tree")

# ── 1. distribution name ─────────────────────────────────────────────────
sub_count("setup.py", "name='dpvo',", "name='dpvo_cuda',", "distribution rename", 1)

# ── 2 + 3. torch 2.x API and 64-bit index types ───────────────────────────
TYPE_SITES = {
    "dpvo/altcorr/correlation_kernel.cu": 4,
    "dpvo/lietorch/src/lietorch_cpu.cpp": 19,
    "dpvo/lietorch/src/lietorch_gpu.cu": 19,
}
LONG_SITES = {
    "dpvo/altcorr/correlation_kernel.cu": 8,
    "dpvo/fastba/ba.cpp": 6,
    "dpvo/fastba/ba_cuda.cu": 19,
    "dpvo/fastba/block_e.cu": 17,
}
ITEM_LONG_SITES = {"dpvo/fastba/ba.cpp": 2, "dpvo/fastba/block_e.cu": 2}

for f, n in TYPE_SITES.items():
    sub_count(f, ".type()", ".scalar_type()", ".type() -> .scalar_type()", n)
for f, n in LONG_SITES.items():
    sub_count(f, "<long,", "<int64_t,", "accessor<long,...> -> int64_t", n)
for f, n in ITEM_LONG_SITES.items():
    sub_count(f, ".item<long>()", ".item<int64_t>()", ".item<long>() -> int64_t", n)
# The farm also replaced `mutable_data_ptr<` -> `data_ptr<`; there is no such
# call at 859bbbf, so that edit was a silent no-op there and is not carried.
for f in set(TYPE_SITES) | set(LONG_SITES):
    require("mutable_data_ptr<" not in pathlib.Path(f).read_text(),
            f"dpvo_cuda: {f} now uses mutable_data_ptr<T>; torch's Windows "
            f"DLL does not export those instantiations -- port the farm's "
            f"data_ptr<> rewrite")
# Nothing that still reads `.type()` may reach a dispatch macro.
for f in TYPE_SITES:
    t = pathlib.Path(f).read_text()
    require(not re.search(r"AT_DISPATCH\w*\([^)]*\.type\(\)", t),
            f"dpvo_cuda: {f} still passes .type() to an AT_DISPATCH macro")

# ── 4. MSVC: compound literals ───────────────────────────────────────────
ba_cuda = "dpvo/fastba/ba_cuda.cu"
sub_count(ba_cuda,
          "Jj = (float[6]){fx*W*d, 0, fx*-X*W*d2, fx*-X*Y*d2, fx*(1+X*X*d2), fx*-Y*d};",
          "Jj[0]=fx*W*d; Jj[1]=0; Jj[2]=fx*-X*W*d2; Jj[3]=fx*-X*Y*d2; Jj[4]=fx*(1+X*X*d2); Jj[5]=fx*-Y*d;",
          "compound literal (x row)", 1)
sub_count(ba_cuda,
          "Jj = (float[6]){0, fy*W*d, fy*-Y*W*d2, fy*(-1-Y*Y*d2), fy*(X*Y*d2), fy*X*d};",
          "Jj[0]=0; Jj[1]=fy*W*d; Jj[2]=fy*-Y*W*d2; Jj[3]=fy*(-1-Y*Y*d2); Jj[4]=fy*(X*Y*d2); Jj[5]=fy*X*d;",
          "compound literal (y row)", 1)
require("(float[6])" not in pathlib.Path(ba_cuda).read_text(),
        "dpvo_cuda: a C99 compound literal survives in ba_cuda.cu")

# ── 5. dead atomicAdd(double*) ───────────────────────────────────────────
sub_count(ba_cuda,
          "      atomicAdd(&r_total[0],  w * r * r);",
          "      // cuda-foundry patch: dead accumulation removed. r_total's only\n"
          "      // reader is the commented-out std::cout below, and atomicAdd(double*)\n"
          "      // needs sm_60 -- keeping this line would cost Maxwell support.",
          "dead atomicAdd(double*) removal", 1)
require("atomicAdd(&r_total" not in pathlib.Path(ba_cuda).read_text(),
        "dpvo_cuda: the r_total atomicAdd is still there")

# ── 6. Eigen: MSVC has no global ::arg ───────────────────────────────────
mf = eigen_dir / "Eigen/src/Core/MathFunctions.h"
sub_count(mf,
          "    #if defined(EIGEN_HIP_DEVICE_COMPILE)\n"
          "    // HIP does not seem to have a native device side implementation for the math routine \"arg\"\n"
          "    using std::arg;\n"
          "    #else\n"
          "    EIGEN_USING_STD(arg);\n"
          "    #endif",
          "    // cuda-foundry patch: MSVC has no global ::arg, which is what\n"
          "    // EIGEN_USING_STD expands to on the nvcc device pass.\n"
          "    using std::arg;",
          "Eigen arg() (complex overload)", 1)
sub_count(mf,
          "    EIGEN_USING_STD(arg);\n    return arg(x);",
          "    using std::arg;\n    return arg(x);",
          "Eigen arg() (real overload)", 1)
require("EIGEN_USING_STD(arg)" not in mf.read_text(),
        "dpvo_cuda: an EIGEN_USING_STD(arg) site survives in MathFunctions.h")

# ── the cell's arch list must stay authoritative ─────────────────────────
for lineno, line in enumerate(pathlib.Path("setup.py").read_text().splitlines(), 1):
    if re.search(r"-arch[= ]|gencode|get_device_capability|TORCH_CUDA_ARCH_LIST", line):
        sys.exit(f"dpvo_cuda patch: setup.py:{lineno} touches the arch list "
                 f"({line.strip()!r}) -- re-check")
print("dpvo_cuda patch: setup.py emits no arch flags; TORCH_CUDA_ARCH_LIST is authoritative")
print("dpvo_cuda patch: done")
