"""Patch cumesh (JeffreyXiang/CuMesh @ cf1a2f0).

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed, on a Linux host with no torch importable. One
tarball feeds both platforms, so nothing here branches on the platform or on
a torch/CUDA version; what is applied is applied to both builds.

Ported from the wheel farm's packages/cumesh/patches/cumesh.py, read against
the pinned revision rather than assumed to still apply:

  1. DROPPED: the "move #if CUDART_VERSION out of the CUDA_CHECK macro
     argument" rewrite of src/atlas.cu. Upstream fixed it itself before
     cf1a2f0 (`auto reduce_op = ...` is hoisted above the ReduceByKey calls,
     atlas.cu:325-329), and the farm's copy has been printing "WARNING:
     Could not find expected code block" ever since. Asserted below that the
     fixed shape is what is there, so a future rev that regresses it fails
     here rather than on a Windows runner.

  2. KEPT: CCCL 3.x removed cub::DeviceScan::ExclusiveSum's 4-argument
     in-place overload; the 5-argument form with in == out is accepted by
     every CCCL. Not needed by the cu12.8 cell this repo builds today, but
     the rewrite is semantically a no-op on older toolkits and applying it
     unconditionally keeps one source tree for every CUDA line the arch
     policy lists. patch_lib.fix_inplace_exclusive_sum_in_files, 12 sites.

  3. CHANGED: the farm rewrote c++17 -> c++20 gated on torch >= 2.13 / CUDA
     >= 13.2, read from env vars the fetch step does not have (and which
     fail closed to "keep c++17" here). Instead every hardcoded standard
     flag is removed and torch's cpp_extension appends the one the installed
     torch needs (-std=c++17 / /std:c++17 for torch 2.8; measured in v2.8.0's
     append_std17_if_no_std_present). Same policy as flash-attn here. On
     win-64 this also avoids a duplicated `-std=` on the nvcc line: torch's
     ninja path prepends its own -std=c++17 unconditionally.

  4. DROPPED: `/permissive-` removal for CUDA < 12.6 on Windows -- a per-cell
     fact this step cannot see, and not applicable to cu12.8. The flags
     upstream passes on Windows (/permissive-, /Zc:__cplusplus, /EHsc,
     -allow-unsupported-compiler) are left exactly as upstream wrote them.

setup.py does not probe the build host's GPU: it emits no -arch/-gencode and
relies on torch's TORCH_CUDA_ARCH_LIST handling, so the cell's arch list is
authoritative without help. Asserted below regardless.
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import (bind_symbol_version, fix_inplace_exclusive_sum_in_files, require,  # noqa: E402
                       strip_std_flags)

# ── 1. atlas.cu: upstream's own fix must still be in place ────────────────
atlas = pathlib.Path("src/atlas.cu").read_text()
require("auto reduce_op = ::cuda::std::plus();" in atlas
        and "auto reduce_op = cub::Sum();" in atlas,
        "cumesh: src/atlas.cu no longer hoists the CUDART_VERSION-dependent "
        "reduce op out of the CUDA_CHECK() argument list -- MSVC cannot take a "
        "preprocessor directive inside a macro argument; re-port the farm's "
        "atlas.cu rewrite")
require(not re.search(r"CUDA_CHECK\([^;]*#if", atlas, re.S),
        "cumesh: src/atlas.cu has a #if inside a CUDA_CHECK() argument list")
print("cumesh patch: atlas.cu reduce-op hoist present upstream; nothing to do")

# ── 2. CCCL 3.x in-place ExclusiveSum ─────────────────────────────────────
n_cub = fix_inplace_exclusive_sum_in_files(
    ["src/shared.h", "src/atlas.cu", "src/simplify.cu", "src/connectivity.cu",
     "src/clean_up.cu", "src/remesh/svox2vert.cu"],
    required=False)
if n_cub == 0:
    # Idempotence: a re-run finds every call already in 5-arg form.
    require("ExclusiveSum(" in pathlib.Path("src/shared.h").read_text(),
            "cumesh: src/shared.h has no ExclusiveSum call at all -- upstream "
            "restructured; re-check the CCCL-3.x fix")
    print("cumesh patch: ExclusiveSum calls already in 5-arg form")
else:
    require(n_cub == 12,
            f"cumesh: rewrote {n_cub} in-place ExclusiveSum call(s), expected 12 "
            f"at cf1a2f0 -- upstream changed; re-read the call sites")
    print(f"cumesh patch: {n_cub} in-place ExclusiveSum call(s) -> 5-arg form")

# ── 3. C++ standard: torch decides ────────────────────────────────────────
setup = pathlib.Path("setup.py")
text = setup.read_text()
new, n_std = strip_std_flags(text)
if n_std == 0:
    require("std=c++" not in new and "std:c++" not in new,
            "cumesh: setup.py still carries a C++-standard flag in a spelling "
            "strip_std_flags does not recognise")
    print("cumesh patch: setup.py already carries no C++-standard flag")
else:
    # cxx (/std:), nvcc (-std= and -Xcompiler=/std:) on Windows; cxx and nvcc
    # -std= on POSIX: five literals at cf1a2f0.
    require(n_std == 5,
            f"cumesh: stripped {n_std} C++-standard flag(s), expected 5 -- "
            f"upstream setup.py changed; re-read it")
    setup.write_text(new)
    print(f"cumesh patch: dropped {n_std} hardcoded C++-standard flag(s); "
          f"torch's cpp_extension now selects the standard")

# ── libstdc++ symbol version the wheel policy admits ─────────────────────
# xatlas's task scheduler waits on a std::condition_variable; linked in a
# conda host env that reference binds to GLIBCXX_3.4.30 and auditwheel then
# refuses the manylinux_2_28 repair (runs 34585896166, 34590139840 -- gcc 13
# and gcc 10 alike). Bind it to the 3.4.11 node every libstdc++ since GCC
# 4.4 exports, which is what the farm's wheel of this same source carries.
bind_symbol_version("third_party/xatlas/xatlas.cpp",
                    "_ZNSt18condition_variable4waitERSt11unique_lockISt5mutexE",
                    "GLIBCXX_3.4.11", label="cumesh: xatlas condition_variable::wait")

# ── the cell's arch list must stay authoritative ─────────────────────────
final = setup.read_text()
for lineno, line in enumerate(final.splitlines(), 1):
    # `--offload-arch` is the ROCm branch (IS_HIP), never taken by a CUDA
    # build, so it is not an arch flag nvcc will ever see.
    if re.search(r"(?<!offload)-arch[= ]|gencode|get_device_capability|TORCH_CUDA_ARCH_LIST", line):
        sys.exit(f"cumesh patch: setup.py:{lineno} touches the arch list "
                 f"({line.strip()!r}); torch drops TORCH_CUDA_ARCH_LIST as soon "
                 f"as any user nvcc flag contains 'arch' -- re-check")
print("cumesh patch: setup.py emits no arch flags; TORCH_CUDA_ARCH_LIST is authoritative")

# ── the nested submodule chain actually resolved ──────────────────────────
for needed in ("third_party/cubvh/src/bvh.cu",
               "third_party/cubvh/third_party/eigen/Eigen/Dense",
               "third_party/xatlas/xatlas.cpp"):
    require(pathlib.Path(needed).is_file(),
            f"cumesh: {needed} is missing -- clone_recursive did not resolve the "
            f"submodule chain (cubvh -> eigen on gitlab); the sandboxed build "
            f"cannot fetch it")
print("cumesh patch: submodule chain (cubvh, eigen, xatlas) present")
