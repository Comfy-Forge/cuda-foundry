"""Patch cumesh_vb (visualbruno/CuMesh @ d10e54c) into a package that
installs and imports beside plain cumesh.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed, on a Linux host with no torch importable. One
tarball feeds both platforms, so nothing here branches on the platform or a
torch version. Ported from the farm's packages/cumesh_vb/patches/cumesh_vb.py
and re-read against the pinned rev; every substitution asserts on its own.

  0. Eigen. visualbruno committed third_party/cubvh flat, so cubvh's own
     eigen submodule is never walked and setup.py's include path
     third_party/cubvh/third_party/eigen is empty. Fetched as a sha256-pinned
     tarball at the commit JeffreyXiang/cubvh pins (e63d9f6) -- see
     patch_lib.EIGEN_E63D9F6 for why that commit and not 3.4.0 or master.
  1. Rename cumesh -> cumesh_vb: pyproject name, setup.py name, package
     list, extension names, the package directory.
  2. C++ standard: drop upstream's `-std=c++20` (five sites) and let torch's
     cpp_extension choose; at torch 2.8 that is c++17. On win-64 torch's ninja
     path prepends -std=c++17 to every nvcc line unconditionally, so leaving
     c++20 in would also have meant two -std= values on one command line.
     The farm's `translate_cxx_flags_for_msvc` (-O3 -> /O2) is NOT ported:
     it ran only on a Windows fetch host, which this repo never has, and it
     is cosmetic -- torch's win_wrap_ninja_compile puts distutils'
     compile_options (/O2 among them) on every cl line, so cl already
     optimises and merely warns D9002 about the -O3 it ignores. On Windows
     the `--extended-lambda` the farm added to the _C nvcc list is added
     unconditionally: it is a plain nvcc flag, harmless on Linux.
  3. CCCL 3.x: 4-arg in-place ExclusiveSum -> 5-arg form (12 sites, same as
     plain cumesh). Not needed on cu12.8; keeps one tree for every line.
  4. module_local() on all seven pybind11 class registrations so this fork
     and plain cumesh can be imported into one interpreter. See the long
     rationale on patch_lib.add_pybind_module_local; short form: both builds
     bind the same C++ types under the same names into one global pybind11
     registry, and the second import dies with "generic_type: type "CuMesh"
     is already registered". Fixed in the FORK, not in plain cumesh.
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import (bind_symbol_version, EIGEN_E63D9F6, add_pybind_module_local,  # noqa: E402
                       fix_inplace_exclusive_sum_in_files, require,
                       strip_std_flags, vendor_eigen)

# ── 0. Eigen at cubvh's pin ───────────────────────────────────────────────
eigen_dir = pathlib.Path("third_party/cubvh/third_party/eigen")
vendor_eigen(eigen_dir, EIGEN_E63D9F6)
mf = eigen_dir / "Eigen/src/Core/MathFunctions.h"
require("There is no official ::arg on device" in mf.read_text(encoding="utf-8", errors="replace"),
        f"{mf}: Eigen is missing the MSVC device-side `arg` fix; expected the "
        f"tree at e63d9f6, got something older")

# ── 1. Rename ─────────────────────────────────────────────────────────────
def sub_once(path, old, new, what, count=1):
    p = pathlib.Path(path)
    t = p.read_text()
    n = t.count(old)
    if n == 0 and new in t:
        print(f"cumesh_vb patch: {what}: already applied")
        return
    require(n == count, f"cumesh_vb: {what}: expected {count} occurrence(s) of "
                        f"{old!r} in {path}, found {n} -- upstream changed")
    p.write_text(t.replace(old, new))
    print(f"cumesh_vb patch: {what} ({n} site{'s' if n != 1 else ''})")

sub_once("pyproject.toml", 'name = "cumesh"', 'name = "cumesh_vb"', "pyproject name")
sub_once("setup.py", 'name="cumesh",', 'name="cumesh_vb",', "setup() name")
sub_once("setup.py", "'cumesh',", "'cumesh_vb',", "packages list")
sub_once("setup.py", 'name="cumesh._C"', 'name="cumesh_vb._C"', "_C extension name")
sub_once("setup.py", "name='cumesh._cubvh'", "name='cumesh_vb._cubvh'", "_cubvh extension name")
sub_once("setup.py", "name='cumesh._xatlas'", "name='cumesh_vb._xatlas'", "_xatlas extension name")

src_dir, dst_dir = pathlib.Path("cumesh"), pathlib.Path("cumesh_vb")
if src_dir.is_dir() and not dst_dir.exists():
    src_dir.rename(dst_dir)
    print("cumesh_vb patch: cumesh/ -> cumesh_vb/")
require(dst_dir.is_dir() and not src_dir.exists(),
        "cumesh_vb: package directory rename did not land")
# Every module in the package imports its siblings relatively (`from . import
# _C`), so nothing inside needs rewriting; assert that stays true.
for py in dst_dir.rglob("*.py"):
    t = py.read_text()
    require(not re.search(r"^\s*(from|import)\s+cumesh(\.|\s|$)", t, re.M),
            f"cumesh_vb: {py} imports the top-level name `cumesh`, which after "
            f"the rename means the OTHER package -- rewrite it to relative")
final_setup = pathlib.Path("setup.py").read_text()
require("cumesh_vb" in final_setup and not re.search(r"""["']cumesh["']""", final_setup)
        and not re.search(r"""["']cumesh\.""", final_setup),
        "cumesh_vb: setup.py still names the unrenamed package somewhere")

# ── 2. C++ standard ───────────────────────────────────────────────────────
setup = pathlib.Path("setup.py")
text = setup.read_text()
new, n_std = strip_std_flags(text)
if n_std:
    require(n_std == 5, f"cumesh_vb: stripped {n_std} C++-standard flag(s), "
                        f"expected 5 at d10e54c -- upstream changed")
    setup.write_text(new)
    print(f"cumesh_vb patch: dropped {n_std} hardcoded C++-standard flag(s)")
else:
    require("std=c++" not in text and "std:c++" not in text,
            "cumesh_vb: a C++-standard flag survives in a spelling strip_std_flags "
            "does not recognise")
    print("cumesh_vb patch: setup.py already carries no C++-standard flag")
sub_once("setup.py",
         '"nvcc": ["-O3",] + cc_flag,\n',
         '"nvcc": ["-O3", "--extended-lambda"] + cc_flag,\n',
         "--extended-lambda on the _C nvcc list")

# ── 3. CCCL 3.x ExclusiveSum ──────────────────────────────────────────────
n_cub = fix_inplace_exclusive_sum_in_files(
    ["src/shared.h", "src/atlas.cu", "src/simplify.cu", "src/connectivity.cu",
     "src/clean_up.cu", "src/remesh/svox2vert.cu"], required=False)
if n_cub:
    require(n_cub == 12, f"cumesh_vb: rewrote {n_cub} ExclusiveSum call(s), "
                         f"expected 12 at d10e54c -- upstream changed")
    print(f"cumesh_vb patch: {n_cub} in-place ExclusiveSum call(s) -> 5-arg form")
else:
    print("cumesh_vb patch: ExclusiveSum calls already in 5-arg form")

# ── 4. module_local() ─────────────────────────────────────────────────────
ml = add_pybind_module_local({
    "src/ext.cpp": 1,
    "third_party/cubvh/src/bindings.cpp": 3,
    "third_party/xatlas/binding.cpp": 3,
})
require(sum(ml.values()) == 7,
        f"cumesh_vb: expected 7 module-local pybind11 class registrations, got "
        f"{sum(ml.values())} {ml}")
print("cumesh_vb patch: 7 py::class_ registrations are module_local")

# ── libstdc++ symbol version the wheel policy admits ─────────────────────
# xatlas's task scheduler waits on a std::condition_variable; linked in a
# conda host env that reference binds to GLIBCXX_3.4.30 and auditwheel then
# refuses the manylinux_2_28 repair (runs 34585896166, 34590139840 -- gcc 13
# and gcc 10 alike). Bind it to the 3.4.11 node every libstdc++ since GCC
# 4.4 exports, which is what the farm's wheel of this same source carries.
bind_symbol_version("third_party/xatlas/xatlas_mod.cpp",
                    "_ZNSt18condition_variable4waitERSt11unique_lockISt5mutexE",
                    "GLIBCXX_3.4.11", label="cumesh_vb: xatlas condition_variable::wait")

# ── the cell's arch list must stay authoritative ─────────────────────────
for lineno, line in enumerate(setup.read_text().splitlines(), 1):
    if re.search(r"(?<!offload)-arch[= ]|gencode|get_device_capability|TORCH_CUDA_ARCH_LIST", line):
        sys.exit(f"cumesh_vb patch: setup.py:{lineno} touches the arch list "
                 f"({line.strip()!r}) -- re-check")
print("cumesh_vb patch: setup.py emits no arch flags; TORCH_CUDA_ARCH_LIST is authoritative")
print("cumesh_vb patch: done")
