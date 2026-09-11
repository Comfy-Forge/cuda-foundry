"""Patch pyg-lib 0.5.0: arch list bridge, RPATH hygiene, no nvrtc link, and
the C++ standard decided by the torch being built against.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed. Two consequences of that timing shape this
file, and they are why it differs from cuda-wheels' packages/pyg_lib/patches:

  * No `import torch`. The fetch step runs on the host, outside any build
    env. The predecessor's first section patched the INSTALLED torch's
    share/cmake files (neutralising legacy nvToolsExt references for torch
    2.4-2.6). That cannot be reached from here, so it is NOT ported: torch
    >= 2.7 rewrote those files to nvtx3 and needs nothing, and a torch < 2.7
    cell of this package is unbuilt until someone adds a CMake-side shim.
    Stated in package.yml rather than left to be discovered.
  * ONE tarball serves every cell. The predecessor gated the C++20 switch on
    CUW_TORCH_VERSION at patch time; here the same decision is moved into
    CMakeLists.txt, where the torch version is a fact the configure step can
    read. The condition is the farm's, unchanged: C++20 for torch >= 2.13.

Every substitution asserts on its own (docs/WINDOWS.md, the fused-ssim
lesson: a whole-file before/after guard passes on a partial match).

── 1. CUDA arch bridge ────────────────────────────────────────────────────
pyg_lib picks CMAKE_CUDA_ARCHITECTURES from a hardcoded if-ladder keyed on
the nvcc version (CMakeLists.txt:54-64) and never reads TORCH_CUDA_ARCH_LIST.
On x86 it builds MORE than asked -- the ladder's tags are unsuffixed, so cmake
emits BOTH -real and -virtual per arch -- and on aarch64 it builds the WRONG
thing (no sm_87 at any CUDA version). The bridge translates the cell's
TORCH_CUDA_ARCH_LIST into explicit NN-real / NN-virtual entries and leaves
the ladder in place as the fallback for a bare build. arch_override.yml is
the union of the old list and what the ladder really built (see its comment).

── 2. RPATH: $ORIGIN only ─────────────────────────────────────────────────
CMakeLists.txt links ${TORCH_LIBRARIES} wholesale, and cmake's default bakes
the build-tree location of every linked library into the binary's RPATH.
libpyg.so shipped four absolute entries (/lib/intel64, /lib/intel64_win,
/lib/win-x64 and the torch/lib of the build machine). BUILD_WITH_INSTALL_RPATH
is the operative property: it applies INSTALL_RPATH to the build-tree binary,
which is the one that gets packaged (no `make install` ever runs). $ORIGIN is
sufficient: `import torch` has loaded libtorch into the process before this
extension is imported. Here this is belt-and-braces -- rattler-build rewrites
RPATHs for the .conda and make_wheel.py drops every non-$ORIGIN entry from
the wheel -- but the binary the two tools start from should be right anyway.

── 3. nvrtc is not linked ─────────────────────────────────────────────────
TorchConfig.cmake puts ${CUDA_NVRTC_LIB} (linux) / caffe2_nvrtc (MSVC) at
the front of TORCH_CUDA_LIBRARIES, so a wholesale link of TORCH_LIBRARIES
records a DT_NEEDED on libnvrtc.so.12 that libpyg.so never calls -- the
farm's own wheel has 0 of its 398 undefined symbols in nvrtc, and its
predecessor's attempt at `--as-needed` demonstrably did not remove it: the
published pyg_lib-0.5.0+cu128torch2.8 wheel vendors a 104 MB
libnvrtc-2bb82d1a.so.12.8.93 for nothing. That patch's own comment names
the right lever -- "filtering TORCH_LIBRARIES, not a link flag whose position
we do not control" -- so that is what this does, on the list itself, before
the link line. Consequence in both outputs: no vendored nvrtc in the wheel,
and no cuda-nvrtc run dep on the .conda.

── 4. C++ standard from the torch version ─────────────────────────────────
Upstream hardcodes `set(CMAKE_CXX_STANDARD 17)` and sets no CUDA standard, so
torch's cpp_extension cannot choose for it. torch 2.13's headers are
C++20-only on MSVC (C7555 designated initializers in c10/util/StringUtil.h,
C7582 bit-field NSDMIs in c10/core/AutogradState.h); GCC accepts both as
extensions, which is why only Windows failed in the farm. Both standards
must move: bumping CXX alone leaves cuda/hash_map.cu failing in the nvcc
frontend with "data member initializer is not allowed". Kept at 17 below
2.13, because torch < 2.7 fails ON WINDOWS at C++20 (nvcc's EDG misparses
ivalue_inl.h; see patch_lib's standard note). The torch version is read at
configure time with the same Python3 upstream already probes for the C++11
ABI -- a configure-time fact, so one tarball serves every torch cell.

The C++20 bump makes nvcc emit a diagnostic it stays quiet about at C++17:
the vendored CUTLASS (b78588d) marks CudaHostAdapter::memsetDevice as
CUTLASS_HOST_DEVICE while its only body calls the host-only pure-virtual
memsetDeviceImpl; torch's cmake injects --Werror cross-execution-space-call,
so it is fatal. The function was never callable from device code, so it is
marked host-only unconditionally -- a no-op at C++17 and the fix at C++20.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require  # noqa: E402

cml = pathlib.Path("CMakeLists.txt")
require(cml.exists(), "pyg_lib: CMakeLists.txt not found at source root")
text = cml.read_text(encoding="utf-8")


def sub_once(haystack: str, old: str, new: str, what: str) -> str:
    n = haystack.count(old)
    require(n == 1, f"pyg_lib: expected exactly one {what} anchor in "
                    f"CMakeLists.txt, found {n} -- upstream changed; re-check "
                    f"against the pinned source_rev")
    return haystack.replace(old, new, 1)


# ── 1. arch bridge ─────────────────────────────────────────────────────────
arch_anchor = """  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;90;100;120")"""
arch_bridge = """  # --- cuda-foundry arch bridge (injected; see packages/pyg-lib/patches) ---
  # Translate the cell's TORCH_CUDA_ARCH_LIST into CMAKE_CUDA_ARCHITECTURES.
  # "8.0 8.7 9.0+PTX" -> "80-real;87-real;90-real;90-virtual".
  # Explicit -real/-virtual matters: a bare "80" makes cmake emit real AND
  # virtual, which is how upstream's ladder shipped PTX for every arch.
  set(_CUW_ARCHS "")
  if (NOT "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
    string(REPLACE ";" " " _CUW_RAW "$ENV{TORCH_CUDA_ARCH_LIST}")
    string(REPLACE "," " " _CUW_RAW "${_CUW_RAW}")
    separate_arguments(_CUW_TOKENS UNIX_COMMAND "${_CUW_RAW}")
    foreach(_CUW_TOK IN LISTS _CUW_TOKENS)
      set(_CUW_PTX FALSE)
      if (_CUW_TOK MATCHES "\\\\+PTX$")
        set(_CUW_PTX TRUE)
        string(REPLACE "+PTX" "" _CUW_TOK "${_CUW_TOK}")
      endif()
      string(STRIP "${_CUW_TOK}" _CUW_TOK)
      string(REPLACE "." "" _CUW_NUM "${_CUW_TOK}")
      if (NOT _CUW_NUM STREQUAL "")
        list(APPEND _CUW_ARCHS "${_CUW_NUM}-real")
        if (_CUW_PTX)
          list(APPEND _CUW_ARCHS "${_CUW_NUM}-virtual")
        endif()
      endif()
    endforeach()
  endif()
  if (NOT _CUW_ARCHS STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "${_CUW_ARCHS}")
    message(STATUS "cuda-foundry: CMAKE_CUDA_ARCHITECTURES from TORCH_CUDA_ARCH_LIST -> ${CMAKE_CUDA_ARCHITECTURES}")
  elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;90;100;120")"""
text = sub_once(text, arch_anchor, arch_bridge, "CMAKE_CUDA_ARCHITECTURES ladder")

# ── 3. nvrtc filter, then 2. RPATH -- both hang off the one link line ───────
link_anchor = "target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})"
link_block = """# --- cuda-foundry: do not link nvrtc (injected; see packages/pyg-lib/patches) ---
# TorchConfig.cmake puts libnvrtc / caffe2_nvrtc at the front of
# TORCH_CUDA_LIBRARIES. libpyg uses no nvrtc symbol, and a DT_NEEDED on it
# costs the wheel a 104 MB vendored library and the conda package a run dep.
list(FILTER TORCH_LIBRARIES EXCLUDE REGEX "nvrtc")
message(STATUS "cuda-foundry: TORCH_LIBRARIES without nvrtc: ${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

# --- cuda-foundry RPATH hygiene (injected; see packages/pyg-lib/patches) ---
# Do not bake the build machine's MKL/torch paths into the shipped .so.
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
message(STATUS "cuda-foundry: ${PROJECT_NAME} RPATH pinned to $ORIGIN")
# --- end cuda-foundry link block ---"""
text = sub_once(text, link_anchor, link_block, "TORCH_LIBRARIES link line")

# ── 4. C++ standard from the torch version ─────────────────────────────────
std_anchor = "set(CMAKE_CXX_STANDARD 17)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
std_block = """# --- cuda-foundry: C++ standard follows the torch version (injected) ---
# torch >= 2.13 headers need C++20 (MSVC: C7555/C7582); torch < 2.7 breaks
# ON WINDOWS at C++20 (nvcc's EDG misparses ivalue_inl.h). Both the CXX and
# the CUDA standard move together: cuda/hash_map.cu fails in the nvcc
# frontend otherwise. Read at configure time so one source tree serves
# every torch cell; the CXX11 ABI probe below already needs Python3+torch.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c"
            "import torch; v=torch.__version__.split('+')[0].split('.'); print('.'.join(v[:2]), end='')"
    RESULT_VARIABLE _CUW_TORCH_RC
    OUTPUT_VARIABLE _CUW_TORCH_VERSION)
if (NOT _CUW_TORCH_RC EQUAL 0 OR "${_CUW_TORCH_VERSION}" STREQUAL "")
  message(FATAL_ERROR "cuda-foundry: could not read torch's version through ${Python3_EXECUTABLE}; the C++ standard cannot be chosen")
endif()
if (_CUW_TORCH_VERSION VERSION_GREATER_EQUAL 2.13)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
  set(CMAKE_CXX_STANDARD 17)
endif()
message(STATUS "cuda-foundry: torch ${_CUW_TORCH_VERSION} -> CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
"""
text = sub_once(text, std_anchor, std_block, "CMAKE_CXX_STANDARD 17")

cml.write_text(text, encoding="utf-8")

# Prove every block landed on disk rather than trusting the replaces.
final = cml.read_text(encoding="utf-8")
require("cuda-foundry arch bridge" in final, "pyg_lib: arch bridge NOT on disk")
require(final.count("set(CMAKE_CUDA_ARCHITECTURES") >= 6,
        "pyg_lib: the upstream ladder fallback was damaged by the bridge")
require('list(FILTER TORCH_LIBRARIES EXCLUDE REGEX "nvrtc")' in final,
        "pyg_lib: nvrtc filter NOT on disk")
require("BUILD_WITH_INSTALL_RPATH TRUE" in final, "pyg_lib: RPATH block NOT on disk")
require("_CUW_TORCH_VERSION VERSION_GREATER_EQUAL 2.13" in final,
        "pyg_lib: C++ standard gate NOT on disk")
require(final.count("set(CMAKE_CXX_STANDARD 17)") == 1
        and final.count("set(CMAKE_CXX_STANDARD 20)") == 1,
        "pyg_lib: the C++ standard is set in more places than the gate")
print("pyg_lib patch: arch bridge, nvrtc filter, RPATH hygiene, C++ standard gate -> CMakeLists.txt")

# ── 4b. CUTLASS memsetDevice is host-only ──────────────────────────────────
cha = pathlib.Path("third_party/cutlass/include/cutlass/cuda_host_adapter.hpp")
require(cha.exists(), "pyg_lib: third_party/cutlass not on disk -- clone_recursive "
                      "should have fetched it")
h = cha.read_text(encoding="utf-8")
needle = "  CUTLASS_HOST_DEVICE\n  Status memsetDevice("
fixed = "  CUTLASS_HOST\n  Status memsetDevice("
require(h.count(needle) == 1,
        "pyg_lib: cutlass cuda_host_adapter.hpp no longer has the expected "
        "memsetDevice declaration -- the submodule pin moved; torch >= 2.13 "
        "Windows cells would fail with cross-execution-space-call")
cha.write_text(h.replace(needle, fixed, 1), encoding="utf-8")
require(fixed in cha.read_text(encoding="utf-8"),
        "pyg_lib: cutlass memsetDevice fix NOT on disk")
print("pyg_lib patch: cutlass memsetDevice -> CUTLASS_HOST")

# ── 5. cmake must probe THE python that is running setup.py ────────────────
# Both the upstream CXX11-ABI probe and the C++ standard gate above run
# `import torch` through find_package(Python3), which takes the first
# python on PATH. Under rattler-build the build env ($BUILD_PREFIX/bin) is
# ahead of the host env on PATH, and the torch we compile against lives in
# the host env. Hand cmake the interpreter pip is already using, so the
# probe cannot answer for a different python than the one the wheel is for.
setup_py = pathlib.Path("setup.py")
s = setup_py.read_text(encoding="utf-8")
anchor = "        cmake_args = [\n            '-DBUILD_TEST=OFF',\n"
fixed = ("        import sys\n"
         "        cmake_args = [\n"
         "            f'-DPython3_EXECUTABLE={sys.executable}',  # cuda-foundry: see patches/pyg_lib.py\n"
         "            '-DBUILD_TEST=OFF',\n"
         "        ]\n"
         "        # cuda-foundry: torch's cuda.cmake runs the LEGACY FindCUDA, which takes\n"
         "        # the first nvcc on PATH. The host env carries cuda-nvcc-tools (12.9,\n"
         "        # dragged in beside torch) while the cell's toolkit is the build env's\n"
         "        # 12.8, so it found the wrong one and refused: 'FindCUDA says CUDA\n"
         "        # version is 12.9 but the headers say 12.8' (run 34592193335). Point it\n"
         "        # at the toolkit the cell pinned; build.sh exports CUDA_HOME, the\n"
         "        # win-64 nvcc activation exports CUDA_PATH.\n"
         "        _cuda_root = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')\n"
         "        if _cuda_root:\n"
         "            cmake_args.append(f'-DCUDA_TOOLKIT_ROOT_DIR={_cuda_root}')\n"
         "        cmake_args += [\n")
require(s.count(anchor) == 1, "pyg_lib: setup.py cmake_args anchor not found "
                              "-- upstream changed; re-check")
setup_py.write_text(s.replace(anchor, fixed, 1), encoding="utf-8")
_final_setup = setup_py.read_text(encoding="utf-8")
require("-DPython3_EXECUTABLE=" in _final_setup and "-DCUDA_TOOLKIT_ROOT_DIR=" in _final_setup,
        "pyg_lib: Python3_EXECUTABLE / CUDA_TOOLKIT_ROOT_DIR hints NOT on disk")
import ast as _ast  # noqa: E402
_ast.parse(_final_setup)
print("pyg_lib patch: cmake gets -DPython3_EXECUTABLE=<the building python> and "
      "-DCUDA_TOOLKIT_ROOT_DIR=<the cell's toolkit>")
