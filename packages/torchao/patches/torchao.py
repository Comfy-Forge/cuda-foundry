"""Stop torchao installing its `test/` directory as a top-level package.

Ported from cuda-wheels (packages/torchao/patches/torchao.py) and re-read
against v0.13.0: setup.py's `find_packages(exclude=["benchmarks",
"benchmarks.*"])` still lets the repo's `test/` tree ship as `import test`,
which is CPython's own stdlib package name. Only the top-level tree is
excluded.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import exclude_top_level_packages, require  # noqa: E402

exclude_top_level_packages(["test"])
require('"test"' in pathlib.Path("setup.py").read_text(),
        "torchao: the test exclusion did not land in setup.py")
print("torchao patch: done")


# ── Windows: CUTLASS is not optional for the sources upstream leaves in ──────
# setup.py v0.13.0 gates the CUTLASS include directories on `not IS_WINDOWS`
# (its own Windows wheels are pure python), but its Windows source filter
# only drops files with "cutlass" in the NAME: activation24/sparsify24.cu and
# sparse_gemm.cu include <cutlass/bfloat16.h> and stay, so the win-64 cell
# died with C1083 on the first of them (run 34586352085). Two honest ways
# out: drop the CUTLASS-dependent sources on Windows, or give Windows the
# include directories upstream withholds. The second keeps the artifact
# equal to Linux's and is what is tried here, with the host flags CUTLASS
# documents for MSVC (/Zc:__cplusplus, /bigobj, /Zc:preprocessor) forwarded
# through nvcc. Decided at BUILD time by setup.py's own IS_WINDOWS, never by
# this fetch-time script.
import re as _re
from pathlib import Path as _Path

_sp = _Path("setup.py")
_t = _sp.read_text()
_old_gate = "    if use_cuda and not IS_WINDOWS:\n        use_cutlass = True"
require(_t.count(_old_gate) == 1, "torchao: the CUTLASS platform gate was not found once")
_t = _t.replace(_old_gate, "    if use_cuda:  # cuda-foundry: CUTLASS on Windows too (see patch)\n        use_cutlass = True", 1)

_old_win = '''    if not IS_WINDOWS:
        extra_compile_args["cxx"].extend(
            ["-O3" if not debug_mode else "-O0", "-fdiagnostics-color=always"]
        )'''
require(_t.count(_old_win) == 1, "torchao: the non-Windows cxx block was not found once")
# A separate `if IS_WINDOWS:` BEFORE upstream's block, not an `else:` after
# its first statement: upstream's `if not IS_WINDOWS:` body continues with a
# nested `if use_cpu_kernels and is_linux:` block, and an else inserted after
# the first extend() lands inside it (SyntaxError at the generated line 431,
# runs 34596208356 and 34596419720 -- parsed here BEFORE writing now).
_t = _t.replace(_old_win, '''    if IS_WINDOWS:
        # cuda-foundry: what CUTLASS 3.x needs from MSVC, on both the host
        # compile and nvcc's host pass.
        _cuw_msvc = ["/Zc:__cplusplus", "/bigobj", "/Zc:preprocessor", "/permissive-"]
        extra_compile_args["cxx"].extend(_cuw_msvc)
        extra_compile_args["nvcc"].extend(f"-Xcompiler={f}" for f in _cuw_msvc)
''' + _old_win, 1)
import ast as _ast
_ast.parse(_t)
_sp.write_text(_t)
print("torchao patch: CUTLASS include dirs and MSVC flags enabled on Windows")


# ── Windows: export the PyInit_ symbol distutils insists on ─────────────────
# With CUTLASS enabled, every CUDA translation unit of _C compiled under MSVC
# (run 34596571344) and the build died at LINK: torchao's _C,
# _C_cutlass_90a and _C_cutlass_100a are torch-ops libraries -- TORCH_LIBRARY
# registrations, loaded with torch.ops.load_library, no Python module in
# them -- and distutils on Windows passes /EXPORT:PyInit_<name> to link.exe
# regardless (LNK2001: unresolved external symbol PyInit__C). ld on Linux
# never asks. One stub source, compiled into each of the three, exports a
# PyInit that refuses to be imported; the name is pasted from the
# -DTORCH_EXTENSION_NAME torch already puts on every compile line, and the
# whole file is empty off Windows (#ifdef _WIN32), so one tarball serves both
# platforms. mxfp8_cuda is a real pybind module and needs nothing.
_stub = _Path("torchao/csrc/cuw_pyinit_stub.cpp")
_stub.write_text('''// cuda-foundry (packages/torchao/patches/torchao.py): this library registers
// torch ops and is loaded with torch.ops.load_library; it has no Python
// module. distutils on Windows still links it with /EXPORT:PyInit_<name>, so
// export one that says so instead of failing the link with LNK2001.
#ifdef _WIN32
#include <Python.h>
#define CUW_PASTE2(a, b) a##b
#define CUW_PASTE(a, b) CUW_PASTE2(a, b)
extern "C" __declspec(dllexport) PyObject *CUW_PASTE(PyInit_, TORCH_EXTENSION_NAME)(void) {
    PyErr_SetString(PyExc_ImportError,
                    "this torchao library registers torch ops and is loaded with "
                    "torch.ops.load_library; it is not an importable module");
    return nullptr;
}
#endif
''')
_t = _sp.read_text()
_old_ext = "    ext_modules = []\n"
require(_t.count(_old_ext) == 1, "torchao: the ext_modules initialiser was not found once")
_t = _t.replace(_old_ext, '''    # cuda-foundry: the PyInit stub every torch-ops library needs on Windows
    # (empty elsewhere); see packages/torchao/patches/torchao.py.
    _cuw_stub = os.path.join(extensions_dir, "cuw_pyinit_stub.cpp")
    sources.append(_cuw_stub)
    if cutlass_90a_sources:
        cutlass_90a_sources.append(_cuw_stub)
    if cutlass_100a_sources:
        cutlass_100a_sources.append(_cuw_stub)
''' + _old_ext, 1)
_ast.parse(_t)
_sp.write_text(_t)
print("torchao patch: PyInit stub attached to the torch-ops libraries")
