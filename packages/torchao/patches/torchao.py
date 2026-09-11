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
_t = _t.replace(_old_win, _old_win + '''
    else:
        # cuda-foundry: what CUTLASS 3.x needs from MSVC, on both the host
        # compile and nvcc's host pass.
        _cuw_msvc = ["/Zc:__cplusplus", "/bigobj", "/Zc:preprocessor", "/permissive-"]
        extra_compile_args["cxx"].extend(_cuw_msvc)
        extra_compile_args["nvcc"].extend(f"-Xcompiler={f}" for f in _cuw_msvc)''', 1)
_sp.write_text(_t)
import ast as _ast
_ast.parse(_t)
print("torchao patch: CUTLASS include dirs and MSVC flags enabled on Windows")
