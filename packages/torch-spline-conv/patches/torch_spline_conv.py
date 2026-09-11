"""Patch torch_spline_conv: drop the CPU twin, build with ninja.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed -- so no `import torch`, and no platform
conditionals (one tarball serves every platform's build).

1. FORCE_ONLY_CUDA (ported from cuda-wheels, see patch_lib.force_only_cuda):
   same shape as torch_scatter -- setup.py builds `_<name>_cpu` and
   `_<name>_cuda` from product(main_files, suffices), the facade prefers the
   cuda spec, and the cuda build already compiles the CPU sources, so the
   `_cpu` twin is compiled and shipped but can never be loaded.

2. use_ninja. setup.py pins use_ninja=False. Not optional here: linux-64
   needs ninja for MAX_JOBS to mean anything, and win-64 reads the compile
   ledger (L3) out of .ninja_log -- build_win.py refuses to ship an extension
   ninja did not build. See packages/torch-scatter/patches/torch_scatter.py.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import force_only_cuda, require  # noqa: E402

force_only_cuda("setup.py")

setup_py = pathlib.Path("setup.py")
text = setup_py.read_text(encoding="utf-8")
old = "BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)"
new = "BuildExtension.with_options(no_python_abi_suffix=True)"
require(text.count(old) == 1,
        f"torch_spline_conv: expected exactly one {old!r} in setup.py, found "
        f"{text.count(old)} -- upstream changed; re-check before building")
setup_py.write_text(text.replace(old, new, 1), encoding="utf-8")
require("use_ninja" not in setup_py.read_text(encoding="utf-8"),
        "torch_spline_conv: use_ninja=False is still in setup.py on disk")
print("torch_spline_conv patch: use_ninja=False dropped -- ninja builds on both platforms")
