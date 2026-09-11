"""Patch torch_scatter: drop the never-loadable CPU twin, and build with ninja.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed -- so no `import torch`, and no platform
conditionals (one tarball serves every platform's build).

1. FORCE_ONLY_CUDA (ported from cuda-wheels, see patch_lib.force_only_cuda).
   setup.py builds every extension twice, `_<name>_cpu` and `_<name>_cuda`,
   via product(main_files, suffices). The facade loads `cuda_spec or
   cpu_spec`, so with both present the CPU twin is never loaded, and it is
   not a fallback either: the cuda build compiles csrc/cpu/*.cpp as well, so
   the CUDA library is a strict superset. Upstream's own switch collapses it.

2. use_ninja. setup.py pins `BuildExtension.with_options(..., use_ninja=False)`
   -- a 2020 decision (commit 05baf9b, "do not use ninja", no reason given,
   torch 1.5 era). Here it is not a preference:

     * linux-64: without ninja torch's BuildExtension falls back to distutils
       and compiles every translation unit SERIALLY, ignoring MAX_JOBS
       (docs/ARCHITECTURE.md, "ninja is required in build:").
     * win-64: the compile ledger (L3 of the from-source guarantee) is read
       out of ninja's own .ninja_log, because nothing sits in the nvcc seat on
       Windows. A distutils build leaves no log, and build_win.py then FAILS
       the artifact: "ninja compiled no object files -- this build did not
       compile what it is shipping". So without this the package cannot
       publish on win-64 at all, by design.

   Dropping the option restores torch's default (ninja when available, which
   the recipe's build: guarantees). Asserted per substitution, not by a
   whole-file before/after guard -- a partial match must not read as success
   (docs/WINDOWS.md, the fused-ssim lesson).
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
        f"torch_scatter: expected exactly one {old!r} in setup.py, found "
        f"{text.count(old)} -- upstream changed; re-check before building")
setup_py.write_text(text.replace(old, new, 1), encoding="utf-8")
require(new in setup_py.read_text(encoding="utf-8") and "use_ninja" not in setup_py.read_text(encoding="utf-8"),
        "torch_scatter: use_ninja=False is still in setup.py on disk")
print("torch_scatter patch: use_ninja=False dropped -- ninja builds on both platforms")
