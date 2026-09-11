"""Patch torch_cluster: BFloat16 dispatch, drop the CPU twin, build with ninja.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed -- so no `import torch`, and no platform
conditionals (one tarball serves every platform's build).

1. BFloat16 (ported from cuda-wheels). The CUDA kernels dispatch for the
   floating/all types plus Half only, so bf16 models hit NotImplementedError.
   radius_cuda.cu already carries BFloat16; fps, knn, nearest, grid and
   graclus do not. `_AND(Half,` -> `_AND2(Half, BFloat16,` -- six sites in
   five files at the pinned rev, asserted exactly.

2. FORCE_ONLY_CUDA (see patch_lib.force_only_cuda): the `_<name>_cpu`
   extension twin is never loaded when the cuda one is present and is not a
   fallback -- the cuda build compiles csrc/cpu/*.cpp too.

3. use_ninja. setup.py pins use_ninja=False. Not optional here: linux-64
   needs ninja for MAX_JOBS to mean anything, and win-64 reads the compile
   ledger (L3) out of .ninja_log -- build_win.py refuses to ship an extension
   ninja did not build. See packages/torch-scatter/patches/torch_scatter.py.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import force_only_cuda, require  # noqa: E402

old_d = "_AND(at::ScalarType::Half,"
new_d = "_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,"
patched, files = 0, 0
for cu in sorted(pathlib.Path("csrc/cuda").glob("*.cu")):
    if cu.name == "radius_cuda.cu":
        require("at::ScalarType::BFloat16" in cu.read_text(encoding="utf-8"),
                "torch_cluster: radius_cuda.cu no longer dispatches BFloat16 "
                "-- the 'already supported' assumption is stale")
        continue
    text = cu.read_text(encoding="utf-8")
    n = text.count(old_d)
    if n:
        cu.write_text(text.replace(old_d, new_d), encoding="utf-8")
        patched += n
        files += 1
        print(f"torch_cluster patch: {cu}: {n} dispatch site(s) -> +BFloat16")
require((patched, files) == (6, 5),
        f"torch_cluster: expected 6 Half-only dispatch sites in 5 files "
        f"(fps, knn, nearest, grid, graclus), found {patched} in {files} -- "
        f"upstream changed; re-check")

force_only_cuda("setup.py")

setup_py = pathlib.Path("setup.py")
text = setup_py.read_text(encoding="utf-8")
old = "BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)"
new = "BuildExtension.with_options(no_python_abi_suffix=True)"
require(text.count(old) == 1,
        f"torch_cluster: expected exactly one {old!r} in setup.py, found "
        f"{text.count(old)} -- upstream changed; re-check before building")
setup_py.write_text(text.replace(old, new, 1), encoding="utf-8")
require("use_ninja" not in setup_py.read_text(encoding="utf-8"),
        "torch_cluster: use_ninja=False is still in setup.py on disk")
print("torch_cluster patch: use_ninja=False dropped -- ninja builds on both platforms")
