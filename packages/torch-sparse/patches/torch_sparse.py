"""Patch torch_sparse: BFloat16 dispatch, drop the CPU twin, build with ninja.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed -- so no `import torch`, and no platform
conditionals (one tarball serves every platform's build).

1. BFloat16 (ported from cuda-wheels). spmm_cuda.cu dispatches for the
   floating types plus Half only; models running in bf16 hit
   NotImplementedError. `_AND(Half,` -> `_AND2(Half, BFloat16,` at every
   dispatch site. Two sites at the pinned rev, both in spmm_cuda.cu; a count
   of zero is an error, not a "may already be supported".

2. FORCE_ONLY_CUDA (see patch_lib.force_only_cuda): the `_<name>_cpu`
   extension twin is never loaded when the cuda one is present and is not a
   fallback -- the cuda build compiles csrc/cpu/*.cpp too. 12 extensions
   instead of 24.

3. use_ninja -- ASSERTED, not patched. Unlike its three siblings (scatter,
   cluster, spline_conv), torch_sparse's setup.py at this rev does NOT pin
   use_ninja=False: it reads `BuildExtension.with_options(no_python_abi_suffix=True)`
   and torch's default is ninja. The sibling substitution was tried here
   first and matched nothing, which is the good failure. What is kept is the
   assertion that this stays true, because win-64 reads the compile ledger
   (L3) out of .ninja_log and build_win.py refuses to ship an extension ninja
   did not build. See packages/torch-scatter/patches/torch_scatter.py.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import force_only_cuda, require  # noqa: E402

old_d = "_AND(at::ScalarType::Half,"
new_d = "_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,"
patched = 0
for cu in sorted(pathlib.Path("csrc/cuda").glob("*.cu")):
    text = cu.read_text(encoding="utf-8")
    n = text.count(old_d)
    if n:
        cu.write_text(text.replace(old_d, new_d), encoding="utf-8")
        patched += n
        print(f"torch_sparse patch: {cu}: {n} dispatch site(s) -> +BFloat16")
require(patched == 2,
        f"torch_sparse: expected 2 Half-only dispatch sites in csrc/cuda "
        f"(spmm_cuda.cu), found {patched} -- upstream changed; re-check")

force_only_cuda("setup.py")

setup_py = pathlib.Path("setup.py")
text = setup_py.read_text(encoding="utf-8")
want = "BuildExtension.with_options(no_python_abi_suffix=True)"
require(text.count(want) == 1 and "use_ninja" not in text,
        "torch_sparse: setup.py no longer builds with torch's default (ninja) "
        "-- a use_ninja=False would leave win-64 with no .ninja_log and no "
        "compile ledger; drop it as torch_scatter's patch does")
print("torch_sparse patch: setup.py builds with ninja (asserted, nothing to drop)")
