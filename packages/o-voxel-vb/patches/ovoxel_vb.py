"""Patch o-voxel-vb (visualbruno/TRELLIS.2 @ 65d1e13, subdirectory o-voxel/)
into the o_voxel_vb package.

Same shape as packages/o-voxel/patches/ovoxel.py (whose ovoxel_common.py this
imports); the differences are the fork's:

  * renamed to o_voxel_vb everywhere (pyproject, setup.py, the package
    directory, its absolute self-imports) so it installs and imports beside
    plain o_voxel;
  * postprocess.py prefers the fork family's siblings and falls back to the
    plain ones: `cumesh_vb` else `cumesh`, `flex_gemm_vb` else `flex_gemm`.
    package.yml declares the fork siblings; the fallback is what lets an env
    with only the plain ones still import;
  * the fork already carries the MSVC source fixes and has no git-URL deps in
    pyproject.toml -- both asserted rather than re-applied;
  * upstream pins `-std=c++20` here (two sites), not c++17.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "o-voxel" / "patches"))
from ovoxel_common import (assert_arch_list_authoritative,  # noqa: E402
                           batched_bvh_queries, hoist_and_vendor_eigen,
                           msvc_source_fixes, rename_package, require,
                           torch_selects_std)

TAG = "o_voxel_vb"

hoist_and_vendor_eigen(TAG)
require("git+" not in pathlib.Path("pyproject.toml").read_text(),
        f"{TAG}: pyproject.toml carries a git+ dependency; this fork was "
        f"expected to have none")
pkg = rename_package(TAG, "o_voxel_vb")

# ── sibling forks first, plain packages as fallback ──────────────────────
post = pkg / "postprocess.py"
t = post.read_text()
if "import cumesh_vb as cumesh" in t:
    print(f"{TAG} patch: flexible sibling imports already applied")
else:
    for old, new in (
        ("from flex_gemm.ops.grid_sample import grid_sample_3d\n",
         "try:\n    from flex_gemm_vb.ops.grid_sample import grid_sample_3d\n"
         "except ImportError:\n    from flex_gemm.ops.grid_sample import grid_sample_3d\n"),
        ("import cumesh\n",
         "try:\n    import cumesh_vb as cumesh\nexcept ImportError:\n    import cumesh\n"),
    ):
        require(t.count(old) == 1, f"{TAG}: postprocess.py: expected exactly one "
                                   f"{old.strip()!r} -- the fork changed")
        t = t.replace(old, new, 1)
    post.write_text(t)
    print(f"{TAG} patch: postprocess.py prefers cumesh_vb / flex_gemm_vb, falls back to plain")

batched_bvh_queries(TAG, post, "except ImportError:\n    import cumesh\n")
msvc_source_fixes(TAG, expect_applied_upstream=True)
torch_selects_std(TAG, expect=2)
assert_arch_list_authoritative(TAG)
require(pathlib.Path("LICENSE").is_file(), f"{TAG}: LICENSE was not carried into the hoisted tree")
print(f"{TAG} patch: done")
