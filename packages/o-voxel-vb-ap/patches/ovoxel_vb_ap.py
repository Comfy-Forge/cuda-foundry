"""Patch o-voxel-vb-ap (PozzettiAndrea/Trellis.2.drtk @ 200ba8c, subdirectory
o-voxel/) into the o_voxel_vb_ap package.

Same shape as packages/o-voxel/patches/ovoxel.py (whose ovoxel_common.py this
imports). This fork is the visualbruno tree with the nvdiffrast-dependent
half removed: no postprocess.py, no rasterize.py, rasterize.cu left out of
the build -- so there is no BVH batching to apply, no sibling-import rewrite, and the
package imports without cumesh, flex_gemm, nvdiffrast or OpenCV (see
package.yml run_deps). It is renamed to o_voxel_vb_ap so it installs and
imports beside the other two. The MSVC source fixes are already in the fork
(asserted); the standard is pinned c++17 here (two sites), dropped so torch
picks it.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "o-voxel" / "patches"))
from ovoxel_common import (assert_arch_list_authoritative,  # noqa: E402
                           hoist_and_vendor_eigen, msvc_source_fixes,
                           rename_package, require, torch_selects_std)

TAG = "o_voxel_vb_ap"

hoist_and_vendor_eigen(TAG)
require("git+" not in pathlib.Path("pyproject.toml").read_text(),
        f"{TAG}: pyproject.toml carries a git+ dependency; this fork was "
        f"expected to have none")
pkg = rename_package(TAG, "o_voxel_vb_ap")
for gone in ("postprocess.py", "rasterize.py"):
    require(not (pkg / gone).exists(),
            f"{TAG}: {gone} is back in this fork -- it would import nvdiffrast/"
            f"cumesh/flex_gemm at load time and run_deps no longer cover that")
# src/rasterize/rasterize.cu is still on disk in this fork; what matters is
# that setup.py does not compile it and no Python module reaches for it.
require("rasterize.cu" not in pathlib.Path("setup.py").read_text(),
        f"{TAG}: the rasterizer is back in this fork's build -- re-check run_deps")
msvc_source_fixes(TAG, expect_applied_upstream=True)
torch_selects_std(TAG, expect=2)
assert_arch_list_authoritative(TAG)
require(pathlib.Path("LICENSE").is_file(), f"{TAG}: LICENSE was not carried into the hoisted tree")
print(f"{TAG} patch: done")
