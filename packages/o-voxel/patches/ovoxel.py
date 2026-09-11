"""Patch o-voxel (microsoft/TRELLIS.2 @ 5565d24, subdirectory o-voxel/).

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked
repository, BEFORE the tarball is sealed, on a Linux host with no torch. One
tarball feeds both platforms, so every edit is unconditional; the ones the
farm applied only on a Windows fetch host are either applied to both (they
are valid C++ on gcc too) or dropped and explained. Ported from the farm's
packages/ovoxel/patches/ovoxel.py; the shared steps live in
ovoxel_common.py beside this file and are used by the two forks as well.

  1. hoist o-voxel/ to the source root (the farm's `build_subdir`), carrying
     the repository's MIT LICENSE in; vendor Eigen 3.4.0 (sha256-pinned
     tarball) at third_party/eigen, the gitlab submodule path setup.py
     hardcodes -- the farm stopped cloning that submodule after gitlab 403s.
  2. pyproject.toml: cumesh / flex_gemm are declared as `git+https://` URLs,
     which no index can satisfy and pip refuses under --no-index. Reduced to
     bare names. The .conda's run: and the wheel's PEP 658 sidecar carry the
     real dependency (package.yml run_deps), not this file.
  3. postprocess.py: batch the BVH unsigned_distance queries (500k points at
     a time) so one call cannot outrun the display driver's kernel timeout.
  4. MSVC: `1e-6d`-style double literals and size_t narrowing in
     initialiser lists, applied to both platforms (8 sites at 5565d24).
  5. The C++ standard: two `-std=c++17` literals dropped; torch picks it.
     The farm's -O3 -> /O2 translation is not ported: it ran only on a
     Windows fetch host and is cosmetic (torch's MSVC ninja path already
     puts distutils' /O2 on every cl line; the -O3 draws warning D9002).
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ovoxel_common import (assert_arch_list_authoritative,  # noqa: E402
                           batched_bvh_queries, hoist_and_vendor_eigen,
                           msvc_source_fixes, require, torch_selects_std)

TAG = "o_voxel"

hoist_and_vendor_eigen(TAG)

# ── 2. git-URL dependencies -> names ─────────────────────────────────────
pyproject = pathlib.Path("pyproject.toml")
t = pyproject.read_text()
n = 0
for name in ("cumesh", "flex_gemm"):
    t, k = re.subn(rf'"{name}\s*@\s*git\+[^"]*"', f'"{name}"', t)
    n += k
if n:
    require(n == 2, f"{TAG}: rewrote {n} git-URL dependency(ies), expected 2")
    pyproject.write_text(t)
    print(f"{TAG} patch: {n} git-URL dependencies reduced to bare names")
else:
    require('"cumesh"' in t and '"flex_gemm"' in t,
            f"{TAG}: pyproject.toml names neither cumesh nor flex_gemm -- upstream changed")
    print(f"{TAG} patch: git-URL dependencies already reduced")
require("git+" not in pyproject.read_text(), f"{TAG}: a git+ URL survives in pyproject.toml")

# ── 3. batched BVH queries ───────────────────────────────────────────────
batched_bvh_queries(TAG, pathlib.Path("o_voxel/postprocess.py"), "import cumesh\n")

# ── 4. MSVC source fixes ─────────────────────────────────────────────────
msvc_source_fixes(TAG, expect_applied_upstream=False)

# ── 5. C++ standard ──────────────────────────────────────────────────────
torch_selects_std(TAG, expect=2)
assert_arch_list_authoritative(TAG)
require(pathlib.Path("LICENSE").is_file(), f"{TAG}: LICENSE was not carried into the hoisted tree")
print(f"{TAG} patch: done")
