"""Patch pytorch3d v0.7.9 for the foundry build.

Ported from cuda-wheels (packages/pytorch3d/patches/pytorch3d.py) and re-read
against the pinned rev. Runs with cwd = the cloned source, once, BEFORE the
tarball is sealed -- so it must not depend on the cell (no torch, no CUDA
version): every cell's build consumes the same tree.

1. Stop hardcoding the C++ standard. setup.py pins `-std=c++17` in two places
   (extra_compile_args["cxx"] and, off Windows, nvcc_args). torch's
   cpp_extension appends the standard the INSTALLED torch needs only when the
   caller gave none, so an explicit pin overrides it: right for torch 2.8,
   wrong the day a torch needs C++20. Dropping it hands the choice to torch,
   on every platform (MSVC included -- pytorch3d's cxx list carries no other
   flag that would need an MSVC spelling).

2. `projects/` must not ship as a top-level package. find_packages excludes
   `projects.*` but not `projects` itself, so the bare top-level directory --
   a licence header and nothing else -- lands in site-packages under one of
   the most collidable names there is. The implicitron trainer is added
   separately with its own package_dir and is unaffected.

3. No console scripts. setup.py registers two entry points,
   pytorch3d_implicitron_runner and pytorch3d_implicitron_visualizer, into
   projects/implicitron_trainer. On win-64 pip renders each as a .exe
   launcher with the BUILD machine's interpreter path baked in
   (D:/a/_temp/.../python.exe, Windows-spelled -- verified on the published artifact), a
   path no user has; and what they import -- hydra-core, visdom, lpips,
   accelerate, sqlalchemy, the implicitron extras -- is not declared by this
   package either. Dropped rather than rendered through the recipe: the
   library is the artifact, the trainer CLI is upstream's research
   scaffolding (its package_dir is added separately in setup.py and stays;
   only the entry points go).

NOT carried from the farm: its CUDA >= 13 branch, which appended
`-static-global-template-stub=false` to NVCC_FLAGS via $GITHUB_ENV. That is a
build-time, per-cell decision (cuda_mm() is read from the environment on the
build machine there); here the patch runs once at fetch time with no cell in
view, and $GITHUB_ENV is not a channel into rattler-build's build script. The
flag only matters for CUDA 13.x (pulsar's explicitly-instantiated __global__
templates lose external linkage under 13's new default); this repo's first
cell is CUDA 12.8. When a 13.x cell is added, express it as a per-CUDA
package.yml knob, not as fetch-time code that cannot know the cell.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import exclude_top_level_packages, require, strip_std_flags  # noqa: E402

setup_file = pathlib.Path("setup.py")
content = setup_file.read_text()
content, n_std = strip_std_flags(content)
require(n_std == 2,
        f"pytorch3d: expected exactly 2 hardcoded C++-standard flags in setup.py "
        f"(cxx list + nvcc append), found {n_std} -- upstream changed; re-read "
        f"setup.py before building against an unverified flag set")
setup_file.write_text(content)
print(f"pytorch3d patch: dropped {n_std} hardcoded std flag(s); torch's "
      f"cpp_extension now selects the standard")

import re  # noqa: E402

content = setup_file.read_text()
if "entry_points=" not in content:
    print("pytorch3d patch: no entry_points block (already removed)")
else:
    content, n_ep = re.subn(
        r"    entry_points=\{\n        \"console_scripts\": \[\n(?:            .*\n)+?        \]\n    \},\n",
        "    # cuda-foundry: the two implicitron console scripts are not shipped\n"
        "    # (see packages/pytorch3d/patches/pytorch3d.py, item 3).\n",
        content)
    require(n_ep == 1 and "entry_points=" not in content and "console_scripts" not in content,
            "pytorch3d: expected exactly one entry_points={console_scripts: [...]} block "
            "in setup.py -- upstream changed; re-read it before dropping the scripts")
    import ast
    ast.parse(content)
    setup_file.write_text(content)
    print("pytorch3d patch: entry_points (implicitron console scripts) removed")

exclude_top_level_packages(["projects"])
require('"projects"' in setup_file.read_text(),
        "pytorch3d: the projects exclusion did not land in setup.py")
print("pytorch3d patch: done")
