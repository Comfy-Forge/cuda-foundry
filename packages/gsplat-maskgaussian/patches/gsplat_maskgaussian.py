"""Patch HY-World 2.0's gsplat_maskgaussian fork into a buildable tree.

Runs on the fetching machine with cwd = the HY-World-2.0 checkout, BEFORE the
tarball is sealed, and produces ONE tarball for every platform: no
`import torch`, no platform conditionals (cuda-wheels' version of this patch
gates the MSVC flag translation on os.name; that condition is moved into
setup.py here, which runs on the build machine).

Five edits, each asserted on its own:

1. RELOCATE. The package lives at hyworld2/worldgen/third_party/
   gsplat_maskgaussian inside a 500 MB research repo (the farm's
   `build_subdir`). This repo's build runs `pip wheel .` at the tarball root,
   so that subdirectory is moved to the root and everything else is deleted.
   The repo's License.txt is copied in first, since the subdirectory carries
   no licence of its own.
2. RENAME the distribution: setup.py says name="gsplat", which would publish
   under the vanilla package's name and version. It becomes
   gsplat_maskgaussian. The import name stays `gsplat` -- HY-World's own code
   imports it that way -- so the two variants share files and only one may be
   installed at a time.
3. glm. HY-World stripped the g-truc/glm submodule out of the fork, but
   setup.py still puts gsplat/cuda/csrc/third_party/glm on the include path
   and Common.h includes glm/gtc/type_ptr.hpp. Cloned back in, pinned to the
   exact commit vanilla gsplat v1.5.3's submodule points at
   (33b4a621a697a305bc3a7610d290677b96beb181) rather than a tag: same headers
   as the package this is a fork of, and a commit cannot be moved. The docs,
   tests and .git are pruned afterwards; the 426 headers stay because the JIT
   fallback in cuda/_backend.py includes them.
4. MSVC host flags / the C++ standard / cg::labeled_partition -- the same three
   fixes as packages/gsplat/patches/gsplat.py, for the same reasons: the
   fork's setup.py and csrc/ are vanilla gsplat's at these lines.
"""

import glob
import pathlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import (guard_labeled_partition_in_files, prune_glm_docs,  # noqa: E402
                       require, strip_std_flags)

SUBDIR = Path("hyworld2/worldgen/third_party/gsplat_maskgaussian")
REPO_LICENSE = Path("License.txt")
GLM_REPO = "https://github.com/g-truc/glm.git"
GLM_REV = "33b4a621a697a305bc3a7610d290677b96beb181"   # gsplat v1.5.3's submodule
GLM_DIR = Path("gsplat/cuda/csrc/third_party/glm")
MARKER = "# patched-by: cuda-foundry gsplat_maskgaussian"


def relocate() -> None:
    """Move the subproject to the root and drop the rest of the checkout."""
    if not SUBDIR.is_dir():
        # Already relocated (re-run), or upstream moved it. Only the first is
        # acceptable, and it is recognisable by the marker the rename leaves.
        require(Path("setup.py").is_file()
                and MARKER in Path("setup.py").read_text(encoding="utf-8"),
                f"{SUBDIR} is not in the checkout and the root is not an "
                f"already-relocated tree -- upstream moved the package; "
                f"re-read the repo layout at this rev")
        print("gsplat_maskgaussian patch: already relocated")
        return
    require(REPO_LICENSE.is_file(),
            f"{REPO_LICENSE} missing at the repo root -- the licence must ship "
            f"with the relocated tree")
    shutil.copy2(REPO_LICENSE, SUBDIR / "LICENSE.HY-World-2.0.txt")
    staging = Path(".cuw-relocate")
    shutil.move(str(SUBDIR), str(staging))
    removed = 0
    for entry in Path(".").iterdir():
        if entry.name in (".git", staging.name):
            continue
        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        removed += 1
    for entry in list(staging.iterdir()):
        shutil.move(str(entry), str(Path(".") / entry.name))
    staging.rmdir()
    require(Path("setup.py").is_file() and Path("gsplat/version.py").is_file(),
            "relocation did not leave setup.py and gsplat/version.py at the root")
    print(f"gsplat_maskgaussian patch: relocated {SUBDIR} to the root, "
          f"dropped {removed} top-level entr(ies)")


def patch_setup_py() -> None:
    setup_file = Path("setup.py")
    content = setup_file.read_text(encoding="utf-8")
    if MARKER in content:
        print("gsplat_maskgaussian patch: setup.py already patched")
        return
    content, n_name = re.subn(r'name="gsplat"', 'name="gsplat_maskgaussian"', content)
    require(n_name == 1,
            f'expected exactly one name="gsplat" in setup.py, found {n_name} -- '
            f"upstream changed; the distribution would publish under the "
            f"vanilla gsplat name")
    content, n_cxx = re.subn(
        r'extra_compile_args = \{"cxx": \["-O3"\]\}',
        'extra_compile_args = {"cxx": (["/O2"] if os.name == "nt" else ["-O3"])}'
        '  # patched: cl.exe ignores -O3 (D9002) and ships unoptimised',
        content)
    require(n_cxx == 1,
            f'expected exactly one extra_compile_args = {{"cxx": ["-O3"]}} in '
            f"setup.py, found {n_cxx} -- upstream moved the block")
    content, n_std = strip_std_flags(content)
    require(n_std > 0, "no hardcoded -std flag in setup.py -- upstream changed; "
                       "refusing to build against an unverified flag set")
    for lineno, line in enumerate(content.splitlines(), 1):
        if "nvcc_flags" in line and ("-arch" in line or "-gencode" in line):
            sys.exit(f"gsplat_maskgaussian patch: setup.py:{lineno} puts an arch "
                     f"flag into nvcc_flags: {line.strip()!r}")
    setup_file.write_text(MARKER + "\n" + content, encoding="utf-8")
    print(f"gsplat_maskgaussian patch: renamed distribution, cxx flags made "
          f"MSVC-aware, dropped {n_std} hardcoded std flag(s)")


def fetch_glm() -> None:
    if (GLM_DIR / "glm" / "glm.hpp").is_file():
        print(f"gsplat_maskgaussian patch: glm already present at {GLM_DIR}")
        return
    if GLM_DIR.exists():
        shutil.rmtree(GLM_DIR)
    GLM_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"gsplat_maskgaussian patch: cloning glm @ {GLM_REV} -> {GLM_DIR}")
    subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
                    GLM_REPO, str(GLM_DIR)], check=True)
    subprocess.run(["git", "checkout", "-q", "--detach", GLM_REV],
                   cwd=GLM_DIR, check=True)
    got = subprocess.run(["git", "rev-parse", "HEAD"], cwd=GLM_DIR,
                         capture_output=True, text=True, check=True).stdout.strip()
    require(got == GLM_REV, f"glm checkout is at {got}, expected {GLM_REV}")
    require((GLM_DIR / "glm" / "gtc" / "type_ptr.hpp").is_file(),
            f"glm cloned but glm/gtc/type_ptr.hpp is not where setup.py's "
            f"include path and Common.h expect it")


def guard_partition() -> None:
    n_lp = guard_labeled_partition_in_files(
        sorted(glob.glob("gsplat/cuda/csrc/*.cu")), required=False)
    if n_lp:
        print(f"gsplat_maskgaussian patch: guarded {n_lp} labeled_partition site(s)")
        return
    already = any("cg::tiled_partition<1>" in Path(f).read_text(encoding="utf-8")
                  for f in glob.glob("gsplat/cuda/csrc/*.cu"))
    require(already, "no cg::labeled_partition sites found and none already "
                     "guarded -- upstream changed; re-read the sources")
    print("gsplat_maskgaussian patch: labeled_partition sites already guarded")


def main() -> None:
    relocate()
    patch_setup_py()
    fetch_glm()
    guard_partition()
    prune_glm_docs(GLM_DIR)     # removes .git too (it is in _GLM_PRUNE_DIRS)
    print("gsplat_maskgaussian patch: done")


if __name__ == "__main__":
    main()
