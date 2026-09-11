"""Stop detectron2 installing its `tools/` directory as a top-level package.

Ported from cuda-wheels (packages/detectron2/patches/detectron2.py), re-read
against v0.6: setup.py's `find_packages(exclude=("configs", "tests*"))` still
lets the repo's `tools/` training scripts ship as `import tools`, one flat
site-packages name claimed for the whole interpreter. Only the top-level tree
is excluded; a genuine `detectron2.tools` subpackage would be untouched.

Runs with cwd = the cloned source, from scripts/fetch_patched_sources.py, and
imports the shared helper by walking up to scripts/ from its own location.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import exclude_top_level_packages, require  # noqa: E402

n = exclude_top_level_packages(["tools"])
require(n == 1 or "\"tools\"" in pathlib.Path("setup.py").read_text(),
        "detectron2: the tools exclusion did not land in setup.py")
print("detectron2 patch: done")
