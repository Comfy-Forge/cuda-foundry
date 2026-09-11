"""Stop torchao installing its `test/` directory as a top-level package.

Ported from cuda-wheels (packages/torchao/patches/torchao.py) and re-read
against v0.13.0: setup.py's `find_packages(exclude=["benchmarks",
"benchmarks.*"])` still lets the repo's `test/` tree ship as `import test`,
which is CPython's own stdlib package name. Only the top-level tree is
excluded.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import exclude_top_level_packages, require  # noqa: E402

exclude_top_level_packages(["test"])
require('"test"' in pathlib.Path("setup.py").read_text(),
        "torchao: the test exclusion did not land in setup.py")
print("torchao patch: done")
