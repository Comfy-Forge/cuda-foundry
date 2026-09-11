"""Patch detectron2 v0.6 for the foundry build.

Ported from cuda-wheels (packages/detectron2/patches/detectron2.py) and
re-read against v0.6. Runs with cwd = the cloned source, from
scripts/fetch_patched_sources.py, and imports the shared helper by walking up
to scripts/ from its own location.

1. Stop detectron2 installing its `tools/` directory as a top-level package.
   setup.py's `find_packages(exclude=("configs", "tests*"))` still lets the
   repo's `tools/` training scripts ship as `import tools`, one flat
   site-packages name claimed for the whole interpreter. Only the top-level
   tree is excluded; a genuine `detectron2.tools` subpackage would be untouched.

2. Pillow 10 removed the `Image.LINEAR` alias (and ANTIALIAS / CUBIC), and
   detectron2/data/transforms/transform.py:46 uses it as a default argument
   -- so `import detectron2.data` raises AttributeError on every Pillow a
   current environment resolves. conda-forge's detectron2 feedstock carries
   the same one-line fix; `Image.BILINEAR` is the constant LINEAR aliased.
   The other constants the tree uses (NEAREST, BILINEAR, BICUBIC) survive in
   Pillow 10+, checked by grep over detectron2/ at this rev; the patch asserts
   none of the removed names remain so a future bump cannot reintroduce one
   silently. With this in, `pillow` needs no upper bound.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import exclude_top_level_packages, require  # noqa: E402

n = exclude_top_level_packages(["tools"])
require(n == 1 or "\"tools\"" in pathlib.Path("setup.py").read_text(),
        "detectron2: the tools exclusion did not land in setup.py")

# 2. Pillow >= 10 -----------------------------------------------------------
transform_py = pathlib.Path("detectron2/data/transforms/transform.py")
text = transform_py.read_text(encoding="utf-8")
if "interp=Image.BILINEAR, fill=0" in text and "interp=Image.LINEAR" not in text:
    print("detectron2 patch: Image.LINEAR already replaced")
else:
    require(text.count("interp=Image.LINEAR, fill=0") == 1,
            "detectron2: expected exactly one `interp=Image.LINEAR, fill=0` default in "
            "detectron2/data/transforms/transform.py -- upstream changed; re-read it")
    text = text.replace("interp=Image.LINEAR, fill=0",
                        "interp=Image.BILINEAR, fill=0", 1)
    transform_py.write_text(text, encoding="utf-8")
    print("detectron2 patch: Image.LINEAR -> Image.BILINEAR (Pillow 10 removed the alias)")

removed = re.compile(r"\bImage\.(LINEAR|ANTIALIAS|CUBIC)\b")
hits = [f"{p}:{i}" for p in sorted(pathlib.Path("detectron2").rglob("*.py"))
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if removed.search(line)]
require(not hits, f"detectron2: Pillow-10-removed Image constants still used at {hits}")
print("detectron2 patch: no Pillow-10-removed Image constant remains under detectron2/")
print("detectron2 patch: done")
