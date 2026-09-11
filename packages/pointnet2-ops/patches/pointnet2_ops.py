"""Let the cell's TORCH_CUDA_ARCH_LIST through to pointnet2_ops.

Ported from cuda-wheels. Run by scripts/fetch_patched_sources.py with cwd set
to the unpacked source ROOT (the repo), before the tarball is sealed; the
build itself happens in pointnet2_ops_lib/ (package.yml `build_subdir`).

Upstream setup.py hardcodes, at import time:

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"

Two problems. It clobbers the arch list the recipe exports, so every artifact
would be compiled for a 2021 GPU set regardless of the cell. And sm_37 (Kepler
K80) was removed in CUDA 12.0, so under cu12.x nvcc fails outright with
"Unsupported gpu architecture 'compute_37'" before compiling a single file.

Deleting the line is the whole fix: setup.py then leaves TORCH_CUDA_ARCH_LIST
alone and torch's BuildExtension picks up the cell's. Exactly one assignment
is expected; anything else is upstream having moved and is an error.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require  # noqa: E402

setup_py = pathlib.Path("pointnet2_ops_lib/setup.py")
require(setup_py.is_file(), "pointnet2_ops: pointnet2_ops_lib/setup.py not found "
                            "-- the build_subdir layout changed")
text = setup_py.read_text(encoding="utf-8")
pattern = re.compile(r'^os\.environ\[\s*["\']TORCH_CUDA_ARCH_LIST["\']\s*\]\s*=.*\n',
                     re.MULTILINE)
patched, n = pattern.subn("", text)
require(n == 1, f"pointnet2_ops: expected exactly 1 TORCH_CUDA_ARCH_LIST assignment "
                f"in {setup_py}, found {n} -- upstream changed; re-check before building")
setup_py.write_text(patched, encoding="utf-8")
require("TORCH_CUDA_ARCH_LIST" not in setup_py.read_text(encoding="utf-8"),
        "pointnet2_ops: the hardcoded arch list is still in setup.py on disk")
print(f"pointnet2_ops patch: removed hardcoded TORCH_CUDA_ARCH_LIST from {setup_py}")
