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

Two more, found by the audit of the published artifact:

  * The licence. The repo's UNLICENSE (public-domain dedication) sits at the
    repo ROOT, outside pointnet2_ops_lib/, so neither the wheel's dist-info
    nor the conda info/licenses got any text at all. It is copied to
    pointnet2_ops_lib/LICENSE -- the repo's own file, byte for byte, under a
    name setuptools' default license_files glob (LICEN[CS]E*) picks up --
    and package.yml's license_files points at that copy.
  * The payload. MANIFEST.in `graft pointnet2_ops/_ext-src` plus
    `include_package_data=True` shipped the 15 C++/CUDA source files inside
    the wheel next to the compiled _ext. They are the build's input, not
    the package's content; include_package_data is switched off and the
    graft removed, so only the Python modules and the extension ship.
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
MARKER = "# cuda-foundry: the hardcoded TORCH_CUDA_ARCH_LIST assignment was removed here\n"
patched, n = pattern.subn(MARKER, text)
require(n == 1 or (n == 0 and MARKER in text),
        f"pointnet2_ops: expected exactly 1 TORCH_CUDA_ARCH_LIST assignment "
        f"in {setup_py}, found {n} -- upstream changed; re-check before building")
setup_py.write_text(patched, encoding="utf-8")
require("os.environ[" not in setup_py.read_text(encoding="utf-8"),
        "pointnet2_ops: the hardcoded arch list is still in setup.py on disk")
print(f"pointnet2_ops patch: removed hardcoded TORCH_CUDA_ARCH_LIST from {setup_py}")

# ── the licence text, into the tree the wheel is built from ───────────────
import shutil  # noqa: E402

unlicense = pathlib.Path("UNLICENSE")
require(unlicense.is_file(), "pointnet2_ops: UNLICENSE is not at the repo root any more")
require("This is free and unencumbered software released into the public domain"
        in unlicense.read_text(encoding="utf-8"),
        "pointnet2_ops: UNLICENSE is not the Unlicense text")
dst = pathlib.Path("pointnet2_ops_lib/LICENSE")
if not dst.is_file() or dst.read_bytes() != unlicense.read_bytes():
    shutil.copyfile(unlicense, dst)
    print(f"pointnet2_ops patch: copied UNLICENSE -> {dst}")
else:
    print(f"pointnet2_ops patch: {dst} already in place")

# ── sources out of the payload ────────────────────────────────────────────
text = setup_py.read_text(encoding="utf-8")
if "include_package_data=False" in text:
    print("pointnet2_ops patch: include_package_data already off")
else:
    require(text.count("include_package_data=True,") == 1,
            "pointnet2_ops: expected exactly one include_package_data=True in setup.py")
    text = text.replace("include_package_data=True,",
                        "include_package_data=False,  # cuda-foundry: _ext-src is build input, not payload", 1)
    setup_py.write_text(text, encoding="utf-8")
    print("pointnet2_ops patch: include_package_data=False (no _ext-src/ in the wheel)")
manifest = pathlib.Path("pointnet2_ops_lib/MANIFEST.in")
if manifest.is_file():
    lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines()
             if "_ext-src" not in ln]
    if lines:
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        manifest.unlink()
    print("pointnet2_ops patch: MANIFEST.in graft of _ext-src removed")
require("_ext-src" not in (manifest.read_text(encoding="utf-8") if manifest.is_file() else ""),
        "pointnet2_ops: MANIFEST.in still grafts _ext-src")
