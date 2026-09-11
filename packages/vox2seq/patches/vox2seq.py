"""Patch vox2seq (microsoft/TRELLIS @ f17fdf1, subdirectory extensions/vox2seq/).

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked
repository, BEFORE the tarball is sealed. The farm needed no source patch for
this package (no flags, no arch probing, no platform code); the one thing it
did declare, `build_subdir: extensions/vox2seq`, has no counterpart in this
repo's build script, so the patch step hoists the subdirectory to the source
root instead (patch_lib.hoist_subdir), carrying the repository's MIT LICENSE
in with it.

Asserted rather than assumed: setup.py emits no arch flag and reads no GPU,
so the cell's TORCH_CUDA_ARCH_LIST is authoritative; and it passes no
extra_compile_args at all, so there is no standard to strip.
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import hoist_subdir, require  # noqa: E402

hoist_subdir("extensions/vox2seq")
setup = pathlib.Path("setup.py")
require(setup.is_file() and pathlib.Path("vox2seq/__init__.py").is_file(),
        "vox2seq: after hoisting, setup.py / vox2seq/ are not at the root")
require(pathlib.Path("LICENSE").is_file(), "vox2seq: LICENSE was not carried in")
text = setup.read_text()
require("extra_compile_args" not in text and "std=c++" not in text,
        "vox2seq: setup.py now passes compiler flags -- re-read it for a "
        "hardcoded C++ standard before building")
for lineno, line in enumerate(text.splitlines(), 1):
    if re.search(r"-arch[= ]|gencode|get_device_capability|TORCH_CUDA_ARCH_LIST", line):
        sys.exit(f"vox2seq patch: setup.py:{lineno} touches the arch list "
                 f"({line.strip()!r}) -- re-check")
print("vox2seq patch: hoisted; no flags, no arch probing; done")
