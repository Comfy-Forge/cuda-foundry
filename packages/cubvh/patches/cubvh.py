"""Patch cubvh: let torch's cpp_extension choose the C++ standard.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed and on a Linux host with no torch importable --
so, unlike the wheel farm's version of this patch, nothing here may branch on
`os.name` or on a torch version. One tarball feeds both platforms' builds.

What the farm did, and why it is not ported as-is:

  * `cpp_standard = 17 -> 20`, gated on `os.name != "nt" or torch >= 2.13`.
    At fetch time os.name is always posix, so porting that gate would bake
    C++20 into the WINDOWS build too -- the exact case the farm's gate
    existed to prevent. And torch 2.8 neither needs nor wants C++20; the
    bump existed for torch 2.13's headers.
  * `strip_permissive_for_old_cuda`: Windows + CUDA < 12.6 only, and again
    gated on facts the fetch step cannot know. Not applicable to cu12.8.

What this does instead: delete every hardcoded C++-standard flag and let
torch's BuildExtension append the one the INSTALLED torch needs (c++17 for
torch 2.8, in both `-std=` and `/std:` spellings -- cpp_extension.py's
append_std17_if_no_std_present, measured in v2.8.0). This is the same policy
flash-attn's patch applies here, and it makes the source correct for either
platform from one tarball.

The farm's patch_lib.strip_std_flags cannot do it: upstream spells the flags
as f-strings (`f"-std=c++{cpp_standard}"`), which its literal-only regex never
matches. Each removal below is asserted individually, because a whole-file
before/after guard passes on a partial match (the fused-ssim lesson).
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require  # noqa: E402

MARKER = "# patched-by: cuda-foundry cubvh std-flag strip"

setup = pathlib.Path("setup.py")
text = setup.read_text()

if MARKER in text:
    print("cubvh patch: setup.py already patched")
    sys.exit(0)

# The four f-string spellings at 7855c00: two `-std=` (nvcc shared, POSIX cxx),
# one `/std:` (Windows cxx) and one `-Xcompiler=/std:` (Windows nvcc). Each is
# a whole list element on its own line, so the line is removed.
removals = {
    "nvcc -std=c++N (shared list)": r'^[ \t]*f"-std=c\+\+\{cpp_standard\}",[ \t]*\n',
    "MSVC /std:c++N (cxx list)":     r'^[ \t]*f"/std:c\+\+\{cpp_standard\}",[ \t]*\n',
    "MSVC -Xcompiler=/std:c++N":     r'^[ \t]*f"-Xcompiler=/std:c\+\+\{cpp_standard\}",[ \t]*\n',
}
counts = {}
for what, pat in removals.items():
    text, n = re.subn(pat, "", text, flags=re.M)
    counts[what] = n
# The shared nvcc list and the POSIX cxx list both carry the `-std=` form.
require(counts["nvcc -std=c++N (shared list)"] == 2,
        f"cubvh: expected 2 `-std=c++{{cpp_standard}}` sites, found "
        f"{counts['nvcc -std=c++N (shared list)']} -- upstream setup.py changed")
require(counts["MSVC /std:c++N (cxx list)"] == 1,
        "cubvh: `/std:c++{cpp_standard}` site not found -- upstream changed")
require(counts["MSVC -Xcompiler=/std:c++N"] == 1,
        "cubvh: `-Xcompiler=/std:c++{cpp_standard}` site not found -- upstream changed")

# Nothing may still hand either compiler a standard.
for lineno, line in enumerate(text.splitlines(), 1):
    if re.search(r"std[:=]c\+\+", line) and not line.strip().startswith("#"):
        sys.exit(f"cubvh patch: setup.py:{lineno} still carries a C++-standard "
                 f"flag after patching: {line.strip()!r}")

setup.write_text(MARKER + "\n" + text)
print(f"cubvh patch: removed {sum(counts.values())} hardcoded C++-standard "
      f"flag(s); torch's cpp_extension now selects the standard")

# The vendored Eigen must actually be there: it is a gitlab submodule, and an
# empty directory compiles to "Eigen/Dense: No such file" three hours later
# on a Windows runner rather than here.
require(pathlib.Path("third_party/eigen/Eigen/Dense").is_file(),
        "cubvh: third_party/eigen is empty -- clone_recursive did not fetch the "
        "gitlab submodule; nothing in the sandboxed build can fetch it later")
print("cubvh patch: third_party/eigen present")
