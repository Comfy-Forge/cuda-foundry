"""Patch flex_gemm_vb (visualbruno/FlexGEMM @ db388bd) into a package that
installs and imports beside plain flex_gemm.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed, on a Linux host with no torch. One tarball
feeds both platforms. Ported from the farm's
packages/flexgemm_vb/patches/flexgemm_vb.py, re-read against the pinned rev;
every substitution asserts its own site count.

  1. Rename flex_gemm -> flex_gemm_vb: pyproject name, setup() name, the nine
     `packages=` entries, the extension name, the five source paths, the
     autotune-cache home directory (~/.flex_gemm -> ~/.flex_gemm_vb, in
     setup.py's post-install copy and in flex_gemm/__init__.py's default
     path), the package directory, and the two `walk_package('flex_gemm', ..)`
     calls in utils/autotuner.py (which enumerate the fork's own modules by
     name -- left alone they would walk the OTHER package's kernels, or fail
     when it is absent).
  2. C++ standard: drop the hardcoded `-std=c++20` / `/std:c++20` (four
     sites; the fork branches on platform itself) and let torch's
     cpp_extension choose. Pinned c++20 breaks torch < 2.7 on Windows and
     duplicates torch's own -std=c++17 on the win-64 nvcc line.
  3. The Autotuner subclass forwards 13 positional arguments to triton's base
     __init__, which triton 3.0/3.1 (torch 2.4/2.5) do not accept. Not a
     cu12.8/torch 2.8 problem (triton 3.4 takes them), but it is a real
     defect and patch_lib.fix_triton_autotuner_super_auto rewrites it into a
     signature-filtered kwargs call for every cell.

setup.py emits no arch flag and reads no GPU (`--use_fast_math` is the only
nvcc flag); the cell's TORCH_CUDA_ARCH_LIST is authoritative. Asserted.
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import (fix_triton_autotuner_super_auto, require,  # noqa: E402
                       strip_std_flags)

TAG = "flex_gemm_vb"


def sub_count(path, old, new, what, expect):
    p = pathlib.Path(path)
    t = p.read_text()
    n = t.count(old)
    if n == 0:
        require(new in t, f"{TAG}: {what}: {old!r} not found in {path} and the "
                          f"replacement is not there either -- upstream changed")
        print(f"{TAG} patch: {what}: already applied")
        return
    require(n == expect, f"{TAG}: {what}: expected {expect} site(s) of {old!r} "
                         f"in {path}, found {n} -- upstream changed")
    p.write_text(t.replace(old, new))
    print(f"{TAG} patch: {what} ({n} site{'s' if n != 1 else ''})")


# ── 1. rename ────────────────────────────────────────────────────────────
sub_count("pyproject.toml", 'name = "flex_gemm"', 'name = "flex_gemm_vb"', "pyproject name", 1)
sub_count("setup.py", 'name="flex_gemm",', 'name="flex_gemm_vb",', "setup() name", 1)
sub_count("setup.py", '"flex_gemm",', '"flex_gemm_vb",', "packages: top level", 1)
sub_count("setup.py", '"flex_gemm.', '"flex_gemm_vb.', "packages: subpackages + extension name", 9)
sub_count("setup.py", '"flex_gemm/', '"flex_gemm_vb/', "extension source paths", 5)
# Exact, delimited spellings: a bare "~/.flex_gemm" is a substring of its own
# replacement and would be renamed again on a re-run.
sub_count("setup.py", '"~/.flex_gemm"', '"~/.flex_gemm_vb"', "autotune cache dir (setup.py)", 1)
sub_count("setup.py", "'~/.flex_gemm/autotune_cache.json'", "'~/.flex_gemm_vb/autotune_cache.json'",
          "autotune cache file (setup.py)", 1)

src, dst = pathlib.Path("flex_gemm"), pathlib.Path("flex_gemm_vb")
if src.is_dir() and not dst.exists():
    src.rename(dst)
    print(f"{TAG} patch: flex_gemm/ -> flex_gemm_vb/")
require(dst.is_dir() and not src.exists(), f"{TAG}: package directory rename did not land")

sub_count(dst / "__init__.py", "~/.flex_gemm/autotune_cache.json",
          "~/.flex_gemm_vb/autotune_cache.json", "autotune cache path (__init__.py)", 1)
sub_count(dst / "utils/autotuner.py", "walk_package('flex_gemm'",
          "walk_package('flex_gemm_vb'", "walk_package() module root", 2)
for py in dst.rglob("*.py"):
    t = py.read_text()
    require(not re.search(r"^\s*(from|import)\s+flex_gemm(\.|\s|$)", t, re.M),
            f"{TAG}: {py} imports the top-level name `flex_gemm`, which after "
            f"the rename means the OTHER package")
    require("'flex_gemm'" not in t and '"flex_gemm"' not in t,
            f"{TAG}: {py} still names 'flex_gemm' as a string -- a module root "
            f"or cache key that would point at the other package")
require(not re.search(r"""["']flex_gemm(["'/.])""", pathlib.Path("setup.py").read_text()),
        f"{TAG}: setup.py still names the unrenamed package somewhere")

# ── 2. C++ standard ──────────────────────────────────────────────────────
setup = pathlib.Path("setup.py")
text = setup.read_text()
new, n_std = strip_std_flags(text)
if n_std:
    require(n_std == 4, f"{TAG}: stripped {n_std} C++-standard flag(s), expected "
                        f"4 at db388bd -- upstream changed")
    setup.write_text(new)
    print(f"{TAG} patch: dropped {n_std} hardcoded C++-standard flag(s); torch selects it")
else:
    require("std=c++" not in text and "std:c++" not in text,
            f"{TAG}: a C++-standard flag survives in setup.py")
    print(f"{TAG} patch: setup.py already carries no C++-standard flag")

# ── 3. triton Autotuner signature ────────────────────────────────────────
n_at = fix_triton_autotuner_super_auto(".")
require(n_at == 1 or "_flexgemm_supported" in (dst / "utils/autotuner.py").read_text(),
        f"{TAG}: the Autotuner super().__init__ forward was not rewritten")
print(f"{TAG} patch: triton Autotuner forward is signature-filtered")

# ── the cell's arch list must stay authoritative ─────────────────────────
for lineno, line in enumerate(setup.read_text().splitlines(), 1):
    if re.search(r"(?<!offload)-arch[= ]|gencode|get_device_capability|TORCH_CUDA_ARCH_LIST", line):
        sys.exit(f"{TAG} patch: setup.py:{lineno} touches the arch list "
                 f"({line.strip()!r}) -- re-check")
print(f"{TAG} patch: setup.py emits no arch flags; TORCH_CUDA_ARCH_LIST is authoritative")
print(f"{TAG} patch: done")
