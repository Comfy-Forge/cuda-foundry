"""Patch gsplat v1.5.3 for the foundry's build matrix.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed, on the fetching machine -- so no `import torch`,
and NO platform conditionals: fetch produces ONE tarball that every platform's
build consumes. cuda-wheels' version of this patch gated the MSVC flag
translation on `os.name == "nt"`, which was correct there because the farm
patches on the build machine; here it would bake the fetching machine's answer
into the Windows build. The condition is moved INTO setup.py, which runs on the
build machine, where it belongs.

Four edits, each asserted on its own (a whole-file before/after comparison
passes as long as ANY edit landed, which is how the predecessor's dead
`-gencode` regex in fused_ssim survived review):

1. MSVC host flags. setup.py hardcodes `{"cxx": ["-O3"]}`; cl.exe does not
   error on a GCC flag, it prints D9002 "ignoring unknown option" and ships an
   UNOPTIMISED extension. Rewritten to pick /O2 under os.name == "nt" at
   setup time.
2. The C++ standard. setup.py appends "-std=c++17" to nvcc_flags AFTER torch's
   own -std, and nvcc honours the last one, so torch >= 2.13's C++20-only
   headers compiled as C++17 and failed. Deleted; torch's cpp_extension
   selects the standard the installed torch needs (patch_lib.strip_std_flags).
3. cg::labeled_partition needs sm_70+. Guarded with a tiled_partition<1>
   fallback rather than dropping old GPUs from the arch list -- the repo's
   rule is never to shrink coverage to make a build pass. Inert at cu12.8+,
   whose policy rows start at 7.0/7.5; load-bearing for the cu12.4/12.6 rows
   which carry 5.0/6.0.
4. glm's documentation. The vendored submodule brings ~1,200 files of doxygen
   HTML, images and a PDF along with its 426 headers -- 19 MB uncompressed,
   18% of the farm's published wheel. The headers stay (gsplat's JIT fallback
   in cuda/_backend.py points extra_include_paths at them); the rest goes.

Upstream never reads its own arch flags from the local GPU -- it relies on
torch's TORCH_CUDA_ARCH_LIST handling, and nothing it puts in nvcc_flags
contains the substring 'arch' (which would make cpp_extension discard the
list entirely). Checked at this rev; asserted below so a future bump that
adds one fails here rather than in the SASS census.
"""

import glob
import pathlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import (guard_labeled_partition_in_files, prune_glm_docs,  # noqa: E402
                       require, strip_std_flags)

MARKER = "# patched-by: cuda-foundry gsplat"

setup_file = Path("setup.py")
content = setup_file.read_text(encoding="utf-8")

if MARKER in content:
    print("gsplat patch: setup.py already patched")
else:
    # 1. host-compiler flags, decided at build time on the build machine
    content, n_cxx = re.subn(
        r'extra_compile_args = \{"cxx": \["-O3"\]\}',
        'extra_compile_args = {"cxx": (["/O2"] if os.name == "nt" else ["-O3"])}'
        '  # patched: cl.exe ignores -O3 (D9002) and ships unoptimised',
        content)
    require(n_cxx == 1,
            f"expected exactly one `extra_compile_args = {{\"cxx\": [\"-O3\"]}}` in "
            f"gsplat setup.py, found {n_cxx} -- upstream moved the block; "
            f"refusing to ship a Windows extension built with flags cl.exe "
            f"silently ignores")
    print("gsplat patch: cxx -O3 made conditional on os.name (/O2 on MSVC)")

    # 2. the C++ standard is torch's call
    content, n_std = strip_std_flags(content)
    require(n_std > 0,
            "no hardcoded -std flag found in gsplat setup.py -- upstream "
            "changed; refusing to build against an unverified flag set")
    print(f"gsplat patch: dropped {n_std} hardcoded std flag(s); torch's "
          f"cpp_extension now selects the standard")

    # Nothing may hand nvcc an arch flag: cpp_extension._get_cuda_arch_flags
    # returns [] as soon as any user nvcc flag contains 'arch', which would
    # silently reduce the cell's arch list to nvcc's default.
    for lineno, line in enumerate(content.splitlines(), 1):
        if "nvcc_flags" in line and ("-arch" in line or "-gencode" in line):
            sys.exit(f"gsplat patch: setup.py:{lineno} puts an arch flag into "
                     f"nvcc_flags: {line.strip()!r}; that makes torch drop "
                     f"TORCH_CUDA_ARCH_LIST entirely")

    setup_file.write_text(MARKER + "\n" + content, encoding="utf-8")

# 3. sm<70 fallback for the match collectives
n_lp = guard_labeled_partition_in_files(
    sorted(glob.glob("gsplat/cuda/csrc/*.cu")), required=False)
if n_lp:
    print(f"gsplat patch: guarded {n_lp} cg::labeled_partition site(s) for sm<70")
else:
    # Idempotence: a second run finds the guard already present. A rev where
    # the sites simply vanished must not pass as "already guarded".
    already = any("cg::tiled_partition<1>" in Path(f).read_text(encoding="utf-8")
                  for f in glob.glob("gsplat/cuda/csrc/*.cu"))
    require(already, "no cg::labeled_partition sites found to guard and none "
                     "already guarded -- upstream changed; re-read the sources")
    print("gsplat patch: labeled_partition sites already guarded")

# 4. glm: headers only
glm = Path("gsplat/cuda/csrc/third_party/glm")
require((glm / "glm" / "glm.hpp").is_file(),
        f"{glm}/glm/glm.hpp is missing -- the submodule was not checked out; "
        f"package.yml must set clone_recursive: true")
prune_glm_docs(glm)
print("gsplat patch: done")
