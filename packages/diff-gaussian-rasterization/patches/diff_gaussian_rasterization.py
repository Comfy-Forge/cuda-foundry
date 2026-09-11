"""Patch diff-gaussian-rasterization: glm through include_dirs, not an -I flag.

Runs on the fetching machine, cwd = the checkout, before the tarball is
sealed. No platform conditionals, no `import torch`.

At the pinned rev setup.py hands glm to nvcc as

    extra_compile_args={"nvcc": ["-I" + os.path.join(<abs path of setup.py>,
                                                     "third_party/glm/")]}

which is an ABSOLUTE build-machine path baked into the nvcc flag list. torch's
cpp_extension._get_cuda_arch_flags returns [] -- discarding
TORCH_CUDA_ARCH_LIST entirely and compiling for nvcc's default arch -- as
soon as any user nvcc flag contains the substring 'arch' (pytorch v2.8.0,
torch/utils/cpp_extension.py; the same trap fused-ssim's patch documents).
Whether that fires here therefore depends on whether the build directory's
path happens to contain 'arch' -- "search", "march", "architecture" -- which
is not a property a build should have. The SASS census would catch the
result, after a full compile.

CUDAExtension's `include_dirs` is the documented way to say the same thing,
reaches both the host and device compiles, and puts nothing in the flag list.
The substitution asserts on itself and the arch-flag check afterwards is
belt-and-braces.

Neither this rev nor camenduru's simple-knn probes the build host's GPU for
an arch (the pattern fused-ssim's patch removes): checked, and asserted below
so a bump that introduces one fails here rather than in the census.
"""

import re
import sys
from pathlib import Path

MARKER = "# patched-by: cuda-foundry diff_gaussian_rasterization"

setup_file = Path("setup.py")
text = setup_file.read_text(encoding="utf-8")
if MARKER in text:
    print("diff_gaussian_rasterization patch: setup.py already patched")
    sys.exit(0)

new, n = re.subn(
    r'extra_compile_args=\{"nvcc": \["-I" \+ os\.path\.join\(os\.path\.dirname\('
    r'os\.path\.abspath\(__file__\)\), "third_party/glm/"\)\]\}',
    # No trailing comment: the match ends just before the call's closing
    # parenthesis, and a comment there would swallow it.
    'include_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)), '
    '"third_party/glm/")]',
    text)
if n != 1:
    sys.exit(f"diff_gaussian_rasterization patch: expected exactly one glm -I "
             f"nvcc flag in setup.py, found {n} -- upstream changed setup.py at "
             f"this pinned rev; re-read it rather than relaxing this check")

for lineno, line in enumerate(new.splitlines(), 1):
    if ("nvcc" in line and ("-arch" in line or "-gencode" in line)) \
            or "get_device_capability" in line or "cuda.is_available" in line:
        sys.exit(f"diff_gaussian_rasterization patch: setup.py:{lineno} decides "
                 f"the arch list itself: {line.strip()!r}")

setup_file.write_text(MARKER + "\n" + new, encoding="utf-8")
print("diff_gaussian_rasterization patch: glm moved from an nvcc -I flag to include_dirs")
