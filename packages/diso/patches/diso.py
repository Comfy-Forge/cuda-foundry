"""Patch diso (SarahWeiii/diso @ 9792ad9).

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed. One tarball feeds both platforms.

Ported from the farm's packages/diso/patches/diso.py:

  1. atomicAdd(double*, double) shim below sm_60. diso's backward pass
     accumulates into doubles with atomicAdd (cumc.cu:440-444,
     cudualmc.cu:738-750), and that overload is a hardware intrinsic only from
     sm_60 (Pascal) on -- below it nvcc fails outright. The shared arch policy
     puts 5.0 on the cu12.4/12.6 rows, and the farm's rule is never to drop an
     architecture to make a build pass, so patch_lib.add_atomicadd_double_shim
     injects NVIDIA's documented CAS-loop emulation (CUDA C Programming Guide
     B.14), guarded to `defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600` so it
     exists only in the device passes that lack the intrinsic. The cu12.8 cell
     this repo builds today starts at sm_70 and never compiles the shim; it is
     kept so one source tree is correct for every row of the policy. The
     helper hard-fails if it finds no atomicAdd to guard.

Not needed, and checked rather than assumed:

  * setup.py emits no arch flag and reads no GPU; the cell's
    TORCH_CUDA_ARCH_LIST is authoritative. Asserted.
  * FORCE_CUDA=1 (set by the recipe template) is what routes get_extensions()
    to CUDAExtension on a GPU-less runner. Asserted to still be the spelling
    upstream checks, because the fallback is a CPU-only build that imports
    fine and has no kernels -- the SASS census would catch it, three hours
    later, on the runner.
  * `"cxx": ["-O3"]` reaches cl.exe on win-64, which warns D9002 and ignores
    it; torch's win_wrap_ninja_compile already puts distutils' /O2 on the
    line, so nothing is lost. Left as upstream wrote it (the farm did too).
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import add_atomicadd_double_shim, require  # noqa: E402

sources = sorted(pathlib.Path("src").glob("*.cu")) + sorted(pathlib.Path("src").glob("*.cuh"))
print(f"diso patch: scanning {len(sources)} CUDA source file(s) for atomicAdd")
marker = "cuda-wheels: atomicAdd(double*) shim for sm < 60"
already = [p for p in sources if marker in p.read_text(errors="replace")]
if len(already) == 2:
    print("diso patch: atomicAdd shim already present in cumc.cu and cudualmc.cu")
else:
    n = add_atomicadd_double_shim(sources)
    require(n == 2, f"diso: shim added to {n} file(s), expected exactly 2 "
                    f"(cumc.cu, cudualmc.cu) at 9792ad9 -- upstream changed")
    print(f"diso patch: atomicAdd(double*) shim added to {n} file(s)")

setup = pathlib.Path("setup.py").read_text()
require('os.getenv(\n        "FORCE_CUDA", "0"\n    ) == "1"' in setup
        or re.search(r'os\.getenv\(\s*"FORCE_CUDA",\s*"0"\s*\)\s*==\s*"1"', setup),
        "diso: setup.py no longer honours FORCE_CUDA=1 -- on a GPU-less runner "
        "it would fall back to a CUDA-less CppExtension and ship no kernels")
for lineno, line in enumerate(setup.splitlines(), 1):
    if re.search(r"-arch[= ]|gencode|get_device_capability|TORCH_CUDA_ARCH_LIST", line):
        sys.exit(f"diso patch: setup.py:{lineno} touches the arch list "
                 f"({line.strip()!r}) -- re-check")
print("diso patch: FORCE_CUDA honoured; no arch flags in setup.py; done")
