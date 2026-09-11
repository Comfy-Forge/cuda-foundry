"""Patch nunchaku v1.2.1 for the foundry build.

Ported from cuda-wheels (packages/nunchaku/patches/nunchaku.py) and re-read
against the pinned rev. Runs with cwd = the cloned source, once, before the
tarball is sealed; nothing here depends on the cell.

1. Local version tag. setup.py appends `+cu{torch.version.cuda}torch{M.m}`
   (dotted CUDA, e.g. +cu12.8torch2.8) to the version. The publisher owns the
   +cu128torch2.8 tag; two local tags is not PEP 440 and the artifact name
   would not match the cell. Keep the base version.

2. Release flags. setup.py hardcodes -g / -Og / -UNDEBUG on the host lines
   and -g / -UNDEBUG on nvcc, ungated: full debug info nothing consumes
   (auditwheel strips it after it was generated at full cost), host code
   compiled at debug optimisation because -Og lands AFTER the default -O2,
   and live assert() in the user's inference path. The two knobs upstream
   DOES gate (-G on DEBUG, --generate-line-info on NUNCHAKU_BUILD_WHEELS) are
   left alone, and so is --allow-expensive-optimizations, which is ptxas's
   own default at -O2+.

3. PTX tail. The gencode loop emits `code=sm_X` only, so no wheel ever
   carried PTX. Add `code=compute_X` for the highest PLAIN target (sm_89 on
   every row): a real JIT path onto sm_90 and newer parts that get no cubin.
   Arch-conditional targets (120a/121a) are skipped -- their PTX loads only on
   that one arch, and the base compute_120 cannot be built from `a`-only
   instructions.

Every substitution asserts on its own.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require  # noqa: E402

MARKER = "# patched-by: cuda-foundry nunchaku"
setup_py = pathlib.Path("setup.py")
text = setup_py.read_text()
if MARKER in text:
    print("nunchaku patch: already applied")
    sys.exit(0)

# 1. version tag --------------------------------------------------------------
old = 'version = f"{version}+cu{cuda_version}torch{torch_major_minor_version}"'
require(text.count(old) == 1, "nunchaku: the version-suffix line was not found once")
text = text.replace(old, "pass  # cuda-foundry: the publisher owns the +cuNNNtorchM.m tag")
print("nunchaku patch: upstream local-version suffix stripped")

# 2. release flags ------------------------------------------------------------
flagsets = {
    "GCC_FLAGS": ('"-g", ', '"-UNDEBUG", ', '"-Og"'),
    "MSVC_FLAGS": ('"/UNDEBUG", ',),
}
removed = []
for name, drop in flagsets.items():
    m = re.search(rf"^(\s*{name} = \[)(.*)(\]\s*)$", text, re.M)
    require(m is not None, f"nunchaku: {name} assignment not found in setup.py")
    body = m.group(2)
    for f in drop:
        require(f in body, f"nunchaku: {f.strip()} not in {name} at this rev")
        body = body.replace(f, "", 1)
        removed.append(f"{name}:{f.strip().strip(chr(34)).strip(',')}")
    body = re.sub(r",\s*\]", "]", body.rstrip().rstrip(","))
    text = text[:m.start()] + m.group(1) + body + m.group(3) + text[m.end():]
for f in ('        "-g",\n', '        "-UNDEBUG",\n'):
    require(text.count(f) == 1, f"nunchaku: NVCC_FLAGS entry {f.strip()} not found once")
    text = text.replace(f, "", 1)
    removed.append(f"NVCC_FLAGS:{f.strip().strip(chr(34)).strip(',')}")
print(f"nunchaku patch: dropped {len(removed)} debug flag(s): {', '.join(removed)}")

# 3. PTX tail -----------------------------------------------------------------
gc_anchor = '''    for target in sm_targets:
        NVCC_FLAGS += ["-gencode", f"arch=compute_{target},code=sm_{target}"]'''
require(text.count(gc_anchor) == 1, "nunchaku: the -gencode loop was not found once")
text = text.replace(gc_anchor, gc_anchor + '''
    # cuda-foundry: portable PTX tail for the highest NON-arch-conditional
    # target. 120a/121a are skipped on purpose -- their PTX would be
    # compute_120a, which loads only on sm_120.
    _cuw_plain = [t for t in sm_targets if t.isdigit()]
    if _cuw_plain:
        _cuw_top = max(_cuw_plain, key=int)
        NVCC_FLAGS += ["-gencode", f"arch=compute_{_cuw_top},code=compute_{_cuw_top}"]
        print(f"[cuda-foundry] PTX tail: compute_{_cuw_top} "
              f"(skipped arch-conditional {[t for t in sm_targets if not t.isdigit()]})")''', 1)

setup_py.write_text(MARKER + "\n" + text)

final = setup_py.read_text()
gcc = re.search(r"^\s*GCC_FLAGS = \[(.*)\]\s*$", final, re.M).group(1)
for bad in ('"-g"', '"-Og"', '"-UNDEBUG"'):
    require(bad not in gcc, f"nunchaku: {bad} still present in GCC_FLAGS")
require('*cond("-G")' in final, "nunchaku: the DEBUG-gated -G entry was damaged")
require("allow-expensive-optimizations=true" in final, "nunchaku: ptxas options were damaged")
require("code=compute_{_cuw_top}" in final, "nunchaku: the PTX tail is not in setup.py on disk")
require("torch_major_minor_version}\"" not in final, "nunchaku: the version suffix survived")
import ast  # noqa: E402
ast.parse(final)
print("nunchaku patch: done")
