"""Patch torchsparse v2.0.0 for torch >= 2.8, MSVC, and a conda-supplied
sparsehash.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed. Two consequences of that timing shape this
file, and they are why it differs from cuda-wheels' packages/torchsparse:

  * ONE tarball serves every platform, so nothing here may branch on
    os.name. The predecessor's Windows block (vendor sparsehash 2.0.4 over
    the network, hand-write an MSVC sparseconfig.h, guard atomic.cuh) ran
    only on a Windows fetcher. Here every fix is applied unconditionally and
    the platform condition, where one exists, lives in the C source
    (#ifndef _WIN32).
  * sparsehash comes from conda-forge on BOTH platforms (host_deps), not from
    EPEL on Linux and a vendored tarball on Windows. It is header-only, so
    nothing is linked; setup.py just has to be told where the headers are,
    and it has no include_dirs at all. package.yml's build_env points
    CUW_SPARSEHASH_INCLUDE at $PREFIX/include (Linux) and
    $PREFIX/Library/include (win-64), and the injected include_dirs reads it.

Every substitution asserts its own count (docs/WINDOWS.md, the fused-ssim
lesson: a whole-file before/after guard passes on a partial match).

1. `.type()` -> `.scalar_type()` in AT_DISPATCH_* arguments. Upstream passes
   a DeprecatedTypeProperties as the dispatch argument; torch >= 2.8 removed
   the implicit conversion to c10::ScalarType ("no suitable conversion
   function from 'const at::DeprecatedTypeProperties'"). Only the dispatch
   position is rewritten (`.type(), "` -- the macro always follows the arg
   with the op-name literal), so `.type().is_cuda()` and friends are
   untouched. 10 sites at the pinned rev.

2. atomic.cuh redefines CUDA's builtin atomicExch for uint64_t. On Linux
   uint64_t is `unsigned long` (a distinct overload); with MSVC uint64_t IS
   `unsigned long long`, so the helper collides with the builtin ("function
   has already been defined"). The builtin covers the Windows case entirely:
   the helper is compiled out under _WIN32, in the source, on every platform.

3. LLP64: `data_ptr<long>` is wrong on Windows. Long tensors hold int64_t;
   on LP64 Linux `long` == int64_t so it happens to work, on MSVC `long` is
   32-bit and the kernel arguments no longer match. int64_t is correct on
   both. 2 sites.

4. sparsehash include_dirs, from CUW_SPARSEHASH_INCLUDE (see above). The
   variable is REQUIRED at build time -- an unset variable would silently
   build against whatever the compiler happens to find, which on a conda
   host prefix is nothing, and the failure would be a missing-header error
   three minutes in rather than a clear one at setup time.

5. `-g` dropped from the cxx flags (patch_lib.strip_debug_flags): a release
   artifact has no use for host DWARF, and it is paid in compile time and
   peak memory. -O3 / -fopenmp / -lgomp are left alone.

6. sparseconfig.h for MSVC. conda-forge's win-64 sparsehash ships the
   tarball's own Windows config under Library/include/windows/, and it is
   for a pre-2015 compiler: HASH_NAMESPACE stdext, SPARSEHASH_HASH
   stdext::hash_compare -- C2039 on VS2022. Nothing else on the include
   path provides sparsehash/internal/sparseconfig.h there, so run
   34592178879 died with C1083 on exactly that file. The patch writes a
   modern config (std::hash from <functional>, stdint types -- the one the
   farm measured working on VS2022) into third_party/sparsehash_msvc/ and
   setup.py puts that directory FIRST in include_dirs on Windows only. The
   condition lives in setup.py, which runs on the build machine, not in this
   fetch-time script; on Linux the directory is never on the path and
   conda-forge's own generated config is used.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require, strip_debug_flags_in_file  # noqa: E402

backend = pathlib.Path("torchsparse/backend")
require(backend.is_dir(), "torchsparse: torchsparse/backend not found")

# ── 1. dispatch arguments ───────────────────────────────────────────────────
pat = re.compile(r"\.type\(\)(\s*,\s*\")")
total = 0
for f in sorted(backend.rglob("*")):
    if f.suffix not in (".cu", ".cpp", ".cc", ".cuh", ".h", ".hpp"):
        continue
    text = f.read_text(encoding="utf-8")
    new, n = pat.subn(r".scalar_type()\1", text)
    if n:
        f.write_text(new, encoding="utf-8")
        print(f"torchsparse patch: {f}: {n} dispatch arg(s) .type() -> .scalar_type()")
        total += n
require(total == 10, f"torchsparse: expected 10 '.type(), \"' dispatch args under "
                     f"torchsparse/backend, rewrote {total} -- upstream changed")

# ── 2. atomic.cuh: helper compiled out on Windows ───────────────────────────
atomic = backend / "utils" / "atomic.cuh"
require(atomic.is_file(), "torchsparse: backend/utils/atomic.cuh not found")
a = atomic.read_text(encoding="utf-8")
require("#ifndef _WIN32" not in a, "torchsparse: atomic.cuh already guarded?")
require(a.startswith("#pragma once\n"), "torchsparse: atomic.cuh does not start "
                                         "with #pragma once -- anchor moved")
a = a.replace("#pragma once\n",
              "#pragma once\n#ifndef _WIN32  // cuda-foundry: uint64_t==ULL on MSVC; "
              "CUDA's builtin already provides this overload\n", 1) + "#endif  // _WIN32\n"
atomic.write_text(a, encoding="utf-8")
require(atomic.read_text(encoding="utf-8").count("#ifndef _WIN32") == 1
        and atomic.read_text(encoding="utf-8").rstrip().endswith("#endif  // _WIN32"),
        "torchsparse: atomic.cuh guard NOT on disk")
print("torchsparse patch: atomic.cuh helper guarded out under _WIN32")

# ── 3. LLP64 ────────────────────────────────────────────────────────────────
ll = 0
for f in sorted(backend.rglob("*")):
    if f.suffix not in (".cu", ".cpp", ".cuh", ".h", ".hpp"):
        continue
    t = f.read_text(encoding="utf-8")
    n = t.count("data_ptr<long>")
    if n:
        f.write_text(t.replace("data_ptr<long>", "data_ptr<int64_t>"), encoding="utf-8")
        print(f"torchsparse patch: {f}: {n} data_ptr<long> -> data_ptr<int64_t>")
        ll += n
require(ll == 2, f"torchsparse: expected 2 data_ptr<long> call(s), rewrote {ll} -- "
                 f"upstream changed")

# ── 4. sparsehash include_dirs ──────────────────────────────────────────────
setup_py = pathlib.Path("setup.py")
s = setup_py.read_text(encoding="utf-8")
needle = ("extension_type('torchsparse.backend',\n"
          "                       sources,\n"
          "                       extra_compile_args=extra_compile_args)")
repl = ("extension_type('torchsparse.backend',\n"
        "                       sources,\n"
        "                       include_dirs=_cuw_sparsehash_include_dirs(),\n"
        "                       extra_compile_args=extra_compile_args)")
require(s.count(needle) == 1, "torchsparse: extension_type call not found in setup.py "
                              "-- upstream changed; update this patch")
s = s.replace(needle, repl, 1)
helper = '''

def _cuw_sparsehash_include_dirs():
    # cuda-foundry (see packages/torchsparse/patches): hashmap_cpu.hpp and
    # query_cpu.cpp include <google/dense_hash_map>, a header-only library
    # supplied by conda-forge's sparsehash in the host prefix. Upstream has no
    # include_dirs at all. Required, not defaulted: silently building against
    # nothing is a missing-header error minutes later, not a clear one here.
    inc = os.environ.get('CUW_SPARSEHASH_INCLUDE', '')
    if not inc or not os.path.isfile(os.path.join(inc, 'google', 'dense_hash_map')):
        raise SystemExit(
            'torchsparse: CUW_SPARSEHASH_INCLUDE must name a directory holding '
            f'google/dense_hash_map (got {inc!r}); package.yml build_env sets it '
            'from $PREFIX and host_deps supplies sparsehash')
    dirs = [inc]
    if os.name == 'nt':
        # conda-forge's win-64 sparsehash carries only the tarball's pre-2015
        # MSVC config (stdext::hash_compare); the modern one the patch wrote
        # goes first so <sparsehash/internal/sparseconfig.h> resolves to it.
        msvc = os.path.abspath(os.path.join('third_party', 'sparsehash_msvc'))
        if not os.path.isfile(os.path.join(msvc, 'sparsehash', 'internal', 'sparseconfig.h')):
            raise SystemExit(f'torchsparse: {msvc} is missing the MSVC sparseconfig.h the patch writes')
        dirs.insert(0, msvc)
    return dirs
'''
anchor2 = "\nextension_type = CUDAExtension if device == 'cuda' else CppExtension\n"
require(s.count(anchor2) == 1, "torchsparse: extension_type assignment anchor not found")
s = s.replace(anchor2, helper + anchor2, 1)
setup_py.write_text(s, encoding="utf-8")
require("_cuw_sparsehash_include_dirs()" in setup_py.read_text(encoding="utf-8")
        and "def _cuw_sparsehash_include_dirs" in setup_py.read_text(encoding="utf-8"),
        "torchsparse: sparsehash include_dirs NOT on disk")
print("torchsparse patch: sparsehash include_dirs injected (CUW_SPARSEHASH_INCLUDE)")

# ── 5. no debug flags in a release artifact ────────────────────────────────
n = strip_debug_flags_in_file("setup.py", "torchsparse setup.py")
require(n == 1, f"torchsparse: expected to strip exactly one debug flag (-g) from "
                f"setup.py, stripped {n}")
require("'-O3', '-fopenmp', '-lgomp'" in setup_py.read_text(encoding="utf-8"),
        "torchsparse: the cxx flag list was damaged by the debug-flag strip")

# ── 6. a sparseconfig.h VS2022 can compile ─────────────────────────────────
cfg = pathlib.Path("third_party/sparsehash_msvc/sparsehash/internal/sparseconfig.h")
require(not cfg.exists(), "torchsparse: the MSVC sparseconfig.h is already there?")
cfg.parent.mkdir(parents=True, exist_ok=True)
cfg.write_text(
    "/* cuda-foundry (packages/torchsparse/patches): sparsehash config for a\n"
    " * modern MSVC. conda-forge's win-64 package ships only the tarball's\n"
    " * pre-2015 config (stdext::hash_compare, C2039 on VS2022). Reached only\n"
    " * on Windows: setup.py puts this directory first in include_dirs there\n"
    " * and nowhere else. */\n"
    "#define GOOGLE_NAMESPACE ::google\n"
    "#define HASH_NAMESPACE std\n"
    "#define HASH_FUN_H <functional>\n"
    "#define SPARSEHASH_HASH HASH_NAMESPACE::hash\n"
    "#define HAVE_STDINT_H 1\n"
    "#define HAVE_UINT16_T 1\n"
    "#define HAVE_LONG_LONG 1\n"
    "#define HAVE_MEMCPY 1\n"
    "#define STL_NAMESPACE std\n"
    "#define _START_GOOGLE_NAMESPACE_ namespace google {\n"
    "#define _END_GOOGLE_NAMESPACE_ }\n", encoding="utf-8")
require(cfg.is_file() and "#define SPARSEHASH_HASH HASH_NAMESPACE::hash\n" in cfg.read_text(encoding="utf-8"),
        "torchsparse: MSVC sparseconfig.h NOT on disk")
print(f"torchsparse patch: wrote {cfg} for win-64 builds")
