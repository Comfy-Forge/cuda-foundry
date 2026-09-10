#!/usr/bin/env python3
"""The win-64 build: one compile, a wheel, and that wheel installed into %PREFIX%.

This is the Windows counterpart of scripts/build_snippets/build.sh. It is a
separate implementation rather than a ported one, because the mechanisms build.sh
relies on -- seccomp, a compiler seat swap on PATH, RPATH, ELF -- either do not
exist on Windows or work differently enough that a translation would be a lie.
What is preserved is the *contract*, because the same downstream tools consume it:

  * exactly one wheel in CUW_WHEELHOUSE, which becomes both published outputs
  * that wheel installed into %PREFIX%, which rattler-build packages as the .conda
  * no direct_url.json and no RECORD in the installed dist-info
  * %PREFIX% otherwise untouched by this script

Why Python and not a .bat: rattler-build renders the embedded build script through
minijinja BEFORE executing it, and minijinja opens a comment on brace-hash, an
expression on brace-brace and a statement on brace-percent. A render failure there
reports only "Script failed to execute" with no output at all. This file is copied
next to the recipe verbatim (like nonet.py) and never rendered, so it can contain
whatever characters it needs. The .bat that invokes it stays short enough to audit
for those three digraphs by eye.

See docs/WINDOWS.md for the measurements behind the checks below.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str) -> "None":
    print(f"::error::{msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        die(f"{name} is not set; it must come from the recipe's build.script.env")
    return val


# ---------------------------------------------------------------------------
# L1 is not available here, and this says so rather than implying otherwise.
# ---------------------------------------------------------------------------
def announce_sandbox_gap() -> None:
    """State the guarantee actually in force on this platform.

    build.sh runs the compile under scripts/build_snippets/nonet.py, a seccomp
    filter denying AF_INET/AF_INET6. That file cannot even load here -- it calls
    os.uname(), which does not exist on Windows -- and Windows offers no
    unprivileged, descendant-inherited equivalent. A per-executable
    `netsh advfirewall` rule keys on a program path rather than being inherited,
    so it is a materially weaker thing and is not pretended to be the same.

    What remains is L2 (per-package force-source flags, checked by the loader),
    L3 (the compile ledger below), and pip's own --no-index, which is a real but
    partial substitute: it stops pip resolving anything from an index, and does
    nothing about a setup.py that opens its own socket. So the claim on win-64 is
    "nothing shipped here lacks a compile record", not "the compile had no
    network". Those are different claims and only the first is enforced.
    """
    log("=== L1 (network denial): NOT ENFORCED on win-64 -- no seccomp equivalent.")
    log("===   In force instead: L2 force-source flags, L3 compile ledger, pip --no-index.")
    log("===   See docs/WINDOWS.md, 'The source-build guarantee is weaker on Windows'.")


# ---------------------------------------------------------------------------
# The host compiler, re-checked against the toolkit that actually got installed
# ---------------------------------------------------------------------------
_MSVC_GUARD = re.compile(r"_MSC_VER\s*<\s*(\d+)\s*\|\|\s*_MSC_VER\s*>=\s*(\d+)")


def msvc_version() -> int | None:
    """_MSC_VER of the cl.exe on PATH, or None if it cannot be determined."""
    cl = shutil.which("cl")
    if not cl:
        return None
    # cl.exe prints its banner on stderr and needs no arguments to do it.
    out = subprocess.run([cl], capture_output=True, text=True).stderr
    m = re.search(r"Version (\d+)\.(\d+)\.", out)
    if not m:
        return None
    log(f"=== cl.exe: {cl}")
    log(f"===   {out.strip().splitlines()[0] if out.strip() else '(no banner)'}")
    return int(m.group(1)) * 100 + int(m.group(2))


def nvcc_msvc_window(build_prefix: Path) -> tuple[int, int] | None:
    """The _MSC_VER window nvcc will actually accept, from its own header."""
    for hdr in build_prefix.glob("**/crt/host_config.h"):
        m = _MSVC_GUARD.search(hdr.read_text(encoding="utf8", errors="replace"))
        if m:
            log(f"=== nvcc host_config.h: {hdr}")
            return int(m.group(1)), int(m.group(2))
    return None


def check_host_compiler(build_prefix: Path) -> None:
    """Fail before compiling if MSVC is outside nvcc's compiled-in window.

    The solver cannot catch this: conda-forge's CUDA packages constrain only
    `vc >=14.2,<15`, which spans every MSVC from 19.2x to 19.5x. The real limit
    is the #error guard in nvcc's crt/host_config.h, and it moves every CUDA
    minor -- vs2022_win-64 (_MSC_VER 1944) is inside the window for CUDA >= 12.4
    and outside it for CUDA <= 12.3, whose ceiling is 1940.

    Checked here, against the headers actually installed, rather than trusted
    from the recipe: nvcc would otherwise report this as a #error in the middle
    of the first translation unit, which reads like a source problem.
    """
    msc = msvc_version()
    window = nvcc_msvc_window(build_prefix)
    if msc is None:
        die("cl.exe not on PATH -- the MSVC activation script did not run. "
            "Check that exactly one vsNNNN_win-64 is in the build environment.")
    if window is None:
        log("=== WARNING: no crt/host_config.h under BUILD_PREFIX; cannot verify "
            "the MSVC window. Proceeding -- nvcc will enforce it itself.")
        return
    lo, hi = window
    log(f"=== MSVC _MSC_VER {msc}; nvcc accepts [{lo}, {hi})")
    if not (lo <= msc < hi):
        die(f"MSVC _MSC_VER {msc} is outside nvcc's supported window [{lo}, {hi}). "
            f"This cell's CUDA line needs a different vsNNNN_win-64 -- run "
            f"`python tools/msvc_ceiling.py` for the per-CUDA table.")


def check_single_msvc(build_prefix: Path) -> None:
    """Exactly one MSVC activation, because two is decided by filename order.

    `cuda-nvcc` (the metapackage) hard-depends on vs2019_win-64, so a recipe that
    also asks for vs2022_win-64 gets both, and which cl.exe wins is settled by
    activate.d running vs2019_compiler_vars.bat before vs2022_compiler_vars.bat.
    That happens to give the newer one today, for no better reason than 2022
    sorting after 2019. Depend on cuda-nvcc_win-64 instead and this stays at one.
    """
    scripts = sorted(build_prefix.glob("etc/conda/activate.d/vs*_compiler_vars.bat"))
    names = [p.name for p in scripts]
    log(f"=== MSVC activation scripts in BUILD_PREFIX: {names or 'none'}")
    if len(scripts) > 1:
        die(f"{len(scripts)} MSVC activation scripts present ({', '.join(names)}). "
            f"Which compiler runs is then decided by activate.d filename order. "
            f"Depend on cuda-nvcc_win-64 rather than the cuda-nvcc metapackage, "
            f"which hard-depends on vs2019_win-64.")


# ---------------------------------------------------------------------------
# L3: the compile ledger, from ninja's own log
# ---------------------------------------------------------------------------
def ninja_log_entries(src_dir: Path) -> list[str]:
    """Outputs ninja recorded building in this tree.

    build.sh gets its ledger from a wrapper sitting in the nvcc seat. That trick
    does not transfer, for a reason narrower than the one this file used to
    give: ninja on Windows does NOT run its commands through cmd.exe -- it hands
    the command line straight to CreateProcess ("Do not prepend \'cmd /c\' on
    Windows, this breaks command lines greater than 8,191 chars",
    src/subprocess-win32.cc) -- and CreateProcess appends only `.exe` to an
    extensionless name. So PATHEXT never gets a say and a .bat in the seat is
    simply never found. The seat is not how caching gets in here either; see
    ccache_launcher() below, which uses torch's own PYTORCH_NVCC hook instead.

    ninja's own .ninja_log is the better ledger source anyway -- it is evidence
    about what was built rather than about what was invoked, and a prebuilt
    binary copied into the source tree appears in it not at all, which is exactly
    the case L3 exists to catch. Each line is
    `start end mtime output command-hash`; the output column is what we want.
    """
    outputs: list[str] = []
    for logfile in src_dir.glob("**/.ninja_log"):
        for line in logfile.read_text(encoding="utf8", errors="replace").splitlines():
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                outputs.append(parts[3])
    return outputs


def dist_info_dir(site: Path, wheel: Path) -> Path:
    """OUR .dist-info, derived from the wheel filename.

    Everything downstream must be scoped to the package we just built. The host
    environment is a full conda prefix -- torch alone brings dozens of extension
    modules and its dependencies bring their own dist-info directories -- so a
    bare glob over site-packages answers questions about torch, not about us.
    """
    name, version = wheel.name.split("-")[:2]
    # PEP 503/427: the dist-info directory uses the escaped name, which for a
    # wheel filename component is already the normalized form.
    d = site / f"{name}-{version}.dist-info"
    if not d.is_dir():
        die(f"expected {d.name} in {site} after installing {wheel.name}, "
            f"but it is not there -- the install did not land where this script "
            f"is looking.")
    return d


def installed_files(dist_info: Path) -> list[Path]:
    """Paths this package installed, from its own RECORD.

    Read BEFORE scrub_dist_info removes RECORD, which is why the order in main()
    matters. RECORD is the only authoritative statement of what belongs to this
    package rather than to the host environment around it.
    """
    record = dist_info / "RECORD"
    if not record.is_file():
        die(f"{dist_info.name}/RECORD is missing; cannot tell which installed "
            f"files belong to this package.")
    out = []
    for line in record.read_text(encoding="utf8", errors="replace").splitlines():
        rel = line.split(",")[0].strip()
        if rel:
            out.append((dist_info.parent / rel).resolve())
    return out


def ninja_edges(src_dir: Path) -> dict:
    """{normalised output -> (rule, source)} for every edge in every build.ninja.

    build.ninja carries the mapping in its `build <out>: <rule> <in>` edges.
    The RULE is kept as well as the source because on Windows it is the only
    honest way to tell which translation units went to nvcc and which went to
    cl: torch writes `cuda_compile` for the files _is_cuda_file() accepts and
    `compile` for the rest, and only the first kind is cached here.
    """
    edges: dict = {}
    for ninja in src_dir.glob("**/build.ninja"):
        for line in ninja.read_text(encoding="utf8", errors="replace").splitlines():
            if not line.startswith("build "):
                continue
            head, rest = _split_edge(line[len("build "):])
            if head is None:
                continue
            parts = rest.split()
            if len(parts) >= 2:
                # parts[0] is the rule name; the first input follows it.
                edges[_norm(_unescape(head))] = (parts[0], _unescape(parts[1]))
    return edges


def ninja_translation_units(src_dir: Path) -> list[str]:
    """Source files ninja actually compiled, by joining .ninja_log to build.ninja.

    .ninja_log records OUTPUTS; the ledger that tools/verify_conda.py consumes
    is a list of translation unit SOURCES, one per line, the same shape the
    Linux nvcc wrapper appends. Joining the two gives the sources for exactly
    the outputs this run produced -- not everything the build file could have
    built.
    """
    edges = ninja_edges(src_dir)
    units = []
    for out in ninja_log_entries(src_dir):
        hit = edges.get(_norm(out))
        if hit:
            units.append(hit[1])
    return units


def ninja_cuda_units(src_dir: Path) -> list[str]:
    """The subset of ninja_translation_units() that nvcc compiled.

    This is the denominator of the zero-miss gate: those are the only TUs that
    pass through ccache, because the C++ ones are handed to a bare `cl` whose
    invocation this build does not intercept (see ccache_launcher()).
    """
    edges = ninja_edges(src_dir)
    units = []
    for out in ninja_log_entries(src_dir):
        hit = edges.get(_norm(out))
        if hit and "cuda" in hit[0]:
            units.append(hit[1])
    return units


def _split_edge(text: str) -> tuple[str | None, str]:
    r"""Split a ninja edge at its separator colon.

    Ninja escapes a literal colon in a path as `$:`, which every Windows path
    with a drive letter has -- `build D$:/a/work/ext.obj: compile D$:/...`.
    Splitting on the first colon therefore lands inside `D$:` and yields
    nonsense. The separator is the first colon NOT preceded by a dollar.

    This is why the first version of this function mapped zero outputs to
    sources and wrote an empty ledger (run 34166741643).
    """
    i = 0
    while i < len(text):
        if text[i] == ":" and (i == 0 or text[i - 1] != "$"):
            return text[:i].strip(), text[i + 1:]
        i += 1
    return None, ""


def _unescape(token: str) -> str:
    """Ninja's path escaping, reversed: `$:` -> `:`, `$ ` -> ' ', `$$` -> '$'."""
    return token.replace("$:", ":").replace("$ ", " ").replace("$$", "$")


def _norm(path: str) -> str:
    """Compare Windows paths without tripping on separator or case differences."""
    return path.replace("\\", "/").lower()


def write_ledger(src_dir: Path) -> int:
    r"""Write CUW_LEDGER so L3 is a real gate on win-64 too.

    Without this the file never exists, tools/verify_conda.py reads an empty
    ledger, and its "compile ledger is non-empty" check is skipped -- reporting
    `0 TUs` while passing. That is precisely the failure mode the L4 canary
    exists to prevent elsewhere: a guarantee reported as being in force during a
    run where its mechanism never started.

    One caveat worth stating rather than leaving to be discovered: the verifier's
    companion check, that no TU came from OUTSIDE the work tree, tests
    `x.startswith("/")` and so cannot judge a Windows path like
    `D:\a\...\work\ssim.cu`. On win-64 the non-empty assertion is real and the
    foreign-path assertion is inert.
    """
    ledger = os.environ.get("CUW_LEDGER")
    if not ledger:
        die("CUW_LEDGER is not set; L3 would silently not run")
    units = ninja_translation_units(src_dir)
    Path(ledger).parent.mkdir(parents=True, exist_ok=True)
    Path(ledger).write_text("\n".join(units) + ("\n" if units else ""), encoding="utf8")
    log(f"=== ledger: wrote {len(units)} translation unit(s) to {ledger}")
    return len(units)


def check_ledger(src_dir: Path, own_files: list[Path]) -> None:
    """L3: every extension module we ship must have been compiled by this run.

    Scoped to THIS package's own files. An earlier version globbed
    site-packages for *.pyd and reported "24 installed extension module(s)" for
    a package that ships exactly one -- it was counting torch's. A check whose
    denominator is dominated by somebody else's binaries cannot fail for the
    reason it exists, so it was passing vacuously.
    """
    compiled = ninja_log_entries(src_dir)
    objects = [o for o in compiled if o.lower().endswith((".obj", ".o", ".lib"))]
    modules = [f for f in own_files if f.suffix.lower() == ".pyd"]
    log(f"ledger: ninja recorded {len(compiled)} output(s), {len(objects)} object(s), "
        f"for {len(modules)} extension module(s) shipped by this package")
    if modules and not objects:
        die(f"shipping {len(modules)} extension module(s) "
            f"({', '.join(f.name for f in modules[:4])}) but ninja compiled no "
            f"object files -- this build did not compile what it is shipping. "
            f"Either a prebuilt binary reached the source tree, or the build did "
            f"not use ninja and this ledger cannot see it.")


# ---------------------------------------------------------------------------
# Sharding: how the compile cache gets into the build on win-64
# ---------------------------------------------------------------------------
# The Linux handoff puts a wrapper script in the nvcc SEAT ($BUILD_PREFIX/bin/
# nvcc, real binary moved to nvcc.real) because torch's cpp_extension invokes
# that path directly. This file used to say the trick "does not transfer", and
# the reason it gave -- PATHEXT -- was the wrong reason for a true statement.
# The real one, measured against ninja's source and pytorch v2.8.0's:
#
#   * ninja on Windows does not use cmd.exe. subprocess-win32.cc hands the
#     command line to CreateProcess directly, deliberately ("Do not prepend
#     'cmd /c' on Windows, this breaks command lines greater than 8,191
#     chars"). CreateProcess appends only ".exe" to an extensionless name, so
#     PATHEXT is never consulted and a .bat in the seat is invisible.
#   * therefore ANY seat occupant on Windows has to be a real .exe.
#
# But the seat is not the only door. torch's _write_ninja_file reads
# PYTORCH_NVCC and writes its value verbatim as the ninja `nvcc` variable
# (cpp_extension.py:2840, v2.8.0, with the comment "user can set nvcc compiler
# with ccache using the environment variable here"). ninja does no tokenising
# of its own, and CreateProcess takes the first token of the command line as
# the executable -- so a TWO-token value is a launcher:
#
#   PYTORCH_NVCC = "<...>\ccache.exe <...>\nvcc.exe"
#
# which is byte-for-byte the invocation the Linux wrapper ends up making,
# `ccache <real nvcc> <args>`. No seat swap, no masquerade, no $PREFIX
# passenger to clean up, and nothing to restore on exit.
#
# ccache's own masquerade mode (copy ccache.exe to nvcc.exe and let it resolve
# the real compiler off PATH) would also work and was the first candidate. It
# is not used because it needs the seat -- or a PATH entry -- to be effective,
# and because "which nvcc did it actually find?" then becomes a question the
# build has to answer at runtime rather than a path this file writes down.
#
# What is NOT cached, stated plainly: the C++ translation units. torch writes
# the literal string `cl` into the ninja compile rule -- `compiler_name =
# "$cxx" if IS_HIP_EXTENSION else "cl"` -- so CXX does not reach it and only a
# real cl.exe earlier on PATH could intercept it. flash-attn has exactly one
# such TU (csrc/flash_attn/flash_api.cpp, measured at ~16 min) against 72 .cu
# files, so the link job recompiles that one and replays the other 72. The
# zero-miss gate below counts nvcc TUs, which is what it can honestly claim.


def ccache_bin() -> str:
    """The ccache to use, checked for the version the '-x cu' path needs."""
    binp = os.environ.get("CUW_CCACHE_BIN") or shutil.which("ccache.exe") or "ccache"
    try:
        out = subprocess.run([binp, "--version"], capture_output=True, text=True)
    except OSError as exc:
        die(f"ccache not usable at {binp!r} ({exc}) -- the shard handoff rides on it")
    if out.returncode != 0:
        die(f"ccache not usable at {binp!r} -- the shard handoff rides on it")
    first = (out.stdout or "").splitlines()[0] if out.stdout else ""
    m = re.search(r"(\d+)", first)
    # ccache 3.x has no "cu" entry in its source-language table, so every TU
    # compiled as `-x cu` is passed through UNCACHED -- not a slow cache, no
    # cache, and the shard lane would silently void itself while reporting
    # success. Same floor, same reason, as build.sh.
    if not m or int(m.group(1)) < 4:
        die(f"ccache {first!r} found; >= 4 required (3.x cannot cache '-x cu')")
    log(f"=== ccache: {first}  dir={os.environ.get('CCACHE_DIR', '(unset)')}")
    # Normalised to a native Windows path. The workflow hands this over in the
    # same mixed git-bash shape everything else uses (D:\a\_temp/ccache-bin/
    # ccache.exe), which bash can execute and Python can open -- but it ends up
    # inside PYTORCH_NVCC, which is a raw command line for CreateProcess, and
    # that deserves the canonical spelling rather than a shape that happens to
    # work.
    try:
        binp = str(Path(binp).resolve())
    except OSError:
        pass
    return binp


def ccache_launcher(build_prefix: Path, ccache: str) -> str:
    """The PYTORCH_NVCC value: `<ccache.exe> <real nvcc.exe>`.

    Both halves must be space-free. CreateProcess resolves the executable from
    the start of the command line and, with no application name given, it will
    happily split "C:\\Program Files\\x.exe y" at the first space and try to run
    "C:\\Program.exe" -- so a space here does not fail loudly, it runs the wrong
    thing. Neither path can contain one today (rattler-build's prefix is a
    padded placeholder and RUNNER_TEMP is D:\\a\\_temp), and if that ever stops
    being true this refuses rather than guesses.
    """
    nvcc = build_prefix / "Library" / "bin" / "nvcc.exe"
    if not nvcc.is_file():
        found = shutil.which("nvcc.exe")
        if not found:
            die(f"no nvcc.exe at {nvcc} and none on PATH -- the CUDA toolkit is "
                f"not in this build environment")
        nvcc = Path(found)
    for part in (ccache, str(nvcc)):
        if " " in part:
            die(f"path {part!r} contains a space. PYTORCH_NVCC is a raw command "
                f"line handed to CreateProcess, which would split it there and "
                f"silently run the wrong executable.")
    return f"{ccache} {nvcc}"


def parse_ccache_stats(text: str) -> tuple:
    """(hits, misses) out of `ccache --print-stats`, which is TAB-separated.

    Only three counters are read, and which three matters. ccache reports
    `direct_cache_miss` for a TU that missed the direct lookup and then HIT the
    preprocessed one -- a hit, not a miss -- so counting every key with "miss"
    in it would report a fully replayed link job as a failed one, and counting
    every key with "hit" in it would double a single compilation. `cache_miss`
    is ccache's own final tally, and it is the only miss counter used here.
    Same three names build.sh sums on Linux.
    """
    hits = misses = 0
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not parts[1].strip().isdigit():
            continue
        key, val = parts[0].strip(), int(parts[1].strip())
        if key in ("direct_cache_hit", "preprocessed_cache_hit"):
            hits += val
        elif key == "cache_miss":
            misses += val
    return hits, misses


def ccache_stats(ccache: str) -> tuple:
    """(hits, misses) since the last `ccache -z`."""
    out = subprocess.run([ccache, "--print-stats"], capture_output=True,
                         text=True).stdout or ""
    return parse_ccache_stats(out)


# ---------------------------------------------------------------------------
# The partition: which translation units this shard actually compiles
# ---------------------------------------------------------------------------
# The Linux wrapper partitions inside the nvcc seat, one TU at a time, and
# emits an empty object for the ones that are not its slice. With no seat here,
# the partition happens one step earlier: the SOURCE files that are not this
# shard's are replaced with a stub before the compile runs. That is a strictly
# weaker mechanism -- it needs to be told which files are translation units --
# and the whole reason it is safe is that a mistake cannot reach an artifact:
#
#   * a shard is never published. Its only output is a ccache directory.
#   * any file this partition gets wrong is compiled for real by the LINK job,
#     whose lookup then misses, and the zero-miss gate fails the build.
#
# So the failure mode of a bad glob is a red link job, never a wrong .conda.
# It is still checked directly, in check_declared_tus(): after the build, the
# set of files ninja actually compiled must EQUAL the declared set. A glob that
# matched too little, too much, or nothing at all is a hard error rather than a
# silently smaller build.
#
# One asymmetry, deliberate: .cu files are partitioned, C++ files are stubbed
# in EVERY shard. A C++ TU does not go through ccache (see above), so a shard
# compiling one buys nothing -- and stubbing it everywhere is what keeps the
# shard's own link step honest. The module entry point lives in a C++ TU
# (PYBIND11_MODULE expands to PyInit_<TORCH_EXTENSION_NAME>), distutils passes
# /EXPORT:PyInit_<name> to link.exe, and an unresolved export is LNK2001. So
# the first stubbed C++ file defines that symbol itself, by pasting the macro
# the compile line already carries. With that in place a shard has no
# unresolved externals at all: the stubs reference nothing and this shard's own
# .cu objects reference only torch and the CUDA runtime.

_EMPTY_STUB = """/* cuw shard stub: this translation unit belongs to another shard.
   Replaced before the compile so nvcc does the cheapest possible work on it.
   The object still has to EXIST -- the link step lists every object. */
static int cuw_shard_stub_translation_unit;
"""

# `##` needs one level of indirection to paste an expanded macro argument.
_PYINIT_STUB = """/* cuw shard stub, module entry point.
   distutils links the extension with /EXPORT:PyInit_<name>, so the symbol has
   to be defined even in a shard whose real sources are all stubbed out; an
   unresolved export is LNK2001 and the shard would fail at link. The name is
   pasted from the -DTORCH_EXTENSION_NAME already on this compile line, so this
   stub carries no package-specific knowledge. */
#define CUW_PASTE2(a, b) a##b
#define CUW_PASTE(a, b) CUW_PASTE2(a, b)
#ifdef TORCH_EXTENSION_NAME
/* extern "C" is not decoration. The file being stubbed is a C++ TU, so
   without it the definition is mangled (?PyInit_x@@YAPEAXXZ) while
   /EXPORT:PyInit_x asks for the plain name -- unresolved, LNK2001, and every
   shard dies at link having compiled its slice perfectly. */
#ifdef __cplusplus
extern "C"
#endif
__declspec(dllexport) void *CUW_PASTE(PyInit_, TORCH_EXTENSION_NAME)(void)
{
    return 0;
}
#endif
static int cuw_shard_stub_translation_unit;
"""


def declared_tus(src_dir: Path, patterns: list) -> list:
    """Every file the package declares as a translation unit, in a stable order.

    A pattern that matches nothing is an error and not an empty slice: it is
    exactly what a moved or renamed source directory looks like, and the shard
    would otherwise compile nothing at all while exiting 0.
    """
    found: list = []
    seen = set()
    for pat in patterns:
        matched = sorted(q for q in src_dir.glob(pat) if q.is_file())
        if not matched:
            die(f"shard_sources pattern {pat!r} matched no file under {src_dir}. "
                f"Sharding cannot partition a set it cannot see; fix the pattern "
                f"in the package's package.yml.")
        for q in matched:
            key = _norm(str(q))
            if key not in seen:
                seen.add(key)
                found.append(q)
    return found


def owns(rel: str, index0: int, count: int) -> bool:
    """Is this translation unit this shard's?

    A hash of the path relative to SRC_DIR, exactly as stateless as the Linux
    wrapper's md5-of-realpath and for the same reason -- but taken on the
    RELATIVE path, because every shard has to reach the same answer and only
    the relative half is guaranteed identical between two runners.
    """
    h = int(hashlib.md5(rel.encode("utf8")).hexdigest()[:8], 16)
    return h % count == index0


def partition_sources(src_dir: Path, patterns: list, index0: int, count: int) -> list:
    """Stub out every declared TU that is not this shard's. Returns the slice."""
    files = declared_tus(src_dir, patterns)
    mine, stub = [], []
    for q in files:
        rel = _norm(str(q.relative_to(src_dir)))
        # C++ TUs are stubbed in every shard: they are compiled by `cl`, which
        # nothing here caches, so a shard that built one would just be slower.
        if q.suffix.lower() != ".cu" or not owns(rel, index0, count):
            stub.append(q)
        else:
            mine.append(q)
    pyinit_done = False
    for q in stub:
        text = q.read_text(encoding="utf8", errors="replace")
        if not pyinit_done and q.suffix.lower() != ".cu" and "PYBIND11_MODULE" in text:
            q.write_text(_PYINIT_STUB, encoding="utf8")
            pyinit_done = True
            log(f"=== partition: {q.name} stubbed WITH the module entry point")
        else:
            q.write_text(_EMPTY_STUB, encoding="utf8")
    log(f"=== partition: shard {index0 + 1}/{count} compiles {len(mine)} of "
        f"{len(files)} declared translation unit(s); {len(stub)} stubbed")
    for q in mine:
        log(f"===   mine: {q.relative_to(src_dir)}")
    return mine


def check_declared_tus(src_dir: Path, patterns: list) -> None:
    """The declared TU set must be exactly what ninja compiled.

    Without this the `shard_sources` globs are an unverified promise, and the
    two ways they can be wrong look nothing alike in the logs: too narrow and a
    shard silently compiles a subset of its own slice; too wide and it stubs a
    file that is not a TU at all. Comparing against .ninja_log turns both into
    one loud failure, and it is the same source of truth the compile ledger
    uses.
    """
    # Compared as resolved ABSOLUTE paths on both sides. build.ninja records
    # os.path.abspath()ed sources, and trying to make them relative to SRC_DIR
    # first would turn any spelling difference between the two (a short 8.3
    # component, a different drive-letter case) into a bogus "compiled but not
    # declared" rather than a match.
    declared = {_norm(str(q.resolve())) for q in declared_tus(src_dir, patterns)}
    built = {_norm(str(Path(_unescape(u)).resolve()))
             for u in ninja_translation_units(src_dir)}
    missing = sorted(declared - built)
    extra = sorted(built - declared)
    if missing or extra:
        die(f"shard_sources does not describe this build's translation units. "
            f"Declared but never compiled: {missing[:5]} ({len(missing)}); "
            f"compiled but not declared: {extra[:5]} ({len(extra)}). "
            f"The partition would silently skip or over-stub those.")
    log(f"=== partition: shard_sources matches ninja exactly ({len(built)} TUs)")


# ---------------------------------------------------------------------------
def one_wheel(wheelhouse: Path) -> Path:
    wheels = sorted(wheelhouse.glob("*.whl"))
    if len(wheels) != 1:
        found = ", ".join(w.name for w in wheels) or "none"
        die(f"expected exactly 1 wheel in {wheelhouse}, found {len(wheels)}: {found}. "
            f"The .conda and the published wheel must come from ONE file; anything "
            f"else means the compile produced nothing or the wheelhouse was not cleared.")
    return wheels[0]


def scrub_dist_info(dist_info: Path) -> None:
    """pip records where it installed FROM; that path exists on no other machine.

    direct_url.json makes `pip freeze` emit a file:// URL instead of a version.
    RECORD goes for the reason conda-forge drops it: with it present `pip
    uninstall` will cheerfully delete files conda owns.

    OUR dist-info only. An earlier version looped over every *.dist-info in
    site-packages and duly deleted RECORD and direct_url.json from filelock,
    fsspec, jinja2, markupsafe and the rest of the host environment -- packages
    this build does not own and must not modify. They are not ours to tidy, and
    the .conda gets no benefit: rattler-build packages files that APPEARED in
    $PREFIX, so deleting somebody else's does nothing but corrupt the prefix.

    INSTALLER is deliberately not written -- rattler-build rewrites it during
    packaging regardless, so a write here would be dead code that looks
    load-bearing.
    """
    for name in ("direct_url.json", "RECORD"):
        target = dist_info / name
        if target.exists():
            target.unlink()
            log(f"=== removed {dist_info.name}/{name}")


def main() -> int:
    mode = env("CUW_MODE", "full")
    if mode not in ("full", "shard", "link"):
        die(f"CUW_MODE={mode!r} is not a mode; expected full, shard or link")

    prefix = Path(env("PREFIX"))
    build_prefix = Path(env("BUILD_PREFIX"))
    src_dir = Path(env("SRC_DIR", os.getcwd()))
    python = env("PYTHON", sys.executable)
    wheelhouse = Path(env("CUW_WHEELHOUSE", r"C:\cuw\wheelhouse"))
    shard_count = int(env("CUW_SHARD_COUNT", "0") or "0")
    shard_index = int(env("CUW_SHARD_INDEX", "0") or "0")
    patterns = [x for x in env("CUW_SHARD_SOURCES", "").replace(";", "\n").split("\n")
                if x.strip()]

    announce_sandbox_gap()
    check_single_msvc(build_prefix)
    check_host_compiler(build_prefix)

    # ---- the compile cache, in shard and link mode only ------------------
    # `full` is left byte-identical to what publishes today for the four
    # unsharded win-64 packages: no ccache, no PYTORCH_NVCC, nothing new in
    # the environment. Only a package that actually shards pays for any of it.
    ccache = ""
    if mode in ("shard", "link"):
        if not patterns:
            die("CUW_SHARD_SOURCES is empty, so there is nothing to partition. "
                "A sharded package must declare `shard_sources` in package.yml; "
                "the recipe template passes it through.")
        ccache = ccache_bin()
        launcher = ccache_launcher(build_prefix, ccache)
        os.environ["PYTORCH_NVCC"] = launcher
        log(f"=== PYTORCH_NVCC={launcher}")
        subprocess.run([ccache, "-z"], capture_output=True)

    if mode == "shard":
        if shard_count < 1:
            die(f"CUW_SHARD_COUNT={shard_count} in shard mode")
        # The matrix numbers shards from 1; the partition arithmetic is 0-based.
        # build.sh converts in the same place and for the same reason: without
        # it the comparison never matches, every TU is stubbed, and the build
        # still succeeds -- having compiled nothing.
        partition_sources(src_dir, patterns, shard_index - 1, shard_count)
        # A shard whose real sources are all stubbed still has to LINK, and
        # /EXPORT:PyInit_<name> is not the only symbol that can go missing --
        # a package whose entry point is not in a PYBIND11_MODULE C++ file
        # would leave it unresolved. The stub above is the mechanism; this is
        # the backstop, and it is scoped to shard mode so the published
        # artifact can never be linked with it. link.exe reads LINK and
        # prepends it to its command line.
        os.environ["LINK"] = ("/FORCE:UNRESOLVED " + os.environ.get("LINK", "")).strip()
        log("=== LINK=/FORCE:UNRESOLVED (shard mode only; this shard's .pyd is discarded)")

    # torch's BuildExtension._check_abi RAISES when the VC environment is active
    # and this is unset, and conda-forge's MSVC activation is exactly what makes
    # it active. Measured in pytorch v2.8.0. Without this, no cell builds.
    os.environ["DISTUTILS_USE_SDK"] = "1"
    log("=== DISTUTILS_USE_SDK=1 (torch raises without it under an active VC env)")

    wheelhouse.mkdir(parents=True, exist_ok=True)
    # Only the wheels, never the directory: this path arrives from the
    # environment and a misspelled variable must not delete a tree.
    for stale in wheelhouse.glob("*.whl"):
        stale.unlink()

    # ---- ONE compile, and it produces a WHEEL ----------------------------
    # --no-index is the partial stand-in for the seccomp filter: with
    # --no-build-isolation and --no-deps nothing legitimate needs an index, so
    # denying it costs nothing and closes pip's own path to a prebuilt wheel.
    cmd = [python, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation",
           "--no-index", "--wheel-dir", str(wheelhouse), "-vv"]
    log(f"=== {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=src_dir).returncode
    if rc != 0:
        die(f"build failed (exit {rc}) -- see the compiler output above")

    # ---- what the cache did, and the gate that makes it mean something ---
    if mode in ("shard", "link"):
        check_declared_tus(src_dir, patterns)
        hits, misses = ccache_stats(ccache)
        cuda_tus = len(ninja_cuda_units(src_dir))
        log(f"=== ccache: {hits} hit(s) / {misses} miss(es) over {cuda_tus} nvcc "
            f"translation unit(s)")
        if hits + misses == 0:
            die("ccache saw zero lookups -- PYTORCH_NVCC did not reach ninja, so "
                "nothing was cached and nothing can be replayed. Check the "
                "`nvcc = ` line in build.ninja.")
        if hits + misses != cuda_tus:
            subprocess.run([ccache, "--show-stats", "-v"])
            die(f"ccache saw {hits + misses} lookup(s) for {cuda_tus} nvcc "
                f"translation unit(s). Some TU bypassed the cache -- an "
                f"uncacheable-* counter in the table above says which -- so the "
                f"handoff covers less than it claims.")

    if mode == "shard":
        # ZERO tolerance in the LINK job is what this exists to make possible,
        # so a shard that stored nothing is a defect even though it exits 0
        # today: it would ship an empty slice and the link job would recompile
        # it -- as one miss, indistinguishable from a wrong-architecture shard.
        if misses == 0:
            die(f"shard {shard_index}/{shard_count} stored nothing in the cache "
                f"({hits} hit(s), 0 miss(es)). Either the partition stubbed "
                f"every translation unit, or the cache was already populated -- "
                f"both mean this shard contributes nothing to the link job.")
        log(f"shard {shard_index}/{shard_count} done; cache populated. "
            f"Exiting before install.")
        return 0

    if mode == "link":
        # Zero, not a ratio. One miss is a whole TU recompiled, and a
        # percentage cannot tell "one nondeterministic TU" from "four shards
        # built for the wrong architecture".
        if misses > 0:
            die(f"link job had {misses} ccache miss(es) of {hits + misses}. "
                f"Shard caches did not transfer cleanly: flag mismatch, path "
                f"mismatch, or wrong-architecture artifacts.")
        log(f"=== link: all {hits} nvcc translation unit(s) replayed from the "
            f"shard caches, zero misses")

    wheel = one_wheel(wheelhouse)
    log(f"=== installing {wheel.name} into %PREFIX%")

    install = [python, "-m", "pip", "install", str(wheel), "--no-deps", "--no-index",
               "--no-build-isolation", "--force-reinstall", "-vv"]
    rc = subprocess.run(install).returncode
    if rc != 0:
        die(f"installing the built wheel failed (exit {rc})")

    site = Path(sysconfig.get_paths()["purelib"])
    if not site.is_dir():
        die(f"site-packages not found at {site}")
    dist_info = dist_info_dir(site, wheel)
    # RECORD is read here and deleted below, in that order.
    own_files = installed_files(dist_info)
    check_ledger(src_dir, own_files)
    write_ledger(src_dir)
    scrub_dist_info(dist_info)
    log("=== win-64 build complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
