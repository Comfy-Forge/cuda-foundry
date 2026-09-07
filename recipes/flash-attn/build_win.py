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
    does not transfer: on Windows the seat holds nvcc.exe, and a .bat cannot take
    an .exe's place when the caller spells the extension (PATHEXT puts .EXE ahead
    of .BAT in any case).

    ninja's own .ninja_log is the better source here anyway -- it is evidence
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


def ninja_translation_units(src_dir: Path) -> list[str]:
    """Source files ninja actually compiled, by joining .ninja_log to build.ninja.

    .ninja_log records OUTPUTS; the ledger that tools/verify_conda.py consumes
    is a list of translation unit SOURCES, one per line, the same shape the
    Linux nvcc wrapper appends. build.ninja carries the mapping in its
    `build <out>: <rule> <in>` edges, so joining the two gives the sources for
    exactly the outputs this run produced -- not everything the build file
    could have built.
    """
    edges: dict[str, str] = {}
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
                edges[_norm(_unescape(head))] = _unescape(parts[1])
    units = []
    for out in ninja_log_entries(src_dir):
        src = edges.get(_norm(out))
        if src:
            units.append(src)
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
    if mode != "full":
        die(f"CUW_MODE={mode!r} is not implemented on win-64; only 'full' is. "
            f"The shard/link handoff on Linux rides on ccache occupying the nvcc "
            f"seat, which does not transfer here (see build_win.py:ninja_log_entries). "
            f"Sharded packages are Windows-blocked until that is built.")

    prefix = Path(env("PREFIX"))
    build_prefix = Path(env("BUILD_PREFIX"))
    src_dir = Path(env("SRC_DIR", os.getcwd()))
    python = env("PYTHON", sys.executable)
    wheelhouse = Path(env("CUW_WHEELHOUSE", r"C:\cuw\wheelhouse"))

    announce_sandbox_gap()
    check_single_msvc(build_prefix)
    check_host_compiler(build_prefix)

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
