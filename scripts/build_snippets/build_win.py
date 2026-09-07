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


def check_ledger(src_dir: Path, site: Path) -> None:
    compiled = ninja_log_entries(src_dir)
    objects = [o for o in compiled if o.lower().endswith((".obj", ".o", ".lib"))]
    modules = [p for p in site.glob("**/*.pyd")]
    log(f"ledger: ninja recorded {len(compiled)} output(s), {len(objects)} object(s), "
        f"for {len(modules)} installed extension module(s)")
    if modules and not objects:
        die(f"installed {len(modules)} extension module(s) "
            f"({', '.join(p.name for p in modules[:4])}) but ninja compiled no "
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


def scrub_dist_info(site: Path) -> None:
    """pip records where it installed FROM; that path exists on no other machine.

    direct_url.json makes `pip freeze` emit a file:// URL instead of a version.
    RECORD goes for the reason conda-forge drops it: with it present `pip
    uninstall` will cheerfully delete files conda owns.

    INSTALLER is deliberately not written -- rattler-build rewrites it during
    packaging regardless, so a write here would be dead code that looks
    load-bearing.
    """
    for di in site.glob("*.dist-info"):
        for name in ("direct_url.json", "RECORD"):
            target = di / name
            if target.exists():
                target.unlink()
                log(f"=== removed {di.name}/{name}")


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
    check_ledger(src_dir, site)
    scrub_dist_info(site)
    log("=== win-64 build complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
