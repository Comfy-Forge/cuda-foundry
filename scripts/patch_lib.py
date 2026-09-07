"""Shared helpers for per-package patch scripts.

Patch scripts run with cwd = the cloned source tree and are invoked as
`python $GITHUB_WORKSPACE/packages/<pkg>/patches/<pkg>.py`, so they import
this module by walking up from their own __file__:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
    from patch_lib import strip_std_flags

Everything here is idempotent and content-gated: applying it twice is a no-op,
and it never rewrites a file it did not match.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# --------------------------------------------------------------------------
# Cell facts. The build-wheel action exports these to patch scripts; a missing
# value FAILS CLOSED (0, 0) so a gate defaults to "old", i.e. the conservative
# branch, rather than silently taking the new-toolchain path.
# --------------------------------------------------------------------------


def env_mm(name: str) -> tuple[int, int]:
    """(major, minor) from an env var like CUW_TORCH_VERSION=2.13.0."""
    parts = os.environ.get(name, "0.0").split(".")[:2]
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return (0, 0)


def torch_mm() -> tuple[int, int]:
    return env_mm("CUW_TORCH_VERSION")


def cuda_mm() -> tuple[int, int]:
    return env_mm("CUW_CUDA_VERSION")


# --------------------------------------------------------------------------
# The C++ standard.
#
# torch's cpp_extension already selects the standard the INSTALLED torch needs
# and appends it only when the caller supplied none (`-std=c++17` through torch
# 2.11, `-std=c++20` from 2.12/2.13; `/std:` spelling on MSVC). A package that
# hardcodes its own standard therefore overrides torch and is wrong at one end
# of the range or the other:
#   * pinned to c++17 -> torch >= 2.13's headers fail on MSVC (C7555 designated
#     initializers in c10/util/StringUtil.h, C7582 bit-field default member
#     initializers in c10/core/AutogradState.h)
#   * pinned to c++20 -> torch < 2.7 fails ON WINDOWS (nvcc's EDG misparses
#     `std::move(ivalue).to<List<Elem>>()` in ATen/core/ivalue_inl.h; and on a
#     pinned MSVC 14.29 host nvcc drops the flag and cudafe++ crashes).
#     "ON WINDOWS" added 2026-08-26: this used to describe the misparse without
#     qualifying the platform, which directly contradicted
#     packages/cubvh/patches/cubvh.py, whose gate only keeps c++17 on Windows.
#     Settled by building it -- cubvh cu12.4 / torch 2.4.1 on Linux compiles
#     clean at c++20 (run 32968713626). The GCC/EDG combination is fine; it is
#     MSVC's front end that chokes.
# The farm's answer is to delete the package's opinion and let torch decide.
# Audited and recommended by the 2026-08-24 review board.
# --------------------------------------------------------------------------

# A string literal that is nothing but a C++-standard flag, in any spelling a
# setup.py uses: "-std=c++17", "/std:c++20", "-Xcompiler=/std:c++17",
# "-Xcompiler", "/std:c++17" pairs are handled by two passes.
_STD_LITERAL = re.compile(
    r"""(['"])(?:-Xcompiler[= ])?[-/]std[:=]c\+\+\d+\1\s*,?[ \t]*"""
)
# `"-Xcompiler", "/std:c++17"` written as two adjacent list elements.
_STD_XCOMPILER_PAIR = re.compile(
    r"""(['"])-Xcompiler\1\s*,\s*(['"])/std:c\+\+\d+\2\s*,?[ \t]*"""
)


# A whole statement whose only argument is a standard flag, e.g.
#     nvcc_args.append("-std=c++17")
# Deleting just the literal would leave `nvcc_args.append()`, which raises
# TypeError at setup.py import time -- it silently destroyed every non-Windows
# pytorch3d cell (2026-08-24). Remove the entire statement instead.
_STD_APPEND_STMT = re.compile(
    r"^[ \t]*[A-Za-z_][\w.]*\.append\(\s*(['\"])(?:-Xcompiler[= ])?"
    r"[-/]std[:=]c\+\+\d+\1\s*\)[ \t]*\r?\n",
    re.M,
)


def strip_std_flags(text: str) -> tuple[str, int]:
    """Remove hardcoded C++-standard flags.

    Handles list elements (`["-O3", "-std=c++17"]`), -Xcompiler pairs, and
    whole `x.append("-std=c++17")` statements. Never leaves a syntactically
    broken call behind: an empty `append()` that was not already present is a
    hard error rather than a silently poisoned setup.py.
    """
    empty_before = len(re.findall(r"\.append\(\s*\)", text))
    # Replace with `pass`, not nothing: the statement may be the only
    # one in an `if` block (pytorch3d guards its append with
    # `if os.name != "nt":`), and deleting it would leave an empty
    # block -> IndentationError.
    text, n0 = _STD_APPEND_STMT.subn(
        lambda m: re.match(r"[ \t]*", m.group(0)).group(0) + "pass\n", text)
    text, n1 = _STD_XCOMPILER_PAIR.subn("", text)
    text, n2 = _STD_LITERAL.subn("", text)
    empty_after = len(re.findall(r"\.append\(\s*\)", text))
    if empty_after > empty_before:
        raise SystemExit(
            "PATCH FAILED: stripping the C++-standard flag would leave an "
            "empty .append() call -- the source uses a form this helper does "
            "not recognise. Fix patch_lib.strip_std_flags rather than "
            "shipping a setup.py that cannot import.")
    return text, n0 + n1 + n2


def strip_std_flags_in_file(path: str | Path, label: str = "") -> int:
    """Apply strip_std_flags to a file in place. Returns the number removed."""
    p = Path(path)
    if not p.exists():
        return 0
    text = p.read_text(encoding="utf-8", errors="surrogateescape")
    new, n = strip_std_flags(text)
    if n:
        p.write_text(new, encoding="utf-8", errors="surrogateescape")
        print(f"patch_lib: {label or p}: dropped {n} hardcoded C++-standard "
              f"flag(s); torch's cpp_extension now selects the standard")
    return n


# --------------------------------------------------------------------------
# GCC-only flags in a "cxx" list reach cl.exe verbatim on Windows, which does
# not error -- it prints `D9002: ignoring unknown option` and carries on. The
# wheel then ships UNOPTIMISED (and without OpenMP). Translate the ones with
# real MSVC equivalents and drop the rest.
# --------------------------------------------------------------------------

_GCC_TO_MSVC = {
    "-O3": "/O2",
    "-O2": "/O2",
    "-O1": "/O1",
    "-fopenmp": "/openmp",
    "-ffast-math": "/fp:fast",
    "-funroll-loops": None,      # on by default under /O2
    "-Wall": None,               # /Wall is far noisier on MSVC; drop
    "-Wno-unused-variable": None,
    "-lgomp": None,              # linker flag, meaningless in a compile list
    "-pthread": None,
}

_CXX_LIST_RE = re.compile(r'("cxx"\s*:\s*\[)([^\]]*)(\])')


def translate_cxx_flags_for_msvc(text: str) -> tuple[str, int]:
    """Map GCC flags to MSVC equivalents inside `"cxx": [...]` lists.

    No-op on non-Windows callers -- guard the call with `os.name == "nt"`.
    Returns (new_text, number_of_flags_changed).
    """
    changed = 0

    def _one(m: re.Match) -> str:
        nonlocal changed
        head, body, tail = m.group(1), m.group(2), m.group(3)
        def _flag(fm: re.Match) -> str:
            nonlocal changed
            quote, flag = fm.group(1), fm.group(2)
            if flag not in _GCC_TO_MSVC:
                return fm.group(0)
            repl = _GCC_TO_MSVC[flag]
            changed += 1
            if repl is None:
                return ""
            return f"{quote}{repl}{quote}"
        body = re.sub(r"""(['"])(-[^'"]*)\1""", _flag, body)
        # tidy the commas left by removed elements
        body = re.sub(r",\s*,", ",", body)
        body = re.sub(r"\[\s*,", "[", body)
        return head + body + tail

    text = _CXX_LIST_RE.sub(_one, text)
    return text, changed


# --------------------------------------------------------------------------
# cg::labeled_partition requires sm_70+ (the match collectives are gated on
# _CG_CUDA_ARCH >= 700 in cooperative_groups/details/info.h). Building it for
# an arch list that includes sm_50/sm_60 fails with
#   namespace "cooperative_groups" has no member "labeled_partition"
# The farm's rule is never to drop architectures to make a build pass, so
# instead compile the fast path only where it exists and fall back to a
# single-thread tile below sm_70: labeled_partition groups the warp's threads
# by a label so one representative can do the atomic for the group; a size-1
# tile makes every thread its own group, which is the same arithmetic with
# less coalescing -- correct on the old GPUs that would otherwise be dropped.
# --------------------------------------------------------------------------

_LABELED_PARTITION_RE = re.compile(
    r"^([ \t]*)auto\s+(\w+)\s*=\s*cg::labeled_partition\(\s*(\w+)\s*,\s*(\w+)\s*\);[ \t]*$",
    re.M,
)


def guard_labeled_partition(text: str) -> tuple[str, int]:
    """Wrap cg::labeled_partition uses in an __CUDA_ARCH__ >= 700 guard."""
    if "__CUDA_ARCH__ >= 700" in text and "cg::tiled_partition<1>" in text:
        return text, 0  # already patched

    def _one(m: re.Match) -> str:
        ind, var, warp, label = m.groups()
        return (
            f"{ind}#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700\n"
            f"{ind}auto {var} = cg::labeled_partition({warp}, {label});\n"
            f"{ind}#else\n"
            f"{ind}// sm_50/sm_60 have no match collectives: every thread is its\n"
            f"{ind}// own group, so each does its own atomic (same result).\n"
            f"{ind}auto {var} = cg::tiled_partition<1>(cg::this_thread_block());\n"
            f"{ind}#endif"
        )

    return _LABELED_PARTITION_RE.subn(_one, text)


def guard_labeled_partition_in_files(files, required: bool = True) -> int:
    total = 0
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="surrogateescape")
        new, n = guard_labeled_partition(text)
        if n:
            p.write_text(new, encoding="utf-8", errors="surrogateescape")
            total += n
            print(f"patch_lib: {f}: guarded {n} cg::labeled_partition use(s) "
                  f"for sm<70")
    if required and total == 0:
        raise SystemExit(
            "patch_lib: no cg::labeled_partition uses found to guard -- "
            "upstream changed; the sm<70 lanes would fail to compile")
    return total


# --------------------------------------------------------------------------
# CCCL 3.x (CUDA 13.x) removed cub::DeviceScan::ExclusiveSum's 4-arg IN-PLACE
# overload (d_temp, bytes, d_data, num). The arguments then bind to the new
# env-based overload and produce garbage template errors (InputIteratorT =
# nullptr_t). The 5-arg form with d_in == d_out is still in-place and works on
# every CUDA line, so duplicate the data argument.
# Hoisted from packages/cumesh/patches/cumesh.py so cumesh and its forks share
# one implementation (review board, 2026-08-24).
# --------------------------------------------------------------------------


def fix_inplace_exclusive_sum(text: str) -> tuple[str, int]:
    """Rewrite 4-arg in-place ExclusiveSum calls to the 5-arg form."""
    out, pos, fixed = [], 0, 0
    for m in re.finditer(r"cub::DeviceScan::ExclusiveSum\(", text):
        start, depth, j = m.end(), 1, m.end()
        while depth and j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        argstr = text[start:j - 1]
        args, d, last = [], 0, 0
        for k, ch in enumerate(argstr):
            if ch in "([":
                d += 1
            elif ch in ")]":
                d -= 1
            elif ch == "," and d == 0:
                args.append(argstr[last:k])
                last = k + 1
        args.append(argstr[last:])
        if len(args) == 4:
            data = args[2]
            new_argstr = ",".join(
                [args[0], args[1], data, " " + data.strip(), args[3]])
            out.append(text[pos:start])
            out.append(new_argstr)
            out.append(")")
            pos = j
            fixed += 1
    out.append(text[pos:])
    return "".join(out), fixed


def fix_inplace_exclusive_sum_in_files(files, required: bool = True) -> int:
    """Apply the ExclusiveSum rewrite across a list of source files."""
    total = 0
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="surrogateescape")
        new, n = fix_inplace_exclusive_sum(text)
        if n:
            p.write_text(new, encoding="utf-8", errors="surrogateescape")
            total += n
            print(f"patch_lib: {f}: {n} in-place ExclusiveSum call(s) "
                  f"-> 5-arg form")
    if required and total == 0:
        raise SystemExit(
            "patch_lib: no in-place ExclusiveSum calls found -- the CCCL-3.x "
            "breakage should live in these files; upstream changed?")
    return total


# --------------------------------------------------------------------------
# FlexGEMM subclasses triton.runtime.Autotuner and forwards 13 POSITIONAL
# arguments to the base __init__. triton 3.0/3.1 (what torch 2.4/2.5 pin)
# has a narrower signature, so the import dies with
#   TypeError: Autotuner.__init__() takes from 7 to 13 positional arguments
#              but 14 were given
# This is a REAL defect, not a CI artifact -- it reproduces on a GPU -- so it
# is fixed rather than forgiven by the verify gate (review board 2026-08-24).
# Pass only the parameters the installed triton actually accepts.
# --------------------------------------------------------------------------

_AUTOTUNER_SUPER_RE = re.compile(
    r"^([ \t]*)super\(\)\.__init__\(\s*\n"          # opening
    r"((?:[ \t]*[A-Za-z_][A-Za-z_0-9]*,\s*\n)+)"     # positional args
    r"[ \t]*\)[ \t]*$",
    re.M,
)


def fix_triton_autotuner_super(path: str | Path) -> int:
    """Rewrite a positional super().__init__ forward into a filtered kwargs call."""
    p = Path(path)
    if not p.exists():
        return 0
    text = p.read_text(encoding="utf-8", errors="surrogateescape")
    if "_flexgemm_supported" in text:
        return 0  # already patched

    def _one(m: re.Match) -> str:
        ind, argblock = m.group(1), m.group(2)
        names = [ln.strip().rstrip(",") for ln in argblock.strip().splitlines()]
        names = [n for n in names if n]
        pairs = "\n".join(f'{ind}    "{n}": {n},' for n in names)
        return (
            f"{ind}# Farm patch: older triton (3.0/3.1, pinned by torch 2.4/2.5)\n"
            f"{ind}# has a narrower Autotuner.__init__; forward only what it takes.\n"
            f"{ind}import inspect as _inspect\n"
            f"{ind}_flexgemm_supported = _inspect.signature(\n"
            f"{ind}    super().__init__).parameters\n"
            f"{ind}_flexgemm_args = {{\n{pairs}\n{ind}}}\n"
            f"{ind}super().__init__(**{{_k: _v for _k, _v in "
            f"_flexgemm_args.items()\n"
            f"{ind}                    if _k in _flexgemm_supported}})"
        )

    new, n = _AUTOTUNER_SUPER_RE.subn(_one, text, count=1)
    if n:
        p.write_text(new, encoding="utf-8", errors="surrogateescape")
        print(f"patch_lib: {p}: triton Autotuner super().__init__ now "
              f"signature-filtered ({len(new.splitlines()) - len(text.splitlines())} lines)")
    return n


def fix_triton_autotuner_super_auto(root: str | Path = ".") -> int:
    """Find the fork's autotuner.py wherever it lives and fix it.

    Forks rename their package directory (flex_gemm -> flex_gemm_vb), and some
    carry no Autotuner subclass at all, so a hardcoded path is wrong in both
    directions. Absent file = nothing to fix (not an error). Present file with
    an unrecognised super() forward = fail loud, because that is the shape we
    rely on.
    """
    files = sorted(Path(root).glob("*/utils/autotuner.py"))
    if not files:
        print("patch_lib: no */utils/autotuner.py -- this fork does not "
              "subclass triton's Autotuner; nothing to fix")
        return 0
    total = 0
    for f in files:
        n = fix_triton_autotuner_super(f)
        if n:
            total += n
            continue
        text = f.read_text(encoding="utf-8", errors="surrogateescape")
        if "_flexgemm_supported" in text:
            print(f"patch_lib: {f}: already signature-filtered")
        elif re.search(r"super\(\)\.__init__\(\s*\n", text):
            raise SystemExit(
                f"PATCH FAILED: {f} forwards to super().__init__ in a shape "
                f"this patch does not recognise -- the wheel would fail to "
                f"import against triton 3.0/3.1")
        else:
            print(f"patch_lib: {f}: no positional super().__init__ forward; "
                  f"nothing to fix")
    return total


# --------------------------------------------------------------------------
# CUDA < 12.6's bundled CCCL/thrust reference `cuda::std` UNQUALIFIED (upstream
# later qualified it to ::cuda::std, which is why 12.6+ toolkits are immune).
# torch's headers bring `c10::cuda` into scope, and MSVC's /permissive- makes
# the lookup strict enough that plain `cuda` becomes ambiguous:
#   error C2872: 'cuda': ambiguous symbol
# Strict conformance is a style choice, not a build requirement, so drop
# /permissive- (and the /Zc:__cplusplus that only matters alongside it) on
# those lanes. Second package to need this (cumesh, then cubvh) -> shared.
# --------------------------------------------------------------------------

_PERMISSIVE_RE = re.compile(
    r'\s*(["\'])(?:-Xcompiler=)?/(?:permissive-|Zc:__cplusplus)\1\s*,?'
)


def strip_permissive_flags(text: str) -> tuple[str, int]:
    """Remove /permissive- and /Zc:__cplusplus (all spellings)."""
    return _PERMISSIVE_RE.subn("", text)


def strip_permissive_for_old_cuda(path: str | Path) -> int:
    """Drop /permissive- when building against CUDA < 12.6 on Windows."""
    if os.name != "nt" or cuda_mm() >= (12, 6):
        return 0
    p = Path(path)
    if not p.exists():
        return 0
    text = p.read_text(encoding="utf-8", errors="surrogateescape")
    new, n = strip_permissive_flags(text)
    if n:
        p.write_text(new, encoding="utf-8", errors="surrogateescape")
        print(f"patch_lib: {p}: dropped {n} /permissive- flag(s) for "
              f"CUDA < 12.6 (C2872 'cuda' ambiguity in bundled CCCL)")
    return n


# --------------------------------------------------------------------------
# Fail-loud helper. Several patch scripts used to print a WARNING and continue
# when an exact-string replace missed, which silently produces a wheel built
# with the wrong flags. Anything load-bearing should use this instead.
# --------------------------------------------------------------------------


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"PATCH FAILED: {message}")


# --------------------------------------------------------------------------
# pybind11 module-local class registrations.
#
# Two builds of the SAME library in one process collide. `cumesh` (JeffreyXiang)
# and `cumesh_vb` (visualbruno) are renamed at the Python level by the fork's
# patch script, but the C++ types keep their upstream identity -- both bind
# `cumesh::CuMesh` as "CuMesh", both bind `cuBVH`/`cuHashTable`/`HashTable`,
# both bind xatlas's `ChartOptions`/`PackOptions`/`XAtlasWrapper`. And the two
# .so sets carry the IDENTICAL pybind11 internals ID (verified on the shipped
# cu130/torch2.11 wheels:
# `__pybind11_internals_v11_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1__`
# in both `_C`), so they share one global type registry. Whichever imports
# second dies with:
#     ImportError: generic_type: type "CuMesh" is already registered!
#
# The fix is pybind11's own remedy for exactly this case. In
# pybind11 v3.0.1 include/pybind11/pybind11.h:1612 the duplicate guard reads
#
#     if ((rec.module_local ? get_local_type_info(*rec.type)
#                           : get_global_type_info(*rec.type)) != nullptr)
#         pybind11_fail("generic_type: type \"...\" is already registered!");
#
# so a `module_local()` registration consults the per-DSO local registry
# (`get_local_internals()`, which lives inside the hidden-visibility `pybind11`
# namespace and is therefore private to each extension module) and cannot see
# -- or collide with -- the other fork's global entry.
#
# WHY NOT a distinct PYBIND11_INTERNALS_ID / PYBIND11_INTERNALS_VERSION:
#   * It is a whole-process ABI statement, not a per-type one. It would split
#     the fork off from the internals it shares with torch and with every
#     other pybind11 module in the interpreter -- exception translators,
#     loader_life_support, the shared type registry -- to solve a name clash
#     in seven classes.
#   * pybind11 v3 hard-#errors on `PYBIND11_INTERNALS_VERSION < 11`
#     (detail/internals.h:45), so the only direction available is *upward*,
#     into the numbers pybind11 reserves for its own future ABI bumps. A
#     future pybind11 that lands on the same number would silently re-merge
#     the registries and the bug returns.
#   * It must be defined identically in EVERY TU of EVERY extension. These
#     forks have three extensions with three separate `extra_compile_args`
#     lists plus vendored third_party sources; one missed TU yields two
#     internals inside a single .so and fails in a far stranger way.
# `module_local()` is local to the registration sites, needs no build-flag
# surgery, and degrades to a no-op if the other fork is never imported.
#
# COST: a module-local type is not shared across modules. That is free here --
# each of the fork's three extensions registers and consumes its own types
# only (the Python layer is the sole glue: bvh.py talks to `_cubvh`, xatlas.py
# to `_xatlas`, cumesh.py to `_C`, and no bound signature names a class
# registered by a different extension). Re-check that before reusing this on
# a package where types DO cross extension boundaries.
# --------------------------------------------------------------------------

# `py::class_<T>(m, "Name")` / `pybind11::class_<T>(scope, "Name")`.
# Non-greedy over the template arguments so `class_<A, B<C>>(m, "X")` still
# terminates at the right `>`. An already-patched site does not re-match: the
# closing paren no longer follows the name string.
_PYBIND_CLASS_RE = re.compile(
    r'((py|pybind11)::class_\s*<.+?>\s*\(\s*[A-Za-z_]\w*\s*,\s*"[^"]+")\s*\)',
    re.S,
)


def add_pybind_module_local(expected: dict[str, int]) -> dict[str, int]:
    """Append `module_local()` to every py::class_ registration in each file.

    `expected` maps a source path to the number of registrations that file
    MUST contain. An exact count is the point: a file that grows a new
    `py::class_` upstream has grown a new collision, and this must fail the
    build rather than localize six of seven types and ship the seventh.

    Returns the per-file counts. Idempotent.
    """
    counts: dict[str, int] = {}
    for rel, want in expected.items():
        p = Path(rel)
        if not p.exists():
            raise SystemExit(
                f"PATCH FAILED: {rel}: pybind11 binding file not found -- "
                "upstream layout changed; the two forks would collide again "
                'with \'generic_type: type "..." is already registered!\'')
        text = p.read_text(encoding="utf-8", errors="surrogateescape")
        already = len(re.findall(r"module_local\s*\(\s*\)", text))
        new, n = _PYBIND_CLASS_RE.subn(r"\1, \2::module_local())", text)
        got = n + already
        if got != want:
            raise SystemExit(
                f"PATCH FAILED: {rel}: expected {want} py::class_ "
                f"registration(s) to make module-local, found {got} "
                f"({n} rewritten, {already} already local) -- upstream "
                "changed; an unlocalized class_ is a live ImportError when "
                "both forks are installed")
        if n:
            p.write_text(new, encoding="utf-8", errors="surrogateescape")
        counts[rel] = got
        print(f"patch_lib: {rel}: {n} py::class_ registration(s) "
              f"-> module_local() ({already} already were)")
    return counts


# --------------------------------------------------------------------------
# glm ships its own documentation: ~1,000 files of doxygen HTML, PNGs and a
# manual PDF, plus its test suite and repo metadata. Packages that vendor glm
# for a header-only include path end up shipping all of it inside a binary
# wheel -- measured on gsplat: 1,236 files, 19.2MB uncompressed / 5.59MB
# compressed, 18% of the wheel and 96% of its file count.
#
# The HEADERS are load-bearing and must survive: gsplat/cuda/_backend.py points
# extra_include_paths at this directory for its runtime JIT fallback, so a
# prune that took them would turn a working fallback into an ImportError on the
# first cache miss. Hence an allowlist of what to delete rather than a filter of
# what to keep, and an explicit assertion afterwards.
# --------------------------------------------------------------------------
_GLM_PRUNE_DIRS = ("doc", "test", ".git", ".github")


def prune_glm_docs(glm_root: str | Path) -> int:
    """Delete glm's docs/tests/repo metadata, keep every header.

    Returns the number of files removed. Raises if the header tree did not
    survive -- silently shipping a glm without headers is worse than the bloat.
    """
    import shutil, stat, sys
    # Windows marks git's pack files read-only, so a plain rmtree of a CLONED
    # glm dies on .git/objects/pack/*.idx with
    #   PermissionError: [WinError 5] Access is denied
    # (measured 2026-08-26, gsplat_maskgaussian windows). Vanilla gsplat never
    # hit it because its glm arrives inside upstream's tree with no .git at all;
    # only the fork, which clones glm itself, has one to delete. Clear the
    # read-only bit and retry -- the standard Windows workaround.
    def _force_rm(func, path, _exc):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except OSError:
            pass          # best effort: leftover metadata is bloat, not breakage
    _rm_kw = ({"onexc": _force_rm} if sys.version_info >= (3, 12)
              else {"onerror": _force_rm})
    root = Path(glm_root)
    if not root.exists():
        print(f"prune_glm_docs: {root} absent -- nothing to prune")
        return 0
    removed = 0
    for sub in _GLM_PRUNE_DIRS:
        d = root / sub
        if d.is_dir():
            removed += sum(1 for _ in d.rglob("*") if _.is_file())
            shutil.rmtree(d, **_rm_kw)
    sentinel = root / "glm" / "gtc" / "type_ptr.hpp"
    require(sentinel.exists(),
            f"prune_glm_docs: {sentinel} is gone after pruning -- the header "
            f"tree was damaged. gsplat's JIT fallback includes this path.")
    n_hdr = sum(1 for _ in (root / "glm").rglob("*")
                if _.suffix in (".hpp", ".inl", ".h"))
    require(n_hdr > 300,
            f"prune_glm_docs: only {n_hdr} headers left under {root}/glm -- "
            f"expected 400+. Refusing to ship a gutted glm.")
    print(f"prune_glm_docs: removed {removed} doc/test/metadata file(s); "
          f"{n_hdr} headers kept")
    return removed


# --------------------------------------------------------------------------
# atomicAdd(double*, double) is a hardware intrinsic from sm_60 (Pascal) on.
# Below that, nvcc reports "no instance of overloaded function atomicAdd
# matches the argument list" and the whole compile dies -- so a package whose
# backward pass accumulates into doubles cannot build for Maxwell.
#
# The farm's rule forbids answering that by deleting sm_50 from the arch list:
# the verify gate compares the wheel against the RESOLVED list, so trimming
# moves both sides of the comparison and the gap vanishes from the record.
#
# NVIDIA publishes the alternative in the CUDA C Programming Guide (section
# B.14, "Atomic Functions"): a compare-and-swap loop over the same 64 bits.
# It is the standard workaround, not an invention, and it is what this injects.
#
# Guarded on `defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600`, which matters in
# both directions: the definition must NOT appear when compiling for sm_60+
# (CUDA already declares it -- redefinition error), and must NOT appear in the
# host pass either (no __device__ context). So the shim exists only in the
# device passes that lack the intrinsic, and is invisible everywhere else.
# --------------------------------------------------------------------------
_ATOMICADD_MARKER = "cuda-wheels: atomicAdd(double*) shim for sm < 60"

_ATOMICADD_SHIM = """
/* %s
   atomicAdd(double*, double) is an intrinsic only from sm_60. Below that,
   emulate it with the CAS loop from the CUDA C Programming Guide (B.14) so
   this translation unit still compiles for Maxwell/Kepler. Compiled ONLY into
   device passes below sm_60; on sm_60+ the real intrinsic is used. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ __forceinline__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);   /* NaN-safe: compares bits, not values */
    return __longlong_as_double(old);
}
#endif
""" % _ATOMICADD_MARKER


def add_atomicadd_double_shim(paths) -> int:
    """Prepend the sm<60 atomicAdd(double*) shim to each file that needs it.

    `paths` is an iterable of .cu/.cuh paths. Files with no `atomicAdd` are
    skipped; already-patched files are skipped (idempotent). Returns the number
    of files changed, and raises if that is zero -- a silent no-op here means
    the arch list was widened on the strength of a fix that never applied.
    """
    changed = 0
    scanned = 0
    for f in paths:
        f = Path(f)
        if not f.is_file():
            continue
        scanned += 1
        text = f.read_text(encoding="utf-8", errors="surrogateescape")
        if "atomicAdd" not in text or _ATOMICADD_MARKER in text:
            continue
        f.write_text(_ATOMICADD_SHIM + text,
                     encoding="utf-8", errors="surrogateescape")
        changed += 1
        print(f"  atomicAdd shim -> {f}")
    require(changed > 0,
            f"add_atomicadd_double_shim: scanned {scanned} file(s) and found no "
            f"atomicAdd call to guard. The arch list was widened below sm_60 on "
            f"the assumption this shim applies -- if upstream no longer uses "
            f"atomicAdd on doubles, restore the floor instead of shipping a "
            f"wheel that cannot compile.")
    return changed


# --------------------------------------------------------------------------
# Debug flags in a RELEASE wheel. Four distinct costs, worst first:
#
#   -G / --device-debug   disables ALL device optimisation and inflates
#                         compile memory enormously. A wheel built with it is
#                         slow at runtime, not merely large.
#   -Og / -O0             "optimise for debugging" / none. Genuinely slower
#                         code. This is the one people miss, because it looks
#                         like an optimisation flag.
#   -lineinfo /           keeps line tables INSIDE the cubin. Note auditwheel
#   --generate-line-info  --strip does NOT remove these: strip only touches
#                         non-allocated ELF sections, and .nv_fatbin is
#                         SHF_ALLOC. So this cost is permanent in the wheel.
#   -g                    host DWARF. Now largely reclaimed by --strip, but
#                         still paid in compile time and peak memory.
#
# Same policy as strip_std_flags: delete the package's opinion. A build farm
# shipping release wheels has no use for any of them, and upstreams hardcode
# them because their setup.py was written for developers building locally.
#
# Deliberately does NOT touch -O1/-O2/-O3 (real optimisation levels) or -DDEBUG
# / -UNDEBUG (assertion control, a correctness choice that is upstream's to
# make).
# --------------------------------------------------------------------------
_DEBUG_FLAGS = ("-G", "--device-debug", "-g", "-Og", "-O0",
                "-lineinfo", "--generate-line-info")


def strip_debug_flags(text: str) -> tuple[str, int]:
    """Remove hardcoded debug/anti-optimisation flags from flag lists.

    Handles quoted list elements ('"-g"', "'-Og'") and the `-Xcompiler=-g`
    spelling. Returns (new_text, count). Leaves -O1/-O2/-O3 and -D/-U alone.
    """
    n = 0

    def _drop(m: re.Match) -> str:
        nonlocal n
        quote, flag = m.group(1), m.group(2)
        base = flag.split("=", 1)[0]
        if base not in _DEBUG_FLAGS and flag not in _DEBUG_FLAGS:
            return m.group(0)
        n += 1
        return ""

    # quoted elements inside any [...] list
    text = re.sub(r"""(['"])(-{1,2}[A-Za-z0-9_-]+)\1,?\s*""", _drop, text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\[\s*,", "[", text)
    return text, n


def strip_debug_flags_in_file(path, label: str = "") -> int:
    """Apply strip_debug_flags to a file in place. Returns the count."""
    p = Path(path)
    if not p.exists():
        return 0
    text = p.read_text(encoding="utf-8", errors="surrogateescape")
    new, n = strip_debug_flags(text)
    if n:
        p.write_text(new, encoding="utf-8", errors="surrogateescape")
        print(f"{label or p}: stripped {n} debug/anti-optimisation flag(s)")
    return n


def force_only_cuda(path: str | Path = "setup.py") -> int:
    """Drop the never-loadable CPU-only extension twin from a pyg-family build.

    torch_scatter / torch_cluster / torch_sparse / torch_spline_conv all build
    the SAME extension twice -- once as `_<name>_cpu`, once as `_<name>_cuda` --
    by iterating `product(main_files, suffices)` with `suffices = ['cpu','cuda']`.

    The `_cpu` half is dead weight in every wheel this farm ships:

      * The package facade loads `cuda_spec or cpu_spec` (torch_scatter/
        __init__.py:15), so with both present the CUDA library ALWAYS wins and
        the CPU twin is never loaded by anything.
      * It is not even a CPU fallback: the `cuda` build compiles
        `csrc/cpu/<name>_cpu.cpp` *and* `csrc/cuda/<name>_cuda.cu`
        (setup.py:90-96), so the CUDA library is a strict superset. CPU tensors
        are served by the same .so.

    Cost measured across the four: 51 of 118 TU compiles, plus ~7.3MB of
    unreachable .so in the wheels.

    Upstream provides the exact switch for this -- `FORCE_ONLY_CUDA=1` sets
    `suffices = ['cuda']` -- but the farm has no per-package env mechanism, so
    set it in setup.py itself, where it is also visible in the post-patch tree.

    Idempotent; raises if the upstream anchor is gone.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="surrogateescape")
    if "cuda-wheels: FORCE_ONLY_CUDA" in text:
        print(f"{p}: FORCE_ONLY_CUDA already applied -- skipping")
        return 0
    anchor = "if os.getenv('FORCE_ONLY_CUDA', '0') == '1':"
    require(
        anchor in text,
        f"{p}: upstream anchor \"{anchor}\" not found. This build no longer "
        "exposes FORCE_ONLY_CUDA, so the CPU twin cannot be dropped safely -- "
        "re-check the patch against the pinned source_tag.",
    )
    shim = (
        "# cuda-wheels: FORCE_ONLY_CUDA -- the '_cpu' extension twin is never\n"
        "# loaded (the facade prefers cuda_spec) and the cuda build already\n"
        "# compiles csrc/cpu/*.cpp, so it is a strict superset. Building it is\n"
        "# 43% of this package's TU compiles for an unreachable .so.\n"
        "#\n"
        "# Gate on FORCE_CUDA, NOT on WITH_CUDA. WITH_CUDA is computed as\n"
        "#     WITH_CUDA = False\n"
        "#     if torch.cuda.is_available():\n"
        "#         WITH_CUDA = CUDA_HOME is not None or torch.version.hip\n"
        "# and a build runner has no GPU, so torch.cuda.is_available() is False\n"
        "# and WITH_CUDA is ALWAYS False here. Keying on it set FORCE_ONLY_CUDA=0\n"
        "# and this whole patch silently did nothing -- caught 2026-08-26 by\n"
        "# tearing the shipped wheel apart, which still contained _scatter_cpu.so\n"
        "# and its three siblings. FORCE_CUDA=1 is what the farm actually exports\n"
        "# (.github/actions/build-wheel/action.yml) and is what makes upstream\n"
        "# choose suffices = ['cuda', 'cpu'] in the first place.\n"
        "_cuw_building_cuda = (WITH_CUDA or os.getenv('FORCE_CUDA', '0') == '1')\n"
        "if _cuw_building_cuda:\n"
        "    os.environ.setdefault('FORCE_ONLY_CUDA', '1')\n"
        "print('[cuda-wheels] building CUDA:', _cuw_building_cuda,\n"
        "      '-> FORCE_ONLY_CUDA=' + os.getenv('FORCE_ONLY_CUDA', '0'))\n"
        + anchor
    )
    text = text.replace(anchor, shim, 1)
    p.write_text(text, encoding="utf-8", errors="surrogateescape")
    print(f"{p}: FORCE_ONLY_CUDA=1 when CUDA is present -- CPU twin not built")
    return 1


def exclude_top_level_packages(names, path: str | Path = "setup.py") -> int:
    """Stop setuptools installing repo scaffolding as top-level packages.

    Every top-level directory a wheel installs lands directly in site-packages,
    which is ONE flat namespace shared by the whole environment. Several
    upstreams ship their dev scaffolding next to the library, so installing
    them silently claims a generic name for the entire interpreter:

        detectron2 -> `tools`     (its training scripts become `import tools`)
        pytorch3d  -> `projects`
        torchao    -> `test`      (collides with CPython's own stdlib `test`)
        flash_attn -> `hopper`    (and it cannot even work -- see below)

    Whichever distribution pip installs last owns the directory, and
    uninstalling that one deletes files the other wrote.

    `names` are added to the existing find_packages(exclude=...) as both the
    bare name and the `name.*` subpackage glob. Adding only one of the two is
    the bug pytorch3d already has: it excludes `projects.*` but not `projects`,
    so the top-level package ships anyway.

    Idempotent. Verifies the result still parses as Python before writing.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="surrogateescape")

    m = re.search(r"find_packages\(\s*exclude\s*=\s*([(\[])", text)
    # Decide "already applied?" from the EXCLUDE CLAUSE ONLY, never from the
    # whole file. A name like "test" or "tools" can easily appear as an
    # unrelated string literal somewhere else in a setup.py, and keying on that
    # would make this helper silently do nothing -- exactly how the
    # FORCE_ONLY_CUDA patch no-op'd for a whole build pass (2026-08-26).
    clause = ""
    if m:
        depth, i = 1, m.end()
        opens, closes = m.group(1), {"(": ")", "[": "]"}[m.group(1)]
        while i < len(text) and depth:
            if text[i] == opens:
                depth += 1
            elif text[i] == closes:
                depth -= 1
            i += 1
        clause = text[m.end():i - 1]
    wanted = [n for n in names
              if f'"{n}"' not in clause and f"'{n}'" not in clause]
    if not wanted:
        print(f"{p}: top-level excludes already present for {list(names)}")
        return 0

    require(
        m is not None,
        f"{p}: no `find_packages(exclude=...)` call found, so the top-level "
        f"package(s) {list(names)} cannot be excluded without rewriting the "
        f"packages= argument. Re-check against the pinned source_tag.",
    )
    ins = m.end()
    added = "".join(f'"{n}", "{n}.*", ' for n in wanted)
    new = text[:ins] + added + text[ins:]

    import ast as _ast
    try:
        _ast.parse(new)
    except SyntaxError as e:
        raise SystemExit(
            f"{p}: injecting the top-level excludes produced invalid Python "
            f"({e}). Refusing to write a broken setup.py.")
    p.write_text(new, encoding="utf-8", errors="surrogateescape")
    print(f"{p}: excluded top-level package(s) {wanted} from the wheel")
    return len(wanted)
