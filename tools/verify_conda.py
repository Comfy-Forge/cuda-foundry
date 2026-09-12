#!/usr/bin/env python3
"""Per-artifact publish gate for a built extension package.

Nothing here is novel: it is cuda-wheels' verify_wheel.py checks (filename,
binary census, ELF sanity, torch linkage, glibc ceiling, SASS archs, import)
re-expressed for .conda, plus conda-torch's packaging invariants (paths.json
matches payload, no RECORD, INSTALLER=conda, $ORIGIN-relative RPATHs), plus
the ones this repo adds:

  * the compile ledger is non-empty and every TU came from THIS build's work
    tree (L3 of the from-source guarantee), anchored on the actual
    rattler-build work directory rather than on the substring "/work/";
  * compiler provenance: every shipped binary was produced by conda-forge's
    toolchain (`.comment` on ELF, the PE linker version on win-64), which is
    what catches an nvcc that fell back to /usr/bin/g++;
  * provenance claims in about.json are DERIVED from the artifact and the
    ledger, not read back as constants;
  * the torch dependency carries a flavour build-glob, without which the
    solver may pair a cu128 extension with a cu130 torch;
  * linkage against the DECLARED closure: every DT_NEEDED / PE import
    resolves through the binary's own RPATH, the declared run deps installed
    into --dep-prefix, or the torch preloader contract; and every declared
    run dep that provides a shared library is actually linked (or
    allowlisted in package.yml `verify.allow_unlinked`).

Every gate here is falsifiable: tools/test_verify_conda.py carries a fixture
that must FAIL for each one. A gate that cannot fail is decoration, and this
file has had several (run 34166741643: `ok no compiled TU came from outside
the build work tree (0 TUs)`).

Usage:
  verify_conda.py <pkg.conda> [...] [--ledger FILE --work-dir DIR]
                  [--expect-arch "8.0 9.0"] [--expect-gcc 13 | --expect-msvc vs2022]
                  [--dep-prefix DIR] [--noarch] [--tmp DIR]
Exit: 0 all pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BUILD_RE = re.compile(
    r"^cuda(?P<cu>\d+)_(?:torch(?P<torch>\d+)_)?py(?P<py>\d+)_h[0-9a-f]+_(?P<n>\d+)$")

# What a linux binary compiled by THIS pipeline may carry in `.comment`:
# conda-forge's gcc, and the crt objects of the conda sysroot (CentOS/RHEL 8
# glibc 2.28, built with Red Hat's gcc 8.5 -- every binary linked against
# that sysroot carries this second entry, from crt1.o/crti.o/crtbegin.o).
# Anything else -- `GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1)` in particular --
# means a translation unit went through the RUNNER's compiler, which is what
# happened to pyg-lib when its recipe clobbered NVCC_PREPEND_FLAGS and nvcc
# fell back to /usr/bin/g++ for its host-side code.
COMMENT_CONDA_FORGE = re.compile(r"^GCC: \(conda-forge gcc (?P<ver>\d+)\.\d+\.\d+-\d+\) ")
COMMENT_SYSROOT = re.compile(r"^GCC: \(GNU\) \d+\.\d+\.\d+ \d{8} \(Red Hat [\d.]+-\d+\)$")
# conda-forge's MSVC activation packages and the linker versions they carry
# (PE optional header MajorLinkerVersion.MinorLinkerVersion == the MSVC
# toolset, 14.2x for VS2019, 14.3x/14.4x for VS2022). MinGW's ld and lld write
# something else entirely (2.x / 14.0), which is the case this exists to
# catch on a platform where there is no `.comment` to read.
MSVC_LINKER = {"vs2019": (20, 30), "vs2022": (30, 50)}

# Libraries the dynamic loader finds without any RPATH, on any manylinux-2.28
# box: glibc itself. Everything else must be resolvable through the artifact,
# the declared closure, or the torch preloader contract.
GLIBC_SONAMES = {
    "libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1",
    "libresolv.so.2", "libutil.so.1", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1",
    "libnsl.so.1", "libcrypt.so.1", "libanl.so.1",
}
# Windows: the OS and the UCRT/VC runtime redistributables, plus the
# interpreter itself (python3XX.dll sits in the env root and is loaded before
# any extension). Everything else must be provided by the artifact, the
# declared closure, or torch/lib.
WIN_SYSTEM_DLLS = re.compile(
    r"^(kernel32|kernelbase|ntdll|user32|gdi32|advapi32|shell32|ole32|oleaut32|"
    r"ws2_32|wsock32|mswsock|ucrtbase|msvcp\d+|vcruntime\d+(_\d)?|concrt\d+|"
    r"api-ms-win-[\w-]+|ext-ms-[\w-]+|bcrypt|crypt32|dbghelp|imagehlp|iphlpapi|"
    r"netapi32|normaliz|psapi|rpcrt4|secur32|shlwapi|userenv|version|winmm|"
    r"wldap32|comdlg32|cfgmgr32|setupapi|powrprof|comctl32|dxgi|d3d1[12]|"
    r"opengl32|glu32|winhttp|wininet|python3\d*)\.dll$", re.I)

# The torch libraries `import torch` loads eagerly. A DT_NEEDED on one of them
# resolves against the already-loaded SONAME in the process, which is why
# make_wheel excludes them from vendoring and why a .conda has no RPATH to
# them: the contract is "torch is imported first". That contract only holds
# for a package that DECLARES pytorch.
TORCH_PRELOADED = re.compile(r"^(lib)?(torch|torch_cpu|torch_cuda|torch_cuda_linalg|"
                             r"torch_python|torch_global_deps|c10|c10_cuda|caffe2_nvrtc|"
                             r"shm|nvfuser_codegen|cudart64_\d+|cublas64_\d+|cublasLt64_\d+|"
                             r"nvrtc64_\d+_\d|nvJitLink_\d+_\d|cudnn64_\d+|cudnn_\w+64_\d+|"
                             r"cufft64_\d+|cufftw64_\d+|curand64_\d+|cusolver64_\d+|"
                             r"cusparse64_\d+|nvToolsExt64_1|libgomp|gomp|uv|asmjit|fbgemm|"
                             r"sleef|libiomp5md|libiompstubs5md|c10_xpu|torch_xpu)"
                             r"(\.so(\.\d+)*|\.dll)$", re.I)


class Report:
    def __init__(self, artifact):
        self.artifact, self.failed = artifact, False

    def ok(self, msg):
        print(f"ok   {msg}")

    def bad(self, msg):
        print(f"FAIL {msg}")
        self.failed = True

    def warn(self, msg):
        print(f"WARN {msg}")

    def check(self, cond, msg):
        (self.ok if cond else self.bad)(msg)
        return cond


# ---------------------------------------------------------------------------
# reading the artifact
# ---------------------------------------------------------------------------

def extract_conda(path: Path, dest: Path) -> Path:
    """Extract a .conda to `dest` (info/ and the payload side by side).

    `rattler-build package extract` when rattler-build is on PATH -- the one
    reader that is guaranteed to agree with what rattler-build wrote -- and
    the zstd CLI otherwise, so this stays runnable on a box with neither
    Rust nor a conda toolchain.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    rb = shutil.which("rattler-build")
    if rb:
        p = subprocess.run([rb, "package", "extract", str(path), "--dest", str(dest)],
                           capture_output=True, text=True)
        if p.returncode == 0 and (dest / "info" / "index.json").is_file():
            return dest
        # fall through: an older rattler-build without the subcommand
    zf = zipfile.ZipFile(path)
    members = [n for n in zf.namelist() if n.endswith(".tar.zst")]
    info_n = [n for n in members if n.startswith("info-")]
    pkg_n = [n for n in members if n.startswith("pkg-")]
    if len(info_n) != 1 or len(pkg_n) != 1:
        sys.exit(f"{path.name}: expected one info- and one pkg- member, "
                 f"found {info_n} / {pkg_n}")
    for member in (info_n[0], pkg_n[0]):
        raw = subprocess.run(["zstd", "-d", "--stdout"], input=zf.read(member),
                             capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            tf.extractall(dest, filter="data")
    return dest


def load_json(root: Path, rel: str, default):
    p = root / rel
    if not p.is_file():
        return default
    return json.loads(p.read_text())


def read_conda(path: Path, tmp: Path | None = None):
    """(index.json, about.json, paths.json, extracted root). Used by verify_wheel."""
    tmp = tmp or Path(os.environ.get("TMPDIR", "/tmp")) / "verify-conda"
    root = extract_conda(path, tmp / path.stem)
    return (load_json(root, "info/index.json", {}), load_json(root, "info/about.json", {}),
            load_json(root, "info/paths.json", {"paths": []}), root)


def payload_files(root: Path) -> dict[str, Path]:
    """Every packaged file (relative path -> absolute), info/ excluded."""
    out = {}
    for p in root.rglob("*"):
        if p.is_dir() and not p.is_symlink():
            continue
        rel = p.relative_to(root).as_posix()
        if rel == "info" or rel.startswith("info/"):
            continue
        out[rel] = p
    return out


# ---------------------------------------------------------------------------
# binary readers
# ---------------------------------------------------------------------------

def is_elf(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def elf_dynamic(p: Path) -> tuple[list[str], list[str]]:
    """(DT_NEEDED sonames, RPATH/RUNPATH entries) via readelf -d."""
    out = subprocess.run(["readelf", "-d", str(p)], capture_output=True, text=True).stdout
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)
    rpaths: list[str] = []
    for line in out.splitlines():
        if "(RPATH)" in line or "(RUNPATH)" in line:
            val = line.split("[", 1)[-1].rstrip("]").strip()
            rpaths.extend(val.split(":"))
    return needed, rpaths


def elf_comment(p: Path) -> list[str]:
    """The `.comment` strings: one per distinct compiler that produced an object."""
    out = subprocess.run(["readelf", "-p", ".comment", str(p)], capture_output=True, text=True).stdout
    return [m.strip() for m in re.findall(r"^\s*\[\s*[0-9a-fx]+\]\s+(.*)$", out, re.M)]


def pe_header(data: bytes):
    """(linker_major, linker_minor, import DLL names) for a PE, or None."""
    if len(data) < 0x40 or data[:2] != b"MZ":
        return None
    (e_lfanew,) = struct.unpack_from("<I", data, 0x3C)
    if len(data) < e_lfanew + 24 or data[e_lfanew:e_lfanew + 4] != b"PE\0\0":
        return None
    coff = e_lfanew + 4
    n_sections, = struct.unpack_from("<H", data, coff + 2)
    size_opt, = struct.unpack_from("<H", data, coff + 16)
    opt = coff + 20
    magic, = struct.unpack_from("<H", data, opt)
    linker_major, linker_minor = struct.unpack_from("<BB", data, opt + 2)
    if magic == 0x20B:      # PE32+
        dd = opt + 112
    elif magic == 0x10B:    # PE32
        dd = opt + 96
    else:
        return None
    n_dd, = struct.unpack_from("<I", data, dd - 4)
    sections = []
    sec = opt + size_opt
    for i in range(n_sections):
        off = sec + i * 40
        if off + 40 > len(data):
            break
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, off + 8)
        sections.append((vaddr, max(vsize, rawsize), rawptr))

    def to_off(rva):
        for vaddr, span, rawptr in sections:
            if vaddr <= rva < vaddr + span:
                return rawptr + (rva - vaddr)
        return None

    def cstr(rva):
        off = to_off(rva)
        if off is None or off >= len(data):
            return None
        end = data.find(b"\0", off)
        return data[off:end if end != -1 else len(data)].decode("ascii", "replace")

    names: set[str] = set()
    # Directory 1: the import table (20-byte descriptors, name RVA at +12);
    # directory 13: delay-load imports (32-byte descriptors, name RVA at +4).
    for index, name_off, stride in ((1, 12, 20), (13, 4, 32)):
        if index >= n_dd:
            continue
        rva, size = struct.unpack_from("<II", data, dd + index * 8)
        if not rva:
            continue
        base = to_off(rva)
        if base is None:
            continue
        for i in range(1024):
            ent = base + i * stride
            if ent + stride > len(data):
                break
            if data[ent:ent + stride] == b"\0" * stride:
                break
            name_rva, = struct.unpack_from("<I", data, ent + name_off)
            if not name_rva:
                continue
            nm = cstr(name_rva)
            if nm:
                names.add(nm)
    return linker_major, linker_minor, names


def pe_imports(data: bytes) -> set[str] | None:
    h = pe_header(data)
    return None if h is None else h[2]


# ---------------------------------------------------------------------------
# package.yml lookups
# ---------------------------------------------------------------------------

# Tests inject a package.yml here rather than writing into packages/.
PACKAGE_CFG_OVERRIDE: dict[str, dict] = {}


def _package_cfg(pkg_name: str) -> dict:
    if not pkg_name:
        return {}
    if pkg_name in PACKAGE_CFG_OVERRIDE:
        return PACKAGE_CFG_OVERRIDE[pkg_name]
    cfg = REPO / "packages" / pkg_name / "package.yml"
    if not cfg.is_file():
        # conda name and folder name agree for every package today; a folder
        # whose package.yml says a different `name` is found by scanning.
        for p in (REPO / "packages").glob("*/package.yml"):
            if re.search(rf"^name:\s*{re.escape(pkg_name)}\s*$", p.read_text(), re.M):
                cfg = p
                break
    if not cfg.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    return yaml.safe_load(cfg.read_text()) or {}


def _verify_field(pkg_name: str, key: str):
    """`verify.<key>` from the package's own package.yml, raw, or None."""
    return (_package_cfg(pkg_name).get("verify") or {}).get(key)


def _expect_linked(pkg_name: str, key: str = "expect_linked") -> list:
    """`verify.expect_linked` (ELF sonames) / `verify.expect_linked_win` (DLLs).

    Read from packages/ rather than passed on the command line so the
    expectation lives beside the host_deps that are supposed to satisfy it,
    and so no caller can forget to pass it. Two lists rather than one plus a
    translation: libjpeg is jpeg8.dll, libpng is libpng16.dll, nvjpeg is
    nvjpeg64_12.dll, and a guessed mapping that silently matched nothing
    would turn the gate into decoration.
    """
    return list(_verify_field(pkg_name, key) or [])


# ---------------------------------------------------------------------------
# the declared closure, from a prefix
# ---------------------------------------------------------------------------

def _credited(pkg: str) -> set[str]:
    """The declared names a package's files count towards.

    conda-forge splits some packages into a metapackage and a per-platform
    payload (`cuda-cudart` -> `cuda-cudart_linux-64`); a recipe declares the
    metapackage, so the payload's files are credited to it. The OpenMP runtime
    is delivered through `libgcc`'s dependency on `_openmp_mutex` -> `libgomp`,
    and the compiler's run_exports say `libgcc`, never `libgomp`, so a
    package linking libgomp.so.1 has declared what conda-forge expects it to
    declare by carrying `libgcc`.
    """
    out = {pkg}
    m = re.match(r"^(.+)_(linux-64|linux-aarch64|win-64|osx-64|osx-arm64)$", pkg)
    if m:
        out.add(m.group(1))
    if pkg in ("_openmp_mutex", "libgomp"):
        out.add("libgcc")
    # Older compiler run_exports (gcc <= 12, which cumm/spconv pin for the
    # manylinux_2_28 GLIBCXX ceiling) say `libgcc-ng` / `libstdcxx-ng`. Those
    # are metapackages whose payload lives in `libgcc` / `libstdcxx`, so the
    # payload's files count towards the legacy name a recipe was made to
    # declare -- run 34626966737 flagged cumm as underlinked against
    # libgcc_s.so.1 while carrying `libgcc-ng >=8`.
    if pkg in ("libgcc", "libgcc_s"):
        out.add("libgcc-ng")
    if pkg == "libstdcxx":
        out.add("libstdcxx-ng")
    # win-64 twin of the libgomp rule: MSVC's OpenMP runtime VCOMP140.DLL is
    # delivered by `vcomp14`, a dependency of the declared `vc14_runtime`
    # (which the `vc` run_export names), so an OpenMP-using .pyd has declared
    # what conda-forge expects -- runs 34695935151 / 34695920949 (torch-sparse,
    # torch-cluster) flagged it as underlinked.
    if pkg == "vcomp14":
        out.add("vc14_runtime")
    return out


def is_link_target(rel: str) -> bool:
    """Whether a file at this prefix-relative path can be a LINK target.

    A dependency provides a shared library only where the dynamic loader
    can be pointed at it: lib/ (and the CUDA packages' targets/<arch>/lib/)
    on linux, Library/bin on win-64. A Python package that ships extension
    modules -- numpy, pillow -- or a Python package's private libraries
    (torchvision-extra-decoders' *.libs/) live under site-packages, are
    imported by the interpreter, and are never DT_NEEDED by anything; the
    first torchvision cell of the rebuilt pipeline was refused as
    "overdepending on numpy, pillow, torchvision-extra-decoders" because
    the census counted them. Excluding site-packages does not weaken the
    torch preloader contract: that resolves through torch/lib by path, not
    through this census.
    """
    p = rel.replace("\\", "/")
    if "/site-packages/" in p or p.startswith("site-packages/"):
        return False
    return (p.startswith("lib/") or re.match(r"^targets/[^/]+/lib/", p) is not None
            or p.startswith("Library/bin/") or "/" not in p)


def closure_providers(prefix: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(soname/dll basename -> packages credited with it, package -> its sonames).

    Read from conda-meta/*.json, which lists every file a package owns. That
    is the exact answer to "which declared dependency provides libcublas.so.12"
    without a hand-written table that would drift the first time NVIDIA
    renumbers a soname.
    """
    owner: dict[str, set[str]] = {}
    provides: dict[str, set[str]] = {}
    meta = prefix / "conda-meta"
    if not meta.is_dir():
        return owner, provides
    for rec in meta.glob("*.json"):
        try:
            d = json.loads(rec.read_text())
        except (OSError, ValueError):
            continue
        name = d.get("name") or rec.stem.rsplit("-", 2)[0]
        for f in d.get("files") or []:
            if not is_link_target(f):
                continue
            base = f.rsplit("/", 1)[-1]
            if re.search(r"\.(so(\.\d+)*|dll)$", base, re.I):
                for credited in _credited(name):
                    owner.setdefault(base.lower(), set()).add(credited)
                    provides.setdefault(credited, set()).add(base.lower())
    return owner, provides


def closure_files(prefix: Path) -> dict[str, set[str]]:
    """directory (prefix-relative, posix) -> basenames (lowercased) of shared libs in it."""
    out: dict[str, set[str]] = {}
    for p in prefix.rglob("*"):
        if p.is_dir():
            continue
        if not re.search(r"\.(so(\.\d+)*|dll)$", p.name, re.I):
            continue
        rel = p.relative_to(prefix).as_posix()
        d, _, b = rel.rpartition("/")
        out.setdefault(d, set()).add(b.lower())
    return out


def _norm_dir(d: str) -> str:
    parts: list[str] = []
    for seg in d.replace("\\", "/").split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/".join(parts)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def _host_solved_build(root: Path, name: str) -> str | None:
    """The build string of `name` in the host env rattler-build finalized, from
    info/recipe/rendered_recipe.yaml; None when the file or entry is absent
    (a hand-written or pre-migration artifact), so the caller can decide."""
    f = root / "info" / "recipe" / "rendered_recipe.yaml"
    if not f.is_file():
        return None
    try:
        import yaml
        d = yaml.safe_load(f.read_text()) or {}
    except Exception:
        return None
    host = (d.get("finalized_dependencies") or {}).get("host") or {}
    for e in (host.get("resolved") if isinstance(host, dict) else None) or []:
        if e.get("name") == name:
            return str(e.get("build") or "") or None
    return None


def verify(path: Path, args, tmp: Path) -> bool:
    rep = Report(path.name)
    print(f"\n=== {path.name} ===")
    root = extract_conda(path, tmp / path.stem)
    index = load_json(root, "info/index.json", {})
    about = load_json(root, "info/about.json", {})
    paths = load_json(root, "info/paths.json", {"paths": []})
    noarch = bool(args.noarch)
    name = index.get("name", "")
    cfg = _package_cfg(name)
    links_torch = cfg.get("links_torch", True) if cfg else None

    # ---- filename / build string agree with the cell it claims -------------
    stem = path.name[: -len(".conda")]
    expected = f"{name}-{index.get('version')}-{index.get('build')}"
    rep.check(stem == expected,
              f"filename matches index.json ({stem} vs {expected})")
    m = None
    if noarch:
        rep.check(index.get("noarch") == "python",
                  f"index.json says noarch: python (got {index.get('noarch')!r})")
        rep.check(index.get("subdir") == "noarch",
                  f"subdir is noarch (got {index.get('subdir')!r})")
    else:
        m = BUILD_RE.match(str(index.get("build", "")))
        rep.check(bool(m), f"build string parses as a cell: {index.get('build')!r}")
        if m and links_torch is not None:
            # A torch-free package has no torch axis in its build string and a
            # torch-linked one must have it; package.yml decides which.
            rep.check(bool(m.group("torch")) == bool(links_torch),
                      f"build string's torch axis agrees with package.yml links_torch="
                      f"{links_torch} ({index.get('build')!r})")
    rep.check("+" not in str(index.get("version", "")),
              "version carries no '+' local tag (conda sorts those BELOW the plain version)")

    # ---- paths.json matches the payload: names, sizes AND hashes -----------
    # The old check compared names only. A file whose bytes differ from what
    # paths.json records installs fine and fails only when a solver verifies
    # the package (or never), so the hash is checked here where it is cheap.
    files = payload_files(root)
    declared = {p["_path"]: p for p in paths.get("paths", [])}
    rep.check(set(declared) == set(files),
              f"paths.json matches payload (declared {len(declared)}, payload {len(files)}; "
              f"only in paths.json: {sorted(set(declared) - set(files))[:3]}, "
              f"only in payload: {sorted(set(files) - set(declared))[:3]})")
    bad_hash, checked_hash = [], 0
    for rel, ent in declared.items():
        p = files.get(rel)
        if p is None or p.is_symlink() or ent.get("path_type") == "softlink":
            continue
        data = p.read_bytes()
        checked_hash += 1
        if ent.get("sha256") and ent["sha256"] != hashlib.sha256(data).hexdigest():
            bad_hash.append(f"{rel}: sha256")
        elif ent.get("size_in_bytes") is not None and ent["size_in_bytes"] != len(data):
            bad_hash.append(f"{rel}: size {ent['size_in_bytes']} != {len(data)}")
        elif not ent.get("sha256"):
            bad_hash.append(f"{rel}: no sha256 recorded")
    rep.check(not bad_hash,
              f"every paths.json sha256/size matches the payload ({checked_hash} files; "
              f"mismatches: {bad_hash[:3]})")

    # ---- the licence travels with the binary --------------------------------
    # An artifact redistributing compiled upstream code carries the upstream
    # licence text under info/licenses/ (recipe about.license_file). A
    # package with no licence file is not publishable; an artifact with none
    # means the recipe forgot, and this is where that is caught.
    lic = [p for p in (root / "info" / "licenses").rglob("*") if p.is_file()] \
        if (root / "info" / "licenses").is_dir() else []
    rep.check(bool(lic), f"info/licenses/ carries the licence text ({len(lic)} file(s))")
    # The PyPI identity is a claim (tools/fragment.py turns it into the purl
    # pixi's conda->pypi map reads). It is stated in package.yml
    # `pypi_project` and recorded in about.extra; index.json carries no
    # `purls` key and is not asked for one. The two statements must agree.
    if cfg:
        want_proj = str(cfg.get("pypi_project") or "").strip()
        got_proj = str((about.get("extra") or {}).get("pypi_project") or "").strip()
        rep.check(want_proj == got_proj,
                  f"about.extra.pypi_project ({got_proj!r}) agrees with package.yml pypi_project "
                  f"({want_proj!r}); a purl is emitted iff it is set")

    # ---- dist-info hygiene (conda-torch's lessons) -------------------------
    rec = [n for n in files if n.endswith(".dist-info/RECORD")]
    rep.check(not rec, "no RECORD in dist-info (pip uninstall would delete conda's files)")
    durl = [n for n in files if n.endswith("direct_url.json")]
    rep.check(not durl, "no direct_url.json (it poisons pip freeze with a build path)")
    inst = [n for n in files if n.endswith(".dist-info/INSTALLER")]
    if inst:
        # rattler-build 0.75 normalises this file during packaging, appending a
        # trailing newline; compare the content, not the trailing whitespace.
        val = files[inst[0]].read_bytes()
        rep.check(val.strip() == b"conda", f"INSTALLER names conda (got {val!r})")

    # ---- dependencies: non-empty, and the torch flavour lock ---------------
    depends = index.get("depends", [])
    rep.check(bool(depends), f"run deps are non-empty ({len(depends)} entries)")
    dep_names = {d.split(" ", 1)[0] for d in depends}
    torch_dep = [d for d in depends if d.split(" ", 1)[0] == "pytorch"]
    if m and m.group("torch"):
        rep.check(bool(torch_dep), "declares a pytorch dependency")
        glob_ok = any(len(d.split()) >= 3 and f"cuda{m.group('cu')}_" in d.split()[2]
                      for d in torch_dep)
        rep.check(glob_ok,
                  f"pytorch dep carries the cuda{m.group('cu')}_* build glob "
                  f"(got {torch_dep!r}) -- the torch we build against exports a "
                  f"flavour-blind run_export, so without this a "
                  f"cu{m.group('cu')} build can pair with another flavour")
    elif m and links_torch is False:
        rep.check(not torch_dep,
                  f"a links_torch: false package declares no pytorch dependency (got {torch_dep!r})")

    # ---- interpreter ABI ----------------------------------------------------
    if m and m.group("py"):
        pytag = m.group("py")
        want = f"{pytag[0]}.{pytag[1:]}"
        abi = [d for d in depends if d.split(" ", 1)[0] == "python_abi"]
        rep.check(bool(abi), f"declares a python_abi dependency (build says py{pytag})")
        rep.check(any(d.split()[1].startswith(want) for d in abi if len(d.split()) >= 2),
                  f"python_abi pins {want} to match the build string (got {abi!r})")

    # ---- states its own GPU requirement -------------------------------------
    if not noarch:
        rep.check("__cuda" in dep_names,
                  "declares __cuda directly (GPU requirement readable without a closure walk)")

    # ---- no build-toolchain passengers -------------------------------------
    STOWAWAY = re.compile(
        r"(^|/)(nvcc|cicc|cudafe\+\+|ptxas|nvlink|fatbinary|nvdisasm|cuobjdump"
        r"|ninja|ccache|patchelf|cc1|cc1plus|ld|as)(\.real|\.exe)?$")
    stowaways = sorted(n for n in files if STOWAWAY.search(n))
    rep.check(not stowaways,
              f"ships no build-toolchain binaries ({stowaways[:3]})")

    # ---- no vendored torch --------------------------------------------------
    # c10, libc10_cuda and libtorch_cuda_linalg were missed by the earlier
    # `(lib)?torch(_cpu|_cuda|_python)?` spelling; a wheel that vendors torch
    # vendors ALL of these, and c10 is the smallest and easiest to overlook.
    VENDORED_TORCH = re.compile(
        r"(^|/)(lib)?(torch(_cpu|_cuda|_python|_cuda_linalg|_global_deps)?|c10|c10_cuda|"
        r"caffe2_nvrtc|shm|nvfuser_codegen)(-[0-9a-f]{8})?\.(so|dll)(\.\d+)*$")
    vendored_torch = [n for n in files if VENDORED_TORCH.search(n)]
    rep.check(not vendored_torch,
              f"ships no vendored torch libraries ({vendored_torch[:3]})")

    # ---- the binaries -------------------------------------------------------
    exts = sorted(n for n in files
                  if re.search(r"\.(so(\.\d+)*|pyd|dll)$", n, re.I) and not files[n].is_symlink())
    elfs = [n for n in exts if is_elf(files[n])]
    pes: dict[str, tuple] = {}
    for n in exts:
        if n in elfs:
            continue
        h = pe_header(files[n].read_bytes())
        if h is not None:
            pes[n] = h
    is_win = bool(pes) and not elfs
    if noarch:
        rep.check(not exts, f"a noarch package ships no compiled modules ({exts[:3]})")
    else:
        rep.check(bool(exts), f"ships compiled extension modules ({len(exts)})")
        rep.check(bool(elfs) != bool(pes),
                  f"binaries are one platform's ({len(elfs)} ELF, {len(pes)} PE)")

    # ---- ELF sanity: $ORIGIN-relative RPATHs, no absolute/empty entries -----
    dyn: dict[str, tuple[list[str], list[str]]] = {}
    bad_rpath = []
    for n in elfs:
        needed, rpaths = elf_dynamic(files[n])
        dyn[n] = (needed, rpaths)
        for entry in rpaths:
            if entry == "" or entry.startswith("/"):
                bad_rpath.append((n, entry))
    if elfs:
        rep.check(not bad_rpath,
                  f"RPATH lint clean over {len(elfs)} ELFs "
                  f"(absolute or empty entries: {bad_rpath[:2]})")

    # ---- declared linkage must be REAL --------------------------------------
    # torchvision prints a warning when a codec is not detected, ships an
    # image extension without it, and still carries libjpeg-turbo/libwebp/
    # libnvjpeg in `depends` via run_exports. Asserting DT_NEEDED here is the
    # fail-closed idea one level out, and it runs on a GPU-less CI box.
    needed_all: set[str] = set()
    for n in elfs:
        needed_all.update(dyn[n][0])
    for n, (_, _, imports) in pes.items():
        needed_all.update(imports)
    expect_linked = _expect_linked(name)
    if expect_linked and exts:
        if is_win:
            expect_win = _expect_linked(name, key="expect_linked_win")
            # FAILS when the Windows list is absent; a warning here is how run
            # 34169055830 published a torchvision win-64 artifact whose .pyd
            # imports libpng16.dll and nothing for jpeg, webp or nvjpeg.
            if rep.check(bool(expect_win),
                         f"declares verify.expect_linked_win, without which the codec "
                         f"gate cannot run on win-64. Imports actually present: "
                         f"{sorted(needed_all)}"):
                low = {x.lower() for x in needed_all}
                missing = [w for w in expect_win if not any(x.startswith(w.lower()) for x in low)]
                interesting = sorted(x for x in needed_all if not WIN_SYSTEM_DLLS.match(x))
                rep.check(not missing,
                          f"imports every DLL package.yml says it must "
                          f"(missing {missing}; non-system imports={interesting})")
        else:
            missing = [w for w in expect_linked if not any(x.startswith(w) for x in needed_all)]
            rep.check(not missing,
                      f"links every library package.yml says it must "
                      f"(missing {missing}; NEEDED={sorted(needed_all)[:8]})")

    # ---- compiler provenance ------------------------------------------------
    # What compiled each shipped binary, read from the binary. On Linux every
    # object records its compiler in `.comment`; the linked module carries the
    # union. On win-64 there is no such section, and the PE optional header's
    # linker version is the toolset that produced it.
    if elfs:
        foreign, no_cf, cf_majors = [], [], set()
        for n in elfs:
            entries = elf_comment(files[n])
            cf = [e for e in entries if COMMENT_CONDA_FORGE.match(e)]
            other = [e for e in entries
                     if not COMMENT_CONDA_FORGE.match(e) and not COMMENT_SYSROOT.match(e)]
            if not cf:
                no_cf.append(n)
            for e in cf:
                cf_majors.add(COMMENT_CONDA_FORGE.match(e).group("ver"))
            for e in other:
                foreign.append((n.rsplit("/", 1)[-1], e))
        rep.check(not foreign and not no_cf,
                  f"every ELF was compiled by conda-forge's gcc and nothing else "
                  f"(foreign compilers: {foreign[:3]}; no conda-forge mark: {no_cf[:3]})")
        if args.expect_gcc:
            rep.check(cf_majors == {str(args.expect_gcc)},
                      f"the conda-forge gcc in .comment is the cell's gcc {args.expect_gcc} "
                      f"(found majors {sorted(cf_majors)})")
    if pes:
        bad_linker = []
        for n, (maj, mino, _) in pes.items():
            lo, hi = MSVC_LINKER.get(args.expect_msvc or "", (20, 50))
            if maj != 14 or not (lo <= mino < hi):
                bad_linker.append((n.rsplit("/", 1)[-1], f"{maj}.{mino}"))
        rep.check(not bad_linker,
                  f"every PE was linked by the conda MSVC toolset "
                  f"({args.expect_msvc or 'vs2019/vs2022'}: linker 14.{MSVC_LINKER.get(args.expect_msvc or '', (20, 50))[0]}-"
                  f"{MSVC_LINKER.get(args.expect_msvc or '', (20, 50))[1] - 1}; "
                  f"offenders: {bad_linker[:3]})")

    # ---- L3: the compile ledger -------------------------------------------
    # What the ledger genuinely proves: (1) an artifact shipping compiled
    # modules compiled SOMETHING, so the binaries did not arrive from a
    # vendored blob or a prebuilt wheel L1 failed to stop; (2) every TU came
    # from THIS build's work tree. It does not prove a particular .so was
    # linked only from ledger TUs -- that needs link-line capture, which the
    # wrapper does not do. Stated plainly rather than implied.
    ledger_ok = None
    if args.ledger is not None and exts and not noarch:
        ledger = args.ledger
        ledger_ok = rep.check(bool(ledger),
                              f"compile ledger is non-empty for an artifact shipping "
                              f"{len(exts)} extension module(s)")
        # "/work/" matched every GitHub runner path (/home/runner/work/...),
        # so the old anchor could not fail. The anchor is now the actual
        # rattler-build work directory, passed in explicitly.
        if rep.check(bool(args.work_dir),
                     "--work-dir given: the build's work tree is known, so 'foreign TU' can be judged"):
            wd = _norm_dir(str(args.work_dir)).lower()

            def _absolute(x):
                return x.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", x) is not None

            def _in_work(x):
                return _norm_dir(x).lower().startswith(wd + "/")

            foreign_tu = sorted(x for x in ledger if _absolute(x) and not _in_work(x))
            ledger_ok = rep.check(not foreign_tu,
                                  f"no compiled TU came from outside {args.work_dir} "
                                  f"({len(ledger)} TUs; foreign: {foreign_tu[:2]})") and ledger_ok
    elif exts and not noarch:
        rep.warn("no --ledger: L3 (compile ledger) not judged; the publish workflow always passes one")

    # ---- provenance: DERIVED, then compared with what about.json claims -----
    # `built_from_source: true` / `prebuilt_wheel_used: false` are template
    # constants; a constant cannot be evidence. The evidence is (a) the
    # ledger -- non-empty, every TU inside the work tree -- and (b) every
    # shipped binary carrying conda-forge's compiler mark, which a wheel
    # fetched from PyPI (manylinux devtoolset, `GCC: (GNU) x.y (Red Hat`
    # alone) or built on the runner (`Ubuntu`) does not. about.json must
    # AGREE with the derivation, so the claim is checked, not copied.
    extra = about.get("extra") or {}
    if not noarch and exts:
        compiled_here = all(
            any(COMMENT_CONDA_FORGE.match(e) for e in elf_comment(files[n])) for n in elfs
        ) if elfs else all(h[0] == 14 for h in pes.values())
        derived_built = bool(compiled_here and (ledger_ok is None or ledger_ok))
        derived_prebuilt = not compiled_here or ledger_ok is False
        rep.check(derived_built,
                  f"derived: built from source (compiler marks on every module: {compiled_here}, "
                  f"ledger: {ledger_ok if ledger_ok is not None else 'not judged'})")
        rep.check(extra.get("built_from_source") is derived_built,
                  f"about.extra.built_from_source agrees with the derivation "
                  f"(claims {extra.get('built_from_source')!r}, derived {derived_built})")
        rep.check(extra.get("prebuilt_wheel_used") is derived_prebuilt,
                  f"about.extra.prebuilt_wheel_used agrees with the derivation "
                  f"(claims {extra.get('prebuilt_wheel_used')!r}, derived {derived_prebuilt})")
    else:
        rep.check(extra.get("built_from_source") is True,
                  f"about.extra.built_from_source is true (got {extra.get('built_from_source')!r})")
        rep.check(extra.get("prebuilt_wheel_used") is False,
                  f"about.extra.prebuilt_wheel_used is false (got {extra.get('prebuilt_wheel_used')!r})")

    # The torch it compiled against: a real build string of the cell's flavour
    # for a torch-linked package, the literal `none` for a torch-free one.
    # `unrecorded` is the template's default and was passing as truthy.
    tb = str(extra.get("torch_build") or "")
    if noarch:
        pass
    elif links_torch is False or (m and not m.group("torch")):
        rep.check(tb == "none",
                  f"torch-free package records torch_build: none (got {tb!r})")
    else:
        # The flavour token must be `repack`: the recipe pins
        # cuda<NNN>_repack_* in host and run, and a stamp naming the mkl mirror
        # records a torch the artifact was never compiled against.
        ok_tb = bool(m) and bool(re.match(rf"^cuda{m.group('cu')}_repack_py{m.group('py')}_h[0-9a-f]+_\d+$", tb))
        rep.check(ok_tb,
                  f"records the exact torch build it compiled against, of the cell's flavour "
                  f"(repack) and python (got {tb!r})")
        # ...and it must be the build the host actually solved, read from the
        # rendered recipe rattler-build embeds. The stamp comes from the matrix
        # generator; run 34694919758 stamped cuda128_mkl_302 while the host had
        # solved cuda128_repack_5. A provenance field that can disagree with
        # the artifact is worse than none.
        solved = _host_solved_build(root, "pytorch")
        rep.check(solved is None or solved == tb,
                  f"torch_build stamp equals the host-solved pytorch build in "
                  f"info/recipe/rendered_recipe.yaml (stamp {tb!r}, solved {solved!r})")
    rev = str(extra.get("source_rev") or "")
    rep.check(bool(re.fullmatch(r"[0-9a-f]{40}", rev)),
              f"source_rev is a 40-hex commit, not a tag or branch ({rev!r})")

    # ---- SASS arch census ---------------------------------------------------
    # Over the UNION of every shipped module: sageattention splits kernels by
    # architecture and keeps the cell's promise as an artifact, not per file.
    if args.expect_arch and exts and not noarch:
        want = {a.replace(".", "").replace("+PTX", "") for a in args.expect_arch.split()}
        got, per_module = set(), {}
        cuobjdump = shutil.which(os.environ.get("CUW_CUOBJDUMP", "cuobjdump"))
        if not rep.check(bool(cuobjdump), "cuobjdump is available for the SASS census (fail closed without it)"):
            exts_for_census = []
        else:
            exts_for_census = exts
        for n in exts_for_census:
            out = subprocess.run([cuobjdump, "--list-elf", str(files[n])],
                                 capture_output=True, text=True).stdout
            archs = set(re.findall(r"sm_(\d+)", out))
            if archs:
                per_module[n.rsplit("/", 1)[-1]] = sorted(archs)
            got |= archs
        no_sass = _verify_field(name, "no_sass")
        if not cuobjdump:
            pass
        elif no_sass:
            # cumm's core_cc is C++ against the runtime and compiles its
            # kernels through NVRTC at run time; "covers the arch list" has no
            # meaning, and the question becomes the opposite one.
            rep.check(not got,
                      f"ships no SASS, as package.yml declares ({str(no_sass).strip()[:60]}...); "
                      f"found {sorted(got)} in {per_module}")
        else:
            # `want <= got or not got` passed an artifact with ZERO SASS -- an
            # extension whose kernels were never compiled in. Empty is a
            # failure unless the package declares verify.no_sass.
            rep.check(bool(got) and want <= got,
                      f"SASS archs cover the cell's arch list (want {sorted(want)}, "
                      f"got {sorted(got)} over {len(exts)} module(s): {per_module})")

    # ---- linkage against the DECLARED closure ------------------------------
    # Two questions, both answered from a prefix holding the declared run
    # closure (the workflow solves one from the local output + the live
    # channels; clean_verify does the same from the live channel alone):
    #   * resolvability: every DT_NEEDED / PE import is found through the
    #     binary's own RPATH inside the artifact or the prefix, through the
    #     torch preloader contract (only if pytorch is declared), or is
    #     glibc / the OS -- anything else fails to load on the user's box;
    #   * overdepending: every declared run dep that provides a shared
    #     library is linked by something in the artifact, unless package.yml
    #     `verify.allow_unlinked` names it with a reason (a dlopen'd library,
    #     a run_export the package needs for headers only).
    if exts and not noarch:
        if args.dep_prefix is None:
            rep.warn("no --dep-prefix: DT_NEEDED resolvability and overdepending not judged; "
                     "the publish workflow always passes one")
        else:
            prefix = Path(args.dep_prefix)
            owner, provides = closure_providers(prefix)
            pfiles = closure_files(prefix)
            rep.check(bool(owner),
                      f"--dep-prefix {prefix} holds an installed closure "
                      f"({len(provides)} package(s) providing shared libraries)")
            torch_declared = bool({"pytorch", "libtorch"} & dep_names)
            torch_libdirs = [d for d in pfiles if d.endswith("torch/lib")] if torch_declared else []
            allow_transitive = {str(x).split()[0].lower()
                                for x in (_verify_field(name, "allow_transitive") or [])}
            # `verify.allow_dso`: libraries that are NOT conda packages by
            # design -- the display driver's libcuda.so.1 / nvcuda.dll for
            # cuTensorMapEncodeTiled (sageattention). The same list feeds
            # rattler-build's missing_dso_allowlist; the resolvability gate
            # honours it too, or run 34627125361 recurs.
            allow_dso = {str(x).split()[0].lower()
                         for x in (_verify_field(name, "allow_dso") or [])}
            unresolved, transitive = [], []

            def _undeclared(so: str) -> str | None:
                """The provider of a resolvable library when NONE of them is declared."""
                provs = owner.get(so.lower(), set())
                if not provs or provs & dep_names or name in provs:
                    return None
                if TORCH_PRELOADED.match(so) and torch_declared:
                    return None
                return "/".join(sorted(provs))

            for n in exts:
                here = n.rsplit("/", 1)[0] if "/" in n else ""
                if n in dyn:
                    needed, rpaths = dyn[n]
                    search = []
                    for r in rpaths:
                        if r.startswith("$ORIGIN"):
                            search.append(_norm_dir(here + r[len("$ORIGIN"):]))
                    for so in needed:
                        key = so.lower()
                        if so in GLIBC_SONAMES or key in allow_dso:
                            continue
                        found = any(
                            (f"{d}/{so}" if d else so) in files or key in pfiles.get(d, ())
                            for d in search)
                        if not found and torch_libdirs and TORCH_PRELOADED.match(so):
                            found = any(key in pfiles[d] for d in torch_libdirs)
                        if not found:
                            unresolved.append((n.rsplit("/", 1)[-1], so, search))
                        elif _undeclared(so) and key not in allow_transitive:
                            transitive.append((so, _undeclared(so)))
                else:
                    _, _, imports = pes[n]
                    # Windows search: the .pyd's own directory, the env root
                    # (python3XX.dll), Library/bin (on PATH once activated),
                    # torch/lib (os.add_dll_directory in `import torch`).
                    search = [here, "", "Library/bin"] + torch_libdirs
                    for dll in imports:
                        key = dll.lower()
                        if WIN_SYSTEM_DLLS.match(dll) or key in allow_dso:
                            continue
                        found = any((f"{d}/{dll}" if d else dll) in files or key in pfiles.get(d, ())
                                    for d in search)
                        if not found:
                            unresolved.append((n.rsplit("/", 1)[-1], dll, search[:3]))
                        elif _undeclared(dll) and key not in allow_transitive:
                            transitive.append((dll, _undeclared(dll)))
            rep.check(not unresolved,
                      f"every DT_NEEDED / PE import resolves through the artifact, the declared "
                      f"closure or the torch preloader contract (unresolved: {unresolved[:3]})")
            # Underlinking, the other direction: a library the binary needs that
            # arrives only because some OTHER dependency happens to drag it in.
            # It loads today and stops loading the day that dependency drops it,
            # which is conda-build's overlinking error and conda-forge's reason
            # for `error_overlinking: true`. `verify.allow_transitive` names a
            # soname with a reason when the indirection is the design.
            rep.check(not transitive,
                      f"every linked library is provided by a DECLARED run dep, not only a "
                      f"transitive one (underlinked: {sorted(set(transitive))[:4]})")

            allow = {str(x).split()[0] for x in (_verify_field(name, "allow_unlinked") or [])}
            needed_low = {x.lower() for x in needed_all}
            over = []
            for dep in sorted(dep_names):
                if dep.startswith("__") or dep in ("python", "python_abi", "pytorch", "libtorch"):
                    continue
                sonames = provides.get(dep)
                if not sonames or dep in allow:
                    continue
                if not (sonames & needed_low):
                    over.append(dep)
            rep.check(not over,
                      f"every declared run dep that provides a shared library is linked "
                      f"(overdepending: {over}; add to verify.allow_unlinked with a reason if "
                      f"it is dlopen'd or header-only)")

    print(f"--- {path.name}: {'FAIL' if rep.failed else 'PASS'}")
    return not rep.failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", nargs="+", type=Path)
    ap.add_argument("--ledger", type=Path,
                    help="compile ledger written by the nvcc wrapper (one TU per line)")
    ap.add_argument("--work-dir", default="",
                    help="rattler-build's work directory for this build "
                         "(.../bld/rattler-build_<name>/work); required with --ledger")
    ap.add_argument("--expect-arch", default="", help="the cell's arch list")
    ap.add_argument("--expect-gcc", default="",
                    help="linux: the cell's gcc major, which .comment must name")
    ap.add_argument("--expect-msvc", default="",
                    help="win-64: vs2019 or vs2022, which the PE linker version must match")
    ap.add_argument("--dep-prefix", type=Path, default=None,
                    help="a prefix holding the package's declared run closure "
                         "(for DT_NEEDED resolvability and overdepending)")
    ap.add_argument("--noarch", action="store_true",
                    help="a pure-python noarch artifact: run the packaging gates only")
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/verify-conda"))
    args = ap.parse_args()
    args.tmp.mkdir(parents=True, exist_ok=True)

    if args.ledger is not None:
        if not args.ledger.is_file():
            print(f"FAIL --ledger {args.ledger} does not exist")
            return 1
        args.ledger = {ln.strip() for ln in args.ledger.read_text().splitlines() if ln.strip()}
    if args.expect_msvc and args.expect_msvc not in MSVC_LINKER:
        sys.exit(f"--expect-msvc must be one of {sorted(MSVC_LINKER)}")

    allok = True
    for a in args.artifacts:
        allok &= verify(a, args, args.tmp)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
