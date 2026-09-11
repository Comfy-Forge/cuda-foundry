#!/usr/bin/env python3
"""Negative controls for every gate in tools/verify_conda.py.

Each gate gets a fixture that must FAIL, next to the one that passes. A gate
without a negative control is decoration: this file exists because several
gates in verify_conda.py were exactly that -- `want <= got or not got` passed
an artifact with no SASS at all, `"/work/" in path` matched every GitHub
runner path, and `bool(extra.get("torch_build"))` accepted the template's
default string "unrecorded".

Fixtures are built here, not downloaded: a tiny shared object compiled with
the host C compiler and then given the `.comment`, DT_NEEDED and RPATH the
case needs (objcopy/patchelf), a hand-assembled PE for the win-64 gates, a
fake conda prefix with conda-meta/ records for the closure gates, and a stub
`cuobjdump` that reports the SASS a fixture claims. Nothing here is a
package build; the C compiler compiles one function so that readelf has an
ELF to read.

Run: python tools/test_verify_conda.py
Needs: cc, objcopy, readelf, patchelf, zstd (or rattler-build on PATH).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_conda as vc  # noqa: E402

CF_GCC = "GCC: (conda-forge gcc 13.4.0-20) 13.4.0"
SYSROOT = "GCC: (GNU) 8.5.0 20210514 (Red Hat 8.5.0-20)"
UBUNTU = "GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
REV = "446e9360ed10d3c8d03925ccb5f220ccb62009de"
TORCH_BUILD = "cuda128_repack_py312_h2bed46fa_7"
WORK = "/home/runner/work/_temp/out/bld/rattler-build_fx/work"

failures: list[str] = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------

def compile_so(dest: Path, comments=(CF_GCC, SYSROOT), needed=(), rpath="$ORIGIN/../../..",
               sass=()):
    """A real ELF shared object with a chosen .comment, DT_NEEDED and RPATH.

    The C compiler's own .comment is REMOVED and replaced, so the fixture says
    what the case needs regardless of which compiler built it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = dest.with_suffix(".c")
    src.write_text("int cuw_fixture(void) { return 42; }\n")
    subprocess.run(["cc", "-shared", "-fPIC", "-o", str(dest), str(src)], check=True)
    src.unlink()
    cbin = dest.with_suffix(".comment")
    cbin.write_bytes(b"".join(c.encode() + b"\0" for c in comments))
    subprocess.run(["objcopy", "--remove-section", ".comment", str(dest)], check=True)
    subprocess.run(["objcopy", "--add-section", f".comment={cbin}", str(dest)], check=True)
    cbin.unlink()
    if sass:
        # The stub cuobjdump (below) reports what this section claims.
        sbin = dest.with_suffix(".sass")
        sbin.write_bytes(("CUW_SASS:" + ",".join(sass)).encode())
        subprocess.run(["objcopy", "--add-section", f".cuw_sass={sbin}", str(dest)], check=True)
        sbin.unlink()
    if rpath is not None:
        subprocess.run(["patchelf", "--set-rpath", rpath, str(dest)], check=True)
    for n in needed:
        subprocess.run(["patchelf", "--add-needed", n, str(dest)], check=True)


def build_pe(dest: Path, imports: list[str], linker=(14, 44)) -> None:
    """A minimal PE32+ DLL carrying an import directory naming `imports`.

    Enough of the format for verify_conda.pe_header: DOS stub with e_lfanew,
    COFF header, PE32+ optional header with 16 data directories, one section
    holding the import descriptors and the DLL name strings.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    sec_rva, sec_raw = 0x1000, 0x200
    # import descriptors (20 bytes each) + null terminator, then names
    names_off = 20 * (len(imports) + 1)
    blob = bytearray()
    name_rvas = []
    cursor = names_off
    for nm in imports:
        name_rvas.append(sec_rva + cursor)
        cursor += len(nm) + 1
    for rva in name_rvas:
        blob += struct.pack("<IIIII", 0, 0, 0, rva, 0)
    blob += b"\0" * 20
    for nm in imports:
        blob += nm.encode() + b"\0"
    raw_size = (len(blob) + 0x1FF) & ~0x1FF
    blob += b"\0" * (raw_size - len(blob))

    dos = bytearray(0x40)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x40)
    coff = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 240, 0x2022)
    opt = bytearray(240)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<BB", opt, 2, linker[0], linker[1])
    struct.pack_into("<Q", opt, 24, 0x180000000)        # ImageBase
    struct.pack_into("<II", opt, 32, 0x1000, 0x200)     # Section/File alignment
    struct.pack_into("<HH", opt, 40, 6, 0)              # OS version
    struct.pack_into("<HH", opt, 48, 6, 0)              # subsystem version
    struct.pack_into("<I", opt, 56, sec_rva + raw_size)  # SizeOfImage
    struct.pack_into("<I", opt, 60, 0x200)              # SizeOfHeaders
    struct.pack_into("<HH", opt, 68, 2, 0x160)          # subsystem, dll characteristics
    struct.pack_into("<I", opt, 108, 16)                # NumberOfRvaAndSizes
    struct.pack_into("<II", opt, 112 + 1 * 8, sec_rva, 20 * (len(imports) + 1))  # import dir
    sec = struct.pack("<8sIIIIIIHHI", b".idata\0\0", len(blob), sec_rva, raw_size, sec_raw,
                      0, 0, 0, 0, 0x40000040)
    header = bytes(dos) + b"PE\0\0" + coff + bytes(opt) + sec
    header += b"\0" * (sec_raw - len(header))
    dest.write_bytes(header + bytes(blob))


def make_conda(out: Path, tree: Path, index: dict, about: dict, corrupt_hash: str | None = None,
               extra_declared: list[str] | None = None, licence: bool = True) -> Path:
    """Package `tree` (payload files) as <name>-<version>-<build>.conda."""
    entries = []
    for p in sorted(tree.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(tree).as_posix()
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if rel == corrupt_hash:
            sha = "0" * 64
        entries.append({"_path": rel, "path_type": "hardlink", "sha256": sha,
                        "size_in_bytes": len(data)})
    for rel in extra_declared or []:
        entries.append({"_path": rel, "path_type": "hardlink", "sha256": "0" * 64, "size_in_bytes": 0})
    info = tree.parent / "info-src"
    shutil.rmtree(info, ignore_errors=True)
    (info / "info").mkdir(parents=True)
    if licence:
        (info / "info" / "licenses").mkdir()
        (info / "info" / "licenses" / "LICENSE").write_text("MIT\n")
    (info / "info" / "index.json").write_text(json.dumps(index))
    (info / "info" / "about.json").write_text(json.dumps(about))
    (info / "info" / "paths.json").write_text(json.dumps({"paths": entries, "paths_version": 1}))

    def tar_zst(src: Path, arcname_root: str | None) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for p in sorted(src.rglob("*")):
                tf.add(p, arcname=p.relative_to(src).as_posix(), recursive=False)
        return subprocess.run(["zstd", "-q", "--stdout"], input=buf.getvalue(),
                              capture_output=True, check=True).stdout

    stem = f"{index['name']}-{index['version']}-{index['build']}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.conda"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("metadata.json", json.dumps({"conda_pkg_format_version": 2}))
        z.writestr(f"info-{stem}.tar.zst", tar_zst(info, None))
        z.writestr(f"pkg-{stem}.tar.zst", tar_zst(tree, None))
    return path


def make_prefix(root: Path, packages: dict[str, list[str]]) -> Path:
    """A fake conda prefix: conda-meta records plus the (empty) files they own."""
    shutil.rmtree(root, ignore_errors=True)
    (root / "conda-meta").mkdir(parents=True)
    for name, files in packages.items():
        names = [f.split("->", 1)[0] for f in files]
        (root / "conda-meta" / f"{name}-1.0-h0_0.json").write_text(
            json.dumps({"name": name, "version": "1.0", "build": "h0_0", "files": names}))
        for f in files:
            # "lib/x.so->../targets/x86_64-linux/lib/x.so" makes a symlink, the
            # layout conda-forge's CUDA packages actually have.
            rel, _, target = f.partition("->")
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if target:
                p.symlink_to(target)
            else:
                p.write_bytes(b"")
    return root


STUB_CUOBJDUMP = """#!/usr/bin/env python3
import re, sys
data = open(sys.argv[-1], "rb").read()
m = re.search(rb"CUW_SASS:([0-9,]+)", data)
if m:
    for a in m.group(1).decode().split(","):
        print(f"ELF file    1: fixture.{a}.sm_{a}.cubin")
"""


def base_index(**over):
    d = {"name": "fx", "version": "1.0", "build": "cuda128_torch28_py312_hdeadbeef_0",
         "build_number": 0, "subdir": "linux-64", "platform": "linux", "arch": "x86_64",
         "depends": ["python", "__cuda", "pytorch 2.8.* cuda128_*", "python_abi 3.12.* *_cp312",
                     "libstdcxx >=13", "libgcc >=13", "cuda-cudart >=12.8"]}
    d.update(over)
    return d


def base_about(**over):
    e = {"built_from_source": True, "prebuilt_wheel_used": False, "torch_build": TORCH_BUILD,
         "source_rev": REV, "arch_list": "8.0 9.0"}
    e.update(over)
    return {"extra": e}


def run_verify(conda: Path, tmp: Path, **kw) -> tuple[bool, str]:
    args = SimpleNamespace(ledger=None, work_dir="", expect_arch="", expect_gcc="",
                           expect_msvc="", dep_prefix=None, noarch=False)
    for k, v in kw.items():
        setattr(args, k, v)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = vc.verify(conda, args, tmp)
    return ok, buf.getvalue()


def fails_on(out: str, needle: str) -> bool:
    return any(ln.startswith("FAIL") and needle in ln for ln in out.splitlines())


# ---------------------------------------------------------------------------

def main() -> int:
    for tool in ("cc", "objcopy", "readelf", "patchelf", "zstd"):
        if not shutil.which(tool):
            print(f"FAIL {tool} not on PATH; this test needs it")
            return 1
    td = Path(tempfile.mkdtemp(prefix="cuw-verify-test-"))
    tmp = td / "tmp"
    stub = td / "bin" / "cuobjdump"
    stub.parent.mkdir()
    stub.write_text(STUB_CUOBJDUMP)
    stub.chmod(0o755)
    os.environ["CUW_CUOBJDUMP"] = str(stub)

    prefix = make_prefix(td / "prefix", {
        "pytorch": ["lib/python3.12/site-packages/torch/lib/libc10.so",
                    "lib/python3.12/site-packages/torch/lib/libtorch_cpu.so"],
        "libtorch": ["lib/libc10.so", "lib/libtorch_cpu.so"],
        "cuda-cudart": ["lib/libcudart.so.12"],
        "cuda-cudart_linux-64": ["targets/x86_64-linux/lib/libcudart.so.12"],
        "libstdcxx": ["lib/libstdc++.so.6"],
        "libgcc": ["lib/libgcc_s.so.1"],
        "_openmp_mutex": ["lib/libgomp.so.1"],
        "libcublas": ["lib/libcublas.so.12", "lib/libcublasLt.so.12"],
        "libpng": ["lib/libpng16.so.16"],
    })
    win_prefix = make_prefix(td / "win-prefix", {
        "pytorch": ["Lib/site-packages/torch/lib/c10.dll", "Lib/site-packages/torch/lib/torch_cpu.dll"],
        "cuda-cudart": [],
        "cuda-cudart_win-64": ["Library/bin/cudart64_12.dll"],
        "python": ["python312.dll"],
        "libpng": ["Library/bin/libpng16.dll"],
    })

    NEEDED = ["libc10.so", "libcudart.so.12", "libstdc++.so.6", "libgcc_s.so.1"]

    def good_tree(root: Path, **so_kw) -> Path:
        shutil.rmtree(root, ignore_errors=True)
        sp = root / "lib" / "python3.12" / "site-packages"
        (sp / "fx").mkdir(parents=True)
        (sp / "fx" / "__init__.py").write_text("from . import _C\n")
        di = sp / "fx-1.0.dist-info"
        di.mkdir()
        (di / "INSTALLER").write_bytes(b"conda\n")
        (di / "METADATA").write_text("Metadata-Version: 2.1\nName: fx\nVersion: 1.0\n")
        kw = dict(needed=NEEDED, sass=("80", "90"))
        kw.update(so_kw)
        compile_so(sp / "fx" / "_C.so", **kw)
        return root

    ledger_good = {f"{WORK}/csrc/a.cu", f"{WORK}/csrc/b.cpp", "../src/relative.cu"}
    common = dict(expect_arch="8.0 9.0", expect_gcc="13", dep_prefix=prefix,
                  ledger=ledger_good, work_dir=WORK)

    # ---- the positive control: a fixture that passes every gate ----------
    good = make_conda(td / "out", good_tree(td / "good"), base_index(), base_about())
    ok, out = run_verify(good, tmp, **common)
    check(ok, "positive control: the good fixture passes every gate")
    if not ok:
        print(out)

    # ---- (e) paths.json: hash and size, not names only -------------------
    bad = make_conda(td / "out-hash", good_tree(td / "hash"), base_index(), base_about(),
                     corrupt_hash="lib/python3.12/site-packages/fx/__init__.py")
    ok, out = run_verify(bad, tmp, **common)
    check(not ok and fails_on(out, "sha256/size"),
          "(e) a payload file whose sha256 disagrees with paths.json FAILS")
    bad = make_conda(td / "out-names", good_tree(td / "names"), base_index(), base_about(),
                     extra_declared=["lib/python3.12/site-packages/fx/ghost.py"])
    ok, out = run_verify(bad, tmp, **common)
    check(not ok and fails_on(out, "paths.json matches payload"),
          "(e) a paths.json entry with no payload file FAILS")

    # ---- (a) SASS census: empty is a failure ------------------------------
    bad = make_conda(td / "out-sass", good_tree(td / "sass", sass=()), base_index(), base_about())
    ok, out = run_verify(bad, tmp, **common)
    check(not ok and fails_on(out, "SASS archs cover"),
          "(a) an extension with ZERO SASS FAILS the census (it used to pass)")
    bad = make_conda(td / "out-sass2", good_tree(td / "sass2", sass=("80",)), base_index(), base_about())
    ok, out = run_verify(bad, tmp, **common)
    check(not ok and fails_on(out, "SASS archs cover"),
          "(a) an extension missing one of the cell's archs FAILS")
    vc.PACKAGE_CFG_OVERRIDE["fx"] = {"verify": {"no_sass": "runtime NVRTC"}}
    ok, out = run_verify(make_conda(td / "out-nosass", good_tree(td / "nosass", sass=()),
                                    base_index(), base_about()), tmp, **common)
    check(ok, "(a) ...unless package.yml declares verify.no_sass, which then PASSES with no SASS")
    ok, out = run_verify(make_conda(td / "out-nosass2", good_tree(td / "nosass2"),
                                    base_index(), base_about()), tmp, **common)
    check(not ok and fails_on(out, "ships no SASS"),
          "(a) ...and a no_sass package that DOES ship SASS FAILS")
    vc.PACKAGE_CFG_OVERRIDE.clear()
    os.environ["CUW_CUOBJDUMP"] = "/nonexistent/cuobjdump"
    ok, out = run_verify(good, tmp, **common)
    check(not ok and fails_on(out, "cuobjdump is available"),
          "(a) a missing cuobjdump FAILS closed instead of skipping the census")
    os.environ["CUW_CUOBJDUMP"] = str(stub)

    # ---- (b) L3: anchored on the real work dir ----------------------------
    ok, out = run_verify(good, tmp, **dict(common, ledger={f"{WORK}/a.cu", "/home/runner/work/other/x.cu"}))
    check(not ok and fails_on(out, "no compiled TU came from outside"),
          "(b) a TU under /home/runner/work/ but outside the build's work dir FAILS "
          "(the old '/work/' substring accepted it)")
    ok, out = run_verify(good, tmp, **dict(common, ledger=set()))
    check(not ok and fails_on(out, "compile ledger is non-empty"),
          "(b) an empty ledger beside a shipped module FAILS")
    ok, out = run_verify(good, tmp, **dict(common, work_dir=""))
    check(not ok and fails_on(out, "--work-dir given"),
          "(b) a ledger without --work-dir FAILS rather than judging nothing")
    ok, out = run_verify(good, tmp, **dict(common, ledger={"D:\\a\\_temp\\out\\bld\\rattler-build_fx\\work\\x.cu"},
                                           work_dir="D:/a/_temp/out/bld/rattler-build_fx/work"))
    check(ok, "(b) a Windows TU path under the Windows work dir PASSES (backslashes, drive letter)")

    # ---- (c) torch_build -------------------------------------------------
    ok, out = run_verify(make_conda(td / "out-tb", good_tree(td / "tb"), base_index(),
                                    base_about(torch_build="unrecorded")), tmp, **common)
    check(not ok and fails_on(out, "torch build"),
          "(c) torch_build 'unrecorded' (the template default) FAILS")
    ok, out = run_verify(make_conda(td / "out-tb2", good_tree(td / "tb2"), base_index(),
                                    base_about(torch_build="cuda130_repack_py312_h2bed46fa_7")), tmp, **common)
    check(not ok and fails_on(out, "torch build"),
          "(c) a torch_build of another CUDA flavour than the build string FAILS")
    vc.PACKAGE_CFG_OVERRIDE["fx"] = {"links_torch": False}
    tf_index = base_index(build="cuda128_py312_hdeadbeef_0",
                          depends=["python", "__cuda", "python_abi 3.12.* *_cp312",
                                   "libstdcxx >=13", "libgcc >=13", "cuda-cudart >=12.8"])
    tree = good_tree(td / "tf", needed=["libcudart.so.12", "libstdc++.so.6", "libgcc_s.so.1"])
    ok, out = run_verify(make_conda(td / "out-tf", tree, tf_index, base_about()), tmp, **common)
    check(not ok and fails_on(out, "torch_build: none"),
          "(c) a links_torch: false package recording a real torch build FAILS")
    ok, out = run_verify(make_conda(td / "out-tf2", tree, tf_index, base_about(torch_build="none")), tmp, **common)
    check(ok, "(c) ...and PASSES with torch_build: none")
    vc.PACKAGE_CFG_OVERRIDE.clear()

    # ---- (d) provenance derived, not copied -------------------------------
    ok, out = run_verify(make_conda(td / "out-bfs", good_tree(td / "bfs"), base_index(),
                                    base_about(built_from_source=False)), tmp, **common)
    check(not ok and fails_on(out, "built_from_source agrees"),
          "(d) about.json claiming built_from_source: false on a from-source artifact FAILS")
    tree = good_tree(td / "prebuilt", comments=(SYSROOT,))   # a manylinux wheel's mark
    ok, out = run_verify(make_conda(td / "out-prebuilt", tree, base_index(), base_about()), tmp, **common)
    check(not ok and fails_on(out, "derived: built from source") and fails_on(out, "prebuilt_wheel_used agrees"),
          "(d) a module without conda-forge's compiler mark is derived as prebuilt, and the "
          "template's `prebuilt_wheel_used: false` constant FAILS against that derivation")

    # ---- (g) compiler provenance ------------------------------------------
    tree = good_tree(td / "ubuntu", comments=(CF_GCC, UBUNTU, SYSROOT))
    ok, out = run_verify(make_conda(td / "out-ubuntu", tree, base_index(), base_about()), tmp, **common)
    check(not ok and fails_on(out, "compiled by conda-forge's gcc and nothing else"),
          "(g) pyg-lib's case: an Ubuntu gcc entry beside the conda-forge one FAILS")
    ok, out = run_verify(good, tmp, **dict(common, expect_gcc="12"))
    check(not ok and fails_on(out, "is the cell's gcc 12"),
          "(g) a conda-forge gcc of a different major than the cell's FAILS")

    # ---- (h) vendored torch ----------------------------------------------
    tree = good_tree(td / "c10")
    (tree / "lib" / "python3.12" / "site-packages" / "fx" / "libc10.so").write_bytes(b"\x7fELFjunk")
    ok, out = run_verify(make_conda(td / "out-c10", tree, base_index(), base_about()), tmp, **common)
    check(not ok and fails_on(out, "vendored torch"),
          "(h) a vendored libc10.so FAILS (the old regex missed c10)")
    tree = good_tree(td / "linalg")
    (tree / "lib" / "python3.12" / "site-packages" / "fx" / "libtorch_cuda_linalg.so").write_bytes(b"\x7fELFjunk")
    ok, out = run_verify(make_conda(td / "out-linalg", tree, base_index(), base_about()), tmp, **common)
    check(not ok and fails_on(out, "vendored torch"),
          "(h) a vendored libtorch_cuda_linalg.so FAILS")

    # ---- (i) source_rev ---------------------------------------------------
    ok, out = run_verify(make_conda(td / "out-rev", good_tree(td / "rev"), base_index(),
                                    base_about(source_rev="v2.8.3")), tmp, **common)
    check(not ok and fails_on(out, "40-hex"),
          "(i) flash-attn's case: source_rev 'v2.8.3' FAILS")

    # ---- (f) resolvability ------------------------------------------------
    tree = good_tree(td / "unres", needed=NEEDED + ["libnvjpeg.so.12"])
    ok, out = run_verify(make_conda(td / "out-unres", tree, base_index(), base_about()), tmp, **common)
    check(not ok and fails_on(out, "resolves through the artifact"),
          "(f) a DT_NEEDED that nothing in the closure provides FAILS")
    tree = good_tree(td / "norpath", rpath="$ORIGIN")
    ok, out = run_verify(make_conda(td / "out-norpath", tree, base_index(), base_about()), tmp, **common)
    check(not ok and fails_on(out, "resolves through the artifact"),
          "(f) an RPATH that does not reach lib/ leaves libcudart unresolved and FAILS")
    tree = good_tree(td / "notorch", needed=NEEDED)
    idx = base_index(build="cuda128_py312_hdeadbeef_0",
                     depends=["python", "__cuda", "python_abi 3.12.* *_cp312", "libstdcxx >=13",
                              "libgcc >=13", "cuda-cudart >=12.8"])
    # The closure a torch-free package's declared deps solve to has no torch
    # in it, so the prefix for this case must not either.
    notorch_prefix = make_prefix(td / "prefix-notorch", {
        "cuda-cudart": ["lib/libcudart.so.12"], "libstdcxx": ["lib/libstdc++.so.6"],
        "libgcc": ["lib/libgcc_s.so.1"]})
    ok, out = run_verify(make_conda(td / "out-notorch", tree, idx, base_about(torch_build="none")), tmp,
                         **dict(common, dep_prefix=notorch_prefix))
    check(not ok and fails_on(out, "resolves through the artifact"),
          "(f) libc10.so is NOT resolvable for a package that does not declare pytorch "
          "(the preloader contract needs the declaration)")
    ok, out = run_verify(make_conda(td / "out-notorch2", tree, idx, base_about(torch_build="none")), tmp, **common)
    check(not ok and fails_on(out, "underlinked"),
          "(f) ...and where a torch happens to sit in the prefix anyway, it FAILS as underlinked")

    # ---- (j) overdepending / underlinking --------------------------------
    idx = base_index(depends=base_index()["depends"] + ["libcublas >=12"])
    ok, out = run_verify(make_conda(td / "out-over", good_tree(td / "over"), idx, base_about()), tmp, **common)
    check(not ok and fails_on(out, "overdepending: ['libcublas']"),
          "(j) a declared libcublas that nothing links FAILS as overdepending")
    vc.PACKAGE_CFG_OVERRIDE["fx"] = {"verify": {"allow_unlinked": ["libcublas"]}}
    ok, out = run_verify(make_conda(td / "out-over2", good_tree(td / "over2"), idx, base_about()), tmp, **common)
    check(ok, "(j) ...and PASSES when package.yml verify.allow_unlinked names it")
    vc.PACKAGE_CFG_OVERRIDE.clear()
    tree = good_tree(td / "under", needed=NEEDED + ["libpng16.so.16"])
    ok, out = run_verify(make_conda(td / "out-under", tree, base_index(), base_about()), tmp, **common)
    check(not ok and fails_on(out, "underlinked: [('libpng16.so.16', 'libpng')]"),
          "(j) a linked libpng16 provided only by an undeclared package FAILS as underlinking")
    vc.PACKAGE_CFG_OVERRIDE["fx"] = {"verify": {"allow_transitive": ["libpng16.so.16"]}}
    ok, out = run_verify(make_conda(td / "out-under2", tree, base_index(), base_about()), tmp, **common)
    check(ok, "(j) ...and PASSES when package.yml verify.allow_transitive names the soname")
    vc.PACKAGE_CFG_OVERRIDE.clear()
    # a DT_NEEDED on the display driver (libcuda.so.1) is not resolvable
    # through any conda package by design; verify.allow_dso is the declared
    # exception -- run 34627125361 (sageattention _qattn_sm90) must pass.
    tree = good_tree(td / "drv", needed=NEEDED + ["libcuda.so.1"])
    ok, out = run_verify(make_conda(td / "out-drv", tree, base_index(), base_about()), tmp, **common)
    check(not ok and fails_on(out, "unresolved: [('"), "(j) a DT_NEEDED on libcuda.so.1 FAILS as unresolvable by default")
    vc.PACKAGE_CFG_OVERRIDE["fx"] = {"verify": {"allow_dso": ["libcuda.so.1"]}}
    ok, out = run_verify(make_conda(td / "out-drv2", tree, base_index(), base_about()), tmp, **common)
    check(ok, "(j) ...and PASSES when package.yml verify.allow_dso names it")
    vc.PACKAGE_CFG_OVERRIDE.clear()
    tree = good_tree(td / "gomp", needed=NEEDED + ["libgomp.so.1"])
    ok, out = run_verify(make_conda(td / "out-gomp", tree, base_index(), base_about()), tmp, **common)
    check(ok, "(j) libgomp.so.1 is credited to the declared libgcc (delivered via _openmp_mutex)")
    # A Python package that ships extension modules or private libraries is
    # imported, never linked: declaring it is not overdepending (torchvision
    # cell 34624209472 was refused for numpy, pillow and
    # torchvision-extra-decoders). A lib/ provider nothing links still is.
    py_prefix = make_prefix(td / "prefix-py", {
        "pytorch": ["lib/python3.12/site-packages/torch/lib/libc10.so"],
        "cuda-cudart": ["lib/libcudart.so.12"],
        "libstdcxx": ["lib/libstdc++.so.6"],
        "libgcc": ["lib/libgcc_s.so.1"],
        "numpy": ["lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so",
                  "lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-ff651d7f.so"],
        "torchvision-extra-decoders": ["lib/python3.12/site-packages/torchvision_extra_decoders/extra_decoders_lib.so",
                                       "lib/python3.12/site-packages/torchvision_extra_decoders.libs/libavif-1a2b3c4d.so.16"],
        "libfoo": ["lib/libfoo.so.1"],
    })
    idx = base_index(depends=base_index()["depends"] + ["numpy >=1.25", "torchvision-extra-decoders"])
    ok, out = run_verify(make_conda(td / "out-pydep", good_tree(td / "pydep"), idx, base_about()), tmp,
                         **dict(common, dep_prefix=py_prefix))
    check(ok, "(j) declared numpy / torchvision-extra-decoders (site-packages DSOs only) are NOT overdepending")
    if not ok:
        print(out)
    idx = base_index(depends=base_index()["depends"] + ["numpy >=1.25", "libfoo"])
    ok, out = run_verify(make_conda(td / "out-libfoo", good_tree(td / "libfoo"), idx, base_about()), tmp,
                         **dict(common, dep_prefix=py_prefix))
    check(not ok and fails_on(out, "overdepending: ['libfoo']"),
          "(j) ...while a declared lib/libfoo.so.1 that nothing links IS flagged, and numpy still is not")
    tree = good_tree(td / "pyprivate", needed=NEEDED + ["libavif.so.16"])
    ok, out = run_verify(make_conda(td / "out-pyprivate", tree, base_index(), base_about()), tmp,
                         **dict(common, dep_prefix=py_prefix))
    check(not ok and fails_on(out, "resolves through the artifact") and not fails_on(out, "underlinked"),
          "(j) a DT_NEEDED on a Python package's private DSO is unresolvable, not 'underlinked against it'")
    ok, out = run_verify(good, tmp, **dict(common, dep_prefix=td / "empty-prefix"))
    check(not ok and fails_on(out, "holds an installed closure"),
          "(f/j) a --dep-prefix with no conda-meta FAILS rather than resolving nothing")

    # ---- licence text, pypi_project ----------------------------------------
    ok, out = run_verify(make_conda(td / "out-lic", good_tree(td / "lic"), base_index(), base_about(),
                                    licence=False), tmp, **common)
    check(not ok and fails_on(out, "info/licenses/"),
          "an artifact with no info/licenses/ FAILS")
    vc.PACKAGE_CFG_OVERRIDE["fx"] = {"pypi_project": "fx"}
    ok, out = run_verify(good, tmp, **common)
    check(not ok and fails_on(out, "pypi_project"),
          "package.yml pypi_project set but about.extra.pypi_project absent FAILS")
    ok, out = run_verify(make_conda(td / "out-proj", good_tree(td / "proj"), base_index(),
                                    base_about(pypi_project="fx")), tmp, **common)
    check(ok, "...and PASSES when about.extra records the same project (no purls key needed)")
    vc.PACKAGE_CFG_OVERRIDE.clear()

    # ---- symlink attribution (rattler-build 0.75's own check gets this wrong)
    # conda-forge's cuda-cudart owns lib/libcudart.so.12 as a SYMLINK into
    # targets/x86_64-linux/lib/, where cuda-cudart_linux-64 owns the real
    # file. A package declaring `cuda-cudart` is neither overdepending nor
    # underlinked: the declared name is credited with the real file through
    # the symlink AND through the _linux-64 split.
    sym_prefix = make_prefix(td / "prefix-sym", {
        "pytorch": ["lib/python3.12/site-packages/torch/lib/libc10.so"],
        "cuda-cudart": ["lib/libcudart.so.12->../targets/x86_64-linux/lib/libcudart.so.12"],
        "cuda-cudart_linux-64": ["targets/x86_64-linux/lib/libcudart.so.12"],
        "libstdcxx": ["lib/libstdc++.so.6"],
        "libgcc": ["lib/libgcc_s.so.1"],
    })
    ok, out = run_verify(good, tmp, **dict(common, dep_prefix=sym_prefix))
    check(ok, "a declared cuda-cudart whose lib/libcudart.so.12 is a symlink into the "
              "_linux-64 split is credited with the real file: no overdepending, no underlinking")
    if not ok:
        print(out)
    split_only = make_prefix(td / "prefix-split", {
        "pytorch": ["lib/python3.12/site-packages/torch/lib/libc10.so"],
        "cuda-cudart": [],
        "cuda-cudart_linux-64": ["lib/libcudart.so.12"],
        "libstdcxx": ["lib/libstdc++.so.6"],
        "libgcc": ["lib/libgcc_s.so.1"],
    })
    ok, out = run_verify(good, tmp, **dict(common, dep_prefix=split_only))
    check(ok, "...and where the split package owns the file outright, the metapackage is still credited")

    # ---- win-64: PE linker version and import resolution ------------------
    def win_tree(root: Path, imports, linker=(14, 44)) -> Path:
        shutil.rmtree(root, ignore_errors=True)
        sp = root / "Lib" / "site-packages"
        (sp / "fx").mkdir(parents=True)
        (sp / "fx" / "__init__.py").write_text("from . import _C\n")
        build_pe(sp / "fx" / "_C.pyd", imports, linker)
        return root

    win_index = base_index(build="cuda128_torch28_py312_h6651153_0", subdir="win-64", platform="win")
    win_imports = ["KERNEL32.dll", "python312.dll", "c10.dll", "torch_cpu.dll", "cudart64_12.dll",
                   "VCRUNTIME140.dll", "api-ms-win-crt-runtime-l1-1-0.dll"]
    wcommon = dict(expect_arch="8.0 9.0", expect_msvc="vs2022", dep_prefix=win_prefix,
                   ledger={"D:\\a\\_temp\\out\\bld\\rattler-build_fx\\work\\x.cu"},
                   work_dir="D:/a/_temp/out/bld/rattler-build_fx/work")
    os.environ["CUW_CUOBJDUMP"] = str(td / "bin" / "cuobjdump-win")
    (td / "bin" / "cuobjdump-win").write_text(
        "#!/usr/bin/env python3\nprint('ELF file 1: x.sm_80.cubin\\nELF file 2: x.sm_90.cubin')\n")
    (td / "bin" / "cuobjdump-win").chmod(0o755)
    ok, out = run_verify(make_conda(td / "out-win", win_tree(td / "win", win_imports), win_index,
                                    base_about()), tmp, **wcommon)
    check(ok, "win-64 positive control: a vs2022-linked .pyd whose imports resolve PASSES")
    if not ok:
        print(out)
    ok, out = run_verify(make_conda(td / "out-win-ld", win_tree(td / "win-ld", win_imports, linker=(2, 42)),
                                    win_index, base_about()), tmp, **wcommon)
    check(not ok and fails_on(out, "linked by the conda MSVC"),
          "win-64: a PE with a non-MSVC linker version (MinGW ld 2.42) FAILS")
    ok, out = run_verify(make_conda(td / "out-win-vs", win_tree(td / "win-vs", win_imports, linker=(14, 29)),
                                    win_index, base_about()), tmp, **wcommon)
    check(not ok and fails_on(out, "linked by the conda MSVC"),
          "win-64: a VS2019 linker on a cell that expects vs2022 FAILS")
    ok, out = run_verify(make_conda(td / "out-win-imp", win_tree(td / "win-imp", win_imports + ["nvjpeg64_12.dll"]),
                                    win_index, base_about()), tmp, **wcommon)
    check(not ok and fails_on(out, "resolves through the artifact"),
          "win-64: an import no declared package provides FAILS")
    ok, out = run_verify(make_conda(td / "out-win-under", win_tree(td / "win-under", win_imports + ["libpng16.dll"]),
                                    win_index, base_about()), tmp, **wcommon)
    check(not ok and fails_on(out, "underlinked"),
          "win-64: an import provided only by an undeclared package FAILS as underlinking")
    os.environ["CUW_CUOBJDUMP"] = str(stub)

    # ---- noarch ------------------------------------------------------------
    def noarch_tree(root: Path, with_so=False) -> Path:
        shutil.rmtree(root, ignore_errors=True)
        sp = root / "site-packages"
        (sp / "pc").mkdir(parents=True)
        (sp / "pc" / "__init__.py").write_text("x = 1\n")
        if with_so:
            compile_so(sp / "pc" / "_C.so")
        return root

    na_index = {"name": "pc", "version": "0.1", "build": "pyh4616a5c_0", "build_number": 0,
                "subdir": "noarch", "noarch": "python", "depends": ["python >=3.10", "fire"]}
    ok, out = run_verify(make_conda(td / "out-na", noarch_tree(td / "na"), na_index, base_about()), tmp, noarch=True)
    check(ok, "noarch positive control PASSES the packaging subset")
    ok, out = run_verify(make_conda(td / "out-na2", noarch_tree(td / "na2", with_so=True), na_index, base_about()),
                         tmp, noarch=True)
    check(not ok and fails_on(out, "noarch package ships no compiled"),
          "noarch: a compiled module inside a noarch artifact FAILS")

    shutil.rmtree(td, ignore_errors=True)
    # Legacy -ng metapackages (gcc<=12 run_exports) own no files; their
    # payload packages do. Declaring `libgcc-ng` has declared the provider of
    # libgcc_s.so.1 -- run 34626966737 flagged cumm (gcc 8) as underlinked.
    check("libgcc-ng" in vc._credited("libgcc"), "libgcc's files are credited to a declared libgcc-ng")
    check("libstdcxx-ng" in vc._credited("libstdcxx"), "libstdcxx's files are credited to a declared libstdcxx-ng")
    check("libgcc-ng" not in vc._credited("libstdcxx"), "libstdcxx does not credit libgcc-ng")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

