#!/usr/bin/env python3
"""Publish gate for the wheel half, and the cross-check against its .conda.

tools/verify_conda.py gates the conda artifact. This is its sibling for the
wheel that came out of the SAME compile, and its most useful checks are the
ones that can only be made by looking at both: the two artifacts must agree
about the version they are, the dependencies they declare, and the GPU
architectures they carry. Nothing else in the pipeline compares them, and
they are produced by different code paths -- rattler-build packages the
prefix, auditwheel repairs the wheel -- so agreement is exactly the property
most likely to rot.

The wheel-only checks are the inverse of the conda ones, on purpose:

  * a conda package must NOT vendor shared libraries; a wheel MUST, and
    `auditwheel repair` is the conversion. So verify_conda asserts no
    vendored libs and this asserts they are present and correct.
  * a conda package declares dependencies in index.json; the root-tree
    wheel declares NONE (cuda-wheels' contract) and its /deps/ twin -- the
    same wheel, same filename, Requires-Dist inside, sidecar identical to
    its METADATA -- declares the curated list. So this asserts the root
    wheel is empty of Requires-Dist, the twin exists, agrees with its
    sidecar byte for byte, and differs from the root wheel in nothing but
    METADATA and RECORD.
  * neither may ever vendor libtorch, and neither may advertise a torch
    dependency: the ABI is pinned in the local version segment, which pip
    ignores for resolution.

Usage:
  verify_wheel.py <wheel> [...] [--conda <pkg.conda>] [--expect-arch "8.0 9.0"]
                  [--expect-version 0.23.0] [--package torchvision]
Exit: 0 all pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import base64
import csv
import email
import hashlib
import io
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tools"))

# Same list make_wheel.py refuses to advertise, for the same reason.
TORCH_NAMES = {"torch", "torchvision", "torchaudio", "triton",
               "pytorch-triton", "pytorch"}
# PEP 427: name-version(-build)?-python-abi-platform.whl. The build tag is
# OPTIONAL and, when present, is what distinguishes a rebuild of one cell from
# its predecessor -- make_wheel.py emits the conda build number there. Without
# the `(?:-(?P<build>\d[^-]*))?` group this regex still MATCHED a six-field
# name, silently reading the build tag as the python tag and the python tag as
# the abi: `fused_ssim-0.0.0+cu128torch2.8-1-cp312-cp312-win_amd64.whl` parsed
# as py="1", abi="cp312", plat="cp312-win_amd64". Every downstream check then
# asserted against nonsense.
WHEEL_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.]+)-(?P<ver>[^-]+)(?:-(?P<build>\d[^-]*))?"
    r"-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>.+)\.whl$")


class Report:
    def __init__(self, artifact):
        self.artifact, self.failed = artifact, False

    def ok(self, msg):
        print(f"ok   {msg}")

    def bad(self, msg):
        print(f"FAIL {msg}")
        self.failed = True

    def check(self, cond, msg):
        (self.ok if cond else self.bad)(msg)
        return cond


def conda_run_deps(path: Path) -> list[str]:
    """The `depends` list from a .conda's index.json."""
    from verify_conda import read_conda
    index = read_conda(path)[0]
    return list(index.get("depends") or [])


def sass_archs(data: bytes, tmp: Path, is_win: bool = False) -> set[str]:
    p = tmp / ("sass.pyd" if is_win else "sass.so")
    p.write_bytes(data)
    out = subprocess.run(["cuobjdump", "--list-elf", str(p)],
                         capture_output=True, text=True).stdout
    return set(re.findall(r"sm_(\d+)", out))


def verify(path: Path, args, tmp: Path) -> bool:
    rep = Report(path.name)
    print(f"=== {path.name}")

    m = WHEEL_RE.match(path.name)
    if not rep.check(bool(m), "filename parses as a wheel"):
        return False

    # ---- the name states the ABI it was compiled for ------------------------
    ver, abi, plat = m["ver"], m["abi"], m["plat"]
    z = zipfile.ZipFile(path)
    names = z.namelist()
    # A module named *.cpython-312-*.so (or *.cp312-win_amd64.pyd) was built
    # without Py_LIMITED_API and binds to that interpreter exactly; the wheel
    # tag must say so. torchao shipped cp39-abi3 with such a module inside
    # (make_wheel now retags; this is the gate that catches a wheel that
    # got past it).
    from make_wheel import cpython_abi_of_modules
    mods = [n for n in names if re.search(r"\.(so|pyd)$", n) and ".libs/" not in n]
    want_abi = cpython_abi_of_modules(mods)
    if want_abi:
        rep.check(abi == want_abi and m["py"] == want_abi,
                  f"wheel tag is {want_abi}-{want_abi}, as its CPython-specific module(s) "
                  f"demand (tag says {m['py']}-{abi})")
    else:
        rep.check(abi == "abi3" or abi == m["py"],
                  f"abi tag {abi} is consistent with python tag {m['py']}")
    rep.check("+cu" in ver and "torch" in ver,
              f"version carries the (cuda, torch) local tag: {ver}")
    is_win = args.platform.startswith("win")
    if is_win:
        # There is no repair step to have run, so there is no repaired tag to
        # assert. A Windows wheel vendors nothing (see below), which is why
        # win_amd64 here is correct rather than under-repaired.
        rep.check("win_amd64" in plat, f"platform tag is win_amd64: {plat}")
    else:
        # manylinux, not linux_x86_64: an unrepaired wheel installs and then
        # fails to find the libraries it was linked against.
        rep.check("manylinux" in plat,
                  f"platform tag is manylinux (auditwheel ran): {plat}")
        rep.check("manylinux_2_28" in plat,
                  f"platform tag includes PyTorch's 2.28 baseline: {plat}")

    di = [n for n in names if n.endswith(".dist-info/METADATA")]
    if not rep.check(len(di) == 1, "exactly one dist-info/METADATA"):
        return False
    meta = email.message_from_string(z.read(di[0]).decode("utf-8"))

    # ---- version agreement --------------------------------------------------
    rep.check(meta.get("Version") == ver,
              f"METADATA version matches the filename ({meta.get('Version')})")
    if args.expect_version:
        rep.check(ver.split("+")[0] == args.expect_version,
                  f"base version is the cell's ({ver.split('+')[0]} == "
                  f"{args.expect_version})")

    # ---- the ROOT wheel declares NOTHING -----------------------------------
    reqs = meta.get_all("Requires-Dist") or []
    rep.check(not reqs,
              f"root wheel declares no Requires-Dist ({len(reqs)} found) -- the "
              f"root index must not let a resolver chase dependencies")
    rep.check(not path.with_name(path.name + ".metadata").exists(),
              "no sidecar beside the root wheel (the root tree advertises none, and a "
              "sidecar that differs from the wheel's METADATA is what pip refuses)")

    # ---- the /deps/ twin: same wheel, dependencies INSIDE ------------------
    # Two files per artifact, same canonical filename, different storage
    # release. PEP 658 says a sidecar and the wheel's METADATA "MUST be
    # identical" and pip >= 26 enforces it (measured: every old-scheme wheel
    # whose sidecar declared a dependency was refused from /deps/), so the
    # deps tree's file carries the curated Requires-Dist in its own METADATA
    # and its sidecar is that METADATA byte for byte.
    deps_whl = args.deps_wheel or (path.parent / "deps" / path.name)
    deps_side = deps_whl.with_name(deps_whl.name + ".metadata")
    sreqs: list[str] = []
    if rep.check(deps_whl.is_file(), f"the /deps/ twin exists ({deps_whl})"):
        zd = zipfile.ZipFile(deps_whl)
        dnames = zd.namelist()
        ddi = [n for n in dnames if n.endswith(".dist-info/METADATA")]
        rep.check(len(ddi) == 1, "deps twin: exactly one dist-info/METADATA")
        dmeta_bytes = zd.read(ddi[0]) if ddi else b""
        dmeta = email.message_from_string(dmeta_bytes.decode("utf-8"))
        sreqs = dmeta.get_all("Requires-Dist") or []
        rep.check(dmeta.get("Version") == ver, "deps twin: METADATA version matches the filename")
        if rep.check(deps_side.is_file(), "deps twin: PEP 658 sidecar exists beside it"):
            rep.check(deps_side.read_bytes() == dmeta_bytes,
                      "deps twin: the sidecar is byte-identical to the wheel's METADATA "
                      "(PEP 658 'MUST be identical'; pip enforces it)")
        bad = [r for r in sreqs
               if re.split(r"[<>=!~\s;\[]", r.strip())[0].lower() in TORCH_NAMES]
        rep.check(not bad,
                  f"deps twin does not advertise torch ({bad}) -- the ABI is "
                  f"pinned in the local version, which pip ignores")
        # Everything but METADATA and RECORD is the same bytes in both files:
        # the twin is a repackaging of the root wheel, never a second build.
        skip = (".dist-info/METADATA", ".dist-info/RECORD")
        differ = sorted(n for n in set(names) | set(dnames)
                        if not n.endswith(skip)
                        and (n not in names or n not in dnames or z.read(n) != zd.read(n)))
        rep.check(not differ,
                  f"deps twin differs from the root wheel in METADATA/RECORD only ({differ[:3]})")
        # Its RECORD hashes its own payload.
        drec = [n for n in dnames if n.endswith(".dist-info/RECORD")]
        if rep.check(len(drec) == 1, "deps twin: exactly one dist-info/RECORD"):
            dbad = []
            for row in csv.reader(io.StringIO(zd.read(drec[0]).decode("utf-8"))):
                if not row or row[0] == drec[0]:
                    continue
                name, digest = row[0], row[1]
                if name not in dnames:
                    dbad.append(f"{name}: absent")
                    continue
                want = "sha256=" + base64.urlsafe_b64encode(
                    hashlib.sha256(zd.read(name)).digest()).rstrip(b"=").decode()
                if digest != want:
                    dbad.append(f"{name}: hash")
            rep.check(not dbad, f"deps twin: every RECORD hash matches ({dbad[:2]})")

        # ---- and they are the SAME dependencies as the conda package --------
        if args.conda:
            cdeps = conda_run_deps(Path(args.conda))
            # The conda names are translated through the SAME table
            # make_wheel.py used (py-opencv -> opencv-python, matplotlib-base
            # -> matplotlib), so the two sides are compared in one namespace.
            from make_wheel import pypi_name_for
            def norm(s):
                n = re.split(r"[<>=!~\s;\[]", s.strip())[0]
                p = pypi_name_for(n, args.cuda, strict=False) or n
                return p.lower().replace("_", "-")
            cnames = {norm(d) for d in cdeps}
            cnames -= {n.lower() for n in TORCH_NAMES}
            # conda's run: legitimately carries things a wheel cannot express
            # (virtual packages, the C/C++ runtimes, and every shared library
            # the wheel VENDORS instead of depending on), so the assertion is
            # one-directional: nothing the wheel claims may be absent from
            # the conda package.
            cnames = {c for c in cnames if not c.startswith("__")}
            snames = {norm(r) for r in sreqs}
            missing = sorted(snames - cnames)
            rep.check(not missing,
                      f"every deps-twin dependency is also a conda run dep "
                      f"(orphans: {missing})")

    # ---- RECORD integrity ---------------------------------------------------
    # make_wheel.py rewrites RECORD, because it renames the dist-info, edits
    # METADATA and rewrites RPATHs -- every one of which invalidates the
    # hashes upstream wrote. Nothing else checks the result, and a wrong
    # RECORD is quiet: the wheel installs, and it is `pip uninstall` and any
    # hash-checking install that break later. So the hashes are recomputed
    # here rather than trusted.
    rec_name = [n for n in names if n.endswith(".dist-info/RECORD")]
    if rep.check(len(rec_name) == 1, "exactly one dist-info/RECORD"):
        rows = list(csv.reader(io.StringIO(z.read(rec_name[0]).decode("utf-8"))))
        listed, bad = set(), []
        for row in rows:
            if not row:
                continue
            name, digest, size = (list(row) + ["", ""])[:3]
            listed.add(name)
            if name == rec_name[0]:
                # RECORD cannot hash itself; PEP 376 leaves both fields empty.
                if digest or size:
                    bad.append(f"{name}: RECORD must carry no hash or size")
                continue
            if name not in names:
                bad.append(f"{name}: listed in RECORD but absent from the wheel")
                continue
            data = z.read(name)
            want = "sha256=" + base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()).rstrip(b"=").decode()
            if digest != want:
                bad.append(f"{name}: hash mismatch")
            elif size and int(size) != len(data):
                bad.append(f"{name}: size mismatch")
        rep.check(not bad, f"every RECORD hash matches the payload ({bad[:2]})")
        unlisted = [n for n in names if n not in listed and not n.endswith("/")]
        rep.check(not unlisted,
                  f"every file in the wheel appears in RECORD ({unlisted[:2]})")

    # ---- vendoring: the inverse of the conda contract -----------------------
    if is_win:
        # On Windows the inverse is that there is nothing to invert: the wheel
        # bundles NO DLL at all. The .pyd links torch's DLLs and finds them
        # because `import torch` calls os.add_dll_directory on torch/lib before
        # any extension loads. Measured in cuda-wheels, whose verify_wheel says
        # so outright. So this asserts the absence rather than auditing the
        # contents -- a vendored DLL here would mean a repair step ran that
        # should not have, or that a build copied one in.
        dlls = sorted(Path(n).name for n in names if n.lower().endswith(".dll"))
        rep.check(not dlls,
                  f"no DLL is vendored ({dlls[:3]}) -- a Windows wheel resolves "
                  f"torch's DLLs through os.add_dll_directory, and bundling one "
                  f"would load a second copy of it")
        exts = [n for n in names if n.lower().endswith(".pyd")]
        rep.check(bool(exts), f"wheel contains extension module(s) ({len(exts)})")
        # The same linker-version gate verify_conda applies on win-64.
        from verify_conda import MSVC_LINKER, pe_header
        bad = []
        for n in exts:
            h = pe_header(z.read(n))
            if h is None:
                bad.append((Path(n).name, "not a PE"))
                continue
            maj, mino, _ = h
            lo, hi = MSVC_LINKER.get(getattr(args, "expect_msvc", "") or "", (20, 50))
            if maj != 14 or not (lo <= mino < hi):
                bad.append((Path(n).name, f"linker {maj}.{mino}"))
        rep.check(not bad, f"every .pyd was linked by the conda MSVC toolset ({bad[:3]})")

        # Torch linkage from the PE import table. PE stores imported DLL names
        # as plain ASCII, so a byte scan finds them without a PE parser; this is
        # evidence rather than proof, and cuda-wheels labels the same technique
        # "PE evidence" for that reason. The assertion is over the wheel as a
        # WHOLE for the same reason as the ELF branch below: a package may ship
        # helper modules that legitimately do not touch torch.
        if args.links_torch and exts:
            torch_linked = [Path(n).name for n in exts
                            if re.search(rb"(torch_cpu|torch_python|torch_cuda|c10)\.dll",
                                         z.read(n), re.I)]
            rep.check(bool(torch_linked),
                      f"at least one extension imports a torch DLL, by PE evidence "
                      f"({torch_linked[:3]})")
    else:
        _verify_elf_binaries(rep, z, names, args, tmp)

    # ---- the arch list, recorded and real -----------------------------------
    stated = (meta.get("Comfy-Forge-Arch-List") or "").strip()
    if args.expect_arch:
        rep.check(stated == args.expect_arch.strip(),
                  f"METADATA records the cell's arch list ({stated!r})")
    return _finish(rep, z, names, meta, args, tmp, is_win)


def _verify_elf_binaries(rep, z, names, args, tmp: Path) -> None:
    libs = [n for n in names if re.search(r"\.libs/lib.*\.so", n)]
    vendored = sorted({re.sub(r"-[0-9a-f]{8}(?=\.so)", "", Path(n).name) for n in libs})
    torch_vendored = [v for v in vendored
                      if v.startswith(("libtorch", "libc10", "libcaffe2"))]
    rep.check(not torch_vendored,
              f"no libtorch/libc10 vendored ({torch_vendored}) -- torch is the "
              f"consumer's, and vendoring it would load a second copy")
    rep.check(not [v for v in vendored if v.startswith("libcuda.so")],
              "the NVIDIA driver is not vendored")
    print(f"     vendored: {', '.join(vendored) if vendored else '(nothing)'}")

    # ---- the extension modules themselves -----------------------------------
    exts = [n for n in names if n.endswith(".so") and ".libs/" not in n]
    rep.check(bool(exts), f"wheel contains extension module(s) ({len(exts)})")

    from verify_conda import COMMENT_CONDA_FORGE, COMMENT_SYSROOT
    torch_linked = []
    for n in exts:
        data = z.read(n)
        p = tmp / "ext.so"
        p.write_bytes(data)
        # Compiler provenance, the same gate verify_conda applies to the
        # .conda: `.comment` must name conda-forge's gcc and nothing foreign.
        # auditwheel does not care who compiled a module; this does.
        com = subprocess.run(["readelf", "-p", ".comment", str(p)],
                             capture_output=True, text=True).stdout
        entries = [x.strip() for x in re.findall(r"^\s*\[\s*[0-9a-fx]+\]\s+(.*)$", com, re.M)]
        cf = [e for e in entries if COMMENT_CONDA_FORGE.match(e)]
        foreign = [e for e in entries
                   if not COMMENT_CONDA_FORGE.match(e) and not COMMENT_SYSROOT.match(e)]
        rep.check(bool(cf) and not foreign,
                  f"{Path(n).name}: compiled by conda-forge's gcc alone "
                  f"(.comment: {entries})")
        dyn = subprocess.run(["readelf", "-d", str(p)],
                             capture_output=True, text=True).stdout
        rpaths = [ln for ln in dyn.splitlines()
                  if "RPATH" in ln or "RUNPATH" in ln]
        absolute = [ln for ln in rpaths
                    if re.search(r"\[(?![^\]]*\$ORIGIN)[^\]]*/", ln)]
        rep.check(not absolute,
                  f"{Path(n).name}: RPATH is $ORIGIN-relative, no build-machine "
                  f"paths ({absolute[:1]})")
        if args.links_torch:
            needed = re.findall(r"NEEDED\).*?\[(.*?)\]", dyn)
            if any(x.startswith(("libtorch", "libc10")) for x in needed):
                torch_linked.append(Path(n).name)

    # Asserted over the wheel as a WHOLE, not per binary. A package that ships
    # helper libraries legitimately has some that do not touch torch:
    # torchaudio's libctc_prefix_decoder.so and pybind11_prefixctc.so are
    # self-contained CUDA and pybind code with ZERO undefined torch or c10
    # symbols, so linking libtorch into them would be over-linking (which
    # upstream does, and which is not a standard to hold ourselves to). What
    # must be true is that a package declaring links_torch actually links it
    # somewhere -- otherwise the flavour lock in its dependencies is decorating
    # a binary that never needed torch at all.
    if args.links_torch and exts:
        rep.check(bool(torch_linked),
                  f"at least one extension links torch ({torch_linked[:3]})")


def _finish(rep, z, names, meta, args, tmp: Path, is_win: bool) -> bool:
    """The SASS census and the verdict, shared by both platform branches.

    cuobjdump reads the fatbin sections of a PE the same way it reads an ELF's,
    so the census itself is platform-neutral; only the file extension it is
    handed differs, and that matters because cuobjdump dispatches on content,
    not name -- the suffix here is for legibility when the temp file is
    inspected after a failure.
    """
    exts = [n for n in names
            if n.lower().endswith(".pyd")] if is_win else [
            n for n in names if n.endswith(".so") and ".libs/" not in n]

    if args.expect_arch:
        want = {a.replace(".", "").replace("+PTX", "")
                for a in args.expect_arch.split()}
        # Union over every module, for the reason verify_conda.py gives at its
        # census: a package that splits kernels by architecture keeps the
        # cell's promise as an artifact and breaks it in any single module.
        if exts:
            got, per_module = set(), {}
            for n in exts:
                archs = sass_archs(z.read(n), tmp, is_win)
                if archs:
                    per_module[Path(n).name] = sorted(archs)
                got |= archs
            # Same declaration verify_conda honours: a package that says it
            # ships no device code (cumm: NVRTC at run time) is asserted to
            # ship none, instead of passing a census that found nothing.
            if getattr(args, "no_sass", None):
                rep.check(not got,
                          f"ships no SASS, as package.yml declares; found "
                          f"{sorted(got)} in {per_module}")
            else:
                rep.check(want <= got or not got,
                          f"SASS covers the cell's arch list "
                          f"(want {sorted(want)}, got {sorted(got)} over "
                          f"{len(exts)} module(s): {per_module})")

    print(f"--- {rep.artifact}: {'FAIL' if rep.failed else 'PASS'}")
    return not rep.failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wheels", nargs="+", type=Path)
    ap.add_argument("--conda", help="the .conda built from the same compile")
    ap.add_argument("--deps-wheel", type=Path, default=None,
                    help="the /deps/ twin (default: deps/<same name> beside the wheel)")
    ap.add_argument("--expect-arch", default="")
    ap.add_argument("--expect-version", default="")
    ap.add_argument("--package", default="",
                    help="packages/<folder>, to read links_torch")
    ap.add_argument("--platform", default="linux-64",
                    help="target platform of the wheel: linux-64 or win-64")
    ap.add_argument("--cuda", default="",
                    help="the cell's CUDA version, for the cu12/cu13 PyPI name suffix")
    ap.add_argument("--expect-msvc", default="",
                    help="win-64: vs2019 or vs2022, which the PE linker version must match")
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/verify-wheel"))
    args = ap.parse_args()
    args.tmp.mkdir(parents=True, exist_ok=True)

    args.links_torch = True
    args.no_sass = None
    if args.package:
        import package_loader as pl
        cfg = pl.load_package(pl.PACKAGES_DIR / args.package)
        args.links_torch = cfg.get("links_torch", True)
        args.no_sass = (cfg.get("verify") or {}).get("no_sass")

    allok = True
    for w in args.wheels:
        allok &= verify(w, args, args.tmp)
    print("\nALL PASS" if allok else "\nFAILURES PRESENT")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
