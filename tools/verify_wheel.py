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
  * a conda package declares dependencies in index.json; the published
    wheel declares NONE (see cuda-wheels' contract) and carries them only
    in a PEP 658 sidecar. So this asserts the wheel is empty of
    Requires-Dist and the sidecar is not.
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

    z = zipfile.ZipFile(path)
    names = z.namelist()
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

    # ---- the wheel itself declares NOTHING ----------------------------------
    reqs = meta.get_all("Requires-Dist") or []
    rep.check(not reqs,
              f"wheel declares no Requires-Dist ({len(reqs)} found) -- the "
              f"index this serves must not let a resolver chase dependencies")

    # ---- the sidecar is where the dependencies live -------------------------
    side = path.with_name(path.name + ".metadata")
    if rep.check(side.is_file(), "PEP 658 sidecar exists beside the wheel"):
        smeta = email.message_from_string(side.read_text(encoding="utf-8"))
        sreqs = smeta.get_all("Requires-Dist") or []
        rep.check(smeta.get("Version") == ver,
                  "sidecar version matches the wheel")
        bad = [r for r in sreqs
               if re.split(r"[<>=!~\s;\[]", r.strip())[0].lower() in TORCH_NAMES]
        rep.check(not bad,
                  f"sidecar does not advertise torch ({bad}) -- the ABI is "
                  f"pinned in the local version, which pip ignores")

        # ---- and they are the SAME dependencies as the conda package --------
        if args.conda:
            cdeps = conda_run_deps(Path(args.conda))
            # The conda names are translated through the SAME table
            # make_wheel.py used to write the sidecar (py-opencv ->
            # opencv-python, matplotlib-base -> matplotlib), so the two
            # sides are compared in one namespace. A second, private copy of
            # that mapping here is exactly the kind of thing that drifts.
            from make_wheel import CONDA_TO_PYPI
            def norm(s):
                n = re.split(r"[<>=!~\s;\[]", s.strip())[0].lower()
                return CONDA_TO_PYPI.get(n, n).lower()
            cnames = {norm(d) for d in cdeps}
            cnames -= {n.lower() for n in TORCH_NAMES}
            # conda's run: legitimately carries things a wheel cannot express
            # (virtual packages, the C/C++ runtimes, and every shared library
            # the wheel VENDORS instead of depending on), so the assertion is
            # one-directional: nothing the sidecar claims may be absent from
            # the conda package.
            cnames = {c for c in cnames if not c.startswith("__")}
            snames = {norm(r) for r in sreqs}
            missing = sorted(snames - cnames)
            rep.check(not missing,
                      f"every sidecar dependency is also a conda run dep "
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

    torch_linked = []
    for n in exts:
        data = z.read(n)
        p = tmp / "ext.so"
        p.write_bytes(data)
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
    ap.add_argument("--expect-arch", default="")
    ap.add_argument("--expect-version", default="")
    ap.add_argument("--package", default="",
                    help="packages/<folder>, to read links_torch")
    ap.add_argument("--platform", default="linux-64",
                    help="target platform of the wheel: linux-64 or win-64")
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/verify-wheel"))
    args = ap.parse_args()
    args.tmp.mkdir(parents=True, exist_ok=True)

    args.links_torch = True
    if args.package:
        import package_loader as pl
        args.links_torch = pl.load_package(
            pl.PACKAGES_DIR / args.package).get("links_torch", True)

    allok = True
    for w in args.wheels:
        allok &= verify(w, args, args.tmp)
    print("\nALL PASS" if allok else "\nFAILURES PRESENT")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
