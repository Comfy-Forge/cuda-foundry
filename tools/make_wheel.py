#!/usr/bin/env python3
"""Turn the build's ONE wheel into the published manylinux wheel + sidecar.

The build compiles exactly once (scripts/build_snippets/build.sh runs
`pip wheel .`) and leaves the result in $CUW_WHEELHOUSE. That same file
becomes both outputs: the .conda is it installed into $PREFIX and packaged
by rattler-build, and the published wheel is what this script makes of it.
Nothing is compiled twice, so the two artifacts cannot disagree about what
they contain.

The conversion is real work, not a rename, because the two package formats
have OPPOSITE contracts about shared libraries. A conda package must not
vendor them -- it declares `libjpeg-turbo` and links libjpeg.so.8 out of the
prefix, so `conda update` can patch it. A manylinux wheel must vendor them,
because a wheel has no way to declare a non-Python dependency at all.
`auditwheel repair` is exactly that conversion: it copies each external
library into <pkg>.libs/, gives it a hash-suffixed name and rewrites the
extension's RPATH to $ORIGIN.

Three outputs land in --out-dir:
  <pkg>-<ver>+cu<NNN>torch<M.m>-<abi>-<manylinux>.whl   deps stripped
  <same>.whl.metadata                                   PEP 658 sidecar
  and a printed summary of what got vendored.

Usage:
  make_wheel.py --package torchvision --wheel <raw.whl> --out-dir dist \\
      --cuda 12.8 --pytorch 2.8 --arch-list "7.5 8.0" --lib-path <host_env>/lib
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# PyTorch's own manylinux baseline. Their wheels are manylinux_2_28 (they
# build in AlmaLinux 8 containers), and our .conda artifacts compute a
# `__glibc >=2.28` floor from the conda sysroot, so the two agree. This is
# POLICY, not the build machine's glibc: auditwheel verifies the binary
# against it either way and fails if the binary actually needs more.
PLAT = {"x86_64": "manylinux_2_28_x86_64", "aarch64": "manylinux_2_28_aarch64"}

# Libraries auditwheel must NOT vendor, ported from cuda-wheels (CW-ADR-0009)
# together with the reasoning, which is the part that matters:
#
# An excluded library gets NO rpath, so it must already be resident in the
# process when the extension is dlopen'd. That is true for exactly one class
# of library here: the ones `import torch` itself loads eagerly. Since a
# consumer of these wheels has always imported torch first (the local version
# tag says which torch), DT_NEEDED resolves against the already-loaded SONAME.
#
# Do NOT add excludes "for safety". A missing exclude costs megabytes; a wrong
# one costs a broken import. The census below catches anything unexpected.
TORCH_EXCLUDES = [
    "libtorch.so", "libtorch_cpu.so", "libtorch_cuda.so",
    "libtorch_cuda_linalg.so", "libtorch_python.so",
    "libcaffe2_nvrtc.so", "libc10.so", "libc10_cuda.so",
    # NVIDIA versions its libraries and PyTorch does not, which is why these
    # need a glob and the torch ones above do not -- auditwheel fnmatches
    # against the SONAME, not the filename.
    "libcudart.so*", "libcublas.so*", "libcublasLt.so*",
    # torch ships torch/lib/libgomp.so.1 and loads it at import. It is in no
    # manylinux whitelist, so this exclude is right ONLY for torch-linked
    # packages -- a torch-free one has no preloader and must vendor it.
    "libgomp.so.1",
]
# The driver is never vendored by anyone: it belongs to the host's NVIDIA
# installation and a wheel that carried one would pin the user's driver.
ALWAYS_EXCLUDE = ["libcuda.so", "libcuda.so.1"]

# Vendoring these is expected and fine; anything else vendored is reported
# loudly, because the usual cause is a package that started linking a CUDA
# math library, where the transitive closure runs to hundreds of megabytes.
EXPECTED_VENDORED = {
    "libnvrtc", "libnvrtc-builtins",          # runtime JIT, deliberately kept
    "libjpeg", "libpng16", "libpng", "libwebp", "libwebpmux", "libwebpdemux",
    "libsharpyuv", "libz", "libzlib", "libnvjpeg",   # torchvision's codecs
    "libsox", "libavutil", "libavcodec", "libavformat", "libavdevice",
    "libavfilter", "libswscale", "libswresample",    # torchaudio's backends
}

# conda spec -> PyPI requirement. The names agree for everything these
# packages declare, but they do NOT agree in general (conda's `pillow` is
# PyPI's `pillow`, but conda has `pytorch` for PyPI's `torch`), so the
# translation is explicit and anything unknown is passed through unchanged
# rather than silently guessed at.
CONDA_TO_PYPI = {"pytorch": "torch", "pillow": "pillow"}

# Never advertised as a wheel dependency, on either index. These wheels are
# compiled against ONE exact (cuda, torch) ABI, and that fact lives only in
# the local version segment -- which pip ignores for resolution. A
# `Requires-Dist: torch` would therefore let a resolver install or swap a
# torch that does not match, producing an import that succeeds and then dies
# on a missing C++ symbol. Installing the right CUDA torch is the consumer's
# job, done before these wheels are touched.
TORCH_NAMES = {"torch", "torchvision", "torchaudio", "triton",
               "pytorch-triton", "pytorch"}


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)


def local_tag(cuda: str, pytorch: str) -> str:
    """cuda 12.8 + torch 2.8 -> '+cu128torch2.8' (the cuda-wheels convention)."""
    return f"+cu{cuda.replace('.', '')}torch{pytorch}"


def conda_spec_to_pep508(spec: str) -> str | None:
    """'pillow >=5.3.0' -> 'pillow>=5.3.0'. None if it must not be advertised.

    Conda separates name and constraint with whitespace; PEP 508 does not
    allow that separation to be meaningful, so it is simply removed.
    """
    parts = spec.strip().split(None, 1)
    name = parts[0]
    if name.startswith("__"):          # virtual package (__cuda, __glibc)
        return None
    pypi = CONDA_TO_PYPI.get(name, name)
    if pypi.lower() in TORCH_NAMES:
        return None
    if len(parts) == 1:
        return pypi
    constraint = parts[1].strip()
    if constraint in ("", "*"):
        return pypi
    return f"{pypi}{constraint.replace(' ', '')}"


def repair(wheel: Path, out_dir: Path, links_torch: bool,
           lib_paths: list[str]) -> Path:
    """auditwheel repair -> a manylinux wheel with its libraries vendored."""
    machine = os.uname().machine
    plat = PLAT.get(machine)
    if not plat:
        sys.exit(f"make_wheel: no manylinux policy mapped for {machine}")

    excludes = list(ALWAYS_EXCLUDE)
    if links_torch:
        excludes += TORCH_EXCLUDES
    else:
        # No preloader: a torch-free package must carry libcudart/libgomp
        # itself or it cannot dlopen on a box with no CUDA toolkit.
        print("  torch-free package: vendoring libcudart/libgomp rather than "
              "excluding them (nothing preloads them)")

    env = dict(os.environ)
    if lib_paths:
        # auditwheel resolves DT_NEEDED with the normal loader search, and the
        # libraries it must vendor live in the conda HOST prefix, which is not
        # on any default path once the build is over. Without this, repair
        # fails with "cannot be located" for exactly the codecs that compiling
        # against conda packages was the point of.
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            lib_paths + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))

    cmd = ["auditwheel", "repair", str(wheel), "--plat", plat,
           "-w", str(out_dir), "--strip"]
    for e in excludes:
        cmd += ["--exclude", e]
    print(f"  $ auditwheel repair --plat {plat} --strip "
          f"({len(excludes)} excludes)")
    try:
        proc = run(cmd, env=env)
    except subprocess.CalledProcessError as e:
        print(e.stdout or "", file=sys.stderr)
        sys.exit(f"make_wheel: auditwheel repair failed for {wheel.name}")
    for line in (proc.stdout or "").splitlines():
        if "Grafting" in line or "Setting RPATH" in line or "previous" in line:
            print(f"    {line.strip()}")

    made = sorted(out_dir.glob("*.whl"))
    if len(made) != 1:
        sys.exit(f"make_wheel: expected 1 repaired wheel, found {len(made)}")
    return made[0]


def census(wheel: Path) -> list[str]:
    """Names of the libraries auditwheel vendored into <pkg>.libs/."""
    names = set()
    with zipfile.ZipFile(wheel) as z:
        for n in z.namelist():
            m = re.search(r"[^/]+\.libs/(lib[^/]+?)(?:-[0-9a-f]{8})?\.so", n)
            if m:
                names.add(m.group(1))
    return sorted(names)


def _hash(data: bytes) -> tuple[str, int]:
    d = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(d).rstrip(b"=").decode(), len(data)


def rebuild_record(root: Path, dist_info: str) -> None:
    rec_rel = f"{dist_info}/RECORD"
    rows = []
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(root).as_posix()
        if rel == rec_rel:
            continue
        h, n = _hash(f.read_bytes())
        rows.append((rel, h, str(n)))
    rows.append((rec_rel, "", ""))
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    (root / rec_rel).write_text(buf.getvalue(), encoding="utf-8")


def strip_requires(text: str) -> tuple[str, int]:
    """Drop every Requires-Dist/Provides-Extra, continuation lines included."""
    head, sep, body = text.partition("\n\n")
    kept, removed, dropping = [], 0, False
    for line in head.splitlines():
        if line.startswith(("Requires-Dist:", "Provides-Extra:")):
            removed += 1
            dropping = True
            continue
        if dropping and line[:1] in (" ", "\t"):
            continue
        dropping = False
        kept.append(line)
    return "\n".join(kept) + (sep + body if sep else "\n"), removed


def set_header(text: str, name: str, value: str) -> str:
    """Insert/replace a header immediately after Metadata-Version.

    Not "at the first blank-looking line": RFC 822 folding means a multi-line
    License: body contains whitespace-only lines, and inserting there lands
    the header INSIDE the licence and reassigns every continuation line under
    it. cuda-wheels corrupted 7 of 41 published packages that way.
    """
    text = re.sub(rf"^{re.escape(name)}:.*(?:\n[ \t].*)*\n", "", text,
                  flags=re.MULTILINE)
    lines = text.split("\n")
    at = 0
    for i, ln in enumerate(lines):
        if ln.lower().startswith("metadata-version:"):
            at = i + 1
            break
    lines.insert(at, f"{name}: {value}")
    return "\n".join(lines)


def finalize(wheel: Path, version_tag: str, arch_list: str,
             run_deps: list[str], expect_version: str = "") -> tuple[Path, Path, int]:
    """Apply the local version, strip deps, write the PEP 658 sidecar."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with zipfile.ZipFile(wheel) as z:
            z.extractall(root)
        di = next(iter(root.glob("*.dist-info")), None)
        if di is None:
            sys.exit(f"make_wheel: no .dist-info in {wheel.name}")
        meta_p = di / "METADATA"
        text = meta_p.read_text(encoding="utf-8")

        m = re.search(r"^Version: (.+)$", text, re.MULTILINE)
        if not m:
            sys.exit(f"make_wheel: {wheel.name} METADATA has no Version")
        base = m.group(1).split("+")[0]
        # The wheel and the .conda around it must claim the same version, and
        # nothing else checks that -- the conda side takes its version from
        # package.yml while the wheel takes whatever setup.py computed. They
        # disagreed for three of the first five packages built here: cc-torch
        # (package.yml 0.1.0 vs setup.py 0.2), fused-ssim (0.0.1 vs an
        # undeclared version that setuptools renders 0.0.0) and torchvision
        # (0.23.0 vs a git-less "0.23.0a0", which sorts BELOW the release).
        # Two were invented numbers and one was a missing BUILD_VERSION; all
        # three are the kind of thing that ships quietly and confuses a
        # resolver much later, so it is a hard error rather than a warning.
        if expect_version and base != expect_version:
            sys.exit(
                f"make_wheel: {wheel.name} builds version {base!r} but the "
                f"cell says {expect_version!r}. The .conda and the wheel would "
                f"claim different versions of the same build. Fix whichever is "
                f"wrong -- package.yml's `version:` if it was invented, or the "
                f"upstream version switch (BUILD_VERSION) if the build is not "
                f"being told which release it is.")
        full = base + version_tag
        text = re.sub(r"^Version: .+$", f"Version: {full}", text,
                      flags=re.MULTILINE, count=1)

        # Record the arch list IN the artifact. A wheel cannot say this in its
        # filename (PEP 427 fixes the fields), and an arch-list change does not
        # change any wheel's name -- so without this, a stale wheel built for
        # the wrong architectures is undetectable after the fact.
        if arch_list:
            text = set_header(text, "Comfy-Forge-Arch-List", arch_list)
            import email
            got = (email.message_from_string(text).get("Comfy-Forge-Arch-List") or "").strip()
            if got != arch_list.strip():
                sys.exit("make_wheel: the arch-list header does not parse back "
                         f"cleanly (wrote {arch_list!r}, read {got[:80]!r})")

        # The sidecar is the dependency-bearing view, captured BEFORE the
        # strip. Its deps come from package.yml's `run_deps` -- the same list
        # the conda package's `run:` is built from -- so the two formats
        # cannot drift into declaring different dependencies.
        sidecar_text = text
        for spec in run_deps:
            pep = conda_spec_to_pep508(spec)
            if pep:
                sidecar_text = sidecar_text.rstrip("\n").split("\n\n")[0] + \
                    f"\nRequires-Dist: {pep}\n" + \
                    ("\n" + sidecar_text.partition("\n\n")[2]
                     if "\n\n" in sidecar_text else "")

        text, removed = strip_requires(text)
        meta_p.write_text(text, encoding="utf-8")

        new_di = root / f"{di.name.split('-')[0]}-{full}.dist-info"
        if new_di != di:
            di.rename(new_di)
        rebuild_record(root, new_di.name)

        stem = wheel.name.split("-")
        stem[1] = full
        out = wheel.with_name("-".join(stem))
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(root))
        if out != wheel:
            wheel.unlink()
        side = out.with_name(out.name + ".metadata")
        side.write_text(sidecar_text, encoding="utf-8")
        n_dist = sum(1 for l in sidecar_text.splitlines()
                     if l.startswith("Requires-Dist:"))
        return out, side, n_dist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="packages/<folder>")
    ap.add_argument("--wheel", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cuda", required=True)
    ap.add_argument("--pytorch", required=True)
    ap.add_argument("--arch-list", default="")
    ap.add_argument("--expect-version", default="",
                    help="the cell's version; the wheel's own base version "
                         "must equal it or this refuses to publish")
    ap.add_argument("--lib-path", action="append", default=[],
                    help="prepended to LD_LIBRARY_PATH so auditwheel can find "
                         "the conda host prefix's libraries (repeatable)")
    args = ap.parse_args()

    import package_loader as pl
    cfg = pl.load_package(pl.PACKAGES_DIR / args.package)
    links_torch = cfg.get("links_torch", True)
    run_deps = cfg.get("run_deps") or []

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"== {args.package}: {args.wheel.name}")
    repaired = repair(args.wheel, args.out_dir, links_torch, args.lib_path)

    vendored = census(repaired)
    unexpected = [v for v in vendored
                  if re.sub(r"\.so.*$", "", v) not in EXPECTED_VENDORED
                  and v not in EXPECTED_VENDORED]
    print(f"  vendored: {', '.join(vendored) if vendored else '(nothing)'}")
    if unexpected:
        print(f"  ::warning:: unexpected vendored librar(ies): {unexpected}. "
              "Either add them to EXPECTED_VENDORED with a reason, or find "
              "out what started linking them.")

    final, side, n = finalize(repaired, local_tag(args.cuda, args.pytorch),
                              args.arch_list, run_deps, args.expect_version)
    print(f"  wheel   : {final.name}")
    print(f"  sidecar : {side.name}  ({n} Requires-Dist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
