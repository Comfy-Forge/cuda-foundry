#!/usr/bin/env python3
"""Derive the Windows host-compiler window for each CUDA line, from nvcc's own headers.

The Linux side of this repo carries a gcc ceiling table read out of conda-forge
repodata (`cuda-nvcc` declares `gcc <N`). Windows has no such declaration: the
conda-forge CUDA packages constrain only `vc >=14.2,<15`, which spans every
MSVC from 19.2x to 19.5x and is therefore not the real limit.

The real limit is compiled into nvcc: `crt/host_config.h` carries

    #if _MSC_VER < 1910 || _MSC_VER >= 1950
    #error -- unsupported Microsoft Visual Studio version! ...

and nvcc hard-errors outside that window regardless of what the solver allowed.
So the table is read from the shipped header, the same way the gcc table is read
from repodata -- computed, not pasted, because the numbers move every CUDA minor
and a stale pasted row is indistinguishable from a correct one.

Why this matters here beyond documentation: `vs2022_win-64` is MSVC 19.44
(_MSC_VER 1944). That is inside the window for CUDA >= 12.4 and OUTSIDE it for
CUDA <= 12.2, whose ceiling is 1940. A cell on an older CUDA line must pin
`vs2019_win-64` instead. Picking one MSVC for the whole matrix is wrong.

Usage:  python tools/msvc_ceiling.py [--json]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import tarfile
import urllib.request
import zipfile

CF = "https://conda.anaconda.org/conda-forge/"
API = "https://api.anaconda.org/package/conda-forge/{}/files"

# host_config.h moved between packages across the 12.x series: up to 12.1 it
# shipped in cuda-nvcc-dev_win-64, from 12.2 it lives in cuda-crt-dev_win-64.
# Try both rather than encoding the boundary, which is itself a moving part.
HEADER_PKGS = ("cuda-crt-dev_win-64", "cuda-nvcc-dev_win-64")

# The activation packages conda-forge ships, and the _MSC_VER each provides.
# Version is the package version; _MSC_VER is major*100+minor of it.
VS_PKGS = ("vs2019_win-64", "vs2022_win-64")

_GUARD = re.compile(
    r"_MSC_VER\s*<\s*(?P<lo>\d+)\s*\|\|\s*_MSC_VER\s*>=\s*(?P<hi>\d+)"
)


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as fh:
        return fh.read()


def _open_conda(basename: str) -> tarfile.TarFile:
    """Return the payload tarball of a .conda (a zip of two zstd tarballs)."""
    z = zipfile.ZipFile(io.BytesIO(_get(CF + basename)))
    inner = next(n for n in z.namelist() if n.startswith("pkg-"))
    raw = subprocess.run(
        ["zstd", "-d", "-c"], input=z.read(inner), capture_output=True, check=True
    ).stdout
    return tarfile.open(fileobj=io.BytesIO(raw))


def _newest_per_minor(pkg: str) -> dict[str, str]:
    """{'12.8': 'noarch/cuda-crt-dev_win-64-12.8.93-h57928b3_3.conda', ...}"""
    files = json.loads(_get(API.format(pkg)))
    best: dict[str, dict] = {}
    for f in files:
        if f.get("attrs", {}).get("subdir") not in ("noarch", "win-64"):
            continue
        minor = ".".join(f["version"].split(".")[:2])
        key = (f["version"], f["attrs"].get("build_number", 0))
        if minor not in best or key > best[minor]["_key"]:
            best[minor] = {"_key": key, "basename": f["basename"]}
    return {k: v["basename"] for k, v in best.items()}


def msvc_window(basename: str) -> tuple[int, int] | None:
    """(min _MSC_VER inclusive, max _MSC_VER exclusive) from crt/host_config.h."""
    tf = _open_conda(basename)
    hdr = [m for m in tf.getmembers() if m.name.endswith("crt/host_config.h")]
    if not hdr:
        return None
    text = tf.extractfile(hdr[0]).read().decode("utf8", "replace")
    # The file guards both host compilers; take the MSVC guard specifically.
    m = _GUARD.search(text)
    return (int(m.group("lo")), int(m.group("hi"))) if m else None


def vs_msc_versions() -> dict[str, int]:
    out = {}
    for pkg in VS_PKGS:
        files = json.loads(_get(API.format(pkg)))
        vers = sorted(
            {f["version"] for f in files if f.get("attrs", {}).get("subdir") == "win-64"},
            key=lambda v: [int(x) for x in v.split(".")],
        )
        if vers:
            major, minor, *_ = vers[-1].split(".")
            out[pkg] = int(major) * 100 + int(minor)
    return out


def build_table() -> dict:
    windows: dict[str, tuple[int, int]] = {}
    for pkg in HEADER_PKGS:
        for minor, basename in _newest_per_minor(pkg).items():
            if minor in windows:
                continue
            win = msvc_window(basename)
            if win:
                windows[minor] = win

    vs = vs_msc_versions()
    rows = []
    for minor in sorted(windows, key=lambda s: [int(x) for x in s.split(".")]):
        lo, hi = windows[minor]
        fits = [p for p, v in sorted(vs.items()) if lo <= v < hi]
        rows.append(
            {
                "cuda": minor,
                "msc_min": lo,
                "msc_max_exclusive": hi,
                "usable": fits,
                # The one we want is the newest that fits: it is the toolset
                # upstream torch itself was built with wherever that is legal.
                "pick": fits[-1] if fits else None,
            }
        )
    return {"vs_packages": vs, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    table = build_table()
    if args.json:
        print(json.dumps(table, indent=2))
        return 0

    vs = table["vs_packages"]
    print("conda-forge MSVC activation packages:")
    for pkg, msc in sorted(vs.items()):
        print(f"  {pkg:<16} _MSC_VER {msc}")
    print()
    print(f"{'CUDA':<6} {'_MSC_VER window':<20} {'pick':<16} usable")
    print("-" * 64)
    bad = 0
    for r in table["rows"]:
        window = f"[{r['msc_min']}, {r['msc_max_exclusive']})"
        pick = r["pick"] or "NONE"
        if r["pick"] is None:
            bad += 1
        print(f"{r['cuda']:<6} {window:<20} {pick:<16} {','.join(r['usable']) or '-'}")
    if bad:
        print(f"\n{bad} CUDA line(s) have no usable conda-forge MSVC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
