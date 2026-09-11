#!/usr/bin/env python3
"""The two index trees, from a local layout of <tag>/ and <tag>-deps/ assets.

Root anchors must point at the stripped file in <subdir> with no PEP 658
attributes; /deps/ anchors at the twin in <subdir>-deps under the SAME
filename, advertising the sidecar's sha256; a wheel with no twin yet is
listed in /deps/ against its root file with nothing advertised (never the
old mismatching sidecar); a known-bad filename is yanked in both trees.

Run: python tools/test_generate_index.py
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
failures: list[str] = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def wheel(path: Path, name: str, version: str, reqs=()) -> None:
    di = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{name}/__init__.py", "")
        z.writestr(f"{di}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                                     + "".join(f"Requires-Dist: {r}\n" for r in reqs))
        z.writestr(f"{di}/WHEEL", "Wheel-Version: 1.0\nTag: cp312-cp312-manylinux_2_28_x86_64\n")
        z.writestr(f"{di}/RECORD", "")


def main() -> int:
    td = Path(tempfile.mkdtemp(prefix="cuw-index-test-"))
    assets = td / "assets"
    W = "fx-1.0+cu128torch2.8-0-cp312-cp312-manylinux_2_28_x86_64.whl"
    L = "legacy-2.0+cu128torch2.8-0-cp312-cp312-manylinux_2_28_x86_64.whl"
    B = "bad-3.0+cu128torch2.8-0-cp312-cp312-manylinux_2_28_x86_64.whl"
    (assets / "linux-64").mkdir(parents=True)
    (assets / "linux-64-deps").mkdir()
    wheel(assets / "linux-64" / W, "fx", "1.0+cu128torch2.8")
    wheel(assets / "linux-64-deps" / W, "fx", "1.0+cu128torch2.8", ["numpy"])
    with zipfile.ZipFile(assets / "linux-64-deps" / W) as z:
        (assets / "linux-64-deps" / (W + ".metadata")).write_bytes(z.read("fx-1.0+cu128torch2.8.dist-info/METADATA"))
    wheel(assets / "linux-64" / L, "legacy", "2.0+cu128torch2.8")
    # the OLD scheme's sidecar beside the root file: must never be advertised
    (assets / "linux-64" / (L + ".metadata")).write_text("Metadata-Version: 2.1\nName: legacy\nRequires-Dist: numpy\n")
    wheel(assets / "linux-64" / B, "bad", "3.0+cu128torch2.8")
    wheel(assets / "linux-64-deps" / B, "bad", "3.0+cu128torch2.8", ["numpy"])
    (assets / "linux-64-deps" / (B + ".metadata")).write_bytes(b"Metadata-Version: 2.1\nName: bad\nRequires-Dist: numpy\n")
    kb = td / "known_bad.json"
    kb.write_text(json.dumps({"linux-64": {B: {"reason": "broken kernel", "fixed_by": "x"}}}))

    site = td / "site"
    p = subprocess.run([sys.executable, str(HERE / "generate_index.py"), "--local-assets", str(assets),
                        "--out", str(site), "--baseline", str(td / "none.json"), "--known-bad", str(kb)],
                       capture_output=True, text=True)
    check(p.returncode == 0, f"generate_index --local-assets runs ({p.stderr[-300:]})")

    root_fx = (site / "fx" / "index.html").read_text()
    deps_fx = (site / "deps" / "fx" / "index.html").read_text()
    side_sha = hashlib.sha256((assets / "linux-64-deps" / (W + ".metadata")).read_bytes()).hexdigest()
    twin_sha = hashlib.sha256((assets / "linux-64-deps" / W).read_bytes()).hexdigest()
    root_sha = hashlib.sha256((assets / "linux-64" / W).read_bytes()).hexdigest()
    check(f"/linux-64/{W.replace('+', '%2B')}#sha256={root_sha}" in root_fx and "data-core-metadata" not in root_fx,
          "root tree links the stripped file in <subdir> and advertises no sidecar")
    check(f"/linux-64-deps/{W.replace('+', '%2B')}#sha256={twin_sha}" in deps_fx
          and f'data-core-metadata="sha256={side_sha}"' in deps_fx
          and f'data-dist-info-metadata="sha256={side_sha}"' in deps_fx,
          "/deps/ tree links the twin in <subdir>-deps with the sidecar's sha256")
    check(re.search(rf">{re.escape(W)}</a>", deps_fx) is not None,
          "/deps/ anchor text is the canonical filename")

    deps_legacy = (site / "deps" / "legacy" / "index.html").read_text()
    check("/linux-64/" in deps_legacy and "-deps/" not in deps_legacy and "data-core-metadata" not in deps_legacy,
          "a wheel with no twin is listed in /deps/ against its root file with NOTHING advertised "
          "(the old mismatching sidecar is never offered)")

    root_bad = (site / "bad" / "index.html").read_text()
    deps_bad = (site / "deps" / "bad" / "index.html").read_text()
    check('data-yanked="broken kernel"' in root_bad and 'data-yanked="broken kernel"' in deps_bad,
          "a known-bad filename is yanked in BOTH trees")

    manifest = json.loads((site / "packages.json").read_text())
    fx = manifest["packages"]["fx"]["wheels"][0]
    check(fx["deps_url"].endswith(f"linux-64-deps/{W.replace('+', '%2B')}") and fx["sidecar_sha256"] == side_sha
          and fx["has_sidecar"] is True,
          "packages.json carries deps_url / deps_sha256 / sidecar_sha256")
    check(manifest["packages"]["legacy"]["wheels"][0]["has_sidecar"] is False,
          "packages.json says the legacy wheel has no sidecar")

    shutil.rmtree(td, ignore_errors=True)
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
