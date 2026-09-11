#!/usr/bin/env python3
"""Emit a repodata fragment for one .conda file.

The channel's repodata.json is assembled from per-package JSON fragments
committed under meta/<subdir>/<filename>.json, so publishing never has to
re-download release assets. This tool produces one fragment: the package's
info/index.json plus the sha256/md5/size of the artifact itself.

Usage: fragment.py <pkg.conda> <subdir> [--meta-dir meta]

Requires the `zstd` CLI (the .conda info tarball is zstd-compressed and
python 3.12 has no stdlib zstd).
"""

from __future__ import annotations  # runs under the target python; may be 3.8

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def read_index_json(conda_path: Path) -> tuple[dict, dict, dict]:
    """(index.json, about.json-or-{}, run_exports.json-or-{}) from the artifact."""
    with zipfile.ZipFile(conda_path) as zf:
        info_names = [n for n in zf.namelist() if n.startswith("info-") and n.endswith(".tar.zst")]
        if len(info_names) != 1:
            sys.exit(f"expected exactly one info-*.tar.zst in {conda_path.name}, found {info_names}")
        zst = zf.read(info_names[0])
    tar = subprocess.run(["zstd", "-d", "--stdout"], input=zst, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        member = tf.extractfile("info/index.json")
        if member is None:
            sys.exit(f"{conda_path.name}: info/index.json missing")
        index = json.load(member)

        def optional(name: str) -> dict:
            try:
                m = tf.extractfile(name)
                return json.load(m) if m is not None else {}
            except KeyError:
                return {}

        # run_exports travels in the fragment so the channel can publish a
        # run_exports.json index: a builder resolving host deps reads the
        # channel index, never the artifacts.
        return index, optional("info/about.json"), optional("info/run_exports.json")


# A purl is a claim: pixi's conda->pypi map reads `pkg:pypi/<project>` and
# treats this artifact as that PyPI project, which is what stops a pack that
# declares the same name as a pypi dependency getting a SECOND copy installed
# from PyPI on top of ours -- the failure conda-torch proved with torch. So
# the purl must name a project this artifact GENUINELY provides. It used to be
# derived from pypi_name for every package, and about 20 of those were false:
# 404 on PyPI, or a different project sharing the name (`nunchaku` on PyPI is
# a data-segmentation library, `drtk` a squatted junk package, `cumesh`
# someone else's, `mmcv` the ops-less distribution). Now it comes only from an
# explicit `pypi_project:` in package.yml -- null, the default, means no purl
# -- which the recipe also records in about.extra.pypi_project, so the
# artifact is the authority on itself where one exists (a hand-written recipe
# such as recipes/pccm states it there directly).
def package_cfg_for(name: str) -> dict | None:
    """packages/<folder>/package.yml whose conda name is `name`, or None."""
    import yaml
    pkgs = Path(__file__).resolve().parent.parent / "packages"
    for d in sorted(pkgs.iterdir()) if pkgs.is_dir() else []:
        y = d / "package.yml"
        if not y.is_file():
            continue
        cfg = yaml.safe_load(y.read_text()) or {}
        conda = (cfg.get("conda_name") or cfg.get("pypi_name") or cfg.get("name") or "").replace("_", "-")
        if cfg.get("name") == name or conda == name:
            return cfg
    return None


def pypi_project_for(name: str) -> str | None:
    """package.yml `pypi_project` for a conda name, or None (no purl)."""
    cfg = package_cfg_for(name)
    proj = str((cfg or {}).get("pypi_project") or "").strip()
    return proj or None


def purl_for(name: str, version: str, about: dict | None = None) -> str | None:
    extra = (about or {}).get("extra") or {}
    # The artifact's own statement first (recipe about.extra.pypi_project, or
    # the hand-written pccm's about.extra.pypi_name), then package.yml.
    pypi = str(extra.get("pypi_project") or extra.get("pypi_name") or "").strip()
    if not pypi:
        pypi = pypi_project_for(name) or ""
    return f"pkg:pypi/{pypi}@{version}" if pypi else None


def hashes(path: Path) -> tuple[str, str, int]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("conda_file", type=Path)
    ap.add_argument("subdir")
    ap.add_argument("--meta-dir", type=Path, default=Path("meta"))
    args = ap.parse_args()

    index, about, run_exports = read_index_json(args.conda_file)
    if index.get("subdir", args.subdir) != args.subdir:
        sys.exit(f"index.json says subdir={index['subdir']!r} but you passed {args.subdir!r}")
    if "+" in str(index.get("version", "")):
        sys.exit(f"refusing local-version '+' in version {index['version']!r}: "
                 "conda orders 2.8.0+cu128 BELOW 2.8.0; encode flavour in the build string")

    sha256, md5, size = hashes(args.conda_file)
    entry = dict(index)
    entry.update({"sha256": sha256, "md5": md5, "size": size, "subdir": args.subdir})
    purl = purl_for(index["name"], index["version"], about)
    if purl:
        entry["purls"] = [purl]
    if run_exports:
        entry["run_exports"] = run_exports
    # provenance records the from-source guarantee in the served metadata,
    # so a third party can audit it without unpacking the artifact
    # distribution_restriction rides along so a consumer reading repodata sees
    # a licence's territorial exclusion without unpacking the artifact.
    prov = {k: v for k, v in (about.get("extra") or {}).items()
            if k in ("run_id", "run_url", "source_commit", "source_repo",
                     "source_rev", "built_from_source", "prebuilt_wheel_used",
                     "torch_build", "cuda_compiler_version", "arch_list",
                     "distribution_restriction")}
    if prov:
        entry["provenance"] = prov

    out = args.meta_dir / args.subdir / f"{args.conda_file.name}.json"
    if out.exists():
        have = json.loads(out.read_text()).get("sha256")
        if have == sha256:
            print(f"{out}: already present with same sha256 — no-op")
            return
        sys.exit(f"{out}: exists with DIFFERENT sha256 ({have} vs {sha256}); "
                 "published artifacts are immutable — bump the build number instead")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entry, indent=1, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
