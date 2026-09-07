#!/usr/bin/env python3
"""Clone a package's source at its pinned rev, patch it, emit a tarball.

This runs OUTSIDE rattler-build, and that placement is the whole point: it is
what lets the build script run with the network denied. Submodules, patches
and any vendoring all happen here, so by the time the sandbox starts there is
nothing left for the build to legitimately download — and therefore nothing
to distinguish a legitimate download from fetching an upstream prebuilt wheel.

Usage:
    fetch_patched_sources.py --package flash_attn --output-dir staging
    fetch_patched_sources.py --package all --output-dir staging
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_SHA1 = re.compile(r"^[0-9a-f]{40}$")


def run(*cmd, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def load_packages(only: str) -> list[tuple[str, dict]]:
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from package_loader import iter_packages  # type: ignore
    except ImportError:
        sys.exit("scripts/package_loader.py not found — it owns the package.yml schema")
    out = [(f, c) for f, c in iter_packages()
           if only == "all" or only in (f, c.get("name"))]
    if not out:
        sys.exit(f"no package matching {only!r}")
    return out


def revs_for(cfg: dict) -> list[tuple[str, str]]:
    """[(version, rev)] this package can be built from.

    A FAMILY package (torchvision, torchaudio) has one revision per torch
    pairing, so the source job must produce a tarball for each: torchvision
    0.26.0 and 0.28.0 are different code, and a single tarball would silently
    build one version's source under another version's label.
    """
    fv = cfg.get("family_versions") or {}
    if fv:
        seen, out = set(), []
        for entry in fv.values():
            key = (entry["version"], entry["source_rev"])
            if key not in seen:
                seen.add(key)
                out.append(key)
        return sorted(out)
    return [(str(cfg.get("version") or ""),
             cfg.get("source_rev") or cfg.get("source_tag"))]


def fetch_one(cfg: dict, outdir: Path, version: str = "", rev: str = "") -> Path:
    name = cfg["name"]
    repo = cfg["source_repo"]
    rev = rev or cfg.get("source_rev") or cfg.get("source_tag")
    if not rev:
        sys.exit(f"{name}: no source_rev — a floating ref makes the build "
                 f"unreproducible and the provenance record a lie")
    if not _SHA1.match(str(rev)):
        # A tag can be moved; a commit cannot. The provenance we stamp into
        # every artifact claims a specific commit, so resolve it here and
        # record what we actually got.
        print(f"{name}: WARNING source_rev {rev!r} is not a 40-hex commit; "
              f"resolving it now and pinning the result", file=sys.stderr)

    # Per-version work dir and tarball, so a family package's revisions do not
    # overwrite each other.
    stem = f"{name}-{version}" if version else name
    work = outdir / stem
    if work.exists():
        shutil.rmtree(work)
    work.parent.mkdir(parents=True, exist_ok=True)

    url = repo if repo.startswith(("http://", "https://", "git@")) \
        else f"https://github.com/{repo}.git"
    run("git", "clone", "--filter=blob:none", "--no-checkout", url, str(work))
    run("git", "checkout", "--detach", str(rev), cwd=work)
    if cfg.get("clone_recursive"):
        run("git", "submodule", "update", "--init", "--recursive", "--depth", "1", cwd=work)

    resolved = subprocess.run(["git", "rev-parse", "HEAD"], cwd=work,
                              capture_output=True, text=True, check=True).stdout.strip()

    patch = cfg.get("patch_script")
    if patch:
        patch_path = REPO / patch
        if not patch_path.is_file():
            sys.exit(f"{name}: patch_script {patch} does not exist")
        env = dict(os.environ, CUW_PACKAGE=name, CUW_SOURCE_REV=resolved)
        subprocess.run([sys.executable, str(patch_path)], cwd=work, env=env, check=True)

    # Drop .git: it is large, it is not an input to the build, and leaving it
    # in makes the tarball non-deterministic.
    shutil.rmtree(work / ".git", ignore_errors=True)

    tarball = outdir / f"{stem}-source.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(work, arcname=name)   # arcname stays the package name
    (outdir / f"{stem}-source.rev").write_text(resolved + "\n")
    print(f"{name}: {resolved} -> {tarball} ({tarball.stat().st_size} bytes)")
    return tarball


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("staging"))
    ap.add_argument("--versions", default="",
                    help="comma-separated versions to fetch (family packages "
                         "have one revision each; default is all of them, "
                         "which is 16 clones for torchvision)")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {v.strip() for v in args.versions.split(",") if v.strip()}
    for _folder, cfg in load_packages(args.package):
        pairs = revs_for(cfg)
        if wanted:
            picked = [(v, r) for v, r in pairs if v in wanted]
            if not picked:
                sys.exit(f"{cfg['name']}: no revision matches --versions "
                         f"{sorted(wanted)}; it has {[v for v, _ in pairs]}")
            pairs = picked
        for version, rev in pairs:
            fetch_one(cfg, args.output_dir, version=version, rev=rev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
