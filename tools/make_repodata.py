#!/usr/bin/env python3
"""Assemble the static channel site from committed repodata fragments.

Reads meta/<subdir>/*.json (one fragment per .conda, produced by
fragment.py) and writes site/<subdir>/repodata.json with CEP-15
info.base_url pointing at the matching GitHub release, so packages are
fetched from release assets while repodata is served by GitHub Pages.

Metadata overlay: patches/<subdir>/patches.json may override selected
repodata keys per exact .conda filename. Fragments on disk stay untouched
(published artifacts are immutable); the overlay is the metadata-fix path.
A patch naming a filename with no fragment is a hard error, so a typo
cannot silently no-op.

Also emits, per subdir: repodata.json.zst, and run_exports.json(.zst) —
the index a builder consults when resolving host dependencies (it reads
the channel index, never the artifacts). Its data comes from each
fragment's `run_exports` key, which fragment.py lifts out of the
artifact's info/run_exports.json.

Usage: make_repodata.py [--meta-dir meta] [--site-dir site] [--patches-dir patches]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RELEASES = "https://github.com/Comfy-Forge/cuda-foundry/releases/download"
CHANNEL = "https://comfy-forge.github.io/cuda-foundry"
# every subdir a client might request must exist (404s abort some solvers)
ALWAYS_SUBDIRS = {"noarch", "linux-64", "linux-aarch64", "win-64", "osx-arm64", "osx-64"}
# keys a patch may override; anything else in a patch entry is a hard error.
# run_exports/license included so a metadata-only fix to either never
# forces a full artifact republish (a review found the original set too
# narrow for exactly that case).
PATCHABLE_KEYS = {"depends", "constrains", "purls", "run_exports", "license", "license_family"}


def write_json(path: Path, payload: dict) -> None:
    """Write an index and its zstd sibling (clients prefer the .zst)."""
    body = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    zst = subprocess.run(["zstd", "-19", "--stdout"], input=body.encode(),
                         capture_output=True, check=True).stdout
    path.with_suffix(path.suffix + ".zst").write_bytes(zst)


def drop_known_bad(known_bad: dict, subdir: str, packages_conda: dict) -> list:
    """Remove defective builds from repodata so no solver can select them.

    known_bad.json was, until now, read only by tools/check_lock.py -- a tool
    someone runs against a lockfile they already have. That protects a person
    who thinks to ask. It does nothing for the solve that has not happened yet,
    and win-64/repodata.json was serving
    torchvision-0.23.0-cuda128_torch28_py312_h6651153_1.conda -- built without
    jpeg, webp or nvjpeg -- to anyone who resolved against this channel, while
    the repo held a file saying we knew it was broken. A record that does not
    change what the system does is the worst of both.

    Conda has no yank: an entry is either in repodata or it is not. Dropping it
    is therefore the strongest available statement, and it costs nothing that
    matters -- the release asset stays exactly where it was, byte for byte, so
    an existing lockfile pinning that URL keeps resolving and immutability
    holds. What changes is that no NEW solve can arrive at it.

    A key naming nothing is a hard error rather than a shrug. These entries are
    written by hand at the worst possible moment, and a typo'd filename in a
    file whose entire job is to neutralise a bad build would protect nobody
    while looking exactly like it had.
    """
    # .conda keys only. The same file also records defective WHEELS, which
    # tools/generate_index.py yanks per PEP 592 -- one list of "this artifact is
    # defective" for both formats rather than two that drift apart.
    entries = {fn: v for fn, v in (known_bad.get(subdir) or {}).items()
               if fn.endswith(".conda")}
    unknown = [fn for fn in entries if fn not in packages_conda]
    if unknown:
        sys.exit(
            f"ERROR: known_bad.json lists {unknown} under {subdir!r}, and no "
            f"such artifact exists in meta/{subdir}/. A known-bad entry that "
            f"matches no filename neutralises nothing. Fix the key to match "
            f"the .conda filename exactly, or remove it.")
    for fn in entries:
        del packages_conda[fn]
    return sorted(entries)


def load_patches(patches_dir: Path, subdir: str, packages_conda: dict) -> int:
    pfile = patches_dir / subdir / "patches.json"
    if not pfile.is_file():
        return 0
    patches = json.loads(pfile.read_text())
    applied = 0
    for filename, override in patches.items():
        if filename not in packages_conda:
            sys.exit(f"{pfile}: patch targets {filename!r} but no such fragment exists "
                     f"in meta/{subdir}/ — fix the filename or drop the patch")
        bad = set(override) - PATCHABLE_KEYS
        if bad:
            sys.exit(f"{pfile}: {filename}: keys {sorted(bad)} are not patchable "
                     f"(allowed: {sorted(PATCHABLE_KEYS)})")
        packages_conda[filename].update(override)
        applied += 1
    return applied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, default=Path("meta"))
    ap.add_argument("--site-dir", type=Path, default=Path("site"))
    ap.add_argument("--patches-dir", type=Path, default=Path("patches"))
    ap.add_argument("--known-bad", type=Path,
                    default=Path(__file__).resolve().parent.parent / "known_bad.json",
                    help="builds to exclude from repodata (see known_bad.json)")
    args = ap.parse_args()

    known_bad = json.loads(args.known_bad.read_text()) if args.known_bad.is_file() else {}

    subdirs = {p.name for p in args.meta_dir.iterdir() if p.is_dir()} | ALWAYS_SUBDIRS
    summary = []
    for subdir in sorted(subdirs):
        packages_conda = {}
        for frag in sorted((args.meta_dir / subdir).glob("*.json")) if (args.meta_dir / subdir).is_dir() else []:
            filename = frag.name[: -len(".json")]
            packages_conda[filename] = json.loads(frag.read_text())
        dropped = drop_known_bad(known_bad, subdir, packages_conda)
        for fn in dropped:
            print(f"{subdir}: EXCLUDED known-bad {fn}")
        patched = load_patches(args.patches_dir, subdir, packages_conda)

        # run_exports lives in the fragments (and is patchable), but ships in
        # its OWN index, not in repodata entries — that is where a builder
        # resolving host deps looks, and it keeps repodata.json standard.
        run_exports = {fn: {"run_exports": entry.pop("run_exports", None) or {}}
                       for fn, entry in packages_conda.items()}

        repodata = {
            "info": {"subdir": subdir, "base_url": f"{RELEASES}/{subdir}/"},
            "packages": {},
            "packages.conda": packages_conda,
            "repodata_version": 2,
        }
        out = args.site_dir / subdir / "repodata.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, repodata)
        write_json(args.site_dir / subdir / "run_exports.json",
                   {"info": {"subdir": subdir, "version": 1},
                    "packages": {}, "packages.conda": run_exports})
        n_rex = sum(1 for v in run_exports.values() if v["run_exports"])
        summary.append((subdir, len(packages_conda), patched, n_rex))

    lines = "".join(
        f"<tr><td>{s}</td><td>{n}</td><td><a href='{s}/repodata.json'>repodata.json</a></td>"
        f"<td><a href='{s}/run_exports.json'>run_exports.json</a></td></tr>"
        for s, n, _, _ in summary
    )
    (args.site_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>comfy-forge conda channel</title>"
        "<style>body{font-family:system-ui;margin:3em auto;max-width:40em}"
        "td{padding:.3em 1em;border-bottom:1px solid #ccc}</style>"
        f"<h1>comfy-forge conda channel</h1><p>Add <code>{CHANNEL}</code> as a conda channel. "
        "Packages are served as GitHub release assets via CEP-15 <code>base_url</code>.</p>"
        f"<table><tr><th>subdir</th><th>packages</th><th></th><th></th></tr>{lines}</table>"
    )
    for s, n, p, r in summary:
        print(f"{s}: {n} packages" + (f" ({p} patched)" if p else "")
              + (f", {r} with run_exports" if r else ""))


if __name__ == "__main__":
    main()
