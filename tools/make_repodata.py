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

purls are re-derived at assembly from packages/<name>/package.yml
`pypi_project` (see fragment.py), so a fragment written when the purl was
still derived from pypi_name -- which produced ~20 false purls -- loses it the
next time the site is built, without the fragment on disk being rewritten.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fragment import package_cfg_for, pypi_project_for  # noqa: E402

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


def _build_number(build: str) -> int:
    """The trailing _N of a build string, which is what conda ranks builds by."""
    tail = str(build).rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def superseded_by(entry: dict, packages_conda: dict, bad: set) -> str | None:
    """A GOOD same-cell build with a higher build number, if one is published.

    Conda's own selection rule: among candidates of the same name and version,
    the highest build number wins. A bad build with a higher sibling is
    therefore unreachable by a fresh solve without anyone doing anything.

    `bad` is excluded, and that exclusion is the whole point rather than a
    detail. A build superseded only by another DEFECTIVE build is not
    protected -- the solver moves off one bad artifact onto another. Checking
    conda-torch's live channel with this function found exactly that:
    libtorch-2.8.0-cuda129_repack_h327d83bf_0 is listed as superseded by _1,
    and _1 is itself in known_bad.json. Counting it would have reported the
    pair as safe.
    """
    for other_fn, other in packages_conda.items():
        if other_fn in bad:
            continue
        if (other.get("name") == entry.get("name")
                and other.get("version") == entry.get("version")
                and _build_number(other.get("build", ""))
                > _build_number(entry.get("build", ""))):
            return other_fn
    return None


def neutralised_by(entry: dict) -> str | None:
    """An unsatisfiable constrain, which makes the solver refuse the build.

    The idiom is an upper bound below every real version of a package that must
    be present -- conda-torch patches `libcudnn <0.0a0` onto its dual-cudnn
    builds for exactly this. Such a build cannot be selected even though it is
    listed.
    """
    for c in entry.get("constrains") or []:
        if "<0.0a0" in str(c).replace(" ", ""):
            return str(c)
    return None


def drop_known_bad(known_bad: dict, subdir: str, packages_conda: dict) -> list:
    """Remove defective builds that a fresh solve can still REACH.

    The test is reachability, not policy, and that is why two channels can look
    like they disagree while following one rule. A known-bad build stays listed
    if and only if something already stops a new solve choosing it:

      superseded   a same-cell build with a higher build number exists, so
                   conda's own ranking never reaches this one;
      neutralised  an unsatisfiable constrain (`<0.0a0` on a package that must
                   be present) makes the solver refuse it outright.

    Otherwise it is dropped, because listed and reachable is an offer.

    conda-torch keeps all six of its known-bad builds listed and is right to:
    four are superseded and two carry `libcudnn <0.0a0`. cuda-foundry dropped
    the no-jpeg torchvision and was right to: at that moment it had neither a
    higher build nor a patch, so a fresh solve would have chosen it. One rule,
    opposite outcomes, because the facts differed -- and this function will
    stop dropping that torchvision by itself once build 2 publishes.

    Staying listed is the better end state wherever it applies. The release
    asset is immutable either way, so a lockfile already pinning a bad build
    keeps resolving; leaving the entry in repodata means the channel and
    known_bad.json agree about what exists, and check_lock.py can still explain
    it to whoever is holding that lock. Dropping is what you do when nothing
    else stops the build being chosen.

    A key naming nothing is a hard error rather than a shrug. These entries are
    written by hand at the worst possible moment, and a typo'd filename in the
    file whose entire job is to neutralise a bad build would protect nobody
    while looking exactly as though it had.
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

    # `fixed_by` must name an artifact that EXISTS: it is what check_lock.py
    # tells someone holding a lockfile to re-pin onto, and a typo there sends
    # them to a filename nothing serves (the win-64 torchvision entry named
    # `_2` when the fix was `_3`). Wheels are checked against the wheel index
    # by generate_index.py; here only .conda targets are checked.
    for fn, entry in entries.items():
        target = str(entry.get("fixed_by") or "")
        if target.endswith(".conda") and target not in packages_conda:
            sys.exit(
                f"ERROR: known_bad.json {subdir!r}/{fn}: fixed_by names "
                f"{target!r}, and no such artifact exists in meta/{subdir}/. "
                f"A superseding build that does not exist supersedes nothing; "
                f"name the published build exactly, or drop fixed_by until it "
                f"is published.")

    dropped = []
    for fn in sorted(entries):
        entry = packages_conda[fn]
        newer = superseded_by(entry, packages_conda, set(entries))
        constrain = None if newer else neutralised_by(entry)
        if newer or constrain:
            why = f"superseded by {newer}" if newer else f"neutralised by {constrain!r}"
            print(f"{subdir}: known-bad {fn} KEPT LISTED -- {why}")
        else:
            del packages_conda[fn]
            dropped.append(fn)
    return dropped


def apply_purl_policy(packages_conda: dict) -> int:
    """Re-derive every entry's purl from package.yml `pypi_project`.

    Fragments are immutable records of the artifact, but a purl is not a fact
    about the artifact -- it is a claim about PyPI, and the claim was wrong
    for ~20 packages when it was derived from pypi_name. So the served
    repodata takes it from package.yml at assembly: set -> that purl, null ->
    none. An entry with no package.yml (the hand-written pccm) keeps what its
    fragment says, because the artifact's own about.extra was the source.
    Returns how many entries lost a purl.
    """
    dropped = 0
    for entry in packages_conda.values():
        proj = pypi_project_for(entry.get("name", ""))
        if proj:
            entry["purls"] = [f"pkg:pypi/{proj}@{entry.get('version')}"]
        elif package_cfg_for(entry.get("name", "")) is not None and entry.pop("purls", None):
            dropped += 1
    return dropped


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
        purls_dropped = apply_purl_policy(packages_conda)
        if purls_dropped:
            print(f"{subdir}: purl removed from {purls_dropped} entr(ies) whose "
                  f"package.yml sets no pypi_project")
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
