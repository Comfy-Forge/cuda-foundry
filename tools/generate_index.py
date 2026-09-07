#!/usr/bin/env python3
"""Build the two PEP 503 index trees served at comfy-forge.github.io/pypi-cuda-wheels.

Ported from cuda-wheels' scripts/generate_index.py. The wheels live as
release assets of THIS repo (same releases the .conda artifacts use); the
index is a static site deployed to the pypi-cuda-wheels repo's Pages. No
wheel is ever copied: both trees, and every per-combo sub-tree, are anchor
tags pointing at the same release URLs.

    /                 plain anchors. pip opens the wheel, finds no
                      Requires-Dist, and installs nothing extra. This is what
                      comfy-env's direct-URL installs need: a resolver must
                      not go chasing a dependency list.
    /deps/            the same anchors plus data-core-metadata, so a resolver
                      fetches the PEP 658 <wheel>.metadata sidecar and
                      installs the declared dependencies.
    /<cu>/<torch>/    both of the above, narrowed to one (CUDA, torch) cell.
                      The flat index cannot be resolved unambiguously: a
                      wheel's CUDA and torch versions live only in its local
                      version tag, which pip does not match on, so an
                      unpinned install picks the highest combo present rather
                      than the one the machine can load. Putting the combo in
                      the URL is what download.pytorch.org does (/whl/cu128/).

`data-core-metadata` is the PEP 714 spelling and `data-dist-info-metadata`
the PEP 658 original that older pip reads; PEP 714 tells index providers to
emit both. The value is "true" rather than a hash -- PEP 658 permits that
when the hash is unavailable, and hashing every sidecar would mean fetching
one file per wheel on every index build to save pip an integrity check it
does not require.

Three deliberate departures from the cuda-wheels original, each recorded
where it bites:

  * A sidecar is advertised only when the .metadata asset is actually
    present in the release (see `Wheel.has_sidecar`). The original
    advertised unconditionally; a 404 on an advertised sidecar is a hard
    resolver error, not a fallback to opening the wheel.
  * The shrinkage-guard baseline is fetched over HTTP from the live index
    rather than from a gh-pages checkout, because this site deploys as a
    Pages artifact and has no gh-pages branch to check out -- with the
    original's branch probe the guard would have been skipped on every run
    while looking like it ran.
  * The torch-free alias expansion is NOT ported. It listed one built wheel
    under every torch in its CUDA line, which made `pip freeze` disagree
    with the installed environment and was findable only through the index.
    It was a transition measure for an index that already had consumers;
    this one has none yet, so it starts without the wart.

Usage: generate_index.py [--out _site] [--repo OWNER/NAME] [--baseline URL|PATH]
"""

import argparse
import datetime as _dt
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Where the wheels live, as release assets. Same releases as the .conda
# artifacts: one compile, one upload, three outputs.
ASSET_REPO = "Comfy-Forge/cuda-foundry"
# Where this site is served from. Used for the shrinkage-guard baseline.
INDEX_URL = "https://comfy-forge.github.io/pypi-cuda-wheels"

_PEP658_ATTRS = ' data-core-metadata="true" data-dist-info-metadata="true"'

# Pulls the cell out of a wheel filename: +cu128torch2.8 -> ("cu128", "torch2.8")
_COMBO_RE = re.compile(r"\+(cu\d+)(torch[\d.]+)")

_STYLE = ("body{font-family:sans-serif;max-width:900px;margin:2rem auto;"
          "padding:0 1rem;line-height:1.5}code{background:#f4f6f8;padding:0 .3em}")


def normalize(name: str) -> str:
    """PEP 503 normalised project name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _next_link(link_header):
    """Return the rel="next" URL from a GitHub Link header, or None."""
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.split(";")
        if len(section) >= 2 and 'rel="next"' in section[1].strip():
            return section[0].strip().strip("<>")
    return None


def get_releases(repo: str, token: str = None) -> list:
    """Every release on `repo`, following the pagination chain to the end.

    The endpoint pages at 30 by default. A single unpaginated fetch silently
    truncates, and the short index that results looks exactly like a healthy
    one -- the packages that fall off simply stop existing for consumers.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"https://api.github.com/repos/{repo}/releases?per_page=100"
    releases, pages = [], 0
    while url:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            releases.extend(json.loads(response.read().decode()))
            url = _next_link(response.headers.get("Link"))
        pages += 1
        if pages > 50:  # 5000 releases; a runaway Link chain is a bug
            raise RuntimeError("release pagination did not terminate")
    print(f"Fetched {len(releases)} release(s) across {pages} page(s) from {repo}")
    # A 200 returning [] is indistinguishable from a total auth failure, and
    # publishing a zero-package index is worse than publishing a stale one:
    # pip hard-fails on a 404 rather than falling through to the next index.
    if not releases:
        sys.exit("ERROR: the releases API returned ZERO releases for "
                 f"{repo}. That is an auth failure or an empty repo; either "
                 "way, refusing to deploy an index built from nothing.")
    return releases


def fetch_baseline(source: str):
    """Previous packages.json, or None if the index was never published.

    Returns None ONLY for a definitive 404. Any other failure raises: an
    inconclusive probe must not silently disable the shrinkage guard, which
    is how the predecessor index came to be 58% dead links.
    """
    if not source.startswith(("http://", "https://")):
        p = Path(source)
        return json.loads(p.read_text()) if p.is_file() else None
    try:
        with urllib.request.urlopen(source, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def parse_wheel(filename: str) -> dict | None:
    """{name, version, cuda, torch, python, abi, platform} or None.

    PEP 427: {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl
    """
    if not filename.endswith(".whl"):
        return None
    parts = filename[:-len(".whl")].split("-")
    if len(parts) < 5:
        return None
    dist, version = parts[0], parts[1]
    python, abi, platform = parts[-3], parts[-2], parts[-1]
    combo = _COMBO_RE.search(version)
    return {
        "name": normalize(dist),
        "version": version,
        "cuda": combo.group(1) if combo else None,
        "torch": combo.group(2) if combo else None,
        # None for abi-agnostic wheels (py3-none, cpXY-abi3): they satisfy
        # every interpreter, and `abi` is what says which shape it is.
        "python": python if re.fullmatch(r"cp\d+", python) else None,
        "abi": abi,
        "platform": platform,
    }


def load_known_bad(path: Path) -> dict:
    """{subdir: {filename: entry}} -- the same file make_repodata.py reads.

    One record of "this artifact is defective" for both formats. Conda has no
    yank, so a bad .conda is dropped from repodata entirely; a wheel index does
    have one, so a bad wheel is yanked rather than removed. Two lists would
    drift, and the drift would be silent in exactly the situation where someone
    is trying to find out whether what they installed is the broken one.
    """
    return json.loads(path.read_text()) if path.is_file() else {}


def anchor(wheel: dict, with_metadata: bool) -> str:
    """One PEP 503 anchor, optionally advertising its PEP 658 sidecar.

    The href carries a `#sha256=` fragment whenever the digest is known (it
    always is -- the releases API serves one per asset, no downloads). Our
    release tags are mutable, so an unhashed anchor pins a URL whose bytes
    can change under it; the fragment is what lets pip/uv/pixi verify the
    artifact and lets a lockfile record a real hash.
    """
    attrs = _PEP658_ATTRS if (with_metadata and wheel["has_sidecar"]) else ""
    # PEP 592. A yanked file stays downloadable, so nobody who pinned this exact
    # filename breaks, but no resolver will SELECT it unless given that exact
    # pin -- which is the wheel analogue of dropping a .conda from repodata,
    # with the additional property that the reason reaches the user.
    if wheel.get("yanked"):
        attrs += f' data-yanked="{html.escape(wheel["yanked"], quote=True)}"'
    url = wheel["url"]
    if wheel.get("sha256"):
        url += f'#sha256={wheel["sha256"]}'
    return f'<a href="{url}"{attrs}>{wheel["filename"]}</a><br>\n'


def write_page(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<!DOCTYPE html>\n<html>\n<head><title>" + title + "</title>\n"
                    f"<style>{_STYLE}</style></head>\n<body>\n" + body + "</body>\n</html>\n")


def write_tree(root: Path, packages: dict, with_metadata: bool, heading: str,
               blurb: str = "") -> None:
    """A complete PEP 503 tree: a root listing plus one page per project."""
    links = "".join(f'<a href="{p}/">{p}</a><br>\n' for p in sorted(packages))
    write_page(root / "index.html", heading, f"<h1>{heading}</h1>\n{blurb}{links}")
    for pkg, wheels in packages.items():
        body = f"<h1>{pkg}</h1>\n" + "".join(
            anchor(w, with_metadata) for w in sorted(wheels, key=lambda w: w["filename"]))
        write_page(root / pkg / "index.html", pkg, body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("_site"),
                    help="directory to write the site into (never committed)")
    ap.add_argument("--repo", default=os.environ.get("ASSET_REPO", ASSET_REPO),
                    help="OWNER/NAME whose releases hold the wheels")
    ap.add_argument("--baseline", default=f"{INDEX_URL}/packages.json",
                    help="URL or path of the live packages.json (shrinkage guard)")
    ap.add_argument("--known-bad", type=Path,
                    default=Path(__file__).resolve().parent.parent / "known_bad.json",
                    help="defective artifacts; wheels listed here are yanked "
                         "per PEP 592 (the same file make_repodata.py reads)")
    args = ap.parse_args()

    releases = get_releases(args.repo, os.environ.get("GITHUB_TOKEN"))

    # Collect wheels, and note which ones have a sidecar asset beside them.
    # A sidecar is an asset named exactly <wheel>.metadata, so its URL is
    # exactly the wheel URL + ".metadata" -- which is where PEP 658 says a
    # resolver looks. Nothing rewrites URLs; the naming IS the contract.
    known_bad = load_known_bad(args.known_bad)
    yanked_count = 0

    packages: dict[str, list] = {}
    for release in releases:
        assets = release.get("assets", [])
        bad_here = known_bad.get(release.get("tag_name", "")) or {}
        sidecars = {a["name"] for a in assets if a["name"].endswith(".whl.metadata")}
        for asset in assets:
            name = asset["name"]
            if not name.endswith(".whl"):
                continue
            parsed = parse_wheel(name)
            if not parsed:
                print(f"WARNING: unparseable wheel name, skipped: {name}")
                continue
            bad = bad_here.get(name)
            if bad:
                yanked_count += 1
            packages.setdefault(parsed["name"], []).append({
                "yanked": (bad or {}).get("reason", "") if bad else "",
                "filename": name,
                "url": asset["browser_download_url"],
                # The releases API serves a digest per asset: free hash
                # verification for every anchor, no downloads.
                "sha256": (asset.get("digest") or "").removeprefix("sha256:"),
                "has_sidecar": f"{name}.metadata" in sidecars,
                "parsed": parsed,
            })

    # ── Shrinkage guard ────────────────────────────────────────────────
    # Discriminate by CAUSE, not by magnitude: a truncated fetch or a bad
    # prune loses ASSETS under releases that still exist, while an operator
    # deleting a package loses the RELEASE OBJECT.
    #
    #   lost package whose release tag is still live -> HARD FAIL
    #   lost package whose release tag is gone       -> loud WARN, proceed
    #
    # So a deliberate removal self-clears in one run, while the truncation
    # case the guard exists for still stops the deploy. No --force flag: a
    # bare --force becomes muscle memory and the guard stops existing.
    live_tags = {r.get("tag_name", "") for r in releases}
    baseline_doc = fetch_baseline(args.baseline)
    lost_live: list[str] = []
    lost_gone: list[str] = []
    if baseline_doc is None:
        print(f"WARNING: no index published at {args.baseline} yet -- the "
              "shrinkage guard has nothing to compare against and is SKIPPED. "
              "Expected on the very first deploy only.")
    elif int(baseline_doc.get("schema", 0)) < 1:
        sys.exit(f"ERROR: baseline at {args.baseline} has an unreadable schema; "
                 "refusing to publish with the shrinkage guard disabled.")
    else:
        prev_pkgs = baseline_doc.get("packages", {})
        for pkg in sorted(set(prev_pkgs) - set(packages)):
            # Every wheel a package had came from some release tag; if any of
            # those tags is gone the package was deliberately removed.
            tags = {w.get("tag", "") for w in prev_pkgs[pkg].get("wheels", [])}
            (lost_live if tags & live_tags else lost_gone).append(pkg)

    if lost_live:
        print(f"ERROR: {len(lost_live)} package(s) vanished from the index while "
              "their release still exists:")
        for name in lost_live:
            print(f"  - {name}  (release tag is LIVE -- assets went missing)")
        sys.exit("That is the signature of a truncated fetch, a partial API page, "
                 "or a prune running mid-generation -- NOT of a deliberate "
                 "removal. Refusing to publish. Re-run; if it persists, the "
                 "release genuinely lost its assets and needs a rebuild.")
    for name in lost_gone:
        print(f"NOTE: {name} is absent because its release was deleted -- "
              "publishing without it.")

    n_wheels = sum(len(v) for v in packages.values())
    n_sidecars = sum(1 for v in packages.values() for w in v if w["has_sidecar"])
    print(f"{len(packages)} package(s), {n_wheels} wheel(s), "
          f"{n_sidecars} with a PEP 658 sidecar, "
          f"{yanked_count} yanked (PEP 592)")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    root_blurb = (
        '<p>PEP 503 simple index of CUDA extensions compiled by '
        f'<a href="https://github.com/{args.repo}">{args.repo}</a>. Wheels here '
        'advertise <b>no</b> dependencies, which is what a direct-URL install '
        'wants. For a resolver that should install dependencies too, use '
        '<a href="deps/">/deps/</a>. Per-cell indexes live at '
        '<code>/cu&lt;ver&gt;/torch&lt;ver&gt;/</code>.</p>\n')
    deps_blurb = (
        '<p>The same wheels as the root index, at the same URLs. This tree '
        'advertises a PEP 658 <code>.metadata</code> sidecar per wheel, so a '
        'resolver installs each package&#39;s third-party dependencies. '
        '<b>torch is deliberately excluded</b> from those sidecars: these '
        'wheels are pinned to one exact (CUDA, torch) ABI in their local '
        'version, which pip ignores when resolving, so torch must be installed '
        'first.</p>\n')

    write_tree(out, packages, False, "CUDA Foundry wheels", root_blurb)
    write_tree(out / "deps", packages, True,
               "CUDA Foundry wheels -- dependency-declaring mirror", deps_blurb)

    # Per-cell trees, both flavours, from the same wheel records.
    combos: dict[tuple[str, str], dict] = {}
    for pkg, wheels in packages.items():
        for w in wheels:
            p = w["parsed"]
            if p["cuda"] and p["torch"]:
                combos.setdefault((p["cuda"], p["torch"]), {}).setdefault(pkg, []).append(w)
    for (cuda, torch), pkgs in sorted(combos.items()):
        write_tree(out / cuda / torch, pkgs, False, f"CUDA Foundry -- {cuda} / {torch}")
        write_tree(out / "deps" / cuda / torch, pkgs, True,
                   f"CUDA Foundry -- {cuda} / {torch} (with dependencies)", deps_blurb)
    print(f"Generated {len(combos)} per-cell index/indexes "
          f"({sum(len(p) for p in combos.values())} package entries)")

    # Machine-readable manifest. Consumers (comfy-env first) read this
    # instead of regex-scraping the HTML, and get sha256 verification with
    # it. `tag` is what the next run's shrinkage guard discriminates on.
    manifest = {
        "schema": 1,
        "asset_repo": args.repo,
        "index_url": INDEX_URL,
        "generated_at": _dt.datetime.now(_dt.timezone.utc)
                            .replace(microsecond=0).isoformat(),
        "source_commit": os.environ.get("GITHUB_SHA", ""),
        "release_count": len(releases),
        "asset_count": n_wheels,
        "removed_since_previous": lost_gone,
        "packages": {},
    }
    tag_of = {a["browser_download_url"]: r.get("tag_name", "")
              for r in releases for a in r.get("assets", [])}
    for pkg, wheels in sorted(packages.items()):
        manifest["packages"][pkg] = {"wheels": sorted(
            ({"filename": w["filename"], "url": w["url"], "sha256": w["sha256"],
              "has_sidecar": w["has_sidecar"], "tag": tag_of.get(w["url"], ""),
              **{k: w["parsed"][k] for k in
                 ("version", "cuda", "torch", "python", "abi", "platform")}}
             for w in wheels), key=lambda e: e["filename"])}
    (out / "packages.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"Wrote packages.json: {n_wheels} wheels, {len(packages)} packages")


if __name__ == "__main__":
    main()
