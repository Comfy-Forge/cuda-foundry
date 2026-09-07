#!/usr/bin/env python3
"""Live solve sweep: one `pixi lock` per published cell, against the channel.

Two things this proves that inspecting a file cannot:

  1. the artifact actually resolves from OUR release URL, and
  2. the pytorch it resolves ALONGSIDE has the same CUDA flavour.

(2) is the assertion this repo exists for. `run_exports` gives the extension
a torch dependency of `pytorch >=2.11,<2.12`, which matches cu126, cu128,
cu129 and cu130 builds alike -- so without the explicit build-glob dep in the
recipe, torch flavour and extension flavour are two independent solver
choices and a cu128 extension can land beside a cu130 torch. `cuda-version`
does not save you: it constrains the CUDA MAJOR only.

Usage: sweep_solve.py [--package NAME] [--work DIR] [--limit N]
Exit: 0 if every cell is OK, 1 otherwise.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIXI = os.environ.get("PIXI", os.path.expanduser("~/.pixi/bin/pixi"))
CHANNEL = "https://comfy-forge.github.io/cuda-foundry"
TORCH_CHANNEL = "https://comfy-forge.github.io/conda-torch"
OURS = "https://github.com/Comfy-Forge/cuda-foundry/releases/download/"

# cuda128_torch211_py312_h<hash>_0
BUILD_RE = re.compile(r"^cuda(\d+)_torch(\d+)_py(\d+)_h[0-9a-f]+_(\d+)$")


def cells_from_meta(meta_dir: Path, only: str = ""):
    """Every published artifact, newest build number per (name, version, build-sans-number)."""
    best = {}
    for sub in sorted(p for p in meta_dir.iterdir() if p.is_dir()):
        for frag in sorted(sub.glob("*.json")):
            e = json.loads(frag.read_text())
            if only and e.get("name") != only:
                continue
            m = BUILD_RE.match(e.get("build", ""))
            if not m:
                continue
            key = (sub.name, e["name"], e["version"], m.group(1), m.group(2), m.group(3))
            n = int(m.group(4))
            if key not in best or n > best[key][1]:
                best[key] = (frag.name[: -len(".json")], n, e)
    return {k: v[0] for k, v in best.items()}, {k: v[2] for k, v in best.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="", help="restrict to one conda package name")
    ap.add_argument("--work", type=Path, default=Path("/tmp/conda-cuda-sweep"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N cells")
    ap.add_argument("--meta-dir", type=Path, default=REPO / "meta")
    args = ap.parse_args()
    os.environ.setdefault("PIXI_CACHE_DIR", str(args.work / "cache"))

    if not args.meta_dir.is_dir():
        print("no meta/ yet: nothing published to sweep")
        return 0
    picks, entries = cells_from_meta(args.meta_dir, args.package)
    if not picks:
        print("no published cells matched")
        return 0

    results = []
    for i, (key, filename) in enumerate(sorted(picks.items())):
        if args.limit and i >= args.limit:
            break
        subdir, name, version, cu, torch_nodot, py = key
        cuda = f"{cu[:2]}.{cu[2:]}"
        python = f"{py[0]}.{py[1:]}"
        build = filename[len(f"{name}-{version}-"):-len(".conda")]
        label = f"{name}/{version}/cuda{cu}/torch{torch_nodot}/py{python}/{subdir}"

        proj = args.work / "proj"
        shutil.rmtree(proj, ignore_errors=True)
        proj.mkdir(parents=True)
        (proj / "pixi.toml").write_text(
            f'[workspace]\nname = "sweep"\n'
            f'channels = ["{CHANNEL}", "{TORCH_CHANNEL}", "conda-forge"]\n'
            f'platforms = [{{ platform = "{subdir}", cuda = "{cuda}" }}]\n\n'
            f'[dependencies]\n'
            f'python = "{python}.*"\n'
            f'{name} = {{ version = "=={version}", build = "{build}" }}\n')
        p = subprocess.run([PIXI, "lock"], cwd=proj, capture_output=True,
                           text=True, timeout=1800)
        if p.returncode != 0:
            tail = (p.stderr or p.stdout).strip().splitlines()
            results.append((label, "UNSAT", tail[-1][:160] if tail else "?"))
        else:
            lock = (proj / "pixi.lock").read_text()
            if f"{OURS}{subdir}/{filename}" not in lock:
                verdict = "WRONG-SOURCE" if filename in lock else "MISSING-IN-LOCK"
                results.append((label, verdict, filename))
            else:
                # the assertion that matters: the torch we resolved beside it
                # must carry the SAME cuda flavour
                m = re.search(r"/pytorch-[\d.]+-cuda(\d+)_", lock)
                if not m:
                    results.append((label, "NO-PYTORCH-IN-LOCK", filename))
                elif m.group(1) != cu:
                    results.append((label, "FLAVOUR-MISMATCH",
                                    f"extension cuda{cu} but pytorch cuda{m.group(1)}"))
                else:
                    results.append((label, "OK", f"pytorch cuda{m.group(1)}"))
        print(*results[-1], sep="\t", flush=True)

    ok = sum(1 for r in results if r[1] == "OK")
    print(f"\n=== SWEEP: {ok}/{len(results)} OK ===")
    for r in results:
        if r[1] != "OK":
            print(*r, sep="\t")
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "results.json").write_text(json.dumps(results, indent=1))
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
