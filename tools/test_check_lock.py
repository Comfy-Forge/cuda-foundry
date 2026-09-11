#!/usr/bin/env python3
"""known_bad.json's `fixed_by` values must name builds that exist.

tools/check_lock.py prints `fixed_by` to a user holding a lockfile that
pinned a defective build, and the user re-locks onto it. It named
torchvision-0.23.0-cuda128_torch28_py312_h6651153_2.conda while the win-64
channel carried _1 and _3 -- a replacement nobody could solve for. So every
fixed_by is checked against what is actually published: a .conda against
the committed fragments under meta/<subdir>/ (offline), a .whl against the
subdir release's asset list (GitHub API; skipped LOUDLY when unreachable).

A negative control runs first: a synthetic known_bad with a phantom
fixed_by must be reported, or the check could not fail.

Run: python tools/test_check_lock.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from check_lock import fixed_by_problems  # noqa: E402

failures: list[str] = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def release_assets(subdir: str) -> set[str] | None:
    url = f"https://api.github.com/repos/Comfy-Forge/cuda-foundry/releases/tags/{subdir}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return {a["name"] for a in json.load(r).get("assets", [])}
    except Exception as e:  # network, rate limit, no such release
        print(f"WARN release asset list for {subdir} unavailable ({e}); wheel fixed_by NOT checked")
        return None


def main() -> int:
    meta = REPO / "meta"

    # ---- negative control -----------------------------------------------
    phantom = {"linux-64": {"x-1.0-cuda128_torch28_py312_h0_0.conda": {
        "reason": "r", "fixed_by": "x-1.0-cuda128_torch28_py312_h0_1.conda"}}}
    probs = fixed_by_problems(phantom, meta, {"linux-64": set()})
    check(any("has no fragment" in p for p in probs),
          "negative control: a fixed_by with no fragment is reported")
    circular = {"linux-64": {"a.conda": {"reason": "r", "fixed_by": "b.conda"},
                             "b.conda": {"reason": "r", "fixed_by": "c.conda"}}}
    probs = fixed_by_problems(circular, meta, {})
    check(any("is itself known-bad" in p for p in probs),
          "negative control: a fixed_by that is itself known-bad is reported")
    probs = fixed_by_problems({"win-64": {"a.whl": {"reason": "r", "fixed_by": "ghost.whl"}}},
                              meta, {"win-64": {"real.whl"}})
    check(any("not an asset" in p for p in probs),
          "negative control: a wheel fixed_by absent from the release is reported")

    # ---- the real file ----------------------------------------------------
    bad = json.loads((REPO / "known_bad.json").read_text())
    subdirs = [s for s in bad if not s.startswith("_")]
    assets = {}
    for s in subdirs:
        if any(str(i.get("fixed_by", "")).endswith(".whl") for i in bad[s].values()):
            got = release_assets(s)
            if got is not None:
                assets[s] = got
    probs = fixed_by_problems(bad, meta, assets)
    for p in probs:
        print(f"FAIL known_bad.json: {p}")
    failures.extend(probs)
    check(not probs, f"every fixed_by in known_bad.json names a published build "
                     f"({sum(len(bad[s]) for s in subdirs)} entries)")
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
