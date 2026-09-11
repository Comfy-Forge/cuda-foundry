#!/usr/bin/env python3
"""Assert the from-source guarantee cannot have silently regressed.

The guarantee is "the build cannot fetch a prebuilt binary", and it is
enforced by scripts/build_snippets/nonet.py: a seccomp filter that denies
AF_INET/AF_INET6 socket creation, installed by the build immediately before
the step that could download a wheel, inherited by every child, and
irreversible once set.

It is NOT enforced by rattler-build's --sandbox. That was the original
design and it does not work: the sandbox needs a separate rattler-sandbox
binary AND unprivileged namespace creation, which is unavailable both in
this project's container and on GitHub-hosted runners (measured on both:
"sandboxing failure: Operation not permitted (os error 1)"). A flag that
silently does nothing is worse than no flag, so this lint checks for the
mechanism that is real.

Failure modes this catches, all of which leave CI green:
  * --allow-network appears anywhere
  * the compile step stops running under nonet.py
  * a generated recipe is hand-edited to drop the guard
  * nonet.py is made to fail open instead of exiting non-zero
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD_SH = REPO / "scripts" / "build_snippets" / "build.sh"
NONET = REPO / "scripts" / "build_snippets" / "nonet.py"
SEARCH = [REPO / ".github", REPO / "scripts", REPO / "templates", REPO / "recipes"]
SUFFIXES = {".yml", ".yaml", ".sh", ".py", ".j2"}


def strip_comments(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("#") or s.startswith("::"):
            continue
        out.append(raw)
    return out


def main() -> int:
    problems: list[str] = []

    # 1. --allow-network must not appear as an argument anywhere.
    for root in SEARCH:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            for line in strip_comments(path.read_text(errors="replace")):
                if "--allow-network" in line:
                    problems.append(
                        f"{path.relative_to(REPO)}: passes --allow-network. Vendor "
                        f"whatever the build needs in scripts/fetch_patched_sources.py.\n"
                        f"      {line.strip()}")

    # 2. the shared build script must run its wheel-capable step under the guard.
    if not BUILD_SH.is_file():
        problems.append("scripts/build_snippets/build.sh is missing")
    else:
        body = BUILD_SH.read_text()
        pip_lines = [l for l in strip_comments(body) if "pip install" in l]
        if not pip_lines:
            problems.append("build.sh no longer runs pip install — has the compile "
                            "step moved? The guard must wrap whatever replaced it.")
        for line in pip_lines:
            if "nonet.py" not in line:
                problems.append(
                    f"build.sh runs the compile WITHOUT nonet.py, so the build can "
                    f"reach the network and fetch a prebuilt binary.\n      {line.strip()}")

    # 3. nonet.py must fail closed: no path may run the command unprotected.
    if not NONET.is_file():
        problems.append("scripts/build_snippets/nonet.py is missing — the guarantee "
                        "has no mechanism at all")
    else:
        src = NONET.read_text()
        if "PR_SET_NO_NEW_PRIVS" not in src or "SECCOMP_SET_MODE_FILTER" not in src:
            problems.append("nonet.py no longer installs a seccomp filter")
        # install() must exit rather than return on failure
        if not re.search(r"sys\.exit\(f?\"nonet: seccomp", src):
            problems.append("nonet.py does not exit when the filter cannot be "
                            "installed — it would fail OPEN, running the build "
                            "with full network access")

    # 4. every generated recipe must carry the guard (a hand-edit would be
    #    caught by regen-check, but this states the invariant directly). The
    #    build script is file-backed -- recipes/<name>/build.sh, named by the
    #    recipe's `build.script.file` -- so the guard is looked for in the
    #    file the build runs, and the recipe must still point at it. A
    #    hand-written recipe (pccm) inlines its script and is checked as text.
    for recipe in sorted((REPO / "recipes").glob("*/recipe.yaml")):
        if recipe.parent.name.startswith("_"):
            continue  # canary and friends assert the property themselves
        text = recipe.read_text()
        sibling = recipe.parent / "build.sh"
        if "file: ${{ \"build_win.py\" if win else \"build.sh\" }}" in text:
            if not sibling.is_file():
                problems.append(f"{recipe.relative_to(REPO)}: names build.sh as its "
                                f"script and there is no build.sh beside it")
            elif not any("nonet.py" in l and "pip" in l
                         for l in strip_comments(sibling.read_text())):
                problems.append(f"{sibling.relative_to(REPO)}: the compile no "
                                f"longer runs under nonet.py")
        elif "nonet.py" not in text:
            problems.append(f"{recipe.relative_to(REPO)}: no nonet.py guard in the "
                            f"build script")

    if problems:
        print("::error::from-source guarantee lint failed", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return 1
    print("lint: no --allow-network; compile runs under nonet.py; guard fails "
          "closed; every recipe's build script carries it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
