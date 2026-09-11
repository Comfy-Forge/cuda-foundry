#!/usr/bin/env python3
"""List the test entries a built .conda carries, by kind, for `--test-index`.

rattler-build writes the recipe's `tests:` list into info/tests/tests.yaml
and `rattler-build test --package-file X --test-index N` runs exactly one
entry. The recipe puts the GPU op in a `script:` entry beside the
`python: {imports}` and `package_contents:` entries, and a GitHub runner has
no GPU -- so the CI step needs the indices of the entries that do NOT need
one. This reads them out of the artifact rather than assuming an order, so
a reordered template cannot make CI run the op on a box without a GPU (a
loud failure) or skip the import test (a silent one).

Usage:
  artifact_tests.py <pkg.conda> [--kind gpu-less|gpu|all]   -> indices, one per line
  artifact_tests.py <pkg.conda> --summary                    -> "N: kind" per entry

An artifact with no tests.yaml prints nothing and exits 3, so a caller that
expects tests fails rather than silently running none.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def entries(conda: Path, tmp: Path) -> list[dict]:
    from verify_conda import extract_conda
    root = extract_conda(conda, tmp / conda.stem)
    t = root / "info" / "tests" / "tests.yaml"
    if not t.is_file():
        return []
    import yaml
    return list(yaml.safe_load(t.read_text()) or [])


def kind(entry: dict) -> str:
    if not isinstance(entry, dict):
        return "unknown"
    if "script" in entry:
        return "gpu"          # the verify op: launches kernels
    if "python" in entry:
        return "python"       # imports (+ pip check)
    if "package_contents" in entry:
        return "package_contents"
    if "perl" in entry or "r" in entry or "downstream" in entry:
        return "other"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("conda", type=Path)
    ap.add_argument("--kind", default="gpu-less", choices=["gpu-less", "gpu", "all"])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/artifact-tests"))
    args = ap.parse_args()
    args.tmp.mkdir(parents=True, exist_ok=True)
    ents = entries(args.conda, args.tmp)
    if not ents:
        print(f"{args.conda.name}: no info/tests/tests.yaml", file=sys.stderr)
        return 3
    for i, e in enumerate(ents):
        k = kind(e)
        if args.summary:
            print(f"{i}: {k}")
        elif args.kind == "all" or (args.kind == "gpu" and k == "gpu") \
                or (args.kind == "gpu-less" and k != "gpu"):
            print(i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
