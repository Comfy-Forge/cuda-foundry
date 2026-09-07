#!/usr/bin/env python3
"""Test the win-64 compile ledger's ninja parsing, including its escaping.

This exists because the first version of `ninja_translation_units` mapped ZERO
outputs to sources and wrote an empty ledger -- and the artifact still passed
its publish gate, because `verify_conda.py` skips the ledger assertion when the
ledger is empty and prints a line that reads like it held (run 34166741643,
`ok no compiled TU came from outside the build work tree (0 TUs)`).

A check that cannot fail is worse than no check, so the parser gets a test with
a negative control: the fixture contains an edge that was NOT built, and the
test asserts it is absent from the result. Without that, a parser that returned
every edge in the file would pass just as happily as a correct one.

Run: python scripts/test_ninja_ledger.py
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "build_snippets" / "build_win.py"

# The escaping that broke it. Ninja writes a literal colon in a path as `$:`,
# so EVERY Windows path with a drive letter carries one, and splitting an edge
# on its first colon lands inside `D$:` rather than at the separator.
BUILD_NINJA = """\
rule compile
  command = cl $in /Fo$out
build D$:/a/work/build/temp/Release/ext.obj: compile D$:/a/work/ext.cpp
build D$:/a/work/build/temp/Release/ssim.obj: cuda_compile D$:/a/work/ssim.cu
build D$:/a/work/build/temp/Release/never.obj: compile D$:/a/work/never.cpp
"""

# Only the first two were actually built. `never.obj` is the negative control.
NINJA_LOG = """\
# ninja log v6
0\t100\t1700000000\tD:/a/work/build/temp/Release/ext.obj\tabc123
101\t900\t1700000001\tD:/a/work/build/temp/Release/ssim.obj\tdef456
"""


def load():
    spec = importlib.util.spec_from_file_location("build_win", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    bw = load()
    failures = []

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sub = root / "build" / "temp" / "Release"
        sub.mkdir(parents=True)
        (sub / "build.ninja").write_text(BUILD_NINJA)
        (sub / ".ninja_log").write_text(NINJA_LOG)

        outputs = bw.ninja_log_entries(root)
        units = bw.ninja_translation_units(root)

        def check(cond, msg):
            print(("ok   " if cond else "FAIL ") + msg)
            if not cond:
                failures.append(msg)

        check(len(outputs) == 2,
              f"the log's two outputs are read, comments skipped (got {len(outputs)})")
        check(units == ["D:/a/work/ext.cpp", "D:/a/work/ssim.cu"],
              f"built outputs map to their sources through the $: escaping "
              f"(got {units})")
        # The control. A parser that ignored .ninja_log and returned every edge
        # would satisfy every other assertion here.
        check("D:/a/work/never.cpp" not in units,
              "an edge that was never built is NOT in the ledger")
        check(bw._split_edge("D$:/a/x.obj: compile D$:/a/x.cpp")[0] == "D$:/a/x.obj",
              "the edge splits at the separator colon, not at the drive letter")
        check(bw._unescape("D$:/a/my$ dir/x$$y.cpp") == "D:/a/my dir/x$y.cpp",
              "ninja's three path escapes are reversed")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
