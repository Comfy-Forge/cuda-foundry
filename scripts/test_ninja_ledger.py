#!/usr/bin/env python3
"""Test the win-64 compile ledger's ninja parsing and the shard partition.

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
import os
import shutil
import subprocess
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
    failures: list = []

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

        # The nvcc/cl split. Only cuda_compile edges pass through ccache, so
        # this set is the denominator of the zero-miss gate -- and if it
        # silently included the cl TU the gate would demand a cache hit for a
        # translation unit nothing ever cached, and no link job could pass.
        cuda = bw.ninja_cuda_units(root)
        check(cuda == ["D:/a/work/ssim.cu"],
              f"only the cuda_compile edge counts as an nvcc TU (got {cuda})")

    failures += continuation_and_fallback_checks()
    failures += partition_checks()
    failures += stats_checks()
    failures += stub_checks()
    failures += tree_checks()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


def continuation_and_fallback_checks() -> list:
    """Two ways a real build file differs from the fixture above.

    ccimport (cumm, spconv) writes its build.ninja through ninja_syntax.Writer,
    which wraps long edges at 78 columns with a trailing `$` -- so the first
    input of an edge sits on the NEXT line, and a line-at-a-time parser reads
    the rule name as the source. And torch's BuildExtension overwrites ONE
    build.ninja per extension in a shared build_temp while .ninja_log there
    accumulates: for a four-extension package only the last extension's edges
    survive, so the object path is the only record left for the other three.
    Both used to produce a ledger that was non-empty and WRONG, which no gate
    downstream can tell from a correct one.
    """
    bw = load()
    failures: list = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "csrc" / "cuda").mkdir(parents=True)
        (root / "csrc" / "scatter.cpp").write_text("int a;\n")
        (root / "csrc" / "cuda" / "scatter_cuda.cu").write_text("int b;\n")
        (root / "csrc" / "version.cpp").write_text("int c;\n")
        (root / "gen" / "src").mkdir(parents=True)
        (root / "gen" / "src" / "long_name_component_number_one.cc").write_text("int d;\n")
        bdir = root / "build" / "temp.win-amd64-cpython-312" / "Release"
        bdir.mkdir(parents=True)
        # The surviving build.ninja: the LAST extension (version), plus a
        # ccimport-style wrapped edge whose input is on a continuation line.
        (bdir / "build.ninja").write_text(
            "rule compile\n  command = cl $in /Fo$out\n"
            f"build {bdir}/csrc/version.obj: compile {root}/csrc/version.cpp\n"
            f"build {bdir}/gen/objs/long_name_component_number_one.o: $\n"
            f"    core_cc_cxx_compiler__cc $\n"
            f"    {root}/gen/src/long_name_component_number_one.cc | $\n"
            f"    {root}/gen/include/x.h\n")
        (bdir / ".ninja_log").write_text(
            "# ninja log v6\n"
            f"0\t1\t170\t{bdir}/csrc/scatter.obj\tabc\n"
            f"0\t1\t170\t{bdir}/csrc/cuda/scatter_cuda.obj\tabc\n"
            f"0\t1\t170\t{bdir}/csrc/version.obj\tabc\n"
            f"0\t1\t170\t{bdir}/gen/objs/long_name_component_number_one.o\tabc\n"
            f"0\t1\t170\t{bdir}/csrc/nowhere.obj\tabc\n")
        units = [u.replace("\\", "/") for u in bw.ninja_translation_units(root)]
        want_edge = f"{root}/csrc/version.cpp"
        want_cont = f"{root}/gen/src/long_name_component_number_one.cc"
        _check(failures, want_edge in units,
               "the extension whose build.ninja survived maps through its edge")
        _check(failures, want_cont in units and "core_cc_cxx_compiler__cc" not in " ".join(units),
               f"a `$`-continued edge maps to its source, not its rule name (got {units})")
        _check(failures, any(u.endswith("/csrc/scatter.cpp") for u in units)
               and any(u.endswith("/csrc/cuda/scatter_cuda.cu") for u in units),
               "objects whose build.ninja was overwritten map back through their path")
        _check(failures, not any("nowhere" in u for u in units),
               "an object with no source anywhere in the tree is NOT invented")
        _check(failures, len(units) == 4, f"exactly the four real TUs (got {len(units)})")
    return failures


def tree_checks() -> list:
    """partition_sources and check_declared_tus against a real directory.

    These two are the whole win-64 partition: one rewrites the files that are
    not this shard's, the other refuses to believe the declaration until
    .ninja_log agrees with it. The controls are the failures that would
    otherwise be silent -- a shard that stubbed its OWN slice, and a
    shard_sources list that describes a build other than the one that ran.
    """
    bw = load()
    failures: list = []
    patterns = ["csrc/flash_api.cpp", "csrc/src/*.cu"]

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "csrc" / "src").mkdir(parents=True)
        (root / "csrc" / "flash_api.cpp").write_text(
            "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { }\n")
        for i in range(12):
            (root / "csrc" / "src" / f"k{i}.cu").write_text(f"__global__ void k{i}(){{}}\n")
        # Not declared, and not a translation unit: a header that happens to
        # live beside the sources must not be touched.
        (root / "csrc" / "src" / "launch.h").write_text("#pragma once\n")

        caught = False
        try:
            bw.partition_sources(root, patterns, 30, 40)
        except SystemExit:
            caught = True
        _check(failures, caught,
               "a shard with an empty slice is REFUSED, not silently run")

        mine = bw.partition_sources(root, patterns, 0, 3)
        _check(failures, all(q.suffix == ".cu" for q in mine),
               "the C++ TU is never this shard's -- nothing caches it")
        for q in mine:
            _check(failures, "__global__" in q.read_text(),
                   f"{q.name} is this shard's slice and was NOT stubbed")
        stubbed = [q for q in (root / "csrc" / "src").glob("*.cu") if q not in mine]
        _check(failures, stubbed and all("stub" in q.read_text() for q in stubbed),
               f"the other {len(stubbed)} .cu file(s) were stubbed")
        api = (root / "csrc" / "flash_api.cpp").read_text()
        _check(failures, "PyInit_" in api and "PYBIND11_MODULE" not in api,
               "the module TU was stubbed WITH an entry point")
        _check(failures, (root / "csrc" / "src" / "launch.h").read_text() == "#pragma once\n",
               "an undeclared neighbouring header was left alone")

        # The declaration must match what ninja compiled. Build a log that
        # says so, then one that does not.
        bdir = root / "build" / "temp" / "Release"
        bdir.mkdir(parents=True)
        srcs = [root / "csrc" / "flash_api.cpp"] + \
               sorted((root / "csrc" / "src").glob("*.cu"))

        def write_ninja(units):
            edges = "\n".join(
                f"build {bdir}/{q.stem}.obj: "
                f"{'cuda_compile' if q.suffix == '.cu' else 'compile'} {q}"
                for q in units)
            (bdir / "build.ninja").write_text("rule compile\n\n" + edges + "\n")
            (bdir / ".ninja_log").write_text(
                "# ninja log v6\n" + "\n".join(
                    f"0\t1\t170\t{bdir}/{q.stem}.obj\tabc" for q in units) + "\n")

        write_ninja(srcs)
        ok = True
        try:
            bw.check_declared_tus(root, patterns)
        except SystemExit:
            ok = False
        _check(failures, ok, "a declaration that matches .ninja_log is accepted")

        # Control: ninja compiled a TU nobody declared. A partition that never
        # saw it would leave it compiled by every shard and cached by none.
        extra = root / "csrc" / "undeclared.cu"
        extra.write_text("__global__ void x(){}\n")
        write_ninja(srcs + [extra])
        caught = False
        try:
            bw.check_declared_tus(root, patterns)
        except SystemExit:
            caught = True
        _check(failures, caught,
               "a TU compiled but not declared is REFUSED")

        # Control: a pattern that matches nothing is an error, not an empty
        # slice -- it is what a renamed source directory looks like.
        caught = False
        try:
            bw.declared_tus(root, ["csrc/nowhere/*.cu"])
        except SystemExit:
            caught = True
        _check(failures, caught, "a pattern matching no file is REFUSED")
    return failures


def _check(failures, cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def partition_checks() -> list:
    """The shard partition: exhaustive, disjoint, and stable across runs.

    The Linux wrapper partitions one nvcc invocation at a time; win-64 has no
    seat to do that from, so it partitions the SOURCE files up front. The
    property that matters is unchanged, and it is the one the link job's
    zero-miss gate depends on: every declared translation unit is compiled for
    real by EXACTLY ONE shard. Compiled by none and the link job misses;
    compiled by two and the shards merely waste time.
    """
    bw = load()
    failures: list = []
    files = [f"csrc/src/f{i}.cu" for i in range(72)]

    for count in (1, 3, 25, 40):
        owners = [bw.slice_for(files, i, count) for i in range(count)]
        flat = [f for slice_ in owners for f in slice_]
        _check(failures, sorted(flat) == sorted(files),
               f"{count} shard(s): every TU is owned by exactly one shard "
               f"({len(flat)} of {len(files)})")
        _check(failures, len(set(flat)) == len(flat),
               f"{count} shard(s): no TU is owned twice")
        # Balance, which is the whole reason this is a stride and not a hash.
        # md5-modulo over these same 72 files at 25 shards left one shard with
        # nothing and another with seven (run 34535747572); the critical path
        # is the largest slice and an empty slice is a wasted Windows runner.
        sizes = [len(o) for o in owners]
        _check(failures, max(sizes) - min(sizes) <= 1,
               f"{count} shard(s): slices differ by at most one "
               f"(min {min(sizes)}, max {max(sizes)})")
        if count <= len(files):
            _check(failures, min(sizes) > 0,
                   f"{count} shard(s): no shard draws an empty slice")

    _check(failures, bw.slice_for(files, 0, 1) == files,
           "with one shard, index 0 owns everything")
    # More shards than TUs is the one case where a slice can legitimately be
    # empty, and partition_sources refuses it rather than running the job.
    _check(failures, bw.slice_for(files, 90, 100) == [],
           "a shard beyond the TU count draws nothing (refused upstream)")
    return failures


def stats_checks() -> list:
    """ccache's counters, and the two that look like the ones we want.

    `direct_cache_miss` on a TU that then hit the preprocessed cache is a HIT.
    A parser that summed every key containing "miss" would report a perfectly
    replayed link job as three misses and fail it; one that summed every key
    containing "hit" would count that TU twice and fail the "one lookup per
    TU" check instead. Both are silent misreadings of a passing build, so both
    get a control here.
    """
    bw = load()
    failures: list = []
    text = "\n".join([
        "cache_size_kibibyte\t123456",
        "direct_cache_hit\t70",
        "direct_cache_miss\t2",
        "preprocessed_cache_hit\t2",
        "preprocessed_cache_miss\t0",
        "cache_miss\t0",
    ])
    hits, misses = bw.parse_ccache_stats(text)
    _check(failures, (hits, misses) == (72, 0),
           f"72 hits / 0 misses read from a table that also carries "
           f"direct_cache_miss (got {hits}/{misses})")
    hits, misses = bw.parse_ccache_stats("cache_miss\t3\ndirect_cache_hit\t69")
    _check(failures, (hits, misses) == (69, 3),
           f"a real miss is counted (got {hits}/{misses})")
    return failures


def stub_checks() -> list:
    """The shard stub, and the one thing in it that cannot be eyeballed.

    A shard stubs the C++ translation unit that carries PYBIND11_MODULE, which
    is where PyInit_<name> comes from -- and distutils hands link.exe
    /EXPORT:PyInit_<name> regardless. If the stub's token paste is wrong the
    symbol is named something else, the export is unresolved, and every shard
    dies at link with LNK2001. The paste is testable without MSVC: run any
    preprocessor over it with the same -D the compile line carries.
    """
    bw = load()
    failures: list = []
    _check(failures, "PYBIND11_MODULE" not in bw._EMPTY_STUB,
           "the empty stub does not itself look like a module TU")

    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        print("skip no C preprocessor available; PyInit paste unchecked")
        return failures
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "stub.c"
        # __declspec is MSVC-only; the paste is what is under test, so it is
        # defined away rather than the test being skipped off-Windows.
        src.write_text("#define __declspec(x)\n" + bw._PYINIT_STUB)
        out = subprocess.run([cc, "-E", "-DTORCH_EXTENSION_NAME=flash_attn_2_cuda",
                              str(src)], capture_output=True, text=True).stdout
        _check(failures, "PyInit_flash_attn_2_cuda" in out,
               "the stub pastes TORCH_EXTENSION_NAME into the PyInit symbol")
        out2 = subprocess.run([cc, "-E", str(src)],
                              capture_output=True, text=True).stdout
        _check(failures, "PyInit_" not in out2,
               "and defines nothing when TORCH_EXTENSION_NAME is absent")

        # The control that matters most, and the one a preprocessor pass
        # cannot give: the file being stubbed is a C++ translation unit, so a
        # definition without extern "C" is MANGLED while /EXPORT: asks for the
        # plain name. Compiled as C++ and read back with nm, the symbol has to
        # be there verbatim.
        cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        nm = shutil.which("nm")
        if cxx and nm:
            cpp = Path(td) / "stub.cpp"
            cpp.write_text("#define __declspec(x)\n" + bw._PYINIT_STUB)
            obj = Path(td) / "stub.o"
            rc = subprocess.run([cxx, "-c", "-DTORCH_EXTENSION_NAME=flash_attn_2_cuda",
                                 str(cpp), "-o", str(obj)],
                                capture_output=True, text=True)
            _check(failures, rc.returncode == 0,
                   f"the stub compiles as C++ ({rc.stderr.strip()[:120]})")
            if rc.returncode == 0:
                syms = subprocess.run([nm, str(obj)], capture_output=True,
                                      text=True).stdout
                _check(failures, " T PyInit_flash_attn_2_cuda" in syms
                       or " T _PyInit_flash_attn_2_cuda" in syms,
                       "compiled as C++, the entry point is UNMANGLED "
                       "(extern \"C\" is doing its job)")
        else:
            print("skip no C++ compiler or nm; mangling control unchecked")
    return failures


if __name__ == "__main__":
    sys.exit(main())
