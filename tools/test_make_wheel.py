#!/usr/bin/env python3
"""Negative controls for the wheel half: make_wheel.py and verify_wheel.py.

  * the abi3 mis-tag (torchao: cp39-abi3 wheel shipping a
    mxfp8_cuda.cpython-312-*.so) -- make_wheel retags, verify_wheel fails
    a wheel that was not retagged;
  * compiler provenance on the raw wheel's modules (pyg-lib's Ubuntu g++);
  * the conda -> PyPI name translation: every mapped case, the CUDA header
    packages per CUDA major, sidecar_omit, and the hard error for a name in
    no table;
  * win-64: the PE linker version gate.

Run: python tools/test_make_wheel.py
Needs: cc, objcopy, readelf, patchelf (fixtures), like test_verify_conda.py.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_wheel as mw  # noqa: E402
import verify_wheel as vw  # noqa: E402
from test_verify_conda import CF_GCC, SYSROOT, UBUNTU, build_pe, compile_so  # noqa: E402

failures: list[str] = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def make_wheel_file(path: Path, files: dict[str, bytes | Path], tags: list[str],
                    name="pkg", version="1.0") -> Path:
    di = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as z:
        for rel, content in files.items():
            z.writestr(rel, content.read_bytes() if isinstance(content, Path) else content)
        z.writestr(f"{di}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                                     f"Requires-Dist: upstream-junk\n\nbody\n")
        z.writestr(f"{di}/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\n"
                                  + "".join(f"Tag: {t}\n" for t in tags))
        z.writestr(f"{di}/RECORD", "")
    return path


def expect_exit(fn, needle: str, msg: str):
    try:
        fn()
    except SystemExit as e:
        check(needle in str(e), f"{msg} (exit message: {str(e)[:90]!r})")
        return
    check(False, f"{msg} -- did not exit")


def main() -> int:
    td = Path(tempfile.mkdtemp(prefix="cuw-wheel-test-"))

    # ---- name translation ---------------------------------------------------
    t = mw.conda_spec_to_pep508
    check(t("cuda-cudart-dev >=12.8", "12.8") == "nvidia-cuda-runtime-cu12>=12.8",
          "cuda-cudart-dev -> nvidia-cuda-runtime-cu12 on a 12.x cell")
    check(t("cuda-cccl", "12.9") == "nvidia-cuda-cccl-cu12", "cuda-cccl -> nvidia-cuda-cccl-cu12")
    check(t("cuda-nvcc-tools", "12.8") == "nvidia-cuda-nvcc-cu12", "cuda-nvcc-tools -> nvidia-cuda-nvcc-cu12")
    check(t("cuda-nvrtc >=12.8.93", "12.8") == "nvidia-cuda-nvrtc-cu12>=12.8.93",
          "cuda-nvrtc -> nvidia-cuda-nvrtc-cu12, constraint kept")
    check(t("cuda-cudart-dev", "13.0") == "nvidia-cuda-runtime",
          "on a 13.x cell the PyPI name is unsuffixed (nvidia-cuda-runtime)")
    check(t("py-opencv >=3", "12.8") == "opencv-python>=3", "py-opencv -> opencv-python")
    check(t("matplotlib-base", "12.8") == "matplotlib", "matplotlib-base -> matplotlib")
    check(t("pytorch 2.8.* cuda128_*", "12.8") is None, "pytorch is never advertised")
    check(t("__cuda", "12.8") is None, "virtual packages are never advertised")
    check(t("libstdcxx >=13", "12.8") is None, "conda-only runtimes are never advertised")
    check(t("numpy >=1.25,<3", "12.8") == "numpy>=1.25,<3", "a same-name dep keeps its constraint")
    check(t("cumm >=0.7.11,<0.8.0 cuda128_*", "12.8") == "cumm>=0.7.11,<0.8.0",
          "a sibling package's build glob is dropped, the constraint kept")
    check(t("flex-gemm", "12.8") == "flex_gemm", "a sibling package maps to its pypi_name")
    check(t("frobnicator", "12.8", sidecar_omit={"frobnicator"}) is None,
          "a name in the package's sidecar_omit list is dropped")
    expect_exit(lambda: t("frobnicator", "12.8"), "no PyPI translation",
                "a conda name in NO table is a hard error, not a pass-through")
    check(mw.pypi_name_for("frobnicator", "12.8", strict=False) == "frobnicator",
          "verify_wheel's lenient lookup passes unknown names through for comparison")

    # ---- a torch-minor clause lands in the sidecar for the cell it names ----
    # torchvision declares torchvision-extra-decoders for torch >=2.7,<2.14
    # on linux; the sidecar written for a 2.8 cell carries it and one for a
    # 2.4 cell does not, and the loader refuses to resolve without a minor.
    sys.path.insert(0, str(HERE.parent / "scripts"))
    import package_loader as pl
    deps = [{"if": 'linux and match(pytorch, ">=2.7,<2.14")', "then": "torchvision-extra-decoders"},
            "numpy"]
    at28 = [mw.conda_spec_to_pep508(d, "12.8") for d in pl.resolve_run_deps(deps, "linux-64", pytorch="2.8")]
    at24 = [mw.conda_spec_to_pep508(d, "12.8") for d in pl.resolve_run_deps(deps, "linux-64", pytorch="2.4")]
    atwin = [mw.conda_spec_to_pep508(d, "12.8") for d in pl.resolve_run_deps(deps, "win-64", pytorch="2.8")]
    check(at28 == ["torchvision-extra-decoders", "numpy"], f"torch 2.8 linux sidecar carries the clause ({at28})")
    check(at24 == ["numpy"], f"torch 2.4 linux sidecar does not ({at24})")
    check(atwin == ["numpy"], f"win-64 sidecar does not ({atwin})")
    try:
        pl.resolve_run_deps(deps, "linux-64")
        check(False, "resolving a torch clause without a minor must raise")
    except ValueError:
        check(True, "resolving a torch clause without a minor raises rather than dropping the dep")

    # ---- abi retag -----------------------------------------------------------
    check(mw.cpython_abi_of_modules(["torchao/_C.abi3.so",
                                     "torchao/prototype/mxfp8_cuda.cpython-312-x86_64-linux-gnu.so"]) == "cp312",
          "torchao's module list demands cp312")
    check(mw.cpython_abi_of_modules(["torchao/_C.abi3.so"]) is None, "an all-abi3 wheel demands nothing")
    check(mw.cpython_abi_of_modules(["fx/_C.cp312-win_amd64.pyd"]) == "cp312", "win-64 spelling is recognised")
    expect_exit(lambda: mw.cpython_abi_of_modules(["a.cpython-312-x86_64-linux-gnu.so",
                                                    "b.cpython-311-x86_64-linux-gnu.so"]),
                "several CPython ABIs", "two interpreters in one wheel is an error")

    so_cf = td / "cf.so"
    compile_so(so_cf, comments=(CF_GCC, SYSROOT), rpath=None)
    raw = make_wheel_file(td / "pkg-1.0-cp39-abi3-manylinux_2_28_x86_64.whl",
                          {"pkg/_C.abi3.so": so_cf,
                           "pkg/prototype/mx.cpython-312-x86_64-linux-gnu.so": so_cf,
                           "pkg/__init__.py": b""}, ["cp39-abi3-manylinux_2_28_x86_64"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out, deps_out, side, n = mw.finalize(raw, "+cu128torch2.8", "8.0 9.0", ["numpy", "cuda-nvrtc"],
                                             expect_version="1.0", build_number="0", cuda="12.8")
    check(out.name == "pkg-1.0+cu128torch2.8-0-cp312-cp312-manylinux_2_28_x86_64.whl",
          f"make_wheel retags cp39-abi3 -> cp312-cp312 (got {out.name})")
    with zipfile.ZipFile(out) as z:
        wheel_txt = z.read("pkg-1.0+cu128torch2.8.dist-info/WHEEL").decode()
        record = z.read("pkg-1.0+cu128torch2.8.dist-info/RECORD").decode()
        root_meta = z.read("pkg-1.0+cu128torch2.8.dist-info/METADATA").decode()
    check("Tag: cp312-cp312-manylinux_2_28_x86_64" in wheel_txt and "abi3" not in wheel_txt,
          "...and rewrites the WHEEL Tag line")
    check("dist-info/WHEEL,sha256=" in record, "...and RECORD hashes the rewritten WHEEL")
    check("Requires-Dist" not in root_meta and not out.with_name(out.name + ".metadata").exists(),
          "the root wheel carries no Requires-Dist and no sidecar")
    # ---- the /deps/ twin ----------------------------------------------------
    check(deps_out.name == out.name and deps_out.parent.name == "deps",
          f"the deps twin has the SAME filename under deps/ ({deps_out})")
    with zipfile.ZipFile(deps_out) as z:
        deps_meta = z.read("pkg-1.0+cu128torch2.8.dist-info/METADATA")
        deps_names = set(z.namelist())
    check(b"Requires-Dist: nvidia-cuda-nvrtc-cu12" in deps_meta and n == 2,
          "the deps twin's METADATA carries the translated CUDA package")
    check(side.read_bytes() == deps_meta, "the deps twin's sidecar is byte-identical to its METADATA")
    with zipfile.ZipFile(out) as zr, zipfile.ZipFile(deps_out) as zd:
        same = all(zr.read(nm) == zd.read(nm) for nm in deps_names
                   if not nm.endswith((".dist-info/METADATA", ".dist-info/RECORD")))
    check(same and deps_names == set(zipfile.ZipFile(out).namelist()),
          "the deps twin differs from the root wheel in METADATA and RECORD only")
    dv, ds = mw.deps_variant_of(out, side, td / "backfill")
    with zipfile.ZipFile(dv) as z:
        check(z.read("pkg-1.0+cu128torch2.8.dist-info/METADATA") == ds.read_bytes() == deps_meta,
              "deps_variant_of() reproduces the twin from a published root wheel + old sidecar")

    raw2 = make_wheel_file(td / "pkg-1.0-cp311-cp311-manylinux_2_28_x86_64.whl",
                           {"pkg/mx.cpython-312-x86_64-linux-gnu.so": so_cf}, ["cp311-cp311-manylinux_2_28_x86_64"])
    with contextlib.redirect_stdout(io.StringIO()):
        expect_exit(lambda: mw.finalize(raw2, "+cu128torch2.8", "", [], expect_version="1.0", cuda="12.8"),
                    "wrong interpreter", "a cp311 wheel shipping a cp312 module is not a retag case")

    # ---- compiler provenance on the raw wheel ------------------------------
    so_ub = td / "ub.so"
    compile_so(so_ub, comments=(CF_GCC, UBUNTU, SYSROOT), rpath=None)
    good_raw = make_wheel_file(td / "good-1.0-cp312-cp312-linux_x86_64.whl", {"good/_C.so": so_cf},
                               ["cp312-cp312-linux_x86_64"], name="good")
    check(len(mw.check_compiler_provenance(good_raw)) == 1, "a conda-forge-compiled module passes provenance")
    bad_raw = make_wheel_file(td / "bad-1.0-cp312-cp312-linux_x86_64.whl", {"bad/_C.so": so_ub},
                              ["cp312-cp312-linux_x86_64"], name="bad")
    expect_exit(lambda: mw.check_compiler_provenance(bad_raw), "not compiled by conda-forge's gcc",
                "pyg-lib's case: an Ubuntu gcc entry in a raw wheel module refuses the repair")

    # ---- verify_wheel gates ------------------------------------------------
    def vargs(**kw):
        a = SimpleNamespace(platform="linux-64", links_torch=False, expect_arch="", expect_version="",
                            conda=None, no_sass=None, cuda="12.8", expect_msvc="", deps_wheel=None)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def run_vw(path, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = vw.verify(path, vargs(**kw), td)
        return ok, buf.getvalue()

    def fails_on(out, needle):
        return any(ln.startswith("FAIL") and needle in ln for ln in out.splitlines())

    def finalized(name: str, files: dict, tags: list[str], retag_expected=True) -> Path:
        """A raw fixture wheel taken through make_wheel.finalize, as CI does."""
        raw = make_wheel_file(td / f"{name}-1.0-{tags[0]}.whl", files, tags, name=name)
        with contextlib.redirect_stdout(io.StringIO()):
            out, _, _, _ = mw.finalize(raw, "+cu128torch2.8", "8.0 9.0", ["numpy"],
                                       expect_version="1.0", build_number="0", cuda="12.8")
        return out

    # An abi3 mis-tag that got PAST make_wheel: build the wheel with the
    # cp312 module renamed so finalize does not retag, then rename the module
    # back inside the zip -- the artifact verify_wheel must refuse.
    fin = finalized("mis", {"mis/mx.so": so_cf}, ["cp39-abi3-manylinux_2_28_x86_64"])
    mis = td / fin.name
    with zipfile.ZipFile(fin) as zin, zipfile.ZipFile(td / "mis.tmp", "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "mis/mx.so":
                item.filename = "mis/mx.cpython-312-x86_64-linux-gnu.so"
            zout.writestr(item, data)
    os.replace(td / "mis.tmp", mis)      # same name; the sidecar beside it still applies
    ok, out = run_vw(mis)
    check(not ok and fails_on(out, "wheel tag is cp312-cp312"),
          "verify_wheel: an abi3-tagged wheel shipping a cp312 module FAILS")

    ub = finalized("ub", {"ub/_C.so": so_ub}, ["cp312-cp312-manylinux_2_28_x86_64"])
    ok, out = run_vw(ub)
    check(not ok and fails_on(out, "compiled by conda-forge's gcc alone"),
          "verify_wheel: a module carrying an Ubuntu gcc entry FAILS")

    okw = finalized("okw", {"okw/_C.so": so_cf}, ["cp312-cp312-manylinux_2_28_x86_64"])
    ok, out = run_vw(okw)
    check(ok, "verify_wheel positive control PASSES")
    if not ok:
        print(out)
    # the old scheme: a sidecar beside the root wheel that differs from it
    stale = okw.with_name(okw.name + ".metadata")
    stale.write_text("Metadata-Version: 2.1\nName: okw\nVersion: 1.0+cu128torch2.8\nRequires-Dist: numpy\n")
    ok, out = run_vw(okw)
    check(not ok and fails_on(out, "no sidecar beside the root wheel"),
          "verify_wheel: a sidecar beside the ROOT wheel (the old scheme) FAILS")
    stale.unlink()
    # a deps twin whose sidecar drifted from its METADATA (what pip refuses)
    twin_side = okw.parent / "deps" / (okw.name + ".metadata")
    orig = twin_side.read_bytes()
    twin_side.write_bytes(orig + b"Requires-Dist: pillow\n")
    ok, out = run_vw(okw)
    check(not ok and fails_on(out, "byte-identical"),
          "verify_wheel: a deps-twin sidecar that differs from the twin's METADATA FAILS")
    twin_side.write_bytes(orig)
    # a missing twin
    twin = okw.parent / "deps" / okw.name
    twin.rename(twin.with_name("moved.whl"))
    ok, out = run_vw(okw)
    check(not ok and fails_on(out, "twin exists"), "verify_wheel: a missing deps twin FAILS")
    twin.with_name("moved.whl").rename(twin)
    # a twin whose payload differs from the root wheel
    with zipfile.ZipFile(twin) as zin, zipfile.ZipFile(td / "twin.tmp", "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            zout.writestr(item, data + (b"x" if item.filename == "okw/_C.so" else b""))
    os.replace(td / "twin.tmp", twin)
    ok, out = run_vw(okw)
    check(not ok and fails_on(out, "METADATA/RECORD only"),
          "verify_wheel: a deps twin whose payload differs from the root wheel FAILS")

    pyd_ok, pyd_bad = td / "ok.pyd", td / "bad.pyd"
    build_pe(pyd_ok, ["KERNEL32.dll", "c10.dll"], linker=(14, 44))
    build_pe(pyd_bad, ["KERNEL32.dll", "c10.dll"], linker=(2, 42))
    for label, pyd, want_ok in (("okwin", pyd_ok, True), ("badwin", pyd_bad, False)):
        w = finalized(label, {f"{label}/_C.cp312-win_amd64.pyd": pyd}, ["cp312-cp312-win_amd64"])
        ok, out = run_vw(w, platform="win-64", links_torch=True, expect_msvc="vs2022")
        if want_ok:
            check(ok, "verify_wheel win-64 positive control PASSES")
            if not ok:
                print(out)
        else:
            check(not ok and fails_on(out, "linked by the conda MSVC"),
                  "verify_wheel win-64: a non-MSVC linker version FAILS")

    shutil.rmtree(td, ignore_errors=True)
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
