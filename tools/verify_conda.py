#!/usr/bin/env python3
"""Per-artifact publish gate for a built extension package.

Nothing here is novel: it is cuda-wheels' verify_wheel.py checks (filename,
binary census, ELF sanity, torch linkage, glibc ceiling, SASS archs, import)
re-expressed for .conda, plus conda-torch's packaging invariants (paths.json
matches payload, no RECORD, INSTALLER=conda, $ORIGIN-relative RPATHs), plus
the three this repo adds:

  * the compile ledger covers every extension module shipped  (L3 of the
    from-source guarantee: an artifact must not contain a binary this build
    did not compile);
  * provenance says built_from_source and NOT prebuilt_wheel_used;
  * the torch dependency carries a flavour build-glob, without which the
    solver may pair a cu128 extension with a cu130 torch.

Usage: verify_conda.py <pkg.conda> [...] [--ledger FILE] [--expect-arch "8.0 9.0"]
Exit: 0 all pass, 1 otherwise.
"""

import argparse
import io
import json
import re
import struct
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

BUILD_RE = re.compile(
    r"^cuda(?P<cu>\d+)_(?:torch(?P<torch>\d+)_)?py(?P<py>\d+)_h[0-9a-f]+_(?P<n>\d+)$")


class Report:
    def __init__(self, artifact):
        self.artifact, self.failed = artifact, False

    def ok(self, msg):
        print(f"ok   {msg}")

    def bad(self, msg):
        print(f"FAIL {msg}")
        self.failed = True

    def check(self, cond, msg):
        (self.ok if cond else self.bad)(msg)
        return cond


def read_conda(path: Path):
    """(index.json, about.json, paths.json, {payload member -> bytes-or-None})."""
    zf = zipfile.ZipFile(path)
    info_n = [n for n in zf.namelist() if n.startswith("info-") and n.endswith(".tar.zst")]
    pkg_n = [n for n in zf.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst")]
    if len(info_n) != 1 or len(pkg_n) != 1:
        sys.exit(f"{path.name}: expected one info- and one pkg- member, "
                 f"found {info_n} / {pkg_n}")

    def untar(member):
        raw = subprocess.run(["zstd", "-d", "--stdout"], input=zf.read(member),
                             capture_output=True, check=True).stdout
        return tarfile.open(fileobj=io.BytesIO(raw))

    itf = untar(info_n[0])

    def jload(name, default=None):
        try:
            m = itf.extractfile(name)
            return json.load(m) if m else default
        except KeyError:
            return default

    return (jload("info/index.json", {}), jload("info/about.json", {}),
            jload("info/paths.json", {"paths": []}), itf, untar(pkg_n[0]))


def elf_dynamic(data: bytes, tmp: Path, name: str):
    """readelf -d output for an in-memory ELF, or None if not an ELF."""
    if data[:4] != b"\x7fELF":
        return None
    p = tmp / name.replace("/", "_")
    p.write_bytes(data)
    return subprocess.run(["readelf", "-d", str(p)], capture_output=True,
                          text=True).stdout


def pe_imports(data: bytes) -> set[str] | None:
    """DLL names in a PE's import and delay-import directories.

    None when the bytes are not a PE, so callers can tell "not applicable" from
    "applicable and empty" -- the distinction the ELF path gets for free by
    checking the \\x7fELF magic.
    """
    if len(data) < 0x40 or data[:2] != b"MZ":
        return None
    (e_lfanew,) = struct.unpack_from("<I", data, 0x3C)
    if len(data) < e_lfanew + 24 or data[e_lfanew:e_lfanew + 4] != b"PE\0\0":
        return None

    coff = e_lfanew + 4
    n_sections, = struct.unpack_from("<H", data, coff + 2)
    size_opt, = struct.unpack_from("<H", data, coff + 16)
    opt = coff + 20
    magic, = struct.unpack_from("<H", data, opt)
    if magic == 0x20B:      # PE32+
        dd = opt + 112
    elif magic == 0x10B:    # PE32
        dd = opt + 96
    else:
        return None
    n_dd, = struct.unpack_from("<I", data, dd - 4)

    sections = []
    sec = opt + size_opt
    for i in range(n_sections):
        off = sec + i * 40
        if off + 40 > len(data):
            break
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, off + 8)
        sections.append((vaddr, max(vsize, rawsize), rawptr))

    def to_off(rva):
        for vaddr, span, rawptr in sections:
            if vaddr <= rva < vaddr + span:
                return rawptr + (rva - vaddr)
        return None

    def cstr(rva):
        off = to_off(rva)
        if off is None or off >= len(data):
            return None
        end = data.find(b"\0", off)
        return data[off:end if end != -1 else len(data)].decode("ascii", "replace")

    names: set[str] = set()

    # Directory 1: the ordinary import table. 20-byte descriptors, DLL name RVA
    # at +12, terminated by an all-zero entry.
    for index, name_off, stride in ((1, 12, 20), (13, 4, 32)):
        if index >= n_dd:
            continue
        rva, size = struct.unpack_from("<II", data, dd + index * 8)
        if not rva:
            continue
        base = to_off(rva)
        if base is None:
            continue
        for i in range(1024):
            ent = base + i * stride
            if ent + stride > len(data):
                break
            if data[ent:ent + stride] == b"\0" * stride:
                break
            name_rva, = struct.unpack_from("<I", data, ent + name_off)
            if not name_rva:
                continue
            # Delay-load descriptors written by older linkers store a virtual
            # ADDRESS rather than an RVA; a name that does not resolve inside a
            # section is that case, and is skipped rather than guessed at.
            nm = cstr(name_rva)
            if nm:
                names.add(nm)
    return names


def _verify_field(pkg_name: str, key: str):
    """`verify.<key>` from the package's own package.yml, raw, or None."""
    if not pkg_name:
        return None
    cfg = Path(__file__).resolve().parent.parent / "packages" / pkg_name / "package.yml"
    if not cfg.is_file():
        return None
    try:
        import yaml
    except ImportError:
        return None
    data = yaml.safe_load(cfg.read_text()) or {}
    return (data.get("verify") or {}).get(key)


def _expect_linked(pkg_name: str, key: str = "expect_linked") -> list:
    """`verify.<key>` from the package's own package.yml, if present.

    Read from packages/ rather than passed on the command line so the
    expectation lives beside the host_deps that are supposed to satisfy it,
    and so no caller can forget to pass it.

    `key` selects the platform's list: `expect_linked` names ELF sonames,
    `expect_linked_win` names DLLs. They are separate lists rather than one
    list plus a translation because there is no reliable mapping -- libjpeg
    is jpeg8.dll, libpng is libpng16.dll, nvjpeg is nvjpeg64_12.dll -- and a
    guessed mapping that silently matched nothing would turn this gate into
    decoration.
    """
    if not pkg_name:
        return []
    cfg = Path(__file__).resolve().parent.parent / "packages" / pkg_name / "package.yml"
    if not cfg.is_file():
        return []
    try:
        import yaml
    except ImportError:
        return []
    data = yaml.safe_load(cfg.read_text()) or {}
    return list((data.get("verify") or {}).get(key) or [])


def verify(path: Path, ledger: set, expect_arch: str, tmp: Path) -> bool:
    rep = Report(path.name)
    print(f"\n=== {path.name} ===")
    index, about, paths, itf, ptf = read_conda(path)

    # ---- filename / build string agree with the cell it claims -------------
    stem = path.name[: -len(".conda")]
    expected = f"{index.get('name')}-{index.get('version')}-{index.get('build')}"
    rep.check(stem == expected,
              f"filename matches index.json ({stem} vs {expected})")
    m = BUILD_RE.match(str(index.get("build", "")))
    rep.check(bool(m), f"build string parses as a cell: {index.get('build')!r}")
    rep.check("+" not in str(index.get("version", "")),
              "version carries no '+' local tag (conda sorts those BELOW the plain version)")

    # ---- paths.json matches the payload ------------------------------------
    payload = {mem.name for mem in ptf.getmembers() if mem.isfile() or mem.issym()}
    declared = {p["_path"] for p in paths.get("paths", [])}
    rep.check(declared == payload,
              f"paths.json matches payload (declared {len(declared)}, payload {len(payload)})")

    # ---- dist-info hygiene (conda-torch's lessons) -------------------------
    rec = [n for n in payload if n.endswith(".dist-info/RECORD")]
    rep.check(not rec, "no RECORD in dist-info (pip uninstall would delete conda's files)")
    durl = [n for n in payload if n.endswith("direct_url.json")]
    rep.check(not durl, "no direct_url.json (it poisons pip freeze with a build path)")
    inst = [n for n in payload if n.endswith(".dist-info/INSTALLER")]
    if inst:
        # rattler-build 0.75 normalises this file during packaging, appending a
        # trailing newline: a recipe writing exactly b"conda" still ships
        # b"conda\n", and there is no knob to stop it. A byte-exact assert here
        # is unsatisfiable, so compare the content, not the trailing whitespace.
        val = ptf.extractfile(inst[0]).read()
        rep.check(val.strip() == b"conda", f"INSTALLER names conda (got {val!r})")

    # ---- dependencies: non-empty, and the torch flavour lock ---------------
    depends = index.get("depends", [])
    rep.check(bool(depends), f"run deps are non-empty ({len(depends)} entries)")
    torch_dep = [d for d in depends if d.split(" ", 1)[0] == "pytorch"]
    if m and m.group("torch"):
        rep.check(bool(torch_dep), "declares a pytorch dependency")
        glob_ok = any(len(d.split()) >= 3 and f"cuda{m.group('cu')}_" in d.split()[2]
                      for d in torch_dep)
        rep.check(glob_ok,
                  f"pytorch dep carries the cuda{m.group('cu')}_* build glob "
                  f"(got {torch_dep!r}) -- the torch we build against exports a "
                  f"flavour-blind run_export, so without this a "
                  f"cu{m.group('cu')} build can pair with another flavour")

    # ---- interpreter ABI ----------------------------------------------------
    # A build string claiming py312 must carry a matching python_abi run dep.
    # A bare `python` bound lets a cp312 extension module install into py3.10;
    # for torch-linked packages pytorch's own python_abi masks that
    # transitively, which is exactly why it must be asserted here rather than
    # assumed -- the mask is absent for any package that does not link torch.
    if m and m.group("py"):
        pytag = m.group("py")                      # e.g. "312"
        want = f"{pytag[0]}.{pytag[1:]}"           # "3.12"
        abi = [d for d in depends if d.split(" ", 1)[0] == "python_abi"]
        rep.check(bool(abi), f"declares a python_abi dependency (build says py{pytag})")
        rep.check(any(d.split()[1].startswith(want) for d in abi if len(d.split()) >= 2),
                  f"python_abi pins {want} to match the build string (got {abi!r})")

    # ---- states its own GPU requirement -------------------------------------
    # Inheriting __cuda through libtorch is enough for the solver, but a
    # consumer that wants to know whether a package needs a GPU should not have
    # to walk a dependency closure to find out -- comfy-test's accelerator lint
    # asks exactly this, on a bare checkout with nothing installed.
    rep.check(any(d.split(" ", 1)[0] == "__cuda" for d in depends),
              "declares __cuda directly (GPU requirement readable without a closure walk)")

    # ---- no build-toolchain passengers -------------------------------------
    # rattler-build ships whatever appeared in $PREFIX during the build, and
    # build.sh mutates $PREFIX on purpose: it moves the real nvcc aside to put
    # a ccache wrapper in its seat. `bin/nvcc.real` is then a NEW file, so it
    # was packaged -- a 27.5 MB CUDA compiler inside torchvision, declared in
    # paths.json, in every artifact this repo had built. build.sh now restores
    # the prefix on exit; this asserts it, because the failure is completely
    # silent otherwise (the package installs, imports and works).
    STOWAWAY = re.compile(
        r"(^|/)(nvcc|cicc|cudafe\+\+|ptxas|nvlink|fatbinary|nvdisasm|cuobjdump"
        r"|ninja|ccache|patchelf|cc1|cc1plus|ld|as)(\.real)?$")
    stowaways = sorted(n for n in payload if STOWAWAY.search(n))
    rep.check(not stowaways,
              f"ships no build-toolchain binaries ({stowaways[:3]})")

    # ---- no vendored torch --------------------------------------------------
    vendored_torch = [n for n in payload
                      if re.search(r"(^|/)(lib)?torch(_cpu|_cuda|_python)?\.(so|dll)", n)]
    rep.check(not vendored_torch,
              f"ships no vendored torch libraries ({vendored_torch[:3]})")

    # ---- ELF sanity: $ORIGIN-relative RPATHs, no absolute/empty entries -----
    exts, bad_rpath, checked = [], [], 0
    for mem in ptf.getmembers():
        if not mem.isfile() or not re.search(r"\.(so|so\.\d+|pyd)$", mem.name):
            continue
        data = ptf.extractfile(mem).read()
        exts.append(mem.name)
        dyn = elf_dynamic(data, tmp, mem.name)
        if dyn is None:
            continue
        checked += 1
        for line in dyn.splitlines():
            if "RPATH" in line or "RUNPATH" in line:
                val = line.split("[", 1)[-1].rstrip("]").strip()
                for entry in val.split(":"):
                    if entry == "" or entry.startswith("/"):
                        bad_rpath.append((mem.name, val))
    rep.check(bool(exts), f"ships compiled extension modules ({len(exts)})")
    rep.check(not bad_rpath,
              f"RPATH lint clean over {checked} ELFs "
              f"(absolute or empty entries: {bad_rpath[:2]})")

    # ---- declared linkage must be REAL --------------------------------------
    # A codec that fails to be detected does not fail the build: torchvision
    # prints a warning, ships an image extension linking libpng alone, still
    # imports, and still carries libjpeg-turbo/libwebp/libnvjpeg in `depends`
    # via run_exports -- so the metadata claims codecs the binary does not
    # have and decode_jpeg raises only on the user's machine. conda-forge hit
    # the same trap and patched setup.py to raise instead of warn
    # (torchvision-feedstock, 0002-Force-nvjpeg-and-force-failure.patch).
    # Asserting the DT_NEEDED here is the same fail-closed idea one level out:
    # it needs no patch per upstream version, and unlike the GPU verify op it
    # runs on a CI box with no GPU, which is where the fan-out happens.
    expect_linked = _expect_linked(index.get("name", ""))
    if expect_linked and exts:
        needed, is_pe = set(), False
        for mem in ptf.getmembers():
            if not mem.isfile() or not re.search(r"\.(so|so\.\d+|pyd|dll)$", mem.name):
                continue
            blob = ptf.extractfile(mem).read()
            # DT_NEEDED is an ELF concept. On win-64 the equivalent statement
            # lives in the PE import directory, and asking readelf about a .pyd
            # returns nothing -- which this gate previously read as "links
            # none of them" and failed on, reporting NEEDED=[] for a binary
            # whose imports it had never looked at.
            imports = pe_imports(blob)
            if imports is not None:
                is_pe, _ = True, needed.update(imports)
                continue
            dyn = elf_dynamic(blob, tmp, mem.name)
            for line in (dyn or "").splitlines():
                m2 = re.search(r"Shared library: \[([^\]]+)\]", line)
                if m2:
                    needed.add(m2.group(1))

        if is_pe:
            # The expectation itself is platform-specific: `libjpeg` is an ELF
            # soname and the same library is `jpeg8.dll` here, so matching the
            # Linux list against PE imports would be a guess at a naming
            # convention. A package states the Windows names itself, or this
            # gate says plainly that it cannot judge -- it does not invent a
            # mapping and then report confidence in it.
            expect_win = _expect_linked(index.get("name", ""), key="expect_linked_win")
            # FAILS when the Windows list is absent, and this was a warning
            # first -- which is how run 34169055830 published a torchvision
            # win-64 artifact whose .pyd imports libpng16.dll and NOTHING for
            # jpeg, webp or nvjpeg. That is the exact trap this gate exists to
            # catch, and warning about it let it through. A package that states
            # the expectation on one platform and cannot be checked on another
            # does not get to publish there.
            if not rep.check(bool(expect_win),
                             f"declares verify.expect_linked_win, without which the "
                             f"codec gate cannot run on win-64. Imports actually "
                             f"present: {sorted(needed)}"):
                pass
            else:
                low = {n.lower() for n in needed}
                missing = [w for w in expect_win
                           if not any(n.startswith(w.lower()) for n in low)]
                # The interesting imports, with the platform's own runtime
                # dropped. Every PE imports KERNEL32, the CRT and the
                # api-ms-win-* apisets, and they sort to the front -- so a
                # truncated list showed ten of those and none of the libraries
                # the check is actually about, which is the opposite of what a
                # failure message is for.
                interesting = sorted(
                    n for n in needed
                    if not n.lower().startswith(("kernel32", "msvcp", "vcruntime",
                                                 "api-ms-win-", "ucrtbase",
                                                 "advapi32", "user32"))
                )
                rep.check(not missing,
                          f"imports every DLL package.yml says it must "
                          f"(missing {missing}; non-system imports={interesting})")
        else:
            missing = [w for w in expect_linked
                       if not any(n.startswith(w) for n in needed)]
            rep.check(not missing,
                      f"links every library package.yml says it must "
                      f"(missing {missing}; NEEDED={sorted(needed)[:8]})")

    # ---- provenance: the from-source guarantee, recorded --------------------
    extra = about.get("extra") or {}
    rep.check(extra.get("built_from_source") is True,
              f"about.extra.built_from_source is true (got {extra.get('built_from_source')!r})")
    rep.check(extra.get("prebuilt_wheel_used") is False,
              f"about.extra.prebuilt_wheel_used is false (got {extra.get('prebuilt_wheel_used')!r})")
    rep.check(bool(extra.get("torch_build")),
              f"records the exact torch build it compiled against ({extra.get('torch_build')!r})")
    rev = str(extra.get("source_rev") or "")
    rep.check(bool(rev) and rev.lower() not in ("main", "master", "head"),
              f"records a non-floating source_rev ({rev!r})")

    # ---- L3: the compile ledger -------------------------------------------
    # What this CAN establish, and what it cannot.
    #
    # The ledger is a list of translation units the nvcc wrapper actually
    # compiled -- source paths like ".../torchvision/csrc/ops/cuda/nms_kernel.cu".
    # The shipped artifact contains linked MODULES, named for the extension
    # (_C.so, image.so). There is no general mapping between the two: _C.so is
    # linked from dozens of TUs and none of them is called "_C".
    #
    # This check used to compare those two name sets directly and require
    # every module stem to appear as a compiled file name. That can only pass
    # for a package whose TU happens to share its module's name, which is none
    # of them -- every artifact this repo has ever built fails it. It never
    # fired because the workflow calls verify_conda.py without --ledger, so
    # the whole branch was dead: L3 was documented as a publish gate, was
    # never run, and could not have passed if it were.
    #
    # So it asserts the two things the ledger genuinely proves:
    #   1. an artifact that ships compiled modules must have compiled
    #      something -- an empty ledger beside a .so means the binary came
    #      from somewhere this build did not look (a vendored blob, or a
    #      prebuilt wheel that L1 failed to stop);
    #   2. every TU came from THIS build's work tree, so nothing was compiled
    #      out of a system path or a foreign checkout.
    # Neither proves a particular .so was linked only from ledger TUs; that
    # needs link-line capture, which the wrapper does not do. Stated plainly
    # rather than implied by a check that looks stronger than it is.
    if ledger is not None and exts:
        rep.check(bool(ledger),
                  f"compile ledger is non-empty for an artifact shipping "
                  f"{len(exts)} extension module(s)")
        # Only ABSOLUTE paths can be judged. cmake invokes nvcc from a build
        # subdirectory inside the work tree and passes the source relatively
        # ("../../../../../src/libtorchaudio/cuctc/src/..."), so a relative
        # entry is by construction under the compiler's cwd, which is the
        # work tree. Rejecting those flagged torchaudio -- a package built
        # entirely from our own source -- so the check was wrong, not the
        # artifact. setuptools-driven builds pass absolute paths and are
        # still covered.
        def _absolute(x):
            # A Windows TU path is "C:\\...\\work\\..." and starts with a
            # drive letter, not "/". Testing startswith("/") alone made this
            # assertion inert on win-64: it judged nothing and still printed
            # ok, which is worse than not running.
            return x.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", x) is not None

        def _in_work(x):
            return "/work/" in x.replace("\\", "/")

        foreign = sorted(x for x in ledger if _absolute(x) and not _in_work(x))
        rep.check(not foreign,
                  f"no compiled TU came from outside the build work tree "
                  f"({len(ledger)} TUs; foreign: {foreign[:2]})")

    # ---- SASS arch census ---------------------------------------------------
    # The census is taken over the UNION of every shipped extension module,
    # not over the largest one. This used to inspect only the biggest .so,
    # which asks the wrong question of any package that splits its kernels
    # by architecture: sageattention builds _qattn_sm89 for sm_89/90/120 only
    # (its FP8 QMMA path is nothing but __brkpt() traps below Ada, so the
    # patch filters those gencodes out on purpose) beside _qattn_sm80 and
    # _fused carrying the whole list -- and the sm89 module is the largest
    # of the four. The cell's promise is "this ARTIFACT carries SASS for
    # every arch in the list"; a per-module reading of it fails an artifact
    # that keeps the promise. For a single-module package the two readings
    # are identical, so nothing is loosened there.
    if expect_arch and exts:
        want = {a.replace(".", "").replace("+PTX", "") for a in expect_arch.split()}
        got, per_module = set(), {}
        for n in exts:
            p = tmp / "sass.so"
            p.write_bytes(ptf.extractfile(n).read())
            out = subprocess.run(["cuobjdump", "--list-elf", str(p)],
                                 capture_output=True, text=True).stdout
            archs = set(re.findall(r"sm_(\d+)", out))
            if archs:
                per_module[n.rsplit("/", 1)[-1]] = sorted(archs)
            got |= archs
        # A package may declare `verify.no_sass: <reason>` -- cumm's core_cc
        # is C++ against the CUDA runtime and compiles its kernels through
        # NVRTC at run time, so it carries no device code by design. For such
        # a package "covers the arch list" has no meaning, and the question
        # becomes the opposite one: any SASS at all means the artifact is not
        # the one the declaration describes. Stated by the package, asserted
        # here, so the census never passes it by finding nothing to count.
        no_sass = _verify_field(index.get("name", ""), "no_sass")
        if no_sass:
            rep.check(not got,
                      f"ships no SASS, as package.yml declares ({str(no_sass).strip()[:60]}...); "
                      f"found {sorted(got)} in {per_module}")
        else:
            rep.check(want <= got or not got,
                      f"SASS archs cover the cell's arch list (want {sorted(want)}, "
                      f"got {sorted(got)} over {len(exts)} module(s): {per_module})")

    print(f"--- {path.name}: {'FAIL' if rep.failed else 'PASS'}")
    return not rep.failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", nargs="+", type=Path)
    ap.add_argument("--ledger", type=Path,
                    help="compile ledger written by the nvcc wrapper (one TU per line)")
    ap.add_argument("--expect-arch", default="", help="the cell's arch list")
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/verify-conda"))
    args = ap.parse_args()
    args.tmp.mkdir(parents=True, exist_ok=True)

    ledger = set()
    if args.ledger and args.ledger.is_file():
        ledger = {ln.strip() for ln in args.ledger.read_text().splitlines() if ln.strip()}

    allok = True
    for a in args.artifacts:
        allok &= verify(a, ledger, args.expect_arch, args.tmp)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
