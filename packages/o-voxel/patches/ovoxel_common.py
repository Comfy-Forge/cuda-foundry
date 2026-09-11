"""The edits shared by the three o-voxel builds (o-voxel, o-voxel-vb,
o-voxel-vb-ap). Each package's own patch script imports this by path and
calls the pieces that apply to its fork; nothing runs on import.

Every step is idempotent and asserts its own site count, so a fork that has
already carried an upstream fix reports "already applied" and a fork whose
source moved fails loudly instead of shipping a half-patched tree. The three
sources really do differ: the vb fork already has the size_t casts and the
double-literal fix, the ap fork also dropped postprocess.py, rasterize.py
and rasterize.cu -- so "apply the same patch" was never the right shape.
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import (EIGEN_3_4_0, hoist_subdir, require,  # noqa: E402
                       strip_std_flags, vendor_eigen)

SUBDIR = "o-voxel"


def hoist_and_vendor_eigen(tag: str) -> None:
    """o-voxel/ becomes the source root; Eigen 3.4.0 lands where setup.py
    looks (third_party/eigen -- upstream's gitlab submodule path)."""
    hoist_subdir(SUBDIR)
    require(pathlib.Path("setup.py").is_file() and pathlib.Path("pyproject.toml").is_file(),
            f"{tag}: after hoisting, setup.py/pyproject.toml are not at the root")
    require('os.path.join(ROOT, "third_party/eigen")' in pathlib.Path("setup.py").read_text(),
            f"{tag}: setup.py no longer points include_dirs at third_party/eigen")
    vendor_eigen("third_party/eigen", EIGEN_3_4_0)


def rename_package(tag: str, new_name: str) -> pathlib.Path:
    """o_voxel -> `new_name` in pyproject, setup.py, the package directory and
    every absolute self-import inside it. Returns the package directory."""
    def sub_once(path, old, new, what):
        p = pathlib.Path(path)
        t = p.read_text()
        n = t.count(old)
        if n == 0 and new in t:
            print(f"{tag} patch: {what}: already applied")
            return
        require(n == 1, f"{tag}: {what}: expected 1 site of {old!r} in {path}, "
                        f"found {n} -- upstream changed")
        p.write_text(t.replace(old, new))
        print(f"{tag} patch: {what}")

    sub_once("pyproject.toml", 'name = "o_voxel"', f'name = "{new_name}"', "pyproject name")
    sub_once("setup.py", 'name="o_voxel",', f'name="{new_name}",', "setup() name")
    sub_once("setup.py", 'name="o_voxel._C"', f'name="{new_name}._C"', "_C extension name")
    sub_once("setup.py", "'o_voxel',", f"'{new_name}',", "packages: o_voxel")
    sub_once("setup.py", "'o_voxel.convert',", f"'{new_name}.convert',", "packages: o_voxel.convert")
    sub_once("setup.py", "'o_voxel.io',", f"'{new_name}.io',", "packages: o_voxel.io")

    src, dst = pathlib.Path("o_voxel"), pathlib.Path(new_name)
    if src.is_dir() and not dst.exists():
        src.rename(dst)
        print(f"{tag} patch: o_voxel/ -> {new_name}/")
    require(dst.is_dir() and not src.exists(), f"{tag}: package directory rename did not land")
    n_imp = 0
    for py in dst.rglob("*.py"):
        t = py.read_text()
        new, k = re.subn(r"^(\s*)(from|import) o_voxel(\b)", rf"\1\2 {new_name}\3", t, flags=re.M)
        if k:
            py.write_text(new)
            n_imp += k
    print(f"{tag} patch: {n_imp} absolute self-import(s) rewritten to {new_name}")
    final = pathlib.Path("setup.py").read_text()
    require(not re.search(r"""["']o_voxel(\.|["'])""", final),
            f"{tag}: setup.py still names the unrenamed package somewhere")
    return dst


BATCHED_BVH = '''
def _batched_unsigned_distance(bvh, positions, batch_size=500000, return_uvw=False):
    """Batch unsigned_distance queries to avoid GPU kernel timeout.
    See: https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2/issues/19
    """
    N = positions.shape[0]
    if N <= batch_size:
        return bvh.unsigned_distance(positions, return_uvw=return_uvw)
    import torch
    distances_list, face_id_list, uvw_list = [], [], []
    for i in range(0, N, batch_size):
        d, f, u = bvh.unsigned_distance(positions[i:min(i+batch_size, N)], return_uvw=return_uvw)
        distances_list.append(d)
        face_id_list.append(f)
        if return_uvw:
            uvw_list.append(u)
    return (
        torch.cat(distances_list),
        torch.cat(face_id_list),
        torch.cat(uvw_list) if return_uvw else None
    )

'''


def batched_bvh_queries(tag: str, postprocess: pathlib.Path, import_anchor: str) -> None:
    """postprocess.py: query the BVH in 500k-point batches. One unbatched
    unsigned_distance over a large voxel grid runs longer than the display
    driver's kernel timeout on desktop GPUs (ComfyUI-TRELLIS2 issue #19).
    `import_anchor` is the exact `import cumesh...` line (with newline) after
    which the helper is inserted; it differs between the forks."""
    t = postprocess.read_text()
    if "_batched_unsigned_distance" in t:
        print(f"{tag} patch: batched BVH queries already applied")
        return
    require(t.count(import_anchor) == 1,
            f"{tag}: postprocess.py: expected exactly one {import_anchor!r} to "
            f"anchor the batched BVH helper on")
    t = t.replace(import_anchor, import_anchor + BATCHED_BVH, 1)
    call_old = "_, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)"
    call_new = "_, face_id, uvw = _batched_unsigned_distance(bvh, valid_pos, return_uvw=True)"
    require(t.count(call_old) == 1,
            f"{tag}: postprocess.py: the unsigned_distance call to batch is not "
            f"where the farm found it -- upstream changed")
    postprocess.write_text(t.replace(call_old, call_new))
    print(f"{tag} patch: batched BVH queries in postprocess.py")


def msvc_source_fixes(tag: str, expect_applied_upstream: bool) -> None:
    """Three MSVC-only rejections in plain C++ files, all equally valid C++
    for gcc and therefore applied to the one shared tarball:
      * `1e-6d` / `0.0d` double-literal suffix (not C++; a gcc extension)
        in convert/flexible_dual_grid.cpp
      * size_t -> int64_t narrowing in torch::zeros({N, C}) initialiser
        lists in io/filter_neighbor.cpp, io/filter_parent.cpp
      * the same in from_blob(..., {v.size()}) in io/svo.cpp
    The vb forks already carry all three; `expect_applied_upstream` says so
    and turns a no-op into an assertion rather than a silent pass."""
    n_total = 0

    fdg = pathlib.Path("src/convert/flexible_dual_grid.cpp")
    t = fdg.read_text()
    pat = r"(?<![A-Za-z_\.])(\d+\.?\d*(?:[eE][+-]?\d+)?)d\b"
    new, n = re.subn(pat, r"\1", t)
    if n:
        fdg.write_text(new)
    print(f"{tag} patch: {n} double-literal suffix(es) removed in flexible_dual_grid.cpp")
    n_total += n

    for f in ("src/io/filter_neighbor.cpp", "src/io/filter_parent.cpp"):
        p = pathlib.Path(f)
        t = p.read_text()
        new, n = re.subn(r"torch::zeros\(\{(\w+),\s*(\w+)\}",
                         r"torch::zeros({(int64_t)\1, (int64_t)\2}", t)
        if n:
            p.write_text(new)
        print(f"{tag} patch: {n} torch::zeros size_t narrowing(s) cast in {f}")
        n_total += n
        require(not re.search(r"torch::zeros\(\{\w+,\s*\w+\}", p.read_text()),
                f"{tag}: {f} still has an uncast torch::zeros({{N, C}})")

    svo = pathlib.Path("src/io/svo.cpp")
    t = svo.read_text()
    new, n = re.subn(r"\{(\w+)\.size\(\)\}", r"{(int64_t)\1.size()}", t)
    if n:
        svo.write_text(new)
    print(f"{tag} patch: {n} from_blob size_t narrowing(s) cast in svo.cpp")
    n_total += n
    require(not re.search(r"\{\w+\.size\(\)\}", svo.read_text()),
            f"{tag}: svo.cpp still has an uncast {{v.size()}}")

    if expect_applied_upstream:
        require(n_total == 0,
                f"{tag}: this fork was expected to carry the MSVC fixes already, "
                f"but {n_total} site(s) needed patching -- re-read the fork")
    else:
        # 2 literals + 2 + 2 zeros + 2 from_blob at 5565d24
        require(n_total in (0, 8),
                f"{tag}: MSVC fixes touched {n_total} site(s); expected 8 on a "
                f"fresh tree (or 0 on a re-run) -- upstream changed")


def torch_selects_std(tag: str, expect: int) -> None:
    """Drop the hardcoded C++ standard from setup.py's cxx/nvcc flag lists
    (`-std=c++17` on upstream and the ap fork, `-std=c++20` on vb) and let
    torch's cpp_extension append the one the installed torch needs. Also
    what keeps win-64 from seeing two `-std=` values on one nvcc line."""
    setup = pathlib.Path("setup.py")
    t = setup.read_text()
    new, n = strip_std_flags(t)
    if n == 0:
        require("std=c++" not in t and "std:c++" not in t,
                f"{tag}: a C++-standard flag survives in setup.py in a spelling "
                f"strip_std_flags does not recognise")
        print(f"{tag} patch: setup.py already carries no C++-standard flag")
        return
    require(n == expect, f"{tag}: stripped {n} C++-standard flag(s), expected "
                         f"{expect} -- upstream changed")
    setup.write_text(new)
    print(f"{tag} patch: dropped {n} hardcoded C++-standard flag(s); torch selects it")


def assert_arch_list_authoritative(tag: str) -> None:
    for lineno, line in enumerate(pathlib.Path("setup.py").read_text().splitlines(), 1):
        if re.search(r"(?<!offload)-arch[= ]|gencode|get_device_capability|TORCH_CUDA_ARCH_LIST", line):
            sys.exit(f"{tag} patch: setup.py:{lineno} touches the arch list "
                     f"({line.strip()!r}) -- re-check")
    print(f"{tag} patch: setup.py emits no arch flags; TORCH_CUDA_ARCH_LIST is authoritative")
