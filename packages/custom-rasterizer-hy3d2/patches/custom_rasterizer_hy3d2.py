"""Make Hunyuan3D-2.1's custom_rasterizer build on Windows, rename it, and
relocate it to the tarball root.

Runs on the fetching machine with cwd = the Hunyuan3D-2.1 checkout, before
the tarball is sealed; one tarball serves every platform, and nothing here
branches on the host. Ported from cuda-wheels; the relocation is this repo's.

THREE JOBS.

1. Windows int64 portability. Upstream assumes LP64 (Linux), where `long` is
   64-bit. On MSVC `long` is 32-bit, which breaks the extension two ways:

     * `.data_ptr<long>()` on an int64 tensor -- torch instantiates data_ptr
       for int64_t, and on Windows `long` is a distinct 32-bit type, so there
       is no matching template and the build fails.
     * `torch::zeros({ some_container.size(), 9 }, ...)` -- the braced list
       must convert to IntArrayRef (int64_t); size_t -> int64_t inside braces
       is a narrowing conversion, which MSVC rejects as an error.

   This is why no prebuilt Windows wheel of the 2.1 rasterizer exists. The fix
   is not invented here: Tencent already shipped it in the Hunyuan3D 2.0 tree
   (hy3dgen/texgen/custom_rasterizer), and this very rev carries it too, as
   lib/custom_rasterizer_kernel_for_windows -- diffing that against
   lib/custom_rasterizer_kernel shows these int64_t/static_cast edits (plus
   two %zu printf formats) are the ONLY difference. So the patch ports the
   kernel tree setup.py actually builds to that already-proven form, and the
   resulting extension is the same on both platforms.

2. Rename + slim the distribution. `custom_rasterizer` is far too generic a
   name to occupy in an index, and both Hunyuan3D families claim it. The dist
   becomes `custom_rasterizer_hy3d2`. The pure-Python `custom_rasterizer`
   package is dropped from the wheel so the only thing installed is the
   `custom_rasterizer_kernel` extension -- consumers vendor their own Python
   wrapper anyway (ComfyUI-3D-Pack does, in both HY families).

   The EXTENSION module keeps its name, `custom_rasterizer_kernel`. That is
   deliberate: it is what upstream's render.py does `import
   custom_rasterizer_kernel` on. Renaming it would force every consumer to be
   patched too.

3. Relocate. The package lives at hy3dpaint/custom_rasterizer inside a repo
   of model code and demo servers (the farm's `build_subdir`). This repo's
   build runs `pip wheel .` at the tarball root, so that directory is moved to
   the root, the repo's LICENSE is copied in beside it, and the rest of the
   checkout is dropped.

Every substitution below asserts its expected hit count, so if upstream edits
these files the build fails here instead of silently producing an extension
that is missing the fix.
"""

import shutil
import sys
from pathlib import Path

SUBDIR = Path("hy3dpaint/custom_rasterizer")
KERNEL = SUBDIR / "lib/custom_rasterizer_kernel"
MARKER = "# patched-by: cuda-foundry custom_rasterizer_hy3d2"

# (path, [(old, new, expected_occurrences), ...])
EDITS = [
    (
        KERNEL / "grid_neighbor.cpp",
        [
            # Braced IntArrayRef init from size_t -> narrowing on MSVC.
            (
                "torch::zeros({seq2pos.size() / 3, 3}, float_options)",
                "torch::zeros({static_cast<int64_t>(seq2pos.size() / 3), "
                "static_cast<int64_t>(3)}, float_options)",
                2,
            ),
            (
                "torch::zeros({seq2pos.size() / 3}, float_options)",
                "torch::zeros({static_cast<int64_t>(seq2pos.size() / 3)}, float_options)",
                2,
            ),
            (
                "torch::zeros({seq2feat.size() / feat_channel, feat_channel}, float_options)",
                "torch::zeros({static_cast<int64_t>(seq2feat.size() / feat_channel), "
                "static_cast<int64_t>(feat_channel)}, float_options)",
                1,
            ),
            (
                "torch::zeros({grids[i].seq2grid.size(), 9}, int64_options)",
                "torch::zeros({static_cast<int64_t>(grids[i].seq2grid.size()), "
                "static_cast<int64_t>(9)}, int64_options)",
                2,
            ),
            (
                "torch::zeros({grids[i].seq2evencorner.size()}, int64_options)",
                "torch::zeros({static_cast<int64_t>(grids[i].seq2evencorner.size())}, "
                "int64_options)",
                2,
            ),
            (
                "torch::zeros({grids[i].seq2oddcorner.size()}, int64_options)",
                "torch::zeros({static_cast<int64_t>(grids[i].seq2oddcorner.size())}, "
                "int64_options)",
                2,
            ),
            (
                "torch::zeros({grids[i].downsample_seq.size()}, int64_options)",
                "torch::zeros({static_cast<int64_t>(grids[i].downsample_seq.size())}, "
                "int64_options)",
                2,
            ),
            # 32-bit `long` pointers into 64-bit tensors.
            ("long* nptr", "int64_t* nptr", 2),
            ("long* dptr", "int64_t* dptr", 4),
            ("data_ptr<long>()", "data_ptr<int64_t>()", 8),
        ],
    ),
    (
        KERNEL / "rasterizer.cpp",
        [
            ("(long)maxint", "(int64_t)maxint", 1),
            ("data_ptr<long>()", "data_ptr<int64_t>()", 3),
        ],
    ),
    (
        KERNEL / "rasterizer_gpu.cu",
        [
            ("(long)maxint", "(int64_t)maxint", 1),
            ("data_ptr<long>()", "data_ptr<int64_t>()", 3),
        ],
    ),
    (
        SUBDIR / "setup.py",
        [
            ('name="custom_rasterizer"', 'name="custom_rasterizer_hy3d2"', 1),
            # Ship only the CUDA extension, not the generic Python package.
            ("packages=find_packages()", "packages=[]", 1),
        ],
    ),
]


def die(msg: str) -> None:
    raise SystemExit(f"custom_rasterizer_hy3d2 patch: {msg}")


def apply_edits() -> None:
    for path, edits in EDITS:
        if not path.exists():
            die(f"missing {path}")
        text = path.read_text(encoding="utf-8")
        for old, new, expected in edits:
            found = text.count(old)
            if found != expected:
                die(f"{path}: expected {expected} occurrence(s) of {old!r}, found "
                    f"{found}. Upstream changed -- re-diff against "
                    f"lib/custom_rasterizer_kernel_for_windows before building.")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
        print(f"Patched {path} ({len(edits)} substitutions)")

    # Nothing may still reach for a 32-bit long view of an int64 tensor.
    for path, _ in EDITS[:3]:
        leftover = path.read_text(encoding="utf-8").count("data_ptr<long>")
        if leftover:
            die(f"{path}: {leftover} data_ptr<long> site(s) survived")
    setup_py = SUBDIR / "setup.py"
    setup_py.write_text(MARKER + "\n" + setup_py.read_text(encoding="utf-8"),
                        encoding="utf-8")
    print("custom_rasterizer -> custom_rasterizer_hy3d2, int64_t port applied")


def relocate() -> None:
    if not Path("LICENSE").is_file():
        die("LICENSE is missing at the repo root")
    shutil.copy2("LICENSE", SUBDIR / "LICENSE")
    staging = Path(".cuw-relocate")
    shutil.move(str(SUBDIR), str(staging))
    dropped = 0
    for entry in Path(".").iterdir():
        if entry.name in (".git", staging.name):
            continue
        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        dropped += 1
    for entry in list(staging.iterdir()):
        shutil.move(str(entry), str(Path(".") / entry.name))
    staging.rmdir()
    if not (Path("setup.py").is_file()
            and Path("lib/custom_rasterizer_kernel/rasterizer_gpu.cu").is_file()):
        die("relocation did not leave setup.py and the kernel tree at the root")
    print(f"custom_rasterizer_hy3d2 patch: relocated {SUBDIR} to the root, "
          f"dropped {dropped} top-level entr(ies)")


def main() -> None:
    if not SUBDIR.is_dir():
        setup_py = Path("setup.py")
        if setup_py.is_file() and MARKER in setup_py.read_text(encoding="utf-8"):
            print("custom_rasterizer_hy3d2 patch: already applied")
            return
        die(f"{SUBDIR} is not in the checkout and the root is not an "
            f"already-patched tree -- upstream moved the package")
    apply_edits()
    relocate()


if __name__ == "__main__":
    main()
