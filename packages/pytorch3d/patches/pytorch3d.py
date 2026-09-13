"""Patch pytorch3d v0.7.9 for the foundry build.

Ported from cuda-wheels (packages/pytorch3d/patches/pytorch3d.py) and re-read
against the pinned rev. Runs with cwd = the cloned source, once, BEFORE the
tarball is sealed -- so it must not depend on the cell (no torch, no CUDA
version): every cell's build consumes the same tree.

1. Stop hardcoding the C++ standard. setup.py pins `-std=c++17` in two places
   (extra_compile_args["cxx"] and, off Windows, nvcc_args). torch's
   cpp_extension appends the standard the INSTALLED torch needs only when the
   caller gave none, so an explicit pin overrides it: right for torch 2.8,
   wrong the day a torch needs C++20. Dropping it hands the choice to torch,
   on every platform (MSVC included -- pytorch3d's cxx list carries no other
   flag that would need an MSVC spelling).

2. `projects/` must not ship as a top-level package. find_packages excludes
   `projects.*` but not `projects` itself, so the bare top-level directory --
   a licence header and nothing else -- lands in site-packages under one of
   the most collidable names there is. The implicitron trainer is added
   separately with its own package_dir and is unaffected.

3. No console scripts. setup.py registers two entry points,
   pytorch3d_implicitron_runner and pytorch3d_implicitron_visualizer, into
   projects/implicitron_trainer. On win-64 pip renders each as a .exe
   launcher with the BUILD machine's interpreter path baked in
   (D:/a/_temp/.../python.exe, Windows-spelled -- verified on the published artifact), a
   path no user has; and what they import -- hydra-core, visdom, lpips,
   accelerate, sqlalchemy, the implicitron extras -- is not declared by this
   package either. Dropped rather than rendered through the recipe: the
   library is the artifact, the trainer CLI is upstream's research
   scaffolding (its package_dir is added separately in setup.py and stays;
   only the entry points go).

4. CUDA >= 13: pulsar's explicitly-instantiated __global__ templates
   (`calc_signature<true>`, `calc_gradients<true>`, `render<true>`, ... --
   the device instantiations, ISONDEVICE=true, defined in separate .gpu.cu
   TUs) lose external linkage under CUDA 13's new default, so every TU that
   references them fails to LINK with "undefined reference"
   (measured, run 34730072697, linux-aarch64 cu130). The farm fixed this by
   appending `-static-global-template-stub=false` to NVCC_FLAGS via
   $GITHUB_ENV -- a build-time, per-cell decision that a fetch-time patch
   cannot make directly. So the patch does not decide it at fetch time; it
   WRITES a build-time gate into setup.py that reads torch.version.cuda (the
   cell's own CUDA flavour) and appends the flag only when the major is >= 13.
   The flag exists only from CUDA 12.5, so an unconditional append would break
   the cu12.4 cell; the >= 13 gate matches exactly where the linkage default
   changed. Inert on cu12.8 (this repo's linux-64 cell), active on cu13.x.

5. The -ccbin={CC} nvcc append. setup.py appends "-ccbin={}".format(CC) to
   nvcc_args. On linux-aarch64 CC is "aarch64-conda-linux-gnu-cc", whose
   "aarch64" contains the substring "arch", and torch's
   _get_cuda_arch_flags returns [] (emits NO -gencode) as soon as any nvcc
   flag contains "arch" -- so the cell's TORCH_CUDA_ARCH_LIST is dropped and
   nvcc builds for its default arch alone, which fails verify_conda's SASS
   census. linux-64 is unaffected (x86_64-... has no "arch" substring). The
   conda build env already passes -ccbin=$CXX through NVCC_PREPEND_FLAGS, so
   the append is redundant on every platform; it is dropped (same fix as
   cc_torch / torch-generic-nms).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import exclude_top_level_packages, require, strip_std_flags  # noqa: E402

setup_file = pathlib.Path("setup.py")
content = setup_file.read_text()
content, n_std = strip_std_flags(content)
require(n_std == 2,
        f"pytorch3d: expected exactly 2 hardcoded C++-standard flags in setup.py "
        f"(cxx list + nvcc append), found {n_std} -- upstream changed; re-read "
        f"setup.py before building against an unverified flag set")
setup_file.write_text(content)
print(f"pytorch3d patch: dropped {n_std} hardcoded std flag(s); torch's "
      f"cpp_extension now selects the standard")

import re  # noqa: E402

content = setup_file.read_text()
if "entry_points=" not in content:
    print("pytorch3d patch: no entry_points block (already removed)")
else:
    content, n_ep = re.subn(
        r"    entry_points=\{\n        \"console_scripts\": \[\n(?:            .*\n)+?        \]\n    \},\n",
        "    # cuda-foundry: the two implicitron console scripts are not shipped\n"
        "    # (see packages/pytorch3d/patches/pytorch3d.py, item 3).\n",
        content)
    require(n_ep == 1 and "entry_points=" not in content and "console_scripts" not in content,
            "pytorch3d: expected exactly one entry_points={console_scripts: [...]} block "
            "in setup.py -- upstream changed; re-read it before dropping the scripts")
    import ast
    ast.parse(content)
    setup_file.write_text(content)
    print("pytorch3d patch: entry_points (implicitron console scripts) removed")

# ── 4/5. CUDA-13 template-stub linkage + drop the -ccbin arch-discard ──────
content = setup_file.read_text()

CCBIN_OLD = (
    "                if existing_CC is None:\n"
    '                    CC_arg = "-ccbin={}".format(CC)\n'
    "                    nvcc_args.append(CC_arg)\n"
)
CCBIN_NEW = (
    "                if existing_CC is None:\n"
    "                    # cuda-foundry: -ccbin={CC} append dropped. On\n"
    '                    # linux-aarch64 CC is "aarch64-conda-linux-gnu-cc",\n'
    '                    # whose "aarch64" contains "arch", and torch\'s\n'
    "                    # _get_cuda_arch_flags returns [] (no -gencode) once any\n"
    '                    # nvcc flag contains "arch", dropping the cell\'s\n'
    "                    # TORCH_CUDA_ARCH_LIST. The conda env already passes\n"
    "                    # -ccbin=$CXX via NVCC_PREPEND_FLAGS. See cc_torch.\n"
    "                    pass\n"
)
require(
    content.count(CCBIN_OLD) == 1,
    f"pytorch3d: the -ccbin={{CC}} append block matched {content.count(CCBIN_OLD)} "
    "time(s) in setup.py, expected exactly 1 -- upstream changed; re-read it",
)
content = content.replace(CCBIN_OLD, CCBIN_NEW, 1)

STUB_OLD = '        extra_compile_args["nvcc"] = nvcc_args\n'
STUB_NEW = (
    "        # cuda-foundry: CUDA 13 made the default for explicitly-instantiated\n"
    "        # __global__ templates a stub with internal linkage, so pulsar's\n"
    "        # calc_signature<true> / calc_gradients<true> / render<true> / ...\n"
    "        # (device instantiations in separate .gpu.cu TUs) become undefined\n"
    "        # references at link (run 34730072697, linux-aarch64 cu130).\n"
    "        # -static-global-template-stub=false restores the old behaviour; it\n"
    "        # exists only from CUDA 12.5, so gate on the toolkit MAJOR >= 13\n"
    "        # (read from torch's own CUDA, i.e. the cell's flavour).\n"
    "        if torch.version.cuda and int(torch.version.cuda.split('.')[0]) >= 13:\n"
    '            nvcc_args.append("-static-global-template-stub=false")\n'
    '        extra_compile_args["nvcc"] = nvcc_args\n'
)
require(
    content.count(STUB_OLD) == 1,
    f"pytorch3d: `extra_compile_args[\"nvcc\"] = nvcc_args` matched "
    f"{content.count(STUB_OLD)} time(s) in setup.py, expected exactly 1 -- "
    "upstream changed; re-read it",
)
content = content.replace(STUB_OLD, STUB_NEW, 1)

import ast as _ast  # noqa: E402
_ast.parse(content)
setup_file.write_text(content)
require(
    '.append(CC_arg)' not in setup_file.read_text()
    and "-static-global-template-stub=false" in setup_file.read_text(),
    "pytorch3d: the ccbin drop or the CUDA-13 stub flag did not land in setup.py",
)
print("pytorch3d patch: dropped -ccbin arch-discard; CUDA-13 static template "
      "stub flag gated on torch.version.cuda >= 13")

exclude_top_level_packages(["projects"])
require('"projects"' in setup_file.read_text(),
        "pytorch3d: the projects exclusion did not land in setup.py")
print("pytorch3d patch: done")
