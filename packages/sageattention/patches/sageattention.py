"""Patch SageAttention v2.2.0 for the foundry build.

Ported from cuda-wheels (packages/sageattention/patches/sageattention.py) and
re-read against the pinned rev. Runs with cwd = the cloned source, once,
before the tarball is sealed; every platform's build consumes the same tree.
The platform branches below are therefore written INTO setup.py, where they
run on the build machine -- never evaluated here.

1. TORCH_CUDA_ARCH_LIST parser. Upstream splits only on ',' and ';'; the
   cell's list is space-separated ("8.0 8.6 8.9 9.0 12.0+PTX"). Without this
   no HAS_SM* flag is ever set, every qattn extension is skipped, the build
   still succeeds, and the wheel raises on real hardware. Not cosmetic.

2. CXX_FLAGS / _GLIBCXX_USE_CXX11_ABI: upstream's lists are GCC-only
   (-fopenmp, -lgomp, -D_GLIBCXX_USE_CXX11_ABI). Made platform-aware inside
   setup.py. `-g` is dropped on both: it is debug info nothing consumes and
   auditwheel strips it after it was generated at full cost.

3. _qattn_sm90 gets ONLY `-gencode arch=compute_90a,code=sm_90a`. Its kernels
   use wgmma, and ptxas fails with "wgmma.mma_async not supported on .target
   sm_80" if the global gencode list reaches it. `-lcuda` becomes
   `libraries=["cuda"]`, which torch turns into the right spelling on both
   platforms (-lcuda / cuda.lib).

4. _qattn_sm89 drops every gencode below sm_89. csrc/mma.cuh:44 gates the
   FP8 QMMA wrappers on __CUDA_ARCH__ >= 890 and otherwise expands them to
   __brkpt() (mma.cuh:55-57): a shipped wheel was measured at 26,880 BPT.TRAP
   and zero QMMA in its sm_80 cubin. Nothing can dispatch to those cubins
   (core.py routes only sm89 here), so they were pure cost. Filtered by
   capability so it keeps working as the arch policy moves.

5. Hardcoded -std=c++17 in both flag lists: stripped so torch's cpp_extension
   selects the standard the installed torch needs.

Every substitution asserts on its own; a whole-file before/after check passes
as long as ANY edit landed (docs/WINDOWS.md, the fused-ssim lesson).
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require, strip_std_flags  # noqa: E402

MARKER = "# patched-by: cuda-foundry sageattention"
setup_file = pathlib.Path("setup.py")
content = setup_file.read_text()
if MARKER in content:
    print("sageattention patch: already applied")
    sys.exit(0)


def sub_once(text: str, old: str, new: str, what: str) -> str:
    require(text.count(old) == 1,
            f"sageattention: {what} matched {text.count(old)} time(s) in setup.py, "
            f"expected exactly 1 -- upstream changed at this rev; re-read it")
    print(f"sageattention patch: {what}")
    return text.replace(old, new, 1)


# 1. arch parser -----------------------------------------------------------
content = sub_once(
    content,
    '    for item in arch_list_env.replace(",", ";").split(";"):',
    '    for item in arch_list_env.replace(",", " ").replace(";", " ").split():',
    "arch parser accepts space-separated TORCH_CUDA_ARCH_LIST")

# 2. platform-aware host flags ---------------------------------------------
content = sub_once(
    content,
    '    CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]',
    '''    import platform
    if platform.system() == "Windows":
        CXX_FLAGS = ["/O2", "/openmp", "-DENABLE_BF16"]
    else:
        CXX_FLAGS = ["-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]''',
    "CXX_FLAGS made platform-aware (-g dropped)")
content = sub_once(
    content,
    '''    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]''',
    '''    if platform.system() != "Windows":
        ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
        CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
        NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]''',
    "_GLIBCXX_USE_CXX11_ABI skipped on Windows")

# 3. _qattn_sm90: sm_90a only, driver library via `libraries` ----------------
content = sub_once(
    content,
    '''                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
                extra_link_args=['-lcuda'],''',
    '''                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": [f for f in NVCC_FLAGS if "gencode" not in f and "arch=" not in f] + ["-gencode", "arch=compute_90a,code=sm_90a"]},
                libraries=["cuda"],''',
    "_qattn_sm90 compiles for sm_90a only and links the driver via libraries=")

# 4. _qattn_sm89: no gencode below sm_89 -------------------------------------
content = sub_once(
    content,
    "import warnings",
    '''import warnings


def _cuw_min_cc_gencodes(flags, minimum):
    """Drop -gencode pairs whose compute capability is below `minimum`.

    Injected by cuda-foundry (packages/sageattention/patches). Pairs are
    removed two elements at a time: setup.py appends them as
    ["-gencode", "arch=compute_NN,code=sm_NN"].
    """
    out, i = [], 0
    while i < len(flags):
        if flags[i] == "-gencode" and i + 1 < len(flags):
            spec = flags[i + 1]
            cc = spec.split("compute_")[1].split(",")[0] if "compute_" in spec else ""
            if cc.isdigit() and int(cc) < minimum:
                i += 2
                continue
            out.extend([flags[i], flags[i + 1]])
            i += 2
            continue
        out.append(flags[i])
        i += 1
    return out''',
    "gencode filter helper injected")
content = sub_once(
    content,
    '''                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu",
                ],
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},''',
    '''                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu",
                ],
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": _cuw_min_cc_gencodes(NVCC_FLAGS, 89)},''',
    "_qattn_sm89 skips gencodes below sm_89 (FP8 QMMA floor)")

# 5. the C++ standard --------------------------------------------------------
content, n_std = strip_std_flags(content)
require(n_std == 2, f"sageattention: expected 2 hardcoded C++-standard flags "
                    f"(CXX_FLAGS, NVCC_FLAGS), stripped {n_std}")
print("sageattention patch: dropped 2 hardcoded std flags; torch selects the standard")

setup_file.write_text(MARKER + "\n" + content)

final = setup_file.read_text()
for needle, what in (
        ('_cuw_min_cc_gencodes(NVCC_FLAGS, 89)', "the _qattn_sm89 FP8 gencode filter"),
        ('arch=compute_90a,code=sm_90a', "the _qattn_sm90 Hopper gencode filter"),
        ('libraries=["cuda"]', "the driver link"),
        ('.replace(";", " ").split()', "the arch parser")):
    require(needle in final, f"sageattention: {what} is NOT PRESENT in setup.py on disk")
require(not re.search(r"""['"](?:-Xcompiler=)?[-/]std[=:]c\+\+\d+['"]""", final),
        "sageattention: a C++-standard flag survived stripping")
import ast  # noqa: E402
ast.parse(final)
print("sageattention patch: done")
