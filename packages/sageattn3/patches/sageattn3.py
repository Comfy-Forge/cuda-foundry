"""Patch SageAttention3 (thu-ml/SageAttention v2.2.0, sageattention3_blackwell/)
for the foundry build.

Ported from cuda-wheels (packages/sageattn3/patches/sageattn3.py) and re-read
against the pinned rev. Runs with cwd = the cloned repo root, once, before the
tarball is sealed; every path below is under sageattention3_blackwell/, the
package.yml build_subdir.

1. VENDOR CUTLASS, pinned. setup.py clones NVIDIA/cutlass at build time --
   unpinned -- and the build has no network. Clone v4.2.1 (f3fde583, the last
   release before this tag) into csrc/cutlass here, where the network exists,
   strip its .git, and pin the fallback clone line to the same tag so the
   build can never reach for a different tree. Only include/ and
   tools/util/include are needed; the rest is dropped to keep the tarball
   small.

2. TORCH_CUDA_ARCH_LIST instead of torch.cuda.get_device_capability(): the
   runner has no GPU. The map is exact and a token outside it is an ERROR,
   not a skip -- the arch list in arch_override.yml is a promise the SASS
   census checks, and a patch that quietly dropped a token would make the
   two disagree hours later.

3. Platform-aware host flags inside setup.py (MSVC needs /Zc:__cplusplus,
   /bigobj, /permissive-, forwarded to nvcc's cudafe pass with -Xcompiler);
   FORCE_CXX11_ABI is a libstdc++ knob and is skipped on Windows.

4. MSVC header patches (kernel_traits.h, kernel_ws.h, launch.h), each under
   #if defined(_MSC_VER) IN THE SOURCE -- one tarball serves both platforms,
   so the condition cannot live here. From mengqin/SageAttention 8bb81e4.

5. Hardcoded -std=c++17 stripped (three sites) so torch selects the standard.

Every substitution asserts on its own.
"""
import pathlib
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require, strip_std_flags  # noqa: E402

SUBDIR = "sageattention3_blackwell"
CUTLASS_TAG = "v4.2.1"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
MARKER = "# patched-by: cuda-foundry sageattn3"


def sub_once(text: str, old: str, new: str, what: str, path: str) -> str:
    require(text.count(old) == 1,
            f"sageattn3: {what} matched {text.count(old)} time(s) in {path}, "
            f"expected exactly 1 -- upstream changed at this rev; re-read it")
    print(f"sageattn3 patch: {path}: {what}")
    return text.replace(old, new, 1)


root = pathlib.Path(".").resolve() / SUBDIR
require((root / "setup.py").is_file() and (root / "sageattn3" / "api.py").is_file(),
        f"sageattn3: {SUBDIR}/ does not look like the sageattention3 project at this rev")

setup_file = root / "setup.py"
content = setup_file.read_text()
if MARKER in content:
    print("sageattn3 patch: already applied")
    sys.exit(0)

# 1. CUTLASS ----------------------------------------------------------------
cutlass = root / "csrc" / "cutlass"
if not (cutlass / "include" / "cutlass" / "cutlass.h").is_file():
    tmp = root / "csrc" / "cutlass.clone"
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", CUTLASS_TAG,
                    "https://github.com/NVIDIA/cutlass.git", str(tmp)], check=True)
    got = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp, capture_output=True,
                         text=True, check=True).stdout.strip()
    require(got == CUTLASS_COMMIT,
            f"sageattn3: CUTLASS {CUTLASS_TAG} resolved to {got}, expected {CUTLASS_COMMIT} "
            f"-- the tag moved; refusing to vendor an unverified tree")
    cutlass.mkdir(parents=True)
    shutil.move(str(tmp / "include"), str(cutlass / "include"))
    (cutlass / "tools" / "util").mkdir(parents=True)
    shutil.move(str(tmp / "tools" / "util" / "include"), str(cutlass / "tools" / "util" / "include"))
    shutil.copy(str(tmp / "LICENSE.txt"), str(cutlass / "LICENSE.txt"))
    shutil.rmtree(tmp)
    (cutlass / "CUW_VENDORED").write_text(f"NVIDIA/cutlass {CUTLASS_TAG} {CUTLASS_COMMIT}\n")
    print(f"sageattn3 patch: vendored CUTLASS {CUTLASS_TAG} ({CUTLASS_COMMIT}) into csrc/cutlass")
require((cutlass / "include" / "cute" / "tensor.hpp").is_file(),
        "sageattn3: csrc/cutlass/include/cute is missing after vendoring")

content = sub_once(
    content,
    '''            ["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", str(cutlass_dir)],''',
    f'''            ["git", "clone", "--depth", "1", "--branch", "{CUTLASS_TAG}",
             "https://github.com/NVIDIA/cutlass.git", str(cutlass_dir)],''',
    f"fallback CUTLASS clone pinned to {CUTLASS_TAG} (never reached: the tree is vendored)",
    "setup.py")

# 2. arch list from the cell -------------------------------------------------
content = sub_once(
    content,
    '''    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")
    cc_major, cc_minor = torch.cuda.get_device_capability()
    if (cc_major, cc_minor) == (10, 0):  # sm_100
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_100a,code=sm_100a")
    elif (cc_major, cc_minor) == (12, 0):  # sm_120
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_120a,code=sm_120a")
    else:
        raise RuntimeError("Unsupported GPU")''',
    '''    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")
    # cuda-foundry: the cell's TORCH_CUDA_ARCH_LIST decides, not a local GPU.
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    arch_map = {
        "10.0": ("compute_100a", "sm_100a"),
        "12.0": ("compute_120a", "sm_120a"),
        "12.1": ("compute_121a", "sm_121a"),
    }
    _unknown = []
    for item in arch_list_env.replace(",", " ").replace(";", " ").split():
        item = item.strip().split("+")[0]
        if item in arch_map:
            compute, sm = arch_map[item]
            cc_flag.extend(["-gencode", f"arch={compute},code={sm}"])
        else:
            _unknown.append(item)
    if _unknown:
        raise RuntimeError(
            f"sageattn3: TORCH_CUDA_ARCH_LIST asks for {_unknown}, which this "
            f"package cannot emit. Known: {sorted(arch_map)}. Fix "
            f"packages/sageattn3/arch_override.yml; do not widen arch_map "
            f"unless upstream grew a kernel for it.")
    if not cc_flag:
        raise RuntimeError(
            f"No supported Blackwell architectures in TORCH_CUDA_ARCH_LIST={arch_list_env!r}; "
            "expected one of 10.0, 12.0, 12.1")''',
    "GPU probe replaced by TORCH_CUDA_ARCH_LIST parsing", "setup.py")

# 3. platform-aware host flags ---------------------------------------------
content = sub_once(
    content,
    '''    ext_modules.append(
        CUDAExtension(
            name="fp4attn_cuda",
            sources=["sageattn3/blackwell/api.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],''',
    '''    if os.name == "nt":
        nvcc_flags += ["-D_WIN32=1", "-DUSE_CUDA=1"]
        # No /std: here -- torch's cpp_extension selects the standard.
        cxx_flags = ["/Zc:__cplusplus", "/bigobj", "/MD", "/permissive-"]
        nvcc_flags += [f"-Xcompiler={flag}" for flag in cxx_flags]
        cxx_flags += ["/O2"]
    else:
        cxx_flags = ["-O3"]

    ext_modules.append(
        CUDAExtension(
            name="fp4attn_cuda",
            sources=["sageattn3/blackwell/api.cu"],
            extra_compile_args={
                "cxx": cxx_flags,''',
    "platform-aware cxx flags (fp4attn_cuda)", "setup.py")
content = sub_once(
    content,
    '''            name="fp4quant_cuda",
            sources=["sageattn3/quantization/fp4_quantization_4d.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],''',
    '''            name="fp4quant_cuda",
            sources=["sageattn3/quantization/fp4_quantization_4d.cu"],
            extra_compile_args={
                "cxx": cxx_flags,''',
    "platform-aware cxx flags (fp4quant_cuda)", "setup.py")
content = sub_once(
    content,
    '''    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True''',
    '''    if FORCE_CXX11_ABI and os.name != "nt":
        torch._C._GLIBCXX_USE_CXX11_ABI = True''',
    "FORCE_CXX11_ABI skipped on Windows", "setup.py")

# 5. the C++ standard (the two cxx lists went with step 3; nvcc_flags remains)
content, n_std = strip_std_flags(content)
require(n_std == 1, f"sageattn3: expected 1 remaining hardcoded C++-standard flag "
                    f"(nvcc_flags), stripped {n_std}")
setup_file.write_text(MARKER + "\n" + content)
final = setup_file.read_text()
require(not re.search(r"""['"](?:-Xcompiler=)?[-/]std[=:]c\+\+\d+['"]""", final),
        "sageattn3: a C++-standard flag survived stripping")
require(f'"--branch", "{CUTLASS_TAG}"' in final and "arch_map" in final,
        "sageattn3: setup.py on disk lacks the CUTLASS pin or the arch map")
import ast  # noqa: E402
ast.parse(final)

# 4. MSVC header patches ----------------------------------------------------
kt_file = root / "sageattn3" / "blackwell" / "kernel_traits.h"
kt = kt_file.read_text()
kt = sub_once(
    kt,
    """\
    using BlkScaledConfig = flash::BlockScaledConfig<SFVectorSize>;
    using LayoutSF = typename BlkScaledConfig::LayoutSF;
    using SfAtom = typename BlkScaledConfig::SfAtom;
    using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFV = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));
    using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));""",
    """\
#if defined(_MSC_VER)
    using BlkScaledConfig = ::flash::BlockScaledConfig<SFVectorSize>;
    // Inline the definitions to avoid MSVC dependent-name quirks
    using SfAtom = Layout<
        Shape< Shape<_16, _4>, Shape<Int<SFVectorSize>, Int<4>>>,
        Stride<Stride<_16, _4>, Stride<_0, _1>>
    >;
    using LayoutSF = decltype(
      blocked_product(
        SfAtom{},
        make_layout(
          make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
          make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0))
        )
      )
    );
    using SmemLayoutAtomSFQ = decltype(::flash::BlockScaledConfig<SFVectorSize>::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFK = decltype(::flash::BlockScaledConfig<SFVectorSize>::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFV = decltype(::flash::BlockScaledConfig<SFVectorSize>::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));
    using SmemLayoutAtomSFVt = decltype(::flash::BlockScaledConfig<SFVectorSize>::deduce_smem_layoutSFVt(TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));
#else
    using BlkScaledConfig = flash::BlockScaledConfig<SFVectorSize>;
    using LayoutSF = typename BlkScaledConfig::LayoutSF;
    using SfAtom = typename BlkScaledConfig::SfAtom;
    using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFV = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));
    using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));
#endif""",
    "MSVC BlkScaledConfig workaround", "kernel_traits.h")
kt_file.write_text(kt)

kw_file = root / "sageattn3" / "blackwell" / "kernel_ws.h"
kw = kw_file.read_text()
kw = sub_once(
    kw,
    """\
template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws(CUTE_GRID_CONSTANT Flash_fwd_params const params,
                    CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const mainloop_params,
                    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits>::Params const epilogue_params,
                    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
                    ) {""",
    """\
template <typename Ktraits, bool Is_causal, typename TileScheduler>
__device__ inline void
compute_attn_ws_impl(Flash_fwd_params const &params,
                     typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const &mainloop_params,
                     typename CollectiveEpilogueFwd<Ktraits>::Params const &epilogue_params,
                     typename TileScheduler::Params const &scheduler_params) {""",
    "kernel body extracted into compute_attn_ws_impl", "kernel_ws.h")
kw = sub_once(
    kw,
    """\
}

} // namespace flash""",
    """\
}

#if defined(_MSC_VER)
// MSVC cannot pass over-aligned structs by value as CUTE_GRID_CONSTANT
// kernel parameters (C2719); take pointers and dereference on the device.

template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
compute_attn_ws(Flash_fwd_params const *params,
                typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const *mainloop_params,
                typename CollectiveEpilogueFwd<Ktraits>::Params const *epilogue_params,
                typename TileScheduler::Params const *scheduler_params) {
    compute_attn_ws_impl<Ktraits, Is_causal, TileScheduler>(
        *params, *mainloop_params, *epilogue_params, *scheduler_params);
}

#else

template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws(CUTE_GRID_CONSTANT Flash_fwd_params const params,
                    CUTE_GRID_CONSTANT
                        typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const
                            mainloop_params,
                    CUTE_GRID_CONSTANT
                        typename CollectiveEpilogueFwd<Ktraits>::Params const
                            epilogue_params,
                    CUTE_GRID_CONSTANT
                        typename TileScheduler::Params const scheduler_params) {

    compute_attn_ws_impl<Ktraits, Is_causal, TileScheduler>(
        params, mainloop_params, epilogue_params, scheduler_params);
}

#endif  // _MSC_VER

} // namespace flash""",
    "MSVC and non-MSVC kernel wrappers", "kernel_ws.h")
kw_file.write_text(kw)

lh_file = root / "sageattn3" / "blackwell" / "launch.h"
lh = lh_file.read_text()
lh = sub_once(
    lh,
    "    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};\n"
    "    cutlass::launch_kernel_on_cluster(launch_params, kernel, params, mainloop_params, epilogue_params, scheduler_params);\n"
    "    \n"
    "    C10_CUDA_KERNEL_LAUNCH_CHECK();",
    """\
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};

#if defined(_MSC_VER)
    // MSVC: pack the over-aligned parameters into device memory and pass
    // pointers, avoiding C2719 on by-value kernel parameters.

    struct DeviceParamsPack {
        Flash_fwd_params params;
        typename CollectiveMainloop::Params mainloop;
        typename CollectiveEpilogue::Params epilogue;
        typename Scheduler::Params scheduler;
    };

    DeviceParamsPack h_pack{params, mainloop_params, epilogue_params, scheduler_params};

    DeviceParamsPack *d_pack = nullptr;
    C10_CUDA_CHECK(cudaMallocAsync(&d_pack, sizeof(DeviceParamsPack), stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        d_pack, &h_pack, sizeof(DeviceParamsPack),
        cudaMemcpyHostToDevice, stream));

    char *base_h = reinterpret_cast<char*>(&h_pack);
    auto off_params   = reinterpret_cast<char*>(&h_pack.params)    - base_h;
    auto off_mainloop = reinterpret_cast<char*>(&h_pack.mainloop)  - base_h;
    auto off_epilogue = reinterpret_cast<char*>(&h_pack.epilogue)  - base_h;
    auto off_sched    = reinterpret_cast<char*>(&h_pack.scheduler) - base_h;

    char *base_d = reinterpret_cast<char*>(d_pack);

    auto d_params = reinterpret_cast<Flash_fwd_params*>(base_d + off_params);
    auto d_mainloop_params =
        reinterpret_cast<typename CollectiveMainloop::Params *>(base_d + off_mainloop);
    auto d_epilogue_params =
        reinterpret_cast<typename CollectiveEpilogue::Params *>(base_d + off_epilogue);
    auto d_scheduler_params =
        reinterpret_cast<typename Scheduler::Params *>(base_d + off_sched);

    cutlass::launch_kernel_on_cluster(
        launch_params, kernel,
        d_params, d_mainloop_params, d_epilogue_params, d_scheduler_params);

    C10_CUDA_CHECK(cudaFreeAsync(d_pack, stream));

#else
    cutlass::launch_kernel_on_cluster(launch_params, kernel, params, mainloop_params, epilogue_params, scheduler_params);
#endif

    C10_CUDA_KERNEL_LAUNCH_CHECK();""",
    "MSVC device-side parameter packing", "launch.h")
lh_file.write_text(lh)

print("sageattn3 patch: done")
