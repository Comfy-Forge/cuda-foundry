"""Patch NATTEN v0.21.6 for the foundry build.

Ported from cuda-wheels (packages/natten/patches/natten.py) and re-read against
the pinned rev. Runs with cwd = the cloned source, once, before the tarball is
sealed: nothing here may depend on the cell or the host. Everything that does
depend on them -- the torch version, the arch list, the shard slice, the
platform -- is written INTO setup.py or CMakeLists.txt and decided at build
time, where those facts exist.

What is applied, each substitution asserting on its own:

  pyproject.toml
    * `where = ["src/"]` -> `["src"]`: newer setuptools' convert_path rejects
      the trailing slash on Windows before the build starts.
    * `name = "NATTEN"` -> `"natten"`: pyproject says NATTEN, setup.py says
      natten, and which one wins depends on the setuptools version, so the
      same cell produced NATTEN-*.whl on some runners and natten-*.whl on
      others. PEP 503 normalises to lowercase anyway.

  csrc/CMakeLists.txt
    * drop -Xcompiler=-Wconversion and -Xcompiler=-fno-strict-aliasing: MSVC
      reads -W<digit> as a warning level (D8021) and has no strict-aliasing
      knob; neither is load-bearing for the kernels.
    * Hopper and Blackwell autogen kernels move into per-arch OBJECT libraries
      (CUDA_ARCHITECTURES "90a-real" / "100a-real"). Upstream compiles every
      .cu against the whole arch list, which for these arch-specific families
      is 5-7x wasted nvcc work per file. set_source_files_properties(...
      CUDA_ARCHITECTURES) silently no-ops (target-scoped property), so
      OBJECT libraries are the working spelling. A shard may hold zero files
      of a family, so the targets are created only when their lists are
      non-empty, and consumed only if(TARGET ...).
    * MSVC warning suppression (/wd...) and nvcc --diag-suppress lists on
      every target: CUTLASS instantiations under MSVC produce ~80k warning
      lines per job otherwise, none actionable in NATTEN.
    * RPATH pinned to $ORIGIN: cmake bakes the build-tree torch/lib into the
      .so, a path that exists on no other machine. The extension resolves
      torch from the already-loaded libtorch, as every torch extension does.
    * CMAKE_CXX_FLAGS' hardcoded -std=c++17 tracks CXX_STD, and setup.py
      passes -DCXX_STD from the torch it builds against (below).

  csrc/include/natten/helpers.h
    * `not x.is_sparse()` -> `!x.is_sparse()`: MSVC does not know the
      alternative token without /permissive-.

  setup.py
    * a shim bridging the cell's conventions to NATTEN's own env vars:
      TORCH_CUDA_ARCH_LIST (space-separated, +PTX suffixes) ->
      NATTEN_CUDA_ARCH (semicolon-separated, no suffix), remembering which
      archs carried +PTX; MAX_JOBS -> NATTEN_N_WORKERS; NATTEN_BUILD_DIR
      pinned to build/natten_cmake inside the source tree (the default is a
      random tempdir, and the win-64 ledger reads .ninja_log from under the
      source tree).
    * arch_list_to_cmake_tags emits `-virtual` for exactly the +PTX archs.
      Upstream's NATTEN_BUILD_WITH_PTX is all-or-nothing and maps 90/100 to
      the arch-conditional `a` form, whose PTX loads only on that arch.
    * the shard partition (package.yml `shard_partition: source`): after
      autogen, when CUW_SHARD_COUNT > 0, delete every .cu outside this
      shard's round-robin slice of the sorted list of autogen kernels plus
      csrc/src/*.cu. The link job runs with CUW_SHARD_COUNT=0 and builds the
      full set from the merged cache.
    * -DCXX_STD=20 for torch >= 2.12, else 17 (torch's own MSVC default
      flipped at 2.12; older torch fails at c++20 under nvcc's EDG front end).

Not carried from the farm: its configure-skip/CMAKE_SUPPRESS_REGENERATION
pair (a resume mechanism for its sequential-checkpoint lane; every job here
configures once, fresh) and its Windows-shard /FORCE:UNRESOLVED cmake block
(build_win.py sets LINK=/FORCE:UNRESOLVED in shard mode, which link.exe reads
however it is invoked).
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require  # noqa: E402

MARKER = "# patched-by: cuda-foundry natten"


def sub_once(text: str, old: str, new: str, what: str, path: str) -> str:
    require(text.count(old) == 1,
            f"natten: {what} matched {text.count(old)} time(s) in {path}, "
            f"expected exactly 1 -- upstream changed at this rev; re-read it")
    print(f"natten patch: {path}: {what}")
    return text.replace(old, new, 1)


setup_file = pathlib.Path("setup.py")
if MARKER in setup_file.read_text():
    print("natten patch: already applied")
    sys.exit(0)

# ── pyproject.toml ─────────────────────────────────────────────────────────
pp = pathlib.Path("pyproject.toml")
t = pp.read_text()
t = sub_once(t, 'where = ["src/"]', 'where = ["src"]', "packages.find.where without the trailing slash", "pyproject.toml")
t = sub_once(t, 'name = "NATTEN"', 'name = "natten"', "[project] name lowercased", "pyproject.toml")
pp.write_text(t)

# ── csrc/CMakeLists.txt ────────────────────────────────────────────────────
cm = pathlib.Path("csrc/CMakeLists.txt")
t = cm.read_text()
for line in ('set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wconversion")',
             'set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-strict-aliasing")'):
    t = sub_once(t, line + "\n", "", f"GCC-only flag removed: {line.split()[-1].rstrip(')')}", "csrc/CMakeLists.txt")

# Torch's headers and import libraries, from torch itself rather than from a
# hardcoded layout. Upstream assumes ${TORCH_DIR}/include, which is the pip
# wheel's layout and conda-torch's repack; the conda-forge-mirrored builds
# (pytorch 2.8.0 cuda128_mkl_*_302, which the solver prefers as the higher
# build number) keep every header under %PREFIX%\Library\include and ship
# no site-packages/torch/include at all. Measured on win-64 run 34587044404:
# cl found Library\include\torch\extension.h through INCLUDE and then could
# not find <torch/all.h>, because that lives in .../csrc/api/include, which
# only the hardcoded (nonexistent) path would have added. torch's own
# cpp_extension.include_paths() / library_paths() know the installed layout
# -- conda-forge patches them for exactly this -- and are what every
# setuptools-driven torch extension already compiles with.
t = sub_once(
    t,
    'set(TORCH_INCLUDE_DIRS "${TORCH_DIR}/include" "${TORCH_DIR}/include/torch/csrc/api/include")',
    '''execute_process(COMMAND ${PYTHON_PATH} "-c" "import torch.utils.cpp_extension as c; print(';'.join(c.include_paths()), end='')"
                RESULT_VARIABLE _PYTHON_SUCCESS
                OUTPUT_VARIABLE TORCH_INCLUDE_DIRS)
if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "torch.utils.cpp_extension.include_paths() failed.")
endif()
execute_process(COMMAND ${PYTHON_PATH} "-c" "import torch.utils.cpp_extension as c; print(';'.join(c.library_paths()), end='')"
                RESULT_VARIABLE _PYTHON_SUCCESS
                OUTPUT_VARIABLE TORCH_LIBRARY_DIRS)
if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "torch.utils.cpp_extension.library_paths() failed.")
endif()
message("cuda-foundry: torch library dirs: ${TORCH_LIBRARY_DIRS}")
link_directories(${TORCH_LIBRARY_DIRS})''',
    "torch include/library dirs taken from torch.utils.cpp_extension, not a hardcoded layout",
    "csrc/CMakeLists.txt")

t = sub_once(
    t,
    'set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17")',
    'set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++${CXX_STD}")',
    "CMAKE_CXX_FLAGS' standard tracks CXX_STD (set by setup.py from the torch built against)",
    "csrc/CMakeLists.txt")

OBJECT_LIBS = '''# --- cuda-foundry: arch-specific OBJECT libraries (injected) ---------------
# Blackwell DC kernels for sm_100a only, Hopper kernels for sm_90a only. Both
# families use arch-specific instructions (TMA, wgmma, tcgen05) and are
# dispatched to only on that arch (checks.py rejects any other device_cc
# before the host entry is called), so compiling them against the whole
# CUDA_ARCHITECTURES list is pure waste. The `a` suffix is required for
# CUTLASS to enable those features; `-real` keeps them SASS-only.
# A shard (package.yml shard_partition: source) may legitimately hold ZERO
# files of a family: an empty add_library() is a hard configure error, so
# the target exists only when its list is non-empty. Bare variable names in
# if(), never ${VAR}: an undefined NATTEN_WITH_*_FNA would otherwise leave
# `if(AND (...))` behind and cmake dies with "Unknown arguments specified".
if(NATTEN_WITH_BLACKWELL_FNA AND (AUTOGEN_BLACKWELL_FNA OR AUTOGEN_BLACKWELL_FMHA))
    list(REMOVE_ITEM ALL_SOURCES ${AUTOGEN_BLACKWELL_FNA} ${AUTOGEN_BLACKWELL_FMHA})
    add_library(natten_blackwell OBJECT
        ${AUTOGEN_BLACKWELL_FNA} ${AUTOGEN_BLACKWELL_FMHA})
    set_target_properties(natten_blackwell PROPERTIES
        CUDA_ARCHITECTURES "100a-real"
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(natten_blackwell SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    target_include_directories(natten_blackwell PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/autogen/include
    )
    list(LENGTH AUTOGEN_BLACKWELL_FNA  _cuw_n_bw_fna)
    list(LENGTH AUTOGEN_BLACKWELL_FMHA _cuw_n_bw_fmha)
    math(EXPR _cuw_n_bw "${_cuw_n_bw_fna} + ${_cuw_n_bw_fmha}")
    message(STATUS "cuda-foundry: ${_cuw_n_bw} Blackwell sources -> natten_blackwell OBJECT (CUDA_ARCHITECTURES=100a-real)")
endif()
if(NATTEN_WITH_HOPPER_FNA AND (AUTOGEN_HOPPER_FNA OR AUTOGEN_HOPPER_FMHA))
    list(REMOVE_ITEM ALL_SOURCES ${AUTOGEN_HOPPER_FNA} ${AUTOGEN_HOPPER_FMHA})
    add_library(natten_hopper OBJECT
        ${AUTOGEN_HOPPER_FNA} ${AUTOGEN_HOPPER_FMHA})
    set_target_properties(natten_hopper PROPERTIES
        CUDA_ARCHITECTURES "90a-real"
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(natten_hopper SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    target_include_directories(natten_hopper PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/autogen/include
    )
    list(LENGTH AUTOGEN_HOPPER_FNA  _cuw_n_hp_fna)
    list(LENGTH AUTOGEN_HOPPER_FMHA _cuw_n_hp_fmha)
    math(EXPR _cuw_n_hp "${_cuw_n_hp_fna} + ${_cuw_n_hp_fmha}")
    message(STATUS "cuda-foundry: ${_cuw_n_hp} Hopper sources -> natten_hopper OBJECT (CUDA_ARCHITECTURES=90a-real)")
endif()
# --- end cuda-foundry arch-specific OBJECT libraries ------------------------

add_library(natten SHARED ${ALL_SOURCES})

# if(TARGET ...), NOT if(NATTEN_WITH_*_FNA): the flag is set from the arch
# list and is true in every shard; the target exists only if THIS shard
# received sources of that family.
if(TARGET natten_blackwell)
    target_link_libraries(natten PRIVATE $<TARGET_OBJECTS:natten_blackwell>)
endif()
if(TARGET natten_hopper)
    target_link_libraries(natten PRIVATE $<TARGET_OBJECTS:natten_hopper>)
endif()'''
t = sub_once(t, "add_library(natten SHARED ${ALL_SOURCES})", OBJECT_LIBS,
             "Hopper/Blackwell autogen kernels split into per-arch OBJECT libraries",
             "csrc/CMakeLists.txt")

t += '''

# --- cuda-foundry: MSVC noise suppression (injected) ------------------------
# CUTLASS instantiations under MSVC produce tens of thousands of C4514/C4100/
# C4623/... lines per job (measured at 82k lines in one farm job), none
# actionable in NATTEN. Applied to every target; NATTEN's own diagnostics
# stay at the /W3 default.
if(NATTEN_IS_WINDOWS)
    set(_cuw_msvc_wd_codes
        4514 4100 4623 4624 4577 4067 4068 4505 4127
        4711 4820 4061 4251 4710 4365 4626 5027 4996 4244 4668 4625 5039 4619 4324 4267
        4275 4686 4355 4800 5031 5246 5026 4582 4583 4018 4242 4310 4459 4201 4189 4191
        5219 5045 4702 4868 4388 4296 4464
    )
    foreach(_cuw_c ${_cuw_msvc_wd_codes})
        foreach(_cuw_t natten natten_blackwell natten_hopper)
            if(TARGET ${_cuw_t})
                target_compile_options(${_cuw_t} PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:/wd${_cuw_c}>
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd${_cuw_c}>
                )
            endif()
        endforeach()
    endforeach()
    message(STATUS "cuda-foundry: MSVC warnings suppressed on natten + per-arch OBJECT libs")
endif()
# --- end cuda-foundry MSVC noise suppression --------------------------------

# --- cuda-foundry: nvcc diagnostic suppression (injected) -------------------
# nvcc's own #NNN-D diagnostics (not MSVC's): 221 (fna_collective_softmax.hpp
# casts -1e300 to float as a sentinel), 20011 (CUTLASS host-function-from-
# host-device noise), and the dllexport/dllimport class-interface family that
# torch's headers trigger in volume on Windows.
set(_cuw_nvcc_diag_codes 221 20011 1394 1388 1390 550)
foreach(_cuw_d ${_cuw_nvcc_diag_codes})
    foreach(_cuw_t natten natten_blackwell natten_hopper)
        if(TARGET ${_cuw_t})
            target_compile_options(${_cuw_t} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=${_cuw_d}>
            )
        endif()
    endforeach()
endforeach()
# --- end cuda-foundry nvcc diagnostic suppression ---------------------------

# --- cuda-foundry: RPATH hygiene (injected) ---------------------------------
# cmake links torch through target_link_libraries and, by default, bakes the
# BUILD-TREE location of every linked library into the binary's RPATH -- a
# torch/lib path that exists only on the builder. rattler-build rewrites
# RPATHs when it packages the .conda, but the wheel is taken before that
# step, and auditwheel does not touch excluded (torch) libraries. The
# extension needs no torch RPATH: `import torch` has loaded libtorch into
# the process before the extension is imported, exactly as for every
# setuptools-built torch extension, which gets no RPATH at all.
set_target_properties(natten PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
# --- end cuda-foundry RPATH hygiene -----------------------------------------
'''
cm.write_text(t)
print("natten patch: csrc/CMakeLists.txt: MSVC/nvcc noise suppression and $ORIGIN RPATH appended")

# ── csrc/include/natten/helpers.h ──────────────────────────────────────────
hp = pathlib.Path("csrc/include/natten/helpers.h")
t = hp.read_text()
t = sub_once(t, "(not x.is_sparse(),", "(!x.is_sparse(),",
             "alternative token `not` replaced for MSVC", "csrc/include/natten/helpers.h")
hp.write_text(t)

# ── setup.py ───────────────────────────────────────────────────────────────
t = setup_file.read_text()

SHIM_ANCHOR = 'CUDA_ARCH = os.getenv("NATTEN_CUDA_ARCH", "")'
SHIM = '''# cuda-foundry shim: bridge the cell's conventions to NATTEN's own env vars.
# TORCH_CUDA_ARCH_LIST -> NATTEN_CUDA_ARCH (NATTEN wants ';' and no +PTX
# suffix; which archs carried +PTX is remembered for arch_list_to_cmake_tags),
# MAX_JOBS -> NATTEN_N_WORKERS, and NATTEN_BUILD_DIR pinned inside the tree.
if not os.getenv("NATTEN_CUDA_ARCH"):
    _torch_arch = os.getenv("TORCH_CUDA_ARCH_LIST", "")
    _raw = [p.strip() for p in _torch_arch.replace(";", " ").split() if p.strip()]
    _parts = [p.replace("+PTX", "").strip() for p in _raw]
    _parts = [p for p in _parts if p]
    if _parts:
        os.environ["NATTEN_CUDA_ARCH"] = ";".join(_parts)
    if not os.getenv("CUW_NATTEN_PTX_ARCH"):
        _ptx_i = []
        for _p in [p.replace("+PTX", "").strip() for p in _raw if "+PTX" in p]:
            try:
                _ptx_i.append(str(int(float(_p) * 10)))
            except ValueError:
                pass
        if _ptx_i:
            os.environ["CUW_NATTEN_PTX_ARCH"] = ";".join(_ptx_i)
            print(f"[cuda-foundry] PTX tail requested for sm_{_ptx_i}")
if not os.getenv("NATTEN_N_WORKERS"):
    _mj = os.getenv("MAX_JOBS", "")
    if _mj.isdigit() and int(_mj) > 0:
        os.environ["NATTEN_N_WORKERS"] = _mj
# NATTEN's default build dir is a per-run tempdir. Pin it inside the source
# tree: the win-64 compile ledger reads .ninja_log from under the tree, and
# a stable path keeps the shard and link jobs' compile lines identical. The
# directory must exist before setup.py reads the variable (it falls back to
# the tempdir otherwise).
if not os.getenv("NATTEN_BUILD_DIR"):
    _cuw_build_dir = os.path.abspath("build/natten_cmake")
    os.makedirs(_cuw_build_dir, exist_ok=True)
    os.environ["NATTEN_BUILD_DIR"] = _cuw_build_dir
    print(f"[cuda-foundry] NATTEN_BUILD_DIR={_cuw_build_dir}")
''' + SHIM_ANCHOR
t = sub_once(t, SHIM_ANCHOR, SHIM, "TORCH_CUDA_ARCH_LIST / MAX_JOBS / build-dir shim", "setup.py")

PTX_ANCHOR = '''    if WITH_PTX:
        ptx_tags = (
            "-virtual;".join(
                [str(x) if x not in [90, 100, 103] else f"{x}a" for x in arch_list]
            )
            + "-virtual"
        )

        return real_tags + ";" + ptx_tags
    return real_tags'''
PTX_NEW = '''    # cuda-foundry: emit `-virtual` (PTX) for exactly the archs the cell marked
    # +PTX. Upstream's WITH_PTX is all-or-nothing and maps 90/100/103 to the
    # arch-conditional `a` form, whose PTX loads only on that one arch and is
    # not a forward-compat tail at all.
    _cuw_ptx = []
    for _a in os.getenv("CUW_NATTEN_PTX_ARCH", "").replace(";", " ").split():
        try:
            _v = int(_a)
        except ValueError:
            continue
        if _v in arch_list and _v not in _cuw_ptx:
            _cuw_ptx.append(_v)
    if _cuw_ptx:
        _cuw_tags = "-virtual;".join(str(x) for x in _cuw_ptx) + "-virtual"
        print(f"[cuda-foundry] CUDA_ARCHITECTURES PTX tail: {_cuw_tags}")
        return real_tags + ";" + _cuw_tags
    if WITH_PTX:
        ptx_tags = (
            "-virtual;".join(
                [str(x) if x not in [90, 100, 103] else f"{x}a" for x in arch_list]
            )
            + "-virtual"
        )

        return real_tags + ";" + ptx_tags
    return real_tags'''
t = sub_once(t, PTX_ANCHOR, PTX_NEW, "arch_list_to_cmake_tags honours the +PTX archs", "setup.py")

AUTOGEN_ANCHOR = '''            autogen_kernel_instantitations(
                this_dir=this_dir,
                autogen_dir=autogen_dir,
                scripts_dir=scripts_dir,
                policy=AUTOGEN_POLICY,
                cuda_arch_list=cuda_arch_list,
            )'''
AUTOGEN_NEW = AUTOGEN_ANCHOR + '''

            # cuda-foundry shard partition (package.yml shard_partition: source).
            # When CUW_SHARD_COUNT > 0 this is a compile shard: keep only this
            # shard's round-robin slice of the sorted autogen kernels plus the
            # hand-written dispatch TUs under csrc/src (which pull in the same
            # CUTLASS headers and would otherwise be compiled by every shard),
            # and delete the rest so cmake builds 1/N of the tree. The link job
            # runs with CUW_SHARD_COUNT=0 and builds everything from the cache.
            _cuw_shard_count = int(os.environ.get("CUW_SHARD_COUNT", "0") or "0")
            if _cuw_shard_count > 0:
                import glob
                _cuw_shard_index = int(os.environ.get("CUW_SHARD_INDEX", "1") or "1")
                _cuw_all = sorted(glob.glob(path.join(autogen_dir, "src", "cuda", "**", "*.cu"),
                                            recursive=True))
                _cuw_shared = sorted(glob.glob(path.join(path.dirname(autogen_dir), "src", "*.cu")))
                _cuw_all = _cuw_all + _cuw_shared
                _cuw_kept = [f for i, f in enumerate(_cuw_all)
                             if i % _cuw_shard_count == _cuw_shard_index - 1]
                for _f in set(_cuw_all) - set(_cuw_kept):
                    os.remove(_f)
                print(f"[cuda-foundry natten shard {_cuw_shard_index}/{_cuw_shard_count}] "
                      f"kept {len(_cuw_kept)}/{len(_cuw_all)} .cu files "
                      f"({len([f for f in _cuw_kept if f in set(_cuw_shared)])}/{len(_cuw_shared)} "
                      f"dispatch TUs); deleted {len(_cuw_all) - len(_cuw_kept)}")
                if not _cuw_kept:
                    raise RuntimeError(
                        f"natten shard {_cuw_shard_index}/{_cuw_shard_count} has an empty "
                        f"slice of {len(_cuw_all)} files; lower `sharding` in package.yml")'''
t = sub_once(t, AUTOGEN_ANCHOR, AUTOGEN_NEW, "shard partition injected after autogen", "setup.py")

CMAKE_ARGS_ANCHOR = '''            cmake_args = [
                f"-DPYTHON_PATH={sys.executable}",'''
CMAKE_ARGS_NEW = '''            cmake_args = [
                # cuda-foundry: the win-64 shard/link lane hands the ccache
                # launcher over as a cmake argument (CUW_CMAKE_ARGS) rather than
                # the CMAKE_CUDA_COMPILER_LAUNCHER environment variable, which
                # would also reach cmake's own try_compile probes and put an
                # uncacheable TU in every job. Empty everywhere else.
                *__import__("shlex").split(os.environ.get("CUW_CMAKE_ARGS", "")),
                # cuda-foundry: the C++ standard the torch being built against
                # needs. torch's own cpp_extension flipped its MSVC default to
                # /std:c++20 at 2.12 (its headers require it from 2.13); torch
                # < 2.7 fails at c++20 under nvcc's EDG front end. CMakeLists
                # reads CXX_STD for both the host and the nvcc lines.
                f"-DCXX_STD={20 if torch_ver >= [2, 12] else 17}",
                f"-DPYTHON_PATH={sys.executable}",'''
t = sub_once(t, CMAKE_ARGS_ANCHOR, CMAKE_ARGS_NEW, "-DCXX_STD from the torch built against", "setup.py")

setup_file.write_text(MARKER + "\n" + t)

# ── prove what is on disk ───────────────────────────────────────────────────
final = setup_file.read_text()
for needle, what in (("CUW_NATTEN_PTX_ARCH", "the PTX shim"),
                     ("CUW_SHARD_COUNT", "the shard partition"),
                     ("-DCXX_STD=", "the C++ standard switch"),
                     ("CUW_CMAKE_ARGS", "the cmake-argument hook"),
                     ('os.environ["NATTEN_BUILD_DIR"]', "the build-dir pin")):
    require(needle in final, f"natten: {what} is NOT PRESENT in setup.py on disk")
import ast  # noqa: E402
ast.parse(final)
cmf = cm.read_text()
for needle, what in (("BUILD_WITH_INSTALL_RPATH TRUE", "the $ORIGIN RPATH"),
                     ("c.include_paths()", "the torch include-dir lookup"),
                     ('CUDA_ARCHITECTURES "100a-real"', "the Blackwell object library"),
                     ('CUDA_ARCHITECTURES "90a-real"', "the Hopper object library"),
                     ("-std=c++${CXX_STD}", "the tracked host standard")):
    require(needle in cmf, f"natten: {what} is NOT PRESENT in csrc/CMakeLists.txt on disk")
require("-Wconversion" not in cmf and "fno-strict-aliasing" not in cmf,
        "natten: a GCC-only flag survived in csrc/CMakeLists.txt")
print("natten patch: done")
