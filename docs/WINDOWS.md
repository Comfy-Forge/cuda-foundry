# Windows

`docs/ARCHITECTURE.md` describes the system as built and measured on linux-64.
This file is the win-64 delta: what carries over unchanged, what has no Windows
analogue and therefore needs a *different* mechanism rather than a ported one,
and — stated plainly — where the guarantee is weaker on Windows than on Linux.

Everything here marked *measured* was established against real packages and
real upstream source. Everything not so marked is design, and says so.

## Where Windows actually stands today

The two halves are not in the same place, and an earlier note in this repo's
history had it backwards:

| half | status |
|---|---|
| **wheel** | **Ported from a live system.** `cuda-wheels` builds win-64 wheels today: `audit.py` has a `windows` lane, `natten` carries `arch_list_by_cuda_windows`, `fused_ssim` carries a real Windows source patch, `spconv` declares `windows` in its platform list, and `verify_wheel.py` has Windows-specific branches. This is working code to port, not a design to invent. |
| **conda** | **New.** `conda-cuda-packages` is linux-64 only; its `win` branch exits 1. Every conda-side Windows fact below is either measured here or explicitly flagged as unproven. |

The prerequisite is satisfied: `conda-torch` publishes win-64. *Measured* —
`https://comfy-forge.github.io/conda-torch/win-64/repodata.json` holds 527
packages (413 `pytorch`, 89 `libtorch`, 25 `libcudnn`), including
`pytorch-2.8.0-cuda128_repack_py312_h2bed46fa_*` and
`pytorch-2.8.0-cuda128_mkl_py312_hc0cb929_302`.

A full win-64 host solve against it succeeds. *Measured* — `python 3.12.*` plus
`pytorch 2.8.* cuda128_*` over `[conda-torch, conda-forge]` resolves, given
`[system-requirements] cuda = "12.8"`; without the CUDA virtual package the only
failure is `__cuda *, for which no candidates were found`, which is a property
of the solving machine and not of the channel.

## The host compiler: named but not constrained, again

The Linux side records that `cuda_compiler_version` names the build string and
constrains nothing, so every cell compiled against CUDA 12.9 regardless of its
label. Windows has the same shape of bug pointed the other way — the CUDA
package constrains the *host compiler*, invisibly:

**`cuda-nvcc` on win-64 hard-depends on `vs2019_win-64`, unversioned.**
*Measured* — `cuda-nvcc-12.8.93-h8f04d04_2.conda` declares exactly
`['cuda-nvcc_win-64 12.8.93.*', 'vs2019_win-64']`. Neither `cuda-nvcc_win-64`
nor `cuda-nvcc-dev_win-64` nor `cuda-nvcc-tools` mentions any MSVC beyond
`vc >=14.2,<15`, which spans 19.2x through 19.5x and constrains nothing useful.

So `cuda-nvcc 12.8.*` in `build:` silently selects **MSVC 19.29 (VS2019)** —
while upstream's own torch 2.8 Windows wheels are built with the VS2022
toolset. And asking for `vs2022_win-64` as well does not replace it: *measured*,
both land in one prefix, and which `cl.exe` wins is decided by `activate.d`
running `vs2019_compiler_vars.bat` before `vs2022_compiler_vars.bat` — correct
today by an accident of the year in the filename. (The `~` prefix on
`~cuda-nvcc_activate.bat` is *not* an accident: it sorts after letters, so
nvcc's activation deliberately runs last.)

**Therefore: depend on `cuda-nvcc_win-64`, never the `cuda-nvcc` metapackage,
and name the MSVC activation package explicitly.** *Measured* — that solve
yields exactly one MSVC (`vs2022_win-64` 19.44.35207) and `cuda-nvcc_win-64`
12.8.93, with no `vs2019_win-64` anywhere in the lock.

## The MSVC window, which is the Windows ceiling table

Linux pins `gcc <N` per CUDA line, read out of conda-forge repodata. Windows has
no such declaration — the real limit is compiled into nvcc, in
`crt/host_config.h`:

```c
#if _MSC_VER < 1910 || _MSC_VER >= 1950
#error -- unsupported Microsoft Visual Studio version!
```

`tools/msvc_ceiling.py` reads that guard out of the shipped headers rather than
pasting the numbers, for the same reason the gcc table is computed: the values
move every CUDA minor, and a stale pasted row looks exactly like a correct one.
Its output, *measured*:

| CUDA | `_MSC_VER` window | pick | usable |
|---|---|---|---|
| 12.0–12.3 | `[1910, 1940)` | `vs2019_win-64` | vs2019 only |
| 12.4–12.9 | `[1910, 1950)` | `vs2022_win-64` | vs2019, vs2022 |
| 13.0–13.1 | `[1920, 1950)` | `vs2022_win-64` | vs2019, vs2022 |
| 13.2–13.3 | `[1920, 1960)` | `vs2022_win-64` | vs2019, vs2022 |

conda-forge ships `vs2019_win-64` at `_MSC_VER` 1929 and `vs2022_win-64` at
1944. The consequence that makes this a table and not a constant: **1944 is
outside the window for CUDA ≤ 12.3.** Pinning one MSVC across the matrix would
be wrong at both ends — vs2019 everywhere silently mismatches upstream's toolset
on modern lines, vs2022 everywhere hard-errors in nvcc on old ones.

Our target cell (cu12.8) picks `vs2022_win-64`, which is both legal for nvcc and
the toolset family upstream torch used.

## `DISTUTILS_USE_SDK=1` is mandatory, not advisory

*Measured*, from pytorch v2.8.0 `torch/utils/cpp_extension.py`,
`BuildExtension._check_abi`:

```python
if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' in os.environ and 'DISTUTILS_USE_SDK' not in os.environ:
    raise UserWarning(msg)
```

It **raises**. conda-forge's MSVC activation runs `vcvarsall`, which sets
`VSCMD_ARG_TGT_ARCH`; so every Windows build in this repo is in exactly the
state that raises unless the build script exports `DISTUTILS_USE_SDK=1`. This is
not a warning to be tidied up later — without it, no cell builds at all.

Torch's own MSVC floor is permissive and offers no protection here:
`MINIMUM_MSVC_VERSION = (19, 0, 24215)`, which both 19.29 and 19.44 clear.

## The source-build guarantee is weaker on Windows. Say so.

L1 on Linux is a seccomp BPF filter denying `AF_INET`/`AF_INET6`, inherited by
every child and irreversible. **Windows has no equivalent mechanism available to
an unprivileged build**, and this repo should not pretend otherwise. What
remains on win-64:

- **L2 (declared force-source flags)** — carries over unchanged. It is
  per-package and declarative, and it is the layer that actually addresses the
  original defect (`FLASH_ATTENTION_FORCE_BUILD` letting pip fetch a prebuilt
  wheel).
- **L3 (compile ledger)** — carries over unchanged, and is the strongest layer
  on Windows precisely because it is evidence about the artifact rather than a
  restriction on the build. An extension module with no matching compile record
  fails the gate whether or not the network was reachable.
- **L4 (canary)** — must be rewritten. Its Linux assertions (`AF_INET` refused,
  `AF_UNIX` intact) are assertions about seccomp. A Windows canary can only
  assert what a Windows mechanism provides.
- **L1** — absent. A per-executable outbound `netsh advfirewall` block rule is
  possible on GitHub's Windows runners, but it keys on the program path rather
  than being inherited by descendants, so it is materially weaker than the
  seccomp filter and must not be described as the same guarantee.

Net: on Windows the claim is "no package ships a binary without a compile record
proving we built it" (L3), not "the compile had no network" (L1). Those are
different claims and the second is not available.

## Gates with no Windows analogue

Several publish-gate checks in `docs/ARCHITECTURE.md` are Linux-shaped. Each
needs a Windows counterpart that checks the same *property*, not a port of the
same *mechanism*:

| Linux gate | Windows counterpart |
|---|---|
| `$ORIGIN`-relative RPATHs, no absolute or empty entries | No RPATH exists in PE. The property — "this binary finds its libraries without depending on the build machine's layout" — becomes: every DLL in the `.pyd`'s import table resolves to the conda prefix (`Library/bin`) or to torch's own `lib` directory, and nothing resolves to a build-time path. |
| computed glibc floor | No glibc. The equivalent declaration is the `vc14_runtime` dependency, which conda-forge's MSVC `run_exports` supplies. |
| no vendored `libtorch` | Same property, different evidence: no `torch_*.dll` / `c10*.dll` inside the artifact. |
| no `nvcc.real` | The ccache seat-swap is a Linux mechanism; on Windows the compiler-cache story is `sccache`, which does not move the compiler aside. The gate still belongs there — cheap, and it fails closed. |
| SASS arch check | Carries over: `cuobjdump` is cross-platform, and `cuda-wheels`' `resolve_windows_arch_list` already exists because the Windows arch list genuinely differs from x86 Linux. |

## Wheels on Windows vendor nothing

*Measured*, from `cuda-wheels/scripts/verify_wheel.py`: on Windows "the wheel
bundles no DLL at all" — there is no `delvewheel` step and no `.libs` directory.
The extension `.pyd` links torch's DLLs and finds them because `import torch`
calls `os.add_dll_directory` on `torch/lib` before any extension loads.

This is a real simplification versus Linux, where `auditwheel repair` grafts
dependencies into `<pkg>.libs/`. It also relocates a problem rather than
removing it: a package linking a **non-torch** third-party library (torchvision
and its jpeg/png/webp codecs) has no vendoring step to carry it, so on Windows
those codecs must be statically linked into the extension or genuinely disabled.

`delvewheel` 1.13.1 and `sccache` 0.17.0 are both available on conda-forge
win-64 (*measured*) if either is needed; neither is currently in the design.

## The codec trap almost certainly recurs, differently

The Linux lesson worth repeating here: torchvision stats its codec headers under
`BUILD_PREFIX`/`CONDA_PREFIX` rather than rattler-build's `$PREFIX`, so jpeg and
webp were silently disabled while `run_exports` still claimed them, and only
running `decode_jpeg` revealed it.

On Windows the same detection code runs against different path conventions
(`Library/include`, `Library/lib`, `.lib` rather than `.so`), so the failure
mode is at least as likely and the structural checks are just as blind to it.
**The win-64 acceptance test must run the op, not import the module.** This is
stated as expectation, not measurement — it has not yet been observed on
win-64, and it should be checked before it is believed.

## Scope for the first Windows cell

py3.12 / CUDA 12.8 / torch 2.8.0 / win-64, the same five packages as linux-64,
on `windows-2022` runners (`defaults/policy.yml` already maps this). The build
environment that follows from the measurements above:

- `build:` — `cuda-nvcc_win-64 12.8.*`, `cuda-version 12.8.*`,
  `vs2022_win-64`, `ninja`; **not** `cuda-nvcc`
- `build.script.env` — `DISTUTILS_USE_SDK=1`
- `host:` — as linux-64, against `pytorch 2.8.* cuda128_*` from conda-torch
