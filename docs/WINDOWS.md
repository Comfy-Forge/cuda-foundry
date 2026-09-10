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
| no `nvcc.real` | The ccache seat-swap is a Linux mechanism. win-64 caches through torch's `PYTORCH_NVCC` hook and never moves the real compiler, so there is nothing to leave behind. The gate still belongs there — cheap, and it fails closed. |
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

## Two things the merge makes easier, and one it makes harder

**Easier: no Windows CUDA installer.** `cuda-wheels` gets its toolkit from a
system install, and its policy file says so — "adding a [CUDA] line requires
wiring its Windows installer URL in `.github/actions/setup-cuda` first (Linux
derives apt package names from the version and needs nothing)". Here CUDA comes
from conda on both platforms (`cuda-nvcc_win-64`), so that asymmetry disappears
and a new CUDA line costs nothing Windows-specific.

**Easier: the target cell is already in policy.** *Measured*, from
`cuda-wheels/defaults/python_cuda_torch_os_policy.yml`: `platforms` includes
`windows`, and cu12.8 / torch 2.8.0 / py3.12 is a live row with no
`pytorch_windows` override. (That key exists, but on exactly one row — cu12.9
pins Windows to torch 2.9.0 where Linux gets 2.9.1 — so it is a per-row
substitution, not a Windows torch floor.)

**Harder: patches can no longer gate on the host platform.**
`fetch_patched_sources.py` emits **one tarball per package**, consumed by every
platform's build. A patch that branches on `os.name` therefore bakes the
*fetching* machine's answer into the Windows build. The predecessor's patches do
branch on it — legitimately, because the wheel farm patches on the build machine
— and *13 of them* do (`natten`, `torchsparse`, `sageattention`, `cubvh`,
`gsplat`, the `ovoxel` family and others). Every one is a future port into this
repo and every one needs its conditional moved from the patch script into the C
source, where `#ifdef _WIN32` expresses it correctly.

`fused-ssim` is the worked example: its Windows prologue is applied
unconditionally and made inert off-Windows by `#ifdef _WIN32`. The same patch
carries the other half of that lesson — the predecessor's arch fix has a
substitution that has never matched this pinned revision, and it went unnoticed
because a whole-file before/after guard passes on a partial match. When porting
those 13, assert per substitution.

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

## What is wired, and what is still unproven

The win branch is no longer an `exit 1` stub. As of commits `d0398d6` and
`333be32` (where it landed by way of an unrelated broad `git add`, so their
messages do not mention it):

| piece | where |
|---|---|
| entry point | `scripts/build_snippets/build.bat` — thin, because rattler-build renders it through minijinja |
| the actual build | `scripts/build_snippets/build_win.py` — a sibling file copied next to the recipe like `nonet.py`, so it is never rendered |
| MSVC selection | the `c_compiler` variant, consumed by `compiler('c')`; table in `defaults/policy.yml` `host_msvc`, computed by `tools/msvc_ceiling.py` |
| win-64 variant config | `defaults/variants-win.yaml` — separate from `variants.yaml` because `c_stdlib` differs (`vs` against `sysroot`) and a variant config has no platform conditionals |
| toolkit | `cuda-nvcc_win-64`, never the `cuda-nvcc` metapackage |
| setuptools | `>=78,<84`, win-64 only |

**Solving is proven; building is not.** *Measured* — `rattler-build
--render-only --with-solve --target-platform win-64` against the live
`conda-torch` channel resolves all five packages at py3.12 / cu12.8 / torch
2.8.0, and linux-64 still resolves after the same template changes. That
retires the structural risk. It says nothing about whether the compile works:
there is no Windows machine here, so `build_win.py` has never been executed.
Everything it does — the MSVC window re-check, the ninja-log ledger, the wheel
handoff — is written from measurement but run for the first time on a runner.

Two things found only by attempting the solve, both worth keeping:

- **`cuda-nvtx` has no win-64 build at all.** conda-forge ships it for
  linux-64/aarch64/ppc64le; win-64 gets only `cuda-nvtx-dev`. Unconditional in
  `host:`, it made every win-64 cell UNSAT. Consistent with why it is there:
  torch 2.4/2.5 link `libnvToolsExt.so.1`, an ELF soname.
- **`build_env` values are shell syntax.** `package.yml` is written once for
  both platforms, and torchaudio declares `FFMPEG_ROOT: $PREFIX`. Copied into a
  `.bat` that would set the *literal* string `$PREFIX` — silently. The hook now
  translates `$VAR`/`${VAR}` to `%VAR%` and refuses anything with a dollar left
  in it. Path separators are deliberately not rewritten; Windows takes forward
  slashes, and a blanket conversion would corrupt non-path values.

**The compile ledger is different, and better.** With no wrapper in the nvcc
seat there is nothing to record invocations, so L3 on win-64 reads ninja's own
`.ninja_log` instead. That is arguably the better source: it is evidence about
what was *built* rather than what was *invoked*, and a prebuilt binary copied
into the source tree appears in it not at all — which is precisely the case L3
exists to catch. It does assume ninja was used, and fails loudly if extension
modules exist with no compiled objects behind them.

It also happens to be immune to the failure the Linux lane hit on torchaudio,
where a cache HIT means the wrapper never runs and the link job's ledger comes
out empty. ninja records an output whether or not ccache served it, so a win-64
link job that replays all 72 of its nvcc translation units still writes all 73.

## Sharding on Windows: there is no seat, and none is needed

This file used to say sharding could not be ported because "a `.bat` cannot
take an `.exe`'s place (PATHEXT puts `.EXE` first in any case)". The conclusion
about the seat was right; the reason was not, and the reason is what mattered,
because it made the seat look like the only door.

*Measured*, from ninja's `src/subprocess-win32.cc` and pytorch v2.8.0's
`torch/utils/cpp_extension.py`:

- **ninja does not run its commands through `cmd.exe`.** It hands the command
  line straight to `CreateProcess`, deliberately — "Do not prepend `cmd /c` on
  Windows, this breaks command lines greater than 8,191 chars".
  `CreateProcess` appends only `.exe` to an extensionless name, so **PATHEXT
  never gets a say at all** and nothing but a real executable can occupy the
  seat. That is a stronger statement than the old one, and it rules the `.bat`
  out on every path rather than only where the caller spells the extension.
- **the seat is not the only door.** `_write_ninja_file` reads `PYTORCH_NVCC`
  and writes its value **verbatim** as the ninja `nvcc` variable
  (`cpp_extension.py:2840`, with the upstream comment "user can set nvcc
  compiler with ccache using the environment variable here"). ninja does no
  tokenising of its own and `CreateProcess` takes the executable from the front
  of the command line, so a **two-token** value is a launcher:

      PYTORCH_NVCC = "<...>\ccache.exe <...>\nvcc.exe"

  which is byte-for-byte the invocation the Linux wrapper ends up making,
  `ccache <real nvcc> <args>`. No seat swap, no `nvcc.real` passenger in
  `$PREFIX`, nothing to restore on exit, and nothing that depends on PATHEXT.

ccache's own masquerade mode — copy `ccache.exe` to `nvcc.exe` and let it find
the real compiler on PATH, skipping itself — also works and was the first
candidate. It is not used, because it still needs the seat (or a PATH entry) to
be reached at all, and because "which nvcc did it actually resolve?" then
becomes a question the build answers at runtime instead of a path this repo
writes down. `CMAKE_CUDA_COMPILER_LAUNCHER` was the third candidate and does
not apply here at all: flash-attn is a `BuildExtension` + ninja build with no
CMake anywhere.

### What partitions the work, since nothing sits in the seat

On Linux the wrapper is also where the shard partition happens: it sees one
translation unit per invocation and emits an empty object for the ones outside
its slice. A launcher cannot do that — it is `ccache`, not our code — so the
win-64 partition happens one step earlier, on the **source files**, before
`pip wheel` runs. Files outside this shard's slice are overwritten with a stub.

That is a strictly weaker mechanism: it has to be *told* which files are
translation units, which is `shard_sources` in `package.yml`. Two things make
it safe to rely on:

- **a mistake cannot reach an artifact.** A shard is never published; its only
  output is a ccache directory. Anything the partition gets wrong is compiled
  for real by the link job, whose lookup then misses, and the zero-miss gate
  fails the build. The failure mode of a bad glob is a red link job, never a
  wrong `.conda`.
- **the declaration is checked against `.ninja_log`.** After the build,
  `check_declared_tus()` requires the declared set to **equal** the set ninja
  actually compiled. Declared-but-not-built and built-but-not-declared are both
  hard errors, so a glob that matched too little, too much or nothing at all
  cannot quietly describe a different build. flash-attn declares 73 entries
  (`csrc/flash_attn/flash_api.cpp` plus `csrc/flash_attn/src/*.cu`) and ninja
  compiles exactly 73.

One asymmetry is deliberate. **C++ translation units are stubbed in every
shard, never partitioned.** torch writes the literal string `cl` into the ninja
compile rule — `compiler_name = "$cxx" if IS_HIP_EXTENSION else "cl"` — so
`CXX` does not reach it and only a real `cl.exe` earlier on PATH could
intercept it. Nothing here does, so a C++ TU is not cached, and a shard that
compiled one would just be slower: flash-attn's single `flash_api.cpp` took
**16 minutes** in run 34195844795. It is stubbed in every shard and compiled
once, for real, in the link job.

Stubbing it needs one piece of care that is not obvious. `PYBIND11_MODULE`
expands to `PyInit_<TORCH_EXTENSION_NAME>`, and distutils passes
`/EXPORT:PyInit_<name>` to `link.exe` regardless of what the sources contain —
so a shard whose module TU is stubbed away has an unresolved export and dies at
link with LNK2001, having compiled its slice perfectly. The stub therefore
defines that symbol itself, pasting the `-DTORCH_EXTENSION_NAME` already on the
compile line, `extern "C"` so the C++ TU does not mangle it.
`LINK=/FORCE:UNRESOLVED` is set as a backstop for a package whose entry point
is somewhere else — scoped to shard mode, so a published artifact can never be
linked with it. (`/FORCE:UNRESOLVED` is documented as ignored when the *entry
point* is unresolved; for a DLL that is the CRT's `_DllMainCRTStartup`, which
is always resolved, so it does still apply to an export.)

### What the zero-miss gate covers, stated exactly

The link job asserts `cache_miss == 0` and, separately, that the number of
ccache lookups **equals** the number of `cuda_compile` edges ninja ran. The
second half is what stops the first from passing vacuously: without it a build
where `PYTORCH_NVCC` never reached ninja would report zero misses because it
reported zero of everything. `direct_cache_miss` is deliberately not counted as
a miss — a TU that misses the direct lookup and then hits the preprocessed one
is a hit — and `scripts/test_ninja_ledger.py` carries the control for that.

The honest limit: the gate covers the **72 nvcc** translation units, not the
one C++ one, because nothing caches that one. It is recompiled in the link job
and it appears in the ledger like everything else.

### What is NOT sharded on win-64, and why

`sharding: 1` means one shard, not none, so on linux-64 every cell runs a
compile-shard job and then a link job that replays it. win-64 shards only where
`sharding > 1`. The four win-64 packages that publish today (`cc-torch`,
`fused-ssim`, `torchvision`, `torchaudio`) fit in one job, and making them
compile everything twice on the slowest runners in the fleet — to prove a
handoff they do not need — would trade real wall clock for nothing. Their path
is unchanged: `CUW_MODE=full`, no ccache, no `PYTORCH_NVCC`, byte-identical to
what shipped.
## Scope for the first Windows cell

py3.12 / CUDA 12.8 / torch 2.8.0 / win-64, the same five packages as linux-64,
on `windows-2022` runners (`defaults/policy.yml` already maps this). The build
environment that follows from the measurements above:

- `build:` — `cuda-nvcc_win-64 12.8.*`, `cuda-version 12.8.*`,
  `vs2022_win-64`, `ninja`; **not** `cuda-nvcc`
- `build.script.env` — `DISTUTILS_USE_SDK=1`
- `host:` — as linux-64, against `pytorch 2.8.* cuda128_*` from conda-torch

## A wheel filename cannot express a rebuild

Not Windows-specific, but found here and worth writing down where the
publishing rules live.

A `.conda`'s build string carries the build number — `..._h6651153_0`,
`..._h6651153_1` — so a rebuild of the same cell publishes under a new name and
the immutability rule holds without anyone having to think about it. A wheel's
filename has no such component: `fused_ssim-0.0.0+cu128torch2.8-cp312-cp312-win_amd64.whl`
is what *every* build of that cell is called, whatever its build number.

`publish-wheels.yml` compares sha256 and hard-errors when the bytes differ,
which is right — an index anchor carries `#sha256=` and a lockfile pins the
bytes. But the two facts together mean **a legitimately rebuilt wheel cannot be
published at all** without first deleting the published asset, which is the one
thing the immutability rule exists to forbid.

That is currently live: the four linux-64 wheels on the release are dev-box
builds, and the CI builds of those same cells will produce the same four
filenames with different bytes. They must be swapped, and the swap is a
deletion.

Two ways out, neither taken yet:

- put the build number in the local version tag (`+cu128torch2.8.1`), which
  makes the filename unique but changes the version every consumer sees and
  what `pip freeze` prints;
- accept that a wheel is replaceable where a `.conda` is not, and say so
  explicitly rather than having `publish-wheels.yml` refuse it.

The first is the honest one and the second is the cheap one. Deciding is a
publishing decision, not a build one.

## known_bad.json covers `.conda` only

`known_bad.json` is keyed by subdir and then by exact `.conda` filename, which
is the right shape for the channel: a published artifact cannot be withdrawn,
so the list is how someone holding a lockfile discovers that what they pinned
is defective (`tools/check_lock.py`).

The wheel built from the same compile has no equivalent. When
`torchvision-0.23.0-cuda128_torch28_py312_h6651153_1.conda` was recorded bad
for shipping without three of its four codecs, the wheel from that identical
build — `torchvision-0.23.0+cu128torch2.8-cp312-cp312-win_amd64.whl` — stayed
on the index with nothing saying so.

Supersession does most of the work now that wheels carry a PEP 427 build tag:
an absent tag sorts below any present one, so the rebuilt `-2-` wheel wins for
anyone resolving. What it does not do is tell someone who pinned the old file
that they should move. Either `known_bad.json` grows a wheel section, or the
wheel index needs its own way to say it — recorded, not decided.
