# Architecture

This repo is the merge of two working systems: `cuda-wheels` (42 packages
compiled into PyPI wheels, with sharding, per-package memory tuning and PEP 658
sidecars) and `conda-cuda-packages` (the same packages compiled into `.conda`
against the `conda-torch` channel). Everything below that is stated as fact was
measured in one of those two, usually after getting it wrong first. The
provenance matters: several of these look like arbitrary detail and are not.

## The central claim, and what must prove it

**One compile, three outputs.** The build produces a wheel; that wheel becomes
both the conda package and the published wheel:

```
pip wheel .                      ← ONE compile, in the conda env, against conda torch
  ├─ pip install <whl> → $PREFIX → rattler-build packages it → .conda
  ├─ auditwheel repair <whl>                                 → manylinux wheel
  └─ METADATA from the same dep list                         → PEP 658 sidecar
```

This is coherent because a conda package must **not** vendor its shared
libraries (it declares `libjpeg-turbo` and links `libjpeg.so.8` from the
prefix) while a manylinux wheel **must** vendor them — and `auditwheel repair`
is exactly the tool that converts the first into the second. The compile is
identical; only the packaging contract differs.

**Unproven until measured:** that a package compiled here is equivalent to the
same package compiled by the old wheel farm, which used pip torch and a
`setup-cuda` toolkit rather than conda torch and conda-forge's toolchain. They
*should* agree — conda-torch's `pytorch` is the PyPI wheel repacked, so the
libtorch linked against is byte-identical. But "should" is doing real work
there, and the mislabelling bug below is precisely the kind of thing that hides
in that gap. **Before the wheel index is cut over, build one package both ways
and diff the resulting extension `.so`.** Either it de-risks the merge or it
tells us something important.

## Compiled from source, in four layers

The defect this repo exists to prevent: the earlier experiment shipped
`flash_attn` conda packages that were never compiled here, because
`FLASH_ATTENTION_FORCE_BUILD: "0"` let `pip install .` fetch Dao-AILab's
prebuilt wheel. Note upstream compares `os.getenv(...) == "TRUE"` — that
experiment's own comment said "set to 1 to force-build", and `"1"` is a no-op.

| layer | mechanism |
|---|---|
| **L1 — no network during compile** | A seccomp filter (`scripts/build_snippets/nonet.py`) denying `AF_INET`/`AF_INET6` socket creation, installed immediately before the compile step. Unprivileged, inherited by every child, irreversible via `PR_SET_NO_NEW_PRIVS`, narrow enough that `AF_UNIX` still works, and it exits rather than proceeding if the filter will not install. **Not** rattler-build's `--sandbox`: that needs a separate `rattler-sandbox` binary plus unprivileged namespaces, and fails with `Operation not permitted` both in containers and on GitHub-hosted runners. Source is fetched and patched *outside* the sandbox, so nothing legitimate needs egress. |
| **L2 — declared force-source flags** | `force_source_build:` in `package.yml` for any upstream whose build can download a binary; the loader hard-errors when it is missing or set to a permissive value. |
| **L3 — compile ledger** | The compiler wrapper records every translation unit; a gate refuses any artifact holding an extension module with no matching compile record. Catches a binary vendored into the source tree, which L1 cannot see. |
| **L4 — canary** | A recipe that proves the guarantee *positively* and must SUCCEED: `AF_INET` refused, a real `pip download` refused, `AF_UNIX` intact. An earlier version inferred "network denied" from the build FAILING, and duly reported the guarantee in force during a run where the mechanism never started. |

A note on honesty in wording: this prevents *accidental* network use during
compile. It does not stop a file descriptor inherited from the parent, nor an
`AF_UNIX` proxy. Against the actual threat — pip transparently fetching
upstream's wheel — it is effective; do not call it more than that.

## Cells, and the mislabelling that hid in them

A cell is `(package, torch, cuda, python, platform)`. Three things about a cell
must be *constrained*, not merely named, and each was learned the hard way:

1. **The CUDA toolkit.** `cuda_compiler_version` names the build string and
   constrains nothing. Every cell in the predecessor repo compiled against CUDA
   12.9 regardless of its label, because pytorch pulls triton and conda-torch's
   triton 3.6.0 exists only for the cuda129 line. A cell stamped `cuda128` was a
   12.9 build. torchaudio is the one upstream that checks, and it refused to load
   beside its own torch. Pinning the toolkit in `host:` is UNSAT (measured); it
   belongs in `build:`, which solves separately and has no torch in it.
2. **The compiler ceiling**, which is a property of the CUDA line, from
   conda-forge repodata: CUDA 12.0–12.4 → `gcc <13`; 12.6 → `<14`; 12.8–12.9 →
   `<15`; 13.0–13.3 → `<16`. A flat `gcc 13` silently worked only because of
   (1) — the solver was quietly using 12.9 for cells labelled cu12.4.
3. **The include order.** Pinning the toolkit is not enough: the host env's
   activation put its `-I` at position 1 and the cell's at position 11.

## Packaging invariants

- **Never pin an exact torch build.** `run_exports` yields a minor range and
  torch's C++ ABI is minor-stable. An exact build pin would have been
  invalidated by conda-torch's 300-build republish wave.
- **Lock the flavour with a build glob**: `pytorch 2.11.* cuda128_*`. A minor
  range alone matches cu126/cu128/cu129/cu130 alike, and `cuda-version`
  constrains only the CUDA *major*. **Correction worth recording:** this is
  currently hand-written in every consumer because of a claim that
  `run_exports` cannot express a build — that claim is false. Only
  `pin_subpackage()` is that limited; a run_export is a MatchSpec string and can
  carry a glob, and conda-forge ships 12,860 such specs on linux-64 alone
  (`adios >=1.13.1,<1.13.2.0a0 mpi_mpich_*`). If conda-torch's `pytorch`
  exported `pytorch >=2.11.0,<2.12.0a0 cuda128_*`, every consumer here could
  delete its glob and inherit the lock — and a consumer that *forgot* it would
  still be correct. That change belongs in conda-torch.
- **Declare `__cuda` directly**, not only transitively through libtorch, so
  "does this package need a GPU?" is one line of the package's own metadata.
  `comfy-test lint --check accel` asks exactly that, on a bare checkout with
  nothing installed, and the alternative was caching a derived list into the
  manifest.
- **Declare `python_abi`.** A cp312 extension with an unbounded `python` may
  legally install into py3.10; that is masked for torch-linked packages by
  pytorch's own `python_abi`, and the mask vanishes for `links_torch: false`
  packages (`cumm`, `spconv`).
- **Do not ship the toolchain.** The ccache seat-swap moves the real `nvcc`
  aside, which makes `bin/nvcc.real` a new file in `$PREFIX` — a 27.5 MB CUDA
  compiler was being packaged inside torchvision, declared in `paths.json`.
  Restore on exit; gate on it.

## What the wheel half costs, measured

The claim above -- one compile, two packaging contracts -- holds, and building
it turned up three things that are properties of the arrangement rather than
bugs in a package. They are recorded here because the next platform hits all
three.

**The conda toolchain cannot always produce a manylinux wheel.** conda-forge's
gcc 13 emits `__throw_bad_array_new_length@GLIBCXX_3.4.29` (GCC 11+ generates
it for `new T[n]`). manylinux_2_28 permits GLIBCXX up to 3.4.25, and
`manylinux_2_31`, `_2_34` and `_2_35` all refuse it as well -- so raising the
tag is not an escape, it would ship a worse glibc floor than torch's own and
still fail. Upstream's manylinux_2_28 torchaudio references nothing above
3.4.25, so the policy is right and our toolchain is the outlier. Of the first
five packages built, **four did not trip it and torchaudio did**: whether a
package uses array `new` anywhere is luck, so this recurs unpredictably across
the grid. The lever is `gcc_version:` in `package.yml`, which may lower the
compiler below the policy's value but never raise it (the policy value is
nvcc's ceiling, not a default to argue with). The `.conda` is unaffected either
way -- it declares `libstdcxx >=13` and gets it.

**RPATH is a defect the wheel half structurally always has.** pip links the
extension against the conda host prefix, so setuptools writes that absolute
path into `DT_RPATH`. rattler-build rewrites RPATHs when it packages, which is
why the `.conda` is clean and `verify_conda`'s lint passes -- but the wheel is
taken *before* that step. auditwheel only rewrites binaries it grafts into, so
a package that vendors nothing keeps a build-machine path in a published
artifact. `make_wheel.py` drops every non-`$ORIGIN` entry.

**auditwheel needs the host prefix, which normally no longer exists.** The
libraries it vendors live in the conda host env, so the build must run with
`--keep-build` and the repair must happen in the same job. Not theoretical:
torchvision vendors libjpeg, libpng16, libwebp, libsharpyuv and libnvjpeg, all
from that prefix.

Measured on one torchvision compile, which is the clearest statement of the
contract inversion the design rests on:

| | `.conda` | wheel |
|---|---|---|
| vendored shared libraries | **none** | libjpeg, libnvjpeg, libpng16, libsharpyuv, libwebp |
| declares | libjpeg-turbo, libpng, libwebp-base, libzlib, libnvjpeg | **nothing** |

Both pass the same op on an RTX 3090 -- `nms`, `encode_jpeg`/`decode_jpeg`,
`encode_png`/`decode_png`, and `decode_jpeg(device="cuda")` -- the conda half
from a real solve, the wheel half in a plain venv on PyPI torch 2.8.0+cu128.

## Dependencies: derive, then curate

`cuda-wheels` ships wheels with **zero** `Requires-Dist` and writes a PEP 658
`<wheel>.metadata` sidecar carrying the dependency-bearing METADATA, with
`torch` excluded on purpose (its ABI is pinned in the local version, which pip
ignores for resolution). So the dependency data **already exists** and is the
right starting point — deriving beats re-reading 42 `setup.py` files.

It is not clean, though: gsplat's sidecar declares `Requires-Dist: ninja`, a
build tool as a runtime dependency. Conda's `build_deps` / `host_deps` /
`run_deps` split is what stops that recurring — the wheel farm's single
`extra_deps` field is how `psutil`, a `setup_requires`, came to be declared as
a runtime dependency of flash-attn.

A correction, because an earlier version of this paragraph used mmcv's `yapf`
as the second example: it is not one. mmcv 1.7.2's `mmcv/utils/config.py`
imports `yapf` at module scope and `import mmcv` reaches it unconditionally
(`mmcv/__init__.py` → `from .utils import *`), so a code formatter really is
a runtime dependency there. "Looks like a build tool" is a prompt to read the
imports, not a verdict.

## Sharding

Heavy packages (flash-attn 25 ways, natten 23) exceed a single job. The handoff
medium is a **content-addressed ccache**, not object files: shards populate it,
the link job re-runs the full build and asserts **zero** misses — zero rather
than a ratio, because one miss is a whole TU recompile and a percentage cannot
distinguish a flaky TU from four shards built for the wrong architecture.

Measured: a cache populated by one `rattler-build` run replays at zero new
misses in a second run in a different work dir *and* prefix, despite conda
injecting path-dependent flags (`-fdebug-prefix-map`, `-isystem $PREFIX`);
`CCACHE_BASEDIR` + `CCACHE_NOHASHDIR=1` absorb them. Requires **ccache ≥ 4** —
3.x has no `cu` language entry and silently passes `-x cu` through uncached.

Two traps found while proving it: rattler-build hands the build script a
**clean environment**, so every `CCACHE_*` setting must live in
`build.script.env`; and its **staging cache key does not cover `$RECIPE_DIR`
files**, so changing the build script left the key identical and restored a
stale tree. Both need the script's own hash to ride in the build command.

### Where a package's own build system owns the cache

Everything above assumes the nvcc wrapper is the outermost thing on the compile
line. For the one CMake-driven package it is not, and the two consequences are
worth stating because the next CMake package hits both (measured on torchaudio,
run 34195839332).

torchaudio's own `CMakeLists.txt` does `find_program(CCACHE_PROGRAM ccache)` and
sets `CMAKE_{C,CXX,CUDA}_COMPILER_LAUNCHER`. That is *load-bearing and welcome*
— the wrapper only ever occupies the **nvcc** seat, so without it the 40 C++
TUs of a 46-TU build would not be cached at all and sharding would buy nothing.
But it puts ccache **outside** the wrapper, and:

- **cmake's `try_compile` probes are uncacheable by construction.** They reach
  ccache through the nvcc seat, and cmake writes each probe's source into
  `CMakeFiles/CMakeScratch/TryCompile-<6 random chars>/`. ccache hashes the
  source **path** — `### inputfile` in the direct hash, and the `# 1 "…"` line
  markers in the preprocessed one — so both lookups miss on every configure,
  and no ccache setting absorbs it: `CCACHE_BASEDIR` rewrites a prefix and the
  randomness is not in the prefix. Two of torchaudio's three CUDA probes are
  like this, `OpenMPTryFlag.cu` and `OpenMPCheckVersion.cu`, and they are the
  whole of "2 ccache miss(es) of 49": all 46 real translation units replayed,
  and so did the third probe, `CMakeCUDACompilerABI.cu`, whose source has a
  fixed path in the cmake install (the random `cmTC_…` name reaches it only
  through `-MT`/`-MF`/`-o`, none of which ccache hashes). The wrapper now sends
  a recognised probe straight to the real compiler, so ccache's counters are
  exactly the package's TUs and **zero stays reachable**. The alternative —
  tolerating two misses — is precisely what a ratio cannot distinguish from two
  shards built for the wrong architecture, and the bypass removes the cause
  instead of widening the gate. Only a positive probe match bypasses; anything
  the patterns fail to classify still goes through ccache and still shows up as
  a miss.
- **on a cache hit the wrapper never runs, so the link job's ledger is empty.**
  Where the wrapper is outermost (flash-attn) it records each TU on its way to
  a hit and the link job's ledger is full — 72 entries for 72 hits. Where it is
  not, the only record of what produced those objects is the **shard's**
  ledger, so it now travels with the cache and L3 reads the union. Without
  that, verify_conda's "compile ledger is non-empty" would fail an artifact
  that was in fact compiled entirely from source in that very run.

One ccache mechanic makes the nesting harmless rather than double-counted:
ccache disables itself when it is the one invoking the compiler, so the
wrapper's own `ccache` call inside a launcher-driven compile reports
`Result: disabled` and never becomes a second lookup.

### Where the nvcc seat sees nothing at all

The seat wrapper is both the ledger and the cache, and it sees nvcc only. A
package whose extension is C++ against the CUDA runtime never reaches it:
cumm's `core_cc` is 39 `.cc` translation units through g++ and zero `.cu`
(measured — ccimport's generated `build.ninja` has `compiler__cu = nvcc`
and no edge that uses it). The seat ledger comes out empty and ccache saw no
lookup, which reads exactly like "the wrapper never occupied the seat", and
the shard job refused it.

The two are told apart by evidence, not by a declaration: ninja's own
`.ninja_log`, the same record `build_win.py` reads for L3 on win-64. If ninja
compiled real translation units and **none of them is a `.cu`**, those TUs
become the ledger and the lookup assertions are skipped as inapplicable
(`build.sh`, via `build_win.py --ninja-ledger`, which is copied beside every
recipe on both platforms). A `.cu` that ninja built and the seat never saw
still fails, so a missing wrapper cannot hide behind this. The shard and link
jobs both compile such a package in full — there is nothing to hand off — and
`verify_conda`'s "ledger non-empty" and "no foreign TU" checks run on the
ninja-derived list.

Two parser facts found on the way, both of which had produced a ledger that
was non-empty and *wrong* — which no downstream gate can tell from a right
one: ccimport writes `build.ninja` through `ninja_syntax.Writer`, which wraps
edges with a trailing `$`, so an edge's first input sits on the next line;
and torch's `BuildExtension` overwrites one `build.ninja` per extension in a
shared `build_temp` while `.ninja_log` there accumulates, so for
torch_scatter's four extensions only the last one's edges survive. The
parser now joins continuation lines and maps an object with no surviving
edge back through its path (distutils keeps the source's relative path under
`build_temp`), refusing to guess when zero or several candidates exist.

### A package that depends on another package this repo builds

spconv's `setup.py` imports cumm to *generate* its translation units, and
both import `pccm` at build time and at import time. pccm is on PyPI and on
no conda channel, so it is carried here as a hand-written noarch recipe
(`recipes/pccm`, the one recipe not generated from a `package.yml`; its
README says why), and the build's host solve lists this repo's own channel
first so a package can depend on a sibling. spconv's dep on cumm carries the
flavour glob, `cumm >=0.7.11,<0.8.0 cuda128_*`, written as recipe jinja in
`package.yml` so it follows the cell; `make_wheel.py` drops the build field
when it writes the sidecar, because PEP 508 has nowhere to put it and the
wheel says the same thing through its local version tag.

A torch-free package's host env inherits **no CUDA window from anywhere** —
a torch-linked one gets it from pytorch's pin — so `cuda-version
${{ cuda_compiler_version }}.*` goes in its `host_deps`, or the host resolves
the newest `cuda-nvrtc-dev` (13.x) against a 12.8 toolkit in `build:`.

### Exact-minor sonames, and where the two outputs disagree

cumm links `libnvrtc-builtins` explicitly so that its wheel's vendored
libnvrtc can find it. The soname is exact-minor, `libnvrtc-builtins.so.12.8`
(libnvrtc's own is `libnvrtc.so.12`), and that is poison for the `.conda`: an
env whose torch pulls `cuda-nvrtc` 12.9 — conda-torch's cu128 triton pins
`cuda-version` 12.9, measured in a live solve — has `.so.12.9` and no
`.so.12.8`, so the artifact fails to load beside the torch it is for, and a
`cuda-nvrtc 12.8.*` run dep makes it UNSAT instead. libnvrtc needs no help
in a conda env: it dlopens `libnvrtc-builtins.so.<major.minor>` by name
through its own `$ORIGIN` RPATH, and both ship in one package (measured — a
lone copy of libnvrtc with the builtins beside it compiles; without them it
fails with "failed to open libnvrtc-builtins.so.12.9"). So the link is
patched out, and the wheel gets the builtins through `wheel_vendor_extra`,
which copies the file under its own SONAME beside the vendored libnvrtc and
gives every vendored library an `$ORIGIN` RPATH — the one case where the two
packaging contracts want *different binaries* and the answer is to post-
process the wheel rather than compromise the `.conda`.

Also measured on cumm, and worth more than the earlier "3.4.25" above:
auditwheel 6.8.2's policy file allows `GLIBCXX <= 3.4.24` and `CXXABI <=
1.3.11` for manylinux_2_28. gcc 13 emitted four symbols above that, and gcc
10 would still have emitted `basic_stringstream::basic_stringstream()@
GLIBCXX_3.4.26` (GCC 9 made the default constructor an exported symbol, and
pccm-generated code builds a stringstream in every `tv::check`). gcc 8, the
oldest conda-forge ships, instantiates it inline and references nothing
above 3.4.24; cumm and spconv pin `gcc_version: "8"`.

## Known per-package facts, carried forward

- **ninja is required in `build:`.** Without it torch's `BuildExtension` falls
  back to distutils and compiles every TU *serially*, ignoring `MAX_JOBS` —
  measured at 12 minutes for one torchvision cell versus 4 with ninja.
- **`setuptools <82` for torchvision ≤ 0.26.0** — setuptools 84 removed
  `pkg_resources`, which 0.22–0.26 import and 0.27.0 does not; that is exactly
  the observed pass/fail boundary.
- **`cuda-nvtx` in the torch floor for torch 2.4/2.5** — they link
  `libnvToolsExt.so.1` and nothing else supplied it; it dies on `import torch`,
  before compiling anything.
- **torchvision stats its codec headers under `BUILD_PREFIX`/`CONDA_PREFIX`,
  never rattler-build's `$PREFIX`** — so jpeg/webp/nvjpeg were silently
  disabled while `run_exports` still claimed them, and `decode_jpeg` would have
  raised at runtime on every published artifact. Structural checks cannot see
  this; only running the op can.

## Verification

- **Per artifact** (publish gate): build string matches the cell; `paths.json`
  matches payload; no `RECORD`; `INSTALLER` names conda; `$ORIGIN`-relative
  RPATHs with no absolute or empty entries; computed (not pasted) glibc floors;
  SASS architectures match the cell's arch list; no vendored `libtorch`; no
  `nvcc.real`; `purls`, `__cuda`, `python_abi` and the torch build-glob present;
  the compile ledger covers every module.
- **Per cell** (solve gate): a live solve from the published channel resolving
  from our own release URL, asserting the resolved `pytorch`'s flavour equals
  the extension's.
- **On hardware**: import plus the package's own minimal op. A mis-arch'd
  kernel shows up as `cudaErrorNoKernelImageForDevice` and nowhere else, and
  the disabled-codec bug above showed up nowhere but a real `decode_jpeg`.

## Defective builds: reachability, not policy

A published artifact is immutable, so a build discovered to be defective can
never be withdrawn — only made unreachable. `known_bad.json` records every such
build, in both formats, and three tools act on it: `make_repodata.py` for the
channel, `generate_index.py` for the wheel index, `check_lock.py` for someone
holding a lockfile that already pinned one.

The rule for the channel is **reachability**, and stating it that way is what
makes two channels that look like they disagree follow one rule. A known-bad
build stays listed in repodata if and only if something already prevents a
fresh solve choosing it:

- **superseded** — a build of the same name and version with a higher build
  number exists, so conda's own ranking never reaches this one;
- **neutralised** — an unsatisfiable constrain (`<0.0a0` on a package that must
  be present) makes the solver refuse it outright.

Otherwise it is dropped, because listed and reachable is an offer.

conda-torch keeps its known-bad builds listed and is right to: they are
superseded or carry `libcudnn <0.0a0`. This repo dropped the no-jpeg
torchvision and was right to: it had neither a higher build nor a patch, so a
fresh solve would have chosen it. One rule, opposite outcomes, because the
facts differed — and it un-drops itself once the superseding build publishes.

Staying listed is the better end state wherever it applies: the release asset
is immutable either way, so a lockfile that already pinned a bad build keeps
resolving, and leaving the entry means the channel and `known_bad.json` agree
about what exists. Dropping is what you do when nothing else stops the build
being chosen.

**Supersession only counts a build that is not itself known-bad.** Not a
detail: a build superseded only by another defective build is not protected,
the solver simply moves from one bad artifact to another. Running this rule
against conda-torch's live channel found exactly that —
`libtorch-2.8.0-cuda129_repack_h327d83bf_0` is superseded by `_1`, and `_1` is
itself in `known_bad.json`.

Wheels cannot be dropped the same way, because a wheel index has a better
mechanism: PEP 592 yanking. `data-yanked` keeps the file downloadable for
anyone who pinned that exact filename while stopping any resolver selecting it,
and the reason string reaches the user. Verified live: unpinned, pip reports
`Ignored the following yanked versions`; pinned exactly, it still installs.

## Scope

linux-64 first, for the same reason as before: prove the machinery where the
compiler story is simplest. `target_platform` stays in the build string and the
templates keep an unwired `win` branch; linux-aarch64 is platform two.
