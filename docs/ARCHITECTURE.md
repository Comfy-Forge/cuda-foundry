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

## Dependencies: derive, then curate

`cuda-wheels` ships wheels with **zero** `Requires-Dist` and writes a PEP 658
`<wheel>.metadata` sidecar carrying the dependency-bearing METADATA, with
`torch` excluded on purpose (its ABI is pinned in the local version, which pip
ignores for resolution). So the dependency data **already exists** and is the
right starting point — deriving beats re-reading 42 `setup.py` files.

It is not clean, though: gsplat's sidecar declares `Requires-Dist: ninja`, a
build tool as a runtime dependency, and mmcv's declares `yapf`. Conda's
`build_deps` / `host_deps` / `run_deps` split is what stops that recurring —
the wheel farm's single `extra_deps` field is how `psutil`, a `setup_requires`,
came to be declared as a runtime dependency of flash-attn.

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

## Scope

linux-64 first, for the same reason as before: prove the machinery where the
compiler story is simplest. `target_platform` stays in the build string and the
templates keep an unwired `win` branch; linux-aarch64 is platform two.
