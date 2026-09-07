"""Patch fused-ssim: make TORCH_CUDA_ARCH_LIST authoritative, and pre-empt the
Windows `small` macro collision.

Run by scripts/fetch_patched_sources.py with cwd set to the unpacked source,
BEFORE the tarball is sealed. Two consequences of that timing shape this file:

  * No `import torch`. The fetch step runs on the host, outside any build env,
    so torch is not importable and its version is not knowable here.
  * **No platform conditionals.** Fetch produces ONE tarball that every
    platform's build consumes. The predecessor patched on the build machine
    and could therefore gate on `os.name == "nt"`; here that gate would bake
    one platform's answer into every platform's source. So the Windows fix is
    applied unconditionally and made inert off-Windows by `#ifdef _WIN32` in
    the C source itself, which is where the condition belongs anyway.

── 1. Arch flags ──────────────────────────────────────────────────────────

Upstream never reads TORCH_CUDA_ARCH_LIST. setup.py either probes the local
GPU and appends `-arch=sm_XX` for that one device, or -- with no GPU visible
-- extends nvcc_args with three hardcoded gencodes (sm_75/80/89).

That is not merely an additional flag: torch's own cpp_extension DISABLES its
TORCH_CUDA_ARCH_LIST handling when the user supplies any arch flag. Measured
in pytorch v2.8.0, torch/utils/cpp_extension.py::_get_cuda_arch_flags:

    if cflags is not None:
        for flag in cflags:
            if 'TORCH_EXTENSION_NAME' in flag:
                continue
            if 'arch' in flag:
                return []

Both upstream spellings contain the substring 'arch', so either branch
silently reduces the cell's arch list to whatever setup.py decided.

**Why this is stricter than the predecessor's version of the same patch.**
cuda-wheels carries this fix, pinned at this exact same revision
(328dc9836f...), and its regex targets `nvcc_args.append(f"-gencode...")` --
which does not appear in this source; the append here reads
`nvcc_args.append(f"-arch={arch}")`. That substitution has therefore never
matched. It went unnoticed because the wheel farm builds on GPU-less runners,
where `torch.cuda.is_available()` is False and only the fallback branch runs
-- which the patch does neutralise. On any machine WITH a visible GPU the
detected-arch append survives and the artifact carries a single architecture.
This repo builds on developer boxes that have one, so the latent bug is live
here.

The predecessor's guard could not catch it either: it compared the whole file
before and after all three substitutions, so two matches out of three still
looked like success. Each substitution below asserts on its own.

── 2. The Windows `small` macro ───────────────────────────────────────────

PyTorch 2.10 uses `bool small` as a parameter name in
c10/cuda/CUDACachingAllocator.h. The Windows SDK's rpcndr.h, reached
transitively through windows.h, does `#define small char`, so nvcc sees
`bool char`. Upstream: https://github.com/pytorch/pytorch/issues/173112

`-Usmall` on the command line does NOT fix it -- that clears only the initial
preprocessor state, and rpcndr.h re-defines the macro mid-compilation. The fix
has to sit in the source file's own include sequence, after windows.h and
before any torch header.

Applied here for torch 2.8 as well, where the collision does not yet occur:
the prologue only #undefs MIDL helper macros that nothing in this source uses,
so it is inert, and gating it on a torch version this script cannot observe
would cost more than it saves.
"""

import re
import sys
from pathlib import Path

MARKER = "pytorch/pytorch#173112"

PROLOGUE = f"""// Workaround for {MARKER}: rpcndr.h on Windows defines `#define small char`,
// which collides with PyTorch 2.10+'s `bool small` parameter in
// c10/cuda/CUDACachingAllocator.h. Force-include windows.h (which triggers the
// definition) then #undef the MIDL helpers BEFORE any torch header is parsed.
//
// WIN32_LEAN_AND_MEAN alone is NOT enough -- windows.h still defines the
// function-like `min`/`max` macros, which collide with torch's std::min/std::max
// across BFloat16.h, Float8_*.h and others, producing `std::std::` nesting and
// "not enough arguments for function-like macro". NOMINMAX is the documented
// opt-out.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef small       // rpcndr.h:  #define small char
#undef hyper       // rpcndr.h:  #define hyper __int64
#undef boolean     // rpcndr.h:  #define boolean unsigned char
#undef byte        // rpcndr.h:  #define byte unsigned char
#endif

"""

SOURCES = ("ssim.cu", "ext.cpp")


def sub_once(text: str, pattern: str, repl: str, what: str) -> str:
    """Apply a substitution and fail loudly if upstream no longer matches it.

    Per-substitution, deliberately: a whole-file before/after comparison passes
    as long as ANY edit landed, which is exactly how the predecessor's dead
    `-gencode` regex survived review.
    """
    new, n = re.subn(pattern, repl, text)
    if n == 0:
        sys.exit(
            f"fused_ssim patch: {what} not found -- upstream setup.py changed "
            f"at this pinned rev; re-read it and update the patch rather than "
            f"relaxing this check"
        )
    print(f"fused_ssim patch: {what} ({n} site{'s' if n != 1 else ''})")
    return new


SETUP_MARKER = "# patched-by: cuda-foundry fused_ssim arch fix"


def patch_setup_py() -> None:
    path = Path("setup.py")
    text = path.read_text()

    # Idempotent like the C sources below. Without this a re-run fails on the
    # second substitution and reports "upstream setup.py changed", which is a
    # confusing lie -- what changed is that we already patched it.
    if SETUP_MARKER in text:
        print("fused_ssim patch: setup.py already patched")
        return

    text = sub_once(
        text,
        r"fallback_archs = \[[^\]]*\]",
        "fallback_archs = []  # patched: cpp_extension derives from TORCH_CUDA_ARCH_LIST",
        "hardcoded gencode fallback list emptied",
    )
    # The detected-GPU append. This is the one the predecessor's regex misses.
    text = sub_once(
        text,
        r'nvcc_args\.append\(f"-arch=\{arch\}"\)',
        "pass  # patched: local-GPU arch must not override TORCH_CUDA_ARCH_LIST",
        "local-GPU -arch append removed",
    )
    # Two sites: the detect-failure branch and the no-CUDA branch.
    text = sub_once(
        text,
        r"nvcc_args\.extend\(fallback_archs\)",
        "pass  # patched: see above",
        "fallback extend removed",
    )

    # Belt and braces: after patching, nothing may hand nvcc an arch flag,
    # because any such flag makes torch drop TORCH_CUDA_ARCH_LIST entirely.
    for lineno, line in enumerate(text.splitlines(), 1):
        if "nvcc_args" in line and ("-arch" in line or "-gencode" in line):
            sys.exit(
                f"fused_ssim patch: setup.py:{lineno} still puts an arch flag "
                f"into nvcc_args after patching: {line.strip()!r}"
            )

    path.write_text(SETUP_MARKER + "\n" + text)


def patch_sources() -> None:
    for name in SOURCES:
        path = Path(name)
        if not path.is_file():
            sys.exit(f"fused_ssim patch: expected source {name} is missing")
        text = path.read_text()
        if MARKER in text:
            print(f"fused_ssim patch: {name} already carries the prologue")
            continue
        path.write_text(PROLOGUE + text)
        print(f"fused_ssim patch: {name} prologue applied")


def main() -> None:
    patch_setup_py()
    patch_sources()
    print("fused_ssim patch: done")


if __name__ == "__main__":
    main()
