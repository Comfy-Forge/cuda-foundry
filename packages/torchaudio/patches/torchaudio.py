"""Patch torchaudio's optional backends to build with no network.

Upstream 2.8.0 builds BOTH sox and ffmpeg support by default on Linux --
tools/setup_helpers/extension.py:37-40 gives BUILD_SOX and USE_FFMPEG a
default of True -- and each reaches the network during the compile:

  * third_party/sox/CMakeLists.txt FetchContent-downloads
    sox-14.4.2.tar.bz2 from SourceForge.
  * third_party/ffmpeg/multi/CMakeLists.txt downloads PRE-BUILT ffmpeg
    binaries for versions 4, 5 and 6 from pytorch.s3.amazonaws.com.

Under L1 the compile has no network, so the sox fetch fails and takes the
whole build with it:

    error: downloading '.../sox-14.4.2.tar.bz2' failed
    CMake Error at .../FetchContent.cmake:1933 (message)
    ERROR: Failed building wheel for torchaudio

The obvious response -- BUILD_SOX=0 USE_FFMPEG=0 -- is the wrong one. It
builds, it passes a resample/lfilter smoke test, and it silently ships a
package missing what upstream's own wheel carries: torchaudio-2.8.0's
manylinux wheel contains _torchaudio_sox.so, libtorchaudio_sox.so and
torio/lib/libtorio_ffmpeg{4,5,6}.so. That is the torchvision-codec trap
exactly -- a feature detected as absent at build time, no error, and a
failure that only appears when a user calls the op.

So this uses upstream's OWN escape hatches instead, and the result is more
in keeping with this repo than upstream's arrangement is:

  * sox is only ever downloaded for its HEADER. CMakeLists.txt sets
    CONFIGURE_COMMAND "" and BUILD_COMMAND "", then compiles third_party's
    own stub.c -- a file of no-op definitions of the sox API -- into a
    shared library. No part of sox is compiled and none of it is shipped.
    So the header can come from conda-forge's `sox` package in the host
    prefix, and the FetchContent block simply goes away.
  * ffmpeg has a documented single-version path: CMakeLists.txt:173-179
    uses third_party/ffmpeg/single -- "searching existing FFmpeg
    installation" -- whenever the FFMPEG_ROOT environment variable is set,
    and third_party/ffmpeg/multi's pre-built download only otherwise. The
    package points FFMPEG_ROOT at the host prefix, so torio links against
    conda-forge's ffmpeg. That trades upstream's three vendored ffmpeg
    versions for one that `conda update` can patch -- the same argument
    that makes torchvision link the host's libjpeg instead of vendoring a
    2018 copy.

Both are build-system edits only; no torchaudio source is touched, and
nothing here gates on os.name -- one tarball serves every platform, so a
host-specific branch would bake this machine's answer into the Windows
build.
"""
import sys as _sys_pl
import pathlib as _pl_pl
_sys_pl.path.insert(0, str(_pl_pl.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require as _require

import re
from pathlib import Path

sox_cmake = Path("third_party/sox/CMakeLists.txt")
_require(sox_cmake.is_file(),
         "third_party/sox/CMakeLists.txt is missing -- torchaudio's layout "
         "changed and this patch must be re-derived, not skipped")
text = sox_cmake.read_text()

# The whole FetchContent preamble, from `include(FetchContent)` down to the
# end of the populate guard. Matched as a block rather than line by line so a
# reshuffle upstream fails loudly here instead of half-applying.
block = re.compile(
    r"include\(FetchContent\).*?"
    r"if\(NOT sox_src_POPULATED\).*?endif\(\)\n",
    re.DOTALL)
text, n = block.subn(
    "# cuda-foundry: upstream downloaded sox-14.4.2 here purely for its\n"
    "# header (CONFIGURE_COMMAND and BUILD_COMMAND are both empty and only\n"
    "# stub.c is compiled). The header now comes from the conda host prefix\n"
    "# via SOX_INCLUDE_DIR, so the compile needs no network.\n",
    text)
_require(n == 1,
         f"expected exactly 1 sox FetchContent block, patched {n}. Upstream "
         "restructured third_party/sox/CMakeLists.txt; re-read it before "
         "assuming this patch still describes reality")

# stub.c does `#include <sox.h>`, so the include dir must be where sox.h
# itself lives -- conda's $PREFIX/include -- not upstream's tarball layout,
# where the header sits one level down in src/.
text, n = re.subn(
    r"target_include_directories\(sox PUBLIC \$\{sox_src_SOURCE_DIR\}/src\)",
    'target_include_directories(sox PUBLIC $ENV{SOX_INCLUDE_DIR})',
    text)
_require(n == 1,
         f"expected exactly 1 sox include-directory line, patched {n}")

sox_cmake.write_text(text)
print("torchaudio: sox header now comes from $SOX_INCLUDE_DIR; "
      "FetchContent download removed")

# Prove the download is really gone rather than trusting the substitutions:
# a leftover URL means the build would still try to reach SourceForge.
leftover = [ln for ln in text.splitlines()
            if "sourceforge" in ln.lower() or "FetchContent" in ln]
_require(not leftover,
         f"sox CMakeLists still references a download after patching: {leftover}")
print("torchaudio: verified no FetchContent/sourceforge reference remains")

# ── $ORIGIN in the installed libraries' RPATH ───────────────────────────────
# The build installs every library side by side (torchaudio/lib/,
# torio/lib/) and the Python-facing ones record DT_NEEDED on their siblings:
# _torchaudio.so -> libtorchaudio.so, _torchaudio_sox.so ->
# libtorchaudio_sox.so, _torio_ffmpeg.so -> libtorio_ffmpeg.so, and the
# ffmpeg one on libav*. cmake's install step strips the build RPATH and
# upstream sets no install RPATH, so the published artifact's four
# python-facing .so carried NO $ORIGIN entry (audit finding): the sibling
# resolves only because torchaudio's Python happens to torch.ops.load_library
# the dependency first, and libav* only through whatever the process already
# has on its search path. Set the install RPATH here, in the one CMake
# project, rather than patchelf-ing afterwards: `$ORIGIN` for the sibling,
# and the link path (the host prefix's lib/, where ffmpeg and sox live) which
# rattler-build then relocates into its own $ORIGIN-relative form when it
# packages, and tools/make_wheel.py drops from the wheel as it drops every
# non-$ORIGIN entry (the wheel vendors neither ffmpeg nor sox, like
# upstream's). ELF only -- guarded on UNIX AND NOT APPLE so the Windows and
# macOS configure are byte-identical to before.
top = Path("CMakeLists.txt")
_require(top.is_file(), "CMakeLists.txt is missing at the source root")
text = top.read_text()
RPATH_BLOCK = """
# cuda-foundry: installed libraries find their siblings through $ORIGIN and
# the host prefix's libraries through the link path (see
# packages/torchaudio/patches/torchaudio.py).
if(UNIX AND NOT APPLE)
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()
"""
if "CMAKE_INSTALL_RPATH" in text:
    print("torchaudio: install RPATH already set in CMakeLists.txt")
else:
    text, n = re.subn(r"^project\(torchaudio\)\n", "project(torchaudio)\n" + RPATH_BLOCK,
                      text, count=1, flags=re.M)
    _require(n == 1, "expected exactly one `project(torchaudio)` line at the top of "
                     "CMakeLists.txt to anchor the RPATH settings on")
    top.write_text(text)
    print("torchaudio: CMAKE_INSTALL_RPATH=$ORIGIN (+ link path) set for ELF builds")
_require('set(CMAKE_INSTALL_RPATH "$ORIGIN")' in top.read_text(),
         "the RPATH block did not land in CMakeLists.txt")
