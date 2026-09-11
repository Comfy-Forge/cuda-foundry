# One build script, three modes. Copied beside every generated recipe (with
# the package's build_env substituted at the hook below) and named by the
# recipe's `build.script.file`, so a shard job and its link job run
# byte-identical compiles — which is what makes the ccache handoff replay
# instead of recompile. File-backed, so rattler-build never renders it through
# minijinja: brace-hash, brace-brace and brace-percent are allowed here again.
#
#   CUW_MODE=full   compile + install (unsharded packages)
#   CUW_MODE=shard  compile only this shard's slice, then exit 0
#   CUW_MODE=link   full build, every compile expected to be a cache hit
#
# The environment this relies on is set in the recipe's build.script.env,
# NOT by the workflow: rattler-build hands the script a clean environment and
# the outer shell's exports do not reach it (measured, step-0 probe).
set -euo pipefail

CUW_DIR="$(dirname "${CUW_LEDGER:-/tmp/cuw/ledger.txt}")"
mkdir -p "$CUW_DIR" "$CCACHE_DIR"
: > "$CUW_LEDGER"

CCACHE_BIN="$(command -v "${CUW_CCACHE_BIN:-ccache}" || echo "${CUW_CCACHE_BIN:-ccache}")"
if ! "$CCACHE_BIN" --version >/dev/null 2>&1; then
  echo "::error::ccache not usable at '$CCACHE_BIN' -- the shard handoff and the compile ledger both run through it" >&2
  exit 1
fi
# ccache 3.x has no "cu" entry in its source-language table, so every TU the
# build compiles as `-x cu` is passed through UNCACHED. That is not a slow
# cache, it is no cache, and the shard lane would silently void itself.
CC_MAJOR="$("$CCACHE_BIN" --version | head -1 | grep -oE '[0-9]+' | head -1)"
if [ "${CC_MAJOR:-0}" -lt 4 ]; then
  echo "::error::ccache $CC_MAJOR.x found; >= 4 required (3.x cannot cache '-x cu')" >&2
  exit 1
fi

# ---- $PREFIX must look untouched by the time rattler-build packages -----
# rattler-build ships "files that appeared in $PREFIX during the build". The
# nvcc seat swap below leaves `bin/nvcc.real` behind and the CUDA header
# bridge further down creates symlinks under `include/`; both are new paths in
# $PREFIX, so both get PACKAGED. Measured, not theorised: the first
# torchvision pilot shipped a 27.5 MB `bin/nvcc.real` -- the real CUDA
# compiler -- fully declared in paths.json, and every package this repo has
# ever built carries the same passenger.
#
# So everything this script creates inside $PREFIX is recorded here and undone
# on EXIT, on every path out: shard mode exits 0 early, a failed compile exits
# non-zero, and both must still leave a clean prefix behind.
CUW_PREFIX_TRACK="$CUW_DIR/prefix-added.txt"
: > "$CUW_PREFIX_TRACK"
cuw_restore_prefix() {
  rc=$?
  # Put the real compiler back in its seat. Guarded on the marker so this
  # cannot clobber a real nvcc if the swap never happened.
  if [ -e "$BUILD_PREFIX/bin/nvcc.real" ] && grep -q 'CUW_WRAPPER_MARKER' "$BUILD_PREFIX/bin/nvcc" 2>/dev/null; then
    mv -f "$BUILD_PREFIX/bin/nvcc.real" "$BUILD_PREFIX/bin/nvcc" || true
  fi
  while IFS= read -r added; do
    [ -n "$added" ] && rm -f "$added" || true
  done < "$CUW_PREFIX_TRACK"
  return $rc
}
trap cuw_restore_prefix EXIT

# ---- the wrapper goes in the nvcc SEAT, not on PATH ---------------------
# torch's cpp_extension invokes "$CUDA_HOME/bin/nvcc" by absolute path, so a
# PATH-based shim is simply never consulted. Move the real binary aside and
# occupy its filename.
#
# The guard below is not paranoia. Without it, a re-entered script (a reused
# prefix, a retried step) moves the WRAPPER onto nvcc.real and installs a
# fresh wrapper on top, so nvcc execs ccache on a script that execs ccache on
# itself. That does not fail — it recurses forever, and in CI it burns the
# whole job timeout with no error. Measured the hard way, step 0.
if [ -f "$BUILD_PREFIX/bin/nvcc" ]; then
  if grep -q 'CUW_WRAPPER_MARKER' "$BUILD_PREFIX/bin/nvcc" 2>/dev/null; then
    # already ours: leave the seat alone, and make sure the real binary is
    # still behind it rather than silently recursing.
    if [ ! -x "$BUILD_PREFIX/bin/nvcc.real" ] || grep -q 'CUW_WRAPPER_MARKER' "$BUILD_PREFIX/bin/nvcc.real" 2>/dev/null; then
      echo "::error::\$BUILD_PREFIX/bin/nvcc is the cuw wrapper but nvcc.real is missing or is itself a wrapper -- the nvcc seat is corrupt and would recurse forever" >&2
      exit 1
    fi
    echo "nvcc seat already wrapped; reusing"
  elif [ ! -f "$BUILD_PREFIX/bin/nvcc.real" ]; then
    mv "$BUILD_PREFIX/bin/nvcc" "$BUILD_PREFIX/bin/nvcc.real"
    install -m 0755 "$RECIPE_DIR/nvcc-wrap.sh" "$BUILD_PREFIX/bin/nvcc"
  fi
fi
# Whatever path we took, the seat must now be a wrapper over a REAL compiler.
if ! "$BUILD_PREFIX/bin/nvcc.real" --version >/dev/null 2>&1; then
  echo "::error::\$BUILD_PREFIX/bin/nvcc.real does not behave like a compiler -- refusing to build with a corrupt nvcc seat" >&2
  exit 1
fi
export CUW_REAL_NVCC="$BUILD_PREFIX/bin/nvcc.real"
export CUW_CCACHE_BIN="$CCACHE_BIN"

export CUDA_HOME="$BUILD_PREFIX"
export PYTORCH_NVCC="$BUILD_PREFIX/bin/nvcc"   # torch's ninja writer honours this
export CUDACXX="$BUILD_PREFIX/bin/nvcc"        # cmake honours this
# Trailing, so it beats any --threads a setup.py hardcodes earlier in the line.
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} --threads ${CUW_NVCC_THREADS:-1}"

# ---- CUDA headers: bridge conda-forge's targets/ layout -----------------
# conda-forge's CUDA packages install headers under
# $<prefix>/targets/<arch>/include, NOT $<prefix>/include. Anything that
# merely COMPILES is fine, because the activation scripts add the -I.
# Anything that PROBES for a header by path is not: it concludes the library
# is absent. torchvision does exactly that for nvjpeg (it stats
# "$CUDA_HOME/include/nvjpeg.h") and built without GPU JPEG support while
# libnvjpeg-dev sat installed in host: and its run_export was already on the
# finished package -- which is what rattler-build's "Overdepending against
# libnvjpeg" warning was really saying.
#
# Sources are searched BUILD_PREFIX first, then PREFIX, and a link is only
# ever created where nothing exists yet. That order is load-bearing: the CUDA
# toolkit is pinned to the cell in the build env while host: still resolves to
# whatever `cuda-version` torch's triton drags in, so build's cuda.h must win.
# The bridge writes into $CUDA_HOME/include, which is inside BUILD_PREFIX and
# therefore never scanned for packaging; the EXIT trap removes it regardless.
mkdir -p "$CUDA_HOME/include"
for cuw_inc in "$BUILD_PREFIX"/targets/*/include "$PREFIX"/targets/*/include; do
  [ -d "$cuw_inc" ] || continue
  cuw_linked=0
  for cuw_h in "$cuw_inc"/*; do
    cuw_b="$(basename "$cuw_h")"
    # -e is false for a dangling symlink, so test -L as well: without it the
    # ln below fails on an existing-but-broken link and the loop is noisy.
    if [ -e "$CUDA_HOME/include/$cuw_b" ] || [ -L "$CUDA_HOME/include/$cuw_b" ]; then continue; fi
    ln -s "$cuw_h" "$CUDA_HOME/include/$cuw_b" || continue
    echo "$CUDA_HOME/include/$cuw_b" >> "$CUW_PREFIX_TRACK"
    cuw_linked=$((cuw_linked + 1))
  done
  echo "cuda header bridge: linked $cuw_linked entr(ies) from $cuw_inc"
done

# ---- and the cell's headers must come FIRST -----------------------------
# Pinning the toolkit in build: is necessary but not sufficient. The HOST
# env's CUDA activation prepends -I$PREFIX/targets/<arch>/include to
# CPPFLAGS/CFLAGS/CXXFLAGS, and host's cuda-version is dragged to whatever
# torch's triton needs -- so the cell's headers are on the command line but
# LOSE. Measured on the TU that bakes in CUDA_VERSION (torchaudio's
# utils.cpp): $PREFIX/targets/.../include at positions 1, 2, 4 and 6, the
# build prefix's headers at position 11. The result compiled 12090 into a
# cell labelled cuda128 and torchaudio refused to load beside its own torch.
# The competing -I lives in these same variables, so putting ours in front of
# them is the whole fix, and the order stops being anyone else's to decide.
export CPPFLAGS="-I$CUDA_HOME/include ${CPPFLAGS:-}"
export CFLAGS="-I$CUDA_HOME/include ${CFLAGS:-}"
export CXXFLAGS="-I$CUDA_HOME/include ${CXXFLAGS:-}"
export NVCC_PREPEND_FLAGS="-I$CUDA_HOME/include ${NVCC_PREPEND_FLAGS:-}"

# ---- no build-machine paths in the shipped binaries ---------------------
# conda-forge's compiler activation already maps the DEBUG records
# (-fdebug-prefix-map); this extends the same mapping to __FILE__ and every
# other place the preprocessor records a path, so an assertion message or a
# CUDA kernel's registered source name reads /usr/local/src/conda/<pkg>/...
# rather than this runner's work dir. The names are conda-forge's own. nvcc
# takes it through -Xcompiler for the host-side compile. Trailing, and read by
# nvcc from the environment, so it never appears on the command line ccache
# hashes; the shard/link handoff is unaffected (and $SRC_DIR is identical
# across those jobs anyway, because the workflow builds with --no-build-id).
CUW_PREFIX_MAP="-ffile-prefix-map=$SRC_DIR=/usr/local/src/conda/${PKG_NAME:-pkg} -ffile-prefix-map=$PREFIX=/usr/local/src/conda-prefix"
export CFLAGS="${CFLAGS:-} $CUW_PREFIX_MAP"
export CXXFLAGS="${CXXFLAGS:-} $CUW_PREFIX_MAP"
export CPPFLAGS="${CPPFLAGS:-} $CUW_PREFIX_MAP"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -Xcompiler -ffile-prefix-map=$SRC_DIR=/usr/local/src/conda/${PKG_NAME:-pkg} -Xcompiler -ffile-prefix-map=$PREFIX=/usr/local/src/conda-prefix"

# ---- package-declared build environment (package.yml `build_env`) -------
# Values computed from $PREFIX cannot live in the recipe's build.script.env:
# rattler-build sets those LITERALLY, with no shell expansion, so a
# "$PREFIX/include" written there arrives at setup.py as those exact 15
# characters. They are rendered here instead, where $PREFIX is real.
export CUMM_DISABLE_JIT="1"
export CUMM_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"

echo "=== cuw build ==================================================="
echo "  mode        : ${CUW_MODE}"
echo "  shard       : ${CUW_SHARD_INDEX}/${CUW_SHARD_COUNT}"
echo "  ccache      : $("$CCACHE_BIN" --version | head -1)  dir=$CCACHE_DIR"
echo "  arch list   : ${TORCH_CUDA_ARCH_LIST:-unset}"
echo "  MAX_JOBS    : ${MAX_JOBS:-unset}  nvcc --threads ${CUW_NVCC_THREADS:-1}"
echo "  SRC_DIR     : $SRC_DIR"
echo "  PREFIX      : $PREFIX"
echo "================================================================="

# In shard mode the wrapper stubs out every TU that is not this slice's, so
# the "build" completes in seconds having compiled only its share. In link
# mode it is a pure ccache pass-through and every compile should hit.
if [ "$CUW_MODE" = "shard" ]; then
  # package.yml `shard_partition: source`: the package's own build reads
  # CUW_SHARD_INDEX / CUW_SHARD_COUNT and compiles only its slice (natten
  # deletes the generated kernels outside it before cmake runs). The seat
  # wrapper then stays a pure ccache pass-through: running BOTH partitions
  # compiles the intersection, ~1/N^2 of the tree per shard, and the link
  # job's zero-miss gate is what catches a package that claims this and
  # then partitions nothing.
  if [ "${CUW_SHARD_PARTITION:-}" = "source" ]; then
    echo "shard partition: by the package's own build (shard_partition: source); nvcc seat passes everything through"
  else
    export CUW_PARTITION=1
  fi
  # The matrix numbers shards from 1 ("shard 1/1"), the partition arithmetic
  # is 0-based. Without this conversion the comparison never matches, the
  # wrapper stubs out EVERY translation unit, and the build still succeeds --
  # shipping a package whose extension module is an empty object. Observed
  # exactly that; only the compile ledger caught it.
  export CUW_SHARD_INDEX0=$(( ${CUW_SHARD_INDEX:-1} - 1 ))
fi

"$CCACHE_BIN" -z >/dev/null 2>&1 || true

# The compile runs with network access permanently removed. This is the
# from-source guarantee (L1), and it is imposed here rather than by
# rattler-build's --sandbox because that flag needs unprivileged namespaces,
# which are unavailable both in this project's container and on
# GitHub-hosted runners (measured: "sandboxing failure: Operation not
# permitted"). The seccomp filter needs no privileges, is inherited by every
# child (setup.py, pip, cmake, nvcc), and cannot be removed once installed.
#
# ---- ONE compile, and it produces a WHEEL --------------------------------
# This was `pip install .`, which builds a wheel internally and then throws it
# away. Building it explicitly changes nothing about what is compiled -- pip
# takes the same PEP 517 path either way -- and it is what makes the second
# and third outputs possible: the .conda comes from installing this wheel into
# $PREFIX, the published manylinux wheel comes from `auditwheel repair` on the
# very same file. There is no second compile anywhere, so the two artifacts
# cannot disagree about what they contain.
#
# The wheelhouse is OUTSIDE the work dir on purpose. rattler-build deletes its
# work directory, and $PREFIX is packaged -- a wheel left in either would be
# gone or shipped. This is the same handoff shape as CCACHE_DIR: a CUW_* path
# the caller owns.
: "${CUW_WHEELHOUSE:=/tmp/cuw/wheelhouse}"
mkdir -p "$CUW_WHEELHOUSE"
# Only the .whl files, never `rm -rf` on the directory itself: this path
# arrives from the environment, and an unset or misspelled variable must not
# be able to delete a tree.
rm -f "$CUW_WHEELHOUSE"/*.whl

# package.yml `build_subdir`: the setup.py to build is not at the source
# root (pointnet2_ops lives in pointnet2_ops_lib/ inside a repo whose root
# setup.py is the training code). Everything above is path-independent;
# everything below runs where setup.py is.
if [ -n "${CUW_BUILD_SUBDIR:-}" ]; then
  [ -d "$SRC_DIR/$CUW_BUILD_SUBDIR" ] || { echo "::error::build_subdir '$CUW_BUILD_SUBDIR' does not exist under $SRC_DIR" >&2; exit 1; }
  cd "$SRC_DIR/$CUW_BUILD_SUBDIR"
  echo "=== building in subdir: $CUW_BUILD_SUBDIR"
fi

BUILD_RC=0
$PYTHON "$RECIPE_DIR/nonet.py" -- $PYTHON -m pip wheel . --no-deps --no-build-isolation \
    --wheel-dir "$CUW_WHEELHOUSE" -vv || BUILD_RC=$?

STATS="$("$CCACHE_BIN" --print-stats 2>/dev/null || true)"
HITS=$(printf '%s\n' "$STATS" | awk -F'\t' '$1 ~ /^(direct_cache_hit|preprocessed_cache_hit)$/ {n+=$2} END{print n+0}')
MISSES=$(printf '%s\n' "$STATS" | awk -F'\t' '$1 == "cache_miss" {n+=$2} END{print n+0}')
COMPILED=$( [ -s "$CUW_LEDGER" ] && wc -l < "$CUW_LEDGER" || echo 0 )
echo "=== ccache: $HITS hit(s) / $MISSES miss(es); ledger records $COMPILED compile(s)"

# ---- a build with no nvcc translation unit at all -----------------------
# The seat wrapper is both the ledger and the cache, and it sees nvcc only.
# A package whose extension is C++ against the CUDA runtime -- cumm: 39 .cc
# through g++, zero .cu -- never reaches it, so the ledger is empty and
# ccache saw nothing, which reads exactly like "the wrapper never occupied
# the seat". The two are told apart by EVIDENCE, not by a declaration:
# ninja's own .ninja_log, the same record build_win.py reads for L3 on win-64
# (that file is copied beside every recipe, so its parser is importable
# here). If ninja compiled real translation units and NONE of them is a .cu,
# there was nothing for the seat to see: the ninja TUs become the ledger and
# the ccache assertions below are skipped as inapplicable. Anything else --
# no ninja log, or a .cu among the ninja TUs -- leaves every assertion in
# force, so this cannot excuse a wrapper that really did go missing.
CUW_NO_NVCC_TU=0
if [ "$BUILD_RC" -eq 0 ] && [ "$COMPILED" -eq 0 ] && [ "$((HITS + MISSES))" -eq 0 ]; then
  NINJA_TUS=$($PYTHON "$RECIPE_DIR/build_win.py" --ninja-ledger "$PWD" "$CUW_LEDGER" 2>/dev/null || echo 0)
  if [ "${NINJA_TUS:-0}" -gt 0 ]; then
    CUW_NO_NVCC_TU=1
    COMPILED=$NINJA_TUS
    echo "=== ledger: no nvcc translation unit in this build; ninja compiled $NINJA_TUS C++ TU(s), recorded from .ninja_log"
  fi
fi

case "$CUW_MODE" in
  shard)
    # Report a real compile failure as itself. Checking the cache first meant
    # a genuine build error (gcc too new for nvcc, say) surfaced as "the
    # wrapper never occupied the nvcc seat", which sent debugging in exactly
    # the wrong direction.
    if [ "$BUILD_RC" -ne 0 ]; then
      echo "::error::shard build failed (exit $BUILD_RC) -- see the compiler output above" >&2
      exit "$BUILD_RC"
    fi
    # A shard that compiled nothing is not automatically wrong (its slice can
    # be empty), but a shard whose wrapper never ran at all is a real defect:
    # it would ship an empty cache and the link job would recompile the world.
    if [ "$((HITS + MISSES))" -eq 0 ] && [ "$CUW_NO_NVCC_TU" -eq 0 ]; then
      echo "::error::shard saw zero ccache lookups -- the wrapper never occupied the nvcc seat, or it stubbed out every translation unit (check CUW_SHARD_INDEX0 against CUW_SHARD_COUNT)" >&2
      exit 1
    fi
    if [ "$COMPILED" -eq 0 ] && [ "$CUW_SHARD_COUNT" -eq 1 ]; then
      # With a single shard there is no slice to be empty: compiling nothing
      # means the partition logic is wrong, not that this shard had no work.
      echo "::error::single-shard build compiled 0 translation units -- the partition stubbed everything out" >&2
      exit 1
    fi
    echo "shard $CUW_SHARD_INDEX done; cache populated. Exiting before install."
    exit 0
    ;;
  link)
    if [ "$BUILD_RC" -ne 0 ]; then exit "$BUILD_RC"; fi
    # ZERO tolerance, not a ratio. One miss is a whole TU recompiled, and a
    # percentage cannot tell "one nondeterministic TU" from "four shards built
    # for the wrong architecture".
    if [ "$((HITS + MISSES))" -eq 0 ] && [ "$CUW_NO_NVCC_TU" -eq 0 ]; then
      echo "::error::link job saw zero ccache lookups -- no shard cache was restored" >&2
      exit 1
    fi
    if [ "$MISSES" -gt 0 ]; then
      echo "::error::link job had $MISSES ccache miss(es) of $((HITS+MISSES)). Shard caches did not transfer cleanly: flag mismatch, path mismatch, or wrong-architecture artifacts." >&2
      printf '%s\n' "$STATS" | grep -iE 'miss|hit' || true
      exit 1
    fi
    ;;
  full)
    if [ "$BUILD_RC" -ne 0 ]; then exit "$BUILD_RC"; fi
    ;;
esac

# ---- install THAT wheel into $PREFIX ------------------------------------
# Everything above this line is shared with the wheel output; everything below
# is conda packaging. Shard mode has already exited, so this runs exactly once
# per artifact.
#
# --no-index as well as --no-deps: --no-deps stops pip resolving dependencies,
# but a missing one would still send it to PyPI for the wheel itself if the
# glob below ever produced nothing. With --no-index there is no index to fall
# back to, so a broken handoff fails loudly here instead of quietly installing
# somebody else's build. (The seccomp filter would also refuse it -- this is
# the same belt-and-braces as declaring force_source_build alongside L1.)
# Counted with positional parameters rather than a bash array. That used to
# be forced: this script was inlined into the recipe, rattler-build renders
# an inline script through minijinja before running it, and minijinja opens a
# comment on the two-character sequence brace-hash -- exactly what a bash
# array-length expansion starts with -- so the render failed with only
#   Error:   x Script failed to execute
# and no script output at all. The script is file-backed now and file-backed
# scripts are not rendered (rattler_build_script/src/execution.rs), so the
# three minijinja openers are no longer forbidden here. The positional form
# stays because it works and is one line shorter; the history stays because
# the failure mode is invisible enough to be worth remembering if this file
# is ever inlined again.
shopt -s nullglob
set -- "$CUW_WHEELHOUSE"/*.whl
shopt -u nullglob
if [ "$#" -ne 1 ]; then
  echo "::error::expected exactly 1 wheel in $CUW_WHEELHOUSE, found $#: ${*:-none}. The .conda and the published wheel must come from ONE file; anything else means the compile produced nothing or the wheelhouse was not cleared." >&2
  exit 1
fi
CUW_WHEEL="$1"
echo "=== installing $(basename "$CUW_WHEEL") into \$PREFIX"
$PYTHON "$RECIPE_DIR/nonet.py" -- $PYTHON -m pip install "$CUW_WHEEL" \
    --no-deps --no-index --no-build-isolation --force-reinstall -vv

# ---- L3: the compile ledger --------------------------------------------
# Every extension module we are about to ship must correspond to at least one
# translation unit this run actually compiled. Catches what the network denial
# cannot see: a binary vendored into the source tree, or a build system that
# copies a prebuilt .so into place.
SITE="$(: | $PYTHON -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
MODULES=$(find "$SITE" -name '*.so' -newermt '-1 day' 2>/dev/null | wc -l)
if [ "$MODULES" -gt 0 ] && [ "$COMPILED" -eq 0 ] && [ "$HITS" -eq 0 ]; then
  echo "::error::installed $MODULES extension module(s) but the compile ledger is empty and nothing was replayed from cache -- this build did not compile what it is shipping" >&2
  exit 1
fi
echo "ledger: $COMPILED compile(s) recorded for $MODULES installed module(s)"

# ---- dist-info hygiene --------------------------------------------------
# pip records where it installed FROM in direct_url.json -- a file:// URL to
# this runner's wheelhouse now that the install is `pip install <wheel>`
# rather than `pip install .` (it was $SRC_DIR before; either way it is a path
# on a machine nobody else has). It makes `pip freeze` emit that URL instead
# of a version, so it must not ship. RECORD goes for the same reason conda-forge drops it: with
# it present, `pip uninstall` will happily delete files conda owns.
#
# INSTALLER is deliberately NOT written here. rattler-build 0.75.0 rewrites it
# during packaging no matter what the build script leaves behind -- measured:
# a script that writes exactly "conda" (5 bytes, verified with od at the end
# of the script) still produces "conda\n" (6 bytes) inside the .conda. So a
# write here would be dead code that looks load-bearing.
# Scoped to the dist-info of the wheel THIS build installed, not every
# dist-info in the prefix. The glob form deleted direct_url.json and RECORD
# from host-environment packages the build does not own -- filelock, fsspec,
# jinja2 and markupsafe were all observed losing theirs. It gains the .conda
# nothing, because rattler-build packages files that APPEARED in $PREFIX and
# those packages were already there; it just corrupts the build environment's
# own metadata. Whether it does visible harm depends on which host packages
# happen to carry a RECORD, which is not a property to rely on.
#
# The name comes from the wheel: PEP 427 fixes the first two filename fields
# as the escaped name and version, and pip installs that as
# "<name>-<version>.dist-info".
CUW_WHL_BASE="$(basename "$CUW_WHEEL")"
CUW_DI="$SITE/$(echo "$CUW_WHL_BASE" | cut -d- -f1,2).dist-info"
if [ -d "$CUW_DI" ]; then
  # ---- strip the shipped shared objects ---------------------------------
  # RECORD is the one authoritative list of what THIS package installed, so
  # it is read here, before it is removed, and only the .so files it names
  # are touched -- never the host environment's. --strip-unneeded removes
  # the symbols a shared object does not need for dynamic linking (debug and
  # local symbols); SASS lives in ELF sections, not symbols, so the
  # cuobjdump census downstream sees exactly what it saw before. The wheel
  # was taken before this point and is unaffected; auditwheel rewrites it
  # anyway. $STRIP is conda-forge's binutils from the compiler activation.
  CUW_STRIP="${STRIP:-strip}"
  CUW_STRIPPED=0
  while IFS=, read -r cuw_rel _; do
    case "$cuw_rel" in *.so|*.so.*) ;; *) continue ;; esac
    cuw_abs="$SITE/$cuw_rel"
    [ -f "$cuw_abs" ] || continue
    if "$CUW_STRIP" --strip-unneeded "$cuw_abs"; then
      CUW_STRIPPED=$((CUW_STRIPPED + 1))
    else
      echo "::error::$CUW_STRIP --strip-unneeded failed on $cuw_rel" >&2
      exit 1
    fi
  done < "$CUW_DI/RECORD"
  echo "stripped $CUW_STRIPPED shared object(s) listed in $(basename "$CUW_DI")/RECORD"
  # REQUESTED is pip's "installed by explicit request" marker, a fact about
  # this build's pip invocation and nothing about the consumer's env.
  # RECORD's removal has a consequence worth knowing: without it `pip
  # uninstall` refuses ("Cannot uninstall ... no RECORD file"), which is the
  # right answer for a file set conda owns. build.files.exclude in the recipe
  # removes all three again at packaging, so an early exit cannot ship them.
  rm -f "$CUW_DI/direct_url.json" "$CUW_DI/RECORD" "$CUW_DI/REQUESTED"
  echo "dist-info hygiene: scrubbed $(basename "$CUW_DI")"
else
  # Not a warning to ignore: it means the wheel's own dist-info is not where
  # its filename says, so direct_url.json would ship with a path from this
  # machine in it.
  echo "::error::expected $CUW_DI from $CUW_WHL_BASE, and it is not there -- refusing to ship a dist-info that may still carry direct_url.json" >&2
  exit 1
fi
if [ -s "$CUW_RSS_LOG" ]; then
  echo "=== nvcc peak RSS per TU (top 10) ==="
  # `|| true` is load-bearing. Under `set -o pipefail`, sort takes SIGPIPE
  # (141) whenever head stops reading before sort has finished writing, and
  # this is the LAST pipeline in the script -- so its status is the script's.
  # A package with a handful of translation units never saw it; spconv's
  # link job (699 entries, run 34592223234) compiled, replayed at zero
  # misses, installed its wheel and scrubbed its dist-info, and then exited
  # 141 on this line with nothing wrong.
  sort -t= -k2 -nr "$CUW_RSS_LOG" | head -10 || true
fi
