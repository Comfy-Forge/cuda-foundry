#!/bin/sh
# CUW_WRAPPER_MARKER — build.sh greps for this string to tell "the seat is
# already ours" from "this is the real compiler". Without that distinction a
# re-entered build moves the wrapper onto nvcc.real and recurses forever.
# Occupies the nvcc seat ($PREFIX/bin/nvcc, real binary moved to nvcc.real),
# because torch's cpp_extension invokes that path directly and never consults
# PATH. Does three jobs at once:
#
#   1. shard partitioning  — compile only this shard's slice, stub the rest
#   2. compile ledger      — record every TU actually compiled (L3)
#   3. peak-RSS logging    — per-TU memory, the input to jobs/nvcc_threads tuning
#
# Kept POSIX sh and dependency-free: it runs inside the build sandbox with
# whatever minimal environment rattler-build constructed.

src=""; out=""; dep=""; prev=""
for a in "$@"; do
  case "$prev" in
    -o) out="$a" ;;
    --dependency-output) dep="$a" ;;
  esac
  case "$a" in
    *.cu|*.cpp|*.cc|*.cxx|*.c) src="$a" ;;
  esac
  prev="$a"
done

# Build-system probes must NEVER be stubbed: cmake's compiler-id and
# try_compile TUs get linked into probe executables, and a stubbed one has no
# main, so configure fails with "Detecting CUDA compiler ABI info - failed".
# Probes cost seconds; when in doubt, compile.
case "$src|$out" in
  *CMakeScratch*|*CompilerId*|*CMakeTmp*|*cmTC_*|*meson-private*|*conftest*) src="" ;;
esac

if [ -n "$src" ] && [ -n "$out" ] && [ "${CUW_PARTITION:-0}" = "1" ] && [ "${CUW_SHARD_COUNT:-0}" != "0" ]; then
  # Stateless partition: the wrapper sees one TU at a time and cannot know a
  # global index, so the slice is a hash of the resolved path. Uneven at small
  # N, but correct, and the link job's zero-miss gate catches any disagreement
  # between shards.
  rp=$(readlink -f "$src" 2>/dev/null || printf %s "$src")
  h=$(printf %s "$rp" | md5sum | cut -c1-8)
  mine=$(( 0x$h % CUW_SHARD_COUNT ))
  # CUW_SHARD_INDEX0 is 0-based; CUW_SHARD_INDEX from the matrix is 1-based.
  if [ "$mine" -ne "${CUW_SHARD_INDEX0:-0}" ]; then
    # Not my slice: emit a valid empty object so the build system proceeds.
    : > "$out.empty.c"
    "${CUW_REAL_NVCC%nvcc.real}"../bin/cc -x c -c "$out.empty.c" -o "$out" 2>/dev/null \
      || cc -x c -c "$out.empty.c" -o "$out" 2>/dev/null \
      || : > "$out"
    rm -f "$out.empty.c"
    [ -n "$dep" ] && printf '%s:\n' "$out" > "$dep"
    exit 0
  fi
fi

# This TU is ours (or we are not partitioning): record it, then compile.
[ -n "$src" ] && [ -n "${CUW_LEDGER:-}" ] && printf '%s\n' "$src" >> "$CUW_LEDGER"

REAL="${CUW_REAL_NVCC:-nvcc.real}"
if [ -n "${CUW_CCACHE_BIN:-}" ]; then
  set -- "$REAL" "$@"
  if [ -n "${CUW_RSS_LOG:-}" ] && command -v /usr/bin/time >/dev/null 2>&1; then
    tmp=$(mktemp 2>/dev/null) || tmp=""
    if [ -n "$tmp" ]; then
      /usr/bin/time -v -o "$tmp" "$CUW_CCACHE_BIN" "$@" ; rc=$?
      kib=$(awk '/Maximum resident set size/ {print $NF}' "$tmp" 2>/dev/null)
      [ -n "$kib" ] && [ -n "$src" ] && printf '%s=peak_kib=%s\n' "$src" "$kib" >> "$CUW_RSS_LOG"
      rm -f "$tmp"
      exit $rc
    fi
  fi
  exec "$CUW_CCACHE_BIN" "$@"
fi
exec "$REAL" "$@"
