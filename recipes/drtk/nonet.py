#!/usr/bin/env python3
"""Run a command with network access permanently removed from it.

Why this exists
---------------
The from-source guarantee is "the build cannot fetch a prebuilt binary". The
obvious mechanism, rattler-build's `--sandbox`, does not work: it needs a
separate `rattler-sandbox` binary AND unprivileged namespace creation, which
is unavailable both in this project's container and on GitHub-hosted runners
(`sandboxing failure: Operation not permitted (os error 1)`, measured on
both). A flag that silently does nothing is worse than no flag.

`sudo unshare -n` would work on hosted runners, but it wraps the *whole*
rattler-build invocation — which legitimately needs the network to solve and
download the build and host environments. Blocking egress around all of it
would break the build for a real reason, and blocking it around nothing is
what we are trying to avoid.

So the restriction is imposed by the build on itself, at the one moment that
matters: the command that could download a wheel. A seccomp filter that
returns EPERM for `socket(AF_INET|AF_INET6, ...)` is:

  * unprivileged — no root, no namespaces, no capabilities;
  * inherited by every child, so setup.py, pip, cmake and nvcc are all covered;
  * irreversible — PR_SET_NO_NEW_PRIVS plus a seccomp filter cannot be removed
    by the process that installed it, so a build script cannot opt back out;
  * narrow — AF_UNIX still works, so anything using local sockets is unaffected.

Usage:
    nonet.py -- <command> [args...]

Exits 127 if the filter cannot be installed. It must never fall through to
running the command unprotected: silently degrading to "no guarantee" is the
failure mode this whole design exists to prevent.
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys

# linux/audit.h
AUDIT_ARCH_X86_64 = 0xC000003E
AUDIT_ARCH_AARCH64 = 0xC00000B7

# syscall numbers for socket(2)
NR_SOCKET = {"x86_64": 41, "aarch64": 198}

# linux/seccomp.h
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_SET_MODE_FILTER = 1

PR_SET_NO_NEW_PRIVS = 38

AF_INET, AF_INET6 = 2, 10
EPERM = 1

# BPF (linux/bpf_common.h)
BPF_LD, BPF_W, BPF_ABS = 0x00, 0x00, 0x20
BPF_JMP, BPF_JEQ, BPF_K = 0x05, 0x10, 0x00
BPF_RET = 0x06

# offsets into struct seccomp_data
OFF_NR = 0
OFF_ARCH = 4
OFF_ARG0 = 16  # args[0], low 32 bits (little-endian)


def stmt(code: int, k: int) -> bytes:
    return struct.pack("HBBI", code, 0, 0, k)


def jump(code: int, k: int, jt: int, jf: int) -> bytes:
    return struct.pack("HBBI", code, jt, jf, k)


def build_filter(arch_token: int, nr_socket: int) -> bytes:
    """Deny socket(AF_INET|AF_INET6, ...) with EPERM; allow everything else."""
    # Jump offsets are relative to the NEXT instruction, so they must be read
    # against the final layout:
    #   0 LD arch | 1 JEQ arch | 2 LD nr | 3 JEQ socket | 4 LD arg0
    #   5 JEQ AF_INET | 6 JEQ AF_INET6 | 7 RET ALLOW | 8 RET EPERM | 9 RET KILL
    # The arch mismatch branch MUST reach 9, not 7: a process whose syscall
    # numbering we cannot map (a 32-bit child reports AUDIT_ARCH_I386, where
    # the numbers mean entirely different calls) cannot be filtered correctly,
    # so it is killed rather than run unprotected. An earlier version jumped
    # to 7 — RET ALLOW — which silently gave exactly those processes
    # unrestricted network, the textbook seccomp bypass.
    prog = b"".join([
        stmt(BPF_LD | BPF_W | BPF_ABS, OFF_ARCH),
        jump(BPF_JMP | BPF_JEQ | BPF_K, arch_token, 0, 7),   # mismatch -> 9 KILL
        # if nr != socket -> allow
        stmt(BPF_LD | BPF_W | BPF_ABS, OFF_NR),
        jump(BPF_JMP | BPF_JEQ | BPF_K, nr_socket, 0, 3),    # not socket -> 7 ALLOW
        # args[0] == AF_INET or AF_INET6 -> EPERM
        stmt(BPF_LD | BPF_W | BPF_ABS, OFF_ARG0),
        jump(BPF_JMP | BPF_JEQ | BPF_K, AF_INET, 2, 0),      # -> 8 EPERM
        jump(BPF_JMP | BPF_JEQ | BPF_K, AF_INET6, 1, 0),     # -> 8 EPERM
        stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),            # 7
        stmt(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),    # 8
        stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),     # 9
    ])
    return prog


def install() -> None:
    machine = os.uname().machine
    if machine == "x86_64":
        arch_token, nr = AUDIT_ARCH_X86_64, NR_SOCKET["x86_64"]
    elif machine in ("aarch64", "arm64"):
        arch_token, nr = AUDIT_ARCH_AARCH64, NR_SOCKET["aarch64"]
    else:
        sys.exit(f"nonet: unsupported architecture {machine!r}; refusing to run "
                 f"a build without the no-network guarantee")

    prog = build_filter(arch_token, nr)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)

    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        sys.exit(f"nonet: PR_SET_NO_NEW_PRIVS failed (errno {ctypes.get_errno()})")

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    buf = ctypes.create_string_buffer(prog, len(prog))
    fprog = SockFprog(len(prog) // 8, ctypes.cast(buf, ctypes.c_void_p))

    # seccomp(2) directly; prctl(PR_SET_SECCOMP) would also work but the
    # syscall is the documented modern entry point.
    rc = libc.syscall(317 if machine == "x86_64" else 277,  # __NR_seccomp
                      SECCOMP_SET_MODE_FILTER, 0, ctypes.byref(fprog))
    if rc != 0:
        sys.exit(f"nonet: seccomp(SECCOMP_SET_MODE_FILTER) failed "
                 f"(errno {ctypes.get_errno()}); refusing to run the build "
                 f"without the no-network guarantee")


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    install()
    try:
        os.execvp(argv[0], argv)
    except OSError as exc:
        print(f"nonet: cannot exec {argv[0]!r}: {exc}", file=sys.stderr)
        return 127
    return 127


if __name__ == "__main__":
    sys.exit(main())
