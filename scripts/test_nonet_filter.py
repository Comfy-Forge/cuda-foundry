#!/usr/bin/env python3
"""Decode the emitted BPF and assert every branch lands where it must.

The canary proves the filter works on the architecture it was built for.
It cannot prove what happens on a MISMATCHED architecture, because that
path only runs for e.g. a 32-bit child on x86_64 — and that is exactly
where a seccomp filter is usually got wrong: an earlier version of this
one sent the mismatch branch to RET ALLOW while its comment claimed it
killed, which would have given any such process unrestricted network.

So decode the program and walk the branches statically instead.

Usage: test_nonet_filter.py   (exit 0 = all assertions hold)
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "build_snippets"))
import nonet  # noqa: E402

BPF_RET = 0x06
BPF_JMP = 0x05
BPF_JEQ = 0x10


def decode(prog: bytes) -> list[tuple[int, int, int, int]]:
    return [struct.unpack("HBBI", prog[i:i + 8]) for i in range(0, len(prog), 8)]


def ret_at(insns, idx: int) -> int:
    """The RET value at idx, or fail if idx is not a RET."""
    code, _, _, k = insns[idx]
    assert code & 0x07 == BPF_RET, f"index {idx} is not a RET (code {code:#x})"
    return k


def main() -> int:
    prog = nonet.build_filter(nonet.AUDIT_ARCH_X86_64, nonet.NR_SOCKET["x86_64"])
    insns = decode(prog)
    failures = []

    def check(cond: bool, msg: str) -> None:
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            failures.append(msg)

    # instruction 1 is the arch comparison; jf is taken on MISMATCH, and a
    # jump offset counts from the following instruction.
    code, jt, jf, k = insns[1]
    check(code == (BPF_JMP | BPF_JEQ | 0x00) and k in
          (nonet.AUDIT_ARCH_X86_64, nonet.AUDIT_ARCH_AARCH64),
          "instruction 1 compares the audit arch")

    mismatch_target = 1 + 1 + jf
    verdict = ret_at(insns, mismatch_target)
    check(verdict == nonet.SECCOMP_RET_KILL_PROCESS,
          f"arch MISMATCH -> index {mismatch_target} = KILL_PROCESS "
          f"(got {verdict:#x}; ALLOW={nonet.SECCOMP_RET_ALLOW:#x} would be a bypass)")

    # a matching arch must fall through to the syscall test, not to a verdict
    match_target = 1 + 1 + jt
    check(insns[match_target][0] & 0x07 != BPF_RET,
          f"arch match -> index {match_target} continues filtering, does not return")

    # non-socket syscalls are allowed
    _, _, jf3, _ = insns[3]
    t = 3 + 1 + jf3
    check(ret_at(insns, t) == nonet.SECCOMP_RET_ALLOW,
          f"non-socket syscall -> index {t} = ALLOW")

    # AF_INET and AF_INET6 are refused
    for i, fam in ((5, "AF_INET"), (6, "AF_INET6")):
        _, jt_i, _, _ = insns[i]
        t = i + 1 + jt_i
        check(ret_at(insns, t) == (nonet.SECCOMP_RET_ERRNO | nonet.EPERM),
              f"{fam} -> index {t} = ERRNO|EPERM")

    # every jump target must be inside the program
    for i, (code, jt, jf, _) in enumerate(insns):
        if code & 0x07 == BPF_JMP:
            for off, name in ((jt, "jt"), (jf, "jf")):
                t = i + 1 + off
                check(0 <= t < len(insns), f"instruction {i} {name} -> {t} is in range")

    print(f"\n{len(insns)} instructions, {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
