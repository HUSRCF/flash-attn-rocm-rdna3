#!/usr/bin/env python3
"""Summarize AMDGPU ISA per kernel label instead of per code object."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


LABEL_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s+<(.+)>:")
DEFAULT_PATTERNS = {
    "v_wmma": r"\bv_wmma",
    "scratch_load": r"\bscratch_load",
    "scratch_store": r"\bscratch_store",
    "ds_load": r"\bds_load",
    "ds_store": r"\bds_store",
    "global_atomic_cmpswap": r"\bglobal_atomic_cmpswap",
    "atomic": r"\b(?:global|flat|buffer)_atomic",
    "s_waitcnt": r"\bs_waitcnt\b",
    "s_waitcnt_vmcnt0": r"\bs_waitcnt\s+vmcnt\(0\)",
    "s_barrier": r"\bs_barrier\b",
    "buffer_load": r"\bbuffer_load",
    "buffer_store": r"\bbuffer_store",
}


def split_kernels(text: str) -> list[tuple[str, str, list[str]]]:
    kernels: list[tuple[str, str, list[str]]] = []
    current_addr: str | None = None
    current_name: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        match = LABEL_RE.match(line)
        if match:
            if current_name is not None:
                kernels.append((current_addr or "", current_name, current_lines))
            current_addr, current_name = match.group(1), match.group(2)
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        kernels.append((current_addr or "", current_name, current_lines))
    return kernels


def summarize(lines: list[str]) -> Counter[str]:
    joined = "\n".join(lines)
    return Counter({key: len(re.findall(pattern, joined)) for key, pattern in DEFAULT_PATTERNS.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isa", type=Path, help="AMDGPU llvm-objdump/roc-obj disassembly file")
    parser.add_argument("--match", default="", help="Regex applied to kernel symbol names")
    parser.add_argument("--index", type=int, default=None, help="Only print the Nth matched kernel, 0-based")
    parser.add_argument("--show-name", action="store_true", help="Print full mangled kernel names")
    args = parser.parse_args()

    text = args.isa.read_text(errors="replace")
    kernels = split_kernels(text)
    name_re = re.compile(args.match) if args.match else None
    matched = [(addr, name, lines) for addr, name, lines in kernels if name_re is None or name_re.search(name)]

    if args.index is not None:
        if args.index < 0 or args.index >= len(matched):
            raise SystemExit(f"--index {args.index} out of range for {len(matched)} matched kernels")
        matched = [matched[args.index]]

    print(f"file={args.isa}")
    print(f"total_kernels={len(kernels)} matched_kernels={len(matched)}")
    for i, (addr, name, lines) in enumerate(matched):
        counts = summarize(lines)
        short_name = name if args.show_name else name[:180] + ("..." if len(name) > 180 else "")
        print(f"\n[{i}] addr=0x{addr} lines={len(lines)}")
        print(f"name={short_name}")
        for key in DEFAULT_PATTERNS:
            print(f"{key}={counts[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
