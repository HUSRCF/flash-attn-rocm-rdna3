#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


PATTERNS = [
    ("split_dv", re.compile(r"BlockFmhaBwdDKDVPipelineKRKTRVRIGLP.*DefaultPolicyELb0ELb1")),
    ("split_dkdv", re.compile(r"BlockFmhaBwdDKDVPipelineKRKTRVRIGLP.*DefaultPolicyELb1ELb1")),
    ("split_dk", re.compile(r"BlockFmhaBwdDKDVPipelineKRKTRVRIGLP.*DefaultPolicyELb1ELb0")),
    ("split_dq_bn64", re.compile(r"BlockFmhaBwdDQPipelineKRKTRVRIGLP.*Li16ELi64ELi256")),
    ("split_dq_bn32", re.compile(r"BlockFmhaBwdDQPipelineKRKTRVRIGLP.*Li16ELi32ELi256")),
]


def count_block(block: str) -> dict[str, int]:
    return {
        "lines": block.count("\n") + 1,
        "scratch_load": block.count("scratch_load"),
        "scratch_store": block.count("scratch_store"),
        "buffer_load": block.count("buffer_load"),
        "buffer_store": block.count("buffer_store"),
        "buffer_atomic": block.count("buffer_atomic"),
        "ds_read": block.count("ds_read") + block.count("ds_load"),
        "ds_write": block.count("ds_write") + block.count("ds_store"),
        "s_waitcnt": block.count("s_waitcnt"),
        "s_barrier": block.count("s_barrier"),
        "wmma_mfma": block.count("v_wmma") + block.count("v_mfma"),
        "s_branch": len(re.findall(r"\bs_branch\b", block)),
        "s_cbranch": block.count("s_cbranch"),
        "v_cmp": block.count("v_cmp"),
        "s_delay_alu": block.count("s_delay_alu"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("isa", type=Path)
    args = parser.parse_args()

    lines = args.isa.read_text(errors="ignore").splitlines()
    starts: list[tuple[int, str]] = [
        (i, line) for i, line in enumerate(lines) if re.match(r"^[0-9a-f]+ <.*>:", line)
    ]
    starts.append((len(lines), "END"))

    for idx, (start, label) in enumerate(starts[:-1]):
        name = next((name for name, pattern in PATTERNS if pattern.search(label)), None)
        if name is None:
            continue
        end = starts[idx + 1][0]
        block = "\n".join(lines[start:end])
        counts = count_block(block)
        print(f"## {name}")
        print(f"file={args.isa}")
        print(f"start_line={start + 1} end_line={end} label_len={len(label)}")
        print(" ".join(f"{key}={value}" for key, value in counts.items()))
        print(f"label={label[:260]}")
        print()


if __name__ == "__main__":
    main()
