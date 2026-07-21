#!/usr/bin/env python3
import csv
import os
import re
import statistics
import sys


def classify_kernel(name: str) -> str | None:
    if "FmhaFwdKernel" in name:
        return "fwd"
    if "FmhaBwdOGradDotOKernel" in name:
        return "dot_do_o"
    if "FmhaBwdConvertQGradKernel" in name:
        return "convert_dq"
    if "BlockFmhaBwdDQPipelineKRKTRVRIGLP" in name:
        return "split_dq"
    if "BlockFmhaBwdDKDVPipelineKRKTRVRIGLP" in name:
        match = re.search(
            r"BlockFmhaBwdPipelineDefaultPolicy,\s*(true|false),\s*(true|false)>",
            name,
        )
        if not match:
            return "dkdv_unknown"
        flags = ",".join(match.groups())
        return {
            "false,true": "split_dv",
            "true,false": "split_dkdv_or_dk",
        }.get(flags, f"dkdv_{flags}")
    return None


def summarize(path: str, chunks_to_keep: int = 3) -> None:
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            category = classify_kernel(row["Kernel_Name"])
            if not category:
                continue
            row["_category"] = category
            row["_ms"] = (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1e6
            rows.append(row)

    chunks: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for row in rows:
        if row["_category"] == "fwd" and current:
            chunks.append(current)
            current = []
        current.append(row)
    if current:
        chunks.append(current)
    chunks = chunks[-chunks_to_keep:]

    values: dict[str, list[float]] = {}
    metas: dict[str, dict[str, str]] = {}
    for chunk in chunks:
        for row in chunk:
            category = row["_category"]
            values.setdefault(category, []).append(row["_ms"])
            metas.setdefault(category, row)

    print(f"\n{path}")
    for category in [
        "fwd",
        "dot_do_o",
        "split_dv",
        "split_dkdv_or_dk",
        "split_dq",
        "convert_dq",
    ]:
        if category not in values:
            continue
        meta = metas[category]
        print(
            "{:<16s} avg_ms={:8.4f} n={:2d} grid={} wg={} LDS={} scratch={} "
            "vgpr={} sgpr={}".format(
                category,
                statistics.mean(values[category]),
                len(values[category]),
                meta["Grid_Size"],
                meta["Workgroup_Size"],
                meta["LDS_Per_Workgroup"],
                meta["Scratch_Per_Workitem"],
                meta["Arch_VGPR"],
                meta["SGPR"],
            )
        )


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {os.path.basename(sys.argv[0])} ROCProfiler.csv [...]", file=sys.stderr)
        return 2
    for path in sys.argv[1:]:
        summarize(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
