#!/usr/bin/env python3
"""Aggregate BF16 non-causal and causal forward ABBA measurements.

Each reported latency is the median of process-level measurements (two
positions per ABBA round). Each process-level measurement is a 10%-trimmed
mean of 100 raw trials. Non-causal and causal bars use their
matching pre-fast-path package so each ratio isolates the dispatch path named
by the row instead of including later, unrelated dispatch additions.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DSE = ROOT / "dse_results" / "bf16_fwd_20260808" / "outputs"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "bf16_fwd_abba_20260808.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noncausal-d64-dir", type=Path, default=DSE / "final_total_abba_d64_nc_h8")
    parser.add_argument(
        "--noncausal-d128-dir",
        type=Path,
        default=DSE / "final_total_abba_d128_nc_h8_recheck5",
    )
    parser.add_argument(
        "--noncausal-d256-dir",
        type=Path,
        default=DSE / "final_total_abba_d256_nc_h8_recheck5",
    )
    parser.add_argument("--causal-d64-dir", type=Path, default=DSE / "final_push_abba_d64_h8")
    parser.add_argument("--causal-d128-dir", type=Path, default=DSE / "final_push_abba_d128_h8")
    parser.add_argument("--causal-d256-dir", type=Path, default=DSE / "final_push_abba_d256_h8")
    parser.add_argument(
        "--causal-d256-gate-dir",
        type=Path,
        default=DSE / "final_push_abba_d256_gate_recheck",
        help="Post-gate run used only for the causal D256 Q64/K64 fallback point.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def trimmed_mean(values: list[float], fraction: float = 0.10) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    kept = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.fmean(kept)


def route(mask: str, headdim: int, q_len: int, k_len: int) -> str:
    if headdim == 256 and mask == "non-causal":
        return "gated fallback"
    if headdim == 256 and mask == "causal" and (q_len, k_len) == (64, 64):
        return "gated fallback"
    return "fast path"


def aggregate(directory: Path, mask: str, measurement_set: str) -> dict[tuple[int, int], dict[str, object]]:
    expected_causal = 1 if mask == "causal" else 0
    samples: dict[tuple[str, int, str, int, int], list[float]] = defaultdict(list)
    metadata: dict[tuple[int, int], dict[str, str]] = {}
    for path in sorted(directory.glob("[0-9]*_*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["dtype"] != "bf16" or int(row["causal"]) != expected_causal:
                    raise RuntimeError(f"unexpected dtype/mask metadata in {path}")
                q_len, k_len = int(row["q_len"]), int(row["k_len"])
                key = (row["label"], int(row["round"]), row["position"], q_len, k_len)
                samples[key].append(float(row["fwd_ms"]))
                metadata[q_len, k_len] = row
    if not samples:
        raise RuntimeError(f"no ABBA samples found in {directory}")
    if {len(values) for values in samples.values()} != {100}:
        raise RuntimeError(f"each process position must contain 100 trials in {directory}")

    per_position = {key: trimmed_mean(values) for key, values in samples.items()}
    round_numbers = sorted({round_number for (_label, round_number, _position, _q, _k) in samples})
    if round_numbers != list(range(1, len(round_numbers) + 1)):
        raise RuntimeError(f"non-contiguous ABBA rounds in {directory}: {round_numbers}")
    positions = len(round_numbers) * 2
    result: dict[tuple[int, int], dict[str, object]] = {}
    for case, meta in metadata.items():
        q_len, k_len = case
        baseline = [
            value
            for (label, _round, _position, q, k), value in per_position.items()
            if label == "baseline" and (q, k) == case
        ]
        candidate = [
            value
            for (label, _round, _position, q, k), value in per_position.items()
            if label == "candidate" and (q, k) == case
        ]
        if len(baseline) != positions or len(candidate) != positions:
            raise RuntimeError(
                f"expected {positions} positions per label for Q={q_len}, K={k_len}; "
                f"found baseline={len(baseline)}, candidate={len(candidate)}"
            )

        round_speedups = []
        for round_index in round_numbers:
            round_baseline = [
                value
                for (label, round_number, _position, q, k), value in per_position.items()
                if label == "baseline" and round_number == round_index and (q, k) == case
            ]
            round_candidate = [
                value
                for (label, round_number, _position, q, k), value in per_position.items()
                if label == "candidate" and round_number == round_index and (q, k) == case
            ]
            if len(round_baseline) != 2 or len(round_candidate) != 2:
                raise RuntimeError(f"incomplete ABBA round {round_index} for Q={q_len}, K={k_len}")
            round_speedups.append(
                statistics.fmean(round_baseline) / statistics.fmean(round_candidate)
            )

        baseline_ms = statistics.median(baseline)
        optimized_ms = statistics.median(candidate)
        headdim = int(meta["dim"])
        result[case] = {
            "dtype": "BF16",
            "direction": "FWD",
            "mask": mask,
            "headdim": headdim,
            "q_len": q_len,
            "k_len": k_len,
            "route": route(mask, headdim, q_len, k_len),
            "baseline": "matching pre-path package",
            "optimized": "final package",
            "baseline_ms": baseline_ms,
            "optimized_ms": optimized_ms,
            "speedup": statistics.median(round_speedups),
            "speedup_min_round": min(round_speedups),
            "speedup_max_round": max(round_speedups),
            "aggregation": "median paired ABBA-round speedup; per-position 10%-trimmed means",
            "rounds": len(round_numbers),
            "positions": positions,
            "warmup": int(meta["warmup"]),
            "trials": 100,
            "iterations": int(meta["iters"]),
            "batch": int(meta["batch"]),
            "heads": int(meta["nheads"]),
            "gpu": f"{meta['device']} (physical GPU1)",
            "arch": meta["arch"],
            "measurement_set": measurement_set,
        }
    return result


def main() -> None:
    args = parse_args()
    combined: list[dict[str, object]] = []
    for directory in (
        args.noncausal_d64_dir,
        args.noncausal_d128_dir,
        args.noncausal_d256_dir,
    ):
        combined.extend(aggregate(directory, "non-causal", "non-causal final ABBA").values())
    for directory in (args.causal_d64_dir, args.causal_d128_dir, args.causal_d256_dir):
        combined.extend(aggregate(directory, "causal", "causal final ABBA").values())

    gate_point = aggregate(
        args.causal_d256_gate_dir,
        "causal",
        "causal post-gate boundary recheck",
    )[64, 64]
    combined = [
        row
        for row in combined
        if not (
            row["mask"] == "causal"
            and row["headdim"] == 256
            and row["q_len"] == 64
            and row["k_len"] == 64
        )
    ]
    combined.append(gate_point)

    mask_order = {"non-causal": 0, "causal": 1}
    combined.sort(
        key=lambda row: (
            row["headdim"],
            row["q_len"],
            row["k_len"],
            mask_order[row["mask"]],
        )
    )
    expected_counts = {64: 12, 128: 12, 256: 16}
    actual_counts = {
        headdim: sum(row["headdim"] == headdim for row in combined)
        for headdim in expected_counts
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"BF16 figure matrix mismatch: {actual_counts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(args.output)


if __name__ == "__main__":
    main()
