#!/usr/bin/env python3
"""Aggregate the final BF16 causal FWD ABBA measurements.

Each plotted latency is the median of six process-level measurements (two
positions in each of three ABBA rounds).  Each process-level measurement is a
10%-trimmed mean of its raw trials.  This makes the checked-in summary robust
to an isolated host-load excursion without hiding repeat-to-repeat behavior.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DSE = ROOT / "dse_results" / "bf16_fwd_20260808" / "outputs"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "bf16_causal_abba_20260808.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d64-dir", type=Path, default=DSE / "final_push_abba_d64_h8")
    parser.add_argument("--d128-dir", type=Path, default=DSE / "final_push_abba_d128_h8")
    parser.add_argument("--d256-dir", type=Path, default=DSE / "final_push_abba_d256_h8")
    parser.add_argument(
        "--d256-gate-dir",
        type=Path,
        default=DSE / "final_push_abba_d256_gate_recheck",
        help="Post-gate run used only for the D256 Q64/K64 legacy-fallback point.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def trimmed_mean(values: list[float], fraction: float = 0.10) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    kept = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.fmean(kept)


def aggregate(directory: Path) -> dict[tuple[int, int], dict[str, object]]:
    samples: dict[tuple[str, int, str, int, int], list[float]] = defaultdict(list)
    metadata: dict[tuple[int, int], dict[str, str]] = {}
    for path in sorted(directory.glob("[0-9]*_*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                q_len, k_len = int(row["q_len"]), int(row["k_len"])
                key = (row["label"], int(row["round"]), row["position"], q_len, k_len)
                samples[key].append(float(row["fwd_ms"]))
                metadata[q_len, k_len] = row
    if not samples:
        raise RuntimeError(f"no ABBA samples found in {directory}")

    per_position = {key: trimmed_mean(values) for key, values in samples.items()}
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
        if len(baseline) != 6 or len(candidate) != 6:
            raise RuntimeError(
                f"expected six positions per label for Q={q_len}, K={k_len}; "
                f"found baseline={len(baseline)}, candidate={len(candidate)}"
            )
        baseline_ms = statistics.median(baseline)
        optimized_ms = statistics.median(candidate)
        round_speedups = []
        for round_index in (1, 2, 3):
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
            round_speedups.append(
                statistics.fmean(round_baseline) / statistics.fmean(round_candidate)
            )
        result[case] = {
            "dtype": "BF16",
            "direction": "FWD",
            "mask": "causal",
            "headdim": int(meta["dim"]),
            "q_len": q_len,
            "k_len": k_len,
            "route": "fast path",
            "baseline_ms": baseline_ms,
            "optimized_ms": optimized_ms,
            "speedup": baseline_ms / optimized_ms,
            "speedup_min_round": min(round_speedups),
            "speedup_max_round": max(round_speedups),
            "aggregation": "median of per-position 10%-trimmed means",
            "rounds": 3,
            "positions": 6,
            "warmup": int(meta["warmup"]),
            "trials": len(next(iter(samples.values()))),
            "iterations": int(meta["iters"]),
            "gpu": f"{meta['device']} (physical GPU1)",
            "measurement_set": "final full ABBA",
        }
    return result


def main() -> None:
    args = parse_args()
    combined: list[dict[str, object]] = []
    for directory in (args.d64_dir, args.d128_dir, args.d256_dir):
        combined.extend(aggregate(directory).values())

    gate_point = aggregate(args.d256_gate_dir)[64, 64]
    gate_point["route"] = "legacy fallback"
    gate_point["measurement_set"] = "post-gate boundary recheck"
    combined = [
        row
        for row in combined
        if not (row["headdim"] == 256 and row["q_len"] == 64 and row["k_len"] == 64)
    ]
    combined.append(gate_point)
    combined.sort(key=lambda row: (row["headdim"], row["q_len"], row["k_len"]))

    fieldnames = list(combined[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    print(args.output)


if __name__ == "__main__":
    main()
