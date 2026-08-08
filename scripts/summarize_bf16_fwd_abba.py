#!/usr/bin/env python3
"""Aggregate the final-vs-old-c18 BF16 forward common-matrix ABBA measurements."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DSE = ROOT / "dse_results" / "bf16_fwd_figure_common_20260808" / "raw"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "bf16_fwd_abba_20260808.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noncausal-dir", type=Path, default=DSE / "noncausal")
    parser.add_argument("--causal-dir", type=Path, default=DSE / "causal")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def trimmed_mean(values: list[float], fraction: float = 0.10) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    kept = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.fmean(kept)


def read_directory(
    directory: Path,
) -> tuple[dict[tuple[object, ...], float], dict[tuple[int, int, int], dict[str, str]]]:
    manifests = sorted(directory.glob("*_abba_manifest.csv"))
    if len(manifests) != 1:
        raise RuntimeError(f"expected one ABBA manifest in {directory}, found {len(manifests)}")
    with manifests[0].open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if {row["status"] for row in manifest} != {"passed"}:
        raise RuntimeError(f"ABBA manifest contains an incomplete position: {manifests[0]}")
    if {row["restore_status"] for row in manifest} != {"restored"}:
        raise RuntimeError(f"ABBA manifest contains a restore failure: {manifests[0]}")
    if {int(row["requested_physical_device"]) for row in manifest} != {1}:
        raise RuntimeError("the BF16 figure accepts physical GPU1 measurements only")

    samples: dict[tuple[object, ...], list[float]] = defaultdict(list)
    metadata: dict[tuple[int, int, int], dict[str, str]] = {}
    for path in sorted(directory.glob("*_raw.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["dtype"] != "bf16":
                    continue
                causal = int(row["causal"])
                headdim, seqlen = int(row["headdim"]), int(row["seqlen"])
                key = (
                    causal,
                    row["binary_label"],
                    int(row["round_index"]),
                    row["binary_position"],
                    headdim,
                    seqlen,
                )
                samples[key].append(float(row["fwd_ms"]))
                metadata[causal, headdim, seqlen] = row
    if not samples:
        raise RuntimeError(f"no BF16 raw trials found in {directory}")
    if {len(values) for values in samples.values()} != {100}:
        raise RuntimeError("each process position must contain exactly 100 trials")
    return {key: trimmed_mean(values) for key, values in samples.items()}, metadata


def route(causal: int, headdim: int, seqlen: int) -> str:
    if headdim == 64:
        return "fast path"
    if headdim == 128 and seqlen >= 1024:
        return "fast path"
    # D256 non-causal was rejected, while its causal specialization only covers
    # Q64/Q128; neither applies to the common square S512+ matrix.
    return "gated fallback"


def main() -> None:
    args = parse_args()
    all_means: dict[tuple[object, ...], float] = {}
    all_metadata: dict[tuple[int, int, int], dict[str, str]] = {}
    for directory in (args.noncausal_dir, args.causal_dir):
        means, metadata = read_directory(directory)
        all_means.update(means)
        all_metadata.update(metadata)

    rows: list[dict[str, object]] = []
    rounds = 3
    positions = rounds * 2
    for case in sorted(all_metadata):
        causal, headdim, seqlen = case
        meta = all_metadata[case]
        baseline = [
            value
            for (mask, label, _round, _position, dim, seq), value in all_means.items()
            if (mask, label, dim, seq) == (causal, "safe", headdim, seqlen)
        ]
        final = [
            value
            for (mask, label, _round, _position, dim, seq), value in all_means.items()
            if (mask, label, dim, seq) == (causal, "native", headdim, seqlen)
        ]
        if len(baseline) != positions or len(final) != positions:
            raise RuntimeError(
                f"expected {positions} positions per package for causal={causal}, D={headdim}, "
                f"S={seqlen}; found baseline={len(baseline)}, final={len(final)}"
            )

        round_speedups = []
        for round_index in range(rounds):
            round_baseline = [
                value
                for (mask, label, number, _position, dim, seq), value in all_means.items()
                if (mask, label, number, dim, seq)
                == (causal, "safe", round_index, headdim, seqlen)
            ]
            round_final = [
                value
                for (mask, label, number, _position, dim, seq), value in all_means.items()
                if (mask, label, number, dim, seq)
                == (causal, "native", round_index, headdim, seqlen)
            ]
            round_speedups.append(statistics.fmean(round_baseline) / statistics.fmean(round_final))

        baseline_ms = statistics.median(baseline)
        final_ms = statistics.median(final)
        rows.append(
            {
                "dtype": "BF16",
                "direction": "FWD",
                "mask": "causal" if causal else "non-causal",
                "headdim": headdim,
                "seqlen": seqlen,
                "route": route(causal, headdim, seqlen),
                "baseline": "old c18",
                "optimized": "final package",
                "baseline_ms": baseline_ms,
                "optimized_ms": final_ms,
                "speedup": baseline_ms / final_ms,
                "speedup_min_round": min(round_speedups),
                "speedup_max_round": max(round_speedups),
                "aggregation": "median of per-position 10%-trimmed means",
                "rounds": rounds,
                "positions": positions,
                "warmup": int(meta["warmup"]),
                "trials": int(meta["trials"]),
                "iterations": int(meta["iters"]),
                "batch": int(meta["batch"]),
                "heads": int(meta["nheads"]),
                "gpu": f"{meta['device_name']} (physical GPU1)",
                "arch": meta["device_arch"],
                "measurement_set": "three-round common square matrix",
            }
        )

    expected_cases = {
        (mask, dim, seq)
        for mask in ("non-causal", "causal")
        for dim in (64, 128, 256)
        for seq in (512, 1024, 2048, 4096)
    }
    actual_cases = {(row["mask"], row["headdim"], row["seqlen"]) for row in rows}
    if actual_cases != expected_cases:
        raise RuntimeError(f"BF16 figure matrix mismatch: {sorted(actual_cases ^ expected_cases)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
