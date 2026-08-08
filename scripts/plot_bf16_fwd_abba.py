#!/usr/bin/env python3
"""Render the BF16 non-causal/causal forward ABBA speedup figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "benchmarks" / "results" / "bf16_fwd_abba_20260808.csv"
DEFAULT_OUTPUT = ROOT / "assets" / "bf16_fwd_abba_speedup"
MASK_ORDER = ["Non-causal", "Causal"]
SEQUENCE_ORDER = ["S512", "S1024", "S2048", "S4096"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def render(data_path: Path, output_base: Path) -> None:
    data = pd.read_csv(data_path)
    required = {"dtype", "direction", "mask", "headdim", "seqlen", "route", "speedup"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if set(data["dtype"]) != {"BF16"} or set(data["direction"]) != {"FWD"}:
        raise ValueError("this figure requires BF16 FWD measurements only")
    if set(data["mask"]) != {"non-causal", "causal"}:
        raise ValueError("this figure requires both non-causal and causal measurements")
    if set(data["route"]) - {"fast path", "gated fallback"}:
        raise ValueError("unknown route label")
    expected_cases = {
        (mask, dim, seq)
        for mask in ("non-causal", "causal")
        for dim in (64, 128, 256)
        for seq in (512, 1024, 2048, 4096)
    }
    actual_cases = set(data[["mask", "headdim", "seqlen"]].itertuples(index=False, name=None))
    if actual_cases != expected_cases:
        raise ValueError("BF16 data does not contain the required common square matrix")

    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        font="DejaVu Serif",
        font_scale=1.05,
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    colors = sns.color_palette("colorblind")
    palette = {"Non-causal": colors[0], "Causal": colors[1]}
    mask_labels = {"non-causal": "Non-causal", "causal": "Causal"}
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharey=False)

    for ax, headdim in zip(axes, (64, 128, 256), strict=True):
        subset = data[data["headdim"] == headdim].copy()
        subset["sequence"] = subset["seqlen"].map(lambda value: f"S{int(value)}")
        subset["mask_label"] = subset["mask"].map(mask_labels)

        sns.barplot(
            data=subset,
            x="sequence",
            y="speedup",
            hue="mask_label",
            order=SEQUENCE_ORDER,
            hue_order=MASK_ORDER,
            palette=palette,
            errorbar=None,
            ax=ax,
        )
        ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
        minimum, maximum = subset["speedup"].min(), subset["speedup"].max()
        lower = max(0.94, min(0.985, minimum - 0.02))
        upper = maximum + max(0.07, 0.10 * maximum)
        ax.set_ylim(lower, upper)
        ax.set_title(f"Head dimension D={headdim}", fontweight="bold", pad=8)
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Speedup (old c18 / final)" if headdim == 64 else "")
        ax.grid(axis="y", color="0.88", linewidth=0.7)

        lookup = {
            (row.sequence, row.mask_label): row
            for row in subset.itertuples(index=False)
        }
        for container, mask_label in zip(ax.containers[:2], MASK_ORDER, strict=True):
            labels = [
                f"{value:.3f}x"
                + ("*" if lookup[sequence, mask_label].route == "gated fallback" else "")
                for value, sequence in zip(container.datavalues, SEQUENCE_ORDER, strict=True)
            ]
            ax.bar_label(container, labels=labels, padding=3, fontsize=7.2)
        if ax.legend_ is not None:
            ax.legend_.remove()

    handles = [
        Patch(facecolor=palette["Non-causal"], label="Non-causal"),
        Patch(facecolor=palette["Causal"], label="Causal"),
    ]
    labels = ["Non-causal", "Causal"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.865), ncol=2, frameon=False)
    fig.suptitle(
        "BF16 Forward Speedup — Final vs Old c18",
        fontsize=14,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.915,
        "* Gated fallback; panels use independent y-scales",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="0.25",
    )
    fig.text(
        0.5,
        0.018,
        "Median across per-position 10%-trimmed means; 1000 warmups, 100 trials, 3 ABBA rounds",
        ha="center",
        fontsize=8.5,
        color="0.28",
    )
    sns.despine(fig=fig)
    fig.tight_layout(rect=(0, 0.065, 1, 0.80), w_pad=1.5)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
    fig.savefig(output_base.with_suffix(".png"), bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(output_base.with_suffix(".pdf"))
    print(output_base.with_suffix(".png"))


def main() -> None:
    args = parse_args()
    render(args.data, args.output)


if __name__ == "__main__":
    main()
