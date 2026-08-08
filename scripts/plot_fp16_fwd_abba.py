#!/usr/bin/env python3
"""Render the final-vs-old-c18 FP16 forward ABBA speedup figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "benchmarks" / "results" / "fp16_fwd_abba_20260808.csv"
DEFAULT_OUTPUT = ROOT / "assets" / "fp16_fwd_abba_speedup"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def render(data_path: Path, output_base: Path) -> None:
    data = pd.read_csv(data_path)
    required = {"dtype", "direction", "mask", "headdim", "seqlen", "speedup"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if set(data["dtype"]) != {"FP16"} or set(data["direction"]) != {"FWD"}:
        raise ValueError("this figure requires FP16 FWD measurements only")
    if set(data["mask"]) != {"non-causal", "causal"}:
        raise ValueError("this figure requires both non-causal and causal measurements")

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
    mask_palette = {"non-causal": colors[0], "causal": colors[1]}
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
            hue_order=["Non-causal", "Causal"],
            palette={
                "Non-causal": mask_palette["non-causal"],
                "Causal": mask_palette["causal"],
            },
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
        for container in ax.containers:
            labels = [f"{value:.3f}x" for value in container.datavalues]
            ax.bar_label(container, labels=labels, padding=3, fontsize=7.2)
        if ax.legend_ is not None:
            ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles = [container for container in axes[0].containers[:2]]
        labels = ["Non-causal", "Causal"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=2, frameon=False)
    fig.suptitle(
        "FP16 Forward Speedup — Final vs Old c18",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.94,
        "Panels use independent y-scales — compare numeric labels, not bar heights",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="0.25",
    )
    fig.text(
        0.5,
        0.01,
        "Median across per-position 10%-trimmed means; 1000 warmups, 100 trials, 3 ABBA rounds "
        "(NC D128/S4096: 5-round recheck)",
        ha="center",
        fontsize=8.5,
        color="0.28",
    )
    sns.despine(fig=fig)
    fig.tight_layout(rect=(0, 0.045, 1, 0.84), w_pad=1.5)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(output_base.with_suffix(".pdf"))
    print(output_base.with_suffix(".png"))


def main() -> None:
    args = parse_args()
    render(args.data, args.output)


if __name__ == "__main__":
    main()
