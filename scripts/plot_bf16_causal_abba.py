#!/usr/bin/env python3
"""Render the final BF16 causal-forward ABBA speedup figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "benchmarks" / "results" / "bf16_causal_abba_20260808.csv"
DEFAULT_OUTPUT = ROOT / "assets" / "bf16_causal_abba_speedup"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def render(data_path: Path, output_base: Path) -> None:
    data = pd.read_csv(data_path)
    required = {"dtype", "direction", "mask", "headdim", "q_len", "k_len", "route", "speedup"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if (
        set(data["dtype"]) != {"BF16"}
        or set(data["direction"]) != {"FWD"}
        or set(data["mask"]) != {"causal"}
    ):
        raise ValueError("this figure requires BF16 causal FWD measurements only")

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
    route_palette = {"fast path": colors[0], "legacy fallback": colors[7]}
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharey=False)

    for ax, headdim in zip(axes, (64, 128, 256), strict=True):
        subset = data[data["headdim"] == headdim].copy()
        subset["case"] = subset.apply(
            lambda row: f"Q{int(row['q_len'])}\nK{int(row['k_len'])}", axis=1
        )
        case_palette = {
            row.case: route_palette[row.route]
            for row in subset.itertuples(index=False)
        }
        sns.barplot(
            data=subset,
            x="case",
            y="speedup",
            hue="case",
            palette=case_palette,
            dodge=False,
            legend=False,
            errorbar=None,
            ax=ax,
        )
        ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
        minimum, maximum = subset["speedup"].min(), subset["speedup"].max()
        lower = max(0.94, min(0.985, minimum - 0.035 * max(1.0, maximum - minimum)))
        upper = maximum + max(0.08, 0.10 * maximum)
        ax.set_ylim(lower, upper)
        ax.set_title(f"Head dimension D={headdim}", fontweight="bold", pad=8)
        ax.set_xlabel("Sequence lengths")
        ax.set_ylabel("Speedup (previous / final)" if headdim == 64 else "")
        ax.grid(axis="y", color="0.88", linewidth=0.7)
        for position, row in enumerate(subset.itertuples(index=False)):
            label = f"{row.speedup:.2f}x" if row.speedup >= 2 else f"{row.speedup:.3f}x"
            ax.text(
                position,
                row.speedup + 0.018 * (upper - lower),
                label,
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=0,
            )
    handles = [
        Patch(facecolor=route_palette[route], label=route)
        for route in ("fast path", "legacy fallback")
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=2, frameon=False)
    fig.suptitle(
        "BF16 Causal Forward Speedup — Final vs Previous Release",
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
        "Median of six per-position 10%-trimmed means; 1000 warmups, 100 trials, 3 ABBA rounds",
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
