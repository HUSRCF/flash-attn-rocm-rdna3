#!/usr/bin/env python3
"""Render the BF16 non-causal/causal forward ABBA speedup figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "benchmarks" / "results" / "bf16_fwd_abba_20260808.csv"
DEFAULT_OUTPUT = ROOT / "assets" / "bf16_fwd_abba_speedup"
MASK_ORDER = ["Non-causal", "Causal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def value_label(speedup: float, fallback: bool) -> str:
    number = f"{speedup:.2f}x" if speedup >= 2 else f"{speedup:.3f}x"
    return f"{number}*" if fallback else number


def render(data_path: Path, output_base: Path) -> None:
    data = pd.read_csv(data_path)
    required = {"dtype", "direction", "mask", "headdim", "q_len", "k_len", "route", "speedup"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if set(data["dtype"]) != {"BF16"} or set(data["direction"]) != {"FWD"}:
        raise ValueError("this figure requires BF16 FWD measurements only")
    if set(data["mask"]) != {"non-causal", "causal"}:
        raise ValueError("this figure requires both non-causal and causal measurements")
    if set(data["route"]) - {"fast path", "gated fallback"}:
        raise ValueError("unknown route label")

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
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.0), sharey=False)

    for ax, headdim in zip(axes, (64, 128, 256), strict=True):
        subset = data[data["headdim"] == headdim].copy()
        subset["case"] = subset.apply(
            lambda row: f"Q{int(row['q_len'])}\nK{int(row['k_len'])}", axis=1
        )
        subset["mask_label"] = subset["mask"].map(mask_labels)
        case_order = list(dict.fromkeys(subset["case"]))
        expected = len(case_order) * len(MASK_ORDER)
        if len(subset) != expected:
            raise ValueError(f"D={headdim} is not a complete mask-paired matrix")

        sns.barplot(
            data=subset,
            x="case",
            y="speedup",
            hue="mask_label",
            order=case_order,
            hue_order=MASK_ORDER,
            palette=palette,
            errorbar=None,
            ax=ax,
        )
        ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
        minimum, maximum = subset["speedup"].min(), subset["speedup"].max()
        lower = max(0.90, min(0.975, minimum - 0.03))
        upper = maximum + max(0.10, 0.11 * maximum)
        ax.set_ylim(lower, upper)
        ax.set_title(f"Head dimension D={headdim}", fontweight="bold", pad=8)
        ax.set_xlabel("Query / key sequence lengths")
        ax.set_ylabel("Speedup (pre-path / final)" if headdim == 64 else "")
        ax.grid(axis="y", color="0.88", linewidth=0.7)
        ax.tick_params(axis="x", labelsize=7.4)

        lookup = {
            (row.case, row.mask_label): row
            for row in subset.itertuples(index=False)
        }
        for container, mask_label in zip(ax.containers[:2], MASK_ORDER, strict=True):
            labels = [
                value_label(
                    float(lookup[case, mask_label].speedup),
                    lookup[case, mask_label].route == "gated fallback",
                )
                for case in case_order
            ]
            rotate = 90 if headdim == 256 and mask_label == "Non-causal" else 0
            padding = 3 if mask_label == "Non-causal" else 9
            ax.bar_label(
                container,
                labels=labels,
                padding=padding,
                fontsize=6.9,
                rotation=rotate,
            )
        if ax.legend_ is not None:
            ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False)
    fig.suptitle(
        "BF16 Forward Speedup — Final vs Matching Pre-Path Baselines",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.955,
        "* Gated fallback (unchanged route); panels use independent y-scales",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="0.25",
    )
    fig.text(
        0.5,
        0.01,
        "Median paired ABBA-round speedup; per-position 10%-trimmed means; 1000 warmups, 100 trials; "
        "3 ABBA rounds (D128/D256 NC: 5-round rechecks)",
        ha="center",
        fontsize=8.5,
        color="0.28",
    )
    sns.despine(fig=fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.85), w_pad=1.5)

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
