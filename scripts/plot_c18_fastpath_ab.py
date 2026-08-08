#!/usr/bin/env python3
"""Render the frozen c18 FP16/D128 forward A/B comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "benchmarks" / "results" / "c18_fastpath_ab_20260808.csv"
DEFAULT_OUTPUT = ROOT / "assets" / "c18_fastpath_ab_performance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def render(data_path: Path, output_base: Path) -> None:
    data = pd.read_csv(data_path)
    required = {
        "protocol",
        "mask",
        "q_len",
        "k_len",
        "route",
        "baseline_ms",
        "optimized_ms",
        "speedup",
        "speedup_min",
        "speedup_max",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

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
    fig, (ax_latency, ax_boundary) = plt.subplots(
        1,
        2,
        figsize=(12.6, 4.6),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    full = data[data["protocol"] == "full-package"].copy()
    full["case"] = full.apply(
        lambda row: f"{'NC' if row['mask'] == 'non-causal' else 'C'}\nK{int(row['k_len'])}",
        axis=1,
    )
    latency = full.melt(
        id_vars=["case"],
        value_vars=["baseline_ms", "optimized_ms"],
        var_name="package",
        value_name="latency_ms",
    )
    latency["package"] = latency["package"].map(
        {"baseline_ms": "A: c18 baseline", "optimized_ms": "B: optimized"}
    )
    sns.barplot(
        data=latency,
        x="case",
        y="latency_ms",
        hue="package",
        hue_order=["A: c18 baseline", "B: optimized"],
        palette=[colors[0], colors[1]],
        errorbar=None,
        ax=ax_latency,
    )
    ax_latency.set_yscale("log")
    ax_latency.set_ylim(0.04, 1.05)
    ax_latency.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1.0]))
    ax_latency.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax_latency.set_xlabel("Attention case (mask, key length)")
    ax_latency.set_ylabel("FWD latency (ms, log scale)")
    ax_latency.legend(title=None, frameon=False, loc="upper left")
    ax_latency.grid(axis="y", which="major", color="0.88", linewidth=0.7)
    for position, row in enumerate(full.itertuples(index=False)):
        top = max(row.baseline_ms, row.optimized_ms)
        speedup_label = (
            f"{row.speedup:.4f}x" if row.route == "legacy fallback" else f"{row.speedup:.3f}x"
        )
        ax_latency.text(
            position,
            top * 1.11,
            speedup_label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    boundary = data[data["protocol"] == "k-boundary"].sort_values("k_len")
    sns.lineplot(
        data=boundary,
        x="k_len",
        y="speedup",
        color="0.35",
        linewidth=1.5,
        marker=None,
        ax=ax_boundary,
        legend=False,
    )
    sns.scatterplot(
        data=boundary,
        x="k_len",
        y="speedup",
        hue="route",
        hue_order=["legacy fallback", "fast path"],
        palette={"legacy fallback": colors[7], "fast path": colors[2]},
        s=72,
        edgecolor="white",
        linewidth=0.7,
        ax=ax_boundary,
    )
    lower = boundary["speedup"] - boundary["speedup_min"]
    upper = boundary["speedup_max"] - boundary["speedup"]
    ax_boundary.errorbar(
        boundary["k_len"],
        boundary["speedup"],
        yerr=[lower, upper],
        fmt="none",
        ecolor="0.45",
        elinewidth=0.9,
        capsize=2.5,
        zorder=1,
    )
    ax_boundary.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
    ax_boundary.axvspan(768, 1060, color=colors[2], alpha=0.08, linewidth=0)
    ax_boundary.axvline(768, color=colors[2], linestyle=":", linewidth=1.1)
    ax_boundary.text(
        758,
        0.985,
        "K=768 production gate",
        color=colors[2],
        fontsize=8,
        rotation=90,
        ha="right",
        va="bottom",
    )
    for row in boundary.itertuples(index=False):
        ax_boundary.text(
            row.k_len,
            row.speedup_max + 0.008,
            f"{row.speedup:.3f}x",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax_boundary.set_xlim(220, 1060)
    ax_boundary.set_ylim(0.975, 1.23)
    ax_boundary.set_xlabel("Key length K (causal, Q=2048)")
    ax_boundary.set_ylabel("Speedup (A latency / B latency)")
    ax_boundary.legend(title="Runtime route", frameon=False, loc="upper left")
    ax_boundary.grid(axis="y", color="0.88", linewidth=0.7)

    ax_latency.text(-0.08, 1.02, "(a)", transform=ax_latency.transAxes, fontweight="bold")
    ax_boundary.text(-0.12, 1.02, "(b)", transform=ax_boundary.transAxes, fontweight="bold")
    sns.despine(fig=fig)
    fig.tight_layout(w_pad=2.0)

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
