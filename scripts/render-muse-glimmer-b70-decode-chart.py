#!/usr/bin/env python3
"""Render the hero chart for docs/muse-glimmer-vllm-xpu-dflash.md."""
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images/muse-glimmer-b70-decode-vllm.png"

ROWS = (
    ("Cookbook llama.cpp  ·  DFlash n=2", 26.8, "#64748b", False),
    ("OpenVINO official IR", 31.7, "#64748b", False),
    ("vLLM + DFlash  ·  writing", 42.6, "#60a5fa", True),
    ("vLLM + DFlash  ·  GSM8K", 89.1, "#34d399", True),
    ("vLLM + DFlash  ·  HumanEval", 101.1, "#34d399", True),
)


def main() -> None:
    labels = [r[0] for r in ROWS]
    values = [r[1] for r in ROWS]
    colors = [r[2] for r in ROWS]
    ours = [r[3] for r in ROWS]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(13.5, 7.6), dpi=160, facecolor="#0b1220")
    ax = fig.add_axes((0.28, 0.14, 0.66, 0.68))
    ax.set_facecolor("#0b1220")

    y = list(range(len(ROWS) - 1, -1, -1))
    bars = ax.barh(y, values, color=colors, height=0.58, zorder=3)
    for bar, val, is_ours in zip(bars, values, ours):
        ax.text(
            val + 1.6,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center",
            ha="left",
            fontsize=17,
            fontweight="bold" if is_ours else "regular",
            color="#f8fafc" if is_ours else "#cbd5e1",
            path_effects=[pe.withStroke(linewidth=3, foreground="#0b1220")],
        )

    ax.axvline(26.8, color="#94a3b8", lw=1.1, ls=(0, (3, 4)), alpha=0.7, zorder=2)
    ax.annotate(
        "published B70 floor",
        xy=(26.8, 4.42),
        xytext=(30.5, 4.42),
        color="#94a3b8",
        fontsize=10,
        va="center",
        arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.8},
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13, color="#e2e8f0")
    ax.set_xlim(0, 118)
    ax.set_ylim(-0.55, 4.7)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel(
        "decode tok/s   ·   greedy   ·   thinking + answer",
        fontsize=12,
        color="#94a3b8",
        labelpad=12,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=11, length=0)
    ax.tick_params(axis="y", colors="#e2e8f0", length=0)
    ax.grid(axis="x", color="#1e293b", lw=0.9, zorder=0)
    ax.set_axisbelow(True)

    fig.text(
        0.06, 0.93,
        "Muse Glimmer 30B on one Arc Pro B70",
        fontsize=26, fontweight="bold", color="#f8fafc", ha="left", va="top",
    )
    fig.text(
        0.06, 0.875,
        "Same card. Public recipes vs vLLM-XPU + DFlash.",
        fontsize=13, color="#94a3b8", ha="left", va="top",
    )
    fig.text(
        0.06, 0.045,
        "One stream. Until stop. GSM8K $18  ·  HumanEval doctest pass.  "
        "Not a 5090, not 8 concurrent slots.",
        fontsize=11, color="#64748b", ha="left", va="center",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=fig.get_facecolor())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
