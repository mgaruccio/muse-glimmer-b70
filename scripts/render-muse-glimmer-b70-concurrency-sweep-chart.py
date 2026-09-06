#!/usr/bin/env python3
"""Render the experimental C8–C128 concurrency-sweep chart."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images/muse-glimmer-b70-concurrency-sweep.png"

# Medians of five short-burst waves from b70-inference@faf4ba9.
ROWS = (
    (8, 345.346, 0.429, 5.644),
    (12, 449.944, 0.621, 6.710),
    (16, 514.712, 0.790, 7.712),
    (24, 482.247, 1.167, 12.438),
    (32, 583.812, 1.452, 13.597),
    (48, 794.194, 2.169, 14.938),
    (64, 823.297, 2.866, 19.213),
    (96, 840.810, 4.308, 28.346),
    (128, 827.754, 5.129, 38.516),
)


def style(ax):
    ax.set_facecolor("#0b1220")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#94a3b8", labelsize=10, length=0)
    ax.grid(color="#1e293b", lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    concurrency = [row[0] for row in ROWS]
    throughput = [row[1] for row in ROWS]
    ttft = [row[2] for row in ROWS]
    request = [row[3] for row in ROWS]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, (rate_ax, latency_ax) = plt.subplots(
        1, 2, figsize=(14.5, 7.8), dpi=160, facecolor="#0b1220"
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.20, top=0.73, wspace=0.18)
    for ax in (rate_ax, latency_ax):
        style(ax)
        ax.set_xlim(4, 132)
        ax.xaxis.set_major_locator(MultipleLocator(16))
        ax.set_xlabel("active requests / client burst concurrency", color="#94a3b8", labelpad=10)

    rate_ax.plot(concurrency, throughput, color="#34d399", lw=2.8, marker="o", ms=6, zorder=3)
    rate_ax.scatter([48], [794.194], color="#fbbf24", s=72, zorder=4)
    rate_ax.scatter([96], [840.810], color="#f472b6", s=72, zorder=4)
    rate_ax.set_ylim(280, 900)
    rate_ax.yaxis.set_major_locator(MultipleLocator(100))
    rate_ax.set_ylabel("aggregate completion tok/s", color="#94a3b8", labelpad=10)
    rate_ax.annotate(
        "C48 knee\n794.2 tok/s",
        xy=(48, 794.194), xytext=(24, 870), color="#fbbf24", fontsize=10,
        arrowprops={"arrowstyle": "-", "color": "#fbbf24", "lw": 1.1},
    )
    rate_ax.annotate(
        "highest observed\nC96: 840.8",
        xy=(96, 840.810), xytext=(76, 705), color="#f472b6", fontsize=10,
        arrowprops={"arrowstyle": "-", "color": "#f472b6", "lw": 1.1},
    )
    rate_ax.set_title("Aggregate throughput", color="#f8fafc", fontsize=15, fontweight="bold", loc="left")

    latency_ax.plot(concurrency, request, color="#60a5fa", lw=2.8, marker="o", ms=6, label="request wall time")
    latency_ax.plot(concurrency, ttft, color="#c084fc", lw=2.1, marker="o", ms=5, label="TTFT")
    latency_ax.set_ylim(0, 42)
    latency_ax.yaxis.set_major_locator(MultipleLocator(5))
    latency_ax.set_ylabel("seconds", color="#94a3b8", labelpad=10)
    latency_ax.set_title("Latency tradeoff", color="#f8fafc", fontsize=15, fontweight="bold", loc="left")
    legend = latency_ax.legend(frameon=False, labelcolor="#cbd5e1", fontsize=10, loc="upper left")
    for handle in legend.legend_handles:
        handle.set_alpha(1)

    fig.text(0.08, 0.94, "Muse Glimmer 30B on one Arc Pro B70", fontsize=25, fontweight="bold", color="#f8fafc", ha="left", va="top")
    fig.text(0.08, 0.885, "Experimental DFlash K3 + frozen 32k shortlist  ·  medians of five waves", fontsize=13, color="#94a3b8", ha="left", va="top")
    fig.text(0.08, 0.085, "Repeated 83-token prompt  ·  256-token capped reasoning-only outputs  ·  prefix cache off  ·  zero measured-wave preemptions", fontsize=10.5, color="#94a3b8", ha="left", va="center")
    fig.text(0.08, 0.05, "Thread-pool bursts, not an exact start barrier or continuous traffic. Not long-context or production-capacity throughput.", fontsize=10.5, color="#64748b", ha="left", va="center")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=fig.get_facecolor())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
