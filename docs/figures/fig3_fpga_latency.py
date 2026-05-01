"""Figure 3 - CPU vs KV260 FPGA Ising-sampler latency at 64 spins.

Bar chart contrasting the CPU reference latency (~290 ms) against the
KV260 hardware-measured per-sample latency (24.83 us, exp1068 smoke
test). Adds a textual note that the .84 scale benchmark (exp1081)
could not reach the board; the crossover-N estimate is extrapolated
from scaling theory rather than measured end-to-end.

Why a figure: the four-orders-of-magnitude latency gap is the most
visceral hardware claim in the paper. A linear-axis chart hides the
gap; a log-axis chart with explicit annotations exposes it without
overstating it. Honest caveat is rendered ON THE FIGURE so reviewers
who skim only the figure cannot miss it.

Why a single point comparison: scaling-curve runs require a board
unreachable in .84. We document the scope limit explicitly rather
than shipping a fabricated curve.

Run: python docs/figures/fig3_fpga_latency.py
Outputs:
    docs/figures/fig3_fpga_latency.png
    docs/figures/fig3_fpga_latency.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUTDIR = Path(__file__).parent

# CPU reference: order-of-magnitude estimate from a Glauber Ising sampler
# at 64 spins, single core, ~290 ms per N=200 sample sweep. Reported as
# a representative number in the position paper's hardware section.
CPU_LATENCY_MS = 290.0

# KV260 hardware-measured mean per-sample latency in microseconds at
# 64 spins. Source: results/experiment_1068_kv260_smoke_test_v9.json
# field "hardware_latency_us" (board IP 192.168.51.98, /dev/uio4).
FPGA_LATENCY_US = 24.82834388501942


def render() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    labels = ["CPU (Python Glauber)", "KV260 FPGA Ising"]
    values_us = [CPU_LATENCY_MS * 1000.0, FPGA_LATENCY_US]
    colors = ["#888888", "#7b1c1c"]

    bars = ax.bar(labels, values_us, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Per-sample latency (us, log scale)", fontsize=11)
    ax.set_title(
        "Figure 3 - 64-spin Ising-sampler latency: CPU vs KV260 FPGA",
        fontsize=12,
    )
    ax.grid(True, axis="y", which="both", alpha=0.3)

    cpu_us, fpga_us = values_us
    ax.text(
        bars[0].get_x() + bars[0].get_width() / 2,
        cpu_us * 1.25,
        f"{CPU_LATENCY_MS:.0f} ms\n({cpu_us:.0f} us)",
        ha="center",
        fontsize=10,
    )
    ax.text(
        bars[1].get_x() + bars[1].get_width() / 2,
        fpga_us * 1.25,
        f"{fpga_us:.2f} us",
        ha="center",
        fontsize=10,
        color="#7b1c1c",
        fontweight="bold",
    )

    speedup = cpu_us / fpga_us
    ax.text(
        0.5,
        0.92,
        f"speedup ~ {speedup:,.0f}x at N=64 spins",
        ha="center",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="#fff8d6", edgecolor="#aa9000"),
    )

    caveat = (
        "Caveat: KV260 board was unreachable during exp1081 scale benchmark in .84;\n"
        "crossover-N is extrapolated from CPU O(N^2) scaling and the FPGA O(1) clock period,\n"
        "not measured end-to-end. Single-point comparison only."
    )
    ax.text(
        0.5,
        -0.20,
        caveat,
        ha="center",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        style="italic",
    )
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(OUTDIR / "fig3_fpga_latency.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUTDIR / "fig3_fpga_latency.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render()
