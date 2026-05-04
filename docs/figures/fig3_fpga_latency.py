"""Figure 3 - KV260 FPGA Ising-sampler measured latency at 64 spins.

Single-bar chart showing the KV260 hardware-measured per-sample latency
from the exp1068 smoke test. The old CPU reference bar is intentionally
absent because it was an order-of-magnitude sweep estimate rather than a
measured same-basis per-sample baseline.

Why a figure: the paper needs a hardware-latency figure, but publication
claims must reduce to measured artifact data. This figure therefore reports
only the measured FPGA datum and keeps the scope limit visible.

Run: python docs/figures/fig3_fpga_latency.py
Outputs:
    docs/figures/fig3_fpga_latency.png
    docs/figures/fig3_fpga_latency.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

OUTDIR = Path(__file__).parent
EXP1068_RESULT_PATH = Path(__file__).resolve().parents[2] / "results" / (
    "experiment_1068_kv260_smoke_test_v9.json"
)
FPGA_LATENCY_FIELD = "hardware_latency_us"


def load_measured_fpga_latency_us(result_path: Path = EXP1068_RESULT_PATH) -> float:
    """Load the measured KV260 latency from the Exp 1068 result artifact."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return float(payload[FPGA_LATENCY_FIELD])


def measured_latency_bars(result_path: Path = EXP1068_RESULT_PATH) -> list[dict]:
    """Return the measured-only bar payload for Figure 3."""
    return [
        {
            "label": "KV260 FPGA Ising\n(exp1068)",
            "latency_us": load_measured_fpga_latency_us(result_path),
            "color": "#7b1c1c",
        }
    ]


def render(outdir: Path = OUTDIR, result_path: Path = EXP1068_RESULT_PATH) -> None:
    bars_data = measured_latency_bars(result_path)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    labels = [item["label"] for item in bars_data]
    values_us = [item["latency_us"] for item in bars_data]
    colors = [item["color"] for item in bars_data]

    bars = ax.bar(labels, values_us, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Measured per-sample latency (us)", fontsize=11)
    ax.set_title(
        "Figure 3 - 64-spin KV260 FPGA Ising-sampler latency\n"
        "Measured hardware datum from exp1068 only",
        fontsize=12,
    )
    ax.grid(True, axis="y", alpha=0.3)

    fpga_us = values_us[0]
    ax.set_ylim(0.0, fpga_us * 1.8)
    ax.text(
        bars[0].get_x() + bars[0].get_width() / 2,
        fpga_us * 1.08,
        f"{fpga_us:.2f} us",
        ha="center",
        fontsize=10,
        color="#7b1c1c",
        fontweight="bold",
    )

    caveat = (
        "Scope: source artifact is results/experiment_1068_kv260_smoke_test_v9.json\n"
        "field hardware_latency_us. No same-basis measured CPU baseline is plotted."
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
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "fig3_fpga_latency.png", dpi=180, bbox_inches="tight")
    fig.savefig(outdir / "fig3_fpga_latency.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    render()
