"""Figure 1 - Carnot 4-tier verification cascade architecture.

Renders a block diagram of the verifier cascade with skip-tier rates
measured in the exp1073 triple-integration end-to-end run (50 questions).

Why a figure: the cascade is the load-bearing architectural claim of
the position paper. A reader unfamiliar with verifier-filtered self-
distillation needs a single diagram showing (a) which tier evaluates
first, (b) the bypass pattern, and (c) where FPGA acceleration sits.

Why these numbers: the skip-tier rates come straight from
results/experiment_1073_triple_integration_e2e_v9.json. They are
empirical, not aspirational, and we keep them visible on the figure so
reviewers can audit them.

Run: python docs/figures/fig1_cascade_architecture.py
Outputs:
    docs/figures/fig1_cascade_architecture.png
    docs/figures/fig1_cascade_architecture.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Empirical skip-tier counts from results/experiment_1073_triple_integration_e2e_v9.json
# (50 questions across all 4 tiers, all_tier_skip_rates_nonzero=True).
TIER_DATA = [
    ("Tier 0a", "ThinkPRM", 4, "Step-level PRM probe"),
    ("Tier 0b", "SpilledEnergy", 25, "Logit-energy gap"),
    ("Tier 2", "SOS-KAN", 13, "AUROC=0.9545 SOS-certified KAN"),
    ("Tier 3", "Ising", 8, "FPGA-accelerated MCMC"),
]
N_QUESTIONS = 50
OUTDIR = Path(__file__).parent


def render() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_width = 2.2
    box_height = 1.6
    y0 = 2.5
    spacing = 2.4

    for idx, (tier, name, skips, descr) in enumerate(TIER_DATA):
        x0 = 0.4 + idx * spacing
        rate_pct = 100.0 * skips / N_QUESTIONS
        face = {
            "Tier 0a": "#cfe8ff",
            "Tier 0b": "#bbe7c4",
            "Tier 2": "#ffe2a8",
            "Tier 3": "#ffc6c6",
        }[tier]
        rect = mpatches.FancyBboxPatch(
            (x0, y0),
            box_width,
            box_height,
            boxstyle="round,pad=0.04",
            linewidth=1.5,
            edgecolor="black",
            facecolor=face,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + box_width / 2,
            y0 + box_height - 0.32,
            f"{tier}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            x0 + box_width / 2,
            y0 + box_height - 0.7,
            name,
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.text(
            x0 + box_width / 2,
            y0 + 0.55,
            descr,
            ha="center",
            va="center",
            fontsize=8.5,
            style="italic",
        )
        ax.text(
            x0 + box_width / 2,
            y0 + 0.18,
            f"skip {skips}/{N_QUESTIONS} = {rate_pct:.0f}%",
            ha="center",
            va="center",
            fontsize=9,
        )
        if idx < len(TIER_DATA) - 1:
            ax.annotate(
                "",
                xy=(x0 + box_width + 0.18, y0 + box_height / 2),
                xytext=(x0 + box_width + 0.02, y0 + box_height / 2),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4),
            )

    # FPGA acceleration callout under Tier 3.
    tier3_x = 0.4 + 3 * spacing
    fpga_box = mpatches.FancyBboxPatch(
        (tier3_x, 0.35),
        box_width,
        1.4,
        boxstyle="round,pad=0.04",
        linewidth=1.2,
        edgecolor="#7b1c1c",
        facecolor="#ffe9e9",
        linestyle="--",
    )
    ax.add_patch(fpga_box)
    ax.text(
        tier3_x + box_width / 2,
        1.55,
        "KV260 FPGA",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#7b1c1c",
    )
    ax.text(
        tier3_x + box_width / 2,
        1.2,
        "24.83 us / 64 spins",
        ha="center",
        va="center",
        fontsize=10,
        color="#7b1c1c",
    )
    ax.text(
        tier3_x + box_width / 2,
        0.85,
        "(exp1068 smoke)",
        ha="center",
        va="center",
        fontsize=8,
        color="#7b1c1c",
        style="italic",
    )
    ax.annotate(
        "",
        xy=(tier3_x + box_width / 2, 1.78),
        xytext=(tier3_x + box_width / 2, 2.45),
        arrowprops=dict(arrowstyle="-|>", color="#7b1c1c", lw=1.2, linestyle="--"),
    )

    # Input arrow on the left.
    ax.annotate(
        "LLM output",
        xy=(0.42, y0 + box_height / 2),
        xytext=(0.0, y0 + box_height / 2 + 0.55),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3),
        fontsize=10,
    )
    # "verified or repaired" arrow on the right.
    last_x = 0.4 + (len(TIER_DATA) - 1) * spacing + box_width
    ax.annotate(
        "verified or repaired",
        xy=(last_x + 1.05, y0 + box_height / 2 + 0.55),
        xytext=(last_x + 0.04, y0 + box_height / 2),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3),
        fontsize=10,
    )

    ax.set_title(
        "Figure 1 - Carnot 4-tier verification cascade (skip rates from exp1073, n=50)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_cascade_architecture.png", dpi=180)
    fig.savefig(OUTDIR / "fig1_cascade_architecture.pdf")
    plt.close(fig)


if __name__ == "__main__":
    render()
