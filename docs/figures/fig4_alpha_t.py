"""Figure 4 - alpha_t self-distillation grounding signal: small vs SOTA model.

Two-bar comparison of Carnot's measured alpha_t grounding signal on
Qwen3.5-0.8B (exp1074) and Qwen3.6-35B-A3B (exp1077). Annotates the
Zenil convergence threshold (alpha_t > 0) and a verbose explanation
of why the SOTA-model number is lower than the small-model number.

Why a figure: alpha_t is the load-bearing quantity in the Phase-3
self-distillation argument. Round-12 of the Zenil derivation
established that any inf_t alpha_t > 0 maintains convergence to the
truth distribution; the size of alpha_t controls the convergence rate.
A figure that makes the inequality visible (both bars sit above 0)
communicates the headline result without requiring the reader to
parse the recursive-self-training equation.

Why annotated: the SOTA result (alpha_t = 0.38) is lower than the
small-model result (alpha_t = 0.78). Without context a reviewer
could read this as a regression. The annotation explains it: a
larger model is closer to the truth distribution to begin with, so
the per-step grounding contribution from the Carnot verifier is
smaller in magnitude even though it is still positive and therefore
load-bearing for convergence.

Run: python docs/figures/fig4_alpha_t.py
Outputs:
    docs/figures/fig4_alpha_t.png
    docs/figures/fig4_alpha_t.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUTDIR = Path(__file__).parent

# alpha_t v1 reference: Qwen3.5-0.8B small-model baseline, exp1074
# (alpha_t_v1_comparison field of exp1077 result JSON).
ALPHA_SMALL = 0.78
# alpha_t v4 SOTA result: Qwen3.6-35B-A3B GGUF, exp1077.
# Source: results/experiment_1077_fr11_alpha_t_sota_v4.json field "alpha_t".
ALPHA_SOTA = 0.38


def render() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    labels = [
        "Qwen3.5-0.8B\n(small, exp1074)",
        "Qwen3.6-35B-A3B\n(SOTA MoE, exp1077)",
    ]
    values = [ALPHA_SMALL, ALPHA_SOTA]
    colors = ["#1f77b4", "#d95f02"]

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.025,
            f"alpha_t = {val:.2f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.axhline(0.0, color="red", lw=1.5, ls="--", label="Zenil bound: alpha_t > 0")
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("Carnot grounding signal alpha_t", fontsize=11)
    ax.set_title(
        "Figure 4 - alpha_t verifier-grounding signal\n"
        "Lower alpha_t on SOTA model is expected: harder to distinguish from temperature",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    note = (
        "Round-12 (Zenil derivation): any inf_t alpha_t > 0 preserves\n"
        "self-distillation convergence to the truth distribution mu_P.\n"
        "Both bars satisfy this; SOTA size of alpha_t reflects the\n"
        "smaller residual gap between Q_t and mu_P at large model scale."
    )
    ax.text(
        0.02,
        0.98,
        note,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#f4f4f4", edgecolor="#888888"),
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig4_alpha_t.png", dpi=180)
    fig.savefig(OUTDIR / "fig4_alpha_t.pdf")
    plt.close(fig)


if __name__ == "__main__":
    render()
