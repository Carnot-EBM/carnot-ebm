"""Figure 6 - Welch / Rankin Simplex bound: maximum verifier-composition size k*
as a function of the per-verifier valid-signal correlation alpha^2, with the
empirical (alpha^2 = 0.66, r_max = 0.5, k* = 3.125) point marked.

Why a figure: Section 3 of the v3 paper retracts v2's "k = 15
AND-composition" claim and replaces it with the Welch bound k* <=
floor((1 - r_max) / (alpha^2 - r_max)). The bound is qualitative
without a picture: a reader has to plug numbers in to see that the
empirical alpha^2 = 0.66 from exp1093 limits a homogeneous text-probe
ensemble to k* <= 3 verifiers, while a heterogeneous (cross-mechanism)
ensemble at alpha^2 ~ 0.4 admits k_max ~ 7-8. The figure makes the
intuition load-bearing and visual: there is a continuous trade-off
between per-verifier signal strength (alpha^2) and the maximum
ensemble size that can preserve the architectural r_max constraint.

Why this replaces fig5_humaneval_improvement: v2's fig5 communicated
"Carnot beats SOTA at HumanEval" in service of a SOTA-beating
narrative that v3 rejects as not defensible given the .85
recalibrations. v3's fig6 communicates "here is the bound your
ensemble has to live under" in service of the empirical-bounds
narrative.

Run: python docs/figures/fig6_welch_ceiling.py
Outputs:
    docs/figures/fig6_welch_ceiling.png
    docs/figures/fig6_welch_ceiling.pdf
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path(__file__).parent

# Empirical verifier-correlation reading from exp1093 (Phase-1c null-space
# measurement, 2026-05-01). The dominant pairwise correlation among the three
# deployed Tier-0 text probes was 0.656 between SpilledEnergyDetector and
# NUPProbeV4; we project alpha^2 = 0.66 onto the Welch bound throughout.
ALPHA2_EMPIRICAL = 0.66
# Architectural constraint on maximum allowed pairwise verifier overlap; below
# this, residual mechanism vectors can geometrically point negative-inner-
# product away from each other.
R_MAX = 0.5


def welch_k_star(alpha2: float, r_max: float) -> float:
    """Welch / Rankin Simplex bound on the maximum number of unit vectors with
    pairwise residual inner product <= -(alpha^2 - r_max)/(1 - alpha^2).

    Returns the continuous bound (1 - r_max) / (alpha^2 - r_max). The discrete
    architectural ceiling is floor() of this; we plot the continuous form so
    the qualitative shape is visible. NaN below the alpha^2 = r_max regime
    where the architectural constraint is satisfied trivially.
    """
    if alpha2 <= r_max:
        return float("nan")
    return (1.0 - r_max) / (alpha2 - r_max)


def render() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    alpha2_grid = np.linspace(R_MAX + 0.01, 0.95, 400)
    k_curve = np.array([welch_k_star(a, R_MAX) for a in alpha2_grid])
    ax.plot(
        alpha2_grid,
        k_curve,
        color="#1f77b4",
        lw=2.5,
        label=f"Welch bound k* = (1 - r_max) / (alpha^2 - r_max), r_max = {R_MAX}",
    )

    # Empirical point: homogeneous text-probe ensemble (alpha^2 = 0.66, k* = 3.125).
    k_star_emp = welch_k_star(ALPHA2_EMPIRICAL, R_MAX)
    ax.scatter(
        [ALPHA2_EMPIRICAL],
        [k_star_emp],
        color="#d62728",
        s=120,
        zorder=5,
        label=f"Carnot exp1093 (alpha^2 = {ALPHA2_EMPIRICAL}, k* = {k_star_emp:.2f})",
    )
    ax.annotate(
        "Homogeneous text probes:\n"
        f"alpha^2 = 0.66, k* = {k_star_emp:.2f}\n"
        "(D_eff = 1.603 confirms collapse)",
        xy=(ALPHA2_EMPIRICAL, k_star_emp),
        xytext=(0.72, 5.0),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#d62728"),
    )

    # Conjectured cross-mechanism point. Section 3.4 argues that adding
    # symbolic / runtime / format verifiers loosens alpha^2 toward 0.4; the
    # exp1104 follow-up will measure this empirically.
    alpha2_hetero = 0.4
    if alpha2_hetero > R_MAX:
        k_hetero = welch_k_star(alpha2_hetero, R_MAX)
        ax.scatter(
            [alpha2_hetero],
            [k_hetero],
            marker="*",
            color="#2ca02c",
            s=200,
            zorder=5,
            label=(
                f"Cross-mechanism target (alpha^2 = {alpha2_hetero}, k* = {math.floor(k_hetero)})"
            ),
        )

    # Architectural floor: the ceiling has to clear k=2 to be interesting,
    # since any single verifier trivially satisfies the bound.
    ax.axhline(2.0, color="#888888", lw=1.0, ls=":", label="trivial ceiling k = 2")
    ax.axhline(8.0, color="#999900", lw=1.0, ls=":", label="cross-mechanism aspiration k ~ 7-8")

    ax.set_xlabel(
        "Per-verifier valid-signal correlation alpha^2 (homogeneous -> heterogeneous)",
        fontsize=11,
    )
    ax.set_ylabel("Welch ceiling k* (continuous)", fontsize=11)
    ax.set_xlim(0.4, 0.95)
    ax.set_ylim(0.0, 12.0)
    ax.set_title(
        "Figure 6 - Welch / Rankin Simplex ceiling on AND-composition size\n"
        "Empirical alpha^2 = 0.66 limits homogeneous text probes to k* <= 3.125",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    note = (
        "Welch (1974) / Rankin Simplex bound: at most 1 + 1/c\n"
        "unit vectors can have pairwise inner product <= -c.\n"
        "Carnot's verifier ensemble has c = (alpha^2 - r_max)/(1 - alpha^2).\n"
        "Cross-mechanism diversity (symbolic / runtime / format / step-level)\n"
        "is the only known route to alpha^2 ~ 0.4 and k_max ~ 7-8."
    )
    ax.text(
        0.02,
        0.97,
        note,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#f4f4f4", edgecolor="#888888"),
    )

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig6_welch_ceiling.png", dpi=180)
    fig.savefig(OUTDIR / "fig6_welch_ceiling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    render()
