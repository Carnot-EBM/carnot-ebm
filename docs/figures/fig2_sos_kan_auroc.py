"""Figure 2 - SOS-KAN v3 vs ThinkPRM ROC curves on FoVer corpus.

Plots receiver operating characteristic curves for the two trained
verifiers reported in the position paper, alongside the random-baseline
diagonal so a reader can read off discriminative power at a glance.

Why a figure: AUROC numbers in the abstract (0.9545 SOS-KAN, 0.9885
ThinkPRM) are abstract until visualized. The ROC curves let a reviewer
verify that the operating points are well-separated from the random
baseline across the full false-positive-rate axis, not just at a
single threshold.

Why synthesized curves: the per-pair scores are not in the result JSONs
that ship in the repo (the JSONs record summary AUROC, training stats,
and validation counts). We synthesize a binormal ROC parametrized by
the published AUROC and a moderate variance assumption so the figure
matches the headline number exactly. The curves are illustrative of
the published AUROC; they are NOT a re-evaluation. The figure caption
in the paper makes this explicit.

Run: python docs/figures/fig2_sos_kan_auroc.py
Outputs:
    docs/figures/fig2_sos_kan_auroc.png
    docs/figures/fig2_sos_kan_auroc.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

OUTDIR = Path(__file__).parent


def binormal_roc(auroc: float, n: int = 401) -> tuple[np.ndarray, np.ndarray]:
    """Return (FPR, TPR) for a binormal ROC with given AUROC.

    Binormal model: positives ~ N(mu, 1), negatives ~ N(0, 1).
    AUROC = Phi(mu / sqrt(2)) so mu = sqrt(2) * Phi^{-1}(AUROC).
    """
    mu = np.sqrt(2.0) * norm.ppf(auroc)
    z = np.linspace(-6.0, 6.0 + mu, n)
    fpr = 1.0 - norm.cdf(z)
    tpr = 1.0 - norm.cdf(z - mu)
    order = np.argsort(fpr)
    return fpr[order], tpr[order]


def render() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    fpr_a, tpr_a = binormal_roc(0.9885)
    fpr_b, tpr_b = binormal_roc(0.9545)

    ax.plot(fpr_a, tpr_a, color="#1f77b4", lw=2.4, label="ThinkPRM (AUROC = 0.9885)")
    ax.plot(fpr_b, tpr_b, color="#d95f02", lw=2.4, label="SOS-KAN v3 (AUROC = 0.9545)")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.0, ls="--", label="Random (AUROC = 0.500)")

    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title(
        "Figure 2 - ROC curves on FoVer corpus (6,548 pairs)\n"
        "Curves illustrate published AUROC; binormal-fit, not a re-evaluation",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_sos_kan_auroc.png", dpi=180)
    fig.savefig(OUTDIR / "fig2_sos_kan_auroc.pdf")
    plt.close(fig)


if __name__ == "__main__":
    render()
