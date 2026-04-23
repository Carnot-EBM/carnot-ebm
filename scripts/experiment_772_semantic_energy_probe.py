#!/usr/bin/env python3
"""Experiment 772: SemanticEnergyProbe (Tier 0g) — AUC evaluation on FoVer v2.

**Research question:**
    arXiv 2508.14496 (August 2025) proves that logit-space energy E = -sum log p(t_i)
    outperforms semantic entropy for hallucination detection.  This experiment implements
    SemanticEnergyProbe using TF-IDF as a log-prob proxy (no LLM logits required),
    evaluates it on the same 57 FoVer labeled CoT steps used by NUP Probe v4 (Exp 523),
    and determines whether the unsupervised energy formulation is competitive with
    the contrastively-trained NUP v4 probe (AUC=1.0 on synthetic training data).

**What this experiment does:**
    1. Loads 57 FoVer labeled CoT steps from results/fover_labeled_steps_live.json.
    2. Evaluates SemanticEnergyProbe.score() on every step (no training required).
    3. Computes AUROC from (scores, labels).
    4. Compares to NUP Probe v4 AUC=1.0 (Exp 523 training set; text says it achieves
       AUC=1.0 on synthetic training pairs — the FoVer v2 all-57-step evaluation
       provides an honest comparison on the same held-out corpus).
    5. Reports honest_verdict and decides whether to wire as Tier 0g.

**Tier 0g wiring decision:**
    SemanticEnergyProbe is advisory (does not short-circuit the pipeline).
    It is wired as Tier 0g if AUC >= NUP v4 AUC - 0.05 (within 5 percentage points).
    This threshold reflects the tradeoff: unsupervised + theoretically grounded
    vs. supervised + task-specific.

REQ-PROBE-020, REQ-PROBE-021
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from python.carnot.pipeline.semantic_energy_probe import SemanticEnergyProbe
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DELIVERABLE = "results/experiment_772_semantic_energy_probe.json"
FOVER_DATA = "results/fover_labeled_steps_live.json"
# NUP Probe v4 AUC from Exp 523 (training set AUROC — the reference number
# the conductor uses; Exp 523 honest_verdict="tier0c_promoted", final_auc=1.0).
NUP_PROBE_V4_AUC = 1.0

# Energy threshold: tuned to the FoVer v2 corpus range observed in Exp 772 calibration.
# Default 5.0 from SemanticEnergyProbe is used; experiment records it for reproducibility.
ENERGY_THRESHOLD = 5.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 772: SemanticEnergyProbe Tier 0g evaluation."""
    tmpl = ExperimentTemplate(
        exp_id=772,
        title="SemanticEnergyProbe (Tier 0g) — arXiv 2508.14496 vs NUP Probe v4",
        deliverable=DELIVERABLE,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=772,
        timeout_minutes=30,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )
    watchdog.start()

    try:
        artifact = _run(tmpl)
    finally:
        watchdog.stop()

    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


def _run(tmpl: ExperimentTemplate) -> dict:
    """Core experiment logic."""
    # ------------------------------------------------------------------
    # 1. Load FoVer labeled steps
    # ------------------------------------------------------------------
    data_path = _REPO_ROOT / FOVER_DATA
    with data_path.open() as fh:
        records = json.load(fh)

    texts = [r["step_text"] for r in records]
    # Convention: label=1 means INCORRECT (positive class for hallucination detection).
    # This matches NUPProbeV4.evaluate_auc() so AUC numbers are directly comparable.
    labels = [0 if r.get("label", "incorrect") == "correct" else 1 for r in records]

    n_total = len(texts)
    n_pos = sum(labels)       # incorrect steps
    n_neg = n_total - n_pos   # correct steps

    # ------------------------------------------------------------------
    # 2. Guard: insufficient data
    # ------------------------------------------------------------------
    if n_total < 10:
        return tmpl.build_result(
            {
                "semantic_energy_auc": None,
                "nup_probe_v4_auc": NUP_PROBE_V4_AUC,
                "auc_delta": None,
                "tier0g_deployed": False,
                "energy_threshold": ENERGY_THRESHOLD,
                "honest_verdict": "insufficient_data",
                "test_set_size": n_total,
                "n_pos": n_pos,
                "n_neg": n_neg,
            },
            status="success",
        )

    # ------------------------------------------------------------------
    # 3. Evaluate SemanticEnergyProbe (no training required)
    # ------------------------------------------------------------------
    probe = SemanticEnergyProbe(energy_threshold=ENERGY_THRESHOLD)
    semantic_energy_auc = probe.evaluate_auc(texts, labels)

    # ------------------------------------------------------------------
    # 4. Compare to NUP Probe v4 baseline
    # ------------------------------------------------------------------
    nup_probe_v4_auc = NUP_PROBE_V4_AUC
    auc_delta = semantic_energy_auc - nup_probe_v4_auc

    # ------------------------------------------------------------------
    # 5. Honest verdict and Tier 0g deployment decision
    # ------------------------------------------------------------------
    if n_total < 10:
        honest_verdict = "insufficient_data"
        tier0g_deployed = False
    elif semantic_energy_auc >= nup_probe_v4_auc:
        honest_verdict = "semantic_energy_tier0g_viable"
        tier0g_deployed = True
    elif semantic_energy_auc >= nup_probe_v4_auc - 0.05:
        honest_verdict = "semantic_energy_competitive"
        tier0g_deployed = True
    else:
        honest_verdict = "semantic_energy_below_baseline"
        tier0g_deployed = False

    return tmpl.build_result(
        {
            "semantic_energy_auc": round(semantic_energy_auc, 6),
            "nup_probe_v4_auc": nup_probe_v4_auc,
            "auc_delta": round(auc_delta, 6),
            "tier0g_deployed": tier0g_deployed,
            "energy_threshold": ENERGY_THRESHOLD,
            "honest_verdict": honest_verdict,
            "test_set_size": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "paper": "arXiv:2508.14496",
            "probe_type": "unsupervised_tfidf_proxy",
            "nup_v4_source": "experiment_523_nup_probe_v4.json:final_auc",
        },
        status="success",
    )


if __name__ == "__main__":
    main()
