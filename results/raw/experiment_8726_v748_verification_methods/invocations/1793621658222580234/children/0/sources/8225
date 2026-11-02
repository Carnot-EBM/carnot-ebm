#!/usr/bin/env python3
"""Experiment 771: EBRM baseline comparison against EORM on FoVer labeled steps.

**Research question:**
    arXiv 2504.13134 (EBRM, April 2025) introduces an energy-based reward model that
    operates at the full-response level.  Carnot's EORM operates at the step level.
    This experiment trains an EBRM baseline on the same FoVer labeled steps used for
    EORM (Exp 732) and compares AUROC.

    If EORM > EBRM on step-level tasks, this is publishable evidence that Carnot's
    step-level granularity is architecturally correct.

**What this experiment does:**
    1. Loads 57 FoVer labeled CoT steps from results/fover_labeled_steps_live.json.
    2. Splits 80/20 train/test.
    3. Trains EBRMEnergy (2-layer MLP over TF-IDF + reward scalar) for 200 epochs.
    4. Evaluates EBRM AUROC on the test split.
    5. Compares to EORM mean_auc=0.992812 from Exp 732.
    6. Reports honest_verdict and architectural_advantage.

REQ-EBRM-001, REQ-EBRM-002
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from repo root or scripts/ directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from python.carnot.pipeline.ebrm_baseline import EBRMEnergy, EBRMTrainer
from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DELIVERABLE = "results/experiment_771_ebrm_comparison.json"
FOVER_DATA = "results/fover_labeled_steps_live.json"
# EORM baseline AUC from Exp 732 5-fold cross-validation.
EORM_AUC = 0.992812  # mean_auc from experiment_732_probe_xval.json


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 771: EBRM vs EORM architectural comparison."""
    tmpl = ExperimentTemplate(
        exp_id=771,
        title="EBRM Baseline Comparison — arXiv 2504.13134 vs EORM Step-Level AUC",
        deliverable=DELIVERABLE,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=771,
        timeout_minutes=30,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )
    watchdog.start()

    try:
        artifact = _run(tmpl)
    finally:
        watchdog.stop()

    # Write the deliverable before calling assert_deliverable_written().
    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


def _run(tmpl: ExperimentTemplate) -> dict:
    """Core experiment logic — returns the result artifact dict."""
    from sklearn.metrics import roc_auc_score  # lazy import; only needed here

    # ------------------------------------------------------------------
    # 1. Load FoVer labeled steps
    # ------------------------------------------------------------------
    data_path = _REPO_ROOT / FOVER_DATA
    with data_path.open() as fh:
        records = json.load(fh)

    texts = [r["step_text"] for r in records]
    # FoVer labels: "correct" -> 1, anything else -> 0
    labels = [1 if r.get("label", "incorrect") == "correct" else 0 for r in records]

    n_total = len(texts)

    # ------------------------------------------------------------------
    # 2. 80/20 train/test split (deterministic — same seed as Exp 770 uses)
    # ------------------------------------------------------------------
    import numpy as np

    rng = np.random.default_rng(42)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    split = int(0.8 * n_total)
    train_idx = idx[:split]
    test_idx = idx[split:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    train_size = len(train_texts)
    test_size = len(test_texts)

    # ------------------------------------------------------------------
    # 3. Guard: insufficient test data
    # ------------------------------------------------------------------
    if test_size < 10:
        return tmpl.build_result(
            {
                "eorm_auc": EORM_AUC,
                "ebrm_auc": None,
                "auroc_delta": None,
                "architectural_advantage": False,
                "train_size": train_size,
                "test_size": test_size,
                "honest_verdict": "insufficient_data",
                "n_pos_test": sum(test_labels),
                "n_neg_test": test_size - sum(test_labels),
            },
            status="success",
        )

    # ------------------------------------------------------------------
    # 4. Train EBRM
    # ------------------------------------------------------------------
    model = EBRMEnergy(feature_dim=128, hidden_dim=64)
    trainer = EBRMTrainer(model, margin=1.0)
    trainer.train(train_texts, train_labels, n_epochs=200, lr=1e-3)

    # ------------------------------------------------------------------
    # 5. Evaluate EBRM on test split
    # ------------------------------------------------------------------
    scores = [trainer.predict(t) for t in test_texts]

    # AUC requires both classes in the test set.
    unique_labels = set(test_labels)
    if len(unique_labels) < 2:
        ebrm_auc = 0.5  # undefined; report as chance level
        auc_note = "test_set_single_class_auc_undefined"
    else:
        ebrm_auc = float(roc_auc_score(test_labels, scores))
        auc_note = "computed"

    # ------------------------------------------------------------------
    # 6. Compare to EORM baseline (REQ-EBRM-002)
    # ------------------------------------------------------------------
    eorm_auc = EORM_AUC
    auroc_delta = eorm_auc - ebrm_auc
    architectural_advantage = eorm_auc > ebrm_auc  # step-level wins

    # ------------------------------------------------------------------
    # 7. Honest verdict
    # ------------------------------------------------------------------
    if test_size < 10:
        honest_verdict = "insufficient_data"
    elif eorm_auc > ebrm_auc + 0.05:
        honest_verdict = "eorm_outperforms_ebrm"
    elif abs(eorm_auc - ebrm_auc) <= 0.05:
        honest_verdict = "ebrm_competitive"
    else:
        honest_verdict = "ebrm_outperforms_eorm"

    return tmpl.build_result(
        {
            "eorm_auc": eorm_auc,
            "ebrm_auc": round(ebrm_auc, 6),
            "auroc_delta": round(auroc_delta, 6),
            "architectural_advantage": architectural_advantage,
            "train_size": train_size,
            "test_size": test_size,
            "honest_verdict": honest_verdict,
            "n_pos_train": sum(train_labels),
            "n_neg_train": train_size - sum(train_labels),
            "n_pos_test": sum(test_labels),
            "n_neg_test": test_size - sum(test_labels),
            "auc_note": auc_note,
            "eorm_source": "experiment_732_probe_xval.json:mean_auc",
            "comparison_paper": "arXiv:2504.13134",
        },
        status="success",
    )


if __name__ == "__main__":
    main()
