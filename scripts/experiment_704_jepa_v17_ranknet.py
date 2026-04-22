"""Experiment 704 — JEPA v17: RankNet Pairwise Ranking Loss, OOD AUC Target >= 0.75.

**Why this experiment exists:**
    JEPA v15 OOD AUC=0.4751 and v16 OOD AUC=0.4759, both BELOW random chance.
    Exp 693 confirmed root cause: pure_loss_anti_correlation. Both BCE and InfoNCE
    allow the model to hedge all outputs to P≈0.5, satisfying the loss without learning
    any step-level discrimination.

    v17 fixes this with RankNet pairwise ranking loss:
        L = -log(sigmoid(score(incorrect) - score(correct)))
    Hedging to P=0.5 gives L=log(2) ≈ 0.693 per pair — non-zero gradient every step.
    Only correct ranking (incorrect score >> correct score) drives L toward 0.
    The model CANNOT hedge.

    Hard negative mining ensures training pairs are maximally informative: for each
    correct step, we select the most cosine-similar incorrect step, forcing the model
    to learn subtle semantic discrimination rather than surface-level differences.

**Gates:**
    - results/experiment_693_jepa_v15_root_cause.json: root_cause confirmed.
    - results/fover_labeled_formal_v1.json: n_pairs >= 100.

**Outputs:**
    - results/experiment_704_jepa_v17_ranknet.json: full artifact.
    - results/jepa_v17_ranknet.npz: trained model weights (if cascade gate opens).

**Spec:** REQ-VERIFY-140, REQ-VERIFY-141, REQ-VERIFY-142,
          SCENARIO-VERIFY-140, SCENARIO-VERIFY-141, SCENARIO-VERIFY-142
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate
from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from python.carnot.inference.jepa_v17_ranknet import (
    train_jepa_v17,
    evaluate_ood_auc,
)

DELIVERABLE = "results/experiment_704_jepa_v17_ranknet.json"
V16_BASELINE_AUC = 0.4759
V15_BASELINE_AUC = 0.4751


def _load_json(path: Path) -> dict | None:
    """Load JSON file; return None if missing or malformed."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> None:
    tmpl = ExperimentTemplate(
        704,
        "JEPA v17: RankNet Pairwise Ranking Loss — OOD AUC Target >= 0.75 on GSM8K 500-699",
        DELIVERABLE,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(
        704, timeout_minutes=60, result_path=DELIVERABLE
    )
    watchdog.start()

    try:
        _run(tmpl)
    finally:
        watchdog.stop()

    tmpl.assert_deliverable_written()


def _run(tmpl: ExperimentTemplate) -> None:
    results_dir = _REPO_ROOT / "results"

    # ---- Gate 1: Exp 693 root cause must be confirmed ----
    exp693 = _load_json(results_dir / "experiment_693_jepa_v15_root_cause.json")
    if exp693 is None or "root_cause" not in exp693:
        artifact = tmpl.build_result(
            {"gate_failed": "exp693_missing_or_no_root_cause"},
            status="blocked",
            honest_verdict="jepa_v17_gate_failed",
        )
        (results_dir / DELIVERABLE.split("/")[-1]).write_text(json.dumps(artifact, indent=2))
        return

    root_cause = exp693["root_cause"]

    # ---- Gate 2: FoVer formal v1 data ----
    fover_data = _load_json(results_dir / "fover_labeled_formal_v1.json")
    if fover_data is None:
        artifact = tmpl.build_result(
            {"gate_failed": "fover_labeled_formal_v1_missing"},
            status="blocked",
            honest_verdict="jepa_v17_gate_failed",
        )
        (results_dir / DELIVERABLE.split("/")[-1]).write_text(json.dumps(artifact, indent=2))
        return

    fover_pairs = fover_data.get("pairs", fover_data.get("labeled_pairs", []))
    if len(fover_pairs) < 100:
        artifact = tmpl.build_result(
            {"gate_failed": f"fover_n_pairs_too_small: {len(fover_pairs)} < 100"},
            status="blocked",
            honest_verdict="jepa_v17_gate_failed",
        )
        (results_dir / DELIVERABLE.split("/")[-1]).write_text(json.dumps(artifact, indent=2))
        return

    n_training_pairs = len(fover_pairs)

    # ---- Step 1: Train JEPARankNetV17 ----
    # 50 epochs is sufficient for RankNet convergence on 200 pairs — v16 used 200
    # epochs for InfoNCE but RankNet's tighter per-pair gradient signal converges faster.
    model, train_losses = train_jepa_v17(fover_pairs, n_epochs=50, lr=1e-3)

    train_loss_final = round(train_losses[-1], 6) if train_losses else None

    # ---- Step 2: Evaluate OOD AUC on GSM8K 500-699 ----
    ood_auc = evaluate_ood_auc(model, gsm8k_indices=range(500, 700))
    v17_ood_auc = round(ood_auc, 4)

    # ---- Step 3: Cascade gate ----
    cascade_gate_open = v17_ood_auc >= 0.75

    # ---- Step 4: Save model if gate opens ----
    model_saved_path = None
    if cascade_gate_open:
        model_path = results_dir / "jepa_v17_ranknet.npz"
        model.save(str(model_path))
        model_saved_path = str(model_path)

    # ---- Step 5: Honest verdict ----
    if v17_ood_auc >= 0.75:
        honest_verdict = "jepa_v17_cascade_unblocked"
    elif v17_ood_auc >= 0.5:
        honest_verdict = "jepa_v17_improved_below_threshold"
    else:
        honest_verdict = "jepa_v17_still_below_random"

    # ---- Step 6: v18 recommendation ----
    if cascade_gate_open:
        v18_recommendation = "cascade_unblocked_no_v18_needed"
    else:
        v18_recommendation = (
            "listwise_lambdarank: rank all steps simultaneously (not just pairs) "
            "using LambdaRank loss with NDCG surrogate. Pairwise RankNet covers all "
            "pairs but treats each pair independently; listwise loss optimises the "
            "full ranking of all steps for a question jointly, which is the true "
            "evaluation metric (AUC = fraction of correct pairwise orderings)."
        )

    # ---- Step 7: Write artifact ----
    artifact = tmpl.build_result(
        {
            "root_cause_addressed": root_cause,
            "v17_architecture_applied": "RankNet pairwise ranking loss + hard negative mining",
            "v17_ood_auc": v17_ood_auc,
            "v16_baseline_auc": V16_BASELINE_AUC,
            "v15_baseline_auc": V15_BASELINE_AUC,
            "ood_auc_delta_vs_v16": round(v17_ood_auc - V16_BASELINE_AUC, 4),
            "v17_train_loss_final": train_loss_final,
            "n_training_pairs": n_training_pairs,
            "cascade_gate_open": cascade_gate_open,
            "model_saved_path": model_saved_path,
            "honest_verdict": honest_verdict,
            "v18_recommendation": v18_recommendation,
        },
        status="success",
    )
    (results_dir / DELIVERABLE.split("/")[-1]).write_text(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
