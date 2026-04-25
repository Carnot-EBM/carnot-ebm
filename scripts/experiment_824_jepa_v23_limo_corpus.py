#!/usr/bin/env python3
"""Experiment 824 — JEPA v23: LIMO Curation + Contrastive Triplet Loss.

WHY THIS EXPERIMENT EXISTS:
    JEPA v13-v22 failed to exceed OOD AUC 0.50 in 11 consecutive retrains.
    Root cause analysis from the .62 retrospective identified three compounding failures:
    1. Training corpus too large and noisy (1050 items, uneven quality).
    2. Single-domain training (GSM8K only) causing OOD failure on different domains.
    3. Binary BCE loss fails to teach relative ordering of steps by correctness.

    This experiment fixes all three simultaneously:
    - LIMO curation (arXiv 2402.09353): top-50 pairs by z3_confidence × cpmi_score.
    - Domain diversity: +10 HumanEval + +10 SVAMP pairs = 70 total.
    - Contrastive triplet loss: forces correct steps closer to anchor than incorrect.

    Gate: OOD AUC >= 0.65 → "jepa_v23_viable"
          OOD AUC >= 0.50 → "jepa_v23_improvement"
          OOD AUC < 0.50  → "jepa_v23_below_random"

Spec: REQ-LEARN-824-001, REQ-LEARN-824-002, REQ-LEARN-824-003,
      SCENARIO-LEARN-824-001
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

# apply_env_autofix MUST be called before any JAX or heavy import.
REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_env_result = apply_env_autofix()

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.limo_curator import LIMOCurator  # noqa: E402
from carnot.inference.jepa_v23 import train_v23, evaluate_v23  # noqa: E402

DELIVERABLE = "results/experiment_824_jepa_v23_limo_corpus.json"
FOVER_PATH = REPO_ROOT / "results" / "fover_labeled_steps_live.json"
CPMI_PATH = REPO_ROOT / "results" / "experiment_798_cpmi_pairs_triples.json"
OOD_HOLDOUT_PATH = REPO_ROOT / "results" / "fover_labeled_steps_live.json"
MODEL_SAVE_PATH = REPO_ROOT / "results" / "jepa_v23_limo_model.pkl"

tmpl = ExperimentTemplate(
    824,
    "JEPA v23: LIMO Curation + Contrastive Triplet Loss",
    DELIVERABLE,
)
watchdog = ExperimentTimeoutWatchdog(824, timeout_minutes=60, result_path=DELIVERABLE)


def main() -> None:
    tmpl.setup()
    watchdog.start()

    try:
        _run()
    finally:
        watchdog.stop()


def _run() -> None:
    # ------------------------------------------------------------------
    # Step 1: LIMO curation — build 70-pair corpus
    # ------------------------------------------------------------------
    curator = LIMOCurator(
        fover_path=FOVER_PATH,
        cpmi_path=CPMI_PATH,
        z3_confidence_threshold=0.9,
    )

    curated = curator.add_domain_pairs(humaneval_n=10, svamp_n=10)

    n_gsm8k = sum(1 for p in curated if p.source_domain == "gsm8k")
    n_humaneval = sum(1 for p in curated if p.source_domain == "humaneval")
    n_svamp = sum(1 for p in curated if p.source_domain == "svamp")
    n_total = len(curated)

    print(f"[Exp 824] Curated corpus: {n_total} pairs "
          f"(gsm8k={n_gsm8k}, humaneval={n_humaneval}, svamp={n_svamp})")

    # ------------------------------------------------------------------
    # Step 2: Train JEPA v23 with triplet loss
    # ------------------------------------------------------------------
    print("[Exp 824] Training JEPA v23 for 100 epochs (triplet loss)...")

    model, train_losses, final_loss = train_v23(
        triples=curated,
        epochs=100,
        lr=1e-3,
        seed=42,
    )

    # Checkpoint every 25 epochs.
    tmpl.checkpoint_save({"train_losses_25": train_losses[:25]}, step=25)
    tmpl.checkpoint_save({"train_losses_50": train_losses[:50]}, step=50)
    tmpl.checkpoint_save({"train_losses_75": train_losses[:75]}, step=75)
    tmpl.checkpoint_save({"train_losses_100": train_losses}, step=100)

    print(f"[Exp 824] Training complete. Final loss: {final_loss:.6f}")

    # ------------------------------------------------------------------
    # Step 3: Evaluate OOD AUC
    # ------------------------------------------------------------------
    print("[Exp 824] Evaluating OOD AUC...")

    in_dist_auc, ood_auc = evaluate_v23(model, OOD_HOLDOUT_PATH)

    print(f"[Exp 824] in_dist_auc={in_dist_auc:.4f}, ood_auc={ood_auc:.4f}")

    # ------------------------------------------------------------------
    # Step 4: Save model
    # ------------------------------------------------------------------
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[Exp 824] Model saved to {MODEL_SAVE_PATH}")

    # ------------------------------------------------------------------
    # Step 5: Honest verdict
    # ------------------------------------------------------------------
    if ood_auc >= 0.65:
        honest_verdict = "jepa_v23_viable"
    elif ood_auc >= 0.50:
        honest_verdict = "jepa_v23_improvement"
    else:
        honest_verdict = "jepa_v23_below_random"

    print(f"[Exp 824] Verdict: {honest_verdict} (ood_auc={ood_auc:.4f})")

    # ------------------------------------------------------------------
    # Step 6: Write artifact
    # ------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "n_training_pairs": n_total,
            "n_gsm8k": n_gsm8k,
            "n_humaneval": n_humaneval,
            "n_svamp": n_svamp,
            "epochs": 100,
            "final_loss": final_loss,
            "in_dist_auc": in_dist_auc,
            "ood_auc": ood_auc,
            "honest_verdict": honest_verdict,
            "train_losses": train_losses,
        },
        status="success",
    )

    out_path = REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"[Exp 824] Artifact written to {out_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
