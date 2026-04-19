#!/usr/bin/env python3
"""Experiment 492: JEPA Curriculum Retrain V3 — recover from RETRO-040 AUC regression.

**Researcher summary:**
    JEPA AUC regressed 0.667 → 0.400 → 0.281 across three milestones.
    Exp 491 diagnosed the root cause: quality-gate filtering removed 73% of pairs,
    leaving an imbalanced corpus that caused majority-class collapse (predict everything
    as "correct"), yielding AUC = 0.281 — below random chance (0.5).

    This experiment applies the three-stage curriculum fix:

    Stage 1 (n_stage1_epochs=100): train only on pairs with label_confidence >= 0.85.
        Establishes a stable energy baseline before exposing the model to the full
        noisy distribution.  The model learns what "clearly correct" vs "clearly wrong"
        looks like before encountering ambiguous cases.

    Stage 2 (n_stage2_epochs=100): fine-tune on ALL pairs with NO confidence gate.
        The information loss from quality-gate filtering IS the root cause — Stage 2
        recovers that information.  The Stage 1 anchor prevents majority-class collapse.

    Stage 3 (n_stage3_epochs=100): augment with EBM-guided synthetic pairs to n_total >= 200.
        The real corpus is only 57 pairs.  Synthetic pairs from the Ising energy landscape
        concentrate on actual failure modes (energy function as ground truth).

**CPU-only experiment:**
    This experiment runs JAX_PLATFORMS=cpu.  GPU is not needed — EORM is tiny (embed_dim=32)
    and the corpus is small (< 300 pairs after augmentation).

**Outputs:**
    results/experiment_492_jepa_curriculum_retrain_v3.json

Spec: REQ-LEARN-040, REQ-LEARN-041, REQ-LEARN-042,
      SCENARIO-LEARN-069, SCENARIO-LEARN-070, RETRO-040
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() before any CUDA import (RETRO-022)
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------

import json
import logging
import os
import time

from carnot.models.jepa_curriculum_trainer import (
    JEPACurriculumTrainer,
    JEPARetrainV3Result,
)
from carnot.models.jepa_curriculum_diagnostic import _compute_auc
from carnot.models.eorm import EORMModel
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 492
EXP_TITLE = "JEPA Curriculum Retrain V3"
DELIVERABLE = "results/experiment_492_jepa_curriculum_retrain_v3.json"

FOVER_PAIRS_PATH = _REPO_ROOT / "results" / "fover_labeled_steps_live.json"
EXP488_PAIRS_PATH = _REPO_ROOT / "results" / "exp488_cot_pairs.json"

MODEL_SAVE_PATH = _REPO_ROOT / "results" / "jepa_model_492_curriculum.safetensors"

# Curriculum hyperparameters (from RETRO-040 diagnosis)
N_STAGE1_EPOCHS = 100
N_STAGE2_EPOCHS = 100
N_STAGE3_EPOCHS = 100
HIGH_CONF_THRESHOLD = 0.85  # more conservative than Exp 477's 0.70

# AUC targets
TARGET_AUC = 0.600          # RETRO-040 closure bar
RECOVERY_AUC = 0.400        # recovery from 0.281 regression
RETRO_CLOSED_AUC = 0.500    # minimum for retro_040_closed


def _load_json_pairs(path: Path) -> list[dict]:
    """Load a JSON file as a list of dicts; return [] if absent or malformed."""
    if not path.exists():
        log.info("Optional pairs file not found, skipping: %s", path)
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        for key in ("pairs", "steps", "data", "results"):
            if isinstance(data, dict) and key in data and isinstance(data[key], list):
                return data[key]
        log.warning("Unexpected format in %s, skipping", path)
        return []
    except Exception as exc:
        log.warning("Failed to load %s: %s", path, exc)
        return []


def main() -> None:
    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=60):
        tmpl = ExperimentTemplate(
            EXP_ID,
            EXP_TITLE,
            DELIVERABLE,
        )
        tmpl.setup()

        guard = DeliverableGuard(DELIVERABLE)

        # --- Load all available real pairs ---
        log.info("Loading real CoT pairs from FOVER and optional Exp 488...")
        fover_pairs = _load_json_pairs(FOVER_PAIRS_PATH)
        exp488_pairs = _load_json_pairs(EXP488_PAIRS_PATH)
        all_pairs = fover_pairs + exp488_pairs
        n_pairs_raw = len(all_pairs)
        log.info("Total real pairs loaded: %d", n_pairs_raw)

        if n_pairs_raw == 0:
            log.warning("No real pairs found — synthetic pairs only (degraded quality)")
            # Create minimal synthetic corpus so experiment can still proceed
            from carnot.models.ising import IsingConfig, IsingModel
            from carnot.models.jepa_retrain_v2 import JEPAQualityAugmentor
            ising = IsingModel(IsingConfig(input_dim=8))
            aug = JEPAQualityAugmentor(ising_model=ising, n_samples=60)
            all_pairs = aug.generate_violation_pairs() + aug.generate_correct_pairs()
            n_pairs_raw = 0  # report zero real pairs

        # --- Compute before_auc on current (untrained baseline) EORM ---
        log.info("Computing before_auc on held-out 20%...")
        import jax.random as jrandom
        n_held = max(1, len(all_pairs) // 5)
        held_out = all_pairs[len(all_pairs) - n_held:]

        key = jrandom.PRNGKey(42)
        baseline_model = EORMModel(
            embed_dim=32,
            n_heads=2,
            n_layers=2,
            max_seq_len=128,
            vocab_size=512,
            key=key,
        )
        before_auc = _compute_auc(baseline_model, held_out)
        log.info("before_auc (untrained baseline) = %.4f", before_auc)

        # --- Three-stage curriculum training ---
        log.info("Starting three-stage curriculum training (high_conf_threshold=%.2f)...", HIGH_CONF_THRESHOLD)
        trainer = JEPACurriculumTrainer(
            n_stage1_epochs=N_STAGE1_EPOCHS,
            n_stage2_epochs=N_STAGE2_EPOCHS,
            n_stage3_epochs=N_STAGE3_EPOCHS,
            high_conf_threshold=HIGH_CONF_THRESHOLD,
        )
        stages = trainer.train(all_pairs)

        for s in stages:
            log.info(
                "Stage %d: n_pairs=%d, n_epochs=%d, auc_after=%.4f",
                s.stage, s.n_pairs, s.n_epochs, s.auc_after,
            )

        # --- Final AUC ---
        after_auc = trainer.get_final_auc(held_out)
        log.info("after_auc (held-out 20%) = %.4f", after_auc)

        # --- Save retrained model ---
        if trainer._model is not None:
            try:
                trainer._model.save(str(MODEL_SAVE_PATH))
                log.info("Retrained JEPA model saved to %s", MODEL_SAVE_PATH)
            except Exception as exc:
                log.warning("Model save failed (non-critical): %s", exc)

        # --- Build result object ---
        result = JEPARetrainV3Result(
            n_pairs_raw=n_pairs_raw,
            curriculum_stages=stages,
            before_auc=before_auc,
            after_auc=after_auc,
        )

        # --- Determine FR-11 Tier 3 status ---
        if result.target_met:
            fr11_tier3_status = "recovered"
        elif result.regression_recovered:
            fr11_tier3_status = "partial_recovery"
        else:
            fr11_tier3_status = "insufficient"

        retro_040_closed = after_auc > RETRO_CLOSED_AUC

        if retro_040_closed:
            honest_verdict = "retro_040_closed"
        elif result.regression_recovered:
            honest_verdict = "partial_recovery"
        else:
            honest_verdict = "insufficient_recovery"

        # --- Build artifact ---
        artifact = tmpl.build_result(
            {
                "n_pairs_raw": n_pairs_raw,
                "curriculum_stages": [
                    {
                        "stage": s.stage,
                        "n_pairs": s.n_pairs,
                        "n_epochs": s.n_epochs,
                        "auc_after": s.auc_after,
                    }
                    for s in stages
                ],
                "before_auc": before_auc,
                "after_auc": after_auc,
                "auc_improvement": result.auc_improvement,
                "target_met": result.target_met,
                "regression_recovered": result.regression_recovered,
                "retro_040_closed": retro_040_closed,
                "fr11_tier3_status": fr11_tier3_status,
                "honest_verdict": honest_verdict,
            },
            status="success",
            schema="carnot.jepa_retrain.v3",
        )

        # --- Write deliverable ---
        deliverable_path = _REPO_ROOT / DELIVERABLE
        deliverable_path.parent.mkdir(parents=True, exist_ok=True)
        with open(deliverable_path, "w") as f:
            json.dump(artifact, f, indent=2)
        log.info("Deliverable written: %s", deliverable_path)

        # FINAL LINE — assert deliverable was actually written
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
