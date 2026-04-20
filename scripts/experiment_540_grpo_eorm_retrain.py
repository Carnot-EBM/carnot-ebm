#!/usr/bin/env python3
"""Experiment 540: GRPO Contrastive EORM Retrain from Live Benchmark Verdicts.

**Researcher summary:**
    GRPO (arXiv 2503.06639) shows that verifiable binary rewards naturally form
    contrastive pairs: every question where the model is right vs. wrong is a free
    training signal.  NUP Probe v4 (Exp 523) validated that energy-gap contrastive
    training achieves AUC=1.0 vs 0.40 for BCE.

    This experiment applies GRPO contrastive retraining to EORM using live binary
    verdicts from Exp 538.  No additional labeling is needed: the benchmark verdicts
    ARE the contrastive signal.

**Pipeline:**
    1. apply_env_autofix()                  — env safety guard
    2. ExperimentTimeoutWatchdog(540, 30)   — hard 30-minute cap
    3. Load Exp 538 benchmark result        — build GRPO pairs from binary verdicts
    4. Fallback to FOVER (Exp 442)          — if fewer than 5 benchmark pairs
    5. Train EORM for 50 epochs             — contrastive loss on pairs
    6. Save retrained model                 — results/eorm_model_540_grpo.safetensors
    7. Build artifact                       — schema='carnot.grpo_eorm_retrain.v1'
    8. tmpl.assert_deliverable_written()    — FINAL LINE

Spec: REQ-LEARN-051, REQ-LEARN-052,
      SCENARIO-LEARN-080, SCENARIO-LEARN-081, SCENARIO-LEARN-082
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() MUST be called before any CUDA import.
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

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.models.eorm import EORMModel
from carnot.models.grpo_eorm_retrain import (
    build_grpo_pairs_from_benchmark,
    build_grpo_pairs_from_fover,
    train_eorm_grpo,
    make_grpo_result,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_RESULTS_DIR = _REPO_ROOT / "results"
_EXP538_RESULT = _RESULTS_DIR / "experiment_538_live_25q_precision_v9.json"
_FOVER_PATH = _RESULTS_DIR / "fover_labeled_steps_live.json"
_MODEL_OUTPUT = _RESULTS_DIR / "eorm_model_540_grpo.safetensors"
_DELIVERABLE = _RESULTS_DIR / "experiment_540_grpo_eorm_retrain.json"

_MIN_BENCHMARK_PAIRS = 5  # fall back to FOVER if fewer pairs extracted from benchmark

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 540: GRPO contrastive EORM retrain."""

    # Step 2: hard timeout guard
    with ExperimentTimeoutWatchdog(540, timeout_minutes=30):

        # Step 3: ExperimentTemplate scaffolding
        tmpl = ExperimentTemplate(
            exp_id=540,
            title="GRPO Contrastive EORM Retrain",
            deliverable=str(_DELIVERABLE),
            requires_gpu=False,
        )
        tmpl.setup()

        # Step 4: Load Exp 538 benchmark result and build GRPO pairs
        _log.info("Building GRPO pairs from benchmark: %s", _EXP538_RESULT)
        benchmark_pairs = build_grpo_pairs_from_benchmark(_EXP538_RESULT)
        _log.info("Benchmark pairs extracted: %d", len(benchmark_pairs))

        is_synthetic_fallback = False

        if len(benchmark_pairs) >= _MIN_BENCHMARK_PAIRS:
            pairs = benchmark_pairs
            data_source = "live_benchmark_exp538"
        else:
            # Step 5: Fall back to FOVER (Exp 442 live annotations)
            _log.info(
                "Fewer than %d benchmark pairs; falling back to FOVER: %s",
                _MIN_BENCHMARK_PAIRS,
                _FOVER_PATH,
            )
            pairs = build_grpo_pairs_from_fover(_FOVER_PATH)
            data_source = "fover_exp442"
            is_synthetic_fallback = True
            _log.info("FOVER pairs extracted: %d", len(pairs))

        _log.info("Training with %d pairs from source: %s", len(pairs), data_source)

        # Step 6: Train EORM for 50 epochs with contrastive loss
        # Uses a small CPU-friendly model (embed_dim=128, n_layers=2).
        # The 55M param model from the paper would be embed_dim=512, n_layers=12 —
        # that requires GPU.  We use the default CPU model for reproducibility.
        eorm_model = EORMModel(embed_dim=128, n_heads=4, n_layers=2)

        _log.info("Starting EORM GRPO contrastive retrain (50 epochs)...")
        training_loss, before_auc, after_auc = train_eorm_grpo(
            eorm_model,
            pairs,
            margin=1.0,
            epochs=50,
            lr=1e-4,
        )
        _log.info(
            "Training complete. loss=%.4f before_auc=%.4f after_auc=%.4f",
            training_loss,
            before_auc,
            after_auc,
        )

        # Step 7: Save retrained model
        _log.info("Saving retrained model to: %s", _MODEL_OUTPUT)
        eorm_model.save(str(_MODEL_OUTPUT))

        # Step 8: Build result with honest verdict
        result = make_grpo_result(
            n_pairs=len(pairs),
            before_auc=before_auc,
            after_auc=after_auc,
            is_synthetic_fallback=is_synthetic_fallback,
        )
        _log.info("honest_verdict: %s", result.honest_verdict)

        # Build artifact with all required schema fields
        artifact = tmpl.build_result(
            {
                "schema": "carnot.grpo_eorm_retrain.v1",
                "n_pairs": result.n_pairs,
                "before_auc": result.before_auc,
                "after_auc": result.after_auc,
                "auc_improvement": result.auc_improvement,
                "honest_verdict": result.honest_verdict,
                "data_source": data_source,
                "training_loss": round(float(training_loss), 6),
                "model_path": str(_MODEL_OUTPUT),
                "env_autofix_applied": getattr(_autofix_result, "auto_fix_applied", False),
            },
            status="success",
        )

        # Write deliverable
        _DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
        with open(_DELIVERABLE, "w") as f:
            json.dump(artifact, f, indent=2)
        _log.info("Deliverable written: %s", _DELIVERABLE)

        # Step 9: FINAL LINE — assert deliverable was written
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
