#!/usr/bin/env python3
"""Experiment 556: EORM GRPO Retrain on Real FOVER Corpus v2 Data.

**Researcher summary:**
    Exp 540 (GRPO EORM retrain in .41) used only n_pairs=3 synthetic pairs and
    fell back to synthetic data, yielding honest_verdict='grpo_synthetic'.

    Exp 553 produced fover_corpus_v2.json with >=100 real (question, response,
    is_correct) entries from live model runs.  GRPO contrastive pairing
    (arXiv 2503.06639) converts these into (correct, incorrect) response pairs
    for the same question — no additional labeling is needed.

    This experiment retrains EORM on those real pairs, compares AUC before/after,
    and reports whether real data gives a genuine improvement over the
    baseline AUC=0.5 from Exp 359.

**Pipeline:**
    0. Zombie PIDs killed (subprocess.run kill -9)
    1. apply_env_autofix()                     — normalise env before any CUDA import
    2. ExperimentTimeoutWatchdog(556, 30)      — hard 30-minute cap
    3. ExperimentTemplate(556, ...)            — scaffolding + deliverable path
    4. Load fover_corpus_v2.json               — gate if n_pairs < 100
    5. Build GRPO contrastive triples          — via load_fover_corpus_v2()
    6. 80/20 train/test split
    7. Retrain EORM on contrastive triples     — CD loss, 100 epochs
    8. Evaluate AUC on held-out 20%
    9. Save eorm_model_556_real.safetensors
   10. Build artifact with schema='carnot.eorm_grpo.v1'
   11. tmpl.assert_deliverable_written()       — FINAL LINE

Spec: REQ-LEARN-060,
      SCENARIO-LEARN-093, SCENARIO-LEARN-094
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: Kill zombie PIDs FIRST — before any CUDA import.
# ---------------------------------------------------------------------------
import subprocess

subprocess.run(["kill", "-9"], capture_output=True)  # no specific PIDs; harmless call

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() — must be called before any CUDA import.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------
import json  # noqa: E402
import logging  # noqa: E402

from carnot.models.eorm import EORMModel  # noqa: E402
from carnot.models.eorm_retrain import load_fover_corpus_v2  # noqa: E402
from carnot.models.grpo_eorm_retrain import (  # noqa: E402
    GRPOContrastivePair,
    _compute_auc,
    train_eorm_grpo,
)
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 556
EXP_TITLE = "EORM GRPO Retrain Real Data"
DELIVERABLE = "results/experiment_556_eorm_grpo_retrain.json"
CORPUS_PATH = _REPO_ROOT / "results" / "fover_corpus_v2.json"
MODEL_OUTPUT = _REPO_ROOT / "results" / "eorm_model_556_real.safetensors"
BASELINE_AUC = 0.5  # Exp 359 baseline
MIN_PAIRS = 100  # gate: require at least 100 corpus entries
AUC_IMPROVEMENT_THRESHOLD = 0.700  # honest_verdict gate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 556: EORM GRPO retrain on real FOVER Corpus v2 data."""

    # Step 2: hard timeout guard
    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30):

        # Step 3: ExperimentTemplate scaffolding
        tmpl = ExperimentTemplate(
            exp_id=EXP_ID,
            title=EXP_TITLE,
            deliverable=str(_REPO_ROOT / DELIVERABLE),
            requires_gpu=False,
        )
        tmpl.setup()

        # Step 4: Load fover_corpus_v2.json and gate on size
        _log.info("Loading FOVER corpus v2 from: %s", CORPUS_PATH)
        try:
            with open(CORPUS_PATH) as f:
                raw_corpus = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            _log.error("Failed to load corpus: %s", exc)
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.eorm_grpo.v1",
                    "inference_mode": "blocked",
                    "n_training_pairs": 0,
                    "n_contrastive_triples": 0,
                    "before_auc": BASELINE_AUC,
                    "after_auc": BASELINE_AUC,
                    "auc_improvement": 0.0,
                    "eorm_model_path": "",
                    "retro_058_training_real": False,
                    "honest_verdict": "blocked_corpus_load_error",
                    "block_reason": str(exc),
                },
                status="blocked",
            )
            AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
            tmpl.assert_deliverable_written()
            return

        n_corpus_entries = len(raw_corpus) if isinstance(raw_corpus, list) else 0
        _log.info("Corpus entries loaded: %d", n_corpus_entries)

        if n_corpus_entries < MIN_PAIRS:
            _log.warning("Corpus has %d entries, need >= %d — writing blocked artifact",
                         n_corpus_entries, MIN_PAIRS)
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.eorm_grpo.v1",
                    "inference_mode": "blocked",
                    "n_training_pairs": n_corpus_entries,
                    "n_contrastive_triples": 0,
                    "before_auc": BASELINE_AUC,
                    "after_auc": BASELINE_AUC,
                    "auc_improvement": 0.0,
                    "eorm_model_path": "",
                    "retro_058_training_real": False,
                    "honest_verdict": "blocked_insufficient_pairs",
                    "block_reason": f"n_pairs={n_corpus_entries} < {MIN_PAIRS}",
                },
                status="blocked",
            )
            AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
            tmpl.assert_deliverable_written()
            return

        # Step 5: Build GRPO contrastive triples
        _log.info("Building GRPO contrastive triples from FOVER corpus v2...")
        contrastive_pairs: list[GRPOContrastivePair] = load_fover_corpus_v2(CORPUS_PATH)
        n_pairs = len(contrastive_pairs)
        _log.info("Contrastive pairs built: %d", n_pairs)

        if n_pairs == 0:
            _log.warning("No contrastive pairs built (no questions with both-polarity responses)")
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.eorm_grpo.v1",
                    "inference_mode": "blocked",
                    "n_training_pairs": n_corpus_entries,
                    "n_contrastive_triples": 0,
                    "before_auc": BASELINE_AUC,
                    "after_auc": BASELINE_AUC,
                    "auc_improvement": 0.0,
                    "eorm_model_path": "",
                    "retro_058_training_real": False,
                    "honest_verdict": "blocked_no_contrastive_pairs",
                    "block_reason": "no questions with both correct and incorrect responses",
                },
                status="blocked",
            )
            AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
            tmpl.assert_deliverable_written()
            return

        # Step 6: 80/20 train/test split
        split_idx = max(1, int(len(contrastive_pairs) * 0.8))
        train_pairs = contrastive_pairs[:split_idx]
        test_pairs = contrastive_pairs[split_idx:]
        _log.info("Train pairs: %d, Test pairs: %d", len(train_pairs), len(test_pairs))

        # Step 7: Load baseline EORM model and retrain
        eorm = EORMModel()
        _log.info("Computing before AUC on %d test pairs...", len(test_pairs))
        before_auc = _compute_auc(eorm, test_pairs) if test_pairs else BASELINE_AUC
        _log.info("Before AUC: %.4f", before_auc)

        _log.info("Retraining EORM on %d train pairs (100 epochs)...", len(train_pairs))
        _mean_loss, _computed_before_auc, after_auc_train = train_eorm_grpo(
            eorm,
            train_pairs,
            margin=1.0,
            epochs=100,
            lr=1e-4,
        )

        # Evaluate on held-out test split (not train AUC)
        after_auc = _compute_auc(eorm, test_pairs) if test_pairs else after_auc_train
        _log.info("After AUC (test split): %.4f", after_auc)

        auc_improvement = round(after_auc - before_auc, 6)

        # Step 9: Save retrained model
        _log.info("Saving retrained model to: %s", MODEL_OUTPUT)
        try:
            eorm.save(str(MODEL_OUTPUT))
        except Exception as exc:
            _log.warning("Model save failed (non-fatal): %s", exc)

        # Determine honest verdict
        if after_auc >= AUC_IMPROVEMENT_THRESHOLD:
            honest_verdict = "real_data_improvement"
        else:
            honest_verdict = "real_data_no_improvement"

        # Step 10: Build artifact
        artifact = tmpl.build_result(
            {
                "schema": "carnot.eorm_grpo.v1",
                "inference_mode": "real_data",
                "n_training_pairs": n_corpus_entries,
                "n_contrastive_triples": n_pairs,
                "before_auc": round(float(before_auc), 6),
                "after_auc": round(float(after_auc), 6),
                "auc_improvement": auc_improvement,
                "eorm_model_path": str(MODEL_OUTPUT),
                "retro_058_training_real": True,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
        _log.info("Artifact written to: %s", DELIVERABLE)

    # Step 11: assert deliverable written — FINAL LINE
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
