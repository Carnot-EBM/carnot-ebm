"""Experiment 430: FOVER Z3 Step Annotation — produce (step, label) pairs for EORM training.

**What this experiment does:**
    Implements the FoVer-style annotation pipeline (arXiv 2505.15960) that solves
    the long-standing FR-11 miss: EORM/JEPA retrains have run on SYNTHETIC data only
    because there was no automated way to annotate live LLM inference with step-level
    correctness labels.

    This experiment bridges that gap:
    1. Load live CoT responses from Exp 427 (if available).
       Fallback: generate 50 synthetic GSM8K-style responses with known-correct and
       known-incorrect arithmetic steps.
    2. Run FOVERAnnotator.annotate_corpus() — pure Z3 CPU, no LLM calls, < 5ms/step.
    3. Write labeled pairs to results/fover_labeled_steps.json.
    4. Build artifact with labeling statistics and honest_verdict.

**Why this matters for FR-11:**
    Carnot's Ising/KAN tier is a Verifiable Process Reward Model (VPRM, arXiv 2601.17223).
    It needs (step_text, correct/incorrect) training pairs to learn which reasoning steps
    have high vs. low energy.  Without these labels, the EORM training signal is zero.
    This experiment produces those labels deterministically using Z3.

**Honest verdict:**
    - 'real_data_labeled':  Live responses loaded from Exp 427.
    - 'synthetic_fallback': Exp 427 data unavailable; synthetic responses used.
      Synthetic data is useful for pipeline validation but NOT as a headline result.

**Output:**
    results/fover_labeled_steps.json  — training data for Exp 431 EORM retrain.
    results/experiment_430_fover_z3_labels.json — experiment artifact.

Spec: REQ-LEARN-030, REQ-LEARN-031,
      SCENARIO-LEARN-054, SCENARIO-LEARN-055, SCENARIO-LEARN-056
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Call apply_env_autofix FIRST — before any other carnot imports — so GPU env
# is correctly set for this process and any children it spawns.
from carnot.pipeline.env_autofix import apply_env_autofix

_env_fix = apply_env_autofix()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.fover_annotator import FOVERAnnotator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
EXP_427_PATH = RESULTS_DIR / "experiment_427_precision_live_confirmed.json"
LABELED_STEPS_PATH = RESULTS_DIR / "fover_labeled_steps.json"
ARTIFACT_PATH = RESULTS_DIR / "experiment_430_fover_z3_labels.json"

EXPERIMENT_ID = 430


# ---------------------------------------------------------------------------
# Synthetic GSM8K-style fallback responses
# ---------------------------------------------------------------------------


def _make_synthetic_responses() -> list[dict]:
    """Generate 50 synthetic GSM8K-style responses with known correct/incorrect steps.

    **Detailed explanation for engineers:**
        These responses are deliberately crafted with a mix of correct and incorrect
        arithmetic steps so the annotation pipeline can be validated end-to-end.

        Pattern A (correct): all arithmetic is right → all labeled steps should be 'correct'.
        Pattern B (incorrect): one step has a wrong equation → that step labeled 'incorrect'.

        We generate 25 of each type for a balanced synthetic corpus.  The 'honest_verdict'
        will be 'synthetic_fallback', clearly marking this as pipeline validation, not a
        real-data result.
    """
    responses = []

    # 25 fully-correct responses
    for i in range(25):
        a, b = i + 1, i + 2
        c = a + b
        d = c * 2
        responses.append(
            {
                "question_id": f"synthetic_correct_{i}",
                "question": f"What is ({a} + {b}) × 2?",
                "response": (
                    f"Step 1: Add the numbers: {a} + {b} = {c}.\n"
                    f"Step 2: Multiply by 2: {c} * 2 = {d}.\n"
                    f"The answer is {d}."
                ),
                "expected_correct": True,
            }
        )

    # 25 responses with one incorrect step each
    for i in range(25):
        a, b = i + 3, i + 4
        c_wrong = a + b + 1  # deliberately wrong
        c_right = a + b
        d = c_right * 3
        responses.append(
            {
                "question_id": f"synthetic_incorrect_{i}",
                "question": f"What is ({a} + {b}) × 3?",
                "response": (
                    f"Step 1: Add the numbers: {a} + {b} = {c_wrong}.\n"  # WRONG
                    f"Step 2: Multiply by 3: {c_right} * 3 = {d}.\n"
                    f"The answer is {d}."
                ),
                "expected_correct": False,
            }
        )

    return responses


# ---------------------------------------------------------------------------
# Load responses
# ---------------------------------------------------------------------------


def _load_responses() -> tuple[list[dict], str]:
    """Load live Exp 427 responses, or fall back to synthetic corpus.

    Returns (responses, honest_verdict).
    """
    if EXP_427_PATH.exists():
        _log.info("Loading live responses from %s", EXP_427_PATH)
        with open(EXP_427_PATH) as f:
            data = json.load(f)

        # Exp 427 result is scaffolding-only — it has no live inference results.
        # Check for a 'responses' key or 'results' key with actual CoT text.
        raw_responses = data.get("responses", data.get("results", []))
        if raw_responses and isinstance(raw_responses, list) and len(raw_responses) > 0:
            # Validate at least one entry has a 'response' key with text.
            if isinstance(raw_responses[0], dict) and "response" in raw_responses[0]:
                _log.info("Loaded %d live responses from Exp 427", len(raw_responses))
                return raw_responses, "real_data_labeled"
            else:
                _log.warning(
                    "Exp 427 data found but no 'response' fields — using synthetic fallback"
                )
        else:
            _log.warning(
                "Exp 427 result is scaffolding-only (no live inference) — using synthetic fallback"
            )
    else:
        _log.info("Exp 427 result not found at %s — using synthetic fallback", EXP_427_PATH)

    return _make_synthetic_responses(), "synthetic_fallback"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _log.info("Experiment %d: FOVER Z3 Step Annotation", EXPERIMENT_ID)
    start_time = datetime.now(tz=timezone.utc)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Watchdog: 30-minute hard cap.
    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXPERIMENT_ID,
        timeout_minutes=30,
        result_path=str(ARTIFACT_PATH),
    )

    with watchdog:
        # 1. Load responses.
        responses, honest_verdict = _load_responses()
        _log.info("honest_verdict=%s, n_responses=%d", honest_verdict, len(responses))

        # 2. Run FOVERAnnotator.
        annotator = FOVERAnnotator(z3_timeout_seconds=5)
        annotated = annotator.annotate_corpus(responses)

        # 3. Collect statistics.
        n_responses_processed = len(annotated)
        n_steps_found = sum(len(steps) for steps in annotated)
        n_labeled_correct = sum(
            1 for steps in annotated for s in steps if s.z3_label == "correct"
        )
        n_labeled_incorrect = sum(
            1 for steps in annotated for s in steps if s.z3_label == "incorrect"
        )
        n_not_verifiable = sum(
            1 for steps in annotated for s in steps if s.z3_label == "not_verifiable"
        )
        n_labeled = n_labeled_correct + n_labeled_incorrect
        labeling_rate = n_labeled / n_steps_found if n_steps_found > 0 else 0.0

        _log.info(
            "Steps found=%d  correct=%d  incorrect=%d  not_verifiable=%d  rate=%.2f",
            n_steps_found,
            n_labeled_correct,
            n_labeled_incorrect,
            n_not_verifiable,
            labeling_rate,
        )

        # 4. Export training pairs.
        training_pairs = annotator.to_training_pairs(annotated, responses=responses)
        labeled_output = {
            "schema": "carnot.fover_labels.v1",
            "experiment": EXPERIMENT_ID,
            "honest_verdict": honest_verdict,
            "n_pairs": len(training_pairs),
            "pairs": training_pairs,
        }
        with open(LABELED_STEPS_PATH, "w") as f:
            json.dump(labeled_output, f, indent=2)
        _log.info("Wrote %d training pairs to %s", len(training_pairs), LABELED_STEPS_PATH)

        # 5. Build artifact.
        end_time = datetime.now(tz=timezone.utc)
        duration_s = (end_time - start_time).total_seconds()

        artifact = {
            "schema": "carnot.fover_labels.v1",
            "experiment": EXPERIMENT_ID,
            "run_date": start_time.date().isoformat(),
            "status": "success",
            "honest_verdict": honest_verdict,
            "env_autofix": {
                "gpu_detected": _env_fix.gpu_detected,
                "auto_fix_applied": _env_fix.auto_fix_applied,
                "final_env_value": _env_fix.final_env_value,
            },
            "n_responses_processed": n_responses_processed,
            "n_steps_found": n_steps_found,
            "n_labeled_correct": n_labeled_correct,
            "n_labeled_incorrect": n_labeled_incorrect,
            "n_not_verifiable": n_not_verifiable,
            "labeling_rate": round(labeling_rate, 4),
            "n_training_pairs": len(training_pairs),
            "labeled_steps_path": str(LABELED_STEPS_PATH),
            "duration_s": round(duration_s, 2),
            "notes": (
                "FOVER annotation pipeline (arXiv 2505.15960). "
                "Labels are Z3-verified, deterministic, and CPU-only (<5ms/step). "
                "Training pairs in fover_labeled_steps.json are inputs to Exp 431 EORM retrain."
                if honest_verdict == "real_data_labeled"
                else (
                    "Synthetic fallback: Exp 427 live data unavailable. "
                    "Pipeline validated but training pairs are synthetic. "
                    "Do NOT use as headline result for FR-11 EORM retrain."
                )
            ),
        }

        with open(ARTIFACT_PATH, "w") as f:
            json.dump(artifact, f, indent=2)
        _log.info("Artifact written to %s", ARTIFACT_PATH)

    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
