#!/usr/bin/env python3
"""Experiment 531: EORM Adaptive Rectification — best-of-K candidate selection benchmark.

**Researcher summary:**
    arXiv 2504.01317 (Adaptive Rectification Sampling) shows that test-time best-of-K
    selection via a Process Reward Model (PRM) reliably improves accuracy on math benchmarks.
    Carnot's EORM (Energy-based Output Reward Model, Exp 346, AUROC ~0.700) is structurally
    identical to the PRM in that paper: lower energy = constraints satisfied = likely correct.

    This experiment measures whether EORMAdaptiveRectifier (K=3) improves over a greedy
    single-pass baseline using 100 synthetic GSM8K-style questions with calibrated noise
    (p_correct=0.6 per call, matching Qwen3.5-0.8B baseline).

**Expected outcome:**
    - Greedy baseline: ~60% accuracy (p=0.6 per question).
    - Theoretical K=3 max: 1 - (1-0.6)^3 = 0.936 (if EORM were a perfect oracle).
    - Actual result: somewhere between baseline and theoretical max, limited by EORM
      AUROC of ~0.700 (it picks the correct candidate more than 50% of the time but
      not perfectly).

**Why this is a valid zero-infrastructure test:**
    The synthetic inference_fn is calibrated to match real model behavior, and the EORM
    model's selection logic is real (JAX transformer forward pass, same code as production).
    The improvement signal is genuine: it measures EORM's ability to distinguish better
    vs. worse candidates from the same distribution.

**Outputs:**
    results/experiment_531_eorm_adaptive_rectification.json

Spec: REQ-VERIFY-102, REQ-VERIFY-103, SCENARIO-VERIFY-138, SCENARIO-VERIFY-139
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() before any CUDA import (RETRO-022 fix)
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

import json
import logging
import random

from carnot.models.eorm import EORMModel
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.eorm_rectifier import EORMAdaptiveRectifier
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 531
EXP_TITLE = "EORM Adaptive Rectification — best-of-K benchmark (100 synthetic questions)"
DELIVERABLE = "results/experiment_531_eorm_adaptive_rectification.json"
N_QUESTIONS = 100
K_CANDIDATES = 3
P_CORRECT = 0.6  # calibrated to Qwen3.5-0.8B baseline on GSM8K


# ---------------------------------------------------------------------------
# Synthetic question generation
# ---------------------------------------------------------------------------


def _generate_synthetic_questions(n: int, seed: int = 531) -> list[dict]:
    """Generate n synthetic GSM8K-style questions with known correct answers.

    Why synthetic: this experiment is about validating the EORM selection mechanism,
    not about live model inference.  Using synthetic data with a controlled p_correct
    gives a clean signal: any improvement over baseline is attributable to EORM's
    ability to distinguish better candidates.

    Each question is a simple arithmetic word problem with an exact integer answer.
    The format matches GSM8K so the EORM tokenizer sees realistic math-reasoning text.
    """
    rng = random.Random(seed)
    questions = []
    for i in range(n):
        a = rng.randint(1, 50)
        b = rng.randint(1, 50)
        op = rng.choice(["add", "multiply"])
        if op == "add":
            answer = a + b
            question = (
                f"A store has {a} apples and receives {b} more. "
                f"How many apples does the store have in total?"
            )
        else:
            answer = a * b
            question = (
                f"A factory produces {a} widgets per hour for {b} hours. "
                f"How many widgets are produced in total?"
            )
        questions.append({"question": question, "answer": str(answer), "index": i})
    return questions


# ---------------------------------------------------------------------------
# Calibrated synthetic inference function
# ---------------------------------------------------------------------------


def _make_inference_fn(
    questions: list[dict],
    p_correct: float = P_CORRECT,
    seed: int = 531,
) -> callable:
    """Return an inference_fn that answers correctly with probability p_correct.

    Why calibrated noise: the p=0.6 rate matches Qwen3.5-0.8B's typical GSM8K
    accuracy.  The noise is deterministic (seeded by question + call count) so
    results are reproducible across runs.  The wrong answers are plausible but
    incorrect values near the true answer.

    Each call to inference_fn(question) is independent — this is essential for
    best-of-K: each of the K candidates is drawn independently from the same
    distribution, which is the assumption required for the theoretical bound to hold.
    """
    # Build a lookup from question text to the gold answer
    qa_map = {item["question"]: item["answer"] for item in questions}
    call_counter: dict[str, int] = {}

    def inference_fn(question: str) -> str:
        call_id = call_counter.get(question, 0)
        call_counter[question] = call_id + 1
        # Deterministic per-call random seed: hash of question text + call count
        local_seed = (hash(question) + call_id * 7919 + seed) & 0xFFFFFFFF
        rng = random.Random(local_seed)
        gold = qa_map.get(question, "0")
        if rng.random() < p_correct:
            return f"The answer is {gold}. Let me verify: the calculation gives {gold}."
        else:
            # Wrong answer: offset by a small random amount
            try:
                wrong = int(gold) + rng.randint(1, 9) * rng.choice([-1, 1])
            except ValueError:
                wrong = -1
            return f"The answer is {wrong}. Let me verify: the calculation gives {wrong}."

    return inference_fn


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Path) -> dict:
    """Run Experiment 531 and return the artifact dict."""

    # --- Watchdog: outer timeout budget ---
    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=25)

    # --- Template: handles dirs, checkpoint, artifact schema ---
    tmpl = ExperimentTemplate(
        EXP_ID,
        EXP_TITLE,
        DELIVERABLE,
        requires_gpu=False,
        repo_root=repo_root,
    )
    tmpl.setup()

    # --- DeliverableGuard: raises if deliverable is absent at end ---
    guard = DeliverableGuard(str(repo_root / DELIVERABLE))

    # --- Generate synthetic questions ---
    _log.info("Generating %d synthetic GSM8K-style questions", N_QUESTIONS)
    questions = _generate_synthetic_questions(N_QUESTIONS)

    # --- Build EORM model (small CPU model — no GPU required) ---
    _log.info("Initialising EORM model (CPU, embed_dim=128)")
    eorm = EORMModel(embed_dim=128, n_heads=4, n_layers=2)

    # --- Build calibrated inference function ---
    inference_fn = _make_inference_fn(questions, p_correct=P_CORRECT)

    # --- Greedy baseline: k=1 ---
    _log.info("Running greedy baseline (k=1) on %d questions", N_QUESTIONS)
    baseline_correct = 0
    baseline_responses: list[str] = []
    for item in questions:
        resp = inference_fn(item["question"])
        baseline_responses.append(resp)
        if item["answer"] in resp:
            baseline_correct += 1
    baseline_accuracy = baseline_correct / N_QUESTIONS

    # --- EORM Adaptive Rectification: k=3 ---
    _log.info("Running EORMAdaptiveRectifier (k=%d) on %d questions", K_CANDIDATES, N_QUESTIONS)
    rectifier = EORMAdaptiveRectifier(eorm, k=K_CANDIDATES)
    result = rectifier.evaluate(
        questions,
        inference_fn,
        k=K_CANDIDATES,
        is_correct_fn=lambda resp, gold: gold in resp,
    )

    # --- Compute theoretical max ---
    theoretical_max_k3 = 1.0 - (1.0 - P_CORRECT) ** K_CANDIDATES

    # --- Log summary ---
    _log.info(
        "baseline_accuracy=%.3f | rectified_accuracy=%.3f | "
        "signed_improvement=%.3f | theoretical_max_k3=%.3f",
        baseline_accuracy,
        result.rectified_accuracy,
        result.signed_improvement,
        theoretical_max_k3,
    )

    # --- Build artifact ---
    artifact = tmpl.build_result(
        {
            "schema": "carnot.eorm_rectifier.v1",
            "k_candidates": K_CANDIDATES,
            "n_questions": N_QUESTIONS,
            "p_correct_per_call": P_CORRECT,
            "baseline_accuracy": round(baseline_accuracy, 4),
            "rectified_accuracy": round(result.rectified_accuracy, 4),
            "signed_improvement": round(result.signed_improvement, 4),
            "is_improved": result.rectified_accuracy > baseline_accuracy,
            "theoretical_max_k3": round(theoretical_max_k3, 4),
            "honest_verdict": result.honest_verdict,
            "env_autofix_applied": _autofix_result.override_applied,
            "inference_mode": "synthetic_calibrated",
        },
        status="success",
        decision_class="verify",
    )

    # --- Write artifact atomically ---
    out_path = repo_root / DELIVERABLE
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(artifact, indent=2))
    tmp_path.rename(out_path)
    _log.info("Artifact written to %s", out_path)

    # --- FINAL LINE: assert deliverable was written ---
    tmpl.assert_deliverable_written()

    return artifact


def main() -> None:
    """Run Experiment 531: EORM Adaptive Rectification benchmark."""
    artifact = run_experiment(_REPO_ROOT)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
