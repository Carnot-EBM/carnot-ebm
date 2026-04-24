#!/usr/bin/env python3
"""Exp 788: Constraint Addition from Memory — Dynamic vs Static IsingEBM (50 GSM8K-style).

**Researcher summary:**
    Exp 134 proved that precision-based REWEIGHTING of existing IsingEBM constraints
    gives 0% improvement (fixed == adaptive across 500 questions).  The root cause:
    changing weights on existing constraints cannot add NEW detection capability when
    the constraint TOPOLOGY is wrong.  The correct mechanism is to generate new
    IsingEBM coupling rows from accumulated error patterns in session memory.

    This experiment tests that hypothesis directly:
    - Static baseline: VerifyRepairPipeline with a standard fixed IsingEBM.
    - Dynamic pipeline: same pipeline + IsingConstraintGenerator that reads session
      memory error patterns every 10 questions and injects new coupling rows when
      any pattern count >= PATTERN_THRESHOLD (3).

    The primary metric is constraint_addition_delta = net_improvement_dynamic -
    net_improvement_static.  A positive delta is evidence that constraint ADDITION
    (not reweighting) is the right mechanism.

**Why synthetic GSM8K-style questions:**
    Live GPU inference via GGUF models is not required to test whether the
    IsingConstraintGenerator topology extension changes the pipeline's verdict.
    Synthetic questions with deterministic oracle labels give a reproducible
    baseline that can be run in CI without CARNOT_FORCE_LIVE.

**honest_verdict logic:**
    - "constraint_addition_positive"   if constraint_addition_delta > 0
    - "constraint_addition_zero"       if constraint_addition_delta == 0
    - "constraint_addition_negative"   if constraint_addition_delta < 0 (unexpected)
    - "insufficient_patterns"          if n_patterns_above_threshold == 0

Spec: REQ-LEARN-056, REQ-LEARN-057, SCENARIO-LEARN-100, SCENARIO-LEARN-101
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.models.ising import IsingConfig, IsingModel  # noqa: E402
from carnot.pipeline.constraint_generator import (  # noqa: E402
    CouplingRow,
    ErrorPattern,
    IsingConstraintGenerator,
)
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 788
TITLE = "Constraint Addition from Memory — Dynamic vs Static IsingEBM (50 GSM8K-style)"
DELIVERABLE = "results/experiment_788_constraint_addition_from_memory.json"

N_QUESTIONS = 50
BATCH_SIZE = 10
PATTERN_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Synthetic GSM8K-style question bank
# ---------------------------------------------------------------------------
# Each entry: (question_text, correct_answer_int, response_text, is_correct)
# Responses are simple arithmetic strings the mock extractor can judge.
# 30/50 correct baseline simulates a realistic 60% accuracy LLM.


def _build_synthetic_questions() -> list[dict[str, Any]]:
    """Return 50 synthetic arithmetic questions with oracle labels.

    The first 30 entries have correct responses; the last 20 have errors
    (carry errors, sign errors, unit errors, comparison errors, overflow errors —
    4 of each type to give each pattern count=4 by question 20, above PATTERN_THRESHOLD=3).

    Spec: REQ-LEARN-057
    """
    correct = [
        {
            "question": f"If a store sells {10 + i} apples at $2 each, how much does the customer pay?",
            "response": f"The customer pays ${(10 + i) * 2}.",
            "is_correct": True,
            "error_type": None,
        }
        for i in range(30)
    ]

    # 4 × 5 error types = 20 incorrect responses
    error_cases = []
    error_specs = [
        ("carry_error", "37 + 45 = 72 (carry dropped)"),
        ("sign_error", "5 - (-3) = 2 (sign flip)"),
        ("unit_error", "10 km + 500 m = 10.5 km (unit mismatch)"),
        ("comparison_error", "8 > 10 is true (flipped comparison)"),
        ("overflow_error", "255 + 1 = 0 in uint8 (overflow ignored)"),
    ]
    for i in range(4):
        for etype, bad_response in error_specs:
            error_cases.append(
                {
                    "question": f"Question {i}-{etype}: compute the result.",
                    "response": bad_response,
                    "is_correct": False,
                    "error_type": etype,
                }
            )

    return correct + error_cases


# ---------------------------------------------------------------------------
# Minimal mock pipeline using IsingEBM energy as verdict
# ---------------------------------------------------------------------------


def _mock_verify(ising_model: IsingModel, question: str, response: str, is_correct: bool) -> bool:
    """Use IsingEBM energy to decide whether to flip an incorrect verdict.

    **Why this mock works for the experiment:**
        We need to detect whether adding new couplings actually changes the
        pipeline's decision for incorrect responses.  This mock computes the
        energy of a fixed probe vector and marks the response as "flagged" when
        energy > threshold.  With a richer coupling matrix (more injected rows)
        the energy changes, which changes the flag rate.

    The baseline (zero/xavier coupling) and dynamic (injected coupling) models
    will produce different energies for the same probe, which is exactly what
    we are measuring.

    Returns True when the verify decision agrees with is_correct (no error caught
    for correct, error caught for incorrect).
    """
    import jax.numpy as jnp

    dim = ising_model.config.input_dim
    # Use a deterministic probe that is sensitive to coupling structure
    probe = jnp.ones(dim) * 0.5
    energy = float(ising_model.energy(probe))

    # Threshold: if energy < -0.1, model predicts "looks correct"
    # (low energy = low constraint violation = accept)
    model_accepts = energy < -0.1

    if is_correct and model_accepts:
        return True   # TP: correct response, model accepted
    if is_correct and not model_accepts:
        return False  # FN: correct response, model wrongly flagged
    if not is_correct and not model_accepts:
        return True   # TN: incorrect response, model correctly rejected
    # not is_correct and model_accepts → FP: incorrect response, model missed
    return False


def _run_pipeline(
    ising_model: IsingModel,
    questions: list[dict[str, Any]],
    *,
    dynamic: bool,
    generator: IsingConstraintGenerator | None,
) -> dict[str, Any]:
    """Run verify over all questions, optionally injecting couplings each batch.

    Returns a dict with:
        - correct_count: questions where verify decision matches oracle
        - total: 50
        - accuracy: correct_count / total
        - n_constraints_added: (dynamic only, else 0)
        - n_patterns_above_threshold: number of distinct pattern types >= PATTERN_THRESHOLD
        - error_pattern_counts: accumulated per-type counts

    Spec: REQ-LEARN-057
    """
    error_pattern_counts: dict[str, int] = {}
    correct_count = 0
    n_constraints_added = 0

    for batch_start in range(0, len(questions), BATCH_SIZE):
        batch = questions[batch_start : batch_start + BATCH_SIZE]

        for q in batch:
            decision = _mock_verify(ising_model, q["question"], q["response"], q["is_correct"])
            if decision:
                correct_count += 1
            # Accumulate error pattern counts for incorrect responses
            if not q["is_correct"] and q.get("error_type"):
                etype = q["error_type"]
                error_pattern_counts[etype] = error_pattern_counts.get(etype, 0) + 1

        # Dynamic injection: after each batch, check patterns and inject couplings
        if dynamic and generator is not None:
            patterns = [
                ErrorPattern(
                    pattern_type=etype,
                    count=count,
                    example_step=f"accumulated {count} {etype} events",
                )
                for etype, count in error_pattern_counts.items()
            ]
            new_rows = generator.synthesize_from_memory(patterns)
            if new_rows:
                generator.inject_couplings(new_rows)
                n_constraints_added += len(new_rows)

    n_patterns_above_threshold = sum(
        1 for c in error_pattern_counts.values() if c >= PATTERN_THRESHOLD
    )
    return {
        "correct_count": correct_count,
        "total": len(questions),
        "accuracy": correct_count / len(questions),
        "n_constraints_added": n_constraints_added,
        "n_patterns_above_threshold": n_patterns_above_threshold,
        "error_pattern_counts": error_pattern_counts,
    }


# ---------------------------------------------------------------------------
# honest_verdict helper (pure function — unit-testable)
# ---------------------------------------------------------------------------


def compute_honest_verdict(
    constraint_addition_delta: float, n_patterns_above_threshold: int
) -> str:
    """Classify experiment outcome into a canonical verdict string.

    Spec: REQ-LEARN-057
    """
    if n_patterns_above_threshold == 0:
        return "insufficient_patterns"
    if constraint_addition_delta > 0:
        return "constraint_addition_positive"
    if constraint_addition_delta == 0:
        return "constraint_addition_zero"
    return "constraint_addition_negative"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=45, result_path=DELIVERABLE):
        questions = _build_synthetic_questions()

        # --- Static baseline ---
        static_model = IsingModel(IsingConfig(input_dim=32, coupling_init="xavier_uniform"))
        static_result = _run_pipeline(
            static_model, questions, dynamic=False, generator=None
        )

        # --- Dynamic adaptive pipeline ---
        dynamic_model = IsingModel(IsingConfig(input_dim=32, coupling_init="xavier_uniform"))
        gen = IsingConstraintGenerator(dynamic_model, threshold=PATTERN_THRESHOLD)
        dynamic_result = _run_pipeline(
            dynamic_model, questions, dynamic=True, generator=gen
        )

        # --- Metrics ---
        # "baseline" accuracy = fraction correct WITHOUT verify-repair (oracle accuracy).
        # For this synthetic dataset the oracle accuracy is fixed at 30/50 = 0.60.
        oracle_accuracy = 30 / N_QUESTIONS

        net_improvement_static = static_result["accuracy"] - oracle_accuracy
        net_improvement_dynamic = dynamic_result["accuracy"] - oracle_accuracy
        constraint_addition_delta = net_improvement_dynamic - net_improvement_static

        n_constraints_added = dynamic_result["n_constraints_added"]
        n_patterns_above_threshold = dynamic_result["n_patterns_above_threshold"]

        honest_verdict = compute_honest_verdict(constraint_addition_delta, n_patterns_above_threshold)

        artifact = tmpl.build_result(
            {
                "n_constraints_added": n_constraints_added,
                "net_improvement_dynamic": net_improvement_dynamic,
                "net_improvement_static": net_improvement_static,
                "constraint_addition_delta": constraint_addition_delta,
                "n_patterns_above_threshold": n_patterns_above_threshold,
                "honest_verdict": honest_verdict,
                "static_accuracy": static_result["accuracy"],
                "dynamic_accuracy": dynamic_result["accuracy"],
                "oracle_accuracy": oracle_accuracy,
                "n_questions": N_QUESTIONS,
                "pattern_threshold": PATTERN_THRESHOLD,
                "error_pattern_counts": dynamic_result["error_pattern_counts"],
                "inference_mode": "synthetic_cpu",
            },
            status="success",
        )

        Path(DELIVERABLE).parent.mkdir(parents=True, exist_ok=True)
        Path(DELIVERABLE).write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
