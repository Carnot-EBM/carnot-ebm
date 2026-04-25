#!/usr/bin/env python3
"""Exp 836 — Constraint Accumulation Fix v3: Write-Path Validation on 30 GSM8K x 3 Sessions.

**Researcher summary (RETRO-CONSTRAINT-ZERO-DELTA):**
    Exp 833 confirmed root_cause=write_path_missing (H1): VerifyRepairPipeline.verify()
    detects violations but never calls EmbeddingConstraintStore.store().  As a result,
    the store remains empty across all sessions, retrieved constraints are always an
    empty list, and energy/precision can never improve across sessions.

    This experiment applies the write-path fix (enable_constraint_accumulation=True)
    and validates it on 30 GSM8K-style questions over 3 sessions:
      - Session 1: empty store; violations detected and written
      - Session 2: store has entries from session 1; retrieved constraints injected
      - Session 3: store has entries from sessions 1+2; largest retrieval pool

    Measurements:
      - n_constraints_written_s1/s2/s3: how many SPO tuples were written per session
      - precision_s1/s2/s3: fraction of known-incorrect responses detected as violated
      - delta_overall: max(precision_s1, precision_s2, precision_s3) - precision_s1

**honest_verdict logic:**
    - "constraint_accumulation_fixed"  if delta_overall > 0
    - "write_path_fixed_no_delta"      if n_constraints_written > 0 but delta_overall <= 0
    - "still_delta_zero"               if n_constraints_written == 0 (fix incomplete)
    - "blocked_no_diagnosis"           if gated by Exp 833

**Gate:**
    Reads results/experiment_833_constraint_delta_root_cause.json first.
    If honest_verdict in ["pipeline_wiring_correct", "diagnosis_inconclusive"],
    writes a blocked artifact and exits.

Spec: REQ-LEARN-048, REQ-LEARN-049, SCENARIO-LEARN-060, SCENARIO-LEARN-836-001
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# apply_env_autofix must run before any JAX/GPU imports to avoid device conflicts.
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 836
TITLE = "Constraint Accumulation Fix v3 — Write-Path Validation on 30 GSM8K x 3 Sessions"
DELIVERABLE = "results/experiment_836_constraint_accumulation_fix_v3.json"
TIMEOUT_MINUTES = 60
GATE_PATH = Path(_REPO / "results/experiment_833_constraint_delta_root_cause.json")
BLOCKED_VERDICTS = {"pipeline_wiring_correct", "diagnosis_inconclusive"}

# ---------------------------------------------------------------------------
# 30 GSM8K-style questions: 15 incorrect, 15 correct, interleaved.
# ground_truth_correct[i] == False means the response contains an arithmetic error.
# These are constructed so the pipeline's arithmetic extractor can catch violations.
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str, bool]] = [
    # (question, response, is_correct)
    ("What is 13 + 29?", "Step 1: 13 + 29 = 41. The answer is 41.", False),      # wrong (42)
    ("What is 50 - 17?", "Step 1: 50 - 17 = 33. The answer is 33.", True),
    ("What is 8 * 7?", "Step 1: 8 * 7 = 54. The answer is 54.", False),           # wrong (56)
    ("What is 100 / 4?", "Step 1: 100 / 4 = 25. The answer is 25.", True),
    ("What is 2^5?", "Step 1: 2^5 = 2*2*2*2*2 = 30. The answer is 30.", False),  # wrong (32)
    ("What is 99 - 44?", "Step 1: 99 - 44 = 55. The answer is 55.", True),
    ("What is 6 + 37?", "Step 1: 6 + 37 = 42. The answer is 42.", False),         # wrong (43)
    ("What is 9 * 8?", "Step 1: 9 * 8 = 72. The answer is 72.", True),
    ("What is 144 / 12?", "Step 1: 144 / 12 = 13. The answer is 13.", False),     # wrong (12)
    ("What is 17 + 58?", "Step 1: 17 + 58 = 75. The answer is 75.", True),
    ("What is 3 * 14?", "Step 1: 3 * 14 = 41. The answer is 41.", False),         # wrong (42)
    ("What is 81 / 9?", "Step 1: 81 / 9 = 9. The answer is 9.", True),
    ("What is 47 + 28?", "Step 1: 47 + 28 = 76. The answer is 76.", False),       # wrong (75)
    ("What is 200 - 63?", "Step 1: 200 - 63 = 137. The answer is 137.", True),
    ("What is 5 * 13?", "Step 1: 5 * 13 = 64. The answer is 64.", False),         # wrong (65)
    ("What is 48 / 6?", "Step 1: 48 / 6 = 8. The answer is 8.", True),
    ("What is 19 + 46?", "Step 1: 19 + 46 = 64. The answer is 64.", False),       # wrong (65)
    ("What is 7 * 9?", "Step 1: 7 * 9 = 63. The answer is 63.", True),
    ("What is 121 / 11?", "Step 1: 121 / 11 = 10. The answer is 10.", False),     # wrong (11)
    ("What is 34 + 19?", "Step 1: 34 + 19 = 53. The answer is 53.", True),
    ("What is 6 * 9?", "Step 1: 6 * 9 = 42. The answer is 42.", False),           # wrong (54)
    ("What is 90 - 27?", "Step 1: 90 - 27 = 63. The answer is 63.", True),
    ("What is 11 * 8?", "Step 1: 11 * 8 = 87. The answer is 87.", False),         # wrong (88)
    ("What is 56 / 7?", "Step 1: 56 / 7 = 8. The answer is 8.", True),
    ("What is 23 + 48?", "Step 1: 23 + 48 = 70. The answer is 70.", False),       # wrong (71)
    ("What is 4 * 16?", "Step 1: 4 * 16 = 64. The answer is 64.", True),
    ("What is 72 / 8?", "Step 1: 72 / 8 = 8. The answer is 8.", False),           # wrong (9)
    ("What is 55 - 18?", "Step 1: 55 - 18 = 37. The answer is 37.", True),
    ("What is 15% of 200?", "Step 1: 0.15 * 200 = 30. The answer is 30.", True),
    ("What is 3^3?", "Step 1: 3 * 3 * 3 = 28. The answer is 28.", False),         # wrong (27)
]

N_QUESTIONS = len(TEST_CASES)
N_SESSIONS = 3
KNOWN_INCORRECT_INDICES = [i for i, (_, _, ok) in enumerate(TEST_CASES) if not ok]


def _count_write_calls(store: Any) -> dict[str, int]:
    """Wrap store.store() to count how many SPO tuples are written in one session.

    Returns a counter dict with a 'n_writes' key.  The wrapper is applied fresh
    each session so counts are per-session, not cumulative.

    Why not subclass:
        We want the real EmbeddingConstraintStore instance across sessions so
        accumulated entries carry over.  Instance-level patching is the minimal
        non-invasive approach (same technique as Exp 833).

    Args:
        store: EmbeddingConstraintStore instance to instrument.

    Returns:
        Dict {'n_writes': 0} — will be mutated in-place by the wrapper.
    """
    counter: dict[str, int] = {"n_writes": 0}
    original_store_fn = store.store

    def _counted(spo: Any) -> None:
        counter["n_writes"] += 1
        original_store_fn(spo)

    store.store = _counted
    return counter


def _restore_store_method(store: Any, original_fn: Any) -> None:
    """Restore the original store.store() method after per-session counting.

    Called at the end of each session to reset the shim for the next session.
    The original_fn must be captured before calling _count_write_calls.

    Args:
        store: Same EmbeddingConstraintStore instance that was instrumented.
        original_fn: The unpatched store() method captured before patching.
    """
    store.store = original_fn


def _run_session(
    pipeline: Any,
    store: Any,
    test_cases: list[tuple[str, str, bool]],
) -> dict[str, Any]:
    """Run one verification session over all test cases and return session metrics.

    Wraps store.store() with a counter for this session only, then runs each
    (question, response) pair through pipeline.verify().  Precision is computed
    as the fraction of known-incorrect responses that are detected as violated.

    A "violation detected" means result.violations is non-empty.  We do not
    require perfect constraint_type matching — any violation on a known-incorrect
    response counts as a correct detection.

    Args:
        pipeline: VerifyRepairPipeline with enable_constraint_accumulation=True.
        store: EmbeddingConstraintStore shared across sessions (carries over entries).
        test_cases: List of (question, response, is_correct) tuples.

    Returns:
        Dict with n_constraints_written, precision, n_detected_incorrect,
        n_known_incorrect, and per-question results list.
    """
    from carnot.pipeline.ising_constraint_injector import IsingConstraintInjector

    injector = IsingConstraintInjector(embedding_dim=384, n_spins=64)

    # Capture original method before patching so we can restore it.
    original_store_fn = store.store
    counter = _count_write_calls(store)

    results = []
    n_detected_incorrect = 0
    known_incorrect = [i for i, (_, _, ok) in enumerate(test_cases) if not ok]

    for idx, (question, response, is_correct) in enumerate(test_cases):
        result = pipeline.verify(
            question=question,
            response=response,
            domain="arithmetic",
            embedding_constraint_store=store,
            ising_constraint_injector=injector,
        )
        violation_detected = len(result.violations) > 0
        if not is_correct and violation_detected:
            n_detected_incorrect += 1
        results.append({
            "idx": idx,
            "question": question[:60],
            "is_correct": is_correct,
            "violation_detected": violation_detected,
            "n_violations": len(result.violations),
        })

    # Restore original store.store() so next session gets a fresh counter.
    store.store = original_store_fn

    n_known_incorrect = len(known_incorrect)
    precision = n_detected_incorrect / n_known_incorrect if n_known_incorrect > 0 else 0.0

    return {
        "n_constraints_written": counter["n_writes"],
        "precision": precision,
        "n_detected_incorrect": n_detected_incorrect,
        "n_known_incorrect": n_known_incorrect,
        "per_question": results,
    }


def compute_honest_verdict(
    n_written_total: int,
    delta_overall: float,
) -> str:
    """Map session metrics to an honest verdict string.

    Decision hierarchy:
        1. still_delta_zero          — no constraints written at all (fix incomplete)
        2. constraint_accumulation_fixed — delta_overall > 0 (precision improved)
        3. write_path_fixed_no_delta — constraints written but no precision gain

    Args:
        n_written_total: Total SPO tuples written across all sessions.
        delta_overall: max(precision_s1..s3) - precision_s1.

    Returns:
        Verdict string — one of the three labels above.

    Spec: REQ-LEARN-048, SCENARIO-LEARN-836-001
    """
    if n_written_total == 0:
        return "still_delta_zero"
    if delta_overall > 0:
        return "constraint_accumulation_fixed"
    return "write_path_fixed_no_delta"


def run_accumulation_experiment(
    test_cases: list[tuple[str, str, bool]] | None = None,
) -> dict[str, Any]:
    """Run the 3-session accumulation experiment and return the result dict.

    Separated from main() so unit tests can call it directly without triggering
    the ExperimentTemplate lifecycle.

    Args:
        test_cases: Optional override for the test dataset.  Defaults to TEST_CASES.

    Returns:
        Dict with all measured fields required by the deliverable schema.

    Spec: REQ-LEARN-048, REQ-LEARN-049, SCENARIO-LEARN-060, SCENARIO-LEARN-836-001
    """
    from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    if test_cases is None:
        test_cases = TEST_CASES

    # One store shared across all sessions so accumulated entries carry over.
    store = EmbeddingConstraintStore()

    # Pipeline with write path enabled (the fix being validated).
    pipeline = VerifyRepairPipeline(
        model=None,
        domains=["arithmetic"],
        max_repairs=0,
        timeout_seconds=60,
        enable_constraint_accumulation=True,
    )

    s1 = _run_session(pipeline, store, test_cases)
    s2 = _run_session(pipeline, store, test_cases)
    s3 = _run_session(pipeline, store, test_cases)

    precisions = [s1["precision"], s2["precision"], s3["precision"]]
    delta_overall = max(precisions) - s1["precision"]
    n_written_total = s1["n_constraints_written"] + s2["n_constraints_written"] + s3["n_constraints_written"]

    honest_verdict = compute_honest_verdict(n_written_total, delta_overall)

    return {
        "precision_s1": s1["precision"],
        "precision_s2": s2["precision"],
        "precision_s3": s3["precision"],
        "n_constraints_written_s1": s1["n_constraints_written"],
        "n_constraints_written_s2": s2["n_constraints_written"],
        "n_constraints_written_s3": s3["n_constraints_written"],
        "n_constraints_in_store_after_s3": len(store._store),
        "delta_overall": delta_overall,
        "honest_verdict": honest_verdict,
        "embedding_mode": store.embedding_mode,
        "n_questions": len(test_cases),
        "n_sessions": N_SESSIONS,
        "session_1": s1,
        "session_2": s2,
        "session_3": s3,
    }


def main() -> None:
    """Run Exp 836 with ExperimentTemplate lifecycle and write deliverable JSON."""
    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES)
    watchdog.start()

    output_path = Path(_REPO / DELIVERABLE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Gate check: if Exp 833 found no actionable root cause, write blocked artifact.
        if GATE_PATH.exists():
            gate_data = json.loads(GATE_PATH.read_text())
            gate_verdict = gate_data.get("honest_verdict", "")
            if gate_verdict in BLOCKED_VERDICTS:
                blocked = tmpl.build_result(
                    {
                        "honest_verdict": "blocked_no_diagnosis",
                        "gate": "exp833_no_root_cause",
                        "gate_verdict": gate_verdict,
                        "blocked": True,
                    },
                    status="blocked",
                    honest_verdict="blocked_no_diagnosis",
                )
                output_path.write_text(json.dumps(blocked, indent=2))
                watchdog.stop()
                tmpl.assert_deliverable_written()
                return

        result = run_accumulation_experiment()

        artifact = tmpl.build_result(
            result,
            status="success",
            honest_verdict=result["honest_verdict"],
        )
        output_path.write_text(json.dumps(artifact, indent=2))

    finally:
        watchdog.stop()

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
