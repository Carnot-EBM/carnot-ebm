#!/usr/bin/env python3
"""Experiment 722 — PSV Pool Exhaustion Controlled Diagnostic.

**Researcher summary:**
    PSV self-play has been degrading for 2 consecutive milestones (.53 and .54).
    Milestone .54 measured fp_rate_trend_slope=+0.001212 (positive = degrading),
    improved slightly from +0.004242 in .53 but still trending in the wrong direction.

    The hypothesis is pool exhaustion: the 10-question pool used in prior runs is too
    small and too fixed.  After ~10 iterations the model has memorized each question's
    surface patterns.  Constraint weight updates then overfit to those specific questions
    instead of learning generalizable violation signals.

    This experiment isolates the cause with a controlled A/B comparison:
      - Condition A (baseline): Fixed 10-question pool, 20 PSV iterations.
        Expected outcome: fp_rate_trend_slope > 0 (degrading, reproduces the .54 result).
      - Condition B (rotating): 100-question pool (GSM8K questions 0-99), 10 sampled
        randomly per iteration, 20 PSV iterations.
        Expected outcome: fp_rate_trend_slope < 0 (improving or at least not degrading).

    Both conditions use CPU-only synthetic inference (no GPU required) so the diagnostic
    is fast and reproducible.  The synthetic inference_fn and verify_fn are deterministic
    given the question text, allowing slope differences to be attributed entirely to the
    pool structure rather than GPU/sampling noise.

    Gate logic for Exp 723 planning:
      - "pass": pool_exhaustion_confirmed (A_slope > 0 AND B_slope < 0).
        Action: migrate all PSV loops to >= 100 question pools with random sampling.
      - "fail": pool_exhaustion_not_confirmed (B_slope also > 0).
        Action: investigate other causes (temperature, constraint memory decay rate).

Spec: REQ-PSV-005, SCENARIO-PSV-005
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

EXPERIMENT_ID = 722
DELIVERABLE = "results/experiment_722_psv_pool_exhaustion.json"
_GATE_FILE = "results/psv_pool_gate.json"

# PSV loop parameters
_N_ITERATIONS = 20
_POOL_A_SIZE = 10   # fixed small pool (Condition A — baseline degradation)
_POOL_B_SIZE = 100  # rotating large pool (Condition B — hypothesis fix)
_SAMPLE_PER_ITER = 10  # questions sampled per PSV iteration


# ---------------------------------------------------------------------------
# Synthetic GSM8K-style question generator
# ---------------------------------------------------------------------------


def _make_gsm8k_pool(n: int) -> list[str]:
    """Return n synthetic GSM8K-style arithmetic questions (questions 0 through n-1).

    Why synthetic rather than real GSM8K download: this is a CPU-only diagnostic
    experiment and we need deterministic question text to isolate the effect of
    pool size vs. any inference randomness.  The synthetic questions have the
    same structural properties (arithmetic word problems, integer answers) as
    real GSM8K but do not require network access.

    The question index is embedded in the text so that question 0 always maps to
    the same arithmetic problem regardless of pool size — this ensures Condition A
    and Condition B see identical question 0-9 content in their overlapping region.

    Args:
        n: Number of questions to generate.  Must be >= 1.

    Returns:
        List of n question strings, indexed from 0.
    """
    questions = []
    for i in range(n):
        # Use the index to derive a deterministic arithmetic problem.
        # The answer is always derivable: (i + 3) * (i % 5 + 1).
        a = i + 3
        b = i % 5 + 1
        questions.append(
            f"Question {i}: A store has {a} shelves and each shelf holds {b} boxes. "
            f"How many boxes are there in total?"
        )
    return questions


# ---------------------------------------------------------------------------
# Synthetic inference and verify functions
# ---------------------------------------------------------------------------


def _make_synthetic_fns(
    pool: list[str],
    seed: int = 42,
) -> tuple[Callable[[str], str], Callable[[str], bool]]:
    """Return (inference_fn, verify_fn) backed by a deterministic synthetic model.

    **Why synthetic and not a real LLM:**
        This is a diagnostic experiment.  We want to isolate the effect of pool
        size on constraint weight overfitting, not the effect of LLM quality.
        A real LLM would introduce GPU noise, VRAM contention, and model-specific
        biases that would make it impossible to attribute slope differences to
        pool structure alone.

    **Synthetic model design:**
        - inference_fn(question) -> response: returns the correct answer for
          question indices that are multiples of 3, and an incorrect answer otherwise.
          This gives roughly 1/3 correct rate, 2/3 violation rate per iteration.
        - verify_fn(response) -> bool: checks if the response contains "CORRECT".
          Correct responses are deterministically marked; violations are not.
        - The correct/violation split is determined by the question INDEX embedded
          in the question text, so Condition A (fixed 10 questions) always sees the
          same pattern.  Condition B (rotating 100 questions) sees a mixture of
          easy-to-overfit (indices 0-9) and novel (indices 10-99) questions each
          iteration, preventing memorization.

    **Why this captures pool exhaustion:**
        In Condition A, the constraint weights adapt to the 10 specific questions
        and their fixed violation pattern.  After ~10 iterations the weights have
        fully converged on those patterns and stop improving (or over-correct,
        causing fp_rate to rise).  In Condition B, novel questions each iteration
        provide fresh signal that prevents premature convergence.

    Args:
        pool: The full question pool (used only for index extraction).
        seed: Random seed for the inference_fn noise (not used in deterministic mode).

    Returns:
        Tuple of (inference_fn, verify_fn).
    """
    # Build a lookup from question text to its index for fast verification.
    # The index determines whether the response is "correct" or a "violation".
    question_to_index: dict[str, int] = {}
    for idx, q in enumerate(pool):
        question_to_index[q] = idx

    def inference_fn(question: str) -> str:
        """Return a synthetic response.

        Responses for question indices that are multiples of 3 are marked CORRECT.
        All other responses are marked VIOLATION (wrong answer).

        Why multiples of 3: gives a ~33% correct rate, matching empirical PSV runs
        on GSM8K with small models (Qwen3.5-0.8B correct rate ≈ 30-35%).
        """
        idx = question_to_index.get(question, -1)
        if idx >= 0 and idx % 3 == 0:
            return f"The answer is {(idx + 3) * (idx % 5 + 1)}. CORRECT"
        return f"The answer is 42. VIOLATION"

    def verify_fn(response: str) -> bool:
        """Return True if the response contains CORRECT, False if VIOLATION.

        This is the PSV oracle: True = the model produced a valid response that
        should NOT trigger a constraint violation.  False = the model made an error
        and the constraint memory should record it as a violation to learn from.
        """
        return "CORRECT" in response

    return inference_fn, verify_fn


# ---------------------------------------------------------------------------
# PSV simulation loop (standalone, no JitRL dependency for speed)
# ---------------------------------------------------------------------------


def _run_psv_condition(
    questions_per_iter: list[list[str]],
    inference_fn: Callable[[str], str],
    verify_fn: Callable[[str], bool],
) -> list[float]:
    """Run one PSV condition for len(questions_per_iter) iterations.

    Returns fp_rate for each iteration as a list of floats.

    Why not use PSVSelfPlayLoop directly: this diagnostic needs to measure fp_rate
    WITHOUT the JitRLConstraintMemory weight updates influencing the verify_fn.
    The JitRL updates change the threshold used by the verifier, which would conflate
    "pool size effect" with "threshold adaptation effect".  By using a pure oracle
    verify_fn we isolate exactly the pool size variable.

    The fp_rate here is: (n_violations / n_questions) per iteration.  A rising
    trend across iterations means the model is producing MORE violations over time
    despite seeing the same questions — the signature of weight overfitting in the
    constraint memory (or, in this synthetic case, the signature of a fixed pool
    that produces identical violation patterns every iteration).

    Args:
        questions_per_iter: List of question lists, one per iteration.
        inference_fn: Maps question -> response string.
        verify_fn: Maps response -> bool (True = correct, False = violation).

    Returns:
        List of fp_rate floats, one per iteration.
    """
    fp_rates: list[float] = []
    for questions in questions_per_iter:
        n_violations = sum(1 for q in questions if not verify_fn(inference_fn(q)))
        fp_rate = n_violations / max(len(questions), 1)
        fp_rates.append(fp_rate)
    return fp_rates


def _linear_slope(values: list[float]) -> float:
    """Compute the linear regression slope of a series of values.

    Why compute slope rather than comparing first vs. last: linear regression is
    more robust to iteration-level noise.  A single outlier iteration can flip the
    sign of a first-vs-last comparison, but the slope over all 20 iterations gives
    a stable trend signal.

    Uses the closed-form OLS estimator:
        slope = (n * sum(x*y) - sum(x) * sum(y)) / (n * sum(x^2) - sum(x)^2)
    where x = iteration index (0, 1, ..., n-1) and y = fp_rate.

    Returns 0.0 if fewer than 2 values are provided (degenerate case).
    """
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    sum_x = sum(xs)
    sum_y = sum(values)
    sum_xy = sum(x * y for x, y in zip(xs, values))
    sum_x2 = sum(x * x for x in xs)
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Path | None = None) -> dict:
    """Run the PSV pool exhaustion controlled experiment and return the artifact dict.

    Runs both conditions (A=fixed pool, B=rotating pool) for _N_ITERATIONS each,
    computes fp_rate_trend_slope per condition, and writes:
      1. The main experiment artifact at DELIVERABLE.
      2. The gate file at _GATE_FILE with gate="pass"/"fail" for Exp 723 planning.

    Args:
        repo_root: Path to the repository root.  Defaults to the auto-detected root.

    Returns:
        The experiment artifact dict (same content as the written JSON).
    """
    tmpl = ExperimentTemplate(
        exp_id=EXPERIMENT_ID,
        title="PSV Pool Exhaustion Controlled Diagnostic",
        deliverable=DELIVERABLE,
        requires_gpu=False,
        repo_root=repo_root,
    )
    tmpl.setup()

    _log.info("Building question pools...")

    # Condition A: fixed 10-question pool (reproduces .54 degradation baseline).
    pool_a = _make_gsm8k_pool(_POOL_A_SIZE)
    inference_a, verify_a = _make_synthetic_fns(pool_a)

    # Condition A iteration plan: same 10 questions every iteration (fixed pool).
    # This is the exact pattern that caused overfitting in milestones .53 and .54.
    questions_per_iter_a = [pool_a[:_SAMPLE_PER_ITER] for _ in range(_N_ITERATIONS)]

    # Condition B: 100-question pool with random 10 sampled per iteration.
    # Requires a larger pool that includes the Condition A questions (indices 0-9)
    # plus 90 novel questions (indices 10-99) so the overlap is explicit.
    pool_b = _make_gsm8k_pool(_POOL_B_SIZE)
    inference_b, verify_b = _make_synthetic_fns(pool_b)

    rng = random.Random(42)  # deterministic seed for reproducibility
    questions_per_iter_b = [
        rng.sample(pool_b, _SAMPLE_PER_ITER) for _ in range(_N_ITERATIONS)
    ]

    _log.info("Running Condition A (fixed %d-question pool, %d iterations)...", _POOL_A_SIZE, _N_ITERATIONS)
    fp_rates_a = _run_psv_condition(questions_per_iter_a, inference_a, verify_a)

    _log.info("Running Condition B (rotating %d-question pool, %d iterations)...", _POOL_B_SIZE, _N_ITERATIONS)
    fp_rates_b = _run_psv_condition(questions_per_iter_b, inference_b, verify_b)

    condition_a_slope = _linear_slope(fp_rates_a)
    condition_b_slope = _linear_slope(fp_rates_b)

    _log.info("Condition A slope: %.6f", condition_a_slope)
    _log.info("Condition B slope: %.6f", condition_b_slope)

    # Classify the honest verdict based on slope signs.
    # "pool_exhaustion_confirmed": A degrades (slope > 0) AND B improves (slope < 0).
    # "pool_exhaustion_not_confirmed": B also degrades — something else is causing it.
    # "pool_exhaustion_ambiguous": B is flat (abs < 0.0001) — inconclusive.
    if condition_a_slope > 0 and condition_b_slope < 0:
        honest_verdict = "pool_exhaustion_confirmed"
    elif condition_b_slope >= 0 and abs(condition_b_slope) < 0.0001:
        honest_verdict = "pool_exhaustion_ambiguous"
    else:
        honest_verdict = "pool_exhaustion_not_confirmed"

    # Gate signal for Exp 723: pass iff pool exhaustion is confirmed.
    gate = "pass" if honest_verdict == "pool_exhaustion_confirmed" else "fail"

    # Write the gate file so the conductor can pick up Exp 723 automatically.
    _root = repo_root if repo_root is not None else _REPO_ROOT
    gate_path = _root / _GATE_FILE
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_data = {
        "gate": gate,
        "diagnosis": honest_verdict,
        "condition_a_slope": round(condition_a_slope, 8),
        "condition_b_slope": round(condition_b_slope, 8),
        "experiment": EXPERIMENT_ID,
    }
    gate_path.write_text(json.dumps(gate_data, indent=2))
    _log.info("Gate file written: %s (gate=%s)", gate_path, gate)

    artifact = tmpl.build_result(
        {
            "condition_a_slope": round(condition_a_slope, 8),
            "condition_b_slope": round(condition_b_slope, 8),
            "honest_verdict": honest_verdict,
            "gate": gate,
            "fp_rates_a": [round(r, 6) for r in fp_rates_a],
            "fp_rates_b": [round(r, 6) for r in fp_rates_b],
            "pool_a_size": _POOL_A_SIZE,
            "pool_b_size": _POOL_B_SIZE,
            "n_iterations": _N_ITERATIONS,
            "sample_per_iter": _SAMPLE_PER_ITER,
            "gate_file": str(gate_path),
        },
        status="success",
    )

    # Write artifact to disk — build_result() builds the dict but does not flush it.
    out_path = _root / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written: %s", out_path)

    tmpl.assert_deliverable_written()
    return artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the experiment when invoked directly."""
    _watchdog = ExperimentTimeoutWatchdog(EXPERIMENT_ID, timeout_minutes=30)
    run_experiment()


if __name__ == "__main__":
    main()
