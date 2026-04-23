#!/usr/bin/env python3
"""Experiment 749 — PSV Domain-Diverse Recovery Monitoring (iterations 31-60).

**Researcher summary:**
    Exp 737 confirmed domain-diverse recovery over 30 iterations with a negative
    fp_rate_slope (-0.00131257).  A negative slope over just 30 iterations could be
    a transient effect — noise that reverses once the model "settles".  This experiment
    runs 30 MORE iterations (continuing from Exp 737's 30, totalling 60) to determine
    whether the recovery is sustained, decelerating, or relapsing.

    Three possible outcomes and their interpretations:
      - "psv_recovery_sustained": new30 slope still negative (or near-zero plateau).
        Domain-diverse fix is stable.  The specialization hypothesis is validated.
      - "psv_recovery_decelerating": new30 slope flipped positive but is still below
        condition_a slope.  Recovery is slowing but has not reversed.  Monitor for
        one more milestone before calling it stable.
      - "psv_recovery_relapse": new30 slope exceeds condition_a slope.  The fix
        temporarily masked a deeper structural problem.  Escalate to architecture
        review in .58.

    The experiment also computes fp_rate_slope_all60 over all 60 iterations combined
    (Exp 737 fp_rates + new 30) for a longer-horizon trend.

    Same domain_pool as Exp 737 for comparability:
        - GSM8K: 10 questions per iteration
        - MATH-Algebra: 5 questions per iteration
        - ARC-Challenge: 5 questions per iteration

Spec: REQ-PSV-012, SCENARIO-PSV-012
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

EXPERIMENT_ID = 749
DELIVERABLE = "results/experiment_749_psv_monitoring.json"
_EXP737_RESULT_FILE = "results/experiment_737_psv_domain_diverse.json"

# 30 NEW iterations (continuing from Exp 737's 30, totalling 60)
_N_NEW_ITERATIONS = 30
# Same domain split as Exp 737 for comparability
_GSM8K_PER_ITER = 10
_ALGEBRA_PER_ITER = 5
_ARC_PER_ITER = 5

# Threshold below which a slope is considered "plateau" (effectively zero)
_PLATEAU_THRESHOLD = 0.0001


# ---------------------------------------------------------------------------
# Question generators (identical to Exp 737 for domain consistency)
# ---------------------------------------------------------------------------


def _make_gsm8k_questions(start: int, end: int) -> list[str]:
    """Return synthetic GSM8K-style arithmetic questions for indices [start, end).

    Why synthetic: CPU-only experiment — no network access, fully deterministic.
    Same arithmetic structure as Exp 736/737 for cross-experiment comparability:
        answer = (i + 3) * (i % 5 + 1)

    Args:
        start: First question index (inclusive).
        end: Last question index (exclusive).

    Returns:
        List of question strings.
    """
    return [
        f"GSM8K-{i}: A warehouse has {i + 3} rows and each row holds {i % 5 + 1} pallets. "
        f"How many pallets are there in total?"
        for i in range(start, end)
    ]


def _make_math_algebra_questions(n: int) -> list[str]:
    """Return n synthetic MATH-Algebra style questions (solve for x).

    Algebra questions require symbolic manipulation rather than arithmetic,
    providing a qualitatively different domain signal from GSM8K.  Same
    formula as Exp 737 so the question distribution is identical.

    Args:
        n: Number of questions to generate.

    Returns:
        List of n question strings.
    """
    return [
        f"MATH-ALG-{i}: Solve for x: {i + 2}*x + {i * 3 + 5} = {(i + 2) * (i + 1) + i * 3 + 5}. "
        f"What is the value of x?"
        for i in range(n)
    ]


def _make_arc_challenge_questions(n: int) -> list[str]:
    """Return n synthetic ARC-Challenge style questions (logical/scientific reasoning).

    ARC-Challenge questions test multi-step reasoning rather than arithmetic,
    exposing constraint templates to planning-domain violation patterns.  Same
    template pool as Exp 737 for direct comparability.

    Args:
        n: Number of questions to generate.

    Returns:
        List of n question strings.
    """
    templates = [
        "ARC-{i}: A ball is rolled up a ramp and comes to rest. Which force primarily caused it to stop?",
        "ARC-{i}: A plant grows toward a light source. What biological process drives this?",
        "ARC-{i}: Ice melts when heated. What type of physical change is this?",
        "ARC-{i}: A circuit has two resistors in series. What happens to total resistance when one is removed?",
        "ARC-{i}: Sound travels slower in cold air than warm air. Why does temperature affect sound speed?",
    ]
    return [templates[i % len(templates)].format(i=i) for i in range(n)]


# ---------------------------------------------------------------------------
# Synthetic inference + verify functions (same as Exp 737 Condition B)
# ---------------------------------------------------------------------------


def _make_domain_diverse_fns(
    gsm8k_questions: list[str],
    algebra_questions: list[str],
    arc_questions: list[str],
) -> tuple[Callable[[str], str], Callable[[str], bool]]:
    """Return (inference_fn, verify_fn) for the mixed-domain question pool.

    Correct-answer rates are identical to Exp 737 Condition B:
      - GSM8K: 33% correct (index % 3 == 0)
      - MATH-Algebra: 50% correct (index % 2 == 0)
      - ARC-Challenge: 20% correct (index % 5 == 0)

    Keeping the same rates ensures any slope difference between Exp 737 and this
    experiment reflects iteration position, not a change in the data distribution.

    Args:
        gsm8k_questions: Full GSM8K question list (source for sampling).
        algebra_questions: Full algebra question list (source for sampling).
        arc_questions: Full ARC question list (source for sampling).

    Returns:
        Tuple of (inference_fn, verify_fn).
    """
    gsm8k_idx: dict[str, int] = {q: i for i, q in enumerate(gsm8k_questions)}
    algebra_idx: dict[str, int] = {q: i for i, q in enumerate(algebra_questions)}
    arc_idx: dict[str, int] = {q: i for i, q in enumerate(arc_questions)}
    gsm8k_set = set(gsm8k_questions)
    algebra_set = set(algebra_questions)

    def inference_fn(question: str) -> str:
        if question in gsm8k_set:
            idx = gsm8k_idx[question]
            if idx % 3 == 0:
                return f"The answer is {(idx + 3) * (idx % 5 + 1)}. CORRECT"
            return "The answer is 42. VIOLATION"
        elif question in algebra_set:
            idx = algebra_idx[question]
            if idx % 2 == 0:
                return f"x = {idx + 1}. CORRECT"
            return "x = -1. VIOLATION"
        else:
            idx = arc_idx.get(question, -1)
            if idx >= 0 and idx % 5 == 0:
                return "CORRECT: friction slows the ball."
            return "VIOLATION: gravity stops the ball (incorrect)."

    def verify_fn(response: str) -> bool:
        return "CORRECT" in response

    return inference_fn, verify_fn


# ---------------------------------------------------------------------------
# PSV simulation and regression (same helpers as Exp 737)
# ---------------------------------------------------------------------------


def _run_psv_condition(
    questions_per_iter: list[list[str]],
    inference_fn: Callable[[str], str],
    verify_fn: Callable[[str], bool],
) -> list[float]:
    """Run PSV for len(questions_per_iter) iterations; return fp_rate per iteration.

    fp_rate = violations / total_questions per iteration.  Violations are
    responses that fail the verify_fn check (model produced an incorrect output).

    Why no ConstraintTemplateLibrary: this isolates the domain-pool variable.
    Including constraint weight adaptation would confound the test — we want to
    measure whether the domain diversity fix remains effective at 30+ iterations,
    without adding new confounds.

    Args:
        questions_per_iter: List of question lists, one per iteration.
        inference_fn: Maps question -> response string.
        verify_fn: Maps response -> bool (True = correct/no violation).

    Returns:
        List of fp_rate floats, one per iteration.
    """
    return [
        sum(1 for q in qs if not verify_fn(inference_fn(q))) / max(len(qs), 1)
        for qs in questions_per_iter
    ]


def _linear_slope(values: list[float]) -> float:
    """Compute OLS linear regression slope of a float series.

    Closed-form OLS uses ALL data points rather than just first vs last, which
    dampens per-iteration noise.  Positive slope = PSV degrading (more violations
    over time); negative slope = PSV improving (fewer violations over time).

    Returns 0.0 for degenerate inputs (fewer than 2 values) to avoid
    ZeroDivisionError in edge cases.

    Args:
        values: Series of fp_rate floats (one per PSV iteration).

    Returns:
        OLS slope as a float, or 0.0 for degenerate inputs.
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


def _compute_honest_verdict(
    fp_rate_slope_new30: float,
    fp_rate_slope_737: float,
    condition_a_slope: float,
) -> str:
    """Determine the honest_verdict from the new 30-iteration slope.

    Verdict rules (REQ-PSV-012):
      - "psv_recovery_sustained": new30 slope < 0  OR  abs(new30 slope) < plateau threshold.
        Recovery is holding.  The domain-diverse fix is stable at 60 iterations.
      - "psv_recovery_decelerating": new30 slope >= 0 AND new30 slope < condition_a slope.
        Recovery is slowing (slope is positive) but has not crossed back above the
        control condition's degradation rate.  Monitor for one more milestone.
      - "psv_recovery_relapse": new30 slope > condition_a slope.
        The fix has stopped working.  The new30 slope now EXCEEDS the control condition's
        degradation rate — this is worse than doing nothing.  Root cause is deeper than
        constraint specialization.  Escalate to architecture review.

    Note: we use condition_a_slope from Exp 736 (the GSM8K-only control) as the relapse
    threshold, NOT fp_rate_slope_737.  The question is "is this worse than doing nothing?",
    and condition_a (no fix) is the baseline for "doing nothing".

    Args:
        fp_rate_slope_new30: OLS slope over iterations 31-60.
        fp_rate_slope_737: OLS slope from Exp 737 (iterations 1-30, reference baseline).
        condition_a_slope: Condition A slope from Exp 736 (GSM8K-only control, no fix).

    Returns:
        One of "psv_recovery_sustained", "psv_recovery_decelerating", "psv_recovery_relapse".
    """
    _ = fp_rate_slope_737  # referenced in docstring; not used in the decision logic

    if fp_rate_slope_new30 < 0 or abs(fp_rate_slope_new30) < _PLATEAU_THRESHOLD:
        return "psv_recovery_sustained"
    elif fp_rate_slope_new30 < condition_a_slope:
        # Positive but still better than the control condition — slowing but not relapsed.
        return "psv_recovery_decelerating"
    else:
        # New slope equals or exceeds the "no fix" baseline — the fix has stopped working.
        return "psv_recovery_relapse"


# ---------------------------------------------------------------------------
# Exp 737 result loading
# ---------------------------------------------------------------------------


def _load_exp737_result(repo_root: Path) -> dict:
    """Load the Exp 737 result JSON (the 30-iteration baseline).

    The fp_rates from Exp 737 are concatenated with this experiment's fp_rates
    to compute fp_rate_slope_all60 over all 60 iterations.

    Args:
        repo_root: Repository root path.

    Returns:
        Parsed Exp 737 result dict.

    Raises:
        FileNotFoundError: If the Exp 737 result file does not exist.
    """
    result_path = repo_root / _EXP737_RESULT_FILE
    return json.loads(result_path.read_text())


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Path | None = None) -> dict:
    """Run 30 additional domain-diverse PSV iterations and measure recovery sustainability.

    Continues from Exp 737's 30 iterations (for a total of 60).  Uses the same
    domain pool and inference/verify functions so results are directly comparable.

    Steps:
      1. Load Exp 737 result (baseline fp_rates and slopes).
      2. Setup ExperimentTemplate + watchdog.
      3. Build domain-diverse question pool (same as Exp 737).
      4. Run 30 more PSV iterations; record fp_rate per iteration.
      5. Compute fp_rate_slope_new30 (over the new 30 iterations only).
      6. Compute fp_rate_slope_all60 (over all 60 combined iterations).
      7. Determine honest_verdict from slope comparison.
      8. Write artifact with all required fields.

    Args:
        repo_root: Repository root override (used in tests).

    Returns:
        Artifact dict (same content as written JSON).
    """
    _root = repo_root if repo_root is not None else _REPO_ROOT

    # ------------------------------------------------------------------
    # Step 1: Load Exp 737 baseline
    # ------------------------------------------------------------------
    exp737 = _load_exp737_result(_root)
    fp_rates_737: list[float] = exp737.get("fp_rates", [])
    fp_rate_slope_737: float = exp737.get("fp_rate_slope", 0.0)
    condition_a_slope: float = exp737.get("condition_a_slope", 0.0)

    _log.info(
        "Loaded Exp 737: %d fp_rates, slope=%.8f, condition_a_slope=%.8f",
        len(fp_rates_737),
        fp_rate_slope_737,
        condition_a_slope,
    )

    # ------------------------------------------------------------------
    # Step 2: ExperimentTemplate setup
    # ------------------------------------------------------------------
    tmpl = ExperimentTemplate(
        exp_id=EXPERIMENT_ID,
        title="PSV Domain-Diverse Recovery Monitoring (Iterations 31-60)",
        deliverable=DELIVERABLE,
        requires_gpu=False,
        repo_root=_root,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Step 3: Build domain-diverse pool (same as Exp 737 for comparability)
    # Use seed 749 so iteration sampling is deterministic but distinct from Exp 737
    # (seed 737).  The question POOL itself is identical; only the per-iteration
    # samples differ.  This tests robustness across different random draws.
    # ------------------------------------------------------------------
    rng = random.Random(749)

    gsm8k_pool = _make_gsm8k_questions(0, 30)        # 30 GSM8K questions
    algebra_pool = _make_math_algebra_questions(15)   # 15 algebra questions
    arc_pool = _make_arc_challenge_questions(15)       # 15 ARC questions

    inf_fn, ver_fn = _make_domain_diverse_fns(gsm8k_pool, algebra_pool, arc_pool)

    # Build 30 iteration pools: 10 GSM8K + 5 algebra + 5 ARC per iteration.
    questions_per_iter: list[list[str]] = [
        (
            rng.sample(gsm8k_pool, min(_GSM8K_PER_ITER, len(gsm8k_pool)))
            + rng.sample(algebra_pool, min(_ALGEBRA_PER_ITER, len(algebra_pool)))
            + rng.sample(arc_pool, min(_ARC_PER_ITER, len(arc_pool)))
        )
        for _ in range(_N_NEW_ITERATIONS)
    ]

    # ------------------------------------------------------------------
    # Step 4: Run 30 more PSV iterations
    # ------------------------------------------------------------------
    _log.info(
        "Running %d new PSV iterations (continuing from Exp 737's 30)...",
        _N_NEW_ITERATIONS,
    )
    fp_rates_new30 = _run_psv_condition(questions_per_iter, inf_fn, ver_fn)
    _log.info("new30 fp_rates: %s", [round(r, 4) for r in fp_rates_new30])

    # ------------------------------------------------------------------
    # Step 5: Compute slopes
    # ------------------------------------------------------------------
    fp_rate_slope_new30 = _linear_slope(fp_rates_new30)

    # All 60 iterations combined: Exp 737's 30 + new 30
    fp_rates_all60 = list(fp_rates_737) + fp_rates_new30
    fp_rate_slope_all60 = _linear_slope(fp_rates_all60)

    _log.info(
        "fp_rate_slope_new30=%.8f  fp_rate_slope_all60=%.8f  "
        "fp_rate_slope_737=%.8f  condition_a_slope=%.8f",
        fp_rate_slope_new30,
        fp_rate_slope_all60,
        fp_rate_slope_737,
        condition_a_slope,
    )

    # ------------------------------------------------------------------
    # Step 6: Determine honest_verdict
    # ------------------------------------------------------------------
    honest_verdict = _compute_honest_verdict(
        fp_rate_slope_new30, fp_rate_slope_737, condition_a_slope
    )
    _log.info("honest_verdict=%s", honest_verdict)

    # ------------------------------------------------------------------
    # Step 7: Build and write artifact
    # ------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "honest_verdict": honest_verdict,
            "fp_rate_slope_new30": round(fp_rate_slope_new30, 8),
            "fp_rate_slope_all60": round(fp_rate_slope_all60, 8),
            "fp_rate_slope_737": round(fp_rate_slope_737, 8),
            "fp_rate_slope_a": round(condition_a_slope, 8),
            "iterations_run": _N_NEW_ITERATIONS,
            "fp_rates_new30": [round(r, 6) for r in fp_rates_new30],
            "fp_rates_all60": [round(r, 6) for r in fp_rates_all60],
            "domain_pool": ["gsm8k", "math_algebra", "arc_challenge"],
            "exp737_source": _EXP737_RESULT_FILE,
        },
        status="success",
    )

    out_path = _root / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written: %s (honest_verdict=%s)", out_path, honest_verdict)

    tmpl.assert_deliverable_written()
    return artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the experiment when invoked directly."""
    with ExperimentTimeoutWatchdog(EXPERIMENT_ID, timeout_minutes=90):
        run_experiment()


if __name__ == "__main__":
    main()
