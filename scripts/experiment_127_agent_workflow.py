#!/usr/bin/env python3
"""Experiment 127: Agent Workflow Verification Benchmark.

**Researcher summary:**
    Exp 125–126 built the ConstraintStateMachine and demonstrated rollback on
    4-step arithmetic reasoning. This experiment broadens the benchmark to three
    *structurally different* workflow types — math (4-step), code (3-step), and
    planning (5-step) — each with 20 synthetic problems. For each workflow type
    we run the full problem set twice: once without the ConstraintStateMachine
    (baseline, errors propagate unchecked) and once with the CSM enabled
    (rollback on first detected violation).

**Workflow types:**
    a. Math (4 steps): extract → compute → verify → answer
       Problems use addition/subtraction in the compute step so the
       ArithmeticExtractor (which checks "A +/- B = C" patterns) can catch
       wrong equations. Faulty compute steps embed a false equation like
       "47 + 28 = 70" (correct: 75). Error propagates into verify and answer.

    b. Code (3 steps): design → implement → test
       The implement step includes an arithmetic self-check (e.g., partial
       sum). A faulty implement step embeds a wrong arithmetic claim such as
       "1 + 2 + 3 + 4 = 15" (correct: 10) that the ArithmeticExtractor flags.
       The test step parrots the wrong result if the implement step is wrong.

    c. Planning (5 steps): goals → constraints → schedule → verify → output
       The verify step checks the budget margin using subtraction: "total -
       budget = 0 (on budget)". A faulty schedule produces a wrong margin
       claim in the verify step, e.g., "55 - 40 = 10 over budget" (correct:
       55-40=15), giving the ArithmeticExtractor a false expression to flag.
       The output step parrots the wrong totals.

**ArithmeticExtractor detection requirement:**
    All three workflow types are designed so the faulty step (or the verify
    step immediately following it) contains a false "+/−" arithmetic expression
    that the built-in ArithmeticExtractor can detect. This is the same
    detection mechanism that caught step-3 errors in Exp 126.

**With-rollback protocol:**
    - If a step's VerificationResult has verified=False *and* at least one
      violation, roll back to the previous step and re-inject the correct text.
    - Subsequent steps then use the correct continuations.
    - If the violation is *not* detected, errors cascade forward as in baseline.

**Metrics (per workflow type and aggregate):**
    - final_accuracy_baseline: fraction of problems where the final step output
      matches the expected answer WITHOUT rollback.
    - final_accuracy_with_csm: fraction correct WITH rollback.
    - avg_steps_baseline: average total steps executed (no retries, always N).
    - avg_steps_with_csm: average total steps including rollback re-runs.
    - rollbacks_triggered: count of problems where CSM triggered a rollback.
    - rollbacks_missed: count of problems where CSM failed to detect violation.
    - violations_per_step: dict mapping step name → total violations detected
      across all 20 problems.

Data flow:
    1. Build 20 problems for each workflow type.
    2. For each problem run: (a) baseline — no CSM, errors propagate;
       (b) with-CSM — CSM enabled, rollback on violations.
    3. Collect per-problem traces + aggregate metrics.
    4. Save to results/experiment_127_results.json.

Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Force JAX to CPU before any carnot imports (avoids ROCm/CUDA instability).
# ---------------------------------------------------------------------------
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Extend path so we can import carnot from the source tree.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from carnot.pipeline.state_machine import ConstraintStateMachine  # noqa: E402
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline  # noqa: E402


# ---------------------------------------------------------------------------
# _SingleArgCompatPipeline — same shim used in Exp 126
# ---------------------------------------------------------------------------


class _SingleArgCompatPipeline(VerifyRepairPipeline):
    """VerifyRepairPipeline shim that accepts single-argument verify() calls.

    **Detailed explanation for engineers:**
        agentic.propagate() calls pipeline.verify(output_text) with one
        positional argument. Standard VerifyRepairPipeline.verify() requires
        two (question, response). This override bridges the gap by treating
        the single argument as the response and passing an empty string as
        the question. This is a local experiment shim — it does not modify
        any shared library code.
    """

    def verify(  # type: ignore[override]
        self,
        question_or_response: str,
        response: str | None = None,
        domain: str | None = None,
    ) -> VerificationResult:
        """Accept (response,) or (question, response) call signatures."""
        if response is None:
            return super().verify("", question_or_response, domain)
        return super().verify(question_or_response, response, domain)


# ===========================================================================
# WORKFLOW A — MATH (4 steps): extract → compute → verify → answer
#
# The ArithmeticExtractor matches: (-?\d+)\s*([+\-])\s*(-?\d+)\s*=\s*(-?\d+)
# All compute-step errors use wrong addition/subtraction so this pattern fires.
# ===========================================================================


@dataclass
class MathWorkflowProblem:
    """One 4-step math workflow problem with correct + faulty step variants.

    **Detailed explanation for engineers:**
        Step sequence: extract → compute → verify → answer.
        The compute step uses "A + B = C" or "A - B = C" format. The faulty
        version has wrong arithmetic (e.g., "47 + 28 = 70" instead of 75).
        The ArithmeticExtractor pattern matches and flags the violation.
        Propagated steps carry the wrong intermediate forward.

    Attributes:
        problem_id: 1-based index within math problems.
        question: Natural-language problem statement.
        correct_steps: All 4 correct step outputs (indices 0..3).
        faulty_step_index: 1-based index of the injected error (always 2).
        faulty_step_text: Wrong compute step with false arithmetic.
        propagated_steps: Steps 3 and 4 when the error propagates forward.
        correct_output: Substring expected in the final correct step.
        error_output: Substring expected in the final error step.
        step_names: Human-readable name for each step position.
    """

    problem_id: int
    question: str
    correct_steps: list[str]
    faulty_step_index: int
    faulty_step_text: str
    propagated_steps: list[str]
    correct_output: str
    error_output: str
    step_names: list[str] = field(
        default_factory=lambda: ["extract", "compute", "verify", "answer"]
    )


def build_math_problems() -> list[MathWorkflowProblem]:
    """Build 20 four-step math workflow problems with false-arithmetic errors.

    **Detailed explanation for engineers:**
        Each problem's compute step uses addition or subtraction that the
        ArithmeticExtractor can verify. The faulty version has a wrong
        claimed result (e.g., 47 + 28 = 70 instead of 75). Propagated steps
        use the wrong intermediate but their own arithmetic is self-consistent
        (so only the injected false expression triggers the violation).

    Returns:
        List of 20 MathWorkflowProblem instances.
    """
    raw: list[dict[str, Any]] = [
        # --- Addition-based problems (compute: a + b = c) ---
        {
            "question": "A warehouse gets 47 boxes Monday and 28 Tuesday. 15 are shipped. How many remain?",
            "correct_steps": [
                "Extract: 47 boxes Monday, 28 Tuesday, 15 shipped.",
                "Compute: 47 + 28 = 75 boxes received.",
                "Verify: Intermediate 75 confirmed.",
                "Answer: 75 - 15 = 60 boxes remain.",
            ],
            "faulty_step_text": "Compute: 47 + 28 = 70 boxes received.",
            "propagated_steps": [
                "Verify: Intermediate 70 confirmed.",
                "Answer: 70 - 15 = 55 boxes remain.",
            ],
            "correct_output": "60 boxes remain",
            "error_output": "55 boxes remain",
        },
        {
            "question": "A store earns $63 on day 1 and $29 on day 2. Rent costs $40. What is the net?",
            "correct_steps": [
                "Extract: Day1=$63, Day2=$29, rent=$40.",
                "Compute: 63 + 29 = 92 dollars earned.",
                "Verify: 92 confirmed.",
                "Answer: 92 - 40 = 52 dollars net.",
            ],
            "faulty_step_text": "Compute: 63 + 29 = 88 dollars earned.",
            "propagated_steps": [
                "Verify: 88 confirmed.",
                "Answer: 88 - 40 = 48 dollars net.",
            ],
            "correct_output": "52 dollars net",
            "error_output": "48 dollars net",
        },
        {
            "question": "A library adds 54 books in spring and 37 in summer. 20 are removed. How many remain?",
            "correct_steps": [
                "Extract: 54 spring, 37 summer, 20 removed.",
                "Compute: 54 + 37 = 91 books added.",
                "Verify: 91 confirmed.",
                "Answer: 91 - 20 = 71 books remain.",
            ],
            "faulty_step_text": "Compute: 54 + 37 = 86 books added.",
            "propagated_steps": [
                "Verify: 86 confirmed.",
                "Answer: 86 - 20 = 66 books remain.",
            ],
            "correct_output": "71 books remain",
            "error_output": "66 books remain",
        },
        {
            "question": "A team scores 38 points in round 1 and 45 in round 2. 12 are deducted. Final score?",
            "correct_steps": [
                "Extract: Round1=38, Round2=45, deduction=12.",
                "Compute: 38 + 45 = 83 points total.",
                "Verify: 83 confirmed.",
                "Answer: 83 - 12 = 71 final score.",
            ],
            "faulty_step_text": "Compute: 38 + 45 = 79 points total.",
            "propagated_steps": [
                "Verify: 79 confirmed.",
                "Answer: 79 - 12 = 67 final score.",
            ],
            "correct_output": "71 final score",
            "error_output": "67 final score",
        },
        {
            "question": "A farm harvests 66 kg grain in week 1 and 34 kg in week 2. 25 kg is sold. Remaining?",
            "correct_steps": [
                "Extract: Week1=66kg, Week2=34kg, sold=25kg.",
                "Compute: 66 + 34 = 100 kg harvested.",
                "Verify: 100 confirmed.",
                "Answer: 100 - 25 = 75 kg remain.",
            ],
            "faulty_step_text": "Compute: 66 + 34 = 95 kg harvested.",
            "propagated_steps": [
                "Verify: 95 confirmed.",
                "Answer: 95 - 25 = 70 kg remain.",
            ],
            "correct_output": "75 kg remain",
            "error_output": "70 kg remain",
        },
        {
            "question": "An account deposits $58 on Monday and $33 on Friday. $20 is withdrawn. Balance?",
            "correct_steps": [
                "Extract: Deposit1=$58, Deposit2=$33, withdrawal=$20.",
                "Compute: 58 + 33 = 91 dollars deposited.",
                "Verify: 91 confirmed.",
                "Answer: 91 - 20 = 71 dollars balance.",
            ],
            "faulty_step_text": "Compute: 58 + 33 = 86 dollars deposited.",
            "propagated_steps": [
                "Verify: 86 confirmed.",
                "Answer: 86 - 20 = 66 dollars balance.",
            ],
            "correct_output": "71 dollars balance",
            "error_output": "66 dollars balance",
        },
        {
            "question": "A pool is filled with 72 liters then 19 liters more. 30 liters evaporate. How much remains?",
            "correct_steps": [
                "Extract: Initial=72L, added=19L, evaporated=30L.",
                "Compute: 72 + 19 = 91 liters total.",
                "Verify: 91 confirmed.",
                "Answer: 91 - 30 = 61 liters remain.",
            ],
            "faulty_step_text": "Compute: 72 + 19 = 85 liters total.",
            "propagated_steps": [
                "Verify: 85 confirmed.",
                "Answer: 85 - 30 = 55 liters remain.",
            ],
            "correct_output": "61 liters remain",
            "error_output": "55 liters remain",
        },
        {
            "question": "A runner logs 43 km in January and 57 km in February. 18 km is training-only. Race km?",
            "correct_steps": [
                "Extract: Jan=43km, Feb=57km, training=18km.",
                "Compute: 43 + 57 = 100 km total.",
                "Verify: 100 confirmed.",
                "Answer: 100 - 18 = 82 km race.",
            ],
            "faulty_step_text": "Compute: 43 + 57 = 96 km total.",
            "propagated_steps": [
                "Verify: 96 confirmed.",
                "Answer: 96 - 18 = 78 km race.",
            ],
            "correct_output": "82 km race",
            "error_output": "78 km race",
        },
        {
            "question": "A project has 85 tasks open and 36 are added. 40 are completed. How many remain?",
            "correct_steps": [
                "Extract: Open=85, added=36, completed=40.",
                "Compute: 85 + 36 = 121 tasks total.",
                "Verify: 121 confirmed.",
                "Answer: 121 - 40 = 81 tasks remain.",
            ],
            "faulty_step_text": "Compute: 85 + 36 = 115 tasks total.",
            "propagated_steps": [
                "Verify: 115 confirmed.",
                "Answer: 115 - 40 = 75 tasks remain.",
            ],
            "correct_output": "81 tasks remain",
            "error_output": "75 tasks remain",
        },
        {
            "question": "A stadium seats 74 in section A and 26 in section B. 30 seats are reserved. Available?",
            "correct_steps": [
                "Extract: A=74, B=26, reserved=30.",
                "Compute: 74 + 26 = 100 seats total.",
                "Verify: 100 confirmed.",
                "Answer: 100 - 30 = 70 seats available.",
            ],
            "faulty_step_text": "Compute: 74 + 26 = 96 seats total.",
            "propagated_steps": [
                "Verify: 96 confirmed.",
                "Answer: 96 - 30 = 66 seats available.",
            ],
            "correct_output": "70 seats available",
            "error_output": "66 seats available",
        },
        # --- Subtraction-based problems (compute: a - b = c) ---
        {
            "question": "A store had 93 items. 35 were sold. 8 were returned. How many are in stock?",
            "correct_steps": [
                "Extract: Initial=93, sold=35, returned=8.",
                "Compute: 93 - 35 = 58 after sales.",
                "Verify: 58 confirmed.",
                "Answer: 58 + 8 = 66 in stock.",
            ],
            "faulty_step_text": "Compute: 93 - 35 = 52 after sales.",
            "propagated_steps": [
                "Verify: 52 confirmed.",
                "Answer: 52 + 8 = 60 in stock.",
            ],
            "correct_output": "66 in stock",
            "error_output": "60 in stock",
        },
        {
            "question": "A tank holds 80 liters. 27 are drained. 15 more are added. Final volume?",
            "correct_steps": [
                "Extract: Initial=80L, drained=27L, added=15L.",
                "Compute: 80 - 27 = 53 liters after draining.",
                "Verify: 53 confirmed.",
                "Answer: 53 + 15 = 68 liters final.",
            ],
            "faulty_step_text": "Compute: 80 - 27 = 48 liters after draining.",
            "propagated_steps": [
                "Verify: 48 confirmed.",
                "Answer: 48 + 15 = 63 liters final.",
            ],
            "correct_output": "68 liters final",
            "error_output": "63 liters final",
        },
        {
            "question": "A budget starts at $120. $44 is spent. $18 is refunded. Remaining budget?",
            "correct_steps": [
                "Extract: Budget=$120, spent=$44, refund=$18.",
                "Compute: 120 - 44 = 76 dollars after spending.",
                "Verify: 76 confirmed.",
                "Answer: 76 + 18 = 94 dollars remaining.",
            ],
            "faulty_step_text": "Compute: 120 - 44 = 70 dollars after spending.",
            "propagated_steps": [
                "Verify: 70 confirmed.",
                "Answer: 70 + 18 = 88 dollars remaining.",
            ],
            "correct_output": "94 dollars remaining",
            "error_output": "88 dollars remaining",
        },
        {
            "question": "A playlist has 95 songs. 38 are removed. 12 are added. Final count?",
            "correct_steps": [
                "Extract: Start=95, removed=38, added=12.",
                "Compute: 95 - 38 = 57 songs after removal.",
                "Verify: 57 confirmed.",
                "Answer: 57 + 12 = 69 songs total.",
            ],
            "faulty_step_text": "Compute: 95 - 38 = 52 songs after removal.",
            "propagated_steps": [
                "Verify: 52 confirmed.",
                "Answer: 52 + 12 = 64 songs total.",
            ],
            "correct_output": "69 songs total",
            "error_output": "64 songs total",
        },
        {
            "question": "A parking lot has 150 spaces. 67 are taken. 20 more cars arrive. Available spaces?",
            "correct_steps": [
                "Extract: Total=150, taken=67, incoming=20.",
                "Compute: 150 - 67 = 83 spaces free.",
                "Verify: 83 confirmed.",
                "Answer: 83 - 20 = 63 spaces available.",
            ],
            "faulty_step_text": "Compute: 150 - 67 = 78 spaces free.",
            "propagated_steps": [
                "Verify: 78 confirmed.",
                "Answer: 78 - 20 = 58 spaces available.",
            ],
            "correct_output": "63 spaces available",
            "error_output": "58 spaces available",
        },
        {
            "question": "A charity collected 200 items. 89 were distributed. 14 were damaged. Usable items?",
            "correct_steps": [
                "Extract: Collected=200, distributed=89, damaged=14.",
                "Compute: 200 - 89 = 111 items remaining.",
                "Verify: 111 confirmed.",
                "Answer: 111 - 14 = 97 usable items.",
            ],
            "faulty_step_text": "Compute: 200 - 89 = 105 items remaining.",
            "propagated_steps": [
                "Verify: 105 confirmed.",
                "Answer: 105 - 14 = 91 usable items.",
            ],
            "correct_output": "97 usable items",
            "error_output": "91 usable items",
        },
        {
            "question": "A river has 175 fish. 62 migrate upstream. 30 return downstream. Net fish count?",
            "correct_steps": [
                "Extract: Start=175, migrated=62, returned=30.",
                "Compute: 175 - 62 = 113 fish remain after migration.",
                "Verify: 113 confirmed.",
                "Answer: 113 + 30 = 143 net fish.",
            ],
            "faulty_step_text": "Compute: 175 - 62 = 108 fish remain after migration.",
            "propagated_steps": [
                "Verify: 108 confirmed.",
                "Answer: 108 + 30 = 138 net fish.",
            ],
            "correct_output": "143 net fish",
            "error_output": "138 net fish",
        },
        {
            "question": "A server farm has 300 cores. 127 are allocated. 50 more jobs come in. Free cores?",
            "correct_steps": [
                "Extract: Total=300, allocated=127, incoming=50.",
                "Compute: 300 - 127 = 173 free cores.",
                "Verify: 173 confirmed.",
                "Answer: 173 - 50 = 123 cores remaining free.",
            ],
            "faulty_step_text": "Compute: 300 - 127 = 167 free cores.",
            "propagated_steps": [
                "Verify: 167 confirmed.",
                "Answer: 167 - 50 = 117 cores remaining free.",
            ],
            "correct_output": "123 cores remaining free",
            "error_output": "117 cores remaining free",
        },
        {
            "question": "A school has 400 students. 136 go on a trip. 25 return early. How many are away?",
            "correct_steps": [
                "Extract: Total=400, on trip=136, returned early=25.",
                "Compute: 400 - 136 = 264 students at school.",
                "Verify: 264 confirmed.",
                "Answer: 136 - 25 = 111 students away.",
            ],
            "faulty_step_text": "Compute: 400 - 136 = 258 students at school.",
            "propagated_steps": [
                "Verify: 258 confirmed.",
                "Answer: 136 - 25 = 111 students away.",
            ],
            "correct_output": "111 students away",
            "error_output": "111 students away",
        },
        {
            "question": "A factory produces 250 units per week. 88 fail QC. 15 are repaired. Passing units?",
            "correct_steps": [
                "Extract: Produced=250, failed=88, repaired=15.",
                "Compute: 250 - 88 = 162 units pass QC.",
                "Verify: 162 confirmed.",
                "Answer: 162 + 15 = 177 passing units.",
            ],
            "faulty_step_text": "Compute: 250 - 88 = 157 units pass QC.",
            "propagated_steps": [
                "Verify: 157 confirmed.",
                "Answer: 157 + 15 = 172 passing units.",
            ],
            "correct_output": "177 passing units",
            "error_output": "172 passing units",
        },
    ]

    problems: list[MathWorkflowProblem] = []
    for idx, r in enumerate(raw):
        problems.append(
            MathWorkflowProblem(
                problem_id=idx + 1,
                question=r["question"],
                correct_steps=r["correct_steps"],
                faulty_step_index=2,
                faulty_step_text=r["faulty_step_text"],
                propagated_steps=r["propagated_steps"],
                correct_output=r["correct_output"],
                error_output=r["error_output"],
            )
        )
    return problems


# ===========================================================================
# WORKFLOW B — CODE (3 steps): design → implement → test
#
# The implement step includes a partial-sum arithmetic self-check like
# "1 + 2 + 3 + 4 = 15" (correct: 10). ArithmeticExtractor catches the
# false addition. The test step reports wrong results when implement is wrong.
# ===========================================================================


@dataclass
class CodeWorkflowProblem:
    """One 3-step code workflow problem.

    **Detailed explanation for engineers:**
        Step sequence: design → implement → test.
        The implement step includes an arithmetic self-check with a wrong
        addition claim (e.g., "1 + 2 + 3 + 4 = 15" instead of 10). The
        ArithmeticExtractor flags this as a violation. The test step then
        explicitly reports the wrong result ("Test FAILED").

    Attributes:
        problem_id: 1-based index within code problems.
        question: Task description.
        correct_steps: Correct design, implement, test texts.
        faulty_step_index: 1-based index with injected error (always 2).
        faulty_step_text: Wrong implementation with false arithmetic.
        propagated_steps: Test step output when implementation is wrong.
        correct_output: Substring expected in the final correct step.
        error_output: Substring expected in the final error step.
        step_names: Names for each step position.
    """

    problem_id: int
    question: str
    correct_steps: list[str]
    faulty_step_index: int
    faulty_step_text: str
    propagated_steps: list[str]
    correct_output: str
    error_output: str
    step_names: list[str] = field(
        default_factory=lambda: ["design", "implement", "test"]
    )


def build_code_problems() -> list[CodeWorkflowProblem]:
    """Build 20 three-step code workflow problems with detectable errors.

    **Detailed explanation for engineers:**
        Each problem's faulty implement step contains a wrong partial-sum
        arithmetic expression (e.g., "1 + 2 + 3 + 4 = 15" for a sum-1-to-4
        problem where correct is 10). This gives the ArithmeticExtractor a
        false "A + B = C" pattern to detect. The test step reflects the
        outcome; when the implement step is wrong, the test fails.

    Returns:
        List of 20 CodeWorkflowProblem instances.
    """
    raw: list[dict[str, Any]] = [
        {
            "question": "Compute sum of integers 1..4. Expected: 10.",
            "correct_steps": [
                "Design: Accumulate 1+2+3+4.",
                "Implement: 1 + 2 = 3; 3 + 3 = 6; 6 + 4 = 10. Result = 10.",
                "Test: assert result == 10. Test PASSED: result is 10.",
            ],
            "faulty_step_text": "Implement: 1 + 2 = 3; 3 + 3 = 6; 6 + 4 = 15. Result = 15.",
            "propagated_steps": [
                "Test: assert result == 10. AssertionError: result is 15. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Compute sum of [5, 8, 7]. Expected: 20.",
            "correct_steps": [
                "Design: Add all elements.",
                "Implement: 5 + 8 = 13; 13 + 7 = 20. Result = 20.",
                "Test: assert result == 20. Test PASSED: result is 20.",
            ],
            "faulty_step_text": "Implement: 5 + 8 = 13; 13 + 7 = 16. Result = 16.",
            "propagated_steps": [
                "Test: assert result == 20. AssertionError: result is 16. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Compute cumulative sum of [3, 7, 2, 8]. Final value: 20.",
            "correct_steps": [
                "Design: Running sum through the list.",
                "Implement: 3 + 7 = 10; 10 + 2 = 12; 12 + 8 = 20. Result = 20.",
                "Test: assert result == 20. Test PASSED: result is 20.",
            ],
            "faulty_step_text": "Implement: 3 + 7 = 10; 10 + 2 = 12; 12 + 8 = 18. Result = 18.",
            "propagated_steps": [
                "Test: assert result == 20. AssertionError: result is 18. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Sum [10, 15, 5]. Expected: 30.",
            "correct_steps": [
                "Design: Add elements sequentially.",
                "Implement: 10 + 15 = 25; 25 + 5 = 30. Result = 30.",
                "Test: assert result == 30. Test PASSED: result is 30.",
            ],
            "faulty_step_text": "Implement: 10 + 15 = 25; 25 + 5 = 27. Result = 27.",
            "propagated_steps": [
                "Test: assert result == 30. AssertionError: result is 27. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Compute sum of even numbers in [1, 2, 3, 4, 6]. Expected: 12.",
            "correct_steps": [
                "Design: Filter evens then add.",
                "Implement: evens=[2,4,6]; 2 + 4 = 6; 6 + 6 = 12. Result = 12.",
                "Test: assert result == 12. Test PASSED: result is 12.",
            ],
            "faulty_step_text": "Implement: evens=[2,4,6]; 2 + 4 = 6; 6 + 6 = 10. Result = 10.",
            "propagated_steps": [
                "Test: assert result == 12. AssertionError: result is 10. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Total cost: 3 items at $4 each plus $6 tax. Expected: $18.",
            "correct_steps": [
                "Design: Compute item total then add tax.",
                "Implement: 4 + 4 = 8; 8 + 4 = 12; 12 + 6 = 18. Result = 18.",
                "Test: assert result == 18. Test PASSED: result is 18.",
            ],
            "faulty_step_text": "Implement: 4 + 4 = 8; 8 + 4 = 12; 12 + 6 = 15. Result = 15.",
            "propagated_steps": [
                "Test: assert result == 18. AssertionError: result is 15. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Compute perimeter of rectangle 9x6. Expected: 30.",
            "correct_steps": [
                "Design: 2*(width+height).",
                "Implement: 9 + 6 = 15; 15 + 15 = 30. Result = 30.",
                "Test: assert result == 30. Test PASSED: result is 30.",
            ],
            "faulty_step_text": "Implement: 9 + 6 = 15; 15 + 15 = 27. Result = 27.",
            "propagated_steps": [
                "Test: assert result == 30. AssertionError: result is 27. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Sum ASCII values of 'abc'. a=97, b=98, c=99. Expected: 294.",
            "correct_steps": [
                "Design: Add ASCII codes.",
                "Implement: 97 + 98 = 195; 195 + 99 = 294. Result = 294.",
                "Test: assert result == 294. Test PASSED: result is 294.",
            ],
            "faulty_step_text": "Implement: 97 + 98 = 195; 195 + 99 = 288. Result = 288.",
            "propagated_steps": [
                "Test: assert result == 294. AssertionError: result is 288. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Merge sorted [2, 5] and [3, 7], compute sum. Expected: 17.",
            "correct_steps": [
                "Design: Concatenate and sum.",
                "Implement: 2 + 5 = 7; 3 + 7 = 10; 7 + 10 = 17. Result = 17.",
                "Test: assert result == 17. Test PASSED: result is 17.",
            ],
            "faulty_step_text": "Implement: 2 + 5 = 7; 3 + 7 = 10; 7 + 10 = 14. Result = 14.",
            "propagated_steps": [
                "Test: assert result == 17. AssertionError: result is 14. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Compute total from [12, 8, 5, 10]. Expected: 35.",
            "correct_steps": [
                "Design: Sequential addition.",
                "Implement: 12 + 8 = 20; 20 + 5 = 25; 25 + 10 = 35. Result = 35.",
                "Test: assert result == 35. Test PASSED: result is 35.",
            ],
            "faulty_step_text": "Implement: 12 + 8 = 20; 20 + 5 = 25; 25 + 10 = 31. Result = 31.",
            "propagated_steps": [
                "Test: assert result == 35. AssertionError: result is 31. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Running total of log bytes [100, 200, 50]. Expected: 350.",
            "correct_steps": [
                "Design: Accumulate byte counts.",
                "Implement: 100 + 200 = 300; 300 + 50 = 350. Result = 350.",
                "Test: assert result == 350. Test PASSED: result is 350.",
            ],
            "faulty_step_text": "Implement: 100 + 200 = 300; 300 + 50 = 340. Result = 340.",
            "propagated_steps": [
                "Test: assert result == 350. AssertionError: result is 340. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Total reward: base=50, bonus=30, streak=20. Expected: 100.",
            "correct_steps": [
                "Design: Sum reward components.",
                "Implement: 50 + 30 = 80; 80 + 20 = 100. Result = 100.",
                "Test: assert result == 100. Test PASSED: result is 100.",
            ],
            "faulty_step_text": "Implement: 50 + 30 = 80; 80 + 20 = 94. Result = 94.",
            "propagated_steps": [
                "Test: assert result == 100. AssertionError: result is 94. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Token count: prefix=15, body=60, suffix=10. Expected: 85.",
            "correct_steps": [
                "Design: Sum token counts.",
                "Implement: 15 + 60 = 75; 75 + 10 = 85. Result = 85.",
                "Test: assert result == 85. Test PASSED: result is 85.",
            ],
            "faulty_step_text": "Implement: 15 + 60 = 75; 75 + 10 = 79. Result = 79.",
            "propagated_steps": [
                "Test: assert result == 85. AssertionError: result is 79. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Sum of [6, 9, 4, 11]. Expected: 30.",
            "correct_steps": [
                "Design: Add sequentially.",
                "Implement: 6 + 9 = 15; 15 + 4 = 19; 19 + 11 = 30. Result = 30.",
                "Test: assert result == 30. Test PASSED: result is 30.",
            ],
            "faulty_step_text": "Implement: 6 + 9 = 15; 15 + 4 = 19; 19 + 11 = 26. Result = 26.",
            "propagated_steps": [
                "Test: assert result == 30. AssertionError: result is 26. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Compute total votes: precinct A=143, B=97, C=60. Expected: 300.",
            "correct_steps": [
                "Design: Sum precinct counts.",
                "Implement: 143 + 97 = 240; 240 + 60 = 300. Result = 300.",
                "Test: assert result == 300. Test PASSED: result is 300.",
            ],
            "faulty_step_text": "Implement: 143 + 97 = 240; 240 + 60 = 294. Result = 294.",
            "propagated_steps": [
                "Test: assert result == 300. AssertionError: result is 294. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Compute total weight: package A=18kg, B=22kg, C=5kg. Expected: 45kg.",
            "correct_steps": [
                "Design: Sum package weights.",
                "Implement: 18 + 22 = 40; 40 + 5 = 45. Result = 45.",
                "Test: assert result == 45. Test PASSED: result is 45.",
            ],
            "faulty_step_text": "Implement: 18 + 22 = 40; 40 + 5 = 41. Result = 41.",
            "propagated_steps": [
                "Test: assert result == 45. AssertionError: result is 41. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Sum distances: leg1=35km, leg2=28km, leg3=17km. Expected: 80km.",
            "correct_steps": [
                "Design: Add leg distances.",
                "Implement: 35 + 28 = 63; 63 + 17 = 80. Result = 80.",
                "Test: assert result == 80. Test PASSED: result is 80.",
            ],
            "faulty_step_text": "Implement: 35 + 28 = 63; 63 + 17 = 76. Result = 76.",
            "propagated_steps": [
                "Test: assert result == 80. AssertionError: result is 76. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Monthly savings: Jan=$55, Feb=$45, Mar=$50. Expected: $150.",
            "correct_steps": [
                "Design: Sum monthly savings.",
                "Implement: 55 + 45 = 100; 100 + 50 = 150. Result = 150.",
                "Test: assert result == 150. Test PASSED: result is 150.",
            ],
            "faulty_step_text": "Implement: 55 + 45 = 100; 100 + 50 = 144. Result = 144.",
            "propagated_steps": [
                "Test: assert result == 150. AssertionError: result is 144. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Total calories: breakfast=400, lunch=600, dinner=500. Expected: 1500.",
            "correct_steps": [
                "Design: Sum meal calories.",
                "Implement: 400 + 600 = 1000; 1000 + 500 = 1500. Result = 1500.",
                "Test: assert result == 1500. Test PASSED: result is 1500.",
            ],
            "faulty_step_text": "Implement: 400 + 600 = 1000; 1000 + 500 = 1490. Result = 1490.",
            "propagated_steps": [
                "Test: assert result == 1500. AssertionError: result is 1490. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
        {
            "question": "Compute CI pipeline time: build=12min, test=18min, deploy=5min. Expected: 35min.",
            "correct_steps": [
                "Design: Sum pipeline stage durations.",
                "Implement: 12 + 18 = 30; 30 + 5 = 35. Result = 35.",
                "Test: assert result == 35. Test PASSED: result is 35.",
            ],
            "faulty_step_text": "Implement: 12 + 18 = 30; 30 + 5 = 32. Result = 32.",
            "propagated_steps": [
                "Test: assert result == 35. AssertionError: result is 32. Test FAILED.",
            ],
            "correct_output": "Test PASSED",
            "error_output": "Test FAILED",
        },
    ]

    problems: list[CodeWorkflowProblem] = []
    for idx, r in enumerate(raw):
        problems.append(
            CodeWorkflowProblem(
                problem_id=idx + 1,
                question=r["question"],
                correct_steps=r["correct_steps"],
                faulty_step_index=2,
                faulty_step_text=r["faulty_step_text"],
                propagated_steps=r["propagated_steps"],
                correct_output=r["correct_output"],
                error_output=r["error_output"],
            )
        )
    return problems


# ===========================================================================
# WORKFLOW C — PLANNING (5 steps): goals → constraints → schedule → verify → output
#
# The verify step computes "schedule_total - budget_cap = X over budget".
# A faulty schedule makes the verify step claim a wrong subtraction result
# (e.g., "55 - 40 = 10 over budget" where correct is 15). This false
# arithmetic triggers the ArithmeticExtractor violation at the verify step.
# ===========================================================================


@dataclass
class PlanningWorkflowProblem:
    """One 5-step planning workflow problem.

    **Detailed explanation for engineers:**
        Step sequence: goals → constraints → schedule → verify → output.
        For correct schedules, the verify step confirms "total - cap = 0 (on
        budget)" which is arithmetically true. For faulty schedules, the
        verify step states a wrong over-budget calculation like
        "55 - 40 = 10 over budget" (correct: 55-40=15), giving the
        ArithmeticExtractor a false expression to flag. The faulty step index
        is 4 (verify) so rollback restores the machine to step 3 (schedule)
        and the schedule is re-run with correct values.

    Attributes:
        problem_id: 1-based index within planning problems.
        question: Scenario description.
        correct_steps: Correct 5-step texts.
        faulty_step_index: 1-based step with injected error (always 4, verify).
        faulty_step_text: Verify step text with wrong subtraction.
        propagated_steps: Output step when schedule is wrong.
        correct_output: Substring expected in the final correct step.
        error_output: Substring expected in the final error step.
        step_names: Names for each step position.
    """

    problem_id: int
    question: str
    correct_steps: list[str]
    faulty_step_index: int
    faulty_step_text: str
    propagated_steps: list[str]
    correct_output: str
    error_output: str
    step_names: list[str] = field(
        default_factory=lambda: ["goals", "constraints", "schedule", "verify", "output"]
    )


def build_planning_problems() -> list[PlanningWorkflowProblem]:
    """Build 20 five-step planning workflow problems with detectable errors.

    **Detailed explanation for engineers:**
        Faulty step is always step 4 (verify). When the schedule exceeds the
        budget, the verify step tries to compute the overage using subtraction,
        but gets the arithmetic wrong (e.g., "55 - 40 = 10" instead of 15).
        ArithmeticExtractor catches the wrong subtraction. Rollback returns to
        step 3 (schedule), then a corrected schedule is re-run, producing a
        verify step that states "total - cap = 0 (on budget)" which is true.

    Returns:
        List of 20 PlanningWorkflowProblem instances.
    """
    raw: list[dict[str, Any]] = [
        {
            "question": "Allocate tasks to 3 engineers with a 40-hour cap.",
            "correct_steps": [
                "Goals: Complete features A, B, C within deadline.",
                "Constraints: Total ≤ 40 engineer-hours.",
                "Schedule: A=15h, B=15h, C=10h; total=40h.",
                "Verify: 40 - 40 = 0 (on budget). Schedule is VALID.",
                "Output: Plan: A=15h, B=15h, C=10h. Total=40h. On budget.",
            ],
            "faulty_step_text": "Verify: 55 - 40 = 10 over budget. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Plan: A=20h, B=20h, C=15h. Total=55h. OVER BUDGET.",
            ],
            "correct_output": "On budget",
            "error_output": "OVER BUDGET",
        },
        {
            "question": "Schedule 4 tasks for solo developer in 30-hour sprint.",
            "correct_steps": [
                "Goals: Deliver T1..T4 in one sprint.",
                "Constraints: Max 30 sprint-hours.",
                "Schedule: T1=8h, T2=7h, T3=8h, T4=7h; total=30h.",
                "Verify: 30 - 30 = 0 (on budget). Schedule is VALID.",
                "Output: Sprint: T1=8h, T2=7h, T3=8h, T4=7h. Total=30h.",
            ],
            "faulty_step_text": "Verify: 40 - 30 = 6 over budget. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Sprint: T1=10h, T2=10h, T3=10h, T4=10h. Total=40h. OVER CAPACITY.",
            ],
            "correct_output": "Total=30h",
            "error_output": "OVER CAPACITY",
        },
        {
            "question": "Plan a conference with a $5000 budget.",
            "correct_steps": [
                "Goals: Venue, catering, AV within $5000.",
                "Constraints: Total ≤ $5000.",
                "Schedule: Venue=$2000, Catering=$2000, AV=$1000; total=$5000.",
                "Verify: 5000 - 5000 = 0 (on budget). Plan is VALID.",
                "Output: Budget: Venue=$2000, Catering=$2000, AV=$1000. Total=$5000.",
            ],
            "faulty_step_text": "Verify: 6500 - 5000 = 1000 over budget. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Budget: Total=$6500. OVER BUDGET.",
            ],
            "correct_output": "Total=$5000",
            "error_output": "OVER BUDGET",
        },
        {
            "question": "Allocate CPU to 3 services; max 100% per server.",
            "correct_steps": [
                "Goals: Run S1, S2, S3 on one server.",
                "Constraints: Total CPU ≤ 100%.",
                "Schedule: S1=40%, S2=35%, S3=25%; total=100%.",
                "Verify: 100 - 100 = 0 (on budget). Schedule is VALID.",
                "Output: CPU: S1=40%, S2=35%, S3=25%. Total=100%.",
            ],
            "faulty_step_text": "Verify: 120 - 100 = 15 over limit. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: CPU: S1=50%, S2=40%, S3=30%. Total=120%. OVER LIMIT.",
            ],
            "correct_output": "Total=100%",
            "error_output": "OVER LIMIT",
        },
        {
            "question": "Schedule 5 meetings in an 8-hour workday.",
            "correct_steps": [
                "Goals: Fit M1..M5 into one workday.",
                "Constraints: Workday = 8 hours.",
                "Schedule: M1-M5 at hours 1-5; total=5h.",
                "Verify: 5 - 8 = -3 (3 hours free). Schedule is VALID.",
                "Output: 5 meetings scheduled, 3 hours free.",
            ],
            "faulty_step_text": "Verify: 9 - 8 = 0 (overruns). CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: 9 meetings. OVERRUNS WORKDAY.",
            ],
            "correct_output": "5 meetings scheduled",
            "error_output": "OVERRUNS WORKDAY",
        },
        {
            "question": "Assign disk quota to 4 users; pool = 200GB.",
            "correct_steps": [
                "Goals: Allocate storage to U1..U4.",
                "Constraints: Total ≤ 200GB.",
                "Schedule: U1=60GB, U2=50GB, U3=50GB, U4=40GB; total=200GB.",
                "Verify: 200 - 200 = 0 (on budget). Allocation is VALID.",
                "Output: Storage: U1=60GB, U2=50GB, U3=50GB, U4=40GB. Total=200GB.",
            ],
            "faulty_step_text": "Verify: 240 - 200 = 30 over budget. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Storage: total=240GB. OVER CAPACITY.",
            ],
            "correct_output": "Total=200GB",
            "error_output": "OVER CAPACITY",
        },
        {
            "question": "Plan deliveries for 3 trucks; max 5 deliveries per truck.",
            "correct_steps": [
                "Goals: Complete 12 deliveries with 3 trucks.",
                "Constraints: Each truck ≤ 5 deliveries.",
                "Schedule: Truck1=4, Truck2=4, Truck3=4; total=12.",
                "Verify: 12 - 15 = -3 (within limit). Schedule is VALID.",
                "Output: Delivery: 3 trucks × 4 = 12 total. On target.",
            ],
            "faulty_step_text": "Verify: 18 - 15 = 2 over limit. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Delivery: total=18. OVER LIMIT per truck.",
            ],
            "correct_output": "12 total",
            "error_output": "OVER LIMIT",
        },
        {
            "question": "Budget a road trip: fuel, food, lodging ≤ $300.",
            "correct_steps": [
                "Goals: Cover fuel, food, lodging within $300.",
                "Constraints: Total trip ≤ $300.",
                "Schedule: Fuel=$100, Food=$100, Lodging=$100; total=$300.",
                "Verify: 300 - 300 = 0 (on budget). Plan is VALID.",
                "Output: Trip: Fuel=$100, Food=$100, Lodging=$100. Total=$300.",
            ],
            "faulty_step_text": "Verify: 360 - 300 = 50 over budget. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Trip: total=$360. OVER BUDGET.",
            ],
            "correct_output": "Total=$300",
            "error_output": "OVER BUDGET",
        },
        {
            "question": "Assign lab shifts: 4 researchers, max 2 shifts each, 6 slots.",
            "correct_steps": [
                "Goals: Cover 6 lab shifts.",
                "Constraints: Each researcher ≤ 2 shifts.",
                "Schedule: R1=2, R2=2, R3=1, R4=1; total=6.",
                "Verify: 6 - 8 = -2 (within cap). Schedule is VALID.",
                "Output: Shifts: R1=2, R2=2, R3=1, R4=1. Total=6.",
            ],
            "faulty_step_text": "Verify: 12 - 8 = 3 over cap. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Shifts: R1=3, R2=3, R3=3, R4=3. Total=12. OVER LIMIT.",
            ],
            "correct_output": "Total=6",
            "error_output": "OVER LIMIT",
        },
        {
            "question": "Schedule 3 batch jobs; total memory ≤ 16GB.",
            "correct_steps": [
                "Goals: Run batch jobs J1, J2, J3 concurrently.",
                "Constraints: Available memory = 16GB.",
                "Schedule: J1=6GB, J2=6GB, J3=4GB; total=16GB.",
                "Verify: 16 - 16 = 0 (on budget). Schedule is VALID.",
                "Output: Jobs: J1=6GB, J2=6GB, J3=4GB. Total=16GB.",
            ],
            "faulty_step_text": "Verify: 24 - 16 = 6 over limit. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Jobs: total=24GB. MEMORY EXCEEDED.",
            ],
            "correct_output": "Total=16GB",
            "error_output": "MEMORY EXCEEDED",
        },
        {
            "question": "Plan ad spend across 4 channels; max $1000 total.",
            "correct_steps": [
                "Goals: Allocate ad budget to 4 channels.",
                "Constraints: Total ≤ $1000.",
                "Schedule: Search=$300, Social=$300, Display=$250, Video=$150; total=$1000.",
                "Verify: 1000 - 1000 = 0 (on budget). Plan is VALID.",
                "Output: Ad plan: total=$1000. Within budget.",
            ],
            "faulty_step_text": "Verify: 1300 - 1000 = 200 over budget. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Ad plan: total=$1300. OVER BUDGET.",
            ],
            "correct_output": "Within budget",
            "error_output": "OVER BUDGET",
        },
        {
            "question": "Allocate training: 5 topics, max 20 hours total.",
            "correct_steps": [
                "Goals: Cover T1..T5 in training.",
                "Constraints: Total ≤ 20 hours.",
                "Schedule: T1=4h, T2=4h, T3=4h, T4=4h, T5=4h; total=20h.",
                "Verify: 20 - 20 = 0 (on budget). Schedule is VALID.",
                "Output: Training: 5 × 4h = 20h total.",
            ],
            "faulty_step_text": "Verify: 30 - 20 = 8 over limit. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Training: total=30h. OVER LIMIT.",
            ],
            "correct_output": "20h total",
            "error_output": "OVER LIMIT",
        },
        {
            "question": "Distribute API rate: 3 clients share 1000 req/min.",
            "correct_steps": [
                "Goals: Serve C1, C2, C3 within rate limit.",
                "Constraints: Total ≤ 1000 req/min.",
                "Schedule: C1=400, C2=350, C3=250; total=1000.",
                "Verify: 1000 - 1000 = 0 (on budget). Allocation is VALID.",
                "Output: Rate: C1=400, C2=350, C3=250. Total=1000.",
            ],
            "faulty_step_text": "Verify: 1500 - 1000 = 400 over limit. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Rate: total=1500. OVER LIMIT.",
            ],
            "correct_output": "Total=1000",
            "error_output": "OVER LIMIT",
        },
        {
            "question": "Schedule 3 construction phases; total ≤ 12 weeks.",
            "correct_steps": [
                "Goals: Complete foundation, framing, finishing.",
                "Constraints: Total ≤ 12 weeks.",
                "Schedule: Foundation=4w, Framing=4w, Finishing=4w; total=12w.",
                "Verify: 12 - 12 = 0 (on schedule). Plan is VALID.",
                "Output: Construction: total=12 weeks. On schedule.",
            ],
            "faulty_step_text": "Verify: 18 - 12 = 4 over schedule. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Construction: total=18w. OVER SCHEDULE.",
            ],
            "correct_output": "On schedule",
            "error_output": "OVER SCHEDULE",
        },
        {
            "question": "Assign bandwidth to 4 VMs sharing 1Gbps.",
            "correct_steps": [
                "Goals: Provision bandwidth for V1..V4.",
                "Constraints: Total ≤ 1000Mbps.",
                "Schedule: V1=300, V2=300, V3=250, V4=150; total=1000Mbps.",
                "Verify: 1000 - 1000 = 0 (on budget). Schedule is VALID.",
                "Output: Bandwidth: total=1000Mbps. Within cap.",
            ],
            "faulty_step_text": "Verify: 1600 - 1000 = 500 over cap. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Bandwidth: total=1600Mbps. OVER CAPACITY.",
            ],
            "correct_output": "Within cap",
            "error_output": "OVER CAPACITY",
        },
        {
            "question": "Plan volunteer shifts: 6 slots, 3 volunteers, max 3 shifts each.",
            "correct_steps": [
                "Goals: Cover 6 event shifts.",
                "Constraints: Each volunteer ≤ 3 shifts.",
                "Schedule: V1=3, V2=2, V3=1; total=6.",
                "Verify: 6 - 9 = -3 (within cap). Schedule is VALID.",
                "Output: Volunteers: V1=3, V2=2, V3=1. Total=6.",
            ],
            "faulty_step_text": "Verify: 12 - 9 = 2 over cap. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Volunteers: total=12. OVER LIMIT.",
            ],
            "correct_output": "Total=6",
            "error_output": "OVER LIMIT",
        },
        {
            "question": "Allocate power to 4 racks; UPS = 20kW.",
            "correct_steps": [
                "Goals: Power R1..R4 from shared UPS.",
                "Constraints: UPS = 20kW.",
                "Schedule: R1=6kW, R2=6kW, R3=5kW, R4=3kW; total=20kW.",
                "Verify: 20 - 20 = 0 (on budget). Allocation is VALID.",
                "Output: Power: total=20kW. Within UPS capacity.",
            ],
            "faulty_step_text": "Verify: 32 - 20 = 8 over limit. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Power: total=32kW. UPS OVERLOADED.",
            ],
            "correct_output": "Within UPS capacity",
            "error_output": "UPS OVERLOADED",
        },
        {
            "question": "Distribute test coverage: 3 suites, max 100 tests total.",
            "correct_steps": [
                "Goals: Write tests for M1, M2, M3.",
                "Constraints: Total ≤ 100 tests.",
                "Schedule: M1=40, M2=35, M3=25; total=100.",
                "Verify: 100 - 100 = 0 (on budget). Schedule is VALID.",
                "Output: Tests: M1=40, M2=35, M3=25. Total=100.",
            ],
            "faulty_step_text": "Verify: 180 - 100 = 70 over limit. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Tests: total=180. OVER LIMIT.",
            ],
            "correct_output": "Total=100",
            "error_output": "OVER LIMIT",
        },
        {
            "question": "Schedule report generation: 4 reports, max 10 hours.",
            "correct_steps": [
                "Goals: Generate R1..R4 in business day.",
                "Constraints: Total ≤ 10h.",
                "Schedule: R1=3h, R2=3h, R3=2h, R4=2h; total=10h.",
                "Verify: 10 - 10 = 0 (on budget). Schedule is VALID.",
                "Output: Reports: total=10h. On time.",
            ],
            "faulty_step_text": "Verify: 16 - 10 = 4 over limit. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Reports: total=16h. OVER TIME.",
            ],
            "correct_output": "On time",
            "error_output": "OVER TIME",
        },
        {
            "question": "Allocate sprint points: 4 stories, max 20 points.",
            "correct_steps": [
                "Goals: Estimate S1..S4 for sprint.",
                "Constraints: Sprint capacity = 20 points.",
                "Schedule: S1=6, S2=6, S3=5, S4=3; total=20.",
                "Verify: 20 - 20 = 0 (on budget). Sprint is VALID.",
                "Output: Sprint: total=20 points. Fits sprint.",
            ],
            "faulty_step_text": "Verify: 32 - 20 = 10 over capacity. CONSTRAINT VIOLATION.",
            "propagated_steps": [
                "Output: Sprint: total=32 points. OVER CAPACITY.",
            ],
            "correct_output": "Fits sprint",
            "error_output": "OVER CAPACITY",
        },
    ]

    problems: list[PlanningWorkflowProblem] = []
    for idx, r in enumerate(raw):
        problems.append(
            PlanningWorkflowProblem(
                problem_id=idx + 1,
                question=r["question"],
                correct_steps=r["correct_steps"],
                faulty_step_index=4,  # always the verify step
                faulty_step_text=r["faulty_step_text"],
                propagated_steps=r["propagated_steps"],
                correct_output=r["correct_output"],
                error_output=r["error_output"],
            )
        )
    return problems


# ===========================================================================
# Generic step-sequence builder (works for all three workflow types)
# ===========================================================================


def faulty_step_sequence(
    correct_steps: list[str],
    faulty_step_index: int,
    faulty_step_text: str,
    propagated_steps: list[str],
) -> list[str]:
    """Build the error-propagated step sequence from problem components.

    **Detailed explanation for engineers:**
        Replaces the faulty step and all subsequent steps with the propagated
        versions. faulty_step_index is 1-based.

    Args:
        correct_steps: All N correct step texts.
        faulty_step_index: 1-based index of the injected error.
        faulty_step_text: Wrong output for the faulty step.
        propagated_steps: Downstream step outputs if error propagates.

    Returns:
        Full step list with error and propagation injected.
    """
    seq = list(correct_steps)
    fi = faulty_step_index  # 1-based
    seq[fi - 1] = faulty_step_text
    for offset, prop_text in enumerate(propagated_steps):
        seq[fi + offset] = prop_text
    return seq


# ===========================================================================
# Generic baseline runner (no CSM)
# ===========================================================================


def run_baseline(
    problem_id: int,
    question: str,
    step_names: list[str],
    steps_seq: list[str],
    correct_output: str,
) -> dict[str, Any]:
    """Run a problem WITHOUT ConstraintStateMachine (baseline).

    **Detailed explanation for engineers:**
        Feeds each step text through without any constraint checking or
        rollback. Checks whether the final step text contains the expected
        correct_output substring. Since errors propagate into the final step,
        baseline rarely produces the correct output.

        'steps_to_completion' is always N (no retries).

    Args:
        problem_id: Identifier for the problem.
        question: Natural-language problem statement.
        step_names: Human-readable names for each step.
        steps_seq: The (possibly error-propagated) step text sequence.
        correct_output: Substring expected in the final step for correctness.

    Returns:
        Dict with per-step trace and outcome metrics.
    """
    step_results: list[dict[str, Any]] = []
    violations_per_step: dict[str, int] = {}

    for i, (name, text) in enumerate(zip(step_names, steps_seq)):
        step_results.append({
            "step": i + 1,
            "step_name": name,
            "output_text": text,
            "verified": None,
            "n_violations": 0,
            "violations": [],
            "n_new_facts": 0,
            "contradictions": [],
            "rollback_triggered": False,
        })
        violations_per_step[name] = 0

    final_text = steps_seq[-1]
    is_correct = correct_output in final_text

    return {
        "problem_id": problem_id,
        "mode": "baseline",
        "is_correct": is_correct,
        "steps_to_completion": len(steps_seq),
        "rollbacks_triggered": 0,
        "steps": step_results,
        "violations_per_step": violations_per_step,
    }


# ===========================================================================
# Generic CSM runner (with rollback)
# ===========================================================================


def run_with_csm(
    problem_id: int,
    question: str,
    step_names: list[str],
    correct_steps: list[str],
    faulty_step_index: int,
    faulty_step_text: str,
    propagated_steps: list[str],
    correct_output: str,
    pipeline: _SingleArgCompatPipeline,
) -> dict[str, Any]:
    """Run a problem WITH ConstraintStateMachine and rollback on violations.

    **Detailed explanation for engineers:**
        Feeds the error-propagated step sequence through a fresh
        ConstraintStateMachine. At each step, if verified=False *and* there
        are violations, we:
          1. Rollback to the previous step (to_step = current_step_idx - 1),
             or stay at 0 if the faulty step is step 1.
          2. Re-run the current step with the correct text.
          3. Continue subsequent steps with the correct texts.

        If the CSM does NOT detect the violation (missed), subsequent steps
        carry the error forward as in baseline mode.

        'steps_to_completion' counts total step() calls including re-runs.

    Args:
        problem_id: Identifier for the problem.
        question: Natural-language problem statement.
        step_names: Human-readable names for each step.
        correct_steps: All N correct step texts.
        faulty_step_index: 1-based index of the injected error.
        faulty_step_text: Wrong output for the faulty step.
        propagated_steps: Downstream step outputs if error propagates.
        correct_output: Substring expected in the final step for correctness.
        pipeline: Shared _SingleArgCompatPipeline for verification.

    Returns:
        Dict with per-step trace, rollback events, and outcome metrics.
    """
    csm = ConstraintStateMachine(pipeline)
    steps_seq = faulty_step_sequence(
        correct_steps, faulty_step_index, faulty_step_text, propagated_steps
    )

    step_results: list[dict[str, Any]] = []
    rollback_events: list[dict[str, Any]] = []
    violations_per_step: dict[str, int] = {name: 0 for name in step_names}

    total_csm_calls = 0
    rollback_triggered = False
    rollback_detected = False
    n_steps = len(correct_steps)

    for step_num in range(1, n_steps + 1):
        step_idx = step_num - 1
        step_name = step_names[step_idx]

        # Use correct text once rollback has been detected.
        if rollback_detected and step_num >= faulty_step_index:
            output_text = correct_steps[step_idx]
        else:
            output_text = steps_seq[step_idx]

        input_text = question if step_num == 1 else f"Step {step_num}: {step_name}"
        result = csm.step(input_text, output_text)
        total_csm_calls += 1

        n_viol = len(result.verification.violations)
        violations_per_step[step_name] += n_viol

        step_record: dict[str, Any] = {
            "step": step_num,
            "step_name": step_name,
            "output_text": output_text,
            "verified": result.verification.verified,
            "n_violations": n_viol,
            "violations": [v.description for v in result.verification.violations],
            "n_new_facts": len(result.new_facts),
            "contradictions": result.contradictions,
            "rollback_triggered": False,
        }

        # Check for rollback: violation at the faulty step.
        is_faulty_step = (step_num == faulty_step_index)
        violation_detected = (not result.verification.verified and n_viol > 0)

        if is_faulty_step and violation_detected and not rollback_triggered:
            rollback_triggered = True
            rollback_detected = True
            step_record["rollback_triggered"] = True

            rollback_to = step_idx - 1
            if rollback_to >= 0:
                csm.rollback(rollback_to)

            # Re-run the faulty step with the correct text.
            correct_text = correct_steps[step_idx]
            rerun_result = csm.step(input_text, correct_text)
            total_csm_calls += 1

            rollback_events.append({
                "at_step": step_num,
                "step_name": step_name,
                "rolled_back_to": max(rollback_to, 0),
                "rerun_verified": rerun_result.verification.verified,
            })

            step_record["rerun_text"] = correct_text
            step_record["rerun_verified"] = rerun_result.verification.verified
            violations_per_step[step_name] += len(rerun_result.verification.violations)

        step_results.append(step_record)

    # Final output text: if rollback was detected, last step used correct text.
    final_output_text = correct_steps[-1] if rollback_detected else steps_seq[-1]
    is_correct = correct_output in final_output_text

    return {
        "problem_id": problem_id,
        "mode": "with_csm",
        "is_correct": is_correct,
        "steps_to_completion": total_csm_calls,
        "rollbacks_triggered": len(rollback_events),
        "rollback_missed": 1 if (not rollback_detected) else 0,
        "steps": step_results,
        "rollback_events": rollback_events,
        "violations_per_step": violations_per_step,
    }


# ===========================================================================
# Per-workflow benchmark runner
# ===========================================================================


def run_workflow_benchmark(
    workflow_name: str,
    problems: list[Any],
    pipeline: _SingleArgCompatPipeline,
) -> dict[str, Any]:
    """Run baseline and CSM for all problems in one workflow type.

    **Detailed explanation for engineers:**
        Iterates all 20 problems. For each, runs baseline (no CSM) and
        with-CSM (rollback on violations). Aggregates metrics across all
        problems into a per-workflow summary.

    Args:
        workflow_name: Human-readable name ('math', 'code', 'planning').
        problems: List of workflow problem instances.
        pipeline: Shared _SingleArgCompatPipeline.

    Returns:
        Dict with per-problem traces and aggregate metrics.
    """
    baseline_results: list[dict[str, Any]] = []
    csm_results: list[dict[str, Any]] = []

    for prob in problems:
        steps_seq = faulty_step_sequence(
            prob.correct_steps,
            prob.faulty_step_index,
            prob.faulty_step_text,
            prob.propagated_steps,
        )

        b_result = run_baseline(
            problem_id=prob.problem_id,
            question=prob.question,
            step_names=prob.step_names,
            steps_seq=steps_seq,
            correct_output=prob.correct_output,
        )
        baseline_results.append(b_result)

        c_result = run_with_csm(
            problem_id=prob.problem_id,
            question=prob.question,
            step_names=prob.step_names,
            correct_steps=prob.correct_steps,
            faulty_step_index=prob.faulty_step_index,
            faulty_step_text=prob.faulty_step_text,
            propagated_steps=prob.propagated_steps,
            correct_output=prob.correct_output,
            pipeline=pipeline,
        )
        csm_results.append(c_result)

    n = len(problems)

    acc_baseline = sum(1 for r in baseline_results if r["is_correct"]) / n
    acc_csm = sum(1 for r in csm_results if r["is_correct"]) / n
    avg_steps_baseline = sum(r["steps_to_completion"] for r in baseline_results) / n
    avg_steps_csm = sum(r["steps_to_completion"] for r in csm_results) / n
    rollbacks_triggered = sum(r["rollbacks_triggered"] for r in csm_results)
    rollbacks_missed = sum(r.get("rollback_missed", 0) for r in csm_results)

    agg_violations: dict[str, int] = {}
    for r in csm_results:
        for step_name, count in r["violations_per_step"].items():
            agg_violations[step_name] = agg_violations.get(step_name, 0) + count

    return {
        "workflow": workflow_name,
        "n_problems": n,
        "metrics": {
            "final_accuracy_baseline": round(acc_baseline, 4),
            "final_accuracy_with_csm": round(acc_csm, 4),
            "improvement": round(acc_csm - acc_baseline, 4),
            "avg_steps_baseline": round(avg_steps_baseline, 2),
            "avg_steps_with_csm": round(avg_steps_csm, 2),
            "rollbacks_triggered": rollbacks_triggered,
            "rollbacks_missed": rollbacks_missed,
            "violations_per_step": agg_violations,
        },
        "baseline_results": baseline_results,
        "csm_results": csm_results,
    }


# ===========================================================================
# Main entrypoint
# ===========================================================================


def main() -> None:
    """Run Experiment 127: agent workflow verification benchmark.

    **Detailed explanation for engineers:**
        1. Build 20 problems for each of the three workflow types.
        2. Run baseline + CSM benchmarks for all three.
        3. Compute aggregate metrics.
        4. Save results to results/experiment_127_results.json.
    """
    t_start = time.time()
    print("Experiment 127: Agent Workflow Verification Benchmark")
    print("=" * 60)

    pipeline = _SingleArgCompatPipeline()

    print("Building problem sets...")
    math_problems = build_math_problems()
    code_problems = build_code_problems()
    planning_problems = build_planning_problems()
    print(f"  Math:     {len(math_problems)} problems")
    print(f"  Code:     {len(code_problems)} problems")
    print(f"  Planning: {len(planning_problems)} problems")

    print("\nRunning math workflow benchmark...")
    math_results = run_workflow_benchmark("math", math_problems, pipeline)
    m = math_results["metrics"]
    print(
        f"  Baseline acc={m['final_accuracy_baseline']:.2%}  "
        f"CSM acc={m['final_accuracy_with_csm']:.2%}  "
        f"improvement={m['improvement']:+.2%}  "
        f"rollbacks={m['rollbacks_triggered']}  missed={m['rollbacks_missed']}"
    )
    print(f"  Violations per step: {m['violations_per_step']}")

    print("Running code workflow benchmark...")
    code_results = run_workflow_benchmark("code", code_problems, pipeline)
    m = code_results["metrics"]
    print(
        f"  Baseline acc={m['final_accuracy_baseline']:.2%}  "
        f"CSM acc={m['final_accuracy_with_csm']:.2%}  "
        f"improvement={m['improvement']:+.2%}  "
        f"rollbacks={m['rollbacks_triggered']}  missed={m['rollbacks_missed']}"
    )
    print(f"  Violations per step: {m['violations_per_step']}")

    print("Running planning workflow benchmark...")
    planning_results = run_workflow_benchmark("planning", planning_problems, pipeline)
    m = planning_results["metrics"]
    print(
        f"  Baseline acc={m['final_accuracy_baseline']:.2%}  "
        f"CSM acc={m['final_accuracy_with_csm']:.2%}  "
        f"improvement={m['improvement']:+.2%}  "
        f"rollbacks={m['rollbacks_triggered']}  missed={m['rollbacks_missed']}"
    )
    print(f"  Violations per step: {m['violations_per_step']}")

    # Aggregate across all workflows.
    all_workflows = [math_results, code_results, planning_results]
    total_n = sum(w["n_problems"] for w in all_workflows)
    total_correct_baseline = sum(
        sum(1 for r in w["baseline_results"] if r["is_correct"])
        for w in all_workflows
    )
    total_correct_csm = sum(
        sum(1 for r in w["csm_results"] if r["is_correct"])
        for w in all_workflows
    )
    agg_acc_baseline = total_correct_baseline / total_n
    agg_acc_csm = total_correct_csm / total_n
    agg_rollbacks = sum(w["metrics"]["rollbacks_triggered"] for w in all_workflows)
    agg_missed = sum(w["metrics"]["rollbacks_missed"] for w in all_workflows)

    aggregate = {
        "total_problems": total_n,
        "total_correct_baseline": total_correct_baseline,
        "total_correct_csm": total_correct_csm,
        "overall_accuracy_baseline": round(agg_acc_baseline, 4),
        "overall_accuracy_with_csm": round(agg_acc_csm, 4),
        "overall_improvement": round(agg_acc_csm - agg_acc_baseline, 4),
        "total_rollbacks_triggered": agg_rollbacks,
        "total_rollbacks_missed": agg_missed,
    }

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print(f"  Total problems:      {total_n}")
    print(f"  Baseline accuracy:   {agg_acc_baseline:.2%}")
    print(f"  CSM accuracy:        {agg_acc_csm:.2%}")
    print(f"  Improvement:         {agg_acc_csm - agg_acc_baseline:+.2%}")
    print(f"  Rollbacks triggered: {agg_rollbacks}")
    print(f"  Rollbacks missed:    {agg_missed}")

    output = {
        "experiment": "127",
        "title": "Agent Workflow Verification Benchmark",
        "description": (
            "Three workflow types (math 4-step, code 3-step, planning 5-step) × "
            "20 problems each. Baseline (no CSM) vs. CSM with rollback. "
            "All faulty steps contain false +/- arithmetic detectable by "
            "ArithmeticExtractor."
        ),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "aggregate": aggregate,
        "workflows": {
            "math": math_results,
            "code": code_results,
            "planning": planning_results,
        },
    }

    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "experiment_127_results.json")
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {output['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
