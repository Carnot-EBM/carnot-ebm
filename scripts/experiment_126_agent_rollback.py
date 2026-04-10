#!/usr/bin/env python3
"""Experiment 126: Agent Rollback on Multi-Step Reasoning Tasks.

**Researcher summary:**
    Exp 125 built the ConstraintStateMachine with rollback capability. This
    experiment stress-tests rollback on structured multi-step math reasoning:
    20 problems × 4 steps each, with deliberate arithmetic errors injected at
    step 2 or step 3. Errors propagate into downstream steps (as they would
    in a real agent), and we measure whether rollback + constraint feedback
    recovers correct answers vs. running without rollback.

**Task structure (4 steps per problem):**
    Step 1: Extract given values — agent states the knowns from the problem.
    Step 2: Compute intermediate result — agent performs first arithmetic.
    Step 3: Apply final operation — agent applies the second arithmetic.
    Step 4: State answer — agent summarizes the final result.

**Error injection and propagation:**
    Each problem has a "faulty step" (step 2 or 3) with wrong arithmetic
    that the ArithmeticExtractor can detect (e.g., "6 * 8 = 40" instead of 48).

    Crucially, errors cascade into downstream steps (as they would in a real
    agent that uses the wrong intermediate value):
    - Error at step 2 -> step 3 uses wrong intermediate -> step 4 states
      wrong answer.
    - Error at step 3 -> step 4 states wrong answer.

    This means without rollback, a wrong intermediate always corrupts the
    final answer, giving a genuine accuracy difference to measure.

**Rollback protocol:**
    - ConstraintStateMachine processes each step's text output.
    - If a violation is detected at step N (verified=False with violations),
      rollback to step N-1, then re-inject the CORRECT text for step N.
    - Remaining steps continue on the now-corrected state.
    - If violation is NOT detected (missed), error propagates unchecked.

**Metrics:**
    - accuracy_no_rollback: fraction correct when errors propagate unchecked.
    - accuracy_with_rollback: fraction correct after rollback+repair.
    - rollback_detected_count: how many errors the CSM detected and recovered.
    - rollback_missed_count: how many errors slipped through undetected.
    - improvement: accuracy_with_rollback - accuracy_no_rollback.

**Why this matters:**
    Real agent frameworks propagate errors forward — a wrong intermediate result
    silently corrupts all downstream steps. Rollback converts detected violations
    into second chances. This experiment quantifies the precision of that recovery.

Data flow:
    1. Build 20 problems with correct steps + faulty step + error-propagated steps.
    2. For each problem, run "no-rollback baseline": feed steps with error
       propagated into downstream steps; measure step-4 answer accuracy.
    3. Run "with-rollback": on violation at step N, rollback to N-1, re-inject
       correct output, then run remaining steps correctly.
    4. Save per-problem trace + aggregate metrics to results/experiment_126_results.json.

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
# Force JAX to CPU before any carnot imports (avoid ROCm/CUDA instability).
# ---------------------------------------------------------------------------
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Extend path so we can import carnot from the source tree.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from carnot.pipeline.state_machine import ConstraintStateMachine  # noqa: E402
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline  # noqa: E402


class _SingleArgCompatPipeline(VerifyRepairPipeline):
    """VerifyRepairPipeline subclass compatible with agentic.propagate().

    **Detailed explanation for engineers:**
        The internal agentic.propagate() function calls
        ``pipeline.verify(step.output_text)`` with a single positional
        argument.  The standard ``VerifyRepairPipeline.verify()`` requires
        two positional arguments (question, response).

        This subclass overrides verify() to accept either one or two
        positional arguments.  When called with a single argument, the
        argument is treated as the response text and an empty string is
        used as the question (matching the propagate() caller's intent:
        verify the agent's output, not a Q/A pair).

        This is a compatibility shim for Exp 126 only — it does not modify
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
            # Called as verify(output_text) from agentic.propagate().
            return super().verify("", question_or_response, domain)
        return super().verify(question_or_response, response, domain)


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

@dataclass
class MathProblem:
    """One 4-step arithmetic problem with correct, faulty, and propagated steps.

    **Detailed explanation for engineers:**
        Each problem has a question, four correct step outputs, a faulty
        version of one step, and "propagated" steps — what downstream steps
        look like when the error carries forward (as a real agent would do).

        Example for step-2 error (baker problem):
            correct_steps[1] = "Intermediate: 6 * 8 = 48 total cookies."
            faulty_step_text = "Intermediate: 6 * 8 = 40 total cookies."
            propagated_steps = [
                "Final operation: 40 - 12 = 28 cookies remaining.",  # step 3 wrong
                "Answer: The baker has 28 cookies left.",              # step 4 wrong
            ]

        In no-rollback mode, the faulty step + propagated_steps replace the
        corresponding correct steps. In with-rollback mode, if the CSM detects
        the violation, we rollback and inject correct steps instead.

    Attributes:
        problem_id: 1-based index.
        question: Natural-language problem statement.
        correct_steps: All 4 correct step outputs (indices 0..3).
        faulty_step_index: Which step has the error (1-based, 2 or 3).
        faulty_step_text: The incorrect output for that step.
        propagated_steps: Downstream step outputs if error propagates.
            If faulty_step_index=2, this is [step3_wrong, step4_wrong].
            If faulty_step_index=3, this is [step4_wrong].
        correct_answer: Ground-truth integer final answer.
        error_answer: What wrong answer the error path produces.
    """
    problem_id: int
    question: str
    correct_steps: list[str]     # Correct outputs for steps 1..4
    faulty_step_index: int       # 1-based (2 or 3)
    faulty_step_text: str        # Injected-error output for that step
    propagated_steps: list[str]  # Downstream steps if error propagates
    correct_answer: int
    error_answer: int            # Wrong answer from propagated error path


def build_problems() -> list[MathProblem]:
    """Build 20 structured 4-step math problems with error propagation.

    **Detailed explanation for engineers:**
        Ten problems have errors at step 2 (wrong intermediate); ten at step 3
        (wrong final operation). Each includes pre-baked "propagated_steps"
        that represent what downstream steps look like when the error carries
        forward into subsequent computation.

        The faulty text encodes a wrong arithmetic expression so the
        ArithmeticExtractor (checking "A op B = C" equalities) fires a
        violation (e.g., "6 * 8 = 40" is detected as wrong since 6*8=48).

    Returns:
        List of 20 MathProblem instances.
    """
    raw: list[dict[str, Any]] = [
        # === Errors at step 2 (problems 1-10) ===
        # When step 2 gives the wrong intermediate, step 3 and step 4 cascade.
        {
            "question": "A baker makes 6 trays of cookies. Each tray has 8 cookies. He sells 12 cookies. How many are left?",
            "correct_steps": [
                "Given: 6 trays, 8 cookies per tray, 12 sold.",
                "Intermediate: 6 * 8 = 48 total cookies.",
                "Final operation: 48 - 12 = 36 cookies remaining.",
                "Answer: The baker has 36 cookies left.",
            ],
            "faulty_step_index": 2,
            "faulty_step_text": "Intermediate: 6 * 8 = 40 total cookies.",
            # If 40 is accepted, step 3 uses 40, step 4 states wrong answer.
            "propagated_steps": [
                "Final operation: 40 - 12 = 28 cookies remaining.",
                "Answer: The baker has 28 cookies left.",
            ],
            "correct_answer": 36,
            "error_answer": 28,
        },
        {
            "question": "A box holds 9 apples. There are 5 boxes. 7 apples are eaten. How many remain?",
            "correct_steps": [
                "Given: 9 apples per box, 5 boxes, 7 eaten.",
                "Intermediate: 9 * 5 = 45 total apples.",
                "Final operation: 45 - 7 = 38 remaining.",
                "Answer: There are 38 apples remaining.",
            ],
            "faulty_step_index": 2,
            "faulty_step_text": "Intermediate: 9 * 5 = 42 total apples.",
            "propagated_steps": [
                "Final operation: 42 - 7 = 35 remaining.",
                "Answer: There are 35 apples remaining.",
            ],
            "correct_answer": 38,
            "error_answer": 35,
        },
        {
            "question": "Each shelf holds 7 books. There are 4 shelves. 5 books are removed. How many books are left?",
            "correct_steps": [
                "Given: 7 books per shelf, 4 shelves, 5 removed.",
                "Intermediate: 7 * 4 = 28 total books.",
                "Final operation: 28 - 5 = 23 remaining.",
                "Answer: There are 23 books left.",
            ],
            "faulty_step_index": 2,
            "faulty_step_text": "Intermediate: 7 * 4 = 24 total books.",
            "propagated_steps": [
                "Final operation: 24 - 5 = 19 remaining.",
                "Answer: There are 19 books left.",
            ],
            "correct_answer": 23,
            "error_answer": 19,
        },
        {
            "question": "A store sells pencils for $3 each. They sell 11 pencils. They spend $8 on supplies. What is the profit?",
            "correct_steps": [
                "Given: $3 per pencil, 11 sold, $8 cost.",
                "Intermediate: 3 * 11 = 33 dollars revenue.",
                "Final operation: 33 - 8 = 25 dollars profit.",
                "Answer: The profit is $25.",
            ],
            "faulty_step_index": 2,
            "faulty_step_text": "Intermediate: 3 * 11 = 30 dollars revenue.",
            "propagated_steps": [
                "Final operation: 30 - 8 = 22 dollars profit.",
                "Answer: The profit is $22.",
            ],
            "correct_answer": 25,
            "error_answer": 22,
        },
        {
            "question": "A garden has 8 rows of plants. Each row has 6 plants. 10 plants die. How many plants remain?",
            "correct_steps": [
                "Given: 8 rows, 6 plants per row, 10 die.",
                "Intermediate: 8 * 6 = 48 total plants.",
                "Final operation: 48 - 10 = 38 remaining.",
                "Answer: There are 38 plants remaining.",
            ],
            "faulty_step_index": 2,
            "faulty_step_text": "Intermediate: 8 * 6 = 46 total plants.",
            "propagated_steps": [
                "Final operation: 46 - 10 = 36 remaining.",
                "Answer: There are 36 plants remaining.",
            ],
            "correct_answer": 38,
            "error_answer": 36,
        },
        {
            "question": "A farmer collects 5 eggs from each of 9 hens daily. He uses 15 eggs. How many does he have left?",
            "correct_steps": [
                "Given: 5 eggs per hen, 9 hens, 15 used.",
                "Intermediate: 5 * 9 = 45 total eggs.",
                "Final operation: 45 - 15 = 30 eggs remaining.",
                "Answer: The farmer has 30 eggs left.",
            ],
            "faulty_step_index": 2,
            "faulty_step_text": "Intermediate: 5 * 9 = 40 total eggs.",
            "propagated_steps": [
                "Final operation: 40 - 15 = 25 eggs remaining.",
                "Answer: The farmer has 25 eggs left.",
            ],
            "correct_answer": 30,
            "error_answer": 25,
        },
        {
            "question": "A factory produces 12 units per hour. It runs for 4 hours. 8 units are defective. How many good units?",
            "correct_steps": [
                "Given: 12 units/hour, 4 hours, 8 defective.",
                "Intermediate: 12 * 4 = 48 total units.",
                "Final operation: 48 - 8 = 40 good units.",
                "Answer: There are 40 good units.",
            ],
            "faulty_step_index": 2,
            "faulty_step_text": "Intermediate: 12 * 4 = 44 total units.",
            "propagated_steps": [
                "Final operation: 44 - 8 = 36 good units.",
                "Answer: There are 36 good units.",
            ],
            "correct_answer": 40,
            "error_answer": 36,
        },
        {
            "question": "A classroom has 7 tables. Each table seats 4 students. 6 students are absent. How many are present?",
            "correct_steps": [
                "Given: 7 tables, 4 seats each, 6 absent.",
                "Intermediate: 7 * 4 = 28 total seats.",
                "Final operation: 28 - 6 = 22 students present.",
                "Answer: There are 22 students present.",
            ],
            "faulty_step_index": 2,
            "faulty_step_text": "Intermediate: 7 * 4 = 25 total seats.",
            "propagated_steps": [
                "Final operation: 25 - 6 = 19 students present.",
                "Answer: There are 19 students present.",
            ],
            "correct_answer": 22,
            "error_answer": 19,
        },
        {
            "question": "A team scores 3 points per game. They play 9 games. They receive a 5-point bonus. Total points?",
            "correct_steps": [
                "Given: 3 points/game, 9 games, 5 bonus.",
                "Intermediate: 3 * 9 = 27 base points.",
                "Final operation: 27 + 5 = 32 total points.",
                "Answer: The team has 32 total points.",
            ],
            "faulty_step_index": 2,
            "faulty_step_text": "Intermediate: 3 * 9 = 24 base points.",
            "propagated_steps": [
                "Final operation: 24 + 5 = 29 total points.",
                "Answer: The team has 29 total points.",
            ],
            "correct_answer": 32,
            "error_answer": 29,
        },
        {
            "question": "A driver travels 60 miles per hour for 3 hours, then drives 15 more miles. Total distance?",
            "correct_steps": [
                "Given: 60 mph, 3 hours, then 15 more miles.",
                "Intermediate: 60 * 3 = 180 miles in first segment.",
                "Final operation: 180 + 15 = 195 total miles.",
                "Answer: The driver travels 195 miles in total.",
            ],
            "faulty_step_index": 2,
            "faulty_step_text": "Intermediate: 60 * 3 = 170 miles in first segment.",
            "propagated_steps": [
                "Final operation: 170 + 15 = 185 total miles.",
                "Answer: The driver travels 185 miles in total.",
            ],
            "correct_answer": 195,
            "error_answer": 185,
        },
        # === Errors at step 3 (problems 11-20) ===
        # Step 2 is correct; step 3 applies the wrong operation; step 4 echoes the wrong answer.
        {
            "question": "A store buys 50 items at $2 each and sells them for a $30 total profit. Revenue?",
            "correct_steps": [
                "Given: 50 items, cost $2 each, profit $30.",
                "Intermediate: 50 * 2 = 100 dollars cost.",
                "Final operation: 100 + 30 = 130 dollars revenue.",
                "Answer: The revenue is $130.",
            ],
            "faulty_step_index": 3,
            "faulty_step_text": "Final operation: 100 + 30 = 125 dollars revenue.",
            "propagated_steps": [
                "Answer: The revenue is $125.",
            ],
            "correct_answer": 130,
            "error_answer": 125,
        },
        {
            "question": "A swimmer does 15 laps per day for 6 days. She rests and swims 10 more laps. Total laps?",
            "correct_steps": [
                "Given: 15 laps/day, 6 days, 10 extra laps.",
                "Intermediate: 15 * 6 = 90 laps base.",
                "Final operation: 90 + 10 = 100 total laps.",
                "Answer: The swimmer swims 100 laps in total.",
            ],
            "faulty_step_index": 3,
            "faulty_step_text": "Final operation: 90 + 10 = 95 total laps.",
            "propagated_steps": [
                "Answer: The swimmer swims 95 laps in total.",
            ],
            "correct_answer": 100,
            "error_answer": 95,
        },
        {
            "question": "A printer prints 8 pages per minute for 7 minutes. Then prints 16 extra pages. Total pages?",
            "correct_steps": [
                "Given: 8 pages/min, 7 minutes, 16 extra.",
                "Intermediate: 8 * 7 = 56 pages initially.",
                "Final operation: 56 + 16 = 72 total pages.",
                "Answer: The printer outputs 72 pages total.",
            ],
            "faulty_step_index": 3,
            "faulty_step_text": "Final operation: 56 + 16 = 68 total pages.",
            "propagated_steps": [
                "Answer: The printer outputs 68 pages total.",
            ],
            "correct_answer": 72,
            "error_answer": 68,
        },
        {
            "question": "A worker earns $14 per hour. Works 8 hours. Pays $20 tax. Take-home pay?",
            "correct_steps": [
                "Given: $14/hour, 8 hours, $20 tax.",
                "Intermediate: 14 * 8 = 112 dollars gross.",
                "Final operation: 112 - 20 = 92 dollars take-home.",
                "Answer: Take-home pay is $92.",
            ],
            "faulty_step_index": 3,
            "faulty_step_text": "Final operation: 112 - 20 = 88 dollars take-home.",
            "propagated_steps": [
                "Answer: Take-home pay is $88.",
            ],
            "correct_answer": 92,
            "error_answer": 88,
        },
        {
            "question": "A cyclist rides 11 km per hour for 5 hours, then rides 8 more km. Total distance?",
            "correct_steps": [
                "Given: 11 km/h, 5 hours, 8 more km.",
                "Intermediate: 11 * 5 = 55 km first leg.",
                "Final operation: 55 + 8 = 63 km total.",
                "Answer: The cyclist travels 63 km in total.",
            ],
            "faulty_step_index": 3,
            "faulty_step_text": "Final operation: 55 + 8 = 59 km total.",
            "propagated_steps": [
                "Answer: The cyclist travels 59 km in total.",
            ],
            "correct_answer": 63,
            "error_answer": 59,
        },
        {
            "question": "A recipe uses 4 cups of flour per batch. They make 7 batches then give away 9 cups. Cups remaining?",
            "correct_steps": [
                "Given: 4 cups/batch, 7 batches, 9 cups given away.",
                "Intermediate: 4 * 7 = 28 cups total.",
                "Final operation: 28 - 9 = 19 cups remaining.",
                "Answer: There are 19 cups remaining.",
            ],
            "faulty_step_index": 3,
            "faulty_step_text": "Final operation: 28 - 9 = 21 cups remaining.",
            "propagated_steps": [
                "Answer: There are 21 cups remaining.",
            ],
            "correct_answer": 19,
            "error_answer": 21,
        },
        {
            "question": "A pool holds 200 liters. It fills at 25 liters per hour for 6 hours. How much total water?",
            "correct_steps": [
                "Given: 200 liters initial, 25 liters/hour, 6 hours.",
                "Intermediate: 25 * 6 = 150 liters added.",
                "Final operation: 200 + 150 = 350 liters total.",
                "Answer: The pool contains 350 liters total.",
            ],
            "faulty_step_index": 3,
            "faulty_step_text": "Final operation: 200 + 150 = 330 liters total.",
            "propagated_steps": [
                "Answer: The pool contains 330 liters total.",
            ],
            "correct_answer": 350,
            "error_answer": 330,
        },
        {
            "question": "A bag contains 100 marbles. Kids take 6 marbles each over 8 turns. How many remain?",
            "correct_steps": [
                "Given: 100 marbles, 6 per turn, 8 turns.",
                "Intermediate: 6 * 8 = 48 marbles taken.",
                "Final operation: 100 - 48 = 52 marbles left.",
                "Answer: There are 52 marbles remaining.",
            ],
            "faulty_step_index": 3,
            "faulty_step_text": "Final operation: 100 - 48 = 56 marbles left.",
            "propagated_steps": [
                "Answer: There are 56 marbles remaining.",
            ],
            "correct_answer": 52,
            "error_answer": 56,
        },
        {
            "question": "A warehouse stores 300 boxes. Workers add 9 boxes per shift over 5 shifts. Total boxes?",
            "correct_steps": [
                "Given: 300 boxes initial, 9 added per shift, 5 shifts.",
                "Intermediate: 9 * 5 = 45 boxes added.",
                "Final operation: 300 + 45 = 345 boxes total.",
                "Answer: There are 345 boxes in the warehouse.",
            ],
            "faulty_step_index": 3,
            "faulty_step_text": "Final operation: 300 + 45 = 340 boxes total.",
            "propagated_steps": [
                "Answer: There are 340 boxes in the warehouse.",
            ],
            "correct_answer": 345,
            "error_answer": 340,
        },
        {
            "question": "A library has 500 books. Donates 7 books per week over 9 weeks. How many remain?",
            "correct_steps": [
                "Given: 500 books, 7 donated/week, 9 weeks.",
                "Intermediate: 7 * 9 = 63 books donated.",
                "Final operation: 500 - 63 = 437 books remaining.",
                "Answer: The library has 437 books remaining.",
            ],
            "faulty_step_index": 3,
            "faulty_step_text": "Final operation: 500 - 63 = 430 books remaining.",
            "propagated_steps": [
                "Answer: The library has 430 books remaining.",
            ],
            "correct_answer": 437,
            "error_answer": 430,
        },
    ]

    problems: list[MathProblem] = []
    for i, r in enumerate(raw, start=1):
        problems.append(
            MathProblem(
                problem_id=i,
                question=r["question"],
                correct_steps=r["correct_steps"],
                faulty_step_index=r["faulty_step_index"],
                faulty_step_text=r["faulty_step_text"],
                propagated_steps=r["propagated_steps"],
                correct_answer=r["correct_answer"],
                error_answer=r["error_answer"],
            )
        )
    return problems


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(step4_text: str) -> int | None:
    """Extract the integer answer from step 4 output text.

    **Detailed explanation for engineers:**
        Step 4 always ends with "Answer: ... <N> ..." where N is the integer
        result. We scan for the last integer in the text, which is robust to
        units like "dollars", "laps", "km", etc.

        Returns None if no integer can be found (signals a failed step).

    Args:
        step4_text: The agent output for step 4.

    Returns:
        The extracted integer answer, or None.
    """
    import re
    matches = re.findall(r"\b(\d+)\b", step4_text)
    if matches:
        return int(matches[-1])
    return None


# ---------------------------------------------------------------------------
# Build step sequences for no-rollback and with-rollback runs
# ---------------------------------------------------------------------------

def no_rollback_step_sequence(problem: MathProblem) -> list[str]:
    """Return the 4-step sequence for the no-rollback (error propagates) run.

    **Detailed explanation for engineers:**
        Replaces the faulty step and all subsequent correct steps with the
        error-propagated versions. The result is a 4-step sequence where
        the error cascades through to the final answer:

        faulty_step_index=2:
            [correct_steps[0], faulty_step_text, propagated_steps[0], propagated_steps[1]]
        faulty_step_index=3:
            [correct_steps[0], correct_steps[1], faulty_step_text, propagated_steps[0]]

    Args:
        problem: The MathProblem instance.

    Returns:
        List of 4 step-output strings for the no-rollback run.
    """
    seq = list(problem.correct_steps)  # start with correct steps

    fi = problem.faulty_step_index  # 1-based
    # Replace faulty step.
    seq[fi - 1] = problem.faulty_step_text
    # Replace all steps after the faulty step with propagated versions.
    for offset, propagated_text in enumerate(problem.propagated_steps):
        seq[fi + offset] = propagated_text

    return seq


# ---------------------------------------------------------------------------
# Run one problem — no rollback baseline
# ---------------------------------------------------------------------------

def run_no_rollback(
    problem: MathProblem, pipeline: _SingleArgCompatPipeline
) -> dict[str, Any]:
    """Run a problem through the CSM without rollback.

    **Detailed explanation for engineers:**
        Feeds all 4 steps through a fresh ConstraintStateMachine. At the
        faulty step index, the buggy text is used and subsequent steps
        carry the error forward. We do NOT check for violations — this
        simulates a naive agent that propagates errors forward unchecked.

        We measure whether the final step 4 answer matches the correct answer.
        Since the error propagates into step 4's text, no-rollback will
        typically produce an incorrect final answer.

    Args:
        problem: The MathProblem to run.
        pipeline: Shared _SingleArgCompatPipeline (verify-only, no LLM).

    Returns:
        Dict with per-step verification results and correctness flag.
    """
    csm = ConstraintStateMachine(pipeline)
    steps_seq = no_rollback_step_sequence(problem)
    step_results: list[dict[str, Any]] = []
    final_text = ""

    for step_num in range(1, 5):
        output_text = steps_seq[step_num - 1]
        input_text = problem.question if step_num == 1 else f"Step {step_num}"
        result = csm.step(input_text, output_text)

        step_results.append({
            "step": step_num,
            "output_text": output_text,
            "verified": result.verification.verified,
            "n_violations": len(result.verification.violations),
            "violations": [v.description for v in result.verification.violations],
            "n_new_facts": len(result.new_facts),
            "contradictions": result.contradictions,
        })

        if step_num == 4:
            final_text = output_text

    extracted = extract_answer(final_text)
    is_correct = extracted == problem.correct_answer

    return {
        "problem_id": problem.problem_id,
        "mode": "no_rollback",
        "faulty_step_index": problem.faulty_step_index,
        "correct_answer": problem.correct_answer,
        "error_answer": problem.error_answer,
        "extracted_answer": extracted,
        "is_correct": is_correct,
        "steps": step_results,
    }


# ---------------------------------------------------------------------------
# Run one problem — with rollback
# ---------------------------------------------------------------------------

def run_with_rollback(
    problem: MathProblem, pipeline: _SingleArgCompatPipeline
) -> dict[str, Any]:
    """Run a problem through the CSM with rollback on detected violations.

    **Detailed explanation for engineers:**
        Feeds steps through a fresh ConstraintStateMachine. At the faulty
        step index, the buggy text is submitted. If the CSM reports a
        violation (verified=False with at least one violation), we:
          1. Rollback to step N-1 (the last known-good state).
          2. Re-run step N with the correct text (constraint-guided repair).
          3. Continue all subsequent steps with the correct step texts.

        If the CSM does NOT detect the violation (missed detection),
        subsequent steps propagate the error (same as no-rollback mode).

        Step 4 always reflects whatever state step 3 produced —
        correct if rollback succeeded, wrong if error slipped through.

    Args:
        problem: The MathProblem to run.
        pipeline: Shared _SingleArgCompatPipeline (verify-only, no LLM).

    Returns:
        Dict with per-step trace, rollback events, and correctness flag.
    """
    csm = ConstraintStateMachine(pipeline)
    steps_seq = no_rollback_step_sequence(problem)  # start with error path
    step_results: list[dict[str, Any]] = []
    rollback_events: list[dict[str, Any]] = []
    final_text = ""

    # Track rollback state so subsequent steps use correct path on success.
    rollback_triggered = False
    rollback_detected = False

    for step_num in range(1, 5):
        idx = step_num - 1

        # Decide which text to use for this step.
        if step_num == problem.faulty_step_index and not rollback_triggered:
            # First attempt at the faulty step: inject the buggy text.
            output_text = problem.faulty_step_text
        elif rollback_triggered:
            # After a successful rollback, use correct steps for all remaining.
            output_text = problem.correct_steps[idx]
        else:
            # Before the faulty step (or missed detection): use error-path text.
            output_text = steps_seq[idx]

        input_text = problem.question if step_num == 1 else f"Step {step_num}"
        result = csm.step(input_text, output_text)

        step_info: dict[str, Any] = {
            "step": step_num,
            "output_text": output_text,
            "verified": result.verification.verified,
            "n_violations": len(result.verification.violations),
            "violations": [v.description for v in result.verification.violations],
            "n_new_facts": len(result.new_facts),
            "contradictions": result.contradictions,
            "rolled_back": False,
            "is_repair": False,
        }

        # Check for violation at the faulty step and attempt rollback.
        if (
            step_num == problem.faulty_step_index
            and not rollback_triggered
            and not result.verification.verified
            and len(result.verification.violations) > 0
        ):
            rollback_detected = True
            rollback_triggered = True

            # Rollback to step N-1 (0-based index of step N-1 in history).
            rollback_target = step_num - 2  # 0-based

            if rollback_target >= 0:
                csm.rollback(rollback_target)
            else:
                # Error was at step 1: reset to fresh CSM.
                csm = ConstraintStateMachine(pipeline)

            rollback_events.append({
                "at_step": step_num,
                "rolled_back_to": max(0, step_num - 1),
                "violations": [v.description for v in result.verification.violations],
            })
            step_info["rolled_back"] = True
            step_results.append(step_info)

            # Re-run faulty step with the CORRECT text (constraint-guided repair).
            correct_text = problem.correct_steps[idx]
            repair_result = csm.step(input_text, correct_text)

            step_results.append({
                "step": step_num,
                "output_text": correct_text,
                "verified": repair_result.verification.verified,
                "n_violations": len(repair_result.verification.violations),
                "violations": [v.description for v in repair_result.verification.violations],
                "n_new_facts": len(repair_result.new_facts),
                "contradictions": repair_result.contradictions,
                "rolled_back": False,
                "is_repair": True,
            })

            if step_num == 4:
                final_text = correct_text
        else:
            step_results.append(step_info)
            if step_num == 4:
                final_text = output_text

    extracted = extract_answer(final_text)
    is_correct = extracted == problem.correct_answer

    return {
        "problem_id": problem.problem_id,
        "mode": "with_rollback",
        "faulty_step_index": problem.faulty_step_index,
        "correct_answer": problem.correct_answer,
        "error_answer": problem.error_answer,
        "extracted_answer": extracted,
        "is_correct": is_correct,
        "rollback_detected": rollback_detected,
        "rollback_events": rollback_events,
        "steps": step_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 126: Agent rollback on 20 multi-step math problems.

    **Detailed explanation for engineers:**
        1. Build 20 math problems (10 with error at step 2, 10 at step 3).
           Errors cascade into downstream steps so no-rollback gives wrong answers.
        2. Initialize a shared VerifyRepairPipeline in verify-only mode (no LLM).
        3. For each problem, run both modes: no-rollback baseline and with-rollback.
        4. Compute aggregate metrics: accuracy with/without rollback, detection rate.
        5. Save all results + metrics to results/experiment_126_results.json.
    """
    print("=" * 70)
    print("Experiment 126: Agent Rollback on Multi-Step Reasoning Tasks")
    print("=" * 70)

    t_start = time.time()

    problems = build_problems()
    n_s2 = sum(1 for p in problems if p.faulty_step_index == 2)
    n_s3 = sum(1 for p in problems if p.faulty_step_index == 3)
    print(f"Built {len(problems)} problems ({n_s2} step-2 errors, {n_s3} step-3 errors).")

    # Single shared pipeline in verify-only mode (no LLM needed).
    # Use the compatibility subclass so agentic.propagate() can call verify()
    # with a single argument (response only) as it does internally.
    print("Initializing VerifyRepairPipeline (verify-only, no model)...")
    pipeline = _SingleArgCompatPipeline(domains=["arithmetic"])
    print("Pipeline ready.")

    no_rollback_results: list[dict[str, Any]] = []
    with_rollback_results: list[dict[str, Any]] = []

    for problem in problems:
        print(f"\n--- Problem {problem.problem_id:02d} | Error at step {problem.faulty_step_index} "
              f"| Correct={problem.correct_answer}, ErrorPath={problem.error_answer} ---")
        print(f"  Q: {problem.question[:70]}...")

        # Baseline: no rollback — errors propagate into step 4.
        nr = run_no_rollback(problem, pipeline)
        no_rollback_results.append(nr)

        # With rollback — detect violation, rollback, inject correct step.
        wr = run_with_rollback(problem, pipeline)
        with_rollback_results.append(wr)

        nr_mark = "CORRECT" if nr["is_correct"] else f"WRONG({nr['extracted_answer']})"
        wr_mark = "CORRECT" if wr["is_correct"] else f"WRONG({wr['extracted_answer']})"
        rb_mark = "detected+fixed" if wr["rollback_detected"] else "missed"
        print(f"  No-rollback: {nr_mark}  |  With-rollback: {wr_mark}  |  Rollback: {rb_mark}")

    # --- Aggregate metrics ---
    n = len(problems)
    acc_no_rb = sum(1 for r in no_rollback_results if r["is_correct"]) / n
    acc_with_rb = sum(1 for r in with_rollback_results if r["is_correct"]) / n
    detected = sum(1 for r in with_rollback_results if r["rollback_detected"])
    missed = n - detected

    # Per-error-step-location subgroups.
    s2_ids = {p.problem_id for p in problems if p.faulty_step_index == 2}
    s3_ids = {p.problem_id for p in problems if p.faulty_step_index == 3}

    def subgroup_acc(results: list[dict[str, Any]], ids: set[int]) -> float:
        sub = [r for r in results if r["problem_id"] in ids]
        return sum(1 for r in sub if r["is_correct"]) / len(sub) if sub else 0.0

    acc_no_rb_s2 = subgroup_acc(no_rollback_results, s2_ids)
    acc_with_rb_s2 = subgroup_acc(with_rollback_results, s2_ids)
    acc_no_rb_s3 = subgroup_acc(no_rollback_results, s3_ids)
    acc_with_rb_s3 = subgroup_acc(with_rollback_results, s3_ids)

    # Detection rate per step group.
    det_s2 = sum(1 for r in with_rollback_results if r["problem_id"] in s2_ids and r["rollback_detected"])
    det_s3 = sum(1 for r in with_rollback_results if r["problem_id"] in s3_ids and r["rollback_detected"])

    t_elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Problems:                        {n}")
    print(f"  Accuracy (no rollback):          {acc_no_rb:.1%}")
    print(f"  Accuracy (with rollback):        {acc_with_rb:.1%}")
    print(f"  Improvement:                     +{(acc_with_rb - acc_no_rb):.1%}")
    print(f"  Rollback detected:               {detected}/{n}")
    print(f"  Rollback missed:                 {missed}/{n}")
    print(f"  Detection rate:                  {detected/n:.1%}")
    print(f"  Step-2 errors | no-rb: {acc_no_rb_s2:.1%}  with-rb: {acc_with_rb_s2:.1%}  detected: {det_s2}/{n_s2}")
    print(f"  Step-3 errors | no-rb: {acc_no_rb_s3:.1%}  with-rb: {acc_with_rb_s3:.1%}  detected: {det_s3}/{n_s3}")
    print(f"  Elapsed: {t_elapsed:.1f}s")

    # --- Save results ---
    output: dict[str, Any] = {
        "experiment": "Exp 126: Agent Rollback on Multi-Step Reasoning Tasks",
        "spec": ["REQ-VERIFY-001", "SCENARIO-VERIFY-005"],
        "n_problems": n,
        "metrics": {
            "accuracy_no_rollback": acc_no_rb,
            "accuracy_with_rollback": acc_with_rb,
            "accuracy_improvement": acc_with_rb - acc_no_rb,
            "rollback_detected_count": detected,
            "rollback_missed_count": missed,
            "detection_rate": detected / n,
            "accuracy_step2_no_rollback": acc_no_rb_s2,
            "accuracy_step2_with_rollback": acc_with_rb_s2,
            "accuracy_step2_detection_rate": det_s2 / n_s2,
            "accuracy_step3_no_rollback": acc_no_rb_s3,
            "accuracy_step3_with_rollback": acc_with_rb_s3,
            "accuracy_step3_detection_rate": det_s3 / n_s3,
        },
        "elapsed_seconds": t_elapsed,
        "no_rollback_results": no_rollback_results,
        "with_rollback_results": with_rollback_results,
    }

    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "experiment_126_results.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
