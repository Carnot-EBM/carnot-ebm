#!/usr/bin/env python3
"""Experiment 101: Agent Verification Workflows — End-to-end agentic constraint propagation.

**Researcher summary:**
    Validates the full agentic verification pipeline (ConstraintState propagation
    from Exp 99 + rollback from Exp 100) on realistic multi-step reasoning workflows.
    Three workflow templates (math tutor, code assistant, research assistant) test
    different reasoning patterns. Each template has 10 instances (5 correct, 5 with
    planted errors) to measure detection rate, detection latency, root cause accuracy,
    repair success, and false positive rate.

**Detailed explanation for engineers:**
    This experiment puts the agentic verification pipeline through its paces by
    simulating realistic agent workflows. Rather than using live LLM inference
    (which would be slow, non-reproducible, and require GPU), we use pre-defined
    workflow instances with known correct and incorrect steps. This lets us measure
    the pipeline's ability to detect errors, locate root causes, and repair chains
    against ground truth.

    The three workflow types exercise different extraction domains:
    - Math tutor: Arithmetic extraction (ArithmeticExtractor) — can the pipeline
      catch wrong computations in step 3 or wrong approach in step 2?
    - Code assistant: Code extraction (CodeExtractor) — can the pipeline catch
      incorrect implementations or wrong test cases?
    - Research assistant: NL/Logic extraction (NLExtractor, LogicExtractor) — can
      the pipeline catch misattributed facts or wrong citations?

    For each workflow instance, we:
    1. Run verify_chain() — propagate constraints through all 4 steps
    2. For error cases: run find_root_cause() and compare to known error step
    3. Attempt rollback_to() the error step and repair via repair_chain()
    4. Compare per-step verification vs final-answer-only verification

    These helper functions (verify_chain, find_root_cause, rollback_to, repair_chain)
    are implemented inline here since they were planned for Exp 99-100 but not yet
    in the library. They compose the existing propagate(), ConstraintState, and
    AgentStep primitives from carnot.pipeline.agentic.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_101_agent_verification.py

REQ: REQ-VERIFY-001, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Force CPU for reproducibility (see CLAUDE.md: ROCm JAX crashes, always use CPU).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from carnot.pipeline.agentic import (
    AgentStep,
    ConstraintState,
    FactStatus,
    TrackedFact,
    propagate,
)
from carnot.pipeline.extract import AutoExtractor
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Agentic workflow helpers (compose existing primitives from Exp 99-100)
# ---------------------------------------------------------------------------


def verify_chain(
    steps: list[AgentStep],
    pipeline: VerifyRepairPipeline,
) -> tuple[ConstraintState, list[VerificationResult]]:
    """Run constraint propagation across all steps in a chain.

    **Detailed explanation for engineers:**
        Iterates through each AgentStep, extracting constraints from the step's
        output text and verifying them via the pipeline. Facts are accumulated
        in the ConstraintState: satisfied constraints are marked VERIFIED,
        violated ones are marked VIOLATED, and cascade invalidation marks
        downstream ASSUMED facts as VIOLATED when an earlier step fails.

        NOTE: We re-implement the propagation loop here rather than calling
        the library's propagate() function because propagate() currently
        calls pipeline.verify(output_text) with only one argument, but
        verify() requires both question and response. This inline version
        correctly passes both arguments.

    Args:
        steps: Ordered list of AgentStep objects forming the reasoning chain.
        pipeline: The VerifyRepairPipeline used for extraction and verification.

    Returns:
        Tuple of (final ConstraintState, list of per-step VerificationResults).
    """
    from carnot.pipeline.agentic import _normalize_fact_key

    state = ConstraintState()
    results: list[VerificationResult] = []

    for step in steps:
        step.state_before = state.snapshot()

        # Extract constraints from this step's output.
        constraints = pipeline.extract_constraints(step.output_text)

        # Add each constraint as an ASSUMED fact.
        for cr in constraints:
            state.add_fact(cr.description, step_index=step.step_index, constraint=cr)

        # Verify the step's output (passing both question and response).
        vr = pipeline.verify(step.input_text, step.output_text)
        step.verification = vr
        results.append(vr)

        # Mark violated facts.
        violated_keys: set[str] = set()
        if vr.violations:
            for v in vr.violations:
                key = _normalize_fact_key(v.description)
                if key in state.facts:
                    state.violate_fact(key, step_index=step.step_index, energy=vr.energy)
                    violated_keys.add(key)

        # Mark satisfied facts (those not violated).
        for cr in constraints:
            key = _normalize_fact_key(cr.description)
            if key not in violated_keys and key in state.facts:
                if state.facts[key].status == FactStatus.ASSUMED:
                    state.verify_fact(key, step_index=step.step_index, energy=0.0)

        # Cascade invalidation: mark ASSUMED facts from later steps as VIOLATED.
        if violated_keys:
            for key, fact in state.facts.items():
                if (
                    key not in violated_keys
                    and fact.status == FactStatus.ASSUMED
                    and fact.introduced_at_step > step.step_index
                ):
                    state.violate_fact(key, step_index=step.step_index, energy=vr.energy)

        step.state_after = state.snapshot()

    return state, results


def find_root_cause(
    state: ConstraintState,
    steps: list[AgentStep],
) -> int | None:
    """Find the earliest step that introduced a violated fact.

    **Detailed explanation for engineers:**
        Scans all VIOLATED facts in the ConstraintState and returns the
        minimum introduced_at_step index. This is our best guess at where
        the error originated — it's the earliest step whose output produced
        a fact that was later found to be incorrect.

        Returns None if no violations exist (chain is clean).

    Args:
        state: The ConstraintState after full chain propagation.
        steps: The list of steps (used for bounds checking).

    Returns:
        Step index of the root cause, or None if no violations.
    """
    violated = state.get_violated()
    if not violated:
        return None
    return min(f.introduced_at_step for f in violated)


def rollback_to(
    steps: list[AgentStep],
    target_step: int,
) -> list[AgentStep]:
    """Return a prefix of the chain up to (but not including) the target step.

    **Detailed explanation for engineers:**
        Simulates rolling back a reasoning chain to just before the error
        occurred. Returns steps[0:target_step] — the clean prefix. The
        caller can then re-run from target_step onward with corrected output.

    Args:
        steps: Full chain of AgentSteps.
        target_step: Index to roll back to (this step and later are dropped).

    Returns:
        List of AgentSteps from step 0 to target_step-1.
    """
    return steps[:target_step]


def repair_chain(
    steps: list[AgentStep],
    error_step: int,
    corrected_outputs: dict[int, str],
    pipeline: VerifyRepairPipeline,
) -> tuple[ConstraintState, list[VerificationResult], list[AgentStep]]:
    """Repair a chain by replacing error step outputs and re-propagating.

    **Detailed explanation for engineers:**
        Takes the original chain, replaces outputs at specified step indices
        with corrected text, then re-runs verify_chain() on the repaired
        chain. This simulates what would happen if the LLM regenerated
        the bad steps with corrected reasoning.

    Args:
        steps: Original chain of AgentSteps.
        error_step: The step where the error was detected.
        corrected_outputs: Dict mapping step_index -> corrected output text.
        pipeline: The VerifyRepairPipeline for re-verification.

    Returns:
        Tuple of (new ConstraintState, new VerificationResults, repaired steps).
    """
    repaired_steps = []
    for step in steps:
        if step.step_index in corrected_outputs:
            repaired_steps.append(
                AgentStep(
                    step_index=step.step_index,
                    input_text=step.input_text,
                    output_text=corrected_outputs[step.step_index],
                )
            )
        else:
            # Reset verification state so propagate() runs fresh.
            repaired_steps.append(
                AgentStep(
                    step_index=step.step_index,
                    input_text=step.input_text,
                    output_text=step.output_text,
                )
            )
    state, results = verify_chain(repaired_steps, pipeline)
    return state, results, repaired_steps


# ---------------------------------------------------------------------------
# Workflow instance dataclass
# ---------------------------------------------------------------------------


@dataclass
class WorkflowInstance:
    """A single workflow instance with known ground truth.

    **Detailed explanation for engineers:**
        Each instance has a name, a workflow type tag, 4 AgentSteps, and
        metadata about whether it contains an error and where. For error
        instances, error_step_index marks the step that has the planted
        error, and corrected_output holds the fixed text for that step.
        This ground truth lets us measure detection accuracy.

    Attributes:
        name: Unique identifier for this instance.
        workflow_type: One of "math_tutor", "code_assistant", "research_assistant".
        steps: List of 4 AgentSteps forming the reasoning chain.
        has_error: True if this instance has a planted error.
        error_step_index: Index of the step with the error (None if no error).
        error_description: Human-readable description of the planted error.
        corrected_output: The corrected output text for the error step.
    """

    name: str
    workflow_type: str
    steps: list[AgentStep]
    has_error: bool
    error_step_index: int | None = None
    error_description: str = ""
    corrected_output: str = ""


# ---------------------------------------------------------------------------
# Workflow template: Math Tutor (4 steps)
# ---------------------------------------------------------------------------


def _make_math_tutor_instances() -> list[WorkflowInstance]:
    """Create 10 math tutor workflow instances (5 correct, 5 with errors).

    **Detailed explanation for engineers:**
        Each math tutor workflow follows 4 steps:
        1. Read and understand the problem
        2. Identify the mathematical approach
        3. Execute computation step by step
        4. Verify and state final answer

        Correct instances have valid arithmetic throughout. Error instances
        have either a wrong operation in step 3 (wrong number used in
        computation) or an incorrect approach in step 2 (wrong formula
        choice leading to wrong computation).

        The arithmetic extractor can catch "X + Y = Z" claims, so we
        embed explicit arithmetic expressions in the step outputs.
    """
    instances: list[WorkflowInstance] = []

    # --- 5 correct instances ---

    correct_problems = [
        {
            "name": "math_correct_1",
            "q": "What is 47 + 28?",
            "steps": [
                "The problem asks me to add two numbers: 47 and 28.",
                "I will use simple addition. Add the ones place first, then tens.",
                "47 + 28 = 75. Breaking it down: ones digit 7 + 8 gives 15, carry the 1, tens digit 4 + 2 plus carry gives 7. Result is 75.",
                "The final answer is 75. Verified: 47 + 28 = 75.",
            ],
        },
        {
            "name": "math_correct_2",
            "q": "What is 156 - 89?",
            "steps": [
                "The problem asks me to subtract 89 from 156.",
                "I will use subtraction with borrowing.",
                "156 - 89 = 67. Breaking down: 16 - 9 = 7, 15 - 8 = 7. So 67.",
                "The final answer is 67. Verified: 156 - 89 = 67.",
            ],
        },
        {
            "name": "math_correct_3",
            "q": "What is 23 + 45?",
            "steps": [
                "The problem asks me to add 23 and 45.",
                "Simple addition: add ones then tens.",
                "23 + 45 = 68. Breaking down: 3 + 5 = 8, 2 + 4 = 6. So 68.",
                "The final answer is 68. Verified: 23 + 45 = 68.",
            ],
        },
        {
            "name": "math_correct_4",
            "q": "What is 100 - 37?",
            "steps": [
                "The problem asks me to subtract 37 from 100.",
                "I will use subtraction with borrowing from hundreds.",
                "100 - 37 = 63. Breaking down: 10 - 7 = 3, 9 - 3 = 6. So 63.",
                "The final answer is 63. Verified: 100 - 37 = 63.",
            ],
        },
        {
            "name": "math_correct_5",
            "q": "What is 55 + 33?",
            "steps": [
                "The problem asks me to add 55 and 33.",
                "Simple addition of two 2-digit numbers.",
                "55 + 33 = 88. Breaking down: 5 + 3 = 8, 5 + 3 = 8. So 88.",
                "The final answer is 88. Verified: 55 + 33 = 88.",
            ],
        },
    ]

    for p in correct_problems:
        step_objs = [
            AgentStep(step_index=i, input_text=p["q"], output_text=p["steps"][i])
            for i in range(4)
        ]
        instances.append(
            WorkflowInstance(
                name=p["name"],
                workflow_type="math_tutor",
                steps=step_objs,
                has_error=False,
            )
        )

    # --- 5 error instances ---

    error_problems = [
        {
            "name": "math_error_1_wrong_op",
            "q": "What is 47 + 28?",
            "steps": [
                "The problem asks me to add two numbers: 47 and 28.",
                "I will use simple addition.",
                # Error: claims 47 + 28 = 73 (wrong)
                "47 + 28 = 73. Breaking down: 7 + 8 = 13, carry 1. 4 + 2 + 1 = 6. So 73.",
                "The final answer is 73. Verified: 47 + 28 = 73.",
            ],
            "error_step": 2,
            "error_desc": "Wrong computation: 47 + 28 claimed as 73 instead of 75",
            "corrected": "47 + 28 = 75. Breaking down: 7 + 8 = 15, carry 1. 4 + 2 + 1 = 7. So 75.",
        },
        {
            "name": "math_error_2_wrong_op",
            "q": "What is 156 - 89?",
            "steps": [
                "The problem asks me to subtract 89 from 156.",
                "I will use subtraction with borrowing.",
                # Error: claims 156 - 89 = 77 (wrong, correct is 67)
                "156 - 89 = 77. Breaking down: 16 - 9 = 7, 15 - 8 = 7. So 77.",
                "The final answer is 77. Verified: 156 - 89 = 77.",
            ],
            "error_step": 2,
            "error_desc": "Wrong computation: 156 - 89 claimed as 77 instead of 67",
            "corrected": "156 - 89 = 67. Breaking down: 16 - 9 = 7, 15 - 8 = 7. So 67.",
        },
        {
            "name": "math_error_3_wrong_approach",
            "q": "What is 23 + 45?",
            "steps": [
                "The problem asks me to add 23 and 45.",
                # Error in approach: says subtraction instead of addition
                "I will use subtraction to solve this.",
                # Carries through the wrong approach with wrong arithmetic
                "23 - 45 = -22. Breaking down: 3 - 5 = -2, 2 - 4 = -2. So -22.",
                "The final answer is -22. Verified: 23 - 45 = -22.",
            ],
            "error_step": 1,
            "error_desc": "Wrong approach: used subtraction instead of addition",
            "corrected": "I will use simple addition: add ones then tens.",
        },
        {
            "name": "math_error_4_wrong_op",
            "q": "What is 100 - 37?",
            "steps": [
                "The problem asks me to subtract 37 from 100.",
                "I will use subtraction with borrowing from hundreds.",
                # Error: claims 100 - 37 = 73 (wrong, correct is 63)
                "100 - 37 = 73. Breaking down: 10 - 7 = 3, 10 - 3 = 7. So 73.",
                "The final answer is 73. Verified: 100 - 37 = 73.",
            ],
            "error_step": 2,
            "error_desc": "Wrong computation: 100 - 37 claimed as 73 instead of 63",
            "corrected": "100 - 37 = 63. Breaking down: 10 - 7 = 3, 9 - 3 = 6. So 63.",
        },
        {
            "name": "math_error_5_wrong_op",
            "q": "What is 55 + 33?",
            "steps": [
                "The problem asks me to add 55 and 33.",
                "Simple addition of two 2-digit numbers.",
                # Error: claims 55 + 33 = 86 (wrong, correct is 88)
                "55 + 33 = 86. Breaking down: 5 + 3 = 6, 5 + 3 = 8. So 86.",
                "The final answer is 86. Verified: 55 + 33 = 86.",
            ],
            "error_step": 2,
            "error_desc": "Wrong computation: 55 + 33 claimed as 86 instead of 88",
            "corrected": "55 + 33 = 88. Breaking down: 5 + 3 = 8, 5 + 3 = 8. So 88.",
        },
    ]

    for p in error_problems:
        step_objs = [
            AgentStep(step_index=i, input_text=p["q"], output_text=p["steps"][i])
            for i in range(4)
        ]
        instances.append(
            WorkflowInstance(
                name=p["name"],
                workflow_type="math_tutor",
                steps=step_objs,
                has_error=True,
                error_step_index=p["error_step"],
                error_description=p["error_desc"],
                corrected_output=p["corrected"],
            )
        )

    return instances


# ---------------------------------------------------------------------------
# Workflow template: Code Assistant (4 steps)
# ---------------------------------------------------------------------------


def _make_code_assistant_instances() -> list[WorkflowInstance]:
    """Create 10 code assistant workflow instances (5 correct, 5 with errors).

    **Detailed explanation for engineers:**
        Each code assistant workflow follows 4 steps:
        1. Parse the specification/requirements
        2. Write the function implementation
        3. Write test cases
        4. Run tests and report results

        The CodeExtractor can verify type annotations, return types, loop bounds,
        and variable initialization. We embed Python code blocks in steps 2 and 3
        so the extractor has something to parse. Error instances have either a
        wrong implementation in step 2 (uninitialized variable, type mismatch)
        or wrong test in step 3 (incorrect expected value).
    """
    instances: list[WorkflowInstance] = []

    # --- 5 correct instances ---

    correct_code = [
        {
            "name": "code_correct_1",
            "q": "Write a function to compute factorial",
            "steps": [
                "The requirement is to write a factorial function that takes an integer n and returns n!.",
                (
                    "Here is the implementation:\n"
                    "```python\n"
                    "def factorial(n: int) -> int:\n"
                    "    result = 1\n"
                    "    for i in range(1, n + 1):\n"
                    "        result = result * i\n"
                    "    return result\n"
                    "```"
                ),
                (
                    "Here are the test cases:\n"
                    "```python\n"
                    "def test_factorial():\n"
                    "    result = factorial(5)\n"
                    "    expected = 120\n"
                    "    assert result == expected\n"
                    "```"
                ),
                "All tests pass. factorial(5) = 120 as expected.",
            ],
        },
        {
            "name": "code_correct_2",
            "q": "Write a function to sum a list",
            "steps": [
                "The requirement is to write a function that sums all elements in a list of integers.",
                (
                    "Here is the implementation:\n"
                    "```python\n"
                    "def sum_list(nums: list) -> int:\n"
                    "    total = 0\n"
                    "    for i in range(len(nums)):\n"
                    "        total = total + nums[i]\n"
                    "    return total\n"
                    "```"
                ),
                (
                    "Here are the test cases:\n"
                    "```python\n"
                    "def test_sum_list():\n"
                    "    result = sum_list([1, 2, 3])\n"
                    "    expected = 6\n"
                    "    assert result == expected\n"
                    "```"
                ),
                "All tests pass. sum_list([1, 2, 3]) = 6 as expected.",
            ],
        },
        {
            "name": "code_correct_3",
            "q": "Write a function to find the maximum",
            "steps": [
                "The requirement is to find the maximum value in a list of integers.",
                (
                    "Here is the implementation:\n"
                    "```python\n"
                    "def find_max(nums: list) -> int:\n"
                    "    best = nums[0]\n"
                    "    for i in range(1, len(nums)):\n"
                    "        if nums[i] > best:\n"
                    "            best = nums[i]\n"
                    "    return best\n"
                    "```"
                ),
                (
                    "Here are the test cases:\n"
                    "```python\n"
                    "def test_find_max():\n"
                    "    result = find_max([3, 1, 4, 1, 5])\n"
                    "    expected = 5\n"
                    "    assert result == expected\n"
                    "```"
                ),
                "All tests pass. find_max([3, 1, 4, 1, 5]) = 5 as expected.",
            ],
        },
        {
            "name": "code_correct_4",
            "q": "Write a function to reverse a string",
            "steps": [
                "The requirement is to reverse a string.",
                (
                    "Here is the implementation:\n"
                    "```python\n"
                    "def reverse_str(s: str) -> str:\n"
                    "    chars = list(s)\n"
                    "    left = 0\n"
                    "    right = len(chars) - 1\n"
                    "    while left < right:\n"
                    "        tmp = chars[left]\n"
                    "        chars[left] = chars[right]\n"
                    "        chars[right] = tmp\n"
                    "        left = left + 1\n"
                    "        right = right - 1\n"
                    "    return ''.join(chars)\n"
                    "```"
                ),
                (
                    "Here are the test cases:\n"
                    "```python\n"
                    "def test_reverse_str():\n"
                    "    result = reverse_str('hello')\n"
                    "    expected = 'olleh'\n"
                    "    assert result == expected\n"
                    "```"
                ),
                "All tests pass. reverse_str('hello') = 'olleh' as expected.",
            ],
        },
        {
            "name": "code_correct_5",
            "q": "Write a function to count vowels",
            "steps": [
                "The requirement is to count the number of vowels in a string.",
                (
                    "Here is the implementation:\n"
                    "```python\n"
                    "def count_vowels(s: str) -> int:\n"
                    "    count = 0\n"
                    "    vowels = 'aeiouAEIOU'\n"
                    "    for i in range(len(s)):\n"
                    "        if s[i] in vowels:\n"
                    "            count = count + 1\n"
                    "    return count\n"
                    "```"
                ),
                (
                    "Here are the test cases:\n"
                    "```python\n"
                    "def test_count_vowels():\n"
                    "    result = count_vowels('hello')\n"
                    "    expected = 2\n"
                    "    assert result == expected\n"
                    "```"
                ),
                "All tests pass. count_vowels('hello') = 2 as expected.",
            ],
        },
    ]

    for p in correct_code:
        step_objs = [
            AgentStep(step_index=i, input_text=p["q"], output_text=p["steps"][i])
            for i in range(4)
        ]
        instances.append(
            WorkflowInstance(
                name=p["name"],
                workflow_type="code_assistant",
                steps=step_objs,
                has_error=False,
            )
        )

    # --- 5 error instances ---

    error_code = [
        {
            "name": "code_error_1_uninit",
            "q": "Write a function to compute factorial",
            "steps": [
                "The requirement is to write a factorial function.",
                (
                    "Here is the implementation:\n"
                    "```python\n"
                    "def factorial(n: int) -> int:\n"
                    # Error: 'result' never initialized before use in loop
                    "    for i in range(1, n + 1):\n"
                    "        result = result * i\n"
                    "    return result\n"
                    "```"
                ),
                (
                    "Here are the test cases:\n"
                    "```python\n"
                    "def test_factorial():\n"
                    "    result = factorial(5)\n"
                    "    expected = 120\n"
                    "    assert result == expected\n"
                    "```"
                ),
                "Tests should pass. factorial(5) = 120.",
            ],
            "error_step": 1,
            "error_desc": "Variable 'result' used before initialization in factorial loop",
            "corrected": (
                "Here is the implementation:\n"
                "```python\n"
                "def factorial(n: int) -> int:\n"
                "    result = 1\n"
                "    for i in range(1, n + 1):\n"
                "        result = result * i\n"
                "    return result\n"
                "```"
            ),
        },
        {
            "name": "code_error_2_return_type",
            "q": "Write a function to check if a number is even",
            "steps": [
                "The requirement is to return True if n is even, False otherwise.",
                (
                    "Here is the implementation:\n"
                    "```python\n"
                    "def is_even(n: int) -> bool:\n"
                    # Error: returns string instead of bool
                    "    if n % 2 == 0:\n"
                    "        return 'yes'\n"
                    "    return 'no'\n"
                    "```"
                ),
                (
                    "Here are the test cases:\n"
                    "```python\n"
                    "def test_is_even():\n"
                    "    result = is_even(4)\n"
                    "    expected = True\n"
                    "    assert result == expected\n"
                    "```"
                ),
                "Tests should pass. is_even(4) returns 'yes'.",
            ],
            "error_step": 1,
            "error_desc": "Returns string 'yes'/'no' instead of bool True/False",
            "corrected": (
                "Here is the implementation:\n"
                "```python\n"
                "def is_even(n: int) -> bool:\n"
                "    if n % 2 == 0:\n"
                "        return True\n"
                "    return False\n"
                "```"
            ),
        },
        {
            "name": "code_error_3_uninit",
            "q": "Write a function to sum a list",
            "steps": [
                "The requirement is to sum all elements in a list of integers.",
                (
                    "Here is the implementation:\n"
                    "```python\n"
                    "def sum_list(nums: list) -> int:\n"
                    # Error: uses undefined variable 'accumulator'
                    "    for i in range(len(nums)):\n"
                    "        accumulator = accumulator + nums[i]\n"
                    "    return accumulator\n"
                    "```"
                ),
                (
                    "Here are the test cases:\n"
                    "```python\n"
                    "def test_sum_list():\n"
                    "    result = sum_list([1, 2, 3])\n"
                    "    expected = 6\n"
                    "    assert result == expected\n"
                    "```"
                ),
                "Tests should pass. sum_list([1, 2, 3]) = 6.",
            ],
            "error_step": 1,
            "error_desc": "Variable 'accumulator' used before initialization",
            "corrected": (
                "Here is the implementation:\n"
                "```python\n"
                "def sum_list(nums: list) -> int:\n"
                "    accumulator = 0\n"
                "    for i in range(len(nums)):\n"
                "        accumulator = accumulator + nums[i]\n"
                "    return accumulator\n"
                "```"
            ),
        },
        {
            "name": "code_error_4_return_type",
            "q": "Write a function to find the maximum",
            "steps": [
                "The requirement is to find the maximum value in a list.",
                (
                    "Here is the implementation:\n"
                    "```python\n"
                    "def find_max(nums: list) -> int:\n"
                    "    best = nums[0]\n"
                    "    for i in range(1, len(nums)):\n"
                    "        if nums[i] > best:\n"
                    "            best = nums[i]\n"
                    # Error: returns string representation instead of int
                    "    return 'max is ' + str(best)\n"
                    "```"
                ),
                (
                    "Here are the test cases:\n"
                    "```python\n"
                    "def test_find_max():\n"
                    "    result = find_max([3, 1, 4, 1, 5])\n"
                    "    expected = 5\n"
                    "    assert result == expected\n"
                    "```"
                ),
                "Tests should pass. find_max([3, 1, 4, 1, 5]) = 5.",
            ],
            "error_step": 1,
            "error_desc": "Returns string concatenation instead of int",
            "corrected": (
                "Here is the implementation:\n"
                "```python\n"
                "def find_max(nums: list) -> int:\n"
                "    best = nums[0]\n"
                "    for i in range(1, len(nums)):\n"
                "        if nums[i] > best:\n"
                "            best = nums[i]\n"
                "    return best\n"
                "```"
            ),
        },
        {
            "name": "code_error_5_wrong_test",
            "q": "Write a function to count vowels",
            "steps": [
                "The requirement is to count the number of vowels in a string.",
                (
                    "Here is the implementation:\n"
                    "```python\n"
                    "def count_vowels(s: str) -> int:\n"
                    "    count = 0\n"
                    "    vowels = 'aeiouAEIOU'\n"
                    "    for i in range(len(s)):\n"
                    "        if s[i] in vowels:\n"
                    "            count = count + 1\n"
                    "    return count\n"
                    "```"
                ),
                (
                    "Here are the test cases:\n"
                    "```python\n"
                    "def test_count_vowels():\n"
                    "    result = count_vowels('hello')\n"
                    # Error: wrong expected value in test (should be 2)
                    "    expected = 3\n"
                    "    assert result == expected\n"
                    "```"
                ),
                # Error propagates: report claims wrong result
                "Tests fail. count_vowels('hello') returns 2 but expected 3. 2 + 1 = 3.",
            ],
            "error_step": 2,
            "error_desc": "Wrong expected value in test: expects 3 vowels in 'hello' but correct is 2",
            "corrected": (
                "Here are the test cases:\n"
                "```python\n"
                "def test_count_vowels():\n"
                "    result = count_vowels('hello')\n"
                "    expected = 2\n"
                "    assert result == expected\n"
                "```"
            ),
        },
    ]

    for p in error_code:
        step_objs = [
            AgentStep(step_index=i, input_text=p["q"], output_text=p["steps"][i])
            for i in range(4)
        ]
        instances.append(
            WorkflowInstance(
                name=p["name"],
                workflow_type="code_assistant",
                steps=step_objs,
                has_error=True,
                error_step_index=p["error_step"],
                error_description=p["error_desc"],
                corrected_output=p["corrected"],
            )
        )

    return instances


# ---------------------------------------------------------------------------
# Workflow template: Research Assistant (4 steps)
# ---------------------------------------------------------------------------


def _make_research_assistant_instances() -> list[WorkflowInstance]:
    """Create 10 research assistant workflow instances (5 correct, 5 with errors).

    **Detailed explanation for engineers:**
        Each research assistant workflow follows 4 steps:
        1. Formulate search query from user question
        2. Gather relevant facts (simulated search results)
        3. Synthesize facts into coherent answer
        4. Cite sources and provide confidence

        The NLExtractor catches "X is Y" factual claims, and the LogicExtractor
        catches "If P then Q" logical claims. Error instances have either a
        misattributed fact in step 3 (says X is Y when the gathered facts say
        X is Z) or a wrong citation in step 4.

        We use real-ish but verifiable factual claims (e.g., geography, basic
        science) so the extractor has structured content to parse.
    """
    instances: list[WorkflowInstance] = []

    # --- 5 correct instances ---

    correct_research = [
        {
            "name": "research_correct_1",
            "q": "What is the capital of France?",
            "steps": [
                "Search query: 'capital city of France'.",
                "Facts gathered: Paris is the capital of France. France is a country in Europe.",
                "Based on the gathered facts: Paris is the capital of France. It is located in Europe.",
                "Source: World Atlas, confidence: high. Paris is the capital of France.",
            ],
        },
        {
            "name": "research_correct_2",
            "q": "What is the boiling point of water?",
            "steps": [
                "Search query: 'boiling point of water at sea level'.",
                "Facts gathered: Water boils at 100 degrees Celsius at sea level. Water is a compound of hydrogen and oxygen.",
                "Based on the gathered facts: Water boils at 100 degrees Celsius at standard atmospheric pressure.",
                "Source: Chemistry textbook, confidence: high. Water boils at 100 degrees Celsius.",
            ],
        },
        {
            "name": "research_correct_3",
            "q": "What is the largest planet?",
            "steps": [
                "Search query: 'largest planet in solar system'.",
                "Facts gathered: Jupiter is the largest planet in the solar system. Jupiter has 95 known moons.",
                "Based on the gathered facts: Jupiter is the largest planet in our solar system.",
                "Source: NASA, confidence: high. Jupiter is the largest planet.",
            ],
        },
        {
            "name": "research_correct_4",
            "q": "What causes rain?",
            "steps": [
                "Search query: 'what causes rain precipitation'.",
                "Facts gathered: Rain is caused by water vapor condensing in the atmosphere. If air rises, then it cools and water condenses.",
                "Based on the gathered facts: Rain occurs when water vapor condenses in the atmosphere as air cools.",
                "Source: Meteorology guide, confidence: high. Rain is caused by atmospheric condensation.",
            ],
        },
        {
            "name": "research_correct_5",
            "q": "Who wrote Romeo and Juliet?",
            "steps": [
                "Search query: 'author of Romeo and Juliet'.",
                "Facts gathered: William Shakespeare is the author of Romeo and Juliet. The play was written around 1597.",
                "Based on the gathered facts: Shakespeare is the author of Romeo and Juliet, written circa 1597.",
                "Source: Literary encyclopedia, confidence: high. Shakespeare is the author of Romeo and Juliet.",
            ],
        },
    ]

    for p in correct_research:
        step_objs = [
            AgentStep(step_index=i, input_text=p["q"], output_text=p["steps"][i])
            for i in range(4)
        ]
        instances.append(
            WorkflowInstance(
                name=p["name"],
                workflow_type="research_assistant",
                steps=step_objs,
                has_error=False,
            )
        )

    # --- 5 error instances ---

    error_research = [
        {
            "name": "research_error_1_misattributed",
            "q": "What is the capital of France?",
            "steps": [
                "Search query: 'capital city of France'.",
                "Facts gathered: Paris is the capital of France. France is a country in Europe.",
                # Error: misattributes — says London instead of Paris
                "Based on the gathered facts: London is the capital of France.",
                "Source: World Atlas, confidence: high. London is the capital of France.",
            ],
            "error_step": 2,
            "error_desc": "Misattributed fact: says London is capital of France instead of Paris",
            "corrected": "Based on the gathered facts: Paris is the capital of France. It is located in Europe.",
        },
        {
            "name": "research_error_2_misattributed",
            "q": "What is the boiling point of water?",
            "steps": [
                "Search query: 'boiling point of water at sea level'.",
                "Facts gathered: Water boils at 100 degrees Celsius at sea level.",
                # Error: says 50 degrees instead of 100
                "Based on the gathered facts: Water boils at 50 degrees Celsius at standard atmospheric pressure.",
                "Source: Chemistry textbook, confidence: high. Water boils at 50 degrees Celsius.",
            ],
            "error_step": 2,
            "error_desc": "Misattributed fact: says water boils at 50C instead of 100C",
            "corrected": "Based on the gathered facts: Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        },
        {
            "name": "research_error_3_misattributed",
            "q": "What is the largest planet?",
            "steps": [
                "Search query: 'largest planet in solar system'.",
                "Facts gathered: Jupiter is the largest planet in the solar system.",
                # Error: says Saturn instead of Jupiter
                "Based on the gathered facts: Saturn is the largest planet in our solar system.",
                "Source: NASA, confidence: high. Saturn is the largest planet.",
            ],
            "error_step": 2,
            "error_desc": "Misattributed fact: says Saturn is largest planet instead of Jupiter",
            "corrected": "Based on the gathered facts: Jupiter is the largest planet in our solar system.",
        },
        {
            "name": "research_error_4_wrong_citation",
            "q": "What causes rain?",
            "steps": [
                "Search query: 'what causes rain precipitation'.",
                "Facts gathered: Rain is caused by water vapor condensing in the atmosphere. If air rises, then it cools and water condenses.",
                "Based on the gathered facts: Rain occurs when water vapor condenses in the atmosphere as air cools.",
                # Error: contradicts step 3 — says rain is caused by temperature inversion
                "Source: Meteorology guide, confidence: high. Rain is caused by temperature inversions preventing condensation.",
            ],
            "error_step": 3,
            "error_desc": "Wrong citation: contradicts synthesis by attributing rain to temperature inversions",
            "corrected": "Source: Meteorology guide, confidence: high. Rain is caused by atmospheric condensation.",
        },
        {
            "name": "research_error_5_wrong_citation",
            "q": "Who wrote Romeo and Juliet?",
            "steps": [
                "Search query: 'author of Romeo and Juliet'.",
                "Facts gathered: William Shakespeare is the author of Romeo and Juliet.",
                "Based on the gathered facts: Shakespeare is the author of Romeo and Juliet.",
                # Error: wrong citation — attributes to Marlowe instead of Shakespeare
                "Source: Literary encyclopedia, confidence: high. Marlowe is the author of Romeo and Juliet.",
            ],
            "error_step": 3,
            "error_desc": "Wrong citation: attributes Romeo and Juliet to Marlowe instead of Shakespeare",
            "corrected": "Source: Literary encyclopedia, confidence: high. Shakespeare is the author of Romeo and Juliet.",
        },
    ]

    for p in error_research:
        step_objs = [
            AgentStep(step_index=i, input_text=p["q"], output_text=p["steps"][i])
            for i in range(4)
        ]
        instances.append(
            WorkflowInstance(
                name=p["name"],
                workflow_type="research_assistant",
                steps=step_objs,
                has_error=True,
                error_step_index=p["error_step"],
                error_description=p["error_desc"],
                corrected_output=p["corrected"],
            )
        )

    return instances


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def _analyze_instance(
    instance: WorkflowInstance,
    pipeline: VerifyRepairPipeline,
) -> dict[str, Any]:
    """Run full agentic verification on one workflow instance and collect metrics.

    **Detailed explanation for engineers:**
        For each instance:
        1. Run verify_chain() to propagate constraints through all 4 steps.
        2. Record per-step verification results and constraint state snapshots.
        3. For error cases: compare find_root_cause() result to actual error step.
        4. Attempt repair_chain() with the known corrected output and re-verify.
        5. Run final-answer-only verification (step 3 / step index 3 only) for comparison.

    Returns:
        Dict with all metrics for this instance, suitable for JSON serialization.
    """
    result: dict[str, Any] = {
        "name": instance.name,
        "workflow_type": instance.workflow_type,
        "has_error": instance.has_error,
        "error_step_index": instance.error_step_index,
        "error_description": instance.error_description,
    }

    # --- Step 1: Full chain verification ---
    t0 = time.monotonic()
    state, step_results = verify_chain(instance.steps, pipeline)
    t1 = time.monotonic()
    result["chain_verify_time_ms"] = (t1 - t0) * 1000

    # Record per-step results.
    per_step: list[dict[str, Any]] = []
    for i, (step, vr) in enumerate(zip(instance.steps, step_results)):
        step_info: dict[str, Any] = {
            "step_index": i,
            "verified": vr.verified if vr else True,
            "n_constraints": len(vr.constraints) if vr else 0,
            "n_violations": len(vr.violations) if vr else 0,
            "energy": vr.energy if vr else 0.0,
        }
        per_step.append(step_info)
    result["per_step"] = per_step

    # Record constraint state snapshot.
    snapshot = state.snapshot()
    result["final_state"] = {
        "n_verified": snapshot["n_verified"],
        "n_assumed": snapshot["n_assumed"],
        "n_violated": snapshot["n_violated"],
        "total_facts": len(snapshot["facts"]),
    }

    # Constraint coverage: fraction of steps that produced extractable constraints.
    steps_with_constraints = sum(1 for s in per_step if s["n_constraints"] > 0)
    result["constraint_coverage"] = steps_with_constraints / len(instance.steps)

    # --- Step 2: Error detection analysis ---
    any_violation_detected = snapshot["n_violated"] > 0
    result["violation_detected"] = any_violation_detected

    if instance.has_error:
        # Find root cause.
        root_cause = find_root_cause(state, instance.steps)
        result["root_cause_step"] = root_cause
        result["root_cause_correct"] = root_cause == instance.error_step_index

        # Detection latency: how many steps after the error before we see a violation?
        first_violation_step = None
        for s in per_step:
            if s["n_violations"] > 0:
                first_violation_step = s["step_index"]
                break
        result["first_violation_at_step"] = first_violation_step
        if first_violation_step is not None and instance.error_step_index is not None:
            result["detection_latency"] = first_violation_step - instance.error_step_index
        else:
            result["detection_latency"] = None

        # --- Step 3: Attempt repair ---
        if instance.corrected_output and instance.error_step_index is not None:
            corrected = {instance.error_step_index: instance.corrected_output}
            repair_state, repair_results, repaired_steps = repair_chain(
                instance.steps, instance.error_step_index, corrected, pipeline
            )
            repair_snapshot = repair_state.snapshot()
            repair_violations = repair_snapshot["n_violated"]
            result["repair_attempted"] = True
            result["repair_success"] = repair_violations == 0
            result["repair_remaining_violations"] = repair_violations
        else:
            result["repair_attempted"] = False
            result["repair_success"] = False
    else:
        result["root_cause_step"] = None
        result["root_cause_correct"] = None
        result["first_violation_at_step"] = None
        result["detection_latency"] = None
        result["repair_attempted"] = False
        result["repair_success"] = False

    # --- Step 4: Final-answer-only verification (comparison baseline) ---
    last_step = instance.steps[-1]
    t2 = time.monotonic()
    final_only_vr = pipeline.verify(last_step.input_text, last_step.output_text)
    t3 = time.monotonic()
    result["final_only"] = {
        "verified": final_only_vr.verified,
        "n_constraints": len(final_only_vr.constraints),
        "n_violations": len(final_only_vr.violations),
        "time_ms": (t3 - t2) * 1000,
    }

    # Flag: did agentic verification catch something that final-only missed?
    result["agentic_catches_more"] = (
        any_violation_detected and final_only_vr.verified
    )

    return result


def _compute_aggregate_metrics(
    all_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate analysis across all workflow instances.

    **Detailed explanation for engineers:**
        Computes the key metrics from the task specification:
        a. Error detection rate: fraction of planted errors detected
        b. Detection latency: steps after error before detection (ideal: 0)
        c. Root cause accuracy: find_root_cause matches actual error step
        d. Repair success: fraction of repairs that eliminated all violations
        e. False positive rate: fraction of correct chains flagged as violated
        f. Per-workflow-type breakdown of all the above
        g. Constraint coverage: fraction of steps producing extractable constraints
        h. Agentic vs final-only comparison
    """
    metrics: dict[str, Any] = {}

    # Split by has_error.
    error_instances = [r for r in all_results if r["has_error"]]
    correct_instances = [r for r in all_results if not r["has_error"]]

    # (a) Error detection rate.
    detected = sum(1 for r in error_instances if r["violation_detected"])
    metrics["error_detection_rate"] = detected / len(error_instances) if error_instances else 0

    # (b) Detection latency.
    latencies = [r["detection_latency"] for r in error_instances if r["detection_latency"] is not None]
    metrics["detection_latency_mean"] = sum(latencies) / len(latencies) if latencies else None
    metrics["detection_latency_at_error_step"] = sum(1 for l in latencies if l == 0)
    metrics["detection_latency_counts"] = {
        "at_error_step": sum(1 for l in latencies if l == 0),
        "1_step_later": sum(1 for l in latencies if l == 1),
        "2_steps_later": sum(1 for l in latencies if l == 2),
        "3_steps_later": sum(1 for l in latencies if l == 3),
        "not_detected": len(error_instances) - len(latencies),
    }

    # (c) Root cause accuracy.
    root_correct = sum(1 for r in error_instances if r["root_cause_correct"])
    metrics["root_cause_accuracy"] = root_correct / len(error_instances) if error_instances else 0

    # (d) Repair success rate.
    repair_attempted = [r for r in error_instances if r["repair_attempted"]]
    repair_succeeded = sum(1 for r in repair_attempted if r["repair_success"])
    metrics["repair_success_rate"] = repair_succeeded / len(repair_attempted) if repair_attempted else 0

    # (e) False positive rate.
    false_positives = sum(1 for r in correct_instances if r["violation_detected"])
    metrics["false_positive_rate"] = false_positives / len(correct_instances) if correct_instances else 0

    # (f) Per-workflow-type breakdown.
    by_type: dict[str, dict[str, Any]] = {}
    for wtype in ["math_tutor", "code_assistant", "research_assistant"]:
        type_results = [r for r in all_results if r["workflow_type"] == wtype]
        type_errors = [r for r in type_results if r["has_error"]]
        type_correct = [r for r in type_results if not r["has_error"]]

        type_detected = sum(1 for r in type_errors if r["violation_detected"])
        type_fp = sum(1 for r in type_correct if r["violation_detected"])
        type_root_correct = sum(1 for r in type_errors if r["root_cause_correct"])
        type_repair_ok = sum(1 for r in type_errors if r.get("repair_success", False))
        type_latencies = [r["detection_latency"] for r in type_errors if r["detection_latency"] is not None]

        by_type[wtype] = {
            "n_instances": len(type_results),
            "n_errors": len(type_errors),
            "n_correct": len(type_correct),
            "error_detection_rate": type_detected / len(type_errors) if type_errors else 0,
            "false_positive_rate": type_fp / len(type_correct) if type_correct else 0,
            "root_cause_accuracy": type_root_correct / len(type_errors) if type_errors else 0,
            "repair_success_rate": type_repair_ok / len(type_errors) if type_errors else 0,
            "mean_detection_latency": sum(type_latencies) / len(type_latencies) if type_latencies else None,
        }
    metrics["by_workflow_type"] = by_type

    # (g) Constraint coverage.
    coverages = [r["constraint_coverage"] for r in all_results]
    metrics["mean_constraint_coverage"] = sum(coverages) / len(coverages) if coverages else 0

    # (h) Agentic vs final-only.
    agentic_catches_more = sum(1 for r in all_results if r["agentic_catches_more"])
    metrics["agentic_catches_more_count"] = agentic_catches_more
    metrics["agentic_catches_more_fraction"] = (
        agentic_catches_more / len(error_instances) if error_instances else 0
    )

    # Final-only detection rate for comparison.
    final_only_detected = sum(
        1 for r in error_instances
        if not r["final_only"]["verified"]
    )
    metrics["final_only_detection_rate"] = (
        final_only_detected / len(error_instances) if error_instances else 0
    )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 101: Agent Verification Workflows.

    **Detailed explanation for engineers:**
        1. Build the VerifyRepairPipeline in verify-only mode (no LLM loaded).
        2. Create all 30 workflow instances (3 templates x 10 each).
        3. Run _analyze_instance() on each to collect per-instance metrics.
        4. Compute aggregate metrics via _compute_aggregate_metrics().
        5. Save everything to results/experiment_101_results.json.
        6. Print a human-readable summary to stdout.
    """
    print("=" * 72)
    print("Experiment 101: Agent Verification Workflows")
    print("=" * 72)
    print()

    # Build pipeline in verify-only mode (no LLM, no model download needed).
    pipeline = VerifyRepairPipeline(model=None, timeout_seconds=60.0)
    print("[OK] VerifyRepairPipeline initialized (verify-only mode)")

    # Create all workflow instances.
    math_instances = _make_math_tutor_instances()
    code_instances = _make_code_assistant_instances()
    research_instances = _make_research_assistant_instances()
    all_instances = math_instances + code_instances + research_instances
    print(f"[OK] Created {len(all_instances)} workflow instances")
    print(f"     Math tutor: {len(math_instances)}")
    print(f"     Code assistant: {len(code_instances)}")
    print(f"     Research assistant: {len(research_instances)}")
    print()

    # Run analysis on each instance.
    all_results: list[dict[str, Any]] = []
    for i, instance in enumerate(all_instances):
        print(f"  [{i+1:2d}/{len(all_instances)}] {instance.name}...", end=" ", flush=True)
        try:
            result = _analyze_instance(instance, pipeline)
            all_results.append(result)
            status = "VIOLATION" if result["violation_detected"] else "CLEAN"
            print(f"{status} ({result['chain_verify_time_ms']:.1f}ms)")
        except Exception as exc:
            print(f"ERROR: {exc}")
            all_results.append({
                "name": instance.name,
                "workflow_type": instance.workflow_type,
                "has_error": instance.has_error,
                "error": str(exc),
            })

    print()

    # Filter out instances that errored out before computing aggregates.
    valid_results = [r for r in all_results if "violation_detected" in r]
    if not valid_results:
        print("ERROR: All instances failed. Cannot compute metrics.")
        return

    # Compute aggregate metrics.
    metrics = _compute_aggregate_metrics(valid_results)

    # Build final output.
    output = {
        "experiment": "101_agent_verification",
        "description": "End-to-end agentic constraint propagation on realistic workflows",
        "n_instances": len(all_instances),
        "n_workflow_types": 3,
        "workflow_types": ["math_tutor", "code_assistant", "research_assistant"],
        "metrics": metrics,
        "per_instance": all_results,
    }

    # Save results.
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "experiment_101_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[OK] Results saved to {results_path}")
    print()

    # Print human-readable summary.
    print("=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print()
    print(f"Error detection rate:        {metrics['error_detection_rate']:.1%}")
    print(f"  (final-only baseline):     {metrics['final_only_detection_rate']:.1%}")
    print(f"Root cause accuracy:         {metrics['root_cause_accuracy']:.1%}")
    print(f"Repair success rate:         {metrics['repair_success_rate']:.1%}")
    print(f"False positive rate:         {metrics['false_positive_rate']:.1%}")
    print(f"Mean constraint coverage:    {metrics['mean_constraint_coverage']:.1%}")
    print(f"Agentic catches more:        {metrics['agentic_catches_more_count']}/{len([r for r in all_results if r.get('has_error')])} errors")
    print()

    latency = metrics.get("detection_latency_counts", {})
    print("Detection latency distribution:")
    print(f"  At error step (ideal):     {latency.get('at_error_step', 0)}")
    print(f"  1 step later:              {latency.get('1_step_later', 0)}")
    print(f"  2 steps later:             {latency.get('2_steps_later', 0)}")
    print(f"  Not detected:              {latency.get('not_detected', 0)}")
    print()

    print("Per-workflow-type breakdown:")
    for wtype, wmetrics in metrics.get("by_workflow_type", {}).items():
        print(f"  {wtype}:")
        print(f"    Detection rate:          {wmetrics['error_detection_rate']:.1%}")
        print(f"    False positive rate:     {wmetrics['false_positive_rate']:.1%}")
        print(f"    Root cause accuracy:     {wmetrics['root_cause_accuracy']:.1%}")
        print(f"    Repair success rate:     {wmetrics['repair_success_rate']:.1%}")
        lat = wmetrics.get("mean_detection_latency")
        print(f"    Mean detection latency:  {lat:.1f} steps" if lat is not None else "    Mean detection latency:  N/A")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
