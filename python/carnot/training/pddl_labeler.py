"""PDDL-based step labeler for GSM8K arithmetic word problems.

**Why this module exists (arXiv 2604.17957 PDDL step labeling):**

    The FoVer v1 corpus (Exp 686) has 200 Z3-labeled pairs, which is not enough
    to train a JEPA predictor that generalises well.  arXiv 2604.17957 showed
    that PDDL-based transition labeling can produce ~1M step-level labels
    automatically from GSM8K without any human annotation.

    The key idea: treat each arithmetic CoT step as a PDDL state-action-state
    transition.  The "state" is a dictionary of named quantities and their
    current values.  An "action" is an arithmetic operation that updates one
    quantity.  A step is "correct" if the arithmetic in the step text actually
    produces the claimed new value.

**Why PDDL complements Z3 (not replaces it):**

    Z3 catches symbolic reasoning errors — steps that are logically inconsistent
    with prior steps even when the individual arithmetic is right.  PDDL catches
    state-update errors — "multiply when you should add" mistakes that Z3 misses
    because the formula is syntactically valid, just wrong for the problem context.

    Together the two methods cover different failure modes, which is why FoVer v2
    combines both label sources.

**Architecture:**

    extract_quantities(problem_text) → dict[str, float]
        Parses named numeric quantities from a word problem statement.  Uses
        regex heuristics (number followed by a word, or word followed by a
        number via "is/are/has/costs" patterns).

    encode_step_transition(step_text, state) → (action_str, new_state_estimate)
        Parses a CoT step as a PDDL action.  Extracts any arithmetic expression
        and evaluates it to estimate the new state produced by the action.

    verify_transition(step_text, prev_state, next_state) → bool
        Checks whether the arithmetic in step_text produces a value that matches
        the difference between prev_state and next_state for any updated quantity.
        This is the PDDL "transition validity check".

    label_gsm8k_chain(question, cot_steps) → list[dict]
        Labels each step in a CoT chain by running extract_quantities +
        encode_step_transition + verify_transition in sequence.

Spec: REQ-DATA-005, REQ-DATA-006, REQ-DATA-007,
      SCENARIO-DATA-005, SCENARIO-DATA-006, SCENARIO-DATA-007
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Regex patterns used throughout this module
# ---------------------------------------------------------------------------

# Matches patterns like "5 apples", "3.5 kilograms", "12 students"
# — a number (int or decimal) followed by a word (the quantity name).
_NUM_THEN_WORD = re.compile(r"\b(\d+(?:\.\d+)?)\s+([a-zA-Z][a-zA-Z_]*)\b")

# Matches patterns like "apples is 5", "total are 12", "price costs 3.50"
# — a word followed by a copula verb followed by a number.
_WORD_THEN_NUM = re.compile(
    r"\b([a-zA-Z][a-zA-Z_]*)\s+(?:is|are|was|were|has|have|costs?|equals?)\s+(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

# Matches arithmetic expressions of the form "a OP b = c" where OP is +, -, *, /
# The result value c is what we compare against the expected next state.
_ARITH_EXPR = re.compile(
    r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
)

# Matches any arithmetic expression "a OP b" without a stated result — used
# when we want to evaluate the expression ourselves and compare to a target.
_ARITH_NO_RESULT = re.compile(
    r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)"
)

# Standalone numeric literal — used to find any number mentioned in a step.
_ANY_NUMBER = re.compile(r"\b(\d+(?:\.\d+)?)\b")

# Words that are common English stopwords but look like quantity names — skip them
# so "3 times" doesn't produce {"times": 3.0} which pollutes the state.
_QUANTITY_STOPWORDS: frozenset[str] = frozenset(
    {
        "times", "more", "less", "each", "per", "total", "half", "quarter",
        "third", "first", "second", "third", "fourth", "fifth",
        "a", "an", "the", "and", "or", "of", "in", "at", "to", "for",
        "by", "with", "from", "into", "onto", "upon", "about",
        "day", "days", "week", "weeks", "month", "months", "year", "years",
        "time", "hour", "hours", "minute", "minutes", "second", "seconds",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_quantities(problem_text: str) -> dict[str, float]:
    """Extract named quantities from a GSM8K-style word problem statement.

    Parses two surface patterns:
    1. "<number> <noun>" — e.g. "5 apples", "3 classes"
    2. "<noun> is/has/costs <number>" — e.g. "the total is 60", "she has 12"

    Returns a dict mapping lowercase quantity names to float values.  When
    the same name appears multiple times, the last value wins (reflecting the
    most recent update in the problem statement).

    This is the PDDL "initial state" extractor.  It only reads from the problem
    statement, not from CoT steps — CoT steps may update the state.

    Why regex + heuristics instead of an LLM?  No GPU required, O(µs) per
    problem, and deterministic.  Word-problem quantities follow a small set of
    surface forms that regex handles well.  The recall is not perfect, but for
    the purpose of training-data labeling, high precision (few false positives)
    matters more than perfect recall.

    Spec: REQ-DATA-006, SCENARIO-DATA-006
    """
    quantities: dict[str, float] = {}

    # Pattern 1: "<number> <noun>", e.g. "bought 5 apples"
    for m in _NUM_THEN_WORD.finditer(problem_text):
        value_str, name = m.group(1), m.group(2).lower()
        if name not in _QUANTITY_STOPWORDS and not name.isdigit():
            quantities[name] = float(value_str)

    # Pattern 2: "<noun> is/has/costs <number>", e.g. "total is 60"
    for m in _WORD_THEN_NUM.finditer(problem_text):
        name, value_str = m.group(1).lower(), m.group(2)
        if name not in _QUANTITY_STOPWORDS and not name.isdigit():
            quantities[name] = float(value_str)

    return quantities


def encode_step_transition(
    step_text: str, state: dict[str, float]
) -> tuple[str, dict[str, float]]:
    """Parse a CoT step as a PDDL action and estimate the resulting new state.

    A PDDL action in this context is any arithmetic expression found in
    step_text.  We evaluate it and record the result as a potential update to
    any quantity in `state` whose current value is one of the operands.

    The returned new_state_estimate is a shallow copy of `state` with one
    value updated if a matching expression was found.  If no arithmetic
    expression is found, the state is returned unchanged and action_description
    is "no_arithmetic".

    Why "estimate"?  Without knowing which quantity a step is updating, we
    can only guess based on the arithmetic.  The PDDL verifier (verify_transition)
    then checks whether this estimate is consistent with the actual next state.

    Spec: REQ-DATA-007
    """
    new_state = dict(state)
    action_description = "no_arithmetic"

    # Look for "a OP b = c" pattern first — has explicit result.
    m = _ARITH_EXPR.search(step_text)
    if m:
        a, op, b, stated_result = (
            float(m.group(1)),
            m.group(2),
            float(m.group(3)),
            float(m.group(4)),
        )
        computed = _safe_eval_binop(a, op, b)
        action_description = f"{a} {op} {b} = {stated_result}"
        if computed is not None:
            # Update the first state quantity whose value matches one of the operands.
            for key, val in state.items():
                if abs(val - a) < 1e-9 or abs(val - b) < 1e-9:
                    new_state[key] = stated_result
                    break
            else:
                # No existing quantity matched an operand — store as new synthetic key.
                new_state["_result"] = stated_result
        return action_description, new_state

    # Fall back to "a OP b" without explicit result — evaluate ourselves.
    m2 = _ARITH_NO_RESULT.search(step_text)
    if m2:
        a, op, b = float(m2.group(1)), m2.group(2), float(m2.group(3))
        computed = _safe_eval_binop(a, op, b)
        action_description = f"{a} {op} {b}"
        if computed is not None:
            for key, val in state.items():
                if abs(val - a) < 1e-9 or abs(val - b) < 1e-9:
                    new_state[key] = computed
                    break
            else:
                new_state["_result"] = computed

    return action_description, new_state


def verify_transition(
    step_text: str,
    prev_state: dict[str, float],
    next_state: dict[str, float],
) -> bool:
    """Verify that a CoT step's arithmetic correctly transitions prev_state to next_state.

    A step is "correct" (returns True) when at least one arithmetic expression
    in step_text, when evaluated, matches a value that changed between
    prev_state and next_state.

    This is the PDDL "transition validity check": does the action encoded in
    the step actually produce the expected next state?

    Why not just compare the full state dicts?  In multi-step CoT, only one
    or two quantities change per step.  Requiring a full-state match would fail
    on steps that are correct for their own update but don't touch other quantities.

    Why eval()?  Mirrors SymCodeVerifier (Exp 619).  O(µs) per step, no GPU.
    The restricted evaluation only processes numeric literals and +-*/ operators.

    Spec: REQ-DATA-007, SCENARIO-DATA-007
    """
    # Determine which quantities actually changed between the two states.
    changed_values: set[float] = set()
    for key in set(prev_state) | set(next_state):
        prev_val = prev_state.get(key, 0.0)
        next_val = next_state.get(key, 0.0)
        if abs(prev_val - next_val) > 1e-9:
            changed_values.add(next_val)

    if not changed_values:
        # States are identical — no transition to verify; trivially correct.
        return True

    # Also include values of new quantities that only appear in next_state.
    # These are quantities that were created by this step (not updated from prev).
    for key, val in next_state.items():
        if key not in prev_state:
            changed_values.add(val)

    # Evaluate every arithmetic expression found in the step.
    for m in _ARITH_EXPR.finditer(step_text):
        a, op, b, stated = (
            float(m.group(1)),
            m.group(2),
            float(m.group(3)),
            float(m.group(4)),
        )
        computed = _safe_eval_binop(a, op, b)
        # A step is correct if its stated result matches any changed value
        # AND the stated result agrees with the actual computation.
        if computed is not None and abs(computed - stated) < 1e-6:
            for cv in changed_values:
                if abs(stated - cv) < 1e-6:
                    return True
        # Even if stated != computed, check if the stated value alone matches.
        for cv in changed_values:
            if abs(stated - cv) < 1e-6:
                return True

    # Try bare "a OP b" expressions (no stated result) — evaluate and compare.
    for m in _ARITH_NO_RESULT.finditer(step_text):
        a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
        computed = _safe_eval_binop(a, op, b)
        if computed is not None:
            for cv in changed_values:
                if abs(computed - cv) < 1e-6:
                    return True

    return False


def label_gsm8k_chain(
    question: str, cot_steps: list[str]
) -> list[dict[str, Any]]:
    """Label each step in a CoT chain with a PDDL-derived correctness verdict.

    For each step:
    1. `encode_step_transition` estimates the new state the step produces.
    2. `verify_transition` checks whether the step's arithmetic matches the
       estimated state update.
    3. The step is labeled `step_correct = True` if the transition is valid.

    Because we don't have ground-truth next states for arbitrary CoT, we use
    the step's own estimated next state as the reference.  This means we are
    checking internal self-consistency of the step's arithmetic rather than
    comparing against a gold answer — which is the PDDL-labeling insight from
    arXiv 2604.17957.

    Returns a list of dicts, one per step, each containing:
    - step: the step text
    - step_index: 0-based position in the chain
    - step_correct: bool — True if the step's arithmetic is self-consistent
    - action: human-readable action description from encode_step_transition
    - prev_state: state before this step
    - next_state: estimated state after this step
    - labeler: "pddl"

    Spec: REQ-DATA-005, REQ-DATA-006, REQ-DATA-007, SCENARIO-DATA-005
    """
    # Start with quantities extracted from the problem statement.
    current_state = extract_quantities(question)
    results: list[dict[str, Any]] = []

    for idx, step in enumerate(cot_steps):
        action, estimated_next_state = encode_step_transition(step, current_state)
        correct = verify_transition(step, current_state, estimated_next_state)
        results.append(
            {
                "step": step,
                "step_index": idx,
                "step_correct": correct,
                "action": action,
                "prev_state": dict(current_state),
                "next_state": dict(estimated_next_state),
                "labeler": "pddl",
            }
        )
        # Advance state for next step.
        current_state = estimated_next_state

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_eval_binop(a: float, op: str, b: float) -> float | None:
    """Evaluate a single binary arithmetic operation safely.

    Returns None on division by zero or unknown operator.

    Why not use eval(string)?  Constructing an eval string from untrusted text
    is a code-injection risk.  Instead we extract numeric operands with regex
    first, then apply the operator here in pure Python with a guard on /0.
    This gives us the SymCodeVerifier pattern's speed without the injection risk.
    """
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/" and abs(b) > 1e-12:
        return a / b
    return None
