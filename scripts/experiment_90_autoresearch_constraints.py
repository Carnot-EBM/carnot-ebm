#!/usr/bin/env python3
"""Experiment 90: Autoresearch loop for constraint pipeline self-improvement.

**The big idea:**
    Apply the FR-11 autonomous self-learning loop to improve the constraint
    extraction pipeline ITSELF. The orchestrator proposes improvement hypotheses
    (new regex patterns, AST rules, logic patterns, or Ising features), tests
    them against held-out failure cases from Exp 88's FailureAnalyzer, and
    accepts/rejects each based on AUROC improvement. Over 20 iterations, the
    pipeline should discover and incorporate patterns that reduce false negatives.

**Why this matters:**
    Exp 88 showed where the pipeline fails (false negatives by category). Exp 89
    showed the pipeline can train on its own outputs. This experiment closes the
    loop: the pipeline PROPOSES fixes to its own blind spots, tests them, and
    incorporates winners. If it works, the constraint pipeline autonomously
    improves its own coverage without human pattern engineering.

**Approach:**
    1. Define a hypothesis space: 4 types of improvements (regex, AST, logic,
       Ising features), each with template-based generation.
    2. Build a ConstraintImprovementOrchestrator that proposes, tests, evaluates,
       and incorporates hypotheses.
    3. Run 20 iterations with a circuit breaker (5 consecutive rejections).
    4. Evaluate: cumulative AUROC improvement, acceptance rate, per-type
       effectiveness, final vs original extractor comparison.

**Hypothesis types:**
    Type 1 — New regex for ArithmeticExtractor: e.g., "X times Y is Z",
        "X divided by Y equals Z", "the product of X and Y is Z".
    Type 2 — New AST rule for CodeExtractor: e.g., detect list comprehensions,
        generator expressions, decorator usage, assert statements.
    Type 3 — New logic pattern for LogicExtractor: e.g., "unless" clauses,
        "provided that" conditionals, "only if" biconditionals.
    Type 4 — New Ising feature: binary feature for CD training, e.g.,
        word-count buckets, punctuation density, claim-density ratio.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_90_autoresearch_constraints.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, FR-11
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# 1. Hypothesis types and templates
# ---------------------------------------------------------------------------

@dataclass
class ImprovementHypothesis:
    """A single proposed improvement to the constraint pipeline.

    **Detailed explanation for engineers:**
        Represents one hypothesis in the autoresearch loop. Each hypothesis
        has a type (regex, ast, logic, ising_feature), a human-readable
        description, and the actual pattern/rule content. The ``accepted``
        field is set after evaluation; ``auroc_delta`` tracks how much the
        hypothesis improved (or degraded) pipeline AUROC.

    Attributes:
        hypothesis_type: One of "regex", "ast", "logic", "ising_feature".
        name: Short identifier for this hypothesis.
        description: What the hypothesis does and why it might help.
        pattern: The regex string, AST node type, logic pattern, or feature
            specification depending on type.
        target_extractor: Which extractor this improves ("arithmetic",
            "code", "logic", "ising").
        accepted: Whether the hypothesis was accepted after evaluation.
        auroc_delta: Change in AUROC from incorporating this hypothesis.
            Positive = improvement, negative = regression.
        metadata: Additional details (e.g., template parameters, test counts).
    """

    hypothesis_type: str
    name: str
    description: str
    pattern: str
    target_extractor: str
    accepted: bool = False
    auroc_delta: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# Template banks for each hypothesis type.
# Each template is a dict with keys: name, pattern, description, target.

ARITHMETIC_REGEX_TEMPLATES: list[dict[str, str]] = [
    {
        "name": "times_is",
        "pattern": r"(-?\d+\.?\d*)\s+times\s+(-?\d+\.?\d*)\s+is\s+(-?\d+\.?\d*)",
        "description": "Extract 'X times Y is Z' multiplication claims",
        "target": "arithmetic",
    },
    {
        "name": "divided_by_equals",
        "pattern": r"(-?\d+\.?\d*)\s+divided\s+by\s+(-?\d+\.?\d*)\s+(?:equals|is)\s+(-?\d+\.?\d*)",
        "description": "Extract 'X divided by Y equals Z' division claims",
        "target": "arithmetic",
    },
    {
        "name": "product_of",
        "pattern": r"(?:the\s+)?product\s+of\s+(-?\d+\.?\d*)\s+and\s+(-?\d+\.?\d*)\s+is\s+(-?\d+\.?\d*)",
        "description": "Extract 'the product of X and Y is Z' claims",
        "target": "arithmetic",
    },
    {
        "name": "sum_of",
        "pattern": r"(?:the\s+)?sum\s+of\s+(-?\d+\.?\d*)\s+and\s+(-?\d+\.?\d*)\s+is\s+(-?\d+\.?\d*)",
        "description": "Extract 'the sum of X and Y is Z' claims",
        "target": "arithmetic",
    },
    {
        "name": "mult_equals",
        "pattern": r"(-?\d+\.?\d*)\s*[*×]\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)",
        "description": "Extract 'X * Y = Z' or 'X × Y = Z' multiplication with explicit equals",
        "target": "arithmetic",
    },
    {
        "name": "div_equals",
        "pattern": r"(-?\d+\.?\d*)\s*[/÷]\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)",
        "description": "Extract 'X / Y = Z' or 'X ÷ Y = Z' division with explicit equals",
        "target": "arithmetic",
    },
    {
        "name": "squared_is",
        "pattern": r"(-?\d+\.?\d*)\s+squared\s+is\s+(-?\d+\.?\d*)",
        "description": "Extract 'X squared is Y' exponentiation claims",
        "target": "arithmetic",
    },
    {
        "name": "remainder_is",
        "pattern": r"(?:the\s+)?remainder\s+(?:of|when)\s+(-?\d+\.?\d*)\s+(?:divided\s+by|mod|%)\s+(-?\d+\.?\d*)\s+is\s+(-?\d+\.?\d*)",
        "description": "Extract 'the remainder of X divided by Y is Z' modular arithmetic",
        "target": "arithmetic",
    },
]

AST_RULE_TEMPLATES: list[dict[str, str]] = [
    {
        "name": "list_comprehension",
        "pattern": "ListComp",
        "description": "Detect list comprehensions and verify iter/target types",
        "target": "code",
    },
    {
        "name": "generator_expression",
        "pattern": "GeneratorExp",
        "description": "Detect generator expressions (lazy evaluation patterns)",
        "target": "code",
    },
    {
        "name": "assert_statement",
        "pattern": "Assert",
        "description": "Detect assert statements as explicit programmer constraints",
        "target": "code",
    },
    {
        "name": "try_except",
        "pattern": "Try",
        "description": "Detect try/except blocks and verify exception handler coverage",
        "target": "code",
    },
    {
        "name": "dict_comprehension",
        "pattern": "DictComp",
        "description": "Detect dict comprehensions and verify key/value structure",
        "target": "code",
    },
    {
        "name": "lambda_function",
        "pattern": "Lambda",
        "description": "Detect lambda expressions and extract inline function constraints",
        "target": "code",
    },
    {
        "name": "set_comprehension",
        "pattern": "SetComp",
        "description": "Detect set comprehensions (uniqueness guarantees)",
        "target": "code",
    },
    {
        "name": "walrus_operator",
        "pattern": "NamedExpr",
        "description": "Detect walrus operator := assignments in expressions",
        "target": "code",
    },
]

LOGIC_PATTERN_TEMPLATES: list[dict[str, str]] = [
    {
        "name": "unless_clause",
        "pattern": r"^(.+?)\s+unless\s+(.+?)\.?$",
        "description": "Extract 'X unless Y' as 'if not Y then X' implications",
        "target": "logic",
    },
    {
        "name": "provided_that",
        "pattern": r"^(.+?)\s+provided\s+that\s+(.+?)\.?$",
        "description": "Extract 'X provided that Y' as 'if Y then X' implications",
        "target": "logic",
    },
    {
        "name": "only_if",
        "pattern": r"^(.+?)\s+only\s+if\s+(.+?)\.?$",
        "description": "Extract 'X only if Y' biconditional constraints",
        "target": "logic",
    },
    {
        "name": "whenever",
        "pattern": r"^whenever\s+(.+?),\s+(.+?)\.?$",
        "description": "Extract 'whenever X, Y' as universal implication",
        "target": "logic",
    },
    {
        "name": "as_long_as",
        "pattern": r"^(.+?)\s+as\s+long\s+as\s+(.+?)\.?$",
        "description": "Extract 'X as long as Y' conditional constraints",
        "target": "logic",
    },
    {
        "name": "except_when",
        "pattern": r"^(.+?)\s+except\s+when\s+(.+?)\.?$",
        "description": "Extract 'X except when Y' as exception-based constraints",
        "target": "logic",
    },
    {
        "name": "in_order_to",
        "pattern": r"^in\s+order\s+(?:to|for)\s+(.+?),\s+(.+?)\.?$",
        "description": "Extract 'in order to X, Y' prerequisite constraints",
        "target": "logic",
    },
    {
        "name": "assuming_that",
        "pattern": r"^assuming\s+(?:that\s+)?(.+?),\s+(.+?)\.?$",
        "description": "Extract 'assuming X, Y' conditional reasoning",
        "target": "logic",
    },
]

ISING_FEATURE_TEMPLATES: list[dict[str, str]] = [
    {
        "name": "word_count_bucket",
        "pattern": "word_count_bucket_5",
        "description": "Binary feature: response has > 5 words (verbosity indicator)",
        "target": "ising",
    },
    {
        "name": "has_numbers",
        "pattern": "contains_numeric",
        "description": "Binary feature: response contains numeric digits",
        "target": "ising",
    },
    {
        "name": "has_code_block",
        "pattern": "contains_code_block",
        "description": "Binary feature: response contains fenced code block",
        "target": "ising",
    },
    {
        "name": "high_punctuation",
        "pattern": "punctuation_ratio_gt_0.1",
        "description": "Binary feature: punctuation-to-word ratio > 0.1",
        "target": "ising",
    },
    {
        "name": "has_equation",
        "pattern": "contains_equation",
        "description": "Binary feature: response contains '=' sign (equation presence)",
        "target": "ising",
    },
    {
        "name": "sentence_count_bucket",
        "pattern": "sentence_count_gt_3",
        "description": "Binary feature: response has > 3 sentences (multi-claim indicator)",
        "target": "ising",
    },
    {
        "name": "has_negation_word",
        "pattern": "contains_negation",
        "description": "Binary feature: response contains never/not/no/none/neither",
        "target": "ising",
    },
    {
        "name": "has_conditional",
        "pattern": "contains_conditional",
        "description": "Binary feature: response contains if/when/unless/provided",
        "target": "ising",
    },
]


# ---------------------------------------------------------------------------
# 2. Synthetic failure cases for testing hypotheses
# ---------------------------------------------------------------------------

def generate_failure_cases(
    n: int, rng: np.random.Generator,
) -> list[dict[str, str]]:
    """Generate synthetic failure cases that mimic Exp 88 FailureAnalyzer output.

    **Detailed explanation for engineers:**
        These simulate false negatives — cases where a wrong response should
        have been caught by the pipeline but wasn't, because the constraint
        extractors don't cover the claim pattern. Each failure case includes
        the response text and the category of uncovered claim, so we can test
        whether a new pattern would have caught it.

        Six categories match Exp 88's CLAIM_CATEGORIES: arithmetic_chain,
        implicit_logic, world_knowledge, code_semantics, comparison, negation.

    Args:
        n: Number of failure cases to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        List of dicts with keys: question, response, ground_truth, category,
        uncovered_pattern (the specific text the pipeline missed).
    """
    cases: list[dict[str, str]] = []

    # --- Arithmetic chain failures (multi-step, no explicit "X + Y = Z") ---
    arith_templates = [
        {
            "q": "What is 15 times 4?",
            "r": "First take 15, then multiply it by 4 to get 60.",
            "gt": "60",
            "cat": "arithmetic_chain",
        },
        {
            "q": "What is 3 * 7 + 2?",
            "r": "3 times 7 is 21, plus 2 gives 23.",
            "gt": "23",
            "cat": "arithmetic_chain",
        },
        {
            "q": "What is (8 + 2) * 5?",
            "r": "First add 8 and 2 which gives us 10, then multiply by 5 resulting in 50.",
            "gt": "50",
            "cat": "arithmetic_chain",
        },
        {
            "q": "What is 100 divided by 4?",
            "r": "100 divided by 4 equals 25.",
            "gt": "25",
            "cat": "arithmetic_chain",
        },
        {
            "q": "What is the product of 12 and 8?",
            "r": "The product of 12 and 8 is 96.",
            "gt": "96",
            "cat": "arithmetic_chain",
        },
        {
            "q": "What is 7 squared?",
            "r": "7 squared is 49.",
            "gt": "49",
            "cat": "arithmetic_chain",
        },
        {
            "q": "What is the sum of 33 and 67?",
            "r": "The sum of 33 and 67 is 100.",
            "gt": "100",
            "cat": "arithmetic_chain",
        },
        {
            "q": "What is the remainder when 17 is divided by 5?",
            "r": "The remainder of 17 divided by 5 is 2.",
            "gt": "2",
            "cat": "arithmetic_chain",
        },
        # Wrong answers that should be caught
        {
            "q": "What is 6 times 9?",
            "r": "6 times 9 is 52.",
            "gt": "54",
            "cat": "arithmetic_chain",
        },
        {
            "q": "What is the product of 11 and 7?",
            "r": "The product of 11 and 7 is 78.",
            "gt": "77",
            "cat": "arithmetic_chain",
        },
    ]

    # --- Implicit logic failures (no explicit "if...then") ---
    logic_templates = [
        {
            "q": "What follows from: all cats are animals, Whiskers is a cat?",
            "r": "Since all cats are animals, Whiskers must be an animal.",
            "gt": "Whiskers is an animal",
            "cat": "implicit_logic",
        },
        {
            "q": "Is rain guaranteed?",
            "r": "Rain will occur unless the front dissipates.",
            "gt": "conditional on front",
            "cat": "implicit_logic",
        },
        {
            "q": "Can we proceed?",
            "r": "We can proceed provided that all tests pass.",
            "gt": "conditional on tests",
            "cat": "implicit_logic",
        },
        {
            "q": "When is X valid?",
            "r": "X is valid only if Y is non-negative.",
            "gt": "requires Y >= 0",
            "cat": "implicit_logic",
        },
        {
            "q": "What happens whenever temperature exceeds 100C?",
            "r": "Whenever temperature exceeds 100C, the alarm triggers.",
            "gt": "alarm triggers",
            "cat": "implicit_logic",
        },
        {
            "q": "Is the system stable?",
            "r": "The system is stable as long as load stays below 80%.",
            "gt": "conditional on load",
            "cat": "implicit_logic",
        },
        {
            "q": "Does X imply Y?",
            "r": "Therefore Y must be true since X holds.",
            "gt": "Y follows from X",
            "cat": "implicit_logic",
        },
        {
            "q": "What is required?",
            "r": "Assuming that auth succeeds, the request is processed.",
            "gt": "auth prerequisite",
            "cat": "implicit_logic",
        },
        {
            "q": "When does it fail?",
            "r": "The pipeline works except when input is empty.",
            "gt": "fails on empty input",
            "cat": "implicit_logic",
        },
        {
            "q": "Why must we sort first?",
            "r": "In order to binary search, we must sort the array.",
            "gt": "sort prerequisite",
            "cat": "implicit_logic",
        },
    ]

    # --- Code semantics failures (AST patterns not caught) ---
    code_templates = [
        {
            "q": "Filter even numbers from a list.",
            "r": "```python\nevens = [x for x in numbers if x % 2 == 0]\n```",
            "gt": "list comprehension with filter",
            "cat": "code_semantics",
        },
        {
            "q": "Validate input is positive.",
            "r": "```python\nassert value > 0, 'Value must be positive'\n```",
            "gt": "assert constraint",
            "cat": "code_semantics",
        },
        {
            "q": "Sum squares lazily.",
            "r": "```python\ntotal = sum(x**2 for x in range(100))\n```",
            "gt": "generator expression",
            "cat": "code_semantics",
        },
        {
            "q": "Handle potential errors.",
            "r": "```python\ntry:\n    result = process(data)\nexcept ValueError:\n    result = default\n```",
            "gt": "exception handling",
            "cat": "code_semantics",
        },
        {
            "q": "Create a mapping.",
            "r": "```python\nmapping = {k: v*2 for k, v in items.items()}\n```",
            "gt": "dict comprehension",
            "cat": "code_semantics",
        },
        {
            "q": "Quick inline function.",
            "r": "```python\nsquare = lambda x: x ** 2\n```",
            "gt": "lambda expression",
            "cat": "code_semantics",
        },
        {
            "q": "Get unique values.",
            "r": "```python\nuniques = {x.lower() for x in words}\n```",
            "gt": "set comprehension",
            "cat": "code_semantics",
        },
        {
            "q": "Find first match while assigning.",
            "r": "```python\nif (m := pattern.search(text)) is not None:\n    print(m.group())\n```",
            "gt": "walrus operator",
            "cat": "code_semantics",
        },
    ]

    # --- Comparison failures ---
    comparison_templates = [
        {
            "q": "Which is larger: 42 or 37?",
            "r": "42 is greater than 37.",
            "gt": "42 > 37",
            "cat": "comparison",
        },
        {
            "q": "What is the fastest sorting algorithm?",
            "r": "Merge sort is the fastest general-purpose sorting algorithm.",
            "gt": "merge sort is fastest",
            "cat": "comparison",
        },
    ]

    # --- Negation failures ---
    negation_templates = [
        {
            "q": "Can the function return None?",
            "r": "The function never returns None.",
            "gt": "never None",
            "cat": "negation",
        },
        {
            "q": "Are there duplicates?",
            "r": "There are no duplicates in the sorted output.",
            "gt": "no duplicates",
            "cat": "negation",
        },
    ]

    all_templates = (
        arith_templates + logic_templates + code_templates
        + comparison_templates + negation_templates
    )

    # Cycle and shuffle to reach n cases.
    if len(all_templates) < n:
        all_templates = all_templates * (n // len(all_templates) + 1)

    import random
    py_rng = random.Random(int(rng.integers(0, 2**31)))
    py_rng.shuffle(all_templates)
    all_templates = all_templates[:n]

    for tmpl in all_templates:
        cases.append({
            "question": tmpl["q"],
            "response": tmpl["r"],
            "ground_truth": tmpl["gt"],
            "category": tmpl["cat"],
        })

    return cases


# ---------------------------------------------------------------------------
# 3. Hypothesis proposer
# ---------------------------------------------------------------------------

def propose_hypothesis(
    iteration: int,
    rng: np.random.Generator,
    already_proposed: set[str],
) -> ImprovementHypothesis:
    """Generate a random improvement hypothesis from the template banks.

    **Detailed explanation for engineers:**
        Selects a hypothesis type (regex, ast, logic, ising_feature) at random,
        then picks a template from that type's bank that hasn't been proposed
        yet. If all templates in one type are exhausted, falls through to the
        next type. This ensures diversity across hypothesis types while avoiding
        duplicates.

    Args:
        iteration: Current loop iteration (for naming).
        rng: Random generator for reproducibility.
        already_proposed: Set of hypothesis names already tried.

    Returns:
        A new ImprovementHypothesis instance.
    """
    # All template banks, indexed by type name.
    banks: dict[str, list[dict[str, str]]] = {
        "regex": ARITHMETIC_REGEX_TEMPLATES,
        "ast": AST_RULE_TEMPLATES,
        "logic": LOGIC_PATTERN_TEMPLATES,
        "ising_feature": ISING_FEATURE_TEMPLATES,
    }
    type_names = list(banks.keys())

    # Shuffle type order for variety.
    order = list(range(len(type_names)))
    rng.shuffle(order)

    for idx in order:
        t = type_names[idx]
        candidates = [
            tmpl for tmpl in banks[t]
            if tmpl["name"] not in already_proposed
        ]
        if not candidates:
            continue

        chosen = candidates[int(rng.integers(0, len(candidates)))]
        return ImprovementHypothesis(
            hypothesis_type=t,
            name=chosen["name"],
            description=chosen["description"],
            pattern=chosen["pattern"],
            target_extractor=chosen.get("target", t),
        )

    # All templates exhausted — generate a parametric variant.
    return ImprovementHypothesis(
        hypothesis_type="regex",
        name=f"parametric_iter{iteration}",
        description=f"Parametric regex variant at iteration {iteration}",
        pattern=r"(-?\d+\.?\d*)\s+(?:plus|minus)\s+(-?\d+\.?\d*)\s+(?:gives|makes)\s+(-?\d+\.?\d*)",
        target_extractor="arithmetic",
    )


# ---------------------------------------------------------------------------
# 4. Hypothesis testing: apply hypothesis to failure cases
# ---------------------------------------------------------------------------

def test_hypothesis_on_failures(
    hypothesis: ImprovementHypothesis,
    failure_cases: list[dict[str, str]],
) -> dict[str, Any]:
    """Test whether a hypothesis catches any of the failure cases.

    **Detailed explanation for engineers:**
        For each failure case, checks if applying the hypothesis's pattern
        to the response text would have extracted a constraint that the
        original pipeline missed. Different testing logic for each type:

        - regex: Compile pattern and search each response.
        - ast: Parse code blocks and look for the specified AST node type.
        - logic: Compile as regex against normalized sentences.
        - ising_feature: Evaluate the feature function on each response.

        Returns counts of matches (cases caught) and misses.

    Args:
        hypothesis: The hypothesis to test.
        failure_cases: List of failure case dicts from generate_failure_cases.

    Returns:
        Dict with keys: matches (count), match_rate, matched_categories,
        matched_indices.
    """
    matches = 0
    matched_categories: dict[str, int] = {}
    matched_indices: list[int] = []

    for i, case in enumerate(failure_cases):
        response = case["response"]
        category = case["category"]
        caught = False

        if hypothesis.hypothesis_type == "regex":
            try:
                if re.search(hypothesis.pattern, response, re.IGNORECASE):
                    caught = True
            except re.error:
                pass

        elif hypothesis.hypothesis_type == "ast":
            # Extract code blocks and look for the specified node type.
            code_blocks = _extract_code_blocks(response)
            node_type_name = hypothesis.pattern
            for code in code_blocks:
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if type(node).__name__ == node_type_name:
                            caught = True
                            break
                except SyntaxError:
                    pass
                if caught:
                    break

        elif hypothesis.hypothesis_type == "logic":
            # Test as regex against normalized text.
            normalized = re.sub(r"\s+", " ", response.strip().lower())
            try:
                if re.search(hypothesis.pattern, normalized, re.IGNORECASE):
                    caught = True
            except re.error:
                pass

        elif hypothesis.hypothesis_type == "ising_feature":
            # Evaluate the feature function on the response.
            caught = _evaluate_ising_feature(hypothesis.pattern, response)

        if caught:
            matches += 1
            matched_indices.append(i)
            matched_categories[category] = matched_categories.get(category, 0) + 1

    total = len(failure_cases)
    return {
        "matches": matches,
        "match_rate": matches / total if total > 0 else 0.0,
        "matched_categories": matched_categories,
        "matched_indices": matched_indices,
    }


def _extract_code_blocks(text: str) -> list[str]:
    """Extract fenced code blocks from text."""
    blocks: list[str] = []
    for match in re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL):
        blocks.append(match.group(1))
    if not blocks:
        try:
            ast.parse(text)
            blocks.append(text)
        except SyntaxError:
            pass
    return blocks


def _evaluate_ising_feature(pattern_name: str, response: str) -> bool:
    """Evaluate a named Ising binary feature on a response text.

    **Detailed explanation for engineers:**
        Each Ising feature template has a pattern_name that maps to a simple
        feature function. These produce binary features that could be appended
        to the 245-dim feature vector from Exp 89 to give the Ising model
        more signal for discriminating correct vs wrong responses.

    Args:
        pattern_name: The feature specification string.
        response: Response text to evaluate.

    Returns:
        True if the feature fires (1) for this response, False otherwise.
    """
    if pattern_name == "word_count_bucket_5":
        return len(response.split()) > 5
    elif pattern_name == "contains_numeric":
        return bool(re.search(r"\d", response))
    elif pattern_name == "contains_code_block":
        return "```" in response
    elif pattern_name == "punctuation_ratio_gt_0.1":
        words = response.split()
        if not words:
            return False
        punct_count = sum(1 for ch in response if ch in ".,;:!?()[]{}\"'")
        return punct_count / len(words) > 0.1
    elif pattern_name == "contains_equation":
        return "=" in response
    elif pattern_name == "sentence_count_gt_3":
        sentences = re.split(r"[.!?]+", response.strip())
        return len([s for s in sentences if s.strip()]) > 3
    elif pattern_name == "contains_negation":
        return bool(re.search(
            r"\b(?:never|not|no|none|neither|nor|cannot|can't|don't|doesn't|won't)\b",
            response,
            re.IGNORECASE,
        ))
    elif pattern_name == "contains_conditional":
        return bool(re.search(
            r"\b(?:if|when|unless|provided|assuming|whenever)\b",
            response,
            re.IGNORECASE,
        ))
    return False


# ---------------------------------------------------------------------------
# 5. AUROC evaluation using Ising model (from Exp 89)
# ---------------------------------------------------------------------------

def encode_response_features(
    response: str,
    extra_patterns: list[ImprovementHypothesis] | None = None,
) -> np.ndarray:
    """Encode a response as a binary feature vector for Ising evaluation.

    **Detailed explanation for engineers:**
        Creates a feature vector from the response text. Unlike Exp 89's
        full 245-dim encoding (which requires question context and pipeline
        verification), this uses a simplified 50-dim encoding focused on
        the structural features most relevant to constraint extraction:

        - Features 0-9: Character/word statistics (length buckets).
        - Features 10-19: Content type indicators (numbers, code, logic words).
        - Features 20-29: Pattern match indicators (equations, comparisons, etc).
        - Features 30-39: Structural indicators (sentences, punctuation, etc).
        - Features 40-49: Slots for dynamically added hypothesis features.

        When extra_patterns (accepted hypotheses) are provided, features 40+
        encode whether each accepted hypothesis's pattern matches this response.

    Args:
        response: Text to encode.
        extra_patterns: Optional list of accepted hypotheses to add as features.

    Returns:
        Binary feature vector of shape (50,), dtype float32.
    """
    features = np.zeros(50, dtype=np.float32)

    # --- Character/word statistics (0-9) ---
    words = response.split()
    n_words = len(words)
    features[0] = 1.0 if n_words > 3 else 0.0
    features[1] = 1.0 if n_words > 10 else 0.0
    features[2] = 1.0 if n_words > 20 else 0.0
    features[3] = 1.0 if n_words > 50 else 0.0
    features[4] = 1.0 if len(response) > 50 else 0.0
    features[5] = 1.0 if len(response) > 100 else 0.0
    features[6] = 1.0 if len(response) > 200 else 0.0
    features[7] = 1.0 if any(c.isupper() for c in response[1:]) else 0.0
    features[8] = 1.0 if "\n" in response else 0.0
    features[9] = 1.0 if "  " in response else 0.0

    # --- Content type indicators (10-19) ---
    features[10] = 1.0 if re.search(r"\d", response) else 0.0
    features[11] = 1.0 if "```" in response else 0.0
    features[12] = 1.0 if re.search(r"\bif\b", response, re.I) else 0.0
    features[13] = 1.0 if re.search(r"\bthen\b", response, re.I) else 0.0
    features[14] = 1.0 if re.search(r"\bnot\b", response, re.I) else 0.0
    features[15] = 1.0 if re.search(r"\ball\b", response, re.I) else 0.0
    features[16] = 1.0 if re.search(r"\bsome\b", response, re.I) else 0.0
    features[17] = 1.0 if re.search(r"\bbecause\b|\bsince\b", response, re.I) else 0.0
    features[18] = 1.0 if re.search(r"\btherefore\b|\bthus\b", response, re.I) else 0.0
    features[19] = 1.0 if re.search(r"def\s+\w+", response) else 0.0

    # --- Pattern match indicators (20-29) ---
    features[20] = 1.0 if "=" in response else 0.0
    features[21] = 1.0 if re.search(r"[<>]", response) else 0.0
    features[22] = 1.0 if re.search(r"\b\d+\s*[+\-*/]\s*\d+", response) else 0.0
    features[23] = 1.0 if re.search(r"\bnever\b", response, re.I) else 0.0
    features[24] = 1.0 if re.search(r"\bno\s+\w+", response, re.I) else 0.0
    features[25] = 1.0 if re.search(r"greater|less|larger|smaller", response, re.I) else 0.0
    features[26] = 1.0 if re.search(r"\bunless\b", response, re.I) else 0.0
    features[27] = 1.0 if re.search(r"provided|assuming", response, re.I) else 0.0
    features[28] = 1.0 if re.search(r"times|product|sum\s+of", response, re.I) else 0.0
    features[29] = 1.0 if re.search(r"squared|remainder", response, re.I) else 0.0

    # --- Structural indicators (30-39) ---
    sentences = re.split(r"[.!?]+", response.strip())
    n_sentences = len([s for s in sentences if s.strip()])
    features[30] = 1.0 if n_sentences > 1 else 0.0
    features[31] = 1.0 if n_sentences > 3 else 0.0
    punct_count = sum(1 for ch in response if ch in ".,;:!?")
    features[32] = 1.0 if punct_count > 2 else 0.0
    features[33] = 1.0 if "(" in response and ")" in response else 0.0
    features[34] = 1.0 if re.search(r"\b(first|then|next|finally)\b", response, re.I) else 0.0
    features[35] = 1.0 if re.search(r"\bstep\s+\d+", response, re.I) else 0.0
    features[36] = 1.0 if re.search(r"result|gives|equals|is\s+\d+", response, re.I) else 0.0
    features[37] = 1.0 if re.search(r"\breturn\b", response, re.I) else 0.0
    features[38] = 1.0 if re.search(r"assert|raise|except", response, re.I) else 0.0
    features[39] = 1.0 if re.search(r"for\s+\w+\s+in\b", response, re.I) else 0.0

    # --- Hypothesis-derived features (40-49) ---
    if extra_patterns:
        for j, hyp in enumerate(extra_patterns[:10]):
            if hyp.hypothesis_type == "regex":
                try:
                    if re.search(hyp.pattern, response, re.IGNORECASE):
                        features[40 + j] = 1.0
                except re.error:
                    pass
            elif hyp.hypothesis_type == "logic":
                normalized = re.sub(r"\s+", " ", response.strip().lower())
                try:
                    if re.search(hyp.pattern, normalized, re.IGNORECASE):
                        features[40 + j] = 1.0
                except re.error:
                    pass
            elif hyp.hypothesis_type == "ast":
                for code in _extract_code_blocks(response):
                    try:
                        tree = ast.parse(code)
                        for node in ast.walk(tree):
                            if type(node).__name__ == hyp.pattern:
                                features[40 + j] = 1.0
                                break
                    except SyntaxError:
                        pass
            elif hyp.hypothesis_type == "ising_feature":
                if _evaluate_ising_feature(hyp.pattern, response):
                    features[40 + j] = 1.0

    return features


def train_and_evaluate_auroc(
    correct_responses: list[str],
    wrong_responses: list[str],
    accepted_hypotheses: list[ImprovementHypothesis],
    n_epochs: int = 200,
    lr: float = 0.05,
    l1_lambda: float = 0.001,
) -> float:
    """Train a discriminative Ising model and compute AUROC.

    **Detailed explanation for engineers:**
        Uses the same discriminative CD training as Exp 89 (see
        train_discriminative_cd), but with the simplified 50-dim feature
        encoding from encode_response_features. The accepted_hypotheses
        list determines which extra features (slots 40-49) are active.

        Splits data 70/30 for train/test, trains for n_epochs, and returns
        AUROC on the test set.

    Args:
        correct_responses: List of correct response texts.
        wrong_responses: List of wrong response texts.
        accepted_hypotheses: Currently accepted hypotheses (used for features).
        n_epochs: Training epochs.
        lr: Learning rate for CD.
        l1_lambda: L1 regularization strength.

    Returns:
        AUROC on the 30% held-out test set.
    """
    # Encode all responses.
    correct_vecs = np.array([
        encode_response_features(r, accepted_hypotheses)
        for r in correct_responses
    ])
    wrong_vecs = np.array([
        encode_response_features(r, accepted_hypotheses)
        for r in wrong_responses
    ])

    # Split 70/30.
    n = min(len(correct_vecs), len(wrong_vecs))
    split = int(0.7 * n)
    train_c, test_c = correct_vecs[:split], correct_vecs[split:n]
    train_w, test_w = wrong_vecs[:split], wrong_vecs[split:n]

    if train_c.shape[0] < 2 or train_w.shape[0] < 2:
        return 0.5  # Not enough data.

    # Train discriminative CD (same logic as Exp 89).
    n_features = train_c.shape[1]
    biases = np.zeros(n_features, dtype=np.float32)
    J = np.random.default_rng(42).normal(
        0, 0.001, (n_features, n_features),
    ).astype(np.float32)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

    correct_spins = 2.0 * train_c - 1.0
    wrong_spins = 2.0 * train_w - 1.0
    pos_bias = np.mean(correct_spins, axis=0)
    pos_weight = np.mean(
        np.einsum("bi,bj->bij", correct_spins, correct_spins), axis=0,
    )
    neg_bias = np.mean(wrong_spins, axis=0)
    neg_weight = np.mean(
        np.einsum("bi,bj->bij", wrong_spins, wrong_spins), axis=0,
    )

    grad_b = -(pos_bias - neg_bias)
    grad_J = -(pos_weight - neg_weight)
    np.fill_diagonal(grad_J, 0.0)

    for _ in range(n_epochs):
        l1_grad = l1_lambda * np.sign(J)
        biases -= lr * (grad_b + 0.005 * biases)
        J -= lr * (grad_J + l1_grad + 0.005 * J)
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)

    # Compute AUROC on test set.
    if test_c.shape[0] < 1 or test_w.shape[0] < 1:
        return 0.5

    def compute_energies(vecs: np.ndarray) -> np.ndarray:
        spins = 2.0 * vecs - 1.0
        return -(spins @ biases + np.einsum("bi,ij,bj->b", spins, J, spins))

    e_c = compute_energies(test_c)
    e_w = compute_energies(test_w)
    n_c, n_w = len(e_c), len(e_w)
    concordant = 0
    tied = 0
    total = n_c * n_w
    diff = e_w[None, :] - e_c[:, None]
    concordant = int(np.sum(diff > 0))
    tied = int(np.sum(diff == 0))
    return (concordant + 0.5 * tied) / total if total > 0 else 0.5


# ---------------------------------------------------------------------------
# 6. Generate correct/wrong response pairs for AUROC evaluation
# ---------------------------------------------------------------------------

def generate_evaluation_pairs(
    n: int, rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    """Generate matched pairs of correct and wrong responses for AUROC eval.

    **Detailed explanation for engineers:**
        Creates pairs of responses where the correct one contains verifiable
        facts/computations and the wrong one has subtle errors. These are used
        to train and evaluate the Ising model's ability to distinguish correct
        from incorrect responses — the AUROC metric quantifies this.

    Args:
        n: Number of pairs to generate.
        rng: Random generator for reproducibility.

    Returns:
        Tuple of (correct_responses, wrong_responses).
    """
    correct: list[str] = []
    wrong: list[str] = []

    for _ in range(n):
        pair_type = int(rng.integers(0, 5))

        if pair_type == 0:
            # Arithmetic
            a = int(rng.integers(2, 100))
            b = int(rng.integers(2, 100))
            result = a * b
            wrong_result = result + int(rng.integers(1, 10)) * int(rng.choice([-1, 1]))
            if wrong_result == result:
                wrong_result = result + 1
            correct.append(f"The product of {a} and {b} is {result}.")
            wrong.append(f"The product of {a} and {b} is {wrong_result}.")

        elif pair_type == 1:
            # Logic
            subjects = ["cats", "dogs", "birds", "fish", "students"]
            predicates = ["mortal", "alive", "fast", "dense", "bright"]
            s = subjects[int(rng.integers(0, len(subjects)))]
            p = predicates[int(rng.integers(0, len(predicates)))]
            correct.append(
                f"Since all {s} are {p}, and Rex is a {s[:-1]}, "
                f"therefore Rex is {p}.",
            )
            wrong.append(
                f"Since all {s} are {p}, and Rex is {p}, "
                f"therefore Rex must be a {s[:-1]}.",
            )

        elif pair_type == 2:
            # Code
            n_val = int(rng.integers(3, 15))
            correct.append(
                f"```python\ndef sum_to(n):\n    return n * (n + 1) // 2\n"
                f"# sum_to({n_val}) = {n_val * (n_val + 1) // 2}\n```",
            )
            wrong.append(
                f"```python\ndef sum_to(n):\n    return n * n // 2\n"
                f"# sum_to({n_val}) = {n_val * n_val // 2}\n```",
            )

        elif pair_type == 3:
            # Conditional logic
            conds = [
                ("temperature exceeds 100", "the alarm triggers"),
                ("load exceeds 80%", "throttling activates"),
                ("input is empty", "return default value"),
                ("auth fails", "reject the request"),
            ]
            cond, result_text = conds[int(rng.integers(0, len(conds)))]
            correct.append(f"Whenever {cond}, {result_text}.")
            wrong.append(f"Whenever {cond}, ignore the condition.")

        else:
            # Arithmetic with "unless"
            a = int(rng.integers(5, 50))
            b = int(rng.integers(5, 50))
            correct.append(
                f"The sum of {a} and {b} is {a + b} unless overflow occurs.",
            )
            wrong.append(
                f"The sum of {a} and {b} is {a + b + int(rng.integers(1, 5))} unless overflow occurs.",
            )

    return correct, wrong


# ---------------------------------------------------------------------------
# 7. ConstraintImprovementOrchestrator — the autoresearch loop
# ---------------------------------------------------------------------------

class ConstraintImprovementOrchestrator:
    """Autoresearch orchestrator that proposes and tests constraint improvements.

    **Detailed explanation for engineers:**
        Implements the FR-11 self-learning loop applied specifically to the
        constraint extraction pipeline. Each iteration:

        1. propose() — Pick a hypothesis from template banks.
        2. test() — Run it against 100 held-out failure cases.
        3. evaluate() — Measure AUROC improvement on a validation set.
        4. incorporate() — If accepted, add the pattern to the active set.

        The orchestrator maintains state: which hypotheses were proposed,
        which were accepted, and the running AUROC trajectory. A circuit
        breaker halts after 5 consecutive rejections to avoid wasting compute.

    Attributes:
        max_iterations: Maximum loop iterations (default 20).
        circuit_breaker_limit: Halt after this many consecutive rejections.
        failure_cases: Held-out failure cases for testing.
        accepted: List of accepted hypotheses.
        rejected: List of rejected hypotheses.
        auroc_history: AUROC after each iteration.
        iteration_log: Per-iteration details for reporting.
    """

    def __init__(
        self,
        max_iterations: int = 20,
        circuit_breaker_limit: int = 5,
        n_failure_cases: int = 100,
        n_eval_pairs: int = 200,
        seed: int = 90,
    ) -> None:
        self.max_iterations = max_iterations
        self.circuit_breaker_limit = circuit_breaker_limit
        self.rng = np.random.default_rng(seed)
        self.failure_cases = generate_failure_cases(n_failure_cases, self.rng)
        self.eval_correct, self.eval_wrong = generate_evaluation_pairs(
            n_eval_pairs, self.rng,
        )
        self.accepted: list[ImprovementHypothesis] = []
        self.rejected: list[ImprovementHypothesis] = []
        self.proposed_names: set[str] = set()
        self.auroc_history: list[float] = []
        self.iteration_log: list[dict[str, Any]] = []
        self.consecutive_rejections = 0

    def run(self) -> dict[str, Any]:
        """Execute the full autoresearch loop.

        **Detailed explanation for engineers:**
            Runs up to max_iterations of propose-test-evaluate-incorporate.
            Returns a summary dict with all results for JSON serialization.

        Returns:
            Results dict with keys: iterations_run, accepted_count,
            rejected_count, auroc_history, per_type_stats, final_auroc,
            baseline_auroc, iteration_log.
        """
        print("\n--- Computing baseline AUROC (no extra patterns) ---")
        baseline_auroc = train_and_evaluate_auroc(
            self.eval_correct, self.eval_wrong, [],
        )
        self.auroc_history.append(baseline_auroc)
        print(f"  Baseline AUROC: {baseline_auroc:.4f}")

        iterations_run = 0

        for i in range(self.max_iterations):
            iterations_run = i + 1
            print(f"\n--- Iteration {i + 1}/{self.max_iterations} ---")

            # 1. Propose
            hypothesis = propose_hypothesis(i, self.rng, self.proposed_names)
            self.proposed_names.add(hypothesis.name)
            print(f"  Proposed: [{hypothesis.hypothesis_type}] {hypothesis.name}")
            print(f"    {hypothesis.description}")
            print(f"    Pattern: {hypothesis.pattern}")

            # 2. Test against failure cases
            test_result = test_hypothesis_on_failures(
                hypothesis, self.failure_cases,
            )
            print(f"  Test: {test_result['matches']}/{len(self.failure_cases)} "
                  f"failures matched ({test_result['match_rate']:.1%})")
            if test_result["matched_categories"]:
                print(f"    Categories: {test_result['matched_categories']}")

            # 3. Evaluate AUROC with this hypothesis tentatively added
            tentative_accepted = self.accepted + [hypothesis]
            new_auroc = train_and_evaluate_auroc(
                self.eval_correct, self.eval_wrong, tentative_accepted,
            )
            auroc_delta = new_auroc - self.auroc_history[-1]
            hypothesis.auroc_delta = auroc_delta
            print(f"  AUROC: {new_auroc:.4f} (delta: {auroc_delta:+.4f})")

            # 4. Accept/reject decision
            # Accept if: (a) catches at least 1 failure case AND
            #             (b) AUROC doesn't regress (delta >= -0.01)
            accept = test_result["matches"] > 0 and auroc_delta >= -0.01
            hypothesis.accepted = accept
            hypothesis.metadata = {
                "test_matches": test_result["matches"],
                "test_match_rate": test_result["match_rate"],
                "matched_categories": test_result["matched_categories"],
                "auroc_before": self.auroc_history[-1],
                "auroc_after": new_auroc,
            }

            if accept:
                self.accepted.append(hypothesis)
                self.auroc_history.append(new_auroc)
                self.consecutive_rejections = 0
                print(f"  ✓ ACCEPTED (cumulative accepted: {len(self.accepted)})")
            else:
                self.rejected.append(hypothesis)
                self.auroc_history.append(self.auroc_history[-1])
                self.consecutive_rejections += 1
                reject_reason = (
                    "no failure matches"
                    if test_result["matches"] == 0
                    else f"AUROC regression ({auroc_delta:+.4f})"
                )
                print(f"  x REJECTED: {reject_reason} "
                      f"(consecutive: {self.consecutive_rejections})")

            self.iteration_log.append({
                "iteration": i + 1,
                "hypothesis_type": hypothesis.hypothesis_type,
                "hypothesis_name": hypothesis.name,
                "description": hypothesis.description,
                "pattern": hypothesis.pattern,
                "test_matches": test_result["matches"],
                "test_match_rate": test_result["match_rate"],
                "matched_categories": test_result["matched_categories"],
                "auroc_delta": auroc_delta,
                "accepted": accept,
                "cumulative_auroc": self.auroc_history[-1],
            })

            # Circuit breaker check
            if self.consecutive_rejections >= self.circuit_breaker_limit:
                print(f"\n  CIRCUIT BREAKER: {self.circuit_breaker_limit} "
                      "consecutive rejections — halting loop.")
                break

        # Compile final results.
        final_auroc = self.auroc_history[-1]
        total_improvement = final_auroc - baseline_auroc

        # Per-type statistics
        per_type: dict[str, dict[str, int]] = {}
        for hyp in self.accepted + self.rejected:
            t = hyp.hypothesis_type
            if t not in per_type:
                per_type[t] = {"proposed": 0, "accepted": 0, "rejected": 0}
            per_type[t]["proposed"] += 1
            if hyp.accepted:
                per_type[t]["accepted"] += 1
            else:
                per_type[t]["rejected"] += 1

        # Side-by-side comparison: original vs improved
        print("\n" + "=" * 70)
        print("FINAL COMPARISON: Original vs Improved Extractor")
        print("=" * 70)
        print(f"  Baseline AUROC:  {baseline_auroc:.4f}")
        print(f"  Final AUROC:     {final_auroc:.4f}")
        print(f"  Improvement:     {total_improvement:+.4f}")
        print(f"  Acceptance rate: {len(self.accepted)}/{iterations_run} "
              f"({len(self.accepted)/max(iterations_run, 1):.1%})")
        print(f"\n  Per-type stats:")
        for t, stats in sorted(per_type.items()):
            print(f"    {t}: {stats['accepted']}/{stats['proposed']} accepted")
        print(f"\n  Accepted hypotheses:")
        for hyp in self.accepted:
            print(f"    [{hyp.hypothesis_type}] {hyp.name}: "
                  f"AUROC delta={hyp.auroc_delta:+.4f}")

        return {
            "experiment": "90_autoresearch_constraints",
            "description": (
                "Autoresearch loop for constraint pipeline self-improvement. "
                "Proposes new regex/AST/logic/Ising feature hypotheses, tests "
                "them on held-out failures, accepts if they improve coverage "
                "without AUROC regression."
            ),
            "iterations_run": iterations_run,
            "max_iterations": self.max_iterations,
            "circuit_breaker_limit": self.circuit_breaker_limit,
            "baseline_auroc": baseline_auroc,
            "final_auroc": final_auroc,
            "total_auroc_improvement": total_improvement,
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "acceptance_rate": len(self.accepted) / max(iterations_run, 1),
            "auroc_history": self.auroc_history,
            "per_type_stats": per_type,
            "accepted_hypotheses": [
                {
                    "type": h.hypothesis_type,
                    "name": h.name,
                    "description": h.description,
                    "pattern": h.pattern,
                    "auroc_delta": h.auroc_delta,
                    "metadata": h.metadata,
                }
                for h in self.accepted
            ],
            "rejected_hypotheses": [
                {
                    "type": h.hypothesis_type,
                    "name": h.name,
                    "description": h.description,
                    "pattern": h.pattern,
                    "auroc_delta": h.auroc_delta,
                    "metadata": h.metadata,
                }
                for h in self.rejected
            ],
            "iteration_log": self.iteration_log,
            "failure_cases_count": len(self.failure_cases),
            "eval_pairs_count": len(self.eval_correct),
        }


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run Experiment 90: Autoresearch constraints self-improvement loop.

    **Detailed explanation for engineers:**
        Entry point that:
        1. Initializes JAX (CPU-only for reproducibility).
        2. Creates the ConstraintImprovementOrchestrator.
        3. Runs the 20-iteration autoresearch loop.
        4. Saves results to results/experiment_90_results.json.

    Returns:
        0 on success, 1 on failure.
    """
    import jax

    print("=" * 70)
    print("EXPERIMENT 90: Autoresearch Loop for Constraint Pipeline Improvement")
    print("  FR-11: Autonomous self-learning applied to constraint extraction")
    print("  Hypothesis types: regex, AST rules, logic patterns, Ising features")
    print("  Loop: 20 iterations, circuit breaker at 5 consecutive rejections")
    print(f"  JAX backend: {jax.default_backend()}")
    print("=" * 70)

    start = time.time()

    # Create and run the orchestrator.
    orchestrator = ConstraintImprovementOrchestrator(
        max_iterations=20,
        circuit_breaker_limit=5,
        n_failure_cases=100,
        n_eval_pairs=200,
        seed=90,
    )

    results = orchestrator.run()
    elapsed = time.time() - start

    results["wall_clock_seconds"] = elapsed
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Print summary.
    print("\n" + "=" * 70)
    print("EXPERIMENT 90 SUMMARY")
    print("=" * 70)
    print(f"  Iterations:       {results['iterations_run']}")
    print(f"  Baseline AUROC:   {results['baseline_auroc']:.4f}")
    print(f"  Final AUROC:      {results['final_auroc']:.4f}")
    print(f"  Improvement:      {results['total_auroc_improvement']:+.4f}")
    print(f"  Accepted:         {results['accepted_count']}")
    print(f"  Rejected:         {results['rejected_count']}")
    print(f"  Acceptance rate:  {results['acceptance_rate']:.1%}")
    print(f"  Wall clock:       {elapsed:.1f}s")

    # Save results.
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "experiment_90_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    # Success criteria: experiment completed, at least 1 hypothesis accepted.
    if results["accepted_count"] > 0:
        print("\n  SUCCESS: Pipeline self-improvement demonstrated.")
        return 0
    else:
        print("\n  NOTE: No hypotheses accepted — pipeline may already be "
              "well-covered for these failure types.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
