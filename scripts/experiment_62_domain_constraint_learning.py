#!/usr/bin/env python3
"""Experiment 62: Learn domain-specific constraint structure from 10K triples.

**The big idea:**
    Experiment 51 showed discriminative CD can learn to separate correct from
    wrong arithmetic answers. Experiment 60 scaled CD training to 200 variables.
    This experiment combines both: train an Ising model (200 vars, sparse) on
    10,000 (question, correct_answer, wrong_answer) triples across three domains
    — arithmetic, logic, and code — to learn which binary features distinguish
    correct from wrong answers.

**Why this matters:**
    If an Ising model can learn domain-specific constraint structure from
    examples alone, it becomes a verifier: given a candidate answer, compute
    its energy. Low energy = likely correct, high energy = likely wrong. This
    is the foundation for EBM-based answer verification without an LLM.

**The three domains:**
    1. Arithmetic (3,333 triples): programmatic generation of addition,
       multiplication, and modular arithmetic with known answers.
    2. Logic (3,333 triples): syllogisms with known validity — modus ponens,
       modus tollens, disjunctive syllogism, etc.
    3. Code (3,334 triples): simple functions with known test outcomes —
       correct implementations vs common bugs.

    All triples are generated from templates with random parameters (no LLM),
    so results are fully reproducible.

**Binary feature encoding (200+ features per answer):**
    Rather than raw text, each answer is encoded as a binary feature vector
    capturing structural properties: numeric features (digit count, number
    presence), structural features (word count, punctuation), domain-specific
    features (has_equation, has_if_then, has_function_def), and consistency
    features (numbers match between question and answer).

**Training approach:**
    Discriminative CD from Exp 51, scaled to 200 variables with L1
    regularization from Exp 60. Train four models:
    - Per-domain: arithmetic-only, logic-only, code-only
    - Combined: all 10K triples together
    Evaluate via AUROC on held-out 20% test split.

**Baselines:**
    - Hand-coded: simple rule-based scoring (does the answer contain the
      expected numbers? is the logic valid?)
    - Logprob: feature-count heuristic (more features = more likely correct)

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_62_domain_constraint_learning.py
"""

from __future__ import annotations

import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Domain 1: Arithmetic triples
# ---------------------------------------------------------------------------

def generate_arithmetic_triples(
    n: int, rng: np.random.Generator,
) -> list[tuple[str, str, str]]:
    """Generate (question, correct_answer, wrong_answer) for arithmetic.

    **Detailed explanation for engineers:**
        We generate three sub-types of arithmetic problems:
        1. Addition: "What is A + B?" — wrong answers off by 1, 10, or carry error.
        2. Multiplication: "What is A * B?" — wrong answers from adjacent products.
        3. Modular arithmetic: "What is A mod B?" — wrong answers from nearby mods.

        Each sub-type gets n/3 triples. Wrong answers are designed to mimic
        common LLM mistakes: off-by-one, digit transposition, carry errors.

    Args:
        n: Number of triples to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        List of (question, correct_answer, wrong_answer) string triples.
    """
    triples = []
    per_type = n // 3
    remainder = n - 3 * per_type

    # --- Addition ---
    for _ in range(per_type + (1 if remainder > 0 else 0)):
        a = int(rng.integers(1, 500))
        b = int(rng.integers(1, 500))
        correct = a + b
        # Generate a plausible wrong answer.
        error_type = rng.choice(["off_by_one", "off_by_ten", "carry", "close"])
        if error_type == "off_by_one":
            wrong = correct + int(rng.choice([-1, 1]))
        elif error_type == "off_by_ten":
            wrong = correct + int(rng.choice([-10, 10]))
        elif error_type == "carry":
            bit_pos = int(rng.integers(1, max(2, correct.bit_length())))
            wrong = correct ^ (1 << bit_pos)
        else:
            wrong = correct + int(rng.integers(2, 8)) * int(rng.choice([-1, 1]))
        if wrong == correct:
            wrong = correct + 1
        if wrong < 0:
            wrong = correct + abs(wrong - correct) + 1
        question = f"What is {a} + {b}?"
        triples.append((question, f"The answer is {correct}.", f"The answer is {wrong}."))
    remainder = max(0, remainder - 1)

    # --- Multiplication ---
    for _ in range(per_type + (1 if remainder > 0 else 0)):
        a = int(rng.integers(2, 50))
        b = int(rng.integers(2, 50))
        correct = a * b
        # Wrong: adjacent product, transposed digits, or off by a factor.
        error_type = rng.choice(["adjacent", "off_factor", "close"])
        if error_type == "adjacent":
            wrong = (a + int(rng.choice([-1, 1]))) * b
        elif error_type == "off_factor":
            wrong = correct + a * int(rng.choice([-1, 1]))
        else:
            wrong = correct + int(rng.integers(1, 10)) * int(rng.choice([-1, 1]))
        if wrong == correct:
            wrong = correct + 1
        if wrong < 0:
            wrong = abs(wrong)
        question = f"What is {a} * {b}?"
        triples.append((question, f"The answer is {correct}.", f"The answer is {wrong}."))
    remainder = max(0, remainder - 1)

    # --- Modular arithmetic ---
    for _ in range(per_type):
        a = int(rng.integers(10, 1000))
        b = int(rng.integers(2, 50))
        correct = a % b
        # Wrong: off by one or confuse with division.
        wrong = correct + int(rng.choice([-1, 1, 2, -2]))
        if wrong == correct:
            wrong = correct + 1
        if wrong < 0:
            wrong = abs(wrong)
        question = f"What is {a} mod {b}?"
        triples.append((question, f"The answer is {correct}.", f"The answer is {wrong}."))

    return triples[:n]


# ---------------------------------------------------------------------------
# Domain 2: Logic triples
# ---------------------------------------------------------------------------

# Template subjects and predicates for syllogism generation.
_SUBJECTS = [
    "cats", "dogs", "birds", "fish", "trees", "rocks", "stars", "clouds",
    "rivers", "mountains", "students", "teachers", "engineers", "doctors",
    "planets", "atoms", "cells", "waves", "crystals", "robots",
]

_PREDICATES = [
    "mortal", "alive", "visible", "heavy", "bright", "fast", "old",
    "complex", "natural", "rare", "symmetric", "stable", "dense", "warm",
    "soluble", "magnetic", "elastic", "transparent", "finite", "periodic",
]


def generate_logic_triples(
    n: int, rng: np.random.Generator,
) -> list[tuple[str, str, str]]:
    """Generate (question, correct_answer, wrong_answer) for logic syllogisms.

    **Detailed explanation for engineers:**
        We generate four types of syllogisms, each with known validity:
        1. Modus ponens: If P then Q. P. Therefore Q. (Valid)
        2. Modus tollens: If P then Q. Not Q. Therefore not P. (Valid)
        3. Affirming the consequent: If P then Q. Q. Therefore P. (INVALID)
        4. Denying the antecedent: If P then Q. Not P. Therefore not Q. (INVALID)

        For valid syllogisms, the correct answer states the valid conclusion
        and the wrong answer states an invalid one (or vice versa for invalid
        syllogisms). This tests whether the Ising model can learn the
        structural difference between valid and invalid reasoning.

    Args:
        n: Number of triples to generate.
        rng: NumPy random generator.

    Returns:
        List of (question, correct_answer, wrong_answer) string triples.
    """
    triples = []
    per_type = n // 4
    remainder = n - 4 * per_type

    def pick_pair(rng):
        """Pick a random subject-predicate pair."""
        s = _SUBJECTS[int(rng.integers(0, len(_SUBJECTS)))]
        p = _PREDICATES[int(rng.integers(0, len(_PREDICATES)))]
        s2 = _SUBJECTS[int(rng.integers(0, len(_SUBJECTS)))]
        # Ensure different subjects.
        while s2 == s:
            s2 = _SUBJECTS[int(rng.integers(0, len(_SUBJECTS)))]
        p2 = _PREDICATES[int(rng.integers(0, len(_PREDICATES)))]
        while p2 == p:
            p2 = _PREDICATES[int(rng.integers(0, len(_PREDICATES)))]
        return s, p, s2, p2

    # --- Modus ponens (valid) ---
    for _ in range(per_type + min(1, remainder)):
        s, p, s2, p2 = pick_pair(rng)
        question = f"If all {s} are {p}, and all {s2} are {s}, what follows?"
        correct = f"All {s2} are {p}. This follows by modus ponens."
        wrong = f"Some {s2} are not {p}. The premises do not guarantee this."
        triples.append((question, correct, wrong))
    remainder = max(0, remainder - 1)

    # --- Modus tollens (valid) ---
    for _ in range(per_type + min(1, remainder)):
        s, p, s2, p2 = pick_pair(rng)
        question = f"If all {s} are {p}, and {s2} are not {p}, what follows?"
        correct = f"{s2} are not {s}. This follows by modus tollens."
        wrong = f"{s2} might still be {s}. We cannot conclude anything."
        triples.append((question, correct, wrong))
    remainder = max(0, remainder - 1)

    # --- Disjunctive syllogism (valid) ---
    for _ in range(per_type + min(1, remainder)):
        s, p, s2, p2 = pick_pair(rng)
        question = f"Either {s} are {p} or {s} are {p2}. {s} are not {p}. What follows?"
        correct = f"{s} are {p2}. This follows by disjunctive syllogism."
        wrong = f"{s} are neither {p} nor {p2}. Both options are eliminated."
        triples.append((question, correct, wrong))
    remainder = max(0, remainder - 1)

    # --- Affirming the consequent (INVALID — test if model learns this) ---
    for _ in range(per_type):
        s, p, s2, p2 = pick_pair(rng)
        question = f"If all {s} are {p}, and {s2} are {p}, are {s2} necessarily {s}?"
        correct = f"No. {s2} being {p} does not mean {s2} are {s}. This is affirming the consequent."
        wrong = f"Yes. Since all {s} are {p} and {s2} are {p}, {s2} must be {s}."
        triples.append((question, correct, wrong))

    return triples[:n]


# ---------------------------------------------------------------------------
# Domain 3: Code triples
# ---------------------------------------------------------------------------

def generate_code_triples(
    n: int, rng: np.random.Generator,
) -> list[tuple[str, str, str]]:
    """Generate (question, correct_answer, wrong_answer) for code snippets.

    **Detailed explanation for engineers:**
        We generate code problems from templates where the correct answer is
        a working implementation and the wrong answer has a common bug:
        1. Off-by-one in range (fence-post error)
        2. Wrong comparison operator (< vs <=, > vs >=)
        3. Wrong return value (returning input instead of result)
        4. Missing edge case (empty list, zero input)
        5. Wrong variable in return statement

        Each template uses random function names and parameters to create
        unique instances. The question describes what the function should do,
        the correct answer is a working implementation, and the wrong answer
        has exactly one bug.

    Args:
        n: Number of triples to generate.
        rng: NumPy random generator.

    Returns:
        List of (question, correct_answer, wrong_answer) string triples.
    """
    triples = []
    templates = _code_templates()

    for i in range(n):
        template_idx = i % len(templates)
        template = templates[template_idx]
        # Randomize the numeric parameter in each template.
        param = int(rng.integers(2, 20))
        triple = template(param, rng)
        triples.append(triple)

    return triples[:n]


def _code_templates() -> list:
    """Return a list of code-triple generator functions.

    **Detailed explanation for engineers:**
        Each template is a callable that takes (param, rng) and returns
        (question, correct_answer, wrong_answer). The param injects
        randomness into the numeric constants used in each problem.
        Templates cover common beginner mistakes: off-by-one errors,
        wrong operators, missing edge cases, and incorrect return values.
    """

    def sum_range(param, rng):
        """Sum of integers 1..n: correct uses n+1 upper bound, wrong uses n."""
        n = param
        question = f"Write a function that returns the sum of integers from 1 to {n}."
        correct = (
            f"def sum_range(n):\n"
            f"    total = 0\n"
            f"    for i in range(1, n + 1):\n"
            f"        total += i\n"
            f"    return total\n"
            f"# sum_range({n}) returns {n * (n + 1) // 2}"
        )
        wrong = (
            f"def sum_range(n):\n"
            f"    total = 0\n"
            f"    for i in range(1, n):\n"
            f"        total += i\n"
            f"    return total\n"
            f"# sum_range({n}) returns {(n - 1) * n // 2}"
        )
        return question, correct, wrong

    def find_max(param, rng):
        """Find max in list: correct initializes to first element, wrong to 0."""
        question = "Write a function that returns the maximum value in a list of integers."
        correct = (
            "def find_max(lst):\n"
            "    if not lst:\n"
            "        return None\n"
            "    result = lst[0]\n"
            "    for x in lst[1:]:\n"
            "        if x > result:\n"
            "            result = x\n"
            "    return result"
        )
        wrong = (
            "def find_max(lst):\n"
            "    result = 0\n"
            "    for x in lst:\n"
            "        if x > result:\n"
            "            result = x\n"
            "    return result"
        )
        return question, correct, wrong

    def is_even(param, rng):
        """Check even: correct uses == 0, wrong uses == 1."""
        question = f"Write a function that returns True if {param} is even."
        correct = (
            f"def is_even(n):\n"
            f"    return n % 2 == 0\n"
            f"# is_even({param}) returns {param % 2 == 0}"
        )
        wrong = (
            f"def is_even(n):\n"
            f"    return n % 2 == 1\n"
            f"# is_even({param}) returns {param % 2 == 1}"
        )
        return question, correct, wrong

    def factorial(param, rng):
        """Factorial: correct handles base case, wrong recurses past 0."""
        n = min(param, 12)
        import math
        question = f"Write a function that computes the factorial of {n}."
        correct = (
            f"def factorial(n):\n"
            f"    if n <= 1:\n"
            f"        return 1\n"
            f"    return n * factorial(n - 1)\n"
            f"# factorial({n}) returns {math.factorial(n)}"
        )
        wrong = (
            f"def factorial(n):\n"
            f"    if n == 1:\n"
            f"        return 1\n"
            f"    return n * factorial(n - 1)\n"
            f"# factorial(0) causes infinite recursion"
        )
        return question, correct, wrong

    def reverse_string(param, rng):
        """Reverse string: correct uses [::-1], wrong uses [::1]."""
        question = "Write a function that reverses a string."
        correct = (
            "def reverse_string(s):\n"
            "    return s[::-1]\n"
            "# reverse_string('hello') returns 'olleh'"
        )
        wrong = (
            "def reverse_string(s):\n"
            "    return s[::1]\n"
            "# reverse_string('hello') returns 'hello'"
        )
        return question, correct, wrong

    def count_vowels(param, rng):
        """Count vowels: correct includes all 5, wrong misses 'u'."""
        question = "Write a function that counts the vowels in a string."
        correct = (
            "def count_vowels(s):\n"
            "    return sum(1 for c in s.lower() if c in 'aeiou')\n"
            "# count_vowels('education') returns 5"
        )
        wrong = (
            "def count_vowels(s):\n"
            "    return sum(1 for c in s.lower() if c in 'aeio')\n"
            "# count_vowels('education') returns 4"
        )
        return question, correct, wrong

    def fibonacci(param, rng):
        """Fibonacci: correct starts with 0,1; wrong starts with 1,1."""
        n = min(param, 15)
        question = f"Write a function that returns the {n}th Fibonacci number (0-indexed)."
        # Correct: F(0)=0, F(1)=1, F(2)=1, ...
        def fib(k):
            a, b = 0, 1
            for _ in range(k):
                a, b = b, a + b
            return a
        correct = (
            f"def fibonacci(n):\n"
            f"    a, b = 0, 1\n"
            f"    for _ in range(n):\n"
            f"        a, b = b, a + b\n"
            f"    return a\n"
            f"# fibonacci({n}) returns {fib(n)}"
        )
        wrong = (
            f"def fibonacci(n):\n"
            f"    a, b = 1, 1\n"
            f"    for _ in range(n):\n"
            f"        a, b = b, a + b\n"
            f"    return a\n"
            f"# fibonacci(0) returns 1 instead of 0"
        )
        return question, correct, wrong

    def binary_search(param, rng):
        """Binary search: correct uses mid+1/mid-1, wrong uses mid/mid."""
        question = "Write a binary search that returns the index of target in a sorted list."
        correct = (
            "def binary_search(lst, target):\n"
            "    lo, hi = 0, len(lst) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if lst[mid] == target:\n"
            "            return mid\n"
            "        elif lst[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1"
        )
        wrong = (
            "def binary_search(lst, target):\n"
            "    lo, hi = 0, len(lst) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if lst[mid] == target:\n"
            "            return mid\n"
            "        elif lst[mid] < target:\n"
            "            lo = mid\n"
            "        else:\n"
            "            hi = mid\n"
            "    return -1"
        )
        return question, correct, wrong

    def is_palindrome(param, rng):
        """Palindrome check: correct compares reversed, wrong compares sorted."""
        question = "Write a function that checks if a string is a palindrome."
        correct = (
            "def is_palindrome(s):\n"
            "    s = s.lower().replace(' ', '')\n"
            "    return s == s[::-1]"
        )
        wrong = (
            "def is_palindrome(s):\n"
            "    s = s.lower().replace(' ', '')\n"
            "    return s == ''.join(sorted(s))"
        )
        return question, correct, wrong

    def flatten_list(param, rng):
        """Flatten nested list: correct recurses on sublists, wrong only goes one level."""
        question = "Write a function that flattens a nested list."
        correct = (
            "def flatten(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result"
        )
        wrong = (
            "def flatten(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(item)\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result"
        )
        return question, correct, wrong

    return [
        sum_range, find_max, is_even, factorial, reverse_string,
        count_vowels, fibonacci, binary_search, is_palindrome, flatten_list,
    ]


# ---------------------------------------------------------------------------
# Binary feature encoding (200+ features per answer)
# ---------------------------------------------------------------------------

def encode_answer(question: str, answer: str) -> np.ndarray:
    """Encode a (question, answer) pair as a binary feature vector.

    **Detailed explanation for engineers:**
        The feature vector captures structural properties of the answer text
        that correlate with correctness across domains. Features are grouped:

        1. Numeric features (20): presence of specific digit counts, whether
           numbers in the answer match numbers in the question, digit sum
           properties, etc.
        2. Structural features (40): word count buckets, sentence count,
           punctuation patterns, capitalization, whitespace patterns.
        3. Domain-specific features (80): has_equation markers, logical
           connectives (if/then, therefore, because), code structure
           (def, return, for, while, indentation).
        4. Consistency features (60): whether numbers from the question
           appear in the answer, whether logical terms are paired correctly,
           whether code has matching brackets/parens.

        Total: 200 binary features. Each is 0 or 1.

    Args:
        question: The question text.
        answer: The answer text (either correct or wrong).

    Returns:
        Binary feature vector, shape (200,), dtype float32.
    """
    features = []

    # ===== Numeric features (20) =====
    q_numbers = [int(x) for x in re.findall(r'\d+', question)]
    a_numbers = [int(x) for x in re.findall(r'\d+', answer)]
    a_digits = re.findall(r'\d', answer)

    # f0: answer contains any number
    features.append(1 if a_numbers else 0)
    # f1-f5: answer number count buckets (0, 1, 2, 3, 4+)
    n_nums = len(a_numbers)
    for k in range(5):
        features.append(1 if n_nums == k else 0)
    # f6: digit count > 3
    features.append(1 if len(a_digits) > 3 else 0)
    # f7: digit sum is even
    digit_sum = sum(int(d) for d in a_digits) if a_digits else 0
    features.append(1 if digit_sum % 2 == 0 else 0)
    # f8: digit sum is odd
    features.append(1 if digit_sum % 2 == 1 else 0)
    # f9-f13: how many question numbers appear in answer (0..4+)
    q_in_a = sum(1 for qn in q_numbers if qn in a_numbers)
    for k in range(5):
        features.append(1 if q_in_a == k else 0)
    # f14: answer has larger number than any question number
    max_q = max(q_numbers) if q_numbers else 0
    max_a = max(a_numbers) if a_numbers else 0
    features.append(1 if max_a > max_q else 0)
    # f15: answer number matches expected sum (for arithmetic)
    if len(q_numbers) >= 2 and "+" in question:
        features.append(1 if (q_numbers[0] + q_numbers[1]) in a_numbers else 0)
    else:
        features.append(0)
    # f16: answer number matches expected product
    if len(q_numbers) >= 2 and "*" in question:
        features.append(1 if (q_numbers[0] * q_numbers[1]) in a_numbers else 0)
    else:
        features.append(0)
    # f17: answer number matches expected mod
    if len(q_numbers) >= 2 and "mod" in question:
        features.append(1 if (q_numbers[0] % q_numbers[1]) in a_numbers else 0)
    else:
        features.append(0)
    # f18-f19: number magnitude buckets
    features.append(1 if max_a < 10 else 0)
    features.append(1 if max_a >= 100 else 0)

    assert len(features) == 20, f"Numeric features: {len(features)}"

    # ===== Structural features (40) =====
    words = answer.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
    n_words = len(words)
    n_sentences = len(sentences)

    # f20-f29: word count buckets (0-2, 3-5, 6-10, 11-15, 16-20, 21-30, 31-50, 51-100, 100+, exact 1)
    wc_bounds = [2, 5, 10, 15, 20, 30, 50, 100]
    for ub in wc_bounds:
        features.append(1 if n_words <= ub else 0)
    features.append(1 if n_words > 100 else 0)
    features.append(1 if n_words == 1 else 0)

    # f30-f34: sentence count buckets (1, 2, 3, 4, 5+)
    for k in range(1, 5):
        features.append(1 if n_sentences == k else 0)
    features.append(1 if n_sentences >= 5 else 0)

    # f35-f39: punctuation features
    features.append(1 if "." in answer else 0)
    features.append(1 if "," in answer else 0)
    features.append(1 if "!" in answer else 0)
    features.append(1 if "?" in answer else 0)
    features.append(1 if ":" in answer else 0)

    # f40-f44: capitalization and formatting
    features.append(1 if answer[0].isupper() else 0)
    features.append(1 if any(c.isupper() for c in answer[1:]) else 0)
    features.append(1 if answer.strip().endswith(".") else 0)
    features.append(1 if "\n" in answer else 0)
    features.append(1 if "  " in answer else 0)

    # f45-f54: character class features
    features.append(1 if any(c == "(" for c in answer) else 0)
    features.append(1 if any(c == ")" for c in answer) else 0)
    features.append(1 if any(c == "[" for c in answer) else 0)
    features.append(1 if any(c == "]" for c in answer) else 0)
    features.append(1 if any(c == "{" for c in answer) else 0)
    features.append(1 if any(c == "}" for c in answer) else 0)
    features.append(1 if "'" in answer else 0)
    features.append(1 if '"' in answer else 0)
    features.append(1 if "#" in answer else 0)
    features.append(1 if "=" in answer else 0)

    # f55-f59: length buckets (character count)
    n_chars = len(answer)
    features.append(1 if n_chars < 20 else 0)
    features.append(1 if 20 <= n_chars < 50 else 0)
    features.append(1 if 50 <= n_chars < 100 else 0)
    features.append(1 if 100 <= n_chars < 200 else 0)
    features.append(1 if n_chars >= 200 else 0)

    assert len(features) == 60, f"Structural features: {len(features)}"

    # ===== Domain-specific features (80) =====
    lower_answer = answer.lower()
    lower_question = question.lower()

    # f60-f69: arithmetic / equation markers
    features.append(1 if "+" in answer else 0)
    features.append(1 if "-" in answer and not answer.startswith("-") else 0)
    features.append(1 if "*" in answer else 0)
    features.append(1 if "/" in answer and "//" not in answer else 0)
    features.append(1 if "//" in answer else 0)
    features.append(1 if "%" in answer else 0)
    features.append(1 if "=" in answer and "==" not in answer else 0)
    features.append(1 if "==" in answer else 0)
    features.append(1 if "the answer is" in lower_answer else 0)
    features.append(1 if "returns" in lower_answer else 0)

    # f70-f79: logic connectives
    features.append(1 if "if " in lower_answer else 0)
    features.append(1 if "then " in lower_answer else 0)
    features.append(1 if "therefore" in lower_answer else 0)
    features.append(1 if "because" in lower_answer else 0)
    features.append(1 if "follows" in lower_answer else 0)
    features.append(1 if " not " in lower_answer else 0)
    features.append(1 if "all " in lower_answer else 0)
    features.append(1 if "some " in lower_answer else 0)
    features.append(1 if "modus" in lower_answer else 0)
    features.append(1 if "syllogism" in lower_answer else 0)

    # f80-f89: code structure markers
    features.append(1 if "def " in answer else 0)
    features.append(1 if "return " in answer else 0)
    features.append(1 if "for " in answer else 0)
    features.append(1 if "while " in answer else 0)
    features.append(1 if "if " in answer else 0)
    features.append(1 if "else" in answer else 0)
    features.append(1 if "import " in answer else 0)
    features.append(1 if "class " in answer else 0)
    features.append(1 if "range(" in answer else 0)
    features.append(1 if "len(" in answer else 0)

    # f90-f99: code quality markers
    features.append(1 if "    " in answer else 0)  # Indentation
    features.append(1 if answer.count("\n") > 2 else 0)  # Multi-line
    features.append(1 if "True" in answer or "False" in answer else 0)
    features.append(1 if "None" in answer else 0)
    features.append(1 if "isinstance" in answer else 0)
    features.append(1 if ".append(" in answer else 0)
    features.append(1 if ".extend(" in answer else 0)
    features.append(1 if "[::" in answer else 0)
    features.append(1 if "recursion" in lower_answer or "recursive" in lower_answer else 0)
    features.append(1 if "edge case" in lower_answer or "empty" in lower_answer else 0)

    # f100-f109: question-type detection
    features.append(1 if "+" in question else 0)
    features.append(1 if "*" in question else 0)
    features.append(1 if "mod " in lower_question else 0)
    features.append(1 if "if all" in lower_question or "if " in lower_question else 0)
    features.append(1 if "write a function" in lower_question else 0)
    features.append(1 if "what is" in lower_question else 0)
    features.append(1 if "what follows" in lower_question else 0)
    features.append(1 if "necessarily" in lower_question else 0)
    features.append(1 if "returns" in lower_question else 0)
    features.append(1 if "check" in lower_question or "sorted" in lower_question else 0)

    # f110-f119: answer sentiment / confidence markers
    features.append(1 if "yes" in lower_answer.split() else 0)
    features.append(1 if "no" in lower_answer.split() else 0)
    features.append(1 if "correct" in lower_answer else 0)
    features.append(1 if "incorrect" in lower_answer or "wrong" in lower_answer else 0)
    features.append(1 if "cannot" in lower_answer or "can't" in lower_answer else 0)
    features.append(1 if "must" in lower_answer else 0)
    features.append(1 if "might" in lower_answer or "may " in lower_answer else 0)
    features.append(1 if "guaranteed" in lower_answer else 0)
    features.append(1 if "invalid" in lower_answer else 0)
    features.append(1 if "valid" in lower_answer and "invalid" not in lower_answer else 0)

    # f120-f139: bigram features (word pairs)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)] if len(words) > 1 else []
    bigram_lower = [b.lower() for b in bigrams]
    target_bigrams = [
        "the answer", "answer is", "is not", "not the", "does not",
        "this follows", "follows by", "by modus", "n +", "+ 1",
        "return total", "return result", "for i", "in range", "if not",
        "not lst", "lst 0", "== 0", "== 1", "def ",
    ]
    for tb in target_bigrams:
        features.append(1 if any(tb in b for b in bigram_lower) else 0)

    assert len(features) == 140, f"Domain features: {len(features)}"

    # ===== Consistency features (60) =====

    # f160-f169: bracket/paren matching
    open_parens = answer.count("(")
    close_parens = answer.count(")")
    features.append(1 if open_parens == close_parens else 0)
    open_brackets = answer.count("[")
    close_brackets = answer.count("]")
    features.append(1 if open_brackets == close_brackets else 0)
    open_braces = answer.count("{")
    close_braces = answer.count("}")
    features.append(1 if open_braces == close_braces else 0)
    features.append(1 if open_parens > 0 and open_parens == close_parens else 0)
    features.append(1 if answer.count("'") % 2 == 0 else 0)
    features.append(1 if answer.count('"') % 2 == 0 else 0)
    # Colon-newline pattern (code blocks)
    features.append(1 if ":\n" in answer else 0)
    features.append(1 if ":\n    " in answer else 0)
    # Return statement present when def is present
    features.append(1 if ("def " in answer and "return" in answer) or "def " not in answer else 0)
    # Consistent indentation (all indented lines use 4 spaces)
    lines = answer.split("\n")
    indented = [l for l in lines if l and l[0] == " "]
    if indented:
        consistent = all(
            (len(l) - len(l.lstrip())) in (4, 8) for l in indented
        )
        features.append(1 if consistent else 0)
    else:
        features.append(0)

    # f170-f179: numeric consistency
    # All question numbers present in answer
    if q_numbers:
        frac_present = sum(1 for qn in q_numbers if str(qn) in answer) / len(q_numbers)
    else:
        frac_present = 0.0
    features.append(1 if frac_present >= 1.0 else 0)
    features.append(1 if frac_present >= 0.5 else 0)
    features.append(1 if frac_present == 0.0 else 0)
    # Answer number is in reasonable range relative to question numbers
    if q_numbers and a_numbers:
        ratio = max_a / max(max_q, 1)
        features.append(1 if 0.01 < ratio < 1000 else 0)
    else:
        features.append(1)
    # Number appears exactly once (no repetition errors)
    if a_numbers:
        unique_ratio = len(set(a_numbers)) / len(a_numbers)
        features.append(1 if unique_ratio > 0.5 else 0)
    else:
        features.append(1)
    # Digit pattern: answer number has same digit count as expected
    features.append(1 if len(a_digits) <= 6 else 0)
    features.append(1 if len(a_digits) > 0 else 0)
    features.append(1 if digit_sum > 0 and digit_sum < 50 else 0)
    # Answer length is reasonable for question type
    features.append(1 if "write a function" in lower_question and n_chars > 30 else 0)
    features.append(1 if "what is" in lower_question and n_chars < 50 else 0)

    # f180-f189: logic consistency
    features.append(1 if "not" in lower_answer and "not" in lower_question else 0)
    features.append(1 if "all" in lower_answer and "all" in lower_question else 0)
    # Subject from question appears in answer
    q_words_set = set(lower_question.split())
    a_words_set = set(lower_answer.split())
    shared_content_words = q_words_set & a_words_set - {"a", "an", "the", "is", "are", "what", "if", "and", "or", "in", "of", "to"}
    features.append(1 if len(shared_content_words) > 2 else 0)
    features.append(1 if len(shared_content_words) > 5 else 0)
    features.append(1 if "affirming" in lower_answer else 0)
    features.append(1 if "consequent" in lower_answer else 0)
    features.append(1 if "tollens" in lower_answer else 0)
    features.append(1 if "ponens" in lower_answer else 0)
    features.append(1 if "disjunctive" in lower_answer else 0)
    features.append(1 if "eliminate" in lower_answer or "eliminated" in lower_answer else 0)

    # f180-f189: cross-domain consistency
    # Arithmetic answer mentions "the answer is" (expected format)
    features.append(1 if "the answer is" in lower_answer and ("+" in question or "*" in question or "mod" in question) else 0)
    # Logic answer mentions reasoning type
    features.append(1 if ("follows" in lower_answer or "cannot" in lower_answer) and ("if " in lower_question) else 0)
    # Code answer has proper structure
    features.append(1 if "def " in answer and "return" in answer and ":\n" in answer else 0)
    # Answer type matches question type
    features.append(1 if "write" in lower_question and "def " in answer else 0)
    features.append(1 if "what is" in lower_question and any(c.isdigit() for c in answer) else 0)
    features.append(1 if "what follows" in lower_question and ("follows" in lower_answer or "not" in lower_answer) else 0)
    # Generic quality signals
    features.append(1 if n_words > 3 else 0)
    features.append(1 if not answer.endswith(" ") else 0)
    features.append(1 if answer == answer.strip() else 0)
    features.append(1 if len(answer) > 0 else 0)

    # f190-f199: token-level features
    # Repeated words (sign of generation errors)
    word_lower = [w.lower() for w in words]
    features.append(1 if len(word_lower) == len(set(word_lower)) else 0)
    # Has numeric suffix (e.g., "returns 42")
    features.append(1 if words and words[-1].isdigit() else 0)
    # Ends with code comment
    features.append(1 if answer.rstrip().startswith("#") or "\n#" in answer else 0)
    # Answer length ratio to question length
    q_len = max(len(question), 1)
    a_q_ratio = len(answer) / q_len
    features.append(1 if a_q_ratio > 2.0 else 0)
    features.append(1 if a_q_ratio < 0.5 else 0)
    # Has mathematical notation
    features.append(1 if "^" in answer or "**" in answer else 0)
    # Has list/tuple literal
    features.append(1 if re.search(r'\[.*\]', answer) else 0)
    # Starts with keyword
    first_word = words[0].lower() if words else ""
    features.append(1 if first_word in ("def", "the", "all", "no", "yes", "some") else 0)
    # Has comparison operator
    features.append(1 if ">=" in answer or "<=" in answer or "!=" in answer else 0)
    # Answer mentions a specific number from the question exactly once
    if q_numbers:
        exact_once = sum(1 for qn in q_numbers if answer.count(str(qn)) == 1)
        features.append(1 if exact_once > 0 else 0)
    else:
        features.append(0)

    # f200-f209: additional discriminative features
    # Answer contains "instead" (common in wrong-answer explanations)
    features.append(1 if "instead" in lower_answer else 0)
    # Answer starts with a number
    features.append(1 if answer and answer[0].isdigit() else 0)
    # Answer has balanced indentation depth (max indent <= 8)
    max_indent = max((len(l) - len(l.lstrip()) for l in lines if l), default=0)
    features.append(1 if max_indent <= 8 else 0)
    # Line count buckets
    n_lines = len(lines)
    features.append(1 if n_lines <= 3 else 0)
    features.append(1 if n_lines > 5 else 0)
    # Has "infinite" or "error" (common in wrong code explanations)
    features.append(1 if "infinite" in lower_answer or "error" in lower_answer else 0)
    # Answer word diversity (unique/total ratio)
    diversity = len(set(word_lower)) / max(len(word_lower), 1)
    features.append(1 if diversity > 0.7 else 0)
    features.append(1 if diversity < 0.4 else 0)
    # Contains "+" and a number (arithmetic answer pattern)
    features.append(1 if "+" in answer and a_numbers else 0)
    # Answer has exactly one sentence
    features.append(1 if n_sentences == 1 and "." in answer else 0)

    assert len(features) == 200, f"Total features: {len(features)}, expected 200"
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Discriminative CD training (from Exp 51, scaled with L1 from Exp 60)
# ---------------------------------------------------------------------------

def train_discriminative_cd_l1(
    correct_vectors: np.ndarray,
    wrong_vectors: np.ndarray,
    n_epochs: int = 300,
    lr: float = 0.05,
    beta: float = 1.0,
    l1_lambda: float = 0.001,
    weight_decay: float = 0.005,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train a sparse Ising model via discriminative CD with L1 regularization.

    **Detailed explanation for engineers:**
        This combines the discriminative CD approach from Exp 51 (positive
        phase = correct answers, negative phase = wrong answers) with the
        L1 regularization from Exp 60 (to enforce sparsity in the 200x200
        coupling matrix).

        The update rule is:
            ΔJ = -β(⟨s_i s_j⟩_correct - ⟨s_i s_j⟩_wrong) - λ·sign(J) - α·J
            Δb = -β(⟨s_i⟩_correct - ⟨s_i⟩_wrong) - α·b

        Where λ is the L1 penalty (encourages exact zeros in J, making the
        model sparse) and α is L2 weight decay (prevents unbounded growth
        since both phases use fixed data).

        After training, features with strong coupling have large |J_ij|
        values — these are the features the model found most discriminative
        between correct and wrong answers.

    Args:
        correct_vectors: Shape (n_correct, n_features), binary {0,1}.
        wrong_vectors: Shape (n_wrong, n_features), binary {0,1}.
        n_epochs: Training iterations.
        lr: Learning rate.
        beta: Inverse temperature.
        l1_lambda: L1 regularization on couplings (sparsity).
        weight_decay: L2 regularization (prevents unbounded growth).
        verbose: Print progress every 50 epochs.

    Returns:
        (biases, coupling_matrix, loss_history)
    """
    n_features = correct_vectors.shape[1]

    # Initialize parameters.
    rng = np.random.default_rng(42)
    biases = np.zeros(n_features, dtype=np.float32)
    J = rng.normal(0, 0.001, (n_features, n_features)).astype(np.float32)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

    # Compute phase statistics (constant since both phases use fixed data).
    correct_spins = 2.0 * correct_vectors - 1.0
    wrong_spins = 2.0 * wrong_vectors - 1.0

    pos_bias_moments = np.mean(correct_spins, axis=0)
    pos_weight_moments = np.mean(
        np.einsum("bi,bj->bij", correct_spins, correct_spins), axis=0
    )
    neg_bias_moments = np.mean(wrong_spins, axis=0)
    neg_weight_moments = np.mean(
        np.einsum("bi,bj->bij", wrong_spins, wrong_spins), axis=0
    )

    # Constant discriminative gradient.
    grad_b = -beta * (pos_bias_moments - neg_bias_moments)
    grad_J = -beta * (pos_weight_moments - neg_weight_moments)
    np.fill_diagonal(grad_J, 0.0)

    losses = []
    for epoch in range(n_epochs):
        # L1 gradient on couplings: pushes small weights to exactly zero.
        l1_grad = l1_lambda * np.sign(J)

        # Update with discriminative gradient + L1 + L2 regularization.
        biases -= lr * (grad_b + weight_decay * biases)
        J -= lr * (grad_J + l1_grad + weight_decay * J)
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)

        # Compute energy gap as loss metric.
        e_correct = compute_energies(correct_vectors, biases, J)
        e_wrong = compute_energies(wrong_vectors, biases, J)
        mean_gap = float(np.mean(e_wrong) - np.mean(e_correct))
        losses.append(mean_gap)

        if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
            sparsity = np.mean(np.abs(J) < 0.001) * 100
            acc = classification_accuracy(correct_vectors, wrong_vectors, biases, J)
            print(f"    Epoch {epoch:3d}: gap={mean_gap:+.4f}  acc={acc:.1%}  "
                  f"sparsity={sparsity:.0f}%")

    return biases, J, losses


def compute_energies(vectors: np.ndarray, biases: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Compute Ising energy for each sample.

    **Detailed explanation for engineers:**
        E(s) = -(b^T s + s^T J s) where s is in {-1,+1} representation.
        Low energy = model "likes" this state. High energy = model "dislikes" it.
        For discriminative training, we want E(correct) < E(wrong).
    """
    spins = 2.0 * vectors - 1.0
    bias_term = spins @ biases
    coupling_term = np.einsum("bi,ij,bj->b", spins, J, spins)
    return -(bias_term + coupling_term)


def classification_accuracy(
    correct_vectors: np.ndarray,
    wrong_vectors: np.ndarray,
    biases: np.ndarray,
    J: np.ndarray,
) -> float:
    """Fraction of pairs where E(correct) < E(wrong)."""
    n_pairs = min(correct_vectors.shape[0], wrong_vectors.shape[0])
    e_correct = compute_energies(correct_vectors[:n_pairs], biases, J)
    e_wrong = compute_energies(wrong_vectors[:n_pairs], biases, J)
    return float(np.mean(e_correct < e_wrong))


def compute_auroc(
    correct_vectors: np.ndarray,
    wrong_vectors: np.ndarray,
    biases: np.ndarray,
    J: np.ndarray,
) -> float:
    """Compute AUROC for the Ising model as a binary classifier.

    **Detailed explanation for engineers:**
        AUROC (Area Under the Receiver Operating Characteristic curve) measures
        how well the energy function separates correct from wrong answers,
        independent of threshold choice. AUROC = 1.0 means perfect separation,
        0.5 means random chance.

        We treat each answer's energy as a score: lower energy = predict correct.
        Correct answers are positive class (label=1), wrong answers are negative
        class (label=0). We compute AUROC via the Wilcoxon-Mann-Whitney statistic:
            AUROC = P(E(correct) < E(wrong)) over all (correct, wrong) pairs.

        This is more informative than accuracy because it doesn't depend on
        a specific decision threshold.
    """
    e_correct = compute_energies(correct_vectors, biases, J)
    e_wrong = compute_energies(wrong_vectors, biases, J)

    # Wilcoxon-Mann-Whitney: fraction of (correct, wrong) pairs where
    # E(correct) < E(wrong). This equals AUROC.
    n_correct = len(e_correct)
    n_wrong = len(e_wrong)

    # Vectorized: count pairs where correct energy < wrong energy.
    # Using broadcasting: e_correct[:, None] < e_wrong[None, :]
    # For large datasets, do this in chunks to avoid memory blowup.
    chunk_size = 1000
    concordant = 0
    tied = 0
    total = n_correct * n_wrong

    for i in range(0, n_correct, chunk_size):
        ec_chunk = e_correct[i:i + chunk_size]
        diff = e_wrong[None, :] - ec_chunk[:, None]  # (chunk, n_wrong)
        concordant += int(np.sum(diff > 0))
        tied += int(np.sum(diff == 0))

    auroc = (concordant + 0.5 * tied) / total
    return auroc


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def hand_coded_score(question: str, answer: str) -> float:
    """Hand-coded rule-based scoring for answer correctness.

    **Detailed explanation for engineers:**
        Simple heuristic baseline: checks domain-specific rules to score
        an answer. Returns a score where higher = more likely correct.

        - Arithmetic: does the answer contain the expected computed result?
        - Logic: does the answer reference a valid reasoning pattern?
        - Code: does the answer have proper structure (def + return + indentation)?

        This represents what you could do without learning — just hard-coding
        domain knowledge into rules.
    """
    lower_q = question.lower()
    lower_a = answer.lower()
    score = 0.0

    q_numbers = [int(x) for x in re.findall(r'\d+', question)]

    # Arithmetic rules.
    if "+" in question and len(q_numbers) >= 2:
        expected = q_numbers[0] + q_numbers[1]
        if str(expected) in answer:
            score += 2.0
    if "*" in question and len(q_numbers) >= 2:
        expected = q_numbers[0] * q_numbers[1]
        if str(expected) in answer:
            score += 2.0
    if "mod" in lower_q and len(q_numbers) >= 2:
        expected = q_numbers[0] % q_numbers[1]
        if str(expected) in answer:
            score += 2.0

    # Logic rules.
    if "what follows" in lower_q or "necessarily" in lower_q:
        if "modus ponens" in lower_a or "modus tollens" in lower_a:
            score += 1.5
        if "disjunctive syllogism" in lower_a:
            score += 1.5
        if "affirming the consequent" in lower_a:
            score += 1.5  # Correctly identifies the fallacy.
        if "might" in lower_a or "cannot conclude" in lower_a:
            score += 0.5  # Hedging is sometimes correct.

    # Code rules.
    if "write a function" in lower_q:
        if "def " in answer and "return" in answer:
            score += 1.0
        if ":\n    " in answer:
            score += 0.5  # Proper indentation.
        if answer.count("(") == answer.count(")"):
            score += 0.5  # Balanced parens.

    return score


def logprob_baseline_score(features: np.ndarray) -> float:
    """Feature-count heuristic: sum of active features as a proxy score.

    **Detailed explanation for engineers:**
        The simplest possible baseline: just count how many features are
        "on" (=1) in the binary feature vector. The intuition is that
        correct answers tend to have more structural features present
        (they mention the right numbers, have proper formatting, etc.).

        This is equivalent to an Ising model with uniform biases and zero
        couplings — it ignores all feature interactions.
    """
    return float(np.sum(features))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 62: Domain Constraint Learning from 10K Triples")
    print("  Discriminative CD on arithmetic + logic + code domains")
    print("  200 binary features, L1-sparse Ising, AUROC evaluation")
    print("=" * 70)

    start = time.time()
    rng = np.random.default_rng(42)

    # --- Step 1: Generate 10,000 triples ---
    print("\n--- Step 1: Generate 10,000 (question, correct, wrong) triples ---")
    t0 = time.time()

    arith_triples = generate_arithmetic_triples(3333, rng)
    logic_triples = generate_logic_triples(3333, rng)
    code_triples = generate_code_triples(3334, rng)

    all_triples = arith_triples + logic_triples + code_triples
    domain_labels = (
        ["arithmetic"] * len(arith_triples)
        + ["logic"] * len(logic_triples)
        + ["code"] * len(code_triples)
    )

    print(f"  Arithmetic: {len(arith_triples)} triples")
    print(f"  Logic:      {len(logic_triples)} triples")
    print(f"  Code:       {len(code_triples)} triples")
    print(f"  Total:      {len(all_triples)} triples ({time.time() - t0:.1f}s)")

    # Show examples.
    for domain, triples_list in [("Arithmetic", arith_triples), ("Logic", logic_triples), ("Code", code_triples)]:
        print(f"\n  {domain} example:")
        q, c, w = triples_list[0]
        print(f"    Q: {q[:80]}")
        print(f"    C: {c[:80]}")
        print(f"    W: {w[:80]}")

    # --- Step 2: Encode as binary features ---
    print("\n--- Step 2: Binary feature encoding (200 features per answer) ---")
    t0 = time.time()

    correct_features = []
    wrong_features = []
    for q, c, w in all_triples:
        correct_features.append(encode_answer(q, c))
        wrong_features.append(encode_answer(q, w))

    correct_all = np.array(correct_features)
    wrong_all = np.array(wrong_features)
    print(f"  Correct features shape: {correct_all.shape}")
    print(f"  Wrong features shape:   {wrong_all.shape}")
    print(f"  Mean features per correct answer: {correct_all.mean(axis=0).sum():.1f}")
    print(f"  Mean features per wrong answer:   {wrong_all.mean(axis=0).sum():.1f}")
    print(f"  Encoding time: {time.time() - t0:.1f}s")

    # --- Step 3: Train/test split (80/20) ---
    print("\n--- Step 3: Train/test split (80/20) ---")
    n_total = len(all_triples)
    indices = rng.permutation(n_total)
    n_train = int(0.8 * n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    correct_train = correct_all[train_idx]
    wrong_train = wrong_all[train_idx]
    correct_test = correct_all[test_idx]
    wrong_test = wrong_all[test_idx]

    domain_labels_arr = np.array(domain_labels)
    train_domains = domain_labels_arr[train_idx]
    test_domains = domain_labels_arr[test_idx]

    print(f"  Train: {n_train} pairs")
    print(f"  Test:  {len(test_idx)} pairs")

    # --- Step 4: Train models ---
    print("\n--- Step 4: Train discriminative Ising models ---")

    models = {}

    # Per-domain models.
    for domain in ["arithmetic", "logic", "code"]:
        d_train_mask = train_domains == domain
        d_correct = correct_train[d_train_mask]
        d_wrong = wrong_train[d_train_mask]

        print(f"\n  Training {domain} model ({d_correct.shape[0]} pairs)...")
        biases, J, losses = train_discriminative_cd_l1(
            d_correct, d_wrong,
            n_epochs=300, lr=0.05, beta=1.0,
            l1_lambda=0.001, weight_decay=0.005,
        )
        models[domain] = (biases, J, losses)

    # Combined model (all domains).
    print(f"\n  Training combined model ({correct_train.shape[0]} pairs)...")
    biases_comb, J_comb, losses_comb = train_discriminative_cd_l1(
        correct_train, wrong_train,
        n_epochs=300, lr=0.05, beta=1.0,
        l1_lambda=0.001, weight_decay=0.005,
    )
    models["combined"] = (biases_comb, J_comb, losses_comb)

    # --- Step 5: Evaluate ---
    print("\n--- Step 5: Evaluate on held-out test set ---")

    # Store results for the summary table.
    results = {}
    domains = ["arithmetic", "logic", "code"]

    for eval_domain in domains:
        d_test_mask = test_domains == eval_domain
        d_correct_test = correct_test[d_test_mask]
        d_wrong_test = wrong_test[d_test_mask]
        n_test_d = d_correct_test.shape[0]

        results[eval_domain] = {"n_test": n_test_d}

        # --- Learned Ising (per-domain) ---
        b, J_m, _ = models[eval_domain]
        auroc_domain = compute_auroc(d_correct_test, d_wrong_test, b, J_m)
        acc_domain = classification_accuracy(d_correct_test, d_wrong_test, b, J_m)
        results[eval_domain]["ising_domain"] = {"auroc": auroc_domain, "acc": acc_domain}

        # --- Learned Ising (combined) ---
        b_c, J_c, _ = models["combined"]
        auroc_combined = compute_auroc(d_correct_test, d_wrong_test, b_c, J_c)
        acc_combined = classification_accuracy(d_correct_test, d_wrong_test, b_c, J_c)
        results[eval_domain]["ising_combined"] = {"auroc": auroc_combined, "acc": acc_combined}

        # --- Hand-coded baseline ---
        hc_correct_scores = []
        hc_wrong_scores = []
        for i in test_idx[d_test_mask[np.arange(len(test_idx))]]:
            q, c, w = all_triples[i]
            hc_correct_scores.append(hand_coded_score(q, c))
            hc_wrong_scores.append(hand_coded_score(q, w))
        hc_correct_scores = np.array(hc_correct_scores)
        hc_wrong_scores = np.array(hc_wrong_scores)
        # AUROC for hand-coded: fraction where score(correct) > score(wrong).
        n_hc = len(hc_correct_scores)
        if n_hc > 0:
            hc_concordant = np.sum(hc_correct_scores > hc_wrong_scores)
            hc_tied = np.sum(hc_correct_scores == hc_wrong_scores)
            hc_auroc = (hc_concordant + 0.5 * hc_tied) / n_hc
            hc_acc = float(hc_concordant / n_hc)
        else:
            hc_auroc = 0.5
            hc_acc = 0.5
        results[eval_domain]["hand_coded"] = {"auroc": hc_auroc, "acc": hc_acc}

        # --- Logprob baseline ---
        lp_correct_scores = np.array([logprob_baseline_score(correct_all[i]) for i in test_idx[d_test_mask[np.arange(len(test_idx))]]])
        lp_wrong_scores = np.array([logprob_baseline_score(wrong_all[i]) for i in test_idx[d_test_mask[np.arange(len(test_idx))]]])
        n_lp = len(lp_correct_scores)
        if n_lp > 0:
            lp_concordant = np.sum(lp_correct_scores > lp_wrong_scores)
            lp_tied = np.sum(lp_correct_scores == lp_wrong_scores)
            lp_auroc = (lp_concordant + 0.5 * lp_tied) / n_lp
            lp_acc = float(lp_concordant / n_lp)
        else:
            lp_auroc = 0.5
            lp_acc = 0.5
        results[eval_domain]["logprob"] = {"auroc": lp_auroc, "acc": lp_acc}

    # Also compute combined AUROC across all domains.
    b_c, J_c, _ = models["combined"]
    overall_auroc = compute_auroc(correct_test, wrong_test, b_c, J_c)
    overall_acc = classification_accuracy(correct_test, wrong_test, b_c, J_c)

    # --- Step 6: Feature importance ---
    print("\n--- Step 6: Feature importance (strongest couplings) ---")
    b_c, J_c, _ = models["combined"]
    # Sum of absolute coupling strengths for each feature.
    feature_importance = np.sum(np.abs(J_c), axis=1) + np.abs(b_c)
    top_features = np.argsort(feature_importance)[::-1][:20]

    feature_names = _feature_names()
    print(f"\n  Top 20 most important features (combined model):")
    for rank, fi in enumerate(top_features):
        name = feature_names[fi] if fi < len(feature_names) else f"f{fi}"
        print(f"    {rank+1:2d}. f{fi:3d} ({name:40s}) importance={feature_importance[fi]:.4f}")

    # --- Step 7: Print AUROC table ---
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 62 RESULTS ({elapsed:.0f}s)")
    print(sep)

    print(f"\n  AUROC Table (per-domain x model-type):")
    print(f"  {'Domain':>12s} | {'Ising(domain)':>14s} | {'Ising(combined)':>16s} | "
          f"{'Hand-coded':>11s} | {'Logprob':>8s} | {'N_test':>6s}")
    print(f"  {'-' * 80}")

    for domain in domains:
        r = results[domain]
        print(f"  {domain:>12s} | "
              f"{r['ising_domain']['auroc']:>14.3f} | "
              f"{r['ising_combined']['auroc']:>16.3f} | "
              f"{r['hand_coded']['auroc']:>11.3f} | "
              f"{r['logprob']['auroc']:>8.3f} | "
              f"{r['n_test']:>6d}")

    print(f"  {'-' * 80}")
    print(f"  {'OVERALL':>12s} | {'---':>14s} | {overall_auroc:>16.3f} | "
          f"{'---':>11s} | {'---':>8s} | {len(test_idx):>6d}")

    print(f"\n  Accuracy Table (E_correct < E_wrong):")
    print(f"  {'Domain':>12s} | {'Ising(domain)':>14s} | {'Ising(combined)':>16s} | "
          f"{'Hand-coded':>11s} | {'Logprob':>8s}")
    print(f"  {'-' * 70}")

    for domain in domains:
        r = results[domain]
        print(f"  {domain:>12s} | "
              f"{r['ising_domain']['acc']:>13.1%} | "
              f"{r['ising_combined']['acc']:>15.1%} | "
              f"{r['hand_coded']['acc']:>10.1%} | "
              f"{r['logprob']['acc']:>7.1%}")

    print(f"  {'-' * 70}")
    print(f"  {'OVERALL':>12s} | {'---':>14s} | {overall_acc:>15.1%} | "
          f"{'---':>11s} | {'---':>8s}")

    # --- Verdict ---
    print(f"\n  Model statistics:")
    for name, (b, J_m, losses) in models.items():
        sparsity = np.mean(np.abs(J_m) < 0.001) * 100
        n_nonzero = np.sum(np.abs(J_m) >= 0.001)
        print(f"    {name:12s}: {n_nonzero:5d} non-zero couplings ({sparsity:.0f}% sparse), "
              f"final gap={losses[-1]:+.4f}")

    # Verdict logic.
    mean_domain_auroc = np.mean([results[d]["ising_domain"]["auroc"] for d in domains])
    mean_hc_auroc = np.mean([results[d]["hand_coded"]["auroc"] for d in domains])
    mean_lp_auroc = np.mean([results[d]["logprob"]["auroc"] for d in domains])

    print(f"\n  Mean AUROC: Ising(domain)={mean_domain_auroc:.3f}, "
          f"Hand-coded={mean_hc_auroc:.3f}, Logprob={mean_lp_auroc:.3f}")

    if mean_domain_auroc > 0.8:
        print(f"\n  VERDICT: Strong domain-specific constraint learning (AUROC > 0.8)")
    elif mean_domain_auroc > 0.65:
        print(f"\n  VERDICT: Moderate constraint learning (AUROC 0.65-0.8)")
    elif mean_domain_auroc > 0.55:
        print(f"\n  VERDICT: Weak but above-chance constraint learning")
    else:
        print(f"\n  VERDICT: Ising model does not learn meaningful constraints")

    if mean_domain_auroc > mean_hc_auroc:
        print(f"  vs Hand-coded: Learned model BEATS hand-coded rules "
              f"({mean_domain_auroc:.3f} vs {mean_hc_auroc:.3f})")
    else:
        print(f"  vs Hand-coded: Hand-coded rules still better "
              f"({mean_hc_auroc:.3f} vs {mean_domain_auroc:.3f})")

    if overall_auroc > mean_domain_auroc:
        print(f"  Combined vs per-domain: Combined model wins "
              f"({overall_auroc:.3f} vs {mean_domain_auroc:.3f})")
    else:
        print(f"  Combined vs per-domain: Per-domain specialists win "
              f"({mean_domain_auroc:.3f} vs {overall_auroc:.3f})")

    print(sep)
    return 0


def _feature_names() -> list[str]:
    """Return human-readable names for each of the 200 features.

    **Detailed explanation for engineers:**
        Maps feature index to a descriptive string for interpretability.
        Organized by feature group matching the encode_answer function.
    """
    names = []
    # Numeric (0-19)
    names.append("has_number")
    for k in range(5):
        names.append(f"num_count=={k}")
    names.append("digit_count>3")
    names.append("digit_sum_even")
    names.append("digit_sum_odd")
    for k in range(5):
        names.append(f"q_nums_in_a=={k}")
    names.append("a_max>q_max")
    names.append("has_expected_sum")
    names.append("has_expected_product")
    names.append("has_expected_mod")
    names.append("max_a<10")
    names.append("max_a>=100")

    # Structural (20-59)
    for ub in [2, 5, 10, 15, 20, 30, 50, 100]:
        names.append(f"words<={ub}")
    names.append("words>100")
    names.append("words==1")
    for k in range(1, 5):
        names.append(f"sentences=={k}")
    names.append("sentences>=5")
    names.append("has_period")
    names.append("has_comma")
    names.append("has_exclamation")
    names.append("has_question")
    names.append("has_colon")
    names.append("starts_uppercase")
    names.append("has_inner_uppercase")
    names.append("ends_with_period")
    names.append("has_newline")
    names.append("has_double_space")
    names.append("has_open_paren")
    names.append("has_close_paren")
    names.append("has_open_bracket")
    names.append("has_close_bracket")
    names.append("has_open_brace")
    names.append("has_close_brace")
    names.append("has_single_quote")
    names.append("has_double_quote")
    names.append("has_hash")
    names.append("has_equals")
    names.append("chars<20")
    names.append("chars_20-50")
    names.append("chars_50-100")
    names.append("chars_100-200")
    names.append("chars>=200")

    # Domain-specific (60-159)
    names.append("has_plus")
    names.append("has_minus")
    names.append("has_star")
    names.append("has_slash")
    names.append("has_doubleslash")
    names.append("has_percent")
    names.append("has_single_eq")
    names.append("has_double_eq")
    names.append("has_the_answer_is")
    names.append("has_returns")
    names.append("has_if")
    names.append("has_then")
    names.append("has_therefore")
    names.append("has_because")
    names.append("has_follows")
    names.append("has_not")
    names.append("has_all")
    names.append("has_some")
    names.append("has_modus")
    names.append("has_syllogism")
    names.append("has_def")
    names.append("has_return")
    names.append("has_for")
    names.append("has_while")
    names.append("has_if_kw")
    names.append("has_else")
    names.append("has_import")
    names.append("has_class")
    names.append("has_range")
    names.append("has_len")
    names.append("has_indent")
    names.append("multiline>2")
    names.append("has_bool_literal")
    names.append("has_None")
    names.append("has_isinstance")
    names.append("has_append")
    names.append("has_extend")
    names.append("has_slice")
    names.append("has_recursion")
    names.append("has_edge_case")
    names.append("q_has_plus")
    names.append("q_has_star")
    names.append("q_has_mod")
    names.append("q_has_if")
    names.append("q_write_function")
    names.append("q_what_is")
    names.append("q_what_follows")
    names.append("q_necessarily")
    names.append("q_returns")
    names.append("q_check_sorted")
    names.append("a_yes")
    names.append("a_no")
    names.append("a_correct")
    names.append("a_incorrect")
    names.append("a_cannot")
    names.append("a_must")
    names.append("a_might")
    names.append("a_guaranteed")
    names.append("a_invalid")
    names.append("a_valid")
    # Bigram features (120-139)
    target_bigrams = [
        "the_answer", "answer_is", "is_not", "not_the", "does_not",
        "this_follows", "follows_by", "by_modus", "n_+", "+_1",
        "return_total", "return_result", "for_i", "in_range", "if_not",
        "not_lst", "lst_0", "==_0", "==_1", "def_",
    ]
    for tb in target_bigrams:
        names.append(f"bigram:{tb}")

    # Consistency (160-199)
    names.append("parens_balanced")
    names.append("brackets_balanced")
    names.append("braces_balanced")
    names.append("parens_balanced_nonzero")
    names.append("quotes_balanced_single")
    names.append("quotes_balanced_double")
    names.append("has_colon_newline")
    names.append("has_colon_indent")
    names.append("def_has_return")
    names.append("consistent_indent")
    names.append("all_q_nums_in_a")
    names.append("half_q_nums_in_a")
    names.append("no_q_nums_in_a")
    names.append("num_ratio_reasonable")
    names.append("unique_numbers")
    names.append("few_digits")
    names.append("has_digits")
    names.append("digit_sum_reasonable")
    names.append("long_for_code_q")
    names.append("short_for_what_is_q")
    names.append("not_in_both")
    names.append("all_in_both")
    names.append("shared_content>2")
    names.append("shared_content>5")
    names.append("has_affirming")
    names.append("has_consequent")
    names.append("has_tollens")
    names.append("has_ponens")
    names.append("has_disjunctive")
    names.append("has_eliminated")
    names.append("arith_format_match")
    names.append("logic_reasoning_match")
    names.append("code_structure_match")
    names.append("write_has_def")
    names.append("what_is_has_digit")
    names.append("what_follows_has_follows")
    names.append("words>3")
    names.append("no_trailing_space")
    names.append("is_stripped")
    names.append("non_empty")
    # Token-level (190-199)
    names.append("all_words_unique")
    names.append("ends_with_digit")
    names.append("has_code_comment")
    names.append("a_q_ratio>2")
    names.append("a_q_ratio<0.5")
    names.append("has_math_notation")
    names.append("has_list_literal")
    names.append("starts_keyword")
    names.append("has_comparison_op")
    names.append("q_num_exact_once")
    # Additional (200-209)
    names.append("has_instead")
    names.append("starts_with_digit")
    names.append("max_indent<=8")
    names.append("lines<=3")
    names.append("lines>5")
    names.append("has_infinite_error")
    names.append("diversity>0.7")
    names.append("diversity<0.4")
    names.append("plus_and_number")
    names.append("single_sentence")

    return names


if __name__ == "__main__":
    sys.exit(main())
