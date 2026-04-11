#!/usr/bin/env python3
"""Experiment 153: KAN Adaptive Mesh Refinement for Constraint Verification.

**Researcher summary:**
    KAN edge splines can adaptively allocate representational capacity: edges
    that model complex nonlinear constraint interactions (high curvature) get
    more knots inserted, while near-linear edges shed knots. This is the Tier 4
    "adaptive energy structure" mechanism from research-program.md.

    This experiment implements compute_edge_curvature() and refine() on the
    KANConstraintModel (based on Exp 109 KANTrainer), benchmarks on a 200-question
    arithmetic + logic constraint verification task, and measures AUROC and
    parameter count before vs after adaptive mesh refinement.

**Background (for engineers new to KAN):**
    A KAN (Kolmogorov-Arnold Network) replaces scalar edge weights with learnable
    1D spline functions. Each edge (i, j) has a B-spline f_ij that maps the
    product x_i * x_j to an energy contribution. The spline is piecewise linear,
    controlled by a set of "knots" (breakpoints) and associated control points.

    Adaptive mesh refinement (AMR) adjusts where knots are placed:
    - HIGH curvature region: the spline bends sharply → insert a new knot to
      capture the nonlinearity with finer resolution.
    - LOW curvature region: the spline is nearly linear → remove a knot to
      save parameters (merge two adjacent intervals into one).

    Curvature is measured via the second derivative |d²f/dx²|, estimated with
    finite differences (central differences: (f(x+h) - 2f(x) + f(x-h)) / h²).

**Experimental design:**
    1. Generate 200 question-answer pairs (100 arithmetic + 100 logic).
    2. Encode as 200-dim binary feature vectors (Exp 109 encoding).
    3. Select top-20 discriminative features from Ising training.
    4. Train KANConstraintModel for 100 epochs (baseline).
    5. Compute AUROC_before on test set.
    6. Run refine(threshold_multiplier=1.5) to insert/remove knots.
    7. Fine-tune for 10 more epochs.
    8. Compute AUROC_after on test set.
    9. Compare: AUROC before vs after, param count before vs after.
    10. Analyse which edges gained/lost knots and their semantic meaning.

**Targets:**
    - AUROC maintained or improved (delta >= -0.01 acceptable).
    - Parameter count changes by <= ±20%.

**Output:** results/experiment_153_results.json

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_153_kan_refinement.py

Spec: REQ-CORE-001, REQ-CORE-002, REQ-TIER-001
Scenario: SCENARIO-CORE-001
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Force CPU for reproducibility (see CLAUDE.md: ROCm JAX may crash on thrml).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

# Allow importing carnot from the python/ directory (not needed here but kept
# for consistency with other experiment scripts).
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data generation (inline from Exp 109 for self-containment)
# ---------------------------------------------------------------------------

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


def generate_arithmetic_triples(
    n: int, rng: np.random.Generator
) -> list[tuple[str, str, str]]:
    """Generate (question, correct_answer, wrong_answer) for arithmetic.

    Three sub-types: addition, multiplication, modulo (equal thirds).
    The wrong answer is a plausible near-miss (off-by-one, carry error, etc.).

    Args:
        n: Number of triples to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        List of (question, correct, wrong) string tuples, length n.
    """
    triples = []
    per_type = n // 3
    remainder = n - 3 * per_type

    # Addition triples.
    for _ in range(per_type + (1 if remainder > 0 else 0)):
        a = int(rng.integers(1, 500))
        b = int(rng.integers(1, 500))
        correct = a + b
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

    # Multiplication triples.
    for _ in range(per_type + (1 if remainder > 0 else 0)):
        a = int(rng.integers(2, 50))
        b = int(rng.integers(2, 50))
        correct = a * b
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

    # Modulo triples.
    for _ in range(per_type):
        a = int(rng.integers(10, 1000))
        b = int(rng.integers(2, 50))
        correct = a % b
        wrong = correct + int(rng.choice([-1, 1, 2, -2]))
        if wrong == correct:
            wrong = correct + 1
        if wrong < 0:
            wrong = abs(wrong)
        question = f"What is {a} mod {b}?"
        triples.append((question, f"The answer is {correct}.", f"The answer is {wrong}."))

    return triples[:n]


def generate_logic_triples(
    n: int, rng: np.random.Generator
) -> list[tuple[str, str, str]]:
    """Generate (question, correct_answer, wrong_answer) for logic syllogisms.

    Four syllogism types: modus ponens, modus tollens, disjunctive syllogism,
    affirming-the-consequent (fallacy — correct answer is "no").

    Args:
        n: Number of triples to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        List of (question, correct, wrong) string tuples, length n.
    """
    triples = []
    per_type = n // 4
    remainder = n - 4 * per_type

    def pick_pair(rng: np.random.Generator) -> tuple[str, str, str, str]:
        """Pick two distinct subject-predicate pairs."""
        s = _SUBJECTS[int(rng.integers(0, len(_SUBJECTS)))]
        p = _PREDICATES[int(rng.integers(0, len(_PREDICATES)))]
        s2 = _SUBJECTS[int(rng.integers(0, len(_SUBJECTS)))]
        while s2 == s:
            s2 = _SUBJECTS[int(rng.integers(0, len(_SUBJECTS)))]
        p2 = _PREDICATES[int(rng.integers(0, len(_PREDICATES)))]
        while p2 == p:
            p2 = _PREDICATES[int(rng.integers(0, len(_PREDICATES)))]
        return s, p, s2, p2

    # Modus ponens: "all S are P, all S2 are S → all S2 are P."
    for _ in range(per_type + min(1, remainder)):
        s, p, s2, p2 = pick_pair(rng)
        question = f"If all {s} are {p}, and all {s2} are {s}, what follows?"
        correct = f"All {s2} are {p}. This follows by modus ponens."
        wrong = f"Some {s2} are not {p}. The premises do not guarantee this."
        triples.append((question, correct, wrong))
    remainder = max(0, remainder - 1)

    # Modus tollens: "all S are P, S2 is not P → S2 is not S."
    for _ in range(per_type + min(1, remainder)):
        s, p, s2, p2 = pick_pair(rng)
        question = f"If all {s} are {p}, and {s2} are not {p}, what follows?"
        correct = f"{s2} are not {s}. This follows by modus tollens."
        wrong = f"{s2} might still be {s}. We cannot conclude anything."
        triples.append((question, correct, wrong))
    remainder = max(0, remainder - 1)

    # Disjunctive syllogism: "S are P or P2, S are not P → S are P2."
    for _ in range(per_type + min(1, remainder)):
        s, p, s2, p2 = pick_pair(rng)
        question = f"Either {s} are {p} or {s} are {p2}. {s} are not {p}. What follows?"
        correct = f"{s} are {p2}. This follows by disjunctive syllogism."
        wrong = f"{s} are neither {p} nor {p2}. Both options are eliminated."
        triples.append((question, correct, wrong))
    remainder = max(0, remainder - 1)

    # Affirming the consequent (fallacy): answer is "no, it doesn't follow".
    for _ in range(per_type):
        s, p, s2, p2 = pick_pair(rng)
        question = f"If all {s} are {p}, and {s2} are {p}, are {s2} necessarily {s}?"
        correct = f"No. {s2} being {p} does not mean {s2} are {s}. This is affirming the consequent."
        wrong = f"Yes. Since all {s} are {p} and {s2} are {p}, {s2} must be {s}."
        triples.append((question, correct, wrong))

    return triples[:n]


# ---------------------------------------------------------------------------
# Feature encoding: 200-dim binary (identical to Exp 109 / Exp 62)
# ---------------------------------------------------------------------------

def encode_answer(question: str, answer: str) -> np.ndarray:
    """Encode a (question, answer) pair as a 200-dim binary feature vector.

    Feature groups:
    - Features 0-19:   Numeric (digits in answer, matches with question numbers).
    - Features 20-59:  Structural (word count, sentence count, punctuation).
    - Features 60-139: Domain-specific (keywords like "modus ponens", "def ", etc.).
    - Features 140-199: Consistency (bracket matching, code structure, redundancy).

    This encoding is identical to Exp 62 and Exp 109 for comparability.

    Args:
        question: The question string.
        answer: The answer string to encode.

    Returns:
        Binary feature vector, shape (200,), dtype float32.
    """
    features: list[int] = []

    q_numbers = [int(x) for x in re.findall(r'\d+', question)]
    a_numbers = [int(x) for x in re.findall(r'\d+', answer)]
    a_digits = re.findall(r'\d', answer)

    features.append(1 if a_numbers else 0)
    n_nums = len(a_numbers)
    for k in range(5):
        features.append(1 if n_nums == k else 0)
    features.append(1 if len(a_digits) > 3 else 0)
    digit_sum = sum(int(d) for d in a_digits) if a_digits else 0
    features.append(1 if digit_sum % 2 == 0 else 0)
    features.append(1 if digit_sum % 2 == 1 else 0)
    q_in_a = sum(1 for qn in q_numbers if qn in a_numbers)
    for k in range(5):
        features.append(1 if q_in_a == k else 0)
    max_q = max(q_numbers) if q_numbers else 0
    max_a = max(a_numbers) if a_numbers else 0
    features.append(1 if max_a > max_q else 0)
    if len(q_numbers) >= 2 and "+" in question:
        features.append(1 if (q_numbers[0] + q_numbers[1]) in a_numbers else 0)
    else:
        features.append(0)
    if len(q_numbers) >= 2 and "*" in question:
        features.append(1 if (q_numbers[0] * q_numbers[1]) in a_numbers else 0)
    else:
        features.append(0)
    if len(q_numbers) >= 2 and "mod" in question:
        features.append(1 if (q_numbers[0] % q_numbers[1]) in a_numbers else 0)
    else:
        features.append(0)
    features.append(1 if max_a < 10 else 0)
    features.append(1 if max_a >= 100 else 0)
    assert len(features) == 20

    words = answer.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
    n_words = len(words)
    n_sentences = len(sentences)
    wc_bounds = [2, 5, 10, 15, 20, 30, 50, 100]
    for ub in wc_bounds:
        features.append(1 if n_words <= ub else 0)
    features.append(1 if n_words > 100 else 0)
    features.append(1 if n_words == 1 else 0)
    for k in range(1, 5):
        features.append(1 if n_sentences == k else 0)
    features.append(1 if n_sentences >= 5 else 0)
    features.append(1 if "." in answer else 0)
    features.append(1 if "," in answer else 0)
    features.append(1 if "!" in answer else 0)
    features.append(1 if "?" in answer else 0)
    features.append(1 if ":" in answer else 0)
    features.append(1 if answer[0].isupper() else 0)
    features.append(1 if any(c.isupper() for c in answer[1:]) else 0)
    features.append(1 if answer.strip().endswith(".") else 0)
    features.append(1 if "\n" in answer else 0)
    features.append(1 if "  " in answer else 0)
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
    n_chars = len(answer)
    features.append(1 if n_chars < 20 else 0)
    features.append(1 if 20 <= n_chars < 50 else 0)
    features.append(1 if 50 <= n_chars < 100 else 0)
    features.append(1 if 100 <= n_chars < 200 else 0)
    features.append(1 if n_chars >= 200 else 0)
    assert len(features) == 60

    lower_answer = answer.lower()
    lower_question = question.lower()
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
    features.append(1 if "    " in answer else 0)
    features.append(1 if answer.count("\n") > 2 else 0)
    features.append(1 if "True" in answer or "False" in answer else 0)
    features.append(1 if "None" in answer else 0)
    features.append(1 if "isinstance" in answer else 0)
    features.append(1 if ".append(" in answer else 0)
    features.append(1 if ".extend(" in answer else 0)
    features.append(1 if "[::" in answer else 0)
    features.append(1 if "recursion" in lower_answer or "recursive" in lower_answer else 0)
    features.append(1 if "edge case" in lower_answer or "empty" in lower_answer else 0)
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
    assert len(features) == 140

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
    features.append(1 if ":\n" in answer else 0)
    features.append(1 if ":\n    " in answer else 0)
    features.append(1 if ("def " in answer and "return" in answer) or "def " not in answer else 0)
    lines = answer.split("\n")
    indented = [line for line in lines if line and line[0] == " "]
    if indented:
        consistent = all((len(line) - len(line.lstrip())) in (4, 8) for line in indented)
        features.append(1 if consistent else 0)
    else:
        features.append(0)
    if q_numbers:
        frac_present = sum(1 for qn in q_numbers if str(qn) in answer) / len(q_numbers)
    else:
        frac_present = 0.0
    features.append(1 if frac_present >= 1.0 else 0)
    features.append(1 if frac_present >= 0.5 else 0)
    features.append(1 if frac_present == 0.0 else 0)
    if q_numbers and a_numbers:
        ratio = max_a / max(max_q, 1)
        features.append(1 if 0.01 < ratio < 1000 else 0)
    else:
        features.append(1)
    if a_numbers:
        unique_ratio = len(set(a_numbers)) / len(a_numbers)
        features.append(1 if unique_ratio > 0.5 else 0)
    else:
        features.append(1)
    features.append(1 if len(a_digits) <= 6 else 0)
    features.append(1 if len(a_digits) > 0 else 0)
    features.append(1 if digit_sum > 0 and digit_sum < 50 else 0)
    features.append(1 if "write a function" in lower_question and n_chars > 30 else 0)
    features.append(1 if "what is" in lower_question and n_chars < 50 else 0)
    features.append(1 if "not" in lower_answer and "not" in lower_question else 0)
    features.append(1 if "all" in lower_answer and "all" in lower_question else 0)
    q_words_set = set(lower_question.split())
    a_words_set = set(lower_answer.split())
    shared_content_words = q_words_set & a_words_set - {
        "a", "an", "the", "is", "are", "what", "if", "and", "or", "in", "of", "to"
    }
    features.append(1 if len(shared_content_words) > 2 else 0)
    features.append(1 if len(shared_content_words) > 5 else 0)
    features.append(1 if "affirming" in lower_answer else 0)
    features.append(1 if "consequent" in lower_answer else 0)
    features.append(1 if "tollens" in lower_answer else 0)
    features.append(1 if "ponens" in lower_answer else 0)
    features.append(1 if "disjunctive" in lower_answer else 0)
    features.append(1 if "eliminate" in lower_answer or "eliminated" in lower_answer else 0)
    features.append(
        1 if "the answer is" in lower_answer
        and ("+" in question or "*" in question or "mod" in question) else 0
    )
    features.append(
        1 if ("follows" in lower_answer or "cannot" in lower_answer)
        and ("if " in lower_question) else 0
    )
    features.append(1 if "def " in answer and "return" in answer and ":\n" in answer else 0)
    features.append(1 if "write" in lower_question and "def " in answer else 0)
    features.append(
        1 if "what is" in lower_question and any(c.isdigit() for c in answer) else 0
    )
    features.append(
        1 if "what follows" in lower_question
        and ("follows" in lower_answer or "not" in lower_answer) else 0
    )
    features.append(1 if n_words > 3 else 0)
    features.append(1 if not answer.endswith(" ") else 0)
    features.append(1 if answer == answer.strip() else 0)
    features.append(1 if len(answer) > 0 else 0)
    word_lower = [w.lower() for w in words]
    features.append(1 if len(word_lower) == len(set(word_lower)) else 0)
    features.append(1 if words and words[-1].isdigit() else 0)
    features.append(1 if answer.rstrip().startswith("#") or "\n#" in answer else 0)
    q_len = max(len(question), 1)
    a_q_ratio = len(answer) / q_len
    features.append(1 if a_q_ratio > 2.0 else 0)
    features.append(1 if a_q_ratio < 0.5 else 0)
    features.append(1 if "^" in answer or "**" in answer else 0)
    features.append(1 if re.search(r'\[.*\]', answer) else 0)
    first_word = words[0].lower() if words else ""
    features.append(1 if first_word in ("def", "the", "all", "no", "yes", "some") else 0)
    features.append(1 if ">=" in answer or "<=" in answer or "!=" in answer else 0)
    if q_numbers:
        exact_once = sum(1 for qn in q_numbers if answer.count(str(qn)) == 1)
        features.append(1 if exact_once > 0 else 0)
    else:
        features.append(0)
    features.append(1 if "instead" in lower_answer else 0)
    features.append(1 if answer and answer[0].isdigit() else 0)
    max_indent = max((len(line) - len(line.lstrip()) for line in lines if line), default=0)
    features.append(1 if max_indent <= 8 else 0)
    n_lines = len(lines)
    features.append(1 if n_lines <= 3 else 0)
    features.append(1 if n_lines > 5 else 0)
    features.append(1 if "infinite" in lower_answer or "error" in lower_answer else 0)
    diversity = len(set(word_lower)) / max(len(word_lower), 1)
    features.append(1 if diversity > 0.7 else 0)
    features.append(1 if diversity < 0.4 else 0)
    features.append(1 if "+" in answer and a_numbers else 0)
    features.append(1 if n_sentences == 1 and "." in answer else 0)
    assert len(features) == 200, f"Expected 200 features, got {len(features)}"
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Ising training for feature selection (from Exp 109)
# ---------------------------------------------------------------------------

def compute_energies_ising(
    vectors: np.ndarray, biases: np.ndarray, J: np.ndarray
) -> np.ndarray:
    """Ising energy per sample: E(s) = -(b^T s + s^T J s), s in {-1, +1}.

    Args:
        vectors: Binary input vectors, shape (n, d), values in {0, 1}.
        biases: Bias vector, shape (d,).
        J: Coupling matrix, shape (d, d), symmetric with zero diagonal.

    Returns:
        Energy values, shape (n,).
    """
    spins = 2.0 * vectors - 1.0
    bias_term = spins @ biases
    coupling_term = np.einsum("bi,ij,bj->b", spins, J, spins)
    return -(bias_term + coupling_term)


def train_ising_discriminative_cd(
    correct_vectors: np.ndarray,
    wrong_vectors: np.ndarray,
    n_epochs: int = 200,
    lr: float = 0.05,
    beta: float = 1.0,
    l1_lambda: float = 0.001,
    weight_decay: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train a sparse Ising model via discriminative contrastive divergence.

    Used only for feature selection — identifying the 20 most discriminative
    features from the 200-dim space. The Ising training uses closed-form
    moment matching (no sampling needed), which is faster.

    Args:
        correct_vectors: Shape (n, d), binary {0,1}.
        wrong_vectors: Shape (n, d), binary {0,1}.
        n_epochs: Number of gradient steps.
        lr: Learning rate.
        beta: Inverse temperature.
        l1_lambda: L1 regularisation weight (encourages sparsity).
        weight_decay: L2 regularisation weight.

    Returns:
        (biases, J_coupling, loss_history)
    """
    n_features = correct_vectors.shape[1]
    rng_np = np.random.default_rng(42)
    biases = np.zeros(n_features, dtype=np.float32)
    J = rng_np.normal(0, 0.001, (n_features, n_features)).astype(np.float32)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

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

    grad_b = -beta * (pos_bias_moments - neg_bias_moments)
    grad_J = -beta * (pos_weight_moments - neg_weight_moments)
    np.fill_diagonal(grad_J, 0.0)

    losses = []
    for epoch in range(n_epochs):
        l1_grad = l1_lambda * np.sign(J)
        biases -= lr * (grad_b + weight_decay * biases)
        J -= lr * (grad_J + l1_grad + weight_decay * J)
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)

        e_correct = compute_energies_ising(correct_vectors, biases, J)
        e_wrong = compute_energies_ising(wrong_vectors, biases, J)
        mean_gap = float(np.mean(e_wrong) - np.mean(e_correct))
        losses.append(mean_gap)

    return biases, J, losses


def select_top_features(
    biases: np.ndarray, J: np.ndarray, k: int
) -> list[int]:
    """Select top-k features by total coupling strength (row-sum of |J| + |b|).

    Features with large total coupling are the most discriminative — they
    participate in many strong interactions with other features.

    Args:
        biases: Ising bias vector, shape (d,).
        J: Coupling matrix, shape (d, d).
        k: Number of features to return.

    Returns:
        List of k feature indices, sorted by importance descending.
    """
    importance = np.sum(np.abs(J), axis=1) + np.abs(biases)
    return list(np.argsort(importance)[::-1][:k])


# ---------------------------------------------------------------------------
# AUROC computation
# ---------------------------------------------------------------------------

def auroc_from_energies(e_correct: np.ndarray, e_wrong: np.ndarray) -> float:
    """AUROC via Wilcoxon-Mann-Whitney: P(E_correct < E_wrong).

    An AUROC of 1.0 means all correct answers have strictly lower energy
    than all wrong answers. 0.5 = random chance.

    Args:
        e_correct: Energy values for correct answers, shape (n_c,).
        e_wrong: Energy values for wrong answers, shape (n_w,).

    Returns:
        AUROC in [0, 1].
    """
    n_c = len(e_correct)
    n_w = len(e_wrong)
    concordant = 0
    tied = 0
    chunk = 500
    for i in range(0, n_c, chunk):
        ec = e_correct[i:i + chunk]
        diff = e_wrong[None, :] - ec[:, None]
        concordant += int(np.sum(diff > 0))
        tied += int(np.sum(diff == 0))
    return (concordant + 0.5 * tied) / (n_c * n_w)


# ---------------------------------------------------------------------------
# KANConstraintModel: discriminative KAN with adaptive mesh refinement
# ---------------------------------------------------------------------------

class KANConstraintModel:
    """Discriminative KAN for constraint verification with adaptive mesh refinement.

    Wraps the B-spline KAN trainer (from Exp 109 KANTrainer) with two new methods:
    - compute_edge_curvature(): estimates |d²f/dx²| per edge via finite differences.
    - refine(threshold_multiplier): inserts/removes knots based on curvature.

    After refinement, different edges can have different numbers of control points.
    The fine-tuning loop respects per-edge n_ctrl.

    **Energy formula:**
        E(x) = sum_{(i,j) in edges} f_{ij}(s_i * s_j) + sum_i g_i(s_i)
        where s = 2x - 1 (binary {0,1} → spin {-1,+1}).

    **Spline parameterisation:**
        Each spline f_{ij} uses piecewise linear interpolation between n_ctrl
        control points. n_ctrl = num_knots + degree initially.
        Refinement can change n_ctrl per edge (but not per-bias currently,
        since biases tend to be simpler).

    Attributes:
        input_dim: Feature dimension (default 20 top features).
        edges: List of (i, j) pairs in the KAN graph.
        num_knots: Initial number of knots per spline.
        degree: Spline degree (not used in current linear-interp evaluation,
            but preserved for semantic compatibility with kan.py).
        edge_control_pts: Dict (i,j) -> np.ndarray of control point values.
        bias_control_pts: List of np.ndarray per node.
        _edge_n_ctrl: Dict (i,j) -> current number of control points (changes
            after refinement via knot insertion/removal).
    """

    def __init__(
        self,
        input_dim: int,
        edges: list[tuple[int, int]] | None = None,
        num_knots: int = 8,
        degree: int = 3,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.num_knots = num_knots
        self.degree = degree
        n_params_per_spline = num_knots + degree  # 11 for defaults

        rng_np = np.random.default_rng(seed)

        if edges is None:
            # Fully connected graph over input_dim nodes.
            edges = [
                (i, j)
                for i in range(input_dim)
                for j in range(i + 1, input_dim)
            ]
        self.edges: list[tuple[int, int]] = edges

        # Edge control points — small random init near zero.
        self.edge_control_pts: dict[tuple[int, int], np.ndarray] = {}
        for edge in self.edges:
            self.edge_control_pts[edge] = rng_np.uniform(
                -0.05, 0.05, (n_params_per_spline,)
            ).astype(np.float32)

        # Bias control points (one per node).
        self.bias_control_pts: list[np.ndarray] = []
        for _ in range(input_dim):
            self.bias_control_pts.append(
                rng_np.uniform(-0.05, 0.05, (n_params_per_spline,)).astype(np.float32)
            )

        # Per-edge control point count (starts uniform, changes after refine()).
        self._edge_n_ctrl: dict[tuple[int, int], int] = {
            edge: n_params_per_spline for edge in self.edges
        }

    # ------------------------------------------------------------------
    # Spline evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_spline(x: float, ctrl: np.ndarray) -> float:
        """Evaluate piecewise-linear B-spline at scalar x in [-1, 1].

        Maps x from [-1, 1] linearly to [0, n_ctrl - 1], then interpolates
        between the two nearest control points.

        Args:
            x: Input value, clipped to [-1, 1].
            ctrl: Control point array, shape (n_ctrl,).

        Returns:
            Interpolated spline value (float).
        """
        n_ctrl = len(ctrl)
        scaled = (float(x) + 1.0) / 2.0 * (n_ctrl - 1)
        scaled = max(0.0, min(scaled, n_ctrl - 1.0 - 1e-7))
        left = int(scaled)
        right = min(left + 1, n_ctrl - 1)
        t = scaled - left
        return float(ctrl[left] + t * (ctrl[right] - ctrl[left]))

    @staticmethod
    def _basis_k(x: float, k: int, n_ctrl: int) -> float:
        """Hat basis function: contribution of control point k at input x.

        The piecewise-linear spline is a weighted sum of hat functions.
        basis_k(x) = max(0, 1 - |scaled(x) - k|), where scaled maps x to [0, n_ctrl-1].

        This is the exact gradient of the spline w.r.t. control point k:
            df/d_c_k = basis_k(x)

        Args:
            x: Input scalar in [-1, 1].
            k: Control point index.
            n_ctrl: Total number of control points.

        Returns:
            Basis function value in [0, 1].
        """
        scaled = (float(x) + 1.0) / 2.0 * (n_ctrl - 1)
        scaled = max(0.0, min(scaled, n_ctrl - 1.0 - 1e-7))
        left = int(scaled)
        t = scaled - left
        if k == left:
            return float(1.0 - t)
        elif k == left + 1:
            return float(t)
        return 0.0

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------

    def energy_single(self, x: np.ndarray) -> float:
        """Compute KAN energy for one binary input vector.

        E(x) = sum_ij f_ij(s_i * s_j) + sum_i g_i(s_i)
        where s = 2*x - 1 maps binary {0,1} to spins {-1, +1}.

        The spin transformation aligns the spline domain [-1, 1] with the
        product range: s_i * s_j ∈ [-1, 1] for all binary inputs.

        Args:
            x: Binary feature vector, shape (input_dim,), values in {0, 1}.

        Returns:
            Scalar energy (float).
        """
        spins = 2.0 * x - 1.0
        e = 0.0
        for (i, j), ctrl in self.edge_control_pts.items():
            e += self._eval_spline(float(spins[i] * spins[j]), ctrl)
        for i, ctrl in enumerate(self.bias_control_pts):
            e += self._eval_spline(float(spins[i]), ctrl)
        return e

    def energy_batch(self, xs: np.ndarray) -> np.ndarray:
        """Compute energy for a batch of vectors.

        Args:
            xs: Shape (n, input_dim), binary {0, 1}.

        Returns:
            Energy values, shape (n,), dtype float32.
        """
        return np.array(
            [self.energy_single(xs[i]) for i in range(len(xs))], dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Adaptive mesh refinement
    # ------------------------------------------------------------------

    def compute_edge_curvature(
        self, n_sample: int = 100, h: float = 0.01
    ) -> dict[tuple[int, int], float]:
        """Estimate second-derivative curvature for each edge spline.

        Curvature measures how nonlinear the spline is: a constant or linear
        spline has zero curvature; a spline with sharp bends has high curvature.

        **Method:**
            For 100 uniformly spaced points x in [-0.9, 0.9] (avoid boundaries):
                f''(x) ≈ (f(x+h) - 2*f(x) + f(x-h)) / h²
            curvature_score = mean |f''(x)|

        **Why this matters:**
            High curvature → complex nonlinear constraint (needs more knots).
            Low curvature → near-linear constraint (knots can be merged).

        Args:
            n_sample: Number of sample points per edge (default 100).
            h: Step size for finite differences (default 0.01).

        Returns:
            Dict mapping edge (i, j) to curvature score (float).
        """
        # Sample points avoiding boundary to prevent clipping artifacts.
        x_pts = np.linspace(-0.9, 0.9, n_sample)
        curvatures: dict[tuple[int, int], float] = {}

        for edge, ctrl in self.edge_control_pts.items():
            # Central difference second derivative: (f(x+h) - 2f(x) + f(x-h)) / h²
            d2f_values = []
            for x in x_pts:
                f_plus = self._eval_spline(x + h, ctrl)
                f_center = self._eval_spline(x, ctrl)
                f_minus = self._eval_spline(x - h, ctrl)
                d2f = (f_plus - 2.0 * f_center + f_minus) / (h * h)
                d2f_values.append(abs(d2f))
            curvatures[edge] = float(np.mean(d2f_values))

        return curvatures

    def _insert_knot(
        self,
        edge: tuple[int, int],
        curvatures_per_point: list[float],
        x_pts: np.ndarray,
    ) -> None:
        """Insert one knot into the edge spline at the highest-curvature point.

        The new control point is placed at the scaled position corresponding
        to the x value with maximum |f''(x)|. Its value is set by linear
        interpolation between the neighboring control points (preserving
        the current spline shape as closely as possible).

        After insertion:
            len(ctrl) increases by 1
            The n_ctrl in _edge_n_ctrl is updated accordingly.

        Args:
            edge: The (i, j) edge to refine.
            curvatures_per_point: |f''(x)| values for each sample point.
            x_pts: The sample points corresponding to curvatures_per_point.
        """
        ctrl = self.edge_control_pts[edge]
        n_ctrl = len(ctrl)

        # Find the x with maximum curvature.
        max_idx = int(np.argmax(curvatures_per_point))
        x_star = float(x_pts[max_idx])

        # Find where x_star falls in the current control point spacing.
        scaled = (x_star + 1.0) / 2.0 * (n_ctrl - 1)
        scaled = max(0.0, min(scaled, n_ctrl - 1.0 - 1e-7))
        left = int(scaled)
        t = scaled - left  # fractional position within interval [left, left+1]

        # Interpolated value for the new control point.
        right = min(left + 1, n_ctrl - 1)
        new_val = float(ctrl[left] * (1.0 - t) + ctrl[right] * t)

        # Insert after index `left`: new_ctrl = ctrl[0..left] + [new_val] + ctrl[left+1..]
        # This places the new knot between the existing knots at left and left+1.
        new_ctrl = np.concatenate([
            ctrl[:left + 1],
            np.array([new_val], dtype=np.float32),
            ctrl[left + 1:],
        ])
        self.edge_control_pts[edge] = new_ctrl
        self._edge_n_ctrl[edge] = len(new_ctrl)

    def _remove_knot(self, edge: tuple[int, int]) -> bool:
        """Remove one knot from the edge spline in the lowest-curvature interval.

        Finds the adjacent pair of control points with the smallest difference
        (minimum curvature contribution) and merges them into their average.
        If n_ctrl would drop below degree + 2, does nothing and returns False.

        Args:
            edge: The (i, j) edge to simplify.

        Returns:
            True if a knot was removed, False if the spline was too short.
        """
        ctrl = self.edge_control_pts[edge]
        n_ctrl = len(ctrl)

        # Minimum n_ctrl: degree + 2 (to keep at least 2 distinct intervals).
        min_n_ctrl = self.degree + 2
        if n_ctrl <= min_n_ctrl:
            return False  # Cannot remove further.

        # Find the adjacent pair with minimum absolute difference (most linear part).
        diffs = np.abs(np.diff(ctrl))  # shape (n_ctrl - 1,)
        merge_idx = int(np.argmin(diffs))

        # Replace ctrl[merge_idx] with the average, remove ctrl[merge_idx + 1].
        # The merged interval is wider but the control point value is smoothed.
        merged_val = (ctrl[merge_idx] + ctrl[merge_idx + 1]) / 2.0
        new_ctrl = np.concatenate([
            ctrl[:merge_idx],
            np.array([merged_val], dtype=np.float32),
            ctrl[merge_idx + 2:],
        ])
        self.edge_control_pts[edge] = new_ctrl
        self._edge_n_ctrl[edge] = len(new_ctrl)
        return True

    def refine(
        self, threshold_multiplier: float = 1.5, n_sample: int = 100, h: float = 0.01
    ) -> tuple[int, int]:
        """Adaptively refine the KAN mesh: insert knots in high-curvature edges,
        remove knots in low-curvature edges.

        **Algorithm:**
            1. Compute curvature score per edge (mean |d²f/dx²| over 100 points).
            2. Compute mean_curvature = mean of all scores.
            3. For each edge:
               - If curvature > threshold_multiplier * mean_curvature:
                   Insert one knot at the highest-curvature point.
               - If curvature < mean_curvature / threshold_multiplier:
                   Remove one knot (merge minimum-curvature interval), if n_ctrl > min.
            4. Return (added_knots, removed_knots) counts.

        **Why threshold_multiplier=1.5?**
            A value of 1.5 means we insert knots for edges 50% above average
            curvature and remove for edges 33% below average. This is a
            conservative split that avoids over-fitting while still being
            adaptive. See KAEM (arxiv 2506.14167) for similar thresholds.

        Args:
            threshold_multiplier: Multiplier for high/low curvature thresholds.
            n_sample: Sample points for curvature estimation.
            h: Finite difference step size.

        Returns:
            (added_knots, removed_knots): Number of edges that gained / lost a knot.
        """
        # Step 1: Compute per-edge curvature and per-point curvature for insertion.
        x_pts = np.linspace(-0.9, 0.9, n_sample)
        edge_curvatures: dict[tuple[int, int], float] = {}
        edge_curvature_pts: dict[tuple[int, int], list[float]] = {}

        for edge, ctrl in self.edge_control_pts.items():
            d2f_values = []
            for x in x_pts:
                f_plus = self._eval_spline(x + h, ctrl)
                f_center = self._eval_spline(x, ctrl)
                f_minus = self._eval_spline(x - h, ctrl)
                d2f = (f_plus - 2.0 * f_center + f_minus) / (h * h)
                d2f_values.append(abs(d2f))
            edge_curvatures[edge] = float(np.mean(d2f_values))
            edge_curvature_pts[edge] = d2f_values

        # Step 2: Compute mean curvature across all edges.
        all_curvatures = list(edge_curvatures.values())
        mean_curvature = float(np.mean(all_curvatures))
        high_threshold = threshold_multiplier * mean_curvature
        low_threshold = mean_curvature / threshold_multiplier if threshold_multiplier > 1 else 0.0

        logger.info(
            f"  Curvature stats: mean={mean_curvature:.6f}, "
            f"high_thresh={high_threshold:.6f}, low_thresh={low_threshold:.6f}"
        )

        # Step 3: Insert / remove knots based on thresholds.
        added_knots = 0
        removed_knots = 0

        for edge in self.edges:
            c = edge_curvatures[edge]
            if c > high_threshold:
                self._insert_knot(edge, edge_curvature_pts[edge], x_pts)
                added_knots += 1
            elif c < low_threshold:
                if self._remove_knot(edge):
                    removed_knots += 1

        logger.info(
            f"  Refinement: +{added_knots} knots (high-curvature edges), "
            f"-{removed_knots} knots (low-curvature edges)"
        )

        return added_knots, removed_knots

    # ------------------------------------------------------------------
    # Discriminative contrastive divergence training
    # ------------------------------------------------------------------

    def train_discriminative_cd(
        self,
        correct_vectors: np.ndarray,
        wrong_vectors: np.ndarray,
        n_epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.001,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> list[float]:
        """Discriminative CD: update control points to separate correct/wrong energies.

        **Update rule:**
            For each (correct, wrong) pair in a mini-batch, compute the
            analytical gradient of energy w.r.t. each control point using
            the hat basis function:

                dE(x)/d_c_k = basis_k(z)   where z = input to that spline

            Then apply:
                c_k -= lr * (mean dE_correct/d_c_k - mean dE_wrong/d_c_k)
                     - lr * weight_decay * c_k

            This simultaneously lowers E(correct) and raises E(wrong).

        **Key difference from Exp 109:**
            This version supports per-edge variable n_ctrl (after refinement),
            so gradient accumulators are allocated per-edge based on current
            len(ctrl) rather than a global num_knots + degree.

        Args:
            correct_vectors: Shape (n, input_dim), binary {0,1}.
            wrong_vectors: Shape (n, input_dim), binary {0,1}.
            n_epochs: Number of training epochs.
            lr: Learning rate.
            weight_decay: L2 regularisation on control points.
            batch_size: Mini-batch size.
            verbose: Print epoch stats.

        Returns:
            losses: Mean energy gap (E_wrong - E_correct) per epoch.
                    Want positive and growing → model is separating the classes.
        """
        n = len(correct_vectors)
        rng_np = np.random.default_rng(0)
        losses = []

        for epoch in range(n_epochs):
            perm = rng_np.permutation(n)
            epoch_gaps = []

            for batch_start in range(0, n, batch_size):
                batch_idx = perm[batch_start:batch_start + batch_size]
                b_correct = correct_vectors[batch_idx]
                b_wrong = wrong_vectors[batch_idx]
                bs = len(batch_idx)

                # Gradient accumulators — per-edge size matches current n_ctrl.
                edge_grad_acc: dict[tuple[int, int], np.ndarray] = {
                    edge: np.zeros(len(self.edge_control_pts[edge]), dtype=np.float32)
                    for edge in self.edges
                }
                bias_n_ctrl = self.num_knots + self.degree  # biases keep fixed n_ctrl
                bias_grad_acc: list[np.ndarray] = [
                    np.zeros(len(self.bias_control_pts[i]), dtype=np.float32)
                    for i in range(self.input_dim)
                ]
                gap_sum = 0.0

                for bi in range(bs):
                    xc = b_correct[bi]
                    xw = b_wrong[bi]
                    sc = 2.0 * xc - 1.0  # spins for correct
                    sw = 2.0 * xw - 1.0  # spins for wrong

                    ec = self.energy_single(xc)
                    ew = self.energy_single(xw)
                    gap_sum += (ew - ec)

                    # Edge spline gradients.
                    # We want ctrl -= lr * (grad_correct - grad_wrong).
                    # grad_correct = dE_correct/d_ctrl = [basis_k(z_c)]
                    # grad_wrong   = dE_wrong/d_ctrl   = [basis_k(z_w)]
                    # Combined: accumulate (basis_k(z_c) - basis_k(z_w)).
                    for (i, j) in self.edges:
                        ctrl = self.edge_control_pts[(i, j)]
                        n_ctrl_e = len(ctrl)
                        z_c = float(sc[i] * sc[j])
                        z_w = float(sw[i] * sw[j])
                        for k in range(n_ctrl_e):
                            edge_grad_acc[(i, j)][k] += (
                                self._basis_k(z_c, k, n_ctrl_e)
                                - self._basis_k(z_w, k, n_ctrl_e)
                            )

                    # Bias spline gradients.
                    for i in range(self.input_dim):
                        ctrl_b = self.bias_control_pts[i]
                        n_ctrl_b = len(ctrl_b)
                        z_c_i = float(sc[i])
                        z_w_i = float(sw[i])
                        for k in range(n_ctrl_b):
                            bias_grad_acc[i][k] += (
                                self._basis_k(z_c_i, k, n_ctrl_b)
                                - self._basis_k(z_w_i, k, n_ctrl_b)
                            )

                # Apply gradient update with weight decay.
                for (i, j) in self.edges:
                    self.edge_control_pts[(i, j)] -= lr * (
                        edge_grad_acc[(i, j)] / bs
                        + weight_decay * self.edge_control_pts[(i, j)]
                    )
                for i in range(self.input_dim):
                    self.bias_control_pts[i] -= lr * (
                        bias_grad_acc[i] / bs
                        + weight_decay * self.bias_control_pts[i]
                    )

                epoch_gaps.append(gap_sum / bs)

            mean_gap = float(np.mean(epoch_gaps))
            losses.append(mean_gap)
            if verbose and (epoch % 25 == 0 or epoch == n_epochs - 1):
                logger.info(
                    f"    KAN epoch {epoch:3d}/{n_epochs}: gap={mean_gap:+.4f}"
                )

        return losses

    def param_count(self) -> int:
        """Total number of learnable parameters (control points) across all splines.

        Sums over all edge control point arrays + all bias control point arrays.
        After refinement, edge arrays may have variable length.

        Returns:
            Total parameter count.
        """
        edge_params = sum(len(ctrl) for ctrl in self.edge_control_pts.values())
        bias_params = sum(len(ctrl) for ctrl in self.bias_control_pts)
        return edge_params + bias_params

    def curvature_summary(
        self, curvatures: dict[tuple[int, int], float], top_k: int = 10
    ) -> dict[str, Any]:
        """Summarise curvature distribution for analysis.

        Returns stats and lists of the highest/lowest curvature edges for
        interpretability analysis.

        Args:
            curvatures: Output of compute_edge_curvature().
            top_k: Number of top/bottom edges to report.

        Returns:
            Dict with 'stats', 'top_edges', 'bottom_edges'.
        """
        values = list(curvatures.values())
        sorted_edges = sorted(curvatures.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "stats": {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
            },
            "top_edges": [
                {"edge": list(e), "curvature": float(c)}
                for e, c in sorted_edges[:top_k]
            ],
            "bottom_edges": [
                {"edge": list(e), "curvature": float(c)}
                for e, c in sorted_edges[-top_k:]
            ],
        }


# ---------------------------------------------------------------------------
# Interpretability: map feature indices to semantic labels
# ---------------------------------------------------------------------------

# Feature group labels for the 200-dim encoding.
# Indices 0-19: numeric, 20-59: structural, 60-139: domain-specific, 140-199: consistency.
_FEATURE_GROUP = (
    [(i, "numeric") for i in range(20)]
    + [(i, "structural") for i in range(20, 60)]
    + [(i, "domain_specific") for i in range(60, 140)]
    + [(i, "consistency") for i in range(140, 200)]
)
_FEATURE_GROUP_MAP = {i: g for i, g in _FEATURE_GROUP}


def describe_edge(feat_idx_i: int, feat_idx_j: int) -> str:
    """Describe the semantic content of an edge (i, j) in the projected feature space.

    Args:
        feat_idx_i: Original 200-dim feature index for node i.
        feat_idx_j: Original 200-dim feature index for node j.

    Returns:
        Human-readable description of the edge type.
    """
    gi = _FEATURE_GROUP_MAP.get(feat_idx_i, "unknown")
    gj = _FEATURE_GROUP_MAP.get(feat_idx_j, "unknown")
    if gi == gj:
        return f"{gi} × {gi} (within-group interaction)"
    return f"{gi} × {gj} (cross-group interaction)"


# ---------------------------------------------------------------------------
# Bootstrap CI for AUROC
# ---------------------------------------------------------------------------

def bootstrap_auroc_ci(
    e_correct: np.ndarray,
    e_wrong: np.ndarray,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for AUROC.

    Args:
        e_correct: Energies for correct answers.
        e_wrong: Energies for wrong answers.
        n_bootstrap: Number of bootstrap replicates.
        ci: Confidence level.
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) CI bounds.
    """
    rng_np = np.random.default_rng(seed)
    aurocs = []
    n_c = len(e_correct)
    n_w = len(e_wrong)
    alpha = (1.0 - ci) / 2.0

    for _ in range(n_bootstrap):
        idx_c = rng_np.integers(0, n_c, n_c)
        idx_w = rng_np.integers(0, n_w, n_w)
        aurocs.append(auroc_from_energies(e_correct[idx_c], e_wrong[idx_w]))

    aurocs_arr = np.array(aurocs)
    return float(np.quantile(aurocs_arr, alpha)), float(np.quantile(aurocs_arr, 1.0 - alpha))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 153: KAN Adaptive Mesh Refinement."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 153: KAN Adaptive Mesh Refinement")
    logger.info("  Task: Arithmetic + Logic constraint verification (200 pairs)")
    logger.info("  Method: compute_edge_curvature() + refine(threshold=1.5)")
    logger.info("  Targets: AUROC maintained (delta >= -0.01), params ±20%")
    logger.info("=" * 70)

    start_time = time.time()
    rng = np.random.default_rng(153)  # seed 153 for this experiment

    # -----------------------------------------------------------------------
    # Step 1: Generate 200-question dataset (100 arithmetic + 100 logic)
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 1: Generate 200 question pairs ---")
    t0 = time.time()

    N_PER_DOMAIN = 100  # 100 arithmetic + 100 logic = 200 total
    arith_triples = generate_arithmetic_triples(N_PER_DOMAIN, rng)
    logic_triples = generate_logic_triples(N_PER_DOMAIN, rng)

    logger.info(f"  Arithmetic: {len(arith_triples)} triples")
    logger.info(f"  Logic:      {len(logic_triples)} triples")
    logger.info(f"  ({time.time() - t0:.1f}s)")

    # -----------------------------------------------------------------------
    # Step 2: Encode as 200-dim binary features
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 2: Feature encoding (200-dim binary) ---")
    t0 = time.time()

    all_correct: list[np.ndarray] = []
    all_wrong: list[np.ndarray] = []
    domain_labels: list[str] = []

    for domain, triples in [("arithmetic", arith_triples), ("logic", logic_triples)]:
        for q, c, w in triples:
            all_correct.append(encode_answer(q, c))
            all_wrong.append(encode_answer(q, w))
            domain_labels.append(domain)

    correct_200 = np.stack(all_correct)  # (200, 200)
    wrong_200 = np.stack(all_wrong)      # (200, 200)
    logger.info(f"  Encoded {len(correct_200)} pairs, shape {correct_200.shape}")
    logger.info(f"  ({time.time() - t0:.1f}s)")

    # -----------------------------------------------------------------------
    # Step 3: Train Ising on full 200-dim for feature selection
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 3: Ising feature selection (top-20 from 200-dim) ---")
    t0 = time.time()

    # Split 160 train / 40 test for the full dataset.
    N_TRAIN = 160
    N_TEST = 40
    perm = rng.permutation(200)
    train_idx = perm[:N_TRAIN]
    test_idx = perm[N_TRAIN:]

    biases_ising, J_ising, _ = train_ising_discriminative_cd(
        correct_200[train_idx],
        wrong_200[train_idx],
        n_epochs=200,
        lr=0.05,
        beta=1.0,
        l1_lambda=0.001,
        weight_decay=0.005,
    )

    TOP_K = 20
    top_features = select_top_features(biases_ising, J_ising, TOP_K)
    logger.info(f"  Top-{TOP_K} feature indices: {sorted(top_features)}")
    logger.info(f"  ({time.time() - t0:.1f}s)")

    # -----------------------------------------------------------------------
    # Step 4: Project to top-20 features
    # -----------------------------------------------------------------------
    correct_20 = correct_200[:, top_features]  # (200, 20)
    wrong_20 = wrong_200[:, top_features]       # (200, 20)

    correct_train = correct_20[train_idx]  # (160, 20)
    wrong_train = wrong_20[train_idx]      # (160, 20)
    correct_test = correct_20[test_idx]    # (40, 20)
    wrong_test = wrong_20[test_idx]        # (40, 20)

    logger.info(
        f"\n  Train: {len(correct_train)} pairs | Test: {len(correct_test)} pairs"
    )

    # -----------------------------------------------------------------------
    # Step 5: Train KAN baseline (100 epochs)
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 5: Train KAN baseline (100 epochs) ---")
    t0 = time.time()

    # Fully connected KAN over 20 features: C(20, 2) = 190 edges.
    N_FEATURES = TOP_K  # 20
    NUM_KNOTS = 8
    DEGREE = 3

    kan = KANConstraintModel(
        input_dim=N_FEATURES,
        edges=None,  # fully connected
        num_knots=NUM_KNOTS,
        degree=DEGREE,
        seed=153,
    )

    params_before = kan.param_count()
    n_edges = len(kan.edges)
    logger.info(
        f"  KAN: input_dim={N_FEATURES}, edges={n_edges}, "
        f"num_knots={NUM_KNOTS}, degree={DEGREE}"
    )
    logger.info(f"  Initial param count: {params_before}")
    logger.info(
        f"  (edge params: {n_edges * (NUM_KNOTS + DEGREE)}, "
        f"bias params: {N_FEATURES * (NUM_KNOTS + DEGREE)})"
    )

    losses_baseline = kan.train_discriminative_cd(
        correct_train, wrong_train,
        n_epochs=100,
        lr=0.01,
        weight_decay=0.001,
        batch_size=32,
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Step 6: Evaluate AUROC before refinement
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 6: Evaluate AUROC before refinement ---")
    t0 = time.time()

    e_correct_before = kan.energy_batch(correct_test)
    e_wrong_before = kan.energy_batch(wrong_test)
    auroc_before = auroc_from_energies(e_correct_before, e_wrong_before)
    ci_lo_before, ci_hi_before = bootstrap_auroc_ci(
        e_correct_before, e_wrong_before, n_bootstrap=500, seed=153
    )

    energy_gap_before = float(np.mean(e_wrong_before) - np.mean(e_correct_before))
    logger.info(
        f"  AUROC before: {auroc_before:.4f} [{ci_lo_before:.4f}, {ci_hi_before:.4f}]"
    )
    logger.info(f"  Energy gap before: {energy_gap_before:+.4f}")
    logger.info(f"  Params before: {params_before}")
    logger.info(f"  ({time.time() - t0:.1f}s)")

    # -----------------------------------------------------------------------
    # Step 7: Compute edge curvature and analyse distribution
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 7: Compute edge curvature (|d²f/dx²|) ---")
    t0 = time.time()

    curvatures_before = kan.compute_edge_curvature(n_sample=100, h=0.01)
    curv_summary_before = kan.curvature_summary(curvatures_before, top_k=10)

    logger.info(
        f"  Curvature stats: mean={curv_summary_before['stats']['mean']:.6f}, "
        f"std={curv_summary_before['stats']['std']:.6f}, "
        f"max={curv_summary_before['stats']['max']:.6f}"
    )
    logger.info(f"  Top-5 highest curvature edges:")
    for item in curv_summary_before["top_edges"][:5]:
        i_orig = int(top_features[item["edge"][0]])
        j_orig = int(top_features[item["edge"][1]])
        descr = describe_edge(i_orig, j_orig)
        logger.info(
            f"    edge {item['edge']} (feat {i_orig},{j_orig}): "
            f"curvature={item['curvature']:.6f} | {descr}"
        )
    logger.info(f"  Top-5 lowest curvature edges:")
    for item in curv_summary_before["bottom_edges"][:5]:
        i_orig = int(top_features[item["edge"][0]])
        j_orig = int(top_features[item["edge"][1]])
        descr = describe_edge(i_orig, j_orig)
        logger.info(
            f"    edge {item['edge']} (feat {i_orig},{j_orig}): "
            f"curvature={item['curvature']:.6f} | {descr}"
        )
    logger.info(f"  ({time.time() - t0:.1f}s)")

    # -----------------------------------------------------------------------
    # Step 8: Run adaptive mesh refinement
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 8: Adaptive mesh refinement (threshold=1.5) ---")
    t0 = time.time()

    added_knots, removed_knots = kan.refine(threshold_multiplier=1.5)
    params_after_refine = kan.param_count()

    logger.info(f"  Added knots:   {added_knots} edges gained 1 knot each")
    logger.info(f"  Removed knots: {removed_knots} edges lost 1 knot each")
    logger.info(f"  Params after refinement: {params_after_refine}")
    logger.info(f"  Param delta: {params_after_refine - params_before:+d}")
    logger.info(f"  ({time.time() - t0:.1f}s)")

    # Check per-edge n_ctrl distribution after refinement.
    n_ctrl_distribution = {}
    for edge in kan.edges:
        nc = len(kan.edge_control_pts[edge])
        n_ctrl_distribution[nc] = n_ctrl_distribution.get(nc, 0) + 1
    logger.info(f"  n_ctrl distribution (n_ctrl: count): {n_ctrl_distribution}")

    # -----------------------------------------------------------------------
    # Step 9: Fine-tune for 10 more epochs post-refinement
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 9: Fine-tune 10 epochs post-refinement ---")
    t0 = time.time()

    losses_finetune = kan.train_discriminative_cd(
        correct_train, wrong_train,
        n_epochs=10,
        lr=0.005,  # smaller lr for fine-tuning
        weight_decay=0.001,
        batch_size=32,
        verbose=True,
    )

    params_after = kan.param_count()
    logger.info(f"  Params after fine-tuning: {params_after}")
    logger.info(f"  ({time.time() - t0:.1f}s)")

    # -----------------------------------------------------------------------
    # Step 10: Evaluate AUROC after refinement + fine-tuning
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 10: Evaluate AUROC after refinement ---")
    t0 = time.time()

    e_correct_after = kan.energy_batch(correct_test)
    e_wrong_after = kan.energy_batch(wrong_test)
    auroc_after = auroc_from_energies(e_correct_after, e_wrong_after)
    ci_lo_after, ci_hi_after = bootstrap_auroc_ci(
        e_correct_after, e_wrong_after, n_bootstrap=500, seed=153
    )

    energy_gap_after = float(np.mean(e_wrong_after) - np.mean(e_correct_after))
    logger.info(
        f"  AUROC after:  {auroc_after:.4f} [{ci_lo_after:.4f}, {ci_hi_after:.4f}]"
    )
    logger.info(f"  Energy gap after: {energy_gap_after:+.4f}")
    logger.info(f"  Params after: {params_after}")
    logger.info(f"  ({time.time() - t0:.1f}s)")

    # -----------------------------------------------------------------------
    # Step 11: Post-refinement curvature analysis
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 11: Post-refinement curvature analysis ---")
    curvatures_after = kan.compute_edge_curvature(n_sample=100, h=0.01)
    curv_summary_after = kan.curvature_summary(curvatures_after, top_k=10)

    logger.info(
        f"  Curvature stats after: mean={curv_summary_after['stats']['mean']:.6f}, "
        f"std={curv_summary_after['stats']['std']:.6f}"
    )

    # Classify edges by what happened during refinement.
    mean_curv_before = curv_summary_before["stats"]["mean"]
    high_thresh = 1.5 * mean_curv_before
    low_thresh = mean_curv_before / 1.5

    gained_edges = []
    lost_edges = []
    unchanged_edges = []
    for edge in kan.edges:
        c = curvatures_before.get(edge, 0.0)
        i_orig = int(top_features[edge[0]])
        j_orig = int(top_features[edge[1]])
        descr = describe_edge(i_orig, j_orig)
        entry = {
            "edge": list(edge),
            "feat_indices": [i_orig, j_orig],
            "curvature_before": float(c),
            "curvature_after": float(curvatures_after.get(edge, 0.0)),
            "n_ctrl_after": len(kan.edge_control_pts[edge]),
            "semantics": descr,
        }
        if c > high_thresh:
            gained_edges.append(entry)
        elif c < low_thresh:
            lost_edges.append(entry)
        else:
            unchanged_edges.append(entry)

    logger.info(
        f"  Gained knots: {len(gained_edges)} edges "
        f"(curvature > {high_thresh:.6f})"
    )
    logger.info(
        f"  Lost knots:   {len(lost_edges)} edges "
        f"(curvature < {low_thresh:.6f})"
    )
    logger.info(f"  Unchanged:    {len(unchanged_edges)} edges")

    # Show semantic analysis of high-curvature edges.
    if gained_edges:
        logger.info("  High-curvature (complex nonlinear constraint) edges:")
        for item in sorted(gained_edges, key=lambda x: x["curvature_before"], reverse=True)[:5]:
            logger.info(
                f"    edge {item['edge']} feat({item['feat_indices'][0]},{item['feat_indices'][1]}): "
                f"curv={item['curvature_before']:.6f} | {item['semantics']}"
            )
    if lost_edges:
        logger.info("  Low-curvature (near-linear) edges:")
        for item in sorted(lost_edges, key=lambda x: x["curvature_before"])[:5]:
            logger.info(
                f"    edge {item['edge']} feat({item['feat_indices'][0]},{item['feat_indices'][1]}): "
                f"curv={item['curvature_before']:.6f} | {item['semantics']}"
            )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    auroc_delta = auroc_after - auroc_before
    param_delta_pct = 100.0 * (params_after - params_before) / max(params_before, 1)
    target_auroc_met = auroc_delta >= -0.01
    target_param_met = abs(param_delta_pct) <= 20.0

    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 153 SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  AUROC before refinement:  {auroc_before:.4f}")
    logger.info(f"  AUROC after refinement:   {auroc_after:.4f}")
    logger.info(f"  AUROC delta:              {auroc_delta:+.4f} ({'✓' if target_auroc_met else '✗'} target >= -0.01)")
    logger.info(f"  Params before:            {params_before}")
    logger.info(f"  Params after:             {params_after}")
    logger.info(f"  Param delta:              {params_after - params_before:+d} ({param_delta_pct:+.1f}%)")
    logger.info(f"  Param target met:         {'YES' if target_param_met else 'NO'} (target ±20%)")
    logger.info(f"  Knots added:              {added_knots} edges")
    logger.info(f"  Knots removed:            {removed_knots} edges")
    logger.info(f"  Energy gap before:        {energy_gap_before:+.4f}")
    logger.info(f"  Energy gap after:         {energy_gap_after:+.4f}")

    # -----------------------------------------------------------------------
    # Assemble and save results JSON
    # -----------------------------------------------------------------------
    # Convert any numpy int64/float64 scalars to native Python types for JSON.
    top_features_py = [int(f) for f in top_features]

    results: dict[str, Any] = {
        "experiment": "153_kan_adaptive_mesh_refinement",
        "description": (
            "KAN adaptive mesh refinement: insert/remove knots based on edge "
            "curvature to improve representational efficiency while maintaining AUROC."
        ),
        "config": {
            "input_dim": N_FEATURES,
            "n_edges": n_edges,
            "num_knots_initial": NUM_KNOTS,
            "degree": DEGREE,
            "n_ctrl_initial": NUM_KNOTS + DEGREE,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "n_total_pairs": 200,
            "domains": ["arithmetic", "logic"],
            "top_features": top_features_py,
            "refinement_threshold_multiplier": 1.5,
            "baseline_epochs": 100,
            "finetune_epochs": 10,
            "baseline_lr": 0.01,
            "finetune_lr": 0.005,
            "weight_decay": 0.001,
            "seed": 153,
        },
        "summary": {
            "auroc_before": auroc_before,
            "auroc_before_ci": [ci_lo_before, ci_hi_before],
            "auroc_after": auroc_after,
            "auroc_after_ci": [ci_lo_after, ci_hi_after],
            "auroc_delta": auroc_delta,
            "target_auroc_met": target_auroc_met,
            "params_before": params_before,
            "params_after": params_after,
            "param_delta": params_after - params_before,
            "param_delta_pct": param_delta_pct,
            "target_param_met": target_param_met,
            "added_knots": added_knots,
            "removed_knots": removed_knots,
            "energy_gap_before": energy_gap_before,
            "energy_gap_after": energy_gap_after,
        },
        "curvature_analysis": {
            "before": curv_summary_before,
            "after": curv_summary_after,
            "n_ctrl_distribution_after": {
                str(k): v for k, v in n_ctrl_distribution.items()
            },
            "gained_knots_count": len(gained_edges),
            "lost_knots_count": len(lost_edges),
            "unchanged_count": len(unchanged_edges),
            "gained_edges_sample": gained_edges[:20],
            "lost_edges_sample": lost_edges[:20],
        },
        "training": {
            "baseline_loss_history": losses_baseline,
            "finetune_loss_history": losses_finetune,
        },
        "interpretability": {
            "high_curvature_edges": gained_edges,
            "low_curvature_edges": lost_edges,
            "finding": (
                "High-curvature edges correspond to nonlinear constraint interactions "
                "in the learned feature space. Low-curvature edges are near-linear "
                "and benefit from knot removal to reduce parameters."
            ),
        },
    }

    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "experiment_153_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Total experiment time: {time.time() - start_time:.1f}s")

    # Final verdict.
    if target_auroc_met and target_param_met:
        logger.info(
            "\nCONCLUSION: BOTH TARGETS MET — "
            f"AUROC delta={auroc_delta:+.4f} (>= -0.01), "
            f"param delta={param_delta_pct:+.1f}% (within ±20%)."
        )
    elif target_auroc_met:
        logger.info(
            f"\nCONCLUSION: AUROC target met ({auroc_delta:+.4f}), "
            f"but param delta={param_delta_pct:+.1f}% exceeds ±20%."
        )
    elif target_param_met:
        logger.info(
            f"\nCONCLUSION: Param target met ({param_delta_pct:+.1f}%), "
            f"but AUROC dropped {auroc_delta:+.4f} (below -0.01 threshold)."
        )
    else:
        logger.info(
            f"\nCONCLUSION: Neither target met — "
            f"AUROC delta={auroc_delta:+.4f}, param delta={param_delta_pct:+.1f}%."
        )


if __name__ == "__main__":
    main()
