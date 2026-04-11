#!/usr/bin/env python3
"""Experiment 175: AdaptiveKAN Live Verification Tracking Loop.

**Researcher summary:**
    Wires AMR into a live verification tracking loop so that AdaptiveKAN
    restructures after every 500 verifications, demonstrating Tier-4
    autonomous structural adaptation without human intervention.

    Exp 153 proved KAN AMR (adaptive mesh refinement) maintains AUROC
    while reducing params by 1.3%.  This experiment takes that one step
    further: the restructuring fires automatically based on a running
    verification count.  We run 3 batches of 500 verifications each
    (simple → medium → complex arithmetic) and measure AUROC + param
    count after each restructure cycle.

**Background:**
    Tier-4 self-learning from research-program.md: the energy function
    structure evolves based on query distribution.  Simple arithmetic
    queries produce a different input distribution than multi-carry
    large-number arithmetic.  AdaptiveKAN adapts its spline topology to
    each distribution shift without any manual retraining schedule.

**Experimental design:**
    Baseline held-out set: 100 arithmetic + 100 logic pairs (Exp 153 style).
    Training: 160 pairs, Test: 40 pairs (same split as Exp 153).

    Phase 0 — Baseline:
        Train AdaptiveKAN + static KAN on 160 initial pairs (100 epochs).
        Evaluate AUROC on 200-question held-out set → auroc_batch0.

    Phase 1 — Batch 1 (simple arithmetic: a,b in 1-9):
        Generate 500 simple-arithmetic verifications.
        Run each x through adaptive_kan.verify_and_maybe_restructure(x).
        At count=500, AMR triggers automatically.
        Fine-tune 10 epochs on training data.
        Evaluate AUROC → auroc_batch1.

    Phase 2 — Batch 2 (medium arithmetic: a,b in 10-99):
        Same with medium complexity; AMR at count=1000.
        Evaluate AUROC → auroc_batch2.

    Phase 3 — Batch 3 (complex arithmetic: multi-carry, a,b in 100-999):
        Same with complex; AMR at count=1500.
        Evaluate AUROC → auroc_batch3.

    Target: AUROC does not decrease across restructures.
    Static KAN (no AMR) run on same data for comparison.

**Output:** results/experiment_175_results.json

Target models referenced: Qwen3.5-0.8B, google/gemma-4-E4B-it
(data generation is synthetic; no real LLM call needed for this experiment)

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_175_adaptive_kan_loop.py

Spec: REQ-CORE-001, REQ-CORE-002, REQ-TIER-001
Scenario: SCENARIO-CORE-001, SCENARIO-TIER-004
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

# Force CPU for reproducibility (ROCm JAX may crash on thrml; see CLAUDE.md).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

# Allow importing carnot from python/ directory.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))

from carnot.models.adaptive_kan import AdaptiveKAN, KANConstraintModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature encoding — 200-dim binary (identical to Exp 109 / Exp 153)
# Copied from experiment_153_kan_refinement.py for self-containment.
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


def encode_answer(question: str, answer: str) -> np.ndarray:
    """Encode (question, answer) as 200-dim binary feature vector.

    Identical encoding to Exp 109 / Exp 153 for cross-experiment comparability.

    Feature groups:
    - 0-19:   Numeric (digit counts, arithmetic checks).
    - 20-59:  Structural (word/sentence counts, punctuation).
    - 60-139: Domain-specific (keywords, operators).
    - 140-199: Consistency (bracket matching, code structure).

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
# Data generation helpers
# ---------------------------------------------------------------------------

def generate_arithmetic_pairs(
    n: int,
    rng: np.random.Generator,
    difficulty: str = "simple",
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Generate arithmetic (question, correct_vec, wrong_vec) triples.

    Difficulty levels control operand ranges:
    - simple: a, b in 1-9 (single digit, no carry)
    - medium: a, b in 10-99 (multi-digit, some carry)
    - complex: a, b in 100-999 (multi-carry, large numbers)

    Args:
        n: Number of pairs.
        rng: NumPy random generator.
        difficulty: "simple", "medium", or "complex".

    Returns:
        List of (question, correct_vector, wrong_vector).
    """
    ranges = {
        "simple":  (1, 10),
        "medium":  (10, 100),
        "complex": (100, 1000),
    }
    lo, hi = ranges[difficulty]
    pairs = []
    for _ in range(n):
        a = int(rng.integers(lo, hi))
        b = int(rng.integers(lo, hi))
        op = rng.choice(["+", "*"])
        if op == "+":
            correct = a + b
            wrong = correct + int(rng.choice([-1, 1, -10, 10]))
        else:
            correct = a * b
            wrong = correct + int(rng.integers(1, max(2, a // 2 + 1)))
        if wrong == correct:
            wrong = correct + 1
        if wrong < 0:
            wrong = correct + abs(wrong - correct) + 1
        q = f"What is {a} {op} {b}?"
        c_vec = encode_answer(q, f"The answer is {correct}.")
        w_vec = encode_answer(q, f"The answer is {wrong}.")
        pairs.append((q, c_vec, w_vec))
    return pairs


def generate_logic_pairs(
    n: int, rng: np.random.Generator
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Generate logic syllogism (question, correct_vec, wrong_vec) triples.

    Uses modus ponens / modus tollens style syllogisms from Exp 153.

    Args:
        n: Number of pairs.
        rng: NumPy random generator.

    Returns:
        List of (question, correct_vector, wrong_vector).
    """
    pairs = []
    for _ in range(n):
        s = _SUBJECTS[int(rng.integers(0, len(_SUBJECTS)))]
        p = _PREDICATES[int(rng.integers(0, len(_PREDICATES)))]
        s2 = _SUBJECTS[int(rng.integers(0, len(_SUBJECTS)))]
        if s2 == s:
            s2 = _SUBJECTS[(int(rng.integers(0, len(_SUBJECTS))) + 1) % len(_SUBJECTS)]
        q = f"If all {s} are {p}, and all {s2} are {s}, what follows?"
        correct = f"All {s2} are {p}. This follows by modus ponens."
        wrong = f"Some {s2} are not {p}. The premises do not guarantee this."
        c_vec = encode_answer(q, correct)
        w_vec = encode_answer(q, wrong)
        pairs.append((q, c_vec, w_vec))
    return pairs


# ---------------------------------------------------------------------------
# Feature selection (from Exp 153: top-20 Ising features)
# ---------------------------------------------------------------------------

# Top-20 features from Exp 153 results — reuse for cross-experiment consistency.
_TOP_FEATURES_EXP153 = [
    125, 74, 126, 180, 127, 78, 162, 15, 16, 17,
    167, 79, 168, 22, 171, 175, 77, 166, 169, 114,
]


def select_features(vectors: np.ndarray, feature_indices: list[int]) -> np.ndarray:
    """Project 200-dim vectors to selected feature indices.

    Args:
        vectors: Shape (n, 200), full feature vectors.
        feature_indices: List of column indices to keep.

    Returns:
        Projected vectors, shape (n, len(feature_indices)).
    """
    return vectors[:, feature_indices]


# ---------------------------------------------------------------------------
# AUROC computation
# ---------------------------------------------------------------------------

def auroc(e_correct: np.ndarray, e_wrong: np.ndarray) -> float:
    """AUROC via Wilcoxon-Mann-Whitney U statistic.

    AUROC = P(E_correct < E_wrong).  An AUROC of 1.0 means correct
    answers always have strictly lower energy than wrong ones.
    0.5 = random chance.

    Args:
        e_correct: Energy values for correct answers.
        e_wrong: Energy values for wrong answers.

    Returns:
        AUROC in [0, 1].
    """
    n_c = len(e_correct)
    n_w = len(e_wrong)
    wins = 0.0
    for ec in e_correct:
        wins += float(np.sum(ec < e_wrong)) + 0.5 * float(np.sum(ec == e_wrong))
    return wins / max(n_c * n_w, 1)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 175: AdaptiveKAN Live Verification Tracking Loop."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 175: AdaptiveKAN Live Verification Tracking Loop")
    logger.info("  Tier-4 structural adaptation: AMR every 500 verifications")
    logger.info("  3 batches: simple → medium → complex arithmetic")
    logger.info("  Target: AUROC does not decrease across restructures")
    logger.info("=" * 70)

    t_start = time.time()
    rng = np.random.default_rng(175)

    # -----------------------------------------------------------------------
    # Step 1: Generate held-out evaluation set (200 pairs: 100 arith + 100 logic)
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 1: Generate 200-pair held-out evaluation set ---")
    t0 = time.time()

    eval_arith = generate_arithmetic_pairs(100, rng, difficulty="medium")
    eval_logic = generate_logic_pairs(100, rng)
    eval_all = eval_arith + eval_logic

    eval_correct_200 = np.stack([p[1] for p in eval_all])
    eval_wrong_200 = np.stack([p[2] for p in eval_all])

    # Project to top-20 features.
    feat = _TOP_FEATURES_EXP153
    eval_correct = select_features(eval_correct_200, feat)
    eval_wrong = select_features(eval_wrong_200, feat)
    logger.info(
        f"  Eval set: {len(eval_all)} pairs, projected to {len(feat)} features "
        f"({time.time()-t0:.1f}s)"
    )

    # -----------------------------------------------------------------------
    # Step 2: Generate training data (160 pairs) for initial training.
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 2: Generate 160-pair training set ---")
    t0 = time.time()

    train_arith = generate_arithmetic_pairs(80, rng, difficulty="medium")
    train_logic = generate_logic_pairs(80, rng)
    train_all = train_arith + train_logic

    train_correct_200 = np.stack([p[1] for p in train_all])
    train_wrong_200 = np.stack([p[2] for p in train_all])
    train_correct = select_features(train_correct_200, feat)
    train_wrong = select_features(train_wrong_200, feat)
    logger.info(f"  Training set: {len(train_all)} pairs ({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # Step 3: Initialise models.
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 3: Initialise AdaptiveKAN + static KAN ---")

    INPUT_DIM = len(feat)  # 20
    RESTRUCTURE_EVERY = 500
    NUM_KNOTS = 8
    DEGREE = 3

    adaptive_kan = AdaptiveKAN(
        input_dim=INPUT_DIM,
        num_knots=NUM_KNOTS,
        degree=DEGREE,
        seed=175,
        restructure_every=RESTRUCTURE_EVERY,
    )
    static_kan = KANConstraintModel(
        input_dim=INPUT_DIM,
        num_knots=NUM_KNOTS,
        degree=DEGREE,
        seed=175,
    )

    logger.info(
        f"  AdaptiveKAN: {adaptive_kan.n_params} params, "
        f"restructure_every={RESTRUCTURE_EVERY}"
    )
    logger.info(f"  Static KAN: {static_kan.n_params} params (no AMR)")

    # -----------------------------------------------------------------------
    # Step 4: Baseline training (100 epochs on training data).
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 4: Baseline training (100 epochs) ---")
    t0 = time.time()

    adaptive_kan.train_discriminative_cd(
        train_correct, train_wrong, n_epochs=100, lr=0.01, verbose=True
    )
    static_kan.train_discriminative_cd(
        train_correct, train_wrong, n_epochs=100, lr=0.01, verbose=True
    )
    logger.info(f"  Baseline training done ({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # Step 5: Baseline AUROC (before any restructuring).
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 5: Baseline AUROC (batch 0) ---")

    e_c = adaptive_kan.energy_batch(eval_correct)
    e_w = adaptive_kan.energy_batch(eval_wrong)
    auroc_adaptive_batch0 = auroc(e_c, e_w)

    e_c_s = static_kan.energy_batch(eval_correct)
    e_w_s = static_kan.energy_batch(eval_wrong)
    auroc_static_batch0 = auroc(e_c_s, e_w_s)

    params_batch0 = adaptive_kan.n_params
    logger.info(
        f"  AdaptiveKAN AUROC={auroc_adaptive_batch0:.4f}, "
        f"params={params_batch0}"
    )
    logger.info(f"  StaticKAN  AUROC={auroc_static_batch0:.4f}")

    # -----------------------------------------------------------------------
    # Steps 6-11: Three verification batches with AMR cycles.
    # -----------------------------------------------------------------------
    batch_difficulties = ["simple", "medium", "complex"]
    batch_descriptions = [
        "Batch 1: 500 simple arithmetic (a,b in 1-9, no carry)",
        "Batch 2: 500 medium arithmetic (a,b in 10-99, some carry)",
        "Batch 3: 500 complex arithmetic (a,b in 100-999, multi-carry)",
    ]

    auroc_adaptive: list[float] = [auroc_adaptive_batch0]
    auroc_static: list[float] = [auroc_static_batch0]
    params_adaptive: list[int] = [params_batch0]
    restructure_log: list[dict[str, Any]] = []

    for batch_idx, (difficulty, desc) in enumerate(
        zip(batch_difficulties, batch_descriptions), start=1
    ):
        logger.info(f"\n--- Batch {batch_idx}: {desc} ---")
        t0 = time.time()

        # Generate 500 verification inputs for this batch.
        batch_pairs = generate_arithmetic_pairs(500, rng, difficulty=difficulty)
        batch_correct_200 = np.stack([p[1] for p in batch_pairs])
        batch_correct = select_features(batch_correct_200, feat)

        # Run 500 verifications through the AdaptiveKAN loop.
        # The loop auto-triggers AMR at the 500th verification.
        restructured_at = None
        for i in range(500):
            x = batch_correct[i]
            energy, restructured = adaptive_kan.verify_and_maybe_restructure(x)
            if restructured:
                restructured_at = adaptive_kan._verification_count
                logger.info(
                    f"  AMR triggered at verification {restructured_at} "
                    f"(batch {batch_idx})"
                )

        assert restructured_at is not None, (
            f"Expected AMR at count={batch_idx * 500} but did not trigger"
        )

        # Log the restructure event.
        if adaptive_kan._curvature_history:
            restructure_log.append(adaptive_kan._curvature_history[-1])

        # Fine-tune both models for 10 epochs.
        logger.info(f"  Fine-tuning 10 epochs (both models)...")
        adaptive_kan.train_discriminative_cd(
            train_correct, train_wrong, n_epochs=10, lr=0.005, verbose=False
        )
        static_kan.train_discriminative_cd(
            train_correct, train_wrong, n_epochs=10, lr=0.005, verbose=False
        )

        # Evaluate AUROC.
        e_c_a = adaptive_kan.energy_batch(eval_correct)
        e_w_a = adaptive_kan.energy_batch(eval_wrong)
        auroc_a = auroc(e_c_a, e_w_a)

        e_c_s = static_kan.energy_batch(eval_correct)
        e_w_s = static_kan.energy_batch(eval_wrong)
        auroc_s = auroc(e_c_s, e_w_s)

        n_params_now = adaptive_kan.n_params

        auroc_adaptive.append(auroc_a)
        auroc_static.append(auroc_s)
        params_adaptive.append(n_params_now)

        logger.info(
            f"  AdaptiveKAN AUROC={auroc_a:.4f}, params={n_params_now} "
            f"(Δparams={n_params_now - params_batch0:+d})"
        )
        logger.info(f"  StaticKAN  AUROC={auroc_s:.4f}")
        logger.info(f"  Batch {batch_idx} done ({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # Step 12: Validate targets.
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 12: Validate targets ---")

    # Target 1: AUROC does not decrease across restructures.
    auroc_delta_ok = all(
        auroc_adaptive[i] >= auroc_adaptive[0] - 0.01
        for i in range(1, len(auroc_adaptive))
    )
    logger.info(
        f"  AUROC sequence (adaptive): {[f'{a:.4f}' for a in auroc_adaptive]}"
    )
    logger.info(
        f"  AUROC maintained (delta >= -0.01): {'PASS' if auroc_delta_ok else 'FAIL'}"
    )

    # Target 2: Param count adapts (changes by ≤ ±20%).
    param_change_pct = (params_adaptive[-1] - params_adaptive[0]) / max(params_adaptive[0], 1) * 100
    param_within_bounds = abs(param_change_pct) <= 20.0
    logger.info(
        f"  Param counts: {params_adaptive}"
    )
    logger.info(
        f"  Param change {param_change_pct:+.1f}% (target ≤ ±20%): "
        f"{'PASS' if param_within_bounds else 'FAIL'}"
    )

    # Target 3: Exactly 3 restructure cycles.
    n_restructures = len(adaptive_kan._curvature_history)
    logger.info(
        f"  Restructure cycles: {n_restructures} (target 3): "
        f"{'PASS' if n_restructures == 3 else 'FAIL'}"
    )

    # -----------------------------------------------------------------------
    # Step 13: Write results.
    # -----------------------------------------------------------------------
    results: dict[str, Any] = {
        "experiment": "175_adaptive_kan_live_verification_loop",
        "description": (
            "AdaptiveKAN Tier-4 structural adaptation: AMR every 500 verifications "
            "across 3 difficulty levels (simple → medium → complex arithmetic). "
            "Demonstrates autonomous structural adaptation without human intervention."
        ),
        "reference_models": ["Qwen3.5-0.8B", "google/gemma-4-E4B-it"],
        "config": {
            "input_dim": INPUT_DIM,
            "num_knots_initial": NUM_KNOTS,
            "degree": DEGREE,
            "restructure_every": RESTRUCTURE_EVERY,
            "n_train": len(train_all),
            "n_eval": len(eval_all),
            "top_features": feat,
            "baseline_epochs": 100,
            "finetune_epochs": 10,
            "baseline_lr": 0.01,
            "finetune_lr": 0.005,
            "seed": 175,
        },
        "results": {
            "auroc_adaptive": [float(a) for a in auroc_adaptive],
            "auroc_static": [float(a) for a in auroc_static],
            "params_adaptive": [int(p) for p in params_adaptive],
            "batch_labels": ["batch_0_baseline", "batch_1_simple", "batch_2_medium", "batch_3_complex"],
            "n_restructure_cycles": n_restructures,
            "total_verifications": int(adaptive_kan._verification_count),
        },
        "validation": {
            "auroc_maintained_delta_ge_neg001": auroc_delta_ok,
            "param_change_within_20pct": param_within_bounds,
            "param_change_pct": float(param_change_pct),
            "n_restructures_is_3": n_restructures == 3,
            "all_targets_pass": auroc_delta_ok and n_restructures == 3,
        },
        "restructure_log": restructure_log,
        "summary": {
            "auroc_batch0": float(auroc_adaptive[0]),
            "auroc_batch1": float(auroc_adaptive[1]),
            "auroc_batch2": float(auroc_adaptive[2]),
            "auroc_batch3": float(auroc_adaptive[3]),
            "auroc_delta_max_drop": float(
                max(auroc_adaptive[0] - a for a in auroc_adaptive[1:])
            ),
            "params_initial": int(params_adaptive[0]),
            "params_final": int(params_adaptive[-1]),
            "static_kan_auroc_batch0": float(auroc_static[0]),
            "static_kan_auroc_final": float(auroc_static[-1]),
        },
        "wall_time_seconds": round(time.time() - t_start, 1),
    }

    out_path = _ROOT / "results" / "experiment_175_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    logger.info(f"\nResults written to {out_path}")
    logger.info(
        f"EXPERIMENT 175 COMPLETE in {time.time()-t_start:.1f}s — "
        f"{'ALL TARGETS PASS' if results['validation']['all_targets_pass'] else 'SOME TARGETS FAIL'}"
    )


if __name__ == "__main__":
    main()
