#!/usr/bin/env python3
"""Experiment 844: JEPA v24b — SVAMP corpus coverage fix + DreamPRM reweighting.

**Researcher summary:**
    Exp 834 (JEPA v24 DG-PRM) achieved auc_arc=0.72 but auc_svamp=0.0 because
    the SVAMP training data in Exp 834 was only 10 correct + 10 incorrect *step
    texts* (not triplets) and the TF-IDF vocabulary had insufficient coverage for
    SVAMP arithmetic phrasing.  DreamPRM per-domain reweighting (arXiv 2505.20241)
    cannot rescue a domain with zero effective training signal.

    This experiment:
    1. Generates 20 new SVAMP triplets (anchor, positive, negative) verified by
       symbolic arithmetic eval.
    2. Builds a combined corpus (Exp 834 GSM8K/HumanEval/ARC steps + 20 SVAMP
       triplets) and asserts all four domains are covered (>= 15 pairs each).
    3. Retrains as JEPA v24b for 250 epochs with updated DreamPRM weights that
       give SVAMP a weight of 8.0 (maximum-deficit domain).
    4. Evaluates per-domain AUC and emits honest_verdict.

**Root cause:**
    The SVAMP domain had only 10 correct + 10 incorrect step texts in Exp 834,
    and the model collapsed to auc_svamp=0.0 — not because of corpus size but
    because the DreamPRM weight was set to 1.5 (low) and the hash-projection
    embedding for SVAMP short arithmetic texts is nearly indistinguishable from
    random noise.  With 20 triplets and weight=8.0 the model receives 5x stronger
    gradient signal specifically for SVAMP pairs.

**Verdict logic:**
    - "jepa_v24b_all_domains_viable": min_domain_auc >= 0.50 AND overall_ood_auc >= 0.65
    - "jepa_v24b_svamp_fixed":        auc_svamp >= 0.40 but min_domain_auc < 0.50
    - "jepa_v24b_svamp_still_collapsed": auc_svamp < 0.40

Spec: REQ-LEARN-010, REQ-LEARN-020, SCENARIO-LEARN-015, SCENARIO-LEARN-020
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 844
TITLE = "JEPA v24b: SVAMP Corpus Coverage Fix + DreamPRM Reweighting"
DELIVERABLE = "results/experiment_844_jepa_v24b_svamp.json"
TRIPLETS_FILE = "results/jepa_v24b_svamp_triplets.json"

EMBED_DIM = 256
HIDDEN1 = 64
HIDDEN2 = 32
N_CORRECTNESS = 1
DOMAIN_NAMES = ["gsm8k", "humaneval", "arc", "svamp"]
N_DOMAINS = len(DOMAIN_NAMES)

# DreamPRM per-domain loss weights for v24b.
# SVAMP receives 8.0 (maximum deficit: auc_svamp=0.0 in Exp 834).
# ARC reduced from 5.0 to 1.3 because it recovered well (auc_arc=0.72).
DREAM_PRM_WEIGHTS_V24B: dict[str, float] = {
    "gsm8k": 1.0,
    "humaneval": 1.2,
    "arc": 1.3,
    "svamp": 8.0,
}

# DG-PRM inference domain weights (carried forward from v24).
DG_PRM_DOMAIN_WEIGHTS: dict[str, float] = {
    "gsm8k": 1.0,
    "humaneval": 1.2,
    "arc": 1.3,
    "svamp": 2.0,
}

DELTA_ENERGY_MIN = 0.5
DELTA_ENERGY_MAX = 3.0

N_EPOCHS = 250
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
VAL_FRACTION = 0.2
TRIPLET_MARGIN = 0.5

# Minimum number of pairs per domain required before training can start.
# This assertion is the whole point of REQ-LEARN-020: if any domain is zero,
# reweighting is mathematically guaranteed to fail (no signal to reweight).
MIN_PAIRS_PER_DOMAIN = 15


# ---------------------------------------------------------------------------
# SVAMP triplets (20 verified arithmetic word problem triplets)
# ---------------------------------------------------------------------------

# Each triplet: anchor (setup), positive (correct answer), negative (wrong answer).
# The positive and negative differ only in the arithmetic result.
# Symbolic verification (step 3 in the task spec) checks Y + Z = W for positive
# and Y + Z != W_error for negative.

SVAMP_TRIPLETS_RAW: list[dict[str, Any]] = [
    {
        "anchor": "There are 14 apples in a basket. 6 more are added.",
        "positive": "Total apples = 14 + 6 = 20.",
        "negative": "Total apples = 14 + 6 = 21.",
        "op": "add", "a": 14, "b": 6, "correct": 20, "wrong": 21,
    },
    {
        "anchor": "A jar has 50 candies. 18 are eaten.",
        "positive": "Remaining candies = 50 - 18 = 32.",
        "negative": "Remaining candies = 50 - 18 = 33.",
        "op": "sub", "a": 50, "b": 18, "correct": 32, "wrong": 33,
    },
    {
        "anchor": "A farmer has 7 rows of corn with 9 plants each.",
        "positive": "Total plants = 7 × 9 = 63.",
        "negative": "Total plants = 7 × 9 = 62.",
        "op": "mul", "a": 7, "b": 9, "correct": 63, "wrong": 62,
    },
    {
        "anchor": "A bag holds 72 oranges. Shared equally among 8 friends.",
        "positive": "Each friend gets 72 ÷ 8 = 9 oranges.",
        "negative": "Each friend gets 72 ÷ 8 = 8 oranges.",
        "op": "div", "a": 72, "b": 8, "correct": 9, "wrong": 8,
    },
    {
        "anchor": "Library has 130 books. 47 are checked out.",
        "positive": "Books remaining = 130 - 47 = 83.",
        "negative": "Books remaining = 130 - 47 = 84.",
        "op": "sub", "a": 130, "b": 47, "correct": 83, "wrong": 84,
    },
    {
        "anchor": "A box contains 6 packs of 12 crayons.",
        "positive": "Total crayons = 6 × 12 = 72.",
        "negative": "Total crayons = 6 × 12 = 70.",
        "op": "mul", "a": 6, "b": 12, "correct": 72, "wrong": 70,
    },
    {
        "anchor": "A child earns $3 per chore. Completes 11 chores.",
        "positive": "Total earnings = 3 × 11 = $33.",
        "negative": "Total earnings = 3 × 11 = $34.",
        "op": "mul", "a": 3, "b": 11, "correct": 33, "wrong": 34,
    },
    {
        "anchor": "A train carries 95 passengers. 28 get off.",
        "positive": "Passengers remaining = 95 - 28 = 67.",
        "negative": "Passengers remaining = 95 - 28 = 66.",
        "op": "sub", "a": 95, "b": 28, "correct": 67, "wrong": 66,
    },
    {
        "anchor": "A pool is filled with 240 litres across 4 equal tanks.",
        "positive": "Each tank holds 240 ÷ 4 = 60 litres.",
        "negative": "Each tank holds 240 ÷ 4 = 58 litres.",
        "op": "div", "a": 240, "b": 4, "correct": 60, "wrong": 58,
    },
    {
        "anchor": "A shop sold 56 items on Monday and 39 items on Tuesday.",
        "positive": "Total items sold = 56 + 39 = 95.",
        "negative": "Total items sold = 56 + 39 = 96.",
        "op": "add", "a": 56, "b": 39, "correct": 95, "wrong": 96,
    },
    {
        "anchor": "A recipe requires 4 cups of flour. Made 5 times.",
        "positive": "Total flour needed = 4 × 5 = 20 cups.",
        "negative": "Total flour needed = 4 × 5 = 18 cups.",
        "op": "mul", "a": 4, "b": 5, "correct": 20, "wrong": 18,
    },
    {
        "anchor": "Tom walks 8 km per day for 6 days.",
        "positive": "Total distance = 8 × 6 = 48 km.",
        "negative": "Total distance = 8 × 6 = 46 km.",
        "op": "mul", "a": 8, "b": 6, "correct": 48, "wrong": 46,
    },
    {
        "anchor": "A shelf had 100 jars. 37 were sold.",
        "positive": "Jars left = 100 - 37 = 63.",
        "negative": "Jars left = 100 - 37 = 62.",
        "op": "sub", "a": 100, "b": 37, "correct": 63, "wrong": 62,
    },
    {
        "anchor": "A class collects 180 stickers shared among 9 students equally.",
        "positive": "Each student gets 180 ÷ 9 = 20 stickers.",
        "negative": "Each student gets 180 ÷ 9 = 19 stickers.",
        "op": "div", "a": 180, "b": 9, "correct": 20, "wrong": 19,
    },
    {
        "anchor": "A garden produces 25 tomatoes on Monday and 38 on Wednesday.",
        "positive": "Total tomatoes = 25 + 38 = 63.",
        "negative": "Total tomatoes = 25 + 38 = 64.",
        "op": "add", "a": 25, "b": 38, "correct": 63, "wrong": 64,
    },
    {
        "anchor": "A bucket holds 15 litres. Fill 7 buckets.",
        "positive": "Total volume = 15 × 7 = 105 litres.",
        "negative": "Total volume = 15 × 7 = 104 litres.",
        "op": "mul", "a": 15, "b": 7, "correct": 105, "wrong": 104,
    },
    {
        "anchor": "A bookshelf has 88 books. 33 are borrowed.",
        "positive": "Books remaining = 88 - 33 = 55.",
        "negative": "Books remaining = 88 - 33 = 54.",
        "op": "sub", "a": 88, "b": 33, "correct": 55, "wrong": 54,
    },
    {
        "anchor": "A group of 5 children each own 13 crayons.",
        "positive": "Total crayons = 5 × 13 = 65.",
        "negative": "Total crayons = 5 × 13 = 63.",
        "op": "mul", "a": 5, "b": 13, "correct": 65, "wrong": 63,
    },
    {
        "anchor": "A store has 144 items in 6 equal sections.",
        "positive": "Each section has 144 ÷ 6 = 24 items.",
        "negative": "Each section has 144 ÷ 6 = 22 items.",
        "op": "div", "a": 144, "b": 6, "correct": 24, "wrong": 22,
    },
    {
        "anchor": "Yesterday 42 visitors came to the zoo. Today 57 visitors came.",
        "positive": "Total visitors = 42 + 57 = 99.",
        "negative": "Total visitors = 42 + 57 = 100.",
        "op": "add", "a": 42, "b": 57, "correct": 99, "wrong": 100,
    },
]

assert len(SVAMP_TRIPLETS_RAW) == 20, "Must have exactly 20 SVAMP triplets"


# ---------------------------------------------------------------------------
# Symbolic verification of SVAMP triplets
# ---------------------------------------------------------------------------

def _eval_op(op: str, a: int, b: int) -> int:
    """Evaluate arithmetic op (add/sub/mul/div) on integers.

    **Why exec-free:** using a lookup table instead of exec() avoids code
    injection risk and keeps the function easily testable.

    Args:
        op: One of 'add', 'sub', 'mul', 'div'.
        a: First operand.
        b: Second operand.

    Returns:
        Integer result of the operation.

    Raises:
        ValueError: If op is unknown or div has remainder.
    """
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        if b == 0:
            raise ValueError("Division by zero")
        if a % b != 0:
            raise ValueError(f"{a} is not evenly divisible by {b}")
        return a // b
    raise ValueError(f"Unknown op: {op!r}")


def verify_and_build_svamp_triplets(
    raw: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Verify 20 SVAMP triplets symbolically and return validated list.

    **What this checks:**
        For each raw triplet entry:
        - anchor + positive is arithmetically consistent: eval(op, a, b) == correct
        - anchor + negative is arithmetically inconsistent: eval(op, a, b) != wrong

    If any triplet fails either check, raises AssertionError with a diagnostic
    message identifying which triplet index failed and why.  This ensures the
    training corpus is clean before it enters the model.

    Args:
        raw: List of raw triplet dicts with keys anchor, positive, negative,
             op, a, b, correct, wrong.

    Returns:
        List of validated dicts with keys: anchor, positive, negative, domain.

    Raises:
        AssertionError: If any triplet fails arithmetic consistency check.

    Spec: REQ-LEARN-020, SCENARIO-LEARN-020
    """
    validated: list[dict[str, Any]] = []
    for i, t in enumerate(raw):
        result = _eval_op(t["op"], t["a"], t["b"])
        assert result == t["correct"], (
            f"SVAMP triplet {i}: anchor+positive inconsistent. "
            f"eval({t['op']}, {t['a']}, {t['b']}) = {result}, expected {t['correct']}"
        )
        assert result != t["wrong"], (
            f"SVAMP triplet {i}: anchor+negative is actually correct! "
            f"eval({t['op']}, {t['a']}, {t['b']}) = {result} == wrong={t['wrong']}"
        )
        validated.append({
            "anchor": t["anchor"],
            "positive": t["positive"],
            "negative": t["negative"],
            "domain": "svamp",
        })
    return validated


# ---------------------------------------------------------------------------
# Corpus from Exp 834 step texts (carried forward verbatim)
# ---------------------------------------------------------------------------

# These are the exact step texts from Exp 834 that produced good AUC for GSM8K,
# HumanEval, and ARC.  SVAMP is replaced entirely by the 20 triplets above.

ARC_CORRECT_STEPS = [
    "If all mammals are warm-blooded and all warm-blooded animals regulate temperature, then all mammals regulate temperature.",
    "If plants need sunlight to grow and sunlight requires clear skies, then plants require clear skies to grow.",
    "If iron conducts electricity and electricity causes heating, then iron causes heating when current flows.",
    "If viruses infect cells and cells are part of organisms, then viruses affect organisms.",
    "All reptiles are cold-blooded. A lizard is a reptile. Therefore, a lizard is cold-blooded.",
    "All prime numbers greater than 2 are odd. 7 is a prime greater than 2. Therefore, 7 is odd.",
    "All acids have pH < 7. Hydrochloric acid is an acid. Therefore, hydrochloric acid has pH < 7.",
    "All gases expand when heated. Oxygen is a gas. Therefore, oxygen expands when heated.",
    "If it rains, the ground gets wet. It is raining. Therefore, the ground is wet.",
    "If a shape has four equal sides and four right angles, it is a square. This shape has four equal sides and four right angles. Therefore, it is a square.",
    "If temperature drops below 0°C, water freezes. Temperature is −5°C. Therefore, water is frozen.",
    "If a number is divisible by 4 it is even. 16 is divisible by 4. Therefore, 16 is even.",
    "If the car has fuel, it can start. The car cannot start. Therefore, the car has no fuel.",
    "If a cell has a nucleus, it is eukaryotic. This cell is not eukaryotic. Therefore, it has no nucleus.",
    "If the experiment is controlled, results are reliable. Results are unreliable. Therefore, the experiment is not controlled.",
    "If all sides of a triangle are equal, it is equilateral. This triangle is not equilateral. Therefore, not all sides are equal.",
    "All metals conduct heat. Copper is a metal. We can conclude copper conducts heat.",
    "All planets orbit a star. Earth is a planet. We can conclude Earth orbits a star.",
    "All chemical reactions obey conservation of mass. Combustion is a chemical reaction. Therefore, combustion obeys conservation of mass.",
    "All living organisms require energy. A bacterium is a living organism. Therefore, a bacterium requires energy.",
]

ARC_INCORRECT_STEPS = [
    "If all mammals are warm-blooded, then all warm-blooded animals must be mammals.",
    "If plants need sunlight, then sunlight exists only to serve plants.",
    "If iron conducts electricity, then anything that heats must be iron.",
    "If viruses infect cells, then all cells must contain viruses.",
    "All reptiles are cold-blooded. A lizard is cold-blooded. Therefore, a lizard must be a reptile.",
    "All prime numbers greater than 2 are odd. 9 is odd. Therefore, 9 must be prime.",
    "All acids have pH < 7. A substance has pH < 7. Therefore, the substance must be an acid.",
    "All gases expand when heated. Oxygen expands. Therefore, oxygen must have been heated.",
    "If it rains, the ground gets wet. It is not raining. Therefore, the ground cannot be wet.",
    "If a shape has four equal sides and four right angles, it is a square. This shape is not a square. Therefore, it cannot have four equal sides.",
    "If temperature drops below 0°C, water freezes. Temperature is 10°C. Therefore, water cannot freeze.",
    "If a number is divisible by 4 it is even. 6 is not divisible by 4. Therefore, 6 is not even.",
    "If the car has fuel, it can start. The car has fuel. Therefore, it will definitely start under all conditions.",
    "If a cell has a nucleus, it is eukaryotic. This cell is eukaryotic. Therefore, it must have a nucleus.",
    "If the experiment is controlled, results are reliable. The experiment is controlled. Therefore, no result can ever be unreliable.",
    "If all sides of a triangle are equal, it is equilateral. This triangle has some equal sides. Therefore, it is equilateral.",
    "All metals conduct heat. Copper conducts heat. Therefore, copper must be a metal and the only substance conducting heat.",
    "All planets orbit a star. Earth orbits the sun. Therefore, anything that orbits must be a planet.",
    "Combustion obeys conservation of mass. Therefore, all mass-conserving processes must be combustion.",
    "A bacterium requires energy. Therefore, all energy-requiring things must be bacteria.",
]

GSM8K_CORRECT_STEPS = [
    "To find the total cost: 3 items × $4 each = $12 total. So the answer is $12.",
    "Mary has 15 apples and gives away 7. Remaining = 15 − 7 = 8 apples.",
    "A recipe needs 2 cups of flour per batch. For 5 batches: 2 × 5 = 10 cups.",
    "Train travels 60 mph for 3 hours. Distance = 60 × 3 = 180 miles.",
    "Store sells 24 items in 4 days equally. Per day = 24 ÷ 4 = 6 items.",
    "Tom saves $5 per week for 8 weeks. Total savings = 5 × 8 = $40.",
    "A box holds 12 oranges. 5 boxes contain 12 × 5 = 60 oranges total.",
    "Temperature drops 3°C per hour for 4 hours. Total drop = 3 × 4 = 12°C.",
    "Class has 30 students, 12 absent. Present = 30 − 12 = 18 students.",
    "Pipe fills 1/6 of a tank per hour. In 6 hours it fills 6 × (1/6) = 1 full tank.",
    "If 4 workers finish in 6 days, 1 worker takes 4 × 6 = 24 days.",
    "Profit = revenue − cost = $150 − $90 = $60.",
    "Area of rectangle = length × width = 8 × 5 = 40 square units.",
    "Speed = distance ÷ time = 120 km ÷ 2 h = 60 km/h.",
    "Discount = 20% of $80 = 0.20 × 80 = $16. Sale price = $80 − $16 = $64.",
    "Perimeter of square = 4 × side = 4 × 7 = 28 units.",
    "Average of 4 numbers: (10 + 20 + 30 + 40) ÷ 4 = 100 ÷ 4 = 25.",
    "Population grows by 5% of 1000 = 50. New total = 1000 + 50 = 1050.",
    "10 pens cost $30 total. Per pen = $30 ÷ 10 = $3 each.",
    "A number doubled and then increased by 3 equals 11. Original = (11 − 3) ÷ 2 = 4.",
]

GSM8K_INCORRECT_STEPS = [
    "To find the total cost: 3 items × $4 each = $8 total. So the answer is $8.",
    "Mary has 15 apples and gives away 7. Remaining = 15 + 7 = 22 apples.",
    "A recipe needs 2 cups of flour per batch. For 5 batches: 2 + 5 = 7 cups.",
    "Train travels 60 mph for 3 hours. Distance = 60 + 3 = 63 miles.",
    "Store sells 24 items in 4 days equally. Per day = 24 × 4 = 96 items.",
    "Tom saves $5 per week for 8 weeks. Total savings = 5 + 8 = $13.",
    "A box holds 12 oranges. 5 boxes contain 12 + 5 = 17 oranges total.",
    "Temperature drops 3°C per hour for 4 hours. Total drop = 3 + 4 = 7°C.",
    "Class has 30 students, 12 absent. Present = 30 × 12 = 360 students.",
    "Pipe fills 1/6 of a tank per hour. In 6 hours it fills 6 + (1/6) ≈ 6.17 tanks.",
    "If 4 workers finish in 6 days, 1 worker takes 6 ÷ 4 = 1.5 days.",
    "Profit = revenue × cost = $150 × $90 = $13500.",
    "Area of rectangle = length + width = 8 + 5 = 13 square units.",
    "Speed = distance × time = 120 km × 2 h = 240 km/h.",
    "Discount = 20% of $80 = 0.20 + 80 = $80.20. Sale price = $80 − $80.20 = −$0.20.",
    "Perimeter of square = 2 × side = 2 × 7 = 14 units.",
    "Average of 4 numbers: (10 + 20 + 30 + 40) × 4 = 400.",
    "Population grows by 5% of 1000 = 50. New total = 1000 − 50 = 950.",
    "10 pens cost $30 total. Per pen = $30 × 10 = $300 each.",
    "A number doubled and then increased by 3 equals 11. Original = (11 + 3) × 2 = 28.",
]

HUMANEVAL_CORRECT_STEPS = [
    "def add(a, b): return a + b  # Correct: returns sum of two numbers.",
    "def is_even(n): return n % 2 == 0  # Correct: True when n divisible by 2.",
    "def max_of_two(a, b): return a if a > b else b  # Correct: returns larger value.",
    "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)  # Correct: recursive factorial.",
    "def reverse_string(s): return s[::-1]  # Correct: reverses string with slice.",
    "def count_vowels(s): return sum(1 for c in s.lower() if c in 'aeiou')  # Correct: counts vowels.",
    "def is_palindrome(s): return s == s[::-1]  # Correct: True when s reads same forwards/backwards.",
    "def sum_list(lst): return sum(lst)  # Correct: sum of all list elements.",
    "def first_element(lst): return lst[0] if lst else None  # Correct: returns first or None.",
    "def square(n): return n * n  # Correct: returns n squared.",
    "def absolute_value(n): return n if n >= 0 else -n  # Correct: returns non-negative magnitude.",
    "def clamp(v, lo, hi): return max(lo, min(hi, v))  # Correct: clamps v to [lo, hi].",
    "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))  # Correct: primality test.",
    "def flatten(lst): return [x for sub in lst for x in sub]  # Correct: flattens one level.",
    "def unique(lst): return list(set(lst))  # Correct: removes duplicates (order not preserved).",
    "def zip_sum(a, b): return [x + y for x, y in zip(a, b)]  # Correct: element-wise sum.",
    "def repeat_str(s, n): return s * n  # Correct: repeats string n times.",
    "def is_sorted(lst): return lst == sorted(lst)  # Correct: True when list is in ascending order.",
    "def count_occurrences(lst, x): return lst.count(x)  # Correct: count of x in lst.",
    "def nth_element(lst, n): return lst[n]  # Correct: returns element at index n.",
]

HUMANEVAL_INCORRECT_STEPS = [
    "def add(a, b): return a - b  # Incorrect: subtracts instead of adding.",
    "def is_even(n): return n % 2 == 1  # Incorrect: True when n is odd, not even.",
    "def max_of_two(a, b): return a if a < b else b  # Incorrect: returns smaller value.",
    "def factorial(n): return n * factorial(n+1)  # Incorrect: infinite recursion, never terminates.",
    "def reverse_string(s): return s  # Incorrect: returns original string unchanged.",
    "def count_vowels(s): return len(s)  # Incorrect: counts all characters, not just vowels.",
    "def is_palindrome(s): return s == s  # Incorrect: always True, not a palindrome check.",
    "def sum_list(lst): return len(lst)  # Incorrect: returns length, not sum.",
    "def first_element(lst): return lst[-1] if lst else None  # Incorrect: returns last element.",
    "def square(n): return n + n  # Incorrect: returns 2n, not n².",
    "def absolute_value(n): return n  # Incorrect: negative numbers remain negative.",
    "def clamp(v, lo, hi): return min(lo, max(hi, v))  # Incorrect: logic is inverted.",
    "def is_prime(n): return n % 2 != 0  # Incorrect: only checks divisibility by 2.",
    "def flatten(lst): return lst  # Incorrect: does not flatten nested lists.",
    "def unique(lst): return lst  # Incorrect: does not remove duplicates.",
    "def zip_sum(a, b): return [x * y for x, y in zip(a, b)]  # Incorrect: multiplies instead of adds.",
    "def repeat_str(s, n): return s + str(n)  # Incorrect: appends n as a string.",
    "def is_sorted(lst): return lst == sorted(lst, reverse=True)  # Incorrect: checks descending order.",
    "def count_occurrences(lst, x): return len(lst)  # Incorrect: returns total length, not count of x.",
    "def nth_element(lst, n): return lst[0]  # Incorrect: always returns first element.",
]


# ---------------------------------------------------------------------------
# Domain coverage assertion (REQ-LEARN-020)
# ---------------------------------------------------------------------------

def assert_domain_coverage(
    n_gsm8k: int,
    n_humaneval: int,
    n_arc: int,
    n_svamp: int,
    min_pairs: int = MIN_PAIRS_PER_DOMAIN,
) -> None:
    """Assert all four domains have sufficient training pairs before training.

    **Why this matters:**
        DreamPRM per-domain loss reweighting (arXiv 2505.20241) multiplies the
        loss for each sample by a domain-specific weight.  If a domain has ZERO
        training samples, the weight is irrelevant — the model never sees any
        gradient signal for that domain.  This is exactly what happened to SVAMP
        in Exp 834 (only 10+10 step texts, which collapsed to auc=0.0).

        By asserting >= MIN_PAIRS_PER_DOMAIN before training, we fail fast with
        a clear diagnostic instead of silently producing a zero-AUC result.

    Args:
        n_gsm8k: Number of labeled pairs in the GSM8K domain.
        n_humaneval: Number of labeled pairs in the HumanEval domain.
        n_arc: Number of labeled pairs in the ARC domain.
        n_svamp: Number of labeled pairs in the SVAMP domain.
        min_pairs: Minimum acceptable pairs per domain. Default 15.

    Raises:
        AssertionError: If any domain is below min_pairs, with a diagnostic
            message naming the domain and the actual count.

    Spec: REQ-LEARN-020, SCENARIO-LEARN-020
    """
    assert n_svamp >= min_pairs, (
        f"SVAMP coverage insufficient: {n_svamp} pairs (need >= {min_pairs}). "
        "DreamPRM cannot improve a domain with zero signal — add SVAMP triplets."
    )
    assert n_arc >= min_pairs, (
        f"ARC coverage insufficient: {n_arc} pairs (need >= {min_pairs})."
    )
    assert n_humaneval >= min_pairs, (
        f"HumanEval coverage insufficient: {n_humaneval} pairs (need >= {min_pairs})."
    )
    assert n_gsm8k >= min_pairs, (
        f"GSM8K coverage insufficient: {n_gsm8k} pairs (need >= {min_pairs})."
    )


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def build_corpus_v24b(svamp_triplets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the v24b corpus: Exp 834 step texts + 20 SVAMP triplets.

    **How SVAMP triplets are converted to pairs:**
        Each triplet provides one correct pair (anchor + positive, label=1)
        and one incorrect pair (anchor + negative, label=0).  This produces
        40 SVAMP labeled pairs (20 correct + 20 incorrect) from 20 triplets,
        exceeding the MIN_PAIRS_PER_DOMAIN=15 threshold by a comfortable margin.

    Args:
        svamp_triplets: Output of verify_and_build_svamp_triplets().

    Returns:
        List of dicts: {text, label, domain, domain_idx}.

    Raises:
        AssertionError: If domain coverage assertion fires.
    """
    pairs: list[dict[str, Any]] = []

    domain_data = [
        ("gsm8k", GSM8K_CORRECT_STEPS, GSM8K_INCORRECT_STEPS),
        ("humaneval", HUMANEVAL_CORRECT_STEPS, HUMANEVAL_INCORRECT_STEPS),
        ("arc", ARC_CORRECT_STEPS, ARC_INCORRECT_STEPS),
    ]

    for domain, corrects, incorrects in domain_data:
        d_idx = DOMAIN_NAMES.index(domain)
        for text in corrects:
            pairs.append({"text": text, "label": 1, "domain": domain, "domain_idx": d_idx})
        for text in incorrects:
            pairs.append({"text": text, "label": 0, "domain": domain, "domain_idx": d_idx})

    # SVAMP: each triplet yields a correct pair and an incorrect pair
    svamp_idx = DOMAIN_NAMES.index("svamp")
    for t in svamp_triplets:
        anchor = t["anchor"]
        pairs.append({
            "text": anchor + " " + t["positive"],
            "label": 1,
            "domain": "svamp",
            "domain_idx": svamp_idx,
        })
        pairs.append({
            "text": anchor + " " + t["negative"],
            "label": 0,
            "domain": "svamp",
            "domain_idx": svamp_idx,
        })

    # Count pairs per domain for assertion
    counts: dict[str, int] = {d: 0 for d in DOMAIN_NAMES}
    for p in pairs:
        counts[p["domain"]] += 1

    assert_domain_coverage(
        n_gsm8k=counts["gsm8k"],
        n_humaneval=counts["humaneval"],
        n_arc=counts["arc"],
        n_svamp=counts["svamp"],
    )

    return pairs


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_text(text: str, dim: int = EMBED_DIM, seed: int = 42) -> np.ndarray:
    """Hash-projection text embedding identical to Exp 834.

    Whitespace tokenisation + random projection matrix indexed by token hash.
    Unit-normalised output of shape (dim,).

    Args:
        text: Input text to embed.
        dim: Output embedding dimension.
        seed: PRNG seed for the projection matrix.

    Returns:
        Float32 numpy array of shape (dim,), unit-normed.
    """
    rng = np.random.RandomState(seed)
    proj = rng.randn(10000, dim).astype(np.float32)
    tokens = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        idx = hash(token) % 10000
        vec += proj[idx]
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Model: dual-head JEPA v24b (identical architecture to v24)
# ---------------------------------------------------------------------------

def _init_v24b_params(key: jax.Array) -> dict[str, jax.Array]:
    """Initialise dual-head MLP: shared trunk + correctness head + domain head.

    Architecture: EMBED_DIM → HIDDEN1 → HIDDEN2 → {1, N_DOMAINS}.

    Args:
        key: JAX PRNG key.

    Returns:
        Dict of JAX float32 arrays: w1, b1, w2, b2, w_corr, b_corr, w_dom, b_dom.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    w1 = jax.random.normal(k1, (EMBED_DIM, HIDDEN1)) * np.sqrt(2.0 / EMBED_DIM)
    w2 = jax.random.normal(k2, (HIDDEN1, HIDDEN2)) * np.sqrt(2.0 / HIDDEN1)
    w_corr = jax.random.normal(k3, (HIDDEN2, N_CORRECTNESS)) * np.sqrt(2.0 / HIDDEN2)
    w_dom = jax.random.normal(k4, (HIDDEN2, N_DOMAINS)) * np.sqrt(2.0 / HIDDEN2)
    return {
        "w1": w1.astype(jnp.float32),
        "b1": jnp.zeros((HIDDEN1,), dtype=jnp.float32),
        "w2": w2.astype(jnp.float32),
        "b2": jnp.zeros((HIDDEN2,), dtype=jnp.float32),
        "w_corr": w_corr.astype(jnp.float32),
        "b_corr": jnp.zeros((N_CORRECTNESS,), dtype=jnp.float32),
        "w_dom": w_dom.astype(jnp.float32),
        "b_dom": jnp.zeros((N_DOMAINS,), dtype=jnp.float32),
    }


def _forward_v24b(
    params: dict[str, jax.Array], x: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Forward pass: embedding → (P(correct), P(domain)).

    Args:
        params: Parameter dict.
        x: Shape (EMBED_DIM,) or (batch, EMBED_DIM).

    Returns:
        Tuple of (corr_prob, dom_prob) with shapes (..., 1) and (..., N_DOMAINS).
    """
    h1 = jax.nn.relu(x @ params["w1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["w2"] + params["b2"])
    corr_logit = h2 @ params["w_corr"] + params["b_corr"]
    dom_logit = h2 @ params["w_dom"] + params["b_dom"]
    return jax.nn.sigmoid(corr_logit), jax.nn.softmax(dom_logit, axis=-1)


def _compute_loss_v24b(
    params: dict[str, jax.Array],
    x_batch: jax.Array,
    y_corr: jax.Array,
    y_dom: jax.Array,
    dream_weights: jax.Array,
    x_pos: jax.Array,
    x_neg: jax.Array,
    delta_weights: jax.Array,
) -> jax.Array:
    """Combined DG-PRM + DreamPRM + ΔEnergy triplet loss (identical to v24).

    Three components:
    1. DreamPRM-weighted BCE correctness loss (forces SVAMP gradient at 8x).
    2. Domain cross-entropy loss (keeps trunk domain-discriminative).
    3. ΔEnergy triplet loss (rewards large energy gaps between correct/incorrect).

    Args:
        params: Model parameters.
        x_batch: Embeddings, shape (batch, EMBED_DIM).
        y_corr: Binary labels, shape (batch, 1).
        y_dom: Domain indices, shape (batch,).
        dream_weights: Per-sample DreamPRM scalar weights, shape (batch,).
        x_pos: Positive anchor embeddings, shape (n_trip, EMBED_DIM).
        x_neg: Negative anchor embeddings, shape (n_trip, EMBED_DIM).
        delta_weights: ΔEnergy weights per triplet, shape (n_trip,).

    Returns:
        Scalar total loss.
    """
    corr_prob, dom_prob = _forward_v24b(params, x_batch)

    corr_loss_per = optax.sigmoid_binary_cross_entropy(
        corr_prob.squeeze(-1), y_corr.squeeze(-1)
    )
    corr_loss = jnp.mean(corr_loss_per * dream_weights)

    dom_log_prob = jnp.log(jnp.clip(dom_prob, 1e-7, 1.0))
    dom_loss = -jnp.mean(dom_log_prob[jnp.arange(len(y_dom)), y_dom])

    corr_pos, _ = _forward_v24b(params, x_pos)
    corr_neg, _ = _forward_v24b(params, x_neg)
    energy_pos = 1.0 - corr_pos.squeeze(-1)
    energy_neg = 1.0 - corr_neg.squeeze(-1)
    triplet_raw = jnp.maximum(0.0, energy_pos - energy_neg + TRIPLET_MARGIN)
    triplet_loss = jnp.mean(triplet_raw * delta_weights)

    return corr_loss + 0.3 * dom_loss + 0.5 * triplet_loss


_grad_loss_v24b = jax.jit(jax.value_and_grad(_compute_loss_v24b))


def _build_triplets_v24b(
    X: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    params: dict[str, jax.Array],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build within-domain (positive, negative) pairs with ΔEnergy weights.

    For each domain, pairs each correct step with a matching incorrect step
    and computes the absolute energy difference as the triplet weight (clamped
    to [DELTA_ENERGY_MIN, DELTA_ENERGY_MAX]).

    Args:
        X: Embeddings array, shape (N, EMBED_DIM).
        labels: Binary labels, shape (N,).
        domains: Domain index per sample, shape (N,).
        params: Current model parameters (used to compute current energy).

    Returns:
        Tuple (x_pos, x_neg, delta_weights) arrays.
    """
    pos_list, neg_list, delta_list = [], [], []
    for d_idx in range(N_DOMAINS):
        pos_mask = (labels == 1) & (domains == d_idx)
        neg_mask = (labels == 0) & (domains == d_idx)
        X_pos = X[pos_mask]
        X_neg = X[neg_mask]
        if len(X_pos) == 0 or len(X_neg) == 0:
            continue
        for i in range(min(len(X_pos), len(X_neg))):
            xp, xn = X_pos[i], X_neg[i]
            cp, _ = _forward_v24b(params, jnp.asarray(xp))
            cn, _ = _forward_v24b(params, jnp.asarray(xn))
            ep = float(1.0 - cp.squeeze())
            en = float(1.0 - cn.squeeze())
            dw = float(np.clip(abs(en - ep), DELTA_ENERGY_MIN, DELTA_ENERGY_MAX))
            pos_list.append(xp)
            neg_list.append(xn)
            delta_list.append(dw)

    if not pos_list:
        dummy = np.zeros((1, EMBED_DIM), dtype=np.float32)
        return dummy, dummy, np.ones(1, dtype=np.float32)

    return (
        np.array(pos_list, dtype=np.float32),
        np.array(neg_list, dtype=np.float32),
        np.array(delta_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_jepa_v24b(
    pairs: list[dict[str, Any]],
    n_epochs: int = N_EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    val_fraction: float = VAL_FRACTION,
    seed: int = 42,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Train JEPA v24b with DG-PRM + DreamPRM (SVAMP weight=8.0) + ΔEnergy.

    Args:
        pairs: Output of build_corpus_v24b().
        n_epochs: Training epochs (default 250 — 50 more than v24).
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        val_fraction: Val split fraction.
        seed: PRNG seed.

    Returns:
        Tuple (final_params, training_log dict).
    """
    rng = np.random.RandomState(seed)

    X = np.array([_embed_text(p["text"]) for p in pairs], dtype=np.float32)
    Y_corr = np.array([[p["label"]] for p in pairs], dtype=np.float32)
    Y_dom = np.array([p["domain_idx"] for p in pairs], dtype=np.int32)
    Y_label_flat = np.array([p["label"] for p in pairs], dtype=np.int32)

    # Stratified split: domain × label combination
    strat_key = Y_dom * 2 + Y_label_flat
    unique_strats = np.unique(strat_key)
    train_idx_list, val_idx_list = [], []
    for s in unique_strats:
        idx = np.where(strat_key == s)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_idx_list.extend(idx[:n_val].tolist())
        train_idx_list.extend(idx[n_val:].tolist())

    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)

    X_tr = X[train_idx]
    Y_corr_tr = Y_corr[train_idx]
    Y_dom_tr = Y_dom[train_idx]
    X_val = X[val_idx]
    Y_corr_val = Y_corr[val_idx]
    Y_dom_val = Y_dom[val_idx]

    # DreamPRM per-sample weights — SVAMP gets 8.0 (maximum deficit correction)
    dream_weights_tr = np.array(
        [DREAM_PRM_WEIGHTS_V24B[DOMAIN_NAMES[d]] for d in Y_dom_tr],
        dtype=np.float32,
    )

    key = jax.random.PRNGKey(seed)
    params = _init_v24b_params(key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def _step(p, state, xb, yc, yd, dw, xpos, xneg, delt):
        loss, grads = _grad_loss_v24b(p, xb, yc, yd, dw, xpos, xneg, delt)
        updates, new_state = optimizer.update(grads, state, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_state, loss

    train_losses: list[float] = []
    val_losses: list[float] = []
    n_tr = len(X_tr)

    # Initialise triplets before first epoch
    label_tr = (Y_corr_tr.squeeze() > 0.5).astype(np.int32)
    x_pos, x_neg, delta_weights = _build_triplets_v24b(X_tr, label_tr, Y_dom_tr, params)

    for epoch in range(n_epochs):
        perm = rng.permutation(n_tr)
        X_sh = X_tr[perm]
        Yc_sh = Y_corr_tr[perm]
        Yd_sh = Y_dom_tr[perm]
        Dw_sh = dream_weights_tr[perm]

        # Rebuild triplets every 50 epochs to track current energy landscape
        if epoch % 50 == 0 and epoch > 0:
            x_pos, x_neg, delta_weights = _build_triplets_v24b(
                X_tr, label_tr, Y_dom_tr, params
            )

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_tr, batch_size):
            end = min(start + batch_size, n_tr)
            xb = jnp.asarray(X_sh[start:end])
            yc = jnp.asarray(Yc_sh[start:end])
            yd = jnp.asarray(Yd_sh[start:end])
            dw = jnp.asarray(Dw_sh[start:end])
            n_trip = len(x_pos)
            t_start = (start // batch_size * batch_size) % n_trip
            t_end = min(t_start + batch_size, n_trip)
            xpos_b = jnp.asarray(x_pos[t_start:t_end])
            xneg_b = jnp.asarray(x_neg[t_start:t_end])
            delt_b = jnp.asarray(delta_weights[t_start:t_end])
            params, opt_state, loss = _step(
                params, opt_state, xb, yc, yd, dw, xpos_b, xneg_b, delt_b
            )
            epoch_loss += float(loss)
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        val_loss_val = float(
            _compute_loss_v24b(
                params,
                jnp.asarray(X_val),
                jnp.asarray(Y_corr_val),
                jnp.asarray(Y_dom_val),
                jnp.ones(len(X_val), dtype=jnp.float32),
                jnp.asarray(x_pos[:min(len(x_pos), len(X_val))]),
                jnp.asarray(x_neg[:min(len(x_neg), len(X_val))]),
                jnp.asarray(delta_weights[:min(len(delta_weights), len(X_val))]),
            )
        )
        val_losses.append(val_loss_val)

    # Evaluate per-domain AUC on validation set
    corr_probs_val, dom_probs_val = _forward_v24b(params, jnp.asarray(X_val))
    corr_probs_np = np.array(corr_probs_val.squeeze(-1))
    dom_probs_np = np.array(dom_probs_val)
    Y_corr_val_flat = Y_corr_val.squeeze(-1)

    predicted_domains = np.argmax(dom_probs_np, axis=-1)
    domain_weight_vec = np.array(
        [DG_PRM_DOMAIN_WEIGHTS[DOMAIN_NAMES[d]] for d in predicted_domains],
        dtype=np.float32,
    )
    adjusted_probs = np.clip(corr_probs_np * domain_weight_vec, 0.0, 1.0)

    auc_per_domain: dict[str, float] = {}
    for d_idx, domain in enumerate(DOMAIN_NAMES):
        mask = Y_dom_val == d_idx
        if mask.sum() == 0:
            auc_per_domain[domain] = 0.5
            continue
        y_true = Y_corr_val_flat[mask]
        y_pred = adjusted_probs[mask]
        if len(np.unique(y_true)) < 2:
            auc_per_domain[domain] = 0.5
            continue
        auc_per_domain[domain] = float(roc_auc_score(y_true, y_pred))

    return params, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "auc_per_domain": auc_per_domain,
        "n_train": int(n_tr),
        "n_val": int(len(X_val)),
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def compute_honest_verdict_v24b(
    auc_gsm8k: float,
    auc_humaneval: float,
    auc_arc: float,
    auc_svamp: float,
) -> str:
    """Map per-domain AUC results to a v24b honest verdict.

    **Logic (priority order):**
        1. jepa_v24b_all_domains_viable: min_domain_auc >= 0.50 AND overall_ood >= 0.65
        2. jepa_v24b_svamp_fixed:        auc_svamp >= 0.40 (progress even if below gate)
        3. jepa_v24b_svamp_still_collapsed: auc_svamp < 0.40

    OOD average excludes GSM8K (in-distribution) — only HumanEval, ARC, SVAMP.

    Args:
        auc_gsm8k: GSM8K domain AUC.
        auc_humaneval: HumanEval domain AUC.
        auc_arc: ARC domain AUC.
        auc_svamp: SVAMP domain AUC.

    Returns:
        One of the three verdict strings above.
    """
    min_domain_auc = min(auc_gsm8k, auc_humaneval, auc_arc, auc_svamp)
    overall_ood_auc = float(np.mean([auc_humaneval, auc_arc, auc_svamp]))
    if min_domain_auc >= 0.50 and overall_ood_auc >= 0.65:
        return "jepa_v24b_all_domains_viable"
    if auc_svamp >= 0.40:
        return "jepa_v24b_svamp_fixed"
    return "jepa_v24b_svamp_still_collapsed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 844: JEPA v24b SVAMP coverage fix and evaluation."""
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # --- Generate and verify SVAMP triplets ---
    with tmpl.phase("svamp_triplet_generation"):
        svamp_triplets = verify_and_build_svamp_triplets(SVAMP_TRIPLETS_RAW)
        triplets_path = Path(_REPO_ROOT) / TRIPLETS_FILE
        triplets_path.parent.mkdir(parents=True, exist_ok=True)
        triplets_path.write_text(json.dumps(svamp_triplets, indent=2))

    # --- Build corpus v24b ---
    with tmpl.phase("corpus_build"):
        corpus = build_corpus_v24b(svamp_triplets)
        domain_counts: dict[str, int] = {d: 0 for d in DOMAIN_NAMES}
        for p in corpus:
            domain_counts[p["domain"]] += 1

    corpus_composition = {
        domain: {
            "total": count,
            "correct": count // 2,
            "incorrect": count // 2,
        }
        for domain, count in domain_counts.items()
    }

    # --- Train JEPA v24b ---
    with tmpl.phase("training", n_epochs=N_EPOCHS, n_pairs=len(corpus)):
        params, train_log = train_jepa_v24b(corpus, n_epochs=N_EPOCHS)
        tmpl.checkpoint_save({"model_trained": True}, step=N_EPOCHS)

    # --- Save checkpoint ---
    checkpoint_path = "results/jepa_v24b_checkpoint/"
    ckpt_dir = Path(_REPO_ROOT) / checkpoint_path
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "params.npz"
    np.savez(str(ckpt_file), **{k: np.array(v) for k, v in params.items()})

    # --- Extract metrics ---
    auc_gsm8k = train_log["auc_per_domain"].get("gsm8k", 0.5)
    auc_humaneval = train_log["auc_per_domain"].get("humaneval", 0.5)
    auc_arc = train_log["auc_per_domain"].get("arc", 0.5)
    auc_svamp = train_log["auc_per_domain"].get("svamp", 0.5)
    overall_ood_auc = float(np.mean([auc_humaneval, auc_arc, auc_svamp]))
    min_domain_auc = float(min(auc_gsm8k, auc_humaneval, auc_arc, auc_svamp))

    verdict = compute_honest_verdict_v24b(auc_gsm8k, auc_humaneval, auc_arc, auc_svamp)

    # --- Write result ---
    artifact = tmpl.build_result(
        {
            "auc_gsm8k": auc_gsm8k,
            "auc_humaneval": auc_humaneval,
            "auc_arc": auc_arc,
            "auc_svamp": auc_svamp,
            "overall_ood_auc": overall_ood_auc,
            "min_domain_auc": min_domain_auc,
            "all_domains_coverage": True,
            "honest_verdict": verdict,
            "checkpoint_path": checkpoint_path,
            "domain_weights_used": DG_PRM_DOMAIN_WEIGHTS,
            "dream_prm_weights_used": DREAM_PRM_WEIGHTS_V24B,
            "corpus_composition": corpus_composition,
            "n_training_pairs": len(corpus),
            "n_train": train_log["n_train"],
            "n_val": train_log["n_val"],
            "n_epochs": N_EPOCHS,
            "final_train_loss": (
                train_log["train_losses"][-1] if train_log["train_losses"] else None
            ),
            "final_val_loss": (
                train_log["val_losses"][-1] if train_log["val_losses"] else None
            ),
            "auc_svamp_v24_baseline": 0.0,
            "auc_arc_v24_baseline": 0.71875,
            "svamp_triplets_file": TRIPLETS_FILE,
        },
        status="success",
        decision_class="verify",
    )

    out_path = Path(_REPO_ROOT) / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
