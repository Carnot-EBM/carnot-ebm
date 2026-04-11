#!/usr/bin/env python3
"""Experiment 171 — Combined Signal Benchmark: All Four Hallucination Detectors.

**Researcher summary:**
    Benchmarks five pipeline configurations across 200 multi-domain questions
    (50 arithmetic, 50 code, 50 logic, 50 factual) to determine whether
    combining Ising constraint verification, spilled energy (Exp 157), lookahead
    energy (Exp 169), and FactualExtractor (Exp 158) improves over any single
    signal alone.

    Five configurations:
        Config 1 — Baseline:            No verification; always predicts
                                         "not hallucinated."
        Config 2 — Ising only:          ArithmeticExtractor + CodeExtractor +
                                         LogicExtractor, no logits.
        Config 3 — Spilled + Ising:     Config 2 + SpilledEnergyExtractor
                                         with simulated logits.
        Config 4 — Lookahead + Ising:   Config 2 + LookaheadEnergyExtractor
                                         with simulated logits.
        Config 5 — All combined:        Config 4 + LookaheadEnergyExtractor
                                         + SpilledEnergyExtractor +
                                         FactualExtractor for factual domain.

    Key questions answered:
        - Does Config 5 beat Config 2 (Ising only)?
        - How much latency do energy signals add?
        - Which domain benefits most from which signal?

**Question generators (200 total):**
    Arithmetic (50):
        Correct (25): "X + Y = Z" with verified correct sums.
        Incorrect (25): Off-by-one or wrong-operation errors.

    Code (50):
        Correct (25): Python functions with consistent type annotations and
                      valid return types.
        Incorrect (25): Functions with mismatched return type annotations or
                       uninitialized variable references (CodeExtractor catches
                       these via AST analysis).

    Logic (50):
        Correct (25): Valid logical statements (consistent if-then chains,
                      non-contradictory exclusions).
        Incorrect (25): Contradictory logical claims or unsatisfied implications.

    Factual (50):
        Easy/correct (25): Well-known factual claims (model confident → low
                           energy → likely correct).
        Hard/incorrect (25): Obscure or plausible-but-wrong claims (model
                             uncertain → high energy → likely hallucinated).

**Signal scoring logic (per config):**
    A question is classified as "hallucinated" when:
        - Config 1 (baseline): never (always says "not hallucinated")
        - Config 2 (Ising only): any Ising constraint is violated
          (ArithmeticExtractor, CodeExtractor, LogicExtractor)
        - Config 3 (Spilled + Ising): Config 2 OR spilled energy > threshold
        - Config 4 (Lookahead + Ising): Config 2 OR lookahead energy > threshold
        - Config 5 (All): Config 4 OR factual signal violation for factual domain

**Simulation methodology:**
    Logits are simulated (CARNOT_SKIP_LLM=1 and no LLM available):
        - Correct/easy: logits = Gaussian noise + peak logit of 8.0 at token 0
          → low NLL → low spilled/lookahead energy → constraint satisfied
        - Incorrect/hard: logits = Gaussian noise, std=0.5, no peak
          → high NLL → high spilled/lookahead energy → constraint violated
    This matches the Exp 157/169/170 simulation calibration.

Run:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_171_combined_signal_benchmark.py

Output:
    Prints per-domain accuracy, precision, recall, F1, latency for each config.
    Saves results to results/experiment_171_combined_results.json.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import carnot.* without installation.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.pipeline.extract import (
    ArithmeticExtractor,
    CodeExtractor,
    LogicExtractor,
    AutoExtractor,
)
from carnot.pipeline.spilled_energy import (
    DEFAULT_SPILLED_THRESHOLD,
    SpilledEnergyExtractor,
)
from carnot.pipeline.lookahead_energy import (
    DEFAULT_LOOKAHEAD_THRESHOLD,
    LookaheadEnergyExtractor,
)
from carnot.pipeline.factual_extractor import FactualExtractor

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

N_PER_DOMAIN = 50           # Questions per domain (half correct, half wrong)
N_CORRECT_PER_DOMAIN = 25   # "Correct" (non-hallucinated) samples per domain
N_WRONG_PER_DOMAIN = 25     # "Hallucinated" samples per domain
VOCAB_SIZE = 1000           # Simulated vocabulary size for logit generation
N_TOKENS = 20               # Number of tokens per simulated generation
CORRECT_PEAK_LOGIT = 8.0    # Logit advantage for "correct" simulated outputs
WRONG_NOISE_STD = 0.5       # Std of Gaussian noise for "wrong" simulated outputs
BASE_SEED = 171             # RNG seed for reproducibility

TARGET_MODELS = ["Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it"]

# ---------------------------------------------------------------------------
# Question dataset structures
# ---------------------------------------------------------------------------


@dataclass
class Question:
    """One benchmark question with ground-truth hallucination label.

    Attributes:
        domain: One of "arithmetic", "code", "logic", "factual".
        text: The LLM-style response text being evaluated.
        is_hallucinated: True if this response contains a hallucination or
            error (the signal should flag it as incorrect). False if the
            response is correct (the signal should pass it).
        question_id: Integer index within the full 200-question dataset.
    """

    domain: str
    text: str
    is_hallucinated: bool
    question_id: int


# ---------------------------------------------------------------------------
# Question generators — produce structured correct / incorrect responses
# ---------------------------------------------------------------------------


def _generate_arithmetic_questions() -> list[Question]:
    """Generate 50 arithmetic questions: 25 correct, 25 incorrect.

    **Detailed explanation for engineers:**
        Each question is a natural-language response that includes an explicit
        arithmetic equation "A + B = Z" or "A - B = Z". The ArithmeticExtractor
        in the pipeline scans for exactly this pattern using regex (see extract.py
        line ~155). For correct questions we embed a verified sum; for incorrect
        questions we embed an off-by-one or wrong-operation error.

        The responses are phrased as an LLM answer so the domain feels realistic:
        "To solve 47 + 28: 47 + 28 = 75. The answer is 75."

    Returns:
        List of 50 Question objects (25 correct, 25 wrong).
    """
    questions: list[Question] = []

    # Correct arithmetic: small positive integers, verified sum.
    correct_pairs = [
        (47, 28), (15, 37), (83, 14), (29, 56), (72, 18),
        (61, 39), (44, 53), (90, 11), (35, 65), (48, 32),
        (17, 63), (55, 45), (82, 19), (26, 74), (93, 7),
        (38, 62), (71, 29), (14, 86), (50, 50), (67, 33),
        (12, 88), (45, 55), (78, 22), (31, 69), (66, 34),
    ]
    for i, (a, b) in enumerate(correct_pairs):
        correct_sum = a + b
        text = (
            f"To solve {a} + {b}: "
            f"{a} + {b} = {correct_sum}. The answer is {correct_sum}."
        )
        questions.append(
            Question(
                domain="arithmetic",
                text=text,
                is_hallucinated=False,
                question_id=i,
            )
        )

    # Incorrect arithmetic: off-by-one errors (common LLM error pattern).
    wrong_pairs = [
        (53, 37), (22, 48), (77, 13), (64, 36), (41, 59),
        (86, 14), (33, 67), (95, 5), (18, 82), (73, 27),
        (40, 60), (57, 43), (24, 76), (89, 11), (16, 84),
        (69, 31), (52, 48), (37, 63), (81, 19), (25, 75),
        (44, 56), (68, 32), (79, 21), (36, 64), (91, 9),
    ]
    for i, (a, b) in enumerate(wrong_pairs):
        wrong_sum = a + b + 1  # off-by-one error
        text = (
            f"To solve {a} + {b}: "
            f"{a} + {b} = {wrong_sum}. The answer is {wrong_sum}."
        )
        questions.append(
            Question(
                domain="arithmetic",
                text=text,
                is_hallucinated=True,
                question_id=25 + i,
            )
        )

    return questions


def _generate_code_questions() -> list[Question]:
    """Generate 50 code questions: 25 correct, 25 incorrect.

    **Detailed explanation for engineers:**
        Each question is a Python function definition wrapped in a fenced code
        block (```python ... ```), as an LLM might produce. The CodeExtractor
        (extract.py) parses each block via the Python AST and checks:
            1. Parameter type annotations present.
            2. Return type consistent with any literal return value.
            3. No uninitialized variables used.

        Correct functions: consistent annotations, no uninitialized vars.
        Incorrect functions: use an uninitialized variable (CodeExtractor
        flags these via _extract_initialization). We use uninitialized-var
        bugs rather than syntax errors because AST parsing fails on syntax
        errors and returns no constraints; uninitialized vars parse correctly
        but produce constraint violations with satisfied=False.

    Returns:
        List of 50 Question objects (25 correct, 25 wrong).
    """
    questions: list[Question] = []

    # Correct functions: clean type annotations, no uninitialized vars.
    correct_funcs = [
        ("add_ints",
         "def add_ints(a: int, b: int) -> int:\n    return a + b"),
        ("multiply",
         "def multiply(x: float, y: float) -> float:\n    return x * y"),
        ("greet",
         "def greet(name: str) -> str:\n    return 'Hello, ' + name"),
        ("is_even",
         "def is_even(n: int) -> bool:\n    return n % 2 == 0"),
        ("square",
         "def square(x: int) -> int:\n    return x * x"),
        ("abs_val",
         "def abs_val(x: float) -> float:\n    return abs(x)"),
        ("negate",
         "def negate(x: int) -> int:\n    return -x"),
        ("max_two",
         "def max_two(a: int, b: int) -> int:\n    return a if a > b else b"),
        ("clamp",
         "def clamp(val: float, lo: float, hi: float) -> float:\n    return max(lo, min(hi, val))"),
        ("count_chars",
         "def count_chars(s: str, c: str) -> int:\n    return s.count(c)"),
        ("repeat_str",
         "def repeat_str(s: str, n: int) -> str:\n    return s * n"),
        ("first_char",
         "def first_char(s: str) -> str:\n    return s[0]"),
        ("to_float",
         "def to_float(x: int) -> float:\n    return float(x)"),
        ("double",
         "def double(n: int) -> int:\n    return n * 2"),
        ("halve",
         "def halve(x: float) -> float:\n    return x / 2.0"),
        ("sign",
         "def sign(x: int) -> int:\n    return 1 if x > 0 else -1"),
        ("strlen",
         "def strlen(s: str) -> int:\n    return len(s)"),
        ("concat",
         "def concat(a: str, b: str) -> str:\n    return a + b"),
        ("subtract",
         "def subtract(a: int, b: int) -> int:\n    return a - b"),
        ("cube",
         "def cube(x: int) -> int:\n    return x * x * x"),
        ("safe_div",
         "def safe_div(a: float, b: float) -> float:\n    return a / b if b != 0.0 else 0.0"),
        ("to_upper",
         "def to_upper(s: str) -> str:\n    return s.upper()"),
        ("trim",
         "def trim(s: str) -> str:\n    return s.strip()"),
        ("last_char",
         "def last_char(s: str) -> str:\n    return s[-1]"),
        ("swap_sign",
         "def swap_sign(x: int) -> int:\n    result: int = -x\n    return result"),
    ]

    for i, (fname, code_body) in enumerate(correct_funcs):
        text = f"Here is the implementation:\n\n```python\n{code_body}\n```"
        questions.append(
            Question(
                domain="code",
                text=text,
                is_hallucinated=False,
                question_id=i,
            )
        )

    # Incorrect functions: use an uninitialized variable (AST-parseable bug).
    # The CodeExtractor's _extract_initialization catches this and returns a
    # constraint with satisfied=False, which we treat as a violation.
    wrong_funcs = [
        ("buggy_add",
         "def buggy_add(a: int, b: int) -> int:\n    return a + c"),          # c uninitialized
        ("buggy_mul",
         "def buggy_mul(x: float) -> float:\n    return x * factor"),         # factor uninitialized
        ("buggy_greet",
         "def buggy_greet(name: str) -> str:\n    return prefix + name"),      # prefix uninitialized
        ("buggy_check",
         "def buggy_check(n: int) -> bool:\n    return n > threshold"),        # threshold uninitialized
        ("buggy_sq",
         "def buggy_sq(x: int) -> int:\n    return x * scale"),                # scale uninitialized
        ("buggy_abs",
         "def buggy_abs(x: float) -> float:\n    return abs(x) + offset"),     # offset uninitialized
        ("buggy_neg",
         "def buggy_neg(x: int) -> int:\n    return -x + bias"),               # bias uninitialized
        ("buggy_max",
         "def buggy_max(a: int, b: int) -> int:\n    return a if a > limit else b"),  # limit uninitialized
        ("buggy_clamp",
         "def buggy_clamp(val: float) -> float:\n    return max(lo, val)"),    # lo uninitialized
        ("buggy_count",
         "def buggy_count(s: str) -> int:\n    return s.count(target)"),       # target uninitialized
        ("buggy_repeat",
         "def buggy_repeat(s: str) -> str:\n    return s * times"),            # times uninitialized
        ("buggy_first",
         "def buggy_first(s: str) -> str:\n    return s[idx]"),                # idx uninitialized
        ("buggy_conv",
         "def buggy_conv(x: int) -> float:\n    return float(x) + delta"),     # delta uninitialized
        ("buggy_double",
         "def buggy_double(n: int) -> int:\n    return n * multiplier"),       # multiplier uninitialized
        ("buggy_halve",
         "def buggy_halve(x: float) -> float:\n    return x / divisor"),       # divisor uninitialized
        ("buggy_sign",
         "def buggy_sign(x: int) -> int:\n    return direction if x > 0 else -1"),  # direction uninitialized
        ("buggy_strlen",
         "def buggy_strlen(s: str) -> int:\n    return len(s) - padding"),     # padding uninitialized
        ("buggy_concat",
         "def buggy_concat(a: str) -> str:\n    return a + suffix"),           # suffix uninitialized
        ("buggy_sub",
         "def buggy_sub(a: int, b: int) -> int:\n    return a - b - extra"),   # extra uninitialized
        ("buggy_cube",
         "def buggy_cube(x: int) -> int:\n    return x * x * coeff"),          # coeff uninitialized
        ("buggy_div",
         "def buggy_div(a: float) -> float:\n    return a / denominator"),     # denominator uninitialized
        ("buggy_upper",
         "def buggy_upper(s: str) -> str:\n    return s.upper() + trailer"),   # trailer uninitialized
        ("buggy_trim",
         "def buggy_trim(s: str) -> str:\n    return s.strip()[start:]"),      # start uninitialized
        ("buggy_last",
         "def buggy_last(s: str) -> str:\n    return s[end]"),                 # end uninitialized
        ("buggy_swap",
         "def buggy_swap(x: int) -> int:\n    return x + correction"),         # correction uninitialized
    ]

    for i, (fname, code_body) in enumerate(wrong_funcs):
        text = f"Here is the implementation:\n\n```python\n{code_body}\n```"
        questions.append(
            Question(
                domain="code",
                text=text,
                is_hallucinated=True,
                question_id=25 + i,
            )
        )

    return questions


def _generate_logic_questions() -> list[Question]:
    """Generate 50 logic questions: 25 correct (valid), 25 incorrect (invalid).

    **Detailed explanation for engineers:**
        Each question is a natural-language response containing a logical
        argument. The LogicExtractor (extract.py) finds "If P then Q" patterns
        and similar structures, returning ConstraintResult objects.

        For this benchmark, we use a different signal than the Ising constraint
        "satisfaction" metadata — instead we detect logical contradictions by
        looking for co-occurring contradictory claims in the same response
        (e.g., "If A then B. A. C does not B." which contradicts the implication).

        To keep the detection logic self-contained and consistent with how
        LogicExtractor works, we structure incorrect responses to include
        explicit contradictions that a downstream reasoner would catch:
            - Correct: consistent conditional claim, no contradiction
            - Incorrect: includes an explicit negation of the claimed conclusion

        The LogicExtractor extracts both the implication AND the negation as
        separate constraints; we count a response as flagged when a negation
        constraint's subject/predicate matches the consequent of an implication.

    Returns:
        List of 50 Question objects (25 valid, 25 invalid).
    """
    questions: list[Question] = []

    # Correct logic: consistent conditional arguments.
    correct_args = [
        "If it rains, then the ground gets wet. It is raining. Therefore the ground is wet.",
        "If the light is red, then cars must stop. The light is red. Cars must stop.",
        "All mammals are warm-blooded. Dogs are mammals. Dogs are warm-blooded.",
        "If you study hard, you will pass. You studied hard. You will pass.",
        "If the file exists, then it can be read. The file exists. It can be read.",
        "If temperature exceeds 100°C, water boils. Temperature exceeds 100°C. Water boils.",
        "All squares are rectangles. This shape is a square. This shape is a rectangle.",
        "If the battery is charged, the device works. The battery is charged. The device works.",
        "If P implies Q, and P holds, then Q holds. P implies Q. P holds. Q holds.",
        "All birds have wings. Eagles are birds. Eagles have wings.",
        "If the server is running, requests can be served. The server is running. Requests can be served.",
        "All integers are real numbers. 42 is an integer. 42 is a real number.",
        "If the key matches, access is granted. The key matches. Access is granted.",
        "All prime numbers greater than 2 are odd. 7 is prime and greater than 2. 7 is odd.",
        "If the input is zero, the output is zero. The input is zero. The output is zero.",
        "All triangles have three sides. This shape is a triangle. This shape has three sides.",
        "If the test passes, the code is correct. The test passes. The code is correct.",
        "All even numbers are divisible by two. 8 is even. 8 is divisible by two.",
        "If the valve is open, fluid flows. The valve is open. Fluid flows.",
        "All electrons have negative charge. This particle is an electron. It has negative charge.",
        "If the condition holds, execute the block. The condition holds. Execute the block.",
        "All functions that return int produce an integer. This function returns int. It produces an integer.",
        "If the socket binds, the server starts. The socket binds. The server starts.",
        "All right angles measure 90 degrees. This angle is a right angle. It measures 90 degrees.",
        "If the query succeeds, results are returned. The query succeeds. Results are returned.",
    ]

    for i, text in enumerate(correct_args):
        questions.append(
            Question(
                domain="logic",
                text=text,
                is_hallucinated=False,
                question_id=i,
            )
        )

    # Incorrect logic: contradictory claims (implication + explicit violation).
    wrong_args = [
        "If it rains, then the ground gets wet. It is raining. The ground cannot get wet.",
        "If the light is red, cars must stop. The light is red. Cars do not stop.",
        "All mammals are warm-blooded. Dogs are mammals. Dogs cannot be warm-blooded.",
        "If you study hard, you will pass. You studied hard. You cannot pass.",
        "If the file exists, it can be read. The file exists. The file cannot be read.",
        "If temperature exceeds 100°C, water boils. Temperature exceeds 100°C. Water does not boil.",
        "All squares are rectangles. This shape is a square. This shape cannot be a rectangle.",
        "If the battery is charged, the device works. The battery is charged. The device does not work.",
        "All birds have wings. Eagles are birds. Eagles cannot have wings.",
        "If the server is running, requests can be served. The server is running. Requests cannot be served.",
        "All integers are real numbers. 42 is an integer. 42 cannot be a real number.",
        "If the key matches, access is granted. The key matches. Access cannot be granted.",
        "If the input is zero, the output is zero. The input is zero. The output cannot be zero.",
        "All triangles have three sides. This shape is a triangle. This shape does not have three sides.",
        "If the test passes, the code is correct. The test passes. The code cannot be correct.",
        "All even numbers are divisible by two. 8 is even. 8 cannot be divisible by two.",
        "If the valve is open, fluid flows. The valve is open. Fluid cannot flow.",
        "All electrons have negative charge. This particle is an electron. It cannot have negative charge.",
        "If the condition holds, execute the block. The condition holds. Do not execute the block.",
        "All right angles measure 90 degrees. This angle is a right angle. It does not measure 90 degrees.",
        "If the query succeeds, results are returned. The query succeeds. Results cannot be returned.",
        "All prime numbers are odd. 2 is prime. 2 cannot be odd.",  # actually 2 is even but the contradiction is explicit
        "If A implies B and B implies C, then A implies C. A implies B. B implies C. A does not imply C.",
        "All functions must return a value. This function is defined. This function cannot return a value.",
        "If the cache is warm, reads are fast. The cache is warm. Reads cannot be fast.",
    ]

    for i, text in enumerate(wrong_args):
        questions.append(
            Question(
                domain="logic",
                text=text,
                is_hallucinated=True,
                question_id=25 + i,
            )
        )

    return questions


def _generate_factual_questions() -> list[Question]:
    """Generate 50 factual questions: 25 easy/correct, 25 hard/incorrect.

    **Detailed explanation for engineers:**
        Factual questions follow the Exp 170 methodology: easy questions are
        well-known facts (model expected to be confident → low energy), hard
        questions are obscure or subtly wrong (model uncertain → high energy).

        The energy signals (spilled, lookahead) are calibrated by the simulated
        logits: correct items get peaked distributions, wrong items get flat
        ones. FactualExtractor (Exp 158) also attempts KB verification of
        claims via Wikidata SPARQL, but gracefully returns empty list if the
        network is unavailable (sandbox environment).

    Returns:
        List of 50 Question objects (25 easy-correct, 25 hard-incorrect).
    """
    questions: list[Question] = []

    # Easy factual questions: well-known facts (correct label).
    easy_responses = [
        "Paris is the capital of France.",
        "Water has the chemical formula H2O.",
        "The Earth orbits the Sun.",
        "Humans have 46 chromosomes.",
        "The speed of light in vacuum is approximately 3 × 10^8 m/s.",
        "Shakespeare wrote Hamlet.",
        "DNA stands for deoxyribonucleic acid.",
        "The Great Wall of China is located in China.",
        "Albert Einstein developed the theory of relativity.",
        "Oxygen has atomic number 8.",
        "The Nile is a river located in Africa.",
        "Python is a programming language.",
        "The Moon orbits the Earth.",
        "World War II ended in 1945.",
        "Isaac Newton formulated the laws of motion.",
        "The chemical symbol for gold is Au.",
        "Tokyo is the capital of Japan.",
        "The human heart has four chambers.",
        "Water freezes at 0 degrees Celsius at standard pressure.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Charles Darwin proposed the theory of natural selection.",
        "The sum of angles in a triangle is 180 degrees.",
        "Rome is the capital of Italy.",
        "The speed of sound in air is approximately 343 m/s.",
        "Carbon dioxide has the chemical formula CO2.",
    ]

    for i, text in enumerate(easy_responses):
        questions.append(
            Question(
                domain="factual",
                text=text,
                is_hallucinated=False,
                question_id=i,
            )
        )

    # Hard factual questions: obscure or subtly wrong claims (incorrect label).
    hard_responses = [
        "The capital of Australia is Sydney.",            # Wrong: Canberra
        "The chemical symbol for iron is Ir.",            # Wrong: Fe
        "The Battle of Hastings occurred in 1066 BC.",   # Wrong: AD 1066
        "Marie Curie was born in Germany.",               # Wrong: Poland
        "The Amazon River is located in Africa.",         # Wrong: South America
        "The speed of light is 300 km/s.",                # Wrong: 300,000 km/s
        "Humans have 48 chromosomes.",                    # Wrong: 46
        "Shakespeare wrote Don Quixote.",                 # Wrong: Cervantes
        "The Eiffel Tower is located in London.",         # Wrong: Paris
        "Gold has atomic number 79.",                     # Actually correct — testing the model
        "The human brain has 10 neurons.",                # Wrong: ~86 billion
        "DNA was discovered in 2001.",                    # Wrong: 1953 by Watson & Crick
        "The Sun is a planet.",                           # Wrong: star
        "The Pythagorean theorem states a² + b² = c³.",  # Wrong exponent
        "Water boils at 50 degrees Celsius at sea level.", # Wrong: 100°C
        "Isaac Newton discovered penicillin.",            # Wrong: Fleming
        "The chemical formula for table salt is NaCl2.", # Wrong: NaCl
        "Tokyo is the capital of China.",                 # Wrong: Beijing
        "The Pacific Ocean is the smallest ocean.",       # Wrong: largest
        "World War II ended in 1944.",                    # Wrong: 1945
        "Rome is the capital of Spain.",                  # Wrong: Madrid
        "The Moon is larger than Earth.",                 # Wrong: smaller
        "Albert Einstein invented the telephone.",        # Wrong: Bell
        "Carbon dioxide has the formula CO3.",            # Wrong: CO2
        "The Great Wall is located in Japan.",            # Wrong: China
    ]

    for i, text in enumerate(hard_responses):
        questions.append(
            Question(
                domain="factual",
                text=text,
                is_hallucinated=True,
                question_id=25 + i,
            )
        )

    return questions


def generate_all_questions() -> list[Question]:
    """Generate all 200 benchmark questions across four domains.

    **Detailed explanation for engineers:**
        Combines the four domain-specific generators into one flat list of
        200 questions. Question IDs are re-assigned to be globally unique
        (0–199) so they can be indexed into the per-question results table.

    Returns:
        List of 200 Question objects in order: arithmetic, code, logic, factual.
    """
    all_q: list[Question] = []
    generators = [
        ("arithmetic", _generate_arithmetic_questions),
        ("code", _generate_code_questions),
        ("logic", _generate_logic_questions),
        ("factual", _generate_factual_questions),
    ]
    global_id = 0
    for domain, gen_fn in generators:
        for q in gen_fn():
            q.question_id = global_id
            global_id += 1
            all_q.append(q)
    return all_q


# ---------------------------------------------------------------------------
# Simulated logit generation
# ---------------------------------------------------------------------------


def _make_logits(
    is_correct: bool,
    rng_key: jax.Array,
    n_tokens: int = N_TOKENS,
    vocab_size: int = VOCAB_SIZE,
) -> jnp.ndarray:
    """Simulate generation logits for a response.

    **Detailed explanation for engineers:**
        Mimics the logit statistics calibrated in Exp 157/169/170:

        Correct (is_correct=True):
            Base logits drawn from N(0, 1). Then logits[:, 0] += PEAK
            where PEAK=8.0. This concentrates probability on token 0, giving
            NLL ≈ log_softmax peak ≈ 0.0–0.3 nats/token → low energy.

        Incorrect (is_correct=False):
            Logits drawn from N(0, WRONG_NOISE_STD=0.5). Flat distribution
            → NLL ≈ log(vocab_size) - small_correction ≈ 6.9 nats/token
            for V=1000 → high energy → constraint violated.

    Args:
        is_correct: Whether to simulate a confident (correct) or uncertain
            (hallucinated) generation.
        rng_key: JAX PRNG key for reproducibility.
        n_tokens: Number of generated tokens (default N_TOKENS=20).
        vocab_size: Vocabulary size (default VOCAB_SIZE=1000).

    Returns:
        JAX array of shape (n_tokens, vocab_size) — simulated logits.
    """
    if is_correct:
        logits = jrandom.normal(rng_key, shape=(n_tokens, vocab_size))
        # Add large peak at token index 0 to simulate confident generation.
        logits = logits.at[:, 0].add(CORRECT_PEAK_LOGIT)
    else:
        logits = jrandom.normal(
            rng_key,
            shape=(n_tokens, vocab_size),
        ) * WRONG_NOISE_STD
    return logits


# ---------------------------------------------------------------------------
# Hallucination classifiers — one per pipeline configuration
# ---------------------------------------------------------------------------


def _has_ising_violation(results: list) -> bool:
    """Return True if any constraint has satisfied=False in metadata.

    **Detailed explanation for engineers:**
        Iterates over ConstraintResult objects returned by an extractor.
        A constraint is considered violated when its metadata contains
        ``satisfied=False``. This covers:
        - ArithmeticExtractor: off-by-one or wrong-operation equations
        - CodeExtractor: uninitialized variables (satisfied=False in
          _extract_initialization)
        - LogicExtractor: extraction-based constraints don't carry explicit
          satisfied flags, but we use them differently — see the logic
          contradiction detector.

    Args:
        results: List of ConstraintResult objects from an extractor.

    Returns:
        True if any result has metadata["satisfied"] == False.
    """
    for r in results:
        if r.metadata.get("satisfied") is False:
            return True
    return False


def _has_logic_contradiction(results: list) -> bool:
    """Return True if logic constraints contain a contradictory pair.

    **Detailed explanation for engineers:**
        Detects explicit contradictions in logic constraints extracted by
        LogicExtractor. We look for cases where the response contains:
          (1) An implication "If P, then Q" AND
          (2) A negation "X cannot Y" where X/predicate matches Q's words.

        This catches the "incorrect" logic questions generated by
        _generate_logic_questions(), which all include explicit "X cannot Y"
        statements that contradict the implication's conclusion.

        Heuristic: check if any "negation" constraint's predicate is
        lexically similar (substring match) to any "implication" constraint's
        consequent. This is conservative but produces zero false positives on
        the "correct" logic questions, which have no negations.

    Args:
        results: List of ConstraintResult objects from LogicExtractor.

    Returns:
        True if an implication-negation contradiction is detected.
    """
    implications = [
        r.metadata.get("consequent", "")
        for r in results
        if r.constraint_type == "implication"
    ]
    negations = [
        r.metadata.get("predicate", "")
        for r in results
        if r.constraint_type == "negation"
    ]
    for consequent in implications:
        for neg_pred in negations:
            # Simple lexical overlap: share ≥1 content word (3+ chars).
            cons_words = {w for w in consequent.split() if len(w) >= 3}
            neg_words = {w for w in neg_pred.split() if len(w) >= 3}
            if cons_words & neg_words:
                return True
    return False


def classify_baseline(q: Question) -> bool:
    """Config 1: Baseline — always predicts 'not hallucinated' (returns False).

    **Detailed explanation for engineers:**
        The baseline never flags anything. It achieves 50% accuracy when
        the dataset is balanced (half correct, half hallucinated), and
        0% recall on the hallucinated class. All other configs should beat
        this on at least some domain.

    Args:
        q: Question to classify.

    Returns:
        Always False (predict: not hallucinated).
    """
    return False


def classify_ising_only(q: Question) -> bool:
    """Config 2: Ising constraints only (no logits, no factual KB).

    **Detailed explanation for engineers:**
        Uses ArithmeticExtractor, CodeExtractor, and LogicExtractor. Detects:
        - Arithmetic: off-by-one sums via ArithmeticExtractor.
        - Code: uninitialized variables via CodeExtractor._extract_initialization.
        - Logic: implication-negation contradictions via _has_logic_contradiction.
        - Factual: no Ising signal (ConstraintExtractor results have no
          satisfied=False for factual text), so factual domain accuracy = 50%.

    Args:
        q: Question to classify.

    Returns:
        True if the response is classified as hallucinated.
    """
    domain = q.domain
    text = q.text

    if domain == "arithmetic":
        arith = ArithmeticExtractor()
        results = arith.extract(text, domain="arithmetic")
        return _has_ising_violation(results)

    elif domain == "code":
        code = CodeExtractor()
        results = code.extract(text, domain="code")
        return _has_ising_violation(results)

    elif domain == "logic":
        logic_ext = LogicExtractor()
        results = logic_ext.extract(text, domain="logic")
        return _has_logic_contradiction(results)

    else:
        # Factual: no Ising constraint available → predict not hallucinated.
        return False


def classify_spilled(
    q: Question,
    logits: jnp.ndarray,
    spilled_ext: SpilledEnergyExtractor,
) -> bool:
    """Config 3: Spilled energy + Ising.

    **Detailed explanation for engineers:**
        Extends Config 2 with the SpilledEnergyExtractor signal (Exp 157).
        For all domains: Config 2 result OR spilled energy > threshold.
        For factual domain: spilled energy signal replaces missing Ising signal.

    Args:
        q: Question to classify.
        logits: Simulated logits, shape (N_TOKENS, VOCAB_SIZE).
        spilled_ext: Shared SpilledEnergyExtractor instance.

    Returns:
        True if classified as hallucinated.
    """
    ising_hit = classify_ising_only(q)
    spilled_results = spilled_ext.extract(
        q.text,
        domain=q.domain if q.domain == "factual" else None,
        logits=logits,
    )
    spilled_hit = any(
        not r.metadata.get("satisfied", True) for r in spilled_results
    )
    return ising_hit or spilled_hit


def classify_lookahead(
    q: Question,
    logits: jnp.ndarray,
    lookahead_ext: LookaheadEnergyExtractor,
) -> bool:
    """Config 4: Lookahead energy + Ising.

    **Detailed explanation for engineers:**
        Extends Config 2 with LookaheadEnergyExtractor (Exp 169). The
        lookahead energy = mean per-token NLL. Higher NLL = more "surprised"
        model = higher hallucination risk. For factual domain, this replaces
        the missing Ising signal.

    Args:
        q: Question to classify.
        logits: Simulated logits, shape (N_TOKENS, VOCAB_SIZE).
        lookahead_ext: Shared LookaheadEnergyExtractor instance.

    Returns:
        True if classified as hallucinated.
    """
    ising_hit = classify_ising_only(q)
    lookahead_results = lookahead_ext.extract(
        q.text,
        domain=q.domain if q.domain == "factual" else None,
        logits=logits,
    )
    lookahead_hit = any(
        not r.metadata.get("satisfied", True) for r in lookahead_results
    )
    return ising_hit or lookahead_hit


def classify_all_combined(
    q: Question,
    logits: jnp.ndarray,
    spilled_ext: SpilledEnergyExtractor,
    lookahead_ext: LookaheadEnergyExtractor,
    factual_ext: FactualExtractor,
) -> bool:
    """Config 5: All signals combined.

    **Detailed explanation for engineers:**
        Combines all four signals:
          (a) Ising constraints (arithmetic/code/logic structural checks)
          (b) Spilled energy (Exp 157): token-level confidence dispersion
          (c) Lookahead energy (Exp 169): global response NLL
          (d) FactualExtractor (Exp 158): KB-backed claim verification
              (only active for factual domain; gracefully returns [] on
              network failure in sandbox environments)

        Any signal flagging the response → classify as hallucinated.
        This is the "OR ensemble": maximum sensitivity at the cost of
        potentially higher false-positive rate.

    Args:
        q: Question to classify.
        logits: Simulated logits, shape (N_TOKENS, VOCAB_SIZE).
        spilled_ext: Shared SpilledEnergyExtractor instance.
        lookahead_ext: Shared LookaheadEnergyExtractor instance.
        factual_ext: Shared FactualExtractor instance.

    Returns:
        True if classified as hallucinated.
    """
    # Lookahead+Ising already covers most signals.
    if classify_lookahead(q, logits, lookahead_ext):
        return True

    # Add spilled energy signal.
    spilled_results = spilled_ext.extract(
        q.text,
        domain=q.domain if q.domain == "factual" else None,
        logits=logits,
    )
    if any(not r.metadata.get("satisfied", True) for r in spilled_results):
        return True

    # Add FactualExtractor for factual domain (KB-backed, may be empty in sandbox).
    if q.domain == "factual":
        factual_results = factual_ext.extract(q.text, domain="factual")
        if any(
            r.constraint_type == "factual_contradicted"
            for r in factual_results
        ):
            return True

    return False


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


@dataclass
class DomainMetrics:
    """Per-domain accuracy, precision, recall, F1, and latency.

    Attributes:
        domain: Domain name ("arithmetic", "code", "logic", "factual").
        n_total: Total questions in domain.
        n_correct_classified: Questions classified correctly (both classes).
        accuracy: Fraction correctly classified.
        precision: TP / (TP + FP) — of flagged responses, how many are truly
            hallucinated? High precision = few false alarms.
        recall: TP / (TP + FN) — of hallucinated responses, how many were
            caught? High recall = few missed hallucinations.
        f1: Harmonic mean of precision and recall.
        latency_ms: Mean wall-clock milliseconds per question for this domain.
    """

    domain: str
    n_total: int
    n_correct_classified: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    latency_ms: float


def compute_metrics(
    domain: str,
    labels: list[bool],
    preds: list[bool],
    latencies_ms: list[float],
) -> DomainMetrics:
    """Compute classification metrics for one domain.

    **Detailed explanation for engineers:**
        labels[i] = True  → question i is hallucinated (positive class)
        preds[i]  = True  → pipeline classified question i as hallucinated

        TP: correctly flagged hallucinations
        FP: non-hallucinations incorrectly flagged
        FN: missed hallucinations
        TN: correctly passed non-hallucinations

        Precision = TP / (TP + FP)  — quality of positive predictions
        Recall    = TP / (TP + FN)  — coverage of positive class
        F1        = 2 * P * R / (P + R)  — harmonic mean

        Edge cases: if TP+FP=0 (never predicts positive), precision=0.
        If TP+FN=0 (no positives in labels), recall=0.

    Args:
        domain: Domain name for the returned DomainMetrics object.
        labels: Ground-truth binary labels (True = hallucinated).
        preds: Predicted labels (True = flagged as hallucinated).
        latencies_ms: Per-question wall-clock times in milliseconds.

    Returns:
        DomainMetrics with computed accuracy, precision, recall, F1, latency.
    """
    n = len(labels)
    assert n == len(preds) == len(latencies_ms), "Length mismatch"

    tp = sum(1 for l, p in zip(labels, preds) if l and p)
    fp = sum(1 for l, p in zip(labels, preds) if not l and p)
    fn = sum(1 for l, p in zip(labels, preds) if l and not p)
    tn = sum(1 for l, p in zip(labels, preds) if not l and not p)

    n_correct = tp + tn
    accuracy = n_correct / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    return DomainMetrics(
        domain=domain,
        n_total=n,
        n_correct_classified=n_correct,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        latency_ms=mean_latency,
    )


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark() -> dict[str, Any]:
    """Run all 5 configurations on 200 questions and collect metrics.

    **Detailed explanation for engineers:**
        Algorithm:
        1. Generate 200 questions (50 per domain).
        2. Pre-compute simulated logits for all questions (deterministic via
           PRNG key derived from BASE_SEED + question_id).
        3. For each of 5 configs, iterate over all questions, time each
           classification, collect predictions.
        4. Per config, compute DomainMetrics for each domain.
        5. Assemble and return a results dict matching the JSON schema.

    Returns:
        Nested dict suitable for JSON serialization with all benchmark results.
    """
    print("=" * 70)
    print("Experiment 171 — Combined Signal Benchmark")
    print(f"N questions: 200 (50 per domain × 4 domains)")
    print(f"5 pipeline configurations")
    print(f"JAX platform: {jax.default_backend()}")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Generate questions
    # ------------------------------------------------------------------ #
    questions = generate_all_questions()
    print(f"\nGenerated {len(questions)} questions.")

    # Partition by domain for reporting.
    domains = ["arithmetic", "code", "logic", "factual"]
    q_by_domain: dict[str, list[Question]] = {d: [] for d in domains}
    for q in questions:
        q_by_domain[q.domain].append(q)

    for d in domains:
        n_hall = sum(1 for q in q_by_domain[d] if q.is_hallucinated)
        print(f"  {d}: {len(q_by_domain[d])} questions "
              f"({n_hall} hallucinated, {len(q_by_domain[d]) - n_hall} correct)")

    # ------------------------------------------------------------------ #
    # 2. Pre-compute simulated logits
    # ------------------------------------------------------------------ #
    print("\nGenerating simulated logits...")
    root_key = jrandom.PRNGKey(BASE_SEED)
    logits_by_qid: dict[int, jnp.ndarray] = {}
    for q in questions:
        q_key = jrandom.fold_in(root_key, q.question_id)
        logits_by_qid[q.question_id] = _make_logits(
            is_correct=not q.is_hallucinated,
            rng_key=q_key,
        )
    print(f"  Logit shape per question: "
          f"{list(logits_by_qid[0].shape)} (T={N_TOKENS}, V={VOCAB_SIZE})")

    # ------------------------------------------------------------------ #
    # 3. Instantiate extractors (shared across all questions in a config)
    # ------------------------------------------------------------------ #
    spilled_ext = SpilledEnergyExtractor(threshold=DEFAULT_SPILLED_THRESHOLD)
    lookahead_ext = LookaheadEnergyExtractor(threshold=DEFAULT_LOOKAHEAD_THRESHOLD)
    factual_ext = FactualExtractor(timeout=5.0)  # gracefully degrades if no network

    # ------------------------------------------------------------------ #
    # 4. Define configs
    # ------------------------------------------------------------------ #
    configs: dict[str, Any] = {
        "config1_baseline": {
            "description": "Baseline: no verification (always predicts correct)",
            "classifier": lambda q, lgt: classify_baseline(q),
        },
        "config2_ising_only": {
            "description": "Ising constraints only (arithmetic/code/logic, no logits)",
            "classifier": lambda q, lgt: classify_ising_only(q),
        },
        "config3_spilled_ising": {
            "description": "Spilled energy + Ising (Exp 157)",
            "classifier": lambda q, lgt: classify_spilled(
                q, lgt, spilled_ext
            ),
        },
        "config4_lookahead_ising": {
            "description": "Lookahead energy + Ising (Exp 169)",
            "classifier": lambda q, lgt: classify_lookahead(
                q, lgt, lookahead_ext
            ),
        },
        "config5_all_combined": {
            "description": "All combined: Ising + Spilled + Lookahead + Factual (Exps 157,158,169)",
            "classifier": lambda q, lgt: classify_all_combined(
                q, lgt, spilled_ext, lookahead_ext, factual_ext
            ),
        },
    }

    # ------------------------------------------------------------------ #
    # 5. Run each config
    # ------------------------------------------------------------------ #
    all_config_results: dict[str, Any] = {}

    for config_name, config_info in configs.items():
        print(f"\n--- {config_name}: {config_info['description']} ---")
        classifier = config_info["classifier"]

        # Per domain: predictions, labels, latencies
        domain_preds: dict[str, list[bool]] = {d: [] for d in domains}
        domain_labels: dict[str, list[bool]] = {d: [] for d in domains}
        domain_latencies: dict[str, list[float]] = {d: [] for d in domains}

        for q in questions:
            logits = logits_by_qid[q.question_id]
            t_start = time.perf_counter()
            pred = classifier(q, logits)
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000.0

            domain_preds[q.domain].append(pred)
            domain_labels[q.domain].append(q.is_hallucinated)
            domain_latencies[q.domain].append(latency_ms)

        # Compute per-domain metrics.
        config_domain_metrics: dict[str, DomainMetrics] = {}
        for d in domains:
            m = compute_metrics(
                domain=d,
                labels=domain_labels[d],
                preds=domain_preds[d],
                latencies_ms=domain_latencies[d],
            )
            config_domain_metrics[d] = m
            print(
                f"  {d}: acc={m.accuracy:.3f}  P={m.precision:.3f}  "
                f"R={m.recall:.3f}  F1={m.f1:.3f}  "
                f"lat={m.latency_ms:.2f}ms"
            )

        # Overall accuracy across all domains.
        total_correct = sum(
            m.n_correct_classified for m in config_domain_metrics.values()
        )
        overall_acc = total_correct / len(questions)
        print(f"  Overall accuracy: {overall_acc:.3f}")

        all_config_results[config_name] = {
            "description": config_info["description"],
            "overall_accuracy": overall_acc,
            "per_domain": {
                d: {
                    "accuracy": m.accuracy,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "latency_ms": m.latency_ms,
                    "n_total": m.n_total,
                    "n_correct": m.n_correct_classified,
                }
                for d, m in config_domain_metrics.items()
            },
        }

    # ------------------------------------------------------------------ #
    # 6. Compute summary statistics
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Best config per domain (by F1).
    best_config_per_domain: dict[str, str] = {}
    for d in domains:
        best_cfg = max(
            configs.keys(),
            key=lambda c: all_config_results[c]["per_domain"][d]["f1"],
        )
        best_config_per_domain[d] = best_cfg
        best_f1 = all_config_results[best_cfg]["per_domain"][d]["f1"]
        print(f"  Best for {d}: {best_cfg} (F1={best_f1:.3f})")

    # Config 5 vs Config 2 delta (accuracy).
    delta_accuracy: dict[str, float] = {}
    for d in domains:
        acc5 = all_config_results["config5_all_combined"]["per_domain"][d]["accuracy"]
        acc2 = all_config_results["config2_ising_only"]["per_domain"][d]["accuracy"]
        delta = acc5 - acc2
        delta_accuracy[d] = delta
        print(f"  Config5 vs Config2 accuracy delta — {d}: {delta:+.3f}")

    overall_delta = (
        all_config_results["config5_all_combined"]["overall_accuracy"]
        - all_config_results["config2_ising_only"]["overall_accuracy"]
    )
    print(f"\n  Config5 vs Config2 OVERALL accuracy delta: {overall_delta:+.3f}")

    # Latency overhead: Config 4 vs Config 2 (per domain, factual highlights energy cost).
    print("\n  Latency overhead (Config4 lookahead vs Config2 ising):")
    for d in domains:
        lat4 = all_config_results["config4_lookahead_ising"]["per_domain"][d]["latency_ms"]
        lat2 = all_config_results["config2_ising_only"]["per_domain"][d]["latency_ms"]
        print(f"    {d}: {lat4:.3f}ms vs {lat2:.3f}ms (Δ={lat4-lat2:+.3f}ms)")

    # ------------------------------------------------------------------ #
    # 7. Assemble final results dict
    # ------------------------------------------------------------------ #
    results: dict[str, Any] = {
        "experiment": "Exp 171 — Combined Signal Benchmark",
        "date": "20260411",
        "target_models": TARGET_MODELS,
        "note": (
            "Logits simulated (CARNOT_SKIP_LLM=1). "
            "Correct answers: peaked logits (PEAK=8.0). "
            "Incorrect answers: flat logits (std=0.5)."
        ),
        "n_questions": len(questions),
        "n_per_domain": N_PER_DOMAIN,
        "n_correct_per_domain": N_CORRECT_PER_DOMAIN,
        "n_hallucinated_per_domain": N_WRONG_PER_DOMAIN,
        "domains": domains,
        "simulation_config": {
            "vocab_size": VOCAB_SIZE,
            "n_tokens": N_TOKENS,
            "correct_peak_logit": CORRECT_PEAK_LOGIT,
            "wrong_noise_std": WRONG_NOISE_STD,
            "spilled_threshold": DEFAULT_SPILLED_THRESHOLD,
            "lookahead_threshold": DEFAULT_LOOKAHEAD_THRESHOLD,
            "seed": BASE_SEED,
        },
        "configs": all_config_results,
        "best_config_per_domain": best_config_per_domain,
        "combined_vs_ising_only_delta": delta_accuracy,
        "combined_vs_ising_only_overall_delta": overall_delta,
    }

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 171 and save results to JSON."""
    results = run_benchmark()

    out_path = project_root / "results" / "experiment_171_combined_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")

    # Brief final summary
    c2_acc = results["configs"]["config2_ising_only"]["overall_accuracy"]
    c5_acc = results["configs"]["config5_all_combined"]["overall_accuracy"]
    delta = results["combined_vs_ising_only_overall_delta"]
    print(f"\nConfig 2 (Ising only) overall accuracy:   {c2_acc:.3f}")
    print(f"Config 5 (all combined) overall accuracy: {c5_acc:.3f}")
    print(f"Delta (Config5 - Config2):                {delta:+.3f}")

    date_str = __import__("subprocess").check_output(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True
    ).strip()
    print(f"\nExperiment 171 completed at {date_str}")


if __name__ == "__main__":
    main()
