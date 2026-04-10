#!/usr/bin/env python3
"""Experiment 92: MATH Benchmark — step-level constraint verification.

**Researcher summary:**
    GSM8K (Exp 91) tests single-step arithmetic. MATH problems require
    multi-step reasoning chains where errors compound. This experiment tests
    whether step-by-step constraint verification catches intermediate errors
    that final-answer-only verification misses.

**Detailed explanation for engineers:**
    The key insight: verify EACH reasoning step, not just the conclusion.
    When an LLM solves a multi-step algebra problem, an error in step 2
    propagates to all later steps. Final-answer-only verification says
    "wrong" but cannot say WHERE the error entered. Step-level verification
    flags the exact step, enabling targeted repair.

    Architecture per problem:
    1. Load 100 MATH problems (Level 1-3, algebra/number_theory subjects).
    2. For each model:
       a. Prompt with "Solve step by step: <problem>"
       b. Parse response into reasoning steps (split on newlines, "Step N:", etc.)
       c. Extract the final answer (boxed or last numerical value)
       d. For EACH reasoning step, extract arithmetic constraints and verify
       e. Track which step FIRST introduces an error
    3. Compare step-level verification vs final-answer-only verification.

    Target models:
    - Qwen/Qwen3.5-0.8B (small, fast, good at arithmetic reasoning)
    - google/gemma-4-E4B-it (instruction-tuned, different architecture)

    Metrics: end-to-end accuracy, step-level error detection, error
    propagation distance, constraint coverage, per-model comparison.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_92_math_benchmark.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-006
"""

from __future__ import annotations

import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make carnot library importable.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))


# ---------------------------------------------------------------------------
# 1. Model configurations (same as Exp 91)
# ---------------------------------------------------------------------------

MODEL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "Qwen3.5-0.8B",
        "candidates": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"],
        "trust_remote_code": True,
    },
    {
        "name": "Gemma4-E4B-it",
        "candidates": ["google/gemma-4-E4B-it"],
        "trust_remote_code": True,
    },
]


# ---------------------------------------------------------------------------
# 2. MATH dataset loading — real first, synthetic fallback
# ---------------------------------------------------------------------------


def load_math_problems(n: int = 100, seed: int = 92) -> list[dict[str, Any]]:
    """Load n problems from the MATH dataset, filtered to Level 1-3 algebra/number_theory.

    **Detailed explanation for engineers:**
        Tries to load the MATH dataset from HuggingFace using the `datasets`
        library. The canonical path is "lighteval/MATH" with split "test".
        We filter to Level 1-3 (easier problems) and algebra/number_theory
        subjects, then sample n problems with a seeded shuffle.

        Each MATH example has:
        - 'problem': the problem statement
        - 'solution': LaTeX chain-of-thought solution ending with \\boxed{answer}
        - 'level': difficulty level string like "Level 1"
        - 'type': subject category like "Algebra", "Number Theory"

        If the datasets library is unavailable or download fails, we generate
        100 synthetic multi-step algebra problems with known solutions. The
        fallback is clearly marked in output so results are not misrepresented.

    Args:
        n: Number of problems to load (default 100).
        seed: Random seed for reproducible sampling.

    Returns:
        List of dicts with keys: problem, ground_truth (str), solution,
        level, subject, source ("math" or "synthetic").
    """
    rng = random.Random(seed)

    # --- Attempt to load real MATH dataset ---
    try:
        from datasets import load_dataset

        # Try multiple known paths for the MATH dataset.
        # EleutherAI/hendrycks_math requires specifying subject configs.
        target_configs = ["algebra", "number_theory"]
        all_examples: list[dict] = []

        # Strategy 1: EleutherAI/hendrycks_math with per-subject configs.
        try:
            for config_name in target_configs:
                print(f"  Loading EleutherAI/hendrycks_math/{config_name}...")
                subset = load_dataset("EleutherAI/hendrycks_math", config_name, split="test")
                for i in range(len(subset)):
                    ex = dict(subset[i])
                    ex["type"] = config_name.replace("_", " ").title()
                    all_examples.append(ex)
                print(f"    Got {len(subset)} examples.")
        except Exception as e:
            print(f"  EleutherAI/hendrycks_math failed: {e}")

        # Strategy 2: lighteval/MATH or hendrycks/competition_math as fallback.
        if not all_examples:
            for ds_name in ["lighteval/MATH", "hendrycks/competition_math"]:
                try:
                    print(f"  Trying {ds_name}...")
                    ds_full = load_dataset(ds_name, split="test")
                    for i in range(len(ds_full)):
                        all_examples.append(dict(ds_full[i]))
                    print(f"  Loaded {len(all_examples)} from {ds_name}.")
                    break
                except Exception as e:
                    print(f"  {ds_name} failed: {e}")

        if not all_examples:
            raise RuntimeError("No MATH dataset variant could be loaded.")

        # Wrap into a list-like for uniform access below.
        ds = all_examples
        print(f"  Total MATH examples loaded: {len(ds)}")

        # Filter to Level 1-3 and algebra/number_theory subjects.
        # Level field is like "Level 1", "Level 2", etc.
        # Type field is like "Algebra", "Number Theory", etc.
        allowed_levels = {"Level 1", "Level 2", "Level 3"}
        allowed_types = {"Algebra", "Number Theory"}

        filtered_indices: list[int] = []
        for i in range(len(ds)):
            example = ds[i]
            level = example.get("level", "")
            subject = example.get("type", "")
            if level in allowed_levels and subject in allowed_types:
                filtered_indices.append(i)

        print(f"  Filtered to {len(filtered_indices)} problems "
              f"(Level 1-3, algebra/number_theory).")

        if len(filtered_indices) == 0:
            print("  No matching problems found. Falling back to synthetic.")
            return _generate_synthetic_math(n, seed=seed)

        rng.shuffle(filtered_indices)
        selected = filtered_indices[:n]

        problems: list[dict[str, Any]] = []
        for idx in selected:
            example = ds[idx]
            gt = _extract_boxed_answer(example.get("solution", ""))
            if gt is None:
                continue

            problems.append({
                "problem": example["problem"],
                "ground_truth": gt,
                "solution": example.get("solution", ""),
                "level": example.get("level", ""),
                "subject": example.get("type", ""),
                "source": "math",
            })

        print(f"  Extracted {len(problems)} problems with valid boxed answers.")

        if len(problems) < n:
            shortfall = n - len(problems)
            print(f"  Padding with {shortfall} synthetic problems.")
            problems.extend(_generate_synthetic_math(shortfall, seed=seed + 1000))

        return problems[:n]

    except ImportError:
        print("  `datasets` library not available (pip install datasets).")
        print("  Falling back to synthetic MATH-style problems.")
    except Exception as e:
        print(f"  Failed to load MATH: {e}")
        print("  Falling back to synthetic MATH-style problems.")

    return _generate_synthetic_math(n, seed=seed)


def _extract_boxed_answer(solution: str) -> str | None:
    """Extract the answer from a \\boxed{...} expression in a MATH solution.

    **Detailed explanation for engineers:**
        MATH dataset solutions end with \\boxed{answer} where answer is the
        final result (may be a number, fraction, expression, etc.). We need
        to handle nested braces since answers can contain expressions like
        \\boxed{\\frac{3}{4}}.

        Returns the string inside \\boxed{} or None if not found.
    """
    # Find the last \boxed{...} allowing nested braces.
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return None

    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(solution) and depth > 0:
        if solution[pos] == "{":
            depth += 1
        elif solution[pos] == "}":
            depth -= 1
        pos += 1

    if depth != 0:
        return None

    return solution[start:pos - 1].strip()


def _generate_synthetic_math(n: int, seed: int = 92) -> list[dict[str, Any]]:
    """Generate n synthetic multi-step algebra problems with known solutions.

    **Detailed explanation for engineers:**
        Creates multi-step algebra and number theory problems that mirror the
        structure of MATH Level 1-3 problems. Each problem requires 2-5
        reasoning steps. The solution text includes step-by-step work so we
        can verify that the step extraction logic works on known-good solutions.

        Problem types:
        - Linear equations (solve for x)
        - Systems of two equations
        - Arithmetic sequences (find nth term or sum)
        - Modular arithmetic
        - Factoring / GCD / LCM
        - Percentage / ratio problems
        - Quadratic equations (integer roots)
        - Divisibility and remainders
    """
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []

    templates = [
        _math_linear_equation,
        _math_system_of_equations,
        _math_arithmetic_sequence,
        _math_modular_arithmetic,
        _math_gcd_lcm,
        _math_percentage_ratio,
        _math_quadratic_integer,
        _math_divisibility,
        _math_linear_combo,
        _math_absolute_value,
    ]

    for i in range(n):
        tmpl = templates[i % len(templates)]
        problem, solution_text, answer = tmpl(random.Random(seed + i * 173))
        level_num = rng.randint(1, 3)
        subject = rng.choice(["Algebra", "Number Theory"])

        problems.append({
            "problem": problem,
            "ground_truth": str(answer),
            "solution": solution_text,
            "level": f"Level {level_num}",
            "subject": subject,
            "source": "synthetic",
        })

    rng.shuffle(problems)
    return problems


# --- Synthetic MATH problem templates ---


def _math_linear_equation(rng: random.Random) -> tuple[str, str, int | str]:
    """Solve ax + b = c for integer x."""
    a = rng.choice([2, 3, 4, 5, 6, 7, 8])
    x = rng.randint(-10, 10)
    b = rng.randint(-20, 20)
    c = a * x + b

    problem = f"Solve for $x$: ${a}x + {b} = {c}$."
    solution = (
        f"Step 1: Subtract {b} from both sides: {a}x = {c} - {b} = {c - b}\n"
        f"Step 2: Divide both sides by {a}: x = {c - b} / {a} = {x}\n"
        f"\\boxed{{{x}}}"
    )
    return problem, solution, x


def _math_system_of_equations(rng: random.Random) -> tuple[str, str, int | str]:
    """Solve a 2x2 system with integer solutions."""
    x = rng.randint(1, 8)
    y = rng.randint(1, 8)
    a1, b1 = rng.randint(1, 5), rng.randint(1, 5)
    a2, b2 = rng.randint(1, 5), rng.randint(1, 5)
    # Ensure the system is not degenerate.
    while a1 * b2 == a2 * b1:
        b2 = rng.randint(1, 5)
    c1 = a1 * x + b1 * y
    c2 = a2 * x + b2 * y

    problem = (
        f"Solve the system: ${a1}x + {b1}y = {c1}$ and "
        f"${a2}x + {b2}y = {c2}$. Find $x + y$."
    )
    answer = x + y
    solution = (
        f"Step 1: From equation 1: {a1}x + {b1}y = {c1}\n"
        f"Step 2: From equation 2: {a2}x + {b2}y = {c2}\n"
        f"Step 3: Multiply eq 1 by {b2} and eq 2 by {b1}:\n"
        f"  {a1 * b2}x + {b1 * b2}y = {c1 * b2}\n"
        f"  {a2 * b1}x + {b1 * b2}y = {c2 * b1}\n"
        f"Step 4: Subtract: {a1 * b2 - a2 * b1}x = {c1 * b2 - c2 * b1}\n"
        f"Step 5: x = {(c1 * b2 - c2 * b1) // (a1 * b2 - a2 * b1)} = {x}\n"
        f"Step 6: Substitute back: y = ({c1} - {a1} * {x}) / {b1} = {y}\n"
        f"Step 7: x + y = {x} + {y} = {answer}\n"
        f"\\boxed{{{answer}}}"
    )
    return problem, solution, answer


def _math_arithmetic_sequence(rng: random.Random) -> tuple[str, str, int | str]:
    """Find the sum of an arithmetic sequence."""
    a1 = rng.randint(1, 10)
    d = rng.randint(1, 5)
    n = rng.randint(5, 15)
    an = a1 + (n - 1) * d
    total = n * (a1 + an) // 2

    problem = (
        f"Find the sum of the first {n} terms of the arithmetic sequence "
        f"with first term {a1} and common difference {d}."
    )
    solution = (
        f"Step 1: Find the nth term: a_{n} = {a1} + ({n} - 1) * {d} = {a1} + {(n-1)*d} = {an}\n"
        f"Step 2: Sum formula: S = n * (a_1 + a_n) / 2\n"
        f"Step 3: S = {n} * ({a1} + {an}) / 2 = {n} * {a1 + an} / 2 = {total}\n"
        f"\\boxed{{{total}}}"
    )
    return problem, solution, total


def _math_modular_arithmetic(rng: random.Random) -> tuple[str, str, int | str]:
    """Find remainder when a^b is divided by m."""
    base = rng.randint(2, 9)
    exp = rng.randint(3, 8)
    mod = rng.choice([3, 5, 7, 11, 13])
    result = pow(base, exp, mod)

    problem = f"What is the remainder when ${base}^{{{exp}}}$ is divided by ${mod}$?"

    # Build step-by-step modular exponentiation.
    steps = [f"Step 1: Compute {base}^{exp} mod {mod}"]
    current = base % mod
    steps.append(f"Step 2: {base} mod {mod} = {current}")
    running = current
    for i in range(2, exp + 1):
        running = (running * (base % mod)) % mod
        steps.append(f"Step {i + 1}: {base}^{i} mod {mod} = {running}")

    steps.append(f"\\boxed{{{result}}}")
    solution = "\n".join(steps)
    return problem, solution, result


def _math_gcd_lcm(rng: random.Random) -> tuple[str, str, int | str]:
    """Find GCD or LCM of two numbers."""
    import math
    a = rng.randint(12, 60)
    b = rng.randint(12, 60)
    use_lcm = rng.choice([True, False])

    if use_lcm:
        answer = math.lcm(a, b)
        problem = f"Find the least common multiple of ${a}$ and ${b}$."
        g = math.gcd(a, b)
        solution = (
            f"Step 1: Find GCD({a}, {b})\n"
            f"Step 2: Using Euclidean algorithm: GCD = {g}\n"
            f"Step 3: LCM = {a} * {b} / GCD = {a * b} / {g} = {answer}\n"
            f"\\boxed{{{answer}}}"
        )
    else:
        answer = math.gcd(a, b)
        problem = f"Find the greatest common divisor of ${a}$ and ${b}$."
        solution = (
            f"Step 1: Factor {a} and {b}\n"
            f"Step 2: Using Euclidean algorithm:\n"
        )
        # Euclidean algorithm steps.
        x, y = max(a, b), min(a, b)
        step_num = 3
        while y != 0:
            solution += f"Step {step_num}: {x} = {x // y} * {y} + {x % y}\n"
            x, y = y, x % y
            step_num += 1
        solution += f"Step {step_num}: GCD = {answer}\n\\boxed{{{answer}}}"

    return problem, solution, answer


def _math_percentage_ratio(rng: random.Random) -> tuple[str, str, int | str]:
    """Multi-step percentage/ratio problem."""
    total = rng.choice([100, 120, 150, 200, 250, 300])
    pct_a = rng.choice([20, 25, 30, 40])
    pct_b = rng.choice([10, 15, 20, 25])
    part_a = total * pct_a // 100
    remainder = total - part_a
    part_b = remainder * pct_b // 100
    final = remainder - part_b

    problem = (
        f"A jar has {total} marbles. {pct_a}% are red. Of the remaining marbles, "
        f"{pct_b}% are blue. How many marbles are neither red nor blue?"
    )
    solution = (
        f"Step 1: Red marbles = {pct_a}% of {total} = {total} * {pct_a} / 100 = {part_a}\n"
        f"Step 2: Remaining = {total} - {part_a} = {remainder}\n"
        f"Step 3: Blue marbles = {pct_b}% of {remainder} = {remainder} * {pct_b} / 100 = {part_b}\n"
        f"Step 4: Neither = {remainder} - {part_b} = {final}\n"
        f"\\boxed{{{final}}}"
    )
    return problem, solution, final


def _math_quadratic_integer(rng: random.Random) -> tuple[str, str, int | str]:
    """Quadratic with integer roots: find sum or product of roots."""
    r1 = rng.randint(-8, 8)
    r2 = rng.randint(-8, 8)
    # x^2 - (r1+r2)x + r1*r2 = 0
    b = -(r1 + r2)
    c = r1 * r2
    ask_sum = rng.choice([True, False])

    if ask_sum:
        answer = r1 + r2
        question_part = "the sum of its solutions"
    else:
        answer = r1 * r2
        question_part = "the product of its solutions"

    bstr = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    cstr = f"+ {c}" if c >= 0 else f"- {abs(c)}"
    problem = f"If $x^2 {bstr}x {cstr} = 0$, what is {question_part}?"

    solution = (
        f"Step 1: For x^2 {bstr}x {cstr} = 0, by Vieta's formulas:\n"
        f"Step 2: Sum of roots = {-b} = {r1 + r2}\n"
        f"Step 3: Product of roots = {c} = {r1 * r2}\n"
        f"Step 4: Answer = {answer}\n"
        f"\\boxed{{{answer}}}"
    )
    return problem, solution, answer


def _math_divisibility(rng: random.Random) -> tuple[str, str, int | str]:
    """Find the largest N-digit number divisible by k."""
    n_digits = rng.choice([2, 3])
    k = rng.choice([3, 4, 6, 7, 8, 9])
    upper = 10**n_digits - 1
    answer = upper - (upper % k)

    problem = (
        f"What is the largest {n_digits}-digit number that is divisible by ${k}$?"
    )
    solution = (
        f"Step 1: The largest {n_digits}-digit number is {upper}\n"
        f"Step 2: {upper} divided by {k} gives remainder {upper % k}\n"
        f"Step 3: Subtract the remainder: {upper} - {upper % k} = {answer}\n"
        f"\\boxed{{{answer}}}"
    )
    return problem, solution, answer


def _math_linear_combo(rng: random.Random) -> tuple[str, str, int | str]:
    """Evaluate a linear expression at a given point."""
    a = rng.randint(2, 10)
    b = rng.randint(1, 10)
    c = rng.randint(1, 10)
    x_val = rng.randint(2, 8)
    result = a * x_val**2 + b * x_val + c

    problem = f"If $f(x) = {a}x^2 + {b}x + {c}$, find $f({x_val})$."
    solution = (
        f"Step 1: f({x_val}) = {a} * {x_val}^2 + {b} * {x_val} + {c}\n"
        f"Step 2: = {a} * {x_val**2} + {b} * {x_val} + {c}\n"
        f"Step 3: = {a * x_val**2} + {b * x_val} + {c}\n"
        f"Step 4: = {result}\n"
        f"\\boxed{{{result}}}"
    )
    return problem, solution, result


def _math_absolute_value(rng: random.Random) -> tuple[str, str, int | str]:
    """Solve |ax + b| = c, find the sum of all solutions."""
    # Pick integer solutions first, then derive a, b, c to guarantee integrality.
    # |ax + b| = c  →  x = (c - b)/a  or  x = (-c - b)/a
    # For both to be integers, we need (c - b) % a == 0 and (-c - b) % a == 0.
    # This requires 2c % a == 0, so: a=1 always works; a=2 needs c even; a=3 needs c divisible by 3.
    a = rng.choice([1, 2, 3])
    if a == 1:
        c = rng.randint(3, 15)
    elif a == 2:
        c = rng.choice([4, 6, 8, 10, 12, 14])
    else:
        c = rng.choice([3, 6, 9, 12, 15])
    # b must satisfy b ≡ c (mod a). Pick from valid values in range.
    valid_b = [bb for bb in range(-10, 11) if (c - bb) % a == 0 and (-c - bb) % a == 0]
    b = rng.choice(valid_b)

    x1 = (c - b) // a
    x2 = (-c - b) // a
    answer = x1 + x2

    bstr = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    if a == 1:
        expr = f"x {bstr}"
    else:
        expr = f"{a}x {bstr}"

    problem = f"Solve $|{expr}| = {c}$. Find the sum of all solutions."
    solution = (
        f"Step 1: |{expr}| = {c} means:\n"
        f"  Case 1: {a}x {bstr} = {c}\n"
        f"  Case 2: {a}x {bstr} = -{c}\n"
        f"Step 2: Case 1: {a}x = {c} - {b} = {c - b}, so x = {x1}\n"
        f"Step 3: Case 2: {a}x = -{c} - {b} = {-c - b}, so x = {x2}\n"
        f"Step 4: Sum = {x1} + {x2} = {answer}\n"
        f"\\boxed{{{answer}}}"
    )
    return problem, solution, answer


# ---------------------------------------------------------------------------
# 3. Answer extraction
# ---------------------------------------------------------------------------


def extract_final_answer(text: str) -> str | None:
    """Extract the final answer from an LLM response to a MATH problem.

    **Detailed explanation for engineers:**
        MATH answers can be numbers, fractions, or algebraic expressions.
        We look for common patterns in priority order:
        1. \\boxed{...} (MATH convention)
        2. "Answer: <value>" or "answer is <value>"
        3. Last number or simple expression found

        Returns the answer as a string (not necessarily numeric) since MATH
        answers can be expressions like "3/4" or "2\\sqrt{3}".
    """
    # Try \boxed{} first.
    boxed = _extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    # Try "Answer: <value>" pattern.
    match = re.search(r"[Aa]nswer[:\s]+(.+?)(?:\n|$)", text)
    if match:
        ans = match.group(1).strip().rstrip(".")
        if ans:
            return ans

    # Try "= <final value>" at end of a line.
    match = re.search(r"=\s*(-?[\d./]+)\s*$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Fall back to last number in text.
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    if numbers:
        return numbers[-1]

    return None


def normalize_answer(answer: str) -> str:
    """Normalize a MATH answer for comparison.

    **Detailed explanation for engineers:**
        MATH answers can be in various formats: "3", "3.0", "\\frac{3}{4}",
        "0.75", "-3", etc. We normalize to a canonical form for comparison.
        For integer-like values, strip trailing .0. For fractions, evaluate
        if possible. For LaTeX fractions, parse and evaluate.
    """
    s = answer.strip()

    # Strip LaTeX formatting.
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\,", "").replace(" ", "")

    # Handle \frac{a}{b}.
    frac_match = re.match(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if frac_match:
        num = int(frac_match.group(1))
        den = int(frac_match.group(2))
        if den != 0:
            if num % den == 0:
                return str(num // den)
            return f"{num}/{den}"

    # Handle a/b.
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                num = int(parts[0])
                den = int(parts[1])
                if den != 0 and num % den == 0:
                    return str(num // den)
            except ValueError:
                pass

    # Handle trailing .0.
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return s
    except ValueError:
        pass

    return s


def answers_match(predicted: str | None, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth after normalization.

    **Detailed explanation for engineers:**
        Both answers are normalized (LaTeX stripped, fractions evaluated,
        trailing .0 removed) then compared as strings. Also tries numeric
        comparison for answers that parse as numbers.
    """
    if predicted is None:
        return False

    norm_pred = normalize_answer(predicted)
    norm_gt = normalize_answer(ground_truth)

    if norm_pred == norm_gt:
        return True

    # Try numeric comparison.
    try:
        # Handle fraction strings.
        def to_float(s: str) -> float:
            if "/" in s:
                parts = s.split("/")
                return float(parts[0]) / float(parts[1])
            return float(s)

        return abs(to_float(norm_pred) - to_float(norm_gt)) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False


# ---------------------------------------------------------------------------
# 4. Step-level parsing and verification
# ---------------------------------------------------------------------------


def parse_reasoning_steps(response: str) -> list[dict[str, Any]]:
    """Parse an LLM response into individual reasoning steps.

    **Detailed explanation for engineers:**
        LLM chain-of-thought responses contain multiple reasoning steps.
        We split on common step delimiters:
        1. "Step N:" or "Step N." patterns
        2. Numbered lines like "1.", "2.", etc.
        3. Lines starting with "First", "Then", "Next", "Finally"
        4. Blank-line-separated paragraphs as a fallback

        Each step is returned with its index, raw text, and any arithmetic
        expressions found within it.
    """
    lines = response.strip().split("\n")
    steps: list[dict[str, Any]] = []

    # Try "Step N:" pattern first.
    step_pattern = re.compile(r"^(?:Step\s+)?(\d+)[.):]\s*(.*)", re.IGNORECASE)
    current_step: dict[str, Any] | None = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = step_pattern.match(line)
        if match:
            if current_step is not None:
                steps.append(current_step)
            current_step = {
                "index": int(match.group(1)),
                "text": match.group(2).strip(),
            }
        elif current_step is not None:
            # Continuation of the current step.
            current_step["text"] += " " + line
        else:
            # Try keyword-based splitting.
            keyword_match = re.match(
                r"^(First|Then|Next|Also|Finally|Therefore|So|Thus)[,:]?\s+(.*)",
                line, re.IGNORECASE,
            )
            if keyword_match:
                if current_step is not None:
                    steps.append(current_step)
                current_step = {
                    "index": len(steps) + 1,
                    "text": line,
                }
            elif current_step is None:
                # Start a new implicit step.
                current_step = {
                    "index": len(steps) + 1,
                    "text": line,
                }

    if current_step is not None:
        steps.append(current_step)

    # If no steps were parsed, treat the whole response as one step.
    if not steps:
        steps = [{"index": 1, "text": response.strip()}]

    return steps


def extract_step_constraints(step_text: str) -> list[dict[str, Any]]:
    """Extract arithmetic constraints from a single reasoning step.

    **Detailed explanation for engineers:**
        For each step, we look for arithmetic expressions that can be verified:
        - "X + Y = Z", "X - Y = Z", "X * Y = Z", "X / Y = Z"
        - "X = Y" (assignment/equality claims)
        - "X^2 = Y" or "X squared = Y"

        Each constraint is verified by direct computation and returned with
        a 'satisfied' flag indicating whether the claim holds.

        This is the core of step-level verification: rather than just checking
        the final answer, we verify EVERY arithmetic claim made at each step.
    """
    constraints: list[dict[str, Any]] = []

    # Pattern: "A op B = C" for +, -, *, /, ×, ÷
    arith_pattern = re.compile(
        r"(-?[\d,]+(?:\.\d+)?)\s*"
        r"([+\-*/×x÷])\s*"
        r"(-?[\d,]+(?:\.\d+)?)\s*"
        r"=\s*(-?[\d,]+(?:\.\d+)?)"
    )

    for match in arith_pattern.finditer(step_text):
        a_str = match.group(1).replace(",", "")
        op = match.group(2)
        b_str = match.group(3).replace(",", "")
        claimed_str = match.group(4).replace(",", "")

        try:
            a = float(a_str)
            b = float(b_str)
            claimed = float(claimed_str)
        except ValueError:
            continue

        if op in ("×", "x"):
            op = "*"
        if op == "÷":
            op = "/"

        if op == "+":
            correct = a + b
        elif op == "-":
            correct = a - b
        elif op == "*":
            correct = a * b
        elif op == "/" and b != 0:
            correct = a / b
        else:
            continue

        # Normalize to int if appropriate.
        if a == int(a) and b == int(b) and claimed == int(claimed):
            a, b, claimed = int(a), int(b), int(claimed)
            correct = int(correct) if correct == int(correct) else correct

        satisfied = abs(claimed - correct) < 0.01

        constraints.append({
            "expression": f"{a} {op} {b}",
            "claimed": claimed,
            "correct": correct,
            "satisfied": satisfied,
            "raw_match": match.group(0),
        })

    # Pattern: "A^2 = B" or "A squared = B"
    sq_pattern = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*(?:\^2|\*\*2|squared)\s*=\s*(-?\d+(?:\.\d+)?)"
    )
    for match in sq_pattern.finditer(step_text):
        try:
            base = float(match.group(1))
            claimed = float(match.group(2))
        except ValueError:
            continue
        correct = base ** 2
        satisfied = abs(claimed - correct) < 0.01
        constraints.append({
            "expression": f"{int(base) if base == int(base) else base}^2",
            "claimed": int(claimed) if claimed == int(claimed) else claimed,
            "correct": int(correct) if correct == int(correct) else correct,
            "satisfied": satisfied,
            "raw_match": match.group(0),
        })

    return constraints


def verify_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify all reasoning steps and find the first error.

    **Detailed explanation for engineers:**
        Iterates through each parsed reasoning step, extracts arithmetic
        constraints, and checks if each constraint is satisfied. Tracks:
        - first_error_step: index of the step where the first error appears
        - total_constraints: how many arithmetic claims were extractable
        - total_violations: how many of those claims were wrong
        - steps_after_first_error: how many steps come after the first error
          (measuring error propagation distance)
        - per_step: detailed results for each step

        The key metric is error propagation: if step 2 has an error and there
        are 5 total steps, the propagation distance is 3 (steps 3-5 are
        potentially corrupted by the step 2 error).
    """
    per_step: list[dict[str, Any]] = []
    first_error_step: int | None = None
    total_constraints = 0
    total_violations = 0

    for step in steps:
        constraints = extract_step_constraints(step["text"])
        step_violations = sum(1 for c in constraints if not c["satisfied"])

        total_constraints += len(constraints)
        total_violations += step_violations

        if step_violations > 0 and first_error_step is None:
            first_error_step = step["index"]

        per_step.append({
            "step_index": step["index"],
            "text_preview": step["text"][:100],
            "n_constraints": len(constraints),
            "n_violations": step_violations,
            "constraints": constraints,
        })

    n_steps = len(steps)
    if first_error_step is not None:
        steps_after_error = n_steps - first_error_step
    else:
        steps_after_error = 0

    return {
        "n_steps": n_steps,
        "total_constraints": total_constraints,
        "total_violations": total_violations,
        "first_error_step": first_error_step,
        "steps_after_first_error": steps_after_error,
        "per_step": per_step,
    }


# ---------------------------------------------------------------------------
# 5. Model loading (reused from Exp 91 pattern)
# ---------------------------------------------------------------------------


def load_model(config: dict[str, Any]) -> tuple[Any, Any, str, bool]:
    """Load a HuggingFace model, trying candidate names in order.

    **Detailed explanation for engineers:**
        Same approach as Exp 91: iterates config["candidates"], loads via
        transformers AutoModelForCausalLM, runs a smoke test. Returns
        (tokenizer, model, device, loaded_ok). Falls back to simulation
        on failure.
    """
    if os.environ.get("CARNOT_SKIP_LLM", ""):
        print(f"    CARNOT_SKIP_LLM set — skipping {config['name']}.")
        return None, None, "cpu", False

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"    torch/transformers not available: {e}")
        return None, None, "cpu", False

    force_cpu = os.environ.get("CARNOT_FORCE_CPU", "1") == "1"
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    trust = config.get("trust_remote_code", True)

    for model_name in config["candidates"]:
        try:
            print(f"    Loading {model_name} on {device}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=trust,
                torch_dtype=torch.float16 if device == "cuda" else None,
            )
            if device == "cuda":
                model = model.cuda()
            model.eval()
            print(f"    Loaded {model_name}. Running smoke test...")

            try:
                test_input = tokenizer("Hi", return_tensors="pt")
                if device == "cuda":
                    test_input = {k: v.cuda() for k, v in test_input.items()}
                with torch.no_grad():
                    _ = model.generate(
                        **test_input, max_new_tokens=4, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                print(f"    Smoke test passed.")
                return tokenizer, model, device, True
            except Exception as e:
                print(f"    Smoke test failed: {e}")

            del model, tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"    Failed to load {model_name}: {e}")

    return None, None, "cpu", False


def unload_model(model: Any, tokenizer: Any, device: str) -> None:
    """Free model memory so the next model can load."""
    del model, tokenizer
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# 6. LLM generation
# ---------------------------------------------------------------------------


def generate_response(
    prompt: str,
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int = 512,
) -> str:
    """Generate a response from a loaded HuggingFace causal LM.

    **Detailed explanation for engineers:**
        Uses greedy decoding (do_sample=False) for reproducibility. Applies
        the model's chat template if available. max_new_tokens is 512 for
        MATH problems (multi-step solutions need more tokens than GSM8K).
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Strip thinking tokens if present (Qwen models).
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return response


# ---------------------------------------------------------------------------
# 7. Simulated LLM responses (fallback when models can't load)
# ---------------------------------------------------------------------------


def simulate_math_response(
    problem: dict[str, Any],
    model_name: str,
    rng: random.Random | None = None,
) -> str:
    """Generate a simulated LLM response for a MATH problem.

    **Detailed explanation for engineers:**
        When the LLM can't be loaded, this simulates chain-of-thought responses
        with realistic error rates for multi-step algebra problems.

        Error modes for MATH are different from GSM8K:
        - Algebraic manipulation errors (40%): wrong sign, wrong coefficient
        - Arithmetic errors (30%): wrong intermediate computation
        - Conceptual errors (20%): wrong formula or approach
        - Transcription errors (10%): correct work but wrong final answer

        The simulation produces step-by-step text with explicit arithmetic
        claims so the step-level verifier has material to check.
    """
    if rng is None:
        rng = random.Random(42)

    gt = problem["ground_truth"]

    # Model-specific base error rates (higher than GSM8K — MATH is harder).
    if "qwen" in model_name.lower():
        base_error = 0.45
    else:
        base_error = 0.38

    is_correct = rng.random() > base_error

    if is_correct:
        # Produce correct step-by-step solution.
        return _simulate_correct_steps(gt, rng)
    else:
        error_type = rng.choices(
            ["algebraic", "arithmetic", "conceptual", "transcription"],
            weights=[40, 30, 20, 10],
            k=1,
        )[0]
        return _simulate_error_steps(gt, error_type, rng)


def _simulate_correct_steps(gt: str, rng: random.Random) -> str:
    """Produce a simulated correct chain-of-thought."""
    try:
        gt_num = int(gt)
    except ValueError:
        try:
            gt_num = float(gt)
        except ValueError:
            return f"Step 1: Working through the problem.\nStep 2: The answer is {gt}.\n\\boxed{{{gt}}}"

    # Multi-step arithmetic that arrives at the correct answer.
    mid = rng.randint(max(1, abs(gt_num) // 3), max(2, abs(gt_num) // 2 + 1))
    remainder = gt_num - mid

    return (
        f"Step 1: First, I'll break this into parts.\n"
        f"Step 2: The first component gives us {mid}.\n"
        f"Step 3: The second component is {gt_num} - {mid} = {remainder}.\n"
        f"Step 4: Combining: {mid} + {remainder} = {gt_num}.\n"
        f"Step 5: Therefore the answer is {gt_num}.\n"
        f"\\boxed{{{gt_num}}}"
    )


def _simulate_error_steps(gt: str, error_type: str, rng: random.Random) -> str:
    """Produce a simulated chain-of-thought with a specific error type.

    **Detailed explanation for engineers:**
        Each error type inserts the error at a different step:
        - algebraic: error in step 2 (wrong manipulation)
        - arithmetic: error in step 3 (wrong computation)
        - conceptual: error in step 1 (wrong approach)
        - transcription: all steps correct but boxed answer is wrong
    """
    try:
        gt_num = int(gt)
    except ValueError:
        try:
            gt_num = float(gt)
        except ValueError:
            wrong = str(gt) + "_wrong"
            return f"Step 1: Working through the problem.\nStep 2: Answer is {wrong}.\n\\boxed{{{wrong}}}"

    mid = max(1, abs(gt_num) // 3)
    remainder = gt_num - mid

    if error_type == "algebraic":
        # Error in step 2: wrong sign or coefficient.
        wrong_mid = mid + rng.choice([-3, -2, -1, 1, 2, 3])
        wrong_total = wrong_mid + remainder  # Propagates.
        return (
            f"Step 1: Break the problem into parts.\n"
            f"Step 2: The first part evaluates to {wrong_mid}.\n"
            f"Step 3: The second part is {remainder}.\n"
            f"Step 4: Total = {wrong_mid} + {remainder} = {wrong_total}.\n"
            f"\\boxed{{{wrong_total}}}"
        )
    elif error_type == "arithmetic":
        # Error in step 3: wrong arithmetic.
        wrong_sum = mid + remainder + rng.choice([-5, -2, 2, 5])
        return (
            f"Step 1: Identify the components.\n"
            f"Step 2: Component A = {mid}, Component B = {remainder}.\n"
            f"Step 3: A + B = {mid} + {remainder} = {wrong_sum}.\n"
            f"Step 4: The answer is {wrong_sum}.\n"
            f"\\boxed{{{wrong_sum}}}"
        )
    elif error_type == "conceptual":
        # Error in step 1: wrong formula.
        wrong = gt_num * rng.choice([2, 3]) + rng.randint(-10, 10)
        return (
            f"Step 1: Using the formula, we compute directly.\n"
            f"Step 2: Result = {wrong}.\n"
            f"\\boxed{{{wrong}}}"
        )
    else:
        # Transcription: correct work, wrong boxed answer.
        wrong_final = gt_num + rng.choice([-1, 1])
        return (
            f"Step 1: Break into parts.\n"
            f"Step 2: First part = {mid}.\n"
            f"Step 3: Second part = {remainder}.\n"
            f"Step 4: Total = {mid} + {remainder} = {gt_num}.\n"
            f"\\boxed{{{wrong_final}}}"
        )


# ---------------------------------------------------------------------------
# 8. Pipeline verification (reused from Exp 91)
# ---------------------------------------------------------------------------


def verify_with_pipeline(problem: str, response: str) -> dict[str, Any]:
    """Run Carnot VerifyRepairPipeline.verify() on a response.

    **Detailed explanation for engineers:**
        Uses the library-level VerifyRepairPipeline in verify-only mode.
        Extracts arithmetic constraints from the response text using the
        AutoExtractor (all domains, not just arithmetic — MATH problems
        can contain code and logical claims too).
    """
    try:
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline(
            model=None,
            domains=["arithmetic"],
            timeout_seconds=30.0,
        )
        vr = pipeline.verify(problem, response, domain="arithmetic")
        return {
            "verified": vr.verified,
            "n_constraints": len(vr.constraints),
            "n_violations": len(vr.violations),
            "energy": vr.energy,
            "error": None,
        }
    except Exception as e:
        return {
            "verified": False,
            "n_constraints": 0,
            "n_violations": 0,
            "energy": 0.0,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 9. Run a single problem through all analyses
# ---------------------------------------------------------------------------


def run_problem(
    problem: dict[str, Any],
    model_name: str,
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live: bool = False,
    sim_rng: random.Random | None = None,
) -> dict[str, Any]:
    """Run a single MATH problem: generate, parse steps, verify step-by-step.

    **Detailed explanation for engineers:**
        This is the core of Experiment 92. For each problem:
        1. Generate a chain-of-thought response (live or simulated).
        2. Extract the final answer and check against ground truth.
        3. Parse the response into individual reasoning steps.
        4. For EACH step, extract arithmetic constraints and verify them.
        5. Find the first step with an error (if any).
        6. Compare step-level error detection vs final-answer-only.
        7. Run the Carnot pipeline for additional verification.

        The key comparison metric: does step-level verification flag errors
        that final-answer-only verification misses? Specifically, does it
        identify errors EARLIER in the chain (before the final answer)?
    """
    t0 = time.time()
    prompt = (
        f"Solve step by step: {problem['problem']}\n"
        f"Show all your work. Put your final answer in \\boxed{{}}."
    )

    if use_live:
        response = generate_response(prompt, tokenizer, model, device)
    else:
        response = simulate_math_response(problem, model_name, rng=sim_rng)

    elapsed = time.time() - t0

    # Final answer extraction and comparison.
    extracted_answer = extract_final_answer(response)
    final_correct = answers_match(extracted_answer, problem["ground_truth"])

    # Step-level parsing and verification.
    steps = parse_reasoning_steps(response)
    step_verification = verify_steps(steps)

    # Pipeline verification.
    pipeline_result = verify_with_pipeline(problem["problem"], response)

    # Determine if step-level caught something final-answer-only missed.
    # "final-answer-only detects error" = not final_correct
    # "step-level detects error" = step_verification has violations
    step_detected = step_verification["total_violations"] > 0
    final_detected = not final_correct

    detection_comparison = "both_clean"
    if step_detected and final_detected:
        detection_comparison = "both_flagged"
    elif step_detected and not final_detected:
        # Step-level caught an error but final answer is correct —
        # this means intermediate errors cancelled out (rare but interesting).
        detection_comparison = "step_only"
    elif not step_detected and final_detected:
        # Final answer is wrong but no step-level arithmetic errors found —
        # the error is conceptual/logical, not arithmetic.
        detection_comparison = "final_only"

    return {
        "problem": problem["problem"][:200],
        "ground_truth": problem["ground_truth"],
        "level": problem.get("level", ""),
        "subject": problem.get("subject", ""),
        "source": problem.get("source", ""),
        "extracted_answer": extracted_answer,
        "final_correct": final_correct,
        "n_steps": step_verification["n_steps"],
        "total_constraints": step_verification["total_constraints"],
        "total_violations": step_verification["total_violations"],
        "first_error_step": step_verification["first_error_step"],
        "steps_after_first_error": step_verification["steps_after_first_error"],
        "step_detected": step_detected,
        "final_detected": final_detected,
        "detection_comparison": detection_comparison,
        "pipeline_verified": pipeline_result["verified"],
        "pipeline_n_constraints": pipeline_result["n_constraints"],
        "pipeline_n_violations": pipeline_result["n_violations"],
        "pipeline_energy": pipeline_result["energy"],
        "time_s": elapsed,
        "response_preview": response[:300],
    }


# ---------------------------------------------------------------------------
# 10. Results saving
# ---------------------------------------------------------------------------


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics from per-problem results.

    **Detailed explanation for engineers:**
        Five categories of metrics:
        1. End-to-end accuracy: fraction of problems with correct final answer.
        2. Step-level error detection: what fraction of wrong answers had
           step-level violations flagged? How early was the first error?
        3. Error propagation: average number of steps between first error
           and final answer (measures how far errors compound).
        4. Coverage: what fraction of steps had extractable constraints?
        5. Detection comparison: how often does step-level verification
           catch something that final-answer-only misses (and vice versa)?
    """
    n = len(results)
    if n == 0:
        return {}

    # 1. End-to-end accuracy.
    n_correct = sum(1 for r in results if r["final_correct"])
    accuracy = n_correct / n

    # 2. Step-level error detection.
    wrong_results = [r for r in results if not r["final_correct"]]
    n_wrong = len(wrong_results)
    n_step_detected_when_wrong = sum(1 for r in wrong_results if r["step_detected"])
    step_detection_rate = n_step_detected_when_wrong / n_wrong if n_wrong else 0.0

    # Average first-error step (among problems where an error was detected).
    first_error_steps = [
        r["first_error_step"] for r in results
        if r["first_error_step"] is not None
    ]
    avg_first_error_step = float(np.mean(first_error_steps)) if first_error_steps else 0.0

    # 3. Error propagation.
    propagation_distances = [
        r["steps_after_first_error"] for r in results
        if r["first_error_step"] is not None
    ]
    avg_propagation = float(np.mean(propagation_distances)) if propagation_distances else 0.0

    # 4. Coverage.
    total_steps = sum(r["n_steps"] for r in results)
    steps_with_constraints = sum(
        sum(1 for s in range(r["n_steps"]) if r["total_constraints"] > 0)
        for r in results
    )
    # More precise: fraction of problems where at least one constraint was extracted.
    problems_with_constraints = sum(1 for r in results if r["total_constraints"] > 0)
    constraint_coverage = problems_with_constraints / n

    total_constraints = sum(r["total_constraints"] for r in results)
    total_violations = sum(r["total_violations"] for r in results)

    # 5. Detection comparison.
    comparison_counts = {
        "both_clean": sum(1 for r in results if r["detection_comparison"] == "both_clean"),
        "both_flagged": sum(1 for r in results if r["detection_comparison"] == "both_flagged"),
        "step_only": sum(1 for r in results if r["detection_comparison"] == "step_only"),
        "final_only": sum(1 for r in results if r["detection_comparison"] == "final_only"),
    }

    return {
        "n_problems": n,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "n_wrong": n_wrong,
        "step_detection_rate": step_detection_rate,
        "avg_first_error_step": avg_first_error_step,
        "avg_error_propagation": avg_propagation,
        "constraint_coverage": constraint_coverage,
        "total_constraints": total_constraints,
        "total_violations": total_violations,
        "total_steps": total_steps,
        "detection_comparison": comparison_counts,
    }


def save_results(
    all_results: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
) -> Path:
    """Save full results to JSON.

    **Detailed explanation for engineers:**
        Writes the complete results dict to results/experiment_92_results.json.
        Strips the 'response_preview' field from individual results to keep the
        file size manageable. Includes per-model metrics and cross-model
        comparison.
    """
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / "experiment_92_results.json"

    # Strip verbose response previews.
    compact: dict[str, Any] = {}
    for model_name, entries in all_results.items():
        compact[model_name] = {
            "metrics": compute_metrics(entries),
            "per_problem": [
                {k: v for k, v in e.items() if k != "response_preview"}
                for e in entries
            ],
        }

    output = {
        "experiment": 92,
        "title": "MATH Benchmark — Step-Level Constraint Verification",
        "metadata": metadata,
        "results": compact,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {path}")
    return path


# ---------------------------------------------------------------------------
# 11. Main benchmark
# ---------------------------------------------------------------------------


def main() -> int:
    """Run MATH benchmark: 100 problems, 2 models, step-level verification."""
    sep = "=" * 78
    print(sep)
    print("EXPERIMENT 92: MATH Benchmark — Step-Level Constraint Verification")
    print("  Multi-step reasoning: verify EACH step, not just the conclusion")
    print("  100 problems × 2 models (Qwen3.5-0.8B, Gemma4-E4B-it)")
    print("  Key question: does step-level verification catch errors earlier?")
    print(sep)

    overall_start = time.time()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # --- Load MATH problems ---
    print("\n[1/3] Loading MATH problems...")
    problems = load_math_problems(n=100, seed=92)
    n_real = sum(1 for p in problems if p.get("source") == "math")
    n_synth = sum(1 for p in problems if p.get("source") == "synthetic")
    print(f"  Problems: {len(problems)} ({n_real} real MATH, {n_synth} synthetic)")

    # --- Run benchmark per model ---
    all_results: dict[str, list[dict[str, Any]]] = {}
    model_metadata: dict[str, dict[str, Any]] = {}
    n_problems = len(problems)

    for mi, config in enumerate(MODEL_CONFIGS):
        model_name = config["name"]
        print(f"\n[2/3] Model {mi + 1}/{len(MODEL_CONFIGS)}: {model_name}")

        # Load model.
        print(f"  Loading {model_name}...")
        tokenizer, model, device, use_live = load_model(config)

        if not use_live:
            print(f"  *** FALLBACK: Using simulated outputs for {model_name} ***")

        model_metadata[model_name] = {"live": use_live, "device": device}

        model_results: list[dict[str, Any]] = []
        model_start = time.time()

        for qi, prob in enumerate(problems):
            seed_base = 92_000 + mi * 10_000 + qi

            result = run_problem(
                prob, model_name,
                tokenizer=tokenizer, model=model, device=device,
                use_live=use_live,
                sim_rng=random.Random(seed_base),
            )
            model_results.append(result)

            if (qi + 1) % 25 == 0:
                n_c = sum(1 for r in model_results if r["final_correct"])
                n_det = sum(1 for r in model_results if r["step_detected"])
                print(f"    {qi + 1}/{n_problems} — "
                      f"correct {n_c}/{qi + 1}, step-errors-detected {n_det}/{qi + 1}")

        model_elapsed = time.time() - model_start
        model_metadata[model_name]["time_s"] = model_elapsed
        all_results[model_name] = model_results

        # Print model summary.
        metrics = compute_metrics(model_results)
        dc = metrics["detection_comparison"]
        print(f"\n  {model_name} summary ({model_elapsed:.1f}s):")
        print(f"    Accuracy: {metrics['n_correct']}/{n_problems} ({metrics['accuracy']:.1%})")
        print(f"    Step detection rate (when wrong): {metrics['step_detection_rate']:.1%}")
        print(f"    Avg first error at step: {metrics['avg_first_error_step']:.1f}")
        print(f"    Avg error propagation: {metrics['avg_error_propagation']:.1f} steps")
        print(f"    Constraint coverage: {metrics['constraint_coverage']:.1%}")
        print(f"    Detection comparison: both_flagged={dc['both_flagged']}, "
              f"step_only={dc['step_only']}, final_only={dc['final_only']}, "
              f"both_clean={dc['both_clean']}")

        # Free model memory.
        if use_live:
            unload_model(model, tokenizer, device)

    # --- Save results ---
    total_elapsed = time.time() - overall_start

    metadata = {
        "timestamp": timestamp,
        "n_problems": n_problems,
        "n_real_math": n_real,
        "n_synthetic": n_synth,
        "dataset_source": "MATH test" if n_real > 0 else "synthetic",
        "total_time_s": total_elapsed,
        "models": model_metadata,
    }

    print(f"\n[3/3] Saving results...")
    results_path = save_results(all_results, metadata)

    # --- Final cross-model comparison ---
    print(f"\n{sep}")
    print(f"EXPERIMENT 92 FINAL RESULTS ({total_elapsed:.1f}s)")
    print(sep)

    print(f"\n  {'Model':<20s} {'Accuracy':>10s} {'StepDet':>10s} "
          f"{'AvgErr@':>10s} {'Propag':>10s} {'Coverage':>10s}")
    print(f"  {'-' * 72}")

    for model_name, entries in all_results.items():
        live = model_metadata[model_name]["live"]
        tag = " (LIVE)" if live else " (SIM)"
        m = compute_metrics(entries)
        print(f"  {model_name + tag:<20s} "
              f"{m['accuracy']:>9.1%} "
              f"{m['step_detection_rate']:>9.1%} "
              f"{m['avg_first_error_step']:>9.1f} "
              f"{m['avg_error_propagation']:>9.1f} "
              f"{m['constraint_coverage']:>9.1%}")

    # Key insight summary.
    print(f"\n  KEY INSIGHT: Step-level vs Final-answer-only verification")
    for model_name, entries in all_results.items():
        m = compute_metrics(entries)
        dc = m["detection_comparison"]
        total_errors = dc["both_flagged"] + dc["step_only"] + dc["final_only"]
        if total_errors > 0:
            step_catches = dc["both_flagged"] + dc["step_only"]
            final_only_catches = dc["final_only"]
            print(f"  {model_name}:")
            print(f"    Step-level catches: {step_catches}/{total_errors} errors "
                  f"({step_catches/total_errors:.0%})")
            print(f"    Final-only catches: {final_only_catches}/{total_errors} "
                  f"(errors invisible to step-level, likely conceptual)")
            if dc["step_only"] > 0:
                print(f"    Step-ONLY catches: {dc['step_only']} "
                      f"(intermediate errors that cancelled out in final answer)")

    # Dataset info.
    print(f"\n  Dataset: {'REAL MATH' if n_real > 0 else 'Synthetic fallback'} "
          f"({n_real} real, {n_synth} synthetic)")

    any_live = any(m_info["live"] for m_info in model_metadata.values())
    if any_live and n_real > 0:
        print(f"  VERDICT: Step-level verification on real MATH dataset with real models.")
    elif any_live:
        print(f"  VERDICT: Real model inference, synthetic data fallback.")
    else:
        print(f"  VERDICT: Simulated run — pipeline logic exercised,")
        print(f"  but numbers are not externally credible.")

    print(f"\n  Architecture: LLM → chain-of-thought → step parsing →")
    print(f"  per-step constraint extraction → step-level verification →")
    print(f"  error localization + propagation tracking")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
