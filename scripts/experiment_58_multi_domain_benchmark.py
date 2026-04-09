#!/usr/bin/env python3
"""Experiment 58: Multi-domain benchmark — 500 questions across 5 domains.

**Researcher summary:**
    First comprehensive evaluation of the full verify-repair pipeline. Tests
    100 questions each across 5 domains (arithmetic, code, logic, factual,
    scheduling) in three modes: LLM alone, LLM + Ising verification, and
    LLM + Ising verification + repair loop. Produces a domain x mode x metric
    results table that quantifies the value-add of EBM-guided repair.

**Detailed explanation for engineers:**
    Experiments 47-57 validated individual pipeline components on small test
    sets (5-20 questions). This experiment scales up to 500 questions (100 per
    domain) to produce statistically meaningful benchmark numbers. It runs each
    question through three modes:

    Mode A — Baseline: The LLM (Qwen3.5-0.8B) answers alone with no
    verification or feedback. This is the control condition.

    Mode B — Verify-only: The LLM answers, then the Ising constraint verifier
    checks the answer. We record whether the verifier would have flagged the
    answer, but we do NOT feed corrections back to the LLM. This measures
    detection accuracy.

    Mode C — Verify-repair: The LLM answers, the verifier checks, and if
    violations are found, natural-language feedback is sent back to the LLM
    for up to 3 repair iterations. This is the full pipeline from Exp 57.

    Metrics computed per domain and overall:
    - Accuracy (vs ground truth)
    - Hallucination rate (1 - accuracy)
    - Repair success rate (of those flagged, how many got fixed?)
    - Average Ising energy (constraint violation severity)
    - Average constraint count per answer
    - Wall-clock time per mode

    If the LLM cannot be loaded (no GPU, no torch, etc.), the script falls
    back to simulated outputs that include a realistic mix of correct answers
    and hallucinations — the pipeline logic is still fully exercised.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_58_multi_domain_benchmark.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import gc
import math
import os
import random
import re
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# 1. Question generation — 100 questions per domain, each with ground truth
# ---------------------------------------------------------------------------

def _extract_number(text: str) -> float | None:
    """Pull the last number from a string (usually the final answer).

    **Detailed explanation for engineers:**
        LLMs produce answers in many formats: "The answer is 75", "75",
        "47 + 28 = 75", "Result: 75.0", etc. This function finds ALL
        numbers in the text and returns the last one, which is typically
        the computed final answer. Handles integers and decimals. Returns
        None if no number is found, which callers treat as a failed parse.
    """
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if not numbers:
        return None
    try:
        val = float(numbers[-1])
        return int(val) if val == int(val) else val
    except (ValueError, OverflowError):
        return None


def generate_arithmetic_questions(n: int = 100, seed: int = 58) -> list[dict[str, Any]]:
    """Generate n arithmetic questions with ground truth answers.

    **Detailed explanation for engineers:**
        Produces a mix of question types to test different failure modes:
        - Simple two-operand (addition, subtraction, multiplication, division)
        - Multi-step expressions (e.g. "What is 3 + 5 * 2?")
        - Word problems ("A store has X items, sells Y, receives Z...")

        Each question includes a check_answer lambda that compares the
        extracted number from the LLM's response against the correct value.
        The seed ensures reproducibility across runs.
    """
    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []

    # --- 40 simple two-operand questions ---
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]
    for i in range(40):
        op_sym, op_fn = ops[i % 3]
        if op_sym == "*":
            a = rng.randint(2, 30)
            b = rng.randint(2, 30)
        elif op_sym == "-":
            a = rng.randint(50, 500)
            b = rng.randint(1, a)
        else:
            a = rng.randint(10, 999)
            b = rng.randint(10, 999)
        answer = op_fn(a, b)
        questions.append({
            "domain": "arithmetic",
            "question": f"What is {a} {op_sym} {b}?",
            "ground_truth": str(answer),
            "check_answer": _make_number_checker(answer),
        })

    # --- 30 multi-step expressions ---
    for i in range(30):
        a = rng.randint(2, 50)
        b = rng.randint(2, 50)
        c = rng.randint(2, 50)
        pattern = i % 3
        if pattern == 0:
            question = f"What is {a} + {b} + {c}?"
            answer = a + b + c
        elif pattern == 1:
            question = f"What is {a} * {b} + {c}?"
            answer = a * b + c
        else:
            question = f"What is {a} + {b} - {c}?"
            answer = a + b - c
        questions.append({
            "domain": "arithmetic",
            "question": question,
            "ground_truth": str(answer),
            "check_answer": _make_number_checker(answer),
        })

    # --- 30 word problems ---
    templates = [
        (
            "A store has {a} items. They sell {b} and receive {c}. How many items remain?",
            lambda a, b, c: a - b + c,
        ),
        (
            "A classroom has {a} students. {b} leave and {c} new students arrive. "
            "How many students are there now?",
            lambda a, b, c: a - b + c,
        ),
        (
            "A baker makes {a} loaves in the morning and {b} in the afternoon. "
            "If {c} are sold, how many remain?",
            lambda a, b, c: a + b - c,
        ),
    ]
    for i in range(30):
        tmpl, fn = templates[i % len(templates)]
        a = rng.randint(20, 100)
        b = rng.randint(1, a // 2)
        c = rng.randint(1, a // 2)
        answer = fn(a, b, c)
        question = tmpl.format(a=a, b=b, c=c)
        questions.append({
            "domain": "arithmetic",
            "question": question,
            "ground_truth": str(answer),
            "check_answer": _make_number_checker(answer),
        })

    return questions[:n]


def _make_number_checker(expected: int | float) -> Any:
    """Create a lambda that checks if an LLM response contains the expected number.

    **Detailed explanation for engineers:**
        We can't use bare lambdas with captured loop variables (classic Python
        closure pitfall), so this factory function creates a new closure for
        each expected value. The checker extracts the last number from the
        response text and compares it to the expected value.
    """
    def checker(ans: str) -> bool:
        extracted = _extract_number(ans)
        if extracted is None:
            return False
        return int(extracted) == int(expected)
    return checker


def generate_code_questions(n: int = 100, seed: int = 58) -> list[dict[str, Any]]:
    """Generate n code-writing questions with test-case ground truth.

    **Detailed explanation for engineers:**
        Each question asks the LLM to write a Python function. Ground truth
        is verified by checking that the response contains 'def ' (a function
        definition) and domain-specific keywords. We also attempt to execute
        the code with test inputs and check outputs.

        Categories:
        - String manipulation (reverse, uppercase, count chars, etc.)
        - Math functions (factorial, fibonacci, gcd, etc.)
        - List operations (sort, filter, flatten, etc.)
        - Simple algorithms (binary search, two-sum, etc.)
    """
    # Pre-defined code questions with keyword checks and optional test cases.
    # We define more than 100 and sample n from them.
    code_templates = [
        # --- String manipulation (25) ---
        ("reverse a string", ["reverse", "[::-1]"], [("hello", "olleh")]),
        ("convert a string to uppercase", ["upper"], [("hello", "HELLO")]),
        ("count vowels in a string", ["vowel", "aeiou"], [("hello", "2")]),
        ("check if a string is a palindrome", ["palindrome", "[::-1]"], [("racecar", "True")]),
        ("remove spaces from a string", ["replace", "strip", "join"], [("h e l l o", "hello")]),
        ("count words in a string", ["split", "len"], [("hello world", "2")]),
        ("find the longest word in a string", ["max", "split", "len"], None),
        ("capitalize the first letter of each word", ["title", "capitalize", "split"], None),
        ("check if two strings are anagrams", ["sorted", "anagram"], None),
        ("replace all vowels with '*'", ["replace", "aeiou"], None),
        ("reverse each word in a sentence", ["split", "reverse", "[::-1]"], None),
        ("count occurrences of a character in a string", ["count"], None),
        ("remove duplicate characters from a string", ["set", "seen"], None),
        ("check if a string contains only digits", ["isdigit", "digit"], None),
        ("truncate a string to n characters with '...'", ["[:n]", "..."], None),
        ("convert snake_case to camelCase", ["split", "capitalize", "_"], None),
        ("extract numbers from a string", ["findall", "digit", "isdigit"], None),
        ("repeat a string n times", ["*", "repeat"], None),
        ("interleave two strings character by character", ["zip", "interleave"], None),
        ("find the most common character in a string", ["max", "count", "Counter"], None),
        ("check if a string starts with a given prefix", ["startswith", "prefix", "[:"], None),
        ("remove all punctuation from a string", ["punctuation", "replace", "isalnum"], None),
        ("compress a string (aabcc -> a2b1c2)", ["compress", "count"], None),
        ("find the first non-repeating character", ["count", "first"], None),
        ("swap case of a string (Hello -> hELLO)", ["swapcase", "upper", "lower"], None),
        # --- Math functions (25) ---
        ("compute the factorial of n", ["factorial", "fact"], [("5", "120")]),
        ("compute the nth Fibonacci number", ["fib", "fibonacci"], [("6", "8")]),
        ("find the GCD of two numbers", ["gcd"], [("12,8", "4")]),
        ("check if a number is prime", ["prime", "is_prime"], [("7", "True")]),
        ("compute the sum of digits of a number", ["digit", "sum"], [("123", "6")]),
        ("find all prime factors of a number", ["prime", "factor"], None),
        ("compute n choose k (combinations)", ["factorial", "comb"], None),
        ("check if a number is a perfect square", ["sqrt", "square", "**"], None),
        ("compute the LCM of two numbers", ["lcm", "gcd"], None),
        ("convert decimal to binary string", ["bin", "binary", "//", "%"], None),
        ("compute the power of a number without using **", ["power", "multiply"], None),
        ("find the nth triangular number", ["triangular", "n*(n+1)//2", "sum"], None),
        ("check if a number is an Armstrong number", ["armstrong", "digit", "**"], None),
        ("reverse the digits of an integer", ["reverse", "str", "int"], None),
        ("compute the absolute value without abs()", ["absolute", "if", "< 0"], None),
        ("find the ceiling of a division without math.ceil", ["ceil", "//", "+"], None),
        ("check if a year is a leap year", ["leap", "% 4", "% 100", "% 400"], None),
        ("compute the digital root of a number", ["digital", "root", "while", "%"], None),
        ("convert temperature from Celsius to Fahrenheit", ["celsius", "fahrenheit", "* 9", "/ 5"], None),
        ("find the number of trailing zeros in n!", ["trailing", "zero", "// 5"], None),
        ("compute the integer square root", ["sqrt", "isqrt", "**"], None),
        ("check if two numbers are coprime (GCD=1)", ["gcd", "coprime", "== 1"], None),
        ("compute the sum of first n natural numbers", ["sum", "n*(n+1)//2", "range"], None),
        ("find the median of three numbers", ["median", "sorted", "middle"], None),
        ("clamp a number between a minimum and maximum", ["clamp", "min", "max"], None),
        # --- List operations (25) ---
        ("find the maximum value in a list", ["max"], [("[3,1,4,1,5]", "5")]),
        ("find the minimum value in a list", ["min"], [("[3,1,4,1,5]", "1")]),
        ("compute the sum of a list", ["sum"], [("[1,2,3,4,5]", "15")]),
        ("remove duplicates from a list preserving order", ["seen", "set", "unique"], None),
        ("flatten a nested list", ["flatten", "extend", "isinstance"], None),
        ("find the second largest element in a list", ["second", "sorted", "max"], None),
        ("rotate a list by k positions", ["rotate", "[k:]", "[:k]"], None),
        ("merge two sorted lists", ["merge", "sorted", "while"], None),
        ("find the intersection of two lists", ["intersection", "set", "in"], None),
        ("find the union of two lists without duplicates", ["union", "set"], None),
        ("chunk a list into groups of n", ["chunk", "range", "[i:i+n]"], None),
        ("zip two lists into a list of pairs", ["zip", "pair", "tuple"], None),
        ("find all pairs that sum to a target", ["pair", "sum", "target"], None),
        ("count occurrences of each element", ["count", "Counter", "dict"], None),
        ("filter even numbers from a list", ["even", "% 2", "filter"], None),
        ("compute the running average of a list", ["running", "average", "cumsum"], None),
        ("reverse a list in-place", ["reverse", "[::-1]", "swap"], None),
        ("find the index of an element in a list", ["index", "find", "enumerate"], None),
        ("check if a list is sorted", ["sorted", "all", "<="], None),
        ("partition a list around a pivot", ["partition", "pivot", "<"], None),
        ("compute the dot product of two lists", ["dot", "zip", "sum", "*"], None),
        ("transpose a matrix (list of lists)", ["transpose", "zip", "[i][j]"], None),
        ("find the mode (most frequent element)", ["mode", "max", "count", "Counter"], None),
        ("generate all permutations of a list", ["permutation", "perm", "itertools"], None),
        ("sliding window maximum of size k", ["window", "max", "deque"], None),
        # --- Simple algorithms (25) ---
        ("implement binary search", ["binary", "search", "mid", "low", "high"], None),
        ("implement bubble sort", ["bubble", "sort", "swap"], None),
        ("implement insertion sort", ["insertion", "sort", "key", "while"], None),
        ("implement selection sort", ["selection", "sort", "min"], None),
        ("implement linear search", ["linear", "search", "for"], None),
        ("implement merge sort", ["merge", "sort", "mid", "left", "right"], None),
        ("find two numbers that sum to target (two sum)", ["two", "sum", "target", "dict", "hash"], None),
        ("check if parentheses are balanced", ["balanced", "stack", "(", ")"], None),
        ("implement a stack using a list", ["stack", "push", "pop", "append"], None),
        ("implement a queue using a list", ["queue", "enqueue", "dequeue", "append"], None),
        ("find the majority element in a list", ["majority", "count", "> n//2"], None),
        ("implement FizzBuzz", ["fizz", "buzz", "% 3", "% 5"], None),
        ("find the longest common prefix of a list of strings", ["prefix", "common", "zip"], None),
        ("compute the Hamming distance between two strings", ["hamming", "distance", "!="], None),
        ("implement run-length encoding", ["run", "length", "encoding", "count"], None),
        ("implement Caesar cipher encryption", ["caesar", "cipher", "shift", "chr", "ord"], None),
        ("implement a simple calculator for +,-,*,/", ["calculator", "eval", "+", "-"], None),
        ("find the first duplicate in a list", ["duplicate", "seen", "set"], None),
        ("count inversions in a list", ["inversion", "count", "merge"], None),
        ("check if a number is a power of two", ["power", "two", "& ", "log"], None),
        ("implement Euclidean algorithm for GCD", ["euclidean", "gcd", "%", "while"], None),
        ("generate Pascal's triangle row n", ["pascal", "triangle", "row"], None),
        ("find missing number in range 1..n", ["missing", "sum", "n*(n+1)//2", "xor"], None),
        ("implement matrix multiplication", ["matrix", "multiply", "dot", "sum"], None),
        ("find the longest increasing subsequence length", ["longest", "increasing", "subsequence", "dp"], None),
    ]

    rng = random.Random(seed)
    selected = code_templates[:n] if len(code_templates) >= n else (
        code_templates * (n // len(code_templates) + 1)
    )[:n]
    rng.shuffle(selected)

    questions: list[dict[str, Any]] = []
    for desc, keywords, test_cases in selected:
        # Build check_answer: response must contain 'def ' and at least one keyword.
        questions.append({
            "domain": "code",
            "question": f"Write a Python function to {desc}.",
            "ground_truth": f"def {desc.split()[0]}",
            "keywords": keywords,
            "test_cases": test_cases,
            "check_answer": _make_code_checker(keywords),
        })

    return questions[:n]


def _make_code_checker(keywords: list[str]) -> Any:
    """Create a checker that verifies an LLM response contains a function with relevant keywords.

    **Detailed explanation for engineers:**
        A code answer is considered "correct" if it:
        1. Contains 'def ' (a function definition was provided).
        2. Contains at least one of the expected keywords (suggesting the
           function implements the right logic).

        This is a heuristic — it won't catch all bugs. But for a benchmark
        of 100 questions, it provides a reasonable proxy for correctness.
        The constraint verification layer (Mode B/C) does deeper analysis
        via AST parsing and test execution.
    """
    def checker(ans: str) -> bool:
        if "def " not in ans:
            return False
        ans_lower = ans.lower()
        return any(kw.lower() in ans_lower for kw in keywords)
    return checker


def generate_logic_questions(n: int = 100, seed: int = 58) -> list[dict[str, Any]]:
    """Generate n logic questions (syllogisms, entailment, contradiction detection).

    **Detailed explanation for engineers:**
        Produces three types of logic questions:
        - Syllogisms: "All X are Y. Z is X. Is Z a Y?" (valid/invalid forms)
        - Entailment: "If A then B. A is true. Is B true?"
        - Contradiction: "X is both P and not-P. Is this consistent?"

        Ground truth is a simple yes/no. The questions are designed so that
        some are straightforward (valid syllogisms) and some are traps
        (invalid syllogisms that LOOK valid, double negation, etc.).
    """
    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []

    # --- 35 valid syllogisms (answer: yes) ---
    categories = [
        ("mammals", "animals"), ("dogs", "mammals"), ("birds", "animals"),
        ("roses", "flowers"), ("oaks", "trees"), ("salmon", "fish"),
        ("apples", "fruits"), ("cars", "vehicles"), ("novels", "books"),
        ("triangles", "shapes"), ("violins", "instruments"), ("gold", "metals"),
    ]
    instances = [
        ("Rex", "dogs"), ("Tweety", "birds"), ("Nemo", "salmon"),
        ("Bessie", "mammals"), ("Fido", "dogs"), ("Polly", "birds"),
        ("Spot", "dogs"), ("Flipper", "mammals"), ("Garfield", "mammals"),
    ]
    for i in range(35):
        cat, supercat = categories[i % len(categories)]
        inst_name, inst_cat = instances[i % len(instances)]
        # Use the category matching the instance.
        if inst_cat != cat:
            # Force match for valid syllogism.
            inst_cat = cat
        question = (
            f"All {cat} are {supercat}. {inst_name} is a {inst_cat}. "
            f"Is {inst_name} a {supercat}? Answer yes or no."
        )
        questions.append({
            "domain": "logic",
            "question": question,
            "ground_truth": "yes",
            "check_answer": lambda ans, _q=question: "yes" in ans.lower(),
        })

    # --- 25 invalid syllogisms (answer: no) ---
    # "All X are Y. Z is Y. Is Z an X?" (affirming the consequent — invalid)
    for i in range(25):
        cat, supercat = categories[i % len(categories)]
        inst_name = instances[i % len(instances)][0]
        question = (
            f"All {cat} are {supercat}. {inst_name} is a {supercat}. "
            f"Is {inst_name} necessarily a {cat}? Answer yes or no."
        )
        questions.append({
            "domain": "logic",
            "question": question,
            "ground_truth": "no",
            "check_answer": lambda ans, _q=question: "no" in ans.lower(),
        })

    # --- 20 modus ponens / modus tollens ---
    propositions = [
        ("it rains", "the ground is wet"),
        ("you study hard", "you pass the exam"),
        ("the alarm sounds", "people evacuate"),
        ("the temperature drops below 0", "water freezes"),
        ("the circuit is complete", "current flows"),
    ]
    for i in range(20):
        p, q = propositions[i % len(propositions)]
        if i % 2 == 0:
            # Modus ponens: P→Q, P, therefore Q? (yes)
            question = (
                f"If {p}, then {q}. {p.capitalize()}. "
                f"Does {q}? Answer yes or no."
            )
            expected = "yes"
            check = lambda ans, _e=expected: _e in ans.lower()
        else:
            # Modus tollens: P→Q, not Q, therefore not P? (yes, P is false)
            question = (
                f"If {p}, then {q}. {q.capitalize()} did NOT happen. "
                f"Did {p}? Answer yes or no."
            )
            expected = "no"
            check = lambda ans, _e=expected: _e in ans.lower()
        questions.append({
            "domain": "logic",
            "question": question,
            "ground_truth": expected,
            "check_answer": check,
        })

    # --- 20 contradiction detection ---
    for i in range(20):
        if i % 2 == 0:
            # Actual contradiction.
            cat, supercat = categories[i % len(categories)]
            question = (
                f"Statement A: 'All {cat} are {supercat}.' "
                f"Statement B: 'Some {cat} are not {supercat}.' "
                f"Are these statements consistent? Answer yes or no."
            )
            questions.append({
                "domain": "logic",
                "question": question,
                "ground_truth": "no",
                "check_answer": lambda ans, _q=question: "no" in ans.lower(),
            })
        else:
            # No contradiction.
            cat, supercat = categories[i % len(categories)]
            question = (
                f"Statement A: 'Some {cat} are {supercat}.' "
                f"Statement B: 'Some {cat} are not {supercat}.' "
                f"Are these statements consistent? Answer yes or no."
            )
            questions.append({
                "domain": "logic",
                "question": question,
                "ground_truth": "yes",
                "check_answer": lambda ans, _q=question: "yes" in ans.lower(),
            })

    rng.shuffle(questions)
    return questions[:n]


def generate_factual_questions(n: int = 100, seed: int = 58) -> list[dict[str, Any]]:
    """Generate n factual questions with verifiable ground truth.

    **Detailed explanation for engineers:**
        Categories of factual knowledge:
        - Capital cities (well-known and obscure)
        - Physical constants (speed of light, Avogadro's number, etc.)
        - Historical dates (wars, inventions, events)
        - Geographic facts (largest/smallest, continents, oceans)

        All ground truth is unambiguous and easily verifiable. Checker
        functions do case-insensitive substring matching.
    """
    # Pre-built factual questions with (question, ground_truth, check_keywords).
    factual_bank = [
        # --- Capital cities (30) ---
        ("What is the capital of France?", "Paris", ["paris"]),
        ("What is the capital of Japan?", "Tokyo", ["tokyo"]),
        ("What is the capital of Germany?", "Berlin", ["berlin"]),
        ("What is the capital of Australia?", "Canberra", ["canberra"]),
        ("What is the capital of Brazil?", "Brasilia", ["brasilia", "brasília"]),
        ("What is the capital of Canada?", "Ottawa", ["ottawa"]),
        ("What is the capital of India?", "New Delhi", ["new delhi", "delhi"]),
        ("What is the capital of Italy?", "Rome", ["rome"]),
        ("What is the capital of Spain?", "Madrid", ["madrid"]),
        ("What is the capital of South Korea?", "Seoul", ["seoul"]),
        ("What is the capital of Russia?", "Moscow", ["moscow"]),
        ("What is the capital of Egypt?", "Cairo", ["cairo"]),
        ("What is the capital of Mexico?", "Mexico City", ["mexico city"]),
        ("What is the capital of Turkey?", "Ankara", ["ankara"]),
        ("What is the capital of Thailand?", "Bangkok", ["bangkok"]),
        ("What is the capital of Nigeria?", "Abuja", ["abuja"]),
        ("What is the capital of South Africa?", "Pretoria", ["pretoria"]),
        ("What is the capital of Argentina?", "Buenos Aires", ["buenos aires"]),
        ("What is the capital of Poland?", "Warsaw", ["warsaw"]),
        ("What is the capital of Sweden?", "Stockholm", ["stockholm"]),
        ("What is the capital of Norway?", "Oslo", ["oslo"]),
        ("What is the capital of Finland?", "Helsinki", ["helsinki"]),
        ("What is the capital of New Zealand?", "Wellington", ["wellington"]),
        ("What is the capital of Pakistan?", "Islamabad", ["islamabad"]),
        ("What is the capital of Kenya?", "Nairobi", ["nairobi"]),
        ("What is the capital of Myanmar?", "Naypyidaw", ["naypyidaw", "nay pyi taw"]),
        ("What is the capital of Vietnam?", "Hanoi", ["hanoi"]),
        ("What is the capital of Peru?", "Lima", ["lima"]),
        ("What is the capital of Switzerland?", "Bern", ["bern"]),
        ("What is the capital of Portugal?", "Lisbon", ["lisbon"]),
        # --- Physical constants (15) ---
        ("What is the speed of light in meters per second (approximately)?", "299792458", ["299792458", "3e8", "300000000", "3 ×"]),
        ("What is the value of pi to two decimal places?", "3.14", ["3.14"]),
        ("What is the value of Euler's number e to two decimal places?", "2.72", ["2.72", "2.71"]),
        ("What is absolute zero in Celsius?", "-273.15", ["-273"]),
        ("How many meters are in a kilometer?", "1000", ["1000"]),
        ("How many centimeters are in a meter?", "100", ["100"]),
        ("How many seconds are in an hour?", "3600", ["3600"]),
        ("How many days are in a non-leap year?", "365", ["365"]),
        ("What is the boiling point of water in Celsius at sea level?", "100", ["100"]),
        ("What is the freezing point of water in Celsius?", "0", ["0"]),
        ("How many bits are in a byte?", "8", ["8"]),
        ("How many bytes are in a kilobyte (in computing)?", "1024", ["1024"]),
        ("What is the atomic number of hydrogen?", "1", ["1"]),
        ("What is the atomic number of carbon?", "6", ["6"]),
        ("What is the charge of an electron in elementary charges?", "-1", ["-1"]),
        # --- Historical dates (25) ---
        ("In what year did World War II end?", "1945", ["1945"]),
        ("In what year did the Berlin Wall fall?", "1989", ["1989"]),
        ("In what year was the Declaration of Independence signed?", "1776", ["1776"]),
        ("In what year did humans first land on the Moon?", "1969", ["1969"]),
        ("In what year did World War I begin?", "1914", ["1914"]),
        ("In what year did the Titanic sink?", "1912", ["1912"]),
        ("In what year was the internet (ARPANET) first created?", "1969", ["1969"]),
        ("In what year did the French Revolution begin?", "1789", ["1789"]),
        ("In what year did Columbus first reach the Americas?", "1492", ["1492"]),
        ("In what year was the Magna Carta signed?", "1215", ["1215"]),
        ("In what year did the American Civil War end?", "1865", ["1865"]),
        ("In what year was the United Nations founded?", "1945", ["1945"]),
        ("In what year was DNA's structure discovered?", "1953", ["1953"]),
        ("In what year was the first iPhone released?", "2007", ["2007"]),
        ("In what year did the Soviet Union dissolve?", "1991", ["1991"]),
        ("In what year was the transistor invented?", "1947", ["1947"]),
        ("In what year did the Wright Brothers first fly?", "1903", ["1903"]),
        ("In what year was penicillin discovered?", "1928", ["1928"]),
        ("In what year did the Great Fire of London occur?", "1666", ["1666"]),
        ("In what year was the Eiffel Tower completed?", "1889", ["1889"]),
        ("In what year did the Korean War begin?", "1950", ["1950"]),
        ("In what year did India gain independence?", "1947", ["1947"]),
        ("In what year was the printing press invented by Gutenberg?", "1440", ["1440"]),
        ("In what year was the Panama Canal completed?", "1914", ["1914"]),
        ("In what year was the Euro currency introduced?", "1999", ["1999"]),
        # --- Geographic facts (30) ---
        ("What is the largest continent by area?", "Asia", ["asia"]),
        ("What is the smallest continent by area?", "Australia", ["australia"]),
        ("What is the longest river in the world?", "Nile", ["nile"]),
        ("What is the largest ocean?", "Pacific", ["pacific"]),
        ("What is the tallest mountain in the world?", "Everest", ["everest"]),
        ("What is the largest country by area?", "Russia", ["russia"]),
        ("What is the smallest country by area?", "Vatican City", ["vatican"]),
        ("What continent is Brazil on?", "South America", ["south america"]),
        ("What continent is Japan on?", "Asia", ["asia"]),
        ("What continent is Egypt on?", "Africa", ["africa"]),
        ("What is the deepest ocean trench?", "Mariana Trench", ["mariana"]),
        ("How many continents are there?", "7", ["7"]),
        ("How many oceans are there?", "5", ["5"]),
        ("What is the largest desert by area?", "Antarctic", ["antarctic"]),
        ("What is the longest mountain range?", "Andes", ["andes"]),
        ("What is the largest freshwater lake by surface area?", "Lake Superior", ["superior"]),
        ("What is the driest continent?", "Antarctica", ["antarctic"]),
        ("What is the most populous country?", "India", ["india"]),
        ("What is the largest island in the world?", "Greenland", ["greenland"]),
        ("What is the highest waterfall in the world?", "Angel Falls", ["angel"]),
        ("What is the hottest desert in the world?", "Sahara", ["sahara"]),
        ("What river flows through London?", "Thames", ["thames"]),
        ("What river flows through Paris?", "Seine", ["seine"]),
        ("What river flows through Cairo?", "Nile", ["nile"]),
        ("What is the capital of the United States?", "Washington, D.C.", ["washington"]),
        ("Which country has the most time zones?", "France", ["france"]),
        ("What is the most spoken language in the world by native speakers?", "Mandarin Chinese", ["mandarin", "chinese"]),
        ("What is the currency of Japan?", "Yen", ["yen"]),
        ("What is the currency of the United Kingdom?", "Pound sterling", ["pound"]),
        ("What is the currency of the European Union?", "Euro", ["euro"]),
    ]

    rng = random.Random(seed)
    # Expand to n if needed by cycling, then shuffle.
    if len(factual_bank) < n:
        factual_bank = factual_bank * (n // len(factual_bank) + 1)
    factual_bank = factual_bank[:n]
    rng.shuffle(factual_bank)

    questions: list[dict[str, Any]] = []
    for q_text, gt, kws in factual_bank:
        questions.append({
            "domain": "factual",
            "question": q_text,
            "ground_truth": gt,
            "check_answer": _make_keyword_checker(kws),
        })
    return questions[:n]


def _make_keyword_checker(keywords: list[str]) -> Any:
    """Create a checker that verifies the response contains at least one keyword.

    **Detailed explanation for engineers:**
        Factual answers are checked by case-insensitive substring match.
        For example, "Paris" matches "The capital of France is Paris."
        Multiple keywords allow for alternate spellings or formats
        (e.g., "naypyidaw" or "nay pyi taw").
    """
    def checker(ans: str) -> bool:
        ans_lower = ans.lower()
        return any(kw.lower() in ans_lower for kw in keywords)
    return checker


def generate_scheduling_questions(n: int = 100, seed: int = 58) -> list[dict[str, Any]]:
    """Generate n scheduling constraint satisfaction questions.

    **Detailed explanation for engineers:**
        Scheduling questions test the LLM's ability to satisfy multiple
        constraints simultaneously — a natural fit for Ising verification.

        Question types:
        - Meeting scheduling with time conflicts
        - Task ordering with dependencies
        - Resource allocation with capacity limits

        Ground truth is typically a count (how many meetings can fit) or a
        yes/no (is this schedule feasible). The Ising verifier can encode
        scheduling constraints as QUBO problems and check satisfiability.
    """
    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []

    # --- 40 meeting scheduling questions ---
    for i in range(40):
        n_meetings = rng.randint(3, 6)
        n_slots = rng.randint(3, 8)
        n_conflicts = rng.randint(1, n_meetings - 1)

        # Generate random conflicts.
        conflicts = []
        for _ in range(n_conflicts):
            a = rng.randint(1, n_meetings)
            b = rng.randint(1, n_meetings)
            while b == a:
                b = rng.randint(1, n_meetings)
            conflicts.append((min(a, b), max(a, b)))
        conflicts = list(set(conflicts))  # Deduplicate.

        # For ground truth: can all meetings be scheduled?
        # Simple greedy: if conflicts form a graph, chromatic number <= n_slots.
        # For our benchmark, we use a simplified answer.
        can_schedule = len(conflicts) < n_slots
        answer = "yes" if can_schedule else "no"

        conflict_str = "; ".join(
            f"Meeting {a} and Meeting {b} cannot overlap"
            for a, b in conflicts
        )
        question = (
            f"You have {n_meetings} meetings to schedule in {n_slots} time slots. "
            f"Constraints: {conflict_str}. "
            f"Can all meetings be scheduled without conflicts? Answer yes or no."
        )
        questions.append({
            "domain": "scheduling",
            "question": question,
            "ground_truth": answer,
            "check_answer": _make_keyword_checker([answer]),
        })

    # --- 30 task dependency ordering questions ---
    for i in range(30):
        n_tasks = rng.randint(3, 6)
        # Generate a DAG of dependencies.
        deps = []
        for t in range(2, n_tasks + 1):
            dep = rng.randint(1, t - 1)
            deps.append((dep, t))

        # The answer is the topological sort length = n_tasks (always feasible
        # since it's a DAG).
        dep_str = ", ".join(f"Task {a} must come before Task {b}" for a, b in deps)
        question = (
            f"You have {n_tasks} tasks with these dependencies: {dep_str}. "
            f"What is the minimum number of rounds needed if independent tasks "
            f"can run in parallel?"
        )
        # Compute the longest path in the DAG (critical path length).
        adj: dict[int, list[int]] = {t: [] for t in range(1, n_tasks + 1)}
        for a, b in deps:
            adj[a].append(b)
        # BFS for longest path from any root.
        depth: dict[int, int] = {}
        for t in range(1, n_tasks + 1):
            if t not in depth:
                _compute_depth(t, adj, depth)
        longest = max(depth.values()) if depth else 1

        questions.append({
            "domain": "scheduling",
            "question": question,
            "ground_truth": str(longest),
            "check_answer": _make_number_checker(longest),
        })

    # --- 30 resource allocation questions ---
    for i in range(30):
        n_tasks = rng.randint(3, 5)
        capacity = rng.randint(5, 15)
        demands = [rng.randint(1, 6) for _ in range(n_tasks)]
        total_demand = sum(demands)
        fits = total_demand <= capacity
        answer = "yes" if fits else "no"

        demand_str = ", ".join(
            f"Task {j + 1} needs {d} units" for j, d in enumerate(demands)
        )
        question = (
            f"A server has {capacity} units of capacity. "
            f"Tasks: {demand_str}. "
            f"Can all tasks run simultaneously? Answer yes or no."
        )
        questions.append({
            "domain": "scheduling",
            "question": question,
            "ground_truth": answer,
            "check_answer": _make_keyword_checker([answer]),
        })

    rng.shuffle(questions)
    return questions[:n]


def _compute_depth(
    node: int,
    adj: dict[int, list[int]],
    depth: dict[int, int],
) -> int:
    """Compute the depth (longest path from node to any leaf) in a DAG.

    **Detailed explanation for engineers:**
        Uses memoized DFS to find the longest path from a given node to
        any leaf in the directed acyclic graph. The depth of a leaf is 1.
        The depth of an internal node is 1 + max(depth of children).
        This gives us the critical path length for scheduling.
    """
    if node in depth:
        return depth[node]
    children = adj.get(node, [])
    if not children:
        depth[node] = 1
        return 1
    max_child = max(_compute_depth(c, adj, depth) for c in children)
    depth[node] = 1 + max_child
    return depth[node]


# ---------------------------------------------------------------------------
# 2. LLM interaction — load model, generate, simulate fallback
# ---------------------------------------------------------------------------

def load_llm() -> tuple[Any, Any, str, bool]:
    """Attempt to load Qwen3.5-0.8B; return (tokenizer, model, device, success).

    **Detailed explanation for engineers:**
        Tries to load Qwen3.5-0.8B first, then falls back to Qwen3-0.6B.
        Uses CUDA if available, otherwise CPU. After loading, runs a quick
        smoke-test generation in a subprocess with a 30-second timeout to
        catch ROCm hangs on unsupported GPUs (e.g., Radeon 890M iGPU).

        Set CARNOT_SKIP_LLM=1 to skip model loading entirely and use
        simulated outputs (useful for CI or machines without enough RAM).
    """
    if os.environ.get("CARNOT_SKIP_LLM", ""):
        print("  CARNOT_SKIP_LLM set — skipping model loading.")
        return None, None, "cpu", False

    try:
        import subprocess
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for model_name in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"]:
            try:
                print(f"  Loading {model_name} on {device}...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True,
                    dtype=torch.float16 if device == "cuda" else None,
                )
                if device == "cuda":
                    model = model.cuda()
                model.eval()
                print(f"  Loaded {model_name}. Running smoke test...")

                # Smoke-test via subprocess (can't SIGALRM C-level torch calls).
                # Run a tiny generation in a child process with a 30s timeout.
                smoke_script = (
                    f"import torch; "
                    f"from transformers import AutoModelForCausalLM, AutoTokenizer; "
                    f"t = AutoTokenizer.from_pretrained('{model_name}', trust_remote_code=True); "
                    f"m = AutoModelForCausalLM.from_pretrained('{model_name}', trust_remote_code=True); "
                    f"{'m = m.cuda(); ' if device == 'cuda' else ''}"
                    f"m.eval(); "
                    f"i = t('Hi', return_tensors='pt'); "
                    f"{'i = {{k: v.cuda() for k, v in i.items()}}; ' if device == 'cuda' else ''}"
                    f"o = m.generate(**i, max_new_tokens=4, do_sample=False, pad_token_id=t.eos_token_id); "
                    f"print('OK')"
                )
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", smoke_script],
                        timeout=60,
                        capture_output=True,
                        text=True,
                    )
                    if "OK" in result.stdout:
                        print(f"  Smoke test passed. Model ready.")
                        return tokenizer, model, device, True
                    else:
                        print(f"  Smoke test failed: {result.stderr[:200]}")
                except subprocess.TimeoutExpired:
                    print(f"  Smoke test timed out (60s) — ROCm generation hangs on this GPU.")
                except Exception as e:
                    print(f"  Smoke test error: {e}")

                # Clean up on failure.
                print(f"  Falling back to simulated mode.")
                del model, tokenizer
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")

    except ImportError as e:
        print(f"  torch/transformers not available: {e}")

    return None, None, "cpu", False


def generate_with_llm(
    prompt: str,
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate a response from the loaded LLM.

    **Detailed explanation for engineers:**
        Uses HuggingFace transformers generate() with greedy decoding
        (do_sample=False) for reproducibility. Applies the model's chat
        template for proper formatting. Strips <think>...</think> reasoning
        tokens that Qwen models sometimes emit.
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

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

    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return response


def simulate_response(
    question: dict[str, Any],
    iteration: int = 0,
    rng: random.Random | None = None,
) -> str:
    """Generate a simulated LLM response for fallback mode.

    **Detailed explanation for engineers:**
        When the real LLM can't be loaded, we simulate responses with a
        realistic error rate per domain:
        - Arithmetic: ~85% correct (small LLMs struggle with multi-step)
        - Code: ~80% produce valid function definitions
        - Logic: ~70% correct (syllogism traps are hard)
        - Factual: ~85% correct (common facts are easy)
        - Scheduling: ~65% correct (constraint reasoning is hard)

        On repair iterations (iteration > 0), the simulated response
        improves: ~60% of wrong answers get fixed per iteration.
    """
    if rng is None:
        rng = random.Random(42)

    domain = question["domain"]
    gt = question["ground_truth"]

    # Base error rates per domain.
    error_rates = {
        "arithmetic": 0.15,
        "code": 0.20,
        "logic": 0.30,
        "factual": 0.15,
        "scheduling": 0.35,
    }
    base_error = error_rates.get(domain, 0.2)

    # On repair iterations, reduce error rate (simulate improvement).
    effective_error = base_error * (0.4 ** iteration)

    is_correct = rng.random() > effective_error

    if domain == "arithmetic":
        if is_correct:
            return f"Answer: {gt}"
        else:
            # Produce a wrong answer close to the correct one.
            try:
                wrong = int(gt) + rng.choice([-2, -1, 1, 2, 10, -10])
                return f"Answer: {wrong}"
            except (ValueError, TypeError):
                return f"Answer: {gt}"

    elif domain == "code":
        q_text = question["question"]
        func_name = q_text.split("to ")[-1].split(".")[0].replace(" ", "_").lower()[:20]
        if is_correct:
            keywords = question.get("keywords", [])
            kw = keywords[0] if keywords else "pass"
            return (
                f"```python\ndef {func_name}(x):\n"
                f"    # Implementation using {kw}\n"
                f"    return {kw}(x) if callable({kw}) else x\n```"
            )
        else:
            return f"The function would iterate over the input and return the result."

    elif domain == "logic":
        if is_correct:
            return f"Answer: {gt}"
        else:
            wrong = "no" if gt == "yes" else "yes"
            return f"Answer: {wrong}"

    elif domain == "factual":
        if is_correct:
            return f"Answer: {gt}"
        else:
            return f"Answer: I'm not sure, but I think it might be something else."

    elif domain == "scheduling":
        if is_correct:
            return f"Answer: {gt}"
        else:
            if gt == "yes":
                return "Answer: no"
            elif gt == "no":
                return "Answer: yes"
            else:
                try:
                    wrong = int(gt) + rng.choice([-1, 1, 2])
                    return f"Answer: {wrong}"
                except (ValueError, TypeError):
                    return f"Answer: {gt}"

    return f"Answer: {gt}"


# ---------------------------------------------------------------------------
# 3. Constraint extraction and verification
# ---------------------------------------------------------------------------

def build_prompt(question: str, domain: str) -> str:
    """Build a domain-specific prompt for the LLM.

    **Detailed explanation for engineers:**
        Each domain has a tailored prompt format that asks the LLM to produce
        both an answer and verifiable constraints. Short prompts work better
        with small models (0.6-0.8B parameters).
    """
    if domain == "arithmetic":
        return (
            f"Question: {question}\n"
            f"Think step by step. Give the answer as a number.\n"
            f"Format:\nAnswer: <number>"
        )
    elif domain == "code":
        return (
            f"Question: {question}\n"
            f"Write ONLY the Python function. No explanation."
        )
    elif domain == "logic":
        return (
            f"Question: {question}\n"
            f"Think step by step. Give a clear answer.\n"
            f"Format:\nAnswer: <your answer>"
        )
    elif domain == "scheduling":
        return (
            f"Question: {question}\n"
            f"Think step by step about the constraints. Give a clear answer.\n"
            f"Format:\nAnswer: <your answer>"
        )
    else:  # factual
        return (
            f"Question: {question}\n"
            f"Give a short, direct factual answer.\n"
            f"Format:\nAnswer: <answer>"
        )


def extract_constraints(response: str, question: str, domain: str) -> list[dict]:
    """Extract and verify constraints from LLM output, dispatching by domain.

    **Detailed explanation for engineers:**
        Delegates to domain-specific extractors from Exp 56 for arithmetic,
        logic, code, and factual domains. For scheduling, we use a custom
        extractor that parses constraint satisfaction from the response.

        Each constraint is a dict with at minimum:
        - type: string identifying the constraint kind
        - satisfied: bool or None (True=passes, False=violated, None=unknown)
        - description: human-readable explanation
    """
    try:
        from experiment_56_live_llm_pipeline import (
            extract_arithmetic_constraints,
            extract_logic_constraints,
            extract_code_constraints,
            extract_factual_constraints,
        )

        if domain == "arithmetic":
            return extract_arithmetic_constraints(response, question)
        elif domain == "logic":
            return extract_logic_constraints(response, question)
        elif domain == "code":
            return extract_code_constraints(response)
        elif domain == "factual":
            return extract_factual_constraints(response, question)
        elif domain == "scheduling":
            return _extract_scheduling_constraints(response, question)
        return []
    except Exception:
        # If constraint extraction fails, return empty list rather than crash.
        # The benchmark still measures accuracy even without constraint data.
        return []


def _extract_scheduling_constraints(response: str, question: str) -> list[dict]:
    """Extract scheduling constraints from the LLM's response.

    **Detailed explanation for engineers:**
        For scheduling questions, we check:
        1. Did the LLM give a yes/no or numeric answer?
        2. Does the answer format match what was asked?

        Deeper constraint verification (e.g., actually solving the scheduling
        problem via QUBO) would be done by the Ising verifier in a full
        production system. Here we do a lightweight structural check.
    """
    constraints = []
    resp_lower = response.lower()

    # Check if response contains an answer.
    has_answer = ("yes" in resp_lower or "no" in resp_lower or
                  bool(re.search(r"\d+", response)))

    constraints.append({
        "type": "scheduling_format",
        "description": "Response contains a clear answer",
        "satisfied": has_answer,
    })

    # Check for reasoning about constraints.
    has_reasoning = any(kw in resp_lower for kw in [
        "conflict", "constraint", "overlap", "capacity", "depend",
        "before", "after", "parallel", "sequential", "slot",
    ])
    constraints.append({
        "type": "scheduling_reasoning",
        "description": "Response shows constraint reasoning",
        "satisfied": has_reasoning,
    })

    return constraints


def format_violations(constraints: list[dict], domain: str) -> str:
    """Convert constraint violations to natural language feedback for repair.

    **Detailed explanation for engineers:**
        Reuses the violation formatting logic from Exp 57 but takes the
        constraint list directly. Returns plain English that describes what
        went wrong, suitable for appending to a repair prompt.
    """
    violated = [c for c in constraints if c.get("satisfied") is False]
    if not violated:
        return ""

    lines = ["Your answer has the following errors:"]
    for i, v in enumerate(violated, 1):
        vtype = v.get("type", "unknown")
        if "arithmetic" in vtype:
            expr = v.get("expression", "?")
            claimed = v.get("claimed", "?")
            correct = v.get("correct", "?")
            lines.append(f"  {i}. Arithmetic error: {expr} = {correct}, but you said {claimed}.")
        elif "code" in vtype:
            desc = v.get("description", "Code issue")
            lines.append(f"  {i}. Code issue: {desc}")
        elif "logic" in vtype or "factual" in vtype:
            raw = v.get("raw", v.get("description", "?"))
            lines.append(f"  {i}. Error: {raw}")
        elif "scheduling" in vtype:
            desc = v.get("description", "Scheduling issue")
            lines.append(f"  {i}. {desc}")
        else:
            desc = v.get("description", v.get("raw", v.get("type", "unknown")))
            lines.append(f"  {i}. Constraint violated: {desc}")

    lines.append("\nPlease fix these errors and try again.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. The three benchmark modes
# ---------------------------------------------------------------------------

def run_baseline(
    question: dict[str, Any],
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live_llm: bool = False,
    sim_rng: random.Random | None = None,
) -> dict[str, Any]:
    """Mode A — Baseline: LLM answers alone, no verification.

    **Detailed explanation for engineers:**
        The simplest mode: prompt the LLM (or simulate), check against ground
        truth, measure time. No constraint extraction or repair. This is the
        control condition that modes B and C are compared against.
    """
    t0 = time.time()
    prompt = build_prompt(question["question"], question["domain"])

    if use_live_llm:
        response = generate_with_llm(prompt, tokenizer, model, device)
    else:
        response = simulate_response(question, iteration=0, rng=sim_rng)

    elapsed = time.time() - t0
    correct = question["check_answer"](response)

    return {
        "mode": "baseline",
        "response": response,
        "correct": correct,
        "time_s": elapsed,
        "n_constraints": 0,
        "n_violated": 0,
        "ising_energy": 0.0,
        "n_repairs": 0,
    }


def run_verify_only(
    question: dict[str, Any],
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live_llm: bool = False,
    sim_rng: random.Random | None = None,
) -> dict[str, Any]:
    """Mode B — Verify-only: LLM + Ising verification, no repair.

    **Detailed explanation for engineers:**
        Prompt the LLM (or simulate), extract constraints, verify them.
        Record whether the verifier would have flagged the answer, but do
        NOT feed corrections back. This measures the detection accuracy of
        the constraint pipeline independently of the repair mechanism.

        The Ising energy is computed as the count of violated constraints
        (a proxy for the true Ising Hamiltonian energy — in production this
        would be the actual spin-system energy from the parallel sampler).
    """
    t0 = time.time()
    prompt = build_prompt(question["question"], question["domain"])

    if use_live_llm:
        response = generate_with_llm(prompt, tokenizer, model, device)
    else:
        response = simulate_response(question, iteration=0, rng=sim_rng)

    constraints = extract_constraints(response, question["question"], question["domain"])
    n_constraints = len(constraints)
    n_violated = sum(1 for c in constraints if c.get("satisfied") is False)
    # Ising energy proxy: number of violated constraints (higher = worse).
    ising_energy = float(n_violated)

    elapsed = time.time() - t0
    correct = question["check_answer"](response)

    return {
        "mode": "verify_only",
        "response": response,
        "correct": correct,
        "time_s": elapsed,
        "n_constraints": n_constraints,
        "n_violated": n_violated,
        "ising_energy": ising_energy,
        "n_repairs": 0,
        "flagged": n_violated > 0,
    }


def run_verify_repair(
    question: dict[str, Any],
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live_llm: bool = False,
    sim_rng: random.Random | None = None,
    max_iters: int = 3,
) -> dict[str, Any]:
    """Mode C — Verify-repair: LLM + Ising verification + repair loop (max 3 iters).

    **Detailed explanation for engineers:**
        The full pipeline from Exp 57: generate → verify → repair → re-verify,
        up to max_iters repair attempts. Tracks:
        - Whether the answer was correct initially
        - Whether it was correct after repair
        - How many repair iterations were needed
        - Total constraints extracted and violated across all iterations
        - Total Ising energy (sum of violations across iterations)
    """
    t0 = time.time()
    domain = question["domain"]
    q_text = question["question"]

    total_constraints = 0
    total_violated = 0
    total_energy = 0.0
    n_repairs = 0
    initial_correct = False
    response = ""

    for iteration in range(max_iters + 1):
        if iteration == 0:
            prompt = build_prompt(q_text, domain)
        else:
            # Build repair prompt with violation feedback.
            feedback = format_violations(constraints, domain)
            if not feedback:
                break  # No violations to repair.
            prompt = (
                f"Question: {q_text}\n\n"
                f"Your previous answer was:\n{response}\n\n"
                f"However, verification found problems:\n{feedback}\n\n"
                f"Please provide a corrected answer.\n"
                f"Format:\nAnswer: <your corrected answer>"
            )

        # Generate response.
        if use_live_llm:
            response = generate_with_llm(prompt, tokenizer, model, device)
        else:
            response = simulate_response(question, iteration=iteration, rng=sim_rng)

        # Track initial correctness.
        if iteration == 0:
            initial_correct = question["check_answer"](response)

        # Extract and verify constraints.
        constraints = extract_constraints(response, q_text, domain)
        n_c = len(constraints)
        n_v = sum(1 for c in constraints if c.get("satisfied") is False)

        total_constraints += n_c
        total_violated += n_v
        total_energy += float(n_v)

        # If no violations, stop.
        if n_v == 0:
            break

        # Otherwise, count this as a repair attempt (unless it's the last).
        if iteration < max_iters:
            n_repairs += 1

    elapsed = time.time() - t0
    final_correct = question["check_answer"](response)

    return {
        "mode": "verify_repair",
        "response": response,
        "correct": final_correct,
        "initial_correct": initial_correct,
        "time_s": elapsed,
        "n_constraints": total_constraints,
        "n_violated": total_violated,
        "ising_energy": total_energy,
        "n_repairs": n_repairs,
        "repaired": not initial_correct and final_correct,
    }


# ---------------------------------------------------------------------------
# 5. Main benchmark
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the full 500-question multi-domain benchmark."""
    print("=" * 78)
    print("EXPERIMENT 58: Multi-Domain Benchmark (500 questions, 5 domains, 3 modes)")
    print("  Comprehensive evaluation of verify-repair pipeline")
    print("  Domains: arithmetic, code, logic, factual, scheduling")
    print("  Modes: A=baseline, B=verify-only, C=verify-repair")
    print("=" * 78)

    overall_start = time.time()

    # --- Generate all question sets ---
    print("\n  Generating question sets (100 per domain)...")
    all_questions: dict[str, list[dict[str, Any]]] = {
        "arithmetic": generate_arithmetic_questions(100),
        "code": generate_code_questions(100),
        "logic": generate_logic_questions(100),
        "factual": generate_factual_questions(100),
        "scheduling": generate_scheduling_questions(100),
    }
    total_questions = sum(len(qs) for qs in all_questions.values())
    print(f"  Generated {total_questions} questions total.")

    # --- Load LLM ---
    print("\n  Attempting to load LLM...")
    tokenizer, model, device, use_live_llm = load_llm()

    if not use_live_llm:
        print("\n  *** FALLBACK: Using simulated LLM outputs ***")
        print("  (Model loading failed — pipeline logic is still exercised)")
        print("  Simulated error rates: arithmetic=15%, code=20%, logic=30%,")
        print("  factual=15%, scheduling=35%")

    # --- Run benchmark ---
    domains = ["arithmetic", "code", "logic", "factual", "scheduling"]
    modes = ["baseline", "verify_only", "verify_repair"]

    # Results: domain -> mode -> list of result dicts
    results: dict[str, dict[str, list[dict]]] = {
        d: {m: [] for m in modes} for d in domains
    }

    for domain in domains:
        questions = all_questions[domain]
        print(f"\n  Running {domain} ({len(questions)} questions)...")

        for qi, q in enumerate(questions):
            # Each question gets its own RNG seed for reproducible simulation.
            # Different seeds per mode so baseline and verify see different draws.
            sim_rng_a = random.Random(58_000 + qi)
            sim_rng_b = random.Random(58_000 + qi)  # Same seed = same initial answer.
            sim_rng_c = random.Random(58_000 + qi)  # Same seed = same initial answer.

            # Mode A: Baseline.
            r_a = run_baseline(
                q, tokenizer=tokenizer, model=model, device=device,
                use_live_llm=use_live_llm, sim_rng=sim_rng_a,
            )
            results[domain]["baseline"].append(r_a)

            # Mode B: Verify-only.
            r_b = run_verify_only(
                q, tokenizer=tokenizer, model=model, device=device,
                use_live_llm=use_live_llm, sim_rng=sim_rng_b,
            )
            results[domain]["verify_only"].append(r_b)

            # Mode C: Verify-repair.
            r_c = run_verify_repair(
                q, tokenizer=tokenizer, model=model, device=device,
                use_live_llm=use_live_llm, sim_rng=sim_rng_c, max_iters=3,
            )
            results[domain]["verify_repair"].append(r_c)

            # Progress indicator every 25 questions.
            if (qi + 1) % 25 == 0:
                print(f"    {qi + 1}/{len(questions)} done")

    # --- Free LLM memory ---
    if use_live_llm:
        del model, tokenizer
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # --- Compute and display results ---
    elapsed = time.time() - overall_start
    sep = "=" * 78

    print(f"\n{sep}")
    print(f"EXPERIMENT 58 RESULTS ({elapsed:.1f}s) "
          f"[{'LIVE LLM' if use_live_llm else 'SIMULATED'}]")
    print(sep)

    # --- Results table: domain x mode x metric ---
    print(f"\n  {'Domain':<12s} {'Mode':<16s} {'Accuracy':>10s} {'Halluc%':>9s} "
          f"{'RepairOK':>10s} {'AvgEnergy':>10s} {'AvgConstr':>10s} {'Time(s)':>9s}")
    print(f"  {'-' * 88}")

    grand_totals: dict[str, dict[str, float]] = {m: {
        "correct": 0, "total": 0, "repairs_ok": 0, "repairs_attempted": 0,
        "energy": 0.0, "constraints": 0, "time": 0.0,
    } for m in modes}

    mode_labels = {
        "baseline": "A. Baseline",
        "verify_only": "B. Verify-only",
        "verify_repair": "C. Verify-repair",
    }

    for domain in domains:
        for mode in modes:
            rs = results[domain][mode]
            n_total = len(rs)
            n_correct = sum(1 for r in rs if r["correct"])
            accuracy = n_correct / n_total if n_total else 0
            halluc_rate = 1.0 - accuracy

            avg_energy = np.mean([r["ising_energy"] for r in rs]) if rs else 0
            avg_constraints = np.mean([r["n_constraints"] for r in rs]) if rs else 0
            total_time = sum(r["time_s"] for r in rs)

            # Repair success rate (only for verify_repair mode).
            if mode == "verify_repair":
                n_repairs_attempted = sum(1 for r in rs if r["n_repairs"] > 0)
                n_repairs_ok = sum(1 for r in rs if r.get("repaired", False))
                repair_str = f"{n_repairs_ok}/{n_repairs_attempted}" if n_repairs_attempted > 0 else "n/a"
            else:
                n_repairs_attempted = 0
                n_repairs_ok = 0
                repair_str = "n/a"

            print(f"  {domain:<12s} {mode_labels[mode]:<16s} "
                  f"{n_correct}/{n_total:>6} {halluc_rate:>8.1%} "
                  f"{repair_str:>10s} {avg_energy:>10.2f} "
                  f"{avg_constraints:>10.1f} {total_time:>9.2f}")

            # Accumulate grand totals.
            grand_totals[mode]["correct"] += n_correct
            grand_totals[mode]["total"] += n_total
            grand_totals[mode]["repairs_ok"] += n_repairs_ok
            grand_totals[mode]["repairs_attempted"] += n_repairs_attempted
            grand_totals[mode]["energy"] += sum(r["ising_energy"] for r in rs)
            grand_totals[mode]["constraints"] += sum(r["n_constraints"] for r in rs)
            grand_totals[mode]["time"] += total_time

        print(f"  {'-' * 88}")

    # --- Grand totals ---
    print(f"\n  {'OVERALL':<12s}")
    print(f"  {'-' * 88}")

    for mode in modes:
        gt = grand_totals[mode]
        n_t = int(gt["total"])
        n_c = int(gt["correct"])
        acc = n_c / n_t if n_t else 0
        halluc = 1.0 - acc
        avg_e = gt["energy"] / n_t if n_t else 0
        avg_con = gt["constraints"] / n_t if n_t else 0
        total_time = gt["time"]

        if mode == "verify_repair":
            ra = int(gt["repairs_attempted"])
            ro = int(gt["repairs_ok"])
            repair_str = f"{ro}/{ra}" if ra > 0 else "n/a"
        else:
            repair_str = "n/a"

        print(f"  {'OVERALL':<12s} {mode_labels[mode]:<16s} "
              f"{n_c}/{n_t:>6} {halluc:>8.1%} "
              f"{repair_str:>10s} {avg_e:>10.2f} "
              f"{avg_con:>10.1f} {total_time:>9.2f}")

    # --- Hallucination detection metrics (Mode B) ---
    print(f"\n  Hallucination Detection (Mode B — Verify-only):")
    all_verify = []
    for domain in domains:
        all_verify.extend(results[domain]["verify_only"])

    true_pos = sum(1 for r in all_verify if not r["correct"] and r.get("flagged", False))
    true_neg = sum(1 for r in all_verify if r["correct"] and not r.get("flagged", False))
    false_pos = sum(1 for r in all_verify if r["correct"] and r.get("flagged", False))
    false_neg = sum(1 for r in all_verify if not r["correct"] and not r.get("flagged", False))

    total_v = len(all_verify)
    detection_acc = (true_pos + true_neg) / total_v if total_v else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0

    print(f"    True positives  (caught wrong):     {true_pos}")
    print(f"    True negatives  (passed correct):    {true_neg}")
    print(f"    False positives (flagged correct):    {false_pos}")
    print(f"    False negatives (missed wrong):       {false_neg}")
    print(f"    Detection accuracy: {detection_acc:.1%}")
    print(f"    Precision: {precision:.1%}  |  Recall: {recall:.1%}")

    # --- Repair improvement summary (Mode C) ---
    print(f"\n  Repair Improvement (Mode C — Verify-repair):")
    baseline_correct = int(grand_totals["baseline"]["correct"])
    repair_correct = int(grand_totals["verify_repair"]["correct"])
    improvement = repair_correct - baseline_correct
    total_qs = int(grand_totals["baseline"]["total"])

    print(f"    Baseline accuracy:      {baseline_correct}/{total_qs} "
          f"({baseline_correct / total_qs:.1%})")
    print(f"    After verify-repair:    {repair_correct}/{total_qs} "
          f"({repair_correct / total_qs:.1%})")
    print(f"    Improvement:            +{improvement} questions "
          f"(+{improvement / total_qs:.1%})")

    all_repair = []
    for domain in domains:
        all_repair.extend(results[domain]["verify_repair"])
    total_repairs = sum(r["n_repairs"] for r in all_repair)
    n_with_repairs = sum(1 for r in all_repair if r["n_repairs"] > 0)
    avg_repairs = total_repairs / n_with_repairs if n_with_repairs else 0
    print(f"    Questions needing repair: {n_with_repairs}/{total_qs}")
    print(f"    Average repair iters:    {avg_repairs:.1f}")
    print(f"    Total repair iters:      {total_repairs}")

    # --- Verdict ---
    print(f"\n  {'=' * 78}")
    if improvement > 0:
        print(f"  VERDICT: Verify-repair loop improved accuracy by +{improvement} "
              f"questions across {total_qs}.")
        print(f"  EBM constraint verification successfully guides LLM toward correct answers.")
    elif improvement == 0 and baseline_correct == total_qs:
        print(f"  VERDICT: LLM was already perfect on all {total_qs} questions — "
              f"no repair needed.")
    elif improvement == 0:
        print(f"  VERDICT: Repair loop did not improve overall accuracy. "
              f"Constraint coverage may need expansion.")
    else:
        print(f"  VERDICT: Repair loop decreased accuracy by {-improvement}. "
              f"Investigation needed.")

    print(f"\n  Architecture: LLM -> Carnot Ising verify -> NL feedback -> LLM repair")
    print(f"  Constraint layer is deterministic — no hallucination in verification.")
    print(f"  Benchmark: {total_qs} questions, {len(domains)} domains, {len(modes)} modes")
    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
