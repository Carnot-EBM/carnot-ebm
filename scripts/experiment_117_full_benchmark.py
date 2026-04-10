#!/usr/bin/env python3
"""Experiment 117: Full v12 Benchmark — all extractors + guided generation mode.

**Researcher summary:**
    Definitive four-mode benchmark: baseline, verify-only, verify-repair, and
    energy-guided generation (Exp 111). Runs both target models across all five
    domains with the complete v11+v12 extractor stack (ArithmeticExtractor,
    CodeExtractor, LogicExtractor, NLExtractor from v11, FactualKBExtractor from
    Exp 113 = v12). Compares accuracy against v10 (Exp 93) and v11 extractor
    generations. Reports bootstrap 95% CI on accuracy improvements, per-extractor
    contribution breakdown, and which domains benefited most from each phase.

**Detailed explanation for engineers:**
    This experiment extends Exp 93 (Multi-Model Comparison) in three ways:

    1. **Fourth mode — Guided Generation**: Energy-guided token-by-token decoding
       via EnergyGuidedSampler (carnot.inference.guided_decoding). Instead of
       post-hoc repair, the energy penalty is applied during generation itself:
       each partial output is checked by AutoExtractor, and high-energy partial
       texts receive a logit penalty (alpha=0.5) that nudges the model away from
       constraint-violating continuations. In simulation mode, this is modeled
       as a stronger error reduction (0.25×) vs. post-hoc repair (0.4× per iter).

    2. **Per-extractor contribution tracking**: For every evaluated response, each
       domain extractor (Arithmetic, Code, Logic, NL, FactualKB) is run
       individually. The number of constraints fired and violations detected per
       extractor is recorded. This shows which extractors are active per domain
       and which ones contribute most to the accuracy gains.

    3. **Multi-generation comparison**: Side-by-side accuracy table comparing
       v10 (Exp 93, ArithmeticExtractor+CodeExtractor+LogicExtractor+NLExtractor),
       v11 (added richer NL patterns), and v12 (adds FactualKBExtractor). False
       negative reduction and bootstrap 95% CI on accuracy deltas are computed
       for every (model, domain) pair.

    Simulation fallback: If real models can't be loaded (no GPU / missing
    packages), the script falls back to deterministic simulation with
    domain-specific error rates. All four pipeline modes are still fully
    exercised in simulation mode; only the LLM generation is mocked.

    Total evaluations: 250 questions × 2 models × 4 modes = 2,000.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_117_full_benchmark.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import subprocess
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Baseline (v10) results from Exp 93 for comparison.
EXP93_RESULTS_PATH = RESULTS_DIR / "experiment_93_results.json"

# ---------------------------------------------------------------------------
# 1. Model configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "Qwen3.5-0.8B",
        "candidates": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"],
        "trust_remote_code": True,
        # Simulated error profile — used when real model is unavailable.
        # These match the Exp 93 calibrated rates so comparisons are valid.
        "sim_error_rates": {
            "arithmetic": 0.12,
            "code": 0.18,
            "logic": 0.28,
            "factual": 0.12,
            "scheduling": 0.32,
        },
        # Per-generation calibration (offset so model RNGs differ).
        "model_offset": 0,
    },
    {
        "name": "Gemma4-E4B-it",
        "candidates": ["google/gemma-4-E4B-it"],
        "trust_remote_code": True,
        # Gemma is instruction-tuned: slightly better at factual/logic,
        # weaker at raw arithmetic.
        "sim_error_rates": {
            "arithmetic": 0.18,
            "code": 0.15,
            "logic": 0.22,
            "factual": 0.10,
            "scheduling": 0.30,
        },
        "model_offset": 100_000,
    },
]


# ---------------------------------------------------------------------------
# 2. Deterministic question generators (identical to Exp 93 for comparability)
# ---------------------------------------------------------------------------


def _extract_number(text: str) -> float | None:
    """Pull the last number from a string (usually the final answer).

    **Detailed explanation for engineers:**
        Handles "The answer is 75", "75", "47 + 28 = 75", "Result: 75.0", etc.
        Returns the last found number, or None if no number is present.
    """
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if not numbers:
        return None
    try:
        val = float(numbers[-1])
        return int(val) if val == int(val) else val
    except (ValueError, OverflowError):
        return None


def _make_number_checker(expected: int | float) -> Any:
    """Create a closure that checks if a response contains the expected number."""
    def checker(ans: str) -> bool:
        extracted = _extract_number(ans)
        if extracted is None:
            return False
        return int(extracted) == int(expected)
    return checker


def _make_keyword_checker(keywords: list[str]) -> Any:
    """Create a checker that verifies the response contains at least one keyword."""
    def checker(ans: str) -> bool:
        ans_lower = ans.lower()
        return any(kw.lower() in ans_lower for kw in keywords)
    return checker


def _make_code_checker(keywords: list[str]) -> Any:
    """Create a checker for code responses: must contain 'def' and a keyword."""
    def checker(ans: str) -> bool:
        if "def " not in ans:
            return False
        ans_lower = ans.lower()
        return any(kw.lower() in ans_lower for kw in keywords)
    return checker


def generate_arithmetic_questions(n: int = 50, seed: int = 117) -> list[dict[str, Any]]:
    """Generate n arithmetic questions with ground truth (seed matches Exp 93 for comparability).

    **Detailed explanation for engineers:**
        Uses seed=117 so questions are traceable to this experiment, but
        same structure as Exp 93 to keep the problem distribution comparable.
        Mix of simple two-operand, multi-step, and word problems.
    """
    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []

    ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)]
    for i in range(20):
        op_sym, op_fn = ops[i % 3]
        if op_sym == "*":
            a, b = rng.randint(2, 30), rng.randint(2, 30)
        elif op_sym == "-":
            a = rng.randint(50, 500)
            b = rng.randint(1, a)
        else:
            a, b = rng.randint(10, 999), rng.randint(10, 999)
        answer = op_fn(a, b)
        questions.append({
            "domain": "arithmetic",
            "question": f"What is {a} {op_sym} {b}?",
            "ground_truth": str(answer),
            "check_answer": _make_number_checker(answer),
        })

    for i in range(15):
        a, b, c = rng.randint(2, 50), rng.randint(2, 50), rng.randint(2, 50)
        pattern = i % 3
        if pattern == 0:
            q = f"What is {a} + {b} + {c}?"
            answer = a + b + c
        elif pattern == 1:
            q = f"What is {a} * {b} + {c}?"
            answer = a * b + c
        else:
            q = f"What is {a} + {b} - {c}?"
            answer = a + b - c
        questions.append({
            "domain": "arithmetic",
            "question": q,
            "ground_truth": str(answer),
            "check_answer": _make_number_checker(answer),
        })

    templates = [
        ("A store has {a} items. They sell {b} and receive {c}. How many items remain?",
         lambda a, b, c: a - b + c),
        ("A classroom has {a} students. {b} leave and {c} arrive. How many now?",
         lambda a, b, c: a - b + c),
        ("A baker makes {a} loaves in the morning and {b} in the afternoon. "
         "If {c} are sold, how many remain?",
         lambda a, b, c: a + b - c),
    ]
    for i in range(15):
        tmpl, fn = templates[i % len(templates)]
        a = rng.randint(20, 100)
        b = rng.randint(1, a // 2)
        c = rng.randint(1, a // 2)
        answer = fn(a, b, c)
        questions.append({
            "domain": "arithmetic",
            "question": tmpl.format(a=a, b=b, c=c),
            "ground_truth": str(answer),
            "check_answer": _make_number_checker(answer),
        })

    return questions[:n]


def generate_code_questions(n: int = 50, seed: int = 117) -> list[dict[str, Any]]:
    """Generate n code-writing questions with keyword-based correctness checks."""
    code_templates = [
        ("reverse a string", ["reverse", "[::-1]"]),
        ("convert a string to uppercase", ["upper"]),
        ("count vowels in a string", ["vowel", "aeiou"]),
        ("check if a string is a palindrome", ["palindrome", "[::-1]"]),
        ("remove spaces from a string", ["replace", "strip", "join"]),
        ("count words in a string", ["split", "len"]),
        ("find the longest word in a string", ["max", "split", "len"]),
        ("capitalize the first letter of each word", ["title", "capitalize", "split"]),
        ("check if two strings are anagrams", ["sorted", "anagram"]),
        ("replace all vowels with '*'", ["replace", "aeiou"]),
        ("compute the factorial of n", ["factorial", "fact"]),
        ("compute the nth Fibonacci number", ["fib", "fibonacci"]),
        ("find the GCD of two numbers", ["gcd"]),
        ("check if a number is prime", ["prime", "is_prime"]),
        ("compute the sum of digits of a number", ["digit", "sum"]),
        ("find all prime factors of a number", ["prime", "factor"]),
        ("compute n choose k (combinations)", ["factorial", "comb"]),
        ("check if a number is a perfect square", ["sqrt", "square", "**"]),
        ("compute the LCM of two numbers", ["lcm", "gcd"]),
        ("convert decimal to binary string", ["bin", "binary", "//", "%"]),
        ("find the maximum value in a list", ["max"]),
        ("find the minimum value in a list", ["min"]),
        ("compute the sum of a list", ["sum"]),
        ("remove duplicates from a list preserving order", ["seen", "set", "unique"]),
        ("flatten a nested list", ["flatten", "extend", "isinstance"]),
        ("find the second largest element in a list", ["second", "sorted", "max"]),
        ("rotate a list by k positions", ["rotate", "[k:]", "[:k]"]),
        ("merge two sorted lists", ["merge", "sorted", "while"]),
        ("find the intersection of two lists", ["intersection", "set", "in"]),
        ("find the union of two lists without duplicates", ["union", "set"]),
        ("implement binary search", ["binary", "search", "mid", "low", "high"]),
        ("implement bubble sort", ["bubble", "sort", "swap"]),
        ("implement insertion sort", ["insertion", "sort", "key", "while"]),
        ("implement selection sort", ["selection", "sort", "min"]),
        ("implement linear search", ["linear", "search", "for"]),
        ("implement merge sort", ["merge", "sort", "mid", "left", "right"]),
        ("find two numbers that sum to target (two sum)", ["two", "sum", "target", "dict"]),
        ("check if parentheses are balanced", ["balanced", "stack", "(", ")"]),
        ("implement a stack using a list", ["stack", "push", "pop", "append"]),
        ("implement a queue using a list", ["queue", "enqueue", "dequeue", "append"]),
        ("find the majority element in a list", ["majority", "count", "> n//2"]),
        ("implement FizzBuzz", ["fizz", "buzz", "% 3", "% 5"]),
        ("find the longest common prefix of strings", ["prefix", "common", "zip"]),
        ("compute the Hamming distance between strings", ["hamming", "distance", "!="]),
        ("implement run-length encoding", ["run", "length", "encoding", "count"]),
        ("implement Caesar cipher encryption", ["caesar", "cipher", "shift", "chr", "ord"]),
        ("check if a number is a power of two", ["power", "two", "& ", "log"]),
        ("implement Euclidean algorithm for GCD", ["euclidean", "gcd", "%", "while"]),
        ("generate Pascal's triangle row n", ["pascal", "triangle", "row"]),
        ("find missing number in range 1..n", ["missing", "sum", "n*(n+1)//2", "xor"]),
    ]

    rng = random.Random(seed)
    selected = code_templates[:n]
    rng.shuffle(selected)

    questions: list[dict[str, Any]] = []
    for desc, keywords in selected:
        questions.append({
            "domain": "code",
            "question": f"Write a Python function to {desc}.",
            "ground_truth": f"def {desc.split()[0]}",
            "keywords": keywords,
            "check_answer": _make_code_checker(keywords),
        })
    return questions[:n]


def generate_logic_questions(n: int = 50, seed: int = 117) -> list[dict[str, Any]]:
    """Generate n logic questions (syllogisms, modus ponens/tollens, contradictions)."""
    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []

    categories = [
        ("mammals", "animals"), ("dogs", "mammals"), ("birds", "animals"),
        ("roses", "flowers"), ("oaks", "trees"), ("salmon", "fish"),
        ("apples", "fruits"), ("cars", "vehicles"), ("novels", "books"),
        ("triangles", "shapes"), ("violins", "instruments"), ("gold", "metals"),
    ]
    instances = [
        ("Rex", "dogs"), ("Tweety", "birds"), ("Nemo", "salmon"),
        ("Bessie", "mammals"), ("Fido", "dogs"), ("Polly", "birds"),
    ]

    for i in range(18):
        cat, supercat = categories[i % len(categories)]
        inst_name = instances[i % len(instances)][0]
        q = (f"All {cat} are {supercat}. {inst_name} is a {cat}. "
             f"Is {inst_name} a {supercat}? Answer yes or no.")
        questions.append({
            "domain": "logic", "question": q,
            "ground_truth": "yes",
            "check_answer": lambda ans: "yes" in ans.lower(),
        })

    for i in range(12):
        cat, supercat = categories[i % len(categories)]
        inst_name = instances[i % len(instances)][0]
        q = (f"All {cat} are {supercat}. {inst_name} is a {supercat}. "
             f"Is {inst_name} necessarily a {cat}? Answer yes or no.")
        questions.append({
            "domain": "logic", "question": q,
            "ground_truth": "no",
            "check_answer": lambda ans: "no" in ans.lower(),
        })

    propositions = [
        ("it rains", "the ground is wet"),
        ("you study hard", "you pass the exam"),
        ("the alarm sounds", "people evacuate"),
        ("the temperature drops below 0", "water freezes"),
        ("the circuit is complete", "current flows"),
    ]
    for i in range(10):
        p, q = propositions[i % len(propositions)]
        if i % 2 == 0:
            q_text = (f"If {p}, then {q}. {p.capitalize()}. "
                      f"Does {q}? Answer yes or no.")
            expected = "yes"
        else:
            q_text = (f"If {p}, then {q}. {q.capitalize()} did NOT happen. "
                      f"Did {p}? Answer yes or no.")
            expected = "no"
        questions.append({
            "domain": "logic", "question": q_text,
            "ground_truth": expected,
            "check_answer": _make_keyword_checker([expected]),
        })

    for i in range(10):
        cat, supercat = categories[i % len(categories)]
        if i % 2 == 0:
            q_text = (f"Statement A: 'All {cat} are {supercat}.' "
                      f"Statement B: 'Some {cat} are not {supercat}.' "
                      f"Are these statements consistent? Answer yes or no.")
            expected = "no"
        else:
            q_text = (f"Statement A: 'Some {cat} are {supercat}.' "
                      f"Statement B: 'Some {cat} are not {supercat}.' "
                      f"Are these statements consistent? Answer yes or no.")
            expected = "yes"
        questions.append({
            "domain": "logic", "question": q_text,
            "ground_truth": expected,
            "check_answer": _make_keyword_checker([expected]),
        })

    rng.shuffle(questions)
    return questions[:n]


def generate_factual_questions(n: int = 50, seed: int = 117) -> list[dict[str, Any]]:
    """Generate n factual questions with verifiable ground truth."""
    factual_bank = [
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
        ("What is the value of pi to two decimal places?", "3.14", ["3.14"]),
        ("How many seconds are in an hour?", "3600", ["3600"]),
        ("How many days are in a non-leap year?", "365", ["365"]),
        ("What is the boiling point of water in Celsius at sea level?", "100", ["100"]),
        ("How many bits are in a byte?", "8", ["8"]),
        ("In what year did World War II end?", "1945", ["1945"]),
        ("In what year did the Berlin Wall fall?", "1989", ["1989"]),
        ("In what year was the Declaration of Independence signed?", "1776", ["1776"]),
        ("In what year did humans first land on the Moon?", "1969", ["1969"]),
        ("In what year did World War I begin?", "1914", ["1914"]),
        ("In what year did the Titanic sink?", "1912", ["1912"]),
        ("In what year did the French Revolution begin?", "1789", ["1789"]),
        ("In what year did Columbus first reach the Americas?", "1492", ["1492"]),
        ("In what year was the United Nations founded?", "1945", ["1945"]),
        ("In what year was the first iPhone released?", "2007", ["2007"]),
        ("What is the largest continent by area?", "Asia", ["asia"]),
        ("What is the longest river in the world?", "Nile", ["nile"]),
        ("What is the largest ocean?", "Pacific", ["pacific"]),
        ("What is the tallest mountain in the world?", "Everest", ["everest"]),
        ("What is the largest country by area?", "Russia", ["russia"]),
        ("What is the smallest country by area?", "Vatican City", ["vatican"]),
        ("What continent is Brazil on?", "South America", ["south america"]),
        ("How many continents are there?", "7", ["7"]),
        ("How many oceans are there?", "5", ["5"]),
        ("What is the largest desert by area?", "Antarctic", ["antarctic"]),
        ("What is the most populous country?", "India", ["india"]),
        ("What is the largest island in the world?", "Greenland", ["greenland"]),
        ("What river flows through London?", "Thames", ["thames"]),
        ("What river flows through Paris?", "Seine", ["seine"]),
        ("What is the capital of the United States?", "Washington, D.C.", ["washington"]),
        ("What is the currency of Japan?", "Yen", ["yen"]),
        ("What is the currency of the United Kingdom?", "Pound sterling", ["pound"]),
        ("What is the currency of the European Union?", "Euro", ["euro"]),
        ("In what year was penicillin discovered?", "1928", ["1928"]),
        ("In what year did the Wright Brothers first fly?", "1903", ["1903"]),
    ]

    rng = random.Random(seed)
    selected = factual_bank[:n]
    rng.shuffle(selected)

    questions: list[dict[str, Any]] = []
    for q_text, gt, kws in selected:
        questions.append({
            "domain": "factual",
            "question": q_text,
            "ground_truth": gt,
            "check_answer": _make_keyword_checker(kws),
        })
    return questions[:n]


def _compute_depth(node: int, adj: dict[int, list[int]], depth: dict[int, int]) -> int:
    """Compute longest path from node to any leaf in a DAG (memoized DFS).

    **Detailed explanation for engineers:**
        Used for task dependency scheduling: the depth of a leaf is 1,
        an internal node is 1 + max(depth of children). Gives the minimum
        number of sequential rounds needed for the critical path.
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


def generate_scheduling_questions(n: int = 50, seed: int = 117) -> list[dict[str, Any]]:
    """Generate n scheduling constraint satisfaction questions."""
    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []

    for i in range(20):
        n_meetings = rng.randint(3, 6)
        n_slots = rng.randint(3, 8)
        n_conflicts = rng.randint(1, n_meetings - 1)
        conflicts = []
        for _ in range(n_conflicts):
            a = rng.randint(1, n_meetings)
            b = rng.randint(1, n_meetings)
            while b == a:
                b = rng.randint(1, n_meetings)
            conflicts.append((min(a, b), max(a, b)))
        conflicts = list(set(conflicts))
        can_schedule = len(conflicts) < n_slots
        answer = "yes" if can_schedule else "no"
        conflict_str = "; ".join(
            f"Meeting {a} and Meeting {b} cannot overlap" for a, b in conflicts
        )
        q = (f"You have {n_meetings} meetings to schedule in {n_slots} time slots. "
             f"Constraints: {conflict_str}. "
             f"Can all meetings be scheduled without conflicts? Answer yes or no.")
        questions.append({
            "domain": "scheduling", "question": q,
            "ground_truth": answer,
            "check_answer": _make_keyword_checker([answer]),
        })

    for i in range(15):
        n_tasks = rng.randint(3, 6)
        deps = []
        for t in range(2, n_tasks + 1):
            dep = rng.randint(1, t - 1)
            deps.append((dep, t))
        dep_str = ", ".join(f"Task {a} must come before Task {b}" for a, b in deps)
        q = (f"You have {n_tasks} tasks with these dependencies: {dep_str}. "
             f"What is the minimum number of rounds needed if independent tasks "
             f"can run in parallel?")
        adj: dict[int, list[int]] = {t: [] for t in range(1, n_tasks + 1)}
        for a, b in deps:
            adj[a].append(b)
        depth: dict[int, int] = {}
        for t in range(1, n_tasks + 1):
            if t not in depth:
                _compute_depth(t, adj, depth)
        longest = max(depth.values()) if depth else 1
        questions.append({
            "domain": "scheduling", "question": q,
            "ground_truth": str(longest),
            "check_answer": _make_number_checker(longest),
        })

    for i in range(15):
        n_tasks = rng.randint(3, 5)
        capacity = rng.randint(5, 15)
        demands = [rng.randint(1, 6) for _ in range(n_tasks)]
        total_demand = sum(demands)
        fits = total_demand <= capacity
        answer = "yes" if fits else "no"
        demand_str = ", ".join(
            f"Task {j + 1} needs {d} units" for j, d in enumerate(demands)
        )
        q = (f"A server has {capacity} units of capacity. "
             f"Tasks: {demand_str}. "
             f"Can all tasks run simultaneously? Answer yes or no.")
        questions.append({
            "domain": "scheduling", "question": q,
            "ground_truth": answer,
            "check_answer": _make_keyword_checker([answer]),
        })

    rng.shuffle(questions)
    return questions[:n]


# ---------------------------------------------------------------------------
# 3. LLM loading and generation
# ---------------------------------------------------------------------------


def load_model(config: dict[str, Any]) -> tuple[Any, Any, str, bool]:
    """Attempt to load a model from config; return (tokenizer, model, device, success).

    **Detailed explanation for engineers:**
        Tries each candidate model name in order. Runs a smoke test in a
        subprocess (60s timeout) to catch ROCm hangs. Falls back to
        (None, None, "cpu", False) and simulated outputs if loading fails.
    """
    if os.environ.get("CARNOT_SKIP_LLM", ""):
        print(f"    CARNOT_SKIP_LLM set — skipping {config['name']}.")
        return None, None, "cpu", False

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for model_name in config["candidates"]:
            try:
                print(f"    Loading {model_name} on {device}...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=config.get("trust_remote_code", True),
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=config.get("trust_remote_code", True),
                    torch_dtype=torch.float16 if device == "cuda" else None,
                )
                if device == "cuda":
                    model = model.cuda()
                model.eval()
                print(f"    Loaded {model_name}. Running smoke test...")

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
                        timeout=60, capture_output=True, text=True,
                    )
                    if "OK" in result.stdout:
                        print(f"    Smoke test passed. {config['name']} ready.")
                        return tokenizer, model, device, True
                    else:
                        print(f"    Smoke test failed: {result.stderr[:200]}")
                except subprocess.TimeoutExpired:
                    print(f"    Smoke test timed out (60s).")
                except Exception as e:
                    print(f"    Smoke test error: {e}")

                del model, tokenizer
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"    Failed to load {model_name}: {e}")

    except ImportError as e:
        print(f"    torch/transformers not available: {e}")

    return None, None, "cpu", False


def generate_with_llm(
    prompt: str, tokenizer: Any, model: Any, device: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate a response from a loaded HuggingFace model (greedy, reproducible)."""
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
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    return response


def simulate_response(
    question: dict[str, Any],
    model_config: dict[str, Any],
    iteration: int = 0,
    rng: random.Random | None = None,
    guided_factor: float | None = None,
) -> str:
    """Generate a simulated LLM response for fallback mode.

    **Detailed explanation for engineers:**
        When the real LLM is unavailable, this simulates responses using the
        model-specific error rates. On repair iterations (iteration > 0),
        the error rate decreases to model constraint feedback helping.

        The ``guided_factor`` parameter overrides the iteration-based factor
        for the guided_generation mode, where energy guidance during decoding
        is modeled as a stronger reduction than a single repair iteration.
        guided_factor=0.25 means the model only makes errors 25% of the time
        it would without guidance — stronger than repair iteration 1 (40%).
    """
    if rng is None:
        rng = random.Random(42)

    domain = question["domain"]
    gt = question["ground_truth"]
    error_rates = model_config.get("sim_error_rates", {})
    base_error = error_rates.get(domain, 0.2)

    # Guided generation applies energy constraints during decoding,
    # effectively reducing errors more than a single repair iteration.
    if guided_factor is not None:
        effective_error = base_error * guided_factor
    else:
        # Repair iterations: each iteration reduces error by 60% (factor 0.4).
        effective_error = base_error * (0.4 ** iteration)

    is_correct = rng.random() > effective_error

    if domain == "arithmetic":
        if is_correct:
            return f"Answer: {gt}"
        else:
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
            return (f"```python\ndef {func_name}(x):\n"
                    f"    # Implementation using {kw}\n"
                    f"    return {kw}(x) if callable({kw}) else x\n```")
        else:
            return "The function would iterate over the input and return the result."

    elif domain == "logic":
        if is_correct:
            return f"Answer: {gt}"
        else:
            return f"Answer: {'no' if gt == 'yes' else 'yes'}"

    elif domain == "factual":
        if is_correct:
            return f"Answer: {gt}"
        else:
            return "Answer: I'm not sure, but I think it might be something else."

    elif domain == "scheduling":
        if is_correct:
            return f"Answer: {gt}"
        else:
            if gt in ("yes", "no"):
                return f"Answer: {'no' if gt == 'yes' else 'yes'}"
            try:
                wrong = int(gt) + rng.choice([-1, 1, 2])
                return f"Answer: {wrong}"
            except (ValueError, TypeError):
                return f"Answer: {gt}"

    return f"Answer: {gt}"


# ---------------------------------------------------------------------------
# 4. Constraint extraction with per-extractor tracking
# ---------------------------------------------------------------------------


def build_prompt(question: str, domain: str) -> str:
    """Build a domain-specific prompt for the LLM.

    **Detailed explanation for engineers:**
        Each domain gets a tailored prompt encouraging structured, verifiable
        output. Short prompts perform better with small models (0.6–0.8B).
    """
    if domain == "arithmetic":
        return (f"Question: {question}\n"
                f"Think step by step. Give the answer as a number.\n"
                f"Format:\nAnswer: <number>")
    elif domain == "code":
        return (f"Question: {question}\n"
                f"Write ONLY the Python function. No explanation.")
    elif domain == "logic":
        return (f"Question: {question}\n"
                f"Think step by step. Give a clear answer.\n"
                f"Format:\nAnswer: <your answer>")
    elif domain == "scheduling":
        return (f"Question: {question}\n"
                f"Think step by step about the constraints. Give a clear answer.\n"
                f"Format:\nAnswer: <your answer>")
    else:  # factual
        return (f"Question: {question}\n"
                f"Give a short, direct factual answer.\n"
                f"Format:\nAnswer: <answer>")


def extract_constraints_with_breakdown(
    response: str, domain: str
) -> tuple[list[dict], dict[str, dict]]:
    """Extract constraints using AutoExtractor and track per-extractor contribution.

    **Detailed explanation for engineers:**
        Runs each extractor (Arithmetic, Code, Logic, NL, FactualKB) individually
        on the response, collecting:
        - n_constraints: how many constraints this extractor found
        - n_violated: how many constraints had satisfied=False in metadata

        This per-extractor breakdown shows which components of the v11+v12
        stack are active for each domain, and which provide the most signal
        for detecting errors.

        Returns a tuple of:
        - constraints: list of unified dicts (type, satisfied, description)
        - extractor_breakdown: {extractor_name: {n_constraints, n_violated}}
    """
    try:
        from carnot.pipeline.extract import (
            ArithmeticExtractor,
            CodeExtractor,
            LogicExtractor,
            NLExtractor,
        )
        from carnot.pipeline.knowledge_base import FactualKBExtractor
    except ImportError as e:
        # Fallback: return empty if imports fail.
        print(f"    Warning: extractor import failed: {e}")
        return [], {}

    extractors = {
        "arithmetic": ArithmeticExtractor(),
        "code": CodeExtractor(),
        "logic": LogicExtractor(),
        "nl": NLExtractor(),
        "factual_kb": FactualKBExtractor(),
    }

    all_constraints: list[dict] = []
    extractor_breakdown: dict[str, dict] = {}
    seen_descriptions: set[str] = set()

    for ext_name, ext in extractors.items():
        try:
            results = ext.extract(response, domain if domain in ext.supported_domains else None)
        except Exception:
            results = []

        n_c = 0
        n_v = 0
        for cr in results:
            if cr.description not in seen_descriptions:
                seen_descriptions.add(cr.description)
                satisfied = cr.metadata.get("satisfied")
                all_constraints.append({
                    "type": cr.constraint_type,
                    "description": cr.description,
                    "satisfied": satisfied,
                    "extractor": ext_name,
                })
                n_c += 1
                if satisfied is False:
                    n_v += 1
            else:
                # Duplicate across extractors: still count for this extractor's breakdown.
                pass

        extractor_breakdown[ext_name] = {
            "n_constraints": n_c,
            "n_violated": n_v,
        }

    return all_constraints, extractor_breakdown


def format_violations(constraints: list[dict], domain: str) -> str:
    """Convert constraint violations to natural-language feedback for repair.

    **Detailed explanation for engineers:**
        Produces a feedback string the LLM can act on. This is the bridge
        between the EBM verification layer and the LLM's natural-language
        interface. Arithmetic violations include the computed correct value;
        other violations include the description text.
    """
    violated = [c for c in constraints if c.get("satisfied") is False]
    if not violated:
        return ""

    lines = ["Your answer has the following errors:"]
    for i, v in enumerate(violated, 1):
        vtype = v.get("type", "unknown")
        desc = v.get("description", vtype)
        if "arithmetic" in vtype:
            lines.append(f"  {i}. Arithmetic error: {desc}")
        else:
            lines.append(f"  {i}. {desc}")
    lines.append("\nPlease fix these errors and try again.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. The four benchmark modes
# ---------------------------------------------------------------------------


def run_baseline(
    question: dict[str, Any],
    model_config: dict[str, Any],
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live_llm: bool = False,
    sim_rng: random.Random | None = None,
) -> dict[str, Any]:
    """Mode A — Baseline: LLM alone, no verification.

    **Detailed explanation for engineers:**
        Control condition. Generates a response, checks correctness, records
        time. No constraint extraction, no repair. Per-extractor breakdown
        is still computed for analysis but does NOT affect the response.
    """
    t0 = time.time()
    prompt = build_prompt(question["question"], question["domain"])

    if use_live_llm:
        response = generate_with_llm(prompt, tokenizer, model, device)
    else:
        response = simulate_response(question, model_config, iteration=0, rng=sim_rng)

    elapsed = time.time() - t0
    correct = question["check_answer"](response)

    # Still run extractors for per-extractor analysis (non-invasive).
    _, breakdown = extract_constraints_with_breakdown(response, question["domain"])

    return {
        "mode": "baseline",
        "response": response,
        "correct": correct,
        "time_s": elapsed,
        "n_constraints": 0,
        "n_violated": 0,
        "ising_energy": 0.0,
        "n_repairs": 0,
        "extractor_breakdown": breakdown,
    }


def run_verify_only(
    question: dict[str, Any],
    model_config: dict[str, Any],
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live_llm: bool = False,
    sim_rng: random.Random | None = None,
) -> dict[str, Any]:
    """Mode B — Verify-only: LLM + constraint verification, no repair.

    **Detailed explanation for engineers:**
        Detects violations but does NOT feed back to the LLM. Measures
        detection accuracy — whether the extractor stack correctly identifies
        wrong answers. The per-extractor breakdown shows which extractors
        fired and which found violations.
    """
    t0 = time.time()
    prompt = build_prompt(question["question"], question["domain"])

    if use_live_llm:
        response = generate_with_llm(prompt, tokenizer, model, device)
    else:
        response = simulate_response(question, model_config, iteration=0, rng=sim_rng)

    constraints, breakdown = extract_constraints_with_breakdown(response, question["domain"])
    n_constraints = len(constraints)
    n_violated = sum(1 for c in constraints if c.get("satisfied") is False)
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
        "extractor_breakdown": breakdown,
    }


def run_verify_repair(
    question: dict[str, Any],
    model_config: dict[str, Any],
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live_llm: bool = False,
    sim_rng: random.Random | None = None,
    max_iters: int = 3,
) -> dict[str, Any]:
    """Mode C — Verify-repair: LLM + verification + iterative repair loop.

    **Detailed explanation for engineers:**
        Full pipeline: generate → verify → repair → re-verify, up to max_iters.
        Tracks initial vs final correctness, repair count, and total constraint
        coverage across all iterations. The extractor breakdown is from the
        final iteration (after all repairs).
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
    constraints: list[dict] = []
    breakdown: dict[str, dict] = {}

    for iteration in range(max_iters + 1):
        if iteration == 0:
            prompt = build_prompt(q_text, domain)
        else:
            feedback = format_violations(constraints, domain)
            if not feedback:
                break
            prompt = (f"Question: {q_text}\n\n"
                      f"Your previous answer was:\n{response}\n\n"
                      f"However, verification found problems:\n{feedback}\n\n"
                      f"Please provide a corrected answer.\n"
                      f"Format:\nAnswer: <your corrected answer>")

        if use_live_llm:
            response = generate_with_llm(prompt, tokenizer, model, device)
        else:
            response = simulate_response(
                question, model_config, iteration=iteration, rng=sim_rng,
            )

        if iteration == 0:
            initial_correct = question["check_answer"](response)

        constraints, breakdown = extract_constraints_with_breakdown(response, domain)
        n_c = len(constraints)
        n_v = sum(1 for c in constraints if c.get("satisfied") is False)

        total_constraints += n_c
        total_violated += n_v
        total_energy += float(n_v)

        if n_v == 0:
            break
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
        "extractor_breakdown": breakdown,
    }


def run_guided_generation(
    question: dict[str, Any],
    model_config: dict[str, Any],
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live_llm: bool = False,
    sim_rng: random.Random | None = None,
    alpha: float = 0.5,
    check_every_k: int = 1,
) -> dict[str, Any]:
    """Mode D — Guided generation: energy-steered decoding via EnergyGuidedSampler.

    **Detailed explanation for engineers:**
        Unlike post-hoc repair (Mode C), this mode applies constraint energy
        as a logit penalty during token generation itself. The EnergyGuidedSampler
        (carnot.inference.guided_decoding) recomputes constraint violations every
        check_every_k tokens and subtracts alpha × energy from all logits, nudging
        the model away from constraint-violating continuations.

        When using a real HuggingFace model:
            EnergyGuidedSampler.generate(prompt, model, tokenizer) is called
            directly, producing a GuidedDecodingResult with the guided text.

        In simulation mode (no real model):
            We simulate the energy guidance effect by using a lower error factor
            (guided_factor=0.25 vs. repair's per-iter 0.40). This models the
            empirical finding from Exp 110 that energy-guided generation reduces
            errors more effectively than a single post-hoc repair step.

        The alpha=0.5 and check_every_k=1 defaults match Exp 110's best-performing
        configuration (see experiment_110_results.json).

    Args:
        question: Question dict with domain, question text, check_answer.
        model_config: Model config with sim_error_rates.
        tokenizer, model, device: Real model if available; None otherwise.
        use_live_llm: Whether a real model was loaded.
        sim_rng: RNG for deterministic simulation.
        alpha: Guidance strength (default 0.5, from Exp 110 sweep).
        check_every_k: Energy recheck interval (default 1 = every token).
    """
    t0 = time.time()
    domain = question["domain"]
    q_text = question["question"]
    prompt = build_prompt(q_text, domain)

    if use_live_llm:
        # Use EnergyGuidedSampler for real models.
        try:
            from carnot.inference.guided_decoding import EnergyGuidedSampler

            sampler = EnergyGuidedSampler(alpha=alpha, check_every_k=check_every_k)
            result = sampler.generate(
                prompt, model, tokenizer, max_tokens=256, temperature=0.0,
                domain=domain,
            )
            response = result.text
            energy_checks = result.energy_checks
            final_energy = result.final_energy
            mean_penalty = result.mean_penalty
        except Exception as e:
            # If guided sampler fails (e.g. architecture mismatch), fall back to
            # standard greedy generation — this ensures we still get a result.
            print(f"      Guided sampler failed, falling back to greedy: {e}")
            response = generate_with_llm(prompt, tokenizer, model, device)
            energy_checks = 0
            final_energy = 0.0
            mean_penalty = 0.0
    else:
        # Simulation mode: model the guidance effect as a stronger error reduction
        # than a single repair step (factor 0.25 vs repair factor 0.40).
        # This reflects that guiding during generation is more effective than
        # post-hoc correction because bad paths are never fully explored.
        response = simulate_response(
            question, model_config, iteration=0, rng=sim_rng,
            guided_factor=0.25,
        )
        energy_checks = 5  # Simulated: typical tokens in a short answer
        final_energy = 0.0
        mean_penalty = alpha * 0.5  # Simulated: moderate penalty applied

    elapsed = time.time() - t0
    correct = question["check_answer"](response)

    # Post-generation verification for metrics (does NOT change the response).
    constraints, breakdown = extract_constraints_with_breakdown(response, domain)
    n_constraints = len(constraints)
    n_violated = sum(1 for c in constraints if c.get("satisfied") is False)

    return {
        "mode": "guided_generation",
        "response": response,
        "correct": correct,
        "time_s": elapsed,
        "n_constraints": n_constraints,
        "n_violated": n_violated,
        "ising_energy": float(n_violated),
        "n_repairs": 0,
        "energy_checks": energy_checks,
        "final_energy": final_energy,
        "mean_penalty": mean_penalty,
        "extractor_breakdown": breakdown,
        "alpha": alpha,
        "check_every_k": check_every_k,
    }


# ---------------------------------------------------------------------------
# 6. Statistical analysis
# ---------------------------------------------------------------------------


def paired_t_test(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute paired t-test statistic and p-value.

    **Detailed explanation for engineers:**
        Tests whether the mean difference between paired observations is
        significantly different from zero. Uses scipy if available, otherwise
        falls back to a normal approximation (good for n >= 30).
    """
    n = len(x)
    if n < 2:
        return 0.0, 1.0

    diffs = [xi - yi for xi, yi in zip(x, y)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)

    if var_d == 0:
        return 0.0, 1.0

    se = math.sqrt(var_d / n)
    t_stat = mean_d / se

    try:
        from scipy import stats
        p_value = stats.t.sf(abs(t_stat), df=n - 1) * 2
    except ImportError:
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))

    return t_stat, p_value


def bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 117,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI on the mean accuracy difference (A - B).

    **Detailed explanation for engineers:**
        Resamples (scores_a, scores_b) pairs with replacement n_bootstrap times.
        For each resample, computes mean(A) - mean(B). Returns (lower, upper, observed_delta)
        where lower/upper are the ci-level percentile bounds.

        This gives a non-parametric confidence interval that is valid even when
        the score distribution is not normal (binary correct/incorrect scores
        satisfy this condition once n >= 30 per the CLT, but bootstrap is safer).

    Args:
        scores_a: List of 0/1 correctness scores for mode A (typically better mode).
        scores_b: List of 0/1 correctness scores for mode B (baseline).
        n_bootstrap: Number of bootstrap resamples (2000 is standard practice).
        ci: Confidence level (default 0.95 = 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        (lower_bound, upper_bound, observed_delta) where observed_delta = mean(A) - mean(B).
    """
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    if n < 2:
        delta = (sum(scores_a) / n if n else 0) - (sum(scores_b) / n if n else 0)
        return delta, delta, delta

    arr_a = np.array(scores_a, dtype=float)
    arr_b = np.array(scores_b, dtype=float)
    observed_delta = float(arr_a.mean() - arr_b.mean())

    bootstrap_deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        delta = float(arr_a[idx].mean() - arr_b[idx].mean())
        bootstrap_deltas.append(delta)

    bootstrap_deltas.sort()
    alpha_half = (1.0 - ci) / 2.0
    lower = float(np.percentile(bootstrap_deltas, 100 * alpha_half))
    upper = float(np.percentile(bootstrap_deltas, 100 * (1.0 - alpha_half)))

    return lower, upper, observed_delta


def load_v10_baseline() -> dict[str, dict[str, float]] | None:
    """Load v10 (Exp 93) accuracy numbers for side-by-side comparison.

    **Detailed explanation for engineers:**
        Reads experiment_93_results.json and extracts per-(model, domain)
        accuracy for baseline and verify_repair modes. Returns a dict keyed
        as {model_name: {domain_mode: accuracy}} or None if the file is missing.
    """
    if not EXP93_RESULTS_PATH.exists():
        print(f"  Note: Exp 93 results not found at {EXP93_RESULTS_PATH}. "
              f"Skipping v10 comparison.")
        return None

    try:
        with open(EXP93_RESULTS_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: Failed to load Exp 93 results: {e}")
        return None

    baseline: dict[str, dict[str, float]] = {}
    for row in data.get("metrics_table", []):
        model = row["model"]
        domain = row["domain"]
        mode = row["mode"]
        if model not in baseline:
            baseline[model] = {}
        baseline[model][f"{domain}_{mode}"] = row["accuracy"]

    return baseline


# ---------------------------------------------------------------------------
# 7. Main benchmark
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the full four-mode benchmark with v12 extractor stack."""
    print("=" * 80)
    print("EXPERIMENT 117: Full v12 Benchmark — All Extractors + Guided Generation")
    print("  Models: Qwen3.5-0.8B vs Gemma4-E4B-it")
    print("  250 questions × 2 models × 4 modes = 2,000 evaluations")
    print("  Domains: arithmetic, code, logic, factual, scheduling")
    print("  Modes: A=baseline, B=verify-only, C=verify-repair, D=guided-generation")
    print("  Extractors: ArithmeticExtractor, CodeExtractor, LogicExtractor,")
    print("              NLExtractor (v11), FactualKBExtractor (v12/Exp 113)")
    print("=" * 80)

    overall_start = time.time()

    # Load v10 baseline for comparison.
    v10_baseline = load_v10_baseline()
    if v10_baseline:
        print(f"  Loaded v10 (Exp 93) baseline from {EXP93_RESULTS_PATH}")
    else:
        print("  No v10 baseline found — skipping version comparison table.")

    # Generate all question sets (identical across models for paired comparison).
    print("\n  Generating question sets (50 per domain)...")
    all_questions: dict[str, list[dict[str, Any]]] = {
        "arithmetic": generate_arithmetic_questions(50),
        "code": generate_code_questions(50),
        "logic": generate_logic_questions(50),
        "factual": generate_factual_questions(50),
        "scheduling": generate_scheduling_questions(50),
    }
    total_questions = sum(len(qs) for qs in all_questions.values())
    print(f"  Generated {total_questions} questions total.")

    domains = ["arithmetic", "code", "logic", "factual", "scheduling"]
    modes = ["baseline", "verify_only", "verify_repair", "guided_generation"]

    # all_results: model_name -> domain -> mode -> list of result dicts
    all_results: dict[str, dict[str, dict[str, list[dict]]]] = {}

    for mc in MODEL_CONFIGS:
        model_name = mc["name"]
        print(f"\n{'=' * 80}")
        print(f"  MODEL: {model_name}")
        print(f"{'=' * 80}")

        print(f"  Attempting to load {model_name}...")
        tokenizer, model, device, use_live_llm = load_model(mc)

        if not use_live_llm:
            print(f"  *** FALLBACK: Using simulated outputs for {model_name} ***")
            rates = mc.get("sim_error_rates", {})
            rate_str = ", ".join(f"{d}={r:.0%}" for d, r in rates.items())
            print(f"  Simulated error rates: {rate_str}")

        results: dict[str, dict[str, list[dict]]] = {
            d: {m: [] for m in modes} for d in domains
        }

        for domain in domains:
            questions = all_questions[domain]
            print(f"\n    {model_name} / {domain} ({len(questions)} questions)...")

            for qi, q in enumerate(questions):
                # Independent RNGs per mode so each mode gets an independent
                # simulated response. Same seed per (model, question index) so
                # the baseline response is deterministically reproducible.
                model_offset = mc.get("model_offset", 0)
                sim_rng_a = random.Random(117_000 + qi + model_offset)
                sim_rng_b = random.Random(117_000 + qi + model_offset)
                sim_rng_c = random.Random(117_000 + qi + model_offset)
                sim_rng_d = random.Random(117_000 + qi + model_offset)

                r_a = run_baseline(
                    q, mc, tokenizer=tokenizer, model=model, device=device,
                    use_live_llm=use_live_llm, sim_rng=sim_rng_a,
                )
                results[domain]["baseline"].append(r_a)

                r_b = run_verify_only(
                    q, mc, tokenizer=tokenizer, model=model, device=device,
                    use_live_llm=use_live_llm, sim_rng=sim_rng_b,
                )
                results[domain]["verify_only"].append(r_b)

                r_c = run_verify_repair(
                    q, mc, tokenizer=tokenizer, model=model, device=device,
                    use_live_llm=use_live_llm, sim_rng=sim_rng_c, max_iters=3,
                )
                results[domain]["verify_repair"].append(r_c)

                r_d = run_guided_generation(
                    q, mc, tokenizer=tokenizer, model=model, device=device,
                    use_live_llm=use_live_llm, sim_rng=sim_rng_d,
                    alpha=0.5, check_every_k=1,
                )
                results[domain]["guided_generation"].append(r_d)

                if (qi + 1) % 25 == 0:
                    print(f"      {qi + 1}/{len(questions)} done")

        # Free model memory before loading the next one.
        if use_live_llm:
            del model, tokenizer
            import torch
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        all_results[model_name] = results

    # -----------------------------------------------------------------------
    # Compute metrics per cell (model × domain × mode)
    # -----------------------------------------------------------------------
    elapsed = time.time() - overall_start
    sep = "=" * 80

    print(f"\n{sep}")
    print(f"EXPERIMENT 117 RESULTS ({elapsed:.1f}s)")
    print(sep)

    mode_labels = {
        "baseline":           "A. Baseline",
        "verify_only":        "B. Verify-only",
        "verify_repair":      "C. Verify+Repair",
        "guided_generation":  "D. Guided-Gen",
    }

    metrics_table: list[dict[str, Any]] = []

    for mc in MODEL_CONFIGS:
        model_name = mc["name"]
        results = all_results[model_name]

        print(f"\n  --- {model_name} ---")
        print(f"  {'Domain':<12s} {'Mode':<18s} {'Acc':>7s} {'Halluc%':>8s} "
              f"{'Repair':>8s} {'AvgConstr':>10s} {'Time(s)':>8s}")
        print(f"  {'-' * 76}")

        for domain in domains:
            for mode in modes:
                rs = results[domain][mode]
                n_total = len(rs)
                n_correct = sum(1 for r in rs if r["correct"])
                accuracy = n_correct / n_total if n_total else 0
                halluc_rate = 1.0 - accuracy
                avg_constraints = np.mean([r["n_constraints"] for r in rs]) if rs else 0
                total_time = sum(r["time_s"] for r in rs)

                if mode == "verify_repair":
                    n_rep_attempted = sum(1 for r in rs if r["n_repairs"] > 0)
                    n_rep_ok = sum(1 for r in rs if r.get("repaired", False))
                    repair_str = f"{n_rep_ok}/{n_rep_attempted}" if n_rep_attempted > 0 else "n/a"
                    repair_rate = n_rep_ok / n_rep_attempted if n_rep_attempted > 0 else 0.0
                else:
                    n_rep_attempted = 0
                    n_rep_ok = 0
                    repair_str = "n/a"
                    repair_rate = 0.0

                print(f"  {domain:<12s} {mode_labels[mode]:<18s} "
                      f"{n_correct}/{n_total:>3} {halluc_rate:>7.1%} "
                      f"{repair_str:>8s} {avg_constraints:>10.1f} {total_time:>8.3f}")

                # Aggregate per-extractor contribution across all questions in cell.
                extractor_agg: dict[str, dict[str, float]] = {}
                for r in rs:
                    bd = r.get("extractor_breakdown", {})
                    for ext_name, ext_stats in bd.items():
                        if ext_name not in extractor_agg:
                            extractor_agg[ext_name] = {"n_constraints": 0, "n_violated": 0}
                        extractor_agg[ext_name]["n_constraints"] += ext_stats.get("n_constraints", 0)
                        extractor_agg[ext_name]["n_violated"] += ext_stats.get("n_violated", 0)

                metrics_table.append({
                    "model": model_name,
                    "domain": domain,
                    "mode": mode,
                    "n_total": n_total,
                    "n_correct": n_correct,
                    "accuracy": round(accuracy, 4),
                    "hallucination_rate": round(halluc_rate, 4),
                    "repair_success_rate": round(repair_rate, 4),
                    "avg_constraints": round(float(avg_constraints), 2),
                    "total_time_s": round(total_time, 3),
                    "avg_time_s": round(total_time / n_total, 4) if n_total else 0,
                    "repairs_attempted": n_rep_attempted,
                    "repairs_successful": n_rep_ok,
                    "extractor_contribution": {
                        k: {
                            "n_constraints": v["n_constraints"],
                            "n_violated": v["n_violated"],
                            "avg_constraints": round(v["n_constraints"] / n_total, 2) if n_total else 0,
                            "avg_violated": round(v["n_violated"] / n_total, 2) if n_total else 0,
                        }
                        for k, v in extractor_agg.items()
                    },
                })

        print(f"  {'-' * 76}")

    # -----------------------------------------------------------------------
    # Cross-model comparison: Baseline → best mode improvement
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("CROSS-MODEL COMPARISON: Best Mode vs Baseline (per domain)")
    print(sep)

    print(f"\n  {'Domain':<12s} ", end="")
    for mc in MODEL_CONFIGS:
        print(f"{'Δ ' + mc['name']:>20s} ", end="")
    print(f"{'Best helped':>14s}")
    print(f"  {'-' * 68}")

    domain_improvements: dict[str, dict[str, float]] = {}
    for domain in domains:
        improvements: dict[str, float] = {}
        for mc in MODEL_CONFIGS:
            model_name = mc["name"]
            base_rs = all_results[model_name][domain]["baseline"]
            # Use the best non-baseline mode for the headline improvement number.
            best_acc = max(
                sum(1 for r in all_results[model_name][domain][m] if r["correct"]) / 50
                for m in ["verify_only", "verify_repair", "guided_generation"]
            )
            base_acc = sum(1 for r in base_rs if r["correct"]) / len(base_rs)
            improvements[model_name] = best_acc - base_acc
        domain_improvements[domain] = improvements

        best = max(improvements, key=improvements.get)  # type: ignore[arg-type]
        print(f"  {domain:<12s} ", end="")
        for mc in MODEL_CONFIGS:
            delta = improvements[mc["name"]]
            sign = "+" if delta >= 0 else ""
            print(f"{sign}{delta:>18.1%} ", end="")
        print(f"  {best:>12s}")

    # Overall improvement per model.
    print(f"  {'-' * 68}")
    print(f"  {'OVERALL':<12s} ", end="")

    overall_deltas: dict[str, float] = {}
    for mc in MODEL_CONFIGS:
        model_name = mc["name"]
        base_total = 0
        best_total = 0
        n_total = 0
        for domain in domains:
            base_rs = all_results[model_name][domain]["baseline"]
            best_mode_total = max(
                sum(1 for r in all_results[model_name][domain][m] if r["correct"])
                for m in ["verify_only", "verify_repair", "guided_generation"]
            )
            base_total += sum(1 for r in base_rs if r["correct"])
            best_total += best_mode_total
            n_total += len(base_rs)
        base_acc = base_total / n_total if n_total else 0
        best_acc = best_total / n_total if n_total else 0
        delta = best_acc - base_acc
        overall_deltas[model_name] = delta
        sign = "+" if delta >= 0 else ""
        print(f"{sign}{delta:>18.1%} ", end="")

    best_overall = max(overall_deltas, key=overall_deltas.get)  # type: ignore[arg-type]
    print(f"  {best_overall:>12s}")

    # -----------------------------------------------------------------------
    # Mode-by-mode comparison (A vs B vs C vs D)
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("MODE COMPARISON: Accuracy per (model × domain) across all four modes")
    print(sep)

    mode_comparison: dict[str, dict[str, dict[str, float]]] = {}
    for mc in MODEL_CONFIGS:
        model_name = mc["name"]
        mode_comparison[model_name] = {}
        print(f"\n  {model_name}")
        print(f"  {'Domain':<12s} {'Baseline':>9s} {'V-only':>8s} {'V+Repair':>9s} {'Guided':>8s} {'Best':>6s}")
        print(f"  {'-' * 58}")
        for domain in domains:
            accs: dict[str, float] = {}
            for mode in modes:
                rs = all_results[model_name][domain][mode]
                acc = sum(1 for r in rs if r["correct"]) / len(rs)
                accs[mode] = acc
            mode_comparison[model_name][domain] = accs
            best_mode = max(accs, key=accs.get)  # type: ignore[arg-type]
            best_acc = accs[best_mode]
            print(f"  {domain:<12s} {accs['baseline']:>8.1%} "
                  f"{accs['verify_only']:>7.1%} "
                  f"{accs['verify_repair']:>8.1%} "
                  f"{accs['guided_generation']:>7.1%} "
                  f"{best_mode.replace('guided_generation', 'guided').replace('verify_repair', 'v+r'):>8s}")

    # -----------------------------------------------------------------------
    # v10 vs v12 comparison (if Exp 93 results available)
    # -----------------------------------------------------------------------
    version_comparison: list[dict] = []
    if v10_baseline:
        print(f"\n{sep}")
        print("VERSION COMPARISON: v10 (Exp 93) vs v12 (this experiment)")
        print("  v10 = ArithmeticExtractor + CodeExtractor + LogicExtractor + NLExtractor")
        print("  v12 = v10 + FactualKBExtractor (Exp 113)")
        print(sep)

        print(f"\n  Model / Domain            v10 baseline  v12 baseline  v10 repair    v12 repair    v12 guided    Δ v12 guided")
        print(f"  {'-' * 110}")

        for mc in MODEL_CONFIGS:
            model_name = mc["name"]
            # Map Gemma4-E4B-it (v12) to the Exp 93 name "Gemma4-E4B-it".
            v10_model_key = model_name  # same names used in Exp 93

            for domain in domains:
                v10_base = v10_baseline.get(v10_model_key, {}).get(f"{domain}_baseline", None)
                v10_repair = v10_baseline.get(v10_model_key, {}).get(f"{domain}_verify_repair", None)

                v12_base = mode_comparison[model_name][domain]["baseline"]
                v12_repair = mode_comparison[model_name][domain]["verify_repair"]
                v12_guided = mode_comparison[model_name][domain]["guided_generation"]

                delta_guided_vs_v10_base = (
                    v12_guided - v10_base if v10_base is not None else float("nan")
                )

                version_comparison.append({
                    "model": model_name,
                    "domain": domain,
                    "v10_baseline": v10_base,
                    "v12_baseline": v12_base,
                    "v10_verify_repair": v10_repair,
                    "v12_verify_repair": v12_repair,
                    "v12_guided_generation": v12_guided,
                    "delta_guided_vs_v10_baseline": (
                        round(delta_guided_vs_v10_base, 4)
                        if not math.isnan(delta_guided_vs_v10_base) else None
                    ),
                })

                v10_base_str = f"{v10_base:.1%}" if v10_base is not None else "n/a"
                v10_rep_str = f"{v10_repair:.1%}" if v10_repair is not None else "n/a"
                delta_str = (f"{delta_guided_vs_v10_base:+.1%}"
                             if not math.isnan(delta_guided_vs_v10_base) else "n/a")
                print(f"  {model_name[:12]:<12s} / {domain:<10s}  "
                      f"{v10_base_str:>12s}  {v12_base:>12.1%}  "
                      f"{v10_rep_str:>12s}  {v12_repair:>12.1%}  "
                      f"{v12_guided:>12.1%}  {delta_str:>12s}")

    # -----------------------------------------------------------------------
    # False negative reduction analysis
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("FALSE NEGATIVE REDUCTION (verify-only misses = wrong answers NOT flagged)")
    print("  A false negative is: answer is WRONG but verifier says no violations.")
    print(sep)

    fn_analysis: list[dict] = []
    for mc in MODEL_CONFIGS:
        model_name = mc["name"]
        print(f"\n  {model_name}")
        print(f"  {'Domain':<12s} {'Wrong':>8s} {'Flagged':>8s} {'FN Rate':>8s} {'True FN':>9s}")
        print(f"  {'-' * 52}")
        for domain in domains:
            base_rs = all_results[model_name][domain]["baseline"]
            verify_rs = all_results[model_name][domain]["verify_only"]
            # Pair baseline (for ground truth correctness) with verify-only (for flagging).
            n_wrong = sum(1 for r in base_rs if not r["correct"])
            n_flagged_when_wrong = 0
            n_false_negative = 0
            for r_base, r_verify in zip(base_rs, verify_rs):
                if not r_base["correct"]:
                    if r_verify.get("flagged", False):
                        n_flagged_when_wrong += 1
                    else:
                        n_false_negative += 1
            fn_rate = n_false_negative / n_wrong if n_wrong > 0 else 0.0
            print(f"  {domain:<12s} {n_wrong:>8d} {n_flagged_when_wrong:>8d} "
                  f"{fn_rate:>8.1%} {n_false_negative:>9d}")
            fn_analysis.append({
                "model": model_name,
                "domain": domain,
                "n_wrong": n_wrong,
                "n_flagged_when_wrong": n_flagged_when_wrong,
                "n_false_negative": n_false_negative,
                "false_negative_rate": round(fn_rate, 4),
            })

    # -----------------------------------------------------------------------
    # Statistical significance: bootstrap 95% CI on all pairwise mode deltas
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("STATISTICAL SIGNIFICANCE: Bootstrap 95% CI on accuracy differences")
    print(sep)

    stat_analysis: list[dict] = []

    comparisons = [
        ("verify_only", "baseline", "Verify-only vs Baseline"),
        ("verify_repair", "baseline", "Verify+Repair vs Baseline"),
        ("guided_generation", "baseline", "Guided-Gen vs Baseline"),
        ("guided_generation", "verify_repair", "Guided-Gen vs Verify+Repair"),
    ]

    for mc in MODEL_CONFIGS:
        model_name = mc["name"]
        print(f"\n  {model_name}")
        for mode_a, mode_b, label in comparisons:
            scores_a: list[float] = []
            scores_b: list[float] = []
            for domain in domains:
                for r in all_results[model_name][domain][mode_a]:
                    scores_a.append(1.0 if r["correct"] else 0.0)
                for r in all_results[model_name][domain][mode_b]:
                    scores_b.append(1.0 if r["correct"] else 0.0)

            lower, upper, delta = bootstrap_ci(scores_a, scores_b)
            t_stat, p_value = paired_t_test(scores_a, scores_b)
            sig = ("***" if p_value < 0.001 else "**" if p_value < 0.01
                   else "*" if p_value < 0.05 else "ns")
            sign = "+" if delta >= 0 else ""
            print(f"    {label:<40s}: {sign}{delta:.3f} "
                  f"[{lower:+.3f}, {upper:+.3f}] "
                  f"p={p_value:.4f} {sig}")

            stat_analysis.append({
                "model": model_name,
                "comparison": label,
                "mode_a": mode_a,
                "mode_b": mode_b,
                "delta": round(delta, 4),
                "ci_lower": round(lower, 4),
                "ci_upper": round(upper, 4),
                "t_stat": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "significant": sig,
            })

    # -----------------------------------------------------------------------
    # Per-extractor contribution summary
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("PER-EXTRACTOR CONTRIBUTION (avg constraints triggered per question by mode)")
    print(sep)

    extractor_names = ["arithmetic", "code", "logic", "nl", "factual_kb"]
    extractor_summary: list[dict] = []

    for mc in MODEL_CONFIGS:
        model_name = mc["name"]
        print(f"\n  {model_name}")
        for domain in domains:
            print(f"    Domain: {domain}")
            # Build per-extractor averages for verify_only (most informative):
            rs = all_results[model_name][domain]["verify_only"]
            n = len(rs)
            for ext_name in extractor_names:
                total_c = sum(r.get("extractor_breakdown", {}).get(ext_name, {}).get("n_constraints", 0) for r in rs)
                total_v = sum(r.get("extractor_breakdown", {}).get(ext_name, {}).get("n_violated", 0) for r in rs)
                avg_c = total_c / n if n else 0
                avg_v = total_v / n if n else 0
                if avg_c > 0 or True:  # always print for completeness
                    print(f"      {ext_name:<14s}: {avg_c:.2f} constraints/q, {avg_v:.2f} violations/q")
                extractor_summary.append({
                    "model": model_name,
                    "domain": domain,
                    "extractor": ext_name,
                    "avg_constraints_per_q": round(avg_c, 3),
                    "avg_violations_per_q": round(avg_v, 3),
                })

    # -----------------------------------------------------------------------
    # Key findings
    # -----------------------------------------------------------------------
    avg_overall_delta = float(np.mean(list(overall_deltas.values())))
    best_model = max(overall_deltas, key=overall_deltas.get)  # type: ignore[arg-type]

    avg_domain_improvement: dict[str, float] = {}
    for domain in domains:
        avg_imp = float(np.mean([domain_improvements[domain][mc["name"]] for mc in MODEL_CONFIGS]))
        avg_domain_improvement[domain] = avg_imp
    sorted_domains = sorted(avg_domain_improvement.items(), key=lambda x: x[1], reverse=True)

    # Compare guided vs repair across all (model, domain) pairs.
    guided_vs_repair_wins = 0
    guided_vs_repair_total = 0
    for mc in MODEL_CONFIGS:
        for domain in domains:
            rs_guided = all_results[mc["name"]][domain]["guided_generation"]
            rs_repair = all_results[mc["name"]][domain]["verify_repair"]
            acc_guided = sum(1 for r in rs_guided if r["correct"]) / len(rs_guided)
            acc_repair = sum(1 for r in rs_repair if r["correct"]) / len(rs_repair)
            if acc_guided >= acc_repair:
                guided_vs_repair_wins += 1
            guided_vs_repair_total += 1
    guided_win_rate = guided_vs_repair_wins / guided_vs_repair_total if guided_vs_repair_total else 0

    print(f"\n{sep}")
    print("KEY FINDINGS:")
    print(f"  Best improvement over baseline: +{avg_overall_delta:.1%} avg across 2 models × 5 domains")
    print(f"  Best-helped model: {best_model} (+{overall_deltas[best_model]:.1%})")
    print(f"  Best domain: {sorted_domains[0][0]} (+{sorted_domains[0][1]:.1%})")
    print(f"  Worst domain: {sorted_domains[-1][0]} ({sorted_domains[-1][1]:+.1%})")
    print(f"  Guided generation ≥ verify+repair in {guided_vs_repair_wins}/{guided_vs_repair_total} cells "
          f"({guided_win_rate:.0%})")
    print(sep)

    # -----------------------------------------------------------------------
    # Save JSON results
    # -----------------------------------------------------------------------
    results_json: dict[str, Any] = {
        "experiment": 117,
        "description": "Full v12 benchmark: all extractors + guided generation mode",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(elapsed, 2),
        "models": [mc["name"] for mc in MODEL_CONFIGS],
        "domains": domains,
        "modes": modes,
        "questions_per_domain": 50,
        "total_questions": total_questions,
        "total_evaluations": total_questions * len(MODEL_CONFIGS) * len(modes),
        "extractor_versions": {
            "v10": ["arithmetic", "code", "logic", "nl"],
            "v11": ["arithmetic", "code", "logic", "nl_enhanced"],
            "v12": ["arithmetic", "code", "logic", "nl_enhanced", "factual_kb"],
        },
        "guided_generation_config": {
            "alpha": 0.5,
            "check_every_k": 1,
        },
        "metrics_table": metrics_table,
        "mode_comparison": mode_comparison,
        "version_comparison": version_comparison,
        "false_negative_analysis": fn_analysis,
        "statistical_analysis": stat_analysis,
        "extractor_summary": extractor_summary,
        "cross_model_comparison": {
            "domain_improvements": {
                domain: {model: round(delta, 4) for model, delta in imps.items()}
                for domain, imps in domain_improvements.items()
            },
            "overall_deltas": {model: round(delta, 4) for model, delta in overall_deltas.items()},
            "avg_improvement": round(avg_overall_delta, 4),
            "best_helped_model": best_model,
            "most_improved_domain": sorted_domains[0][0],
            "least_improved_domain": sorted_domains[-1][0],
            "guided_win_rate": round(guided_win_rate, 4),
        },
        "key_finding": (
            f"v12 extractor stack (with FactualKBExtractor) improves accuracy by "
            f"+{avg_overall_delta:.1%} on average across 2 models and 5 domains. "
            f"{best_model} benefits most (+{overall_deltas[best_model]:.1%}). "
            f"Best domain: {sorted_domains[0][0]} (+{sorted_domains[0][1]:.1%}). "
            f"Guided generation wins in {guided_vs_repair_wins}/{guided_vs_repair_total} cells "
            f"({guided_win_rate:.0%}) vs verify+repair."
        ),
    }

    results_path = RESULTS_DIR / "experiment_117_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  JSON results saved to {results_path}")

    # -----------------------------------------------------------------------
    # Save Markdown benchmark report
    # -----------------------------------------------------------------------
    md_lines: list[str] = [
        "# Full Benchmark v12: All Extractors + Guided Generation",
        "",
        f"**Experiment 117** | {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} | "
        f"{elapsed:.1f}s runtime",
        "",
        "## Setup",
        "",
        f"- **Models**: {', '.join(mc['name'] for mc in MODEL_CONFIGS)}",
        f"- **Domains**: {', '.join(domains)}",
        f"- **Questions**: {total_questions} ({total_questions // len(domains)} per domain)",
        "- **Modes**: Baseline (A), Verify-only (B), Verify+Repair (C), Guided-Gen (D)",
        f"- **Total evaluations**: {total_questions * len(MODEL_CONFIGS) * len(modes)}",
        "- **Extractor stack (v12)**:",
        "  - `ArithmeticExtractor` — X±Y=Z equation verification",
        "  - `CodeExtractor` — AST-based type/return/bound checks",
        "  - `LogicExtractor` — implication/exclusion/negation patterns",
        "  - `NLExtractor` (v11) — factual IS/relation/quantity patterns",
        "  - `FactualKBExtractor` (v12, Exp 113) — 5000-fact KB entity lookups",
        "- **Guided generation**: `EnergyGuidedSampler(alpha=0.5, check_every_k=1)`",
        "",
        "## Results: Accuracy by Mode",
        "",
    ]

    # Table per model.
    for mc in MODEL_CONFIGS:
        model_name = mc["name"]
        md_lines.append(f"### {model_name}")
        md_lines.append("")
        md_lines.append("| Domain | Baseline | Verify-only | Verify+Repair | Guided-Gen | Δ (best−base) |")
        md_lines.append("|--------|----------|-------------|---------------|------------|---------------|")
        for domain in domains:
            accs = mode_comparison[model_name][domain]
            best = max(accs[m] for m in ["verify_only", "verify_repair", "guided_generation"])
            delta = best - accs["baseline"]
            sign = "+" if delta >= 0 else ""
            md_lines.append(
                f"| {domain} | {accs['baseline']:.1%} | {accs['verify_only']:.1%} | "
                f"{accs['verify_repair']:.1%} | {accs['guided_generation']:.1%} | "
                f"{sign}{delta:.1%} |"
            )
        md_lines.append("")

    # Version comparison table if v10 data available.
    if version_comparison:
        md_lines.extend([
            "## Version Comparison: v10 (Exp 93) vs v12 (Exp 117)",
            "",
            "| Model | Domain | v10 Base | v12 Base | v10 Repair | v12 Repair | v12 Guided | Δ Guided−v10Base |",
            "|-------|--------|----------|----------|------------|------------|------------|-----------------|",
        ])
        for row in version_comparison:
            v10b = f"{row['v10_baseline']:.1%}" if row["v10_baseline"] is not None else "n/a"
            v10r = f"{row['v10_verify_repair']:.1%}" if row["v10_verify_repair"] is not None else "n/a"
            delta = row.get("delta_guided_vs_v10_baseline")
            delta_str = f"{delta:+.1%}" if delta is not None else "n/a"
            md_lines.append(
                f"| {row['model']} | {row['domain']} | {v10b} | "
                f"{row['v12_baseline']:.1%} | {v10r} | "
                f"{row['v12_verify_repair']:.1%} | "
                f"{row['v12_guided_generation']:.1%} | {delta_str} |"
            )
        md_lines.append("")

    # Statistical significance.
    md_lines.extend([
        "## Statistical Significance (Bootstrap 95% CI)",
        "",
        "| Model | Comparison | Delta | CI Lower | CI Upper | p-value | Sig |",
        "|-------|-----------|-------|----------|----------|---------|-----|",
    ])
    for row in stat_analysis:
        sign = "+" if row["delta"] >= 0 else ""
        md_lines.append(
            f"| {row['model']} | {row['comparison']} | "
            f"{sign}{row['delta']:.3f} | {row['ci_lower']:+.3f} | {row['ci_upper']:+.3f} | "
            f"{row['p_value']:.4f} | {row['significant']} |"
        )
    md_lines.append("")

    # False negative analysis.
    md_lines.extend([
        "## False Negative Analysis",
        "",
        "False negatives = wrong answers NOT detected by verify-only.",
        "",
        "| Model | Domain | # Wrong | # Flagged | FN Rate | True FN |",
        "|-------|--------|---------|-----------|---------|---------|",
    ])
    for row in fn_analysis:
        md_lines.append(
            f"| {row['model']} | {row['domain']} | {row['n_wrong']} | "
            f"{row['n_flagged_when_wrong']} | {row['false_negative_rate']:.1%} | "
            f"{row['n_false_negative']} |"
        )
    md_lines.append("")

    # Per-domain improvement summary.
    md_lines.extend([
        "## Domain Impact (avg across models)",
        "",
        "| Domain | Avg Improvement (best mode vs baseline) |",
        "|--------|-----------------------------------------|",
    ])
    for domain, imp in sorted_domains:
        sign = "+" if imp >= 0 else ""
        md_lines.append(f"| {domain} | {sign}{imp:.1%} |")
    md_lines.append("")

    # Key finding.
    md_lines.extend([
        "## Key Finding",
        "",
        f"> {results_json['key_finding']}",
        "",
        "---",
        f"*Generated by Experiment 117 | {time.strftime('%Y-%m-%d', time.gmtime())}*",
        "",
    ])

    md_path = REPO_ROOT / "ops" / "full-benchmark-v12.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Markdown report saved to {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
