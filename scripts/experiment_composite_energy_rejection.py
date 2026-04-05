#!/usr/bin/env python3
"""Experiment: Composite energy = logprob + structural verification.

The married pair: LLM confidence (logprobs) + EBM constraint checking
(execution-based test cases) as a single composite score for rejection
sampling. Structural checks catch confident hallucinations that logprobs
miss; logprobs break ties between structurally-valid candidates.

Scoring: composite_energy = -mean_logprob + penalty * n_failed_tests
  - Lower is better (high logprob + zero test failures)
  - A candidate that fails 1 test gets a large penalty even if confident
  - Among candidates that pass all tests, highest logprob wins

Usage:
    python scripts/experiment_composite_energy_rejection.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

N_CANDIDATES = 5
TEST_FAILURE_PENALTY = 10.0  # Large penalty per failed test

# QA tasks with STRUCTURAL CHECKS — not just substring matching
# Each has: question, func for LLM, test_cases for structural verification
TASKS = [
    {
        "question": "Write a Python function called add(a, b) that returns a + b.",
        "func_name": "add",
        "test_cases": [((1, 2), 3), ((0, 0), 0), ((-1, 1), 0), ((100, 200), 300)],
    },
    {
        "question": "Write a Python function called factorial(n) that returns n! for non-negative integers. factorial(0)=1.",
        "func_name": "factorial",
        "test_cases": [((0,), 1), ((1,), 1), ((5,), 120), ((7,), 5040)],
    },
    {
        "question": "Write a Python function called reverse_string(s) that returns the reversed string.",
        "func_name": "reverse_string",
        "test_cases": [(("hello",), "olleh"), (("",), ""), (("a",), "a"), (("abcd",), "dcba")],
    },
    {
        "question": "Write a Python function called is_prime(n) that returns True if n is prime, False otherwise.",
        "func_name": "is_prime",
        "test_cases": [((2,), True), ((1,), False), ((17,), True), ((15,), False), ((0,), False)],
    },
    {
        "question": "Write a Python function called max_of_three(a, b, c) that returns the largest of three numbers.",
        "func_name": "max_of_three",
        "test_cases": [((1, 2, 3), 3), ((3, 2, 1), 3), ((-1, -2, -3), -1), ((5, 5, 5), 5)],
    },
    {
        "question": "Write a Python function called count_vowels(s) that returns the number of vowels (a,e,i,o,u, case-insensitive) in a string.",
        "func_name": "count_vowels",
        "test_cases": [(("hello",), 2), (("AEIOU",), 5), (("xyz",), 0), (("",), 0)],
    },
    {
        "question": "Write a Python function called fizzbuzz(n) that returns 'Fizz' if n is divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if by both, else the number as a string.",
        "func_name": "fizzbuzz",
        "test_cases": [((3,), "Fizz"), ((5,), "Buzz"), ((15,), "FizzBuzz"), ((7,), "7"), ((1,), "1")],
    },
    {
        "question": "Write a Python function called flatten(lst) that takes a nested list and returns a flat list.",
        "func_name": "flatten",
        "test_cases": [(([1, [2, [3]]],), [1, 2, 3]), (([],), []), (([[1], [2]],), [1, 2])],
    },
    {
        "question": "Write a Python function called gcd(a, b) that returns the greatest common divisor using the Euclidean algorithm.",
        "func_name": "gcd",
        "test_cases": [((12, 8), 4), ((7, 13), 1), ((100, 75), 25), ((0, 5), 5)],
    },
    {
        "question": "Write a Python function called binary_search(arr, target) that returns the index of target in sorted arr, or -1 if not found.",
        "func_name": "binary_search",
        "test_cases": [(([1, 3, 5, 7, 9], 5), 2), (([1, 3, 5], 4), -1), (([], 1), -1), (([1], 1), 0)],
    },
]


def generate_with_logprobs(model, tokenizer, prompt, do_sample=False, temperature=1.0):
    """Generate code and compute mean logprob."""
    import re
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(max_new_tokens=150)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, **gen_kwargs,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0, prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract code from markdown blocks
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
        code = match.group(1).strip() if match else response.strip()

    # Compute mean logprob
    total_logprob = 0.0
    n_tokens = 0
    if hasattr(outputs, "scores") and outputs.scores:
        for step_idx, scores in enumerate(outputs.scores):
            if step_idx >= len(generated_ids):
                break
            token_id = generated_ids[step_idx].item()
            log_probs = torch.log_softmax(scores[0], dim=-1)
            total_logprob += log_probs[token_id].item()
            n_tokens += 1

    mean_logprob = total_logprob / max(n_tokens, 1)
    return code, mean_logprob


def run_tests(code: str, func_name: str, test_cases: list) -> tuple[int, int]:
    """Run test cases. Returns (n_passed, n_total)."""
    from carnot.verify.python_types import safe_exec_function

    n_passed = 0
    for args, expected in test_cases:
        result, error = safe_exec_function(code, func_name, args)
        if error is None and result == expected:
            n_passed += 1
    return n_passed, len(test_cases)


def main() -> int:
    print("=" * 60)
    print("EXPERIMENT: Composite Energy Rejection Sampling")
    print("Score = -logprob + penalty * test_failures")
    print(f"Candidates: {N_CANDIDATES}, Penalty: {TEST_FAILURE_PENALTY}")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    print("Loaded.")

    system = "You are a precise Python programmer. Write ONLY the function definition — no explanation, no imports, no test code."

    # Run each approach: greedy, logprob-only, structural-only, composite
    greedy_passed = []
    logprob_passed = []
    structural_passed = []
    composite_passed = []

    for task_idx, task in enumerate(TASKS):
        q = task["question"]
        fn = task["func_name"]
        tests = task["test_cases"]
        prompt = f"{system}\n\n{q}"

        print(f"\n{'─'*60}")
        print(f"Task {task_idx+1}/{len(TASKS)}: {fn}")

        # Greedy
        code, lp = generate_with_logprobs(model, tokenizer, prompt)
        n_pass, n_total = run_tests(code, fn, tests)
        greedy_ok = n_pass == n_total
        greedy_passed.append(greedy_ok)
        g_icon = "✓" if greedy_ok else "✗"
        print(f"  Greedy: [{g_icon}] {n_pass}/{n_total} tests, lp={lp:.2f}")

        # Generate N candidates
        candidates = []
        for c in range(N_CANDIDATES):
            code_c, lp_c = generate_with_logprobs(
                model, tokenizer, prompt, do_sample=True, temperature=0.8,
            )
            n_pass_c, n_total_c = run_tests(code_c, fn, tests)
            n_fail = n_total_c - n_pass_c
            composite_score = -lp_c + TEST_FAILURE_PENALTY * n_fail
            candidates.append({
                "code": code_c,
                "logprob": lp_c,
                "n_pass": n_pass_c,
                "n_total": n_total_c,
                "n_fail": n_fail,
                "composite": composite_score,
                "all_pass": n_pass_c == n_total_c,
            })

        # Logprob-only: pick highest logprob
        best_lp = max(candidates, key=lambda c: c["logprob"])
        lp_ok = best_lp["all_pass"]
        logprob_passed.append(lp_ok)

        # Structural-only: pick fewest test failures, break ties by logprob
        best_struct = min(candidates, key=lambda c: (c["n_fail"], -c["logprob"]))
        struct_ok = best_struct["all_pass"]
        structural_passed.append(struct_ok)

        # Composite: pick lowest composite score (= most confident + fewest failures)
        best_comp = min(candidates, key=lambda c: c["composite"])
        comp_ok = best_comp["all_pass"]
        composite_passed.append(comp_ok)

        # Report
        lp_icon = "✓" if lp_ok else "✗"
        s_icon = "✓" if struct_ok else "✗"
        c_icon = "✓" if comp_ok else "✗"

        any_pass = sum(1 for c in candidates if c["all_pass"])
        print(f"  Logprob-best:    [{lp_icon}] {best_lp['n_pass']}/{best_lp['n_total']}, lp={best_lp['logprob']:.2f}")
        print(f"  Structural-best: [{s_icon}] {best_struct['n_pass']}/{best_struct['n_total']}, lp={best_struct['logprob']:.2f}")
        print(f"  Composite-best:  [{c_icon}] {best_comp['n_pass']}/{best_comp['n_total']}, lp={best_comp['logprob']:.2f}, score={best_comp['composite']:.2f}")
        print(f"  Oracle: {any_pass}/{N_CANDIDATES} candidates pass all tests")

    # Summary
    n = len(TASKS)
    g_acc = sum(greedy_passed) / n
    l_acc = sum(logprob_passed) / n
    s_acc = sum(structural_passed) / n
    c_acc = sum(composite_passed) / n

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  {'Approach':<20} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'─'*40}")
    print(f"  {'Greedy':<20} {sum(greedy_passed):>5}/{n}   {g_acc:>8.0%}")
    print(f"  {'Logprob-only':<20} {sum(logprob_passed):>5}/{n}   {l_acc:>8.0%}")
    print(f"  {'Structural-only':<20} {sum(structural_passed):>5}/{n}   {s_acc:>8.0%}")
    print(f"  {'Composite (married)':<20} {sum(composite_passed):>5}/{n}   {c_acc:>8.0%}")
    print(f"{'='*60}")

    best_approach = max(
        [("Greedy", g_acc), ("Logprob", l_acc), ("Structural", s_acc), ("Composite", c_acc)],
        key=lambda x: x[1],
    )
    print(f"Best approach: {best_approach[0]} at {best_approach[1]:.0%}")

    if c_acc >= g_acc and c_acc >= l_acc:
        print("SUCCESS: Composite (married LLM+EBM) is the best or tied for best!")
    elif s_acc > l_acc:
        print("Structural verification alone beats logprob confidence.")
    else:
        print("Logprob confidence dominates — structural checks didn't add value here.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
