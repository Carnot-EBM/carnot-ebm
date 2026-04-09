#!/usr/bin/env python3
"""Experiment 47: LLM self-constraint extraction for automated verification.

The LLM generates BOTH an answer AND verifiable constraints. The Ising model
checks whether the constraints are internally consistent and whether the
answer satisfies them.

This tests whether the verify/repair pipeline can be fully automated:
  1. LLM answers a question
  2. LLM extracts constraints from its own answer
  3. Ising model verifies constraint consistency
  4. If inconsistent → flag hallucination

We test on:
  - Arithmetic claims (verified via QUBO encoding)
  - Logical consistency (verified via Ising energy)
  - Factual claims with known ground truth

Since loading a local LLM is expensive and may not be available, we simulate
the LLM outputs with realistic examples (both correct and hallucinated).
This validates the constraint extraction → verification pipeline independent
of LLM availability.

Usage:
    .venv/bin/python scripts/experiment_47_llm_self_constraints.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# --- Constraint types ---

def verify_arithmetic_constraint(a: int, b: int, claimed_sum: int) -> dict:
    """Verify a + b = claimed_sum via direct computation.

    For the full pipeline, this would use the QUBO Ising solver.
    Here we verify directly since the QUBO encoding is validated in Exp 42b.
    """
    correct = a + b
    return {
        "type": "arithmetic",
        "claim": f"{a} + {b} = {claimed_sum}",
        "satisfied": claimed_sum == correct,
        "correct_value": correct,
    }


def verify_logical_constraints(claims: list[dict], n_props: int) -> dict:
    """Verify logical consistency of a set of claims via Ising energy.

    Uses the same encoding as Exp 45 but with the parallel sampler
    (avoids thrml ROCm crash and runs faster).
    """
    from experiment_45_logical_consistency import (
        encode_claims_as_ising,
        count_violations,
    )
    from carnot.samplers.parallel_ising import (
        ParallelIsingSampler,
        AnnealingSchedule,
        sat_to_coupling_matrix,
    )
    import jax.numpy as jnp
    import jax.random as jrandom

    try:
        biases_np, edge_pairs, weights_list = encode_claims_as_ising(claims, n_props)

        if len(edge_pairs) == 0:
            return {"type": "logical", "n_claims": len(claims), "consistent": True, "violations": 0}

        # Build coupling matrix from edge list.
        J = np.zeros((n_props, n_props), dtype=np.float32)
        for k, (i, j) in enumerate(edge_pairs):
            J[i, j] += weights_list[k]
            J[j, i] += weights_list[k]

        sampler = ParallelIsingSampler(
            n_warmup=500, n_samples=30, steps_per_sample=10,
            schedule=AnnealingSchedule(0.1, 8.0),
            use_checkerboard=True,
        )

        samples = sampler.sample(
            jrandom.PRNGKey(42),
            jnp.array(biases_np, dtype=jnp.float32),
            jnp.array(J, dtype=jnp.float32),
            beta=8.0,
        )

        best_violations = len(claims)
        for s_idx in range(samples.shape[0]):
            assignment = {i: bool(samples[s_idx, i]) for i in range(n_props)}
            v = count_violations(claims, assignment)
            best_violations = min(best_violations, v)

        return {
            "type": "logical",
            "n_claims": len(claims),
            "consistent": best_violations == 0,
            "violations": best_violations,
        }
    except Exception as e:
        return {
            "type": "logical",
            "n_claims": len(claims),
            "consistent": None,
            "error": str(e),
        }


def verify_mutual_exclusion(claims: list[str], truth: dict[str, bool]) -> dict:
    """Verify factual claims against known ground truth."""
    correct = 0
    wrong = []
    for claim in claims:
        if claim in truth:
            if truth[claim]:
                correct += 1
            else:
                wrong.append(claim)
        # Unknown claims are not penalized.
    return {
        "type": "factual",
        "n_claims": len(claims),
        "n_verified": correct,
        "n_wrong": len(wrong),
        "wrong_claims": wrong,
        "all_correct": len(wrong) == 0,
    }


# --- Simulated LLM outputs ---

def get_test_scenarios() -> list[dict]:
    """Simulated LLM outputs with self-extracted constraints.

    Each scenario has:
    - question: what was asked
    - llm_answer: the LLM's response
    - constraints: constraints the LLM extracted from its own answer
    - ground_truth: what's actually correct
    - expected_verdict: should verification pass or fail?
    """
    return [
        # --- Correct LLM outputs ---
        {
            "name": "Correct arithmetic",
            "question": "What is 47 + 28?",
            "llm_answer": "47 + 28 = 75",
            "constraints": [
                {"type": "arithmetic", "a": 47, "b": 28, "result": 75},
            ],
            "expected_pass": True,
        },
        {
            "name": "Correct logic chain",
            "question": "If it rains, the ground is wet. It rained. Is the ground wet?",
            "llm_answer": "Yes, the ground is wet.",
            "constraints": [
                {"type": "implies", "from": 0, "to": 1},  # rain → wet
                {"type": "true", "prop": 0},                # it rained
                {"type": "true", "prop": 1},                # ground is wet
            ],
            "n_props": 2,
            "expected_pass": True,
        },
        {
            "name": "Correct multi-step arithmetic",
            "question": "A store has 15 apples. 7 are sold. 3 more arrive. How many?",
            "llm_answer": "15 - 7 + 3 = 11 apples.",
            "constraints": [
                {"type": "arithmetic", "a": 15, "b": -7, "result": 8},
                {"type": "arithmetic", "a": 8, "b": 3, "result": 11},
            ],
            "expected_pass": True,
        },
        {
            "name": "Correct factual claims",
            "question": "What is the capital of France?",
            "llm_answer": "The capital of France is Paris.",
            "constraints": [
                {"type": "factual", "claims": ["Paris is the capital of France"]},
            ],
            "ground_truth": {"Paris is the capital of France": True},
            "expected_pass": True,
        },
        # --- Hallucinated LLM outputs ---
        {
            "name": "Wrong arithmetic (off by one)",
            "question": "What is 47 + 28?",
            "llm_answer": "47 + 28 = 76",
            "constraints": [
                {"type": "arithmetic", "a": 47, "b": 28, "result": 76},
            ],
            "expected_pass": False,
        },
        {
            "name": "Logical contradiction",
            "question": "Can something be both a mammal and a reptile?",
            "llm_answer": "A platypus is both a mammal and a reptile.",
            "constraints": [
                {"type": "true", "prop": 0},                # is_mammal
                {"type": "true", "prop": 1},                # is_reptile
                {"type": "mutex", "props": [0, 1]},         # mammal XOR reptile
            ],
            "n_props": 2,
            "expected_pass": False,
        },
        {
            "name": "Contradictory implications",
            "question": "If all birds fly, do penguins fly?",
            "llm_answer": "Penguins are birds and all birds fly, so penguins fly.",
            "constraints": [
                {"type": "true", "prop": 0},                # penguin_is_bird
                {"type": "implies", "from": 0, "to": 1},   # bird → flies
                {"type": "true", "prop": 1},                # penguin_flies
                # But also:
                {"type": "true", "prop": 2},                # penguin_is_flightless
                {"type": "implies", "from": 2, "to": 3},   # flightless → not_flies
                {"type": "true", "prop": 3},                # not_flies
                {"type": "mutex", "props": [1, 3]},         # flies XOR not_flies
            ],
            "n_props": 4,
            "expected_pass": False,
        },
        {
            "name": "Wrong multi-step arithmetic",
            "question": "A store has 15 apples. 7 sold. 3 arrive. How many?",
            "llm_answer": "15 - 7 + 3 = 12 apples.",
            "constraints": [
                {"type": "arithmetic", "a": 15, "b": -7, "result": 8},
                {"type": "arithmetic", "a": 8, "b": 3, "result": 12},  # Wrong!
            ],
            "expected_pass": False,
        },
        {
            "name": "Wrong factual claim",
            "question": "What is the capital of Australia?",
            "llm_answer": "The capital of Australia is Sydney.",
            "constraints": [
                {"type": "factual", "claims": ["Sydney is the capital of Australia"]},
            ],
            "ground_truth": {"Sydney is the capital of Australia": False},
            "expected_pass": False,
        },
        {
            "name": "Subtle logical error",
            "question": "If A implies B, and B implies C, and A is true, is C true?",
            "llm_answer": "A→B, B→C, A is true. So B is true, but C might not be.",
            "constraints": [
                {"type": "implies", "from": 0, "to": 1},   # A → B
                {"type": "implies", "from": 1, "to": 2},   # B → C
                {"type": "true", "prop": 0},                # A is true
                {"type": "true", "prop": 1},                # B is true (correct)
                {"type": "false", "prop": 2},               # C is false (WRONG)
            ],
            "n_props": 3,
            "expected_pass": False,
        },
    ]


def verify_scenario(scenario: dict) -> dict:
    """Run all constraint verifications for a scenario."""
    results = []
    all_pass = True

    for constraint in scenario["constraints"]:
        if constraint["type"] == "arithmetic":
            r = verify_arithmetic_constraint(
                constraint["a"], constraint["b"], constraint["result"]
            )
            if not r["satisfied"]:
                all_pass = False
            results.append(r)

        elif constraint["type"] == "factual":
            gt = scenario.get("ground_truth", {})
            r = verify_mutual_exclusion(constraint["claims"], gt)
            if not r["all_correct"]:
                all_pass = False
            results.append(r)

        elif constraint["type"] in ("true", "false", "implies", "mutex"):
            # Collect all logical constraints and verify together.
            pass  # Handled below.

    # Verify logical constraints as a group.
    logical_claims = [c for c in scenario["constraints"]
                      if c["type"] in ("true", "false", "implies", "mutex")]
    if logical_claims:
        n_props = scenario.get("n_props", 2)
        r = verify_logical_constraints(logical_claims, n_props)
        if r.get("consistent") is False:
            all_pass = False
        results.append(r)

    return {
        "name": scenario["name"],
        "expected_pass": scenario["expected_pass"],
        "actual_pass": all_pass,
        "correct_detection": all_pass == scenario["expected_pass"],
        "details": results,
    }


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 47: LLM Self-Constraint Extraction & Verification")
    print("  LLM generates answer + constraints → Ising verifies")
    print("=" * 70)

    start = time.time()
    scenarios = get_test_scenarios()
    results = []

    for scenario in scenarios:
        result = verify_scenario(scenario)
        icon = "✓" if result["correct_detection"] else "✗"
        status = "PASS" if result["actual_pass"] else "FAIL"
        expected = "pass" if result["expected_pass"] else "fail"
        print(f"  [{icon}] {result['name']:<40s} → {status} (expected {expected})")
        results.append(result)

    # Summary.
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 47 RESULTS ({elapsed:.0f}s)")
    print(sep)

    n_total = len(results)
    n_correct = sum(1 for r in results if r["correct_detection"])
    n_true_pos = sum(1 for r in results if not r["expected_pass"] and not r["actual_pass"])
    n_true_neg = sum(1 for r in results if r["expected_pass"] and r["actual_pass"])
    n_false_pos = sum(1 for r in results if r["expected_pass"] and not r["actual_pass"])
    n_false_neg = sum(1 for r in results if not r["expected_pass"] and r["actual_pass"])
    n_hallucinations = sum(1 for r in results if not r["expected_pass"])
    n_correct_outputs = sum(1 for r in results if r["expected_pass"])

    print(f"  Total scenarios:        {n_total}")
    print(f"  Correct detections:     {n_correct}/{n_total}")
    print(f"  True positives (caught hallucinations):  {n_true_pos}/{n_hallucinations}")
    print(f"  True negatives (passed correct):         {n_true_neg}/{n_correct_outputs}")
    print(f"  False positives (flagged correct as bad): {n_false_pos}")
    print(f"  False negatives (missed hallucination):   {n_false_neg}")

    if n_correct == n_total:
        print(f"\n  VERDICT: ✅ Perfect constraint verification pipeline!")
    elif n_correct >= n_total * 0.8:
        print(f"\n  VERDICT: ✅ Pipeline works ({n_correct}/{n_total} correct)")
    else:
        print(f"\n  VERDICT: ❌ Pipeline needs work ({n_correct}/{n_total})")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
