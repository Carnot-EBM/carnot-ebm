#!/usr/bin/env python3
"""Experiment 45: Logical consistency verification via Ising/thrml.

Encode logical claims as Ising constraints. If an LLM makes a chain of
claims that are mutually inconsistent, the Ising model detects the
contradiction as high energy.

Logical rules encoded:
  - A implies B: if A=1 then B=1 → coupling A→B
  - A and B: both must be 1 → positive biases
  - A or B: at least one must be 1 → penalty when both 0
  - not A: A must be 0 → negative bias
  - A xor B: exactly one → coupling

Test: give the Ising model sets of consistent and inconsistent claims,
check if it correctly identifies contradictions.

Usage:
    .venv/bin/python scripts/experiment_45_logical_consistency.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def encode_claims_as_ising(claims: list[dict], n_props: int) -> tuple:
    """Encode logical claims as Ising biases and couplings.

    Each proposition is a spin. Claims add biases and couplings.
    Returns (biases, edges, weights).
    """
    biases = np.zeros(n_props)
    edges = []
    weights = []
    penalty = 3.0

    for claim in claims:
        ctype = claim["type"]

        if ctype == "true":
            # Proposition is true → strong positive bias
            biases[claim["prop"]] += penalty

        elif ctype == "false":
            # Proposition is false → strong negative bias
            biases[claim["prop"]] -= penalty

        elif ctype == "implies":
            # A implies B: penalize A=1, B=0
            # Encoded as: if A=1, encourage B=1
            a, b = claim["from"], claim["to"]
            biases[b] += penalty * 0.3  # Slight push toward B=1
            edges.append((a, b))
            weights.append(penalty * 0.5)  # A and B tend to agree

        elif ctype == "and":
            # Both must be true
            a, b = claim["props"]
            biases[a] += penalty * 0.5
            biases[b] += penalty * 0.5
            edges.append((a, b))
            weights.append(penalty * 0.3)

        elif ctype == "or":
            # At least one must be true
            a, b = claim["props"]
            biases[a] += penalty * 0.3
            biases[b] += penalty * 0.3
            edges.append((a, b))
            weights.append(-penalty * 0.2)  # Anti-correlate slightly (diversity)

        elif ctype == "xor":
            # Exactly one must be true
            a, b = claim["props"]
            edges.append((a, b))
            weights.append(-penalty)  # Strong anti-correlation

        elif ctype == "mutex":
            # At most one can be true
            a, b = claim["props"]
            edges.append((a, b))
            weights.append(-penalty)

    return biases, edges, np.array(weights) if weights else np.array([])


def check_consistency(claims: list[dict], n_props: int) -> dict:
    """Check logical consistency of claims via Ising sampling."""
    import jax.numpy as jnp
    import jax.random as jrandom
    from thrml import SpinNode, Block, SamplingSchedule, sample_states
    from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init

    biases, edge_pairs, weights = encode_claims_as_ising(claims, n_props)

    if len(edge_pairs) == 0:
        # No couplings — just check biases
        assignment = {i: biases[i] > 0 for i in range(n_props)}
        return {"consistent": True, "assignment": assignment, "energy": 0}

    nodes = [SpinNode() for _ in range(n_props)]
    thrml_edges = [(nodes[e[0]], nodes[e[1]]) for e in edge_pairs]

    model = IsingEBM(
        nodes=nodes,
        edges=thrml_edges,
        biases=jnp.array(biases, dtype=jnp.float32),
        weights=jnp.array(weights, dtype=jnp.float32),
        beta=jnp.array(8.0),
    )

    free_blocks = [Block([nodes[i]]) for i in range(n_props)]
    program = IsingSamplingProgram(model, free_blocks, [])
    init_state = hinton_init(jrandom.PRNGKey(42), model, free_blocks, ())
    schedule = SamplingSchedule(500, 20, 10)

    samples = sample_states(
        jrandom.PRNGKey(123), program, schedule,
        init_state, [], free_blocks,
    )

    # Check if any sample satisfies all claims
    n_samples = samples[0].shape[0]
    best_violations = len(claims)
    best_assignment = None

    for s_idx in range(n_samples):
        assignment = {i: bool(samples[i][s_idx, 0]) for i in range(n_props)}
        violations = count_violations(claims, assignment)
        if violations < best_violations:
            best_violations = violations
            best_assignment = assignment

    return {
        "consistent": best_violations == 0,
        "violations": best_violations,
        "assignment": best_assignment,
    }


def count_violations(claims: list[dict], assignment: dict[int, bool]) -> int:
    """Count how many claims are violated by an assignment."""
    violations = 0
    for claim in claims:
        ctype = claim["type"]
        if ctype == "true" and not assignment.get(claim["prop"], False):
            violations += 1
        elif ctype == "false" and assignment.get(claim["prop"], True):
            violations += 1
        elif ctype == "implies":
            a_val = assignment.get(claim["from"], False)
            b_val = assignment.get(claim["to"], False)
            if a_val and not b_val:
                violations += 1
        elif ctype == "and":
            a, b = claim["props"]
            if not (assignment.get(a, False) and assignment.get(b, False)):
                violations += 1
        elif ctype == "or":
            a, b = claim["props"]
            if not (assignment.get(a, False) or assignment.get(b, False)):
                violations += 1
        elif ctype == "xor":
            a, b = claim["props"]
            if assignment.get(a, False) == assignment.get(b, False):
                violations += 1
        elif ctype == "mutex":
            a, b = claim["props"]
            if assignment.get(a, False) and assignment.get(b, False):
                violations += 1
    return violations


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 45: Logical Consistency Verification")
    print("  Encode logical claims as Ising → detect contradictions")
    print("=" * 70)

    start = time.time()

    # Test cases
    test_cases = [
        {
            "name": "Simple consistent",
            "n_props": 3,
            "claims": [
                {"type": "true", "prop": 0},    # A is true
                {"type": "implies", "from": 0, "to": 1},  # A implies B
                {"type": "true", "prop": 1},    # B is true
            ],
            "expected_consistent": True,
        },
        {
            "name": "Simple contradiction",
            "n_props": 3,
            "claims": [
                {"type": "true", "prop": 0},    # A is true
                {"type": "implies", "from": 0, "to": 1},  # A implies B
                {"type": "false", "prop": 1},   # B is false ← contradiction!
            ],
            "expected_consistent": False,
        },
        {
            "name": "Mutual exclusion + both true",
            "n_props": 2,
            "claims": [
                {"type": "true", "prop": 0},    # A is true
                {"type": "true", "prop": 1},    # B is true
                {"type": "mutex", "props": [0, 1]},  # A and B can't both be true
            ],
            "expected_consistent": False,
        },
        {
            "name": "XOR consistent",
            "n_props": 2,
            "claims": [
                {"type": "true", "prop": 0},
                {"type": "false", "prop": 1},
                {"type": "xor", "props": [0, 1]},
            ],
            "expected_consistent": True,
        },
        {
            "name": "Chain implication (consistent)",
            "n_props": 4,
            "claims": [
                {"type": "true", "prop": 0},           # A
                {"type": "implies", "from": 0, "to": 1},  # A→B
                {"type": "implies", "from": 1, "to": 2},  # B→C
                {"type": "implies", "from": 2, "to": 3},  # C→D
                {"type": "true", "prop": 3},            # D
            ],
            "expected_consistent": True,
        },
        {
            "name": "Chain implication (contradiction)",
            "n_props": 4,
            "claims": [
                {"type": "true", "prop": 0},           # A
                {"type": "implies", "from": 0, "to": 1},  # A→B
                {"type": "implies", "from": 1, "to": 2},  # B→C
                {"type": "implies", "from": 2, "to": 3},  # C→D
                {"type": "false", "prop": 3},           # NOT D ← contradiction with A→B→C→D
            ],
            "expected_consistent": False,
        },
        {
            "name": "LLM-style: 'Paris is capital AND London is capital' (mutex)",
            "n_props": 2,
            "claims": [
                {"type": "true", "prop": 0},    # Paris is capital
                {"type": "true", "prop": 1},    # London is capital
                {"type": "mutex", "props": [0, 1]},  # Only one can be capital of France
            ],
            "expected_consistent": False,
        },
        {
            "name": "Complex consistent",
            "n_props": 5,
            "claims": [
                {"type": "true", "prop": 0},
                {"type": "implies", "from": 0, "to": 1},
                {"type": "or", "props": [2, 3]},
                {"type": "mutex", "props": [3, 4]},
                {"type": "true", "prop": 2},
            ],
            "expected_consistent": True,
        },
    ]

    results = []
    for tc in test_cases:
        result = check_consistency(tc["claims"], tc["n_props"])
        expected = tc["expected_consistent"]
        got = result["consistent"]
        correct = got == expected

        icon = "✓" if correct else "✗"
        status = "consistent" if got else f"CONTRADICTION ({result['violations']} violations)"
        print(f"  [{icon}] {tc['name']:45s} → {status}")

        results.append({"name": tc["name"], "correct": correct, "expected": expected, "got": got})

    # Summary
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 45 RESULTS ({elapsed:.0f}s)")
    print(sep)

    n_correct = sum(1 for r in results if r["correct"])
    n_consistent_correct = sum(1 for r in results if r["expected"] and r["correct"])
    n_contradictions_detected = sum(1 for r in results if not r["expected"] and not r["got"])
    n_contradictions_total = sum(1 for r in results if not r["expected"])

    print(f"  Overall accuracy: {n_correct}/{len(results)} ({n_correct/len(results):.0%})")
    print(f"  Contradictions detected: {n_contradictions_detected}/{n_contradictions_total}")
    print(f"  Consistent correctly identified: {n_consistent_correct}/{len(results) - n_contradictions_total}")

    if n_correct == len(results):
        print(f"\n  VERDICT: ✅ Perfect logical consistency detection!")
    elif n_correct > len(results) * 0.75:
        print(f"\n  VERDICT: ✅ Good consistency detection ({n_correct}/{len(results)})")
    else:
        print(f"\n  VERDICT: ⚠️ Partial success ({n_correct}/{len(results)})")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
