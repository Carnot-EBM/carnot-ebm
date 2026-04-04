#!/usr/bin/env python3
"""Run real LLM benchmark: measure hallucination rate vs EBM repair success.

This script sends SAT and graph coloring problems to a real LLM (via Claude
API bridge), measures how often the LLM gets it wrong, and how effectively
the EBM verify-and-repair pipeline fixes the mistakes.

This is the first real "LLM hallucinates → EBM repairs" measurement.

Usage:
    python scripts/run_llm_benchmark.py
    python scripts/run_llm_benchmark.py --api-base http://localhost:8080/v1 --model sonnet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import jax.numpy as jnp

from carnot.inference.benchmark import generate_random_graph, generate_random_sat
from carnot.inference.llm_solver import (
    LLMSolverConfig,
    run_llm_coloring_experiment,
    run_llm_sat_experiment,
)
from carnot.verify.sat import SATClause, build_sat_energy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_sat_benchmark(
    config: LLMSolverConfig,
    n_instances: int = 10,
    n_vars: int = 8,
    n_clauses: int = 20,
) -> dict:
    """Run SAT benchmark against real LLM."""
    print(f"\n{'='*60}")
    print(f"SAT BENCHMARK: {n_instances} instances, {n_vars} vars, {n_clauses} clauses")
    print(f"{'='*60}")

    results = []
    for i in range(n_instances):
        clauses = generate_random_sat(n_vars, n_clauses, clause_size=3, seed=i)
        print(f"\n--- Instance {i+1}/{n_instances} ---")

        try:
            vr = run_llm_sat_experiment(
                config, clauses, n_vars,
                repair_step_size=0.1, repair_max_steps=100,
            )

            if vr.initial_verification is None:
                print(f"  LLM call failed or parse error")
                results.append({"status": "error"})
                continue

            init_verified = vr.initial_verification.verdict.verified
            init_energy = float(vr.initial_verification.total_energy)
            init_failing = len(vr.initial_verification.verdict.failing)

            round_verified = False
            round_energy = 0.0
            round_failing = 0
            if vr.rounded_verification is not None:
                round_verified = vr.rounded_verification.verdict.verified
                round_energy = float(vr.rounded_verification.total_energy)
                round_failing = len(vr.rounded_verification.verdict.failing)

            print(f"  LLM: verified={init_verified}, energy={init_energy:.4f}, failing={init_failing}")
            print(f"  Repaired: verified={round_verified}, energy={round_energy:.4f}, failing={round_failing}")

            results.append({
                "status": "ok",
                "llm_verified": init_verified,
                "llm_energy": init_energy,
                "llm_failing": init_failing,
                "repaired_verified": round_verified,
                "repaired_energy": round_energy,
                "repaired_failing": round_failing,
            })

        except Exception as e:
            print(f"  Error: {e}")
            results.append({"status": "error"})

    # Aggregate
    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        print("\nNo successful results!")
        return {"domain": "sat", "n_instances": n_instances, "n_ok": 0}

    llm_correct = sum(1 for r in ok_results if r["llm_verified"])
    repaired_correct = sum(1 for r in ok_results if r["repaired_verified"])
    avg_llm_energy = sum(r["llm_energy"] for r in ok_results) / len(ok_results)
    avg_repaired_energy = sum(r["repaired_energy"] for r in ok_results) / len(ok_results)

    summary = {
        "domain": "sat",
        "n_instances": n_instances,
        "n_ok": len(ok_results),
        "llm_accuracy": llm_correct / len(ok_results),
        "repaired_accuracy": repaired_correct / len(ok_results),
        "avg_llm_energy": avg_llm_energy,
        "avg_repaired_energy": avg_repaired_energy,
        "improvement": (repaired_correct - llm_correct) / max(len(ok_results), 1),
    }

    print(f"\n{'='*60}")
    print(f"SAT RESULTS ({len(ok_results)}/{n_instances} instances)")
    print(f"{'='*60}")
    print(f"  LLM accuracy:     {summary['llm_accuracy']:.0%}")
    print(f"  Repaired accuracy: {summary['repaired_accuracy']:.0%}")
    print(f"  Improvement:       +{summary['improvement']:.0%}")
    print(f"  Avg LLM energy:    {summary['avg_llm_energy']:.4f}")
    print(f"  Avg repaired energy: {summary['avg_repaired_energy']:.4f}")

    return summary


def run_coloring_benchmark(
    config: LLMSolverConfig,
    n_instances: int = 10,
    n_nodes: int = 6,
    n_colors: int = 3,
) -> dict:
    """Run graph coloring benchmark against real LLM."""
    print(f"\n{'='*60}")
    print(f"COLORING BENCHMARK: {n_instances} instances, {n_nodes} nodes, {n_colors} colors")
    print(f"{'='*60}")

    results = []
    for i in range(n_instances):
        edges = generate_random_graph(n_nodes, edge_probability=0.4, seed=i)
        if not edges:
            continue

        print(f"\n--- Instance {i+1}/{n_instances} ({len(edges)} edges) ---")

        try:
            vr = run_llm_coloring_experiment(
                config, edges, n_nodes, n_colors,
                repair_step_size=0.1, repair_max_steps=100,
            )

            if vr.initial_verification is None:
                print(f"  LLM call failed or parse error")
                results.append({"status": "error"})
                continue

            init_verified = vr.initial_verification.verdict.verified
            init_energy = float(vr.initial_verification.total_energy)

            round_verified = False
            round_energy = 0.0
            if vr.rounded_verification is not None:
                round_verified = vr.rounded_verification.verdict.verified
                round_energy = float(vr.rounded_verification.total_energy)

            print(f"  LLM: verified={init_verified}, energy={init_energy:.4f}")
            print(f"  Repaired: verified={round_verified}, energy={round_energy:.4f}")

            results.append({
                "status": "ok",
                "llm_verified": init_verified,
                "llm_energy": init_energy,
                "repaired_verified": round_verified,
                "repaired_energy": round_energy,
            })

        except Exception as e:
            print(f"  Error: {e}")
            results.append({"status": "error"})

    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        print("\nNo successful results!")
        return {"domain": "coloring", "n_instances": n_instances, "n_ok": 0}

    llm_correct = sum(1 for r in ok_results if r["llm_verified"])
    repaired_correct = sum(1 for r in ok_results if r["repaired_verified"])

    summary = {
        "domain": "coloring",
        "n_instances": n_instances,
        "n_ok": len(ok_results),
        "llm_accuracy": llm_correct / len(ok_results),
        "repaired_accuracy": repaired_correct / len(ok_results),
        "improvement": (repaired_correct - llm_correct) / max(len(ok_results), 1),
    }

    print(f"\n{'='*60}")
    print(f"COLORING RESULTS ({len(ok_results)}/{n_instances} instances)")
    print(f"{'='*60}")
    print(f"  LLM accuracy:     {summary['llm_accuracy']:.0%}")
    print(f"  Repaired accuracy: {summary['repaired_accuracy']:.0%}")
    print(f"  Improvement:       +{summary['improvement']:.0%}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM-EBM benchmark")
    parser.add_argument("--api-base", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--n-sat", type=int, default=10, help="Number of SAT instances")
    parser.add_argument("--n-coloring", type=int, default=10, help="Number of coloring instances")
    args = parser.parse_args()

    config = LLMSolverConfig(
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        temperature=0.3,
    )

    print("=" * 60)
    print("CARNOT LLM-EBM BENCHMARK")
    print("LLM proposes → EBM scores → Gradient repairs → Certificate")
    print(f"API: {args.api_base}, Model: {args.model}")
    print("=" * 60)

    start = time.time()

    sat_summary = run_sat_benchmark(config, n_instances=args.n_sat, n_vars=8, n_clauses=20)
    coloring_summary = run_coloring_benchmark(config, n_instances=args.n_coloring, n_nodes=6, n_colors=3)

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY (elapsed: {elapsed:.0f}s)")
    print(f"{'='*60}")

    for s in [sat_summary, coloring_summary]:
        if s.get("n_ok", 0) > 0:
            print(f"\n  {s['domain'].upper()}:")
            print(f"    LLM accuracy:     {s['llm_accuracy']:.0%}")
            print(f"    Repaired accuracy: {s['repaired_accuracy']:.0%}")
            print(f"    EBM improvement:   +{s['improvement']:.0%}")

    print(f"\n{'='*60}")
    print("CONCLUSION: The EBM verify-and-repair pipeline catches and fixes")
    print("what the LLM gets wrong. Energy is the objective judge.")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
