#!/usr/bin/env python3
"""Run self-improving code verification autoresearch loop.

This script runs the autoresearch pipeline where the optimization target
is CODE VERIFICATION ACCURACY (not sampler performance). The loop proposes
better verifier architectures/training strategies, evaluates them on a
held-out test set, and accumulates lessons about what works.

This is the first demonstration of directed self-learning applied to
verification quality.

Usage:
    python scripts/run_code_verification_autoresearch.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from carnot.autoresearch.code_improvement import (
    build_code_verification_baselines,
    code_verification_hypothesis_template,
    run_code_verification_autoresearch,
)
from carnot.autoresearch.orchestrator import AutoresearchConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    print("=" * 60)
    print("CARNOT CODE VERIFICATION AUTORESEARCH")
    print("Self-improving loop: EBM learns to verify code better")
    print("=" * 60)

    # Phase 1: Establish baselines
    print("\n--- Phase 1: Establishing baseline code verifier ---")
    start = time.time()

    try:
        baselines, benchmark_data = build_code_verification_baselines()
    except Exception as e:
        print(f"Baseline creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"Baseline established in {time.time() - start:.1f}s")
    for name, metrics in sorted(baselines.benchmarks.items()):
        print(f"  {name}: energy={metrics.final_energy:.4f}")

    # Phase 2: Generate hypotheses and evaluate
    print("\n--- Phase 2: Running autoresearch loop ---")
    strategies = ["wider_model", "deeper_model", "more_epochs", "more_data"]

    hypotheses = []
    for strategy in strategies:
        code = code_verification_hypothesis_template(strategy)
        hypotheses.append((f"Strategy: {strategy}", code))
        print(f"  Hypothesis: {strategy}")

    # Run the loop with pre-built hypotheses
    from carnot.autoresearch.orchestrator import run_loop

    config = AutoresearchConfig(
        max_iterations=len(hypotheses),
        max_consecutive_failures=4,
    )

    print(f"\nEvaluating {len(hypotheses)} hypotheses...")
    result = run_loop(
        hypotheses=hypotheses,
        baselines=baselines,
        benchmark_data=benchmark_data,
        config=config,
    )

    # Print results
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"CODE VERIFICATION AUTORESEARCH RESULTS ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Iterations:       {result.iterations}")
    print(f"  Accepted:         {result.accepted}")
    print(f"  Rejected:         {result.rejected}")
    print(f"  Pending review:   {result.pending_review}")
    print(f"  Circuit breaker:  {'TRIPPED' if result.circuit_breaker_tripped else 'OK'}")

    print("\nEXPERIMENT LOG:")
    print("-" * 60)
    for entry in result.experiment_log.entries:
        icon = {"accepted": "+", "rejected": "x", "pending_review": "?"}.get(entry.outcome, " ")
        print(f"  [{icon}] {entry.hypothesis_description}")
        print(f"      Verdict: {entry.eval_verdict} — {entry.eval_reason}")
        if entry.sandbox_error:
            print(f"      Error: {entry.sandbox_error[:200]}")
        print()

    if result.final_baselines:
        print("FINAL BASELINES:")
        print("-" * 60)
        for name, metrics in sorted(result.final_baselines.benchmarks.items()):
            print(f"  {name}: energy={metrics.final_energy:.4f}")

    print(f"\n{'='*60}")
    print(f"  {result.accepted} hypothesis(es) accepted!")
    print("  The energy function served as the objective judge.")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
