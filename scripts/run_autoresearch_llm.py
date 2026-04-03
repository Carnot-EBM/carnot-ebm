#!/usr/bin/env python3
"""Run the autoresearch loop with an LLM hypothesis generator.

**What this does:**
    Connects the Carnot autoresearch pipeline to an LLM via an
    OpenAI-compatible API (e.g., the Claude API bridge). The LLM proposes
    EBM improvement hypotheses, which are evaluated in the sandbox against
    real benchmarks.

**How to run:**

    # 1. Start the Claude API bridge (if using Claude Code):
    cd tools/claude-api-bridge
    docker compose up -d

    # 2. Run the autoresearch loop:
    python scripts/run_autoresearch_llm.py

    # Or with custom settings:
    python scripts/run_autoresearch_llm.py \\
        --api-base http://localhost:8080/v1 \\
        --model sonnet \\
        --max-iterations 10

Spec: FR-11, REQ-AUTO-003, SCENARIO-AUTO-001
"""

import argparse
import logging
import sys
import os

# Add the Python package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from carnot.autoresearch import (
    AutoresearchConfig,
    BaselineRecord,
    BenchmarkMetrics,
    run_loop_with_generator,
)
from carnot.autoresearch.hypothesis_generator import (
    GeneratorConfig,
    generate_hypotheses,
)
from carnot.benchmarks import DoubleWell, Rosenbrock, Ackley, Rastrigin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_initial_baselines() -> tuple[BaselineRecord, dict]:
    """Create initial baselines by running Langevin sampling on real benchmarks.

    Uses the actual benchmark energy functions (DoubleWell, Rosenbrock, etc.)
    with a default Langevin sampler to establish baseline performance.

    Returns:
        Tuple of (baselines, benchmark_data) where benchmark_data maps
        benchmark names to their configurations for hypothesis code.
    """
    import time
    import jax.numpy as jnp
    import jax.random as jrandom
    from carnot.samplers.langevin import LangevinSampler

    record = BaselineRecord(version="0.1.0")

    # Set up the benchmarks we'll evaluate against
    benchmarks = {
        "double_well": DoubleWell(dim=2),
        "rosenbrock": Rosenbrock(dim=2),
    }

    # Default sampler: Langevin with step_size=0.001 (small enough for Rosenbrock)
    sampler = LangevinSampler(step_size=0.001)
    key = jrandom.PRNGKey(0)
    n_steps = 5000

    for name, energy_fn in benchmarks.items():
        k1, key = jrandom.split(key)
        x0 = jrandom.normal(k1, (energy_fn.input_dim,)) * 0.5  # small init

        # Warm up JIT on first call so baseline timing is fair
        _ = sampler.sample(energy_fn, x0, 10, key=key)

        start = time.time()
        x_final = sampler.sample(energy_fn, x0, n_steps, key=key)
        wall_clock = time.time() - start

        final_energy = float(energy_fn.energy(x_final))

        record.benchmarks[name] = BenchmarkMetrics(
            benchmark_name=name,
            final_energy=final_energy,
            convergence_steps=n_steps,
            wall_clock_seconds=wall_clock,
            peak_memory_mb=50.0,
        )

    # benchmark_data passed to hypotheses — tells them which benchmarks exist
    # and their dimensionality so they can construct the right energy functions
    benchmark_data = {
        "benchmarks": {
            name: {
                "dim": fn.input_dim,
                "global_min_energy": fn.info().global_min_energy,
            }
            for name, fn in benchmarks.items()
        }
    }

    return record, benchmark_data


def make_generator(gen_config: GeneratorConfig):
    """Create a generator callable for run_loop_with_generator.

    Returns a function with signature:
        (baselines, recent_failures, iteration) -> list[tuple[str, str]]
    """

    def generator(baselines, recent_failures, iteration):
        result = generate_hypotheses(
            gen_config,
            baselines,
            recent_failures=recent_failures,
            iteration=iteration,
        )
        if result.error:
            logger.error("Hypothesis generation failed: %s", result.error)
            return []

        logger.info(
            "Generated %d hypothesis(es) from LLM", len(result.hypotheses)
        )
        for desc, _ in result.hypotheses:
            logger.info("  - %s", desc)

        return result.hypotheses

    return generator


def print_results(result) -> None:
    """Pretty-print the loop results."""
    print()
    print("=" * 70)
    print("AUTORESEARCH RESULTS (LLM-powered)")
    print("=" * 70)
    print(f"  Iterations:       {result.iterations}")
    print(f"  Accepted:         {result.accepted}")
    print(f"  Rejected:         {result.rejected}")
    print(f"  Pending review:   {result.pending_review}")
    print(
        f"  Circuit breaker:  "
        f"{'TRIPPED' if result.circuit_breaker_tripped else 'OK'}"
    )
    print()

    print("EXPERIMENT LOG:")
    print("-" * 70)
    for entry in result.experiment_log.entries:
        status_icon = {
            "accepted": "+",
            "rejected": "x",
            "pending_review": "?",
        }.get(entry.outcome, " ")
        print(f"  [{status_icon}] {entry.hypothesis_description}")
        print(f"      Verdict: {entry.eval_verdict} -- {entry.eval_reason}")
        if entry.sandbox_error:
            print(f"      Error: {entry.sandbox_error}")
        print()

    print("FINAL BASELINES:")
    print("-" * 70)
    if result.final_baselines:
        for name, metrics in sorted(result.final_baselines.benchmarks.items()):
            print(
                f"  {name}: energy={metrics.final_energy:.4f}, "
                f"steps={metrics.convergence_steps}, "
                f"time={metrics.wall_clock_seconds:.1f}s"
            )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Carnot autoresearch with LLM hypothesis generator"
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8080/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--model",
        default="sonnet",
        help="Model name to use (default: sonnet)",
    )
    parser.add_argument(
        "--api-key",
        default="not-needed",
        help="API key (default: not-needed for Claude bridge)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of hypotheses to evaluate (default: 5)",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="Circuit breaker threshold (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM sampling temperature (default: 0.7)",
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  CARNOT AUTORESEARCH -- LLM-Powered Self-Improvement")
    print("=" * 70)
    print(f"  API: {args.api_base}")
    print(f"  Model: {args.model}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Circuit breaker: {args.max_failures} consecutive failures")
    print("=" * 70)
    print()

    # Set up baselines using REAL benchmark energy functions
    print("Computing initial baselines with Langevin sampling...")
    baselines, benchmark_data = create_initial_baselines()
    print("Initial baselines (from real benchmarks):")
    for name, metrics in sorted(baselines.benchmarks.items()):
        binfo = benchmark_data["benchmarks"][name]
        print(
            f"  {name}: energy={metrics.final_energy:.4f} "
            f"(optimal={binfo['global_min_energy']:.4f}, "
            f"gap={metrics.final_energy - binfo['global_min_energy']:.4f})"
        )
    print()

    # Configure the generator
    gen_config = GeneratorConfig(
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
    )

    generator = make_generator(gen_config)

    # Configure the loop
    loop_config = AutoresearchConfig(
        max_iterations=args.max_iterations,
        max_consecutive_failures=args.max_failures,
    )

    # Run the loop
    print("Starting autoresearch loop...")
    print()
    result = run_loop_with_generator(
        generator=generator,
        baselines=baselines,
        benchmark_data=benchmark_data,
        config=loop_config,
    )

    # Print results
    print_results(result)

    # Summary — compare final vs initial baselines
    if result.final_baselines:
        for name in sorted(baselines.benchmarks.keys()):
            initial = baselines.benchmarks.get(name)
            final = result.final_baselines.benchmarks.get(name)
            if initial and final and final.final_energy < initial.final_energy:
                improvement = (1 - final.final_energy / initial.final_energy) * 100
                print(
                    f"  {name}: {initial.final_energy:.4f} -> "
                    f"{final.final_energy:.4f} ({improvement:+.1f}%)"
                )

    if result.accepted == 0:
        print("\n  No hypotheses were accepted. Try adjusting the model or temperature.")
    else:
        print(f"\n  {result.accepted} hypothesis(es) accepted!")
        print("  The energy function served as the objective judge.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
