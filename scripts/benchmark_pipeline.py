"""Benchmark the verify-repair pipeline: latency, throughput, memory.

Measures baseline performance of the Carnot verification pipeline across
four dimensions:

1. verify() latency per domain (p50/p95/p99 over 100 calls each)
2. extract_constraints() scaling vs input length (50-5000 chars)
3. Batch throughput for 1000 sequential verify() calls
4. Peak RSS memory during init and during batch processing

Results are saved to ops/benchmark-results.md.

Spec: REQ-VERIFY-001
"""

from __future__ import annotations

import gc
import os
import resource
import statistics
import time
from pathlib import Path

# Force CPU for reproducibility (see CLAUDE.md).
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def get_rss_mb() -> float:
    """Return current peak RSS in megabytes."""
    # ru_maxrss is in KB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def percentile(data: list[float], p: float) -> float:
    """Compute percentile without numpy dependency."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


# ---------------------------------------------------------------------------
# Test inputs per domain
# ---------------------------------------------------------------------------

DOMAIN_INPUTS = {
    "arithmetic": "The sum of 47 + 28 = 75. Also 100 - 37 = 63. And 15 + 22 = 37.",
    "code": (
        "Here is a function:\n```python\n"
        "def add(a: int, b: int) -> int:\n"
        "    result = a + b\n"
        "    return result\n```\n"
    ),
    "logic": (
        "If it rains, then the ground is wet. "
        "All cats are mammals. "
        "Either the light is on or the room is dark."
    ),
    "nl": (
        "Paris is the capital of France. "
        "Water is composed of hydrogen and oxygen. "
        "There are 7 continents."
    ),
}

# Template for scaling test: repeat a sentence to reach target length.
SCALING_SENTENCE = "The result of 10 + 5 = 15. If x is positive, then x squared is positive. "


def build_input_of_length(target_len: int) -> str:
    """Repeat the scaling sentence to approximate target character count."""
    repeats = max(1, target_len // len(SCALING_SENTENCE) + 1)
    text = SCALING_SENTENCE * repeats
    return text[:target_len]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def benchmark_verify_latency(
    n_calls: int = 100,
) -> dict[str, dict[str, float]]:
    """Benchmark verify() latency per domain.

    Returns dict mapping domain -> {p50, p95, p99, mean} in milliseconds.
    """
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(timeout_seconds=60.0)
    results: dict[str, dict[str, float]] = {}

    for domain, text in DOMAIN_INPUTS.items():
        # Warm up JIT / caches with one call.
        pipeline.verify(f"Check this {domain} text", text, domain=domain)

        latencies: list[float] = []
        for _ in range(n_calls):
            t0 = time.perf_counter()
            pipeline.verify(f"Check this {domain} text", text, domain=domain)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        results[domain] = {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "mean": statistics.mean(latencies),
        }
    return results


def benchmark_extract_scaling() -> list[dict]:
    """Benchmark extract_constraints() vs input length (50-5000 chars).

    Returns list of {length, time_ms, n_constraints}.
    """
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(timeout_seconds=60.0)
    lengths = [50, 100, 250, 500, 1000, 2000, 3000, 5000]
    results: list[dict] = []

    for target_len in lengths:
        text = build_input_of_length(target_len)
        actual_len = len(text)

        # Warm up.
        pipeline.extract_constraints(text)

        times: list[float] = []
        n_constraints = 0
        for _ in range(20):
            t0 = time.perf_counter()
            cr = pipeline.extract_constraints(text)
            times.append((time.perf_counter() - t0) * 1000.0)
            n_constraints = len(cr)

        results.append({
            "length": actual_len,
            "time_ms_mean": statistics.mean(times),
            "time_ms_p95": percentile(times, 95),
            "n_constraints": n_constraints,
        })
    return results


def benchmark_batch_throughput(n_batch: int = 1000) -> dict[str, float]:
    """Benchmark 1000 sequential verify() calls, mixed domains.

    Returns {total_seconds, calls_per_second, mean_ms}.
    """
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(timeout_seconds=60.0)
    domains = list(DOMAIN_INPUTS.keys())
    inputs = [(d, DOMAIN_INPUTS[d]) for d in domains]

    # Warm up.
    for d, text in inputs:
        pipeline.verify(f"Check {d}", text, domain=d)

    t0 = time.perf_counter()
    for i in range(n_batch):
        d, text = inputs[i % len(inputs)]
        pipeline.verify(f"Check {d}", text, domain=d)
    elapsed = time.perf_counter() - t0

    return {
        "total_seconds": elapsed,
        "calls_per_second": n_batch / elapsed,
        "mean_ms": (elapsed / n_batch) * 1000.0,
    }


def benchmark_memory() -> dict[str, float]:
    """Benchmark peak RSS during init and during batch processing.

    Returns {rss_before_init_mb, rss_after_init_mb, rss_after_batch_mb}.
    """
    gc.collect()
    rss_before = get_rss_mb()

    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(timeout_seconds=60.0)
    # Force JAX init by running one call.
    pipeline.verify("test", "1 + 1 = 2", domain="arithmetic")

    rss_after_init = get_rss_mb()

    # Run a batch to measure memory growth.
    for i in range(500):
        d = list(DOMAIN_INPUTS.keys())[i % len(DOMAIN_INPUTS)]
        pipeline.verify(f"Check {d}", DOMAIN_INPUTS[d], domain=d)

    rss_after_batch = get_rss_mb()

    return {
        "rss_before_init_mb": rss_before,
        "rss_after_init_mb": rss_after_init,
        "rss_after_batch_mb": rss_after_batch,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def format_report(
    latency: dict,
    scaling: list[dict],
    throughput: dict,
    memory: dict,
) -> str:
    """Format benchmark results as Markdown."""
    lines = [
        "# Pipeline Benchmark Results",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"**Platform**: CPU (JAX_PLATFORMS=cpu)",
        "",
        "## 1. verify() Latency per Domain (100 calls each)",
        "",
        "| Domain | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) |",
        "|--------|----------|----------|----------|-----------|",
    ]
    for domain, stats in latency.items():
        lines.append(
            f"| {domain} | {stats['p50']:.2f} | {stats['p95']:.2f} | "
            f"{stats['p99']:.2f} | {stats['mean']:.2f} |"
        )

    lines += [
        "",
        "## 2. extract_constraints() vs Input Length",
        "",
        "| Input Length (chars) | Mean (ms) | p95 (ms) | Constraints Found |",
        "|---------------------|-----------|----------|-------------------|",
    ]
    for row in scaling:
        lines.append(
            f"| {row['length']} | {row['time_ms_mean']:.2f} | "
            f"{row['time_ms_p95']:.2f} | {row['n_constraints']} |"
        )

    lines += [
        "",
        "## 3. Batch Throughput (1000 sequential calls)",
        "",
        f"- **Total time**: {throughput['total_seconds']:.2f} s",
        f"- **Throughput**: {throughput['calls_per_second']:.1f} calls/s",
        f"- **Mean latency**: {throughput['mean_ms']:.2f} ms/call",
        "",
        "## 4. Memory Usage (Peak RSS)",
        "",
        f"- **Before init**: {memory['rss_before_init_mb']:.1f} MB",
        f"- **After init + JAX warmup**: {memory['rss_after_init_mb']:.1f} MB",
        f"- **After 500-call batch**: {memory['rss_after_batch_mb']:.1f} MB",
        f"- **Growth during batch**: "
        f"{memory['rss_after_batch_mb'] - memory['rss_after_init_mb']:.1f} MB",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print("=== Carnot Pipeline Benchmark ===\n")

    print("[1/4] Benchmarking verify() latency per domain...")
    latency = benchmark_verify_latency(n_calls=100)
    for domain, stats in latency.items():
        print(f"  {domain}: p50={stats['p50']:.2f}ms p95={stats['p95']:.2f}ms")

    print("[2/4] Benchmarking extract_constraints() scaling...")
    scaling = benchmark_extract_scaling()
    for row in scaling:
        print(f"  {row['length']} chars: {row['time_ms_mean']:.2f}ms mean")

    print("[3/4] Benchmarking batch throughput (1000 calls)...")
    throughput = benchmark_batch_throughput(n_batch=1000)
    print(f"  {throughput['calls_per_second']:.1f} calls/s")

    print("[4/4] Benchmarking memory usage...")
    memory = benchmark_memory()
    print(f"  Peak RSS after batch: {memory['rss_after_batch_mb']:.1f} MB")

    report = format_report(latency, scaling, throughput, memory)

    out_path = Path(__file__).resolve().parent.parent / "ops" / "benchmark-results.md"
    out_path.write_text(report)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
