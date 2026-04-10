#!/usr/bin/env python3
"""Experiment 102: Latency Benchmark — Is differentiable constraint verification fast enough for guided decoding?

**Researcher summary:**
    Measures the exact latency of the differentiable constraint pipeline (Exp 66)
    at various scales to determine whether energy-guided decoding is viable during
    LLM token generation. The critical threshold: <1ms for per-token use, 1-10ms
    for per-sentence batched verification, >10ms for post-hoc only.

**Detailed explanation for engineers:**
    The Carnot verification pipeline has three backends:
    1. Python VerifyRepairPipeline.verify() — the production path (extraction + JAX energy)
    2. JAX JIT-compiled differentiable forward pass — the Exp 66 differentiable verifier
    3. Rust VerifyPipeline — native speed via carnot-python bindings (if available)

    This experiment benchmarks ALL of them with identical inputs and timing methodology:
    - 100 warm-up iterations (to trigger JIT compilation and cache warming)
    - 1000 timed iterations, reporting p50/p95/p99/mean/std
    - Scale sweeps across input length (10-1000 tokens) and constraint count (1-50)
    - Component-level timing: embedding, extraction, Ising energy, MLP scoring

    The key question answered: at 50 tokens/second LLM generation speed, each token
    has a 20ms budget. If the constraint check fits in <10ms of that budget, we can
    do energy-guided decoding (the Kona path). Otherwise, we're limited to post-hoc
    verify-repair loops.

    Results feed directly into research-program.md Goal #4 and inform whether to
    pursue Phase 17 guided decoding or stick with Phase 16 verify-repair.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_102_latency_benchmark.py

REQ: REQ-EBT-001, REQ-VERIFY-001, REQ-CORE-005
SCENARIO: SCENARIO-VERIFY-004
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Force CPU for reproducibility (see CLAUDE.md: ROCm JAX crashes, always use CPU).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.pipeline.extract import AutoExtractor

# Import Exp 66 components for the differentiable verifier forward pass.
from experiment_66_differentiable_constraints import (
    DifferentiableVerifierParams,
    continuous_ising_energy,
    differentiable_verifier_forward,
    init_verifier_params,
    _check_constraints,
    EMBED_DIM,
    N_CONSTRAINTS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP_ITERS = 100
MEASURE_ITERS = 1000
INPUT_LENGTHS = [10, 50, 100, 500, 1000]  # simulated token counts
CONSTRAINT_COUNTS = [1, 5, 10, 50]  # synthetic constraint counts
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OPS_DIR = Path(__file__).resolve().parent.parent / "ops"

# Simulated text fragments at various "token" lengths.
# Each "token" is roughly 5 characters of natural-language text.
_BASE_TEXT = (
    "The answer is 42. Because 20 + 22 = 42. "
    "If x is positive, then x squared is also positive. "
    "All mammals are warm-blooded. Paris is the capital of France. "
    "There are 7 continents on Earth. "
)


def _build_text(n_tokens: int) -> str:
    """Build simulated text of approximately n_tokens tokens.

    **Detailed explanation for engineers:**
        Repeats the base text (which contains arithmetic, logic, factual, and
        NL content) until we reach roughly n_tokens * 5 characters. This gives
        the extractors realistic mixed-domain content to parse, not just
        padding. The constraint count scales naturally with text length.

    Args:
        n_tokens: Approximate number of tokens (1 token ≈ 5 chars).

    Returns:
        Text string of approximately the target length.
    """
    target_chars = n_tokens * 5
    repeats = max(1, target_chars // len(_BASE_TEXT) + 1)
    return (_BASE_TEXT * repeats)[:target_chars]


def _build_constraint_vec(n_constraints: int, seed: int = 102) -> jax.Array:
    """Build a synthetic binary constraint vector of given size.

    **Detailed explanation for engineers:**
        Creates a random binary vector simulating n_constraints constraint
        pass/fail results. About 70% pass (1.0) and 30% fail (0.0), matching
        typical real-world distributions where most constraints are satisfied.

    Args:
        n_constraints: Number of constraints to simulate.
        seed: Random seed for reproducibility.

    Returns:
        JAX array of shape (n_constraints,) with values in {0.0, 1.0}.
    """
    rng = np.random.default_rng(seed)
    return jnp.array((rng.random(n_constraints) > 0.3).astype(np.float32))


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def percentile(data: list[float], p: float) -> float:
    """Compute percentile without numpy dependency.

    **Detailed explanation for engineers:**
        Linear interpolation between the two nearest ranks, identical to
        numpy's "linear" method. Avoids pulling in numpy just for stats.
    """
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


@dataclass
class TimingResult:
    """Latency statistics from a benchmark run.

    Attributes:
        name: Human-readable label for what was measured.
        n_iters: Number of timed iterations.
        p50_ms: Median latency in milliseconds.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        mean_ms: Arithmetic mean latency.
        std_ms: Standard deviation of latency.
        raw_ms: All individual measurements (for JSON export).
    """
    name: str
    n_iters: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    raw_ms: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_iters": self.n_iters,
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "mean_ms": round(self.mean_ms, 4),
            "std_ms": round(self.std_ms, 4),
        }


def measure(fn, warmup: int = WARMUP_ITERS, iters: int = MEASURE_ITERS, name: str = "") -> TimingResult:
    """Run a callable with warmup + timed iterations, return latency stats.

    **Detailed explanation for engineers:**
        Uses time.perf_counter_ns() for nanosecond resolution. Runs warmup
        iterations first (critical for JAX JIT — first call compiles the
        trace, subsequent calls use cached XLA code). Then collects iters
        measurements and computes percentile statistics.

        Forces garbage collection before the timed loop to minimize GC
        pause interference with measurements.

    Args:
        fn: Zero-argument callable to benchmark.
        warmup: Number of warmup iterations (default 100).
        iters: Number of timed iterations (default 1000).
        name: Human-readable label.

    Returns:
        TimingResult with full statistics.
    """
    # Warmup: trigger JIT compilation and cache warming.
    for _ in range(warmup):
        fn()

    # Force GC before timed loop.
    gc.collect()

    latencies_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        elapsed_ns = time.perf_counter_ns() - t0
        latencies_ms.append(elapsed_ns / 1_000_000.0)

    return TimingResult(
        name=name,
        n_iters=iters,
        p50_ms=percentile(latencies_ms, 50),
        p95_ms=percentile(latencies_ms, 95),
        p99_ms=percentile(latencies_ms, 99),
        mean_ms=statistics.mean(latencies_ms),
        std_ms=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        raw_ms=latencies_ms,
    )


# ---------------------------------------------------------------------------
# 1. Component-level benchmarks for differentiable pipeline (Exp 66)
# ---------------------------------------------------------------------------


def benchmark_components() -> list[TimingResult]:
    """Benchmark each component of the differentiable verifier independently.

    **Detailed explanation for engineers:**
        Measures five components in isolation:
        1. Sentence embedding computation (simulated — deterministic hash-based
           mock when sentence-transformers is unavailable, real model otherwise)
        2. Constraint extraction via AutoExtractor on mixed-domain text
        3. Continuous Ising energy computation via JAX
        4. Joint MLP scoring (the final classification layer)
        5. Full end-to-end differentiable forward pass

        Each component is measured with the standard warmup + 1000 iterations
        methodology. The simulated embedding uses a fixed random vector to
        isolate the compute cost from I/O and model loading overhead.

    Returns:
        List of TimingResult, one per component.
    """
    print("\n[1/4] Component-level benchmarks (differentiable pipeline)")
    results: list[TimingResult] = []

    # Set up shared state.
    key = jrandom.PRNGKey(102)
    params = init_verifier_params(
        n_spins=N_CONSTRAINTS,
        embed_dim=EMBED_DIM,
        hidden_dim=64,
        key=key,
    )

    # Test input: moderate-length text with mixed constraints.
    test_text = _build_text(100)
    test_question = "What is 20 + 22?"
    test_embedding = jnp.array(np.random.default_rng(102).standard_normal(EMBED_DIM).astype(np.float32))
    test_constraint_vec = jnp.array(
        [float(x) for x in _check_constraints(test_question, test_text)],
        dtype=jnp.float32,
    )

    extractor = AutoExtractor()

    # --- Component 1: Sentence embedding (simulated) ---
    # Use a deterministic mock: hash-based embedding that's fast but exercises
    # the same array shapes as a real model would produce.
    def embed_fn():
        # Simulate the cost of converting text → 384-dim vector.
        # Real all-MiniLM-L6-v2 would be much slower; we measure that separately
        # if the model is available.
        words = test_text.lower().split()
        # Simple deterministic "embedding": hash words → accumulate into vector.
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        for i, w in enumerate(words):
            idx = hash(w) % EMBED_DIM
            vec[idx] += 1.0
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return jnp.array(vec)

    results.append(measure(embed_fn, name="simulated_embedding"))
    print(f"  Simulated embedding: {results[-1].mean_ms:.3f}ms mean")

    # Try real sentence-transformers if available.
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")

        def real_embed_fn():
            return encoder.encode(f"Q: {test_question} A: {test_text}", show_progress_bar=False)

        results.append(measure(real_embed_fn, warmup=10, iters=100, name="real_embedding_MiniLM"))
        print(f"  Real MiniLM embedding: {results[-1].mean_ms:.3f}ms mean")
    except ImportError:
        print("  [SKIP] sentence-transformers not installed, skipping real embedding benchmark")

    # --- Component 2: Constraint extraction ---
    def extract_fn():
        return extractor.extract(test_text)

    results.append(measure(extract_fn, name="constraint_extraction"))
    print(f"  Constraint extraction: {results[-1].mean_ms:.3f}ms mean")

    # --- Component 3: Continuous Ising energy ---
    # JIT-compile the Ising energy function for fair comparison.
    jit_ising = jax.jit(continuous_ising_energy)
    # Trigger compilation.
    _ = jit_ising(
        jax.nn.sigmoid(10.0 * (test_constraint_vec - 0.5)),
        params.ising_biases,
        params.ising_J,
    )

    def ising_fn():
        relaxed = jax.nn.sigmoid(10.0 * (test_constraint_vec - 0.5))
        return jit_ising(relaxed, params.ising_biases, params.ising_J).block_until_ready()

    results.append(measure(ising_fn, name="ising_energy_jit"))
    print(f"  Ising energy (JIT): {results[-1].mean_ms:.3f}ms mean")

    # --- Component 4: Joint MLP scoring ---
    relaxed_spins = jax.nn.sigmoid(10.0 * (test_constraint_vec - 0.5))
    ising_e = continuous_ising_energy(relaxed_spins, params.ising_biases, params.ising_J)
    ising_e_norm = jnp.tanh(ising_e * 0.1)
    mlp_input = jnp.concatenate([
        test_embedding, test_constraint_vec, relaxed_spins, ising_e_norm.reshape(1),
    ])

    @jax.jit
    def mlp_fn(x):
        h = jax.nn.silu(x @ params.mlp_w1 + params.mlp_b1)
        logit = (h @ params.mlp_w2 + params.mlp_b2).squeeze()
        return jax.nn.sigmoid(logit)

    # Trigger JIT.
    _ = mlp_fn(mlp_input).block_until_ready()

    def mlp_bench():
        return mlp_fn(mlp_input).block_until_ready()

    results.append(measure(mlp_bench, name="mlp_scoring_jit"))
    print(f"  MLP scoring (JIT): {results[-1].mean_ms:.3f}ms mean")

    # --- Component 5: Full end-to-end differentiable forward pass ---
    jit_forward = jax.jit(differentiable_verifier_forward, static_argnums=())

    def full_forward():
        return jit_forward(params, test_embedding, test_constraint_vec, 10.0).block_until_ready()

    # Trigger JIT.
    _ = full_forward()

    results.append(measure(full_forward, name="full_differentiable_forward_jit"))
    print(f"  Full differentiable forward (JIT): {results[-1].mean_ms:.3f}ms mean")

    return results


# ---------------------------------------------------------------------------
# 2. Scale sweep: input length × constraint count
# ---------------------------------------------------------------------------


def benchmark_scale_sweep() -> dict[str, list[dict]]:
    """Benchmark latency across input lengths and constraint counts.

    **Detailed explanation for engineers:**
        Creates a matrix of (input_length, constraint_count) and measures
        two things at each cell:
        1. AutoExtractor.extract() latency — scales with input text length
        2. Differentiable forward pass latency — scales with constraint count

        The extraction cost depends on text length (regex scanning), while
        the Ising + MLP cost depends on constraint count (matrix operations).
        This tells us which dimension is the bottleneck.

    Returns:
        Dict with "extraction_sweep" and "forward_sweep" keys, each
        containing a list of measurement dicts.
    """
    print("\n[2/4] Scale sweep: input length × constraint count")

    extractor = AutoExtractor()
    key = jrandom.PRNGKey(102)
    extraction_results: list[dict] = []
    forward_results: list[dict] = []

    for n_tokens in INPUT_LENGTHS:
        text = _build_text(n_tokens)

        # Measure extraction latency at this input length.
        def extract_fn():
            return extractor.extract(text)

        tr = measure(extract_fn, warmup=50, iters=500, name=f"extract_{n_tokens}tok")
        n_found = len(extractor.extract(text))
        extraction_results.append({
            "input_tokens": n_tokens,
            "input_chars": len(text),
            "constraints_found": n_found,
            **tr.to_dict(),
        })
        print(f"  Extract {n_tokens:>5} tokens ({len(text):>5} chars): "
              f"{tr.mean_ms:.3f}ms mean, {n_found} constraints found")

    for n_constraints in CONSTRAINT_COUNTS:
        # Build Ising model and MLP sized for this constraint count.
        params = init_verifier_params(
            n_spins=n_constraints,
            embed_dim=EMBED_DIM,
            hidden_dim=64,
            key=key,
        )
        constraint_vec = _build_constraint_vec(n_constraints)
        embedding = jnp.array(np.random.default_rng(102).standard_normal(EMBED_DIM).astype(np.float32))

        # JIT the forward pass for this specific size.
        @jax.jit
        def forward_fn(p, e, c):
            return differentiable_verifier_forward(p, e, c, 10.0)

        # Trigger JIT.
        _ = forward_fn(params, embedding, constraint_vec).block_until_ready()

        def bench_fn():
            return forward_fn(params, embedding, constraint_vec).block_until_ready()

        tr = measure(bench_fn, warmup=50, iters=500, name=f"forward_{n_constraints}c")
        forward_results.append({
            "n_constraints": n_constraints,
            **tr.to_dict(),
        })
        print(f"  Forward {n_constraints:>3} constraints: {tr.mean_ms:.3f}ms mean")

    # Cross-product matrix: for each (input_length, constraint_count), measure
    # the combined extraction + forward latency.
    matrix_results: list[dict] = []
    print("\n  Cross-product matrix (extraction + forward):")
    for n_tokens in INPUT_LENGTHS:
        text = _build_text(n_tokens)
        for n_constraints in CONSTRAINT_COUNTS:
            params = init_verifier_params(
                n_spins=n_constraints, embed_dim=EMBED_DIM, hidden_dim=64, key=key,
            )
            constraint_vec = _build_constraint_vec(n_constraints)
            embedding = jnp.array(np.random.default_rng(102).standard_normal(EMBED_DIM).astype(np.float32))

            @jax.jit
            def fwd(p, e, c):
                return differentiable_verifier_forward(p, e, c, 10.0)

            _ = fwd(params, embedding, constraint_vec).block_until_ready()

            def combined_fn():
                extractor.extract(text)
                return fwd(params, embedding, constraint_vec).block_until_ready()

            tr = measure(combined_fn, warmup=30, iters=200, name=f"combined_{n_tokens}t_{n_constraints}c")
            matrix_results.append({
                "input_tokens": n_tokens,
                "n_constraints": n_constraints,
                **tr.to_dict(),
            })

    # Print matrix.
    header = f"  {'tokens':>8}" + "".join(f"  {nc}c" for nc in CONSTRAINT_COUNTS)
    print(header)
    idx = 0
    for nt in INPUT_LENGTHS:
        row = f"  {nt:>8}"
        for _ in CONSTRAINT_COUNTS:
            row += f"  {matrix_results[idx]['mean_ms']:.2f}ms"
            idx += 1
        print(row)

    return {
        "extraction_sweep": extraction_results,
        "forward_sweep": forward_results,
        "matrix": matrix_results,
    }


# ---------------------------------------------------------------------------
# 3. Backend comparison
# ---------------------------------------------------------------------------


def benchmark_backends() -> list[TimingResult]:
    """Compare Python verify(), JAX JIT forward, and Rust verify (if available).

    **Detailed explanation for engineers:**
        Uses the same 100 test inputs for all backends so measurements are
        directly comparable. Each backend processes the full set sequentially.

        The Python path goes through VerifyRepairPipeline.verify(), which
        includes extraction + JAX energy evaluation. The JAX JIT path uses
        the Exp 66 differentiable forward pass (pre-compiled). The Rust path
        uses the native carnot-python bindings if available.

    Returns:
        List of TimingResult, one per backend.
    """
    print("\n[3/4] Backend comparison (100 identical inputs)")
    results: list[TimingResult] = []

    # Build 100 test inputs.
    test_inputs = []
    rng = np.random.default_rng(102)
    questions = [
        "What is 20 + 22?",
        "If it rains, the ground is wet. It rained. Is the ground wet?",
        "What is the capital of France?",
        "Write a function to add two numbers.",
        "Meeting A is 9-10, Meeting B is 11-12. Any conflict?",
    ]
    for i in range(100):
        q = questions[i % len(questions)]
        a_len = int(rng.integers(50, 200))
        a = _build_text(a_len)
        test_inputs.append((q, a))

    # --- Backend A: Python VerifyRepairPipeline.verify() ---
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(timeout_seconds=60.0)

    def python_verify():
        for q, a in test_inputs:
            pipeline.verify(q, a)

    tr = measure(python_verify, warmup=5, iters=50, name="python_verify_100inputs")
    per_call_ms = tr.mean_ms / 100.0
    results.append(tr)
    print(f"  Python VerifyRepairPipeline: {tr.mean_ms:.1f}ms total, {per_call_ms:.3f}ms/call")

    # --- Backend B: JAX JIT differentiable forward pass ---
    key = jrandom.PRNGKey(102)
    params = init_verifier_params(n_spins=N_CONSTRAINTS, embed_dim=EMBED_DIM, hidden_dim=64, key=key)
    jit_forward = jax.jit(differentiable_verifier_forward, static_argnums=())

    # Pre-compute embeddings and constraint vectors for all 100 inputs.
    embeddings = []
    constraint_vecs = []
    for q, a in test_inputs:
        emb = np.random.default_rng(hash(a) % (2**31)).standard_normal(EMBED_DIM).astype(np.float32)
        embeddings.append(jnp.array(emb))
        cv = jnp.array([float(x) for x in _check_constraints(q, a)], dtype=jnp.float32)
        constraint_vecs.append(cv)

    # Trigger JIT.
    _ = jit_forward(params, embeddings[0], constraint_vecs[0], 10.0).block_until_ready()

    def jax_forward():
        for emb, cv in zip(embeddings, constraint_vecs):
            jit_forward(params, emb, cv, 10.0).block_until_ready()

    tr = measure(jax_forward, warmup=5, iters=50, name="jax_jit_forward_100inputs")
    per_call_ms = tr.mean_ms / 100.0
    results.append(tr)
    print(f"  JAX JIT forward pass: {tr.mean_ms:.1f}ms total, {per_call_ms:.3f}ms/call")

    # --- Backend C: Rust VerifyPipeline (if available) ---
    try:
        os.environ["CARNOT_USE_RUST"] = "1"
        from carnot._rust_compat import RUST_AVAILABLE, RustVerifyPipeline

        if RUST_AVAILABLE:
            rust_pipeline = RustVerifyPipeline()

            def rust_verify():
                for q, a in test_inputs:
                    rust_pipeline.verify(q, a)

            tr = measure(rust_verify, warmup=5, iters=50, name="rust_verify_100inputs")
            per_call_ms = tr.mean_ms / 100.0
            results.append(tr)
            print(f"  Rust VerifyPipeline: {tr.mean_ms:.1f}ms total, {per_call_ms:.3f}ms/call")
        else:
            print("  [SKIP] Rust bindings not available (RUST_AVAILABLE=False)")
    except ImportError:
        print("  [SKIP] carnot._rust_compat not importable, Rust backend unavailable")
    finally:
        os.environ.pop("CARNOT_USE_RUST", None)

    return results


# ---------------------------------------------------------------------------
# 4. Analysis and classification
# ---------------------------------------------------------------------------


def analyze_results(
    components: list[TimingResult],
    scale_sweep: dict[str, list[dict]],
    backends: list[TimingResult],
) -> dict[str, Any]:
    """Analyze benchmark results and classify viability.

    **Detailed explanation for engineers:**
        Applies the three-tier classification from the research program:
        - <1ms mean → VIABLE for per-token guided decoding
        - 1-10ms mean → VIABLE for per-sentence or batched verification
        - >10ms mean → POST-HOC only, not viable for generation-time use

        Also identifies the bottleneck component and projects requirements
        for 50 tokens/second generation (20ms budget per token, constraint
        check must fit in <10ms of that budget).

    Returns:
        Dict with classification, bottleneck analysis, and recommendations.
    """
    analysis: dict[str, Any] = {}

    # Find the full forward pass result.
    full_forward = None
    for r in components:
        if "full_differentiable" in r.name:
            full_forward = r
            break

    if full_forward:
        mean = full_forward.mean_ms
        if mean < 1.0:
            classification = "VIABLE for per-token guided decoding"
            tier = "per-token"
        elif mean < 10.0:
            classification = "VIABLE for per-sentence or batched verification"
            tier = "per-sentence"
        else:
            classification = "POST-HOC only, not viable for generation-time use"
            tier = "post-hoc"

        analysis["differentiable_forward"] = {
            "mean_ms": round(mean, 4),
            "classification": classification,
            "tier": tier,
        }

    # Identify bottleneck component (which component has highest mean latency).
    component_latencies = [(r.name, r.mean_ms) for r in components]
    component_latencies.sort(key=lambda x: x[1], reverse=True)
    analysis["bottleneck"] = {
        "component": component_latencies[0][0],
        "mean_ms": round(component_latencies[0][1], 4),
        "ranking": [{"name": n, "mean_ms": round(m, 4)} for n, m in component_latencies],
    }

    # Backend comparison: find fastest.
    if backends:
        backend_speeds = [(r.name, r.mean_ms / 100.0) for r in backends]  # per-call ms
        backend_speeds.sort(key=lambda x: x[1])
        analysis["fastest_backend"] = {
            "name": backend_speeds[0][0],
            "per_call_ms": round(backend_speeds[0][1], 4),
            "all_backends": [{"name": n, "per_call_ms": round(m, 4)} for n, m in backend_speeds],
        }

    # Generation budget projection.
    # At 50 tokens/second, each token has 20ms budget.
    # Constraint check should use <50% of budget → <10ms.
    tokens_per_second = 50
    budget_per_token_ms = 1000.0 / tokens_per_second  # 20ms

    if full_forward:
        check_ms = full_forward.mean_ms
        budget_fraction = check_ms / budget_per_token_ms
        analysis["generation_budget"] = {
            "target_tokens_per_second": tokens_per_second,
            "budget_per_token_ms": budget_per_token_ms,
            "constraint_check_ms": round(check_ms, 4),
            "budget_fraction": round(budget_fraction, 4),
            "fits_in_budget": budget_fraction < 0.5,
            "recommendation": (
                "Guided decoding viable — constraint check uses "
                f"{budget_fraction*100:.1f}% of per-token budget"
                if budget_fraction < 0.5
                else f"Guided decoding NOT viable at {tokens_per_second} tok/s — "
                f"constraint check uses {budget_fraction*100:.1f}% of budget. "
                "Consider: (1) batching every N tokens, "
                "(2) async verification on separate thread, "
                "(3) Rust backend for lower latency."
            ),
        }

    # Extraction scaling analysis.
    extraction_sweep = scale_sweep.get("extraction_sweep", [])
    if len(extraction_sweep) >= 2:
        first = extraction_sweep[0]
        last = extraction_sweep[-1]
        scaling_factor = last["mean_ms"] / max(first["mean_ms"], 0.001)
        input_factor = last["input_chars"] / max(first["input_chars"], 1)
        analysis["extraction_scaling"] = {
            "input_range": f"{first['input_chars']}-{last['input_chars']} chars",
            "latency_range": f"{first['mean_ms']:.3f}-{last['mean_ms']:.3f} ms",
            "scaling_factor": round(scaling_factor, 2),
            "input_factor": round(input_factor, 2),
            "roughly_linear": abs(scaling_factor / input_factor - 1.0) < 1.0,
        }

    return analysis


# ---------------------------------------------------------------------------
# 5. Report generation
# ---------------------------------------------------------------------------


def save_results_json(
    components: list[TimingResult],
    scale_sweep: dict,
    backends: list[TimingResult],
    analysis: dict,
) -> Path:
    """Save full results to JSON."""
    results = {
        "experiment": 102,
        "title": "Latency Benchmark — Differentiable Constraint Pipeline",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": "CPU (JAX_PLATFORMS=cpu)",
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "components": [r.to_dict() for r in components],
        "scale_sweep": scale_sweep,
        "backends": [r.to_dict() for r in backends],
        "analysis": analysis,
    }
    out_path = RESULTS_DIR / "experiment_102_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    return out_path


def save_summary_md(
    components: list[TimingResult],
    scale_sweep: dict,
    backends: list[TimingResult],
    analysis: dict,
) -> Path:
    """Save human-readable Markdown summary to ops/latency-benchmark.md."""
    lines = [
        "# Experiment 102: Latency Benchmark Results",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"**Platform**: CPU (JAX_PLATFORMS=cpu)",
        f"**Warmup**: {WARMUP_ITERS} iterations | **Measured**: {MEASURE_ITERS} iterations",
        "",
        "## 1. Component-Level Latency (Differentiable Pipeline)",
        "",
        "| Component | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Std (ms) |",
        "|-----------|----------|----------|----------|-----------|----------|",
    ]
    for r in components:
        lines.append(
            f"| {r.name} | {r.p50_ms:.4f} | {r.p95_ms:.4f} | "
            f"{r.p99_ms:.4f} | {r.mean_ms:.4f} | {r.std_ms:.4f} |"
        )

    # Classification banner.
    da = analysis.get("differentiable_forward", {})
    if da:
        tier = da.get("tier", "unknown")
        emoji_map = {"per-token": "GREEN", "per-sentence": "YELLOW", "post-hoc": "RED"}
        lines += [
            "",
            f"### Classification: **{da.get('classification', 'N/A')}**",
            f"- Full differentiable forward pass mean: **{da.get('mean_ms', 0):.4f} ms**",
            f"- Verdict: **{emoji_map.get(tier, '?')}** — {tier} tier",
        ]

    # Bottleneck.
    bn = analysis.get("bottleneck", {})
    if bn:
        lines += [
            "",
            f"### Bottleneck: **{bn.get('component', 'N/A')}** ({bn.get('mean_ms', 0):.4f} ms)",
        ]

    # Scale sweep.
    lines += [
        "",
        "## 2. Scale Sweep",
        "",
        "### Extraction vs Input Length",
        "",
        "| Tokens | Chars | Constraints | Mean (ms) | p95 (ms) |",
        "|--------|-------|-------------|-----------|----------|",
    ]
    for row in scale_sweep.get("extraction_sweep", []):
        lines.append(
            f"| {row['input_tokens']} | {row['input_chars']} | "
            f"{row['constraints_found']} | {row['mean_ms']:.4f} | {row['p95_ms']:.4f} |"
        )

    lines += [
        "",
        "### Forward Pass vs Constraint Count",
        "",
        "| Constraints | Mean (ms) | p95 (ms) |",
        "|-------------|-----------|----------|",
    ]
    for row in scale_sweep.get("forward_sweep", []):
        lines.append(f"| {row['n_constraints']} | {row['mean_ms']:.4f} | {row['p95_ms']:.4f} |")

    # Matrix.
    matrix = scale_sweep.get("matrix", [])
    if matrix:
        lines += [
            "",
            "### Combined (Extraction + Forward) Matrix",
            "",
            "| Tokens \\ Constraints | " + " | ".join(str(c) for c in CONSTRAINT_COUNTS) + " |",
            "|" + "---|" * (len(CONSTRAINT_COUNTS) + 1),
        ]
        idx = 0
        for nt in INPUT_LENGTHS:
            row = f"| {nt} |"
            for _ in CONSTRAINT_COUNTS:
                row += f" {matrix[idx]['mean_ms']:.3f}ms |"
                idx += 1
            lines.append(row)

    # Backend comparison.
    lines += [
        "",
        "## 3. Backend Comparison (100 inputs each)",
        "",
        "| Backend | Total (ms) | Per-call (ms) |",
        "|---------|------------|---------------|",
    ]
    for r in backends:
        lines.append(f"| {r.name} | {r.mean_ms:.2f} | {r.mean_ms / 100.0:.4f} |")

    fb = analysis.get("fastest_backend", {})
    if fb:
        lines.append(f"\n**Fastest**: {fb.get('name', 'N/A')} at {fb.get('per_call_ms', 0):.4f} ms/call")

    # Generation budget.
    gb = analysis.get("generation_budget", {})
    if gb:
        lines += [
            "",
            "## 4. Generation Budget Analysis",
            "",
            f"- **Target generation speed**: {gb.get('target_tokens_per_second', 50)} tokens/second",
            f"- **Budget per token**: {gb.get('budget_per_token_ms', 20):.1f} ms",
            f"- **Constraint check cost**: {gb.get('constraint_check_ms', 0):.4f} ms",
            f"- **Budget fraction**: {gb.get('budget_fraction', 0)*100:.1f}%",
            f"- **Fits in budget**: {'Yes' if gb.get('fits_in_budget') else 'No'}",
            "",
            f"**Recommendation**: {gb.get('recommendation', 'N/A')}",
        ]

    # Extraction scaling.
    es = analysis.get("extraction_scaling", {})
    if es:
        lines += [
            "",
            "## 5. Extraction Scaling",
            "",
            f"- Input range: {es.get('input_range', 'N/A')}",
            f"- Latency range: {es.get('latency_range', 'N/A')}",
            f"- Scaling factor: {es.get('scaling_factor', 0):.2f}x "
            f"(vs {es.get('input_factor', 0):.2f}x input growth)",
            f"- Roughly linear: {'Yes' if es.get('roughly_linear') else 'No'}",
        ]

    lines.append("")
    out_path = OPS_DIR / "latency-benchmark.md"
    OPS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Experiment 102: Latency Benchmark")
    print("Is differentiable constraint verification fast enough")
    print("for energy-guided decoding during LLM generation?")
    print("=" * 60)
    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    t_start = time.time()

    # 1. Component-level benchmarks.
    components = benchmark_components()

    # 2. Scale sweep.
    scale_sweep = benchmark_scale_sweep()

    # 3. Backend comparison.
    backends = benchmark_backends()

    # 4. Analysis.
    print("\n[4/4] Analysis")
    analysis = analyze_results(components, scale_sweep, backends)

    da = analysis.get("differentiable_forward", {})
    if da:
        print(f"\n  === CLASSIFICATION: {da.get('classification', 'N/A')} ===")
        print(f"  Full forward pass: {da.get('mean_ms', 0):.4f} ms")

    bn = analysis.get("bottleneck", {})
    if bn:
        print(f"  Bottleneck: {bn.get('component', 'N/A')} ({bn.get('mean_ms', 0):.4f} ms)")

    gb = analysis.get("generation_budget", {})
    if gb:
        print(f"  Budget fraction at 50 tok/s: {gb.get('budget_fraction', 0)*100:.1f}%")
        print(f"  Recommendation: {gb.get('recommendation', 'N/A')}")

    elapsed = time.time() - t_start

    # 5. Save results.
    json_path = save_results_json(components, scale_sweep, backends, analysis)
    print(f"\nResults JSON: {json_path}")

    md_path = save_summary_md(components, scale_sweep, backends, analysis)
    print(f"Summary MD: {md_path}")

    print(f"\nTotal benchmark time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
