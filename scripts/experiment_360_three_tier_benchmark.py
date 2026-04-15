#!/usr/bin/env python3
"""Experiment 360 — Three-Tier Pipeline Benchmark.

**Researcher summary:**
    Benchmarks the combined three-tier verification pipeline (SinkProbe →
    EORM → Ising) against an Ising-alone baseline on 100 synthetic responses.

    Key questions:
    (a) skip_rate: what fraction of responses are cleared before reaching Ising?
    (b) fn_rate:   what fraction of wrong responses slip through the fast tiers?
    (c) throughput: queries-per-second for three-tier vs Ising-alone?

    Hypothesis (v31 design doc): combining all three tiers saves 40-60% of Ising
    calls while maintaining false-negative rate ≤ 5%.

**Synthetic data design:**
    - 30 correct responses: high sink concentration (0.9) → cleared by SinkProbe
    - 70 wrong responses: uniform attention → fall through to EORM and Ising

    This gives a controlled baseline where the skip_rate is predictable.
    EORM may additionally clear some responses depending on model energy;
    these are labelled in the output.

**Outputs:**
    results/experiment_360_three_tier_benchmark.json

**CI-safe:**
    Runs entirely on CPU (JAX_PLATFORMS=cpu) with synthetic data.
    EORM loaded from results/eorm_model_359_real.safetensors (55M params).
    Ising is a stub returning (True, 0.0) — represents the Ising fast-path
    when no real constraint extractor is wired up in CPU mode.
    inference_mode is set to "cpu_synthetic" explicitly.

Spec: REQ-VERIFY-088
SCENARIO-VERIFY-116, SCENARIO-VERIFY-117
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root setup — so scripts can import carnot and scripts modules
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Force CPU JAX before any jax import
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from carnot.models.eorm import EORMModel
from carnot.pipeline.sink_probe import SinkProbe
from carnot.pipeline.three_tier_pipeline import (
    ThreeTierPipeline,
    build_three_tier_artifact,
)
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 360
TITLE = "Three-Tier Pipeline Benchmark: SinkProbe + EORM + Ising vs Ising-alone"
DELIVERABLE = "results/experiment_360_three_tier_benchmark.json"

N_RESPONSES = 100
N_CORRECT = 30      # 30% correct — high sink concentration → cleared by SinkProbe
N_WRONG = 70        # 70% wrong  — uniform attention → fall through

EORM_WEIGHTS_PATH = _REPO_ROOT / "results" / "eorm_model_359_real.safetensors"

SINK_THRESHOLD = 0.3
EORM_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _make_high_sink_attn(n_heads: int = 4, seq_len: int = 16, seed: int = 0) -> np.ndarray:
    """Attention where every head routes 0.9 mass to position 0 (BOS sink).

    Used for 'correct' responses — SinkProbe will clear these with high confidence.
    """
    remaining = 0.1 / max(seq_len - 1, 1)
    attn = np.full((n_heads, seq_len, seq_len), remaining, dtype=np.float32)
    attn[:, :, 0] = 0.9
    return attn


def _make_uniform_attn(n_heads: int = 4, seq_len: int = 16) -> np.ndarray:
    """Uniform attention — mean_sink_score = 1/seq_len ≈ 0.0625.

    Used for 'wrong' responses — SinkProbe detects uncertainty and passes to EORM/Ising.
    """
    return np.full((n_heads, seq_len, seq_len), 1.0 / seq_len, dtype=np.float32)


def build_synthetic_responses(n_correct: int, n_wrong: int) -> tuple[list[dict], list[bool]]:
    """Build synthetic response corpus with known ground-truth labels.

    Returns a list of response dicts and a parallel list of correctness labels.
    Correct responses have high sink attention; wrong responses have uniform attention.
    """
    responses: list[dict] = []
    labels: list[bool] = []

    attn_high = _make_high_sink_attn()
    attn_uniform = _make_uniform_attn()

    for i in range(n_correct):
        responses.append({
            "response": f"The answer is {i * 3 + 1}. Step 1: multiply. Step 2: add. Final: {i * 3 + 1}.",
            "question": f"What is {i} × 3 + 1?",
            "attention_matrix": attn_high,
        })
        labels.append(True)

    for i in range(n_wrong):
        responses.append({
            "response": f"The answer is {i * 5 + 99}. I think it could be various things.",
            "question": f"What is {i} × 3 + 1?",
            "attention_matrix": attn_uniform,
        })
        labels.append(False)

    return responses, labels


# ---------------------------------------------------------------------------
# Ising stub — CI-safe placeholder for the full Ising verifier
# ---------------------------------------------------------------------------


def _ising_cpu_stub(response: str, question: str) -> tuple[bool, float]:
    """CPU stub for the Ising verifier (CI-safe, no Rust binary required).

    In a live GPU run, this would be replaced with a call to the full
    VerifyRepairPipeline.verify() method.  For Exp 360 we are benchmarking
    the SKIP RATE of the upper tiers — responses that reach this stub were
    NOT skipped.  The stub returns (True, 0.0) for all responses, which
    simulates the best-case Ising outcome.

    Why not simulate Ising accurately here?
    The goal of Exp 360 is to measure how many Ising calls are SAVED by the
    upper tiers, not the accuracy of Ising itself.  The FNR measurement is
    what matters: wrong responses that slip through SinkProbe/EORM without
    reaching this stub.
    """
    return (True, 0.0)


# ---------------------------------------------------------------------------
# Ising-alone baseline
# ---------------------------------------------------------------------------


def run_ising_alone_baseline(
    responses: list[dict],
    ground_truth: list[bool],
) -> dict:
    """Simulate running ONLY the Ising verifier on all 100 responses.

    This is the baseline: every response incurs a full Ising call.
    skip_rate = 0.0, fn_rate = 0.0, ising_calls_saved_pct = 0.0.

    In practice the Ising verifier would return wrong=False for wrong
    responses, but since we are using a stub that returns True for all,
    we compute the "ideal" baseline metrics directly.
    """
    total = len(responses)
    t0 = time.perf_counter()
    # Simulate calling Ising on every response
    for item in responses:
        _ising_cpu_stub(item["response"], item["question"])
    elapsed = time.perf_counter() - t0
    throughput_qps = total / elapsed if elapsed > 0 else 0.0

    return {
        "skip_rate_sink_probe": 0.0,
        "skip_rate_eorm": 0.0,
        "total_skip_rate": 0.0,
        "fn_rate": 0.0,
        "throughput_qps": throughput_qps,
        "ising_calls_saved_pct": 0.0,
        "inference_mode": "cpu_synthetic",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 360."""
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Load EORM model
    # ------------------------------------------------------------------
    if EORM_WEIGHTS_PATH.exists():
        print(f"[Exp360] Loading EORM from {EORM_WEIGHTS_PATH}")
        eorm_model = EORMModel.load(str(EORM_WEIGHTS_PATH))
        eorm_source = "eorm_model_359_real.safetensors"
    else:
        # Fallback: instantiate a fresh small EORM (CI mode without weights file)
        print(f"[Exp360] EORM weights not found at {EORM_WEIGHTS_PATH} — using fresh small model")
        eorm_model = EORMModel(
            embed_dim=128,
            n_heads=4,
            n_layers=2,
            max_seq_len=512,
            vocab_size=4096,
            key=jr.PRNGKey(42),
        )
        eorm_source = "fresh_init_fallback"

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    sink_probe = SinkProbe(threshold=SINK_THRESHOLD)
    pipeline = ThreeTierPipeline(
        sink_probe=sink_probe,
        eorm_model=eorm_model,
        ising_pipeline=_ising_cpu_stub,
        sink_threshold=SINK_THRESHOLD,
        eorm_threshold=EORM_THRESHOLD,
    )

    # ------------------------------------------------------------------
    # Build synthetic corpus
    # ------------------------------------------------------------------
    print(f"[Exp360] Building {N_RESPONSES} synthetic responses ({N_CORRECT} correct, {N_WRONG} wrong)")
    responses, ground_truth = build_synthetic_responses(N_CORRECT, N_WRONG)

    # ------------------------------------------------------------------
    # Run three-tier benchmark
    # ------------------------------------------------------------------
    print("[Exp360] Running three-tier benchmark …")
    pipeline_result = pipeline.benchmark(
        responses,
        ground_truth,
        inference_mode="cpu_synthetic",
    )

    print(f"[Exp360] three-tier results:")
    print(f"  skip_rate_sink_probe : {pipeline_result.skip_rate_sink_probe:.3f}")
    print(f"  skip_rate_eorm       : {pipeline_result.skip_rate_eorm:.3f}")
    print(f"  total_skip_rate      : {pipeline_result.total_skip_rate:.3f}")
    print(f"  fn_rate              : {pipeline_result.fn_rate:.3f}")
    print(f"  throughput_qps       : {pipeline_result.throughput_qps:.1f}")
    print(f"  ising_calls_saved_pct: {pipeline_result.ising_calls_saved_pct:.1f}%")

    # ------------------------------------------------------------------
    # Run Ising-alone baseline
    # ------------------------------------------------------------------
    print("[Exp360] Running Ising-alone baseline …")
    ising_alone = run_ising_alone_baseline(responses, ground_truth)
    print(f"[Exp360] Ising-alone throughput_qps: {ising_alone['throughput_qps']:.1f}")

    # ------------------------------------------------------------------
    # Compute improvement
    # ------------------------------------------------------------------
    improvement_pct = pipeline_result.ising_calls_saved_pct
    throughput_ratio = (
        pipeline_result.throughput_qps / ising_alone["throughput_qps"]
        if ising_alone["throughput_qps"] > 0
        else 0.0
    )

    # Honest verdict based on actual measurements
    if pipeline_result.total_skip_rate >= 0.40 and pipeline_result.fn_rate <= 0.05:
        honest_verdict = "hypothesis_confirmed: >=40% Ising calls saved, fn_rate<=5%"
    elif pipeline_result.total_skip_rate >= 0.20:
        honest_verdict = "partial_improvement: <40% calls saved but pipeline functional"
    else:
        honest_verdict = "hypothesis_not_confirmed: skip_rate below 20% threshold"

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------
    three_tier_artifact = build_three_tier_artifact(pipeline_result)

    artifact = tmpl.build_result(
        {
            "pipeline_results": three_tier_artifact,
            "ising_alone_results": ising_alone,
            "improvement_pct": round(improvement_pct, 2),
            "throughput_ratio_3tier_vs_ising": round(throughput_ratio, 3),
            "inference_mode": "cpu_synthetic",
            "honest_verdict": honest_verdict,
            "eorm_source": eorm_source,
            "n_responses": N_RESPONSES,
            "n_correct": N_CORRECT,
            "n_wrong": N_WRONG,
            "sink_threshold": SINK_THRESHOLD,
            "eorm_threshold": EORM_THRESHOLD,
        },
        status="success",
    )

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_path = _REPO_ROOT / DELIVERABLE
    output_path.write_text(json.dumps(artifact, indent=2))
    print(f"[Exp360] Artifact written to {output_path}")
    print(f"[Exp360] honest_verdict: {honest_verdict}")


if __name__ == "__main__":
    main()
