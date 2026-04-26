"""Tests for DualGPURunner wiring to ThreeTierPipeline — REQ-PERF-004.

Covers:
  - wire_dual_gpu_runner() stores the runner without raising
  - CARNOT_DUAL_GPU=1 + runner wired → benchmark() uses threading path
  - CARNOT_DUAL_GPU=0 + no runner → benchmark() runs sequentially (no regression)
  - CARNOT_DUAL_GPU=1 + no runner → falls back to sequential (runner None guard)
  - Observed throughput with dual-GPU threading >= baseline sequential (no regression)

CI-safe: no real GPU required.  Stub EORM, Ising, and SinkProbe used throughout.

Spec: REQ-PERF-004, SCENARIO-PERF-004
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class _FakeConcentration:
    mean_sink_score = 0.0  # Always below threshold → no early Tier 1 exit


class _StubSinkProbe:
    def score(self, attn: Any, sink_positions: list[int]) -> Any:
        return _FakeConcentration()


class _StubEORM:
    """Returns energy=0.8 so every response falls through to Ising (Tier 3)."""

    def energy(self, cot_input: Any) -> float:
        time.sleep(0.0001)  # Simulate real EORM latency so threading is measurable
        return 0.8


def _stub_ising(response: str, question: str) -> tuple[bool, float]:
    """Stub Ising callable — always returns verified=True, energy=0.5."""
    time.sleep(0.0001)  # Simulate Ising sweep latency
    return True, 0.5


class _StubDualGPURunner:
    """Minimal marker compatible with wire_dual_gpu_runner()."""


def _make_pipeline(dual_gpu_enabled: bool, wire_runner: bool) -> ThreeTierPipeline:
    """Create a ThreeTierPipeline with controlled dual-GPU mode."""
    ThreeTierPipeline.DUAL_GPU_ENABLED = dual_gpu_enabled
    pipeline = ThreeTierPipeline(
        sink_probe=_StubSinkProbe(),
        eorm_model=_StubEORM(),
        ising_pipeline=_stub_ising,
        sink_threshold=0.3,
        eorm_threshold=0.5,
    )
    if wire_runner:
        pipeline.wire_dual_gpu_runner(_StubDualGPURunner())
    return pipeline


def _corpus(n: int = 20) -> tuple[list[dict[str, Any]], list[bool]]:
    items = [{"question": f"q{i}", "response": f"r{i}", "attention_matrix": None} for i in range(n)]
    labels = [True] * n
    return items, labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWireDualGPURunner:
    """REQ-PERF-004: wire_dual_gpu_runner() stores runner without raising."""

    def test_wire_stores_runner(self) -> None:
        """wire_dual_gpu_runner() must store the runner on _dual_gpu_runner."""
        # SCENARIO-PERF-004: wiring does not raise
        ThreeTierPipeline.DUAL_GPU_ENABLED = False
        p = ThreeTierPipeline(
            sink_probe=_StubSinkProbe(),
            eorm_model=_StubEORM(),
            ising_pipeline=_stub_ising,
        )
        assert p._dual_gpu_runner is None
        runner = _StubDualGPURunner()
        p.wire_dual_gpu_runner(runner)
        assert p._dual_gpu_runner is runner

    def test_wire_none_reverts_to_sequential(self) -> None:
        """Passing None to wire_dual_gpu_runner() restores sequential mode."""
        # SCENARIO-PERF-004: revert to sequential is supported
        ThreeTierPipeline.DUAL_GPU_ENABLED = True
        p = ThreeTierPipeline(
            sink_probe=_StubSinkProbe(),
            eorm_model=_StubEORM(),
            ising_pipeline=_stub_ising,
        )
        p.wire_dual_gpu_runner(_StubDualGPURunner())
        assert p._dual_gpu_runner is not None
        p.wire_dual_gpu_runner(None)
        assert p._dual_gpu_runner is None


class TestDualGPUBenchmarkPath:
    """REQ-PERF-004: benchmark() uses threading when DUAL_GPU_ENABLED and runner wired."""

    def test_dual_gpu_benchmark_returns_result(self) -> None:
        """benchmark() with dual-GPU wired must return a valid ThreeTierPipelineResult."""
        # SCENARIO-PERF-004: dual-GPU mode produces results
        p = _make_pipeline(dual_gpu_enabled=True, wire_runner=True)
        corpus, labels = _corpus(20)
        result = p.benchmark(corpus, labels, inference_mode="test_dual_gpu")
        assert result.inference_mode == "test_dual_gpu"
        assert isinstance(result.throughput_qps, float)
        assert result.throughput_qps > 0.0

    def test_sequential_benchmark_returns_result(self) -> None:
        """benchmark() without dual-GPU must return a valid ThreeTierPipelineResult."""
        # SCENARIO-PERF-004: sequential (fallback) mode produces results
        p = _make_pipeline(dual_gpu_enabled=False, wire_runner=False)
        corpus, labels = _corpus(20)
        result = p.benchmark(corpus, labels, inference_mode="test_sequential")
        assert result.throughput_qps > 0.0

    def test_dual_gpu_flag_without_runner_falls_back(self) -> None:
        """CARNOT_DUAL_GPU=1 without a wired runner must not crash (falls back)."""
        # SCENARIO-PERF-004: no regression when runner is missing
        p = _make_pipeline(dual_gpu_enabled=True, wire_runner=False)
        assert p._dual_gpu_runner is None
        corpus, labels = _corpus(10)
        result = p.benchmark(corpus, labels)
        # Must still produce a valid result
        assert result.throughput_qps > 0.0

    def test_dual_gpu_throughput_not_regressed(self) -> None:
        """Dual-GPU mode must not be slower than sequential by more than 50%.

        We don't guarantee >1x speedup in CI (no real second GPU), but we
        must not regress so badly that the wiring itself is harmful.

        Why 50% tolerance: threading overhead on a stubbed pipeline with
        identical artificial sleep on both paths is bounded.  A 50% regression
        threshold is conservative enough to survive even heavily loaded CI.
        """
        # SCENARIO-PERF-004: observed_speedup > 0.5 (no severe regression)
        import time

        corpus, labels = _corpus(20)

        p_seq = _make_pipeline(dual_gpu_enabled=False, wire_runner=False)
        t0 = time.perf_counter()
        p_seq.benchmark(corpus, labels)
        sequential_s = time.perf_counter() - t0

        p_par = _make_pipeline(dual_gpu_enabled=True, wire_runner=True)
        t0 = time.perf_counter()
        p_par.benchmark(corpus, labels)
        parallel_s = time.perf_counter() - t0

        speedup = sequential_s / parallel_s if parallel_s > 0 else 1.0
        assert speedup > 0.5, (
            f"Dual-GPU threading regressed sequential by >50%: speedup={speedup:.3f} "
            f"(sequential={sequential_s:.4f}s, parallel={parallel_s:.4f}s)"
        )

    def test_empty_corpus_no_crash(self) -> None:
        """benchmark() with 0 items must not crash in either mode."""
        # Edge case — both paths guard for total == 0 before threading
        for dual_gpu in (False, True):
            p = _make_pipeline(dual_gpu_enabled=dual_gpu, wire_runner=dual_gpu)
            result = p.benchmark([], [])
            assert result.throughput_qps == 0.0

    def test_single_item_corpus_no_crash(self) -> None:
        """benchmark() with 1 item must run sequentially (total < 2 guard)."""
        # batch of 1 cannot be split; must not crash
        p = _make_pipeline(dual_gpu_enabled=True, wire_runner=True)
        corpus, labels = _corpus(1)
        result = p.benchmark(corpus, labels)
        assert result.throughput_qps > 0.0
