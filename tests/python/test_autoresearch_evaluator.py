"""Tests for the autoresearch hypothesis evaluator.

Spec coverage: REQ-AUTO-005, REQ-AUTO-002,
               SCENARIO-AUTO-001, SCENARIO-AUTO-002, SCENARIO-AUTO-007
"""

from pathlib import Path

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
from carnot.autoresearch.evaluator import EvalResult, evaluate_hypothesis
from carnot.autoresearch.sandbox import SandboxResult


def _make_baselines() -> BaselineRecord:
    """Create test baselines."""
    record = BaselineRecord(version="0.1.0")
    record.benchmarks["double_well"] = BenchmarkMetrics(
        benchmark_name="double_well",
        final_energy=-5.0,
        convergence_steps=1000,
        wall_clock_seconds=1.0,
        peak_memory_mb=100.0,
    )
    record.benchmarks["rosenbrock"] = BenchmarkMetrics(
        benchmark_name="rosenbrock",
        final_energy=-3.0,
        convergence_steps=2000,
        wall_clock_seconds=2.0,
        peak_memory_mb=100.0,
    )
    return record


class TestEvaluator:
    """Tests for REQ-AUTO-005: three-gate evaluation."""

    def test_pass_on_improvement(self) -> None:
        """SCENARIO-AUTO-001: hypothesis that improves energy passes."""
        baselines = _make_baselines()
        sandbox_result = SandboxResult(
            success=True,
            metrics={
                "double_well": {"final_energy": -6.0, "wall_clock_seconds": 0.8},
                "rosenbrock": {"final_energy": -3.5, "wall_clock_seconds": 1.5},
            },
        )
        result = evaluate_hypothesis(sandbox_result, baselines)
        assert result.verdict == "PASS"
        assert result.primary_gate
        assert result.secondary_gate
        assert len(result.improvements) == 2

    def test_fail_on_sandbox_failure(self) -> None:
        """SCENARIO-AUTO-002: failed sandbox = FAIL verdict."""
        baselines = _make_baselines()
        sandbox_result = SandboxResult(
            success=False,
            error="ValueError: divergent energy",
        )
        result = evaluate_hypothesis(sandbox_result, baselines)
        assert result.verdict == "FAIL"
        assert "Sandbox failed" in result.reason

    def test_fail_on_energy_regression(self) -> None:
        """SCENARIO-AUTO-002: energy regression = FAIL."""
        baselines = _make_baselines()
        sandbox_result = SandboxResult(
            success=True,
            metrics={
                "double_well": {"final_energy": -2.0, "wall_clock_seconds": 0.5},
                "rosenbrock": {"final_energy": -1.0, "wall_clock_seconds": 1.0},
            },
        )
        result = evaluate_hypothesis(sandbox_result, baselines)
        assert result.verdict == "FAIL"
        assert not result.primary_gate

    def test_fail_on_time_budget(self) -> None:
        """REQ-AUTO-005: time budget exceeded = FAIL."""
        baselines = _make_baselines()
        sandbox_result = SandboxResult(
            success=True,
            metrics={
                "double_well": {"final_energy": -5.5, "wall_clock_seconds": 10.0},
            },
        )
        result = evaluate_hypothesis(sandbox_result, baselines)
        assert result.verdict == "FAIL"
        assert not result.secondary_gate

    def test_fail_on_memory_budget(self) -> None:
        """REQ-AUTO-005: memory budget exceeded = FAIL."""
        baselines = _make_baselines()
        sandbox_result = SandboxResult(
            success=True,
            metrics={
                "double_well": {"final_energy": -5.5, "wall_clock_seconds": 0.5},
                "peak_memory_mb": 500.0,
            },
        )
        result = evaluate_hypothesis(sandbox_result, baselines)
        assert result.verdict == "FAIL"
        assert not result.tertiary_gate

    def test_review_on_mixed_results(self) -> None:
        """SCENARIO-AUTO-007: mixed improvement/regression = REVIEW."""
        baselines = _make_baselines()
        sandbox_result = SandboxResult(
            success=True,
            metrics={
                "double_well": {"final_energy": -8.0, "wall_clock_seconds": 0.5},
                "rosenbrock": {"final_energy": -1.0, "wall_clock_seconds": 1.0},
            },
        )
        result = evaluate_hypothesis(sandbox_result, baselines)
        assert result.verdict == "REVIEW"
        assert len(result.improvements) > 0
        assert len(result.regressions) > 0

    def test_pass_no_regression(self) -> None:
        """REQ-AUTO-005: no regression = PASS even without improvement."""
        baselines = _make_baselines()
        sandbox_result = SandboxResult(
            success=True,
            metrics={
                "double_well": {"final_energy": -5.0, "wall_clock_seconds": 1.0},
            },
        )
        result = evaluate_hypothesis(sandbox_result, baselines)
        assert result.verdict == "PASS"

    def test_missing_benchmark_skipped(self) -> None:
        """REQ-AUTO-005: benchmarks not in metrics are skipped."""
        baselines = _make_baselines()
        sandbox_result = SandboxResult(
            success=True,
            metrics={},  # no benchmark results at all
        )
        result = evaluate_hypothesis(sandbox_result, baselines)
        assert result.verdict == "PASS"  # nothing to regress on

    def test_non_dict_metrics_skipped(self) -> None:
        """REQ-AUTO-005: non-dict benchmark metrics are skipped."""
        baselines = _make_baselines()
        sandbox_result = SandboxResult(
            success=True,
            metrics={
                "double_well": "not a dict",
            },
        )
        result = evaluate_hypothesis(sandbox_result, baselines)
        assert result.verdict == "PASS"

    def test_missing_energy_skipped(self) -> None:
        """REQ-AUTO-005: missing final_energy in metrics is skipped."""
        baselines = _make_baselines()
        sandbox_result = SandboxResult(
            success=True,
            metrics={
                "double_well": {"convergence_steps": 100},  # no final_energy
            },
        )
        result = evaluate_hypothesis(sandbox_result, baselines)
        assert result.verdict == "PASS"


class TestBaselines:
    """Tests for REQ-AUTO-002: baseline persistence."""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """REQ-AUTO-002: baselines can be saved and loaded."""
        record = _make_baselines()
        record.commit = "abc123"
        record.timestamp = "2026-04-03T00:00:00Z"

        path = tmp_path / "baseline.json"
        record.save(path)
        loaded = BaselineRecord.load(path)

        assert loaded.version == "0.1.0"
        assert loaded.commit == "abc123"
        assert len(loaded.benchmarks) == 2
        assert loaded.benchmarks["double_well"].final_energy == -5.0
        assert loaded.benchmarks["rosenbrock"].convergence_steps == 2000
