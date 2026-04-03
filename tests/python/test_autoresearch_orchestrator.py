"""Tests for the autoresearch orchestrator — the self-improvement loop.

Spec coverage: FR-11, REQ-AUTO-003, REQ-AUTO-005, REQ-AUTO-007,
               REQ-AUTO-008, REQ-AUTO-009, REQ-AUTO-010,
               SCENARIO-AUTO-001, SCENARIO-AUTO-002, SCENARIO-AUTO-005
"""

from datetime import timedelta, timezone
from pathlib import Path

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
from carnot.autoresearch.experiment_log import ExperimentEntry, ExperimentLog
from carnot.autoresearch.orchestrator import AutoresearchConfig, LoopResult, run_loop


def _make_baselines() -> BaselineRecord:
    """Create test baselines with known metrics."""
    record = BaselineRecord(version="0.1.0")
    record.benchmarks["double_well"] = BenchmarkMetrics(
        benchmark_name="double_well",
        final_energy=-5.0,
        convergence_steps=1000,
        wall_clock_seconds=1.0,
        peak_memory_mb=100.0,
    )
    return record


# A hypothesis that improves on the baseline
GOOD_HYPOTHESIS = (
    "better energy",
    """
def run(benchmark_data):
    return {"double_well": {"final_energy": -6.0, "wall_clock_seconds": 0.8}}
""",
)

# A hypothesis that regresses
BAD_HYPOTHESIS = (
    "worse energy",
    """
def run(benchmark_data):
    return {"double_well": {"final_energy": -1.0, "wall_clock_seconds": 0.5}}
""",
)

# A hypothesis that crashes
CRASHING_HYPOTHESIS = (
    "crash",
    """
def run(benchmark_data):
    raise ValueError("divergent energy")
""",
)

# A hypothesis that tries to import os
UNSAFE_HYPOTHESIS = (
    "unsafe",
    """
import os
def run(benchmark_data):
    return {}
""",
)


class TestOrchestrator:
    """Tests for FR-11: the autoresearch loop."""

    def test_successful_improvement_cycle(self) -> None:
        """SCENARIO-AUTO-001: good hypothesis is accepted, baselines updated."""
        baselines = _make_baselines()
        result = run_loop([GOOD_HYPOTHESIS], baselines, {})

        assert result.iterations == 1
        assert result.accepted == 1
        assert result.rejected == 0
        assert not result.circuit_breaker_tripped
        # Baselines should be updated with the improved energy
        assert result.final_baselines is not None
        assert result.final_baselines.benchmarks["double_well"].final_energy == -6.0

    def test_rejected_hypothesis(self) -> None:
        """SCENARIO-AUTO-002: bad hypothesis is rejected, baselines unchanged."""
        baselines = _make_baselines()
        original_energy = baselines.benchmarks["double_well"].final_energy
        result = run_loop([BAD_HYPOTHESIS], baselines, {})

        assert result.iterations == 1
        assert result.accepted == 0
        assert result.rejected == 1
        # Baselines should NOT be updated
        assert result.final_baselines is not None
        assert result.final_baselines.benchmarks["double_well"].final_energy == original_energy

    def test_crashing_hypothesis_rejected(self) -> None:
        """SCENARIO-AUTO-002: crashing hypothesis is rejected gracefully."""
        baselines = _make_baselines()
        result = run_loop([CRASHING_HYPOTHESIS], baselines, {})

        assert result.rejected == 1
        assert result.accepted == 0
        # Error should be logged
        entry = result.experiment_log.entries[0]
        assert not entry.sandbox_success
        assert "ValueError" in (entry.sandbox_error or "")

    def test_unsafe_hypothesis_rejected(self) -> None:
        """SCENARIO-AUTO-004: unsafe import is blocked and hypothesis rejected."""
        baselines = _make_baselines()
        result = run_loop([UNSAFE_HYPOTHESIS], baselines, {})

        assert result.rejected == 1
        entry = result.experiment_log.entries[0]
        assert "Blocked imports" in (entry.sandbox_error or "")

    def test_circuit_breaker(self) -> None:
        """SCENARIO-AUTO-005: consecutive failures halt the loop."""
        baselines = _make_baselines()
        # 5 bad hypotheses with circuit breaker at 3
        hypotheses = [BAD_HYPOTHESIS] * 5
        config = AutoresearchConfig(max_consecutive_failures=3)
        result = run_loop(hypotheses, baselines, {}, config=config)

        assert result.circuit_breaker_tripped
        assert result.rejected == 3  # stopped after 3
        assert result.iterations == 3

    def test_max_iterations_limit(self) -> None:
        """REQ-AUTO-009: loop respects max_iterations."""
        baselines = _make_baselines()
        hypotheses = [GOOD_HYPOTHESIS] * 10
        config = AutoresearchConfig(max_iterations=3)
        result = run_loop(hypotheses, baselines, {}, config=config)

        assert result.iterations == 3
        assert not result.circuit_breaker_tripped

    def test_mixed_hypotheses(self) -> None:
        """REQ-AUTO-005: mix of good and bad hypotheses."""
        baselines = _make_baselines()
        hypotheses = [GOOD_HYPOTHESIS, BAD_HYPOTHESIS, GOOD_HYPOTHESIS]
        result = run_loop(hypotheses, baselines, {})

        assert result.iterations == 3
        assert result.accepted == 2  # first and third
        assert result.rejected == 1  # second

    def test_baselines_rise_over_loop(self) -> None:
        """REQ-AUTO-002: baselines update during the loop, raising the bar."""
        baselines = _make_baselines()
        # Two hypotheses: first improves to -6, second improves to -7
        hypotheses = [
            ("step 1", 'def run(d): return {"double_well": {"final_energy": -6.0, "wall_clock_seconds": 0.5}}'),
            ("step 2", 'def run(d): return {"double_well": {"final_energy": -7.0, "wall_clock_seconds": 0.5}}'),
        ]
        result = run_loop(hypotheses, baselines, {})

        assert result.accepted == 2
        assert result.final_baselines is not None
        # Baselines should reflect the latest improvement
        assert result.final_baselines.benchmarks["double_well"].final_energy == -7.0

    def test_experiment_log_populated(self) -> None:
        """REQ-AUTO-008: experiment log captures full lifecycle."""
        baselines = _make_baselines()
        result = run_loop([GOOD_HYPOTHESIS, BAD_HYPOTHESIS], baselines, {})

        assert len(result.experiment_log) == 2
        # First entry should be accepted
        e0 = result.experiment_log.entries[0]
        assert e0.outcome == "accepted"
        assert e0.eval_verdict == "PASS"
        assert e0.sandbox_success
        # Second entry should be rejected
        e1 = result.experiment_log.entries[1]
        assert e1.outcome == "rejected"
        assert e1.eval_verdict == "FAIL"

    def test_benchmark_data_passed_through(self) -> None:
        """REQ-AUTO-004: benchmark data reaches the hypothesis."""
        baselines = _make_baselines()
        hypotheses = [
            ("echo dim", 'def run(d): return {"double_well": {"final_energy": -6.0, "wall_clock_seconds": 0.5, "dim": d["dim"]}}'),
        ]
        result = run_loop(hypotheses, baselines, {"dim": 42})

        assert result.accepted == 1
        entry = result.experiment_log.entries[0]
        assert entry.sandbox_metrics["double_well"]["dim"] == 42

    def test_skip_already_rejected(self) -> None:
        """REQ-AUTO-007: hypotheses with IDs in the rejected registry are skipped.

        We pre-populate the experiment log with a rejected entry whose ID
        matches what the loop will generate. Since IDs include a timestamp,
        we monkey-patch datetime to make it predictable.
        """
        import unittest.mock
        from datetime import datetime as real_datetime

        baselines = _make_baselines()

        # Pre-populate the log with a "rejected" entry whose ID matches
        # what the loop will generate. The ID format is auto-YYYYMMDD-HHMMSS-NNN.
        # We mock datetime to produce a known timestamp.
        fake_now = real_datetime(2099, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(0)))
        with unittest.mock.patch(
            "carnot.autoresearch.orchestrator.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)
            # The loop will generate ID "auto-20990101-000000-000"
            pre_log = ExperimentLog()
            pre_log.append(ExperimentEntry(
                id="auto-20990101-000000-000",
                timestamp="",
                hypothesis_code="",
                outcome="rejected",
            ))

            result = run_loop(
                [GOOD_HYPOTHESIS],
                baselines,
                {},
                experiment_log=pre_log,
            )

        # The hypothesis should have been skipped — no new entries
        assert result.iterations == 0
        assert result.accepted == 0

    def test_review_verdict(self) -> None:
        """SCENARIO-AUTO-007: mixed results produce REVIEW verdict, not auto-accepted."""
        baselines = _make_baselines()
        # Add a second benchmark so we can have mixed results
        baselines.benchmarks["rosenbrock"] = BenchmarkMetrics(
            benchmark_name="rosenbrock",
            final_energy=-3.0,
            convergence_steps=2000,
            wall_clock_seconds=2.0,
        )
        # Improve double_well but regress rosenbrock
        hypotheses = [
            ("mixed", 'def run(d): return {"double_well": {"final_energy": -8.0, "wall_clock_seconds": 0.5}, "rosenbrock": {"final_energy": -1.0, "wall_clock_seconds": 1.0}}'),
        ]
        result = run_loop(hypotheses, baselines, {})

        assert result.pending_review == 1
        assert result.accepted == 0
        entry = result.experiment_log.entries[0]
        assert entry.eval_verdict == "REVIEW"
        assert entry.outcome == "pending_review"

    def test_hypothesis_with_non_dict_metrics(self) -> None:
        """REQ-AUTO-002: non-dict metrics in accepted hypothesis are skipped during baseline update."""
        baselines = _make_baselines()
        # Return a mix of dict and non-dict metric values
        hypotheses = [
            ("partial", 'def run(d): return {"double_well": {"final_energy": -6.0, "wall_clock_seconds": 0.5}, "extra_info": "not a dict"}'),
        ]
        result = run_loop(hypotheses, baselines, {})

        assert result.accepted == 1
        # Non-dict "extra_info" should not appear in baselines
        assert "extra_info" not in result.final_baselines.benchmarks

    def test_hypothesis_missing_energy_key(self) -> None:
        """REQ-AUTO-002: metrics dict without final_energy is skipped during baseline update."""
        baselines = _make_baselines()
        hypotheses = [
            ("no energy", 'def run(d): return {"double_well": {"final_energy": -6.0, "wall_clock_seconds": 0.5}, "new_bench": {"convergence_steps": 100}}'),
        ]
        result = run_loop(hypotheses, baselines, {})

        assert result.accepted == 1
        # "new_bench" has no final_energy, should not be added to baselines
        assert "new_bench" not in result.final_baselines.benchmarks


class TestExperimentLog:
    """Tests for REQ-AUTO-008: experiment log persistence."""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """REQ-AUTO-008: experiment log can be saved and loaded."""
        log = ExperimentLog()
        log.append(ExperimentEntry(
            id="test-001",
            timestamp="2026-04-03T00:00:00Z",
            hypothesis_code="def run(d): return {}",
            eval_verdict="PASS",
            outcome="accepted",
        ))
        log.append(ExperimentEntry(
            id="test-002",
            timestamp="2026-04-03T00:01:00Z",
            hypothesis_code="def run(d): raise ValueError('bad')",
            eval_verdict="FAIL",
            outcome="rejected",
        ))

        path = tmp_path / "experiments.json"
        log.save(path)
        loaded = ExperimentLog.load(path)

        assert len(loaded) == 2
        assert loaded.entries[0].id == "test-001"
        assert loaded.entries[1].outcome == "rejected"

    def test_rejected_ids(self) -> None:
        """REQ-AUTO-007: rejected registry prevents re-proposal."""
        log = ExperimentLog()
        log.append(ExperimentEntry(id="a", timestamp="", hypothesis_code="", outcome="accepted"))
        log.append(ExperimentEntry(id="b", timestamp="", hypothesis_code="", outcome="rejected"))
        log.append(ExperimentEntry(id="c", timestamp="", hypothesis_code="", outcome="rejected"))

        rejected = log.rejected_ids()
        assert "b" in rejected
        assert "c" in rejected
        assert "a" not in rejected

    def test_consecutive_failures(self) -> None:
        """REQ-AUTO-009: consecutive failure counting for circuit breaker."""
        log = ExperimentLog()
        log.append(ExperimentEntry(id="1", timestamp="", hypothesis_code="", outcome="accepted"))
        assert log.consecutive_failures() == 0

        log.append(ExperimentEntry(id="2", timestamp="", hypothesis_code="", outcome="rejected"))
        assert log.consecutive_failures() == 1

        log.append(ExperimentEntry(id="3", timestamp="", hypothesis_code="", outcome="rejected"))
        assert log.consecutive_failures() == 2

        log.append(ExperimentEntry(id="4", timestamp="", hypothesis_code="", outcome="accepted"))
        assert log.consecutive_failures() == 0  # streak broken

    def test_empty_log(self) -> None:
        """REQ-AUTO-008: empty log operations work."""
        log = ExperimentLog()
        assert len(log) == 0
        assert log.consecutive_failures() == 0
        assert log.rejected_ids() == set()

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """REQ-AUTO-008: loading a nonexistent file returns empty log."""
        path = tmp_path / "nonexistent.json"
        log = ExperimentLog.load(path)
        assert len(log) == 0
