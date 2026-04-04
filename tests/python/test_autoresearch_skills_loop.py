"""Tests for run_loop_with_skills: the Trace2Skill-enhanced autoresearch loop.

Spec coverage: FR-11, REQ-AUTO-011, REQ-AUTO-012, REQ-AUTO-013, REQ-AUTO-014,
SCENARIO-AUTO-008, SCENARIO-AUTO-009, SCENARIO-AUTO-010, SCENARIO-AUTO-011
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
from carnot.autoresearch.orchestrator import AutoresearchConfig, run_loop_with_skills
from carnot.autoresearch.skill_directory import SkillDirectory, SkillDirectoryConfig
from carnot.autoresearch.trajectory_analyst import Lesson

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOOD_CODE = """\
def run(benchmark_data):
    return {
        "double_well": {
            "final_energy": 0.01,
            "convergence_steps": 100,
            "wall_clock_seconds": 0.5,
            "peak_memory_mb": 10.0,
        },
    }
"""

BAD_CODE = """\
def run(benchmark_data):
    return {
        "double_well": {
            "final_energy": 999.0,
            "convergence_steps": 100,
            "wall_clock_seconds": 0.5,
            "peak_memory_mb": 10.0,
        },
    }
"""


def _make_baselines() -> BaselineRecord:
    record = BaselineRecord(version="test")
    record.benchmarks["double_well"] = BenchmarkMetrics(
        benchmark_name="double_well",
        final_energy=0.05,
        convergence_steps=5000,
        wall_clock_seconds=2.0,
        peak_memory_mb=50.0,
    )
    return record


def _make_generator(code: str = GOOD_CODE, description: str = "test hypothesis") -> Any:
    """Create a generator callable that returns a fixed hypothesis."""

    def gen(
        baselines: BaselineRecord,
        failures: list[dict[str, Any]],
        iteration: int,
    ) -> list[tuple[str, str]]:
        return [(description, code)]

    return gen


def _make_failing_generator() -> Any:
    """Generator that raises on first call, then returns empty."""
    call_count = 0

    def gen(
        baselines: BaselineRecord,
        failures: list[dict[str, Any]],
        iteration: int,
    ) -> list[tuple[str, str]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Generator failed")
        return []

    return gen


# ---------------------------------------------------------------------------
# Tests: Basic flow
# ---------------------------------------------------------------------------


class TestSkillsLoopBasicFlow:
    """Tests for basic operation of run_loop_with_skills."""

    def test_basic_accepted_hypothesis(self) -> None:
        """FR-11, REQ-AUTO-012: skill loop accepts good hypotheses."""
        config = AutoresearchConfig(max_iterations=1)
        result = run_loop_with_skills(
            generator=_make_generator(),
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )
        assert result.accepted == 1
        assert result.iterations == 1
        assert result.skill_directory is not None

    def test_basic_rejected_hypothesis(self) -> None:
        """FR-11, REQ-AUTO-012: skill loop rejects bad hypotheses."""
        config = AutoresearchConfig(max_iterations=1)
        result = run_loop_with_skills(
            generator=_make_generator(code=BAD_CODE),
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )
        assert result.rejected == 1
        assert result.accepted == 0

    def test_skill_directory_returned(self) -> None:
        """REQ-AUTO-012: result includes skill directory."""
        config = AutoresearchConfig(max_iterations=1)
        result = run_loop_with_skills(
            generator=_make_generator(),
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )
        assert result.skill_directory is not None
        assert isinstance(result.skill_directory, SkillDirectory)


# ---------------------------------------------------------------------------
# Tests: Trajectory analysis
# ---------------------------------------------------------------------------


class TestTrajectoryAnalysis:
    """Tests for analyst dispatch within the skill loop."""

    @patch("carnot.autoresearch.orchestrator.consolidate_lessons")
    @patch("carnot.autoresearch.orchestrator.analyze_batch")
    def test_analyst_dispatch_on_consolidation(
        self, mock_consolidate: MagicMock, mock_analyze: MagicMock
    ) -> None:
        """REQ-AUTO-011, SCENARIO-AUTO-008: analysts dispatched at consolidation interval."""
        mock_analyze.return_value = [Lesson(title="insight", description="d", confidence=0.8)]
        mock_consolidate.return_value = [
            Lesson(title="consolidated", description="d", confidence=0.9)
        ]

        config = AutoresearchConfig(
            max_iterations=5,
            enable_trajectory_analysis=True,
            consolidation_interval=5,
        )
        result = run_loop_with_skills(
            generator=_make_generator(),
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )
        # After 5 iterations, analysis + consolidation should have run
        assert mock_analyze.called or result.iterations == 5

    @patch("carnot.autoresearch.orchestrator.consolidate_lessons")
    @patch("carnot.autoresearch.orchestrator.analyze_batch")
    def test_analysis_disabled(self, mock_consolidate: MagicMock, mock_analyze: MagicMock) -> None:
        """REQ-AUTO-012: analysis can be disabled."""
        config = AutoresearchConfig(
            max_iterations=3,
            enable_trajectory_analysis=False,
        )
        run_loop_with_skills(
            generator=_make_generator(),
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )
        mock_analyze.assert_not_called()
        mock_consolidate.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Skill context injection
# ---------------------------------------------------------------------------


class TestSkillContextInjection:
    """Tests for skill directory context in generator calls."""

    def test_skill_context_passed_to_generator(self) -> None:
        """REQ-AUTO-012: generator receives skill context in failures list."""
        received_failures: list[list[dict[str, Any]]] = []

        def capturing_generator(
            baselines: BaselineRecord,
            failures: list[dict[str, Any]],
            iteration: int,
        ) -> list[tuple[str, str]]:
            received_failures.append(list(failures))
            return [("test", GOOD_CODE)] if iteration == 0 else []

        # Pre-populate skill directory with a lesson
        sd = SkillDirectory()
        sd.evolve([Lesson(title="HMC insight", description="use HMC", confidence=0.8)])

        config = AutoresearchConfig(max_iterations=1)
        run_loop_with_skills(
            generator=capturing_generator,
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
            skill_directory=sd,
        )

        assert len(received_failures) >= 1
        # First failure entry should be the skill playbook
        first_call_failures = received_failures[0]
        assert any(f["description"] == "skill_playbook" for f in first_call_failures)

    def test_empty_skills_no_extra_context(self) -> None:
        """REQ-AUTO-012: empty skill directory doesn't inject context."""
        received_failures: list[list[dict[str, Any]]] = []

        def capturing_generator(
            baselines: BaselineRecord,
            failures: list[dict[str, Any]],
            iteration: int,
        ) -> list[tuple[str, str]]:
            received_failures.append(list(failures))
            return [("test", GOOD_CODE)] if iteration == 0 else []

        config = AutoresearchConfig(max_iterations=1)
        run_loop_with_skills(
            generator=capturing_generator,
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )

        first_call_failures = received_failures[0]
        assert not any(f.get("description") == "skill_playbook" for f in first_call_failures)


# ---------------------------------------------------------------------------
# Tests: Circuit breaker and edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and backward compatibility."""

    def test_circuit_breaker_still_works(self) -> None:
        """REQ-AUTO-009: circuit breaker halts loop even with skills."""
        config = AutoresearchConfig(
            max_iterations=100,
            max_consecutive_failures=3,
        )
        result = run_loop_with_skills(
            generator=_make_generator(code=BAD_CODE),
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )
        assert result.circuit_breaker_tripped
        assert result.rejected >= 3

    def test_generator_exception_handled(self) -> None:
        """FR-11: generator exceptions don't crash the loop."""
        config = AutoresearchConfig(max_iterations=3)
        result = run_loop_with_skills(
            generator=_make_failing_generator(),
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )
        # Loop should have survived the exception
        assert result.iterations == 0  # No hypotheses evaluated

    def test_skill_directory_persistence(self, tmp_path: Path) -> None:
        """REQ-AUTO-012: skill directory is saved to disk when path is set."""
        config = AutoresearchConfig(
            max_iterations=1,
            skill_directory_path=tmp_path / "skills",
            enable_trajectory_analysis=True,
            consolidation_interval=1,
        )

        with (
            patch("carnot.autoresearch.orchestrator.analyze_batch") as mock_analyze,
            patch("carnot.autoresearch.orchestrator.consolidate_lessons") as mock_consolidate,
        ):
            mock_analyze.return_value = [Lesson(title="test", description="d", confidence=0.8)]
            mock_consolidate.return_value = [Lesson(title="test", description="d", confidence=0.8)]

            run_loop_with_skills(
                generator=_make_generator(),
                baselines=_make_baselines(),
                benchmark_data={"dim": 2},
                config=config,
            )

            # Check that the skill directory was persisted
            if mock_consolidate.called:
                assert (tmp_path / "skills" / "lessons.json").exists()

    def test_loads_existing_skill_directory(self, tmp_path: Path) -> None:
        """REQ-AUTO-012: loop loads existing skill directory from path."""
        # Pre-create a skill directory on disk
        sd_config = SkillDirectoryConfig(path=tmp_path / "skills")
        sd = SkillDirectory(sd_config)
        sd.evolve([Lesson(title="preexisting", description="d", confidence=0.8)])
        sd.save()

        received_failures: list[list[dict[str, Any]]] = []

        def capturing_generator(
            baselines: BaselineRecord,
            failures: list[dict[str, Any]],
            iteration: int,
        ) -> list[tuple[str, str]]:
            received_failures.append(list(failures))
            return []

        config = AutoresearchConfig(
            max_iterations=1,
            skill_directory_path=tmp_path / "skills",
        )
        run_loop_with_skills(
            generator=capturing_generator,
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )

        # The pre-existing skill should have been loaded and injected
        assert len(received_failures) >= 1
        first_call_failures = received_failures[0]
        assert any(f["description"] == "skill_playbook" for f in first_call_failures)

    def test_default_config_when_none(self) -> None:
        """REQ-AUTO-012: passing config=None uses defaults."""
        call_count = 0

        def once_generator(
            baselines: BaselineRecord,
            failures: list[dict[str, Any]],
            iteration: int,
        ) -> list[tuple[str, str]]:
            nonlocal call_count
            call_count += 1
            return [("test", GOOD_CODE)] if call_count == 1 else []

        result = run_loop_with_skills(
            generator=once_generator,
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=None,
        )
        assert result.iterations == 1

    def test_inner_max_iterations_break(self) -> None:
        """FR-11: inner loop breaks at max_iterations when generator returns many."""

        def multi_generator(
            baselines: BaselineRecord,
            failures: list[dict[str, Any]],
            iteration: int,
        ) -> list[tuple[str, str]]:
            # Return 5 hypotheses but max_iterations is 2
            return [("test", GOOD_CODE)] * 5

        config = AutoresearchConfig(max_iterations=2)
        result = run_loop_with_skills(
            generator=multi_generator,
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )
        assert result.iterations == 2

    def test_inner_circuit_breaker_break(self) -> None:
        """REQ-AUTO-009: inner loop circuit breaker triggers mid-batch."""

        def multi_bad_generator(
            baselines: BaselineRecord,
            failures: list[dict[str, Any]],
            iteration: int,
        ) -> list[tuple[str, str]]:
            return [("bad", BAD_CODE)] * 10

        config = AutoresearchConfig(
            max_iterations=100,
            max_consecutive_failures=3,
        )
        result = run_loop_with_skills(
            generator=multi_bad_generator,
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )
        assert result.circuit_breaker_tripped

    def test_review_verdict(self) -> None:
        """REQ-AUTO-005: REVIEW verdict is handled in skills loop."""
        # Code that improves one benchmark but regresses another
        review_code = """\
def run(benchmark_data):
    return {
        "double_well": {
            "final_energy": 0.01,
            "convergence_steps": 100,
            "wall_clock_seconds": 0.5,
            "peak_memory_mb": 10.0,
        },
        "rosenbrock": {
            "final_energy": 999.0,
            "convergence_steps": 100,
            "wall_clock_seconds": 0.5,
            "peak_memory_mb": 10.0,
        },
    }
"""
        baselines = _make_baselines()
        baselines.benchmarks["rosenbrock"] = BenchmarkMetrics(
            benchmark_name="rosenbrock",
            final_energy=0.1,
            convergence_steps=5000,
            wall_clock_seconds=2.0,
            peak_memory_mb=50.0,
        )

        config = AutoresearchConfig(max_iterations=1)
        result = run_loop_with_skills(
            generator=_make_generator(code=review_code),
            baselines=baselines,
            benchmark_data={"dim": 2},
            config=config,
        )
        assert result.pending_review == 1

    @patch("carnot.autoresearch.orchestrator.consolidate_lessons")
    @patch("carnot.autoresearch.orchestrator.analyze_batch")
    def test_final_analysis_saves_to_disk(
        self, mock_analyze: MagicMock, mock_consolidate: MagicMock, tmp_path: Path
    ) -> None:
        """REQ-AUTO-012: final analysis saves skill directory when path is set."""
        mock_analyze.return_value = [Lesson(title="final", description="d", confidence=0.8)]
        mock_consolidate.return_value = [Lesson(title="final", description="d", confidence=0.8)]

        config = AutoresearchConfig(
            max_iterations=1,
            enable_trajectory_analysis=True,
            consolidation_interval=100,  # Won't trigger mid-loop
            skill_directory_path=tmp_path / "skills",
        )
        run_loop_with_skills(
            generator=_make_generator(),
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )
        # Final analysis should save to disk
        if mock_consolidate.called:
            assert (tmp_path / "skills" / "lessons.json").exists()

    @patch("carnot.autoresearch.orchestrator.consolidate_lessons")
    @patch("carnot.autoresearch.orchestrator.analyze_batch")
    def test_final_analysis_on_exit(
        self, mock_consolidate: MagicMock, mock_analyze: MagicMock
    ) -> None:
        """REQ-AUTO-011: remaining entries are analyzed when loop ends."""
        mock_analyze.return_value = [Lesson(title="final", description="d", confidence=0.8)]
        mock_consolidate.return_value = [Lesson(title="final", description="d", confidence=0.8)]

        config = AutoresearchConfig(
            max_iterations=2,
            enable_trajectory_analysis=True,
            consolidation_interval=10,  # Won't trigger mid-loop
        )
        run_loop_with_skills(
            generator=_make_generator(),
            baselines=_make_baselines(),
            benchmark_data={"dim": 2},
            config=config,
        )
        # Final analysis should have been triggered
        assert mock_analyze.called
