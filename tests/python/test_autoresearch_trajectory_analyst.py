"""Tests for trajectory analyst: error/success analysis and lesson extraction.

Spec coverage: REQ-AUTO-011, SCENARIO-AUTO-008, SCENARIO-AUTO-009
"""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
from carnot.autoresearch.experiment_log import ExperimentEntry
from carnot.autoresearch.trajectory_analyst import (
    AnalystConfig,
    Lesson,
    _build_analyst_context,
    _parse_lesson_json,
    analyze_batch,
    analyze_error,
    analyze_success,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_baselines() -> BaselineRecord:
    record = BaselineRecord(version="test")
    record.benchmarks["double_well"] = BenchmarkMetrics(
        benchmark_name="double_well",
        final_energy=0.05,
        convergence_steps=5000,
        wall_clock_seconds=2.0,
    )
    return record


def _make_failed_entry() -> ExperimentEntry:
    return ExperimentEntry(
        id="test-fail-001",
        timestamp="2026-04-04T00:00:00Z",
        hypothesis_code="def run(d): return {'double_well': {'final_energy': float('nan')}}",
        hypothesis_description="Large step size causes NaN",
        sandbox_success=False,
        sandbox_metrics={"double_well": {"final_energy": float("nan")}},
        sandbox_error="NaN energy detected",
        eval_verdict="FAIL",
        eval_reason="Energy regression on: double_well",
        eval_regressions=["double_well"],
        outcome="rejected",
    )


def _make_accepted_entry() -> ExperimentEntry:
    return ExperimentEntry(
        id="test-pass-001",
        timestamp="2026-04-04T00:00:00Z",
        hypothesis_code="def run(d): return {'double_well': {'final_energy': 0.01}}",
        hypothesis_description="Annealing schedule improved convergence",
        sandbox_success=True,
        sandbox_metrics={"double_well": {"final_energy": 0.01, "wall_clock_seconds": 1.5}},
        eval_verdict="PASS",
        eval_reason="All benchmarks improved",
        eval_improvements=["double_well"],
        outcome="accepted",
    )


def _make_config() -> AnalystConfig:
    return AnalystConfig(api_base="http://test:8080/v1", model="test-model")


# ---------------------------------------------------------------------------
# Tests: Lesson dataclass
# ---------------------------------------------------------------------------


class TestLesson:
    """Tests for the Lesson data type."""

    def test_lesson_defaults(self) -> None:
        """REQ-AUTO-011: Lesson has sensible defaults."""
        lesson = Lesson(title="test", description="desc")
        assert lesson.confidence == 0.5
        assert lesson.applicable_benchmarks == ["all"]
        assert lesson.model_tier == "all"
        assert lesson.lesson_type == "error_pattern"

    def test_lesson_to_dict(self) -> None:
        """REQ-AUTO-011: Lesson serializes to dict."""
        lesson = Lesson(title="test", description="desc", confidence=0.9)
        d = lesson.to_dict()
        assert d["title"] == "test"
        assert d["confidence"] == 0.9

    def test_lesson_from_dict(self) -> None:
        """REQ-AUTO-011: Lesson deserializes from dict."""
        d = {"title": "test", "description": "desc", "confidence": 0.8}
        lesson = Lesson.from_dict(d)
        assert lesson.title == "test"
        assert lesson.confidence == 0.8

    def test_lesson_from_dict_ignores_unknown_keys(self) -> None:
        """REQ-AUTO-011: from_dict handles extra keys gracefully."""
        d = {"title": "test", "description": "desc", "unknown_field": "ignored"}
        lesson = Lesson.from_dict(d)
        assert lesson.title == "test"

    def test_lesson_roundtrip(self) -> None:
        """REQ-AUTO-011: to_dict -> from_dict roundtrip is lossless."""
        original = Lesson(
            title="gradient explosion",
            description="Step size too large for curvature",
            examples=["exp-001"],
            confidence=0.9,
            applicable_benchmarks=["rosenbrock"],
            model_tier="ising",
            lesson_type="error_pattern",
            source_experiment_id="exp-001",
        )
        restored = Lesson.from_dict(original.to_dict())
        assert restored.title == original.title
        assert restored.confidence == original.confidence
        assert restored.applicable_benchmarks == original.applicable_benchmarks

    def test_lesson_confidence_range(self) -> None:
        """REQ-AUTO-011: confidence can be set to boundary values."""
        low = Lesson(title="t", description="d", confidence=0.0)
        high = Lesson(title="t", description="d", confidence=1.0)
        assert low.confidence == 0.0
        assert high.confidence == 1.0


# ---------------------------------------------------------------------------
# Tests: _parse_lesson_json
# ---------------------------------------------------------------------------


class TestParseLessonJson:
    """Tests for JSON extraction from LLM responses."""

    def test_direct_json(self) -> None:
        """REQ-AUTO-011: parses direct JSON."""
        raw = '{"title": "test", "description": "desc", "confidence": 0.7}'
        result = _parse_lesson_json(raw)
        assert result is not None
        assert result["title"] == "test"

    def test_code_block_json(self) -> None:
        """REQ-AUTO-011: extracts JSON from code block."""
        raw = 'Some text\n```json\n{"title": "test"}\n```\nMore text'
        result = _parse_lesson_json(raw)
        assert result is not None
        assert result["title"] == "test"

    def test_embedded_json(self) -> None:
        """REQ-AUTO-011: extracts JSON from surrounding text."""
        raw = 'Here is the result: {"title": "test"} done.'
        result = _parse_lesson_json(raw)
        assert result is not None
        assert result["title"] == "test"

    def test_invalid_json_returns_none(self) -> None:
        """REQ-AUTO-011: returns None for unparseable responses."""
        assert _parse_lesson_json("no json here") is None
        assert _parse_lesson_json("") is None

    def test_code_block_with_invalid_json(self) -> None:
        """REQ-AUTO-011: code block with invalid JSON falls through."""
        raw = "```json\nnot valid json\n```"
        # Falls through to embedded search, which also fails
        assert _parse_lesson_json(raw) is None

    def test_embedded_invalid_json(self) -> None:
        """REQ-AUTO-011: embedded braces with invalid JSON falls through."""
        raw = "result: {not: valid: json} done"
        assert _parse_lesson_json(raw) is None


# ---------------------------------------------------------------------------
# Tests: _build_analyst_context
# ---------------------------------------------------------------------------


class TestBuildAnalystContext:
    """Tests for context building."""

    def test_includes_code(self) -> None:
        """REQ-AUTO-011: context includes hypothesis code."""
        entry = _make_failed_entry()
        ctx = _build_analyst_context(entry, _make_baselines())
        assert "def run(d)" in ctx

    def test_includes_error(self) -> None:
        """REQ-AUTO-011: context includes sandbox error."""
        entry = _make_failed_entry()
        ctx = _build_analyst_context(entry, _make_baselines())
        assert "NaN energy" in ctx

    def test_includes_baselines(self) -> None:
        """REQ-AUTO-011: context includes baseline metrics."""
        ctx = _build_analyst_context(_make_failed_entry(), _make_baselines())
        assert "double_well" in ctx
        assert "0.050000" in ctx


# ---------------------------------------------------------------------------
# Tests: analyze_error
# ---------------------------------------------------------------------------


class TestAnalyzeError:
    """Tests for error analysis."""

    @patch("openai.OpenAI")
    def test_error_analyst_produces_lesson(self, mock_openai_cls: MagicMock) -> None:
        """REQ-AUTO-011, SCENARIO-AUTO-008: error analyst extracts lesson."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"title": "Gradient explosion on steep landscapes", '
                        '"description": "Step size exceeded curvature", '
                        '"applicable_benchmarks": ["rosenbrock"], "confidence": 0.8}'
                    )
                )
            ]
        )

        lesson = analyze_error(_make_config(), _make_failed_entry(), _make_baselines())
        assert lesson is not None
        assert lesson.title == "Gradient explosion on steep landscapes"
        assert lesson.lesson_type == "error_pattern"
        assert lesson.confidence == 0.8
        assert "rosenbrock" in lesson.applicable_benchmarks

    @patch("openai.OpenAI")
    def test_error_analyst_handles_api_failure(self, mock_openai_cls: MagicMock) -> None:
        """REQ-AUTO-011: returns None when API call fails."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        lesson = analyze_error(_make_config(), _make_failed_entry(), _make_baselines())
        assert lesson is None

    @patch("openai.OpenAI")
    def test_error_analyst_handles_bad_response(self, mock_openai_cls: MagicMock) -> None:
        """REQ-AUTO-011: returns None when response is unparseable."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="This is not JSON"))]
        )

        lesson = analyze_error(_make_config(), _make_failed_entry(), _make_baselines())
        assert lesson is None

    def test_error_analyst_missing_openai(self) -> None:
        """REQ-AUTO-011: returns None when openai not installed."""
        with patch.dict("sys.modules", {"openai": None}):
            # Force reimport to trigger ImportError path
            import importlib

            import carnot.autoresearch.trajectory_analyst as ta

            importlib.reload(ta)
            lesson = ta.analyze_error(_make_config(), _make_failed_entry(), _make_baselines())
            assert lesson is None
            importlib.reload(ta)  # Restore


# ---------------------------------------------------------------------------
# Tests: analyze_success
# ---------------------------------------------------------------------------


class TestAnalyzeSuccess:
    """Tests for success analysis."""

    @patch("openai.OpenAI")
    def test_success_analyst_extracts_pattern(self, mock_openai_cls: MagicMock) -> None:
        """REQ-AUTO-011, SCENARIO-AUTO-009: success analyst extracts pattern."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"title": "Step-size annealing for multi-basin landscapes", '
                        '"description": "Annealing prevents divergence", '
                        '"applicable_benchmarks": ["double_well"], "confidence": 0.9}'
                    )
                )
            ]
        )

        lesson = analyze_success(_make_config(), _make_accepted_entry(), _make_baselines())
        assert lesson is not None
        assert lesson.title == "Step-size annealing for multi-basin landscapes"
        assert lesson.lesson_type == "success_pattern"
        assert lesson.confidence == 0.9

    @patch("openai.OpenAI")
    def test_success_analyst_handles_api_failure(self, mock_openai_cls: MagicMock) -> None:
        """REQ-AUTO-011: returns None when API call fails."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        lesson = analyze_success(_make_config(), _make_accepted_entry(), _make_baselines())
        assert lesson is None

    @patch("openai.OpenAI")
    def test_success_analyst_handles_bad_response(self, mock_openai_cls: MagicMock) -> None:
        """REQ-AUTO-011: returns None when response is unparseable."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Not JSON at all"))]
        )
        lesson = analyze_success(_make_config(), _make_accepted_entry(), _make_baselines())
        assert lesson is None

    def test_success_analyst_missing_openai(self) -> None:
        """REQ-AUTO-011: returns None when openai not installed."""
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import carnot.autoresearch.trajectory_analyst as ta

            importlib.reload(ta)
            lesson = ta.analyze_success(_make_config(), _make_accepted_entry(), _make_baselines())
            assert lesson is None
            importlib.reload(ta)


# ---------------------------------------------------------------------------
# Tests: analyze_batch
# ---------------------------------------------------------------------------


class TestAnalyzeBatch:
    """Tests for parallel batch analysis."""

    def test_empty_batch(self) -> None:
        """REQ-AUTO-011: empty input returns empty output."""
        assert analyze_batch(_make_config(), [], _make_baselines()) == []

    @patch("carnot.autoresearch.trajectory_analyst.analyze_success")
    @patch("carnot.autoresearch.trajectory_analyst.analyze_error")
    def test_batch_routes_by_outcome(self, mock_error: MagicMock, mock_success: MagicMock) -> None:
        """REQ-AUTO-011: routes to correct analyst based on outcome."""
        mock_error.return_value = Lesson(
            title="err",
            description="d",
            lesson_type="error_pattern",
        )
        mock_success.return_value = Lesson(
            title="suc",
            description="d",
            lesson_type="success_pattern",
        )

        entries = [_make_failed_entry(), _make_accepted_entry()]

        # Patch ThreadPoolExecutor to run sequentially for deterministic test
        with patch("carnot.autoresearch.trajectory_analyst.ThreadPoolExecutor") as mock_pool:
            mock_executor = MagicMock()
            mock_pool.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)

            # Set up futures
            future_err = Future()
            future_err.set_result(
                Lesson(
                    title="err",
                    description="d",
                    lesson_type="error_pattern",
                )
            )
            future_suc = Future()
            future_suc.set_result(
                Lesson(
                    title="suc",
                    description="d",
                    lesson_type="success_pattern",
                )
            )

            mock_executor.submit.side_effect = [future_err, future_suc]

            lessons = analyze_batch(_make_config(), entries, _make_baselines())
            assert len(lessons) == 2

    @patch("carnot.autoresearch.trajectory_analyst.analyze_error")
    def test_batch_handles_none_results(self, mock_error: MagicMock) -> None:
        """REQ-AUTO-011: None results are filtered out."""
        mock_error.return_value = None
        entries = [_make_failed_entry()]

        with patch("carnot.autoresearch.trajectory_analyst.ThreadPoolExecutor") as mock_pool:
            mock_executor = MagicMock()
            mock_pool.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)

            future = Future()
            future.set_result(None)
            mock_executor.submit.return_value = future

            lessons = analyze_batch(_make_config(), entries, _make_baselines())
            assert len(lessons) == 0

    def test_batch_skips_unknown_outcome(self) -> None:
        """REQ-AUTO-011: entries with no outcome are skipped."""
        entry = _make_failed_entry()
        entry.outcome = ""

        with patch("carnot.autoresearch.trajectory_analyst.ThreadPoolExecutor") as mock_pool:
            mock_executor = MagicMock()
            mock_pool.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)

            lessons = analyze_batch(_make_config(), [entry], _make_baselines())
            assert len(lessons) == 0
            mock_executor.submit.assert_not_called()

    def test_batch_handles_future_exception(self) -> None:
        """REQ-AUTO-011: exceptions in future.result() are caught."""
        entries = [_make_failed_entry()]

        with patch("carnot.autoresearch.trajectory_analyst.ThreadPoolExecutor") as mock_pool:
            mock_executor = MagicMock()
            mock_pool.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)

            future = Future()
            future.set_exception(RuntimeError("Analyst crashed"))
            mock_executor.submit.return_value = future

            lessons = analyze_batch(_make_config(), entries, _make_baselines())
            assert len(lessons) == 0

    @patch("carnot.autoresearch.trajectory_analyst.analyze_error")
    def test_batch_handles_pending_review(self, mock_error: MagicMock) -> None:
        """REQ-AUTO-011: pending_review entries are analyzed as errors."""
        entry = _make_failed_entry()
        entry.outcome = "pending_review"

        with patch("carnot.autoresearch.trajectory_analyst.ThreadPoolExecutor") as mock_pool:
            mock_executor = MagicMock()
            mock_pool.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)

            future = Future()
            future.set_result(Lesson(title="review", description="d"))
            mock_executor.submit.return_value = future

            lessons = analyze_batch(_make_config(), [entry], _make_baselines())
            assert len(lessons) == 1
