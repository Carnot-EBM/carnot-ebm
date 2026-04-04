"""Tests for LLM hypothesis generator and generator-based orchestrator loop.

Spec coverage: FR-11, REQ-AUTO-003, REQ-AUTO-009
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
from carnot.autoresearch.hypothesis_generator import (
    GeneratorConfig,
    _build_user_prompt,
    _extract_hypotheses,
    generate_hypotheses,
    generate_hypotheses_batch,
)
from carnot.autoresearch.orchestrator import (
    AutoresearchConfig,
    run_loop_with_generator,
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


# ---------------------------------------------------------------------------
# Tests: _build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    """Tests for prompt construction."""

    def test_basic_prompt(self) -> None:
        """REQ-AUTO-003: prompt includes baseline info."""
        baselines = _make_baselines()
        prompt = _build_user_prompt(baselines)
        assert "double_well" in prompt
        assert "0.050000" in prompt

    def test_with_failures(self) -> None:
        """REQ-AUTO-003: prompt includes failure context."""
        baselines = _make_baselines()
        failures = [{"description": "bad idea", "reason": "energy regressed"}]
        prompt = _build_user_prompt(baselines, recent_failures=failures)
        assert "bad idea" in prompt
        assert "energy regressed" in prompt

    def test_iteration_zero(self) -> None:
        """REQ-AUTO-003: first iteration has conservative guidance."""
        prompt = _build_user_prompt(_make_baselines(), iteration=0)
        assert "first iteration" in prompt.lower()

    def test_later_iteration(self) -> None:
        """REQ-AUTO-003: later iterations encourage creativity."""
        prompt = _build_user_prompt(_make_baselines(), iteration=10)
        assert "creative" in prompt.lower()

    def test_extra_context(self) -> None:
        """REQ-AUTO-003: extra context is included."""
        prompt = _build_user_prompt(
            _make_baselines(), extra_context="Focus on HMC sampling."
        )
        assert "Focus on HMC sampling" in prompt

    def test_empty_baselines(self) -> None:
        """REQ-AUTO-003: handles empty baselines."""
        baselines = BaselineRecord(version="test")
        prompt = _build_user_prompt(baselines)
        assert "No baselines" in prompt


# ---------------------------------------------------------------------------
# Tests: _extract_hypotheses
# ---------------------------------------------------------------------------


class TestExtractHypotheses:
    """Tests for code extraction from LLM responses."""

    def test_single_python_block(self) -> None:
        """REQ-AUTO-003: extracts code from python code block."""
        response = """Here's my hypothesis:

Smaller step size for better convergence.

```python
def run(benchmark_data):
    return {"double_well": {"final_energy": 0.01}}
```
"""
        hypotheses = _extract_hypotheses(response)
        assert len(hypotheses) == 1
        assert "def run(" in hypotheses[0][1]
        assert "Smaller step size" in hypotheses[0][0]

    def test_no_code_blocks(self) -> None:
        """REQ-AUTO-003: returns empty list when no code blocks."""
        hypotheses = _extract_hypotheses("I think we should try something.")
        assert hypotheses == []

    def test_code_without_run(self) -> None:
        """REQ-AUTO-003: skips code blocks without run function."""
        response = """```python
x = 42
```"""
        hypotheses = _extract_hypotheses(response)
        assert hypotheses == []

    def test_multiple_blocks(self) -> None:
        """REQ-AUTO-003: extracts multiple hypotheses."""
        response = """First idea:

```python
def run(benchmark_data):
    return {"a": {"final_energy": 0.01}}
```

Second idea:

```python
def run(benchmark_data):
    return {"b": {"final_energy": 0.02}}
```
"""
        hypotheses = _extract_hypotheses(response)
        assert len(hypotheses) == 2

    def test_generic_code_block(self) -> None:
        """REQ-AUTO-003: extracts from generic (non-python) code blocks."""
        response = """Try this:

```
def run(benchmark_data):
    return {"x": {"final_energy": 0.1}}
```
"""
        hypotheses = _extract_hypotheses(response)
        assert len(hypotheses) == 1


# ---------------------------------------------------------------------------
# Tests: generate_hypotheses (with mocked OpenAI)
# ---------------------------------------------------------------------------


class TestGenerateHypotheses:
    """Tests for the full generation pipeline."""

    def test_missing_openai(self) -> None:
        """REQ-AUTO-003: graceful error when openai not installed."""
        config = GeneratorConfig()
        with patch.dict("sys.modules", {"openai": None}):
            result = generate_hypotheses(config, _make_baselines())
            assert result.error is not None
            assert "openai" in result.error.lower()

    def test_successful_generation(self) -> None:
        """REQ-AUTO-003: successful LLM call returns hypotheses."""
        config = GeneratorConfig()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
Try a smaller step size:

```python
def run(benchmark_data):
    return {"double_well": {"final_energy": 0.01}}
```
"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            result = generate_hypotheses(config, _make_baselines())
            assert result.error is None
            assert len(result.hypotheses) == 1
            assert "def run(" in result.hypotheses[0][1]

    def test_no_valid_code(self) -> None:
        """REQ-AUTO-003: error when LLM returns no valid code."""
        config = GeneratorConfig()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I think we should improve things."

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            result = generate_hypotheses(config, _make_baselines())
            assert result.error is not None
            assert "no valid" in result.error.lower()

    def test_api_error(self) -> None:
        """REQ-AUTO-003: API error returns error result."""
        config = GeneratorConfig()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("timeout")

        with patch("openai.OpenAI", return_value=mock_client):
            result = generate_hypotheses(config, _make_baselines())
            assert result.error is not None
            assert "timeout" in result.error


class TestGenerateHypothesesBatch:
    """Tests for batch generation."""

    def test_batch_collects_results(self) -> None:
        """REQ-AUTO-003: batch returns multiple hypotheses."""
        config = GeneratorConfig()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
```python
def run(benchmark_data):
    return {"x": {"final_energy": 0.01}}
```
"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            hypotheses = generate_hypotheses_batch(
                config, _make_baselines(), count=2
            )
            assert len(hypotheses) == 2

    def test_batch_handles_failures(self) -> None:
        """REQ-AUTO-003: batch continues on individual failures."""
        config = GeneratorConfig()

        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("first call fails")
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = """
```python
def run(benchmark_data):
    return {"x": {"final_energy": 0.01}}
```
"""
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = mock_create

        with patch("openai.OpenAI", return_value=mock_client):
            hypotheses = generate_hypotheses_batch(
                config, _make_baselines(), count=2
            )
            # First call fails, second succeeds
            assert len(hypotheses) == 1


# ---------------------------------------------------------------------------
# Tests: run_loop_with_generator
# ---------------------------------------------------------------------------


class TestRunLoopWithGenerator:
    """Tests for the generator-based orchestrator loop."""

    def test_accepts_good_hypothesis(self) -> None:
        """REQ-AUTO-005: loop accepts hypothesis that improves energy."""
        baselines = _make_baselines()

        def generator(bl, failures, iteration):
            return [(
                "better step size",
                """
def run(benchmark_data):
    return {"double_well": {"final_energy": 0.01, "wall_clock_seconds": 1.0}}
""",
            )]

        config = AutoresearchConfig(max_iterations=1)
        result = run_loop_with_generator(generator, baselines, {}, config)
        assert result.accepted == 1

    def test_rejects_bad_hypothesis(self) -> None:
        """REQ-AUTO-005: loop rejects hypothesis that regresses energy."""
        baselines = _make_baselines()

        def generator(bl, failures, iteration):
            return [(
                "worse step size",
                """
def run(benchmark_data):
    return {"double_well": {"final_energy": 10.0, "wall_clock_seconds": 1.0}}
""",
            )]

        config = AutoresearchConfig(max_iterations=1)
        result = run_loop_with_generator(generator, baselines, {}, config)
        assert result.rejected == 1

    def test_empty_generator_stops(self) -> None:
        """REQ-AUTO-009: loop stops when generator returns no hypotheses."""
        baselines = _make_baselines()

        def generator(bl, failures, iteration):
            return []

        config = AutoresearchConfig(max_iterations=10)
        result = run_loop_with_generator(generator, baselines, {}, config)
        assert result.iterations == 0

    def test_circuit_breaker(self) -> None:
        """REQ-AUTO-009: circuit breaker trips after consecutive failures."""
        baselines = _make_baselines()

        def generator(bl, failures, iteration):
            return [(
                f"crash hypothesis {iteration}",
                "def run(benchmark_data):\n    raise ValueError('boom')",
            )]

        config = AutoresearchConfig(
            max_iterations=20,
            max_consecutive_failures=3,
        )
        result = run_loop_with_generator(generator, baselines, {}, config)
        assert result.circuit_breaker_tripped

    def test_generator_exception(self) -> None:
        """REQ-AUTO-009: loop handles generator exceptions gracefully."""
        baselines = _make_baselines()
        call_count = 0

        def generator(bl, failures, iteration):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("generator broken")
            return []

        config = AutoresearchConfig(max_iterations=5)
        result = run_loop_with_generator(generator, baselines, {}, config)
        # Should not crash — just continue
        assert result.iterations == 0

    def test_default_config(self) -> None:
        """REQ-AUTO-005: loop works with default config (None)."""
        baselines = _make_baselines()

        def generator(bl, failures, iteration):
            return []

        result = run_loop_with_generator(generator, baselines, {})
        assert result.iterations == 0

    def test_max_iterations_inner_break(self) -> None:
        """REQ-AUTO-009: loop stops mid-batch when max_iterations reached."""
        baselines = _make_baselines()

        def generator(bl, failures, iteration):
            # Return more hypotheses than max_iterations allows
            code = (
                "def run(d):\n"
                '    return {"double_well": '
                '{"final_energy": 99.0, "wall_clock_seconds": 1.0}}'
            )
            return [("h1", code), ("h2", code), ("h3", code)]

        config = AutoresearchConfig(max_iterations=2, max_consecutive_failures=100)
        result = run_loop_with_generator(generator, baselines, {}, config)
        assert result.iterations == 2

    def test_review_verdict(self) -> None:
        """REQ-AUTO-005: loop handles REVIEW verdict (mixed improvements/regressions)."""
        baselines = _make_baselines()
        # Add a second benchmark so we can have mixed results
        baselines.benchmarks["rosenbrock"] = BenchmarkMetrics(
            benchmark_name="rosenbrock",
            final_energy=0.5,
            convergence_steps=10000,
            wall_clock_seconds=5.0,
        )

        def generator(bl, failures, iteration):
            # Improve one benchmark, regress the other -> REVIEW
            return [(
                "mixed results",
                """
def run(benchmark_data):
    return {
        "double_well": {"final_energy": 0.01, "wall_clock_seconds": 1.0},
        "rosenbrock": {"final_energy": 99.0, "wall_clock_seconds": 1.0},
    }
""",
            )]

        config = AutoresearchConfig(max_iterations=1)
        result = run_loop_with_generator(generator, baselines, {}, config)
        assert result.pending_review == 1

    def test_circuit_breaker_inner(self) -> None:
        """REQ-AUTO-009: circuit breaker trips mid-batch."""
        baselines = _make_baselines()

        call_count = 0

        def generator(bl, failures, iteration):
            nonlocal call_count
            call_count += 1
            # Return many crash hypotheses in one batch
            return [
                (f"crash {i}", "def run(d):\n    raise ValueError('boom')")
                for i in range(5)
            ]

        config = AutoresearchConfig(
            max_iterations=20,
            max_consecutive_failures=3,
        )
        result = run_loop_with_generator(generator, baselines, {}, config)
        assert result.circuit_breaker_tripped

    def test_feedback_loop(self) -> None:
        """REQ-AUTO-003: failures are fed back to generator."""
        baselines = _make_baselines()
        received_failures: list[list] = []

        def generator(bl, failures, iteration):
            received_failures.append(list(failures))
            if iteration == 0:
                return [(
                    "bad idea",
                    """
def run(benchmark_data):
    return {"double_well": {"final_energy": 99.0, "wall_clock_seconds": 1.0}}
""",
                )]
            return []

        config = AutoresearchConfig(max_iterations=5)
        run_loop_with_generator(generator, baselines, {}, config)

        # Second call should have received the failure from the first
        assert len(received_failures) >= 2
        if len(received_failures) > 1:
            assert len(received_failures[1]) > 0
            assert "bad idea" in received_failures[1][0]["description"]
