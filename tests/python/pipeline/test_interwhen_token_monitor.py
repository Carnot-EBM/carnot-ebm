"""Tests for InterwhenTokenMonitor — async token-level PySAT constraint polling.

Verifies that the monitor correctly interrupts generation on constraint violations,
avoids false positives on compliant responses, and accurately tracks compute
avoided metrics.

Spec: REQ-VERIFY-175, SCENARIO-VERIFY-175
"""

from __future__ import annotations

import pytest

from carnot.pipeline.interwhen_token_monitor import (
    InterwhenTokenMonitor,
    TokenMonitorResult,
    _check_constraint_violated,
    _committed_numbers,
    _count_tool_calls,
    _count_words,
    _has_bold_markdown,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def length_constraint() -> dict:
    """CCTU length constraint: max 10 words."""
    return {
        "id": "test-length-001",
        "type": "length",
        "description": "Max 10 words.",
        "validator": {"name": "word_count", "max": 10},
    }


@pytest.fixture()
def numeric_constraint() -> dict:
    """CCTU numeric constraint: score in [10, 100]."""
    return {
        "id": "test-numeric-001",
        "type": "numeric",
        "description": "Score in [10, 100].",
        "validator": {"name": "numeric_score_range", "min": 10, "max": 100},
    }


@pytest.fixture()
def tool_constraint() -> dict:
    """CCTU resource constraint: exactly 1 tool call."""
    return {
        "id": "test-tool-001",
        "type": "resource",
        "description": "Exactly 1 tool call.",
        "validator": {"name": "tool_call_protocol", "count": 1, "required_tool": "calculate"},
    }


@pytest.fixture()
def format_constraint() -> dict:
    """CCTU format constraint: markdown bold required."""
    return {
        "id": "test-format-001",
        "type": "format",
        "description": "Markdown bold required.",
        "validator": {"name": "format_style", "style": "markdown_bold"},
    }


@pytest.fixture()
def monitor_length_only(length_constraint) -> InterwhenTokenMonitor:
    """Monitor with only the length constraint, polling every 5 tokens."""
    return InterwhenTokenMonitor(poll_every_n=5, constraints=[length_constraint])


@pytest.fixture()
def monitor_numeric_only(numeric_constraint) -> InterwhenTokenMonitor:
    """Monitor with only the numeric constraint, polling every 3 tokens."""
    return InterwhenTokenMonitor(poll_every_n=3, constraints=[numeric_constraint])


@pytest.fixture()
def monitor_multi(length_constraint, tool_constraint) -> InterwhenTokenMonitor:
    """Monitor with length + tool constraints, polling every 5 tokens."""
    return InterwhenTokenMonitor(
        poll_every_n=5, constraints=[length_constraint, tool_constraint]
    )


# ---------------------------------------------------------------------------
# Helper functions — REQ-VERIFY-175 unit tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for the pure-function helpers used by _check_pysat."""

    def test_count_words_basic(self) -> None:
        """_count_words returns correct word count from token list."""
        # REQ-VERIFY-175-3: length check relies on word counting
        tokens = ["hello", "world", "foo"]
        assert _count_words(tokens) == 3

    def test_count_words_empty(self) -> None:
        """_count_words returns 0 for empty token list."""
        assert _count_words([]) == 0

    def test_count_words_single(self) -> None:
        """_count_words returns 1 for a single token."""
        assert _count_words(["hello"]) == 1

    def test_count_tool_calls_detects_xml_marker(self) -> None:
        """_count_tool_calls finds <tool_call> XML markers."""
        # REQ-VERIFY-175-3: resource constraint checking
        tokens = ["Here", "is", "<tool_call>", "calculate(2+2)", "</tool_call>"]
        assert _count_tool_calls(tokens) >= 1

    def test_count_tool_calls_zero_when_absent(self) -> None:
        """_count_tool_calls returns 0 when no tool calls present."""
        tokens = ["The", "answer", "is", "42"]
        assert _count_tool_calls(tokens) == 0

    def test_has_bold_markdown_true(self) -> None:
        """_has_bold_markdown detects ** bold markers."""
        tokens = ["The", "**answer**", "is", "42"]
        assert _has_bold_markdown(tokens) is True

    def test_has_bold_markdown_false(self) -> None:
        """_has_bold_markdown returns False when no bold present."""
        tokens = ["The", "answer", "is", "42"]
        assert _has_bold_markdown(tokens) is False

    def test_committed_numbers_extracts_integers(self) -> None:
        """_committed_numbers extracts integer values from tokens."""
        tokens = ["score", "is", "75"]
        numbers = _committed_numbers(tokens)
        assert 75.0 in numbers

    def test_committed_numbers_empty_when_no_numbers(self) -> None:
        """_committed_numbers returns empty list when no numerics present."""
        tokens = ["hello", "world"]
        assert _committed_numbers(tokens) == []


# ---------------------------------------------------------------------------
# _check_constraint_violated — REQ-VERIFY-175-3
# ---------------------------------------------------------------------------


class TestCheckConstraintViolated:
    """Tests for the per-constraint violation detection helper."""

    def test_length_not_violated_under_limit(self, length_constraint) -> None:
        """Length constraint is NOT violated when word count is under max."""
        # REQ-VERIFY-175-3: length violation only when count > max
        tokens = "one two three".split()
        assert _check_constraint_violated(length_constraint, tokens) is False

    def test_length_violated_over_limit(self, length_constraint) -> None:
        """Length constraint IS violated when word count exceeds max=10."""
        tokens = "one two three four five six seven eight nine ten eleven".split()
        assert _check_constraint_violated(length_constraint, tokens) is True

    def test_length_not_violated_at_exact_limit(self, length_constraint) -> None:
        """Length constraint is NOT violated when count equals max (boundary)."""
        tokens = "one two three four five six seven eight nine ten".split()
        assert _check_constraint_violated(length_constraint, tokens) is False

    def test_numeric_not_violated_within_range(self, numeric_constraint) -> None:
        """Numeric constraint NOT violated when number in [10, 100]."""
        tokens = ["The", "score", "is", "50"]
        assert _check_constraint_violated(numeric_constraint, tokens) is False

    def test_numeric_violated_below_min(self, numeric_constraint) -> None:
        """Numeric constraint IS violated when number < min=10."""
        tokens = ["The", "score", "is", "5"]
        assert _check_constraint_violated(numeric_constraint, tokens) is True

    def test_numeric_violated_above_max(self, numeric_constraint) -> None:
        """Numeric constraint IS violated when number > max=100."""
        tokens = ["The", "score", "is", "200"]
        assert _check_constraint_violated(numeric_constraint, tokens) is True

    def test_tool_not_violated_when_within_count(self, tool_constraint) -> None:
        """Tool constraint NOT violated when tool calls <= allowed count."""
        tokens = ["Use", "<tool_call>", "calculate(1+1)"]
        # 1 tool call, allowed count=1 → not violated
        assert _check_constraint_violated(tool_constraint, tokens) is False

    def test_tool_violated_when_over_count(self, tool_constraint) -> None:
        """Tool constraint IS violated when tool calls exceed allowed count=1."""
        tokens = ["<tool_call>", "foo", "<tool_call>", "bar"]
        assert _check_constraint_violated(tool_constraint, tokens) is True

    def test_format_not_violated_by_partial_trace(self, format_constraint) -> None:
        """Format constraint is NEVER violated mid-generation (deferred check)."""
        # Format requires seeing the full response; partial trace never triggers it.
        tokens = ["No", "bold", "yet"]
        assert _check_constraint_violated(format_constraint, tokens) is False


# ---------------------------------------------------------------------------
# InterwhenTokenMonitor — REQ-VERIFY-175-1, 175-2, 175-4
# ---------------------------------------------------------------------------


class TestInterwhenTokenMonitor:
    """Tests for the main InterwhenTokenMonitor class."""

    def test_no_interrupt_on_compliant_short_response(
        self, monitor_length_only
    ) -> None:
        """Compliant response (5 words) must NOT be interrupted.

        SCENARIO-VERIFY-175: correct responses → interrupted = False.
        REQ-VERIFY-175-6: zero false accepts.
        """
        tokens = InterwhenTokenMonitor.tokenize_response("The answer is forty two")
        result = monitor_length_only.monitor_generation(tokens)
        assert result.interrupted is False
        assert result.tokens_avoided == 0
        assert result.compute_avoided_pct == 0.0

    def test_interrupt_on_overlength_response(self, monitor_length_only) -> None:
        """Overlength response (20 words > max=10) MUST be interrupted mid-sequence.

        SCENARIO-VERIFY-175: violated constraint → interrupted = True, tokens_avoided > 0.
        REQ-VERIFY-175-2: monitor_generation returns TokenMonitorResult with interruption.

        We use 20 words with poll_every_n=5: violation is detected at token-15 (idx=14),
        leaving 5 ungenerated tokens → tokens_avoided > 0.
        """
        response = (
            "one two three four five six seven eight nine ten "
            "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
        )
        tokens = InterwhenTokenMonitor.tokenize_response(response)
        assert len(tokens) == 20, "fixture must be 20 tokens"
        result = monitor_length_only.monitor_generation(tokens)
        assert result.interrupted is True
        assert result.tokens_avoided > 0
        assert result.compute_avoided_pct > 0.0
        assert "test-length-001" in result.violations_detected

    def test_interrupt_sets_interrupt_token_idx(self, monitor_length_only) -> None:
        """interrupt_token_idx is set to a valid index when interrupted.

        REQ-VERIFY-175-4: TokenMonitorResult.interrupt_token_idx is non-None when interrupted.
        """
        response = " ".join(["word"] * 20)
        tokens = InterwhenTokenMonitor.tokenize_response(response)
        result = monitor_length_only.monitor_generation(tokens)
        assert result.interrupt_token_idx is not None
        assert 0 <= result.interrupt_token_idx < result.tokens_total

    def test_no_interrupt_on_compliant_numeric_response(
        self, monitor_numeric_only
    ) -> None:
        """Response with score in [10, 100] must NOT be interrupted."""
        tokens = InterwhenTokenMonitor.tokenize_response("The score is 75 points total")
        result = monitor_numeric_only.monitor_generation(tokens)
        assert result.interrupted is False

    def test_interrupt_on_out_of_range_numeric(self, monitor_numeric_only) -> None:
        """Response with out-of-range score (5 < min=10) MUST be interrupted."""
        tokens = InterwhenTokenMonitor.tokenize_response("The score is 5 and done")
        result = monitor_numeric_only.monitor_generation(tokens)
        assert result.interrupted is True
        assert "test-numeric-001" in result.violations_detected

    def test_tokens_total_equals_sequence_length(self, monitor_length_only) -> None:
        """tokens_total always equals the full input sequence length.

        REQ-VERIFY-175-4: tokens_total is the full sequence, not the generated subset.
        """
        response = "a b c d e f g h i j k"
        tokens = InterwhenTokenMonitor.tokenize_response(response)
        result = monitor_length_only.monitor_generation(tokens)
        assert result.tokens_total == len(tokens)

    def test_compute_avoided_pct_in_range(self, monitor_length_only) -> None:
        """compute_avoided_pct is always in [0, 100]."""
        for word_count in [5, 15, 25]:
            response = " ".join(["word"] * word_count)
            tokens = InterwhenTokenMonitor.tokenize_response(response)
            result = monitor_length_only.monitor_generation(tokens)
            assert 0.0 <= result.compute_avoided_pct <= 100.0

    def test_pysat_checks_run_is_positive(self, monitor_length_only) -> None:
        """At least one PySAT check is always performed.

        REQ-VERIFY-175-3: _check_pysat is invoked at every poll boundary.
        """
        tokens = InterwhenTokenMonitor.tokenize_response("hello world")
        result = monitor_length_only.monitor_generation(tokens)
        assert result.pysat_checks_run >= 1

    def test_tokens_generated_plus_avoided_equals_total(
        self, monitor_length_only
    ) -> None:
        """tokens_generated + tokens_avoided == tokens_total always holds."""
        response = " ".join(["word"] * 20)
        tokens = InterwhenTokenMonitor.tokenize_response(response)
        result = monitor_length_only.monitor_generation(tokens)
        assert result.tokens_generated + result.tokens_avoided == result.tokens_total

    def test_no_interrupt_empty_sequence(self, monitor_length_only) -> None:
        """Empty token sequence produces no interrupt and 0 avoided compute."""
        result = monitor_length_only.monitor_generation([])
        assert result.interrupted is False
        assert result.tokens_avoided == 0
        assert result.compute_avoided_pct == 0.0

    def test_multi_constraint_no_interrupt_compliant(self, monitor_multi) -> None:
        """Compliant response (short, 0 extra tool calls) is not interrupted."""
        response = "The result is 42 from a single calculation"
        tokens = InterwhenTokenMonitor.tokenize_response(response)
        result = monitor_multi.monitor_generation(tokens)
        assert result.interrupted is False

    def test_violations_detected_empty_on_no_interrupt(
        self, monitor_length_only
    ) -> None:
        """violations_detected is empty when no interrupt occurs."""
        response = "short answer"
        tokens = InterwhenTokenMonitor.tokenize_response(response)
        result = monitor_length_only.monitor_generation(tokens)
        assert result.violations_detected == []

    def test_tokenize_response_splits_whitespace(self) -> None:
        """tokenize_response splits on whitespace and drops empty strings."""
        tokens = InterwhenTokenMonitor.tokenize_response("  hello   world  ")
        assert tokens == ["hello", "world"]

    def test_tokenize_response_empty_string(self) -> None:
        """tokenize_response returns empty list for empty string."""
        assert InterwhenTokenMonitor.tokenize_response("") == []

    def test_result_is_token_monitor_result_instance(
        self, monitor_length_only
    ) -> None:
        """monitor_generation always returns a TokenMonitorResult."""
        result = monitor_length_only.monitor_generation(["a", "b"])
        assert isinstance(result, TokenMonitorResult)
