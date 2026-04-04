"""Tests for hierarchical lesson consolidation.

Spec coverage: REQ-AUTO-013, SCENARIO-AUTO-010
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from carnot.autoresearch.consolidator import (
    ConsolidatorConfig,
    _merge_batch_with_llm,
    _parse_consolidated_lessons,
    consolidate_lessons,
)
from carnot.autoresearch.trajectory_analyst import Lesson

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_lesson(
    title: str = "test",
    confidence: float = 0.7,
    lesson_type: str = "success_pattern",
) -> Lesson:
    return Lesson(
        title=title,
        description=f"Description for {title}",
        confidence=confidence,
        lesson_type=lesson_type,
    )


def _make_config(**kwargs: object) -> ConsolidatorConfig:
    return ConsolidatorConfig(
        api_base="http://test:8080/v1",
        model="test-model",
        **kwargs,  # type: ignore[arg-type]
    )


def _mock_llm_response(lessons: list[dict]) -> MagicMock:  # type: ignore[type-arg]
    """Create a mock OpenAI response returning a JSON array of lessons."""
    return MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(lessons)))])


# ---------------------------------------------------------------------------
# Tests: _parse_consolidated_lessons
# ---------------------------------------------------------------------------


class TestParseConsolidatedLessons:
    """Tests for parsing LLM consolidation responses."""

    def test_direct_json_array(self) -> None:
        """REQ-AUTO-013: parses direct JSON array."""
        raw = json.dumps([{"title": "merged", "description": "d", "confidence": 0.8}])
        fallback = [_make_lesson()]
        result = _parse_consolidated_lessons(raw, fallback)
        assert len(result) == 1
        assert result[0].title == "merged"

    def test_code_block_json(self) -> None:
        """REQ-AUTO-013: extracts from code block."""
        raw = '```json\n[{"title": "merged", "description": "d"}]\n```'
        result = _parse_consolidated_lessons(raw, [_make_lesson()])
        assert len(result) == 1

    def test_embedded_array(self) -> None:
        """REQ-AUTO-013: extracts array from surrounding text."""
        raw = 'Result: [{"title": "merged", "description": "d"}] done'
        result = _parse_consolidated_lessons(raw, [_make_lesson()])
        assert len(result) == 1

    def test_invalid_returns_fallback(self) -> None:
        """REQ-AUTO-013: falls back to originals on parse failure."""
        fallback = [_make_lesson(title="original")]
        result = _parse_consolidated_lessons("garbage", fallback)
        assert len(result) == 1
        assert result[0].title == "original"

    def test_empty_array_returns_fallback(self) -> None:
        """REQ-AUTO-013: empty array falls back to originals."""
        fallback = [_make_lesson(title="original")]
        result = _parse_consolidated_lessons("[]", fallback)
        assert len(result) == 1
        assert result[0].title == "original"

    def test_non_list_returns_fallback(self) -> None:
        """REQ-AUTO-013: non-array JSON falls back to originals."""
        fallback = [_make_lesson(title="original")]
        result = _parse_consolidated_lessons('{"title": "not array"}', fallback)
        assert len(result) == 1
        assert result[0].title == "original"

    def test_code_block_with_invalid_json(self) -> None:
        """REQ-AUTO-013: code block with invalid JSON falls through."""
        fallback = [_make_lesson(title="original")]
        raw = "```json\nnot valid\n```"
        result = _parse_consolidated_lessons(raw, fallback)
        assert result[0].title == "original"

    def test_embedded_array_with_invalid_json(self) -> None:
        """REQ-AUTO-013: embedded array with invalid JSON falls back."""
        fallback = [_make_lesson(title="original")]
        raw = "result: [not valid json] done"
        result = _parse_consolidated_lessons(raw, fallback)
        assert result[0].title == "original"

    def test_array_with_non_dict_items(self) -> None:
        """REQ-AUTO-013: non-dict items in array are skipped."""
        raw = json.dumps([{"title": "good", "description": "d"}, "bad_item", 42])
        result = _parse_consolidated_lessons(raw, [_make_lesson()])
        assert len(result) == 1
        assert result[0].title == "good"


# ---------------------------------------------------------------------------
# Tests: _merge_batch_with_llm
# ---------------------------------------------------------------------------


class TestMergeBatch:
    """Tests for single-batch merging."""

    @patch("openai.OpenAI")
    def test_merge_produces_consolidated_lessons(self, mock_cls: MagicMock) -> None:
        """REQ-AUTO-013: LLM merge produces consolidated output."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            [{"title": "merged A+B", "description": "combined", "confidence": 0.9}]
        )

        batch = [_make_lesson(title="A"), _make_lesson(title="B")]
        result = _merge_batch_with_llm(_make_config(), batch)
        assert len(result) == 1
        assert result[0].title == "merged A+B"

    @patch("openai.OpenAI")
    def test_merge_handles_api_failure(self, mock_cls: MagicMock) -> None:
        """REQ-AUTO-013: API failure returns batch unchanged."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("down")

        batch = [_make_lesson(title="A")]
        result = _merge_batch_with_llm(_make_config(), batch)
        assert len(result) == 1
        assert result[0].title == "A"

    def test_merge_missing_openai(self) -> None:
        """REQ-AUTO-013: missing openai returns batch unchanged."""
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import carnot.autoresearch.consolidator as cons

            importlib.reload(cons)
            batch = [_make_lesson(title="A")]
            result = cons._merge_batch_with_llm(_make_config(), batch)
            assert len(result) == 1
            assert result[0].title == "A"
            importlib.reload(cons)  # Restore


# ---------------------------------------------------------------------------
# Tests: consolidate_lessons
# ---------------------------------------------------------------------------


class TestConsolidateLessons:
    """Tests for the full consolidation pipeline."""

    def test_empty_list(self) -> None:
        """REQ-AUTO-013: empty input returns empty."""
        assert consolidate_lessons(_make_config(), []) == []

    def test_single_lesson_above_threshold(self) -> None:
        """REQ-AUTO-013: single lesson above threshold is returned."""
        lesson = _make_lesson(confidence=0.8)
        result = consolidate_lessons(_make_config(min_confidence=0.3), [lesson])
        assert len(result) == 1
        assert result[0].title == "test"

    def test_single_lesson_below_threshold(self) -> None:
        """REQ-AUTO-013: single lesson below threshold is filtered."""
        lesson = _make_lesson(confidence=0.1)
        result = consolidate_lessons(_make_config(min_confidence=0.5), [lesson])
        assert len(result) == 0

    @patch("carnot.autoresearch.consolidator._merge_batch_with_llm")
    def test_small_batch_single_merge(self, mock_merge: MagicMock) -> None:
        """REQ-AUTO-013: lessons within batch_size get one merge pass."""
        mock_merge.return_value = [_make_lesson(title="merged", confidence=0.9)]

        lessons = [_make_lesson(title=f"l{i}") for i in range(5)]
        config = _make_config(batch_size=32, min_confidence=0.3)
        result = consolidate_lessons(config, lessons)

        mock_merge.assert_called_once()
        assert len(result) == 1
        assert result[0].title == "merged"

    @patch("carnot.autoresearch.consolidator._merge_batch_with_llm")
    def test_large_batch_multiple_levels(self, mock_merge: MagicMock) -> None:
        """REQ-AUTO-013, SCENARIO-AUTO-010: tree reduction across levels."""
        # First level: 2 batches of 3 -> 2 merged lessons
        # Second level: 1 batch of 2 -> 1 final lesson
        call_count = 0

        def side_effect(config: ConsolidatorConfig, batch: list[Lesson]) -> list[Lesson]:
            nonlocal call_count
            call_count += 1
            return [_make_lesson(title=f"merged-{call_count}", confidence=0.8)]

        mock_merge.side_effect = side_effect

        lessons = [_make_lesson(title=f"l{i}") for i in range(6)]
        config = _make_config(batch_size=3, min_confidence=0.3)
        result = consolidate_lessons(config, lessons)

        assert call_count >= 2  # At least 2 merge passes
        assert len(result) >= 1

    @patch("carnot.autoresearch.consolidator._merge_batch_with_llm")
    def test_confidence_filter_applied(self, mock_merge: MagicMock) -> None:
        """REQ-AUTO-013: post-merge confidence filter removes weak lessons."""
        mock_merge.return_value = [
            _make_lesson(title="strong", confidence=0.9),
            _make_lesson(title="weak", confidence=0.1),
        ]

        lessons = [_make_lesson() for _ in range(3)]
        config = _make_config(min_confidence=0.5)
        result = consolidate_lessons(config, lessons)

        assert len(result) == 1
        assert result[0].title == "strong"

    @patch("carnot.autoresearch.consolidator._merge_batch_with_llm")
    def test_exhausted_levels_final_merge(self, mock_merge: MagicMock) -> None:
        """REQ-AUTO-013: for-else triggers final merge when levels exhausted."""
        # Make merge return MORE lessons than input (prevents convergence)
        # With batch_size=2 and 4 lessons: level 0 -> 2 batches -> 6 lessons
        # level 1 -> 3 batches -> 9 lessons (still > batch_size)
        # After exhausting levels, for-else clause does final merge

        def expanding_merge(config: ConsolidatorConfig, batch: list[Lesson]) -> list[Lesson]:
            return [_make_lesson(title=f"m{i}", confidence=0.8) for i in range(3)]

        mock_merge.side_effect = expanding_merge

        lessons = [_make_lesson(title=f"l{i}") for i in range(4)]
        config = _make_config(batch_size=2, min_confidence=0.3)
        result = consolidate_lessons(config, lessons)
        # Should have completed without error
        assert len(result) >= 1

    @patch("carnot.autoresearch.consolidator._merge_batch_with_llm")
    def test_preserves_benchmark_tags(self, mock_merge: MagicMock) -> None:
        """REQ-AUTO-013: consolidated lessons retain benchmark metadata."""
        mock_merge.return_value = [
            Lesson(
                title="merged",
                description="d",
                confidence=0.8,
                applicable_benchmarks=["rosenbrock", "double_well"],
            )
        ]

        lessons = [_make_lesson() for _ in range(2)]
        result = consolidate_lessons(_make_config(), lessons)
        assert "rosenbrock" in result[0].applicable_benchmarks
