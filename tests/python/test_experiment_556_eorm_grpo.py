"""Tests for load_fover_corpus_v2 in python/carnot/models/eorm_retrain.py.

Spec coverage: REQ-LEARN-060, SCENARIO-LEARN-093, SCENARIO-LEARN-094
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from carnot.models.eorm_retrain import load_fover_corpus_v2
from carnot.models.grpo_eorm_retrain import GRPOContrastivePair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_corpus(path: Path, entries: list[dict]) -> None:
    """Write FOVER corpus entries as JSON to path."""
    with open(path, "w") as f:
        json.dump(entries, f)


def _make_entry(question: str, response: str, is_correct: bool) -> dict:
    """Build a minimal FOVERCorpusEntry-compatible dict."""
    return {
        "question": question,
        "response": response,
        "model_id": "test_model",
        "is_correct": is_correct,
        "constraint_types": ["correct" if is_correct else "incorrect"],
        "cot_steps": [],
    }


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-094: missing file → empty list
# ---------------------------------------------------------------------------


def test_load_fover_corpus_v2_missing_file() -> None:
    """SCENARIO-LEARN-094: load_fover_corpus_v2 returns empty list for missing file."""
    result = load_fover_corpus_v2("/tmp/this_file_does_not_exist_556.json")
    assert result == []


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-093: both-polarity question → one pair per question
# ---------------------------------------------------------------------------


def test_load_fover_corpus_v2_basic_pair() -> None:
    """SCENARIO-LEARN-093: questions with both correct and incorrect responses produce pairs."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        path = Path(f.name)

    entries = [
        _make_entry("What is 2+2?", "4", True),
        _make_entry("What is 2+2?", "5 (wrong)", False),
    ]
    _write_corpus(path, entries)

    pairs = load_fover_corpus_v2(path)
    assert len(pairs) == 1
    pair = pairs[0]
    assert isinstance(pair, GRPOContrastivePair)
    assert pair.correct_response == "4"
    assert pair.incorrect_response == "5 (wrong)"
    assert "What is 2+2?" in pair.question_id


def test_load_fover_corpus_v2_only_correct_no_pair() -> None:
    """Questions with only correct responses produce no pair (no incorrect counterpart)."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        path = Path(f.name)

    entries = [
        _make_entry("Q1", "answer A", True),
        _make_entry("Q1", "answer B", True),
    ]
    _write_corpus(path, entries)

    pairs = load_fover_corpus_v2(path)
    assert pairs == []


def test_load_fover_corpus_v2_only_incorrect_no_pair() -> None:
    """Questions with only incorrect responses produce no pair."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        path = Path(f.name)

    entries = [
        _make_entry("Q1", "wrong A", False),
        _make_entry("Q1", "wrong B", False),
    ]
    _write_corpus(path, entries)

    pairs = load_fover_corpus_v2(path)
    assert pairs == []


def test_load_fover_corpus_v2_multiple_questions() -> None:
    """Multiple questions each with both polarities → one pair per question."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        path = Path(f.name)

    entries = [
        _make_entry("Q1", "right1", True),
        _make_entry("Q1", "wrong1", False),
        _make_entry("Q2", "right2", True),
        _make_entry("Q2", "wrong2", False),
        # Q3 only has correct — no pair
        _make_entry("Q3", "right3", True),
    ]
    _write_corpus(path, entries)

    pairs = load_fover_corpus_v2(path)
    assert len(pairs) == 2
    question_ids = {p.question_id for p in pairs}
    assert any("Q1" in qid for qid in question_ids)
    assert any("Q2" in qid for qid in question_ids)


def test_load_fover_corpus_v2_first_correct_used() -> None:
    """When a question has multiple correct responses, only the first is used."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        path = Path(f.name)

    entries = [
        _make_entry("Q1", "first_correct", True),
        _make_entry("Q1", "second_correct", True),
        _make_entry("Q1", "wrong", False),
    ]
    _write_corpus(path, entries)

    pairs = load_fover_corpus_v2(path)
    assert len(pairs) == 1
    assert pairs[0].correct_response == "first_correct"
    assert pairs[0].incorrect_response == "wrong"


def test_load_fover_corpus_v2_skips_empty_response() -> None:
    """Entries with empty response strings are skipped."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        path = Path(f.name)

    entries = [
        {"question": "Q1", "response": "", "is_correct": True,
         "model_id": "m", "constraint_types": [], "cot_steps": []},
        _make_entry("Q1", "wrong", False),
    ]
    _write_corpus(path, entries)

    pairs = load_fover_corpus_v2(path)
    # No pair because the correct response is empty and was skipped
    assert pairs == []


def test_load_fover_corpus_v2_skips_empty_question() -> None:
    """Entries with empty question strings are skipped."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        path = Path(f.name)

    entries = [
        {"question": "", "response": "answer", "is_correct": True,
         "model_id": "m", "constraint_types": [], "cot_steps": []},
        {"question": "", "response": "wrong", "is_correct": False,
         "model_id": "m", "constraint_types": [], "cot_steps": []},
    ]
    _write_corpus(path, entries)

    pairs = load_fover_corpus_v2(path)
    assert pairs == []


def test_load_fover_corpus_v2_invalid_json() -> None:
    """Malformed JSON returns empty list without raising."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        path = Path(f.name)
        f.write("not valid json {{{")

    pairs = load_fover_corpus_v2(path)
    assert pairs == []


def test_load_fover_corpus_v2_not_a_list() -> None:
    """JSON that is not a list returns empty list."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        path = Path(f.name)
        json.dump({"not": "a list"}, f)

    pairs = load_fover_corpus_v2(path)
    assert pairs == []


def test_load_fover_corpus_v2_question_id_truncated() -> None:
    """Long question text is truncated to 120 chars in question_id."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        path = Path(f.name)

    long_q = "x" * 200
    entries = [
        _make_entry(long_q, "right", True),
        _make_entry(long_q, "wrong", False),
    ]
    _write_corpus(path, entries)

    pairs = load_fover_corpus_v2(path)
    assert len(pairs) == 1
    assert len(pairs[0].question_id) == 120
