"""Tests for Exp 535 helpers — jepa_retrain_v7 module.

Coverage: load_v7_cot_corpus, summarize_corpus.

Spec: REQ-LEARN-049, REQ-LEARN-050, SCENARIO-LEARN-078, SCENARIO-LEARN-079
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from carnot.embeddings.jepa_retrain import ViolationPair
from carnot.models.jepa_retrain_v7 import (
    load_v7_cot_corpus,
    summarize_corpus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fover_file(path: Path, n: int = 3) -> Path:
    """Write n FOVER-style entries and return path."""
    data = [
        {"step_text": f"step {i}", "label": "correct" if i % 2 == 0 else "incorrect", "question_id": f"q{i}"}
        for i in range(n)
    ]
    path.write_text(json.dumps(data))
    return path


def _make_pairs(n: int = 5) -> List[ViolationPair]:
    return [
        ViolationPair(
            partial_response=f"text {i}",
            full_response=f"text {i} full",
            has_violation=(i % 2 == 1),
            model_id=f"model_{i % 2}",
            question_id=f"q{i}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# load_v7_cot_corpus tests
# ---------------------------------------------------------------------------


class TestLoadV7CotCorpus:
    """Spec: REQ-LEARN-049, SCENARIO-LEARN-078, SCENARIO-LEARN-079"""

    def test_prefers_preferred_paths_when_present(self, tmp_path: Path):
        """SCENARIO-LEARN-078: preferred files yield live_exp527_528."""
        pref = tmp_path / "exp527_cot_pairs.json"
        _make_fover_file(pref, n=4)
        fb = tmp_path / "fover.json"
        _make_fover_file(fb, n=2)

        pairs, source = load_v7_cot_corpus([str(pref)], [str(fb)])

        assert source == "live_exp527_528"
        assert len(pairs) == 4
        assert all(isinstance(p, ViolationPair) for p in pairs)

    def test_falls_back_when_preferred_absent(self, tmp_path: Path):
        """SCENARIO-LEARN-079: missing preferred files → fallback."""
        fb = tmp_path / "fover.json"
        _make_fover_file(fb, n=3)

        pairs, source = load_v7_cot_corpus(
            [str(tmp_path / "nonexistent_527.json")],
            [str(fb)],
        )

        assert source == "live_fover_442"
        assert len(pairs) == 3

    def test_returns_synthetic_when_all_absent(self, tmp_path: Path):
        """SCENARIO-LEARN-079: nothing on disk → synthetic sentinel."""
        pairs, source = load_v7_cot_corpus(
            [str(tmp_path / "no527.json")],
            [str(tmp_path / "nofover.json")],
        )

        assert source == "synthetic"
        assert pairs == []

    def test_accumulates_multiple_preferred_files(self, tmp_path: Path):
        """Both exp527 and exp528 files are merged when both exist."""
        f527 = tmp_path / "exp527_cot_pairs.json"
        f528 = tmp_path / "exp528_cot_pairs.json"
        _make_fover_file(f527, n=3)
        _make_fover_file(f528, n=2)

        pairs, source = load_v7_cot_corpus([str(f527), str(f528)], [])

        assert source == "live_exp527_528"
        assert len(pairs) == 5

    def test_fallback_tries_all_fallback_paths(self, tmp_path: Path):
        """Fallback accumulates across multiple fallback files."""
        fb1 = tmp_path / "fover.json"
        fb2 = tmp_path / "exp514.json"
        _make_fover_file(fb1, n=2)
        _make_fover_file(fb2, n=1)

        pairs, source = load_v7_cot_corpus(
            [str(tmp_path / "no527.json")],
            [str(fb1), str(fb2)],
        )

        assert source == "live_fover_442"
        assert len(pairs) == 3

    def test_empty_preferred_list_goes_to_fallback(self, tmp_path: Path):
        """Empty preferred_paths list → try fallback immediately."""
        fb = tmp_path / "fover.json"
        _make_fover_file(fb, n=2)

        pairs, source = load_v7_cot_corpus([], [str(fb)])

        assert source == "live_fover_442"
        assert len(pairs) == 2

    def test_returns_violation_pair_objects(self, tmp_path: Path):
        """Each returned item is a ViolationPair with non-empty partial_response."""
        pref = tmp_path / "exp527.json"
        _make_fover_file(pref, n=3)

        pairs, _ = load_v7_cot_corpus([str(pref)], [])

        for p in pairs:
            assert isinstance(p, ViolationPair)
            assert p.partial_response

    def test_never_raises_on_missing_files(self, tmp_path: Path):
        """SCENARIO-LEARN-079: function must not raise even with all paths missing."""
        try:
            pairs, source = load_v7_cot_corpus(
                [str(tmp_path / "x.json")],
                [str(tmp_path / "y.json"), str(tmp_path / "z.json")],
            )
        except Exception as exc:
            pytest.fail(f"load_v7_cot_corpus raised unexpectedly: {exc}")

        assert source == "synthetic"
        assert pairs == []


# ---------------------------------------------------------------------------
# summarize_corpus tests
# ---------------------------------------------------------------------------


class TestSummarizeCorpus:
    """Spec: REQ-LEARN-049"""

    def test_empty_corpus(self):
        result = summarize_corpus([])
        assert result["n_pairs"] == 0
        assert result["n_correct"] == 0
        assert result["n_incorrect"] == 0
        assert result["source_breakdown"] == {}

    def test_counts_correct_and_incorrect(self):
        pairs = _make_pairs(6)
        # has_violation is True when i%2==1 → 3 incorrect, 3 correct
        result = summarize_corpus(pairs)
        assert result["n_pairs"] == 6
        assert result["n_correct"] == 3
        assert result["n_incorrect"] == 3

    def test_source_breakdown_counts_by_model_id(self):
        pairs = _make_pairs(4)
        # model_id is 'model_0' for even i, 'model_1' for odd i → 2 each
        result = summarize_corpus(pairs)
        bd = result["source_breakdown"]
        assert bd["model_0"] == 2
        assert bd["model_1"] == 2

    def test_all_correct(self):
        pairs = [
            ViolationPair("text", "text", False, "m", f"q{i}")
            for i in range(4)
        ]
        result = summarize_corpus(pairs)
        assert result["n_correct"] == 4
        assert result["n_incorrect"] == 0

    def test_all_incorrect(self):
        pairs = [
            ViolationPair("text", "text", True, "m", f"q{i}")
            for i in range(3)
        ]
        result = summarize_corpus(pairs)
        assert result["n_correct"] == 0
        assert result["n_incorrect"] == 3

    def test_single_pair(self):
        pairs = [ViolationPair("text", "text", True, "mymodel", "q1")]
        result = summarize_corpus(pairs)
        assert result["n_pairs"] == 1
        assert result["n_correct"] == 0
        assert result["n_incorrect"] == 1
        assert result["source_breakdown"] == {"mymodel": 1}

    def test_unknown_model_id_when_empty_string(self):
        """Empty model_id is stored as 'unknown' in source_breakdown."""
        pairs = [ViolationPair("text", "text", False, "", "q1")]
        result = summarize_corpus(pairs)
        assert "unknown" in result["source_breakdown"]

    def test_returns_required_keys(self):
        result = summarize_corpus(_make_pairs(2))
        assert set(result.keys()) == {"n_pairs", "n_correct", "n_incorrect", "source_breakdown"}

    def test_n_pairs_equals_correct_plus_incorrect(self):
        pairs = _make_pairs(10)
        result = summarize_corpus(pairs)
        assert result["n_pairs"] == result["n_correct"] + result["n_incorrect"]
