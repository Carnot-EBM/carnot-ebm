"""Tests for Exp 558 helpers — 100% targeted coverage on new functions.

Spec: REQ-VERIFY-115-B,
      SCENARIO-VERIFY-131, SCENARIO-VERIFY-132, SCENARIO-VERIFY-133
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import helpers directly from the experiment module.
from scripts.experiment_558_internal_probe_real import (
    PAPER_RATIO,
    PROBE_LAYER,
    _load_corpus,
    _load_eorm_auc,
    _make_probe_pairs,
    _split_corpus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entries(n: int = 120, *, n_incorrect: int | None = None) -> list[dict]:
    """Build n minimal fover_corpus_v2 entries."""
    if n_incorrect is None:
        n_incorrect = n // 2
    entries = []
    for i in range(n):
        entries.append(
            {
                "question": f"q{i}",
                "response": f"r{i}",
                "model_id": "test",
                "is_correct": i >= n_incorrect,
                "constraint_types": ["carry" if i % 2 else "arithmetic"],
            }
        )
    return entries


# ---------------------------------------------------------------------------
# _load_corpus
# ---------------------------------------------------------------------------


class TestLoadCorpus:
    """SCENARIO-VERIFY-131: corpus loading from plain JSON list."""

    def test_loads_list(self, tmp_path: Path) -> None:
        entries = _make_entries(10)
        p = tmp_path / "corpus.json"
        p.write_text(json.dumps(entries))
        result = _load_corpus(p)
        assert len(result) == 10

    def test_loads_dict_pairs_key(self, tmp_path: Path) -> None:
        entries = _make_entries(5)
        p = tmp_path / "corpus.json"
        p.write_text(json.dumps({"pairs": entries}))
        result = _load_corpus(p)
        assert len(result) == 5

    def test_loads_dict_labeled_pairs_key(self, tmp_path: Path) -> None:
        entries = _make_entries(3)
        p = tmp_path / "corpus.json"
        p.write_text(json.dumps({"labeled_pairs": entries}))
        result = _load_corpus(p)
        assert len(result) == 3

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            _load_corpus(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# _split_corpus
# ---------------------------------------------------------------------------


class TestSplitCorpus:
    """SCENARIO-VERIFY-131: deterministic 80/20 split."""

    def test_split_sizes(self) -> None:
        entries = _make_entries(100)
        train, test = _split_corpus(entries, 0.80, seed=42)
        assert len(train) == 80
        assert len(test) == 20

    def test_no_overlap(self) -> None:
        entries = _make_entries(100)
        train, test = _split_corpus(entries, 0.80, seed=42)
        train_qs = {e["question"] for e in train}
        test_qs = {e["question"] for e in test}
        assert train_qs.isdisjoint(test_qs)

    def test_deterministic(self) -> None:
        entries = _make_entries(100)
        t1, v1 = _split_corpus(entries, 0.80, seed=42)
        t2, v2 = _split_corpus(entries, 0.80, seed=42)
        assert [e["question"] for e in t1] == [e["question"] for e in t2]
        assert [e["question"] for e in v1] == [e["question"] for e in v2]

    def test_different_seeds_differ(self) -> None:
        entries = _make_entries(100)
        t1, _ = _split_corpus(entries, 0.80, seed=42)
        t2, _ = _split_corpus(entries, 0.80, seed=99)
        assert [e["question"] for e in t1] != [e["question"] for e in t2]

    def test_covers_all_entries(self) -> None:
        entries = _make_entries(50)
        train, test = _split_corpus(entries, 0.80, seed=1)
        assert len(train) + len(test) == 50


# ---------------------------------------------------------------------------
# _make_probe_pairs
# ---------------------------------------------------------------------------


class TestMakeProbePairs:
    """SCENARIO-VERIFY-132: hidden-state assignment and label convention."""

    def test_returns_correct_length(self) -> None:
        entries = _make_entries(20)
        pairs = _make_probe_pairs(entries, hidden_size=64, seed=7)
        assert len(pairs) == 20

    def test_hidden_state_shape(self) -> None:
        entries = _make_entries(10)
        pairs = _make_probe_pairs(entries, hidden_size=128, seed=0)
        for hs, label in pairs:
            assert hs.shape == (128,)
            assert label in (0, 1)

    def test_correct_entries_get_label_0(self) -> None:
        # All correct entries should have label=0 (correct → probe should output low score)
        entries = [
            {"question": "q", "response": "r", "model_id": "t", "is_correct": True, "constraint_types": []}
            for _ in range(10)
        ]
        pairs = _make_probe_pairs(entries, hidden_size=32, seed=5)
        labels = [label for _, label in pairs]
        assert all(l == 0 for l in labels)

    def test_incorrect_entries_get_label_1(self) -> None:
        entries = [
            {"question": "q", "response": "r", "model_id": "t", "is_correct": False, "constraint_types": []}
            for _ in range(10)
        ]
        pairs = _make_probe_pairs(entries, hidden_size=32, seed=5)
        labels = [label for _, label in pairs]
        assert all(l == 1 for l in labels)

    def test_empty_entries(self) -> None:
        pairs = _make_probe_pairs([], hidden_size=64, seed=0)
        assert pairs == []

    def test_hidden_states_are_float64(self) -> None:
        entries = _make_entries(5)
        pairs = _make_probe_pairs(entries, hidden_size=16, seed=3)
        for hs, _ in pairs:
            assert hs.dtype == np.float64


# ---------------------------------------------------------------------------
# _load_eorm_auc
# ---------------------------------------------------------------------------


class TestLoadEormAuc:
    def test_loads_after_auc(self, tmp_path: Path) -> None:
        p = tmp_path / "exp556.json"
        p.write_text(json.dumps({"experiment": 556, "after_auc": 0.87}))
        assert _load_eorm_auc(p) == pytest.approx(0.87)

    def test_missing_file_returns_default(self, tmp_path: Path) -> None:
        result = _load_eorm_auc(tmp_path / "nonexistent.json")
        assert result == pytest.approx(0.5)

    def test_invalid_json_returns_default(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not json")
        result = _load_eorm_auc(p)
        assert result == pytest.approx(0.5)

    def test_missing_key_returns_default(self, tmp_path: Path) -> None:
        p = tmp_path / "exp556.json"
        p.write_text(json.dumps({"experiment": 556}))
        result = _load_eorm_auc(p)
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# PAPER_RATIO constant
# ---------------------------------------------------------------------------


class TestPaperRatio:
    """Sanity-check the 1/810 arXiv 2511.06209 headline figure."""

    def test_paper_ratio_value(self) -> None:
        # PAPER_RATIO is pre-rounded to 8 decimal places; compare with tolerance
        assert abs(PAPER_RATIO - 1.0 / 810) < 1e-7

    def test_probe_layer_is_minus_four(self) -> None:
        assert PROBE_LAYER == -4
