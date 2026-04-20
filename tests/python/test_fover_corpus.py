"""Tests for python/carnot/pipeline/fover_corpus.py.

Coverage target: 100% of fover_corpus.py.

Spec: REQ-DATA-003, REQ-DATA-004,
      SCENARIO-DATA-007, SCENARIO-DATA-008, SCENARIO-DATA-009
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from carnot.pipeline.fover_corpus import (
    FOVERCorpusEntry,
    balance_corpus,
    compute_corpus_diversity,
    merge_fover_sources,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# FOVERCorpusEntry
# ---------------------------------------------------------------------------


class TestFOVERCorpusEntry:
    """REQ-DATA-003 — dataclass structure."""

    def test_fields_populated(self) -> None:
        e = FOVERCorpusEntry(
            question="q",
            response="r",
            model_id="m",
            is_correct=True,
            constraint_types=["correct"],
            cot_steps=[{"step_text": "s", "z3_label": "correct"}],
        )
        assert e.question == "q"
        assert e.model_id == "m"
        assert e.is_correct is True
        assert e.constraint_types == ["correct"]
        assert len(e.cot_steps) == 1

    def test_default_cot_steps(self) -> None:
        e = FOVERCorpusEntry(
            question="q", response="r", model_id="m", is_correct=False,
            constraint_types=[],
        )
        assert e.cot_steps == []


# ---------------------------------------------------------------------------
# merge_fover_sources
# ---------------------------------------------------------------------------


class TestMergeFoverSources:
    """SCENARIO-DATA-007 — merge and deduplicate all sources."""

    def test_missing_file_skipped(self, tmp_path: Path) -> None:
        result = merge_fover_sources([str(tmp_path / "nonexistent.json")])
        assert result == []

    def test_empty_list(self) -> None:
        assert merge_fover_sources([]) == []

    def test_exp442_step_level_schema(self, tmp_path: Path) -> None:
        # Exp 442 format: flat list of step-level dicts.
        data = [
            {"question_id": "q1", "step_text": "step A", "label": "correct", "confidence": 1.0},
            {"question_id": "q1", "step_text": "step B", "label": "incorrect", "confidence": 1.0},
            {"question_id": "q2", "step_text": "step C", "label": "not_verifiable", "confidence": 0.0},
        ]
        f = tmp_path / "fover_steps.json"
        _write_json(f, data)

        entries = merge_fover_sources([str(f)])
        # Two unique question_ids → two entries.
        assert len(entries) == 2
        q1 = next(e for e in entries if e.question == "q1")
        assert set(q1.constraint_types) == {"correct", "incorrect"}
        assert q1.model_id == "unknown"

    def test_exp551_552_entry_schema(self, tmp_path: Path) -> None:
        # Exps 551/552 format: entry-level dicts.
        data = [
            {
                "question_index": 0,
                "question": "What is 2+2?",
                "model": "Qwen3.5-0.8B",
                "response": "4",
                "is_correct": True,
                "cot_steps": [{"step_idx": 0, "step_text": "2+2=4", "z3_label": "correct"}],
                "fover_labels": ["correct"],
            }
        ]
        f = tmp_path / "live_pairs.json"
        _write_json(f, data)

        entries = merge_fover_sources([str(f)])
        assert len(entries) == 1
        assert entries[0].question == "What is 2+2?"
        assert entries[0].model_id == "Qwen3.5-0.8B"
        assert entries[0].is_correct is True
        assert entries[0].constraint_types == ["correct"]

    def test_exp538_cot_schema(self, tmp_path: Path) -> None:
        # Exp 538 indirect format: cot_text, model_id, correct.
        data = [
            {"question": "Q1", "cot_text": "some CoT", "correct": False, "model_id": "ModelA", "latency_s": 1.0}
        ]
        f = tmp_path / "cot_pairs.json"
        _write_json(f, data)

        entries = merge_fover_sources([str(f)])
        assert len(entries) == 1
        assert entries[0].response == "some CoT"
        assert entries[0].is_correct is False
        assert entries[0].constraint_types == []

    def test_deduplication_by_question_model(self, tmp_path: Path) -> None:
        # Same (question, model) from two files → only first kept.
        data = [
            {
                "question": "Same Q",
                "model": "ModelA",
                "response": "r1",
                "is_correct": True,
                "cot_steps": [],
                "fover_labels": ["correct"],
            }
        ]
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        _write_json(f1, data)
        _write_json(f2, data)

        entries = merge_fover_sources([str(f1), str(f2)])
        assert len(entries) == 1

    def test_empty_json_list_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.json"
        _write_json(f, [])
        assert merge_fover_sources([str(f)]) == []

    def test_unknown_schema_skipped(self, tmp_path: Path) -> None:
        # No recognized keys → silently skipped.
        data = [{"foo": "bar", "baz": 1}]
        f = tmp_path / "unknown.json"
        _write_json(f, data)
        # Should not raise; returns empty list.
        result = merge_fover_sources([str(f)])
        assert result == []

    def test_non_list_json_skipped(self, tmp_path: Path) -> None:
        # Dict at top level → not a list → skipped.
        data = {"key": "value"}
        f = tmp_path / "dict.json"
        _write_json(f, data)
        assert merge_fover_sources([str(f)]) == []


# ---------------------------------------------------------------------------
# compute_corpus_diversity
# ---------------------------------------------------------------------------


class TestComputeCorpusDiversity:
    """SCENARIO-DATA-008 — diversity metrics are correct."""

    def test_empty_corpus(self) -> None:
        result = compute_corpus_diversity([])
        assert result["constraint_type_entropy"] == 0.0
        assert result["carry_pct"] == 0.0
        assert result["correct_pct"] == 0.0
        assert result["n_labeled"] == 0

    def test_single_type_zero_entropy(self) -> None:
        # All steps are 'incorrect' → entropy = 0.
        entries = [
            FOVERCorpusEntry("q1", "r", "m", False, ["incorrect", "incorrect"]),
            FOVERCorpusEntry("q2", "r", "m", False, ["incorrect"]),
        ]
        result = compute_corpus_diversity(entries)
        assert result["constraint_type_entropy"] == pytest.approx(0.0, abs=1e-9)
        assert result["carry_pct"] == pytest.approx(1.0)

    def test_two_equal_types_one_bit(self) -> None:
        # 50/50 split of 'correct' and 'incorrect' → entropy = 1.0 bit.
        entries = [
            FOVERCorpusEntry("q1", "r", "m", True, ["correct"]),
            FOVERCorpusEntry("q2", "r", "m", False, ["incorrect"]),
        ]
        result = compute_corpus_diversity(entries)
        assert result["constraint_type_entropy"] == pytest.approx(1.0, abs=1e-9)
        assert result["carry_pct"] == pytest.approx(0.5)
        assert result["correct_pct"] == pytest.approx(0.5)

    def test_three_equal_types_entropy(self) -> None:
        # Three types equally represented → H = log2(3) ≈ 1.585 bits.
        entries = [
            FOVERCorpusEntry("q1", "r", "m", True, ["correct"]),
            FOVERCorpusEntry("q2", "r", "m", False, ["incorrect"]),
            FOVERCorpusEntry("q3", "r", "m", False, ["not_verifiable"]),
        ]
        result = compute_corpus_diversity(entries)
        assert result["constraint_type_entropy"] == pytest.approx(math.log2(3), abs=1e-9)

    def test_entry_no_steps(self) -> None:
        # Entries with no steps don't affect step-level counts but do affect n_labeled.
        entries = [
            FOVERCorpusEntry("q1", "r", "m", True, []),
            FOVERCorpusEntry("q2", "r", "m", True, ["correct"]),
        ]
        result = compute_corpus_diversity(entries)
        assert result["n_labeled"] == 2
        # Only 'correct' steps — entropy = 0.
        assert result["constraint_type_entropy"] == pytest.approx(0.0, abs=1e-9)

    def test_constraint_type_counts_returned(self) -> None:
        entries = [
            FOVERCorpusEntry("q1", "r", "m", True, ["correct", "correct", "incorrect"]),
        ]
        result = compute_corpus_diversity(entries)
        assert result["constraint_type_counts"]["correct"] == 2
        assert result["constraint_type_counts"]["incorrect"] == 1


# ---------------------------------------------------------------------------
# balance_corpus
# ---------------------------------------------------------------------------


class TestBalanceCorpus:
    """SCENARIO-DATA-009 — balance_corpus raises entropy by downsampling."""

    def test_already_balanced_no_change(self) -> None:
        # Three equal types → entropy ~1.585 >= 1.5, no change.
        entries = [
            FOVERCorpusEntry(f"q{i}", "r", "m", False, ["correct"]) for i in range(10)
        ] + [
            FOVERCorpusEntry(f"q{i+10}", "r", "m", False, ["incorrect"]) for i in range(10)
        ] + [
            FOVERCorpusEntry(f"q{i+20}", "r", "m", False, ["not_verifiable"]) for i in range(10)
        ]
        balanced = balance_corpus(entries, target_entropy=1.5)
        assert len(balanced) == len(entries)

    def test_carry_dominated_corpus_balanced(self) -> None:
        # 20 incorrect + 10 correct = ~67% carry → balance to entropy >= 1.0.
        # With 10 correct entries and ~10 incorrect, 50/50 split hits exactly 1.0 bits.
        entries = [
            FOVERCorpusEntry(f"carry_{i}", "r", "m", False, ["incorrect"]) for i in range(20)
        ] + [
            FOVERCorpusEntry(f"ok_{i}", "r", "m", True, ["correct"]) for i in range(10)
        ]
        balanced = balance_corpus(entries, target_entropy=1.0)
        div = compute_corpus_diversity(balanced)
        assert div["constraint_type_entropy"] >= 1.0
        assert len(balanced) < len(entries)

    def test_empty_corpus_returns_empty(self) -> None:
        assert balance_corpus([]) == []

    def test_too_few_entries_stops_early(self) -> None:
        # 11 incorrect + 1 correct — stops when len would go to <=10.
        entries = [
            FOVERCorpusEntry(f"c_{i}", "r", "m", False, ["incorrect"]) for i in range(11)
        ] + [
            FOVERCorpusEntry("ok", "r", "m", True, ["correct"]),
        ]
        balanced = balance_corpus(entries, target_entropy=1.5)
        # Should stop before removing too many; at least 10 remain.
        assert len(balanced) >= 10

    def test_single_type_below_min_size_stops(self) -> None:
        # Only 5 entries, all 'correct' — guard triggers at <=10, no removal possible.
        entries = [
            FOVERCorpusEntry(f"q{i}", "r", "m", True, ["correct"]) for i in range(5)
        ]
        balanced = balance_corpus(entries, target_entropy=1.5)
        # Too few to remove; returns as-is (len=5 already <= 10).
        assert len(balanced) == len(entries)

    def test_returns_copy_not_mutation(self) -> None:
        entries = [
            FOVERCorpusEntry(f"carry_{i}", "r", "m", False, ["incorrect"]) for i in range(20)
        ] + [
            FOVERCorpusEntry(f"ok_{i}", "r", "m", True, ["correct"]) for i in range(5)
        ]
        original_len = len(entries)
        balance_corpus(entries, target_entropy=1.0)
        # Original list must not be mutated.
        assert len(entries) == original_len


# ---------------------------------------------------------------------------
# Integration: pipeline __init__ exports
# ---------------------------------------------------------------------------


class TestPipelineExports:
    """Verify that fover_corpus symbols are exported from carnot.pipeline."""

    def test_imports_from_pipeline(self) -> None:
        from carnot.pipeline import (  # noqa: F401
            FOVERCorpusEntry,
            balance_corpus,
            compute_corpus_diversity,
            merge_fover_sources,
        )
