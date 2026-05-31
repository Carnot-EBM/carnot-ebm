"""Tests for experiment_3543 cross-corpus transfer of step->final aggregation.

REQ-KONA-3543, SCENARIO-KONA-3543: Transfer evaluation of the step->final
aggregation mechanism confirmed in exp3532.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the scripts directory importable.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiment_3543_fover_step_aggregation_cross_corpus_generalize_v1 import (
    _distinct_pipeline_assert,
    _checksum,
    _BASE_SEED,
    _is_usable,
    _normalise_sample,
)


def _make_records(n: int = 10) -> list[dict]:
    """Build minimal records matching the corpus schema for testing."""
    records = []
    for i in range(n):
        records.append({
            "problem_id": f"prob_{i}",
            "gold_answer": str(i),
            "samples": [
                {
                    "reasoning_steps": [f"x = {i}", f"so the answer is {i}"],
                    "correct": True,
                    "extracted_answer": str(i),
                },
                {
                    "steps": [f"y = {i + 1}", f"therefore y = {i + 1}"],
                    "correct": False,
                    "extracted_answer": str(i + 1),
                },
                {
                    "reasoning_steps": [f"z = {i}", "compute"],
                    "correct": i % 2 == 0,
                    "extracted_answer": str(i) if i % 2 == 0 else str(i + 2),
                },
            ],
        })
    return records


class TestDistinctPipelineAssert:
    def test_empty_arrays_counts_as_distinct(self):
        assert _distinct_pipeline_assert([], []) is True

    def test_different_lengths_distinct(self):
        assert _distinct_pipeline_assert([1.0, 2.0], [1.0]) is True

    def test_identical_arrays_not_distinct(self):
        a = [1.0, 2.0, 3.0]
        assert _distinct_pipeline_assert(a, a[:]) is False

    def test_different_arrays_are_distinct(self):
        last_scores = [1.5, 2.0, 0.8]
        min_scores = [0.5, 0.3, 0.8]
        assert _distinct_pipeline_assert(last_scores, min_scores) is True

    def test_one_element_diff_is_distinct(self):
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 4.0]
        assert _distinct_pipeline_assert(a, b) is True

    def test_all_equal_returns_false(self):
        a = [0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5]
        assert _distinct_pipeline_assert(a, b) is False


class TestChecksum:
    def test_returns_16_char_hex(self):
        recs_a = _make_records(3)
        recs_b = _make_records(4)
        cs = _checksum(recs_a, recs_b, 42)
        assert len(cs) == 16
        assert all(c in "0123456789abcdef" for c in cs)

    def test_different_seeds_different_checksum(self):
        recs_a = _make_records(3)
        recs_b = _make_records(4)
        cs1 = _checksum(recs_a, recs_b, 42)
        cs2 = _checksum(recs_a, recs_b, 43)
        assert cs1 != cs2

    def test_same_inputs_same_checksum(self):
        recs_a = _make_records(3)
        recs_b = _make_records(4)
        cs1 = _checksum(recs_a, recs_b, 42)
        cs2 = _checksum(recs_a, recs_b, 42)
        assert cs1 == cs2


class TestIsUsable:
    def test_usable_record(self):
        recs = _make_records(1)
        assert _is_usable(recs[0]) is True

    def test_missing_gold_answer(self):
        recs = _make_records(1)
        del recs[0]["gold_answer"]
        assert _is_usable(recs[0]) is False

    def test_insufficient_samples(self):
        recs = _make_records(1)
        recs[0]["samples"] = recs[0]["samples"][:1]
        assert _is_usable(recs[0]) is False


class TestNormaliseSample:
    def test_steps_renamed_to_reasoning_steps(self):
        s = {"steps": ["a"], "correct": True}
        norm = _normalise_sample(s)
        assert "reasoning_steps" in norm
        assert "steps" not in norm
        assert norm["reasoning_steps"] == ["a"]

    def test_already_reasoning_steps_preserved(self):
        s = {"reasoning_steps": ["a"], "correct": True}
        norm = _normalise_sample(s)
        assert norm == s


class TestModuleConstants:
    def test_base_seed_is_not_experiment_number(self):
        assert _BASE_SEED != 3543
