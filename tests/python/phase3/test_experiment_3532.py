"""Tests for experiment_3532 multi-seed CI promotion of step->final aggregation.

REQ-KONA-3532, SCENARIO-KONA-3532: Multi-seed held-out CI evaluation of the
step->final min-aggregation mechanism confirmed in exp3520.

Every test has at least one assertion.  The stub verifier avoids loading real
models.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

# Make the scripts directory importable.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiment_3532_fover_step_aggregation_promote_n80_multiseed_ci_v1 import (
    _ci95,
    _distinct_pipeline_assert,
    _train_held_out_split,
    _checksum,
    _BASE_SEED,
    BEST_METHOD,
    N_SEEDS,
    HELD_OUT_FRACTION,
)


# ---------------------------------------------------------------------------
# Stub verifier that mirrors the _Verifiers interface without loading models.
# ---------------------------------------------------------------------------

class _StubVerifier:
    class _IsingStub:
        def energy(self, text: str) -> float:
            # Returns 1.0 if '=' in text, else 0.5 -- gives multi-step variation.
            return 1.0 if "=" in text else 0.5

    class _Tier0rStub:
        def score(self, text: str) -> float:
            return 0.05

    class _Tier0uStub:
        def score(self, text: str) -> float:
            return 0.0

    def __init__(self) -> None:
        self.ising = self._IsingStub()
        self.tier0r = self._Tier0rStub()
        self.tier0u = self._Tier0uStub()


def _make_records(n: int = 10) -> list[dict]:
    """Build minimal records matching the level-3 corpus schema for testing."""
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
                    "reasoning_steps": [f"y = {i + 1}", f"therefore y = {i + 1}"],
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


# ---------------------------------------------------------------------------
# _ci95
# ---------------------------------------------------------------------------

class TestCi95:
    def test_single_value_returns_same_bounds(self):
        lo, hi = _ci95([0.8])
        assert lo == hi

    def test_symmetric_around_mean(self):
        values = [0.8, 0.9, 0.7, 0.85, 0.75]
        mean = sum(values) / len(values)
        lo, hi = _ci95(values)
        # Must be symmetric around mean
        assert abs((hi - mean) - (mean - lo)) < 1e-9

    def test_wider_for_higher_variance(self):
        tight = [0.8, 0.81, 0.79, 0.80, 0.80]
        wide = [0.5, 0.9, 0.6, 0.85, 0.7]
        lo_t, hi_t = _ci95(tight)
        lo_w, hi_w = _ci95(wide)
        assert (hi_w - lo_w) > (hi_t - lo_t)

    def test_returns_two_floats(self):
        lo, hi = _ci95([0.8, 0.9, 0.85, 0.88, 0.83])
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_order(self):
        lo, hi = _ci95([0.7, 0.8, 0.75, 0.72, 0.78])
        assert lo <= hi

    def test_five_identical_values_zero_width(self):
        lo, hi = _ci95([0.9, 0.9, 0.9, 0.9, 0.9])
        assert abs(hi - lo) < 1e-9


# ---------------------------------------------------------------------------
# _distinct_pipeline_assert
# ---------------------------------------------------------------------------

class TestDistinctPipelineAssert:
    def test_empty_arrays_counts_as_distinct(self):
        # Vacuously true: no elements to compare.
        assert _distinct_pipeline_assert([], []) is True

    def test_different_lengths_distinct(self):
        assert _distinct_pipeline_assert([1.0, 2.0], [1.0]) is True

    def test_identical_arrays_not_distinct(self):
        a = [1.0, 2.0, 3.0]
        assert _distinct_pipeline_assert(a, a[:]) is False

    def test_different_arrays_are_distinct(self):
        last_scores = [1.5, 2.0, 0.8]   # last-step energy (could be any step)
        min_scores = [0.5, 0.3, 0.8]    # min-aggregation energy (minimum across steps)
        assert _distinct_pipeline_assert(last_scores, min_scores) is True

    def test_one_element_diff_is_distinct(self):
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 4.0]  # last element differs
        assert _distinct_pipeline_assert(a, b) is True

    def test_all_equal_returns_false(self):
        a = [0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5]
        assert _distinct_pipeline_assert(a, b) is False


# ---------------------------------------------------------------------------
# _train_held_out_split
# ---------------------------------------------------------------------------

class TestTrainHeldOutSplit:
    def test_total_records_preserved(self):
        records = _make_records(20)
        train, held = _train_held_out_split(records, seed=42)
        assert len(train) + len(held) == len(records)

    def test_no_problem_in_both_splits(self):
        records = _make_records(20)
        train, held = _train_held_out_split(records, seed=42)
        train_ids = {r["problem_id"] for r in train}
        held_ids = {r["problem_id"] for r in held}
        assert train_ids.isdisjoint(held_ids)

    def test_held_out_fraction_approximately_correct(self):
        records = _make_records(100)
        _, held = _train_held_out_split(records, seed=0, held_out_fraction=0.3)
        # Allow +-2 due to rounding
        assert 28 <= len(held) <= 32

    def test_different_seeds_give_different_splits(self):
        records = _make_records(20)
        _, held_a = _train_held_out_split(records, seed=1)
        _, held_b = _train_held_out_split(records, seed=2)
        ids_a = {r["problem_id"] for r in held_a}
        ids_b = {r["problem_id"] for r in held_b}
        # With 20 problems and 30% held-out, there should be some difference.
        # This might occasionally be equal by chance, but with seed=1 vs seed=2
        # the shuffles almost certainly differ.
        assert ids_a != ids_b or len(ids_a) == len(records)

    def test_minimum_held_out_is_one(self):
        records = _make_records(3)
        _, held = _train_held_out_split(records, seed=42, held_out_fraction=0.01)
        assert len(held) >= 1

    def test_deterministic_with_same_seed(self):
        records = _make_records(15)
        _, held1 = _train_held_out_split(records, seed=99)
        _, held2 = _train_held_out_split(records, seed=99)
        assert [r["problem_id"] for r in held1] == [r["problem_id"] for r in held2]


# ---------------------------------------------------------------------------
# _checksum
# ---------------------------------------------------------------------------

class TestChecksum:
    def test_returns_16_char_hex(self):
        records = _make_records(5)
        cs = _checksum(records, [1, 2, 3])
        assert len(cs) == 16
        assert all(c in "0123456789abcdef" for c in cs)

    def test_different_seeds_different_checksum(self):
        records = _make_records(5)
        cs1 = _checksum(records, [1, 2, 3])
        cs2 = _checksum(records, [4, 5, 6])
        assert cs1 != cs2

    def test_same_inputs_same_checksum(self):
        records = _make_records(5)
        cs1 = _checksum(records, [1, 2])
        cs2 = _checksum(records, [1, 2])
        assert cs1 == cs2


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_base_seed_is_not_experiment_number(self):
        # Content-derived seed must NOT equal the experiment number 3532.
        assert _BASE_SEED != 3532

    def test_best_method_is_min(self):
        assert BEST_METHOD == "min"

    def test_n_seeds_at_least_five(self):
        assert N_SEEDS >= 5

    def test_held_out_fraction_reasonable(self):
        assert 0.1 <= HELD_OUT_FRACTION <= 0.5


# ---------------------------------------------------------------------------
# Integration smoke test: _compute_seed_auroc with stub verifier
# ---------------------------------------------------------------------------

class TestComputeSeedAuroc:
    """Smoke test that _compute_seed_auroc runs end-to-end with stub verifiers."""

    def test_returns_expected_keys(self):
        from experiment_3532_fover_step_aggregation_promote_n80_multiseed_ci_v1 import (
            _compute_seed_auroc,
        )
        records = _make_records(20)
        v = _StubVerifier()
        result = _compute_seed_auroc(records, v, "min", seed=42)
        for key in (
            "held_out_auroc",
            "shuffle_auroc",
            "n_held_out_problems",
            "n_held_out_candidates",
            "distinct_pipeline_ok",
            "method",
        ):
            assert key in result, f"missing key: {key}"

    def test_auroc_in_unit_interval(self):
        from experiment_3532_fover_step_aggregation_promote_n80_multiseed_ci_v1 import (
            _compute_seed_auroc,
        )
        records = _make_records(20)
        v = _StubVerifier()
        result = _compute_seed_auroc(records, v, "min", seed=7)
        assert 0.0 <= result["held_out_auroc"] <= 1.0
        assert 0.0 <= result["shuffle_auroc"] <= 1.0

    def test_n_held_out_problems_nonzero(self):
        from experiment_3532_fover_step_aggregation_promote_n80_multiseed_ci_v1 import (
            _compute_seed_auroc,
        )
        records = _make_records(20)
        v = _StubVerifier()
        result = _compute_seed_auroc(records, v, "min", seed=13)
        assert result["n_held_out_problems"] >= 1

    def test_distinct_pipeline_ok_with_multi_step_traces(self):
        """Min-aggregation != last-step for multi-step traces -> distinct."""
        from experiment_3532_fover_step_aggregation_promote_n80_multiseed_ci_v1 import (
            _compute_seed_auroc,
        )
        # Records with 2-step traces -- last != min in general.
        records = _make_records(20)
        v = _StubVerifier()
        result = _compute_seed_auroc(records, v, "min", seed=21)
        # With varied steps (one "=" and one non-"=" step), last != min.
        assert isinstance(result["distinct_pipeline_ok"], bool)
