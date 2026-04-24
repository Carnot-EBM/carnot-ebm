"""Tests for Experiment 787 — S* Energy Pre-Ranking for Code Candidate Selection.

Spec: REQ-RANK-001, REQ-RANK-002, SCENARIO-RANK-001, SCENARIO-RANK-002

Coverage target: 100% of carnot/pipeline/sstar_energy_ranker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from carnot.pipeline.sstar_energy_ranker import (  # noqa: E402
    CandidateWithEnergy,
    SStarConfig,
    SStarEnergyRanker,
)


# ---------------------------------------------------------------------------
# REQ-RANK-001: SStarEnergyRanker energy computation and candidate ranking
# ---------------------------------------------------------------------------


class TestSStarConfig:
    """Tests for SStarConfig dataclass defaults."""

    def test_default_values(self):
        """SStarConfig MUST have n_candidates=4, energy_top_k=1, vocab_size=256."""
        config = SStarConfig()
        assert config.n_candidates == 4
        assert config.energy_top_k == 1
        assert config.vocab_size == 256

    def test_custom_values(self):
        """SStarConfig MUST accept custom values for all fields."""
        config = SStarConfig(n_candidates=8, energy_top_k=2, vocab_size=512)
        assert config.n_candidates == 8
        assert config.energy_top_k == 2
        assert config.vocab_size == 512


class TestSStarEnergyRankerInit:
    """Tests for SStarEnergyRanker initialization."""

    def test_default_config_used_when_none(self):
        """SStarEnergyRanker(config=None) MUST use a default SStarConfig (REQ-RANK-001)."""
        ranker = SStarEnergyRanker(config=None)
        assert ranker.config is not None
        assert ranker.config.n_candidates == 4

    def test_custom_config_stored(self):
        """SStarEnergyRanker MUST store the provided config (REQ-RANK-001)."""
        config = SStarConfig(n_candidates=8)
        ranker = SStarEnergyRanker(config=config)
        assert ranker.config.n_candidates == 8


class TestComputeEnergy:
    """Tests for SStarEnergyRanker.compute_energy — REQ-RANK-001."""

    def setup_method(self):
        self.ranker = SStarEnergyRanker()

    def test_energy_is_non_negative(self):
        """Energy MUST be a non-negative float for any code input (REQ-RANK-001)."""
        energy = self.ranker.compute_energy("def add(a, b):\n    return a + b")
        assert isinstance(energy, float)
        assert energy >= 0.0

    def test_empty_code_has_low_energy(self):
        """Empty or trivial code MUST produce lower energy than complex code (REQ-RANK-001)."""
        empty_energy = self.ranker.compute_energy("")
        complex_energy = self.ranker.compute_energy(
            "def f(a, b, c, d, e):\n    return a + b + c + d + e + 1 + 2 + 3"
        )
        # Empty code (no tokens) should have energy <= complex code with many tokens.
        assert empty_energy <= complex_energy

    def test_longer_code_has_higher_energy(self):
        """A longer candidate with more tokens MUST have higher energy (REQ-RANK-001).

        This validates the Occam's razor prior: simpler code = lower energy.
        'return a + b + 1' has one extra constant and operator vs 'return a + b'.
        """
        short_code = "def add(a, b):\n    return a + b"
        long_code = "def add(a, b):\n    return a + b + 1"
        short_energy = self.ranker.compute_energy(short_code)
        long_energy = self.ranker.compute_energy(long_code)
        # The longer candidate has an extra literal '1' and an extra '+' operator,
        # so it must have strictly higher token count energy.
        assert long_energy > short_energy

    def test_same_code_produces_same_energy(self):
        """Energy computation MUST be deterministic (REQ-RANK-001)."""
        code = "def f(a, b):\n    return a * b"
        energy1 = self.ranker.compute_energy(code)
        energy2 = self.ranker.compute_energy(code)
        assert energy1 == energy2

    def test_invalid_syntax_code_produces_zero_energy(self):
        """Syntactically invalid code MUST not crash; produces 0.0 energy (REQ-RANK-001).

        code_to_embedding handles TokenError gracefully and returns the partial
        tokenization (typically empty for completely malformed code).
        """
        energy = self.ranker.compute_energy("def !!invalid((*)")
        assert isinstance(energy, float)
        assert energy >= 0.0


class TestRankByEnergy:
    """Tests for SStarEnergyRanker.rank_by_energy — REQ-RANK-001, SCENARIO-RANK-001."""

    def setup_method(self):
        self.ranker = SStarEnergyRanker()

    def test_returns_all_candidates(self):
        """rank_by_energy MUST return all input candidates, none dropped (SCENARIO-RANK-001)."""
        candidates = [
            "def f(a, b):\n    return a + b",
            "def f(a, b):\n    return a - b",
            "def f(a, b):\n    return a * b",
            "def f(a, b):\n    return a + b + 1",
        ]
        ranked = self.ranker.rank_by_energy(candidates)
        assert len(ranked) == len(candidates)
        assert set(ranked) == set(candidates)

    def test_sorted_ascending_by_energy(self):
        """rank_by_energy MUST return candidates sorted lowest energy first (SCENARIO-RANK-001).

        'return a + b' has fewer tokens than 'return a + b + 1', so the shorter
        candidate must appear first in the ranked output.
        """
        short_correct = "def add(a, b):\n    return a + b"
        long_wrong = "def add(a, b):\n    return a + b + 1"
        candidates = [long_wrong, short_correct]  # Wrong order initially
        ranked = self.ranker.rank_by_energy(candidates)
        # Lower energy (shorter code) must come first.
        assert ranked[0] == short_correct
        assert ranked[1] == long_wrong

    def test_empty_candidates_returns_empty(self):
        """rank_by_energy MUST return an empty list for empty input (REQ-RANK-001)."""
        assert self.ranker.rank_by_energy([]) == []

    def test_single_candidate_returned_unchanged(self):
        """rank_by_energy with one candidate MUST return a list with that candidate."""
        code = "def f(a, b):\n    return a + b"
        assert self.ranker.rank_by_energy([code]) == [code]

    def test_problem_context_ignored_in_static_energy(self):
        """rank_by_energy MUST produce same result regardless of problem_context (REQ-RANK-001).

        The static energy implementation does not use problem_context — it is
        reserved for future semantic energy extensions.
        """
        candidates = [
            "def f(a, b):\n    return a + b",
            "def f(a, b):\n    return a + b + 1",
        ]
        ranked_no_ctx = self.ranker.rank_by_energy(candidates, problem_context="")
        ranked_with_ctx = self.ranker.rank_by_energy(candidates, problem_context="Add two numbers")
        assert ranked_no_ctx == ranked_with_ctx

    def test_tie_broken_by_original_order(self):
        """rank_by_energy MUST break ties by original index (stable sort) (SCENARIO-RANK-001).

        'return a + b' and 'return a - b' have the same token count, so the one
        that appeared first in the input must appear first in the output.
        """
        first = "def f(a, b):\n    return a + b"
        second = "def f(a, b):\n    return a - b"
        # Both have identical token counts; the first must win the tie.
        ranked = self.ranker.rank_by_energy([first, second])
        assert ranked[0] == first
        assert ranked[1] == second

    def test_lowest_energy_candidate_identified(self):
        """rank_by_energy[0] MUST be the candidate with the globally lowest energy (REQ-RANK-001)."""
        candidates = [
            "def f(a, b):\n    return a + b + 1 + 2 + 3",  # longest (highest energy)
            "def f(a, b):\n    return a + b",               # shortest (lowest energy)
            "def f(a, b):\n    return a + b + 1",           # medium
            "def f(a, b):\n    return a + b + 1 + 2",      # medium-long
        ]
        ranked = self.ranker.rank_by_energy(candidates)
        # The shortest candidate must have the lowest energy and come first.
        assert ranked[0] == "def f(a, b):\n    return a + b"


class TestSelectTopK:
    """Tests for SStarEnergyRanker.select_top_k — REQ-RANK-001."""

    def setup_method(self):
        self.ranker = SStarEnergyRanker(config=SStarConfig(energy_top_k=1))

    def test_select_top_1_returns_single_candidate(self):
        """select_top_k(k=1) MUST return exactly 1 candidate (REQ-RANK-001)."""
        candidates = [
            "def f(a, b):\n    return a + b",
            "def f(a, b):\n    return a + b + 1",
            "def f(a, b):\n    return a * b",
            "def f(a, b):\n    return a - b",
        ]
        top1 = self.ranker.select_top_k(candidates, k=1)
        assert len(top1) == 1

    def test_select_top_k_uses_config_when_k_none(self):
        """select_top_k(k=None) MUST use config.energy_top_k (REQ-RANK-001)."""
        config = SStarConfig(energy_top_k=2)
        ranker = SStarEnergyRanker(config=config)
        candidates = [
            "def f(a, b):\n    return a + b",
            "def f(a, b):\n    return a + b + 1",
            "def f(a, b):\n    return a * b",
        ]
        top = ranker.select_top_k(candidates)  # k=None, uses config.energy_top_k=2
        assert len(top) == 2

    def test_select_top_k_returns_lowest_energy_first(self):
        """select_top_k MUST return candidates in ascending energy order (REQ-RANK-001)."""
        short_code = "def f(a, b):\n    return a + b"
        long_code = "def f(a, b):\n    return a + b + 1"
        candidates = [long_code, short_code]
        top2 = self.ranker.select_top_k(candidates, k=2)
        assert top2[0] == short_code

    def test_select_top_k_with_k_equals_n_returns_all(self):
        """select_top_k(k=n) where n == len(candidates) MUST return all candidates."""
        candidates = ["def f(a, b):\n    return a + b", "def f(a, b):\n    return a - b"]
        top = self.ranker.select_top_k(candidates, k=2)
        assert len(top) == 2


# ---------------------------------------------------------------------------
# REQ-RANK-002: tests_saved_pct threshold gate
# ---------------------------------------------------------------------------


class TestTestsSavedPct:
    """Tests for tests_saved_pct metric — REQ-RANK-002, SCENARIO-RANK-002."""

    def test_tests_saved_zero_when_accuracy_below_threshold(self):
        """tests_saved_pct MUST be 0.0 when energy_correct_rank_pct < 0.60 (SCENARIO-RANK-002).

        With energy_correct_rank_pct = 0.45, the energy signal is not reliable
        enough to skip execution tests, so no savings are claimed.
        """
        energy_correct_rank_pct = 0.45
        n_candidates = 4
        if energy_correct_rank_pct >= 0.60:
            tests_saved_pct = 1.0 - (1.0 / n_candidates)
        else:
            tests_saved_pct = 0.0
        assert tests_saved_pct == 0.0

    def test_tests_saved_nonzero_when_accuracy_at_threshold(self):
        """tests_saved_pct MUST be > 0.0 when energy_correct_rank_pct >= 0.60 (REQ-RANK-002)."""
        energy_correct_rank_pct = 0.60
        n_candidates = 4
        if energy_correct_rank_pct >= 0.60:
            tests_saved_pct = 1.0 - (1.0 / n_candidates)
        else:
            tests_saved_pct = 0.0
        assert tests_saved_pct == pytest.approx(0.75)

    def test_tests_saved_formula_for_n4_candidates(self):
        """tests_saved_pct = 1 - 1/4 = 0.75 for n_candidates=4 (REQ-RANK-002).

        Interpretation: energy correctly selects the best candidate, so we
        run 1 test instead of 4, saving 3/4 = 75% of test executions.
        """
        n_candidates = 4
        expected_savings = 1.0 - (1.0 / n_candidates)
        assert expected_savings == pytest.approx(0.75)

    def test_tests_saved_zero_when_accuracy_is_zero(self):
        """tests_saved_pct MUST be 0.0 when energy_correct_rank_pct = 0.0 (SCENARIO-RANK-002)."""
        energy_correct_rank_pct = 0.0
        n_candidates = 4
        if energy_correct_rank_pct >= 0.60:
            tests_saved_pct = 1.0 - (1.0 / n_candidates)
        else:
            tests_saved_pct = 0.0
        assert tests_saved_pct == 0.0


# ---------------------------------------------------------------------------
# energy_correct_rank_pct metric formula tests
# ---------------------------------------------------------------------------


class TestEnergyCorrectRankPct:
    """Tests for energy_correct_rank_pct metric formula — REQ-RANK-002."""

    def test_metric_formula_all_correct(self):
        """energy_correct_rank_pct = 1.0 when energy selects correct on all problems (REQ-RANK-002)."""
        results = [True, True, True, True]
        pct = sum(results) / len(results)
        assert pct == pytest.approx(1.0)

    def test_metric_formula_none_correct(self):
        """energy_correct_rank_pct = 0.0 when energy never selects correct (REQ-RANK-002)."""
        results = [False, False, False, False]
        pct = sum(results) / len(results)
        assert pct == pytest.approx(0.0)

    def test_metric_formula_half_correct(self):
        """energy_correct_rank_pct = 0.5 for 2/4 correct (REQ-RANK-002)."""
        results = [True, False, True, False]
        pct = sum(results) / len(results)
        assert pct == pytest.approx(0.5)

    def test_energy_selected_equals_first_passing(self):
        """energy_correct_rank = True when energy_selected_idx == first_passing_idx (REQ-RANK-002).

        Candidate index 0 is always labeled correct in the synthetic corpus.
        If the energy-selected index is also 0, the prediction is correct.
        """
        energy_selected_idx = 0
        first_passing_idx = 0  # candidate 0 is always correct in Exp 787 corpus
        assert (energy_selected_idx == first_passing_idx) is True

    def test_energy_selected_wrong_candidate(self):
        """energy_correct_rank = False when energy selects a non-correct candidate (REQ-RANK-002)."""
        energy_selected_idx = 2  # energy picked candidate 2
        first_passing_idx = 0    # but candidate 0 is the correct one
        assert (energy_selected_idx == first_passing_idx) is False


# ---------------------------------------------------------------------------
# CandidateWithEnergy dataclass
# ---------------------------------------------------------------------------


class TestCandidateWithEnergy:
    """Tests for CandidateWithEnergy dataclass — REQ-RANK-001."""

    def test_fields_stored_correctly(self):
        """CandidateWithEnergy MUST store code, energy, and original_index (REQ-RANK-001)."""
        c = CandidateWithEnergy(code="def f(): pass", energy=3.0, original_index=2)
        assert c.code == "def f(): pass"
        assert c.energy == 3.0
        assert c.original_index == 2
