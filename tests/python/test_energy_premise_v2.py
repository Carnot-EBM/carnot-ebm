"""Tests for the four-condition matched-compute premise helpers (exp3426).

Every test traces to REQ-KONA-3426 / SCENARIO-KONA-3426: the v2 premise test
whose PRIMARY comparison is energy-weighted vote vs majority-vote
self-consistency at matched compute. We cover the two NEW cheap baselines
energy must beat (self-certainty Best-of-N) and the headline energy-weighted
vote, plus the new verdict mapping. The shared primitives (mcnemar, bootstrap,
extraction) are re-exported from the v1 module and already tested in
``test_energy_descent_premise.py``; here we test only the v2 additions.
"""

from __future__ import annotations

import math

import pytest

from carnot.phase3.energy_premise_v2 import (
    PremiseV2Verdict,
    derive_premise_v2_verdict,
    energy_weighted_vote,
    mean_token_confidence,
    self_certainty_select,
)


# REQ-KONA-3426: self-certainty proxy is the mean chosen-token probability.
class TestMeanTokenConfidence:
    def test_empty_or_none_scores_zero(self) -> None:
        # A candidate that produced no scorable tokens must never win by default.
        assert mean_token_confidence(None) == 0.0
        assert mean_token_confidence([]) == 0.0

    def test_mean_of_probabilities(self) -> None:
        # logprob 0.0 -> prob 1.0; ln(0.5) -> prob 0.5; mean = 0.75.
        score = mean_token_confidence([0.0, math.log(0.5)])
        assert score == pytest.approx(0.75)

    def test_filters_non_finite_logprobs(self) -> None:
        # None and -inf tokens are dropped; only the finite 0.0 (prob 1.0) counts.
        score = mean_token_confidence([0.0, float("-inf")])
        assert score == pytest.approx(1.0)

    def test_all_non_finite_scores_zero(self) -> None:
        assert mean_token_confidence([float("-inf"), float("nan")]) == 0.0


# REQ-KONA-3426: self-certainty Best-of-N selects the most-confident sample.
class TestSelfCertaintySelect:
    def test_empty_candidates_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one candidate"):
            self_certainty_select([])

    def test_picks_highest_confidence(self) -> None:
        # Candidate 1 is more confident (logprobs near 0 -> prob near 1).
        idx = self_certainty_select([[math.log(0.3)], [0.0], [math.log(0.5)]])
        assert idx == 1

    def test_tie_breaks_to_earliest(self) -> None:
        # Equal confidence -> earliest candidate wins (deterministic).
        idx = self_certainty_select([[0.0], [0.0]])
        assert idx == 0

    def test_missing_logprobs_lose(self) -> None:
        # A None-logprob candidate scores 0.0 and cannot win against a real one.
        idx = self_certainty_select([None, [math.log(0.1)]])
        assert idx == 1


# REQ-KONA-3426: energy-weighted vote = softmax(-E/T) over distinct answers.
class TestEnergyWeightedVote:
    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            energy_weighted_vote([1, 2], [0.1])

    def test_non_positive_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be positive"):
            energy_weighted_vote([1], [0.1], temperature=0.0)

    def test_all_none_returns_none(self) -> None:
        assert energy_weighted_vote([None, None], [0.1, 0.2]) is None

    def test_low_energy_answer_dominates(self) -> None:
        # Answer 7 has far lower energy than the two votes for 3 -> 7 wins
        # despite being a minority by count, at a sharp temperature.
        result = energy_weighted_vote([3, 3, 7], [5.0, 5.0, 0.0], temperature=0.5)
        assert result == 7

    def test_high_temperature_recovers_majority(self) -> None:
        # T -> large makes weights ~uniform, so the modal answer (3) wins even
        # though answer 7 has slightly lower energy.
        result = energy_weighted_vote([3, 3, 7], [1.0, 1.0, 0.9], temperature=1e6)
        assert result == 3

    def test_tie_breaks_to_earliest_answer(self) -> None:
        # Identical energies -> equal weights; answers 5 and 9 tie 1-1; the
        # earliest-appearing answer (5) wins.
        result = energy_weighted_vote([5, 9], [1.0, 1.0], temperature=1.0)
        assert result == 5

    def test_null_answers_contribute_no_weight(self) -> None:
        # The None candidate is skipped; the single real answer wins.
        result = energy_weighted_vote([None, 4], [0.0, 9.0], temperature=1.0)
        assert result == 4


# REQ-KONA-3426: verdict gates are stated against self-consistency.
class TestDerivePremiseV2Verdict:
    def test_g2_validated_when_energy_significantly_beats_sc(self) -> None:
        v = derive_premise_v2_verdict(
            self_consistency_accuracy=0.80,
            energy_weighted_vote_accuracy=0.88,
            p_value=0.01,
            ci=(0.02, 0.14),
            direction=1.0,
        )
        assert isinstance(v, PremiseV2Verdict)
        assert v.g2_energy_adds_value is True
        assert v.g1_energy_non_inferior is True
        assert v.verdict == "complete: energy_beats_self_consistency_premise_validated"

    def test_g1_when_energy_matches_but_not_significant(self) -> None:
        # Energy strictly higher but the CI lower bound does not clear 0 -> G1 only.
        v = derive_premise_v2_verdict(
            self_consistency_accuracy=0.80,
            energy_weighted_vote_accuracy=0.82,
            p_value=0.30,
            ci=(-0.03, 0.09),
            direction=1.0,
        )
        assert v.g2_energy_adds_value is False
        assert v.g1_energy_non_inferior is True
        assert "matches_but_does_not_beat" in v.verdict

    def test_g1_via_non_significant_shortfall(self) -> None:
        # Energy slightly worse but the shortfall is not significant -> still G1.
        v = derive_premise_v2_verdict(
            self_consistency_accuracy=0.85,
            energy_weighted_vote_accuracy=0.83,
            p_value=0.40,
            ci=(-0.10, 0.04),
            direction=-1.0,
        )
        assert v.g1_energy_non_inferior is True
        assert v.g2_energy_adds_value is False

    def test_g1_fails_when_energy_significantly_below_sc(self) -> None:
        v = derive_premise_v2_verdict(
            self_consistency_accuracy=0.90,
            energy_weighted_vote_accuracy=0.78,
            p_value=0.01,
            ci=(-0.20, -0.04),
            direction=-1.0,
        )
        assert v.g1_energy_non_inferior is False
        assert v.g2_energy_adds_value is False
        assert "premise_unsupported_retire_superiority_framing" in v.verdict
