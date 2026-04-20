"""Tests for PROGRSCentering in jepa_cpmi_pairs — 100% coverage of new code.

**Why these tests exist:**
    PROGRSCentering (arXiv 2604.02341) is the novel component added in Exp 593.
    It prevents reward hacking on easy questions by normalising energy gaps within
    each question group.  These tests verify:

    1. center_pairs() correctly computes raw gaps and group-mean-centered gaps.
    2. compute_centered_loss() returns the expected mean hinge loss on centered gaps.
    3. Empty pair list returns [] / 0.0 without errors.
    4. Single-pair-per-group case: centering has no effect (centered_gap = 0).
    5. Multi-pair-per-group case: centering shifts gaps by the group mean.

Spec: REQ-LEARN-069, SCENARIO-LEARN-107, SCENARIO-LEARN-108, SCENARIO-LEARN-109
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from carnot.inference.jepa_cpmi_pairs import (
    JEPACPMIPair,
    PROGRSCentering,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _const_model(value: float):
    """Return a model callable that always returns a fixed constant energy.

    Useful for constructing pairs with known energy gaps without needing real embeddings.
    """
    def _model(emb: jnp.ndarray) -> float:
        return value
    return _model


def _model_from_dict(energy_map: dict[str, float]):
    """Return a model that maps embedding (encoded as its first element) to energy.

    We encode the embedding as jnp.array([value]) and look up by that integer key.
    """
    def _model(emb: jnp.ndarray) -> float:
        key = int(float(emb[0]))
        return energy_map.get(key, 0.0)
    return _model


def _pair(
    question_id: str,
    correct_energy: float,
    incorrect_energy: float,
) -> JEPACPMIPair:
    """Build a JEPACPMIPair with single-step chains whose energies are constants.

    We encode correct_energy and incorrect_energy into the embeddings as jnp.array([val])
    and use _model_from_dict to retrieve them during tests.

    Args:
        question_id:       Identifier for the source question group.
        correct_energy:    The energy the model should return for the correct chain.
        incorrect_energy:  The energy the model should return for the incorrect chain.
    """
    return JEPACPMIPair(
        question_id=question_id,
        correct_embeddings=[jnp.array([correct_energy])],
        incorrect_embeddings=[jnp.array([incorrect_energy])],
        hard_negative_step_idx=0,
        pair_quality=1.0,
    )


def _identity_model(emb: jnp.ndarray) -> float:
    """Model that returns the first element of the embedding as energy.

    Used with _pair() so that energy(emb) = emb[0].
    """
    return float(emb[0])


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-107: center_pairs computes correct centered gaps
# ---------------------------------------------------------------------------


class TestCenterPairs:
    """Tests for PROGRSCentering.center_pairs().

    Spec: SCENARIO-LEARN-107
    """

    def test_empty_pairs_returns_empty_list(self):
        """center_pairs([]) returns [] without calling model.

        Spec: SCENARIO-LEARN-107
        """
        centering = PROGRSCentering()
        # Model would fail if called — verifies no model call on empty input.
        result = centering.center_pairs([], model=None)
        assert result == []

    def test_single_pair_single_group_centered_gap_is_zero(self):
        """Single pair in its own group: centered_gap = raw_gap - mean([raw_gap]) = 0.

        PROGRS centering with one pair per group leaves the centered gap at 0,
        meaning the loss at margin=1.0 will be max(0, 1.0 - 0) = 1.0 per pair.

        Spec: SCENARIO-LEARN-107, SCENARIO-LEARN-109
        """
        centering = PROGRSCentering()
        # correct_energy=2.0, incorrect_energy=5.0 → raw_gap = 3.0
        # group mean = 3.0 → centered_gap = 0.0
        pair = _pair("q1", correct_energy=2.0, incorrect_energy=5.0)
        result = centering.center_pairs([pair], model=_identity_model)
        assert len(result) == 1
        returned_pair, centered_gap = result[0]
        assert returned_pair is pair
        assert abs(centered_gap - 0.0) < 1e-5

    def test_two_pairs_same_group_centering_shifts_by_mean(self):
        """Two pairs in same group: centered gaps sum to 0 (each is raw_gap - mean).

        Spec: SCENARIO-LEARN-107
        """
        centering = PROGRSCentering()
        # pair_a: raw_gap = 5 - 2 = 3.0
        # pair_b: raw_gap = 6 - 5 = 1.0
        # group_mean = (3.0 + 1.0) / 2 = 2.0
        # centered_a = 3.0 - 2.0 = 1.0
        # centered_b = 1.0 - 2.0 = -1.0
        pair_a = _pair("q1", correct_energy=2.0, incorrect_energy=5.0)
        pair_b = _pair("q1", correct_energy=5.0, incorrect_energy=6.0)
        result = centering.center_pairs([pair_a, pair_b], model=_identity_model)
        assert len(result) == 2
        gaps = [g for _, g in result]
        assert abs(sum(gaps)) < 1e-5  # centered gaps sum to zero
        assert abs(gaps[0] - 1.0) < 1e-5
        assert abs(gaps[1] - (-1.0)) < 1e-5

    def test_two_pairs_different_groups_independent_centering(self):
        """Pairs in different groups are centered independently.

        q1: raw_gap = 3.0, group_mean = 3.0, centered = 0.0
        q2: raw_gap = 1.0, group_mean = 1.0, centered = 0.0

        Spec: SCENARIO-LEARN-107
        """
        centering = PROGRSCentering()
        pair_a = _pair("q1", correct_energy=2.0, incorrect_energy=5.0)
        pair_b = _pair("q2", correct_energy=5.0, incorrect_energy=6.0)
        result = centering.center_pairs([pair_a, pair_b], model=_identity_model)
        gaps = [g for _, g in result]
        assert abs(gaps[0]) < 1e-5
        assert abs(gaps[1]) < 1e-5

    def test_pairs_order_preserved(self):
        """Output list is in the same order as input pairs.

        Spec: SCENARIO-LEARN-107
        """
        centering = PROGRSCentering()
        pairs = [
            _pair("q2", correct_energy=1.0, incorrect_energy=4.0),
            _pair("q1", correct_energy=2.0, incorrect_energy=5.0),
            _pair("q2", correct_energy=3.0, incorrect_energy=7.0),
        ]
        result = centering.center_pairs(pairs, model=_identity_model)
        assert len(result) == 3
        for i, (returned_pair, _) in enumerate(result):
            assert returned_pair is pairs[i]


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-108: compute_centered_loss returns expected mean hinge loss
# ---------------------------------------------------------------------------


class TestComputeCenteredLoss:
    """Tests for PROGRSCentering.compute_centered_loss().

    Spec: SCENARIO-LEARN-108
    """

    def test_empty_pairs_returns_zero(self):
        """compute_centered_loss([]) returns 0.0 without calling model.

        Spec: SCENARIO-LEARN-108
        """
        centering = PROGRSCentering()
        loss = centering.compute_centered_loss(model=None, pairs=[], margin=1.0)
        assert loss == 0.0

    def test_well_separated_pairs_contribute_full_margin_when_centered_gap_zero(self):
        """When centered_gap = 0 for all pairs, each pair contributes margin to the loss.

        This happens when there is one pair per group (each is its own group mean).
        Expected loss = margin = 1.0 per pair, averaged = 1.0.

        Spec: SCENARIO-LEARN-108
        """
        centering = PROGRSCentering()
        # Two pairs in separate groups; centered_gap = 0 for each.
        pairs = [
            _pair("q1", correct_energy=2.0, incorrect_energy=5.0),
            _pair("q2", correct_energy=3.0, incorrect_energy=7.0),
        ]
        loss = centering.compute_centered_loss(_identity_model, pairs, margin=1.0)
        # Each pair: max(0, 1.0 - 0.0) = 1.0; mean = 1.0.
        assert abs(loss - 1.0) < 1e-5

    def test_two_pairs_same_group_loss_accounts_for_centering(self):
        """Two pairs in same group: only the below-mean pair contributes to loss.

        pair_a: raw_gap=3, pair_b: raw_gap=1; group_mean=2
        centered_a = 1.0; centered_b = -1.0
        L_a = max(0, 1.0 - 1.0) = 0.0
        L_b = max(0, 1.0 - (-1.0)) = 2.0
        mean_loss = (0.0 + 2.0) / 2 = 1.0

        Spec: SCENARIO-LEARN-108
        """
        centering = PROGRSCentering()
        pair_a = _pair("q1", correct_energy=2.0, incorrect_energy=5.0)
        pair_b = _pair("q1", correct_energy=5.0, incorrect_energy=6.0)
        loss = centering.compute_centered_loss(_identity_model, [pair_a, pair_b], margin=1.0)
        assert abs(loss - 1.0) < 1e-5

    def test_margin_zero_loss_is_zero_for_positive_centered_gaps(self):
        """With margin=0, any positive centered gap yields zero loss.

        Spec: SCENARIO-LEARN-108
        """
        centering = PROGRSCentering()
        # Two pairs in different groups: centered_gap = 0 for each.
        # L_p = max(0, 0 - 0) = 0.0 for both.
        pairs = [
            _pair("q1", correct_energy=1.0, incorrect_energy=4.0),
            _pair("q2", correct_energy=2.0, incorrect_energy=3.0),
        ]
        loss = centering.compute_centered_loss(_identity_model, pairs, margin=0.0)
        assert loss == 0.0

    def test_single_pair_loss_equals_margin(self):
        """Single pair: centered_gap=0, loss = max(0, margin - 0) = margin.

        Spec: SCENARIO-LEARN-108
        """
        centering = PROGRSCentering()
        pair = _pair("q1", correct_energy=1.0, incorrect_energy=10.0)
        for margin in [0.5, 1.0, 2.0]:
            loss = centering.compute_centered_loss(_identity_model, [pair], margin=margin)
            assert abs(loss - margin) < 1e-5


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-109: single-pair-per-group centering has no effect on raw loss order
# ---------------------------------------------------------------------------


class TestSinglePairGroupCentering:
    """Tests the SCENARIO-LEARN-109 property: centering with one pair per group is neutral.

    When every question_id appears exactly once (the output of JEPACPMIPairBuilder),
    centered_gap = raw_gap - raw_gap = 0 for every pair.  The loss then equals
    margin * n_pairs / n_pairs = margin.  This is documented behaviour, not a bug.

    Spec: SCENARIO-LEARN-109
    """

    def test_one_pair_per_group_centering_produces_zero_centered_gap(self):
        """PROGRS centering on unique question_ids gives centered_gap=0 for all pairs.

        Spec: SCENARIO-LEARN-109
        """
        centering = PROGRSCentering()
        pairs = [
            _pair(f"q{i}", correct_energy=float(i), incorrect_energy=float(i) + 2.0)
            for i in range(5)
        ]
        result = centering.center_pairs(pairs, model=_identity_model)
        for _, centered_gap in result:
            assert abs(centered_gap) < 1e-5

    def test_progrs_is_importable_from_inference_package(self):
        """PROGRSCentering is exported from carnot.inference.

        Spec: REQ-LEARN-069
        """
        from carnot.inference import PROGRSCentering as PROGRSFromInit  # noqa: PLC0415
        assert PROGRSFromInit is PROGRSCentering
