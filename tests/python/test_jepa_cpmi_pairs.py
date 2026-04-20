"""Tests for jepa_cpmi_pairs — CPMI contrastive pair builder.

Spec: REQ-LEARN-065, REQ-LEARN-066,
      SCENARIO-LEARN-101, SCENARIO-LEARN-102, SCENARIO-LEARN-103
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp
import pytest

from carnot.inference.jepa_cpmi_pairs import (
    CPMIContrastiveLoss,
    JEPACPMIPair,
    JEPACPMIPairBuilder,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeEntry:
    """Minimal FOVERCorpusEntry-like object for testing."""
    question: str
    is_correct: bool
    cot_steps: list = field(default_factory=list)
    response: str = ""
    model_id: str = "test_model"
    constraint_types: list = field(default_factory=list)


def _dummy_embed(text: str) -> jnp.ndarray:
    """Hash-based embed: deterministic, no ML needed."""
    return jnp.array([hash(text) % 128], dtype=jnp.float32)


def _const_model(value: float):
    """Returns a model that always outputs a fixed scalar."""
    def _model(emb: jnp.ndarray) -> float:
        return value
    return _model


# ---------------------------------------------------------------------------
# JEPACPMIPairBuilder.build_pairs
# ---------------------------------------------------------------------------


class TestBuildPairsGrouping:
    """SCENARIO-LEARN-101: build_pairs groups by question_id correctly."""

    def test_single_question_yields_one_pair(self):
        entries = [
            _FakeEntry("q1", is_correct=True,  cot_steps=[{"step_text": "a"}]),
            _FakeEntry("q1", is_correct=False, cot_steps=[{"step_text": "b"}]),
        ]
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_pairs(entries)
        assert len(pairs) == 1
        assert pairs[0].question_id == "q1"

    def test_two_questions_yield_two_pairs(self):
        entries = [
            _FakeEntry("q1", is_correct=True,  cot_steps=[{"step_text": "a"}]),
            _FakeEntry("q1", is_correct=False, cot_steps=[{"step_text": "b"}]),
            _FakeEntry("q2", is_correct=True,  cot_steps=[{"step_text": "c"}]),
            _FakeEntry("q2", is_correct=False, cot_steps=[{"step_text": "d"}]),
        ]
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_pairs(entries)
        assert len(pairs) == 2
        qids = {p.question_id for p in pairs}
        assert qids == {"q1", "q2"}

    def test_only_correct_entries_skipped(self):
        entries = [
            _FakeEntry("q1", is_correct=True, cot_steps=[{"step_text": "a"}]),
            _FakeEntry("q1", is_correct=True, cot_steps=[{"step_text": "b"}]),
        ]
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_pairs(entries)
        assert len(pairs) == 0

    def test_only_incorrect_entries_skipped(self):
        entries = [
            _FakeEntry("q1", is_correct=False, cot_steps=[{"step_text": "a"}]),
        ]
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_pairs(entries)
        assert len(pairs) == 0

    def test_empty_corpus_yields_no_pairs(self):
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_pairs([])
        assert pairs == []

    def test_pair_has_embeddings(self):
        entries = [
            _FakeEntry("q1", is_correct=True,  cot_steps=[{"step_text": "step A"}, {"step_text": "step B"}]),
            _FakeEntry("q1", is_correct=False, cot_steps=[{"step_text": "step C"}]),
        ]
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_pairs(entries)
        assert len(pairs[0].correct_embeddings) == 2
        assert len(pairs[0].incorrect_embeddings) == 1

    def test_pair_quality_computed(self):
        entries = [
            _FakeEntry("q1", is_correct=True,  cot_steps=[{"step_text": "a"}, {"step_text": "b"}]),
            _FakeEntry("q1", is_correct=False, cot_steps=[{"step_text": "c"}, {"step_text": "d"}, {"step_text": "e"}]),
        ]
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_pairs(entries)
        # pair_quality = len(correct_steps) / len(incorrect_steps) = 2/3
        assert abs(pairs[0].pair_quality - 2 / 3) < 1e-6


class TestHardNegativeSelection:
    """SCENARIO-LEARN-101: hardest incorrect = entry with most cot_steps."""

    def test_hardest_incorrect_picked_by_step_count(self):
        # incorrect_short has 1 step, incorrect_long has 3 steps
        entries = [
            _FakeEntry("q1", is_correct=True,  cot_steps=[{"step_text": "ok"}]),
            _FakeEntry("q1", is_correct=False, cot_steps=[{"step_text": "bad"}], model_id="short"),
            _FakeEntry("q1", is_correct=False, cot_steps=[
                {"step_text": "bad1"}, {"step_text": "bad2"}, {"step_text": "bad3"}
            ], model_id="long"),
        ]
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_pairs(entries)
        # hardest incorrect has 3 embeddings
        assert len(pairs[0].incorrect_embeddings) == 3

    def test_hard_negative_step_idx_is_last_step(self):
        entries = [
            _FakeEntry("q1", is_correct=True,  cot_steps=[{"step_text": "ok"}]),
            _FakeEntry("q1", is_correct=False, cot_steps=[
                {"step_text": "w1"}, {"step_text": "w2"}, {"step_text": "w3"},
            ]),
        ]
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_pairs(entries)
        assert pairs[0].hard_negative_step_idx == 2  # index of last step

    def test_no_steps_fallback_to_empty_string_embedding(self):
        """Entry with no cot_steps still gets one embedding (empty string fallback)."""
        entries = [
            _FakeEntry("q1", is_correct=True,  cot_steps=[]),
            _FakeEntry("q1", is_correct=False, cot_steps=[]),
        ]
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_pairs(entries)
        assert len(pairs) == 1
        assert len(pairs[0].correct_embeddings) == 1
        assert len(pairs[0].incorrect_embeddings) == 1

    def test_non_dict_step_converted_to_string(self):
        """Steps that are plain strings (not dicts) are handled via str()."""
        entries = [
            _FakeEntry("q1", is_correct=True,  cot_steps=["step text as string"]),
            _FakeEntry("q1", is_correct=False, cot_steps=["wrong step as string"]),
        ]
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_pairs(entries)
        assert len(pairs) == 1
        assert len(pairs[0].correct_embeddings) == 1


# ---------------------------------------------------------------------------
# CPMIContrastiveLoss
# ---------------------------------------------------------------------------


class TestCPMIContrastiveLossZeroGap:
    """SCENARIO-LEARN-102: loss returns 0.0 when gap > margin."""

    def _make_pair(self) -> JEPACPMIPair:
        emb = jnp.array([1.0])
        return JEPACPMIPair(
            question_id="q1",
            correct_embeddings=[emb],
            incorrect_embeddings=[emb],
            hard_negative_step_idx=0,
            pair_quality=1.0,
        )

    def test_zero_loss_when_gap_exceeds_margin(self):
        # correct_model returns 0.0 for correct, 5.0 for incorrect
        # gap = 5.0 - 0.0 = 5.0 > margin=1.0 → loss=0
        call_count = [0]

        def _model(emb: jnp.ndarray) -> float:
            call_count[0] += 1
            # first call: correct chain; second call: incorrect chain
            return 0.0 if call_count[0] % 2 == 1 else 5.0

        loss_fn = CPMIContrastiveLoss(margin=1.0)
        pair = self._make_pair()
        loss = loss_fn.compute_loss(_model, [pair])
        assert loss == 0.0

    def test_positive_loss_when_gap_less_than_margin(self):
        # correct=2.0, incorrect=2.5 → gap=0.5 < margin=1.0 → loss=0.5
        calls = [0]

        def _model(emb):
            calls[0] += 1
            return 2.0 if calls[0] % 2 == 1 else 2.5

        loss_fn = CPMIContrastiveLoss(margin=1.0)
        pair = self._make_pair()
        loss = loss_fn.compute_loss(_model, [pair])
        assert loss > 0.0
        assert abs(loss - 0.5) < 1e-5

    def test_loss_non_negative(self):
        """Loss must always be >= 0."""
        loss_fn = CPMIContrastiveLoss(margin=1.0)
        pair = self._make_pair()
        loss = loss_fn.compute_loss(_const_model(0.0), [pair])
        assert loss >= 0.0


class TestCPMIContrastiveLossEmpty:
    """SCENARIO-LEARN-103: empty pair list returns 0.0."""

    def test_empty_pairs_returns_zero(self):
        loss_fn = CPMIContrastiveLoss(margin=1.0)
        loss = loss_fn.compute_loss(_const_model(0.0), [])
        assert loss == 0.0

    def test_zero_if_empty_returns_zero(self):
        loss_fn = CPMIContrastiveLoss(margin=1.0)
        assert loss_fn.zero_if_empty([]) == 0.0
        assert loss_fn.zero_if_empty(["anything"]) == 0.0


class TestChainEnergyModes:
    """chain_energy aggregation modes produce correct values."""

    def _make_model(self, scores):
        idx = [0]

        def _m(emb):
            v = scores[idx[0] % len(scores)]
            idx[0] += 1
            return v

        return _m

    def test_mean_mode(self):
        emb = jnp.array([0.0])
        embeddings = [emb, emb, emb]
        scores = [1.0, 3.0, 5.0]
        loss_fn = CPMIContrastiveLoss(chain_energy_mode="mean")
        result = loss_fn.chain_energy(self._make_model(scores), embeddings)
        assert abs(result - 3.0) < 1e-5

    def test_max_mode(self):
        emb = jnp.array([0.0])
        embeddings = [emb, emb, emb]
        scores = [1.0, 3.0, 5.0]
        loss_fn = CPMIContrastiveLoss(chain_energy_mode="max")
        result = loss_fn.chain_energy(self._make_model(scores), embeddings)
        assert abs(result - 5.0) < 1e-5

    def test_min_mode(self):
        emb = jnp.array([0.0])
        embeddings = [emb, emb, emb]
        scores = [1.0, 3.0, 5.0]
        loss_fn = CPMIContrastiveLoss(chain_energy_mode="min")
        result = loss_fn.chain_energy(self._make_model(scores), embeddings)
        assert abs(result - 1.0) < 1e-5

    def test_empty_embeddings_returns_zero(self):
        loss_fn = CPMIContrastiveLoss(chain_energy_mode="mean")
        result = loss_fn.chain_energy(_const_model(99.0), [])
        assert result == 0.0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="chain_energy_mode"):
            CPMIContrastiveLoss(chain_energy_mode="invalid")


# ---------------------------------------------------------------------------
# build_synthetic_pairs
# ---------------------------------------------------------------------------


class TestBuildSyntheticPairs:
    """SCENARIO-LEARN-103: build_synthetic_pairs returns exactly n_pairs."""

    def test_returns_requested_count(self):
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_synthetic_pairs(10)
        assert len(pairs) == 10

    def test_zero_pairs(self):
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_synthetic_pairs(0)
        assert pairs == []

    def test_synthetic_question_ids_are_unique(self):
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_synthetic_pairs(5)
        qids = [p.question_id for p in pairs]
        assert len(set(qids)) == 5

    def test_synthetic_pair_quality_is_one(self):
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_synthetic_pairs(3)
        for pair in pairs:
            assert pair.pair_quality == 1.0

    def test_synthetic_has_two_steps_each(self):
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_synthetic_pairs(2)
        for pair in pairs:
            assert len(pair.correct_embeddings) == 2
            assert len(pair.incorrect_embeddings) == 2

    def test_synthetic_hard_negative_idx_set(self):
        builder = JEPACPMIPairBuilder(embed_fn=_dummy_embed)
        pairs = builder.build_synthetic_pairs(1)
        assert pairs[0].hard_negative_step_idx == 1  # last of 2 steps


# ---------------------------------------------------------------------------
# Mean loss over multiple pairs
# ---------------------------------------------------------------------------


class TestMeanLossMultiplePairs:
    """Mean loss is computed correctly over multiple pairs."""

    def test_mean_of_two_pairs(self):
        # pair1: gap=0.0 → loss=1.0 (margin=1.0)
        # pair2: gap=2.0 → loss=0.0
        # mean = 0.5
        emb = jnp.array([0.0])
        pair1 = JEPACPMIPair("q1", [emb], [emb], None, 1.0)
        pair2 = JEPACPMIPair("q2", [emb], [emb], None, 1.0)

        call_log = [0]

        def _model(e):
            call_log[0] += 1
            # pair1 correct=0, incorrect=0 → gap=0 → loss=1.0
            # pair2 correct=0, incorrect=2 → gap=2 → loss=0.0
            if call_log[0] <= 2:
                return 0.0
            elif call_log[0] == 3:
                return 0.0   # correct for pair2
            else:
                return 2.0   # incorrect for pair2

        loss_fn = CPMIContrastiveLoss(margin=1.0, chain_energy_mode="mean")
        loss = loss_fn.compute_loss(_model, [pair1, pair2])
        assert abs(loss - 0.5) < 1e-5
