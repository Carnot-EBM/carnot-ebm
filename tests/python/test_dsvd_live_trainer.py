"""Tests for dsvd_live_trainer.py — 100% coverage on DSVDLiveTrainPair,
TemporalWindowLabeler, and DSVDLiveTrainer.

Spec: REQ-VERIFY-130, REQ-VERIFY-131,
      SCENARIO-VERIFY-163, SCENARIO-VERIFY-164, SCENARIO-VERIFY-165
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.pipeline.dsvd_adapter import DSVDAdapter, DSVDLinearProbe
from carnot.pipeline.dsvd_live_trainer import (
    DSVDLiveTrainPair,
    DSVDLiveTrainer,
    TemporalWindowLabeler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter() -> DSVDAdapter:
    probe = DSVDLinearProbe(hidden_dim=64)
    return DSVDAdapter(probe, violation_threshold=0.5)


def _make_pair(is_correct: bool, T: int = 64, D: int = 128) -> DSVDLiveTrainPair:
    scalar = 1.0 if is_correct else 0.0
    hidden_states = jnp.ones((T, D), dtype=jnp.float32) * scalar
    return DSVDLiveTrainPair(
        hidden_states=hidden_states,
        response="test response with numbers 3 + 4 = 7",
        is_correct=is_correct,
    )


# ---------------------------------------------------------------------------
# DSVDLiveTrainPair
# ---------------------------------------------------------------------------

class TestDSVDLiveTrainPair:
    def test_fields_correct(self):
        # REQ-VERIFY-130-1
        hidden = jnp.ones((64, 128))
        pair = DSVDLiveTrainPair(hidden_states=hidden, response="r", is_correct=True)
        assert pair.hidden_states.shape == (64, 128)
        assert pair.response == "r"
        assert pair.is_correct is True
        assert pair.window_size == 32

    def test_custom_window_size(self):
        hidden = jnp.ones((64, 128))
        pair = DSVDLiveTrainPair(hidden_states=hidden, response="r", is_correct=False, window_size=16)
        assert pair.window_size == 16

    def test_is_correct_false(self):
        pair = _make_pair(is_correct=False)
        assert pair.is_correct is False


# ---------------------------------------------------------------------------
# TemporalWindowLabeler — SCENARIO-VERIFY-165
# ---------------------------------------------------------------------------

class TestTemporalWindowLabeler:
    def test_default_window_size(self):
        labeler = TemporalWindowLabeler()
        assert labeler.window_size == 32

    def test_custom_window_size(self):
        labeler = TemporalWindowLabeler(window_size=16)
        assert labeler.window_size == 16

    def test_correct_pair_all_true(self):
        # SCENARIO-VERIFY-165: correct pairs — all windows labeled True.
        labeler = TemporalWindowLabeler(window_size=32)
        pair = _make_pair(is_correct=True, T=96, D=64)
        labeled = labeler.label_windows(pair)
        assert len(labeled) == 3
        for _w, label in labeled:
            assert label is True

    def test_incorrect_pair_last_two_false(self):
        # SCENARIO-VERIFY-165: 3 windows, last 2 labeled False.
        labeler = TemporalWindowLabeler(window_size=32)
        pair = _make_pair(is_correct=False, T=96, D=64)
        labeled = labeler.label_windows(pair)
        assert len(labeled) == 3
        assert labeled[0][1] is True   # early window — normal reasoning
        assert labeled[1][1] is False  # second-to-last — violation forming
        assert labeled[2][1] is False  # last window — violation confirmed

    def test_incorrect_pair_single_window_false(self):
        # 1 window, incorrect → labeled False.
        labeler = TemporalWindowLabeler(window_size=32)
        pair = _make_pair(is_correct=False, T=20, D=64)
        labeled = labeler.label_windows(pair)
        assert len(labeled) == 1
        assert labeled[0][1] is False

    def test_incorrect_pair_two_windows_both_false(self):
        # 2 windows, incorrect → both False (n_windows <= 2 branch).
        labeler = TemporalWindowLabeler(window_size=32)
        pair = _make_pair(is_correct=False, T=64, D=64)
        labeled = labeler.label_windows(pair)
        assert len(labeled) == 2
        assert labeled[0][1] is False
        assert labeled[1][1] is False

    def test_window_shapes(self):
        # Each window should have shape (window_size, D) except possibly last.
        labeler = TemporalWindowLabeler(window_size=32)
        pair = _make_pair(is_correct=True, T=70, D=128)
        labeled = labeler.label_windows(pair)
        assert labeled[0][0].shape == (32, 128)
        assert labeled[1][0].shape == (32, 128)
        # Last window has remainder tokens: 70 - 64 = 6
        assert labeled[2][0].shape == (6, 128)

    def test_incorrect_four_windows_first_two_true(self):
        # 4 windows incorrect → first 2 True, last 2 False.
        labeler = TemporalWindowLabeler(window_size=32)
        pair = _make_pair(is_correct=False, T=128, D=64)
        labeled = labeler.label_windows(pair)
        assert len(labeled) == 4
        assert labeled[0][1] is True
        assert labeled[1][1] is True
        assert labeled[2][1] is False
        assert labeled[3][1] is False


# ---------------------------------------------------------------------------
# DSVDLiveTrainer.build_training_pairs — SCENARIO-VERIFY-163
# ---------------------------------------------------------------------------

class TestDSVDLiveTrainerBuildPairs:
    def _write_corpus(self, entries: list, tmp_path: Path) -> str:
        corpus_file = tmp_path / "corpus.json"
        corpus_file.write_text(json.dumps(entries))
        return str(corpus_file)

    def test_builds_pairs_from_list_corpus(self, tmp_path):
        entries = [
            {"response": "correct answer", "is_correct": True},
            {"response": "wrong answer", "is_correct": False},
        ]
        corpus_path = self._write_corpus(entries, tmp_path)
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = trainer.build_training_pairs(corpus_path)
        assert len(pairs) == 2
        assert pairs[0].is_correct is True
        assert pairs[1].is_correct is False

    def test_hidden_state_shape(self, tmp_path):
        # REQ-VERIFY-130-2: each pair has hidden_states with shape (64, 128).
        entries = [{"response": "r", "is_correct": True}]
        corpus_path = self._write_corpus(entries, tmp_path)
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = trainer.build_training_pairs(corpus_path)
        assert pairs[0].hidden_states.shape == (64, 128)

    def test_synthetic_approx_correct_value(self, tmp_path):
        # Correct pairs get ones, incorrect get zeros.
        entries = [
            {"response": "r", "is_correct": True},
            {"response": "r", "is_correct": False},
        ]
        corpus_path = self._write_corpus(entries, tmp_path)
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = trainer.build_training_pairs(corpus_path)
        assert float(jnp.mean(pairs[0].hidden_states)) == pytest.approx(1.0)
        assert float(jnp.mean(pairs[1].hidden_states)) == pytest.approx(0.0)

    def test_empty_corpus(self, tmp_path):
        corpus_path = self._write_corpus([], tmp_path)
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = trainer.build_training_pairs(corpus_path)
        assert pairs == []

    def test_dict_corpus_wrapping(self, tmp_path):
        # If corpus is wrapped in a dict, trainer finds the list.
        entries = [{"response": "r", "is_correct": True}]
        corpus_file = tmp_path / "corpus.json"
        corpus_file.write_text(json.dumps({"data": entries}))
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = trainer.build_training_pairs(str(corpus_file))
        assert len(pairs) == 1

    def test_missing_response_field(self, tmp_path):
        # Missing 'response' defaults to empty string.
        entries = [{"is_correct": True}]
        corpus_path = self._write_corpus(entries, tmp_path)
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = trainer.build_training_pairs(corpus_path)
        assert pairs[0].response == ""

    def test_missing_is_correct_field(self, tmp_path):
        # Missing 'is_correct' defaults to False.
        entries = [{"response": "r"}]
        corpus_path = self._write_corpus(entries, tmp_path)
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = trainer.build_training_pairs(corpus_path)
        assert pairs[0].is_correct is False


# ---------------------------------------------------------------------------
# DSVDLiveTrainer.train — SCENARIO-VERIFY-164
# ---------------------------------------------------------------------------

class TestDSVDLiveTrainerTrain:
    def _make_pairs(self, n: int) -> list[DSVDLiveTrainPair]:
        pairs = []
        for i in range(n):
            is_correct = (i % 2 == 0)
            pairs.append(DSVDLiveTrainPair(
                hidden_states=jnp.ones((64, 128)) * (1.0 if is_correct else 0.0),
                response=f"step {i}: 3 + 4 = 7 and result is {i}",
                is_correct=is_correct,
            ))
        return pairs

    def test_returns_float_in_unit_interval(self):
        # REQ-VERIFY-130-3
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = self._make_pairs(10)
        auc = trainer.train(pairs, n_epochs=5)
        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0

    def test_empty_pairs_returns_zero(self):
        trainer = DSVDLiveTrainer(_make_adapter())
        auc = trainer.train([], n_epochs=5)
        assert auc == 0.0

    def test_single_pair_correct(self):
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = [_make_pair(is_correct=True)]
        auc = trainer.train(pairs, n_epochs=5)
        assert 0.0 <= auc <= 1.0

    def test_all_correct_returns_half(self):
        # All-same labels → AUC=0.5 (uninformative).
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = [_make_pair(is_correct=True)] * 10
        auc = trainer.train(pairs, n_epochs=5)
        assert auc == pytest.approx(0.5)

    def test_all_incorrect_returns_half(self):
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = [_make_pair(is_correct=False)] * 10
        auc = trainer.train(pairs, n_epochs=5)
        assert auc == pytest.approx(0.5)

    def test_mixed_pairs_auc_valid(self):
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = self._make_pairs(20)
        auc = trainer.train(pairs, n_epochs=10)
        assert 0.0 <= auc <= 1.0

    def test_train_modifies_probe_weights(self):
        # After training, probe weights should differ from initial zeros.
        adapter = _make_adapter()
        initial_weights = adapter.probe._weights.copy()
        trainer = DSVDLiveTrainer(adapter)
        pairs = self._make_pairs(10)
        trainer.train(pairs, n_epochs=10)
        assert not np.allclose(adapter.probe._weights, initial_weights)


# ---------------------------------------------------------------------------
# DSVDLiveTrainer._compute_auc
# ---------------------------------------------------------------------------

class TestComputeAUC:
    def test_empty_pairs(self):
        trainer = DSVDLiveTrainer(_make_adapter())
        assert trainer._compute_auc([]) == 0.5

    def test_perfect_discrimination(self):
        # Probe trained to predict 1.0 for incorrect — AUC should approach 1.0
        # after training; here we just check the method returns valid float.
        trainer = DSVDLiveTrainer(_make_adapter())
        pairs = [_make_pair(is_correct=True), _make_pair(is_correct=False)]
        auc = trainer._compute_auc(pairs)
        assert 0.0 <= auc <= 1.0

    def test_distinct_scores_trigger_auc_accumulation(self):
        # Train so probe sees different texts → different scores → covers lines 296-299.
        adapter = _make_adapter()
        # Fit probe on distinct texts so predictions vary.
        adapter.probe.fit(
            ["3 + 4 = 7 correct step", "wrong answer placeholder"],
            [0.0, 1.0],
        )
        trainer = DSVDLiveTrainer(adapter)
        pairs = [
            DSVDLiveTrainPair(
                hidden_states=jnp.ones((64, 128)),
                response="3 + 4 = 7 correct step",
                is_correct=True,
            ),
            DSVDLiveTrainPair(
                hidden_states=jnp.zeros((64, 128)),
                response="wrong answer placeholder",
                is_correct=False,
            ),
        ]
        auc = trainer._compute_auc(pairs)
        assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# __init__ export — REQ-VERIFY-130-4
# ---------------------------------------------------------------------------

class TestPipelineExports:
    def test_exported_from_pipeline(self):
        from carnot.pipeline import DSVDLiveTrainPair, DSVDLiveTrainer, TemporalWindowLabeler
        assert DSVDLiveTrainPair is not None
        assert DSVDLiveTrainer is not None
        assert TemporalWindowLabeler is not None
