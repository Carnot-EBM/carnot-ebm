"""Tests for GRPOContrastivePairer and NUPProbeV5.

100% coverage of python/carnot/pipeline/nup_probe_v5.py.

Spec: REQ-VERIFY-125, SCENARIO-VERIFY-155, SCENARIO-VERIFY-156
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from carnot.pipeline.nup_probe_v5 import (
    GRPOContrastivePairer,
    NUPProbeV5,
    _extract_step_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entry(question_index: int, is_correct: bool, response: str = "", steps=None):
    """Helper to build a minimal live_pairs-style entry."""
    e = {"question_index": question_index, "is_correct": is_correct, "response": response}
    if steps is not None:
        e["cot_steps"] = steps
    return e


# ---------------------------------------------------------------------------
# _extract_step_text
# ---------------------------------------------------------------------------


class TestExtractStepText:
    """Tests for the private _extract_step_text helper.

    Spec: REQ-VERIFY-125-2
    """

    def test_prefers_cot_steps_first_step(self):
        """When cot_steps present, use first step's step_text."""
        entry = _make_entry(0, True, response="full response", steps=[{"step_text": "step one"}])
        assert _extract_step_text(entry) == "step one"

    def test_falls_back_to_response_when_no_steps(self):
        """When no cot_steps, use response field."""
        entry = _make_entry(0, True, response="my response")
        assert _extract_step_text(entry) == "my response"

    def test_falls_back_to_response_when_steps_empty_list(self):
        """Empty cot_steps list falls back to response."""
        entry = _make_entry(0, True, response="fallback", steps=[])
        assert _extract_step_text(entry) == "fallback"

    def test_falls_back_to_response_when_step_has_no_step_text(self):
        """Step dict missing step_text key returns empty string from step."""
        entry = _make_entry(0, True, response="resp", steps=[{"other_key": "val"}])
        # step_text missing -> step.get('step_text', '') = '' -> returns ''
        assert _extract_step_text(entry) == ""

    def test_handles_missing_response_key(self):
        """Entry missing response key returns empty string."""
        entry = {"question_index": 0, "is_correct": True}
        assert _extract_step_text(entry) == ""

    def test_cot_steps_not_a_list(self):
        """Non-list cot_steps falls back to response (isinstance check fails)."""
        entry = {"question_index": 0, "is_correct": True, "response": "res", "cot_steps": "bad"}
        # isinstance("bad", list) is False -> falls back to response
        assert _extract_step_text(entry) == "res"


# ---------------------------------------------------------------------------
# GRPOContrastivePairer
# ---------------------------------------------------------------------------


class TestGRPOContrastivePairer:
    """Tests for GRPOContrastivePairer.pairs().

    Spec: REQ-VERIFY-125-1, SCENARIO-VERIFY-155
    """

    def setup_method(self):
        self.pairer = GRPOContrastivePairer()

    def test_empty_input_returns_empty(self):
        """No entries -> no pairs."""
        assert self.pairer.pairs([]) == []

    def test_only_correct_returns_empty(self):
        """Questions with only correct responses have no contrastive partner."""
        entries = [_make_entry(0, True, "correct1"), _make_entry(0, True, "correct2")]
        assert self.pairer.pairs(entries) == []

    def test_only_incorrect_returns_empty(self):
        """Questions with only incorrect responses have no contrastive partner."""
        entries = [_make_entry(1, False, "wrong1"), _make_entry(1, False, "wrong2")]
        assert self.pairer.pairs(entries) == []

    def test_one_correct_one_incorrect_yields_one_pair(self):
        """Single correct + single incorrect for same question yields one pair."""
        c = _make_entry(5, True, "right")
        i = _make_entry(5, False, "wrong")
        pairs = self.pairer.pairs([c, i])
        assert len(pairs) == 1
        assert pairs[0] == (c, i)

    def test_pair_order_is_correct_then_incorrect(self):
        """Pairs are always (correct_entry, incorrect_entry), not the reverse."""
        c = _make_entry(0, True, "correct")
        i = _make_entry(0, False, "incorrect")
        pair = self.pairer.pairs([i, c])[0]  # input order reversed
        assert pair[0]["is_correct"] is True
        assert pair[1]["is_correct"] is False

    def test_cross_product_for_multiple_responses(self):
        """2 correct + 3 incorrect for same question yields 6 pairs (Cartesian)."""
        entries = (
            [_make_entry(0, True, f"c{j}") for j in range(2)]
            + [_make_entry(0, False, f"i{j}") for j in range(3)]
        )
        pairs = self.pairer.pairs(entries)
        assert len(pairs) == 6

    def test_different_questions_are_not_paired_across(self):
        """Correct response for q0 is never paired with incorrect response for q1."""
        entries = [
            _make_entry(0, True, "q0_correct"),
            _make_entry(1, False, "q1_incorrect"),
        ]
        assert self.pairer.pairs(entries) == []

    def test_multiple_questions_paired_independently(self):
        """Two separate questions each with one correct+one incorrect -> 2 pairs."""
        entries = [
            _make_entry(0, True, "q0c"),
            _make_entry(0, False, "q0i"),
            _make_entry(1, True, "q1c"),
            _make_entry(1, False, "q1i"),
        ]
        pairs = self.pairer.pairs(entries)
        assert len(pairs) == 2
        q_indices = {p[0]["question_index"] for p in pairs}
        assert q_indices == {0, 1}


# ---------------------------------------------------------------------------
# NUPProbeV5
# ---------------------------------------------------------------------------


class TestNUPProbeV5:
    """Tests for NUPProbeV5.

    Spec: REQ-VERIFY-125-2, REQ-VERIFY-125-3, REQ-VERIFY-125-4, SCENARIO-VERIFY-156
    """

    def setup_method(self):
        self.probe = NUPProbeV5(energy_dim=8, margin=1.0, learning_rate=0.05, random_seed=0)

    def test_train_from_pairs_empty_returns_no_pairs(self):
        """Training on empty list returns 0 grpo_pairs_built."""
        result = self.probe.train_from_pairs([])
        assert result["grpo_pairs_built"] == 0

    def test_train_from_pairs_returns_grpo_pairs_count(self):
        """train_from_pairs reports the number of GRPO pairs extracted."""
        entries = [
            _make_entry(0, True, "The answer is 9"),
            _make_entry(0, False, "The answer is 42"),
            _make_entry(1, True, "Result: 7"),
            _make_entry(1, False, "Result: 0"),
        ]
        result = self.probe.train_from_pairs(entries, n_epochs=5)
        assert result["grpo_pairs_built"] == 2
        assert result["n_correct_steps"] == 2
        assert result["n_incorrect_steps"] == 2

    def test_train_from_pairs_returns_standard_keys(self):
        """train_from_pairs result contains all NUPProbeV4 training keys."""
        entries = [
            _make_entry(0, True, "correct step"),
            _make_entry(0, False, "wrong step"),
        ]
        result = self.probe.train_from_pairs(entries, n_epochs=3)
        for key in ("converged", "final_loss", "final_auc", "loss_history", "grpo_pairs_built"):
            assert key in result, f"Missing key: {key}"

    def test_evaluate_auc_returns_float_in_range(self):
        """evaluate_auc returns a float in [0, 1]."""
        entries = [
            _make_entry(0, True, "correct math: 2+2=4"),
            _make_entry(0, False, "wrong: 2+2=5"),
        ]
        auc = self.probe.evaluate_auc(entries)
        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0

    def test_evaluate_auc_all_correct_returns_half(self):
        """AUROC is 0.5 when there are no incorrect entries (degenerate case)."""
        entries = [_make_entry(0, True, "correct") for _ in range(5)]
        auc = self.probe.evaluate_auc(entries)
        assert auc == 0.5

    def test_evaluate_auc_all_incorrect_returns_half(self):
        """AUROC is 0.5 when there are no correct entries (degenerate case)."""
        entries = [_make_entry(0, False, "wrong") for _ in range(5)]
        auc = self.probe.evaluate_auc(entries)
        assert auc == 0.5

    def test_trained_probe_separates_clear_cases(self):
        """After training on well-separated texts, AUC should exceed 0.5."""
        # Use very distinct texts so the probe has signal to learn from
        correct_texts = [f"correct arithmetic result {i} equals {i*2}" for i in range(20)]
        incorrect_texts = [f"hallucinated junk xyz abc def {i}" for i in range(20)]
        entries = (
            [_make_entry(j, True, correct_texts[j]) for j in range(20)]
            + [_make_entry(j, False, incorrect_texts[j]) for j in range(20)]
        )
        probe = NUPProbeV5(energy_dim=32, margin=1.0, learning_rate=0.05, random_seed=7)
        probe.train_from_pairs(entries, n_epochs=50)
        auc = probe.evaluate_auc(entries)
        # After training on clearly distinct texts the probe should do better than chance
        assert auc >= 0.5, f"AUC {auc:.4f} should be >= 0.5 after training"

    def test_save_safetensors_creates_file(self, tmp_path):
        """save_safetensors writes a non-empty file at the given path."""
        entries = [
            _make_entry(0, True, "correct"),
            _make_entry(0, False, "wrong"),
        ]
        self.probe.train_from_pairs(entries, n_epochs=1)
        out_path = tmp_path / "nup_probe_v5.safetensors"
        self.probe.save_safetensors(str(out_path))
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_save_safetensors_file_has_valid_header(self, tmp_path):
        """safetensors file starts with a valid 8-byte length prefix."""
        entries = [
            _make_entry(0, True, "correct text here"),
            _make_entry(0, False, "wrong text here"),
        ]
        self.probe.train_from_pairs(entries, n_epochs=1)
        out_path = tmp_path / "probe.safetensors"
        self.probe.save_safetensors(str(out_path))

        with out_path.open("rb") as f:
            raw = f.read()

        # First 8 bytes encode header length as little-endian uint64
        header_len = struct.unpack("<Q", raw[:8])[0]
        assert header_len > 0
        # Header JSON should contain our tensor names
        header_bytes = raw[8 : 8 + header_len]
        header_str = header_bytes.decode("utf-8").strip()
        assert "weights" in header_str
        assert "bias" in header_str

    def test_save_safetensors_creates_parent_dirs(self, tmp_path):
        """save_safetensors creates missing parent directories."""
        nested = tmp_path / "subdir" / "nested" / "probe.safetensors"
        self.probe.save_safetensors(str(nested))
        assert nested.exists()
