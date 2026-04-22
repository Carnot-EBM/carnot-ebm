"""Tests for carnot.pipeline.adaptive_gate.ModelAdaptiveThresholdGate (Exp 707).

Covers all public methods plus save/load round-trip so the module
reaches 100% coverage.

Spec: REQ-VERIFY-146, REQ-VERIFY-147, SCENARIO-VERIFY-146, SCENARIO-VERIFY-147
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from carnot.pipeline.adaptive_gate import ModelAdaptiveThresholdGate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gate(tmp_path: Path) -> ModelAdaptiveThresholdGate:
    """Return a fresh gate backed by a temp state file."""
    return ModelAdaptiveThresholdGate(state_file=tmp_path / "gate_state.json")


# ---------------------------------------------------------------------------
# REQ-VERIFY-146-4 / REQ-VERIFY-146-5: update() increments correct counter
# ---------------------------------------------------------------------------


class TestUpdate:
    """REQ-VERIFY-146-4/146-5: update increments tp or fp correctly."""

    def test_update_true_increments_tp(self, tmp_path: Path) -> None:
        """REQ-VERIFY-146-4: was_tp=True increments tp_count."""
        gate = _gate(tmp_path)
        gate.update("model_a", "ArithmeticExtractor", was_tp=True)
        entry = gate.state["model_a"]["ArithmeticExtractor"]
        assert entry["tp"] == 1
        assert entry["fp"] == 0

    def test_update_false_increments_fp(self, tmp_path: Path) -> None:
        """REQ-VERIFY-146-5: was_tp=False increments fp_count."""
        gate = _gate(tmp_path)
        gate.update("model_b", "SymCodeVerifier", was_tp=False)
        entry = gate.state["model_b"]["SymCodeVerifier"]
        assert entry["fp"] == 1
        assert entry["tp"] == 0

    def test_multiple_updates_accumulate(self, tmp_path: Path) -> None:
        """Successive updates accumulate independently."""
        gate = _gate(tmp_path)
        for _ in range(3):
            gate.update("m", "ct", was_tp=True)
        for _ in range(7):
            gate.update("m", "ct", was_tp=False)
        entry = gate.state["m"]["ct"]
        assert entry["tp"] == 3
        assert entry["fp"] == 7

    def test_different_pairs_are_independent(self, tmp_path: Path) -> None:
        """Updates to different (model_id, constraint_type) pairs are isolated."""
        gate = _gate(tmp_path)
        gate.update("model_a", "type_x", was_tp=True)
        gate.update("model_b", "type_x", was_tp=False)
        assert gate.state["model_a"]["type_x"]["tp"] == 1
        assert gate.state["model_b"]["type_x"]["fp"] == 1
        assert gate.state["model_a"]["type_x"]["fp"] == 0


# ---------------------------------------------------------------------------
# REQ-VERIFY-146-1 / REQ-VERIFY-146-2: is_suppressed
# SCENARIO-VERIFY-146
# ---------------------------------------------------------------------------


class TestIsSuppressed:
    """REQ-VERIFY-146-1/146-2: suppression logic is correct."""

    def test_suppressed_when_precision_below_half(self, tmp_path: Path) -> None:
        """REQ-VERIFY-146-1 / SCENARIO-VERIFY-146: precision < 0.5 → suppressed."""
        gate = _gate(tmp_path)
        for _ in range(10):
            gate.update("google/gemma-4-E4B-it", "SymCodeVerifier", was_tp=False)
        assert gate.is_suppressed("google/gemma-4-E4B-it", "SymCodeVerifier") is True

    def test_not_suppressed_when_no_observations(self, tmp_path: Path) -> None:
        """REQ-VERIFY-146-2: zero observations → default allow."""
        gate = _gate(tmp_path)
        assert gate.is_suppressed("Qwen/Qwen3.5-0.8B", "SymCodeVerifier") is False

    def test_not_suppressed_when_precision_above_half(self, tmp_path: Path) -> None:
        """Mostly TPs → precision > 0.5 → not suppressed."""
        gate = _gate(tmp_path)
        for _ in range(8):
            gate.update("model_a", "ArithmeticExtractor", was_tp=True)
        for _ in range(2):
            gate.update("model_a", "ArithmeticExtractor", was_tp=False)
        assert gate.is_suppressed("model_a", "ArithmeticExtractor") is False

    def test_not_suppressed_when_precision_exactly_half(self, tmp_path: Path) -> None:
        """Precision == 0.5 is NOT suppressed (threshold is strict: < 0.5)."""
        gate = _gate(tmp_path)
        gate.update("m", "ct", was_tp=True)
        gate.update("m", "ct", was_tp=False)
        # tp=1, fp=1 → precision = 0.5 → not suppressed
        assert gate.is_suppressed("m", "ct") is False

    def test_unknown_model_not_suppressed(self, tmp_path: Path) -> None:
        """Model with no data is never suppressed."""
        gate = _gate(tmp_path)
        gate.update("known_model", "ct", was_tp=False)
        assert gate.is_suppressed("unknown_model", "ct") is False


# ---------------------------------------------------------------------------
# REQ-VERIFY-146-3: precision() returns 0.5 when no data
# ---------------------------------------------------------------------------


class TestPrecision:
    """REQ-VERIFY-146-3: precision helper returns correct values."""

    def test_precision_no_data_returns_half(self, tmp_path: Path) -> None:
        """REQ-VERIFY-146-3: no observations → neutral prior of 0.5."""
        gate = _gate(tmp_path)
        assert gate.precision("model_a", "ArithmeticExtractor") == pytest.approx(0.5)

    def test_precision_all_fp_returns_zero(self, tmp_path: Path) -> None:
        """All FPs → precision = 0.0."""
        gate = _gate(tmp_path)
        for _ in range(5):
            gate.update("m", "ct", was_tp=False)
        assert gate.precision("m", "ct") == pytest.approx(0.0)

    def test_precision_all_tp_returns_one(self, tmp_path: Path) -> None:
        """All TPs → precision = 1.0."""
        gate = _gate(tmp_path)
        for _ in range(5):
            gate.update("m", "ct", was_tp=True)
        assert gate.precision("m", "ct") == pytest.approx(1.0)

    def test_precision_mixed(self, tmp_path: Path) -> None:
        """3 TP, 1 FP → precision = 0.75."""
        gate = _gate(tmp_path)
        for _ in range(3):
            gate.update("m", "ct", was_tp=True)
        gate.update("m", "ct", was_tp=False)
        assert gate.precision("m", "ct") == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# REQ-VERIFY-147-1 / REQ-VERIFY-147-2 / REQ-VERIFY-147-3: save / load
# SCENARIO-VERIFY-147
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """REQ-VERIFY-147: state persistence across sessions."""

    def test_load_nonexistent_file_is_noop(self, tmp_path: Path) -> None:
        """REQ-VERIFY-147-3: load() with missing file leaves state empty."""
        gate = ModelAdaptiveThresholdGate(state_file=tmp_path / "missing.json")
        gate.load()
        assert gate.state == {}

    def test_save_load_roundtrip_preserves_suppression(self, tmp_path: Path) -> None:
        """REQ-VERIFY-147-1/147-2 / SCENARIO-VERIFY-147: suppression survives save/load."""
        state_file = tmp_path / "gate.json"
        gate1 = ModelAdaptiveThresholdGate(state_file=state_file)
        for _ in range(10):
            gate1.update("google/gemma-4-E4B-it", "SymCodeVerifier", was_tp=False)
        gate1.save()

        gate2 = ModelAdaptiveThresholdGate(state_file=state_file)
        gate2.load()

        assert gate2.is_suppressed("google/gemma-4-E4B-it", "SymCodeVerifier") is True
        assert gate2.is_suppressed("Qwen/Qwen3.5-0.8B", "SymCodeVerifier") is False

    def test_save_creates_valid_json(self, tmp_path: Path) -> None:
        """Saved file is valid JSON with expected structure."""
        state_file = tmp_path / "gate.json"
        gate = ModelAdaptiveThresholdGate(state_file=state_file)
        gate.update("m", "ct", was_tp=True)
        gate.save()

        with state_file.open() as fh:
            data = json.load(fh)
        assert "m" in data
        assert "ct" in data["m"]
        assert data["m"]["ct"]["tp"] == 1

    def test_load_bad_json_raises(self, tmp_path: Path) -> None:
        """Malformed JSON raises json.JSONDecodeError on load."""
        state_file = tmp_path / "bad.json"
        state_file.write_text("{not valid json}")
        gate = ModelAdaptiveThresholdGate(state_file=state_file)
        with pytest.raises(json.JSONDecodeError):
            gate.load()

    def test_load_wrong_type_raises(self, tmp_path: Path) -> None:
        """Non-dict JSON raises ValueError on load."""
        state_file = tmp_path / "bad_type.json"
        state_file.write_text("[1, 2, 3]")
        gate = ModelAdaptiveThresholdGate(state_file=state_file)
        with pytest.raises(ValueError):
            gate.load()

    def test_update_auto_saves(self, tmp_path: Path) -> None:
        """update() persists state without an explicit save() call."""
        state_file = tmp_path / "gate.json"
        gate = ModelAdaptiveThresholdGate(state_file=state_file)
        gate.update("m", "ct", was_tp=False)
        # File should exist immediately after update
        assert state_file.exists()

    def test_save_is_atomic_via_rename(self, tmp_path: Path) -> None:
        """No .tmp files left after a successful save."""
        state_file = tmp_path / "gate.json"
        gate = ModelAdaptiveThresholdGate(state_file=state_file)
        gate.update("m", "ct", was_tp=True)
        gate.save()
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Unexpected temp files: {tmp_files}"
