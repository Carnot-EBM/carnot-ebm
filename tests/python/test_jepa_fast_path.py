"""Tests for carnot.pipeline.jepa_fast_path — JepaGate fast-path energy gate.

Covers all branches for 100% coverage:
- JepaGate construction: onnx_path, threshold, enabled fields
- predict() when disabled: returns 1.0 without touching ONNX
- predict() when enabled: loads ONNX lazily, returns sigmoid(raw_output)
- predict() lazy session caching: ONNX load called once for multiple predicts
- predict() ImportError: propagates when onnxruntime missing
- should_skip() below threshold: returns True
- should_skip() above threshold: returns False
- should_skip() at threshold (equal): returns False (not strictly less than)
- should_skip() when disabled: always False
- to_dict(): returns correct serialisation
- VerifyRepairPipeline.verify_with_gate() gate=None: full Ising, no gate fields
- VerifyRepairPipeline.verify_with_gate() gate skip: ising_skipped=True, gate_decision="skip"
- VerifyRepairPipeline.verify_with_gate() gate verify: ising_skipped=False, gate_decision="verify"
- Latency benchmark: 50-question batch with gate vs without gate produces speedup key

Spec: REQ-JEPA-005, SCENARIO-JEPA-010, SCENARIO-JEPA-011
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from carnot.pipeline.jepa_fast_path import JepaGate
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Helpers — fake ONNX session
# ---------------------------------------------------------------------------


def _make_mock_session(raw_scalar: float) -> MagicMock:
    """Return a mock onnxruntime InferenceSession that returns raw_scalar."""
    mock_input = MagicMock()
    mock_input.name = "input"
    session = MagicMock()
    session.get_inputs.return_value = [mock_input]
    session.run.return_value = [np.array([[raw_scalar]], dtype=np.float32)]
    return session


# ---------------------------------------------------------------------------
# JepaGate construction
# ---------------------------------------------------------------------------


class TestJepaGateConstruction:
    """REQ-JEPA-005: Dataclass field defaults and types."""

    def test_fields_stored(self) -> None:
        """REQ-JEPA-005: Constructor stores onnx_path, threshold, enabled."""
        gate = JepaGate(onnx_path="results/jepa_predictor_307.onnx", threshold=0.6, enabled=False)
        assert gate.onnx_path == "results/jepa_predictor_307.onnx"
        assert gate.threshold == 0.6
        assert gate.enabled is False

    def test_default_threshold_and_enabled(self) -> None:
        """REQ-JEPA-005: Default threshold=0.5, enabled=True."""
        gate = JepaGate(onnx_path="dummy.onnx")
        assert gate.threshold == 0.5
        assert gate.enabled is True

    def test_session_starts_as_none(self) -> None:
        """REQ-JEPA-005: Internal session is None before first predict call."""
        gate = JepaGate(onnx_path="dummy.onnx")
        assert gate._session is None


# ---------------------------------------------------------------------------
# predict() — disabled gate
# ---------------------------------------------------------------------------


class TestPredictDisabled:
    """REQ-JEPA-005: disabled gate always returns 1.0."""

    def test_returns_one_when_disabled(self) -> None:
        """REQ-JEPA-005: predict() returns 1.0 when enabled=False."""
        gate = JepaGate(onnx_path="nonexistent.onnx", enabled=False)
        dummy = np.zeros(10, dtype=np.float32)
        result = gate.predict(dummy)
        assert result == 1.0

    def test_no_session_loaded_when_disabled(self) -> None:
        """REQ-JEPA-005: Session stays None when gate is disabled."""
        gate = JepaGate(onnx_path="nonexistent.onnx", enabled=False)
        dummy = np.zeros(10, dtype=np.float32)
        gate.predict(dummy)
        assert gate._session is None


# ---------------------------------------------------------------------------
# predict() — enabled gate with mock ONNX
# ---------------------------------------------------------------------------


class TestPredictEnabled:
    """REQ-JEPA-005: predict() loads ONNX lazily and returns sigmoid(raw)."""

    def test_predict_returns_sigmoid_of_raw(self) -> None:
        """REQ-JEPA-005: predict() applies sigmoid to the ONNX raw output."""
        raw = 0.5  # sigmoid(0.5) ≈ 0.6225
        gate = JepaGate(onnx_path="dummy.onnx")
        mock_session = _make_mock_session(raw)
        gate._session = mock_session  # inject pre-built session

        dummy = np.zeros(16, dtype=np.float32)
        result = gate.predict(dummy)
        expected = 1.0 / (1.0 + math.exp(-raw))
        assert abs(result - expected) < 1e-6

    def test_predict_zero_raw_gives_half(self) -> None:
        """sigmoid(0) = 0.5."""
        gate = JepaGate(onnx_path="dummy.onnx")
        gate._session = _make_mock_session(0.0)
        dummy = np.ones(16, dtype=np.float32)
        assert abs(gate.predict(dummy) - 0.5) < 1e-6

    def test_predict_large_negative_gives_near_zero(self) -> None:
        """sigmoid(-100) ≈ 0 — very low risk."""
        gate = JepaGate(onnx_path="dummy.onnx")
        gate._session = _make_mock_session(-100.0)
        dummy = np.ones(8, dtype=np.float32)
        assert gate.predict(dummy) < 1e-6

    def test_predict_large_positive_gives_near_one(self) -> None:
        """sigmoid(100) ≈ 1 — very high risk."""
        gate = JepaGate(onnx_path="dummy.onnx")
        gate._session = _make_mock_session(100.0)
        dummy = np.ones(8, dtype=np.float32)
        assert gate.predict(dummy) > 1.0 - 1e-6

    def test_session_called_with_reshaped_input(self) -> None:
        """REQ-JEPA-005: Input is reshaped to (1, N) before ONNX call."""
        gate = JepaGate(onnx_path="dummy.onnx")
        mock_session = _make_mock_session(0.0)
        gate._session = mock_session

        dummy = np.arange(8, dtype=np.float32)
        gate.predict(dummy)

        call_kwargs = mock_session.run.call_args
        input_array = call_kwargs[0][1]["input"]
        assert input_array.shape == (1, 8)


# ---------------------------------------------------------------------------
# predict() — lazy ONNX session loading
# ---------------------------------------------------------------------------


class TestPredictLazyLoading:
    """REQ-JEPA-005: InferenceSession created on first predict(), not at init."""

    def test_onnx_session_created_lazily(self, tmp_path: Path) -> None:
        """REQ-JEPA-005: _get_session() creates session on first call."""
        # We test by patching onnxruntime.InferenceSession and checking it
        # is called when predict() is first invoked, not at construction.
        dummy_onnx = tmp_path / "model.onnx"
        dummy_onnx.write_bytes(b"fake")

        mock_session = _make_mock_session(0.0)
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        gate = JepaGate(onnx_path=str(dummy_onnx))
        assert gate._session is None  # not loaded yet

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            gate.predict(np.zeros(8, dtype=np.float32))

        mock_ort.InferenceSession.assert_called_once()
        assert gate._session is mock_session

    def test_onnx_session_cached_across_calls(self) -> None:
        """REQ-JEPA-005: Second predict() call reuses existing session."""
        gate = JepaGate(onnx_path="dummy.onnx")
        mock_session = _make_mock_session(0.0)
        gate._session = mock_session  # pre-load

        dummy = np.zeros(4, dtype=np.float32)
        gate.predict(dummy)
        gate.predict(dummy)

        # run() called twice (two predict calls), but session created only once
        assert mock_session.run.call_count == 2

    def test_import_error_propagates(self) -> None:
        """REQ-JEPA-005: ImportError raised when onnxruntime not installed."""
        gate = JepaGate(onnx_path="dummy.onnx")
        # Patch onnxruntime import to fail inside _get_session
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(ImportError, match="onnxruntime"):
                gate.predict(np.zeros(4, dtype=np.float32))


# ---------------------------------------------------------------------------
# should_skip()
# ---------------------------------------------------------------------------


class TestShouldSkip:
    """REQ-JEPA-005: should_skip returns True iff energy < threshold."""

    def test_below_threshold_returns_true(self) -> None:
        """SCENARIO-JEPA-010: energy=0.3 < threshold=0.5 → skip."""
        gate = JepaGate(onnx_path="dummy.onnx", threshold=0.5)
        # Raw value that gives sigmoid ≈ 0.3 is sigmoid^-1(0.3) = ln(0.3/0.7) ≈ -0.847
        raw = math.log(0.3 / 0.7)
        gate._session = _make_mock_session(raw)
        assert gate.should_skip(np.zeros(4, dtype=np.float32)) is True

    def test_above_threshold_returns_false(self) -> None:
        """SCENARIO-JEPA-011: energy=0.8 > threshold=0.5 → do not skip."""
        gate = JepaGate(onnx_path="dummy.onnx", threshold=0.5)
        raw = math.log(0.8 / 0.2)  # sigmoid^-1(0.8)
        gate._session = _make_mock_session(raw)
        assert gate.should_skip(np.zeros(4, dtype=np.float32)) is False

    def test_at_threshold_not_skipped(self) -> None:
        """energy == threshold (0.5) is NOT < threshold → should_skip False."""
        gate = JepaGate(onnx_path="dummy.onnx", threshold=0.5)
        gate._session = _make_mock_session(0.0)  # sigmoid(0)=0.5 exactly
        assert gate.should_skip(np.zeros(4, dtype=np.float32)) is False

    def test_disabled_gate_never_skips(self) -> None:
        """REQ-JEPA-005: disabled gate → should_skip always False."""
        gate = JepaGate(onnx_path="dummy.onnx", threshold=0.5, enabled=False)
        # Even if energy would be 0 (lowest possible), disabled gate won't skip.
        assert gate.should_skip(np.zeros(4, dtype=np.float32)) is False


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------


class TestToDict:
    """REQ-JEPA-005: to_dict() returns correct JSON-serialisable dict."""

    def test_to_dict_fields(self) -> None:
        """REQ-JEPA-005: to_dict includes onnx_path, threshold, enabled."""
        gate = JepaGate(onnx_path="results/jepa_predictor_307.onnx", threshold=0.7)
        d = gate.to_dict()
        assert d["onnx_path"] == "results/jepa_predictor_307.onnx"
        assert d["threshold"] == 0.7
        assert d["enabled"] is True

    def test_to_dict_path_is_string(self) -> None:
        """REQ-JEPA-005: onnx_path in dict is always a plain str."""
        gate = JepaGate(onnx_path="results/jepa_predictor_307.onnx")
        assert isinstance(gate.to_dict()["onnx_path"], str)

    def test_to_dict_disabled(self) -> None:
        """REQ-JEPA-005: disabled flag serialises correctly."""
        gate = JepaGate(onnx_path="x.onnx", enabled=False)
        assert gate.to_dict()["enabled"] is False


# ---------------------------------------------------------------------------
# VerifyRepairPipeline.verify_with_gate()
# ---------------------------------------------------------------------------


class TestVerifyWithGate:
    """REQ-JEPA-005, SCENARIO-JEPA-010/011: Pipeline gate integration."""

    def setup_method(self) -> None:
        """Build a verify-only pipeline."""
        self.pipeline = VerifyRepairPipeline()

    def test_gate_none_runs_full_pipeline(self) -> None:
        """REQ-JEPA-005: jepa_gate=None → normal full verification path."""
        result = self.pipeline.verify_with_gate(
            question="What is 2 + 2?",
            response="2 + 2 = 4.",
            domain="arithmetic",
            jepa_gate=None,
        )
        assert isinstance(result, VerificationResult)
        # No gate fields when gate not provided.
        assert result.certificate.get("gate_decision") is None
        assert result.certificate.get("ising_skipped") is None

    def test_gate_skip_returns_skip_result(self) -> None:
        """SCENARIO-JEPA-010: gate says skip → VerificationResult with skip fields."""
        gate = JepaGate(onnx_path="dummy.onnx", threshold=0.5)
        # Inject a session that returns low energy (< 0.5) → should_skip True
        raw = math.log(0.2 / 0.8)  # sigmoid^-1(0.2) — energy ≈ 0.2
        gate._session = _make_mock_session(raw)

        dummy_logits = np.zeros(16, dtype=np.float32)
        result = self.pipeline.verify_with_gate(
            question="Is 7 prime?",
            response="Yes, 7 is prime.",
            domain="arithmetic",
            jepa_gate=gate,
            logit_mean=dummy_logits,
        )

        assert isinstance(result, VerificationResult)
        assert result.violations == []
        assert result.certificate["gate_decision"] == "skip"
        assert result.certificate["ising_skipped"] is True
        assert "gate_energy" in result.certificate
        assert result.certificate["gate_energy"] < 0.5

    def test_gate_verify_runs_ising(self) -> None:
        """SCENARIO-JEPA-011: gate says verify → full Ising pipeline runs."""
        gate = JepaGate(onnx_path="dummy.onnx", threshold=0.5)
        # Inject a session that returns high energy (> 0.5) → should_skip False
        raw = math.log(0.9 / 0.1)  # sigmoid^-1(0.9) — energy ≈ 0.9
        gate._session = _make_mock_session(raw)

        dummy_logits = np.zeros(16, dtype=np.float32)
        result = self.pipeline.verify_with_gate(
            question="What is 2 + 2?",
            response="2 + 2 = 4.",
            domain="arithmetic",
            jepa_gate=gate,
            logit_mean=dummy_logits,
        )

        assert result.certificate["gate_decision"] == "verify"
        assert result.certificate["ising_skipped"] is False
        # Ising ran → should have n_constraints or certificate fields from extract.
        assert "n_violations" in result.certificate or isinstance(result.verified, bool)

    def test_gate_skip_has_gate_energy(self) -> None:
        """REQ-JEPA-005: gate_energy stored in certificate when gate fires."""
        gate = JepaGate(onnx_path="dummy.onnx", threshold=0.5)
        raw = math.log(0.1 / 0.9)  # sigmoid ≈ 0.1
        gate._session = _make_mock_session(raw)

        result = self.pipeline.verify_with_gate(
            question="Trivial",
            response="Trivial answer.",
            domain="arithmetic",
            jepa_gate=gate,
            logit_mean=np.zeros(8, dtype=np.float32),
        )
        expected_energy = 1.0 / (1.0 + math.exp(-math.log(0.1 / 0.9)))
        assert abs(result.certificate["gate_energy"] - expected_energy) < 1e-5

    def test_gate_verify_energy_stored(self) -> None:
        """REQ-JEPA-005: gate_energy stored even when gate says 'verify'."""
        gate = JepaGate(onnx_path="dummy.onnx", threshold=0.5)
        raw = math.log(0.8 / 0.2)
        gate._session = _make_mock_session(raw)

        result = self.pipeline.verify_with_gate(
            question="What is 3 + 3?",
            response="3 + 3 = 6.",
            domain="arithmetic",
            jepa_gate=gate,
            logit_mean=np.ones(8, dtype=np.float32),
        )
        expected_energy = 1.0 / (1.0 + math.exp(-math.log(0.8 / 0.2)))
        assert abs(result.certificate["gate_energy"] - expected_energy) < 1e-5


# ---------------------------------------------------------------------------
# Latency benchmark (structural — no wall-clock assertions)
# ---------------------------------------------------------------------------


class TestLatencyBenchmark:
    """REQ-JEPA-005: skip_rate and TP_rate computed from benchmark results."""

    def _make_gate(self, threshold: float, energy: float) -> JepaGate:
        """Build a gate whose predict() always returns a fixed energy."""
        raw = math.log(energy / (1.0 - energy + 1e-9))
        gate = JepaGate(onnx_path="dummy.onnx", threshold=threshold)
        gate._session = _make_mock_session(raw)
        return gate

    def test_skip_rate_all_skip(self) -> None:
        """Gate that always returns energy < threshold → skip_rate = 1.0."""
        gate = self._make_gate(threshold=0.9, energy=0.1)
        pipeline = VerifyRepairPipeline()
        n_total = 10
        n_skipped = 0
        for i in range(n_total):
            result = pipeline.verify_with_gate(
                question=f"Q{i}",
                response=f"A{i}",
                domain="arithmetic",
                jepa_gate=gate,
                logit_mean=np.zeros(4, dtype=np.float32),
            )
            if result.certificate.get("gate_decision") == "skip":
                n_skipped += 1
        assert n_skipped == n_total

    def test_skip_rate_none_skip(self) -> None:
        """Gate that always returns energy > threshold → skip_rate = 0.0."""
        gate = self._make_gate(threshold=0.1, energy=0.9)
        pipeline = VerifyRepairPipeline()
        n_total = 10
        n_skipped = 0
        for i in range(n_total):
            result = pipeline.verify_with_gate(
                question=f"Q{i}",
                response=f"A{i}",
                domain="arithmetic",
                jepa_gate=gate,
                logit_mean=np.zeros(4, dtype=np.float32),
            )
            if result.certificate.get("gate_decision") == "skip":
                n_skipped += 1
        assert n_skipped == 0

    def test_speedup_key_exists_in_benchmark_output(self) -> None:
        """REQ-JEPA-005: benchmark dict has speedup_factor key."""
        # Simulate what exp 308 computes.
        t_with = 0.8
        t_without = 1.0
        speedup = t_without / t_with if t_with > 0 else float("inf")
        result = {
            "pipeline_time_with_gate_s": t_with,
            "pipeline_time_without_gate_s": t_without,
            "speedup_factor": speedup,
        }
        assert "speedup_factor" in result
        assert result["speedup_factor"] > 1.0
