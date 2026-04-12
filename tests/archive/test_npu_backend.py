"""Tests for NpuJEPAPredictor (AMD XDNA NPU backend).

**Spec coverage:**
    REQ-JEPA-001: JEPA Tier-3 predictor backed by AMD XDNA NPU (falls back to CPU).
    SCENARIO-JEPA-001: predict() returns per-domain violation probabilities.
    SCENARIO-JEPA-002: is_high_risk() gates on max domain probability.

All tests mock onnxruntime so no NPU hardware or ONNX model file is required.
The ``onnxruntime`` import lives inside ``NpuJEPAPredictor.__init__``, so patching
``sys.modules`` at construction time is sufficient.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from carnot.samplers.npu_backend import NpuJEPAPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_ort(
    available_providers: list[str] | None = None,
    npu_session_raises: Exception | None = None,
) -> MagicMock:
    """Build a mock onnxruntime module.

    REQ-JEPA-001: session creation strategy depends on available providers.

    Args:
        available_providers: List returned by ``get_available_providers()``.
            Defaults to ``["CPUExecutionProvider"]``.
        npu_session_raises: If set, the first ``InferenceSession`` call for
            AMDXDNAExecutionProvider raises this exception, testing the
            graceful fallback path.
    """
    if available_providers is None:
        available_providers = ["CPUExecutionProvider"]

    ort = MagicMock()
    ort.get_available_providers.return_value = available_providers

    # Build a realistic mock session.
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    # Returns a batch-1 result: [[p_arithmetic, p_code, p_logic]].
    mock_session.run.return_value = [np.array([[0.8, 0.3, 0.1]], dtype=np.float32)]

    if npu_session_raises is not None:
        call_count: dict[str, int] = {"n": 0}

        def session_side_effect(path: str, providers: list[str] | None = None) -> MagicMock:
            call_count["n"] += 1
            if providers and "VitisAIExecutionProvider" in providers:
                raise npu_session_raises  # type: ignore[misc]
            return mock_session

        ort.InferenceSession.side_effect = session_side_effect
    else:
        ort.InferenceSession.return_value = mock_session

    return ort


@pytest.fixture()
def fake_onnx(tmp_path: Path) -> Path:
    """Create a placeholder ONNX file so Path.exists() is satisfied.

    REQ-JEPA-001: constructor requires the ONNX file to exist before loading.
    """
    onnx_file = tmp_path / "jepa_predictor_146.onnx"
    onnx_file.write_bytes(b"fake-onnx-content")
    return onnx_file


def _make_predictor(
    fake_onnx: Path,
    available_providers: list[str] | None = None,
    npu_session_raises: Exception | None = None,
    prefer_npu: bool = False,
) -> NpuJEPAPredictor:
    """Construct an NpuJEPAPredictor with mocked onnxruntime.

    The import of onnxruntime happens inside __init__, so patching sys.modules
    for the duration of the constructor call is sufficient.

    REQ-JEPA-001: helper shared across init and inference tests.
    """
    mock_ort = _make_mock_ort(
        available_providers=available_providers,
        npu_session_raises=npu_session_raises,
    )
    with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
        return NpuJEPAPredictor(onnx_path=fake_onnx, prefer_npu=prefer_npu)


# ---------------------------------------------------------------------------
# Tests: NpuJEPAPredictor.__init__
# ---------------------------------------------------------------------------


class TestNpuJEPAPredictorInit:
    """REQ-JEPA-001: constructor selects the appropriate execution backend."""

    def test_missing_onnx_raises(self) -> None:
        """REQ-JEPA-001: non-existent ONNX path raises FileNotFoundError."""
        mock_ort = _make_mock_ort()
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            with pytest.raises(FileNotFoundError, match="ONNX model not found"):
                NpuJEPAPredictor(onnx_path="/tmp/does_not_exist_xyz.onnx")

    def test_cpu_fallback_when_prefer_npu_false(self, fake_onnx: Path) -> None:
        """REQ-JEPA-001: prefer_npu=False skips NPU check and uses CPU."""
        pred = _make_predictor(fake_onnx, prefer_npu=False)
        assert pred.backend_name == "cpu_fallback"

    def test_cpu_fallback_when_npu_provider_absent(self, fake_onnx: Path) -> None:
        """REQ-JEPA-001: NPU provider absent → logs warning, uses CPU."""
        pred = _make_predictor(
            fake_onnx,
            available_providers=["CPUExecutionProvider"],
            prefer_npu=True,
        )
        assert pred.backend_name == "cpu_fallback"

    def test_npu_backend_when_provider_available(self, fake_onnx: Path) -> None:
        """REQ-JEPA-001: VitisAIExecutionProvider present → backend_name='npu'."""
        pred = _make_predictor(
            fake_onnx,
            available_providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            prefer_npu=True,
        )
        assert pred.backend_name == "npu"

    def test_npu_session_fails_falls_back_to_cpu(self, fake_onnx: Path) -> None:
        """REQ-JEPA-001: NPU session creation error → graceful CPU fallback."""
        pred = _make_predictor(
            fake_onnx,
            available_providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            npu_session_raises=RuntimeError("device busy"),
            prefer_npu=True,
        )
        assert pred.backend_name == "cpu_fallback"


# ---------------------------------------------------------------------------
# Tests: predict() and is_high_risk()
# ---------------------------------------------------------------------------


class TestNpuJEPAPredictorInference:
    """SCENARIO-JEPA-001, SCENARIO-JEPA-002: inference methods."""

    @pytest.fixture()
    def predictor(self, fake_onnx: Path) -> NpuJEPAPredictor:
        """Return a CPU-fallback NpuJEPAPredictor with mocked onnxruntime.

        SCENARIO-JEPA-001: fixture used by inference tests.
        """
        return _make_predictor(fake_onnx, prefer_npu=False)

    def test_predict_returns_domain_dict(self, predictor: NpuJEPAPredictor) -> None:
        """SCENARIO-JEPA-001: predict() returns dict with all three domain keys."""
        embedding = np.zeros(256, dtype=np.float32)
        result = predictor.predict(embedding)
        assert set(result.keys()) == {"arithmetic", "code", "logic"}

    def test_predict_values_in_zero_one(self, predictor: NpuJEPAPredictor) -> None:
        """SCENARIO-JEPA-001: all probability values lie in [0, 1]."""
        embedding = np.zeros(256, dtype=np.float32)
        result = predictor.predict(embedding)
        for domain, prob in result.items():
            assert 0.0 <= prob <= 1.0, f"{domain} probability {prob} out of range"

    def test_predict_correct_values(self, predictor: NpuJEPAPredictor) -> None:
        """SCENARIO-JEPA-001: predict() maps ONNX outputs to correct domain probs."""
        # Mock session returns [[0.8, 0.3, 0.1]] → arithmetic=0.8, code=0.3, logic=0.1.
        embedding = np.zeros(256, dtype=np.float32)
        result = predictor.predict(embedding)
        assert result["arithmetic"] == pytest.approx(0.8)
        assert result["code"] == pytest.approx(0.3)
        assert result["logic"] == pytest.approx(0.1)

    def test_is_high_risk_true_when_above_threshold(
        self, predictor: NpuJEPAPredictor
    ) -> None:
        """SCENARIO-JEPA-002: max prob 0.8 ≥ threshold 0.5 → True."""
        embedding = np.zeros(256, dtype=np.float32)
        assert predictor.is_high_risk(embedding, threshold=0.5) is True

    def test_is_high_risk_false_when_below_threshold(
        self, predictor: NpuJEPAPredictor
    ) -> None:
        """SCENARIO-JEPA-002: max prob 0.8 < threshold 0.9 → False."""
        embedding = np.zeros(256, dtype=np.float32)
        assert predictor.is_high_risk(embedding, threshold=0.9) is False

    def test_is_high_risk_default_threshold(self, predictor: NpuJEPAPredictor) -> None:
        """SCENARIO-JEPA-002: default threshold=0.5 used when not specified."""
        embedding = np.zeros(256, dtype=np.float32)
        # max prob = 0.8 ≥ 0.5 → True.
        assert predictor.is_high_risk(embedding) is True

    def test_predict_reshapes_input(self, predictor: NpuJEPAPredictor) -> None:
        """SCENARIO-JEPA-001: non-float32 input is cast and reshaped correctly."""
        # Integer array should be cast to float32 without error.
        embedding = np.zeros(256, dtype=np.int32)
        result = predictor.predict(embedding)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Tests: backend_name and __repr__
# ---------------------------------------------------------------------------


class TestNpuJEPAPredictorMeta:
    """REQ-JEPA-001: backend_name property and repr."""

    def test_backend_name_cpu_fallback(self, fake_onnx: Path) -> None:
        """REQ-JEPA-001: cpu_fallback backend_name when NPU not available."""
        pred = _make_predictor(fake_onnx, prefer_npu=False)
        assert pred.backend_name == "cpu_fallback"

    def test_repr_contains_backend(self, fake_onnx: Path) -> None:
        """REQ-JEPA-001: __repr__ includes the active backend name."""
        pred = _make_predictor(fake_onnx, prefer_npu=False)
        assert "cpu_fallback" in repr(pred)
        assert "NpuJEPAPredictor" in repr(pred)
