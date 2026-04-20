"""Tests for carnot.pipeline.live_assertion — 100% coverage.

Spec: REQ-INFRA-082, SCENARIO-INFRA-089, SCENARIO-INFRA-090
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

import carnot.pipeline.live_assertion as _mod
from carnot.pipeline.live_assertion import assert_live_gpu_available, assert_live_or_ci_skip


# ---------------------------------------------------------------------------
# assert_live_gpu_available — no-GPU branches
# ---------------------------------------------------------------------------


def test_no_torch_returns_silently(monkeypatch):
    """When torch cannot be imported, the assertion does not apply and returns None.

    REQ-INFRA-082-2: assert_live_gpu_available() returns silently when torch is not importable.
    """
    # Simulate torch not being installed by making the import fail.
    with patch.dict(sys.modules, {"torch": None}):
        # None in sys.modules causes ImportError on `import torch`
        result = assert_live_gpu_available()
    assert result is None


def test_cuda_not_available_returns_silently(monkeypatch):
    """When torch is present but CUDA is unavailable, the assertion returns silently.

    REQ-INFRA-082-3: returns silently when torch.cuda.is_available() is False.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = assert_live_gpu_available()
    assert result is None


# ---------------------------------------------------------------------------
# assert_live_gpu_available — GPU present, var missing/wrong
# ---------------------------------------------------------------------------


def test_raises_when_cuda_available_and_var_absent(monkeypatch):
    """RuntimeError raised when CUDA is available and CARNOT_FORCE_LIVE is not set.

    SCENARIO-INFRA-089: raises RuntimeError when cuda available and var missing.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)

    with patch.dict(sys.modules, {"torch": mock_torch}):
        with pytest.raises(RuntimeError, match="CARNOT_FORCE_LIVE must be set to 1"):
            assert_live_gpu_available()


def test_raises_when_cuda_available_and_var_is_zero(monkeypatch):
    """RuntimeError raised when CUDA is available and CARNOT_FORCE_LIVE='0'.

    REQ-INFRA-082-1: raises when CARNOT_FORCE_LIVE != '1'.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")

    with patch.dict(sys.modules, {"torch": mock_torch}):
        with pytest.raises(RuntimeError, match="CARNOT_FORCE_LIVE must be set to 1"):
            assert_live_gpu_available()


def test_raises_when_cuda_available_and_var_is_false_string(monkeypatch):
    """RuntimeError raised when CUDA is available and CARNOT_FORCE_LIVE='false'.

    REQ-INFRA-082-1: any non-'1' value triggers the error.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    monkeypatch.setenv("CARNOT_FORCE_LIVE", "false")

    with patch.dict(sys.modules, {"torch": mock_torch}):
        with pytest.raises(RuntimeError, match="CARNOT_FORCE_LIVE must be set to 1"):
            assert_live_gpu_available()


# ---------------------------------------------------------------------------
# assert_live_gpu_available — GPU present, var correctly set
# ---------------------------------------------------------------------------


def test_does_not_raise_when_cuda_available_and_var_is_one(monkeypatch):
    """No exception when CUDA is available and CARNOT_FORCE_LIVE='1'.

    REQ-INFRA-082-1 (happy path): function returns None when correctly configured.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = assert_live_gpu_available()
    assert result is None


# ---------------------------------------------------------------------------
# assert_live_or_ci_skip — CI bypass
# ---------------------------------------------------------------------------


def test_ci_skip_returns_silently_regardless_of_gpu(monkeypatch):
    """When CARNOT_IS_CI=1, assert_live_or_ci_skip returns without GPU check.

    SCENARIO-INFRA-090: CI skip bypasses GPU assertion entirely.
    REQ-INFRA-082-4: assert_live_or_ci_skip returns silently when CARNOT_IS_CI=1.
    """
    mock_torch = MagicMock()
    # Simulate a live GPU that would normally trigger the error.
    mock_torch.cuda.is_available.return_value = True

    monkeypatch.setenv("CARNOT_IS_CI", "1")
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = assert_live_or_ci_skip()
    assert result is None


def test_ci_skip_delegates_to_assert_when_not_ci(monkeypatch):
    """When CARNOT_IS_CI is not '1', assert_live_or_ci_skip delegates to assert_live_gpu_available.

    Non-CI + CUDA available + missing var → RuntimeError propagates.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    monkeypatch.delenv("CARNOT_IS_CI", raising=False)
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)

    with patch.dict(sys.modules, {"torch": mock_torch}):
        with pytest.raises(RuntimeError, match="CARNOT_FORCE_LIVE must be set to 1"):
            assert_live_or_ci_skip()


def test_ci_skip_no_raise_when_not_ci_but_no_gpu(monkeypatch):
    """Non-CI + no CUDA → assert_live_or_ci_skip returns silently.

    Verifies the delegation path returns cleanly when CUDA is absent.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    monkeypatch.delenv("CARNOT_IS_CI", raising=False)
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = assert_live_or_ci_skip()
    assert result is None


# ---------------------------------------------------------------------------
# Export check — both symbols present in carnot.pipeline
# ---------------------------------------------------------------------------


def test_exports_present_in_pipeline():
    """Both assert functions are exported from carnot.pipeline.

    REQ-INFRA-082-5: Both functions exported from carnot.pipeline.__init__.
    """
    import carnot.pipeline as pipeline

    assert hasattr(pipeline, "assert_live_gpu_available"), (
        "assert_live_gpu_available not exported from carnot.pipeline"
    )
    assert hasattr(pipeline, "assert_live_or_ci_skip"), (
        "assert_live_or_ci_skip not exported from carnot.pipeline"
    )
    assert callable(pipeline.assert_live_gpu_available)
    assert callable(pipeline.assert_live_or_ci_skip)
