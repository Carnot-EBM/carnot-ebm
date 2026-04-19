"""Tests for JITVRAMCheck and JITVRAMResult.

Spec: REQ-INFRA-064, REQ-INFRA-065, REQ-INFRA-066,
      SCENARIO-INFRA-073, SCENARIO-INFRA-074, SCENARIO-INFRA-075
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from carnot.pipeline.jit_vram_check import (
    JITVRAMCheck,
    JITVRAMResult,
    _CI_STUB_AVAILABLE_GB,
)


# ---------------------------------------------------------------------------
# JITVRAMResult dataclass tests
# ---------------------------------------------------------------------------


class TestJITVRAMResult:
    def test_fields_are_set(self):
        r = JITVRAMResult(
            device_id=0,
            model_id="test-model",
            required_gb=10.0,
            available_gb=20.0,
            is_cleared=True,
            attempts=1,
        )
        assert r.device_id == 0
        assert r.model_id == "test-model"
        assert r.required_gb == pytest.approx(10.0)
        assert r.available_gb == pytest.approx(20.0)
        assert r.is_cleared is True
        assert r.attempts == 1
        assert r.wait_applied is False

    def test_wait_applied_defaults_false(self):
        r = JITVRAMResult(
            device_id=0,
            model_id="m",
            required_gb=5.0,
            available_gb=10.0,
            is_cleared=True,
            attempts=1,
        )
        assert r.wait_applied is False

    def test_wait_applied_can_be_true(self):
        r = JITVRAMResult(
            device_id=0,
            model_id="m",
            required_gb=5.0,
            available_gb=10.0,
            is_cleared=True,
            attempts=2,
            wait_applied=True,
        )
        assert r.wait_applied is True


# ---------------------------------------------------------------------------
# JITVRAMCheck.get_available_gb tests
# ---------------------------------------------------------------------------


class TestGetAvailableGb:
    def test_ci_stub_when_pynvml_not_installed(self):
        # SCENARIO-INFRA-075 (CI stub path): pynvml ImportError → returns 24.0
        checker = JITVRAMCheck(device_id=0)
        with patch.dict("sys.modules", {"pynvml": None}):
            result = checker.get_available_gb()
        assert result == pytest.approx(_CI_STUB_AVAILABLE_GB)

    def test_returns_real_value_when_pynvml_available(self):
        # Simulate pynvml returning 12 GiB free
        free_bytes = int(12.0 * 1024 ** 3)
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(free=free_bytes)

        checker = JITVRAMCheck(device_id=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = checker.get_available_gb()

        assert result == pytest.approx(12.0)
        mock_pynvml.nvmlInit.assert_called_once()
        mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)

    def test_ci_stub_on_pynvml_runtime_error(self):
        # If pynvml raises (e.g. driver not loaded), fall back to stub
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = RuntimeError("driver not loaded")

        checker = JITVRAMCheck(device_id=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = checker.get_available_gb()

        assert result == pytest.approx(_CI_STUB_AVAILABLE_GB)


# ---------------------------------------------------------------------------
# JITVRAMCheck.gate_model_load tests
# ---------------------------------------------------------------------------


class TestGateModelLoad:
    def _make_checker(self, available_sequence):
        """Return a JITVRAMCheck whose get_available_gb() returns successive values."""
        checker = JITVRAMCheck(device_id=0)
        checker.get_available_gb = MagicMock(side_effect=available_sequence)
        return checker

    def test_cleared_on_first_check(self):
        # SCENARIO-INFRA-073: available=20 >= required=10 → cleared, attempts=1
        checker = self._make_checker([20.0])
        result = checker.gate_model_load("model-a", required_gb=10.0)

        assert result.is_cleared is True
        assert result.available_gb == pytest.approx(20.0)
        assert result.attempts == 1
        assert result.wait_applied is False
        checker.get_available_gb.assert_called_once()

    def test_exact_boundary_is_cleared(self):
        # available == required → cleared (>= semantics)
        checker = self._make_checker([10.0])
        result = checker.gate_model_load("model-b", required_gb=10.0)
        assert result.is_cleared is True
        assert result.attempts == 1

    def test_retry_clears_on_second_check(self):
        # SCENARIO-INFRA-074: first check fails (8 < 10), retry succeeds (12 >= 10)
        checker = self._make_checker([8.0, 12.0])
        with patch("carnot.pipeline.jit_vram_check.time.sleep") as mock_sleep:
            result = checker.gate_model_load("model-c", required_gb=10.0, retry_wait_s=30.0)

        assert result.is_cleared is True
        assert result.available_gb == pytest.approx(12.0)
        assert result.attempts == 2
        assert result.wait_applied is True
        mock_sleep.assert_called_once_with(30.0)

    def test_both_checks_fail_is_cleared_false(self):
        # SCENARIO-INFRA-075: first=5, retry=6 — both < 10 → is_cleared=False
        checker = self._make_checker([5.0, 6.0])
        with patch("carnot.pipeline.jit_vram_check.time.sleep"):
            result = checker.gate_model_load("model-d", required_gb=10.0)

        assert result.is_cleared is False
        assert result.available_gb == pytest.approx(6.0)
        assert result.attempts == 2
        assert result.wait_applied is True

    def test_result_carries_device_id_and_model_id(self):
        checker = self._make_checker([20.0])
        checker.device_id = 1
        result = checker.gate_model_load("my-model", required_gb=5.0)
        assert result.device_id == 1
        assert result.model_id == "my-model"
        assert result.required_gb == pytest.approx(5.0)

    def test_default_retry_wait_is_30s(self):
        # Confirm default retry_wait_s=30 is passed to sleep
        checker = self._make_checker([1.0, 1.0])
        with patch("carnot.pipeline.jit_vram_check.time.sleep") as mock_sleep:
            checker.gate_model_load("m", required_gb=10.0)
        mock_sleep.assert_called_once_with(30.0)


# ---------------------------------------------------------------------------
# JITVRAMCheck.sequential_load_gate tests
# ---------------------------------------------------------------------------


class TestSequentialLoadGate:
    def test_returns_result_per_spec(self):
        checker = JITVRAMCheck(device_id=0)
        checker.get_available_gb = MagicMock(return_value=20.0)

        specs = [
            {"model_id": "model-a", "required_gb": 5.0},
            {"model_id": "model-b", "required_gb": 8.0},
        ]
        results = checker.sequential_load_gate(specs)

        assert len(results) == 2
        assert results[0].model_id == "model-a"
        assert results[1].model_id == "model-b"
        assert all(r.is_cleared for r in results)

    def test_empty_specs_returns_empty_list(self):
        checker = JITVRAMCheck()
        results = checker.sequential_load_gate([])
        assert results == []

    def test_sequential_order_preserved(self):
        # get_available_gb returns: first model=15 (pass), second model first=7 (fail), retry=7 (fail)
        checker = JITVRAMCheck(device_id=0)
        checker.get_available_gb = MagicMock(side_effect=[15.0, 7.0, 7.0])

        specs = [
            {"model_id": "first", "required_gb": 10.0},
            {"model_id": "second", "required_gb": 10.0},
        ]
        with patch("carnot.pipeline.jit_vram_check.time.sleep"):
            results = checker.sequential_load_gate(specs)

        assert results[0].is_cleared is True   # 15 >= 10
        assert results[1].is_cleared is False  # 7 < 10 on both attempts


# ---------------------------------------------------------------------------
# CI stub integration: pynvml import fails → gate always clears
# ---------------------------------------------------------------------------


class TestCIStubIntegration:
    def test_gate_clears_when_pynvml_not_installed(self):
        # Even with required_gb=22, stub returns 24.0 → cleared
        checker = JITVRAMCheck(device_id=0)
        with patch.dict("sys.modules", {"pynvml": None}):
            result = checker.gate_model_load("ci-model", required_gb=22.0)

        assert result.is_cleared is True
        assert result.available_gb == pytest.approx(_CI_STUB_AVAILABLE_GB)
        assert result.attempts == 1
