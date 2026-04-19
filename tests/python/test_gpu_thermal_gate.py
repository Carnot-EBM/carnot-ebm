"""Tests for GPUThermalGate, ThermalStatus, and GPUThermalThrottleError.

REQ-INFRA-054: check_temperature() returns ThermalStatus with temperature in Celsius
REQ-INFRA-055: wait_for_cool() blocks with exponential backoff until cool or timeout
REQ-INFRA-056: ExperimentTemplate.setup_gpu() calls GPUThermalGate before model load

SCENARIO-INFRA-062: GPU already cool — wait_for_cool returns True immediately
SCENARIO-INFRA-063: GPU hot, cools after one backoff sleep — wait_for_cool returns True
SCENARIO-INFRA-064: GPU stays hot for max_wait_seconds — GPUThermalThrottleError raised
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call

from carnot.pipeline.gpu_thermal_gate import (
    GPUThermalGate,
    GPUThermalThrottleError,
    ThermalStatus,
)


# ---------------------------------------------------------------------------
# ThermalStatus tests
# ---------------------------------------------------------------------------


class TestThermalStatus:
    """Tests for ThermalStatus dataclass and is_throttling property."""

    def test_is_throttling_false_when_temperature_none(self):
        """CPU-only machine: temperature_c=None means no GPU, never throttling."""
        status = ThermalStatus(gpu_index=0, temperature_c=None, is_safe=True)
        assert status.is_throttling is False

    def test_is_throttling_false_when_temperature_below_threshold(self):
        """Temperature at 80°C is well below 85°C throttle onset."""
        status = ThermalStatus(gpu_index=0, temperature_c=80.0, is_safe=True)
        assert status.is_throttling is False

    def test_is_throttling_false_when_temperature_at_exactly_85(self):
        """85.0°C is NOT > 85.0, so is_throttling is False at the boundary."""
        status = ThermalStatus(gpu_index=0, temperature_c=85.0, is_safe=True)
        assert status.is_throttling is False

    def test_is_throttling_true_when_temperature_above_85(self):
        """86°C exceeds 85°C threshold — GPU is throttling."""
        status = ThermalStatus(gpu_index=0, temperature_c=86.0, is_safe=False)
        assert status.is_throttling is True

    def test_is_throttling_true_when_temperature_at_93(self):
        """93°C is the RTX 3090 thermal limit — should be flagged as throttling."""
        status = ThermalStatus(gpu_index=0, temperature_c=93.0, is_safe=False)
        assert status.is_throttling is True

    def test_dataclass_fields(self):
        """ThermalStatus fields are accessible and correct."""
        status = ThermalStatus(gpu_index=1, temperature_c=72.5, is_safe=True)
        assert status.gpu_index == 1
        assert status.temperature_c == 72.5
        assert status.is_safe is True


# ---------------------------------------------------------------------------
# GPUThermalThrottleError tests
# ---------------------------------------------------------------------------


class TestGPUThermalThrottleError:
    """Tests for GPUThermalThrottleError exception."""

    def test_attributes_stored(self):
        """Error stores gpu_index, temperature_c, max_wait_seconds."""
        err = GPUThermalThrottleError(gpu_index=0, temperature_c=90.0, max_wait_seconds=300)
        assert err.gpu_index == 0
        assert err.temperature_c == 90.0
        assert err.max_wait_seconds == 300

    def test_message_contains_key_info(self):
        """Error message includes GPU index and wait time for debuggability."""
        err = GPUThermalThrottleError(gpu_index=1, temperature_c=88.0, max_wait_seconds=120)
        msg = str(err)
        assert "GPU 1" in msg
        assert "120" in msg

    def test_is_exception(self):
        """GPUThermalThrottleError is an Exception subclass."""
        err = GPUThermalThrottleError(gpu_index=0, temperature_c=None, max_wait_seconds=300)
        assert isinstance(err, Exception)

    def test_temperature_none_in_message(self):
        """Error message handles temperature_c=None gracefully."""
        err = GPUThermalThrottleError(gpu_index=0, temperature_c=None, max_wait_seconds=300)
        # Should not raise; message just includes 'None'
        assert "None" in str(err) or "300" in str(err)


# ---------------------------------------------------------------------------
# GPUThermalGate.check_temperature() tests
# ---------------------------------------------------------------------------


class TestCheckTemperature:
    """Tests for check_temperature() — REQ-INFRA-054."""

    def test_returns_none_when_pynvml_unavailable(self):
        """CPU-only machine: ImportError from pynvml → temperature_c=None, is_safe=True."""
        gate = GPUThermalGate()
        with patch("builtins.__import__", side_effect=ImportError("no pynvml")):
            status = gate.check_temperature(0)
        assert status.temperature_c is None
        assert status.is_safe is True
        assert status.gpu_index == 0

    def test_returns_none_when_pynvml_raises(self):
        """Any pynvml exception (driver error, no device) → temperature_c=None."""
        gate = GPUThermalGate()
        mock_nvml = MagicMock()
        mock_nvml.nvmlInit.side_effect = Exception("NVML init failed")
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            status = gate.check_temperature(0)
        assert status.temperature_c is None
        assert status.is_safe is True

    def test_returns_temperature_when_gpu_available(self):
        """When pynvml returns 72°C, ThermalStatus reflects that."""
        gate = GPUThermalGate()
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetTemperature.return_value = 72
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            status = gate.check_temperature(0)
        assert status.temperature_c == 72.0
        assert status.is_safe is True  # 72 <= 85

    def test_is_safe_false_when_above_hot_threshold(self):
        """Temperature above hot_threshold_c → is_safe=False."""
        gate = GPUThermalGate(hot_threshold_c=85.0)
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetTemperature.return_value = 90
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            status = gate.check_temperature(0)
        assert status.temperature_c == 90.0
        assert status.is_safe is False

    def test_is_throttling_true_when_above_85(self):
        """check_temperature returns status with is_throttling=True when temp > 85."""
        gate = GPUThermalGate()
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetTemperature.return_value = 88
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            status = gate.check_temperature(0)
        assert status.is_throttling is True

    def test_gpu_index_stored_in_status(self):
        """ThermalStatus.gpu_index matches the queried device index."""
        gate = GPUThermalGate()
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetTemperature.return_value = 65
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            status = gate.check_temperature(1)
        assert status.gpu_index == 1


# ---------------------------------------------------------------------------
# GPUThermalGate.wait_for_cool() tests
# ---------------------------------------------------------------------------


class TestWaitForCool:
    """Tests for wait_for_cool() — REQ-INFRA-055."""

    def test_returns_true_immediately_when_no_gpu(self):
        """CPU-only machine (temperature_c=None) → True immediately, no sleep.

        SCENARIO-INFRA-062 (CPU-only variant)
        """
        gate = GPUThermalGate()
        with patch.object(gate, "check_temperature") as mock_check:
            mock_check.return_value = ThermalStatus(0, None, True)
            with patch("time.sleep") as mock_sleep:
                result = gate.wait_for_cool(0)
        assert result is True
        mock_sleep.assert_not_called()

    def test_returns_true_immediately_when_already_cool(self):
        """Temperature below cool_threshold_c → True with no sleep.

        SCENARIO-INFRA-062
        """
        gate = GPUThermalGate(cool_threshold_c=80.0)
        with patch.object(gate, "check_temperature") as mock_check:
            mock_check.return_value = ThermalStatus(0, 75.0, True)
            with patch("time.sleep") as mock_sleep:
                result = gate.wait_for_cool(0)
        assert result is True
        mock_sleep.assert_not_called()

    def test_returns_true_after_cooling(self):
        """GPU hot initially, then cools after one backoff sleep.

        SCENARIO-INFRA-063
        """
        gate = GPUThermalGate(
            hot_threshold_c=85.0,
            cool_threshold_c=80.0,
            max_wait_seconds=300,
            backoff_base_seconds=15.0,
        )
        # First call: hot (90°C), second call: cool (78°C)
        hot = ThermalStatus(0, 90.0, False)
        cool = ThermalStatus(0, 78.0, True)
        with patch.object(gate, "check_temperature", side_effect=[hot, hot, cool]):
            with patch("time.sleep"):
                result = gate.wait_for_cool(0)
        assert result is True

    def test_returns_false_when_stays_hot(self):
        """GPU stays above cool_threshold_c for entire max_wait_seconds.

        SCENARIO-INFRA-064
        """
        gate = GPUThermalGate(
            hot_threshold_c=85.0,
            cool_threshold_c=80.0,
            max_wait_seconds=30,
            backoff_base_seconds=15.0,
        )
        hot = ThermalStatus(0, 90.0, False)
        with patch.object(gate, "check_temperature", return_value=hot):
            with patch("time.sleep"):
                result = gate.wait_for_cool(0)
        assert result is False

    def test_gpu_disappears_during_wait_returns_true(self):
        """If GPU disappears mid-wait (temperature_c→None), treat as no-op and return True."""
        gate = GPUThermalGate(
            hot_threshold_c=85.0,
            cool_threshold_c=80.0,
            max_wait_seconds=300,
            backoff_base_seconds=15.0,
        )
        hot = ThermalStatus(0, 90.0, False)
        gone = ThermalStatus(0, None, True)
        with patch.object(gate, "check_temperature", side_effect=[hot, hot, gone]):
            with patch("time.sleep"):
                result = gate.wait_for_cool(0)
        assert result is True


# ---------------------------------------------------------------------------
# GPUThermalGate context manager tests
# ---------------------------------------------------------------------------


class TestContextManager:
    """Tests for __enter__ / __exit__ — raises on persistent throttle."""

    def test_enter_succeeds_when_cool(self):
        """No exception when GPU is already cool."""
        gate = GPUThermalGate()
        with patch.object(gate, "wait_for_cool", return_value=True):
            with gate:
                pass  # should not raise

    def test_enter_raises_when_stays_hot(self):
        """GPUThermalThrottleError raised when wait_for_cool returns False.

        SCENARIO-INFRA-064
        """
        gate = GPUThermalGate()
        hot = ThermalStatus(0, 92.0, False)
        with patch.object(gate, "check_temperature", return_value=hot):
            with patch.object(gate, "wait_for_cool", return_value=False):
                with pytest.raises(GPUThermalThrottleError) as exc_info:
                    gate.__enter__()
        assert exc_info.value.gpu_index == 0

    def test_exit_is_noop(self):
        """__exit__ does not raise and performs no cleanup."""
        gate = GPUThermalGate()
        gate.__exit__(None, None, None)  # should not raise


# ---------------------------------------------------------------------------
# GPUThermalGate no-pynvml integration test
# ---------------------------------------------------------------------------


class TestNoPynvml:
    """Integration: entire gate is a no-op on CPU-only machines."""

    def test_full_noop_on_cpu_only(self):
        """On a machine without pynvml, check_temperature returns None and
        wait_for_cool returns True without sleeping.

        This ensures every CPU-only CI run is unaffected by the thermal gate.
        """
        gate = GPUThermalGate()
        # Simulate pynvml not installed by patching the import to raise
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else None  # type: ignore[attr-defined]

        def _fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "pynvml":
                raise ImportError("no module named pynvml")
            import builtins
            return builtins.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            status = gate.check_temperature(0)
            assert status.temperature_c is None
            assert status.is_safe is True

            with patch("time.sleep") as mock_sleep:
                result = gate.wait_for_cool(0)
            assert result is True
            mock_sleep.assert_not_called()
