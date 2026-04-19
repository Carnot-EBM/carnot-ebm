"""Tests for GPUVRAMGate, VRAMStatus, and GPUVRAMInsufficientError.

Covers REQ-INFRA-039, REQ-INFRA-040, REQ-INFRA-041,
       SCENARIO-INFRA-047, SCENARIO-INFRA-048, SCENARIO-INFRA-049
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.gpu_vram_gate import (
    GPUVRAMGate,
    GPUVRAMInsufficientError,
    VRAMStatus,
)


# ---------------------------------------------------------------------------
# VRAMStatus tests (SCENARIO-INFRA-047)
# ---------------------------------------------------------------------------


class TestVRAMStatus:
    """Unit tests for VRAMStatus dataclass properties."""

    def test_free_gb_converts_mb_to_gb(self):
        """free_gb should be free_mb / 1024."""
        s = VRAMStatus(gpu_index=0, total_mb=24576, used_mb=8192, free_mb=16384)
        assert s.free_gb == pytest.approx(16.0, abs=0.01)

    def test_free_gb_zero(self):
        s = VRAMStatus(gpu_index=0, total_mb=24576, used_mb=24576, free_mb=0)
        assert s.free_gb == pytest.approx(0.0)

    def test_is_zombie_saturated_true_when_over_90pct_and_zero_util(self):
        """Zombie saturation: >90% used AND 0% GPU utilisation."""
        s = VRAMStatus(
            gpu_index=0,
            total_mb=24576,
            used_mb=23000,   # ~93.6% used
            free_mb=1576,
            utilization_pct=0,
        )
        assert s.is_zombie_saturated is True

    def test_is_zombie_saturated_false_when_high_util(self):
        """Not saturated when compute is running (active model, not zombie)."""
        s = VRAMStatus(
            gpu_index=0,
            total_mb=24576,
            used_mb=23000,
            free_mb=1576,
            utilization_pct=85,
        )
        assert s.is_zombie_saturated is False

    def test_is_zombie_saturated_false_when_under_90pct(self):
        """Not saturated when VRAM is only moderately used."""
        s = VRAMStatus(
            gpu_index=0,
            total_mb=24576,
            used_mb=10000,   # ~40.7% used
            free_mb=14576,
            utilization_pct=0,
        )
        assert s.is_zombie_saturated is False

    def test_is_zombie_saturated_false_when_total_zero(self):
        """No GPU hardware — total_mb==0 means gate is no-op, not zombie-saturated."""
        s = VRAMStatus(gpu_index=0, total_mb=0, used_mb=0, free_mb=0, utilization_pct=0)
        assert s.is_zombie_saturated is False

    def test_default_utilization_pct_is_zero(self):
        s = VRAMStatus(gpu_index=0, total_mb=1024, used_mb=512, free_mb=512)
        assert s.utilization_pct == 0


# ---------------------------------------------------------------------------
# GPUVRAMInsufficientError tests (SCENARIO-INFRA-049)
# ---------------------------------------------------------------------------


class TestGPUVRAMInsufficientError:
    """Unit tests for the custom exception."""

    def test_attributes_preserved(self):
        err = GPUVRAMInsufficientError(gpu_index=1, free_gb=2.5, min_free_gb=8.0)
        assert err.gpu_index == 1
        assert err.free_gb == pytest.approx(2.5)
        assert err.min_free_gb == pytest.approx(8.0)

    def test_is_runtime_error(self):
        err = GPUVRAMInsufficientError(gpu_index=0, free_gb=1.0, min_free_gb=8.0)
        assert isinstance(err, RuntimeError)

    def test_message_contains_verdict(self):
        err = GPUVRAMInsufficientError(gpu_index=0, free_gb=1.0, min_free_gb=8.0)
        assert "gpu_vram_insufficient" in str(err)


# ---------------------------------------------------------------------------
# GPUVRAMGate.check_vram tests
# ---------------------------------------------------------------------------


class TestCheckVram:
    """Test check_vram() with mocked pynvml."""

    def test_returns_vram_status_from_nvml(self):
        mem_mock = MagicMock()
        mem_mock.total = 24 * 1024 * 1024 * 1024   # 24 GB in bytes
        mem_mock.used = 8 * 1024 * 1024 * 1024    # 8 GB
        mem_mock.free = 16 * 1024 * 1024 * 1024   # 16 GB
        util_mock = MagicMock()
        util_mock.gpu = 50

        with patch.dict("sys.modules", {"pynvml": MagicMock()}):
            import pynvml
            pynvml.nvmlInit = MagicMock()
            pynvml.nvmlDeviceGetHandleByIndex = MagicMock(return_value="handle")
            pynvml.nvmlDeviceGetMemoryInfo = MagicMock(return_value=mem_mock)
            pynvml.nvmlDeviceGetUtilizationRates = MagicMock(return_value=util_mock)

            gate = GPUVRAMGate()
            status = gate.check_vram(0)

        assert status.gpu_index == 0
        assert status.total_mb == 24 * 1024
        assert status.used_mb == 8 * 1024
        assert status.free_mb == 16 * 1024
        assert status.utilization_pct == 50

    def test_returns_synthetic_healthy_when_pynvml_absent(self):
        """When pynvml is not available, check_vram returns a zero-total no-op status."""
        gate = GPUVRAMGate()
        with patch.dict("sys.modules", {"pynvml": None}):
            status = gate.check_vram(0)
        assert status.total_mb == 0
        assert status.free_mb == 0


# ---------------------------------------------------------------------------
# GPUVRAMGate.kill_zombies tests
# ---------------------------------------------------------------------------


class TestKillZombies:
    """Test kill_zombies() returns count of killed processes."""

    def test_returns_zero_when_pynvml_absent(self):
        gate = GPUVRAMGate()
        with patch.dict("sys.modules", {"pynvml": None}):
            count = gate.kill_zombies(0)
        assert count == 0

    def test_kills_zombie_process(self):
        """Simulate a zombie: PID 9999 holds 10 GB VRAM, 0 CPU time, age > 60 s."""
        proc_mock = MagicMock()
        proc_mock.pid = 9999
        proc_mock.usedGpuMemory = 10 * 1024 * 1024 * 1024  # 10 GB

        import time as _time

        with patch.dict("sys.modules", {"pynvml": MagicMock(), "psutil": MagicMock()}):
            import pynvml
            import psutil

            pynvml.nvmlInit = MagicMock()
            pynvml.nvmlDeviceGetHandleByIndex = MagicMock(return_value="handle")
            pynvml.nvmlDeviceGetComputeRunningProcesses = MagicMock(return_value=[proc_mock])

            psutil_proc = MagicMock()
            cpu_times = MagicMock()
            cpu_times.user = 0.0
            cpu_times.system = 0.0
            psutil_proc.cpu_times = MagicMock(return_value=cpu_times)
            psutil_proc.create_time = MagicMock(return_value=_time.time() - 200)
            psutil.Process = MagicMock(return_value=psutil_proc)

            import os
            with patch("os.kill") as mock_kill:
                gate = GPUVRAMGate(auto_kill=True)
                count = gate.kill_zombies(0)

        assert count == 1
        mock_kill.assert_called_once_with(9999, 9)  # SIGKILL == 9


# ---------------------------------------------------------------------------
# GPUVRAMGate.wait_for_vram tests
# ---------------------------------------------------------------------------


class TestWaitForVram:
    """Test wait_for_vram() polling logic."""

    def test_returns_true_immediately_when_threshold_met(self):
        gate = GPUVRAMGate(min_free_gb=8.0, wait_seconds=60)
        # Patch check_vram to return plenty of free VRAM
        gate.check_vram = MagicMock(
            return_value=VRAMStatus(
                gpu_index=0, total_mb=24576, used_mb=4096, free_mb=20480
            )
        )
        result = gate.wait_for_vram(0)
        assert result is True
        gate.check_vram.assert_called_once()

    def test_returns_true_when_no_gpu(self):
        """total_mb==0 means no GPU → gate is no-op → returns True."""
        gate = GPUVRAMGate(min_free_gb=8.0, wait_seconds=10)
        gate.check_vram = MagicMock(
            return_value=VRAMStatus(gpu_index=0, total_mb=0, used_mb=0, free_mb=0)
        )
        result = gate.wait_for_vram(0)
        assert result is True

    def test_returns_false_when_wait_exhausted(self):
        """When VRAM never meets threshold, returns False after timeout."""
        gate = GPUVRAMGate(min_free_gb=8.0, wait_seconds=1)
        # Always returns insufficient VRAM
        gate.check_vram = MagicMock(
            return_value=VRAMStatus(
                gpu_index=0, total_mb=24576, used_mb=23000, free_mb=1576
            )
        )
        with patch("time.sleep"):  # speed up the test
            result = gate.wait_for_vram(0)
        assert result is False

    def test_returns_true_after_retry(self):
        """VRAM becomes available on second poll — returns True."""
        gate = GPUVRAMGate(min_free_gb=8.0, wait_seconds=60)
        responses = [
            VRAMStatus(gpu_index=0, total_mb=24576, used_mb=23000, free_mb=1576),  # low
            VRAMStatus(gpu_index=0, total_mb=24576, used_mb=4096, free_mb=20480),  # OK
        ]
        gate.check_vram = MagicMock(side_effect=responses)
        with patch("time.sleep"):
            result = gate.wait_for_vram(0)
        assert result is True


# ---------------------------------------------------------------------------
# GPUVRAMGate context manager tests (SCENARIO-INFRA-047/048/049)
# ---------------------------------------------------------------------------


class TestGPUVRAMGateContextManager:
    """Integration tests for the context manager __enter__ / __exit__."""

    def test_no_op_on_cpu_only_machine(self):
        """When no GPUs detected, context manager is a complete no-op (SCENARIO-INFRA-047)."""
        gate = GPUVRAMGate(min_free_gb=8.0)
        gate._n_gpus = MagicMock(return_value=0)
        with gate:
            pass  # must not raise

    def test_passes_without_killing_when_vram_sufficient(self):
        """When free VRAM >= threshold, no kill and no wait (SCENARIO-INFRA-048)."""
        gate = GPUVRAMGate(min_free_gb=8.0)
        gate._n_gpus = MagicMock(return_value=1)
        gate.check_vram = MagicMock(
            return_value=VRAMStatus(
                gpu_index=0, total_mb=24576, used_mb=4096, free_mb=20480
            )
        )
        gate.kill_zombies = MagicMock()
        gate.wait_for_vram = MagicMock()
        with gate:
            pass
        gate.kill_zombies.assert_not_called()
        gate.wait_for_vram.assert_not_called()

    def test_calls_kill_zombies_when_vram_insufficient(self):
        """When free VRAM < threshold with auto_kill=True, kill_zombies is called (SCENARIO-INFRA-048)."""
        gate = GPUVRAMGate(min_free_gb=8.0, auto_kill=True)
        gate._n_gpus = MagicMock(return_value=1)
        gate.check_vram = MagicMock(
            return_value=VRAMStatus(
                gpu_index=0, total_mb=24576, used_mb=23000, free_mb=1576
            )
        )
        gate.kill_zombies = MagicMock(return_value=1)
        gate.wait_for_vram = MagicMock(return_value=True)
        with gate:
            pass
        gate.kill_zombies.assert_called_once_with(0)

    def test_skips_kill_when_auto_kill_false(self):
        """When auto_kill=False, zombie kill is skipped even if VRAM is low."""
        gate = GPUVRAMGate(min_free_gb=8.0, auto_kill=False)
        gate._n_gpus = MagicMock(return_value=1)
        gate.check_vram = MagicMock(
            return_value=VRAMStatus(
                gpu_index=0, total_mb=24576, used_mb=23000, free_mb=1576
            )
        )
        gate.kill_zombies = MagicMock()
        gate.wait_for_vram = MagicMock(return_value=True)
        with gate:
            pass
        gate.kill_zombies.assert_not_called()

    def test_raises_gpu_vram_insufficient_error_when_wait_exhausted(self):
        """When wait_for_vram returns False, raises GPUVRAMInsufficientError (SCENARIO-INFRA-049)."""
        gate = GPUVRAMGate(min_free_gb=8.0, auto_kill=True)
        gate._n_gpus = MagicMock(return_value=1)
        low_vram = VRAMStatus(gpu_index=0, total_mb=24576, used_mb=23000, free_mb=1576)
        gate.check_vram = MagicMock(return_value=low_vram)
        gate.kill_zombies = MagicMock(return_value=1)
        gate.wait_for_vram = MagicMock(return_value=False)
        with pytest.raises(GPUVRAMInsufficientError) as exc_info:
            with gate:
                pass
        err = exc_info.value
        assert err.gpu_index == 0
        assert err.min_free_gb == pytest.approx(8.0)

    def test_skips_zero_total_mb_gpu(self):
        """total_mb==0 for a detected index means unavailable — skip gracefully."""
        gate = GPUVRAMGate(min_free_gb=8.0)
        gate._n_gpus = MagicMock(return_value=1)
        gate.check_vram = MagicMock(
            return_value=VRAMStatus(gpu_index=0, total_mb=0, used_mb=0, free_mb=0)
        )
        gate.kill_zombies = MagicMock()
        gate.wait_for_vram = MagicMock()
        with gate:
            pass
        gate.kill_zombies.assert_not_called()
        gate.wait_for_vram.assert_not_called()

    def test_exit_returns_none(self):
        """__exit__ must not suppress exceptions (returns None, not True)."""
        gate = GPUVRAMGate()
        result = gate.__exit__(None, None, None)
        assert result is None


# ---------------------------------------------------------------------------
# GPUVRAMGate._n_gpus tests
# ---------------------------------------------------------------------------


class TestNGpus:
    """Test _n_gpus() helper."""

    def test_returns_count_from_pynvml(self):
        with patch.dict("sys.modules", {"pynvml": MagicMock()}):
            import pynvml
            pynvml.nvmlInit = MagicMock()
            pynvml.nvmlDeviceGetCount = MagicMock(return_value=2)
            gate = GPUVRAMGate()
            count = gate._n_gpus()
        assert count == 2

    def test_returns_zero_when_pynvml_absent(self):
        gate = GPUVRAMGate()
        with patch.dict("sys.modules", {"pynvml": None}):
            count = gate._n_gpus()
        assert count == 0


# ---------------------------------------------------------------------------
# Additional kill_zombies edge cases
# ---------------------------------------------------------------------------


class TestKillZombiesEdgeCases:
    """Cover NoSuchProcess and ImportError fallback branches."""

    def _make_proc_mock(self, pid: int, vram_bytes: int) -> MagicMock:
        m = MagicMock()
        m.pid = pid
        m.usedGpuMemory = vram_bytes
        return m

    def _base_pynvml_patch(self, proc_list):
        pynvml_mock = MagicMock()
        pynvml_mock.nvmlInit = MagicMock()
        pynvml_mock.nvmlDeviceGetHandleByIndex = MagicMock(return_value="handle")
        pynvml_mock.nvmlDeviceGetComputeRunningProcesses = MagicMock(return_value=proc_list)
        return pynvml_mock

    def test_skips_process_with_small_vram(self):
        """Processes holding < 100 MB should be skipped (system processes)."""
        proc_mock = self._make_proc_mock(pid=1234, vram_bytes=50 * 1024 * 1024)  # 50 MB
        pynvml_mock = self._base_pynvml_patch([proc_mock])
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            gate = GPUVRAMGate()
            count = gate.kill_zombies(0)
        assert count == 0

    def test_kills_via_no_such_process_branch(self):
        """When psutil raises NoSuchProcess, fall through to direct os.kill."""
        import psutil as _psutil_real

        proc_mock = self._make_proc_mock(pid=8888, vram_bytes=10 * 1024 * 1024 * 1024)
        pynvml_mock = self._base_pynvml_patch([proc_mock])
        psutil_mock = MagicMock()
        psutil_mock.NoSuchProcess = _psutil_real.NoSuchProcess
        psutil_mock.Process = MagicMock(side_effect=_psutil_real.NoSuchProcess(pid=8888))

        with patch.dict("sys.modules", {"pynvml": pynvml_mock, "psutil": psutil_mock}):
            with patch("os.kill") as mock_kill:
                gate = GPUVRAMGate()
                count = gate.kill_zombies(0)

        assert count == 1
        mock_kill.assert_called_once_with(8888, 9)

    def test_no_such_process_already_gone(self):
        """When NoSuchProcess + os.kill raises ProcessLookupError, count stays 0."""
        import psutil as _psutil_real

        proc_mock = self._make_proc_mock(pid=7777, vram_bytes=10 * 1024 * 1024 * 1024)
        pynvml_mock = self._base_pynvml_patch([proc_mock])
        psutil_mock = MagicMock()
        psutil_mock.NoSuchProcess = _psutil_real.NoSuchProcess
        psutil_mock.Process = MagicMock(side_effect=_psutil_real.NoSuchProcess(pid=7777))

        with patch.dict("sys.modules", {"pynvml": pynvml_mock, "psutil": psutil_mock}):
            with patch("os.kill", side_effect=ProcessLookupError):
                gate = GPUVRAMGate()
                count = gate.kill_zombies(0)
        assert count == 0

    def test_kills_via_proc_fallback_when_psutil_absent(self):
        """When psutil is not importable and /proc/<pid> does not exist, kill via os."""
        proc_mock = self._make_proc_mock(pid=6666, vram_bytes=10 * 1024 * 1024 * 1024)
        pynvml_mock = self._base_pynvml_patch([proc_mock])

        # Remove psutil from sys.modules to simulate ImportError
        with patch.dict("sys.modules", {"pynvml": pynvml_mock, "psutil": None}):
            with patch("os.path.exists", return_value=False):
                with patch("os.kill") as mock_kill:
                    gate = GPUVRAMGate()
                    count = gate.kill_zombies(0)
        assert count == 1
        mock_kill.assert_called_once_with(6666, 9)

    def test_proc_fallback_noop_when_proc_exists(self):
        """When psutil absent but /proc/<pid> exists, process is alive — skip kill."""
        proc_mock = self._make_proc_mock(pid=4444, vram_bytes=10 * 1024 * 1024 * 1024)
        pynvml_mock = self._base_pynvml_patch([proc_mock])

        with patch.dict("sys.modules", {"pynvml": pynvml_mock, "psutil": None}):
            with patch("os.path.exists", return_value=True):
                with patch("os.kill") as mock_kill:
                    gate = GPUVRAMGate()
                    count = gate.kill_zombies(0)
        assert count == 0
        mock_kill.assert_not_called()

    def test_proc_fallback_handles_lookup_error(self):
        """When psutil absent and os.kill raises ProcessLookupError, count stays 0."""
        proc_mock = self._make_proc_mock(pid=3333, vram_bytes=10 * 1024 * 1024 * 1024)
        pynvml_mock = self._base_pynvml_patch([proc_mock])

        with patch.dict("sys.modules", {"pynvml": pynvml_mock, "psutil": None}):
            with patch("os.path.exists", return_value=False):
                with patch("os.kill", side_effect=ProcessLookupError):
                    gate = GPUVRAMGate()
                    count = gate.kill_zombies(0)
        assert count == 0

    def test_proc_with_active_cpu_not_killed(self):
        """Process with CPU activity > 0.1 s should NOT be killed."""
        import time as _time

        proc_mock = self._make_proc_mock(pid=5555, vram_bytes=10 * 1024 * 1024 * 1024)
        pynvml_mock = self._base_pynvml_patch([proc_mock])
        psutil_mock = MagicMock()
        psutil_mock.NoSuchProcess = Exception  # won't be raised

        cpu_times = MagicMock()
        cpu_times.user = 100.0
        cpu_times.system = 50.0
        psutil_proc = MagicMock()
        psutil_proc.cpu_times = MagicMock(return_value=cpu_times)
        psutil_proc.create_time = MagicMock(return_value=_time.time() - 200)
        psutil_mock.Process = MagicMock(return_value=psutil_proc)

        with patch.dict("sys.modules", {"pynvml": pynvml_mock, "psutil": psutil_mock}):
            with patch("os.kill") as mock_kill:
                gate = GPUVRAMGate()
                count = gate.kill_zombies(0)
        assert count == 0
        mock_kill.assert_not_called()
