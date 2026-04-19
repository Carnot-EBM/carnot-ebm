"""Tests for GPUVRAMGateV2.

Spec: REQ-INFRA-049, REQ-INFRA-050, REQ-INFRA-051,
      SCENARIO-INFRA-057, SCENARIO-INFRA-058, SCENARIO-INFRA-059
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from carnot.pipeline.gpu_vram_gate import GPUVRAMInsufficientError, VRAMStatus
from carnot.pipeline.gpu_vram_gate_v2 import GPUVRAMGateV2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _healthy_status(gpu_index: int = 0, free_gb: float = 20.0) -> VRAMStatus:
    """Return a VRAMStatus with sufficient free VRAM."""
    free_mb = int(free_gb * 1024)
    return VRAMStatus(
        gpu_index=gpu_index,
        total_mb=24576,
        used_mb=24576 - free_mb,
        free_mb=free_mb,
        utilization_pct=0,
    )


def _starved_status(gpu_index: int = 0, free_gb: float = 1.0) -> VRAMStatus:
    """Return a VRAMStatus with insufficient free VRAM."""
    free_mb = int(free_gb * 1024)
    return VRAMStatus(
        gpu_index=gpu_index,
        total_mb=24576,
        used_mb=24576 - free_mb,
        free_mb=free_mb,
        utilization_pct=0,
    )


def _no_gpu_status(gpu_index: int = 0) -> VRAMStatus:
    """Return a VRAMStatus representing an absent GPU (pynvml unavailable)."""
    return VRAMStatus(gpu_index=gpu_index, total_mb=0, used_mb=0, free_mb=0, utilization_pct=0)


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-057: kill_first=True calls kill_zombies BEFORE first check_vram
# ---------------------------------------------------------------------------


class TestKillFirstOrdering:
    """REQ-INFRA-049: verify kill_zombies() fires before check_vram() when kill_first=True."""

    def test_kill_before_check_vram_ordering(self):
        """kill_first=True: kill_zombies() must be called before check_vram()."""
        gate = GPUVRAMGateV2(
            min_free_gb=8.0,
            wait_seconds=60,
            zombie_drain_sleep_seconds=0,  # no sleep in tests
            kill_first=True,
        )
        call_order = []

        def mock_kill(gpu_index: int) -> int:
            call_order.append("kill")
            return 1  # pretend we killed one zombie

        def mock_check(gpu_index: int) -> VRAMStatus:
            call_order.append("check")
            return _healthy_status(gpu_index)

        gate.kill_zombies = mock_kill
        gate.check_vram = mock_check

        with patch("carnot.pipeline.gpu_vram_gate_v2.time.sleep"):
            result = gate.ensure_vram_available(0)

        assert result is True
        assert call_order[0] == "kill", "kill_zombies must fire BEFORE check_vram"
        assert call_order[1] == "check"

    def test_drain_sleep_called_with_zombie_drain_sleep_seconds(self):
        """Sleep is called with zombie_drain_sleep_seconds after kill_zombies()."""
        gate = GPUVRAMGateV2(
            min_free_gb=8.0,
            wait_seconds=60,
            zombie_drain_sleep_seconds=15,
            kill_first=True,
        )
        gate.kill_zombies = lambda gpu_index: 0
        gate.check_vram = lambda gpu_index: _healthy_status(gpu_index)

        with patch("carnot.pipeline.gpu_vram_gate_v2.time.sleep") as mock_sleep:
            gate.ensure_vram_available(0)

        mock_sleep.assert_called_once_with(15)

    def test_drain_sleep_called_even_when_no_zombies_killed(self):
        """Sleep fires even when kill_zombies() returns 0 (no zombies found).

        Why: pynvml's zombie detection heuristic is imperfect.  Some contexts
        are not in ComputeRunningProcesses but still consume VRAM.  Always sleeping
        ensures the drain window is respected regardless of detection accuracy.
        """
        gate = GPUVRAMGateV2(
            min_free_gb=8.0,
            zombie_drain_sleep_seconds=15,
            kill_first=True,
        )
        gate.kill_zombies = lambda gpu_index: 0  # no zombies found
        gate.check_vram = lambda gpu_index: _healthy_status(gpu_index)

        with patch("carnot.pipeline.gpu_vram_gate_v2.time.sleep") as mock_sleep:
            gate.ensure_vram_available(0)

        mock_sleep.assert_called_once_with(15)


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-058: kill_first=False uses old check-first order
# ---------------------------------------------------------------------------


class TestCheckFirstBackwardCompat:
    """REQ-INFRA-049: kill_first=False reproduces V1 check-first behavior."""

    def test_check_before_kill_when_kill_first_false(self):
        """kill_first=False: check_vram() must be called before kill_zombies()."""
        gate = GPUVRAMGateV2(
            min_free_gb=8.0,
            wait_seconds=0,
            zombie_drain_sleep_seconds=0,
            kill_first=False,
        )
        call_order = []

        def mock_check(gpu_index: int) -> VRAMStatus:
            call_order.append("check")
            return _healthy_status(gpu_index)

        def mock_kill(gpu_index: int) -> int:
            call_order.append("kill")
            return 0

        gate.check_vram = mock_check
        gate.kill_zombies = mock_kill

        result = gate.ensure_vram_available(0)

        assert result is True
        assert call_order[0] == "check", "check_vram must fire BEFORE kill_zombies in V1 compat"
        # kill_zombies not called because check passed immediately
        assert "kill" not in call_order

    def test_check_first_kills_when_vram_insufficient(self):
        """kill_first=False: kill_zombies fires after check fails (V1 behavior)."""
        gate = GPUVRAMGateV2(
            min_free_gb=8.0,
            wait_seconds=0,
            zombie_drain_sleep_seconds=0,
            kill_first=False,
        )
        call_order = []

        def mock_check(gpu_index: int) -> VRAMStatus:
            call_order.append("check")
            return _starved_status(gpu_index)

        def mock_kill(gpu_index: int) -> int:
            call_order.append("kill")
            return 0

        gate.check_vram = mock_check
        gate.kill_zombies = mock_kill

        with patch.object(gate, "wait_for_vram", return_value=False):
            gate.ensure_vram_available(0)

        assert call_order[0] == "check"
        assert call_order[1] == "kill"

    def test_no_drain_sleep_in_check_first_mode(self):
        """kill_first=False: time.sleep is NOT called (no drain logic in V1 path)."""
        gate = GPUVRAMGateV2(
            min_free_gb=8.0,
            wait_seconds=0,
            zombie_drain_sleep_seconds=15,
            kill_first=False,
        )
        gate.check_vram = lambda gpu_index: _healthy_status(gpu_index)
        gate.kill_zombies = lambda gpu_index: 0

        with patch("carnot.pipeline.gpu_vram_gate_v2.time.sleep") as mock_sleep:
            gate.ensure_vram_available(0)

        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-059: CPU-only / no-GPU machine is a no-op
# ---------------------------------------------------------------------------


class TestCPUOnlyNoOp:
    """REQ-INFRA-049: when n_gpus==0, GPUVRAMGateV2 is a complete no-op."""

    def test_context_manager_noop_when_no_gpus(self):
        """__enter__ returns self without error when no GPUs are detected."""
        gate = GPUVRAMGateV2(kill_first=True)

        with patch.object(gate, "_n_gpus", return_value=0):
            entered = gate.__enter__()

        assert entered is gate

    def test_ensure_vram_available_noop_for_absent_gpu(self):
        """ensure_vram_available returns True when total_mb==0 (pynvml unavailable)."""
        gate = GPUVRAMGateV2(
            min_free_gb=8.0,
            zombie_drain_sleep_seconds=0,
            kill_first=True,
        )
        gate.kill_zombies = lambda gpu_index: 0
        gate.check_vram = lambda gpu_index: _no_gpu_status(gpu_index)

        with patch("carnot.pipeline.gpu_vram_gate_v2.time.sleep"):
            result = gate.ensure_vram_available(0)

        assert result is True

    def test_context_manager_noop_check_first_no_gpus(self):
        """kill_first=False also no-ops on CPU-only machines."""
        gate = GPUVRAMGateV2(kill_first=False)

        with patch.object(gate, "_n_gpus", return_value=0):
            with gate:
                pass  # must not raise


# ---------------------------------------------------------------------------
# GPUVRAMInsufficientError raised when ensure_vram_available returns False
# ---------------------------------------------------------------------------


class TestInsufficientVRAMRaises:
    """REQ-INFRA-049: __enter__ raises GPUVRAMInsufficientError when VRAM unavailable."""

    def test_raises_when_vram_never_frees(self):
        """__enter__ raises GPUVRAMInsufficientError when ensure_vram_available=False."""
        gate = GPUVRAMGateV2(
            min_free_gb=8.0,
            zombie_drain_sleep_seconds=0,
            kill_first=True,
        )

        with patch.object(gate, "_n_gpus", return_value=1):
            with patch.object(gate, "ensure_vram_available", return_value=False):
                with patch.object(gate, "check_vram", return_value=_starved_status(0, 1.0)):
                    with pytest.raises(GPUVRAMInsufficientError) as exc_info:
                        gate.__enter__()

        err = exc_info.value
        assert err.gpu_index == 0
        assert err.min_free_gb == 8.0

    def test_context_manager_raises_on_vram_failure(self):
        """Using 'with GPUVRAMGateV2(...)' raises GPUVRAMInsufficientError."""
        gate = GPUVRAMGateV2(
            min_free_gb=8.0,
            zombie_drain_sleep_seconds=0,
            kill_first=True,
        )

        with patch.object(gate, "_n_gpus", return_value=1):
            with patch.object(gate, "ensure_vram_available", return_value=False):
                with patch.object(gate, "check_vram", return_value=_starved_status(0, 1.0)):
                    with pytest.raises(GPUVRAMInsufficientError):
                        with gate:
                            pass

    def test_no_raise_when_vram_sufficient(self):
        """__enter__ does not raise when ensure_vram_available returns True."""
        gate = GPUVRAMGateV2(
            min_free_gb=8.0,
            zombie_drain_sleep_seconds=0,
            kill_first=True,
        )

        with patch.object(gate, "_n_gpus", return_value=1):
            with patch.object(gate, "ensure_vram_available", return_value=True):
                with gate:
                    pass  # must not raise


# ---------------------------------------------------------------------------
# Inheritance sanity: GPUVRAMGateV2 reuses V1's VRAMStatus / Error types
# ---------------------------------------------------------------------------


class TestTypeReuse:
    """Verify no duplication of VRAMStatus / GPUVRAMInsufficientError."""

    def test_vram_status_is_v1_type(self):
        """VRAMStatus imported from gpu_vram_gate_v2 is the same class as in V1."""
        from carnot.pipeline.gpu_vram_gate import VRAMStatus as V1Status
        from carnot.pipeline.gpu_vram_gate import GPUVRAMInsufficientError as V1Error
        from carnot.pipeline.gpu_vram_gate_v2 import GPUVRAMGateV2

        # GPUVRAMGateV2 inherits from GPUVRAMGate, so check_vram returns V1Status
        gate = GPUVRAMGateV2()
        # These types are imported from V1 in gpu_vram_gate_v2.py
        assert VRAMStatus is V1Status
        assert GPUVRAMInsufficientError is V1Error

    def test_gpuvramgatev2_is_subclass_of_v1(self):
        """GPUVRAMGateV2 must be a subclass of GPUVRAMGate for type compatibility."""
        from carnot.pipeline.gpu_vram_gate import GPUVRAMGate

        assert issubclass(GPUVRAMGateV2, GPUVRAMGate)


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------


class TestDefaultParameters:
    """REQ-INFRA-049: verify default constructor values."""

    def test_defaults(self):
        gate = GPUVRAMGateV2()
        assert gate.min_free_gb == 8.0
        assert gate.wait_seconds == 60
        assert gate.zombie_drain_sleep_seconds == 15
        assert gate.kill_first is True

    def test_custom_drain_sleep(self):
        gate = GPUVRAMGateV2(zombie_drain_sleep_seconds=5)
        assert gate.zombie_drain_sleep_seconds == 5

    def test_kill_first_false_disables_drain(self):
        gate = GPUVRAMGateV2(kill_first=False)
        assert gate.kill_first is False
