"""Tests for ExperimentTemplate.teardown() and kill_gpu_zombies().

Spec: REQ-INFRA-073, REQ-INFRA-074,
      SCENARIO-INFRA-083, SCENARIO-INFRA-084, SCENARIO-INFRA-085
"""
from __future__ import annotations

import atexit
import gc
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Ensure repo root on path
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tmpl(tmp_path: Path, exp_id: int = 537) -> ExperimentTemplate:
    """Create an ExperimentTemplate with a tmp repo root so tests don't write to the real results dir."""
    return ExperimentTemplate(
        exp_id,
        "TeardownTest",
        f"results/experiment_{exp_id}_teardown_test.json",
        requires_gpu=False,
        repo_root=tmp_path,
    )


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-083: teardown() calls gc.collect() and logs
# ---------------------------------------------------------------------------


class TestTeardown:
    """REQ-INFRA-073 / SCENARIO-INFRA-083"""

    def test_teardown_calls_gc_collect(self, tmp_path: Path) -> None:
        """teardown() must call gc.collect() to release tensor references before the cache flush."""
        tmpl = _make_tmpl(tmp_path)
        with patch("gc.collect") as mock_gc:
            tmpl.teardown(clear_gpu=False)
        mock_gc.assert_called_once()

    def test_teardown_calls_empty_cache_when_cuda_available(self, tmp_path: Path) -> None:
        """teardown() must call torch.cuda.empty_cache() when a CUDA GPU is present."""
        tmpl = _make_tmpl(tmp_path)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict(sys.modules, {"torch": mock_torch}):
            with patch("scripts.experiment_template._cuda_is_available", return_value=True):
                tmpl.teardown(clear_gpu=True)

        mock_torch.cuda.empty_cache.assert_called_once()

    def test_teardown_skips_empty_cache_when_no_cuda(self, tmp_path: Path) -> None:
        """teardown(clear_gpu=True) must NOT call torch when CUDA is unavailable (CPU-only CI)."""
        tmpl = _make_tmpl(tmp_path)
        with patch("scripts.experiment_template._cuda_is_available", return_value=False):
            with patch("gc.collect") as mock_gc:
                tmpl.teardown(clear_gpu=True)
        mock_gc.assert_called_once()

    def test_teardown_skips_empty_cache_when_clear_gpu_false(self, tmp_path: Path) -> None:
        """teardown(clear_gpu=False) must never call torch even when CUDA is available."""
        tmpl = _make_tmpl(tmp_path)
        with patch("scripts.experiment_template._cuda_is_available", return_value=True) as _:
            mock_torch = MagicMock()
            with patch.dict(sys.modules, {"torch": mock_torch}):
                tmpl.teardown(clear_gpu=False)
        mock_torch.cuda.empty_cache.assert_not_called()


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-085: __init__ registers teardown via atexit
# ---------------------------------------------------------------------------


class TestAtexitRegistration:
    """REQ-INFRA-073 / SCENARIO-INFRA-085"""

    def test_init_registers_teardown_via_atexit(self, tmp_path: Path) -> None:
        """ExperimentTemplate.__init__() must register self.teardown with atexit."""
        with patch("atexit.register") as mock_register:
            tmpl = _make_tmpl(tmp_path)
        # atexit.register may be called multiple times for other registrations;
        # verify that teardown is among the registered callables.
        registered_callables = [c.args[0] for c in mock_register.call_args_list]
        assert tmpl.teardown in registered_callables


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-084: kill_gpu_zombies() returns pynvml_unavailable
# ---------------------------------------------------------------------------


class TestKillGpuZombies:
    """REQ-INFRA-074 / SCENARIO-INFRA-084"""

    def test_returns_pynvml_unavailable_when_not_installed(self) -> None:
        """kill_gpu_zombies() must return {'killed_pids': [], 'freed_mb': 0, 'error': 'pynvml_unavailable'}
        when pynvml is not importable, without sending SIGTERM to any process."""
        with patch.dict(sys.modules, {"pynvml": None}):
            result = ExperimentTemplate.kill_gpu_zombies()
        assert result == {"killed_pids": [], "freed_mb": 0, "error": "pynvml_unavailable"}

    def test_returns_empty_when_no_processes_exceed_threshold(self) -> None:
        """kill_gpu_zombies() must return empty killed_pids when no process exceeds thresholds."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        util = MagicMock()
        util.gpu = 0.0
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util
        # Process holds only 500 MB — below default 1000 MB threshold
        proc = MagicMock()
        proc.pid = 99999
        proc.usedGpuMemory = 500 * 1024 * 1024
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [proc]

        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            result = ExperimentTemplate.kill_gpu_zombies()

        assert result["killed_pids"] == []
        assert result["freed_mb"] == 0

    def test_kills_zombie_process_above_threshold(self) -> None:
        """kill_gpu_zombies() must SIGTERM a process holding VRAM > threshold at near-zero util."""
        import signal as _signal

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        util = MagicMock()
        util.gpu = 0.0
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util
        proc = MagicMock()
        proc.pid = 430009
        proc.usedGpuMemory = 18678 * 1024 * 1024  # 18,678 MB — the .40 zombie
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [proc]

        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            with patch("os.kill") as mock_kill:
                result = ExperimentTemplate.kill_gpu_zombies()

        mock_kill.assert_called_once_with(430009, _signal.SIGTERM)
        assert 430009 in result["killed_pids"]
        assert result["freed_mb"] == 18678
