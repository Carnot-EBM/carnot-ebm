"""Tests for python/carnot/pipeline/session_health_check.py — 100% coverage.

Coverage targets
----------------
- GPUHealth dataclass and computed properties (is_zombie_saturated, is_overheating, is_idle)
- ZombieProcess dataclass and computed property (should_kill)
- SessionHealthResult dataclass and to_dict()
- ConductorSessionHealthCheck.run() — non-destructive (auto_remediate=False)
- ConductorSessionHealthCheck._check_env() — both True and False paths
- ConductorSessionHealthCheck._check_gpu_health() — pynvml absent path
- ConductorSessionHealthCheck._find_zombie_processes() — empty and pynvml absent paths
- ConductorSessionHealthCheck._kill_zombies() — count returned
- ConductorSessionHealthCheck._get_process_age_s() — psutil absent path

Spec: REQ-INFRA-036, REQ-INFRA-037, REQ-INFRA-038,
      SCENARIO-INFRA-044, SCENARIO-INFRA-045, SCENARIO-INFRA-046
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.session_health_check import (  # noqa: E402
    ConductorSessionHealthCheck,
    GPUHealth,
    SessionHealthResult,
    ZombieProcess,
)


# ---------------------------------------------------------------------------
# GPUHealth
# ---------------------------------------------------------------------------


class TestGPUHealth:
    """REQ-INFRA-036: GPUHealth computed properties."""

    def test_zombie_saturated_true(self) -> None:
        """SCENARIO-INFRA-044: 23 GB held at 0% util → zombie signature."""
        g = GPUHealth(gpu_index=0, vram_used_mb=23000, utilization_pct=0, temp_c=45)
        assert g.is_zombie_saturated is True

    def test_zombie_saturated_false_low_vram(self) -> None:
        """Low VRAM + 0% util is NOT zombie saturated (driver context only)."""
        g = GPUHealth(gpu_index=0, vram_used_mb=100, utilization_pct=0, temp_c=45)
        assert g.is_zombie_saturated is False

    def test_zombie_saturated_false_active_util(self) -> None:
        """High VRAM + nonzero util = active model, not zombie."""
        g = GPUHealth(gpu_index=0, vram_used_mb=20000, utilization_pct=80, temp_c=60)
        assert g.is_zombie_saturated is False

    def test_zombie_saturated_boundary_exact_1000(self) -> None:
        """Exactly 1000 MB at 0% util is NOT zombie (boundary: must be > 1000)."""
        g = GPUHealth(gpu_index=0, vram_used_mb=1000, utilization_pct=0, temp_c=45)
        assert g.is_zombie_saturated is False

    def test_zombie_saturated_boundary_1001(self) -> None:
        """1001 MB at 0% util IS zombie."""
        g = GPUHealth(gpu_index=0, vram_used_mb=1001, utilization_pct=0, temp_c=45)
        assert g.is_zombie_saturated is True

    def test_is_overheating_true(self) -> None:
        """SCENARIO-INFRA-046: 82°C triggers thermal gate."""
        g = GPUHealth(gpu_index=0, vram_used_mb=100, utilization_pct=0, temp_c=82)
        assert g.is_overheating is True

    def test_is_overheating_false(self) -> None:
        """79°C is below the 80°C threshold."""
        g = GPUHealth(gpu_index=0, vram_used_mb=100, utilization_pct=0, temp_c=79)
        assert g.is_overheating is False

    def test_is_overheating_boundary_80(self) -> None:
        """Exactly 80°C triggers thermal gate (>= 80)."""
        g = GPUHealth(gpu_index=0, vram_used_mb=100, utilization_pct=0, temp_c=80)
        assert g.is_overheating is True

    def test_is_idle_true(self) -> None:
        """SCENARIO-INFRA-044: 100 MB → idle (driver context only)."""
        g = GPUHealth(gpu_index=0, vram_used_mb=100, utilization_pct=0, temp_c=45)
        assert g.is_idle is True

    def test_is_idle_false(self) -> None:
        """200 MB is NOT idle (boundary: must be < 200)."""
        g = GPUHealth(gpu_index=0, vram_used_mb=200, utilization_pct=0, temp_c=45)
        assert g.is_idle is False

    def test_is_idle_boundary_199(self) -> None:
        """199 MB IS idle."""
        g = GPUHealth(gpu_index=0, vram_used_mb=199, utilization_pct=0, temp_c=45)
        assert g.is_idle is True

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict() must include all fields for artifact serialisation."""
        g = GPUHealth(gpu_index=1, vram_used_mb=5000, utilization_pct=0, temp_c=45)
        d = g.to_dict()
        assert d["gpu_index"] == 1
        assert d["vram_used_mb"] == 5000
        assert d["utilization_pct"] == 0
        assert d["temp_c"] == 45
        assert "is_zombie_saturated" in d
        assert "is_overheating" in d
        assert "is_idle" in d


# ---------------------------------------------------------------------------
# ZombieProcess
# ---------------------------------------------------------------------------


class TestZombieProcess:
    """REQ-INFRA-037: ZombieProcess.should_kill logic."""

    def test_should_kill_true(self) -> None:
        """SCENARIO-INFRA-045: 11.5-hour-old process holding 5.4 GB → should kill."""
        z = ZombieProcess(pid=12345, gpu_index=0, vram_mb=5418, wall_time_s=41335)
        assert z.should_kill is True

    def test_should_kill_false_young(self) -> None:
        """Process started 60 seconds ago — not a zombie candidate yet."""
        z = ZombieProcess(pid=12345, gpu_index=0, vram_mb=5418, wall_time_s=60)
        assert z.should_kill is False

    def test_should_kill_false_small_vram(self) -> None:
        """Process is old but holds only 200 MB — not significant enough to kill."""
        z = ZombieProcess(pid=12345, gpu_index=0, vram_mb=200, wall_time_s=41335)
        assert z.should_kill is False

    def test_should_kill_boundary_wall_time_300(self) -> None:
        """Exactly 300 seconds wall time is NOT enough (must be > 300)."""
        z = ZombieProcess(pid=12345, gpu_index=0, vram_mb=5000, wall_time_s=300)
        assert z.should_kill is False

    def test_should_kill_boundary_wall_time_301(self) -> None:
        """301 seconds IS enough."""
        z = ZombieProcess(pid=12345, gpu_index=0, vram_mb=5000, wall_time_s=301)
        assert z.should_kill is True

    def test_should_kill_boundary_vram_500(self) -> None:
        """Exactly 500 MB is NOT enough (must be > 500)."""
        z = ZombieProcess(pid=12345, gpu_index=0, vram_mb=500, wall_time_s=400)
        assert z.should_kill is False

    def test_should_kill_boundary_vram_501(self) -> None:
        """501 MB IS enough."""
        z = ZombieProcess(pid=12345, gpu_index=0, vram_mb=501, wall_time_s=400)
        assert z.should_kill is True

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict() round-trips correctly."""
        z = ZombieProcess(pid=99, gpu_index=1, vram_mb=6000, wall_time_s=3600.7)
        d = z.to_dict()
        assert d["pid"] == 99
        assert d["gpu_index"] == 1
        assert d["vram_mb"] == 6000
        assert d["should_kill"] is True
        assert isinstance(d["wall_time_s"], float)


# ---------------------------------------------------------------------------
# SessionHealthResult
# ---------------------------------------------------------------------------


class TestSessionHealthResult:
    """SessionHealthResult: to_dict() serialises all fields."""

    def test_to_dict_healthy(self) -> None:
        r = SessionHealthResult(
            env_ok=True,
            gpu_ok=True,
            zombies_killed=0,
            thermal_ok=True,
            honest_verdict="session_healthy",
        )
        d = r.to_dict()
        assert d["env_ok"] is True
        assert d["gpu_ok"] is True
        assert d["zombies_killed"] == 0
        assert d["thermal_ok"] is True
        assert d["honest_verdict"] == "session_healthy"

    def test_to_dict_thermal_blocked(self) -> None:
        r = SessionHealthResult(
            env_ok=True,
            gpu_ok=False,
            zombies_killed=0,
            thermal_ok=False,
            honest_verdict="session_thermal_blocked",
        )
        assert r.to_dict()["honest_verdict"] == "session_thermal_blocked"


# ---------------------------------------------------------------------------
# ConductorSessionHealthCheck.run() — non-destructive (CI safe)
# ---------------------------------------------------------------------------


class TestConductorSessionHealthCheckRun:
    """REQ-INFRA-036: run() returns SessionHealthResult without killing processes."""

    def test_run_returns_session_health_result(self) -> None:
        """SCENARIO-INFRA-044: run(auto_remediate=False) always returns SessionHealthResult."""
        chk = ConductorSessionHealthCheck(auto_remediate=False)
        result = chk.run()
        assert isinstance(result, SessionHealthResult)

    def test_run_no_destructive_action_in_ci(self) -> None:
        """auto_remediate=False must never kill processes (CI safety)."""
        chk = ConductorSessionHealthCheck(auto_remediate=False)
        with patch.object(chk, "_kill_zombies") as mock_kill:
            chk.run()
            mock_kill.assert_not_called()

    def test_run_result_has_honest_verdict(self) -> None:
        """honest_verdict must be one of the three valid strings."""
        chk = ConductorSessionHealthCheck(auto_remediate=False)
        result = chk.run()
        assert result.honest_verdict in (
            "session_healthy",
            "session_remediated",
            "session_thermal_blocked",
        )

    def test_run_env_ok_reflects_carnot_force_live(self) -> None:
        """env_ok mirrors CARNOT_FORCE_LIVE presence after autofix."""
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}, clear=False):
            result = ConductorSessionHealthCheck(auto_remediate=False).run()
            assert result.env_ok is True

    def test_run_no_gpus_gpu_ok_true(self) -> None:
        """When no GPUs are present, gpu_ok defaults to True (CI / CPU-only)."""
        chk = ConductorSessionHealthCheck(auto_remediate=False)
        with patch.object(chk, "_check_gpu_health", return_value=[]):
            result = chk.run()
        assert result.gpu_ok is True

    def test_run_no_gpus_thermal_ok_true(self) -> None:
        """When no GPUs are present, thermal_ok defaults to True."""
        chk = ConductorSessionHealthCheck(auto_remediate=False)
        with patch.object(chk, "_check_gpu_health", return_value=[]):
            result = chk.run()
        assert result.thermal_ok is True

    def test_run_with_zombie_saturated_gpu_no_remediate(self) -> None:
        """Zombie-saturated GPU detected but auto_remediate=False → zombies_killed=0."""
        chk = ConductorSessionHealthCheck(auto_remediate=False)
        zombie_gpu = GPUHealth(gpu_index=0, vram_used_mb=23000, utilization_pct=0, temp_c=45)
        with patch.object(chk, "_check_gpu_health", return_value=[zombie_gpu]):
            with patch.object(chk, "_find_zombie_processes", return_value=[]):
                result = chk.run()
        assert result.zombies_killed == 0

    def test_run_thermal_blocked_verdict(self) -> None:
        """SCENARIO-INFRA-046: overheating GPU → session_thermal_blocked verdict."""
        chk = ConductorSessionHealthCheck(auto_remediate=False)
        hot_gpu = GPUHealth(gpu_index=0, vram_used_mb=100, utilization_pct=0, temp_c=82)
        with patch.object(chk, "_check_gpu_health", return_value=[hot_gpu]):
            result = chk.run()
        assert result.honest_verdict == "session_thermal_blocked"
        assert result.thermal_ok is False

    def test_run_remediated_verdict_after_zombie_kill(self) -> None:
        """After killing zombies, verdict is session_remediated."""
        chk = ConductorSessionHealthCheck(auto_remediate=True)
        zombie_gpu = GPUHealth(gpu_index=0, vram_used_mb=23000, utilization_pct=0, temp_c=45)
        idle_gpu = GPUHealth(gpu_index=0, vram_used_mb=100, utilization_pct=0, temp_c=45)
        z = ZombieProcess(pid=99, gpu_index=0, vram_mb=20000, wall_time_s=50000)
        call_count = [0]

        def mock_gpu_health():
            call_count[0] += 1
            if call_count[0] == 1:
                return [zombie_gpu]
            return [idle_gpu]

        with patch.object(chk, "_check_gpu_health", side_effect=mock_gpu_health):
            with patch.object(chk, "_find_zombie_processes", return_value=[z]):
                with patch.object(chk, "_kill_zombies", return_value=1) as mock_kill:
                    with patch("time.sleep"):
                        result = chk.run()
        mock_kill.assert_called_once_with([z])
        assert result.zombies_killed == 1
        assert result.honest_verdict == "session_remediated"


# ---------------------------------------------------------------------------
# ConductorSessionHealthCheck._check_env()
# ---------------------------------------------------------------------------


class TestCheckEnv:
    """REQ-INFRA-036: _check_env() correctly reads CARNOT_FORCE_LIVE."""

    def test_env_ok_when_set(self) -> None:
        """_check_env() returns True when CARNOT_FORCE_LIVE=1."""
        chk = ConductorSessionHealthCheck()
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}, clear=False):
            assert chk._check_env() is True

    def test_env_not_ok_when_absent(self) -> None:
        """_check_env() returns False when CARNOT_FORCE_LIVE is not set."""
        chk = ConductorSessionHealthCheck()
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with patch.dict(os.environ, env, clear=True):
            assert chk._check_env() is False

    def test_env_not_ok_when_not_1(self) -> None:
        """_check_env() returns False when CARNOT_FORCE_LIVE=0."""
        chk = ConductorSessionHealthCheck()
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}, clear=False):
            assert chk._check_env() is False


# ---------------------------------------------------------------------------
# ConductorSessionHealthCheck._check_gpu_health()
# ---------------------------------------------------------------------------


class TestCheckGPUHealth:
    """REQ-INFRA-036: _check_gpu_health() degrades gracefully without pynvml."""

    def test_returns_empty_list_when_pynvml_absent(self) -> None:
        """SCENARIO-INFRA-044: no pynvml → empty list (CI safe)."""
        chk = ConductorSessionHealthCheck()
        with patch.dict("sys.modules", {"pynvml": None}):
            result = chk._check_gpu_health()
        assert result == []

    def test_returns_empty_list_on_pynvml_error(self) -> None:
        """Any pynvml exception → empty list, no crash."""
        chk = ConductorSessionHealthCheck()
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = RuntimeError("driver not loaded")
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = chk._check_gpu_health()
        assert result == []

    def test_returns_gpu_health_list_when_pynvml_works(self) -> None:
        """When pynvml is available, returns one GPUHealth per GPU."""
        chk = ConductorSessionHealthCheck()
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_mem = MagicMock()
        mock_mem.used = 100 * 1024 * 1024  # 100 MB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_util = MagicMock()
        mock_util.gpu = 0
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 45
        mock_pynvml.NVML_TEMPERATURE_GPU = 0
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = chk._check_gpu_health()
        assert len(result) == 1
        assert isinstance(result[0], GPUHealth)
        assert result[0].vram_used_mb == 100
        assert result[0].temp_c == 45


# ---------------------------------------------------------------------------
# ConductorSessionHealthCheck._find_zombie_processes()
# ---------------------------------------------------------------------------


class TestFindZombieProcesses:
    """REQ-INFRA-037: zombie detection."""

    def test_returns_empty_when_no_gpu_indices(self) -> None:
        """No zombie-saturated GPUs → no zombie scan needed."""
        chk = ConductorSessionHealthCheck()
        result = chk._find_zombie_processes([])
        assert result == []

    def test_returns_empty_when_pynvml_absent(self) -> None:
        """pynvml absent → empty list, no crash."""
        chk = ConductorSessionHealthCheck()
        with patch.dict("sys.modules", {"pynvml": None}):
            result = chk._find_zombie_processes([0])
        assert result == []

    def test_finds_zombie_when_process_is_old_and_large(self) -> None:
        """SCENARIO-INFRA-045: old, large process on saturated GPU → found."""
        chk = ConductorSessionHealthCheck()
        mock_pynvml = MagicMock()
        mock_proc = MagicMock()
        mock_proc.pid = 42
        mock_proc.usedGpuMemory = 5418 * 1024 * 1024
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [mock_proc]
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch.object(chk, "_get_process_age_s", return_value=41335.0):
                result = chk._find_zombie_processes([0])

        assert len(result) == 1
        assert result[0].pid == 42
        assert result[0].should_kill is True

    def test_skips_young_processes(self) -> None:
        """Young process (60s) not included in zombie list."""
        chk = ConductorSessionHealthCheck()
        mock_pynvml = MagicMock()
        mock_proc = MagicMock()
        mock_proc.pid = 43
        mock_proc.usedGpuMemory = 5418 * 1024 * 1024
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [mock_proc]

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch.object(chk, "_get_process_age_s", return_value=60.0):
                result = chk._find_zombie_processes([0])

        assert result == []

    def test_handles_nvml_get_processes_exception(self) -> None:
        """If nvmlDeviceGetComputeRunningProcesses raises, treats as empty process list."""
        chk = ConductorSessionHealthCheck()
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.side_effect = RuntimeError("nvml error")

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = chk._find_zombie_processes([0])

        assert result == []


# ---------------------------------------------------------------------------
# ConductorSessionHealthCheck._get_process_age_s()
# ---------------------------------------------------------------------------


class TestGetProcessAgeS:
    """_get_process_age_s() returns 0.0 when psutil unavailable."""

    def test_returns_zero_when_psutil_absent(self) -> None:
        """If psutil is not installed, return 0.0 (safe default, won't kill)."""
        chk = ConductorSessionHealthCheck()
        with patch.dict("sys.modules", {"psutil": None}):
            age = chk._get_process_age_s(99999, time.time())
        assert age == 0.0

    def test_returns_zero_on_no_such_process(self) -> None:
        """If the process has already exited, return 0.0."""
        chk = ConductorSessionHealthCheck()
        mock_psutil = MagicMock()
        mock_psutil.Process.side_effect = Exception("no such process")
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            age = chk._get_process_age_s(99999, time.time())
        assert age == 0.0

    def test_returns_age_when_psutil_works(self) -> None:
        """When psutil is available, returns positive age."""
        chk = ConductorSessionHealthCheck()
        now = time.time()
        mock_psutil = MagicMock()
        mock_proc = MagicMock()
        mock_proc.create_time.return_value = now - 1000.0
        mock_psutil.Process.return_value = mock_proc
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            age = chk._get_process_age_s(42, now)
        assert abs(age - 1000.0) < 1.0


# ---------------------------------------------------------------------------
# ConductorSessionHealthCheck._kill_zombies()
# ---------------------------------------------------------------------------


class TestKillZombies:
    """REQ-INFRA-037: _kill_zombies() returns count and handles errors."""

    def test_returns_count_killed(self) -> None:
        """SCENARIO-INFRA-045: count returned = number of PIDs processed."""
        chk = ConductorSessionHealthCheck()
        zombies = [
            ZombieProcess(pid=100, gpu_index=0, vram_mb=6000, wall_time_s=40000),
            ZombieProcess(pid=101, gpu_index=0, vram_mb=6000, wall_time_s=40000),
        ]
        with patch("os.kill") as mock_kill:
            count = chk._kill_zombies(zombies)
        assert count == 2
        assert mock_kill.call_count == 2

    def test_counts_already_dead_process(self) -> None:
        """ProcessLookupError (already dead) still counts as killed."""
        chk = ConductorSessionHealthCheck()
        zombies = [ZombieProcess(pid=999, gpu_index=0, vram_mb=6000, wall_time_s=40000)]
        with patch("os.kill", side_effect=ProcessLookupError):
            count = chk._kill_zombies(zombies)
        assert count == 1

    def test_permission_error_not_counted(self) -> None:
        """PermissionError → not counted (can't kill), logged as error."""
        chk = ConductorSessionHealthCheck()
        zombies = [ZombieProcess(pid=1, gpu_index=0, vram_mb=6000, wall_time_s=40000)]
        with patch("os.kill", side_effect=PermissionError("operation not permitted")):
            count = chk._kill_zombies(zombies)
        assert count == 0

    def test_empty_list_returns_zero(self) -> None:
        """No zombies → return 0 immediately."""
        chk = ConductorSessionHealthCheck()
        with patch("os.kill") as mock_kill:
            count = chk._kill_zombies([])
        assert count == 0
        mock_kill.assert_not_called()
