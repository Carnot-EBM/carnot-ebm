"""Tests for DualGPUMonitor and ExperimentTemplate.setup_gpu() integration.

Spec coverage:
  REQ-INFRA-003  — GPU zombie process detection (SCENARIO-INFRA-004, SCENARIO-INFRA-006)
  REQ-INFRA-004  — Dual-GPU utilisation check   (SCENARIO-INFRA-005, SCENARIO-INFRA-006)
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.dual_gpu_monitor import DualGPUMonitor, GPUProcessInfo


# ---------------------------------------------------------------------------
# GPUProcessInfo dataclass
# ---------------------------------------------------------------------------


class TestGPUProcessInfo:
    """REQ-INFRA-003: GPUProcessInfo holds per-process GPU metadata."""

    def test_fields_are_accessible(self) -> None:
        """GPUProcessInfo stores pid, gpu_index, vram_mb, utilization_pct, is_zombie."""
        info = GPUProcessInfo(
            pid=12345,
            gpu_index=0,
            vram_mb=600,
            utilization_pct=0,
            is_zombie=True,
        )
        assert info.pid == 12345
        assert info.gpu_index == 0
        assert info.vram_mb == 600
        assert info.utilization_pct == 0
        assert info.is_zombie is True

    def test_non_zombie_has_is_zombie_false(self) -> None:
        """A process with >0 utilisation and >100 MB VRAM is not a zombie."""
        info = GPUProcessInfo(
            pid=99,
            gpu_index=1,
            vram_mb=300,
            utilization_pct=45,
            is_zombie=False,
        )
        assert info.is_zombie is False

    def test_small_vram_zero_util_is_not_zombie_by_construction(self) -> None:
        """Caller determines is_zombie; monitor uses vram_mb>100 AND util==0 rule."""
        # When vram_mb <= 100, is_zombie should be False regardless of util
        info = GPUProcessInfo(
            pid=50,
            gpu_index=0,
            vram_mb=50,
            utilization_pct=0,
            is_zombie=False,
        )
        assert info.is_zombie is False


# ---------------------------------------------------------------------------
# is_zombie classification rule
# ---------------------------------------------------------------------------


class TestIsZombieRule:
    """REQ-INFRA-003: zombie = utilization_pct==0 AND vram_mb>100."""

    @pytest.mark.parametrize(
        "vram_mb,util,expected",
        [
            (600, 0, True),    # large VRAM, zero util → zombie
            (101, 0, True),    # just above threshold
            (100, 0, False),   # exactly at boundary → not zombie (>100 required)
            (50, 0, False),    # small VRAM, zero util → not zombie
            (600, 1, False),   # large VRAM, non-zero util → not zombie
            (600, 100, False), # large VRAM, full util → not zombie
            (0, 0, False),     # zero VRAM → not zombie
        ],
    )
    def test_zombie_classification(
        self, vram_mb: int, util: int, expected: bool
    ) -> None:
        """SCENARIO-INFRA-004: zombie rule applied by DualGPUMonitor._is_zombie()."""
        monitor = DualGPUMonitor()
        assert monitor._is_zombie(vram_mb=vram_mb, utilization_pct=util) is expected


# ---------------------------------------------------------------------------
# DualGPUMonitor.list_gpu_processes()
# ---------------------------------------------------------------------------


_NVIDIA_SMI_COMPUTE_APPS_OUTPUT = (
    "2592400, 0, 600 MiB\n"
    "2595103, 0, 450 MiB\n"
    "1234567, 1, 50 MiB\n"
)

_NVIDIA_SMI_GPU_UTIL_OUTPUT = "0 %\n15 %\n"


class TestListGpuProcesses:
    """REQ-INFRA-003: list_gpu_processes() parses nvidia-smi output."""

    def _make_run(
        self, compute_out: str = _NVIDIA_SMI_COMPUTE_APPS_OUTPUT,
        util_out: str = _NVIDIA_SMI_GPU_UTIL_OUTPUT,
    ):
        """Return a fake subprocess.run that mimics nvidia-smi CSV responses.

        Uses substring matching on each argument (not list-membership) because
        the actual commands contain full flag strings like
        ``--query-compute-apps=pid,gpu_index,used_memory``.
        """
        def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
            cmd_str = " ".join(cmd)
            if "query-compute-apps" in cmd_str:
                return SimpleNamespace(returncode=0, stdout=compute_out, stderr="")
            if "query-gpu" in cmd_str:
                return SimpleNamespace(returncode=0, stdout=util_out, stderr="")
            raise AssertionError(f"Unexpected command: {cmd}")

        return fake_run

    def test_parses_processes_from_nvidia_smi(self) -> None:
        """REQ-INFRA-003: parses pid, gpu_index, vram_mb from nvidia-smi CSV."""
        monitor = DualGPUMonitor()
        with patch("subprocess.run", side_effect=self._make_run()):
            procs = monitor.list_gpu_processes()

        assert len(procs) == 3
        pids = {p.pid for p in procs}
        assert {2592400, 2595103, 1234567} == pids

    def test_zombie_flag_set_for_large_vram_zero_util(self) -> None:
        """SCENARIO-INFRA-004: GPU 0 processes with 600/450 MB at 0% are zombies."""
        monitor = DualGPUMonitor()
        with patch("subprocess.run", side_effect=self._make_run()):
            procs = monitor.list_gpu_processes()

        gpu0_procs = [p for p in procs if p.gpu_index == 0]
        # Both GPU 0 processes are at 0% utilisation → zombies
        assert all(p.is_zombie for p in gpu0_procs)

    def test_small_vram_process_not_zombie(self) -> None:
        """SCENARIO-INFRA-004: GPU 1 process with 50 MB at 15% is NOT a zombie."""
        monitor = DualGPUMonitor()
        with patch("subprocess.run", side_effect=self._make_run()):
            procs = monitor.list_gpu_processes()

        gpu1_procs = [p for p in procs if p.gpu_index == 1]
        assert len(gpu1_procs) == 1
        assert gpu1_procs[0].is_zombie is False

    def test_returns_empty_list_when_nvidia_smi_absent(self) -> None:
        """SCENARIO-INFRA-006: CI-safe — FileNotFoundError returns [] without raising."""
        monitor = DualGPUMonitor()
        with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
            procs = monitor.list_gpu_processes()

        assert procs == []

    def test_returns_empty_list_when_nvidia_smi_fails(self) -> None:
        """SCENARIO-INFRA-006: non-zero returncode is handled gracefully."""
        monitor = DualGPUMonitor()
        failed = SimpleNamespace(returncode=1, stdout="", stderr="no devices")
        with patch("subprocess.run", return_value=failed):
            procs = monitor.list_gpu_processes()

        assert procs == []

    def test_returns_empty_list_when_no_processes(self) -> None:
        """SCENARIO-INFRA-006: empty nvidia-smi output → empty list."""
        monitor = DualGPUMonitor()

        def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            procs = monitor.list_gpu_processes()

        assert procs == []


# ---------------------------------------------------------------------------
# DualGPUMonitor.detect_zombies()
# ---------------------------------------------------------------------------


class TestDetectZombies:
    """REQ-INFRA-003: detect_zombies() filters is_zombie=True processes."""

    def test_detect_zombies_returns_only_zombie_processes(self) -> None:
        """SCENARIO-INFRA-004: two zombies on GPU 0, one healthy on GPU 1."""
        monitor = DualGPUMonitor()
        fake_procs = [
            GPUProcessInfo(pid=1, gpu_index=0, vram_mb=600, utilization_pct=0, is_zombie=True),
            GPUProcessInfo(pid=2, gpu_index=0, vram_mb=450, utilization_pct=0, is_zombie=True),
            GPUProcessInfo(pid=3, gpu_index=1, vram_mb=50, utilization_pct=15, is_zombie=False),
        ]
        with patch.object(monitor, "list_gpu_processes", return_value=fake_procs):
            zombies = monitor.detect_zombies()

        assert len(zombies) == 2
        assert all(z.is_zombie for z in zombies)
        assert {z.pid for z in zombies} == {1, 2}

    def test_detect_zombies_returns_empty_when_none(self) -> None:
        """detect_zombies() returns [] when no processes are zombies."""
        monitor = DualGPUMonitor()
        healthy_procs = [
            GPUProcessInfo(pid=10, gpu_index=0, vram_mb=300, utilization_pct=60, is_zombie=False),
        ]
        with patch.object(monitor, "list_gpu_processes", return_value=healthy_procs):
            zombies = monitor.detect_zombies()

        assert zombies == []


# ---------------------------------------------------------------------------
# DualGPUMonitor.check_dual_gpu_health()
# ---------------------------------------------------------------------------


class TestCheckDualGpuHealth:
    """REQ-INFRA-004: check_dual_gpu_health() returns structured health dict."""

    def _monitor_with_procs(self, procs: list[GPUProcessInfo]) -> DualGPUMonitor:
        monitor = DualGPUMonitor()
        monitor.list_gpu_processes = MagicMock(return_value=procs)  # type: ignore[method-assign]
        return monitor

    def test_healthy_two_gpu_config(self) -> None:
        """SCENARIO-INFRA-005: two GPUs, active processes, no zombies → all_healthy=True."""
        monitor = self._monitor_with_procs([
            GPUProcessInfo(pid=1, gpu_index=0, vram_mb=400, utilization_pct=70, is_zombie=False),
            GPUProcessInfo(pid=2, gpu_index=1, vram_mb=400, utilization_pct=65, is_zombie=False),
        ])
        with patch.object(monitor, "_get_gpu_count", return_value=2):
            health = monitor.check_dual_gpu_health()

        assert health["all_healthy"] is True
        assert health["n_gpus_detected"] == 2
        assert health["n_zombies"] == 0
        assert health["idle_gpus"] == []

    def test_unhealthy_when_zombie_present(self) -> None:
        """REQ-INFRA-003: zombie on GPU 0 makes all_healthy=False."""
        monitor = self._monitor_with_procs([
            GPUProcessInfo(pid=1, gpu_index=0, vram_mb=600, utilization_pct=0, is_zombie=True),
            GPUProcessInfo(pid=2, gpu_index=1, vram_mb=400, utilization_pct=65, is_zombie=False),
        ])
        with patch.object(monitor, "_get_gpu_count", return_value=2):
            health = monitor.check_dual_gpu_health()

        assert health["all_healthy"] is False
        assert health["n_zombies"] == 1

    def test_unhealthy_when_only_one_gpu_detected(self) -> None:
        """REQ-INFRA-004: fewer than 2 GPUs → all_healthy=False."""
        monitor = self._monitor_with_procs([
            GPUProcessInfo(pid=1, gpu_index=0, vram_mb=400, utilization_pct=70, is_zombie=False),
        ])
        with patch.object(monitor, "_get_gpu_count", return_value=1):
            health = monitor.check_dual_gpu_health()

        assert health["all_healthy"] is False
        assert health["n_gpus_detected"] == 1

    def test_unhealthy_when_idle_gpu_detected(self) -> None:
        """REQ-INFRA-004: GPU 1 idle (no processes) → idle_gpus=[1], all_healthy=False."""
        monitor = self._monitor_with_procs([
            GPUProcessInfo(pid=1, gpu_index=0, vram_mb=400, utilization_pct=70, is_zombie=False),
        ])
        # Simulate GPU 1 existing but having no processes
        with patch.object(monitor, "_get_gpu_count", return_value=2):
            health = monitor.check_dual_gpu_health()

        assert health["all_healthy"] is False
        assert 1 in health["idle_gpus"]

    def test_no_gpus_detected_when_nvidia_smi_absent(self) -> None:
        """SCENARIO-INFRA-006: no nvidia-smi → n_gpus_detected=0, all_healthy=False."""
        monitor = DualGPUMonitor()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            health = monitor.check_dual_gpu_health()

        assert health["all_healthy"] is False
        assert health["n_gpus_detected"] == 0
        assert health["n_zombies"] == 0
        assert health["idle_gpus"] == []

    def test_required_keys_present(self) -> None:
        """REQ-INFRA-004: returned dict always has the four required keys."""
        monitor = self._monitor_with_procs([])
        with patch.object(monitor, "_get_gpu_count", return_value=0):
            health = monitor.check_dual_gpu_health()

        for key in ("n_gpus_detected", "n_zombies", "idle_gpus", "all_healthy"):
            assert key in health, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# DualGPUMonitor.to_dict()
# ---------------------------------------------------------------------------


class TestToDictSerialization:
    """REQ-INFRA-004: to_dict() produces a JSON-serialisable artifact."""

    def test_to_dict_contains_health_and_processes(self) -> None:
        """to_dict() bundles both health summary and per-process list."""
        monitor = DualGPUMonitor()
        fake_procs = [
            GPUProcessInfo(pid=1, gpu_index=0, vram_mb=400, utilization_pct=70, is_zombie=False),
        ]
        with patch.object(monitor, "list_gpu_processes", return_value=fake_procs), \
             patch.object(monitor, "_get_gpu_count", return_value=1):
            d = monitor.to_dict()

        assert "health" in d
        assert "processes" in d
        assert isinstance(d["processes"], list)
        assert d["processes"][0]["pid"] == 1

    def test_to_dict_is_json_serialisable(self) -> None:
        """to_dict() must be JSON-serialisable without custom encoder."""
        import json

        monitor = DualGPUMonitor()
        with patch.object(monitor, "list_gpu_processes", return_value=[]), \
             patch.object(monitor, "_get_gpu_count", return_value=0):
            d = monitor.to_dict()

        # Should not raise
        json.dumps(d)


# ---------------------------------------------------------------------------
# ExperimentTemplate.setup_gpu() integration
# ---------------------------------------------------------------------------


class TestSetupGpuIntegration:
    """REQ-INFRA-004: ExperimentTemplate.setup_gpu() adds gpu_monitor_results."""

    def _make_template(self, tmp_path):
        from scripts.experiment_template import ExperimentTemplate

        tmpl = ExperimentTemplate(
            326,
            "Dual GPU Config",
            "results/experiment_326_dual_gpu_config.json",
            repo_root=tmp_path,
        )
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        return tmpl

    def test_setup_gpu_returns_gpu_monitor_results_key(self, tmp_path) -> None:
        """REQ-INFRA-004: returned dict gains gpu_monitor_results key (additive)."""
        from scripts.experiment_template import ExperimentTemplate

        tmpl = self._make_template(tmp_path)

        fake_prewarm = MagicMock(return_value=SimpleNamespace(
            health_ok=True, load_time_s=1.0, stall_root_cause=None
        ))

        result = tmpl.setup_gpu(
            [{"name": "TestModel", "hf_id": "org/test", "gpu": 0}],
            prewarm_fn=fake_prewarm,
        )

        assert "gpu_monitor_results" in result

    def test_setup_gpu_preserves_existing_keys(self, tmp_path) -> None:
        """REQ-INFRA-004: existing keys all_healthy, models, prewarm_time_s are preserved."""
        from scripts.experiment_template import ExperimentTemplate

        tmpl = self._make_template(tmp_path)

        fake_prewarm = MagicMock(return_value=SimpleNamespace(
            health_ok=True, load_time_s=0.5, stall_root_cause=None
        ))

        result = tmpl.setup_gpu(
            [{"name": "TestModel", "hf_id": "org/test", "gpu": 0}],
            prewarm_fn=fake_prewarm,
        )

        for key in ("all_healthy", "models", "prewarm_time_s"):
            assert key in result, f"Existing key missing: {key}"

    def test_setup_gpu_gpu_monitor_results_has_required_keys(self, tmp_path) -> None:
        """REQ-INFRA-004: gpu_monitor_results has n_gpus_detected, n_zombies, etc."""
        from scripts.experiment_template import ExperimentTemplate

        tmpl = self._make_template(tmp_path)

        fake_prewarm = MagicMock(return_value=SimpleNamespace(
            health_ok=True, load_time_s=0.5, stall_root_cause=None
        ))

        result = tmpl.setup_gpu(
            [{"name": "TestModel", "hf_id": "org/test", "gpu": 0}],
            prewarm_fn=fake_prewarm,
        )

        gmr = result["gpu_monitor_results"]
        for key in ("n_gpus_detected", "n_zombies", "idle_gpus", "all_healthy"):
            assert key in gmr, f"Missing gpu_monitor key: {key}"

    def test_setup_gpu_warns_but_does_not_fail_on_unhealthy_monitor(
        self, tmp_path, monkeypatch
    ) -> None:
        """REQ-INFRA-004: unhealthy monitor logs warning; setup_gpu still returns dict."""
        import logging

        from scripts.experiment_template import ExperimentTemplate

        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

        tmpl = self._make_template(tmp_path)

        fake_prewarm = MagicMock(return_value=SimpleNamespace(
            health_ok=True, load_time_s=0.5, stall_root_cause=None
        ))

        # Force an unhealthy monitor result
        unhealthy = {
            "n_gpus_detected": 0,
            "n_zombies": 2,
            "idle_gpus": [0, 1],
            "all_healthy": False,
        }

        with patch(
            "carnot.pipeline.dual_gpu_monitor.DualGPUMonitor.check_dual_gpu_health",
            return_value=unhealthy,
        ):
            with patch(
                "carnot.pipeline.dual_gpu_monitor.DualGPUMonitor.list_gpu_processes",
                return_value=[],
            ):
                result = tmpl.setup_gpu(
                    [{"name": "TestModel", "hf_id": "org/test", "gpu": 0}],
                    prewarm_fn=fake_prewarm,
                )

        # Must still return a dict (no exception raised)
        assert isinstance(result, dict)
        assert "gpu_monitor_results" in result


# ---------------------------------------------------------------------------
# Export availability
# ---------------------------------------------------------------------------


class TestPipelineExports:
    """REQ-INFRA-003/004: DualGPUMonitor and GPUProcessInfo exported from carnot.pipeline."""

    def test_dual_gpu_monitor_exported(self) -> None:
        """DualGPUMonitor is importable from carnot.pipeline."""
        from carnot.pipeline import DualGPUMonitor as Exported

        assert Exported is DualGPUMonitor

    def test_gpu_process_info_exported(self) -> None:
        """GPUProcessInfo is importable from carnot.pipeline."""
        from carnot.pipeline import GPUProcessInfo as Exported

        assert Exported is GPUProcessInfo
