"""Tests for ExpandedGPUReaper, ExpandedGPUReaperConfig, ExpandedGPUReapResult.

Spec: REQ-INFRA-067, REQ-INFRA-068, REQ-INFRA-069,
      SCENARIO-INFRA-076, SCENARIO-INFRA-077, SCENARIO-INFRA-078
"""

from __future__ import annotations

import os
import signal
from unittest.mock import MagicMock, call, patch

import pytest

from carnot.pipeline.expanded_gpu_reaper import (
    ExpandedGPUReaper,
    ExpandedGPUReaperConfig,
    ExpandedGPUReapResult,
)


# ---------------------------------------------------------------------------
# ExpandedGPUReaperConfig
# ---------------------------------------------------------------------------


class TestExpandedGPUReaperConfig:
    def test_defaults(self):
        cfg = ExpandedGPUReaperConfig()
        assert cfg.min_vram_mb == 1024
        assert cfg.min_age_s == 1800
        assert cfg.dry_run is False

    def test_custom_values(self):
        cfg = ExpandedGPUReaperConfig(min_vram_mb=512, min_age_s=300, dry_run=True)
        assert cfg.min_vram_mb == 512
        assert cfg.min_age_s == 300
        assert cfg.dry_run is True


# ---------------------------------------------------------------------------
# ExpandedGPUReapResult
# ---------------------------------------------------------------------------


class TestExpandedGPUReapResult:
    def test_defaults(self):
        r = ExpandedGPUReapResult()
        assert r.killed == []
        assert r.skipped == []
        assert r.total_vram_freed_mb == 0
        assert r.honest_verdict == "reap_complete"

    def test_custom_values(self):
        r = ExpandedGPUReapResult(
            killed=[{"pid": 1}],
            skipped=[{"pid": 2}],
            total_vram_freed_mb=2048,
            honest_verdict="reap_dry_run_complete",
        )
        assert len(r.killed) == 1
        assert r.total_vram_freed_mb == 2048
        assert r.honest_verdict == "reap_dry_run_complete"


# ---------------------------------------------------------------------------
# Helpers: make a reaper with mocked subtree/age helpers
# ---------------------------------------------------------------------------


def _make_reaper(cfg: ExpandedGPUReaperConfig | None = None) -> ExpandedGPUReaper:
    return ExpandedGPUReaper(cfg or ExpandedGPUReaperConfig())


# ---------------------------------------------------------------------------
# _list_gpu_processes
# ---------------------------------------------------------------------------


class TestListGpuProcesses:
    def test_returns_empty_when_no_nvidia_smi(self):
        # SCENARIO-INFRA-078 (CI stub): no nvidia-smi → empty list
        reaper = _make_reaper()
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value=None):
            result = reaper._list_gpu_processes()
        assert result == []

    def test_parses_csv_output(self):
        nvidia_smi_output = "1234, 4096, python3\n5678, 2048, /usr/bin/python\n"
        reaper = _make_reaper()
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch(
                "carnot.pipeline.expanded_gpu_reaper.subprocess.check_output",
                return_value=nvidia_smi_output,
            ):
                result = reaper._list_gpu_processes()
        assert len(result) == 2
        assert result[0] == {"pid": 1234, "used_memory_mb": 4096, "process_name": "python3"}
        assert result[1] == {"pid": 5678, "used_memory_mb": 2048, "process_name": "/usr/bin/python"}

    def test_skips_malformed_lines(self):
        output = "badline\n1234, 4096, python3\n"
        reaper = _make_reaper()
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch(
                "carnot.pipeline.expanded_gpu_reaper.subprocess.check_output",
                return_value=output,
            ):
                result = reaper._list_gpu_processes()
        # Only the valid line should be parsed
        assert len(result) == 1
        assert result[0]["pid"] == 1234

    def test_returns_empty_on_calledprocesserror(self):
        import subprocess as _sub

        reaper = _make_reaper()
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch(
                "carnot.pipeline.expanded_gpu_reaper.subprocess.check_output",
                side_effect=_sub.CalledProcessError(1, "nvidia-smi", "error"),
            ):
                result = reaper._list_gpu_processes()
        assert result == []

    def test_skips_empty_lines(self):
        output = "\n  \n1234, 1000, python\n\n"
        reaper = _make_reaper()
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch(
                "carnot.pipeline.expanded_gpu_reaper.subprocess.check_output",
                return_value=output,
            ):
                result = reaper._list_gpu_processes()
        assert len(result) == 1

    def test_skips_line_with_non_numeric_pid(self):
        # Exercises the ValueError branch when pid/memory can't be cast to int
        output = "notapid, notmb, python\n1234, 1000, python\n"
        reaper = _make_reaper()
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch(
                "carnot.pipeline.expanded_gpu_reaper.subprocess.check_output",
                return_value=output,
            ):
                result = reaper._list_gpu_processes()
        assert len(result) == 1
        assert result[0]["pid"] == 1234


# ---------------------------------------------------------------------------
# _process_age_s
# ---------------------------------------------------------------------------


class TestProcessAgeS:
    def test_returns_age_from_ps(self):
        reaper = _make_reaper()
        with patch(
            "carnot.pipeline.expanded_gpu_reaper.subprocess.check_output",
            return_value="3600\n",
        ):
            age = reaper._process_age_s(1234)
        assert age == 3600

    def test_returns_minus_one_on_failure(self):
        import subprocess as _sub

        reaper = _make_reaper()
        with patch(
            "carnot.pipeline.expanded_gpu_reaper.subprocess.check_output",
            side_effect=_sub.CalledProcessError(1, "ps"),
        ):
            age = reaper._process_age_s(99999)
        assert age == -1

    def test_returns_minus_one_on_valueerror(self):
        reaper = _make_reaper()
        with patch(
            "carnot.pipeline.expanded_gpu_reaper.subprocess.check_output",
            return_value="not_a_number\n",
        ):
            age = reaper._process_age_s(1234)
        assert age == -1


# ---------------------------------------------------------------------------
# _in_our_subtree
# ---------------------------------------------------------------------------


class TestInOurSubtree:
    def test_pid_equals_root(self):
        reaper = _make_reaper()
        assert reaper._in_our_subtree(42, 42) is True

    def test_direct_child(self):
        # /proc/<child>/stat: ppid == root
        reaper = _make_reaper()
        root = 100
        child = 200
        # stat content: "200 (python3) S 100 ..." — ppid is fields[1] after last ')'
        stat_content = f"200 (python3) S {root} 200 200 0 -1 0 0 0\n"

        def fake_open(path, *args, **kwargs):
            if str(child) in path:
                m = MagicMock()
                m.__enter__ = lambda s: MagicMock(read=MagicMock(return_value=stat_content))
                m.__exit__ = MagicMock(return_value=False)
                # Use a real StringIO-like approach
                import io

                return io.StringIO(stat_content)
            raise OSError("not found")

        with patch("builtins.open", side_effect=fake_open):
            result = reaper._in_our_subtree(child, root)
        assert result is True

    def test_unrelated_process(self):
        # Process chain: pid=999 -> ppid=1 (init) — not in our subtree
        reaper = _make_reaper()
        root = 100

        stat_999 = "999 (bash) S 1 999 999 0 -1 0 0 0\n"

        import io

        def fake_open(path, *args, **kwargs):
            if "/proc/999/" in path:
                return io.StringIO(stat_999)
            raise OSError("not found")

        with patch("builtins.open", side_effect=fake_open):
            result = reaper._in_our_subtree(999, root)
        assert result is False

    def test_oserror_returns_false(self):
        reaper = _make_reaper()
        with patch("builtins.open", side_effect=OSError("no such file")):
            result = reaper._in_our_subtree(12345, 1)
        assert result is False

    def test_pid_1_not_in_subtree(self):
        reaper = _make_reaper()
        # root_pid=100, pid=1 → should be False without opening /proc/1/stat
        # (loop terminates when current <= 1)
        result = reaper._in_our_subtree(1, 100)
        assert result is False


# ---------------------------------------------------------------------------
# reap() — no nvidia-smi path (CI stub)
# ---------------------------------------------------------------------------


class TestReapNoNvidiaSmi:
    def test_returns_no_nvidia_smi_verdict(self):
        # SCENARIO-INFRA-078: nvidia-smi not in PATH → no-op
        reaper = _make_reaper()
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value=None):
            result = reaper.reap()
        assert result.honest_verdict == "no_nvidia_smi_no_reap"
        assert result.killed == []
        assert result.skipped == []
        assert result.total_vram_freed_mb == 0


# ---------------------------------------------------------------------------
# reap() — dry_run path
# ---------------------------------------------------------------------------


class TestReapDryRun:
    def _setup_reaper(self, gpu_procs, age_map=None, subtree_pids=None):
        """Return a reaper in dry_run mode with mocked helpers."""
        cfg = ExpandedGPUReaperConfig(min_vram_mb=1024, min_age_s=1800, dry_run=True)
        reaper = ExpandedGPUReaper(cfg)
        reaper._list_gpu_processes = MagicMock(return_value=gpu_procs)
        subtree_set = set(subtree_pids or [])
        reaper._in_our_subtree = MagicMock(
            side_effect=lambda pid, root: pid in subtree_set
        )
        age_map_ = age_map or {}
        reaper._process_age_s = MagicMock(side_effect=lambda pid: age_map_.get(pid, 9999))
        return reaper

    def test_dry_run_candidate_recorded_not_killed(self):
        # SCENARIO-INFRA-076: dry_run=True → candidate goes to skipped with reason='dry_run_candidate'
        procs = [{"pid": 555, "used_memory_mb": 4096, "process_name": "python3"}]
        reaper = self._setup_reaper(procs, age_map={555: 3600})
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            result = reaper.reap()
        assert result.honest_verdict == "reap_dry_run_complete"
        assert result.killed == []
        assert len(result.skipped) == 1
        assert result.skipped[0]["reason"] == "dry_run_candidate"
        assert result.skipped[0]["pid"] == 555
        assert result.total_vram_freed_mb == 0

    def test_below_min_vram_skipped(self):
        procs = [{"pid": 111, "used_memory_mb": 100, "process_name": "tiny"}]
        reaper = self._setup_reaper(procs)
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            result = reaper.reap()
        assert result.skipped[0]["reason"] == "below_min_vram"

    def test_in_subtree_skipped(self):
        procs = [{"pid": 222, "used_memory_mb": 8192, "process_name": "conductor_child"}]
        reaper = self._setup_reaper(procs, age_map={222: 3600}, subtree_pids=[222])
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            result = reaper.reap()
        assert result.skipped[0]["reason"] == "in_our_subtree"

    def test_below_min_age_skipped(self):
        procs = [{"pid": 333, "used_memory_mb": 8192, "process_name": "fresh_worker"}]
        reaper = self._setup_reaper(procs, age_map={333: 60})  # 60s < 1800s
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            result = reaper.reap()
        assert result.skipped[0]["reason"] == "below_min_age"

    def test_age_minus_one_still_candidate(self):
        # age_s==-1 means process already gone → treat as eligible
        procs = [{"pid": 444, "used_memory_mb": 8192, "process_name": "ghost"}]
        reaper = self._setup_reaper(procs, age_map={444: -1})
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            result = reaper.reap()
        assert result.skipped[0]["reason"] == "dry_run_candidate"


# ---------------------------------------------------------------------------
# reap() — live kill path
# ---------------------------------------------------------------------------


class TestReapLiveKill:
    def _setup_reaper(self, gpu_procs, age_map=None, subtree_pids=None):
        cfg = ExpandedGPUReaperConfig(min_vram_mb=1024, min_age_s=1800, dry_run=False)
        reaper = ExpandedGPUReaper(cfg)
        reaper._list_gpu_processes = MagicMock(return_value=gpu_procs)
        subtree_set = set(subtree_pids or [])
        reaper._in_our_subtree = MagicMock(
            side_effect=lambda pid, root: pid in subtree_set
        )
        age_map_ = age_map or {}
        reaper._process_age_s = MagicMock(side_effect=lambda pid: age_map_.get(pid, 9999))
        return reaper

    def test_eligible_process_is_killed(self):
        # SCENARIO-INFRA-077: eligible process → killed, vram_freed updated
        procs = [{"pid": 777, "used_memory_mb": 6144, "process_name": "stale_pytest"}]
        reaper = self._setup_reaper(procs, age_map={777: 7200})
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("carnot.pipeline.expanded_gpu_reaper.os.kill") as mock_kill:
                result = reaper.reap()
        mock_kill.assert_called_once_with(777, signal.SIGKILL)
        assert len(result.killed) == 1
        assert result.killed[0]["pid"] == 777
        assert result.killed[0]["action"] == "killed"
        assert result.total_vram_freed_mb == 6144
        assert result.honest_verdict == "reap_complete"
        assert result.skipped == []

    def test_process_lookup_error_goes_to_skipped(self):
        procs = [{"pid": 888, "used_memory_mb": 4096, "process_name": "gone"}]
        reaper = self._setup_reaper(procs, age_map={888: 7200})
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch(
                "carnot.pipeline.expanded_gpu_reaper.os.kill",
                side_effect=ProcessLookupError("no such process"),
            ):
                result = reaper.reap()
        assert result.skipped[0]["reason"] == "kill_error"
        assert result.killed == []
        assert result.total_vram_freed_mb == 0

    def test_permission_error_goes_to_skipped(self):
        procs = [{"pid": 999, "used_memory_mb": 4096, "process_name": "root_owned"}]
        reaper = self._setup_reaper(procs, age_map={999: 7200})
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch(
                "carnot.pipeline.expanded_gpu_reaper.os.kill",
                side_effect=PermissionError("not permitted"),
            ):
                result = reaper.reap()
        assert result.skipped[0]["reason"] == "kill_error"

    def test_multiple_processes_mixed_outcomes(self):
        procs = [
            {"pid": 100, "used_memory_mb": 50, "process_name": "tiny"},   # below_min_vram
            {"pid": 200, "used_memory_mb": 4096, "process_name": "child"},  # in subtree
            {"pid": 300, "used_memory_mb": 4096, "process_name": "fresh"},  # below_min_age
            {"pid": 400, "used_memory_mb": 8192, "process_name": "stale"},  # kill
        ]
        age_map = {200: 9999, 300: 60, 400: 9999}
        reaper = self._setup_reaper(procs, age_map=age_map, subtree_pids=[200])
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("carnot.pipeline.expanded_gpu_reaper.os.kill"):
                result = reaper.reap()
        assert len(result.killed) == 1
        assert result.killed[0]["pid"] == 400
        assert len(result.skipped) == 3
        assert result.total_vram_freed_mb == 8192

    def test_verdict_is_reap_complete_for_live_run(self):
        reaper = self._setup_reaper([])
        with patch("carnot.pipeline.expanded_gpu_reaper.shutil.which", return_value="/usr/bin/nvidia-smi"):
            result = reaper.reap()
        assert result.honest_verdict == "reap_complete"


# ---------------------------------------------------------------------------
# __init__ export
# ---------------------------------------------------------------------------


class TestInitExports:
    def test_importable_from_pipeline(self):
        from carnot.pipeline import (  # noqa: F401
            ExpandedGPUReaper,
            ExpandedGPUReaperConfig,
            ExpandedGPUReapResult,
        )
