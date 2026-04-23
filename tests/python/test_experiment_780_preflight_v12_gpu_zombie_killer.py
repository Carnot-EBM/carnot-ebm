"""Tests for Exp 780: GPU Zombie Killer (gpu_zombie_killer.py).

Each test traces to a spec requirement.  Tests are designed to run without a real
GPU — all nvidia-smi calls are mocked so the test suite passes on CPU-only CI.

Spec: REQ-INFRA-055, REQ-INFRA-056, SCENARIO-INFRA-064, SCENARIO-INFRA-065
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.gpu_zombie_killer import (
    GPUZombieResult,
    _get_vram_used_mb,
    _parse_float_from_smi_output,
    _query_nvidia_smi,
    get_gpu_memory_pids,
    kill_gpu_zombies,
)


# ---------------------------------------------------------------------------
# Tests for get_gpu_memory_pids()
# ---------------------------------------------------------------------------


class TestGetGpuMemoryPids:
    """Unit tests for get_gpu_memory_pids().

    Spec: REQ-INFRA-055, REQ-INFRA-056
    """

    def test_returns_empty_list_when_no_processes(self) -> None:
        """No compute processes → empty list returned.

        SCENARIO-INFRA-064: clean GPU reports no PIDs.
        Spec: REQ-INFRA-056
        """
        with patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", return_value=""):
            pids = get_gpu_memory_pids(0)
        assert pids == []

    def test_parses_single_pid(self) -> None:
        """Single-line nvidia-smi output → one PID extracted.

        Spec: REQ-INFRA-055
        """
        with patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", return_value="12345\n"):
            pids = get_gpu_memory_pids(0)
        assert pids == [12345]

    def test_parses_multiple_pids(self) -> None:
        """Multi-line nvidia-smi output → all PIDs extracted.

        Spec: REQ-INFRA-055
        """
        with patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", return_value="111\n222\n333\n"):
            pids = get_gpu_memory_pids(0)
        assert pids == [111, 222, 333]

    def test_returns_empty_when_smi_unavailable(self) -> None:
        """nvidia-smi unavailable (returns None) → empty list, no exception.

        Spec: REQ-INFRA-056
        """
        with patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", return_value=None):
            pids = get_gpu_memory_pids(0)
        assert pids == []

    def test_skips_non_numeric_lines(self) -> None:
        """Malformed output lines are silently skipped.

        Spec: REQ-INFRA-055
        """
        with patch(
            "carnot.pipeline.gpu_zombie_killer._query_nvidia_smi",
            return_value="999\nbad_line\n888\n",
        ):
            pids = get_gpu_memory_pids(0)
        assert pids == [999, 888]


# ---------------------------------------------------------------------------
# Tests for kill_gpu_zombies()
# ---------------------------------------------------------------------------


class TestKillGpuZombies:
    """Unit tests for kill_gpu_zombies().

    Spec: REQ-INFRA-055, REQ-INFRA-056, SCENARIO-INFRA-064, SCENARIO-INFRA-065
    """

    def test_no_zombies_found_on_empty_gpu(self) -> None:
        """Empty GPU → no_zombies_found verdict, pids_killed=[].

        SCENARIO-INFRA-064: clean GPU state produces no-op result.
        Spec: REQ-INFRA-056
        """
        with (
            patch(
                "carnot.pipeline.gpu_zombie_killer._query_nvidia_smi",
                return_value="0\n",  # vram query returns 0
            ),
            patch(
                "carnot.pipeline.gpu_zombie_killer.get_gpu_memory_pids",
                return_value=[],
            ),
        ):
            result = kill_gpu_zombies(gpu_index=0)

        assert result.honest_verdict == "no_zombies_found"
        assert result.pids_killed == []
        assert result.pids_found == []
        assert isinstance(result, GPUZombieResult)

    def test_excludes_calling_process_pid(self) -> None:
        """Calling process PID is never sent SIGKILL.

        SCENARIO-INFRA-065: os.getpid() always in exclude set.
        Spec: REQ-INFRA-055, REQ-INFRA-056
        """
        my_pid = os.getpid()

        with (
            patch(
                "carnot.pipeline.gpu_zombie_killer._query_nvidia_smi",
                return_value="8000\n",
            ),
            patch(
                "carnot.pipeline.gpu_zombie_killer.get_gpu_memory_pids",
                return_value=[my_pid],  # only the calling process is on the GPU
            ),
            patch("os.kill") as mock_kill,
        ):
            result = kill_gpu_zombies(gpu_index=0)

        # SIGKILL must NOT have been sent to our own PID
        for call_args in mock_kill.call_args_list:
            sent_pid = call_args[0][0]
            assert sent_pid != my_pid, f"kill() called on calling process PID {my_pid}"

        # No zombies to kill after exclusion → no_zombies_found
        assert result.pids_killed == []
        assert result.honest_verdict == "no_zombies_found"

    def test_vram_freed_mb_computed_correctly(self) -> None:
        """vram_freed_mb = vram_before_mb - vram_after_mb.

        Spec: REQ-INFRA-055
        """
        target_pid = 99999  # fake zombie PID, not os.getpid()

        # _query_nvidia_smi is called twice: once for vram_before, once for vram_after.
        # We return 5000 MB before, 200 MB after.
        side_effects = ["5000\n", "200\n"]
        call_count = 0

        def mock_smi(args: list) -> str:
            nonlocal call_count
            # get_gpu_memory_pids also calls _query_nvidia_smi but we mock that separately
            val = side_effects[min(call_count, len(side_effects) - 1)]
            call_count += 1
            return val

        with (
            patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", side_effect=mock_smi),
            patch(
                "carnot.pipeline.gpu_zombie_killer.get_gpu_memory_pids",
                return_value=[target_pid],
            ),
            patch("os.kill"),
            patch("time.sleep"),  # skip the 2-second wait
        ):
            result = kill_gpu_zombies(gpu_index=0)

        assert result.vram_before_mb == 5000.0
        assert result.vram_after_mb == 200.0
        assert result.vram_freed_mb == pytest.approx(4800.0)

    def test_honest_verdict_zombies_killed_vram_freed(self) -> None:
        """pids_killed non-empty AND vram_freed > 100 → zombies_killed_vram_freed.

        Spec: REQ-INFRA-055
        """
        target_pid = 77777
        smi_calls = ["10000\n", "500\n"]
        smi_iter = iter(smi_calls)

        with (
            patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", side_effect=lambda _: next(smi_iter)),
            patch("carnot.pipeline.gpu_zombie_killer.get_gpu_memory_pids", return_value=[target_pid]),
            patch("os.kill"),
            patch("time.sleep"),
        ):
            result = kill_gpu_zombies(gpu_index=0)

        assert result.honest_verdict == "zombies_killed_vram_freed"
        assert target_pid in result.pids_killed

    def test_honest_verdict_zombies_killed_vram_unclear(self) -> None:
        """pids_killed non-empty but vram_freed_mb <= 100 → zombies_killed_vram_unclear.

        Spec: REQ-INFRA-055
        """
        target_pid = 66666
        # before=5000, after=4950 → freed=50 (< 100)
        smi_calls = ["5000\n", "4950\n"]
        smi_iter = iter(smi_calls)

        with (
            patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", side_effect=lambda _: next(smi_iter)),
            patch("carnot.pipeline.gpu_zombie_killer.get_gpu_memory_pids", return_value=[target_pid]),
            patch("os.kill"),
            patch("time.sleep"),
        ):
            result = kill_gpu_zombies(gpu_index=0)

        assert result.honest_verdict == "zombies_killed_vram_unclear"

    def test_honest_verdict_nvidia_smi_unavailable(self) -> None:
        """nvidia-smi not found → nvidia_smi_unavailable verdict, no kill attempted.

        Spec: REQ-INFRA-056
        """
        with patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", return_value=None):
            result = kill_gpu_zombies(gpu_index=0)

        assert result.honest_verdict == "nvidia_smi_unavailable"
        assert result.pids_killed == []
        assert not result.kill_attempted

    def test_kill_attempted_false_when_all_excluded(self) -> None:
        """When all found PIDs are in exclude list, kill_attempted=False.

        Spec: REQ-INFRA-056
        """
        my_pid = os.getpid()
        extra_excluded = 55555

        with (
            patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", return_value="100\n"),
            patch(
                "carnot.pipeline.gpu_zombie_killer.get_gpu_memory_pids",
                return_value=[my_pid, extra_excluded],
            ),
            patch("os.kill") as mock_kill,
        ):
            result = kill_gpu_zombies(gpu_index=0, exclude_pids=[extra_excluded])

        mock_kill.assert_not_called()
        assert not result.kill_attempted
        assert result.pids_killed == []

    def test_custom_exclude_pids_honored(self) -> None:
        """Caller-supplied exclude_pids are respected alongside os.getpid().

        Spec: REQ-INFRA-055, REQ-INFRA-056
        """
        protected_pid = 44444
        real_zombie = 33333
        smi_calls = ["8000\n", "1000\n"]
        smi_iter = iter(smi_calls)

        with (
            patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", side_effect=lambda _: next(smi_iter)),
            patch(
                "carnot.pipeline.gpu_zombie_killer.get_gpu_memory_pids",
                return_value=[protected_pid, real_zombie],
            ),
            patch("os.kill") as mock_kill,
            patch("time.sleep"),
        ):
            result = kill_gpu_zombies(gpu_index=0, exclude_pids=[protected_pid])

        # Only real_zombie should be killed
        killed_pids_in_calls = [c[0][0] for c in mock_kill.call_args_list]
        assert protected_pid not in killed_pids_in_calls
        assert real_zombie in killed_pids_in_calls
        assert real_zombie in result.pids_killed


# ---------------------------------------------------------------------------
# Tests for GPUZombieResult dataclass
# ---------------------------------------------------------------------------


class TestGPUZombieResult:
    """Unit tests for the GPUZombieResult dataclass.

    Spec: REQ-INFRA-055
    """

    def test_default_construction(self) -> None:
        """GPUZombieResult can be constructed with only gpu_index.

        Spec: REQ-INFRA-055
        """
        r = GPUZombieResult(gpu_index=0)
        assert r.gpu_index == 0
        assert r.pids_found == []
        assert r.pids_killed == []
        assert r.vram_before_mb == 0.0
        assert r.vram_after_mb == 0.0
        assert r.vram_freed_mb == 0.0
        assert r.kill_attempted is False
        assert r.honest_verdict == "no_zombies_found"

    def test_vram_freed_mb_field_is_float(self) -> None:
        """vram_freed_mb is a float, not an int.

        Spec: REQ-INFRA-055
        """
        r = GPUZombieResult(gpu_index=1, vram_before_mb=5000.0, vram_after_mb=200.0, vram_freed_mb=4800.0)
        assert isinstance(r.vram_freed_mb, float)
        assert r.vram_freed_mb == pytest.approx(4800.0)


# ---------------------------------------------------------------------------
# Tests for internal helpers (_query_nvidia_smi, _parse_float_from_smi_output,
# _get_vram_used_mb) — needed to cover exception paths not reached by mocking.
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    """Tests for internal helper functions.

    Spec: REQ-INFRA-055, REQ-INFRA-056
    """

    def test_query_nvidia_smi_returns_stdout_on_success(self) -> None:
        """_query_nvidia_smi returns subprocess stdout on success.

        Spec: REQ-INFRA-055
        """
        fake_result = MagicMock()
        fake_result.stdout = "1234\n"
        with patch("subprocess.run", return_value=fake_result):
            out = _query_nvidia_smi(["--query-gpu=memory.used", "--format=csv"])
        assert out == "1234\n"

    def test_query_nvidia_smi_returns_none_on_file_not_found(self) -> None:
        """FileNotFoundError (nvidia-smi not installed) → returns None.

        Spec: REQ-INFRA-056
        """
        with patch("subprocess.run", side_effect=FileNotFoundError("no nvidia-smi")):
            out = _query_nvidia_smi(["--query-gpu=memory.used"])
        assert out is None

    def test_query_nvidia_smi_returns_none_on_generic_exception(self) -> None:
        """Generic subprocess exception → returns None, no propagation.

        Spec: REQ-INFRA-056
        """
        with patch("subprocess.run", side_effect=RuntimeError("broken")):
            out = _query_nvidia_smi(["--query-gpu=memory.used"])
        assert out is None

    def test_parse_float_from_smi_output_valid(self) -> None:
        """Valid numeric string → correct float returned.

        Spec: REQ-INFRA-055
        """
        assert _parse_float_from_smi_output("8192\n") == pytest.approx(8192.0)

    def test_parse_float_from_smi_output_invalid(self) -> None:
        """Non-numeric string → 0.0 returned without exception.

        Spec: REQ-INFRA-056
        """
        assert _parse_float_from_smi_output("N/A\n") == 0.0

    def test_parse_float_from_smi_output_empty(self) -> None:
        """Empty string → 0.0 returned without exception.

        Spec: REQ-INFRA-056
        """
        assert _parse_float_from_smi_output("") == 0.0

    def test_get_vram_used_mb_returns_zero_when_smi_unavailable(self) -> None:
        """nvidia-smi unavailable → _get_vram_used_mb returns 0.0.

        Spec: REQ-INFRA-056
        """
        with patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", return_value=None):
            result = _get_vram_used_mb(0)
        assert result == 0.0

    def test_get_vram_used_mb_returns_parsed_value(self) -> None:
        """Valid nvidia-smi output → correct float returned.

        Spec: REQ-INFRA-055
        """
        with patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", return_value="4096\n"):
            result = _get_vram_used_mb(0)
        assert result == pytest.approx(4096.0)


class TestKillGpuZombiesEdgeCases:
    """Additional edge case tests for kill_gpu_zombies().

    Spec: REQ-INFRA-055, REQ-INFRA-056
    """

    def test_oserror_during_kill_is_handled_gracefully(self) -> None:
        """OSError during os.kill() (process already dead) → not in pids_killed.

        Spec: REQ-INFRA-055
        """
        target_pid = 22222
        smi_calls = ["8000\n", "8000\n"]
        smi_iter = iter(smi_calls)

        def raise_oserror(pid: int, sig: int) -> None:
            raise OSError("no such process")

        with (
            patch("carnot.pipeline.gpu_zombie_killer._query_nvidia_smi", side_effect=lambda _: next(smi_iter)),
            patch("carnot.pipeline.gpu_zombie_killer.get_gpu_memory_pids", return_value=[target_pid]),
            patch("os.kill", side_effect=raise_oserror),
            patch("time.sleep"),
        ):
            result = kill_gpu_zombies(gpu_index=0)

        # Kill was attempted but OSError means it wasn't recorded in pids_killed
        assert result.kill_attempted is True
        assert result.pids_killed == []
        # All kills failed → no_zombies_found verdict
        assert result.honest_verdict == "no_zombies_found"

    def test_empty_line_in_smi_output_is_skipped(self) -> None:
        """Empty lines in nvidia-smi output are silently skipped.

        Spec: REQ-INFRA-056
        """
        with patch(
            "carnot.pipeline.gpu_zombie_killer._query_nvidia_smi",
            return_value="\n\n555\n\n666\n",
        ):
            pids = get_gpu_memory_pids(0)
        assert pids == [555, 666]
