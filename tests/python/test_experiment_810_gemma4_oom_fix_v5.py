"""Tests for Experiment 810: Gemma4 OOM Fix v5 — nvidia-smi Verification Loop.

Spec: REQ-LOADER-014, SCENARIO-LOADER-014, SCENARIO-LOADER-015

All tests are unit tests — no GPU, no model download, no live inference.
Every code path in vram_loop_eviction.py and the experiment script is covered
via mocking of subprocess (nvidia-smi), os.kill, and time.sleep.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.vram_loop_eviction import (  # noqa: E402
    VRAMLoopEvictionResult,
    evict_vram_with_loop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zombie_result(pids_killed: list[int] | None = None) -> MagicMock:
    """Build a mock GPUZombieResult for patching kill_gpu_zombies."""
    r = MagicMock()
    r.pids_killed = pids_killed or []
    r.honest_verdict = "no_zombies_found"
    return r


def _smi_probe_ok(vram_value: float = 100.0) -> str:
    """Return a valid nvidia-smi --query-gpu=memory.used line."""
    return f"{vram_value}\n"


def _smi_apps_line(pid: int, mem_mb: float) -> str:
    """Return a single --query-compute-apps=pid,used_memory CSV line."""
    return f"{pid}, {mem_mb} MiB"


# ---------------------------------------------------------------------------
# VRAMLoopEvictionResult dataclass — REQ-LOADER-014
# ---------------------------------------------------------------------------


class TestVRAMLoopEvictionResult:
    """REQ-LOADER-014: VRAMLoopEvictionResult must have all required fields."""

    def test_defaults(self) -> None:
        """REQ-LOADER-014: dataclass initialises with safe defaults."""
        r = VRAMLoopEvictionResult(gpu_index=1)
        assert r.gpu_index == 1
        assert r.n_retries_attempted == 0
        assert r.vram_mb_per_retry == []
        assert r.final_vram_mb == 0.0
        assert r.vram_cleared is False
        assert r.abort_reason is None
        assert r.honest_verdict == "max_retries_exceeded"

    def test_field_types(self) -> None:
        """REQ-LOADER-014: field types match the spec."""
        r = VRAMLoopEvictionResult(
            gpu_index=1,
            n_retries_attempted=2,
            vram_mb_per_retry=[800.0, 300.0],
            final_vram_mb=300.0,
            vram_cleared=True,
            abort_reason=None,
            honest_verdict="vram_cleared",
        )
        assert isinstance(r.gpu_index, int)
        assert isinstance(r.n_retries_attempted, int)
        assert isinstance(r.vram_mb_per_retry, list)
        assert isinstance(r.final_vram_mb, float)
        assert isinstance(r.vram_cleared, bool)
        assert r.abort_reason is None
        assert isinstance(r.honest_verdict, str)


# ---------------------------------------------------------------------------
# evict_vram_with_loop — REQ-LOADER-014, SCENARIO-LOADER-014, SCENARIO-LOADER-015
# ---------------------------------------------------------------------------


class TestEvictVramWithLoop:
    """REQ-LOADER-014: evict_vram_with_loop must implement the retry-loop protocol."""

    def test_nvidia_smi_unavailable_returns_safe_result(self) -> None:
        """REQ-LOADER-014: when nvidia-smi is missing, return nvidia_smi_unavailable verdict."""
        with (
            patch("carnot.pipeline.vram_loop_eviction._query_nvidia_smi", return_value=None),
            patch(
                "carnot.pipeline.vram_loop_eviction.kill_gpu_zombies",
                return_value=_zombie_result(),
            ),
        ):
            result = evict_vram_with_loop(gpu_index=1, max_retries=3)

        assert result.honest_verdict == "nvidia_smi_unavailable"
        assert result.vram_cleared is False
        assert result.n_retries_attempted == 0

    def test_max_retries_exceeded_when_vram_stays_high(self) -> None:
        """SCENARIO-LOADER-014: 3 retries x 10s; VRAM still 600 MB; abort with max_retries_exceeded.

        Spec: REQ-LOADER-014, SCENARIO-LOADER-014
        """
        # Probe (availability check) returns a value, then the loop runs 3 retries.
        # Each retry: _get_compute_apps_with_memory returns empty, sleep, _get_vram_used_mb returns 600.
        probe_response = "700\n"

        def _smi_side_effect(args: list[str]) -> str | None:
            # Distinguish the probe query (--query-gpu) from apps query (--query-compute-apps).
            joined = " ".join(args)
            if "--query-compute-apps" in joined:
                return ""  # no compute apps
            if "--query-gpu" in joined:
                return "600\n"  # always 600 MB — above threshold
            return None

        with (
            patch(
                "carnot.pipeline.vram_loop_eviction._query_nvidia_smi",
                side_effect=lambda args: probe_response if "--query-gpu" in " ".join(args) else "",
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction.kill_gpu_zombies",
                return_value=_zombie_result(),
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction._get_vram_used_mb",
                return_value=600.0,
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction._get_compute_apps_with_memory",
                return_value=[],
            ),
            patch("carnot.pipeline.vram_loop_eviction.time.sleep") as mock_sleep,
        ):
            result = evict_vram_with_loop(
                gpu_index=1, max_retries=3, retry_sleep_s=10.0, threshold_mb=500.0
            )

        assert result.vram_cleared is False
        assert result.abort_reason == "max_retries_exceeded"
        assert result.honest_verdict == "max_retries_exceeded"
        assert result.n_retries_attempted == 3
        assert len(result.vram_mb_per_retry) == 3
        assert all(v == 600.0 for v in result.vram_mb_per_retry)
        # Ensure sleep was called 3 times (once per retry).
        assert mock_sleep.call_count == 3
        mock_sleep.assert_called_with(10.0)

    def test_clears_on_retry_2_returns_early(self) -> None:
        """SCENARIO-LOADER-015: VRAM 700->800->300 MB; cleared on retry 2; loop exits early.

        Spec: REQ-LOADER-014, SCENARIO-LOADER-015
        """
        # vram_mb_per_retry sequence: retry 1 = 800 (too high), retry 2 = 300 (cleared)
        vram_sequence = [800.0, 300.0]
        vram_iter = iter(vram_sequence)

        with (
            patch(
                "carnot.pipeline.vram_loop_eviction._query_nvidia_smi",
                return_value="700\n",
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction.kill_gpu_zombies",
                return_value=_zombie_result(),
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction._get_vram_used_mb",
                side_effect=vram_sequence,
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction._get_compute_apps_with_memory",
                return_value=[],
            ),
            patch("carnot.pipeline.vram_loop_eviction.time.sleep") as mock_sleep,
        ):
            result = evict_vram_with_loop(
                gpu_index=1, max_retries=3, retry_sleep_s=10.0, threshold_mb=500.0
            )

        assert result.vram_cleared is True
        assert result.honest_verdict == "vram_cleared"
        assert result.abort_reason is None
        assert result.n_retries_attempted == 2
        assert result.vram_mb_per_retry == [800.0, 300.0]
        assert result.final_vram_mb == 300.0
        # Only 2 sleeps because loop exits after retry 2.
        assert mock_sleep.call_count == 2

    def test_kills_processes_above_threshold(self) -> None:
        """REQ-LOADER-014-2: processes using > 100 MB are SIGKILLed in each retry.

        Spec: REQ-LOADER-014
        """
        large_pid = 55555
        small_pid = 66666

        with (
            patch(
                "carnot.pipeline.vram_loop_eviction._query_nvidia_smi",
                return_value="15000\n",
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction.kill_gpu_zombies",
                return_value=_zombie_result(),
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction._get_vram_used_mb",
                return_value=200.0,  # clears on first retry
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction._get_compute_apps_with_memory",
                return_value=[(large_pid, 14000.0), (small_pid, 50.0)],
            ),
            patch("carnot.pipeline.vram_loop_eviction.os.kill") as mock_kill,
            patch("carnot.pipeline.vram_loop_eviction.os.getpid", return_value=1),
            patch("carnot.pipeline.vram_loop_eviction.time.sleep"),
        ):
            result = evict_vram_with_loop(
                gpu_index=1, max_retries=3, retry_sleep_s=10.0, threshold_mb=500.0
            )

        # large_pid (14 GB) must be killed; small_pid (50 MB) must not be.
        mock_kill.assert_called_once_with(large_pid, signal.SIGKILL)
        assert result.vram_cleared is True

    def test_does_not_kill_own_pid(self) -> None:
        """REQ-LOADER-014: evict_vram_with_loop never kills its own PID.

        Spec: REQ-LOADER-014
        """
        my_pid = os.getpid()

        with (
            patch(
                "carnot.pipeline.vram_loop_eviction._query_nvidia_smi",
                return_value="15000\n",
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction.kill_gpu_zombies",
                return_value=_zombie_result(),
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction._get_vram_used_mb",
                return_value=200.0,
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction._get_compute_apps_with_memory",
                return_value=[(my_pid, 15000.0)],
            ),
            patch("carnot.pipeline.vram_loop_eviction.os.kill") as mock_kill,
            patch("carnot.pipeline.vram_loop_eviction.time.sleep"),
        ):
            evict_vram_with_loop(gpu_index=1, max_retries=1)

        # The calling process must never be SIGKILL'd.
        for c in mock_kill.call_args_list:
            assert c.args[0] != my_pid

    def test_clears_on_first_retry(self) -> None:
        """REQ-LOADER-014: loop exits after first retry if VRAM is below threshold.

        Spec: REQ-LOADER-014
        """
        with (
            patch(
                "carnot.pipeline.vram_loop_eviction._query_nvidia_smi",
                return_value="600\n",
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction.kill_gpu_zombies",
                return_value=_zombie_result(),
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction._get_vram_used_mb",
                return_value=100.0,  # below 500 MB on first read
            ),
            patch(
                "carnot.pipeline.vram_loop_eviction._get_compute_apps_with_memory",
                return_value=[],
            ),
            patch("carnot.pipeline.vram_loop_eviction.time.sleep") as mock_sleep,
        ):
            result = evict_vram_with_loop(gpu_index=1, max_retries=3, threshold_mb=500.0)

        assert result.vram_cleared is True
        assert result.n_retries_attempted == 1
        assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# Experiment 810 main() — integration-level tests
# ---------------------------------------------------------------------------


class TestExperiment810Main:
    """Integration tests for experiment_810 main() using mocked GPU/model deps."""

    def _run_main(
        self,
        tmp_path: Path,
        eviction_result: VRAMLoopEvictionResult,
        load_result: dict,
        generate_responses: list[str] | None = None,
        force_live: str = "1",
    ) -> dict:
        """Run experiment 810 main() with mocked dependencies; return parsed artifact."""
        import scripts.experiment_810_gemma4_oom_fix_v5 as exp810  # noqa: PLC0415

        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        fake_questions = [
            {"question": f"What is {i} + {i}?", "answer": str(i * 2)} for i in range(20)
        ]

        generate_responses = generate_responses or ["The answer is 42."] * 20

        mock_loader = MagicMock()
        mock_loader.generate.side_effect = generate_responses
        load_result_with_loader = {**load_result, "loader": mock_loader}

        mock_ckpt = MagicMock()

        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": force_live}),
            patch.object(exp810, "_REPO_ROOT", tmp_path),
            patch("scripts.experiment_810_gemma4_oom_fix_v5.apply_env_autofix"),
            patch(
                "scripts.experiment_810_gemma4_oom_fix_v5.evict_vram_with_loop",
                return_value=eviction_result,
            ),
            patch(
                "scripts.experiment_810_gemma4_oom_fix_v5.AtomicResultWriter",
                return_value=mock_ckpt,
            ),
            patch(
                "carnot.pipeline.gemma_isolation.load_gemma4_on_gpu1",
                return_value=load_result_with_loader,
            ),
            patch(
                "scripts.experiment_810_gemma4_oom_fix_v5._load_gsm8k_questions",
                return_value=fake_questions,
            ),
            patch(
                "scripts.experiment_810_gemma4_oom_fix_v5.GemmaTransformersLoader.is_valid_output",
                side_effect=lambda t: bool(t and t.strip()),
            ),
            patch(
                "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__enter__",
                return_value=None,
            ),
            patch(
                "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__exit__",
                return_value=False,
            ),
        ):
            with (
                patch.object(exp810.ExperimentTemplate, "assert_deliverable_written"),
                patch.object(exp810.ExperimentTemplate, "check_exclusion_manifest"),
                patch.object(exp810.ExperimentTemplate, "checkpoint_save"),
            ):
                # Also patch the load_gemma4_on_gpu1 import inside main()
                with patch(
                    "carnot.pipeline.gemma_isolation.load_gemma4_on_gpu1",
                    return_value=load_result_with_loader,
                ):
                    # We need to patch the import that happens inside main()
                    import unittest.mock as um
                    import carnot.pipeline.gemma_isolation as gi_mod

                    with um.patch.object(
                        gi_mod,
                        "load_gemma4_on_gpu1",
                        return_value=load_result_with_loader,
                    ):
                        exp810.main()

        artifact_path = results_dir / "experiment_810_gemma4_oom_fix_v5.json"
        if artifact_path.exists():
            return json.loads(artifact_path.read_text())
        return {}

    def test_blocked_no_live_gpu(self, tmp_path: Path) -> None:
        """SCENARIO-LOADER-014: blocked_no_live_gpu when CARNOT_FORCE_LIVE not set.

        Spec: REQ-LOADER-014
        """
        import scripts.experiment_810_gemma4_oom_fix_v5 as exp810  # noqa: PLC0415

        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}),
            patch.object(exp810, "_REPO_ROOT", tmp_path),
            patch("scripts.experiment_810_gemma4_oom_fix_v5.apply_env_autofix"),
            patch(
                "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__enter__",
                return_value=None,
            ),
            patch(
                "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__exit__",
                return_value=False,
            ),
            patch("scripts.experiment_810_gemma4_oom_fix_v5.AtomicResultWriter"),
        ):
            with (
                patch.object(exp810.ExperimentTemplate, "assert_deliverable_written"),
                patch.object(exp810.ExperimentTemplate, "check_exclusion_manifest"),
            ):
                exp810.main()

        artifact_path = results_dir / "experiment_810_gemma4_oom_fix_v5.json"
        assert artifact_path.exists(), "blocked artifact must be written"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["honest_verdict"] == "blocked_no_live_gpu"
        assert artifact["retro_028_closed"] is False
        assert artifact["step1_vram_cleared"] is False

    def test_blocked_vram_stuck_no_model_load(self, tmp_path: Path) -> None:
        """SCENARIO-LOADER-014: when eviction loop fails, abort without model load.

        Spec: REQ-LOADER-014, SCENARIO-LOADER-014
        """
        import scripts.experiment_810_gemma4_oom_fix_v5 as exp810  # noqa: PLC0415

        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        stuck_eviction = VRAMLoopEvictionResult(
            gpu_index=1,
            n_retries_attempted=3,
            vram_mb_per_retry=[700.0, 650.0, 600.0],
            final_vram_mb=600.0,
            vram_cleared=False,
            abort_reason="max_retries_exceeded",
            honest_verdict="max_retries_exceeded",
        )

        mock_load = MagicMock()
        mock_ckpt = MagicMock()

        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch.object(exp810, "_REPO_ROOT", tmp_path),
            patch("scripts.experiment_810_gemma4_oom_fix_v5.apply_env_autofix"),
            patch(
                "scripts.experiment_810_gemma4_oom_fix_v5.evict_vram_with_loop",
                return_value=stuck_eviction,
            ),
            patch(
                "scripts.experiment_810_gemma4_oom_fix_v5.AtomicResultWriter",
                return_value=mock_ckpt,
            ),
            patch(
                "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__enter__",
                return_value=None,
            ),
            patch(
                "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__exit__",
                return_value=False,
            ),
        ):
            with (
                patch.object(exp810.ExperimentTemplate, "assert_deliverable_written"),
                patch.object(exp810.ExperimentTemplate, "check_exclusion_manifest"),
            ):
                exp810.main()

        artifact_path = results_dir / "experiment_810_gemma4_oom_fix_v5.json"
        assert artifact_path.exists(), "blocked_vram_stuck artifact must be written"
        artifact = json.loads(artifact_path.read_text())

        # Core assertions per SCENARIO-LOADER-014
        assert artifact["honest_verdict"] == "blocked_vram_stuck"
        assert artifact["step1_vram_cleared"] is False
        assert artifact["step2_model_loaded"] is False
        assert artifact["retro_028_closed"] is False
        assert artifact["step1_vram_mb_per_retry"] == [700.0, 650.0, 600.0]
        # Critically: model load must NOT have been attempted
        mock_load.assert_not_called()

    def test_required_artifact_fields_present(self, tmp_path: Path) -> None:
        """REQ-LOADER-014: artifact must contain all required schema fields.

        Spec: REQ-LOADER-014
        """
        import scripts.experiment_810_gemma4_oom_fix_v5 as exp810  # noqa: PLC0415

        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        cleared_eviction = VRAMLoopEvictionResult(
            gpu_index=1,
            n_retries_attempted=2,
            vram_mb_per_retry=[800.0, 300.0],
            final_vram_mb=300.0,
            vram_cleared=True,
            abort_reason=None,
            honest_verdict="vram_cleared",
        )

        mock_loader = MagicMock()
        mock_loader.generate.return_value = "The answer is 42."
        load_ok = {
            "loaded": True,
            "device": "cuda:1",
            "reason": None,
            "vram_before_mb": 16000.0,
            "vram_after_mb": 200.0,
            "vram_clear": True,
            "pids_killed": [12345],
            "pkill_attempts": 1,
            "loader": mock_loader,
        }

        fake_questions = [
            {"question": f"What is {i}+{i}?", "answer": str(i * 2)} for i in range(20)
        ]

        mock_ckpt = MagicMock()

        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch.object(exp810, "_REPO_ROOT", tmp_path),
            patch("scripts.experiment_810_gemma4_oom_fix_v5.apply_env_autofix"),
            patch(
                "scripts.experiment_810_gemma4_oom_fix_v5.evict_vram_with_loop",
                return_value=cleared_eviction,
            ),
            patch(
                "scripts.experiment_810_gemma4_oom_fix_v5.AtomicResultWriter",
                return_value=mock_ckpt,
            ),
            patch(
                "scripts.experiment_810_gemma4_oom_fix_v5._load_gsm8k_questions",
                return_value=fake_questions,
            ),
            patch(
                "scripts.experiment_810_gemma4_oom_fix_v5.GemmaTransformersLoader.is_valid_output",
                return_value=True,
            ),
            patch(
                "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__enter__",
                return_value=None,
            ),
            patch(
                "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__exit__",
                return_value=False,
            ),
        ):
            with (
                patch.object(exp810.ExperimentTemplate, "assert_deliverable_written"),
                patch.object(exp810.ExperimentTemplate, "check_exclusion_manifest"),
                patch.object(exp810.ExperimentTemplate, "checkpoint_save"),
            ):
                import carnot.pipeline.gemma_isolation as gi_mod
                import unittest.mock as um

                with um.patch.object(gi_mod, "load_gemma4_on_gpu1", return_value=load_ok):
                    exp810.main()

        artifact_path = results_dir / "experiment_810_gemma4_oom_fix_v5.json"
        assert artifact_path.exists()
        artifact = json.loads(artifact_path.read_text())

        required_fields = [
            "step1_vram_mb_per_retry",
            "step1_vram_cleared",
            "step2_model_loaded",
            "step3_n_valid_responses",
            "retro_028_closed",
            "honest_verdict",
        ]
        for field_name in required_fields:
            assert field_name in artifact, f"Missing required field: {field_name}"

        # 20 valid responses → retro_028_closed
        assert artifact["step1_vram_cleared"] is True
        assert artifact["step2_model_loaded"] is True
        assert artifact["step3_n_valid_responses"] == 20
        assert artifact["honest_verdict"] == "retro_028_closed"
        assert artifact["retro_028_closed"] is True
