"""Tests for Experiment 795: Gemma4 OOM Fix v4 — Four-Step VRAM Isolation.

Spec: REQ-LOADER-012, REQ-LOADER-013, SCENARIO-LOADER-012, SCENARIO-LOADER-013

All tests are unit tests — no GPU, no model download, no live inference.
Every code path in gemma_isolation.py and the experiment script is covered
via mocking of subprocess (nvidia-smi), os.kill, and GemmaTransformersLoader.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import types
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.gemma_isolation import (  # noqa: E402
    VRAMEvictionResult,
    _VRAM_CLEAR_THRESHOLD_MB,
    evict_gpu_vram,
    load_gemma4_on_gpu1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_smi_output(value: float) -> str:
    """Return a nvidia-smi --query-gpu=memory.used formatted line."""
    return f"{value}\n"


def _make_zombie_result(pids_killed: list[int] | None = None) -> MagicMock:
    """Build a mock GPUZombieResult."""
    r = MagicMock()
    r.pids_killed = pids_killed if pids_killed is not None else []
    r.honest_verdict = "zombies_killed_vram_freed" if r.pids_killed else "no_zombies_found"
    r.vram_before_mb = 16000.0
    r.vram_after_mb = 400.0
    return r


# ---------------------------------------------------------------------------
# VRAMEvictionResult dataclass — REQ-LOADER-012
# ---------------------------------------------------------------------------


class TestVRAMEvictionResult:
    """REQ-LOADER-012: VRAMEvictionResult must have all required typed fields."""

    def test_field_names(self) -> None:
        """SCENARIO-LOADER-012: dataclass has all required fields with correct defaults."""
        result = VRAMEvictionResult(gpu_index=1)
        assert result.gpu_index == 1
        assert result.vram_before_mb == 0.0
        assert result.pids_killed == []
        assert result.pkill_attempts == 0
        assert result.vram_after_mb == 0.0
        assert result.vram_clear is False
        assert result.honest_verdict == "vram_not_cleared"

    def test_field_types(self) -> None:
        """REQ-LOADER-012: VRAMEvictionResult fields are typed correctly."""
        result = VRAMEvictionResult(
            gpu_index=1,
            vram_before_mb=15000.0,
            pids_killed=[111, 222],
            pkill_attempts=2,
            vram_after_mb=300.0,
            vram_clear=True,
            honest_verdict="vram_cleared",
        )
        assert isinstance(result.gpu_index, int)
        assert isinstance(result.vram_before_mb, float)
        assert isinstance(result.pids_killed, list)
        assert isinstance(result.pkill_attempts, int)
        assert isinstance(result.vram_after_mb, float)
        assert isinstance(result.vram_clear, bool)
        assert isinstance(result.honest_verdict, str)


# ---------------------------------------------------------------------------
# evict_gpu_vram — REQ-LOADER-012, SCENARIO-LOADER-012
# ---------------------------------------------------------------------------


class TestEvictGpuVram:
    """REQ-LOADER-012: evict_gpu_vram must implement the four-step protocol."""

    def test_nvidia_smi_unavailable_returns_correct_verdict(self) -> None:
        """REQ-LOADER-012: when nvidia-smi is missing, return nvidia_smi_unavailable verdict."""
        with patch(
            "carnot.pipeline.gemma_isolation._query_nvidia_smi", return_value=None
        ):
            result = evict_gpu_vram(gpu_index=1)

        assert result.honest_verdict == "nvidia_smi_unavailable"
        assert result.vram_clear is False
        assert result.vram_before_mb == 0.0

    def test_clears_to_below_500mb(self) -> None:
        """SCENARIO-LOADER-012: when eviction succeeds, vram_clear=True and verdict=vram_cleared."""
        zombie_result = _make_zombie_result(pids_killed=[12345])

        # nvidia-smi calls: (1) availability check, (2) vram_before, (3) vram_after
        # also _get_vram_used_mb is called, and _get_compute_pids for the pkill sweep
        smi_side_effects = [
            "400\n",   # availability check (memory.used)
            "400\n",   # _get_vram_used_mb (vram_before)
            "",        # _get_compute_pids (no residual pids after primary kill)
            "200\n",   # _get_vram_used_mb (vram_after)
        ]

        with patch(
            "carnot.pipeline.gemma_isolation._query_nvidia_smi",
            side_effect=smi_side_effects,
        ), patch(
            "carnot.pipeline.gemma_isolation.kill_gpu_zombies",
            return_value=zombie_result,
        ), patch(
            "carnot.pipeline.gemma_isolation.time.sleep"
        ):
            result = evict_gpu_vram(gpu_index=1)

        assert result.vram_clear is True
        assert result.vram_after_mb == 200.0
        assert result.honest_verdict == "vram_cleared"
        assert 12345 in result.pids_killed

    def test_fails_when_vram_above_threshold(self) -> None:
        """REQ-LOADER-012: when vram_after >= 500 MB, vram_clear=False."""
        zombie_result = _make_zombie_result(pids_killed=[])

        smi_side_effects = [
            "16000\n",  # availability check
            "16000\n",  # vram_before
            "",         # compute pids (none)
            "600\n",    # vram_after (still above 500 MB)
        ]

        with patch(
            "carnot.pipeline.gemma_isolation._query_nvidia_smi",
            side_effect=smi_side_effects,
        ), patch(
            "carnot.pipeline.gemma_isolation.kill_gpu_zombies",
            return_value=zombie_result,
        ), patch(
            "carnot.pipeline.gemma_isolation.time.sleep"
        ):
            result = evict_gpu_vram(gpu_index=1)

        assert result.vram_clear is False
        assert result.vram_after_mb == 600.0
        assert result.honest_verdict == "vram_not_cleared"

    def test_pkill_sweep_kills_residual_pids(self) -> None:
        """SCENARIO-LOADER-012: pkill sweep SIGKILLs residual PIDs not caught by primary kill."""
        zombie_result = _make_zombie_result(pids_killed=[])

        residual_pid = 99999
        # smi calls: availability, vram_before, compute_pids (residual pid found), vram_after
        smi_side_effects = [
            "15000\n",           # availability check
            "15000\n",           # vram_before
            f"{residual_pid}\n", # _get_compute_pids — one residual pid
            "100\n",             # vram_after — now clear
        ]

        with patch(
            "carnot.pipeline.gemma_isolation._query_nvidia_smi",
            side_effect=smi_side_effects,
        ), patch(
            "carnot.pipeline.gemma_isolation.kill_gpu_zombies",
            return_value=zombie_result,
        ), patch(
            "carnot.pipeline.gemma_isolation.os.kill"
        ) as mock_kill, patch(
            "carnot.pipeline.gemma_isolation.os.getpid", return_value=1
        ), patch(
            "carnot.pipeline.gemma_isolation.time.sleep"
        ):
            result = evict_gpu_vram(gpu_index=1)

        # The residual pid must have received SIGKILL
        mock_kill.assert_called_once_with(residual_pid, signal.SIGKILL)
        assert result.pkill_attempts == 1
        assert residual_pid in result.pids_killed

    def test_does_not_kill_own_pid(self) -> None:
        """REQ-LOADER-012: evict_gpu_vram never kills its own PID."""
        my_pid = os.getpid()
        zombie_result = _make_zombie_result(pids_killed=[])

        smi_side_effects = [
            "15000\n",
            "15000\n",
            f"{my_pid}\n",  # nvidia-smi reports our own PID
            "50\n",
        ]

        with patch(
            "carnot.pipeline.gemma_isolation._query_nvidia_smi",
            side_effect=smi_side_effects,
        ), patch(
            "carnot.pipeline.gemma_isolation.kill_gpu_zombies",
            return_value=zombie_result,
        ), patch(
            "carnot.pipeline.gemma_isolation.os.kill"
        ) as mock_kill, patch(
            "carnot.pipeline.gemma_isolation.time.sleep"
        ):
            result = evict_gpu_vram(gpu_index=1)

        # Our PID must never be killed
        for c in mock_kill.call_args_list:
            assert c.args[0] != my_pid


# ---------------------------------------------------------------------------
# load_gemma4_on_gpu1 — REQ-LOADER-013, SCENARIO-LOADER-013
# ---------------------------------------------------------------------------


class TestLoadGemma4OnGpu1:
    """REQ-LOADER-013: load_gemma4_on_gpu1 skips model load when vram_not_cleared."""

    def test_skips_load_when_vram_not_cleared(self) -> None:
        """SCENARIO-LOADER-013: when evict_gpu_vram returns vram_clear=False, loaded=False."""
        failed_eviction = VRAMEvictionResult(
            gpu_index=1,
            vram_before_mb=20000.0,
            pids_killed=[],
            pkill_attempts=0,
            vram_after_mb=15000.0,
            vram_clear=False,
            honest_verdict="vram_not_cleared",
        )

        with patch(
            "carnot.pipeline.gemma_isolation.evict_gpu_vram",
            return_value=failed_eviction,
        ):
            result = load_gemma4_on_gpu1("google/gemma-4-E4B-it")

        assert result["loaded"] is False
        assert result["reason"] == "vram_not_cleared"
        assert result["device"] is None
        assert result["vram_clear"] is False

    def test_loads_on_cuda_1_when_vram_clear(self) -> None:
        """SCENARIO-LOADER-013: when eviction succeeds, model loads on cuda:1."""
        cleared_eviction = VRAMEvictionResult(
            gpu_index=1,
            vram_before_mb=16000.0,
            pids_killed=[12345],
            pkill_attempts=1,
            vram_after_mb=200.0,
            vram_clear=True,
            honest_verdict="vram_cleared",
        )

        mock_loader = MagicMock()
        mock_loader_cls = MagicMock(return_value=mock_loader)

        with patch(
            "carnot.pipeline.gemma_isolation.evict_gpu_vram",
            return_value=cleared_eviction,
        ), patch(
            "carnot.pipeline.gemma_isolation.GemmaTransformersLoader",
            mock_loader_cls,
        ):
            result = load_gemma4_on_gpu1("google/gemma-4-E4B-it")

        assert result["loaded"] is True
        assert result["device"] == "cuda:1"
        assert result["reason"] is None
        # Confirm device_map pointed at cuda:1
        mock_loader_cls.assert_called_once_with(
            model_id="google/gemma-4-E4B-it",
            device={"": "cuda:1"},
        )
        mock_loader.load.assert_called_once()

    def test_returns_loader_object_on_success(self) -> None:
        """REQ-LOADER-013: the returned dict includes the loader instance for inference."""
        cleared_eviction = VRAMEvictionResult(
            gpu_index=1,
            vram_before_mb=500.0,
            pids_killed=[],
            pkill_attempts=0,
            vram_after_mb=50.0,
            vram_clear=True,
            honest_verdict="vram_cleared",
        )

        mock_loader = MagicMock()

        with patch(
            "carnot.pipeline.gemma_isolation.evict_gpu_vram",
            return_value=cleared_eviction,
        ), patch(
            "carnot.pipeline.gemma_isolation.GemmaTransformersLoader",
            return_value=mock_loader,
        ):
            result = load_gemma4_on_gpu1("google/gemma-4-E4B-it")

        assert "loader" in result
        assert result["loader"] is mock_loader

    def test_returns_error_when_loader_raises(self) -> None:
        """REQ-LOADER-013: if GemmaTransformersLoader.load() raises, loaded=False."""
        cleared_eviction = VRAMEvictionResult(
            gpu_index=1,
            vram_before_mb=200.0,
            pids_killed=[],
            pkill_attempts=0,
            vram_after_mb=50.0,
            vram_clear=True,
            honest_verdict="vram_cleared",
        )

        mock_loader = MagicMock()
        mock_loader.load.side_effect = RuntimeError("CUDA OOM simulated")

        with patch(
            "carnot.pipeline.gemma_isolation.evict_gpu_vram",
            return_value=cleared_eviction,
        ), patch(
            "carnot.pipeline.gemma_isolation.GemmaTransformersLoader",
            return_value=mock_loader,
        ):
            result = load_gemma4_on_gpu1("google/gemma-4-E4B-it")

        assert result["loaded"] is False
        assert "CUDA OOM" in result["reason"]


# ---------------------------------------------------------------------------
# Experiment 795 main() — integration path tests
# ---------------------------------------------------------------------------


class TestExperiment795Main:
    """Integration-level tests for the experiment main() using mocked live deps."""

    def _run_main(
        self,
        tmp_path: Path,
        env_overrides: dict,
        eviction_result: VRAMEvictionResult,
        load_result: dict,
        generate_responses: list[str] | None = None,
    ) -> dict:
        """Run experiment 795 main() with mocked GPU/model deps; return artifact dict."""
        import scripts.experiment_795_gemma4_oom_fix_v4 as exp795  # noqa: PLC0415

        deliverable = tmp_path / "experiment_795_gemma4_oom_fix_v4.json"
        zombie_mock = _make_zombie_result(pids_killed=[12345])

        # Mock question loader to avoid network
        fake_questions = [
            {"question": f"What is {i} + {i}?", "answer": str(i * 2)}
            for i in range(10)
        ]

        generate_responses = generate_responses or ["The answer is 42."] * 10

        def _fake_generate(prompt: str, max_new_tokens: int = 512) -> str:
            idx = len([r for r in generate_responses if r])
            return generate_responses[min(idx, len(generate_responses) - 1)]

        mock_loader = MagicMock()
        mock_loader.generate.side_effect = generate_responses
        load_result_with_loader = {**load_result, "loader": mock_loader}

        env = {"CARNOT_FORCE_LIVE": "1", **env_overrides}

        with patch.dict(os.environ, env), patch.object(
            exp795, "_REPO_ROOT", tmp_path
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.apply_env_autofix"
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.kill_gpu_zombies",
            return_value=zombie_mock,
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.evict_gpu_vram",
            return_value=eviction_result,
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.load_gemma4_on_gpu1",
            return_value=load_result_with_loader,
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4._load_gsm8k_questions",
            return_value=fake_questions,
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.GemmaTransformersLoader"
                ".is_valid_output",
            side_effect=lambda t: bool(t and t.strip()),
        ), patch(
            "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__enter__",
            return_value=None,
        ), patch(
            "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__exit__",
            return_value=False,
        ):
            # Patch deliverable path in template
            with patch.object(
                exp795.ExperimentTemplate, "assert_deliverable_written"
            ), patch.object(
                exp795.ExperimentTemplate, "check_exclusion_manifest"
            ), patch.object(
                exp795.ExperimentTemplate, "checkpoint_save"
            ):
                exp795.main()

        artifact_path = tmp_path / "results" / "experiment_795_gemma4_oom_fix_v4.json"
        if artifact_path.exists():
            return json.loads(artifact_path.read_text())
        # Fallback: check if written directly to tmp_path
        if deliverable.exists():
            return json.loads(deliverable.read_text())
        return {}

    def test_blocked_no_live_gpu(self, tmp_path: Path) -> None:
        """SCENARIO-LOADER-012: blocked_no_live_gpu when CARNOT_FORCE_LIVE not set."""
        import scripts.experiment_795_gemma4_oom_fix_v4 as exp795  # noqa: PLC0415

        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}), patch.object(
            exp795, "_REPO_ROOT", tmp_path
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.apply_env_autofix"
        ), patch(
            "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__enter__",
            return_value=None,
        ), patch(
            "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__exit__",
            return_value=False,
        ):
            with patch.object(
                exp795.ExperimentTemplate, "assert_deliverable_written"
            ), patch.object(
                exp795.ExperimentTemplate, "check_exclusion_manifest"
            ):
                exp795.main()

        artifact_path = results_dir / "experiment_795_gemma4_oom_fix_v4.json"
        assert artifact_path.exists(), "blocked artifact must be written"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["honest_verdict"] == "blocked_no_live_gpu"
        assert artifact["retro_028_closed"] is False

    def test_vram_not_cleared_writes_blocked_artifact(self, tmp_path: Path) -> None:
        """REQ-LOADER-012: when eviction fails, artifact has honest_verdict=vram_not_cleared."""
        import scripts.experiment_795_gemma4_oom_fix_v4 as exp795  # noqa: PLC0415

        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        failed_eviction = VRAMEvictionResult(
            gpu_index=1,
            vram_before_mb=20000.0,
            pids_killed=[],
            pkill_attempts=0,
            vram_after_mb=15000.0,
            vram_clear=False,
            honest_verdict="vram_not_cleared",
        )
        zombie_mock = _make_zombie_result(pids_killed=[])

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}), patch.object(
            exp795, "_REPO_ROOT", tmp_path
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.apply_env_autofix"
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.kill_gpu_zombies",
            return_value=zombie_mock,
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.evict_gpu_vram",
            return_value=failed_eviction,
        ), patch(
            "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__enter__",
            return_value=None,
        ), patch(
            "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__exit__",
            return_value=False,
        ):
            with patch.object(
                exp795.ExperimentTemplate, "assert_deliverable_written"
            ), patch.object(
                exp795.ExperimentTemplate, "check_exclusion_manifest"
            ):
                exp795.main()

        artifact_path = results_dir / "experiment_795_gemma4_oom_fix_v4.json"
        assert artifact_path.exists()
        artifact = json.loads(artifact_path.read_text())
        assert artifact["honest_verdict"] == "vram_not_cleared"
        assert artifact["step3_vram_clear"] is False
        assert artifact["retro_028_closed"] is False

    def test_required_artifact_fields_present(self, tmp_path: Path) -> None:
        """REQ-LOADER-012/013: artifact must contain all required schema fields."""
        import scripts.experiment_795_gemma4_oom_fix_v4 as exp795  # noqa: PLC0415

        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        cleared_eviction = VRAMEvictionResult(
            gpu_index=1,
            vram_before_mb=16000.0,
            pids_killed=[12345],
            pkill_attempts=1,
            vram_after_mb=200.0,
            vram_clear=True,
            honest_verdict="vram_cleared",
        )
        zombie_mock = _make_zombie_result(pids_killed=[12345])

        mock_loader = MagicMock()
        # 10 valid responses → retro_028_closed
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
            {"question": f"What is {i}+{i}?", "answer": str(i * 2)}
            for i in range(10)
        ]

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}), patch.object(
            exp795, "_REPO_ROOT", tmp_path
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.apply_env_autofix"
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.kill_gpu_zombies",
            return_value=zombie_mock,
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.evict_gpu_vram",
            return_value=cleared_eviction,
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.load_gemma4_on_gpu1",
            return_value=load_ok,
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4._load_gsm8k_questions",
            return_value=fake_questions,
        ), patch(
            "scripts.experiment_795_gemma4_oom_fix_v4.GemmaTransformersLoader"
                ".is_valid_output",
            return_value=True,
        ), patch(
            "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__enter__",
            return_value=None,
        ), patch(
            "carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog.__exit__",
            return_value=False,
        ):
            with patch.object(
                exp795.ExperimentTemplate, "assert_deliverable_written"
            ), patch.object(
                exp795.ExperimentTemplate, "check_exclusion_manifest"
            ), patch.object(
                exp795.ExperimentTemplate, "checkpoint_save"
            ):
                exp795.main()

        artifact_path = results_dir / "experiment_795_gemma4_oom_fix_v4.json"
        assert artifact_path.exists()
        artifact = json.loads(artifact_path.read_text())

        required_fields = [
            "step1_zombies_killed",
            "step2_vram_before_mb",
            "step2_pids_killed",
            "step3_vram_after_mb",
            "step3_vram_clear",
            "step4_model_loaded",
            "n_valid_responses",
            "honest_verdict",
            "retro_028_closed",
        ]
        for field in required_fields:
            assert field in artifact, f"Missing required field: {field}"

        assert artifact["honest_verdict"] == "retro_028_closed"
        assert artifact["retro_028_closed"] is True
        assert artifact["step3_vram_clear"] is True
        assert artifact["step4_model_loaded"] is True
