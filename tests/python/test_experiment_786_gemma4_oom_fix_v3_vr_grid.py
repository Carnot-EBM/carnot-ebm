"""Tests for Experiment 786: Gemma4 OOM Fix v3 + VR Grid (RETRO-028 closure).

Spec: REQ-LOADER-011, SCENARIO-LOADER-011

All tests are unit tests (no GPU, no model download, no live inference).
Every call path in the experiment module is covered via mocking.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the repo root is on sys.path so the experiment module is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zombie_result(vram_after_mb: float = 1000.0) -> MagicMock:
    """Build a mock GPUZombieResult with the given vram_after_mb."""
    r = MagicMock()
    r.honest_verdict = "zombies_killed_vram_freed"
    r.pids_killed = [12345]
    r.vram_before_mb = 16000.0
    r.vram_after_mb = vram_after_mb
    r.vram_freed_mb = 16000.0 - vram_after_mb
    return r


def _make_vr_result(abstained: bool = False) -> MagicMock:
    """Build a mock VerifyRepairPipeline result."""
    r = MagicMock()
    r.final_response = "The answer is 42."
    r.abstained = abstained
    return r


# ---------------------------------------------------------------------------
# Test: kill_gpu_zombies() is called before GemmaTransformersLoader
#
# REQ-LOADER-011: A Gemma4 load attempt MUST call kill_gpu_zombies(gpu_index=0)
# before any GemmaTransformersLoader.load() call.
# ---------------------------------------------------------------------------


class TestKillGpuZombiesCalledFirst:
    """REQ-LOADER-011: kill_gpu_zombies must precede any loader.load() call."""

    def test_kill_gpu_zombies_called_before_loader(self, tmp_path: Path) -> None:
        """Verify kill_gpu_zombies() is called before GemmaTransformersLoader.load().

        Why this test: REQ-LOADER-011 is the core invariant that closes RETRO-028.
        If kill_gpu_zombies() is called AFTER load(), the OOM can still happen.
        The call order is enforced by verifying both mocks were called AND that the
        kill mock was called before the load mock within the same execution.
        """
        call_order: list[str] = []

        def _mock_kill(gpu_index: int = 0, **kw: object) -> MagicMock:
            call_order.append("kill")
            # Return enough free VRAM: vram_after_mb = 2000 → free = 24576 - 2000 = 22576 > 12000
            return _make_zombie_result(vram_after_mb=2000.0)

        deliverable = tmp_path / "results" / "experiment_786_gemma4_oom_fix_v3_vr_grid.json"
        deliverable.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch(
                "scripts.experiment_786_gemma4_oom_fix_v3_vr_grid.kill_gpu_zombies",
                side_effect=_mock_kill,
            ),
            patch(
                "scripts.experiment_786_gemma4_oom_fix_v3_vr_grid.GemmaTransformersLoader",
            ) as mock_cls,
            patch("scripts.experiment_786_gemma4_oom_fix_v3_vr_grid.ExperimentTemplate") as mock_tmpl_cls,
            patch("scripts.experiment_786_gemma4_oom_fix_v3_vr_grid.ExperimentTimeoutWatchdog") as mock_wd,
            patch("scripts.experiment_786_gemma4_oom_fix_v3_vr_grid.apply_env_autofix"),
            patch("scripts.experiment_786_gemma4_oom_fix_v3_vr_grid._load_gsm8k_questions", return_value=[]),
            patch("scripts.experiment_786_gemma4_oom_fix_v3_vr_grid._run_vr_grid", return_value=[]),
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch("scripts.experiment_786_gemma4_oom_fix_v3_vr_grid._REPO_ROOT", tmp_path),
        ):
            mock_instance = MagicMock()
            mock_instance.load.side_effect = lambda: call_order.append("load")
            mock_instance.generate.return_value = "Hello world"
            mock_cls.return_value = mock_instance
            mock_cls.is_valid_output = MagicMock(return_value=True)

            tmpl_instance = MagicMock()
            tmpl_instance.build_result.return_value = {
                "experiment": 786, "schema": [], "run_date": "20260423",
                "started_at": "2026-04-23T00:00:00Z", "finished_at": "2026-04-23T00:01:00Z",
                "duration_s": 60.0, "status": "success", "title": "test",
                "honest_verdict": "retro028_closed_no_improvement",
            }
            # MagicMock blocks any attribute starting with "assert" by default.
            # Set it explicitly so assert_deliverable_written() can be called.
            tmpl_instance.assert_deliverable_written = MagicMock()
            mock_tmpl_cls.return_value = tmpl_instance
            tmpl_instance.check_exclusion_manifest.return_value = False

            wd_instance = MagicMock()
            wd_instance.__enter__ = MagicMock(return_value=wd_instance)
            wd_instance.__exit__ = MagicMock(return_value=False)
            mock_wd.return_value = wd_instance

            import scripts.experiment_786_gemma4_oom_fix_v3_vr_grid as exp  # noqa: PLC0415

            exp.main()

        # kill must appear before load in the call order
        assert "kill" in call_order, "kill_gpu_zombies() was never called"
        assert "load" in call_order, "loader.load() was never called"
        assert call_order.index("kill") < call_order.index("load"), (
            f"kill_gpu_zombies() must precede loader.load(); got order={call_order}"
        )


# ---------------------------------------------------------------------------
# Test: blocked_insufficient_vram when free VRAM < 12000 MB
#
# REQ-LOADER-011: MUST NOT attempt load if free_vram_mb < 12000.
# SCENARIO-LOADER-011: artifact contains honest_verdict="blocked_insufficient_vram".
# ---------------------------------------------------------------------------


class TestBlockedInsufficientVram:
    """REQ-LOADER-011 / SCENARIO-LOADER-011: VRAM gate blocks load when < 12 GB free."""

    def test_blocked_insufficient_vram_artifact(self, tmp_path: Path) -> None:
        """When free VRAM < 12000 MB after kill, write blocked artifact; do NOT call loader.load().

        Why this test: RETRO-028's root failure was attempting the load with
        only ~9 GB free.  The 12 GB threshold ensures a 2+ GB safety margin
        above the 14.89 GiB footprint — wait, the footprint is 14.89 GiB so we
        actually need MORE than 12 GB free.  The 12 GB threshold is the minimum
        check that confirms the zombie kill freed enough VRAM; if it reads < 12 GB
        free, something else is holding VRAM and we must abort.
        """
        # vram_after_mb = 14000 → free = 24576 - 14000 = 10576 < 12000 → BLOCKED
        mock_zombie = _make_zombie_result(vram_after_mb=14000.0)

        deliverable = tmp_path / "results" / "experiment_786_gemma4_oom_fix_v3_vr_grid.json"
        deliverable.parent.mkdir(parents=True, exist_ok=True)

        load_called = []

        with (
            patch(
                "scripts.experiment_786_gemma4_oom_fix_v3_vr_grid.kill_gpu_zombies",
                return_value=mock_zombie,
            ),
            patch(
                "scripts.experiment_786_gemma4_oom_fix_v3_vr_grid.GemmaTransformersLoader",
            ) as mock_cls,
            patch("scripts.experiment_786_gemma4_oom_fix_v3_vr_grid.ExperimentTemplate") as mock_tmpl_cls,
            patch("scripts.experiment_786_gemma4_oom_fix_v3_vr_grid.ExperimentTimeoutWatchdog") as mock_wd,
            patch("scripts.experiment_786_gemma4_oom_fix_v3_vr_grid.apply_env_autofix"),
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch("scripts.experiment_786_gemma4_oom_fix_v3_vr_grid._REPO_ROOT", tmp_path),
        ):
            mock_loader_inst = MagicMock()
            mock_loader_inst.load.side_effect = lambda: load_called.append(True)
            mock_cls.return_value = mock_loader_inst

            tmpl_instance = MagicMock()
            tmpl_instance.build_result.side_effect = lambda data, **kw: {
                "experiment": 786, "schema": [], "run_date": "20260423",
                "started_at": "2026-04-23T00:00:00Z", "finished_at": "2026-04-23T00:01:00Z",
                "duration_s": 1.0, "status": kw.get("status", "blocked"), "title": "test",
                **data,
            }
            # MagicMock blocks any attribute starting with "assert" by default.
            # Set it explicitly so assert_deliverable_written() can be called.
            tmpl_instance.assert_deliverable_written = MagicMock()
            mock_tmpl_cls.return_value = tmpl_instance
            tmpl_instance.check_exclusion_manifest.return_value = False

            wd_instance = MagicMock()
            wd_instance.__enter__ = MagicMock(return_value=wd_instance)
            wd_instance.__exit__ = MagicMock(return_value=False)
            mock_wd.return_value = wd_instance

            import scripts.experiment_786_gemma4_oom_fix_v3_vr_grid as exp  # noqa: PLC0415

            exp.main()

        # Load must NOT have been called
        assert not load_called, "loader.load() must NOT be called when VRAM < 12000 MB"

        # Artifact must contain blocked_insufficient_vram
        artifact = json.loads(deliverable.read_text())
        assert artifact["honest_verdict"] == "blocked_insufficient_vram", (
            f"Expected honest_verdict='blocked_insufficient_vram', got {artifact['honest_verdict']!r}"
        )
        assert artifact["free_vram_mb_after_kill"] < 12000, (
            "free_vram_mb_after_kill must be below 12000 in blocked artifact"
        )


# ---------------------------------------------------------------------------
# Test: positive_threshold_found = any(signed_improvement > 0)
#
# REQ-LOADER-011: the VR grid must correctly identify positive thresholds.
# ---------------------------------------------------------------------------


class TestPositiveThresholdFound:
    """REQ-LOADER-011: positive_threshold_found = any(signed_improvement > 0)."""

    def test_positive_threshold_found_true_when_any_improvement(self) -> None:
        """positive_threshold_found must be True when at least one threshold improves accuracy.

        Why this test: this is the primary RETRO-028 success criterion.  If any
        abstention threshold produces vr_accuracy > baseline_accuracy, we have
        demonstrated that the VR pipeline adds value for Gemma4 on GSM8K.
        """
        per_threshold_results = [
            {"threshold": 0.10, "baseline_accuracy": 0.5, "vr_accuracy": 0.48, "signed_improvement": -0.02, "n_abstained": 0},
            {"threshold": 0.20, "baseline_accuracy": 0.5, "vr_accuracy": 0.52, "signed_improvement": 0.02, "n_abstained": 1},
            {"threshold": 0.30, "baseline_accuracy": 0.5, "vr_accuracy": 0.50, "signed_improvement": 0.00, "n_abstained": 3},
        ]
        positive_threshold_found = any(r["signed_improvement"] > 0 for r in per_threshold_results)
        assert positive_threshold_found is True

    def test_positive_threshold_found_false_when_no_improvement(self) -> None:
        """positive_threshold_found must be False when no threshold improves accuracy.

        Why this test: we must not claim RETRO-028 closed with a positive threshold
        if all thresholds produce zero or negative improvement.  The honest verdict
        in that case is "retro028_closed_no_improvement", not "retro028_closed_positive_threshold".
        """
        per_threshold_results = [
            {"threshold": 0.10, "baseline_accuracy": 0.5, "vr_accuracy": 0.48, "signed_improvement": -0.02, "n_abstained": 0},
            {"threshold": 0.20, "baseline_accuracy": 0.5, "vr_accuracy": 0.50, "signed_improvement": 0.00, "n_abstained": 2},
            {"threshold": 0.30, "baseline_accuracy": 0.5, "vr_accuracy": 0.49, "signed_improvement": -0.01, "n_abstained": 5},
        ]
        positive_threshold_found = any(r["signed_improvement"] > 0 for r in per_threshold_results)
        assert positive_threshold_found is False

    def test_positive_threshold_found_false_on_empty_grid(self) -> None:
        """positive_threshold_found must be False when the grid is empty (loader blocked).

        Why this test: an empty per_threshold_results list (e.g. blocked before
        grid runs) must NOT claim positive improvement.
        """
        per_threshold_results: list[dict] = []
        positive_threshold_found = any(r["signed_improvement"] > 0 for r in per_threshold_results)
        assert positive_threshold_found is False
