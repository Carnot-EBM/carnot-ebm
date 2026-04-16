"""Tests for scripts/experiment_426_dual_gpu_fix.py — 100% coverage.

Coverage targets
----------------
- run_experiment():
  - Normal path: retro file present, CI GPU defaults, builds artifact
  - Retro file missing: proceeds without context, no exception
  - Retro file malformed JSON: logs warning, proceeds
  - Artifact contains all required fields (REQUIRED_RESULT_FIELDS)
  - honest_verdict / retro_025_status forwarded from build_gpu_fix_artifact
  - env_autofix block embedded in artifact
- main():
  - Calls run_experiment() inside ExperimentTimeoutWatchdog
  - Logs headline results

Spec: REQ-INFRA-025, REQ-INFRA-026,
      SCENARIO-INFRA-031, SCENARIO-INFRA-032, SCENARIO-INFRA-033 (Exp 426)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.dual_gpu_health import DualGPUHealthResult  # noqa: E402
from scripts.experiment_template import REQUIRED_RESULT_FIELDS  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CI_HEALTH = DualGPUHealthResult(
    gpu0_util_pct=0.0,
    gpu1_util_pct=0.0,
    gpu0_temp_c=0.0,
    gpu1_temp_c=0.0,
    gpu0_vram_mb=0.0,
    gpu1_vram_mb=0.0,
    gpu1_is_zombie=False,
    temperature_warning=False,
    recommended_batch_size_factor=1.0,
)

_ZOMBIE_HEALTH = DualGPUHealthResult(
    gpu0_util_pct=88.0,
    gpu1_util_pct=0.0,
    gpu0_temp_c=75.0,
    gpu1_temp_c=60.0,
    gpu0_vram_mb=15000.0,
    gpu1_vram_mb=1786.0,
    gpu1_is_zombie=True,
    temperature_warning=False,
    recommended_batch_size_factor=1.0,
)

_TEMP_WARN_HEALTH = DualGPUHealthResult(
    gpu0_util_pct=88.0,
    gpu1_util_pct=0.0,
    gpu0_temp_c=82.0,
    gpu1_temp_c=70.0,
    gpu0_vram_mb=15000.0,
    gpu1_vram_mb=100.0,
    gpu1_is_zombie=False,
    temperature_warning=True,
    recommended_batch_size_factor=0.75,
)

_RETRO_DATA = {
    "schema": "carnot.operational_retro.v5",
    "milestone": "2026.04.31",
    "gpu_utilization": {
        "gpu_1_vram_used_mb": 1786,
        "gpu_1_utilization_pct": 0,
    },
}


def _make_autofix_result(gpu_detected=False, auto_fix_applied=False):
    from carnot.pipeline.env_autofix import EnvironmentAutoFix
    return EnvironmentAutoFix(
        gpu_detected=gpu_detected,
        carnot_force_live_was_set=False,
        auto_fix_applied=auto_fix_applied,
        final_env_value=None,
    )


# ---------------------------------------------------------------------------
# run_experiment() tests
# ---------------------------------------------------------------------------


class TestRunExperiment:
    """run_experiment() — all paths."""

    def _run(self, health, retro_content=None, retro_missing=False, retro_raw: str | None = None):
        """Helper: patch dependencies and run the experiment in a temp dir.

        Parameters
        ----------
        retro_raw : str | None
            When set, write this raw string directly to the retro file (bypassing
            json.dumps) so tests can produce genuinely malformed JSON.
        """
        import scripts.experiment_426_dual_gpu_fix as exp426  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            # Write fake retro file
            retro_path = tmp_root / "results" / "operational_retro_2026_04_31.json"
            if not retro_missing:
                retro_path.parent.mkdir(parents=True, exist_ok=True)
                if retro_raw is not None:
                    retro_path.write_text(retro_raw)
                else:
                    retro_path.write_text(
                        json.dumps(retro_content if retro_content is not None else _RETRO_DATA)
                    )

            with (
                patch.object(exp426, "_REPO_ROOT", tmp_root),
                patch.object(exp426, "_autofix_result", _make_autofix_result()),
                patch(
                    "scripts.experiment_426_dual_gpu_fix.check_dual_gpu_health",
                    return_value=health,
                ),
            ):
                artifact = exp426.run_experiment()

        return artifact

    def test_ci_defaults_artifact_structure(self):
        """CI path: all required fields present in output artifact."""
        artifact = self._run(_CI_HEALTH)
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_ci_defaults_schema(self):
        """Artifact schema key is list of sorted field names."""
        artifact = self._run(_CI_HEALTH)
        assert isinstance(artifact["schema"], list)

    def test_ci_defaults_verdict_healthy(self):
        """CI health → honest_verdict='gpu1_healthy', retro_025_status='zombie_cleared'."""
        artifact = self._run(_CI_HEALTH)
        assert artifact["honest_verdict"] == "gpu1_healthy"
        assert artifact["retro_025_status"] == "zombie_cleared"

    def test_zombie_detected_verdict(self):
        """SCENARIO-INFRA-031: zombie health → zombie_detected verdict in artifact."""
        artifact = self._run(_ZOMBIE_HEALTH)
        assert artifact["honest_verdict"] == "zombie_detected"
        assert artifact["retro_025_status"] == "zombie_confirmed"
        assert artifact["gpu1_is_zombie"] is True

    def test_temperature_warning_in_artifact(self):
        """SCENARIO-INFRA-033: temperature_warning=True propagated to artifact."""
        artifact = self._run(_TEMP_WARN_HEALTH)
        assert artifact["temperature_warning"] is True
        assert artifact["recommended_batch_size_factor"] == pytest.approx(0.75)

    def test_retro_context_embedded(self):
        """When retro file present, incident VRAM and util values embedded."""
        artifact = self._run(_CI_HEALTH, retro_content=_RETRO_DATA)
        assert artifact["retro_025_incident_gpu1_vram_mb"] == 1786
        assert artifact["retro_025_incident_gpu1_util_pct"] == 0

    def test_retro_file_missing(self):
        """When retro file is missing, experiment completes without exception."""
        artifact = self._run(_CI_HEALTH, retro_missing=True)
        assert artifact["status"] == "success"
        assert artifact["retro_025_incident_gpu1_vram_mb"] is None

    def test_retro_file_malformed_json(self):
        """When retro file has malformed JSON, warning logged and experiment proceeds."""
        artifact = self._run(_CI_HEALTH, retro_raw="not_valid_json{{{")
        assert artifact["status"] == "success"

    def test_env_autofix_block_present(self):
        """env_autofix dict is always embedded in artifact."""
        artifact = self._run(_CI_HEALTH)
        assert "env_autofix" in artifact
        assert "gpu_detected" in artifact["env_autofix"]
        assert "auto_fix_applied" in artifact["env_autofix"]

    def test_artifact_written_to_disk(self):
        """Output JSON file is written to deliverable path."""
        import scripts.experiment_426_dual_gpu_fix as exp426  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            retro_path = tmp_root / "results" / "operational_retro_2026_04_31.json"
            retro_path.parent.mkdir(parents=True, exist_ok=True)
            retro_path.write_text(json.dumps(_RETRO_DATA))

            with (
                patch.object(exp426, "_REPO_ROOT", tmp_root),
                patch.object(exp426, "_autofix_result", _make_autofix_result()),
                patch(
                    "scripts.experiment_426_dual_gpu_fix.check_dual_gpu_health",
                    return_value=_CI_HEALTH,
                ),
            ):
                exp426.run_experiment()

            output_path = tmp_root / "results" / "experiment_426_dual_gpu_fix.json"
            assert output_path.exists(), "Output JSON file was not written"
            data = json.loads(output_path.read_text())
            assert data["experiment"] == 426


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


class TestMain:
    """main() — calls run_experiment() inside watchdog, logs headline."""

    def test_main_runs_without_error(self):
        """main() completes without raising when run_experiment() succeeds."""
        import scripts.experiment_426_dual_gpu_fix as exp426  # noqa: PLC0415

        mock_artifact = {
            "honest_verdict": "gpu1_healthy",
            "retro_025_status": "zombie_cleared",
            "temperature_warning": False,
            "recommended_batch_size_factor": 1.0,
            "experiment": 426,
        }

        with (
            patch("scripts.experiment_426_dual_gpu_fix.run_experiment", return_value=mock_artifact),
            patch("scripts.experiment_426_dual_gpu_fix.get_timeout_minutes", return_value=1),
            patch("scripts.experiment_426_dual_gpu_fix.ExperimentTimeoutWatchdog") as mock_watchdog,
        ):
            mock_watchdog.return_value.__enter__ = lambda s: s
            mock_watchdog.return_value.__exit__ = MagicMock(return_value=False)
            exp426.main()

        mock_watchdog.assert_called_once_with(
            experiment_id=426,
            timeout_minutes=1,
            result_path=str(exp426._REPO_ROOT / exp426.DELIVERABLE),
        )

    def test_main_calls_run_experiment(self):
        """main() calls run_experiment() exactly once."""
        import scripts.experiment_426_dual_gpu_fix as exp426  # noqa: PLC0415

        mock_artifact = {
            "honest_verdict": "gpu1_healthy",
            "retro_025_status": "zombie_cleared",
            "temperature_warning": False,
            "recommended_batch_size_factor": 1.0,
        }

        with (
            patch(
                "scripts.experiment_426_dual_gpu_fix.run_experiment",
                return_value=mock_artifact,
            ) as mock_run,
            patch("scripts.experiment_426_dual_gpu_fix.get_timeout_minutes", return_value=1),
            patch("scripts.experiment_426_dual_gpu_fix.ExperimentTimeoutWatchdog") as mock_watchdog,
        ):
            mock_watchdog.return_value.__enter__ = lambda s: s
            mock_watchdog.return_value.__exit__ = MagicMock(return_value=False)
            exp426.main()

        mock_run.assert_called_once()
