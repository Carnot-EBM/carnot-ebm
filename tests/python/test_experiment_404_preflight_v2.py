"""Tests for experiment_404_preflight_v2 script — 100% coverage.

Coverage targets
----------------
- main(): gpu_confirmed_live path (no cloud script), gpu_hardware_not_live path
  (cloud script generated), prints ACTION REQUIRED when not live
- All branches: cloud_gpu_script_generated=True/False, retro_022_resolved=True/False
- Artifact schema validation: all required fields present

Spec: REQ-INFRA-019, REQ-INFRA-020,
      SCENARIO-INFRA-022, SCENARIO-INFRA-023, SCENARIO-INFRA-024
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import once — do NOT reload inside tests (reload resets module-level _REPO_ROOT)
import scripts.experiment_404_preflight_v2 as _mod404  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to build mock GPUPreflightResult
# ---------------------------------------------------------------------------

_ALL_KNOWN_FILES = [
    "python/carnot/models/cikan_energy.py",
    "python/carnot/pipeline/jitrl_memory.py",
    "python/carnot/models/safety_kan.py",
    "python/carnot/pipeline/semantic_energy_scorer.py",
    "python/carnot/pipeline/crane_extractor.py",
]


def _all_missing_audit() -> dict[str, str]:
    return {k: "missing" for k in _ALL_KNOWN_FILES}


def _all_valid_audit() -> dict[str, str]:
    return {k: "valid_python" for k in _ALL_KNOWN_FILES}


def _make_preflight_result(
    honest_verdict: str,
    is_live_capable: bool = False,
) -> MagicMock:
    """Return a mock GPUPreflightResult."""
    r = MagicMock()
    r.honest_verdict = honest_verdict
    r.is_live_capable = is_live_capable
    r.env_var_set = False
    r.subprocess_inherits_env = False
    r.session_startup_exists = True
    r.conductor_gpu_env_exists = True
    r.smoke_test_passed = False
    return r


def _make_live_preflight_result() -> MagicMock:
    """Return a mock result representing a fully live GPU node."""
    r = MagicMock()
    r.honest_verdict = "gpu_confirmed_live"
    r.is_live_capable = True
    r.env_var_set = True
    r.subprocess_inherits_env = True
    r.session_startup_exists = True
    r.conductor_gpu_env_exists = True
    r.smoke_test_passed = True
    return r


def _run_main_with_patches(
    tmp_path: Path,
    preflight_result: MagicMock,
    audit: dict[str, str],
) -> dict:
    """Run main() with standard patches applied to the already-imported module."""
    with (
        patch.object(_mod404, "_REPO_ROOT", tmp_path),
        patch(
            "scripts.experiment_404_preflight_v2.run_gpu_preflight",
            return_value=preflight_result,
        ),
        patch(
            "scripts.experiment_404_preflight_v2.DeliverableContentValidator.audit_known_corrupt_files",
            return_value=audit,
        ),
    ):
        return _mod404.main()


# ---------------------------------------------------------------------------
# main() — gpu_hardware_not_live path (cloud script generated)
# ---------------------------------------------------------------------------


class TestMainGPUNotLive:
    """SCENARIO-INFRA-024: cloud GPU script generated when GPU not live."""

    def test_artifact_written_to_disk(self, tmp_path: Path) -> None:
        preflight_mock = _make_preflight_result(
            "gpu_hardware_not_live", is_live_capable=False
        )
        _run_main_with_patches(tmp_path, preflight_mock, _all_missing_audit())
        out = tmp_path / "results" / "experiment_404_preflight_v2.json"
        assert out.exists(), "artifact file must be written"
        saved = json.loads(out.read_text())
        assert saved["honest_verdict"] == "gpu_hardware_not_live"

    def test_cloud_gpu_script_generated(self, tmp_path: Path) -> None:
        preflight_mock = _make_preflight_result(
            "gpu_hardware_not_live", is_live_capable=False
        )
        artifact = _run_main_with_patches(tmp_path, preflight_mock, _all_missing_audit())
        assert artifact["cloud_gpu_script_generated"] is True
        script_path = tmp_path / "scripts" / "setup_cloud_gpu.sh"
        assert script_path.exists(), "setup_cloud_gpu.sh must be created"

    def test_artifact_has_required_fields(self, tmp_path: Path) -> None:
        preflight_mock = _make_preflight_result(
            "gpu_hardware_not_live", is_live_capable=False
        )
        artifact = _run_main_with_patches(tmp_path, preflight_mock, _all_missing_audit())
        required = {
            "experiment",
            "title",
            "run_date",
            "started_at",
            "finished_at",
            "duration_s",
            "status",
            "honest_verdict",
            "retro_022_resolved",
            "retro_023_root_cause_fixed",
            "corrupt_files_found",
            "n_corrupt_files",
            "cloud_gpu_script_generated",
        }
        for key in required:
            assert key in artifact, f"missing required artifact key: {key}"

    def test_retro_022_not_resolved_when_not_live(self, tmp_path: Path) -> None:
        preflight_mock = _make_preflight_result(
            "gpu_hardware_not_live", is_live_capable=False
        )
        artifact = _run_main_with_patches(tmp_path, preflight_mock, _all_missing_audit())
        assert artifact["retro_022_resolved"] is False

    def test_retro_023_always_fixed(self, tmp_path: Path) -> None:
        preflight_mock = _make_preflight_result(
            "gpu_hardware_not_live", is_live_capable=False
        )
        artifact = _run_main_with_patches(tmp_path, preflight_mock, _all_missing_audit())
        assert artifact["retro_023_root_cause_fixed"] is True

    def test_corrupt_files_listed_in_artifact(self, tmp_path: Path) -> None:
        corrupt_path = "python/carnot/models/cikan_energy.py"
        preflight_mock = _make_preflight_result(
            "gpu_hardware_not_live", is_live_capable=False
        )
        audit = {
            corrupt_path: "corrupt_json",
            "python/carnot/pipeline/jitrl_memory.py": "missing",
            "python/carnot/models/safety_kan.py": "missing",
            "python/carnot/pipeline/semantic_energy_scorer.py": "missing",
            "python/carnot/pipeline/crane_extractor.py": "missing",
        }
        artifact = _run_main_with_patches(tmp_path, preflight_mock, audit)
        assert corrupt_path in artifact["corrupt_files_found"]
        assert artifact["n_corrupt_files"] == 1

    def test_action_required_printed_when_not_live(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        preflight_mock = _make_preflight_result(
            "gpu_hardware_not_live", is_live_capable=False
        )
        _run_main_with_patches(tmp_path, preflight_mock, _all_missing_audit())
        captured = capsys.readouterr()
        assert "ACTION REQUIRED" in captured.out
        assert "gpu_hardware_not_live" in captured.out

    def test_status_success_in_artifact(self, tmp_path: Path) -> None:
        preflight_mock = _make_preflight_result(
            "gpu_hardware_not_live", is_live_capable=False
        )
        artifact = _run_main_with_patches(tmp_path, preflight_mock, _all_missing_audit())
        assert artifact["status"] == "success"
        assert artifact["experiment"] == 404


# ---------------------------------------------------------------------------
# main() — gpu_confirmed_live path (no cloud script)
# ---------------------------------------------------------------------------


class TestMainGPULive:
    """GPU confirmed live: no cloud script, retro_022_resolved=True."""

    def test_no_cloud_script_when_live(self, tmp_path: Path) -> None:
        artifact = _run_main_with_patches(
            tmp_path, _make_live_preflight_result(), _all_valid_audit()
        )
        assert artifact["cloud_gpu_script_generated"] is False
        script_path = tmp_path / "scripts" / "setup_cloud_gpu.sh"
        assert not script_path.exists(), "cloud script must NOT be created when GPU is live"

    def test_retro_022_resolved_when_live(self, tmp_path: Path) -> None:
        artifact = _run_main_with_patches(
            tmp_path, _make_live_preflight_result(), _all_valid_audit()
        )
        assert artifact["retro_022_resolved"] is True
        assert artifact["honest_verdict"] == "gpu_confirmed_live"

    def test_no_action_required_output_when_live(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _run_main_with_patches(
            tmp_path, _make_live_preflight_result(), _all_valid_audit()
        )
        captured = capsys.readouterr()
        assert "ACTION REQUIRED" not in captured.out

    def test_status_success_in_artifact(self, tmp_path: Path) -> None:
        artifact = _run_main_with_patches(
            tmp_path, _make_live_preflight_result(), _all_valid_audit()
        )
        assert artifact["status"] == "success"
        assert artifact["experiment"] == 404

    def test_n_corrupt_files_zero_when_all_valid(self, tmp_path: Path) -> None:
        artifact = _run_main_with_patches(
            tmp_path, _make_live_preflight_result(), _all_valid_audit()
        )
        assert artifact["n_corrupt_files"] == 0
        assert artifact["corrupt_files_found"] == []


# ---------------------------------------------------------------------------
# main() — scripts_missing and env_not_propagating verdicts
# ---------------------------------------------------------------------------


class TestMainOtherVerdicts:
    """Other honest_verdict values: scripts_missing, env_not_propagating."""

    def test_scripts_missing_generates_cloud_script(self, tmp_path: Path) -> None:
        preflight_mock = _make_preflight_result("scripts_missing", is_live_capable=False)
        artifact = _run_main_with_patches(tmp_path, preflight_mock, _all_missing_audit())
        assert artifact["honest_verdict"] == "scripts_missing"
        assert artifact["cloud_gpu_script_generated"] is True

    def test_env_not_propagating_generates_cloud_script(self, tmp_path: Path) -> None:
        preflight_mock = _make_preflight_result(
            "env_not_propagating", is_live_capable=False
        )
        artifact = _run_main_with_patches(tmp_path, preflight_mock, _all_missing_audit())
        assert artifact["honest_verdict"] == "env_not_propagating"
        assert artifact["cloud_gpu_script_generated"] is True


# ---------------------------------------------------------------------------
# Artifact schema validation
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    """Artifact must contain schema field and all required fields."""

    def test_schema_field_present(self, tmp_path: Path) -> None:
        preflight_mock = _make_preflight_result("gpu_hardware_not_live", is_live_capable=False)
        artifact = _run_main_with_patches(tmp_path, preflight_mock, _all_missing_audit())
        # ExperimentTemplate.build_result() sets schema to sorted list of keys
        assert "schema" in artifact

    def test_corrupt_audit_detail_in_artifact(self, tmp_path: Path) -> None:
        preflight_mock = _make_preflight_result("gpu_hardware_not_live", is_live_capable=False)
        audit = {
            "python/carnot/models/cikan_energy.py": "corrupt_json",
            "python/carnot/pipeline/jitrl_memory.py": "corrupt_json",
            "python/carnot/models/safety_kan.py": "missing",
            "python/carnot/pipeline/semantic_energy_scorer.py": "missing",
            "python/carnot/pipeline/crane_extractor.py": "missing",
        }
        artifact = _run_main_with_patches(tmp_path, preflight_mock, audit)
        assert artifact["n_corrupt_files"] == 2
        assert len(artifact["corrupt_files_found"]) == 2
        assert artifact["corrupt_audit_detail"] == audit

    def test_cloud_gpu_script_path_in_artifact(self, tmp_path: Path) -> None:
        preflight_mock = _make_preflight_result("gpu_hardware_not_live", is_live_capable=False)
        artifact = _run_main_with_patches(tmp_path, preflight_mock, _all_missing_audit())
        assert artifact["cloud_gpu_script_path"] == "scripts/setup_cloud_gpu.sh"

    def test_cloud_gpu_script_path_none_when_live(self, tmp_path: Path) -> None:
        artifact = _run_main_with_patches(
            tmp_path, _make_live_preflight_result(), _all_valid_audit()
        )
        assert artifact["cloud_gpu_script_path"] is None
