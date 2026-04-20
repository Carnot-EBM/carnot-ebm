"""Tests for exclusion manifest and nvidia-smi zombie kill fallback.

Spec coverage:
  REQ-INFRA-062  — ExperimentTemplate.check_exclusion_manifest() exits for excluded experiments
  REQ-INFRA-063  — kill_gpu_zombies() uses nvidia-smi subprocess fallback when pynvml unavailable
  SCENARIO-INFRA-086 — check_exclusion_manifest() returns False when manifest file missing
  SCENARIO-INFRA-087 — check_exclusion_manifest() writes excluded artifact and exits when in manifest
  SCENARIO-INFRA-088 — kill_gpu_zombies() falls back to nvidia-smi when pynvml not installed
  SCENARIO-INFRA-089 — kill_gpu_zombies() returns no_gpu_tooling when nvidia-smi not found
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_template(exp_id: int, tmp_path: Path, repo_root: Path | None = None):
    """Create an ExperimentTemplate with a temp deliverable path for testing."""
    from scripts.experiment_template import ExperimentTemplate

    deliverable = str(tmp_path / f"exp_{exp_id}_result.json")
    return ExperimentTemplate(
        exp_id,
        f"Test Exp {exp_id}",
        deliverable,
        requires_gpu=False,
        repo_root=repo_root or tmp_path,
    )


class TestCheckExclusionManifest:
    """Tests for ExperimentTemplate.check_exclusion_manifest().

    REQ-INFRA-062, SCENARIO-INFRA-086, SCENARIO-INFRA-087
    """

    def test_returns_false_when_manifest_missing(self, tmp_path: Path):
        """SCENARIO-INFRA-086: returns False when manifest file is absent (no FileNotFoundError)."""
        # tmp_path has no conductor_exclusion_manifest.json — the method must gracefully return False.
        tmpl = _make_template(999, tmp_path)
        result = tmpl.check_exclusion_manifest()
        assert result is False

    def test_returns_false_when_exp_id_not_in_manifest(self, tmp_path: Path):
        """Returns False when the manifest exists but this experiment ID is not excluded."""
        manifest = {
            "version": 1,
            "excluded_experiments": [308, 260, 309, 425, 410],
            "reason": "already modern",
            "added_milestone": "2026.04.42",
        }
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "conductor_exclusion_manifest.json").write_text(json.dumps(manifest))

        # exp_id=999 is not in the manifest
        tmpl = _make_template(999, tmp_path, repo_root=tmp_path)
        result = tmpl.check_exclusion_manifest()
        assert result is False

    def test_writes_excluded_artifact_and_exits_when_in_manifest(self, tmp_path: Path):
        """SCENARIO-INFRA-087: writes excluded artifact and calls sys.exit(0) when exp in manifest."""
        manifest = {
            "version": 1,
            "excluded_experiments": [308, 260],
            "reason": "fully modern per Exp 547",
            "added_milestone": "2026.04.42",
        }
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "conductor_exclusion_manifest.json").write_text(json.dumps(manifest))

        # results dir must exist for the deliverable write
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        tmpl = _make_template(308, results_dir, repo_root=tmp_path)
        # Override the output path to be inside results_dir (which exists)
        tmpl._output_path = results_dir / "exp_308_result.json"
        tmpl._guard._path = str(tmpl._output_path)

        with pytest.raises(SystemExit) as exc_info:
            tmpl.check_exclusion_manifest()

        assert exc_info.value.code == 0

        # The artifact must have been written before sys.exit(0)
        assert tmpl._output_path.exists()
        artifact = json.loads(tmpl._output_path.read_text())
        assert artifact["excluded"] is True
        assert artifact["honest_verdict"] == "excluded_already_modern"
        assert artifact["experiment"] == 308
        assert artifact["schema"] == "carnot.excluded.v1"
        assert artifact["reason"] == "fully modern per Exp 547"

    def test_returns_false_on_corrupt_manifest(self, tmp_path: Path):
        """Gracefully returns False when the manifest JSON is malformed."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "conductor_exclusion_manifest.json").write_text("not valid json {{{")

        tmpl = _make_template(308, tmp_path, repo_root=tmp_path)
        result = tmpl.check_exclusion_manifest()
        assert result is False


class TestKillGpuZombiesNvidiaSmi:
    """Tests for kill_gpu_zombies() nvidia-smi fallback path.

    REQ-INFRA-063, SCENARIO-INFRA-088, SCENARIO-INFRA-089
    """

    def test_nvidia_smi_fallback_when_pynvml_missing(self):
        """SCENARIO-INFRA-088: returns method='nvidia_smi_fallback' when pynvml not installed."""
        from scripts.experiment_template import ExperimentTemplate

        # Patch the import so pynvml appears unavailable, then let _kill_gpu_zombies_nvidia_smi
        # be called. Mock nvidia-smi to return an empty process list (no zombies this run).
        nvidia_smi_vram_output = ""  # no compute processes running
        nvidia_smi_util_output = "0\n"  # GPU at 0% utilization

        def fake_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if any("--query-compute-apps" in c for c in cmd):
                result.stdout = nvidia_smi_vram_output
            else:
                result.stdout = nvidia_smi_util_output
            return result

        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("scripts.experiment_template.subprocess.run", side_effect=fake_subprocess_run):
                result = ExperimentTemplate.kill_gpu_zombies(
                    vram_threshold_mb=1000,
                    util_threshold_pct=5.0,
                )

        assert result.get("method") == "nvidia_smi_fallback"
        assert "killed_pids" in result
        assert isinstance(result["killed_pids"], list)
        assert "freed_mb" in result

    def test_nvidia_smi_kills_zombie_process(self):
        """kill_gpu_zombies() sends SIGTERM to processes above VRAM threshold at low utilization."""
        import os
        import signal

        from scripts.experiment_template import ExperimentTemplate

        # nvidia-smi reports one process: PID 99999, 5000 MB VRAM; GPU at 0% utilization
        nvidia_smi_vram_output = "99999, 5000\n"
        nvidia_smi_util_output = "0\n"

        def fake_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if any("--query-compute-apps" in c for c in cmd):
                result.stdout = nvidia_smi_vram_output
            else:
                result.stdout = nvidia_smi_util_output
            return result

        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("scripts.experiment_template.subprocess.run", side_effect=fake_subprocess_run):
                with patch("scripts.experiment_template.os.kill") as mock_kill:
                    result = ExperimentTemplate.kill_gpu_zombies(
                        vram_threshold_mb=1000,
                        util_threshold_pct=5.0,
                    )
                    mock_kill.assert_called_once_with(99999, signal.SIGTERM)

        assert 99999 in result["killed_pids"]
        assert result["freed_mb"] == 5000
        assert result.get("method") == "nvidia_smi_fallback"

    def test_no_gpu_tooling_when_nvidia_smi_not_found(self):
        """SCENARIO-INFRA-089: returns error='no_gpu_tooling' when nvidia-smi not in PATH."""
        from scripts.experiment_template import ExperimentTemplate

        with patch.dict(sys.modules, {"pynvml": None}):
            with patch(
                "scripts.experiment_template.subprocess.run",
                side_effect=FileNotFoundError("nvidia-smi not found"),
            ):
                result = ExperimentTemplate.kill_gpu_zombies(
                    vram_threshold_mb=1000,
                    util_threshold_pct=5.0,
                )

        assert result == {"killed_pids": [], "freed_mb": 0, "error": "no_gpu_tooling"}

    def test_no_kill_when_vram_below_threshold(self):
        """Does not kill processes whose VRAM is below vram_threshold_mb."""
        from scripts.experiment_template import ExperimentTemplate

        # Process uses only 500 MB, threshold is 1000 MB — should NOT be killed
        nvidia_smi_vram_output = "11111, 500\n"
        nvidia_smi_util_output = "0\n"

        def fake_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if any("--query-compute-apps" in c for c in cmd):
                result.stdout = nvidia_smi_vram_output
            else:
                result.stdout = nvidia_smi_util_output
            return result

        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("scripts.experiment_template.subprocess.run", side_effect=fake_subprocess_run):
                with patch("scripts.experiment_template.os.kill") as mock_kill:
                    result = ExperimentTemplate.kill_gpu_zombies(
                        vram_threshold_mb=1000,
                        util_threshold_pct=5.0,
                    )
                    mock_kill.assert_not_called()

        assert result["killed_pids"] == []
        assert result["freed_mb"] == 0

    def test_no_kill_when_gpu_util_above_threshold(self):
        """Does not kill processes when GPU utilization is above util_threshold_pct."""
        from scripts.experiment_template import ExperimentTemplate

        # Process has 5000 MB VRAM but GPU is at 50% utilization — NOT a zombie
        nvidia_smi_vram_output = "22222, 5000\n"
        nvidia_smi_util_output = "50\n"  # 50% > 5% threshold

        def fake_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if any("--query-compute-apps" in c for c in cmd):
                result.stdout = nvidia_smi_vram_output
            else:
                result.stdout = nvidia_smi_util_output
            return result

        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("scripts.experiment_template.subprocess.run", side_effect=fake_subprocess_run):
                with patch("scripts.experiment_template.os.kill") as mock_kill:
                    result = ExperimentTemplate.kill_gpu_zombies(
                        vram_threshold_mb=1000,
                        util_threshold_pct=5.0,
                    )
                    mock_kill.assert_not_called()

        assert result["killed_pids"] == []


# ---------------------------------------------------------------------------
# Tests for the new ExclusionManifest class (REQ-INFRA-070, REQ-INFRA-071)
# RETRO-056: these five experiments consumed 2,485 minutes over 7 milestones.
# ---------------------------------------------------------------------------


class TestExclusionManifestClass:
    """Tests for carnot.pipeline.exclusion_manifest.ExclusionManifest.

    Spec: REQ-INFRA-070, REQ-INFRA-071,
          SCENARIO-INFRA-075, SCENARIO-INFRA-076, SCENARIO-INFRA-077
    """

    def test_load_returns_empty_list_when_file_missing(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-075: load() returns [] for a nonexistent file (no crash).

        REQ-INFRA-070: A missing manifest must not crash the conductor — it simply
        means no experiments are excluded yet.
        """
        from carnot.pipeline.exclusion_manifest import ExclusionManifest

        manifest = ExclusionManifest(str(tmp_path / "does_not_exist.json"))
        result = manifest.load()
        assert result == []

    def test_is_excluded_returns_true_for_exp_308(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-076: is_excluded(308) returns True after loading default manifest.

        REQ-INFRA-071: The conductor must skip excluded experiments — 308 is one of five.
        """
        import json

        from carnot.pipeline.exclusion_manifest import ExclusionManifest, build_default_manifest

        manifest_file = tmp_path / "manifest.json"
        entries = build_default_manifest()
        manifest_file.write_text(
            json.dumps({"excluded": [{"experiment_id": e.experiment_id,
                                      "completed_milestone": e.completed_milestone,
                                      "reason": e.reason} for e in entries]})
        )
        manifest = ExclusionManifest(str(manifest_file))
        manifest.load()
        assert manifest.is_excluded(308) is True

    def test_is_excluded_returns_false_for_unknown_id(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-076: is_excluded(999) returns False — unknown = allowed to run.

        REQ-INFRA-070: The manifest is an opt-in exclusion list, not a whitelist.
        """
        import json

        from carnot.pipeline.exclusion_manifest import ExclusionManifest, build_default_manifest

        manifest_file = tmp_path / "manifest.json"
        entries = build_default_manifest()
        manifest_file.write_text(
            json.dumps({"excluded": [{"experiment_id": e.experiment_id,
                                      "completed_milestone": e.completed_milestone,
                                      "reason": e.reason} for e in entries]})
        )
        manifest = ExclusionManifest(str(manifest_file))
        manifest.load()
        assert manifest.is_excluded(999) is False

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-077: save() + load() roundtrip preserves all entries exactly.

        REQ-INFRA-070: The JSON format must survive a write/read cycle without data loss.
        """
        from carnot.pipeline.exclusion_manifest import ExclusionEntry, ExclusionManifest

        manifest_file = tmp_path / "roundtrip.json"
        manifest = ExclusionManifest(str(manifest_file))

        original = [
            ExclusionEntry(experiment_id=308, completed_milestone="2026.04.37", reason="stuck"),
            ExclusionEntry(experiment_id=260, completed_milestone="2026.04.37", reason="loop"),
        ]
        manifest.save(original)

        fresh = ExclusionManifest(str(manifest_file))
        loaded = fresh.load()

        assert len(loaded) == 2
        assert loaded[0].experiment_id == 308
        assert loaded[1].experiment_id == 260
        assert loaded[0].reason == "stuck"
