"""
Tests for experiment_620_live_vr_attempt_15.py

Coverage requirement: 100% of the new code in that script.

WHY these tests exist: RETRO-033 requires every experiment to have a
corresponding test suite so the gate logic can be verified independently of
running against a live GPU.
"""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# Make the scripts/ directory importable without installing the package.
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import experiment_620_live_vr_attempt_15 as exp620  # noqa: E402


class TestBlockedArtifact:
    """Exp 617 timed out, so Exp 620 must emit a blocked artifact."""

    def test_main_writes_artifact(self, tmp_path, monkeypatch):
        """main() must create the result file at the configured path."""
        result_path = tmp_path / "experiment_620_live_vr_attempt_15.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        exp620.main()

        assert result_path.exists(), "Result file must be written by main()"

    def test_artifact_schema(self, tmp_path, monkeypatch):
        """Artifact must carry schema='carnot.live_vr_15.v1'."""
        result_path = tmp_path / "result.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        exp620.main()

        artifact = json.loads(result_path.read_text())
        assert artifact["schema"] == "carnot.live_vr_15.v1"

    def test_artifact_status_blocked(self, tmp_path, monkeypatch):
        """status must be 'blocked' because gate_open=False."""
        result_path = tmp_path / "result.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        exp620.main()

        artifact = json.loads(result_path.read_text())
        assert artifact["status"] == "blocked"

    def test_artifact_gate_open_false(self, tmp_path, monkeypatch):
        """gate_open must be False — Exp 617 did not confirm it True."""
        result_path = tmp_path / "result.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        exp620.main()

        artifact = json.loads(result_path.read_text())
        assert artifact["gate_open"] is False

    def test_artifact_signed_improvement_zero(self, tmp_path, monkeypatch):
        """signed_improvement must be 0.0 — no VR was executed."""
        result_path = tmp_path / "result.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        exp620.main()

        artifact = json.loads(result_path.read_text())
        assert artifact["signed_improvement"] == 0.0

    def test_artifact_retro_033_not_resolved(self, tmp_path, monkeypatch):
        """retro_033_resolved must be False — the gate is still closed."""
        result_path = tmp_path / "result.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        exp620.main()

        artifact = json.loads(result_path.read_text())
        assert artifact["retro_033_resolved"] is False

    def test_artifact_honest_verdict(self, tmp_path, monkeypatch):
        """honest_verdict must be 'blocked_gate_closed_do_not_retry'."""
        result_path = tmp_path / "result.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        exp620.main()

        artifact = json.loads(result_path.read_text())
        assert artifact["honest_verdict"] == "blocked_gate_closed_do_not_retry"

    def test_artifact_experiment_number(self, tmp_path, monkeypatch):
        """experiment field must be 620."""
        result_path = tmp_path / "result.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        exp620.main()

        artifact = json.loads(result_path.read_text())
        assert artifact["experiment"] == 620

    def test_artifact_exp617_status(self, tmp_path, monkeypatch):
        """exp617_status must record that Exp 617 timed out."""
        result_path = tmp_path / "result.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        exp620.main()

        artifact = json.loads(result_path.read_text())
        assert artifact["exp617_status"] == "timed_out"

    def test_artifact_no_extractor_used(self, tmp_path, monkeypatch):
        """best_extractor_used must be None — no extractor was invoked."""
        result_path = tmp_path / "result.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        exp620.main()

        artifact = json.loads(result_path.read_text())
        assert artifact["best_extractor_used"] is None

    def test_artifact_n_questions_zero(self, tmp_path, monkeypatch):
        """n_questions must be 0 — no questions were processed."""
        result_path = tmp_path / "result.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        exp620.main()

        artifact = json.loads(result_path.read_text())
        assert artifact["n_questions"] == 0

    def test_artifact_block_reason_contains_threshold(self, tmp_path, monkeypatch):
        """block_reason must mention the 0.20 recall threshold."""
        result_path = tmp_path / "result.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        exp620.main()

        artifact = json.loads(result_path.read_text())
        assert "0.20" in artifact["block_reason"]

    def test_artifact_parent_dir_created(self, tmp_path, monkeypatch):
        """main() must create the parent directory if it does not exist."""
        deep_path = tmp_path / "deep" / "nested" / "result.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", deep_path)

        exp620.main()

        assert deep_path.exists()

    def test_main_as_script(self, tmp_path, monkeypatch):
        """Running as __main__ must also write a valid artifact."""
        result_path = tmp_path / "result_main.json"
        monkeypatch.setattr(exp620, "RESULT_PATH", result_path)

        # Simulate __main__ execution by calling main() directly.
        exp620.main()

        artifact = json.loads(result_path.read_text())
        assert artifact["status"] == "blocked"
