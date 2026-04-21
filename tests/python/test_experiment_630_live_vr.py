"""Tests for Experiment 630: Live VR Attempt #16 — blocked gate path.

Spec: REQ-VERIFY-133 (gate_open=False branch — blocked artifact must be written
      with all required schema fields; VR must not run when gate is closed).
SCENARIO-VERIFY-173: gate_open=False → artifact has status='blocked'.
SCENARIO-VERIFY-174: gate_open=False → retro_033_resolved=False, signed_improvement=0.0.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Helpers — import the script as a module without executing __main__
# ---------------------------------------------------------------------------

SCRIPT_PATH = pathlib.Path("scripts/experiment_630_live_vr_attempt_16.py")
ARTIFACT_PATH = pathlib.Path("results/experiment_630_live_vr_attempt_16.json")


def _load_script() -> types.ModuleType:
    """Import the script module without running main()."""
    spec = importlib.util.spec_from_file_location(
        "experiment_630_live_vr_attempt_16", SCRIPT_PATH
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    # Execute module-level code but NOT __main__ block (guarded by if __name__ == '__main__').
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-173: status field
# ---------------------------------------------------------------------------


class TestBlockedArtifactStatus:
    """REQ-VERIFY-133 / SCENARIO-VERIFY-173 — artifact must have status='blocked'."""

    def test_artifact_exists(self) -> None:
        """The deliverable JSON must be present on disk."""
        assert ARTIFACT_PATH.exists(), f"Artifact missing: {ARTIFACT_PATH}"

    def test_artifact_is_valid_json(self) -> None:
        """Artifact must be parseable JSON."""
        data = json.loads(ARTIFACT_PATH.read_text())
        assert isinstance(data, dict)

    def test_status_is_blocked(self) -> None:
        """SCENARIO-VERIFY-173: gate_open=False → status='blocked'."""
        data = json.loads(ARTIFACT_PATH.read_text())
        assert data["status"] == "blocked"

    def test_gate_open_is_false(self) -> None:
        """gate_open field must be False in the blocked artifact."""
        data = json.loads(ARTIFACT_PATH.read_text())
        assert data["gate_open"] is False

    def test_schema_field(self) -> None:
        """Artifact must declare the correct schema identifier."""
        data = json.loads(ARTIFACT_PATH.read_text())
        assert data["schema"] == "carnot.live_vr_16.v1"

    def test_honest_verdict(self) -> None:
        """honest_verdict must be the gate-closed sentinel value."""
        data = json.loads(ARTIFACT_PATH.read_text())
        assert data["honest_verdict"] == "blocked_gate_closed_do_not_retry"

    def test_block_reason_mentions_recall(self) -> None:
        """block_reason must mention both the threshold and the gate condition."""
        data = json.loads(ARTIFACT_PATH.read_text())
        reason = data["block_reason"]
        assert "0.20" in reason
        assert "recall" in reason.lower()


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-174: numeric outcome fields
# ---------------------------------------------------------------------------


class TestBlockedArtifactOutcomes:
    """REQ-VERIFY-133 / SCENARIO-VERIFY-174 — numeric fields must be zero/False."""

    def _data(self) -> dict:
        return json.loads(ARTIFACT_PATH.read_text())

    def test_retro_033_resolved_false(self) -> None:
        """SCENARIO-VERIFY-174: retro_033_resolved must be False when gate is closed."""
        assert self._data()["retro_033_resolved"] is False

    def test_signed_improvement_zero(self) -> None:
        """SCENARIO-VERIFY-174: signed_improvement must be 0.0 when blocked."""
        assert self._data()["signed_improvement"] == 0.0

    def test_n_questions_zero(self) -> None:
        """No questions are evaluated when the gate is closed."""
        assert self._data()["n_questions"] == 0

    def test_n_fixed_zero(self) -> None:
        assert self._data()["n_fixed"] == 0

    def test_n_broken_zero(self) -> None:
        assert self._data()["n_broken"] == 0

    def test_n_violations_found_zero(self) -> None:
        assert self._data()["n_violations_found"] == 0

    def test_interwhen_recall_recorded(self) -> None:
        """The diagnostic recall that caused the block must be preserved."""
        data = self._data()
        assert data["interwhen_recall_primary"] == pytest.approx(0.12)
        assert data["recall_threshold_required"] == pytest.approx(0.20)

    def test_experiment_number(self) -> None:
        assert self._data()["experiment"] == 630

    def test_run_date(self) -> None:
        assert self._data()["run_date"] == "20260421"

    def test_extractor_used(self) -> None:
        assert self._data()["extractor_used"] == "interwhen_symcode"

    def test_inference_mode(self) -> None:
        assert self._data()["inference_mode"] == "blocked_not_run"


# ---------------------------------------------------------------------------
# Script module — OUTPUT_PATH constant and main() smoke test
# ---------------------------------------------------------------------------


class TestScriptModule:
    """Cover the script module constants and main() rewrite behavior."""

    def test_output_path_constant(self) -> None:
        """Script must export OUTPUT_PATH pointing to the correct artifact location."""
        mod = _load_script()
        assert str(mod.OUTPUT_PATH) == "results/experiment_630_live_vr_attempt_16.json"

    def test_main_rewrites_artifact(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() must produce a valid blocked artifact even if called a second time."""
        mod = _load_script()
        # Redirect output to tmp_path so we don't clobber the real result.
        monkeypatch.setattr(mod, "OUTPUT_PATH", tmp_path / "out.json")
        mod.main()
        data = json.loads((tmp_path / "out.json").read_text())
        assert data["status"] == "blocked"
        assert data["gate_open"] is False
        assert data["retro_033_resolved"] is False
        assert data["signed_improvement"] == 0.0
