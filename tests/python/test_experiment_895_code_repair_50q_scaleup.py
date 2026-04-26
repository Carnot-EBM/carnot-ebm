"""Tests for Exp 895: Code Repair 50Q Scale-Up gate check and artifact schema.

Spec: REQ-CODE-042 (50-question scale-up), SCENARIO-CODE-042

These tests verify:
  1. The gate check correctly reads Exp 881's signed_improvement and blocks when <= 0.
  2. The blocked artifact has all required schema fields.
  3. The honest_verdict mapping logic is correct.
  4. The artifact written to disk is valid JSON with the expected structure.

Why the gate check matters: Exp 881 showed zero_constraints (signed_improvement=0.0),
meaning the repair pipeline produced no improvement on 25 HumanEval problems.
Running 50 problems without first fixing the root cause would waste GPU time
and compound failure without diagnostic value. The gate enforces this discipline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers that replicate the gate check logic from the experiment script
# (REQ-CODE-042)
# ---------------------------------------------------------------------------

REQUIRED_ARTIFACT_FIELDS = [
    "experiment",
    "schema",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "status",
    "title",
]


def _gate_check_881(signed_improvement: float) -> bool:
    """Return True if Exp 881's signed_improvement passes the gate (> 0).

    The gate exists because Exp 895 is a scale-up of Exp 881. Running it when
    the baseline shows no improvement would produce meaningless results and
    waste GPU time. This mirrors the Failed-Experiment Rerun Discipline in
    CLAUDE.md — we only scale up what has already shown positive signal.
    """
    return signed_improvement > 0.0


def _honest_verdict(
    signed_improvement_50q: float | None,
    gate_passed: bool,
    gpu_available: bool,
) -> str:
    """Map experiment outcome to a canonical honest_verdict string.

    The verdict strings are consumed by the conductor's retrospective scripts;
    changing them would break trend analysis in results/operational_retro_*.json.
    """
    if not gpu_available:
        return "blocked_no_gpu"
    if not gate_passed:
        return "blocked_gate_881_not_met"
    if signed_improvement_50q is None:
        return "blocked_gate_881_not_met"
    if signed_improvement_50q > 0.0:
        return "code_repair_50q_positive"
    if signed_improvement_50q == 0.0:
        return "code_repair_50q_neutral"
    return "code_repair_50q_regression"


# ---------------------------------------------------------------------------
# Gate check unit tests (REQ-CODE-042, SCENARIO-CODE-042)
# ---------------------------------------------------------------------------


class TestGateCheck:
    """Unit tests for the Exp 881 gate check (REQ-CODE-042)."""

    def test_gate_passes_when_improvement_positive(self):
        # SCENARIO-CODE-042: gate allows scale-up when baseline is positive
        assert _gate_check_881(0.04) is True

    def test_gate_fails_when_improvement_zero(self):
        # SCENARIO-CODE-042: gate blocks scale-up when baseline is exactly zero
        assert _gate_check_881(0.0) is False

    def test_gate_fails_when_improvement_negative(self):
        # SCENARIO-CODE-042: gate blocks scale-up when baseline regressed
        assert _gate_check_881(-0.08) is False

    def test_gate_fails_on_small_positive_still_passes(self):
        # Any strictly positive value (even tiny) satisfies the gate
        assert _gate_check_881(0.001) is True


class TestHonestVerdictMapping:
    """Unit tests for honest_verdict mapping (REQ-CODE-042)."""

    def test_blocked_no_gpu(self):
        assert _honest_verdict(None, gate_passed=True, gpu_available=False) == "blocked_no_gpu"

    def test_blocked_gate_not_met(self):
        assert (
            _honest_verdict(None, gate_passed=False, gpu_available=True)
            == "blocked_gate_881_not_met"
        )

    def test_positive_improvement(self):
        assert (
            _honest_verdict(0.08, gate_passed=True, gpu_available=True)
            == "code_repair_50q_positive"
        )

    def test_neutral_no_improvement(self):
        assert (
            _honest_verdict(0.0, gate_passed=True, gpu_available=True) == "code_repair_50q_neutral"
        )

    def test_regression(self):
        assert (
            _honest_verdict(-0.04, gate_passed=True, gpu_available=True)
            == "code_repair_50q_regression"
        )


# ---------------------------------------------------------------------------
# Artifact schema tests (REQ-CODE-042)
# ---------------------------------------------------------------------------


DELIVERABLE = (
    Path(__file__).parent.parent.parent / "results" / "experiment_895_code_repair_50q_scaleup.json"
)


class TestArtifactSchema:
    """Verify the written artifact has the correct schema (REQ-CODE-042).

    These tests run against the real artifact on disk so they confirm that
    the experiment script actually wrote what it claimed. A missing or
    malformed artifact means the conductor cannot parse the result.
    """

    @pytest.fixture(scope="class")
    def artifact(self):
        assert DELIVERABLE.exists(), (
            f"Deliverable not found at {DELIVERABLE}. "
            "The experiment must have written results/experiment_895_code_repair_50q_scaleup.json."
        )
        return json.loads(DELIVERABLE.read_text())

    def test_required_fields_present(self, artifact):
        # Every conductor-consumed artifact must carry these fields (REQ-VERIFY-083)
        for field in REQUIRED_ARTIFACT_FIELDS:
            assert field in artifact, f"Required field '{field}' missing from artifact"

    def test_experiment_id_correct(self, artifact):
        assert artifact["experiment"] == 895

    def test_schema_version(self, artifact):
        assert artifact["schema"] == "carnot-experiment-v1"

    def test_honest_verdict_is_known_value(self, artifact):
        known_verdicts = {
            "code_repair_50q_positive",
            "code_repair_50q_neutral",
            "code_repair_50q_regression",
            "blocked_gate_881_not_met",
            "blocked_no_gpu",
        }
        assert artifact["honest_verdict"] in known_verdicts, (
            f"Unknown honest_verdict '{artifact['honest_verdict']}'. "
            "Add it to known_verdicts if this is intentional."
        )

    def test_blocked_artifact_has_gate_check_info(self, artifact):
        # When blocked, the artifact must document WHY it was blocked so
        # the retrospective can diagnose the root cause without re-running.
        if artifact["status"] == "blocked":
            assert "gate_check" in artifact or "honest_verdict" in artifact
            assert artifact["honest_verdict"] in (
                "blocked_gate_881_not_met",
                "blocked_no_gpu",
            )

    def test_gate_check_actual_value_recorded(self, artifact):
        # The gate_check block must record the actual signed_improvement value
        # from Exp 881 so the conductor knows what blocked this run.
        if artifact["honest_verdict"] == "blocked_gate_881_not_met":
            assert "gate_check" in artifact
            assert "actual" in artifact["gate_check"]

    def test_duration_non_negative(self, artifact):
        assert artifact["duration_s"] >= 0

    def test_status_valid(self, artifact):
        assert artifact["status"] in ("success", "blocked", "partial", "failed")
