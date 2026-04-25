"""Tests for Experiment 873: JEPA v25 deploy gate check.

Traces to: REQ-LEARN-050

**What we test:**
    - Gate check logic: blocked when Exp 872 ood_auc <= 0.65.
    - Deliverable JSON exists with all required schema fields.
    - cascade_deployed is False when gate fails.
    - honest_verdict is "blocked" when gate fails.
    - blocked_by is the canonical sentinel string.

These tests run entirely on CPU; no GPU or live model required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DELIVERABLE = Path("results/experiment_873_jepa_v25_deploy.json")
EXP_872_ARTIFACT = Path("results/experiment_872_jepa_v25_dg_prm.json")

OOD_AUC_GATE = 0.65

REQUIRED_FIELDS = {
    "experiment",
    "title",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "status",
    "honest_verdict",
    "cascade_deployed",
    "blocked_by",
}


@pytest.fixture(scope="module")
def artifact() -> dict:
    """Load the deliverable artifact once for all tests in this module."""
    assert DELIVERABLE.exists(), f"Deliverable not found: {DELIVERABLE}"
    with DELIVERABLE.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def exp872() -> dict:
    """Load the Exp 872 artifact for gate-check verification."""
    assert EXP_872_ARTIFACT.exists(), f"Exp 872 artifact not found: {EXP_872_ARTIFACT}"
    with EXP_872_ARTIFACT.open() as f:
        return json.load(f)


class TestGateCheckLogic:
    """Verify the gate check produces the correct blocked outcome."""

    def test_exp872_ood_auc_below_gate(self, exp872: dict) -> None:
        # REQ-LEARN-050: gate requires ood_auc > 0.65; Exp 872 did not meet it.
        assert exp872["ood_auc"] <= OOD_AUC_GATE, (
            f"Exp 872 ood_auc={exp872['ood_auc']} unexpectedly meets gate {OOD_AUC_GATE}; "
            "Exp 873 should have deployed, not blocked."
        )

    def test_artifact_status_blocked(self, artifact: dict) -> None:
        # Gate miss must produce status="blocked".
        assert artifact["status"] == "blocked"

    def test_cascade_not_deployed(self, artifact: dict) -> None:
        # When gate fails, JEPA v25 must NOT be deployed as Tier 2 default.
        assert artifact["cascade_deployed"] is False

    def test_honest_verdict_blocked(self, artifact: dict) -> None:
        assert artifact["honest_verdict"] == "blocked"

    def test_blocked_by_sentinel(self, artifact: dict) -> None:
        # The canonical sentinel lets downstream reconcilers identify this gate failure.
        assert artifact["blocked_by"] == "exp872_ood_auc_below_0.65"

    def test_gate_values_recorded(self, artifact: dict) -> None:
        # Artifact must record both the required threshold and actual value for auditability.
        assert artifact["gate_ood_auc_required"] == OOD_AUC_GATE
        assert artifact["gate_ood_auc_actual"] == pytest.approx(0.484375, abs=1e-6)


class TestDeliverableSchema:
    """Verify the deliverable artifact has all required fields."""

    def test_all_required_fields_present(self, artifact: dict) -> None:
        missing = REQUIRED_FIELDS - artifact.keys()
        assert not missing, f"Deliverable is missing required fields: {missing}"

    def test_experiment_id(self, artifact: dict) -> None:
        assert artifact["experiment"] == 873

    def test_schema_field_is_sorted_list(self, artifact: dict) -> None:
        # schema field must be a sorted list (conductor convention).
        schema = artifact.get("schema", [])
        assert isinstance(schema, list)
        assert schema == sorted(schema)

    def test_invariant_violations_empty(self, artifact: dict) -> None:
        assert artifact.get("invariant_violations", []) == []
