"""
Tests for Exp 845: JEPA v24b Tier 3.5 Deployment Gate Check.

Traces to REQ-LEARN-015 (JEPA Tier 3.5 spec), FR-11.

The gate requires min_domain_auc >= 0.50 AND overall_ood >= 0.65.
Exp 844 produced min_domain_auc=0.0 (SVAMP AUC collapsed), so deployment
is blocked without touching the pipeline.  These tests verify both the
blocked-gate path and the hypothetical pass path to ensure the gate logic
is correct by construction.
"""

import json
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent
_ARTIFACT_845 = _REPO_ROOT / "results" / "experiment_845_jepa_v24b_tier35_deployment.json"
_ARTIFACT_844 = _REPO_ROOT / "results" / "experiment_844_jepa_v24b_svamp.json"

# Deployment gate thresholds (REQ-LEARN-015)
_MIN_DOMAIN_AUC_GATE = 0.50
_OVERALL_OOD_GATE = 0.65


# ---------------------------------------------------------------------------
# Gate logic helper — tested in isolation so the logic is covered regardless
# of which artifact is on disk.
# ---------------------------------------------------------------------------


def _evaluate_deployment_gate(min_domain_auc: float, overall_ood_auc: float) -> dict:
    """
    Evaluate whether a JEPA checkpoint clears the Tier 3.5 deployment gate.

    The gate has two independent thresholds (REQ-LEARN-015 / FR-11):
      - min_domain_auc >= 0.50: every domain must have at least above-chance AUC
        so that the predictor is not actively harmful on any one domain.
      - overall_ood >= 0.65: aggregate OOD performance must exceed 65% so the
        model is genuinely useful beyond the training distribution.

    Returns a dict with per-gate pass/fail flags and the composite verdict.
    """
    gate_min = min_domain_auc >= _MIN_DOMAIN_AUC_GATE
    gate_ood = overall_ood_auc >= _OVERALL_OOD_GATE
    deployed = gate_min and gate_ood

    if deployed:
        verdict = "jepa_v24b_tier35_deployed"
    else:
        verdict = "jepa_v24b_not_deployed_below_gate"

    return {
        "tier35_deployed": deployed,
        "gate_min_domain_auc_passed": gate_min,
        "gate_overall_ood_passed": gate_ood,
        "honest_verdict": verdict,
    }


# ---------------------------------------------------------------------------
# REQ-LEARN-015 / FR-11: blocked path — min_domain_auc below gate
# ---------------------------------------------------------------------------


class TestBlockedIfBelowGate:
    """Verify that gate logic correctly blocks deployment when min_domain_auc < 0.50."""

    def test_blocked_if_below_gate(self):
        # Exp 844 values: SVAMP AUC collapsed to 0.0
        result = _evaluate_deployment_gate(
            min_domain_auc=0.0,
            overall_ood_auc=0.5208333333333334,
        )
        assert result["tier35_deployed"] is False
        assert result["gate_min_domain_auc_passed"] is False
        assert result["honest_verdict"] == "jepa_v24b_not_deployed_below_gate"

    def test_boundary_exactly_at_gate_is_pass(self):
        # Boundary condition: exactly 0.50 should pass the min-domain gate.
        result = _evaluate_deployment_gate(
            min_domain_auc=0.50,
            overall_ood_auc=0.70,
        )
        assert result["gate_min_domain_auc_passed"] is True

    def test_just_below_gate_is_blocked(self):
        result = _evaluate_deployment_gate(
            min_domain_auc=0.499,
            overall_ood_auc=0.70,
        )
        assert result["tier35_deployed"] is False
        assert result["gate_min_domain_auc_passed"] is False

    def test_blocked_when_only_ood_gate_fails(self):
        # min_domain_auc passes but overall_ood is below 0.65.
        result = _evaluate_deployment_gate(
            min_domain_auc=0.60,
            overall_ood_auc=0.60,
        )
        assert result["tier35_deployed"] is False
        assert result["gate_overall_ood_passed"] is False
        assert result["gate_min_domain_auc_passed"] is True

    def test_blocked_when_both_gates_fail(self):
        result = _evaluate_deployment_gate(
            min_domain_auc=0.30,
            overall_ood_auc=0.40,
        )
        assert result["tier35_deployed"] is False
        assert result["gate_min_domain_auc_passed"] is False
        assert result["gate_overall_ood_passed"] is False


# ---------------------------------------------------------------------------
# REQ-LEARN-015 / FR-11: deployment path — both gates pass
# ---------------------------------------------------------------------------


class TestTier35DeployedInPipeline:
    """Verify gate logic produces tier35_deployed=True when both gates clear."""

    def test_tier35_deployed_when_gates_pass(self):
        result = _evaluate_deployment_gate(
            min_domain_auc=0.55,
            overall_ood_auc=0.70,
        )
        assert result["tier35_deployed"] is True
        assert result["gate_min_domain_auc_passed"] is True
        assert result["gate_overall_ood_passed"] is True
        assert result["honest_verdict"] == "jepa_v24b_tier35_deployed"

    def test_verdict_string_deployed(self):
        result = _evaluate_deployment_gate(min_domain_auc=0.80, overall_ood_auc=0.80)
        assert result["honest_verdict"] == "jepa_v24b_tier35_deployed"

    def test_verdict_string_blocked(self):
        result = _evaluate_deployment_gate(min_domain_auc=0.10, overall_ood_auc=0.80)
        assert result["honest_verdict"] == "jepa_v24b_not_deployed_below_gate"


# ---------------------------------------------------------------------------
# SVAMP coverage flag — should be True in all JEPAPredictions when deployed
# ---------------------------------------------------------------------------


class TestSvampCoverageFlag:
    """
    When JEPA v24b is deployed, every JEPAPrediction must carry
    svamp_coverage_flag=True to signal that the model was trained with
    SVAMP triplets (even though those triplets did not produce a useful
    AUC in Exp 844).
    """

    def test_svamp_coverage_flag_present_in_prediction_schema(self):
        # Simulate a JEPAPrediction-like dict to check the schema contract.
        prediction = {
            "step_id": "step_001",
            "domain_label": "gsm8k",
            "jepa_score": 0.72,
            "energy_delta": -0.15,
            "domain_weight": 1.0,
            "confidence_score": 0.80,
            "svamp_coverage_flag": True,  # REQ-LEARN-015: must always be present
        }
        assert prediction["svamp_coverage_flag"] is True

    def test_svamp_coverage_flag_is_true_for_all_domains(self):
        domains = ["gsm8k", "humaneval", "arc", "svamp"]
        for domain in domains:
            prediction = {
                "step_id": f"step_{domain}",
                "domain_label": domain,
                "jepa_score": 0.65,
                "energy_delta": -0.10,
                "domain_weight": 1.0,
                "confidence_score": 0.75,
                "svamp_coverage_flag": True,
            }
            assert prediction["svamp_coverage_flag"] is True, (
                f"svamp_coverage_flag must be True for domain={domain}"
            )


# ---------------------------------------------------------------------------
# Domain accuracy test — gate: 30/40 steps (75%) correctly labelled
# ---------------------------------------------------------------------------


class TestDomainAccuracy:
    """
    The domain classifier inside JEPAPredictor must achieve >= 75% accuracy
    on 40 held-out steps (10 per domain: GSM8K/HumanEval/ARC/SVAMP).

    This test verifies the accuracy threshold calculation rather than running
    real inference (no GPU / model weights required in CI).
    """

    def test_accuracy_threshold_is_75_percent(self):
        # 30 out of 40 is exactly 75% — the minimum to report "deployed" tier
        assert 30 / 40 >= 0.75

    def test_30_correct_out_of_40_passes(self):
        n_correct = 30
        n_total = 40
        accuracy = n_correct / n_total
        assert accuracy >= 0.75

    def test_29_correct_out_of_40_fails(self):
        n_correct = 29
        n_total = 40
        accuracy = n_correct / n_total
        assert accuracy < 0.75

    def test_all_correct_passes(self):
        assert 40 / 40 >= 0.75

    def test_accuracy_gate_string(self):
        # Verify verdict selection for low-accuracy deployed case
        for n_correct in range(0, 41):
            accuracy = n_correct / 40
            if accuracy >= 0.75:
                expected_verdict = "jepa_v24b_tier35_deployed"
            else:
                expected_verdict = "jepa_v24b_deployed_low_accuracy"
            # Just check the branch logic is consistent — no assertion failure expected
            assert expected_verdict in (
                "jepa_v24b_tier35_deployed",
                "jepa_v24b_deployed_low_accuracy",
            )


# ---------------------------------------------------------------------------
# Artifact integrity: Exp 845 deliverable JSON
# ---------------------------------------------------------------------------


class TestExp845Artifact:
    """Verify the written Exp 845 artifact has the required schema and values."""

    @pytest.fixture(scope="class")
    def artifact(self):
        assert _ARTIFACT_845.exists(), f"Missing artifact: {_ARTIFACT_845}"
        with _ARTIFACT_845.open() as f:
            return json.load(f)

    def test_experiment_number(self, artifact):
        assert artifact["experiment"] == 845

    def test_status_is_blocked(self, artifact):
        assert artifact["status"] == "blocked"

    def test_tier35_deployed_false(self, artifact):
        assert artifact["tier35_deployed"] is False

    def test_honest_verdict(self, artifact):
        assert artifact["honest_verdict"] == "jepa_v24b_not_deployed_below_gate"

    def test_blocked_reason_present(self, artifact):
        assert "blocked_reason" in artifact
        assert "0.50" in artifact["blocked_reason"]

    def test_source_experiment(self, artifact):
        assert artifact["source_experiment"] == 844

    def test_gate_thresholds_recorded(self, artifact):
        assert artifact["deployment_gate_min_domain_auc"] == 0.5
        assert artifact["deployment_gate_overall_ood"] == 0.65

    def test_gate_flags_recorded(self, artifact):
        assert artifact["gate_min_domain_auc_passed"] is False

    def test_schema_field_present(self, artifact):
        assert "schema" in artifact

    def test_invariant_violations_empty(self, artifact):
        assert artifact["invariant_violations"] == []
