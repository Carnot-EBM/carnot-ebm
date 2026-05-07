"""Tests for Exp 1500 latent-vs-deterministic discipline gate.

Spec: REQ-VERIFY-1500, SCENARIO-VERIFY-1500.
"""

from __future__ import annotations

import pytest

from carnot.verify.latent_deterministic_gate import (
    REQUIRED_ARTIFACT_FIELDS,
    _validate_required_fields,
    build_discipline_gate_artifact,
    render_policy_markdown,
)


def test_scenario_verify_1500_demotes_confounded_latent_signals() -> None:
    """SCENARIO-VERIFY-1500: confounded latent signals are retired from claims."""

    artifact = build_discipline_gate_artifact(
        exp1499=_exp1499_with_matrix(),
        exp1481={
            "status": "complete",
            "claim_allowed": False,
            "diagnostic_lineage_retired": True,
            "signal_beats_superficial_baselines": False,
        },
        exp1487={
            "status": "complete",
            "improvement_allowed": False,
            "energy_ranking_accuracy": 1.0,
            "superficial_baseline_accuracy": 1.0,
        },
        ops_note_path="ops/latent_deterministic_discipline_gate_1500.md",
    )

    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["discipline_gate_ready"] is True
    assert artifact["gated_inputs_present"] is True
    assert "deterministic_executable_validators" in artifact["headline_allowed_signals"]
    assert "conservative_deterministic_bounds" in artifact["headline_allowed_signals"]
    assert "partial_trace_energy_localization_for_repair" in artifact["auxiliary_allowed_signals"]
    assert "semantic_energy_headline_telemetry" in artifact["retired_signals"]
    assert "v1_pairwise_self_verification_active_gate" in artifact["retired_signals"]
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1500_blocks_without_orthogonality_matrix() -> None:
    """REQ-VERIFY-1500: missing Exp 1499 matrix gates the discipline artifact."""

    artifact = build_discipline_gate_artifact(
        exp1499={"status": "complete", "orthogonality_matrix_written": False},
        exp1481={"status": "complete"},
        exp1487={"status": "complete"},
        ops_note_path="ops/latent_deterministic_discipline_gate_1500.md",
    )

    assert artifact["status"] == "blocked"
    assert artifact["discipline_gate_ready"] is False
    assert "missing_exp1499_orthogonality_matrix" in artifact["blockers"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1500_blocks_without_gated_inputs() -> None:
    """REQ-VERIFY-1500: all cited retirement and orthogonality inputs are required."""

    artifact = build_discipline_gate_artifact(
        exp1499=_exp1499_with_matrix(),
        exp1481={},
        exp1487={},
        ops_note_path="ops/latent_deterministic_discipline_gate_1500.md",
    )

    assert artifact["status"] == "blocked"
    assert artifact["gated_inputs_present"] is False
    assert "missing_exp1481_semantic_energy_retirement" in artifact["blockers"]
    assert "missing_exp1487_v1_pairwise_retirement" in artifact["blockers"]


def test_req_verify_1500_policy_markdown_has_required_tables() -> None:
    """REQ-VERIFY-1500: policy note renders headline, auxiliary, triage, retired tables."""

    artifact = build_discipline_gate_artifact(
        exp1499=_exp1499_with_matrix(),
        exp1481={"status": "complete", "claim_allowed": False},
        exp1487={"status": "complete", "improvement_allowed": False},
        ops_note_path="ops/latent_deterministic_discipline_gate_1500.md",
    )

    markdown = render_policy_markdown(artifact)

    assert "Spec: REQ-VERIFY-1500, SCENARIO-VERIFY-1500." in markdown
    assert "## Headline Evidence" in markdown
    assert "## Auxiliary Ranking Evidence" in markdown
    assert "## Triage Evidence" in markdown
    assert "## Retired / No-Claim Evidence" in markdown
    assert "deterministic validator comparison" in markdown
    assert "superficial-baseline comparison" in markdown
    assert "held-out calibration" in markdown
    assert "false-accept accounting" in markdown


def test_req_verify_1500_schema_validation_rejects_bad_artifacts() -> None:
    """REQ-VERIFY-1500: schema guard rejects missing fields and bad verdict prefixes."""

    with pytest.raises(ValueError, match="missing required artifact fields"):
        _validate_required_fields({"honest_verdict": "complete: incomplete"})

    artifact = build_discipline_gate_artifact(
        exp1499=_exp1499_with_matrix(),
        exp1481={"status": "complete", "claim_allowed": False},
        exp1487={"status": "complete", "improvement_allowed": False},
        ops_note_path="ops/latent_deterministic_discipline_gate_1500.md",
    )
    artifact["honest_verdict"] = "blocked_without_allowed_prefix"

    with pytest.raises(ValueError, match="disallowed prefix"):
        _validate_required_fields(artifact)


def _exp1499_with_matrix() -> dict[str, object]:
    return {
        "status": "complete",
        "orthogonality_matrix_written": True,
        "conditional_acceptance_matrix": {
            "labels": ["cctu_full_executable_verifier", "beaver_lite_bound"]
        },
        "deterministic_first_recommendations": [
            "Gate generation first on executable deterministic validators."
        ],
    }
