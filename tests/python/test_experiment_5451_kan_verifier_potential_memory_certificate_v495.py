"""Tests for Exp5451 verifier-potential/governed-memory KAN certificate.

Spec refs: REQ-KAN-5451, SCENARIO-KAN-5451.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5451_kan_verifier_potential_memory_certificate_v495 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_kan_5451_spec_declares_bounded_certificate_contract() -> None:
    """REQ-KAN-5451: OpenSpec anchors the V495 certificate boundary."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-KAN-5451") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-KAN-5451",
        "SCENARIO-KAN-5451",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp5443 reports `verifier_potential_fixture_ready=true`",
        "Exp5446 reports `governed_csl_loop_ready=true`",
        "unsupported hardware speedup claims",
        "unsupported token or internal-state claims",
        "broad KAN soundness claims",
        "`bounded_measurement_access_certificate`",
        "`scripts/research_conductor.py`",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f"{field}`: {principle}" in normalized


def test_req_kan_5451_preconditions_and_claim_set_cover_required_families() -> None:
    """REQ-KAN-5451: upstream gates and claim families are checked first."""

    upstreams = exp.load_upstream_artifacts()
    claims = exp.build_claim_set()
    evaluated = exp.evaluate_claims(upstreams, claims)

    assert exp.upstream_gates_ready(upstreams) is True
    assert upstreams["exp5443"]["verifier_potential_fixture_ready"] is True
    assert upstreams["exp5446"]["governed_csl_loop_ready"] is True
    assert {claim["claim_kind"] for claim in claims} == {
        "true_measured",
        "false_property",
        "unsupported",
        "broad_soundness",
    }
    assert {claim["claim_domain"] for claim in claims} >= {
        "verifier_potential",
        "governed_memory",
        "hardware_speedup",
        "token_internal",
        "kan_soundness",
    }
    assert len(claims) == len(evaluated) == 15
    assert all(row["rejected"] or row["preserved"] for row in evaluated)


def test_scenario_kan_5451_preserves_true_measured_claims() -> None:
    """SCENARIO-KAN-5451: true measured artifact properties stay certified."""

    diagnostic = exp.evaluate_certificate()
    true_claims = [
        row for row in diagnostic["claim_records"] if row["claim_kind"] == "true_measured"
    ]

    assert diagnostic["true_measured_claim_preservation_rate"] == pytest.approx(1.0)
    assert {row["claim_id"] for row in true_claims} == {
        "true_exp5443_fixture_ready",
        "true_exp5443_exact_final_authority",
        "true_exp5443_prefix_final_disagreements_measured",
        "true_exp5446_governed_loop_ready",
        "true_exp5446_no_weight_mutation",
        "true_exp5446_zero_unsafe_false_accepts",
    }
    assert all(row["classification"] == "measured_supported" for row in true_claims)
    assert all(row["preserved"] is True for row in true_claims)
    assert all(row["missing_evidence"] == [] for row in true_claims)


def test_scenario_kan_5451_rejects_false_verifier_and_memory_properties() -> None:
    """SCENARIO-KAN-5451: false properties are rejected from measured fields."""

    diagnostic = exp.evaluate_certificate()
    false_claims = [
        row for row in diagnostic["claim_records"] if row["claim_kind"] == "false_property"
    ]

    assert diagnostic["false_property_rejection_rate"] == pytest.approx(1.0)
    assert {row["claim_id"] for row in false_claims} == {
        "false_exp5443_no_prefix_final_disagreements",
        "false_exp5443_metric_independence_failed",
        "false_exp5446_ungated_memory_safe",
        "false_exp5446_all_memories_active_for_routing",
        "false_exp5446_model_weights_mutated",
    }
    assert all(row["classification"] == "measured_contradicted" for row in false_claims)
    assert all(row["rejected"] is True for row in false_claims)
    assert any(
        row["measured_values"]["exp5443.prefix_final_disagreement_cases"] == 6
        for row in false_claims
    )
    assert any(
        row["measured_values"].get("exp5446.control_metrics.ungated_memory.unsafe_false_accepts")
        == 3
        for row in false_claims
    )


def test_scenario_kan_5451_rejects_unsupported_and_broad_claims() -> None:
    """SCENARIO-KAN-5451: absent fields reject unsupported and broad claims."""

    diagnostic = exp.evaluate_certificate()
    unsupported = [
        row
        for row in diagnostic["claim_records"]
        if row["claim_kind"] in {"unsupported", "broad_soundness"}
    ]

    assert diagnostic["unsupported_claim_rejection_rate"] == pytest.approx(1.0)
    assert diagnostic["hardware_speedup_claim_rejected"] is True
    assert diagnostic["token_internal_claim_rejected"] is True
    assert diagnostic["broad_kan_claim_made"] is False
    assert {row["claim_id"] for row in unsupported} == {
        "unsupported_hardware_speedup_from_certificate",
        "unsupported_token_level_access_from_certificate",
        "unsupported_internal_state_access_from_certificate",
        "unsupported_broad_kan_soundness",
    }
    assert all(row["classification"] == "missing_evidence_unsupported" for row in unsupported)
    assert all(row["rejected"] is True for row in unsupported)
    assert all(row["missing_evidence"] for row in unsupported)
    assert any(
        "exp5446.authenticated_hardware_speedup" in row["missing_evidence"] for row in unsupported
    )


def test_req_kan_5451_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-KAN-5451: run() writes the required terminal artifact."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=exp.default_tests_run())

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["gated_upstreams_ready"] is True
    assert artifact["claim_count"] == len(artifact["claim_records"]) == 15
    assert artifact["true_measured_claim_preservation_rate"] == pytest.approx(1.0)
    assert artifact["false_property_rejection_rate"] == pytest.approx(1.0)
    assert artifact["unsupported_claim_rejection_rate"] == pytest.approx(1.0)
    assert artifact["verifier_potential_claims_checked"] == 5
    assert artifact["governed_memory_claims_checked"] == 6
    assert artifact["hardware_speedup_claim_rejected"] is True
    assert artifact["token_internal_claim_rejected"] is True
    assert artifact["broad_kan_claim_made"] is False
    assert artifact["certificate_checksum"].startswith("sha256:")
    assert artifact["kan_certificate_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == exp.default_tests_run()
    exp.validate_artifact(artifact)


def test_req_kan_5451_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-KAN-5451: checked-in deliverable is stable under deterministic replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=checked_in["tests_run"])

    assert checked_in == replay
    assert checked_in["kan_certificate_ready"] is True
    assert checked_in["broad_kan_claim_made"] is False
    exp.validate_artifact(checked_in)


def test_req_kan_5451_validation_rejects_claim_drift() -> None:
    """REQ-KAN-5451: validation fails closed on unsupported or broad drift."""

    artifact = exp.build_artifact(tests_run=exp.default_tests_run())
    blocked = exp.build_artifact(tests_run=[])

    assert blocked["status"] == "blocked"
    assert blocked["kan_certificate_ready"] is False
    assert blocked["honest_verdict"].startswith("blocked:")
    exp.validate_artifact(blocked)

    missing = deepcopy(artifact)
    missing.pop("claim_count")
    with pytest.raises(ValueError, match="claim_count"):
        exp.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    broad_claim = deepcopy(artifact)
    broad_claim["broad_kan_claim_made"] = True
    with pytest.raises(ValueError, match="broad_kan_claim_made"):
        exp.validate_artifact(broad_claim)

    broad_verdict = deepcopy(artifact)
    broad_verdict["honest_verdict"] = "complete: broad KAN soundness proved"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(broad_verdict)

    unsupported_accepted = deepcopy(artifact)
    unsupported_accepted["claim_records"][-1]["rejected"] = False
    with pytest.raises(ValueError, match="claim_records"):
        exp.validate_artifact(unsupported_accepted)

    false_accepted = deepcopy(artifact)
    false_accepted["claim_records"][6]["rejected"] = False
    with pytest.raises(ValueError, match="claim_records"):
        exp.validate_artifact(false_accepted)

    true_rejected = deepcopy(artifact)
    true_rejected["claim_records"][0]["preserved"] = False
    with pytest.raises(ValueError, match="claim_records"):
        exp.validate_artifact(true_rejected)

    bad_checksum = deepcopy(artifact)
    bad_checksum["certificate_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="certificate_checksum"):
        exp.validate_artifact(bad_checksum)


def test_req_kan_5451_upstream_gate_blocks_certificate() -> None:
    """REQ-KAN-5451: failed upstream gates produce a blocked certificate."""

    upstreams = exp.load_upstream_artifacts()
    upstreams["exp5443"] = upstreams["exp5443"] | {"verifier_potential_fixture_ready": False}
    artifact = exp.build_artifact(upstreams=upstreams, tests_run=exp.default_tests_run())

    assert artifact["gated_upstreams_ready"] is False
    assert artifact["kan_certificate_ready"] is False
    assert artifact["status"] == "blocked"
    assert "exp5443_verifier_potential_fixture_not_ready" in artifact["readiness_blockers"]
    assert artifact["honest_verdict"].startswith("blocked:")
    exp.validate_artifact(artifact)


def test_req_kan_5451_defensive_helper_edges(tmp_path: Path) -> None:
    """REQ-KAN-5451: defensive helper branches remain deterministic."""

    upstreams = exp.load_upstream_artifacts()
    upstreams["exp5446"] = upstreams["exp5446"] | {"governed_csl_loop_ready": False}
    diagnostic = exp.evaluate_certificate(upstreams=upstreams)
    blockers = exp._readiness_blockers(diagnostic, "bad", [])

    assert "exp5446_governed_csl_loop_not_ready" in blockers
    assert "true_measured_claim_preservation" in blockers
    assert "certificate_checksum" in blockers
    assert "tests_recorded" in blockers

    broken = exp.evaluate_certificate()
    broken["false_property_rejection_rate"] = 0.0
    broken["unsupported_claim_rejection_rate"] = 0.0
    broken["hardware_speedup_claim_rejected"] = False
    broken["token_internal_claim_rejected"] = False
    broken["broad_kan_claim_made"] = True
    broken["claim_limits"] = []
    edge_blockers = exp._readiness_blockers(
        broken,
        "bad",
        [{"command": "edge", "outcome": "passed"}],
    )

    assert "false_property_rejection" in edge_blockers
    assert "unsupported_claim_rejection" in edge_blockers
    assert "hardware_speedup_claim_rejected" in edge_blockers
    assert "token_internal_claim_rejected" in edge_blockers
    assert "broad_kan_claim_made" in edge_blockers
    assert "certificate_checksum" in edge_blockers
    assert "claim_limits" in edge_blockers

    with pytest.raises(ValueError, match="unsupported claim check op"):
        exp._check(
            exp.load_upstream_artifacts(),
            {"path": "exp5443.fixture_count", "op": "bad", "value": 8},
        )
    assert exp._verdict_is_bounded("done") is False
    assert exp._sha256_if_exists(tmp_path / "missing.json") is None
