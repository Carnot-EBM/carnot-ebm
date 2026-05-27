"""Tests for Exp 3198 repair-gate decision v5.

Spec refs: REQ-VERIFY-3198, SCENARIO-VERIFY-3198.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import repair_gate_decision_v5 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "source_artifacts",
    "receipt_gate_state",
    "clean_verifier_state",
    "adaptive_policy_state",
    "domain_preview_state",
    "invariant_certificate_state",
    "repair_gate_state",
    "repair_allowed_scope",
    "blocker_reasons",
    "downstream_gated_skip_expected",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md before changes.\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nWrite Tests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("Adversarial Artifact Verification\n", encoding="utf-8")
    spec = root / mod.SPEC_REL_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-VERIFY-3198\nSCENARIO-VERIFY-3198\n"
        "results/experiment_3198_repair_gate_decision_v5.json\n",
        encoding="utf-8",
    )


def _write_standard_sources(
    root: Path,
    *,
    receipt_contract: dict[str, Any] | None = None,
    offload_probe: dict[str, Any] | None = None,
    clean_verifier: dict[str, Any] | None = None,
    adaptive_policy: dict[str, Any] | None = None,
    domain_preview: dict[str, Any] | None = None,
    invariant_certificate: dict[str, Any] | None = None,
    include_clean_verifier: bool = True,
) -> None:
    _write_text_sources(root)
    contract_payload = {
        "experiment_id": "exp3192",
        "receipt_adversarial_contract_v4_ready": True,
        "current_evidence_assessment": {
            "proof_execution_sufficient": True,
            "clean_rerun_allowed": True,
            "headline_claim_allowed": True,
            "substrate_classification": "full_local_sota_receipt",
            "proof_receipt_count": 2,
        },
        "source_errors": [],
        "honest_verdict": "complete: contract fixture",
    }
    if receipt_contract:
        contract_payload.update(receipt_contract)
    _write_json(root, mod.EXP3192_REL_PATH, contract_payload)

    probe_payload = {
        "experiment_id": "exp3193",
        "substrate_classification": "full_local_sota_receipt",
        "clean_rerun_allowed": True,
        "headline_claim_allowed": True,
        "receipt_count": 1,
        "flagged_adversarial": False,
        "blocker_reasons": [],
        "honest_verdict": "complete: offload fixture",
    }
    if offload_probe:
        probe_payload.update(offload_probe)
    _write_json(root, mod.EXP3193_REL_PATH, probe_payload)

    clean_payload = {
        "experiment_id": "exp3194",
        "clean_live_sota_verifier_rerun_v11_ready": True,
        "status": "complete",
        "gated_skip": False,
        "metrics_computed": True,
        "live_call_count": 6,
        "flagged_adversarial": False,
        "headline_claim_allowed": True,
        "false_accept_rate": 0.0,
        "known_false_accepts_accepted": [],
        "honest_verdict": "complete: clean verifier fixture",
    }
    if clean_verifier:
        clean_payload.update(clean_verifier)
    if include_clean_verifier:
        _write_json(root, mod.EXP3194_REL_PATH, clean_payload)

    policy_payload = {
        "experiment_id": "exp3195",
        "adaptive_verification_granularity_policy_v1_ready": True,
        "exact_rows_used": 72,
        "false_accept_risk_increase": 0.0,
        "source_errors": [],
        "honest_verdict": "complete: adaptive policy fixture",
    }
    if adaptive_policy:
        policy_payload.update(adaptive_policy)
    _write_json(root, mod.EXP3195_REL_PATH, policy_payload)

    preview_payload = {
        "experiment_id": "exp3196",
        "preview_domain_count": 5,
        "average_candidate_domain_size": 1.2,
        "source_errors": [],
        "preview_manifest": [{"row_id": f"row-{idx}"} for idx in range(5)],
        "repair_call_ready": False,
        "promotion_allowed": False,
        "honest_verdict": "complete: domain preview fixture",
    }
    if domain_preview:
        preview_payload.update(domain_preview)
    _write_json(root, mod.EXP3196_REL_PATH, preview_payload)

    invariant_payload = {
        "experiment_id": "exp3197",
        "invariant_record_count": 3,
        "exact_guard_count": 3,
        "anti_overfit_test_count": 3,
        "linked_domain_preview_count": 3,
        "source_errors": [],
        "invariant_records": [
            {"row_id": f"row-{idx}", "linked_domain_preview_record": f"row-{idx}"}
            for idx in range(3)
        ],
        "repair_call_ready": False,
        "honest_verdict": "complete: invariant fixture",
    }
    if invariant_certificate:
        invariant_payload.update(invariant_certificate)
    _write_json(root, mod.EXP3197_REL_PATH, invariant_payload)


def test_req_verify_3198_spec_anchor_exists() -> None:
    """REQ-VERIFY-3198: OpenSpec declares the v5 repair gate artifact."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3198" in spec
    assert "SCENARIO-VERIFY-3198" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "repair_gate_state" in spec


def test_scenario_verify_3198_current_blocked_evidence_is_machine_readable(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3198: skipped and flagged upstreams force downstream skip."""

    _write_standard_sources(
        tmp_path,
        receipt_contract={
            "current_evidence_assessment": {
                "proof_execution_sufficient": True,
                "clean_rerun_allowed": False,
                "headline_claim_allowed": False,
                "substrate_classification": "cpu_fallback_receipt_only",
                "proof_receipt_count": 2,
            }
        },
        offload_probe={
            "substrate_classification": "cuda_unavailable",
            "clean_rerun_allowed": False,
            "headline_claim_allowed": False,
            "receipt_count": 0,
            "flagged_adversarial": True,
            "blocker_reasons": ["selected Python torch.cuda.is_available() is false"],
        },
        clean_verifier={
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp3193.clean_rerun_allowed actual=False expected=True"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp3193",
                    "artifact_field": "clean_rerun_allowed",
                    "expected": True,
                    "actual": False,
                    "passed": False,
                }
            ],
        },
    )

    artifact = mod.build_artifact(tmp_path, tests_run=["focused"])

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3198"
    assert artifact["receipt_gate_state"] == "blocked_cuda_unavailable"
    assert artifact["clean_verifier_state"] == "blocked_gate_skipped_conductor_pre_gate"
    assert artifact["adaptive_policy_state"] == "ready_adaptive_schedule"
    assert artifact["domain_preview_state"] == "ready_bounded_domain_preview"
    assert artifact["invariant_certificate_state"] == "ready_exact_guard_coverage"
    assert artifact["repair_gate_state"] == "blocked_clean_verifier_gate_skipped"
    assert artifact["repair_allowed_scope"] is None
    assert artifact["downstream_gated_skip_expected"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == ["focused"]

    codes = {row["code"] for row in artifact["blocker_reasons"]}
    assert "exp3192_current_evidence_cpu_fallback_only" in codes
    assert "exp3193_clean_rerun_not_allowed" in codes
    assert "exp3193_adversarially_flagged" in codes
    assert "exp3194_gate_skipped" in codes
    assert len([row for row in artifact["source_artifacts"] if row["required"]]) == 10


def test_scenario_verify_3198_unblocks_only_with_bounded_scope(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3198: clean upstreams open only a finite repair scope."""

    _write_standard_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        tests_run=["focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert artifact["repair_gate_state"] == "unblocked_for_bounded_repair_ladder"
    assert artifact["blocker_reasons"] == []
    assert artifact["downstream_gated_skip_expected"] is False
    assert artifact["repair_allowed_scope"] == {
        "enabled": True,
        "source_domain_preview_artifact": mod.EXP3196_REL_PATH.as_posix(),
        "source_invariant_certificate_artifact": mod.EXP3197_REL_PATH.as_posix(),
        "row_selection": "intersection_of_preview_domains_and_invariant_guards",
        "max_distinct_rows": 2,
        "max_attempts_per_row": 2,
        "max_total_repair_attempts": 4,
        "requires_mandated_local_sota": True,
        "requires_clean_verifier_unflagged": True,
        "requires_exact_authority_acceptance": True,
        "requires_anti_overfit_guard": True,
        "no_headline_claim_from_gate_alone": True,
        "allowed_row_budget_source": {
            "preview_domain_count": 5,
            "invariant_record_count": 3,
        },
    }
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_3198_missing_required_upstream_fails_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3198: missing required upstreams produce a precise blocked state."""

    _write_standard_sources(tmp_path, include_clean_verifier=False)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["clean_verifier_state"] == "missing_clean_verifier_artifact"
    assert artifact["repair_gate_state"] == "blocked_missing_required_upstream"
    assert artifact["repair_allowed_scope"] is None
    assert artifact["downstream_gated_skip_expected"] is True
    assert any(
        row["code"] == "missing_required_upstream"
        and row["source_artifact"] == mod.EXP3194_REL_PATH.as_posix()
        for row in artifact["blocker_reasons"]
    )


@pytest.mark.parametrize(
    ("updates", "expected_state", "expected_code"),
    [
        (
            {"adaptive_policy": {"false_accept_risk_increase": 0.1}},
            "blocked_adaptive_policy_not_ready",
            "exp3195_false_accept_risk_increase",
        ),
        (
            {"domain_preview": {"preview_domain_count": 0}},
            "blocked_domain_preview_not_ready",
            "exp3196_preview_domain_missing",
        ),
        (
            {"invariant_certificate": {"exact_guard_count": 1}},
            "blocked_invariant_certificate_not_ready",
            "exp3197_exact_guard_coverage_insufficient",
        ),
    ],
)
def test_req_verify_3198_specific_non_verifier_blockers(
    tmp_path: Path,
    updates: dict[str, dict[str, Any]],
    expected_state: str,
    expected_code: str,
) -> None:
    """REQ-VERIFY-3198: each later readiness layer maps to a concrete state."""

    _write_standard_sources(tmp_path, **updates)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_gate_state"] == expected_state
    assert any(row["code"] == expected_code for row in artifact["blocker_reasons"])
    assert artifact["downstream_gated_skip_expected"] is True


def test_req_verify_3198_validation_rejects_unsafe_shapes() -> None:
    """REQ-VERIFY-3198: validation rejects contradictory terminal artifacts."""

    artifact = {
        "schema_version": mod.SCHEMA_VERSION,
        "experiment_id": "exp3198",
        "source_artifacts": [],
        "receipt_gate_state": "eligible_full_local_sota_receipt",
        "clean_verifier_state": "eligible_unflagged_clean_verifier",
        "adaptive_policy_state": "ready_adaptive_schedule",
        "domain_preview_state": "ready_bounded_domain_preview",
        "invariant_certificate_state": "ready_exact_guard_coverage",
        "repair_gate_state": "unblocked_for_bounded_repair_ladder",
        "repair_allowed_scope": {
            "enabled": True,
            "max_total_repair_attempts": 1,
            "max_attempts_per_row": 1,
            "max_distinct_rows": 1,
        },
        "blocker_reasons": [],
        "downstream_gated_skip_expected": False,
        "honest_verdict": "complete: valid",
        "inference_substrate": {"live_model_calls": 0, "repair_calls": 0},
    }

    mod.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({k: v for k, v in artifact.items() if k != "honest_verdict"})
    with pytest.raises(ValueError, match="allowed repair gate state"):
        mod.validate_artifact({**artifact, "repair_gate_state": "maybe"})
    with pytest.raises(ValueError, match="blocker reasons"):
        mod.validate_artifact({**artifact, "blocker_reasons": [{"code": "blocked"}]})
    with pytest.raises(ValueError, match="repair scope"):
        mod.validate_artifact({**artifact, "repair_allowed_scope": None})
    with pytest.raises(ValueError, match="downstream skip"):
        mod.validate_artifact({**artifact, "downstream_gated_skip_expected": True})
    with pytest.raises(ValueError, match="must not perform live"):
        mod.validate_artifact({**artifact, "inference_substrate": {"live_model_calls": 1}})
    with pytest.raises(ValueError, match="complete:"):
        mod.validate_artifact({**artifact, "honest_verdict": "blocked"})

    blocked = {
        **artifact,
        "repair_gate_state": "blocked_receipt_precondition",
        "repair_allowed_scope": None,
        "blocker_reasons": [{"code": "blocked"}],
        "downstream_gated_skip_expected": True,
        "honest_verdict": "complete: blocked",
    }
    mod.validate_artifact(blocked)
    with pytest.raises(ValueError, match="repair scope"):
        mod.validate_artifact({**blocked, "repair_allowed_scope": {"enabled": False}})
    with pytest.raises(ValueError, match="blocker reasons"):
        mod.validate_artifact({**blocked, "blocker_reasons": []})
    with pytest.raises(ValueError, match="downstream skip"):
        mod.validate_artifact({**blocked, "downstream_gated_skip_expected": False})


def test_req_verify_3198_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3198: malformed inputs normalize to blocked, non-authoritative evidence."""

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad json", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.finite_rate("not a rate") is None
    assert mod.finite_rate(float("inf")) is None
    assert mod.finite_rate(1.2) is None
    assert mod.finite_rate(0.05) == pytest.approx(0.05)
    assert mod.finite_nonnegative_number("bad") is None
    assert mod.finite_nonnegative_number(float("inf")) is None
    assert mod.finite_nonnegative_number(-1.0) is None
    assert mod.finite_nonnegative_number(2.5) == pytest.approx(2.5)
    assert mod.int_value("3") == 0
    assert mod.int_value(3) == 3


def test_req_verify_3198_state_helpers_cover_fail_closed_edges() -> None:
    """REQ-VERIFY-3198: helper states keep each malformed layer blocked."""

    assert mod.receipt_gate_state({}, {}) == "missing_receipt_contract_artifact"
    assert mod.receipt_gate_state({"receipt_adversarial_contract_v4_ready": True}, {}) == (
        "missing_offload_probe_artifact"
    )
    assert mod.clean_verifier_state({"schema": "blocked_gate_check_v1"}) == (
        "blocked_clean_verifier_artifact"
    )
    assert mod.clean_verifier_state({"gated_skip": True}) == "blocked_clean_verifier_gated_skip"
    assert mod.clean_verifier_state({"flagged_adversarial": True}) == (
        "blocked_clean_verifier_adversarially_flagged"
    )
    assert mod.clean_verifier_state({"clean_live_sota_verifier_rerun_v11_ready": True}) == (
        "blocked_clean_verifier_not_eligible"
    )
    assert mod.adaptive_policy_state({}) == "missing_adaptive_policy_artifact"
    assert mod.adaptive_policy_state({"adaptive_verification_granularity_policy_v1_ready": False}) == (
        "blocked_adaptive_policy_not_ready"
    )
    assert mod.adaptive_policy_state(
        {"adaptive_verification_granularity_policy_v1_ready": True, "source_errors": ["x"]}
    ) == "blocked_adaptive_policy_source_errors"
    assert mod.adaptive_policy_state(
        {
            "adaptive_verification_granularity_policy_v1_ready": True,
            "exact_rows_used": 0,
            "false_accept_risk_increase": 0.0,
        }
    ) == "blocked_adaptive_policy_no_exact_rows"
    assert mod.domain_preview_state({}) == "missing_domain_preview_artifact"
    assert mod.domain_preview_state({"source_errors": ["x"]}) == "blocked_domain_preview_source_errors"
    assert mod.domain_preview_state(
        {"preview_domain_count": 1, "average_candidate_domain_size": 0.0}
    ) == "blocked_domain_preview_unbounded"
    assert mod.invariant_certificate_state({}) == "missing_invariant_certificate_artifact"
    assert mod.invariant_certificate_state({"source_errors": ["x"]}) == (
        "blocked_invariant_certificate_source_errors"
    )
    assert mod.invariant_certificate_state({"invariant_record_count": 0}) == (
        "blocked_invariant_certificate_empty"
    )
    assert mod.invariant_certificate_state(
        {
            "invariant_record_count": 2,
            "exact_guard_count": 2,
            "anti_overfit_test_count": 1,
            "linked_domain_preview_count": 2,
        }
    ) == "blocked_invariant_certificate_anti_overfit_gap"
    assert mod.invariant_certificate_state(
        {
            "invariant_record_count": 2,
            "exact_guard_count": 2,
            "anti_overfit_test_count": 2,
            "linked_domain_preview_count": 0,
        }
    ) == "blocked_invariant_certificate_no_domain_links"


def test_req_verify_3198_blocker_helpers_cover_all_machine_codes() -> None:
    """REQ-VERIFY-3198: blocker helpers emit stable codes for all blocked inputs."""

    receipt_rows = mod.receipt_blockers(
        {
            "receipt_adversarial_contract_v4_ready": False,
            "current_evidence_assessment": {
                "clean_rerun_allowed": False,
                "substrate_classification": "cpu_fallback_receipt_only",
            },
        },
        {
            "clean_rerun_allowed": False,
            "headline_claim_allowed": False,
            "substrate_classification": "cpu_fallback_receipt_only",
            "receipt_count": 0,
            "flagged_adversarial": True,
        },
    )
    assert {
        "exp3192_contract_not_ready",
        "exp3192_current_evidence_cpu_fallback_only",
        "exp3192_clean_rerun_not_allowed",
        "exp3193_clean_rerun_not_allowed",
        "exp3193_headline_claim_not_allowed",
        "exp3193_nonfull_substrate",
        "exp3193_no_offload_receipt",
        "exp3193_adversarially_flagged",
    } <= {row["code"] for row in receipt_rows}

    clean_rows = mod.clean_verifier_blockers(
        {
            "gated_skip": True,
            "flagged_adversarial": True,
            "clean_live_sota_verifier_rerun_v11_ready": False,
            "metrics_computed": False,
            "headline_claim_allowed": False,
            "false_accept_rate": 0.5,
            "known_false_accepts_accepted": ["row-1"],
        }
    )
    assert {
        "exp3194_gated_skip",
        "exp3194_adversarially_flagged",
        "exp3194_not_ready",
        "exp3194_metrics_not_computed",
        "exp3194_headline_claim_not_allowed",
        "exp3194_false_accept_gate_failed",
        "exp3194_known_false_accepts_accepted",
    } <= {row["code"] for row in clean_rows}

    policy_rows = mod.adaptive_policy_blockers(
        {
            "adaptive_verification_granularity_policy_v1_ready": False,
            "source_errors": ["missing"],
            "exact_rows_used": 0,
            "false_accept_risk_increase": 0.25,
        }
    )
    assert {
        "exp3195_policy_not_ready",
        "exp3195_source_errors",
        "exp3195_no_exact_rows",
        "exp3195_false_accept_risk_increase",
    } <= {row["code"] for row in policy_rows}

    domain_rows = mod.domain_preview_blockers(
        {"source_errors": ["missing"], "preview_domain_count": 0, "average_candidate_domain_size": 0}
    )
    assert {
        "exp3196_source_errors",
        "exp3196_preview_domain_missing",
        "exp3196_average_domain_size_invalid",
    } <= {row["code"] for row in domain_rows}

    invariant_rows = mod.invariant_certificate_blockers(
        {
            "source_errors": ["missing"],
            "invariant_record_count": 2,
            "exact_guard_count": 1,
            "anti_overfit_test_count": 1,
            "linked_domain_preview_count": 0,
        }
    )
    assert {
        "exp3197_source_errors",
        "exp3197_exact_guard_coverage_insufficient",
        "exp3197_anti_overfit_coverage_insufficient",
        "exp3197_domain_links_missing",
    } <= {row["code"] for row in invariant_rows}
    assert {
        row["code"]
        for row in mod.invariant_certificate_blockers(
            {
                "invariant_record_count": 0,
                "exact_guard_count": 0,
                "anti_overfit_test_count": 0,
                "linked_domain_preview_count": 0,
            }
        )
    } == {"exp3197_invariant_records_missing", "exp3197_domain_links_missing"}

    assert mod.receipt_blockers({}, {}) == []
    assert mod.clean_verifier_blockers({}) == []
    assert mod.adaptive_policy_blockers({}) == []
    assert mod.domain_preview_blockers({}) == []
    assert mod.invariant_certificate_blockers({}) == []


def test_req_verify_3198_repair_state_priority_edges() -> None:
    """REQ-VERIFY-3198: repair state priority gives precise downstream gates."""

    ready = {
        "receipt_state": "eligible_full_local_sota_receipt",
        "clean_verifier_state": "eligible_unflagged_clean_verifier",
        "adaptive_policy_state": "ready_adaptive_schedule",
        "domain_preview_state": "ready_bounded_domain_preview",
        "invariant_certificate_state": "ready_exact_guard_coverage",
    }

    assert mod.repair_gate_state(
        **ready,
        blockers=[{"code": "exp3193_adversarially_flagged"}],
    ) == "blocked_upstream_adversarially_flagged"
    assert mod.repair_gate_state(
        **{**ready, "receipt_state": "blocked_cuda_unavailable"},
        blockers=[],
    ) == "blocked_receipt_precondition"
    assert mod.repair_gate_state(
        **{**ready, "clean_verifier_state": "blocked_clean_verifier_not_eligible"},
        blockers=[],
    ) == "blocked_clean_verifier_not_eligible"
    assert mod.repair_gate_state(**ready, blockers=[{"code": "unexpected"}]) == (
        "blocked_other_precondition"
    )
