"""Tests for Exp6309 V543 adversarial capstone.

Spec refs: REQ-INTEG-6309, SCENARIO-INTEG-6309-1,
SCENARIO-INTEG-6309-2, SCENARIO-INTEG-6309-3,
SCENARIO-INTEG-6309-4, SCENARIO-INTEG-6309-5,
SCENARIO-INTEG-6309-6.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_6309_v543_adversarial_capstone as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _write_artifact(
    path: Path,
    status: str = "complete",
    verdict: str = "complete: fixture",
    **extra: object,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "honest_verdict": verdict,
        "duration_s": 1.0,
        "inference_substrate": "aggregation_from_exact_declared_artifacts",
        "verifier_is_oracle": False,
        "preconditions_checked": {"fixture": True},
        "field_principles": {"status": "fixture"},
        "field_provenance": {"status": {"sources": ["fixture"]}},
        "test_commands": ["fixture"],
        "test_exit_codes": {"fixture": 0},
        "random_seed": 6309,
        "reproducibility_checksum": "sha256:fixture",
        **extra,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_req_integ_6309_spec_declares_fields_and_scenarios() -> None:
    """REQ-INTEG-6309: OpenSpec records the capstone contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INTEG-6309") :]

    for token in (
        "REQ-INTEG-6309",
        "SCENARIO-INTEG-6309-1",
        "SCENARIO-INTEG-6309-2",
        "SCENARIO-INTEG-6309-3",
        "SCENARIO-INTEG-6309-4",
        "SCENARIO-INTEG-6309-5",
        "SCENARIO-INTEG-6309-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.VERIFIER_ORACLE_BOUNDARY,
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_exact_declared_matrix_preserves_missing_and_aliases(tmp_path: Path) -> None:
    """SCENARIO-INTEG-6309-1: exact paths outrank aliases and receipts."""

    declared = tmp_path / "results/experiment_6300_declared.json"
    alias = tmp_path / "results/experiment_6303_alias.json"
    _write_artifact(declared, flagged_adversarial=True)
    _write_artifact(alias)
    tasks = [
        {
            "id": "exp6300-three-family-universal-activation-bus",
            "title": "Bus fixture",
            "track": "shared_state",
            "deliverable": "results/experiment_6300_declared.json",
        },
        {
            "id": "exp6303-live-three-family-shared-state-benchmark",
            "title": "Missing exact fixture",
            "track": "shared_state",
            "deliverable": "results/experiment_6303_declared_missing.json",
        },
    ]
    receipts = {
        "exp6303-live-three-family-shared-state-benchmark": {"status": "OK"},
    }

    matrix = mod.build_exact_declared_task_artifact_matrix(
        tmp_path,
        tasks,
        conductor_receipts=receipts,
    )

    flagged = matrix["exp6300-three-family-universal-activation-bus"]
    assert flagged["terminal_class"] == "flagged"
    assert flagged["flagged_adversarial_stamped"] is True
    assert flagged["same_number_alias_used"] is False

    missing = matrix["exp6303-live-three-family-shared-state-benchmark"]
    assert missing["terminal_class"] == "missing"
    assert missing["terminal"] is False
    assert missing["receipt_override_attempted"] is True
    assert missing["receipt_overrode"] is False
    assert missing["same_number_alias_candidates_ignored"] == ["results/experiment_6303_alias.json"]


def test_report_preserves_current_rule_flags_counts_and_branch_ledgers() -> None:
    """SCENARIO-INTEG-6309-2 and SCENARIO-INTEG-6309-4: branches stay independent."""

    report = mod.build_report(
        REPO,
        date="20260811",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    assert mod.validate_report(report) == []
    counts = report[
        "missing_nonterminal_blocked_skipped_null_flagged_retired_ready_oracle_only_replay_only_safety_only_and_unlicensed_counts"
    ]
    assert counts["task_count"] == 13
    assert counts["terminal_class_task_count_sum"] == 13
    assert counts["missing"] == 1
    assert counts["raw_blocked_status"] == 2
    assert counts["flagged"] >= 2
    assert counts["oracle_only"] >= 2
    assert counts["replay_only"] >= 1
    assert counts["safety_only"] == 1
    assert counts["unlicensed_transfer"] == 1

    reviews = report["current_rule_adversarial_results_by_task"]
    bus = reviews["exp6300-three-family-universal-activation-bus"]
    assert bus["stamped_flagged_adversarial"] is True
    assert bus["current_rule_critical_flag_count"] >= 1
    assert any(flag["kind"] == "DURATION_TOO_SHORT" for flag in bus["current_rule_flags"])
    missing = reviews["exp6303-live-three-family-shared-state-benchmark"]
    assert missing["present"] is False
    assert missing["current_rule_critical_flag_count"] >= 1

    ledger = report["branch_independent_promotion_ledger"]
    assert ledger["infrastructure_source_integrity"]["promotion_allowed"] is True
    assert ledger["shared_model_state_to_exact_energy"]["promotion_allowed"] is False
    assert ledger["continuous_online_learning_and_licensed_transfer"]["promotion_allowed"] is False
    assert ledger["arc_target_validated_routing"]["route_audit_promotion_allowed"] is True
    assert ledger["arc_target_validated_routing"]["solve_claim_allowed"] is False

    assert report["shared_activation_bus_verdict"]["promotion_allowed"] is False
    assert report["shared_state_initializer_verdict"]["promotion_allowed"] is False
    assert report["live_three_family_value_verdict"]["terminal_class"] == "missing"
    assert report["continuous_self_learning_verdict"]["utility_ready_score"] == 1.0
    assert report["online_learning_safety_verdict"]["safety_only"] is True
    assert report["evidence_licensed_transfer_verdict"]["promotion_allowed"] is False
    assert report["arc_target_validation_verdict"]["solve_claim_allowed"] is False


def test_structured_gates_retirement_and_claim_boundaries() -> None:
    """SCENARIO-INTEG-6309-3 and SCENARIO-INTEG-6309-5: gates and boundaries are exact."""

    report = mod.build_report(
        REPO,
        date="20260811",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )
    gates = report["branch_independent_promotion_ledger"]["structured_gate_replay"]
    by_task = {row["task_id"]: row for row in gates["gates"]}

    assert by_task["exp6302-shared-activation-state-initializer"]["passed"] is False
    assert by_task["exp6305-evidence-licensed-cross-family-transfer"]["passed"] is False
    assert by_task["exp6308-arc-target-validated-route-holdout"]["passed"] is True

    assert report["oracle_claim_boundary"]["oracle_promoted_as_verifier_value"] is False
    assert report["replay_is_not_transfer_boundary"]["replay_promoted_as_transfer"] is False
    assert report["safety_cannot_promote_utility_boundary"]["safety_promoted_as_utility"] is False
    assert report["arc_no_solve_claim_boundary"]["arc_proxy_promoted_as_solve"] is False
    assert report["prior_failure_retirement_actions"]["rule_fired_count"] == 2
    fired_tasks = {
        action["task_id"]
        for action in report["prior_failure_retirement_actions"]["actions"]
        if action["rule_fired"]
    }
    assert fired_tasks == {
        "exp6302-shared-activation-state-initializer",
        "exp6305-evidence-licensed-cross-family-transfer",
    }
    assert report["exclusion_manifest_updates"]["updated"] is True

    same = mod.prior_failure_retirement_actions(
        [
            {
                "id": "same",
                "prior_failures": [
                    {
                        "experiment_id": "exp1",
                        "verdict": "complete: same",
                        "retire_if_same_verdict": True,
                    }
                ],
            }
        ],
        {"same": {"terminal_class": "complete", "honest_verdict_raw": "complete: same"}},
    )
    assert same["rule_fired_count"] == 1
    assert same["actions"][0]["action"] == "retire_if_same_verdict_rule_fired"


def test_schema_validation_and_artifact_root_override(tmp_path: Path) -> None:
    """SCENARIO-INTEG-6309-6: schema, checksum, and isolated writes are enforced."""

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    report = mod.build_report(
        REPO,
        date="20260811",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_provenance"])
    assert report["verifier_is_oracle"] == mod.VERIFIER_ORACLE_BOUNDARY
    assert report["honest_verdict"].startswith("complete:")
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert report["architecture_reconciliation_receipt"]["architecture_stale_over_30_days"] is True
    assert (
        report["openspec_traceability_status_changelog_and_reference_reconciliation_receipts"][
            "traceability_status_changelog_touched_by_this_task"
        ]
        is False
    )

    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})

    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_validation_helper_edges_fail_closed() -> None:
    """REQ-INTEG-6309: validation rejects malformed reports."""

    invalid = {"status": "complete"}
    errors = mod.validate_report(invalid)
    assert "missing required field: milestone_roadmap_path_and_hash" in errors
    assert "field_principles is not a mapping" in errors
    assert "field_provenance is not a mapping" in errors
    assert "counts field is not a mapping" in errors
    assert "verifier_is_oracle boundary is wrong" in errors
    assert "honest_verdict lacks terminal prefix" in errors
    assert "reproducibility_checksum missing" in errors

    broken = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    broken.update(
        {
            "status": "complete",
            "field_principles": dict(mod.FIELD_PRINCIPLES),
            "field_provenance": {},
            "missing_nonterminal_blocked_skipped_null_flagged_retired_ready_oracle_only_replay_only_safety_only_and_unlicensed_counts": {
                "task_count": 13,
                "terminal_class_task_count_sum": 12,
                "count_principles": {},
            },
            "branch_independent_promotion_ledger": {"structured_gate_replay": {"gates": [{}]}},
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "verifier_is_oracle": mod.VERIFIER_ORACLE_BOUNDARY,
            "honest_verdict": "complete: fixture",
            "reproducibility_checksum": "sha256:bad",
        }
    )
    broken_errors = mod.validate_report(broken)
    assert "terminal class counts must conserve 13 tasks" in broken_errors
    assert "missing count principle: missing" in broken_errors
    assert "gate missing principle" in broken_errors
    assert "missing field_provenance entry: status" in broken_errors
    assert "reproducibility_checksum mismatch" in broken_errors

    try:
        mod.write_report({"status": "complete"}, REPO)
    except ValueError as exc:
        assert "invalid Exp6309 report" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid report unexpectedly wrote")


def test_fail_closed_helper_edges_cover_malformed_inputs(tmp_path: Path) -> None:
    """SCENARIO-INTEG-6309-1: malformed inputs stay non-promotable."""

    assert mod.read_yaml_mapping(tmp_path / "missing.yaml") == {}
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- not-a-map\n", encoding="utf-8")
    assert mod.read_yaml_mapping(list_yaml) == {}

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    _, bad_meta = mod.read_json_mapping(bad_json)
    assert bad_meta["loadable"] is False
    assert str(bad_meta["error"]).startswith("json_error:")

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[1, 2, 3]", encoding="utf-8")
    _, scalar_meta = mod.read_json_mapping(scalar_json)
    assert scalar_meta["error"] == "json_not_mapping"

    assert mod.roadmap_tasks({"tasks": "not-a-list"}) == []
    assert mod._bare_value({"score": {"value": 1.0, "principle": "fixture"}}, "score") == 1.0

    assert mod.evaluate_operator("x", "exists", True) is True
    assert mod.evaluate_operator(None, "==", 1) is False
    assert mod.evaluate_operator(2, "!=", 1) is True
    assert mod.evaluate_operator(2, ">", 1) is True
    assert mod.evaluate_operator(2, ">=", 2) is True
    assert mod.evaluate_operator(1, "<", 2) is True
    assert mod.evaluate_operator(2, "<=", 2) is True
    assert mod.evaluate_operator("x", ">", 1) is False
    assert mod.evaluate_operator(1, "??", 1) is False

    missing_gate = mod.evaluate_structured_gates(
        tmp_path,
        [
            {
                "id": "downstream",
                "deliverable": "results/downstream.json",
                "gated_on": [
                    {
                        "upstream": "missing",
                        "artifact_field": "score",
                        "op": "==",
                        "value": 1.0,
                    }
                ],
            }
        ],
    )
    assert missing_gate["gates"][0]["gate_results"][0]["reason"] == "missing_upstream_task"

    skipped_prior = mod.prior_failure_retirement_actions(
        [
            {
                "id": "same",
                "prior_failures": [
                    "not-a-map",
                    {"verdict": "complete: same", "retire_if_same_verdict": False},
                ],
            }
        ],
        {"same": {"terminal_class": "complete", "honest_verdict_raw": "complete: same"}},
    )
    assert skipped_prior["actions"] == []

    publication = mod.publication_gate_replay([])
    assert publication["paper_ready"] is None
    assert publication["unmet_gates"] == []
    assert publication["gates"] == {}

    assert mod._status_from_commands([{"command": "bad", "exit_code": 2}])[0] == "blocked"
