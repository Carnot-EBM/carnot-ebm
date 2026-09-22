"""Tests for REQ-REPORT-7525 and SCENARIO-REPORT-7525-*.

All writer checks use ``tmp_path``. Published research evidence stays read-only.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7525_v658_decision_audit as exp
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from scripts.verdict_row_consistency_lint import check_artifact


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, *, passed: bool = True) -> dict[str, Any]:
    """Build one bounded command receipt without starting a child process."""

    return {
        "name": name,
        "command": f"private {name}",
        "command_argv": ["private", name],
        "scope": "private_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "1" * 64,
        "passed": passed,
        "timed_out": False,
        "required": True,
    }


def _validation(*, passed: bool = True, terminal: bool = True) -> list[dict[str, Any]]:
    """Supply the exact required receipt names for pure artifact tests."""

    names = list(validation_scope.REQUIRED_CHECK_NAMES)
    if terminal:
        names.extend(exp.TERMINAL_CHECK_NAMES)
    rows = [_receipt(name) for name in names]
    if not passed:
        rows[1] = _receipt(names[1], passed=False)
    return rows


def _present_inventory() -> list[dict[str, Any]]:
    """Create authenticated inventory rows for private complete fixtures."""

    return [
        {
            "role": role,
            "branch": branch,
            "path": path.as_posix(),
            "required_field": field,
            "expected": 1,
            "observed": 1,
            "exists": True,
            "sha256": "sha256:" + f"{index:064x}",
            "passed": True,
        }
        for index, (role, branch, path, field) in enumerate(exp.INPUT_CONTRACT, start=1)
    ]


def _artifact(
    *,
    receipts: list[dict[str, Any]] | None = None,
    publication_state: str = "terminal",
) -> dict[str, Any]:
    """Build a deterministic complete-null terminal artifact."""

    static = exp.reduce_static_evidence(exp.static_fixture())
    online = exp.reduce_online_evidence(exp.online_fixture())
    return exp.build_artifact(
        inventory=_present_inventory(),
        static_reduction=static,
        online_reduction=online,
        mutation_results=exp.run_private_mutations(),
        validation_receipts=receipts or _validation(),
        publication_state=publication_state,
        started_at_utc="2026-09-22T12:00:00+00:00",
        ended_at_utc="2026-09-22T12:00:01+00:00",
        started_monotonic_ns=10,
        ended_monotonic_ns=1_000_000_010,
        phase_spans=[],
    )


def test_inventory_names_each_external_failure_and_preserves_branches(tmp_path: Path) -> None:
    """REQ-REPORT-7525; SCENARIO-REPORT-7525-INVENTORY."""

    protocol = tmp_path / exp.PROTOCOL_PATH
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps({"source_protocol_ready_score": 0}), encoding="utf-8")

    rows = exp.inventory_inputs(tmp_path)
    assert [row["role"] for row in rows] == [row[0] for row in exp.INPUT_CONTRACT]
    protocol_row = next(row for row in rows if row["role"] == "source_protocol")
    assert protocol_row["exists"] is True
    assert protocol_row["observed"] == 0
    assert protocol_row["passed"] is False
    static = exp.branch_gate_summary(rows, "static")
    online = exp.branch_gate_summary(rows, "online")
    assert static["all_passed"] is False and online["all_passed"] is False
    assert all(
        {"path", "required_field", "expected", "observed"} <= set(row)
        for row in static["failures"] + online["failures"]
    )
    assert exp.classify_overall(static="blocked", online="null", validation_passed=True) == (
        "blocked",
        "complete_blocked_required_science_incomplete",
    )


def test_static_rows_recompute_to_a_qualified_honest_null(tmp_path: Path) -> None:
    """REQ-REPORT-7525; SCENARIO-REPORT-7525-STATIC."""

    reduction = exp.reduce_static_evidence(exp.static_fixture())
    assert reduction["errors"] == []
    assert reduction["claims_qualified_score"] == 1
    assert reduction["value_score"] == 0
    assert reduction["verdict_class"] == "null"
    assert reduction["support"]["complete_groups"] == 4
    assert all(row["arm_a_cost"] == row["arm_b_cost"] for row in reduction["rows"])
    assert all(row["no_headroom"] and not row["positive_claim"] for row in reduction["rows"])
    assert all(row["headroom_explanation"] for row in reduction["rows"])

    guard = tmp_path / "honest-null.json"
    guard.write_text(json.dumps(exp.strict_null_fixture()), encoding="utf-8")
    assert check_artifact(guard) == ("ok", [])


def test_online_events_recompute_counts_chronology_and_retention() -> None:
    """REQ-REPORT-7525; SCENARIO-REPORT-7525-ONLINE."""

    reduction = exp.reduce_online_evidence(exp.online_fixture())
    assert reduction["errors"] == []
    assert reduction["claims_qualified_score"] == 1
    assert reduction["value_score"] == 0
    assert reduction["verdict_class"] == "null"
    assert reduction["support"] == {
        "complete_sources": 4,
        "delivered_labels": 4,
        "class_0": 2,
        "class_1": 2,
        "mixed_batch_labels": 4,
    }
    assert reduction["chronology_violations"] == 0
    assert reduction["restart_parity_score"] == 1
    assert reduction["retention_upper95_deterioration"] == 0.0


def test_all_nine_private_mutations_fail_closed() -> None:
    """REQ-REPORT-7525; SCENARIO-REPORT-7525-MUTATIONS."""

    expected = {
        "label_leakage",
        "swapped_option_mapping",
        "omitted_failed_group",
        "future_release",
        "duplicated_update",
        "changed_comparator",
        "edited_gate",
        "false_positive_class",
        "majority_identical_cost",
    }
    rows = exp.run_private_mutations()
    assert {row["mutation"] for row in rows} == expected
    assert all(row["rejected"] is True for row in rows)
    assert all(row["errors"] and row["corrupted_payload_retained"] is False for row in rows)


def test_named_mutation_errors_are_specific() -> None:
    """SCENARIO-REPORT-7525-MUTATIONS names the protected boundary."""

    expected = {
        "label_leakage": "private_label_visible_before_reveal:s0",
        "swapped_option_mapping": "option_mapping_mismatch:g0",
        "omitted_failed_group": "planned_group_missing:g_failed",
        "future_release": "update_before_release:u0",
        "duplicated_update": "duplicate_update:u0",
        "changed_comparator": "registered_comparator_changed",
        "edited_gate": "registered_gates_changed",
        "false_positive_class": "static_false_positive_claim",
        "majority_identical_cost": "positive_claim_without_headroom:g0",
    }
    for name, error in expected.items():
        static = exp.static_fixture()
        online = exp.online_fixture()
        exp.apply_private_mutation(static, online, name)
        errors = [
            *exp.reduce_static_evidence(static)["errors"],
            *exp.reduce_online_evidence(online)["errors"],
        ]
        assert error in errors


def test_terminal_artifact_keeps_qualification_separate_from_value() -> None:
    """REQ-REPORT-7525; SCENARIO-REPORT-7525-ARTIFACT."""

    artifact = _artifact()
    assert exp.validate_artifact(artifact) == []
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_null_both_decision_branches_qualified"
    assert artifact["audit_complete_score"] == 1
    assert artifact["decision_claims_qualified_score"] == 1
    assert artifact["online_claims_qualified_score"] == 1
    assert artifact["qualified_static_value_score"] == 0
    assert artifact["qualified_online_value_score"] == 0
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert set(artifact["field_principles"]) == set(artifact)


def test_blocked_artifact_has_complete_branch_dispositions() -> None:
    """SCENARIO-REPORT-7525-INVENTORY keeps external absence terminal and explicit."""

    inventory = exp.inventory_inputs(ROOT)
    artifact = exp.build_artifact(
        inventory=inventory,
        static_reduction=exp.blocked_reduction("static", inventory),
        online_reduction=exp.blocked_reduction("online", inventory),
        mutation_results=exp.run_private_mutations(),
        validation_receipts=_validation(),
        publication_state="terminal",
        started_at_utc="2026-09-22T12:00:00+00:00",
        ended_at_utc="2026-09-22T12:00:01+00:00",
        started_monotonic_ns=10,
        ended_monotonic_ns=1_000_000_010,
        phase_spans=[],
    )
    assert exp.validate_artifact(artifact) == []
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["audit_complete_score"] == 1
    assert artifact["decision_claims_qualified_score"] == 0
    assert artifact["online_claims_qualified_score"] == 0
    assert artifact["gate_check_summary"]["failed_count"] > 0
    assert {row["branch"] for row in artifact["rows"]} == {"static", "online"}


def test_failed_required_validation_disqualifies_audit() -> None:
    """SCENARIO-REPORT-7525-ARTIFACT gives validity precedence over science."""

    artifact = _artifact(receipts=_validation(passed=False))
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["audit_complete_score"] == 0
    assert artifact["decision_claims_qualified_score"] == 0
    assert artifact["online_claims_qualified_score"] == 0
    assert artifact["qualified_static_value_score"] == 0
    assert artifact["qualified_online_value_score"] == 0
    assert exp.validate_artifact(artifact) == []


def test_cold_validator_rejects_identity_scores_hashes_and_receipts() -> None:
    """SCENARIO-REPORT-7525-ARTIFACT rejects terminal contract drift."""

    mutations: tuple[tuple[str, Any, str], ...] = (
        ("schema", "wrong", "identity_mismatch"),
        ("model_invoked", True, "current_model_declaration_mismatch"),
        ("inference_substrate", "other", "inference_substrate_mismatch"),
        ("audit_complete_score", 0, "terminal_scores_mismatch"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum_mismatch"),
    )
    for field, value, expected in mutations:
        artifact = _artifact()
        artifact[field] = value
        assert expected in exp.validate_artifact(artifact)

    artifact = _artifact()
    artifact["validation_receipts"] = artifact["validation_receipts"][:-1]
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert "required_validation_incomplete" in exp.validate_artifact(artifact)

    artifact = _artifact()
    artifact["field_principles"].pop("rows")
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert "field_principles_incomplete" in exp.validate_artifact(artifact)


def test_candidate_validation_does_not_invent_terminal_receipts() -> None:
    """SCENARIO-REPORT-7525-ARTIFACT permits a measured pre-publication candidate."""

    artifact = _artifact(receipts=_validation(terminal=False), publication_state="candidate")
    assert exp.validate_artifact(artifact) == []
    artifact["publication_state"] = "terminal"
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert "required_validation_incomplete" in exp.validate_artifact(artifact)


def test_validation_plan_is_scoped_and_imports_this_worktree(tmp_path: Path) -> None:
    """REQ-REPORT-7525; SCENARIO-REPORT-7525-ARTIFACT."""

    plan = exp.build_validation_plan(ROOT, tmp_path / "private")
    assert [row.name for row in plan] == list(validation_scope.REQUIRED_CHECK_NAMES)
    assert exp.validate_validation_plan(ROOT, plan) == []
    by_name = {row.name: row for row in plan}
    focused = by_name["focused_pytest"]
    assert "tests/python" not in focused.argv
    assert "-n" in focused.argv and "0" in focused.argv
    assert "-o" in focused.argv and "addopts=" in focused.argv
    assert "--no-cov" in focused.argv
    assert any(part.startswith("--basetemp=") for part in focused.argv)
    coverage_env = dict(by_name["changed_module_coverage_report"].command_environment)
    assert coverage_env["COVERAGE_FILE"].startswith(str(tmp_path))

    terminal = exp.terminal_commands(ROOT, tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert all(row.required for row in terminal)
    assert str(tmp_path / "candidate.json") in terminal[0].spec.argv


def test_parser_independent_replay_and_atomic_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7525-ARTIFACT exercises the thin public reader surface."""

    candidate = tmp_path / "candidate.json"
    artifact = _artifact()
    exp.atomic_json(candidate, artifact)
    assert exp.independent_replay(candidate) == []
    parsed = exp.parse_args(["--date", "20260922", "--validate-candidate", str(candidate)])
    assert parsed.date == "20260922"
    assert parsed.validate_candidate == candidate
    assert exp.main(["--date", "20260922", "--validate-candidate", str(candidate)]) == 0

    artifact["rows"][0]["disposition"] = "positive"
    exp.atomic_json(candidate, artifact)
    assert "reproducibility_checksum_mismatch" in exp.independent_replay(candidate)


def test_metric_helpers_preserve_absolute_costs_and_clip_only_log_loss() -> None:
    """SCENARIO-REPORT-7525-STATIC checks proper losses and typed policy costs."""

    assert exp.metric_losses(0.0, 1)["brier"] == 1.0
    assert exp.metric_losses(1.0, 1)["brier"] == 0.0
    assert exp.metric_losses(0.0, 1)["log_loss"] > 20.0
    action, cost = exp.typed_decision(0.5, 1)
    assert action == "escalate" and cost == 0.2
    assert exp.typed_decision(0.01, 1) == ("accept", 5.0)
    assert exp.typed_decision(0.99, 0) == ("reject", 1.0)


def test_static_defensive_reader_names_malformed_operands(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7525-STATIC covers each raw custody failure."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    malformed.write_text("[]", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    assert exp._load_object(tmp_path / "missing.json") == {}
    with pytest.raises(ValueError, match="coefficient_shape_invalid"):
        exp._probability({"bias": 1.0}, {"feature_order": ["bias"], "values": []})

    fixture = exp.static_fixture()
    fixture["raw_rows"].append(deepcopy(fixture["raw_rows"][0]))
    fixture["changed_source_rows"].extend(
        [None, {"group_id": "leak", "label": 1, "label_origin": "original"}]
    )
    fixture["hash_boundaries"]["model"] = "sha256:" + "c" * 64
    fixture["per_source_results"][0]["candidate_brier"] = 9.0
    errors = exp.reduce_static_evidence(fixture)["errors"]
    assert "duplicate_group:g0" in errors
    assert "changed_source_row_invalid" in errors
    assert "changed_source_original_label_exposed:leak" in errors
    assert "static_hash_boundary_mismatch" in errors
    assert "static_operand_mismatch:g0:candidate_brier" in errors

    missing = exp.static_fixture()
    missing["per_source_results"] = missing["per_source_results"][1:]
    assert "result_row_missing:g0" in exp.reduce_static_evidence(missing)["errors"]
    invalid = exp.static_fixture()
    invalid["raw_rows"][0]["features"] = None
    assert "prediction_operand_invalid:g0" in exp.reduce_static_evidence(invalid)["errors"]


def test_online_defensive_reader_names_custody_and_metric_drift() -> None:
    """SCENARIO-REPORT-7525-ONLINE covers chronology, donor, metric, and restart guards."""

    fixture = exp.online_fixture()
    fixture["registered_gates"]["block_length"] = 99
    fixture["hash_boundaries"]["model"] = "sha256:" + "d" * 64
    fixture["events"][1]["event_index"] = fixture["events"][0]["event_index"]
    release = next(row for row in fixture["events"] if row.get("release_id") == "r0")
    release["donors"]["s0"] = "s0"
    update = next(row for row in fixture["events"] if row.get("update_id") == "u1")
    update["release_event_index"] = 99
    update["labels"]["local"] = 0
    fixture["events"].append({"event_type": "unknown", "event_index": 20})
    fixture["per_source_results"][0]["label"] = 1
    fixture["per_source_results"][0]["probabilities"]["local"] = 0.9
    fixture["per_source_results"][0]["brier"]["local"] = 0.9
    fixture["per_source_results"][0]["log_loss"]["local"] = 0.9
    fixture["retention_rows"] = [None]
    fixture["checkpoint_manifest"]["resumed_hash"] = "sha256:" + "e" * 64
    fixture["producer_summary"].update(
        {"verdict_class": "positive", "online_information_value_score": 1}
    )
    errors = exp.reduce_online_evidence(fixture)["errors"]
    for expected in (
        "online_registered_gates_changed",
        "online_hash_boundary_mismatch",
        "duplicate_event_index",
        "donor_role_invalid:s0",
        "release_reference_mismatch:u1",
        "update_label_mismatch:u1:local",
        "event_type_invalid:unknown",
        "online_label_mismatch:s0",
        "online_stored_probability_mismatch:s0:local",
        "online_brier_mismatch:s0:local",
        "online_log_loss_mismatch:s0:local",
        "retention_row_not_read_only",
        "restart_checkpoint_mismatch",
        "online_false_positive_claim",
    ):
        assert expected in errors


def test_online_reader_rejects_missing_late_and_unreported_sources() -> None:
    """SCENARIO-REPORT-7525-ONLINE does not average away missing sources."""

    missing_release = exp.online_fixture()
    missing_release["events"] = [
        row
        for row in missing_release["events"]
        if not (row.get("event_type") == "release" and row.get("release_id") == "r1")
        and row.get("update_id") not in {"u2", "u3"}
    ]
    errors = exp.reduce_online_evidence(missing_release)["errors"]
    assert "source_not_released:s2" in errors
    assert "source_not_released:s3" in errors

    late = exp.online_fixture()
    release = next(row for row in late["events"] if row.get("release_id") == "r0")
    release["event_index"] = 0.5
    assert "reveal_not_after_prediction:s1" in exp.reduce_online_evidence(late)["errors"]

    missing_result = exp.online_fixture()
    missing_result["per_source_results"] = missing_result["per_source_results"][1:]
    assert "online_result_missing:s0" in exp.reduce_online_evidence(missing_result)["errors"]


def test_closed_classification_and_unknown_mutation_boundaries() -> None:
    """REQ-REPORT-7525 preserves disqualified and positive terminal precedence."""

    assert (
        exp.classify_overall(static="disqualified", online="null", validation_passed=True)[0]
        == "disqualified"
    )
    assert exp.classify_overall(static="positive", online="null", validation_passed=True)[0] == (
        "positive"
    )
    with pytest.raises(ValueError, match="unknown_mutation"):
        exp.apply_private_mutation(exp.static_fixture(), exp.online_fixture(), "unknown")


def test_cold_validator_names_each_schema_boundary(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7525-ARTIFACT covers every terminal schema guard."""

    assert exp.validate_artifact(None) == ["artifact_not_object"]
    assert exp._source_revision(tmp_path) is None
    changes: tuple[tuple[str, Any, str], ...] = (
        ("verdict_class", "unknown", "verdict_class_invalid"),
        ("honest_verdict", "blocked_without_complete", "honest_verdict_prefix_invalid"),
        ("publication_state", "draft", "publication_state_invalid"),
        ("rows", [], "branch_rows_incomplete"),
        ("audit_complete_score", True, "score_not_bare_binary:audit_complete_score"),
    )
    for field, value, expected in changes:
        artifact = _artifact()
        artifact[field] = value
        artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
        assert expected in exp.validate_artifact(artifact)

    artifact = _artifact()
    artifact["qualified_static_value_score"] = 1
    artifact["decision_claims_qualified_score"] = 0
    artifact["qualified_online_value_score"] = 1
    artifact["online_claims_qualified_score"] = 0
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    errors = exp.validate_artifact(artifact)
    assert "static_value_exceeds_qualification" in errors
    assert "online_value_exceeds_qualification" in errors

    blocked = exp.build_artifact(
        inventory=exp.inventory_inputs(ROOT),
        static_reduction=exp.blocked_reduction("static", exp.inventory_inputs(ROOT)),
        online_reduction=exp.blocked_reduction("online", exp.inventory_inputs(ROOT)),
        mutation_results=exp.run_private_mutations(),
        validation_receipts=_validation(),
        publication_state="terminal",
        started_at_utc="2026-09-22T12:00:00+00:00",
        ended_at_utc="2026-09-22T12:00:01+00:00",
        started_monotonic_ns=10,
        ended_monotonic_ns=1_000_000_010,
        phase_spans=[],
    )
    blocked["gate_check_summary"]["first_failure"] = None
    blocked["decision_claims_qualified_score"] = 1
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    errors = exp.validate_artifact(blocked)
    assert "blocked_gate_summary_missing" in errors
    assert "blocked_claim_qualification_nonzero" in errors


def test_main_rejects_wrong_date() -> None:
    """REQ-REPORT-7525 binds the run date before any output is read."""

    with pytest.raises(SystemExit, match="--date must be 20260922"):
        exp.main(["--date", "20260921"])
