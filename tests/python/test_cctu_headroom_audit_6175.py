"""Tests for the Exp6175 CCTU headroom audit.

Spec refs: REQ-CONSTRAINT-VERIFY-6175,
SCENARIO-CONSTRAINT-VERIFY-6175-RAW-REVALIDATION,
SCENARIO-CONSTRAINT-VERIFY-6175-NO-HELD-ROWS,
SCENARIO-CONSTRAINT-VERIFY-6175-RETIRE-PARSE-FAILURE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import cctu_headroom_audit_6175 as audit
from carnot.verify import cctu_item_bank_6173 as exp6173


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
NAMED_GATE_REQS = (
    "REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY",
    "REQ-CONSTRAINT-VERIFY-6175-PARSEABILITY",
    "REQ-CONSTRAINT-VERIFY-6175-EXACT-FLOOR",
    "REQ-CONSTRAINT-VERIFY-6175-COMPETENCE",
    "REQ-CONSTRAINT-VERIFY-6175-UNSATURATION",
    "REQ-CONSTRAINT-VERIFY-6175-CONSENSUS",
    "REQ-CONSTRAINT-VERIFY-6175-ORACLE-K",
    "REQ-CONSTRAINT-VERIFY-6175-ERROR-DIVERSITY",
    "REQ-CONSTRAINT-VERIFY-6175-CLUSTERED-INFERENCE",
    "REQ-CONSTRAINT-VERIFY-6175-HELD-AGGREGATE",
    "REQ-CONSTRAINT-VERIFY-6175-NO-SELECTOR",
    "REQ-CONSTRAINT-VERIFY-6175-FAIL-CLOSED-RETIREMENT",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_req_6175_spec_declares_fail_closed_headroom_contract() -> None:
    """REQ-CONSTRAINT-VERIFY-6175: OpenSpec declares the required audit gates."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONSTRAINT-VERIFY-6175") :]

    for marker in (
        "SCENARIO-CONSTRAINT-VERIFY-6175-RAW-REVALIDATION",
        "SCENARIO-CONSTRAINT-VERIFY-6175-NO-HELD-ROWS",
        "SCENARIO-CONSTRAINT-VERIFY-6175-RETIRE-PARSE-FAILURE",
        "deterministic_exact_tool_trace_headroom_audit",
        "future_rows_allowed_by_this_artifact",
        *NAMED_GATE_REQS,
    ):
        assert marker in section


def test_req_6175_named_gates_are_reflected_in_receipts_and_provenance(
    tmp_path: Path,
) -> None:
    """REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY: named gates are auditable."""

    artifact = audit.run(result_path=tmp_path / audit.RESULT_RELATIVE_PATH.name, duration_s=1.0)
    checks = artifact["preconditions_checked"]["checks"]
    provenance = artifact["field_provenance"]

    for check_name in (
        "raw_trace_hash_matches_exp6174",
        "label_sidecar_hashes_match",
        "calibration_and_held_seals",
        "output_paths_declared",
    ):
        assert checks[check_name] is True

    expected_field_reqs = {
        "preconditions_checked": "REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY",
        "all_sample_and_parseable_denominators": "REQ-CONSTRAINT-VERIFY-6175-PARSEABILITY",
        "exact_floor_definition_value_and_provenance": "REQ-CONSTRAINT-VERIFY-6175-EXACT-FLOOR",
        "per_candidate_competence_and_clustered_interval": "REQ-CONSTRAINT-VERIFY-6175-COMPETENCE",
        "saturation_and_error_diversity_metrics": "REQ-CONSTRAINT-VERIFY-6175-ERROR-DIVERSITY",
        "tuned_oracle_blind_consensus_definition_freeze_and_accuracy": "REQ-CONSTRAINT-VERIFY-6175-CONSENSUS",
        "oracle_at_8_accuracy": "REQ-CONSTRAINT-VERIFY-6175-ORACLE-K",
        "oracle_minus_consensus_delta_and_clustered_interval": "REQ-CONSTRAINT-VERIFY-6175-CLUSTERED-INFERENCE",
        "held_aggregate_qualification_and_row_label_seal_hash": "REQ-CONSTRAINT-VERIFY-6175-HELD-AGGREGATE",
        "duplicate_and_shortcut_audits": "REQ-CONSTRAINT-VERIFY-6175-NO-SELECTOR",
        "parseability_competence_unsaturation_headroom_and_minority_gate_matrix": "REQ-CONSTRAINT-VERIFY-6175-FAIL-CLOSED-RETIREMENT",
    }
    for field, req_id in expected_field_reqs.items():
        assert req_id in provenance[field]


def test_scenario_6175_revalidates_exp6174_labels_from_raw_text() -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6175-RAW-REVALIDATION: labels replay exactly."""

    paths = audit.default_paths()
    raw_rows = audit.read_jsonl(paths["raw_trace"])
    calibration = audit.read_jsonl(paths["calibration_label"])
    held = audit.read_jsonl(paths["held_label"])
    receipt = audit.revalidate_labels_from_raw(raw_rows, calibration + held)

    assert receipt["raw_rows_revalidated"] == 960
    assert receipt["label_rows_checked"] == 960
    assert receipt["validator_result_mismatch_count"] == 0
    assert receipt["raw_row_hash_mismatch_count"] == 0
    assert receipt["validator_version"] == exp6173.VALIDATOR_VERSION
    assert receipt["terminal_pass_count"] == 0


def test_scenario_6175_retired_artifact_keeps_failures_in_denominators(
    tmp_path: Path,
) -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6175-RETIRE-PARSE-FAILURE: Exp6174 retires."""

    artifact = audit.run(
        result_path=tmp_path / audit.RESULT_RELATIVE_PATH.name,
        duration_s=6.175,
        test_exit_codes={command: 0 for command in audit.DEFAULT_TEST_COMMANDS},
    )

    assert set(audit.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "retired"
    assert artifact["honest_verdict"].startswith("retired:")
    assert artifact["phase_d_headroom_ready_score"] == 0.0
    assert artifact["future_rows_allowed_by_this_artifact"] is False
    assert artifact["inference_substrate"] == audit.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    denominators = artifact["all_sample_and_parseable_denominators"]
    assert denominators["all_samples"]["count"] == 960
    assert denominators["all_samples"]["terminal_pass_count"] == 0
    assert denominators["parseable_samples"]["count"] == 0
    assert denominators["headline_denominator_policy"]["never_drop_failures"] is True

    competence = artifact["per_candidate_competence_and_clustered_interval"]
    assert competence["accuracy_all_sample"] == 0.0
    assert competence["clustered_interval"]["lower"] == 0.0
    assert competence["above_exact_floor_gate_passed"] is False

    gate_matrix = artifact["parseability_competence_unsaturation_headroom_and_minority_gate_matrix"]
    failed = {name for name, row in gate_matrix["conjuncts"].items() if not row["passed"]}
    assert {"parseability", "competence", "headroom", "minority", "family_support"} <= failed


def test_scenario_6175_tuned_consensus_is_oracle_blind_and_held_rows_are_sealed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6175-NO-HELD-ROWS: held output is aggregate only."""

    artifact = audit.run(result_path=tmp_path / audit.RESULT_RELATIVE_PATH.name, duration_s=1.0)
    consensus = artifact["tuned_oracle_blind_consensus_definition_freeze_and_accuracy"]
    held = artifact["held_aggregate_qualification_and_row_label_seal_hash"]

    assert consensus["forbidden_selection_inputs"] == {
        "validator_labels_at_selection_time": False,
        "held_labels_for_tuning": False,
        "hidden_states": False,
        "arbitrary_ids": False,
        "answer_positions": False,
        "sample_indexes": False,
    }
    assert consensus["tuning_split"] == "calibration"
    assert held["held_rows_exposed"] is False
    assert held["sealed_row_label_hash"].startswith("sha256:")
    assert held["aggregate_signature_sha256"].startswith("sha256:")

    serialized = json.dumps(held, sort_keys=True)
    forbidden_fragments = (
        "sample_key",
        "raw_row_hash",
        "label_row_hash",
        "validator_result",
        "cctu-6173-resource-008::k00",
    )
    assert not any(fragment in serialized for fragment in forbidden_fragments)


def test_req_6175_helper_branches_and_schema_validation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CONSTRAINT-VERIFY-6175: helper branches and schema checks stay covered."""

    assert audit.normalize_action_terminal_cluster("{not json").startswith("unparseable:")
    assert audit.normalize_action_terminal_cluster('{"ok": true} trailing').startswith(
        "unparseable:trailing-or-nonobject"
    )
    trace = exp6173.known_valid_trace(exp6173.build_item_bank()[0])
    normalized = audit.normalize_action_terminal_cluster(json.dumps(trace))
    assert "tool:math.aggregate" in normalized
    assert "final:15|abstain:false" in normalized

    assert audit.clustered_interval([])["estimate"] is None
    interval = audit.clustered_interval([0.0, 1.0, 1.0], seed=6175, resamples=25)
    assert 0.0 <= interval["lower"] <= interval["estimate"] <= interval["upper"] <= 1.0
    assert audit.rate(0, 0) is None
    assert audit.sha256_json({"b": 2, "a": 1}).startswith("sha256:")

    valid_artifact = audit.run(result_path=tmp_path / "artifact.json", duration_s=1.0)
    validation = audit.validate_artifact(valid_artifact)
    assert validation["ok"] is True

    missing = dict(valid_artifact)
    missing.pop("honest_verdict")
    assert audit.validate_artifact(missing)["ok"] is False

    bad_ready = dict(valid_artifact)
    bad_ready["phase_d_headroom_ready_score"] = 1.0
    assert audit.validate_artifact(bad_ready)["ok"] is False

    bad_schema = dict(valid_artifact)
    bad_schema["inference_substrate"] = "wrong"
    bad_schema["verifier_is_oracle"] = False
    bad_schema["future_rows_allowed_by_this_artifact"] = True
    bad_schema["status"] = "complete_ready"
    bad_schema["held_aggregate_qualification_and_row_label_seal_hash"] = {"sample_key": "leak"}
    validation = audit.validate_artifact(bad_schema)
    assert validation["ok"] is False
    assert {
        "bad_inference_substrate",
        "verifier_is_oracle_not_true",
        "nonready_allows_future_rows",
        "complete_ready_without_ready_score_one",
        "held_rows_exposed",
    } <= set(validation["errors"])

    with pytest.raises(ValueError, match="duplicate raw sample_key"):
        audit.index_raw_rows([{"sample_key": "dup"}, {"sample_key": "dup"}])

    records = [
        {
            "case_id": "case-1",
            "split": "calibration",
            "cluster": "bad",
            "parseable": False,
            "terminal_passed": False,
            "timeout": True,
            "refusal": True,
            "truncated": False,
        },
        {
            "case_id": "case-1",
            "split": "calibration",
            "cluster": "bad",
            "parseable": False,
            "terminal_passed": False,
            "timeout": False,
            "refusal": False,
            "truncated": False,
        },
        {
            "case_id": "case-1",
            "split": "calibration",
            "cluster": "good",
            "parseable": True,
            "terminal_passed": True,
            "timeout": False,
            "refusal": False,
            "truncated": False,
        },
    ]
    assert audit.selected_cluster_correct(records, {"min_agreement": 9}) is False
    assert audit.selected_cluster_correct([], {"min_agreement": 1}) is False
    assert audit.select_consensus_cluster(records, {"parseable_only": True}) == "good"
    minority = audit.consensus_wrong_oracle_right_group_count(records, {"min_agreement": 1})
    assert minority["count"] == 1
    assert audit._consensus_accuracy([], {"min_agreement": 1}) == 0.0
    assert audit._dominant_failure_surface(records)["name"] == "parse_failure"

    blocked = audit.honest_verdict("blocked", {"failed_conjuncts": ["preconditions"]})
    ready = audit.honest_verdict("complete_ready", {"failed_conjuncts": []})
    null = audit.honest_verdict("complete_null", {"failed_conjuncts": []})
    assert blocked.startswith("blocked:")
    assert ready.startswith("complete_ready:")
    assert null.startswith("complete_null:")

    parent_file = tmp_path / "not-a-dir"
    parent_file.write_text("x", encoding="utf-8")
    assert audit._parent_writable(parent_file / "artifact.json") is False

    output = tmp_path / "cli-artifact.json"
    assert audit.main(["--output", str(output)]) == 0
    assert "retired" in capsys.readouterr().out
    assert audit.main(["--validate", "--output", str(output)]) == 0
    assert '"ok":true' in capsys.readouterr().out


def test_scenario_6175_revalidation_detects_mismatches_without_row_output() -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6175-RAW-REVALIDATION: mismatches fail closed."""

    case = exp6173.build_item_bank()[0]
    raw = {
        "sample_key": "fixture::k00",
        "case_id": case.case_id,
        "split": "calibration",
        "raw_completion_text": json.dumps(exp6173.known_valid_trace(case), sort_keys=True),
    }
    raw["row_hash"] = audit.raw_row_hash(raw)
    validation = exp6173.validate_candidate_trace(case, raw["raw_completion_text"])
    good_label = {
        "sample_key": raw["sample_key"],
        "case_id": case.case_id,
        "split": "calibration",
        "raw_row_hash": raw["row_hash"],
        "validator_version": exp6173.VALIDATOR_VERSION,
        "validator_result": validation,
    }

    bad_hash = dict(good_label, raw_row_hash="sha256:bad")
    bad_version = dict(good_label, validator_version="old")
    bad_result = dict(good_label, validator_result={**validation, "terminal_passed": False})
    missing_raw = dict(good_label, sample_key="fixture::missing")
    receipt = audit.revalidate_labels_from_raw(
        [raw],
        [bad_hash, bad_version, bad_result, missing_raw],
    )

    assert receipt["raw_row_hash_mismatch_count"] == 1
    assert receipt["validator_version_mismatch_count"] == 1
    assert receipt["validator_result_mismatch_count"] == 1
    assert receipt["label_rows_without_raw_count"] == 1
    assert receipt["all_labels_match_revalidation"] is False
    assert receipt["held_row_labels_exposed"] is False
