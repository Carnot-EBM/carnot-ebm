"""Tests for the V656 independent raw-row science audit."""

from __future__ import annotations

from copy import deepcopy

import pytest

from carnot import experiment_7498_v656_independent_audit as audit


def _protocol_fixture() -> dict[str, list[dict[str, object]]]:
    response = "First sentence. Second sentence."
    return {
        "groups": [
            {"group_id": "fit", "role": "training", "eligible": True},
            {"group_id": "eval", "role": "test", "eligible": True},
        ],
        "predictors": [
            {"group_id": "fit", "role": "training", "response_text": response},
            {"group_id": "eval", "role": "test", "response_text": response},
        ],
        "windows": [
            {"group_id": group, "window_index": index, "byte_start": start, "byte_end": end}
            for group in ("fit", "eval")
            for index, (start, end) in enumerate(((0, 16), (16, len(response.encode()))))
        ],
        "requests": [
            {
                "request_id": f"{group}-{order}",
                "group_id": group,
                "role": role,
                "eligible": True,
                "arm": "whole_response",
                "option_order": list(order),
            }
            for group, role in (("fit", "training"), ("eval", "test"))
            for order in (
                ("supported", "contains_unsupported"),
                ("contains_unsupported", "supported"),
            )
        ],
        "evaluators": [
            {"group_id": "fit", "role": "training", "label": 1},
            {"group_id": "eval", "role": "test", "label": 0},
        ],
    }


def _capture_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    plan = _protocol_fixture()["requests"]
    rows = []
    for planned in plan:
        order = list(planned["option_order"])
        rows.append(
            {
                **planned,
                "disposition": "complete",
                "label_to_option_id": {" A": order[0], " B": order[1]},
                "raw_logits_by_option_id": {
                    "supported": 2.0,
                    "contains_unsupported": 1.0,
                },
            }
        )
    return plan, rows


# REQ-REPORT-7498; SCENARIO-REPORT-7498-RAW
def test_protocol_reduction_checks_roles_coverage_and_frozen_labels() -> None:
    fixture = _protocol_fixture()
    reduced = audit.reduce_protocol_rows(**fixture)

    assert reduced["passed"] is True
    assert reduced["role_disjoint"] is True
    assert reduced["complete_response_coverage"] is True
    assert reduced["frozen_label_boundary"] is True
    assert reduced["group_counts"] == {"test": 1, "training": 1}

    overlap = deepcopy(fixture)
    overlap["groups"].append({"group_id": "fit", "role": "test", "eligible": True})
    assert audit.reduce_protocol_rows(**overlap)["role_disjoint"] is False

    gap = deepcopy(fixture)
    gap["windows"][1]["byte_start"] = 17
    assert audit.reduce_protocol_rows(**gap)["complete_response_coverage"] is False

    opened = deepcopy(fixture)
    opened["requests"][0]["gold_label"] = 1
    assert audit.reduce_protocol_rows(**opened)["frozen_label_boundary"] is False


# REQ-REPORT-7498; SCENARIO-REPORT-7498-RAW
def test_capture_reduction_rebuilds_option_mapping_and_counts() -> None:
    plan, rows = _capture_fixture()
    reduced = audit.reduce_capture_rows(plan, rows)

    assert reduced["passed"] is True
    assert reduced["complete_calls"] == 4
    assert reduced["group_counts"] == {"test": 1, "training": 1}
    assert reduced["option_mapping_valid"] is True

    swapped = deepcopy(rows)
    swapped[0]["label_to_option_id"] = {
        " A": "contains_unsupported",
        " B": "supported",
    }
    assert audit.reduce_capture_rows(plan, swapped)["option_mapping_valid"] is False

    missing = rows[:-1]
    assert audit.reduce_capture_rows(plan, missing)["complete_roster"] is False

    with_exclusion = deepcopy(plan)
    excluded_rows = deepcopy(rows)
    with_exclusion[-1]["eligible"] = False
    excluded_rows[-1] = {
        **with_exclusion[-1],
        "disposition": "excluded",
        "label_to_option_id": None,
        "raw_logits_by_option_id": None,
    }
    exclusion = audit.reduce_capture_rows(with_exclusion, excluded_rows)
    assert exclusion["passed"] is True
    assert exclusion["complete_calls"] == 3


# REQ-REPORT-7498; SCENARIO-REPORT-7498-MUTATIONS
def test_future_label_and_terminal_checkpoint_mutations_are_detected() -> None:
    events = [
        {
            "event_id": "e0",
            "prediction_time": 0,
            "feedback_time": 1,
            "label_available_time": 1,
            "state_before_hash": "s0",
            "state_after_hash": "s1",
            "next_state_before_hash": "s1",
            "actual_label": 0,
            "shuffled_label": 1,
        },
        {
            "event_id": "e1",
            "prediction_time": 2,
            "feedback_time": 3,
            "label_available_time": 3,
            "state_before_hash": "s1",
            "state_after_hash": "s2",
            "next_state_before_hash": None,
            "actual_label": 1,
            "shuffled_label": 0,
        },
    ]
    assert audit.replay_feedback_rows(events)["passed"] is True

    future = deepcopy(events)
    future[0]["label_available_time"] = 2
    assert audit.replay_feedback_rows(future)["future_label_count"] == 1

    checkpoint = audit.make_checkpoint_receipt({"terminal_state_hash": "s2"})
    assert audit.verify_checkpoint_receipt(checkpoint) is True
    checkpoint["payload"]["terminal_state_hash"] = "mutated"
    assert audit.verify_checkpoint_receipt(checkpoint) is False

    attacks = audit.run_attack_controls()
    assert attacks == {
        "future_label_mutation_detected": True,
        "terminal_checkpoint_mutation_detected": True,
        "passed": True,
    }


# REQ-REPORT-7498; SCENARIO-REPORT-7498-MISSING
@pytest.mark.parametrize(
    ("dispositions", "reduction_errors", "expected_class", "expected_score"),
    [
        (["available", "absent"], [], "blocked", 1),
        (["available", "disqualified"], ["invalid_present"], "disqualified", 0),
        (["available", "available"], [], "null", 1),
    ],
)
def test_terminal_classification_preserves_missing_and_invalid_branches(
    dispositions: list[str],
    reduction_errors: list[str],
    expected_class: str,
    expected_score: int,
) -> None:
    terminal = audit.classify_terminal(dispositions, reduction_errors)
    assert terminal["verdict_class"] == expected_class
    assert terminal["science_audit_complete_score"] == expected_score
    assert terminal["honest_verdict"].startswith("complete_")


# REQ-REPORT-7498; SCENARIO-REPORT-7498-ARTIFACT
def test_artifact_contract_rejects_provenance_gate_and_checksum_drift(tmp_path) -> None:
    artifact = audit.fixture_artifact()
    assert audit.validate_artifact(artifact, verify_files=False) == []

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "aggregation"
    assert "current_provenance_invalid" in audit.validate_artifact(
        bad_substrate, verify_files=False
    )

    bad_gate = deepcopy(artifact)
    del bad_gate["acceptance_gate_results"][0]["principle"]
    assert "gate_contract_invalid" in audit.validate_artifact(bad_gate, verify_files=False)

    bad_checksum = deepcopy(artifact)
    bad_checksum["science_audit_complete_score"] = 0
    assert "reproducibility_checksum_mismatch" in audit.validate_artifact(
        bad_checksum, verify_files=False
    )

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"].pop("schema")
    assert "field_principles_incomplete" in audit.validate_artifact(
        bad_principles, verify_files=False
    )

    identity = deepcopy(artifact)
    identity["experiment_id"] = "wrong"
    assert "identity_invalid" in audit.validate_artifact(identity, verify_files=False)

    bad_class = deepcopy(artifact)
    bad_class["verdict_class"] = "unknown"
    assert "verdict_class_invalid" in audit.validate_artifact(bad_class, verify_files=False)

    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = True
    assert "oracle_declaration_invalid" in audit.validate_artifact(bad_oracle, verify_files=False)

    bad_score = deepcopy(artifact)
    bad_score["science_audit_complete_score"] = 2
    assert "science_audit_score_invalid" in audit.validate_artifact(bad_score, verify_files=False)

    bad_dispositions = deepcopy(artifact)
    bad_dispositions["claim_dispositions"] = []
    assert "claim_dispositions_invalid" in audit.validate_artifact(
        bad_dispositions, verify_files=False
    )

    bad_block = deepcopy(artifact)
    for row in bad_block["claim_dispositions"]:
        row["disposition"] = "valid"
    assert "blocked_without_missing_branch" in audit.validate_artifact(
        bad_block, verify_files=False
    )

    bad_receipts = deepcopy(artifact)
    bad_receipts["validation_receipts"] = []
    assert "required_validation_failed" in audit.validate_artifact(bad_receipts, verify_files=False)

    bad_budget = deepcopy(artifact)
    bad_budget["sample_size_budget"]["complete"] = 0
    assert "sample_size_budget_mismatch" in audit.validate_artifact(bad_budget, verify_files=False)

    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    bad_file = deepcopy(artifact)
    bad_file["source_artifact_hashes"] = [{"path": str(source), "sha256": "sha256:wrong"}]
    assert f"source_hash_invalid:{source}" in audit.validate_artifact(bad_file)


# REQ-REPORT-7498; SCENARIO-REPORT-7498-ARTIFACT
def test_roadmap_resolution_uses_exact_current_deliverables() -> None:
    tasks = [
        {
            "id": f"exp{number}-task",
            "milestone": "2026.09.656",
            "deliverable": f"results/experiment_{number}_v656_task.json",
        }
        for number in range(7491, 7498)
    ]
    resolved = audit.resolve_roadmap_producers(
        {"milestone": "2026.09.656", "tasks": [None, *tasks]}
    )
    assert [row["number"] for row in resolved] == list(range(7491, 7498))

    with pytest.raises(ValueError, match="roadmap_producer_sequence_invalid"):
        audit.resolve_roadmap_producers({"milestone": "2026.09.656", "tasks": tasks[:-1]})
