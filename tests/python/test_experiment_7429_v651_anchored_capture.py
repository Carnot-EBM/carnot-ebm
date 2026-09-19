"""Tests for the repaired bounded anchored extraction capture.

Spec refs: REQ-VERIFY-7429 and SCENARIO-VERIFY-7429-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7416_v650_anchored_extraction as frozen
from carnot import experiment_7429_v651_anchored_capture as exp


def _terminal_row(
    call: dict[str, object],
    *,
    reply: str | None = None,
    finish_reason: str = "stop",
) -> dict[str, object]:
    """Create one raw-first terminal row through the shipped Exp7416 reader."""

    answer = str(call["answer"])
    subject = answer.split()[0]
    object_text = answer.rstrip(".").split()[-1]
    if call["arm"] == "free":
        payload: dict[str, object] = {
            "triples": [
                {
                    "subject": subject,
                    "relation": "mentions",
                    "object": object_text,
                    "qualifiers": [],
                }
            ]
        }
    else:
        object_start = answer.rfind(object_text)
        payload = {
            "triples": [
                {
                    "subject": {"text": subject, "span": [0, len(subject)]},
                    "relation": "mentions",
                    "object": {
                        "text": object_text,
                        "span": [object_start, object_start + len(object_text)],
                    },
                    "qualifiers": [],
                }
            ]
        }
    raw_reply = reply if reply is not None else json.dumps(payload, sort_keys=True)
    response = {
        "raw_request": {"messages": [{"role": "user", "content": call["prompt"]}]},
        "raw_response": {"choices": [{"message": {"content": raw_reply}}]},
        "raw_reply": raw_reply,
        "attempted": True,
        "terminal_state": "response",
        "error": None,
        "finish_reason": finish_reason,
        "prompt_tokens": 20,
        "completion_tokens": 12,
        "latency_s": 0.25,
    }
    return frozen.parse_capture_row(frozen.build_raw_capture_row(call, response, {}))


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-FROZEN
def test_frozen_schedule_is_byte_identical_to_exp7416() -> None:
    cases = frozen._fixture_cases()
    expected = frozen.build_schedule(cases)
    observed = exp.build_frozen_schedule(cases)

    assert observed == expected
    assert exp.schedule_identity(observed) == frozen.canonical_hash(expected)
    assert len(observed) == 96
    assert all(row["seed"] == 6_501_601 for row in observed)
    assert all(row["max_new_tokens"] == 384 for row in observed)


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-FROZEN
def test_runtime_ownership_prerequisites_are_exact_and_fail_closed() -> None:
    ready = {
        "experiment_id": "exp7422-v651-runtime-ownership",
        "runtime_ownership_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    checks = exp.runtime_ownership_gate_rows(ready)
    assert all(row["passed"] is True for row in checks)

    changed = deepcopy(ready)
    changed["runtime_ownership_ready_score"] = 0
    failed = exp.runtime_ownership_gate_rows(changed)
    score = next(row for row in failed if row["check"] == "runtime_ownership_ready")
    assert score["expected_value"] == 1
    assert score["observed_value"] == 0
    assert score["passed"] is False

    changed = deepcopy(ready)
    changed["verdict_class"] = "blocked"
    assert any(row["passed"] is False for row in exp.runtime_ownership_gate_rows(changed))


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-DEVELOPMENT
def test_development_gate_requires_three_usable_untruncated_outputs() -> None:
    schedule = exp.build_development_schedule(frozen.build_schedule(frozen._fixture_cases()))
    rows = [_terminal_row(call) for call in schedule]
    gate = exp.reduce_development_gate(rows)
    assert gate["attempted"] == 4
    assert gate["usable_output_count"] == 4
    assert gate["capture_open"] is True

    rows[0] = _terminal_row(schedule[0], finish_reason="length")
    assert exp.reduce_development_gate(rows)["capture_open"] is True
    rows[1] = _terminal_row(schedule[1], reply="not json")
    failed = exp.reduce_development_gate(rows)
    assert failed["usable_output_count"] == 2
    assert failed["capture_open"] is False
    assert failed["terminal_class"] == "null"


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-RAW
def test_content_addressed_shard_binds_raw_row_before_next_call(tmp_path: Path) -> None:
    call = frozen.build_schedule(frozen._fixture_cases())[0]
    row = _terminal_row(call)
    receipt = exp.write_content_addressed_shard(tmp_path, "measured", row)

    shard = tmp_path / str(receipt["path"])
    assert shard.is_file()
    assert receipt["sha256"] == exp.sha256_file(shard)
    assert json.loads(shard.read_text(encoding="utf-8")) == row
    assert exp.write_content_addressed_shard(tmp_path, "measured", row) == receipt


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-REDUCTION
def test_independent_reducer_keeps_all_calls_and_separates_arm_endpoints() -> None:
    cases = frozen._fixture_cases()
    schedule = exp.build_frozen_schedule(cases)
    rows = [_terminal_row(call) for call in schedule]
    rows[-1] = frozen.parse_capture_row(
        frozen.build_raw_capture_row(
            schedule[-1],
            {
                "attempted": False,
                "terminal_state": "cancelled",
                "error": "capture_deadline",
                "raw_reply": "",
            },
            {},
        )
    )

    reduced = exp.reduce_capture(rows, cases)
    assert reduced["planned_call_count"] == 96
    assert reduced["unstarted_call_count"] == 1
    assert reduced["usable_output_count"] == 95
    assert set(reduced["arm_metrics"]) == {"free", "anchored"}
    for arm in ("free", "anchored"):
        metrics = reduced["arm_metrics"][arm]
        assert metrics["json_parse_validity"]["denominator"] == 48
        assert metrics["claim_coverage"]["denominator"] == 48
        assert metrics["relation_direction"]["denominator"] == 48
        assert metrics["negation_retention"]["denominator"] == 48
        assert metrics["quantifier_retention"]["denominator"] == 48
    assert reduced["official_scope"]["authority"] == "machine_annotation_not_truth"
    assert reduced["constructed_scope"]["authority"] == "constructed_exact_fixture"


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-REDUCTION
def test_relation_direction_and_quantifier_checks_use_constructed_authority_only() -> None:
    row = {
        "corpus": "constructed_challenge",
        "family": "count_mismatch",
        "answer": "Mira approved 3 permits.",
        "parse_valid": True,
        "decoded_triples": [
            {
                "subject": "Mira",
                "relation": "approved",
                "object": "3 permits",
                "qualifiers": ["3"],
            }
        ],
        "source_relation": ["Mira", "approved", "3 permits"],
    }
    endpoints = exp.independent_semantic_endpoints(row)
    assert endpoints == {
        "relation_direction_valid": True,
        "negation_retained": None,
        "quantifier_retained": True,
    }

    reversed_row = deepcopy(row)
    reversed_row["decoded_triples"][0]["subject"] = "3 permits"
    reversed_row["decoded_triples"][0]["object"] = "Mira"
    assert exp.independent_semantic_endpoints(reversed_row)["relation_direction_valid"] is False

    official = deepcopy(row)
    official["corpus"] = "official_test"
    assert exp.independent_semantic_endpoints(official) == {
        "relation_direction_valid": None,
        "negation_retained": None,
        "quantifier_retained": None,
    }


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-TERMINAL
def test_complete_fixture_replays_and_mutations_fail_closed() -> None:
    artifact = exp.build_artifact_for_test()
    assert exp.validate_artifact(artifact, require_terminal=True) == []
    assert exp.independent_reduce_artifact(artifact) == []
    assert artifact["schema"] == "carnot.exp7429.v651.anchored_capture.v1"
    assert artifact["milestone"] == "2026.09.651"
    assert artifact["extraction_capture_complete_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["usable_output_count"] == 96

    changed = deepcopy(artifact)
    changed["extraction_rows"][0]["raw_reply"] = "changed"
    changed["rows"] = deepcopy(changed["extraction_rows"])
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "raw_reply_hash_mismatch:0" in exp.validate_artifact(changed, require_terminal=True)

    changed = deepcopy(artifact)
    changed["usable_output_count"] = 95
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "usable_output_count_mismatch" in exp.validate_artifact(changed, require_terminal=True)


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-TERMINAL
def test_blocked_artifact_reports_no_model_load() -> None:
    failed = exp.runtime_ownership_gate_rows({})
    artifact = exp.build_blocked_artifact(failed)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["MODEL_SPECS"] == ["unsloth/Qwen3.8-27B-GGUF"]
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["invocation_counts"] == frozen.zero_counts()
    assert artifact["extraction_capture_complete_score"] == 0
    assert artifact["extraction_value_score"] is None


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-RAW
def test_pure_helpers_reject_drift_and_content_collision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="measured_schedule_count"):
        exp.build_development_schedule([])
    with pytest.raises(ValueError, match="fixed_panel_shape"):
        exp.reduce_capture([], [])

    row = {"value": 1}
    receipt = exp.write_content_addressed_shard(tmp_path, "rows", row)
    (tmp_path / str(receipt["path"])).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="content_address_collision"):
        exp.write_content_addressed_shard(tmp_path, "rows", row)

    assert exp._argument_text({"text": "anchored"}) == "anchored"
    assert exp._argument_text(1) == ""
    assert exp._date_argument(exp.RUN_DATE) == exp.RUN_DATE
    with pytest.raises(ValueError, match="date must be"):
        exp._date_argument("20260918")


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-REDUCTION
def test_semantic_endpoint_unknown_and_retention_branches() -> None:
    malformed = {
        "corpus": "constructed_challenge",
        "family": "negation",
        "answer": "Mira did not approve it.",
        "parse_valid": True,
        "decoded_triples": [{"qualifiers": ["not"]}],
        "source_relation": None,
    }
    endpoints = exp.independent_semantic_endpoints(malformed)
    assert endpoints["relation_direction_valid"] is None
    assert endpoints["negation_retained"] is True
    assert endpoints["quantifier_retained"] is None

    no_markers = deepcopy(malformed)
    no_markers["answer"] = "Mira approved it."
    assert exp.independent_semantic_endpoints(no_markers)["negation_retained"] is None

    invalid = deepcopy(malformed)
    invalid["parse_valid"] = False
    assert exp.independent_semantic_endpoints(invalid)["relation_direction_valid"] is None


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-REDUCTION
def test_value_gate_requires_known_non_degrading_endpoints() -> None:
    def metric(rate: float, known: int = 1) -> dict[str, object]:
        return {"rate_all_assigned": rate, "known": known}

    names = (
        "json_parse_validity",
        "claim_coverage",
        "relation_direction",
        "negation_retention",
        "quantifier_retention",
    )
    equal = {arm: {name: metric(0.5) for name in names} for arm in ("free", "anchored")}
    assert exp._value_score(equal) == 1

    unknown = deepcopy(equal)
    unknown["free"]["relation_direction"]["known"] = 0
    assert exp._value_score(unknown) == 0

    degraded = deepcopy(equal)
    degraded["anchored"]["claim_coverage"]["rate_all_assigned"] = 0.4
    assert exp._value_score(degraded) == 0


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-TERMINAL
def test_validator_rejects_declaration_model_and_row_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = exp.build_artifact_for_test()

    missing = deepcopy(artifact)
    missing.pop("schema")
    assert "missing_field:schema" in exp.validate_artifact(missing)

    changed = deepcopy(artifact)
    changed["schema"] = "wrong"
    changed["field_principles"] = {}
    changed["verdict_class"] = "unknown"
    changed["inference_substrate_class"] = "no_model_load"
    changed["model_invoked"] = False
    changed["rows"] = []
    errors = exp.validate_artifact(changed)
    assert "declaration_mismatch:schema" in errors
    assert "field_principles_mismatch" in errors
    assert "verdict_class_invalid" in errors
    assert "substrate_class_mismatch" in errors
    assert "model_invoked_mismatch" in errors
    assert "rows_alias_mismatch" in errors

    assert exp._model_spec_errors({"model_invoked": True, "model_specs": []}) == [
        "resolved_model_spec_missing"
    ]
    bad_spec = deepcopy(artifact)
    bad_spec["model_specs"] = [
        {
            "hf_id": "wrong",
            "quantization": "wrong",
            "native_chat_template": False,
            "decoding": {"max_new_tokens": 1, "retry_budget": 1},
        }
    ]
    model_errors = exp._model_spec_errors(bad_spec)
    assert "resolved_model_spec_mismatch:hf_id" in model_errors
    assert "resolved_model_spec_missing:revision" in model_errors
    assert "resolved_model_decoding_mismatch" in model_errors

    blocked = exp.build_blocked_artifact(exp.runtime_ownership_gate_rows({}))
    blocked["extraction_rows"] = [{}]
    blocked["rows"] = [{}]
    blocked["honest_verdict"] = "complete_wrong"
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    blocked_errors = exp.validate_artifact(blocked)
    assert "blocked_rows_or_score_invalid" in blocked_errors
    assert "blocked_verdict_prefix_invalid" in blocked_errors

    short = deepcopy(artifact)
    short["extraction_rows"] = short["extraction_rows"][:-1]
    short["rows"] = deepcopy(short["extraction_rows"])
    short["reproducibility_checksum"] = exp.artifact_checksum(short)
    assert "extraction_row_count_mismatch" in exp.validate_artifact(short)

    duplicate = deepcopy(artifact)
    duplicate["extraction_rows"][0]["case_id"] = duplicate["extraction_rows"][2]["case_id"]
    duplicate["extraction_rows"][1]["case_id"] = duplicate["extraction_rows"][2]["case_id"]
    duplicate["rows"] = deepcopy(duplicate["extraction_rows"])
    duplicate["reproducibility_checksum"] = exp.artifact_checksum(duplicate)
    assert "case_identity_mismatch" in exp.validate_artifact(duplicate)

    broken_row = deepcopy(artifact)
    broken_row["extraction_rows"][0] = "bad"
    broken_row["rows"] = deepcopy(broken_row["extraction_rows"])
    broken_row["reproducibility_checksum"] = exp.artifact_checksum(broken_row)
    assert "row_shape:0" in exp.validate_artifact(broken_row)

    not_persisted = deepcopy(artifact)
    not_persisted["extraction_rows"][0]["persisted_before_parse"] = False
    not_persisted["rows"] = deepcopy(not_persisted["extraction_rows"])
    not_persisted["reproducibility_checksum"] = exp.artifact_checksum(not_persisted)
    assert "raw_not_persisted_before_parse:0" in exp.validate_artifact(not_persisted)

    monkeypatch.setattr(
        exp, "reduce_capture", lambda *_args: (_ for _ in ()).throw(ValueError("x"))
    )
    assert any(
        error.startswith("independent_reduction_failed:ValueError")
        for error in exp.validate_artifact(artifact)
    )


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-TERMINAL
def test_validator_rejects_budget_development_and_terminal_receipt_mutations() -> None:
    artifact = exp.build_artifact_for_test()

    changed = deepcopy(artifact)
    changed["sample_size_budget"]["attempted"] = 95
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "sample_size_budget_mismatch" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["development_rows"] = changed["development_rows"][:-1]
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "development_row_count_mismatch" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["development_gate"]["usable_output_count"] = 3
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "development_gate_mismatch" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["validation_receipts"] = []
    changed["extraction_capture_complete_score"] = 1
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    terminal_errors = exp.validate_artifact(changed, require_terminal=True)
    assert any(
        error.startswith("required_validation_receipts_missing:") for error in terminal_errors
    )
    assert "extraction_capture_complete_score_mismatch" in terminal_errors

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "wrong"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)


# REQ-VERIFY-7429; SCENARIO-VERIFY-7429-RAW
def test_native_row_normalizer_and_receipt_set_are_exact() -> None:
    schedule = exp.build_frozen_schedule(frozen._fixture_cases())[0]
    raw = _terminal_row(schedule)
    normalized = exp._normalize_raw_row(schedule, raw)
    assert normalized["call_id"] == schedule["call_id"]

    filler = exp._normalize_raw_row(
        schedule,
        {
            "attempted": False,
            "terminal_state": "cancelled",
            "error": "deadline",
            "runtime_identity_receipt": {"pid": 1},
        },
    )
    assert filler["attempted"] is False
    assert filler["runtime_identity_receipt"] == {"pid": 1}

    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in ("one", "two")
    ]
    assert exp._receipts_pass(receipts, ("one", "two")) is True
    receipts.append({"name": "one", "passed": True, "exit_code": 0, "timed_out": False})
    assert exp._receipts_pass(receipts, ("one", "two")) is False
