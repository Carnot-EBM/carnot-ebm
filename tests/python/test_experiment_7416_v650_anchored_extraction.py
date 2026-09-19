"""Tests for the bounded source-anchored extraction capture."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7416_v650_anchored_extraction as exp


def _official_rows(count: int = 30) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    predictors: list[dict[str, object]] = []
    evaluators: list[dict[str, object]] = []
    for index in range(count):
        sentence = f"Person {index} founded Place {index} in 20{index % 10:02d}."
        prefix = f"Background {index}. "
        answer = prefix + sentence
        predictors.append(
            {
                "row_key": f"row-{index:03d}",
                "group_id": f"group-{index:03d}",
                "partition": "final_test",
                "question": "Who founded the place and when?",
                "context": sentence,
                "answer": answer,
                "sentence": sentence,
            }
        )
        evaluators.append(
            {
                "row_key": f"row-{index:03d}",
                "group_id": f"group-{index:03d}",
                "partition": "final_test",
                "label": index % 2,
                "label_authority": "machine_annotation",
            }
        )
    return predictors, evaluators


def _challenge_manifest() -> dict[str, object]:
    predictors: list[dict[str, object]] = []
    evaluators: list[dict[str, object]] = []
    families = (
        "negation",
        "subject_object_reversal",
        "comparator_reversal",
        "time_qualifier",
        "unit_mismatch",
        "count_mismatch",
        "omitted_condition",
        "coreference_ambiguity",
    )
    for pair_index, family in enumerate(families, start=1):
        pair_id = f"pair-{pair_index:02d}-{family}"
        for member in ("base", "contrast", "control"):
            answer = f"Agent {pair_index} {member} value {pair_index}."
            case_id = f"{pair_id}-{member}"
            predictors.append(
                {
                    "case_id": case_id,
                    "pair_id": pair_id,
                    "case_kind": (
                        "equivalent_control" if member == "control" else "minimal_pair_member"
                    ),
                    "family": family,
                    "question": "Extract the relation and preserve its qualifiers.",
                    "source": f"Agent {pair_index} base value {pair_index}.",
                    "answer": answer,
                }
            )
            evaluators.append(
                {
                    "case_id": case_id,
                    "answer_span": [0, len(answer)],
                    "answer_span_text": answer,
                    "authority": "constructed_source_defined_fixture",
                    "expected_scope": member,
                    "expected_verdict": (
                        "unsupported_or_ambiguous" if member == "contrast" else "supported"
                    ),
                    "source_relation": [f"Agent {pair_index}", "value", str(pair_index)],
                }
            )
    return {
        "schema": "carnot.exp7412.challenge_manifest.v1",
        "case_count": 24,
        "pair_count": 8,
        "training_use": "prohibited",
        "external_accuracy_evidence": False,
        "predictor_records": predictors,
        "evaluator_records": evaluators,
    }


def _response_for(call: dict[str, object]) -> dict[str, object]:
    answer = str(call["answer"])
    subject = answer.split()[0]
    subject_start = answer.index(subject)
    object_text = answer.rstrip(".").split()[-1]
    object_start = answer.rfind(object_text)
    if call["arm"] == "free":
        payload = {
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
        payload = {
            "triples": [
                {
                    "subject": {
                        "text": subject,
                        "span": [subject_start, subject_start + len(subject)],
                    },
                    "relation": "mentions",
                    "object": {
                        "text": object_text,
                        "span": [object_start, object_start + len(object_text)],
                    },
                    "qualifiers": [],
                }
            ]
        }
    raw_reply = json.dumps(payload, sort_keys=True)
    return {
        "raw_request": {"messages": [{"role": "user", "content": call["prompt"]}]},
        "raw_response": {"choices": [{"message": {"content": raw_reply}}]},
        "raw_reply": raw_reply,
        "attempted": True,
        "terminal_state": "response",
        "error": None,
        "finish_reason": "stop",
        "prompt_tokens": 20,
        "completion_tokens": 12,
        "latency_s": 0.25,
    }


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-SELECTION
def test_label_blind_selection_and_two_arm_schedule() -> None:
    predictors, evaluators = _official_rows()
    selected = exp.select_official_cases(predictors)
    assert len(selected) == 24
    expected = sorted(
        predictors,
        key=lambda row: exp.canonical_hash({"seed": exp.RANDOM_SEED, "row_key": row["row_key"]}),
    )[:24]
    assert [row["row_key"] for row in selected] == [row["row_key"] for row in expected]

    changed_labels = deepcopy(evaluators)
    for row in changed_labels:
        row["label"] = 1 - int(row["label"])
    cases_a = exp.build_cases(selected, evaluators, _challenge_manifest())
    cases_b = exp.build_cases(selected, changed_labels, _challenge_manifest())
    assert [row["case_id"] for row in cases_a] == [row["case_id"] for row in cases_b]

    schedule = exp.build_schedule(cases_a)
    assert len(schedule) == 96
    assert {row["arm"] for row in schedule} == {"free", "anchored"}
    assert all(row["max_new_tokens"] == 384 for row in schedule)
    for case in cases_a:
        calls = [row for row in schedule if row["case_id"] == case["case_id"]]
        assert len(calls) == 2
        assert calls[0]["arm"] != calls[1]["arm"]
        assert "expected_verdict" not in calls[0]["prompt"]
        assert "source_relation" not in calls[0]["prompt"]
        assert str(case["question"]) in calls[0]["prompt"]
        assert str(case["source_sentence"]) in calls[0]["prompt"]
        assert str(case["answer"]) in calls[0]["prompt"]


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-RAW
def test_raw_capture_is_hashed_before_parser_and_bad_json_is_not_repaired() -> None:
    predictors, evaluators = _official_rows()
    case = exp.build_cases(
        exp.select_official_cases(predictors), evaluators, _challenge_manifest()
    )[0]
    call = exp.build_schedule([case])[0]
    response = _response_for(call)
    raw = exp.build_raw_capture_row(call, response, {"owned_by_task": True})
    assert raw["parse_status"] == "not_parsed"
    assert raw["raw_request_sha256"].startswith("sha256:")
    assert raw["raw_response_sha256"].startswith("sha256:")

    parsed = exp.parse_capture_row(raw)
    assert parsed["parse_valid"] is True
    assert parsed["coverage_valid"] is True
    malformed = deepcopy(raw)
    malformed["raw_reply"] = "not json"
    malformed["raw_reply_sha256"] = exp.canonical_hash("not json")
    invalid = exp.parse_capture_row(malformed)
    assert invalid["parse_valid"] is False
    assert invalid["parse_error"] == "json_decode_error"
    assert invalid["retry_count"] == 0


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-ENDPOINTS
def test_free_and_anchored_parsers_keep_span_and_semantic_endpoints_separate() -> None:
    answer = "Mira did not approve the permit in 2021."
    free = exp.parse_extraction(
        json.dumps(
            {
                "triples": [
                    {
                        "subject": "Mira",
                        "relation": "approve",
                        "object": "permit",
                        "qualifiers": ["not", "2021"],
                    }
                ]
            }
        ),
        arm="free",
        answer=answer,
    )
    anchored = exp.parse_extraction(
        json.dumps(
            {
                "triples": [
                    {
                        "subject": {"text": "Mira", "span": [0, 4]},
                        "relation": "approve",
                        "object": {"text": "permit", "span": [25, 31]},
                        "qualifiers": [
                            {"text": "not", "span": [9, 12]},
                            {"text": "2021", "span": [35, 39]},
                        ],
                    }
                ]
            }
        ),
        arm="anchored",
        answer=answer,
    )
    assert free["parse_valid"] is True
    assert free["argument_anchoring_valid"] is True
    assert anchored["parse_valid"] is True
    assert anchored["argument_anchoring_valid"] is True
    assert anchored["qualifier_anchoring_valid"] is True
    assert anchored["semantic_judgment"] == "unknown"

    forged = json.dumps(
        {
            "triples": [
                {
                    "subject": {"text": "Mira", "span": [1, 5]},
                    "relation": "approve",
                    "object": {"text": "permit", "span": [25, 31]},
                    "qualifiers": [],
                }
            ]
        }
    )
    bad = exp.parse_extraction(forged, arm="anchored", answer=answer)
    assert bad["parse_valid"] is True
    assert bad["argument_anchoring_valid"] is False

    with pytest.raises(ValueError, match="unsupported_arm"):
        exp.parse_extraction("{}", arm="repair", answer=answer)


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-ENDPOINTS
def test_reducer_uses_all_calls_and_eight_pairs_as_denominators() -> None:
    predictors, evaluators = _official_rows()
    cases = exp.build_cases(
        exp.select_official_cases(predictors), evaluators, _challenge_manifest()
    )
    schedule = exp.build_schedule(cases)
    rows = [
        exp.parse_capture_row(exp.build_raw_capture_row(call, _response_for(call), {}))
        for call in schedule
    ]
    rows[-1]["terminal_state"] = "failed"
    rows[-1]["parse_valid"] = False
    reduced = exp.reduce_extractions(rows, cases)
    assert reduced["planned_call_count"] == 96
    assert reduced["assigned_case_count"] == 48
    assert reduced["independent_constructed_pair_count"] == 8
    assert len(reduced["semantic_pair_rows"]) == 8
    assert all(row["independent_unit_weight"] == 1 for row in reduced["semantic_pair_rows"])
    assert reduced["endpoint_metrics"]["json_parse_validity"]["denominator"] == 96
    assert reduced["endpoint_metrics"]["coverage"]["denominator"] == 96
    assert reduced["endpoint_metrics"]["latency_s"]["denominator"] == 96
    assert reduced["endpoint_metrics"]["semantic_fidelity"]["unknown"] == 96
    assert reduced["failed_call_count"] == 1


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-PROVENANCE
def test_substrate_class_tracks_owned_events_without_historical_counts() -> None:
    assert exp.substrate_class_from_counts(exp.zero_counts()) == "no_model_load"
    load_only = exp.zero_counts()
    load_only["model_loads_attempted"] = 1
    load_only["model_loads_completed"] = 1
    assert exp.substrate_class_from_counts(load_only) == "model_load_no_generation"
    generated = deepcopy(load_only)
    generated["generation_calls_attempted"] = 1
    generated["generation_calls_completed"] = 1
    assert exp.substrate_class_from_counts(generated) == "model_bounded_generation"


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-TERMINAL
def test_artifact_fixture_replays_and_mutations_fail_closed() -> None:
    artifact = exp.build_artifact_for_test()
    assert exp.validate_artifact(artifact, require_terminal=True) == []
    assert artifact["extraction_capture_complete_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["verifier_is_oracle"] is True

    changed = deepcopy(artifact)
    changed["sample_size_budget"]["planned"] = 95
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "sample_size_budget_mismatch" in exp.validate_artifact(changed, require_terminal=True)

    changed = deepcopy(artifact)
    changed["extraction_rows"][0]["raw_reply"] = "changed"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "raw_reply_hash_mismatch:0" in exp.validate_artifact(changed, require_terminal=True)

    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = []
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "model_specs_invalid" in exp.validate_artifact(changed, require_terminal=True)


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-TERMINAL
def test_blocked_artifact_has_no_model_attempts() -> None:
    failed = exp.gate_row(
        "source_feature_protocol_ready",
        "results/experiment_7412_v650_source_features.json",
        "source_feature_protocol_ready_score",
        "==",
        1,
        None,
        principle="Only a ready producer can authorize capture.",
    )
    artifact = exp.build_blocked_artifact([failed])
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == exp.zero_counts()
    assert artifact["extraction_capture_complete_score"] == 0
    assert artifact["gate_check_summary"]["observed_value"] is None
    assert exp.independent_reduce_artifact(artifact) == []


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-PROVENANCE
def test_runtime_preflight_accepts_more_than_one_available_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited = exp.gate_row(
        "one_owned_rtx3090_slot",
        "legacy_exact_count_gate",
        "owned_runtime_receipt",
        "==",
        {"minimum_rtx3090_slots": 1},
        {"minimum_rtx3090_slots": 2},
        principle="The inherited gate is deliberately too strict.",
    )
    monkeypatch.setattr(exp.canary, "_runtime_preconditions", lambda *_args: [inherited])
    context = {
        "available_gpu_uuids": ["gpu-1", "gpu-2"],
        "model_spec": {"decoding": {}},
    }
    checks = exp._runtime_preconditions(exp.REPO_ROOT, context, 0.0)
    slot_checks = [row for row in checks if row["check"] == "one_owned_rtx3090_slot"]
    assert len(slot_checks) == 1
    assert slot_checks[0]["operator"] == ">="
    assert slot_checks[0]["observed_value"] == 2
    assert slot_checks[0]["passed"] is True


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-RAW
def test_compare_and_gate_reject_unknown_operators() -> None:
    assert exp.compare("in", "null", ["null", "positive"]) is True
    assert exp.compare(">=", 1, 1) is True
    with pytest.raises(ValueError, match="unsupported_operator"):
        exp.compare("!=", 1, 2)

    rejected = exp.gate_row(
        "bad_operator",
        "fixture",
        "value",
        "!=",
        1,
        2,
        principle="An undeclared operator cannot pass a gate.",
    )
    assert rejected["passed"] is False


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-PROVENANCE
def test_file_receipts_are_atomic_and_fail_closed(tmp_path: Path) -> None:
    payload = {"answer": 42}
    target = tmp_path / "receipt.json"
    exp.atomic_json(target, payload)
    assert exp.load_object(target) == payload
    assert exp.sha256_file(target).startswith("sha256:")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    assert exp.load_object(invalid) == {}
    assert exp.load_object(tmp_path / "missing.json") == {}
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    assert exp.load_object(sequence) == {}


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-SELECTION
def test_case_construction_rejects_incomplete_or_ambiguous_sources() -> None:
    predictors, evaluators = _official_rows()
    selected = exp.select_official_cases(predictors)
    challenge = _challenge_manifest()

    with pytest.raises(ValueError, match="fewer_than_24"):
        exp.select_official_cases(predictors[:23])
    with pytest.raises(ValueError, match="official_case_count"):
        exp.build_cases(selected[:23], evaluators, challenge)
    with pytest.raises(ValueError, match="official_authority_missing"):
        exp.build_cases(selected, evaluators[1:], challenge)
    with pytest.raises(ValueError, match="challenge_manifest_shape"):
        exp.build_cases(selected, evaluators, {})

    changed = deepcopy(challenge)
    changed["predictor_records"].pop()
    with pytest.raises(ValueError, match="challenge_case_count"):
        exp.build_cases(selected, evaluators, changed)

    changed = deepcopy(challenge)
    changed["predictor_records"][0] = "invalid"
    with pytest.raises(ValueError, match="challenge_predictor_shape"):
        exp.build_cases(selected, evaluators, changed)

    changed = deepcopy(challenge)
    changed["evaluator_records"][0]["case_id"] = changed["evaluator_records"][1]["case_id"]
    with pytest.raises(ValueError, match="challenge_authority_missing"):
        exp.build_cases(selected, evaluators, changed)

    duplicated = deepcopy(selected)
    duplicated[1]["row_key"] = duplicated[0]["row_key"]
    with pytest.raises(ValueError, match="case_identity"):
        exp.build_cases(duplicated, evaluators, challenge)

    joined = exp.build_cases(selected, evaluators, challenge)
    with pytest.raises(ValueError, match="unsupported_arm"):
        exp._prompt(joined[0], "repair")


def _valid_free_triple() -> dict[str, object]:
    return {
        "subject": "Mira",
        "relation": "approve",
        "object": "permit",
        "qualifiers": [],
    }


def _valid_anchored_triple() -> dict[str, object]:
    return {
        "subject": {"text": "Mira", "span": [0, 4]},
        "relation": "approve",
        "object": {"text": "permit", "span": [25, 31]},
        "qualifiers": [],
    }


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-RAW
@pytest.mark.parametrize(
    ("arm", "raw_reply", "expected_error"),
    [
        ("free", "```json\n{}\n```", "markdown_fence_forbidden"),
        ("free", "[]", "top_level_schema"),
        ("free", '{"triples":{}}', "triples_not_list"),
        ("free", '{"triples":[[]]}', "triple_schema"),
        (
            "free",
            json.dumps({"triples": [{**_valid_free_triple(), "relation": 1}]}),
            "triple_field_type",
        ),
        (
            "free",
            json.dumps({"triples": [{**_valid_free_triple(), "subject": 1}]}),
            "free_argument_type",
        ),
        (
            "free",
            json.dumps({"triples": [{**_valid_free_triple(), "qualifiers": [1]}]}),
            "free_qualifier_type",
        ),
        (
            "anchored",
            json.dumps({"triples": [{**_valid_anchored_triple(), "subject": "Mira"}]}),
            "anchored_argument_schema",
        ),
        (
            "anchored",
            json.dumps(
                {
                    "triples": [
                        {
                            **_valid_anchored_triple(),
                            "subject": {"text": 1, "span": [0, 4]},
                        }
                    ]
                }
            ),
            "anchored_argument_text_type",
        ),
        (
            "anchored",
            json.dumps({"triples": [{**_valid_anchored_triple(), "qualifiers": ["not"]}]}),
            "anchored_qualifier_schema",
        ),
        (
            "anchored",
            json.dumps(
                {
                    "triples": [
                        {
                            **_valid_anchored_triple(),
                            "qualifiers": [{"text": 1, "span": [9, 12]}],
                        }
                    ]
                }
            ),
            "anchored_qualifier_text_type",
        ),
    ],
)
def test_parser_rejects_each_undeclared_transport_shape(
    arm: str, raw_reply: str, expected_error: str
) -> None:
    result = exp.parse_extraction(
        raw_reply,
        arm=arm,
        answer="Mira did not approve the permit in 2021.",
    )
    assert result["parse_valid"] is False
    assert result["parse_error"] == expected_error


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-ENDPOINTS
def test_invalid_span_and_missing_free_text_remain_parseable_but_unanchored() -> None:
    anchored = _valid_anchored_triple()
    anchored["subject"] = {"text": "Mira", "span": [0]}
    result = exp.parse_extraction(
        json.dumps({"triples": [anchored]}),
        arm="anchored",
        answer="Mira did not approve the permit in 2021.",
    )
    assert result["parse_valid"] is True
    assert result["argument_anchoring_valid"] is False

    free = _valid_free_triple()
    free["subject"] = ""
    result = exp.parse_extraction(
        json.dumps({"triples": [free]}),
        arm="free",
        answer="Mira did not approve the permit in 2021.",
    )
    assert result["parse_valid"] is True
    assert result["argument_anchoring_valid"] is False


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-RAW
def test_capture_without_response_and_wrong_panel_shape_fail_closed() -> None:
    predictors, evaluators = _official_rows()
    case = exp.build_cases(
        exp.select_official_cases(predictors), evaluators, _challenge_manifest()
    )[0]
    call = exp.build_schedule([case])[0]
    response = _response_for(call)
    response["terminal_state"] = "failed"
    parsed = exp.parse_capture_row(exp.build_raw_capture_row(call, response, {}))
    assert parsed["parse_error"] == "no_terminal_response"
    with pytest.raises(ValueError, match="fixed_panel_shape"):
        exp.reduce_extractions([], [])


def _rechecksum(artifact: dict[str, object]) -> None:
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)


def _replace_rows(artifact: dict[str, object], rows: list[object]) -> None:
    artifact["extraction_rows"] = rows
    artifact["rows"] = deepcopy(rows)
    _rechecksum(artifact)


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-TERMINAL
@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda value: value.__setitem__("milestone", "wrong"), "declaration_mismatch:milestone"),
        (lambda value: value.__setitem__("verdict_class", "unknown"), "verdict_class_invalid"),
        (lambda value: value["field_principles"].pop("schema"), "field_principles_mismatch"),
        (
            lambda value: value.__setitem__("inference_substrate_class", "no_model_load"),
            "substrate_class_mismatch",
        ),
        (lambda value: value.__setitem__("model_invoked", False), "model_invoked_mismatch"),
        (
            lambda value: value.__setitem__("independent_constructed_pair_count", 7),
            "constructed_pair_count_mismatch",
        ),
        (lambda value: value["semantic_pair_rows"].pop(), "semantic_pair_rows_mismatch"),
        (lambda value: value["raw_capture_manifest"].pop(), "raw_capture_manifest_count_mismatch"),
        (
            lambda value: value.__setitem__("validation_receipts", []),
            "required_validation_receipts_missing:",
        ),
        (
            lambda value: value.__setitem__("extraction_capture_complete_score", 0),
            "extraction_capture_complete_score_mismatch",
        ),
    ],
)
def test_artifact_validator_rejects_each_terminal_contract_mutation(
    mutation: object, expected_error: str
) -> None:
    artifact = exp.build_artifact_for_test()
    mutation(artifact)
    _rechecksum(artifact)
    assert any(
        error.startswith(expected_error)
        for error in exp.validate_artifact(artifact, require_terminal=True)
    )


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-TERMINAL
def test_artifact_validator_rejects_row_identity_and_shape_mutations() -> None:
    artifact = exp.build_artifact_for_test()
    rows = deepcopy(artifact["extraction_rows"])
    rows.pop()
    _replace_rows(artifact, rows)
    assert "extraction_row_count_mismatch" in exp.validate_artifact(artifact)

    artifact = exp.build_artifact_for_test()
    rows = deepcopy(artifact["extraction_rows"])
    rows[1]["call_id"] = rows[0]["call_id"]
    _replace_rows(artifact, rows)
    assert "call_identity_mismatch" in exp.validate_artifact(artifact)

    artifact = exp.build_artifact_for_test()
    rows = deepcopy(artifact["extraction_rows"])
    rows[2]["case_id"] = rows[0]["case_id"]
    rows[3]["case_id"] = rows[0]["case_id"]
    _replace_rows(artifact, rows)
    assert "case_identity_mismatch" in exp.validate_artifact(artifact)

    artifact = exp.build_artifact_for_test()
    rows = deepcopy(artifact["extraction_rows"])
    rows[0] = "invalid"
    _replace_rows(artifact, rows)
    assert "row_shape:0" in exp.validate_artifact(artifact)

    artifact = exp.build_artifact_for_test()
    artifact["extraction_rows"][0]["persisted_before_parse"] = False
    artifact["rows"] = deepcopy(artifact["extraction_rows"])
    _rechecksum(artifact)
    assert "raw_not_persisted_before_parse:0" in exp.validate_artifact(artifact)


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-TERMINAL
def test_blocked_and_checksum_claims_fail_closed_when_mutated() -> None:
    failed = exp.gate_row(
        "missing",
        "fixture",
        "field",
        "==",
        1,
        None,
        principle="The fixture deliberately blocks.",
    )
    blocked = exp.build_blocked_artifact([failed])
    blocked["extraction_rows"] = [{"unexpected": True}]
    blocked["rows"] = deepcopy(blocked["extraction_rows"])
    blocked["honest_verdict"] = "complete_invalid"
    _rechecksum(blocked)
    errors = exp.validate_artifact(blocked)
    assert "blocked_rows_or_score_invalid" in errors
    assert "blocked_verdict_prefix_invalid" in errors

    artifact = exp.build_artifact_for_test()
    artifact["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(artifact)


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-TERMINAL
def test_independent_reducer_reparses_rows_and_receipt_sets() -> None:
    artifact = exp.build_artifact_for_test()
    assert exp.independent_reduce_artifact(artifact) == []

    too_short = deepcopy(artifact)
    too_short["extraction_rows"].pop()
    assert exp.independent_reduce_artifact(too_short) == ["independent_row_count"]

    malformed = deepcopy(artifact)
    malformed["extraction_rows"][0] = "invalid"
    malformed["rows"] = deepcopy(malformed["extraction_rows"])
    _rechecksum(malformed)
    assert "independent_row_shape:0" in exp.independent_reduce_artifact(malformed)

    changed = deepcopy(artifact)
    changed["extraction_rows"][0]["parse_valid"] = False
    changed["rows"] = deepcopy(changed["extraction_rows"])
    _rechecksum(changed)
    assert "independent_parse_mismatch:0" in exp.independent_reduce_artifact(changed)

    receipts = [{"name": "one", "passed": True, "exit_code": 0, "timed_out": False}]
    assert exp._receipts_pass(receipts, ("one",)) is True
    receipts[0]["timed_out"] = True
    assert exp._receipts_pass(receipts, ("one",)) is False


# REQ-VERIFY-7416; SCENARIO-VERIFY-7416-PROVENANCE
def test_fixed_execution_date_guard() -> None:
    assert exp._date_argument("20260919") == "20260919"
    with pytest.raises(ValueError, match="date must be 20260919"):
        exp._date_argument("20260920")
