"""Tests for the V665 evidence-schema repair.

Spec refs: REQ-REPORT-7616 and SCENARIO-REPORT-7616-SCHEMA/REPLAY/
LOSSLESS/ROLES/E2E/TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7588_v663_evidence_protocol as v663
from carnot import experiment_7616_v665_evidence_schema as schema


ROOT = Path(__file__).resolve().parents[2]


def _record() -> dict:
    source = "Alpha is documented. Beta is absent."
    question = "Which statement supports the response?"
    response = "Alpha is documented. Gamma is unknown."
    return {
        "component_hash": schema.canonical_hash({"fixture": 1}),
        "role": "pilot",
        "source_role": "fit",
        "official_split": "train",
        "learning_partition": "pilot_only",
        "complete_source": source,
        "complete_question": question,
        "complete_answer": response,
        "source_sha256": schema.text_sha256(source),
        "question_sha256": schema.text_sha256(question),
        "answer_sha256": schema.text_sha256(response),
        "source_sentences": v663.segment_lossless(source, "S"),
        "question_sentences": v663.segment_lossless(question, "S"),
        "answer_sentences": v663.segment_lossless(response, "R"),
        "maximum_proposed_links": 6,
        "allowed_relations": ["contradicts", "supports", "unknown"],
        "labels_accessible": False,
        "raw_probability_accessible": False,
    }


def _linked(**updates: object) -> dict:
    row = {
        "response_sentence_id": "R001",
        "source_sentence_ids": ["S001"],
        "relation": "supports",
        "entity_type": "other",
        "abstention_reason": "",
    }
    row.update(updates)
    return row


def _valid_text() -> str:
    return json.dumps(
        [
            _linked(),
            {
                "response_sentence_id": "R002",
                "source_sentence_ids": [],
                "relation": "unknown",
                "entity_type": "none",
                "abstention_reason": "No source sentence supports this response sentence.",
            },
        ]
    )


def test_authority_renders_prompt_decoder_schema_and_installed_grammar() -> None:
    """SCENARIO-REPORT-7616-SCHEMA: one authority renders every decoder surface."""

    authority = schema.build_schema_authority(_record())
    prompt = schema.render_system_prompt(authority)
    request = schema.build_extraction_request(_record())
    grammar = schema.compile_decoder_grammar(authority)

    assert authority["entity_types"] == [
        "person",
        "organization",
        "location",
        "date",
        "numeric",
        "code",
        "other",
        "none",
    ]
    assert authority["pointer_fields"] == [
        "response_sentence_id",
        "source_sentence_ids",
        "relation",
        "entity_type",
        "abstention_reason",
    ]
    assert authority["maximum_pointers"] == 6
    assert all(value in prompt for value in authority["entity_types"])
    assert request["messages"][0]["content"] == prompt
    assert request["response_format"] == {
        "type": "json_object",
        "schema": authority["json_schema"],
    }
    assert "root" in grammar and "entity-type" in grammar
    assert authority["unsupported_decoder_keywords"] == ["uniqueItems"]
    assert authority["independent_validator_required"] is True


def test_canonical_input_is_single_copy_lossless_and_label_free() -> None:
    """SCENARIO-REPORT-7616-LOSSLESS: canonical arrays preserve exact bytes once."""

    record = _record()
    contract = schema.build_canonical_input(record)
    request = schema.build_extraction_request(record)
    visible = json.loads(request["messages"][1]["content"])

    assert set(visible) == {"source_sentences", "question_sentences", "response_sentences"}
    assert "complete_source" not in visible and "complete_answer" not in visible
    assert "label" not in json.dumps(visible).lower()
    assert "probability" not in json.dumps(visible).lower()
    assert schema.reconstruct_canonical_text(contract, "source") == record["complete_source"]
    assert schema.reconstruct_canonical_text(contract, "question") == record["complete_question"]
    assert schema.reconstruct_canonical_text(contract, "response") == record["complete_answer"]
    assert contract["reconstruction_hashes"] == {
        "source": record["source_sha256"],
        "question": record["question_sha256"],
        "response": record["answer_sha256"],
    }


def test_independent_validator_accepts_valid_output_and_hashes_features() -> None:
    """SCENARIO-REPORT-7616-E2E: valid pointers reduce to one hashed feature row."""

    record = _record()
    outcome = schema.validate_evidence_output(record, _valid_text(), finish_reason="stop")
    row = schema.build_hashed_feature_row(record, outcome)

    assert outcome["accepted"] is True
    assert outcome["error"] is None
    assert len(outcome["evidence"]) == 2
    assert row["component_hash"] == record["component_hash"]
    assert len(row["features"]) == 8
    assert row["feature_sha256"] == schema.canonical_hash(row["features"])
    assert row["censored"] is False


@pytest.mark.parametrize(
    ("payload", "finish_reason", "expected"),
    [
        ([_linked(entity_type="code_snippet")], "stop", "evidence_output_value_invalid"),
        ([_linked(source_sentence_ids=["S999"])], "stop", "source_sentence_id_invalid"),
        ([_linked(response_sentence_id="R999")], "stop", "response_sentence_id_invalid"),
        ([{**_linked(), "extra": True}], "stop", "evidence_output_fields_invalid"),
        ([_linked()] * 7, "stop", "evidence_link_budget_exceeded"),
        (
            [_linked(relation="unknown", abstention_reason="missing")],
            "stop",
            "unknown_relation_contract_invalid",
        ),
        (
            [_linked(source_sentence_ids=[])],
            "stop",
            "linked_relation_contract_invalid",
        ),
        ([_linked()], "length", "truncated_output"),
    ],
)
def test_independent_validator_rejects_every_required_failure(
    payload: list[dict], finish_reason: str, expected: str
) -> None:
    """SCENARIO-REPORT-7616-SCHEMA: every registered bad output fails closed."""

    outcome = schema.validate_evidence_output(
        _record(), json.dumps(payload), finish_reason=finish_reason
    )

    assert outcome["accepted"] is False
    assert outcome["error"] == expected
    assert outcome["evidence"] == []


def test_independent_validator_rejects_malformed_and_duplicate_links() -> None:
    """SCENARIO-REPORT-7616-SCHEMA: parser-only constraints remain fail closed."""

    malformed = schema.validate_evidence_output(_record(), "[{", finish_reason="stop")
    duplicate = schema.validate_evidence_output(
        _record(),
        json.dumps([_linked(source_sentence_ids=["S001", "S001"])]),
        finish_reason="stop",
    )

    assert malformed["error"] == "truncated_json"
    assert duplicate["error"] == "source_sentence_id_invalid"


def test_authority_rejects_unsupported_decoder_keywords() -> None:
    """SCENARIO-REPORT-7616-SCHEMA: unsupported keywords cannot be silently ignored."""

    authority = schema.build_schema_authority(_record())
    authority["json_schema"]["uniqueItems"] = True

    with pytest.raises(ValueError, match="unsupported_schema_keyword:uniqueItems"):
        schema.compile_decoder_grammar(authority)


def test_replay_preserves_all_eight_historical_rejections() -> None:
    """SCENARIO-REPORT-7616-REPLAY: old invalid output is never relabeled."""

    replay = schema.replay_exp7604_outputs(ROOT)

    assert len(replay) == 8
    assert [row["pilot_index"] for row in replay] == list(range(1, 9))
    assert all(
        row["original_parser_error"] == "ValueError:evidence_output_value_invalid" for row in replay
    )
    assert all(row["observed_error"] == "evidence_output_value_invalid" for row in replay)
    assert all(row["original_rejection_preserved"] is True for row in replay)
    assert all(row["semantic_verdict"] is None for row in replay)
    assert all(str(row["request_sha256"]).startswith("sha256:") for row in replay)
    assert all(str(row["response_sha256"]).startswith("sha256:") for row in replay)


def test_role_manifest_authenticates_every_sidecar_and_partition() -> None:
    """SCENARIO-REPORT-7616-ROLES: role custody preserves all frozen counts."""

    receipt = schema.authenticate_role_contract(ROOT)

    assert receipt["ready"] is True
    assert receipt["selection_salt"] == "v663-evidence-20260924"
    assert receipt["restored_group_count"] == 480
    assert receipt["selected_scored_group_count"] == 240
    assert receipt["role_counts"] == {
        "fit": 80,
        "tune": 20,
        "policy": 20,
        "online": 80,
        "evaluation": 40,
        "pilot": 8,
    }
    assert receipt["fit_partition_counts"] == {"optimization": 64, "anchor": 16}
    assert receipt["pilot_disjoint"] is True
    assert len(receipt["sidecars"]) == 12
    assert all(row["authenticated"] is True for row in receipt["sidecars"])


def test_guarded_lifecycle_authenticates_and_restarts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7616-E2E: existing lifecycle state survives a cold reload."""

    receipt = schema.run_guarded_update_restart_e2e(ROOT, tmp_path)

    assert receipt["authenticated"] is True
    assert receipt["restart_parity"] is True
    assert receipt["configuration_matches"] is True
    assert receipt["upstream_guarded_update_ready_score"] == 1
    assert receipt["before_parameter_hash"] == receipt["after_parameter_hash"]
    assert receipt["model_calls"] == 0
    assert receipt["fixture_positive_class"] == "circular_positive"


def test_freeze_sidecars_writes_exact_schema_config_and_inputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7616-LOSSLESS: frozen sidecars bind schema and input bytes."""

    records = [_record()]
    frozen = schema.freeze_protocol_sidecars(tmp_path, records)

    assert {row["name"] for row in frozen} == {
        "schema_authority",
        "capture_configuration",
        "canonical_inputs",
    }
    assert all(Path(row["path"]).is_file() for row in frozen)
    assert all(row["sha256"] == schema.sha256_file(Path(row["path"])) for row in frozen)
    canonical = json.loads((tmp_path / "canonical_inputs.json").read_text())
    assert canonical[0]["model_input"]["source_sentences"][0]["text"].startswith("Alpha")
    assert "complete_source" not in json.dumps(canonical)


def _passing_receipts() -> list[dict]:
    names = [*schema.validation_scope.REQUIRED_CHECK_NAMES, *schema.TERMINAL_CHECK_NAMES]
    return [
        {
            "name": name,
            "command": f"fixture:{name}",
            "command_argv": ["fixture", name],
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": schema.canonical_hash({"name": name}),
        }
        for name in names
    ]


def test_complete_artifact_is_null_ready_and_independently_reducible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7616-TERMINAL: protocol readiness stays separate from benefit."""

    artifact = schema.build_test_artifact(ROOT, tmp_path, _passing_receipts())
    errors = schema.validate_artifact(artifact, root=ROOT, require_terminal=True)
    reduction = schema.independent_reduce(artifact, root=ROOT)

    assert errors == []
    assert reduction["passed"] is True
    assert artifact["honest_verdict"] == "complete_null_evidence_schema_ready"
    assert artifact["verdict_class"] == "null"
    assert artifact["evidence_schema_ready_score"] == 1
    assert artifact["role_contract_ready_score"] == 1
    assert artifact["guarded_update_ready_score"] == 1
    assert artifact["fresh_confirmatory_claim_allowed"] is False
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert all(value == 0 for value in artifact["invocation_counts"].values())
    gates = {row["category"]: row for row in artifact["acceptance_gate_results"]}
    assert gates["validity"]["passed"] is True
    assert gates["readiness"]["passed"] is True
    assert gates["benefit"]["passed"] is False
    assert gates["freshness"]["passed"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["flagged_adversarial"] is False
    assert len(artifact["rows"]) == 8
    assert len(artifact["schema_case_rows"]) >= 10


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("MODEL_SPECS",), ["unsloth/Qwen3.8-27B-GGUF"], "model_specs_must_be_empty"),
        (("model_invoked",), True, "model_invoked_must_be_false"),
        (("evidence_schema_ready_score",), 0, "schema_ready_score_mismatch"),
        (("fresh_confirmatory_claim_allowed",), True, "freshness_invalid"),
        (("rows", 0, "observed_error"), None, "historical_replay_invalid"),
    ],
)
def test_terminal_validator_rejects_claim_and_row_mutations(
    tmp_path: Path, path: tuple[object, ...], value: object, expected: str
) -> None:
    """SCENARIO-REPORT-7616-TERMINAL: terminal claim mutations fail closed."""

    artifact = schema.build_test_artifact(ROOT, tmp_path, _passing_receipts())
    changed = deepcopy(artifact)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    changed["reproducibility_checksum"] = schema.reproducibility_checksum(changed)

    assert expected in schema.validate_artifact(changed, root=ROOT, require_terminal=True)


def test_blocked_artifact_has_complete_diagnostic() -> None:
    """REQ-REPORT-7616: external absence is blocked with exact gate operands."""

    failure = schema.precondition(
        "missing_input",
        "exp7604",
        ROOT / "missing.json",
        "filesystem.is_file",
        "eq",
        True,
        False,
    )
    artifact = schema.build_blocked_artifact([failure], duration_s=0.01)

    assert artifact["honest_verdict"] == "complete_blocked_missing_input"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["first_failure"] == failure
    assert artifact["sample_size_budget"]["censored"] == 8
    assert schema.validate_artifact(artifact, root=ROOT, require_terminal=False) == []


def test_canonical_contract_rejects_missing_changed_and_malformed_content() -> None:
    """SCENARIO-REPORT-7616-LOSSLESS: every byte-custody mutation fails closed."""

    missing = _record()
    missing["complete_source"] = None
    with pytest.raises(ValueError, match="canonical_source_absent"):
        schema.build_canonical_input(missing)
    changed = _record()
    changed["source_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="canonical_source_hash_invalid"):
        schema.build_canonical_input(changed)
    unidentified = _record()
    unidentified["component_hash"] = ""
    with pytest.raises(ValueError, match="component_hash_absent"):
        schema.build_canonical_input(unidentified)

    contract = schema.build_canonical_input(_record())
    with pytest.raises(ValueError, match="canonical_side_invalid"):
        schema.reconstruct_canonical_text(contract, "bad")
    absent = deepcopy(contract)
    absent["model_input"] = None
    with pytest.raises(ValueError, match="canonical_source_absent"):
        schema.reconstruct_canonical_text(absent, "source")
    malformed = deepcopy(contract)
    malformed["model_input"]["source_sentences"].append("bad")
    with pytest.raises(ValueError, match="canonical_source_row_invalid"):
        schema.reconstruct_canonical_text(malformed, "source")
    wrong_hash = deepcopy(contract)
    wrong_hash["reconstruction_hashes"]["source"] = "sha256:changed"
    with pytest.raises(ValueError, match="canonical_source_reconstruction_invalid"):
        schema.reconstruct_canonical_text(wrong_hash, "source")


def test_validator_rejects_nonarray_relation_and_feature_on_failure() -> None:
    """SCENARIO-REPORT-7616-SCHEMA: structural and semantic failures stay distinct."""

    nonarray = schema.validate_evidence_output(_record(), "{}", finish_reason="stop")
    relation = schema.validate_evidence_output(
        _record(), json.dumps([_linked(relation="maybe")]), finish_reason="stop"
    )
    assert nonarray["error"] == "evidence_output_not_array"
    assert relation["error"] == "evidence_relation_invalid"
    with pytest.raises(ValueError, match="accepted_evidence_required"):
        schema.build_hashed_feature_row(_record(), relation)


def test_json_loader_and_frozen_sidecar_conflicts_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7616-TERMINAL: non-object and changed sidecars are rejected."""

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        schema._load_json(scalar)

    path = tmp_path / "frozen.json"
    schema._write_frozen_json(path, {"value": 1})
    schema._write_frozen_json(path, {"value": 1})
    with pytest.raises(FileExistsError, match="frozen_sidecar_conflict"):
        schema._write_frozen_json(path, {"value": 2})


def test_missing_preconditions_stop_before_dependent_authentication(tmp_path: Path) -> None:
    """REQ-REPORT-7616: absent external inputs become explicit precondition failures."""

    checks, sources = schema.collect_preconditions(tmp_path)

    assert checks
    assert any(row["passed"] is False for row in checks)
    assert sources == []


def test_adversarial_flag_and_independent_nonmapping_branches() -> None:
    """SCENARIO-REPORT-7616-TERMINAL: reader flags and bad roots remain explicit."""

    assert schema._adversarial_flag([]) is False
    assert (
        schema._adversarial_flag(
            [{"name": "adversarial_verify", "output_tail": "Scanned 1 artifact(s); 1 flagged."}]
        )
        is True
    )
    assert schema.independent_reduce([], root=ROOT)["passed"] is False


def test_terminal_validator_rejects_remaining_contract_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7616-TERMINAL: all governed terminal claims fail closed."""

    artifact = schema.build_test_artifact(ROOT, tmp_path, _passing_receipts())
    mutations = [
        ({"schema": "bad"}, "experiment_identity_mismatch"),
        ({"honest_verdict": "null"}, "terminal_verdict_prefix_required"),
        ({"verdict_class": "unknown"}, "verdict_class_invalid"),
        ({"invocation_counts": {}}, "current_model_calls_nonzero"),
        ({"verifier_is_oracle": False}, "verifier_oracle_class_invalid"),
        ({"flagged_adversarial": None}, "adversarial_flag_invalid"),
        ({"reproducibility_checksum": "sha256:bad"}, "reproducibility_checksum_mismatch"),
        ({"field_principles": {}}, "field_principles_incomplete"),
        ({"inference_substrate_class": "aggregation"}, "substrate_class_invalid"),
        ({"role_contract_ready_score": 0}, "role_ready_score_mismatch"),
        ({"guarded_update_ready_score": 0}, "guarded_update_ready_score_mismatch"),
        ({"acceptance_gate_results": []}, "acceptance_gate_shape_invalid"),
        ({"current_work_receipt": {}}, "current_work_receipt_invalid"),
        ({"validation_receipts": []}, "terminal_validation_incomplete"),
    ]
    for updates, expected in mutations:
        changed = {**deepcopy(artifact), **updates}
        if expected != "reproducibility_checksum_mismatch":
            changed["reproducibility_checksum"] = schema.reproducibility_checksum(changed)
        assert expected in schema.validate_artifact(changed, root=ROOT, require_terminal=True)

    benefit = deepcopy(artifact)
    next(row for row in benefit["acceptance_gate_results"] if row["category"] == "benefit")[
        "passed"
    ] = True
    benefit["reproducibility_checksum"] = schema.reproducibility_checksum(benefit)
    assert "benefit_gate_invalid" in schema.validate_artifact(
        benefit, root=ROOT, require_terminal=True
    )

    bad_schema_path = deepcopy(artifact)
    bad_schema_path["schema_path"]["sha256"] = "sha256:bad"
    bad_schema_path["reproducibility_checksum"] = schema.reproducibility_checksum(bad_schema_path)
    assert "schema_path_hash_mismatch" in schema.validate_artifact(
        bad_schema_path, root=ROOT, require_terminal=True
    )

    blocked = schema.build_blocked_artifact(
        [
            schema.precondition(
                "missing", "upstream", tmp_path / "missing", "field", "eq", True, False
            )
        ],
        duration_s=0.0,
    )
    blocked["gate_check_summary"]["first_failure"] = {}
    blocked["reproducibility_checksum"] = schema.reproducibility_checksum(blocked)
    assert "blocked_gate_summary_incomplete" in schema.validate_artifact(
        blocked, root=ROOT, require_terminal=False
    )
    assert schema.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]


def test_cold_replay_manifests_commands_and_argument_guards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7616-TERMINAL: fresh readers and bounded plans are reproducible."""

    artifact = schema.build_test_artifact(ROOT, tmp_path, _passing_receipts())
    candidate = tmp_path / "candidate.json"
    schema.atomic_json(candidate, artifact)

    assert schema.cold_replay(candidate, root=ROOT)["passed"] is True
    manifest = schema.affected_file_manifest()
    assert manifest["requirement"] == "REQ-REPORT-7616"
    scoped = schema.build_validation_commands(ROOT, tmp_path / "private")
    terminal = schema.terminal_commands(ROOT, candidate)
    assert [row.name for row in scoped] == list(schema.validation_scope.REQUIRED_CHECK_NAMES)
    assert [row.name for row in terminal] == list(schema.TERMINAL_CHECK_NAMES)

    receipts = [{"log_path": str(ROOT / "relative.log"), "name": "one"}]
    assert schema._normalize_receipts(receipts, ROOT)[0]["log_path"] == "relative.log"
    assert schema._terminal_receipts_equal([], []) is True
    assert schema._terminal_receipts_equal([], [{"name": "extra"}]) is False
    span = schema._span("unit", 1.0, 1.0, 1, 1)
    assert span["completed_units"] == 1

    assert schema._date_argument(schema.RUN_DATE) == schema.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        schema._date_argument("20260101")
    assert schema._output_argument(schema.RESULT_PATH.as_posix()) == schema.RESULT_PATH
    with pytest.raises(Exception, match="output must be"):
        schema._output_argument("results/wrong.json")
    parsed = schema.parse_args(["--cold-replay", str(candidate)])
    assert parsed.cold_replay == candidate
