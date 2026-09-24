"""Focused tests for the V664 evidence extraction pilot.

Spec refs: REQ-VERIFY-7604 and SCENARIO-VERIFY-7604-AUTH/REQUESTS/PARSER/
COST/NULL/E2E.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7604_v664_evidence_pilot as pilot


def _record(index: int = 1) -> dict:
    return {
        "component_hash": f"sha256:component-{index}",
        "role": "pilot",
        "source_role": "fit",
        "official_split": "train",
        "learning_partition": "pilot_only",
        "complete_source": "Alpha is documented. Beta is absent.",
        "complete_question": "Which statement supports the answer?",
        "complete_answer": "Alpha is documented. Gamma is unknown.",
        "source_sha256": pilot.text_sha256("Alpha is documented. Beta is absent."),
        "question_sha256": pilot.text_sha256("Which statement supports the answer?"),
        "answer_sha256": pilot.text_sha256("Alpha is documented. Gamma is unknown."),
        "source_sentences": [
            {
                "sentence_id": "S001",
                "byte_start": 0,
                "byte_end": 21,
                "text": "Alpha is documented. ",
                "text_sha256": pilot.text_sha256("Alpha is documented. "),
                "boundary_kind": "whole_sentence",
            },
            {
                "sentence_id": "S002",
                "byte_start": 21,
                "byte_end": 36,
                "text": "Beta is absent.",
                "text_sha256": pilot.text_sha256("Beta is absent."),
                "boundary_kind": "whole_sentence",
            },
        ],
        "question_sentences": [
            {
                "sentence_id": "S001",
                "byte_start": 0,
                "byte_end": 36,
                "text": "Which statement supports the answer?",
                "text_sha256": pilot.text_sha256("Which statement supports the answer?"),
                "boundary_kind": "whole_sentence",
            }
        ],
        "answer_sentences": [
            {
                "sentence_id": "R001",
                "byte_start": 0,
                "byte_end": 21,
                "text": "Alpha is documented. ",
                "text_sha256": pilot.text_sha256("Alpha is documented. "),
                "boundary_kind": "whole_sentence",
            },
            {
                "sentence_id": "R002",
                "byte_start": 21,
                "byte_end": 38,
                "text": "Gamma is unknown.",
                "text_sha256": pilot.text_sha256("Gamma is unknown."),
                "boundary_kind": "whole_sentence",
            },
        ],
        "maximum_proposed_links": 6,
        "allowed_relations": ["contradicts", "supports", "unknown"],
        "labels_accessible": False,
        "raw_probability_accessible": False,
    }


def _proposal(response_id: str = "R001", source_id: str = "S001") -> dict:
    return {
        "response_sentence_id": response_id,
        "source_sentence_ids": [source_id],
        "relation": "supports",
        "entity_type": "other",
        "abstention_reason": "",
    }


def _pilot_row(index: int, *, usable: bool = True) -> dict:
    return {
        "pilot_index": index,
        "component_hash": f"sha256:component-{index}",
        "arm": "evidence_link",
        "usable_schema": usable,
        "invalid_pointer_accepted": False,
        "lossless_input": True,
        "parser_outcome": "usable_schema" if usable else "invalid_output",
        "censoring": "none" if usable else "invalid_output",
        "prompt_tokens": 100 + index,
        "output_tokens": 10,
        "prefill_s": 1.0,
        "decode_s": 2.0,
        "call_s": 3.5,
        "finish_reason": "stop",
        "seed": pilot.RANDOM_SEED,
        "direction": "higher_is_more_transport_usable",
        "numerator": int(usable),
        "denominator": 1,
        "provenance": {"source": "exp7602", "call_id": f"generation-{index}"},
    }


def _complete_artifact() -> dict:
    rows = [_pilot_row(index) for index in range(8)]
    projection = pilot.estimate_capture(
        rows, [100] * 120, model_load_s=20.0, validation_reserve_s=120.0
    )
    return pilot.build_complete_artifact(
        pilot_rows=rows,
        runtime_receipt={
            "transport_authenticated": True,
            "model_load_completed": True,
            "model_load_s": 20.0,
            "gpu_uuid": "GPU-test",
            "server_pid": 123,
            "offload_layers": {"loaded_layers": 65, "total_layers": 65},
        },
        source_hashes=[],
        fit_projection=projection,
        eval_projection=projection,
        duration_s=40.0,
        validation_receipts=[],
    )


def test_request_is_frozen_label_free_and_lossless() -> None:
    """SCENARIO-VERIFY-7604-REQUESTS: exact text enters one fixed request."""

    record = _record()
    request = pilot.build_extraction_request(record)

    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 512
    assert request["seed"] == 7_604_001
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    encoded = json.dumps(request, sort_keys=True)
    assert record["complete_source"] in encoded
    assert record["complete_question"] in encoded
    assert record["complete_answer"] in encoded
    assert "S001" in encoded and "R002" in encoded
    assert "label" not in encoded.lower()
    assert "probability" not in encoded.lower()
    assert pilot.validate_input_record(record) is True


def test_input_contract_rejects_label_access_and_byte_loss() -> None:
    """SCENARIO-VERIFY-7604-AUTH: predictor custody fails closed."""

    labeled = {**_record(), "label": 1}
    with pytest.raises(ValueError, match="predictor_label_access"):
        pilot.validate_input_record(labeled)
    changed = _record()
    changed["complete_source"] += " changed"
    with pytest.raises(ValueError, match="source_hash_invalid"):
        pilot.validate_input_record(changed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"labels_accessible": True}, "predictor_label_access"),
        ({"raw_probability_accessible": True}, "predictor_probability_access"),
        ({"maximum_proposed_links": 7}, "evidence_link_budget_changed"),
        ({"allowed_relations": ["supports"]}, "evidence_relations_changed"),
        ({"complete_question": ""}, "complete_question_absent"),
        ({"component_hash": ""}, "component_hash_absent"),
    ],
)
def test_input_contract_rejects_each_custody_change(mutation: dict, message: str) -> None:
    """SCENARIO-VERIFY-7604-AUTH: each frozen input operand fails closed."""

    record = {**_record(), **mutation}
    with pytest.raises(ValueError, match=message):
        pilot.validate_input_record(record)


def test_input_contract_rejects_bad_sentence_ids() -> None:
    """SCENARIO-VERIFY-7604-AUTH: duplicate or wrong-prefix IDs do not enter calls."""

    record = _record()
    record["answer_sentences"][1]["sentence_id"] = "S002"
    with pytest.raises(ValueError, match="answer_sentence_ids_invalid"):
        pilot.validate_input_record(record)


def test_parser_normalizes_valid_links_and_omissions() -> None:
    """SCENARIO-VERIFY-7604-PARSER: valid IDs normalize with explicit unknowns."""

    outcome = pilot.parse_extraction_response(_record(), json.dumps([_proposal()]), "stop")

    assert outcome["usable_schema"] is True
    assert outcome["parser_outcome"] == "usable_schema"
    assert outcome["invalid_pointer_accepted"] is False
    assert [row["response_sentence_id"] for row in outcome["evidence"]] == ["R001", "R002"]
    assert outcome["evidence"][1]["relation"] == "unknown"
    assert outcome["censoring"] == "none"


@pytest.mark.parametrize(
    ("text", "finish_reason", "expected"),
    [
        ("", "stop", "empty_output"),
        (json.dumps([_proposal(source_id="S999")]), "stop", "invalid_output"),
        (json.dumps([_proposal()]), "length", "truncated_output"),
        ("not-json", "stop", "invalid_output"),
    ],
)
def test_parser_retains_empty_bad_and_truncated_outcomes(
    text: str, finish_reason: str, expected: str
) -> None:
    """SCENARIO-VERIFY-7604-PARSER: failed output stays in the denominator."""

    outcome = pilot.parse_extraction_response(_record(), text, finish_reason)

    assert outcome["usable_schema"] is False
    assert outcome["parser_outcome"] == expected
    assert outcome["invalid_pointer_accepted"] is False
    assert outcome["censoring"] != "none"


def test_parser_rejects_non_array_json() -> None:
    """SCENARIO-VERIFY-7604-PARSER: JSON objects are not accepted as pointer arrays."""

    outcome = pilot.parse_extraction_response(_record(), "{}", "stop")
    assert outcome["parser_outcome"] == "invalid_output"


def test_readiness_requires_six_usable_rows_and_authenticated_transport() -> None:
    """SCENARIO-VERIFY-7604-NULL: readiness is a transport-only score."""

    rows = [_pilot_row(index, usable=index < 5) for index in range(8)]
    failed = pilot.reduce_pilot_rows(rows, transport_authenticated=True)
    rows[5]["usable_schema"] = True
    rows[5]["numerator"] = 1
    rows[5]["parser_outcome"] = "usable_schema"
    passed = pilot.reduce_pilot_rows(rows, transport_authenticated=True)
    unauthenticated = pilot.reduce_pilot_rows(rows, transport_authenticated=False)

    assert failed["usable_schema_count"] == 5
    assert failed["evidence_transport_ready_score"] == 0
    assert passed["usable_schema_count"] == 6
    assert passed["evidence_transport_ready_score"] == 1
    assert unauthenticated["evidence_transport_ready_score"] == 0
    with pytest.raises(ValueError, match="exactly_eight_pilot_rows_required"):
        pilot.reduce_pilot_rows(rows[:-1], transport_authenticated=True)
    bad = deepcopy(rows)
    bad[0]["invalid_pointer_accepted"] = True
    assert (
        pilot.reduce_pilot_rows(bad, transport_authenticated=True)["evidence_transport_ready_score"]
        == 0
    )
    duplicate = deepcopy(rows)
    duplicate[0]["component_hash"] = duplicate[1]["component_hash"]
    with pytest.raises(ValueError, match="pilot_components_not_disjoint"):
        pilot.reduce_pilot_rows(duplicate, transport_authenticated=True)


def test_capture_estimator_uses_upper_rates_and_fixed_budget() -> None:
    """SCENARIO-VERIFY-7604-COST: projections charge measured upper costs."""

    rows = [_pilot_row(index) for index in range(8)]
    estimate = pilot.estimate_capture(
        rows,
        target_prompt_tokens=[100] * 120,
        model_load_s=20.0,
        validation_reserve_s=120.0,
    )
    slow = pilot.estimate_capture(
        [{**row, "decode_s": 40.0, "call_s": 42.0} for row in rows],
        target_prompt_tokens=[1_000] * 120,
        model_load_s=20.0,
        validation_reserve_s=120.0,
    )

    assert estimate["group_count"] == 120
    assert estimate["output_tokens_per_group"] == 512
    assert estimate["uncertainty_multiplier"] == 1.25
    assert estimate["projected_s"] >= 140.0
    assert estimate["feasible_score"] == int(estimate["projected_s"] <= 3000.0)
    assert slow["feasible_score"] == 0
    with pytest.raises(ValueError, match="sample_size"):
        pilot.estimate_capture(rows[:-1], [100] * 120, model_load_s=1.0)
    with pytest.raises(ValueError, match="target_prompt_length"):
        pilot.estimate_capture(rows, [0] * 120, model_load_s=1.0)
    no_cost = [{**row, "prompt_tokens": 0, "prefill_s": -1.0, "decode_s": -1.0} for row in rows]
    with pytest.raises(ValueError, match="pilot_cost_components_absent"):
        pilot.estimate_capture(no_cost, [100] * 120, model_load_s=1.0)


def test_prompt_projection_uses_exact_request_bytes() -> None:
    """SCENARIO-VERIFY-7604-COST: target lengths derive from frozen request bytes."""

    rows = [{**_pilot_row(index), "request_bytes": 100} for index in range(8)]
    projected = pilot.prompt_token_projections([_record()], rows)
    exact_bytes = len(
        json.dumps(
            pilot.build_extraction_request(_record()),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    assert projected == [math.ceil(exact_bytes * 1.07)]
    with pytest.raises(ValueError, match="prompt_token_ratio_absent"):
        pilot.prompt_token_projections([_record()], [{**row, "request_bytes": 0} for row in rows])


def test_blocked_artifact_names_exact_failed_gate() -> None:
    """SCENARIO-VERIFY-7604-AUTH: a no-run block remains schema complete."""

    check = pilot.gate_row(
        "model_cache",
        upstream="cached_current_model",
        path="/abs/model.gguf",
        field="Q4_K_M",
        operator="exists",
        expected=True,
        observed=False,
        passed=False,
    )
    artifact = pilot.build_blocked_artifact(
        checks=[check],
        source_hashes=[],
        duration_s=0.25,
        reason="model_cache",
    )

    assert artifact["honest_verdict"] == "complete_blocked_model_cache"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["planned_inference_substrate_class"] == "model_bounded_generation"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["invocation_counts"]["generation_calls_attempted"] == 0
    assert artifact["gate_check_summary"]["first_failure"] == check
    assert pilot.validate_artifact(artifact) == []


def test_complete_artifact_reduces_rows_and_detects_mutation() -> None:
    """SCENARIO-VERIFY-7604-E2E: cold validation recomputes terminal fields."""

    artifact = _complete_artifact()

    assert artifact["honest_verdict"] == "complete_null_transport_feasibility_measured"
    assert artifact["verdict_class"] == "null"
    assert artifact["evidence_transport_ready_score"] == 1
    assert artifact["sample_size_budget"]["observed"] == 8
    assert artifact["verifier_is_oracle"] is True
    assert pilot.validate_artifact(artifact) == []
    changed = deepcopy(artifact)
    changed["evidence_transport_ready_score"] = 0
    assert "evidence_transport_ready_score_mismatch" in pilot.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in pilot.validate_artifact(changed)


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("schema",), "bad", "schema_invalid"),
        (("honest_verdict",), "null", "honest_verdict_not_terminal"),
        (("verdict_class",), "bad", "verdict_class_invalid"),
        (("flagged_adversarial",), None, "flagged_adversarial_invalid"),
        (("random_seed",), 1, "random_seed_invalid"),
        (("verifier_is_oracle",), False, "verifier_is_oracle_invalid"),
        (("field_principles",), {}, "field_principles_incomplete"),
        (("pilot_rows",), [], "pilot_rows_invalid"),
        (("sample_size_budget",), None, "sample_size_budget_invalid"),
        (("invocation_counts",), None, "invocation_counts_invalid"),
        (("inference_substrate_class",), None, "inference_substrate_class_invalid"),
        (("pilot_reduction",), {}, "pilot_reduction_mismatch"),
        (("MODEL_SPECS",), [], "model_specs_mandate_invalid"),
        (("invocation_counts", "generation_calls_attempted"), 7, "generation_call_count_invalid"),
        (("invocation_counts", "model_loads_attempted"), 0, "model_load_count_invalid"),
        (("planned_inference_substrate_class",), "bad", "planned_substrate_invalid"),
        (("inference_substrate_class",), "bad", "measured_substrate_invalid"),
        (("duration_s",), 1.0, "model_bounded_generation_duration_implausible"),
        (("fit_capture_feasible_score",), 1, "fit_capture_feasible_score_mismatch"),
        (("eval_capture_feasible_score",), 1, "eval_capture_feasible_score_mismatch"),
    ],
)
def test_cold_validator_rejects_each_governed_mutation(
    path: tuple[str, ...], value: object, expected: str
) -> None:
    """SCENARIO-VERIFY-7604-E2E: cold validation distrusts summary fields."""

    artifact = deepcopy(_complete_artifact())
    target = artifact
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert expected in pilot.validate_artifact(artifact)


def test_cold_validator_rejects_bad_rows_and_current_receipt() -> None:
    """SCENARIO-VERIFY-7604-E2E: row identity and owned receipt are recomputed."""

    artifact = _complete_artifact()
    artifact["pilot_rows"][0]["component_hash"] = artifact["pilot_rows"][1]["component_hash"]
    artifact["rows"] = artifact["pilot_rows"]
    assert any(
        error.startswith("pilot_reduction_invalid") for error in pilot.validate_artifact(artifact)
    )
    artifact = _complete_artifact()
    artifact["current_work_receipt"] = {"bad": True}
    assert any(
        error.startswith("current_work_receipt:") for error in pilot.validate_artifact(artifact)
    )


def test_blocked_validator_checks_substrate_counts_and_diagnostic() -> None:
    """SCENARIO-VERIFY-7604-AUTH: blocked records cannot hide started work."""

    check = pilot.gate_row(
        "cache",
        upstream="resolver",
        path="/missing",
        field="path",
        operator="exists",
        expected=True,
        observed=False,
        passed=False,
    )
    artifact = pilot.build_blocked_artifact(
        checks=[check], source_hashes=[], duration_s=1.0, reason="cache"
    )
    artifact["inference_substrate_class"] = "bad"
    artifact["MODEL_SPECS"] = pilot.MODEL_SPECS
    artifact["invocation_counts"]["generation_calls_attempted"] = 1
    artifact["gate_check_summary"]["first_failure"] = {}
    errors = pilot.validate_artifact(artifact)
    assert "blocked_substrate_invalid" in errors
    assert "blocked_invocations_nonzero" in errors
    assert "blocked_gate_diagnostic_incomplete" in errors
    counts = pilot._zero_invocations()
    counts.update({"model_loads_attempted": 1, "model_loads_failed": 1})
    load_failed = pilot.build_blocked_artifact(
        checks=[check],
        source_hashes=[],
        duration_s=12.0,
        reason="load",
        invocation_counts=counts,
        actual_substrate_class="model_load_no_generation",
    )
    assert pilot.validate_artifact(load_failed) == []


def test_cold_replay_and_independent_reducers(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7604-E2E: fresh readers reload exact persisted bytes."""

    complete = tmp_path / "complete.json"
    pilot.current_work_receipt.atomic_json(complete, _complete_artifact())
    assert pilot.cold_replay(complete, root=tmp_path)["valid"] is True
    assert pilot.independent_reduce_artifact(complete, root=tmp_path)["passed"] is True
    check = pilot.gate_row(
        "cache",
        upstream="resolver",
        path=str(tmp_path / "missing"),
        field="path",
        operator="exists",
        expected=True,
        observed=False,
        passed=False,
    )
    blocked = tmp_path / "blocked.json"
    pilot.current_work_receipt.atomic_json(
        blocked,
        pilot.build_blocked_artifact(
            checks=[check], source_hashes=[], duration_s=1.0, reason="cache"
        ),
    )
    reduced = pilot.independent_reduce_artifact(blocked, root=tmp_path)
    assert reduced["reduction"] == {"blocked_no_run": True, "pilot_count": 8}


def test_json_readers_and_hashes_reject_non_objects(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7604-AUTH: source readers accept object records only."""

    object_path = tmp_path / "object.json"
    object_path.write_text('{"a": 1}', encoding="utf-8")
    assert pilot.load_json(object_path) == {"a": 1}
    assert (
        pilot.sha256_file(object_path)
        == "sha256:" + hashlib.sha256(object_path.read_bytes()).hexdigest()
    )
    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        pilot.load_json(list_path)
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    assert pilot.load_jsonl(jsonl) == [{"a": 1}, {"b": 2}]
    jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_object_required"):
        pilot.load_jsonl(jsonl)


def test_raw_response_receipts_keep_costs_and_transport_failures(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7604-PARSER: raw call receipts retain timing and failure state."""

    request = pilot.build_extraction_request(_record())
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    response_path = tmp_path / "response.json"
    raw = {
        "choices": [
            {
                "message": {
                    "content": json.dumps([_proposal()]),
                    "reasoning_content": "",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 101, "completion_tokens": 11},
        "timings": {"prompt_ms": 1200.0, "predicted_ms": 2200.0},
    }
    response_bytes = json.dumps(raw).encode("utf-8")
    response_path.write_bytes(response_bytes)
    row = pilot._completed_row(
        index=1,
        record=_record(),
        request_payload=request,
        request_path=request_path,
        response_path=response_path,
        response_bytes=response_bytes,
        call_start_ns=1_000_000_000,
        call_end_ns=5_000_000_000,
    )
    assert row["prompt_tokens"] == 101
    assert row["output_tokens"] == 11
    assert row["prefill_s"] == 1.2
    assert row["decode_s"] == 2.2
    assert row["response_sha256"] == "sha256:" + hashlib.sha256(response_bytes).hexdigest()
    failed = pilot._failed_row(
        index=2,
        record=_record(2),
        request_payload=request,
        request_path=request_path,
        error=RuntimeError("transport"),
        call_start_ns=1,
        call_end_ns=11,
    )
    assert failed["parser_outcome"] == "transport_failure"
    assert failed["denominator"] == 1
    with pytest.raises(ValueError, match="response_object_required"):
        pilot._completed_row(
            index=1,
            record=_record(),
            request_payload=request,
            request_path=request_path,
            response_path=response_path,
            response_bytes=b"[]",
            call_start_ns=1,
            call_end_ns=2,
        )


def test_timing_token_and_event_helpers_cover_runtime_shapes(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7604-REQUESTS: runtime counters use observed server fields."""

    assert pilot._timing_value({"timings": {"x": 1000}}, "x", "y") == 1.0
    assert pilot._timing_value({"timings": {"y": 500}}, "x", "y") == 0.5
    assert pilot._timing_value({}, "x", "y") == 0.0
    assert pilot._token_count({"timings": {"x": 3}}, "x", "y") == 3
    assert pilot._token_count({"usage": {"y": 4}}, "x", "y") == 4
    assert pilot._token_count({}, "x", "y") == 0
    event = pilot._event("run", "call", "generation", "attempted")
    assert event["scope"] == "current" and event["owner_pid"] > 1
    log = tmp_path / "server.log"
    log.write_text(
        "- CUDA0 : card\nloading model\nwarming up the model with an empty run\nmodel loaded\n",
        encoding="utf-8",
    )
    fallback = pilot._offload_receipt(log, 17_000, {"loaded_layers": None})
    assert fallback["actual_offload"] is True
    assert fallback["warmup_calls"] == 1
    counted = pilot._offload_receipt(
        log,
        17_000,
        {"loaded_layers": 65, "total_layers": 65, "evidence": "stderr"},
    )
    assert counted["authentication_method"] == "reported_layer_count"
    assert pilot._offload_receipt(None, None, {"loaded_layers": None})["actual_offload"] is False


def test_validation_manifest_commands_and_receipt_reduction(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7604-E2E: validation scope and terminal readers stay fixed."""

    manifest = pilot.affected_validation_manifest()
    assert manifest["test_paths"] == [pilot.TEST_PATH.as_posix()]
    commands = pilot.terminal_commands(tmp_path, tmp_path / "candidate.json")
    assert [row.name for row in commands] == [
        "declared_entrypoint",
        "fresh_process_cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    receipts = [
        {
            "name": "reader",
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "log_sha256": "sha256:x",
        }
    ]
    assert pilot._all_passed(receipts) is True
    assert pilot._all_passed([]) is False
    assert pilot._all_passed([{**receipts[0], "timed_out": True}]) is False
    assert pilot._reader_outcomes(receipts)[0]["worktree"] == str(pilot.REPO_ROOT)
    checks: list[dict] = []
    pilot._check(
        checks,
        "x",
        upstream="u",
        path=tmp_path,
        field="f",
        operator="eq",
        expected=1,
        observed=1,
        passed=True,
    )
    assert checks[0]["passed"] is True


def test_current_receipt_source_hashes_and_cli_helpers(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7604-E2E: helper receipts bind files and read-only modes."""

    receipt = pilot._empty_current_receipt(10, 20, "cache")
    assert receipt["duration_s"] == pytest.approx(1e-8)
    assert receipt["MODEL_SPECS"] == []
    for relative in (pilot.MODULE_PATH, pilot.WRAPPER_PATH, pilot.TEST_PATH, pilot.SPEC_PATH):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative.as_posix(), encoding="utf-8")
    hashes: list[dict] = []
    pilot._add_changed_source_hashes(tmp_path, hashes)
    pilot._add_changed_source_hashes(tmp_path, hashes)
    assert len(hashes) == 4
    assert (
        pilot._source_hash_row(
            tmp_path / pilot.MODULE_PATH, producer="test", source_class="fixture"
        )["producer"]
        == "test"
    )
    args = pilot.parse_args(["--root", str(tmp_path), "--date", pilot.RUN_DATE])
    assert args.root == tmp_path
    assert pilot._argument_path(Path("x"), tmp_path) == tmp_path / "x"
    assert pilot._argument_path(tmp_path / "x", Path("/other")) == tmp_path / "x"


def test_capture_rosters_and_projection_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7604-COST: both frozen capture rosters contain 120 groups."""

    counts = {"fit": 80, "tune": 20, "policy": 20, "online": 80, "evaluation": 40}
    receipts = {}
    for role, count in counts.items():
        path = tmp_path / f"{role}.jsonl"
        lines = [json.dumps(_record(index)) for index in range(count)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        receipts[role] = {"path": path.name, "rows": count}
    fit, evaluation = pilot._capture_records(tmp_path, receipts)
    assert len(fit) == len(evaluation) == 120
    rows = [_pilot_row(index) for index in range(8)]
    measured = pilot._projection_or_blocked(
        rows, [100] * 120, {"model_load_s": 20.0, "transport_authenticated": True}
    )
    assert measured["status"] == "measured_projection"
    unauthenticated = pilot._projection_or_blocked(
        rows, [100] * 120, {"model_load_s": 20.0, "transport_authenticated": False}
    )
    assert unauthenticated["status"] == "transport_not_authenticated"
    insufficient = pilot._projection_or_blocked(
        rows[:-1], [100] * 120, {"model_load_s": 20.0, "transport_authenticated": True}
    )
    assert insufficient["status"] == "insufficient_measured_cost"
    receipts["fit"]["rows"] = 79
    with pytest.raises(ValueError, match="capture_role_count_changed"):
        pilot._capture_records(tmp_path, receipts)
    receipts["fit"]["rows"] = 80
    monkeypatch.setattr(pilot, "validate_input_record", lambda _row: False)
    with pytest.raises(ValueError, match="capture_role_invalid"):
        pilot._capture_records(tmp_path, receipts)
    monkeypatch.undo()
    fit_path = tmp_path / "fit.jsonl"
    fit_path.write_text(
        "\n".join(json.dumps(_record(index)) for index in range(79)) + "\n",
        encoding="utf-8",
    )
    receipts["fit"]["rows"] = 79
    with pytest.raises(ValueError, match="capture_roster_not_120_groups"):
        pilot._capture_records(tmp_path, receipts)
