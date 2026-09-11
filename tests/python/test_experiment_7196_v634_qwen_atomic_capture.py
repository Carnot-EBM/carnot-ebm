"""Focused tests for REQ-VERIFY-7196 and SCENARIO-VERIFY-7196-*.

The tests use frozen public bytes or private temporary paths. They do not load
the model, acquire a GPU, or change the checked-in research result.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7196_v634_qwen_atomic_capture as exp


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
UPSTREAM = REPO / "results/experiment_7195_v634_typed_grounding.json"
PUBLIC = REPO / "results/experiment_7195_v634_typed_grounding_public.jsonl"


@pytest.fixture(scope="module")
def frozen() -> tuple[dict[str, object], bytes, list[dict[str, object]], list[dict[str, object]]]:
    """Load the public contract once so all schedule tests use identical bytes."""

    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    public_bytes = PUBLIC.read_bytes()
    public_rows = [json.loads(line) for line in public_bytes.splitlines()]
    schedule = exp.build_schedule(
        public_rows,
        upstream["fixture_manifest"]["frozen_generation_contract"],
    )
    return upstream, public_bytes, public_rows, schedule


def _typed_output() -> str:
    """Return one syntax-valid typed response without asserting semantics."""

    return exp.canonical_json(
        {
            "entity_bindings": [
                {
                    "entity_id": "e1",
                    "surface": "entity-one",
                    "source_start": 0,
                    "source_end": 10,
                }
            ],
            "relations": [
                {
                    "subject_id": "e1",
                    "operator": "precedes",
                    "object_id": "e2",
                    "polarity": "positive",
                    "source_start": 0,
                    "source_end": 10,
                }
            ],
            "missing_fields": [],
        }
    )


def _response(raw_output: str, *, tokens: int = 8, finish: str = "stop") -> dict[str, object]:
    """Make the native-response shape consumed by the receipt builder."""

    return {
        "raw_request": {"messages": [], "max_tokens": 128},
        "raw_request_bytes_b64": "e30=",
        "raw_output": raw_output,
        "raw_response": {"choices": []},
        "prompt_tokens": 12,
        "completion_tokens": tokens,
        "latency_s": 0.25,
        "finish_reason": finish,
        "error": None,
    }


def _resource() -> dict[str, object]:
    """Bind test receipts to one fictional but internally stable owner."""

    return {
        "server_pid": 101,
        "server_pid_start_ticks": 202,
        "gpu_uuid": "GPU-test",
        "lease_id": "lease:test",
    }


def _completion_bank(schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build one complete parse-valid logical bank with exact source reuse."""

    rows: list[dict[str, object]] = []
    by_call_id: dict[str, dict[str, object]] = {}
    for sealed in schedule:
        if sealed["cache_hit_expected"]:
            row = exp.build_cache_hit_row(sealed, by_call_id[sealed["reuse_from_call_id"]])
        else:
            raw = '{"decision":"supported"}' if sealed["call_type"] == "direct" else _typed_output()
            row = exp.build_completion_row(sealed, _response(raw), _resource())
        rows.append(row)
        by_call_id[row["call_id"]] = row
    return rows


def test_req_verify_7196_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7196 defines each focused runtime and artifact scenario."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7196") :]
    for scenario in (
        "PREFLIGHT",
        "BLINDING",
        "PARSING",
        "CACHE",
        "CHECKPOINT",
        "RUNTIME",
        "TERMINAL",
        "ARTIFACT",
    ):
        assert f"SCENARIO-VERIFY-7196-{scenario}" in section


def test_scenario_verify_7196_blinding_and_cache_freeze_exact_calls(
    frozen: tuple[dict[str, object], bytes, list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7196-BLINDING keeps each call's inputs separate."""

    upstream, _, public_rows, schedule = frozen
    contract = upstream["fixture_manifest"]["frozen_generation_contract"]

    assert exp.schedule_errors(schedule, public_rows, contract) == []
    assert len(schedule) == 576
    assert [row["call_type"] for row in schedule[:3]] == ["source", "claim", "direct"]
    assert {row["output_token_budget"] for row in schedule if row["call_type"] == "source"} == {128}
    assert {row["output_token_budget"] for row in schedule if row["call_type"] == "claim"} == {64}
    assert {row["output_token_budget"] for row in schedule if row["call_type"] == "direct"} == {16}
    assert all(set(row["model_input"]) == {"source_text"} for row in schedule[0::3])
    assert all(set(row["model_input"]) == {"claim_text"} for row in schedule[1::3])
    assert all(set(row["model_input"]) == {"source_text", "claim_text"} for row in schedule[2::3])
    assert sum(bool(row["cold_request_expected"]) for row in schedule) == 481
    assert sum(bool(row["cache_hit_expected"]) for row in schedule) == 95
    assert all(not row["cache_hit_expected"] for row in schedule if row["call_type"] != "source")
    assert len({row["input_sha256"] for row in schedule[0::3]}) == 97
    assert all(row["prompt_sha256"] == exp.sha256_text(row["prompt"]) for row in schedule)
    assert all(
        row["prompt_template_sha256"]
        == exp.sha256_text(contract["atomic_prompts"][row["call_type"]]["template"])
        for row in schedule
    )

    changed = deepcopy(schedule)
    changed[0]["model_input"]["claim_text"] = "private leak"
    assert "call_0:model_input_shape" in exp.schedule_errors(changed, public_rows, contract)


def test_scenario_verify_7196_blinding_rejects_every_schedule_drift(
    frozen: tuple[dict[str, object], bytes, list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7196-BLINDING cold-checks each frozen call property."""

    upstream, _, public_rows, schedule = frozen
    contract = upstream["fixture_manifest"]["frozen_generation_contract"]
    replacements = {
        "logical_order": 9,
        "input_sha256": "sha256:changed",
        "prompt": "changed",
        "prompt_template_sha256": "sha256:changed",
        "output_token_budget": 1,
        "response_schema": {},
        "decoding_parameters": {},
        "cache_key": "sha256:changed",
        "cold_request_expected": False,
    }
    expected_errors = {
        "logical_order": "call_0:logical_order",
        "input_sha256": "call_0:input_hash",
        "prompt": "call_0:prompt_hash",
        "prompt_template_sha256": "call_0:template_hash",
        "output_token_budget": "call_0:token_budget",
        "response_schema": "call_0:schema",
        "decoding_parameters": "call_0:decoding",
        "cache_key": "call_0:cache_key",
        "cold_request_expected": "cold_request_count_mismatch",
    }
    for field, replacement in replacements.items():
        changed = deepcopy(schedule)
        changed[0][field] = replacement
        assert expected_errors[field] in exp.schedule_errors(changed, public_rows, contract)

    changed = deepcopy(schedule)
    changed[1]["cache_key"] = "not-allowed"
    assert "call_1:non_source_cache" in exp.schedule_errors(changed, public_rows, contract)
    changed = deepcopy(schedule)
    hit_index = next(index for index, row in enumerate(changed) if row["cache_hit_expected"])
    changed[hit_index]["cache_hit_expected"] = False
    errors = exp.schedule_errors(changed, public_rows, contract)
    assert f"call_{hit_index}:cache_hit" in errors
    assert "source_cache_hit_count_mismatch" in errors
    assert "public_row_count_mismatch" in exp.schedule_errors(schedule, public_rows[:-1], contract)
    assert "logical_call_count_mismatch" in exp.schedule_errors(
        schedule[:-1], public_rows, contract
    )

    bad_public = deepcopy(public_rows)
    bad_public[0]["private"] = True
    with pytest.raises(ValueError, match="public_view_shape"):
        exp.build_schedule(bad_public, contract)
    bad_contract = deepcopy(contract)
    bad_contract["atomic_prompts"]["source"]["max_tokens"] = 127
    with pytest.raises(ValueError, match="schedule_invalid"):
        exp.build_schedule(public_rows, bad_contract)


def test_scenario_verify_7196_parsing_retains_failures_and_unknowns(
    frozen: tuple[dict[str, object], bytes, list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7196-PARSING never repairs or retries raw output."""

    _, _, _, schedule = frozen
    source = exp.build_completion_row(schedule[0], _response(_typed_output()), _resource())
    malformed = exp.build_completion_row(schedule[1], _response("{"), _resource())
    direct = exp.build_completion_row(
        schedule[2], _response('{"decision":"abstain"}', tokens=16, finish="length"), _resource()
    )
    failed_response = _response("")
    failed_response["error"] = "TimeoutError:request"
    request_error = exp.build_completion_row(schedule[3], failed_response, _resource())

    assert source["parse_status"] == "valid"
    assert source["unknown"] is False
    assert malformed["parse_status"] == "failed"
    assert malformed["parse_error"] == "invalid_json"
    assert direct["unknown"] is True
    assert direct["abstention"] is True
    assert direct["truncated"] is True
    assert request_error["terminal_state"] == "request_error"
    assert request_error["parse_status"] == "failed"
    assert request_error["request_error"] == "TimeoutError:request"
    assert exp.completion_row_errors(source, schedule[0]) == []

    changed = deepcopy(source)
    changed["raw_output_sha256"] = "sha256:changed"
    assert "raw_output_hash_mismatch" in exp.completion_row_errors(changed, schedule[0])

    wrong_shape = exp.parse_output("source", "[]")
    wrong_fields = exp.parse_output("direct", '{"decision":"maybe"}')
    assert wrong_shape["parse_error"] == "root_not_object"
    assert wrong_fields["parse_error"] == "direct_decision_invalid"


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(entity_bindings="bad"),
        lambda value: value.update(missing_fields=[1]),
        lambda value: value["entity_bindings"].append([]),
        lambda value: value["entity_bindings"][0].update(entity_id=1),
        lambda value: value["entity_bindings"][0].update(source_start=True),
        lambda value: value["entity_bindings"][0].update(source_start=11),
        lambda value: value["relations"].append([]),
        lambda value: value["relations"][0].update(subject_id=1),
        lambda value: value["relations"][0].update(polarity="maybe"),
        lambda value: value["relations"][0].update(source_start=True),
        lambda value: value["relations"][0].update(source_start=11),
    ],
)
def test_scenario_verify_7196_parsing_rejects_typed_shape_defects(change: object) -> None:
    """SCENARIO-VERIFY-7196-PARSING treats each syntax defect as invalid."""

    value = json.loads(_typed_output())
    change(value)
    assert exp.parse_output("source", json.dumps(value))["parse_error"] == "typed_output_invalid"


def test_scenario_verify_7196_parsing_checks_every_receipt_hash(
    frozen: tuple[dict[str, object], bytes, list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7196-PARSING detects raw or projected receipt drift."""

    _, _, _, schedule = frozen
    row = exp.build_completion_row(schedule[0], _response(_typed_output()), _resource())
    mutations = {
        "logical_order": (99, "logical_order_mismatch"),
        "prompt": ("changed", "prompt_hash_mismatch"),
        "prompt_bytes_b64": ("changed", "prompt_bytes_mismatch"),
        "raw_output_bytes_b64": ("changed", "raw_output_bytes_mismatch"),
        "raw_request_sha256": ("sha256:changed", "raw_request_hash_mismatch"),
        "raw_response_sha256": ("sha256:changed", "raw_response_hash_mismatch"),
        "unknown": (True, "unknown_mismatch"),
        "cache_hit": (True, "cache_state_mismatch"),
        "output_token_budget": (1, "token_budget_mismatch"),
        "terminal_state": ("running", "terminal_state_invalid"),
    }
    for field, (replacement, expected) in mutations.items():
        changed = deepcopy(row)
        changed[field] = replacement
        assert expected in exp.completion_row_errors(changed, schedule[0])

    hit = next(item for item in schedule if item["cache_hit_expected"])
    cold = next(item for item in schedule if item["call_id"] == hit["reuse_from_call_id"])
    cached = exp.build_cache_hit_row(
        hit, exp.build_completion_row(cold, _response(_typed_output()), _resource())
    )
    cached["reuse_from_call_id"] = "changed"
    assert "cache_origin_mismatch" in exp.completion_row_errors(cached, hit)

    response = _response(_typed_output())
    response.pop("raw_request_bytes_b64")
    fallback = exp.build_completion_row(schedule[0], response, _resource())
    assert fallback["raw_request_bytes_b64"] == "eyJtYXhfdG9rZW5zIjoxMjgsIm1lc3NhZ2VzIjpbXX0="
    response["raw_request_bytes_b64"] = "!"
    invalid_b64 = exp.build_completion_row(schedule[0], response, _resource())
    assert invalid_b64["raw_request_bytes_b64"] == fallback["raw_request_bytes_b64"]

    with pytest.raises(ValueError, match="cache_source_only"):
        exp.build_cache_hit_row(schedule[1], row)


def test_scenario_verify_7196_cache_clones_exact_source_receipt(
    frozen: tuple[dict[str, object], bytes, list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7196-CACHE names the cold receipt behind each hit."""

    _, _, _, schedule = frozen
    hit = next(row for row in schedule if row["cache_hit_expected"])
    cold_id = hit["reuse_from_call_id"]
    cold_schedule = next(row for row in schedule if row["call_id"] == cold_id)
    cold = exp.build_completion_row(cold_schedule, _response(_typed_output()), _resource())
    cached = exp.build_cache_hit_row(hit, cold)

    assert cached["cache_hit"] is True
    assert cached["cold_request"] is False
    assert cached["reuse_from_call_id"] == cold["call_id"]
    assert cached["raw_output"] == cold["raw_output"]
    assert cached["raw_output_sha256"] == cold["raw_output_sha256"]
    assert cached["request_ordinal"] is None
    assert exp.completion_row_errors(cached, hit) == []

    changed = deepcopy(hit)
    changed["input_sha256"] = "sha256:different"
    with pytest.raises(ValueError, match="cache_input_hash_mismatch"):
        exp.build_cache_hit_row(changed, cold)


def test_scenario_verify_7196_checkpoint_binds_schedule_model_and_rows(
    frozen: tuple[dict[str, object], bytes, list[dict[str, object]], list[dict[str, object]]],
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7196-CHECKPOINT rejects changed frozen identity."""

    _, _, _, schedule = frozen
    identity = exp.checkpoint_identity(schedule, "sha256:public", "sha256:model")
    row = exp.build_completion_row(schedule[0], _response(_typed_output()), _resource())
    path = tmp_path / "checkpoints" / "latest.json"
    payload = exp.write_checkpoint(path, identity, [row])

    assert exp.resume_checkpoint(path, identity) == [row]
    assert payload["row_hashes"] == [exp.sha256_json(row)]
    changed = deepcopy(identity)
    changed["model_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="checkpoint_identity_mismatch"):
        exp.resume_checkpoint(path, changed)
    payload["row_hashes"][0] = "sha256:changed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_row_hash_mismatch"):
        exp.resume_checkpoint(path, identity)
    payload["row_hashes"] = [exp.sha256_json(row)]
    payload["row_count"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_row_count_mismatch"):
        exp.resume_checkpoint(path, identity)


def test_scenario_verify_7196_preflight_rejects_quarantine_before_fields(
    frozen: tuple[dict[str, object], bytes, list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7196-PREFLIGHT rejects a structured quarantine flag."""

    upstream, public_bytes, _, _ = frozen
    clean = exp.upstream_gate_rows(upstream, public_bytes, exclusion_manifest={})
    assert all(row["passed"] for row in clean)
    assert next(row for row in clean if row["check"] == "known_failed_value_preserved")[
        "observed_value"
    ] == [0]

    quarantined = deepcopy(upstream)
    quarantined["flagged_adversarial"] = True
    rows = exp.upstream_gate_rows(quarantined, public_bytes, exclusion_manifest={})
    first = exp.first_failed_gate(rows)
    assert first is not None
    assert first["check"] == "upstream_structured_quarantine"
    assert first["field"] == "flagged_adversarial"
    assert first["observed_value"] is True

    wrapped = deepcopy(upstream)
    wrapped["flagged_adversarial"] = {"principle": "audited", "value": True}
    wrapped_rows = exp.upstream_gate_rows(wrapped, public_bytes, exclusion_manifest={})
    assert exp.first_failed_gate(wrapped_rows)["check"] == "upstream_structured_quarantine"

    excluded = exp.upstream_gate_rows(
        upstream,
        public_bytes,
        exclusion_manifest={"retired": [{"experiment_id": 7195}]},
    )
    assert exp.first_failed_gate(excluded)["check"] == "upstream_manifest_quarantine"


def test_scenario_verify_7196_terminal_class_uses_work_and_not_parse_success() -> None:
    """SCENARIO-VERIFY-7196-TERMINAL keeps complete parse-poor capture usable."""

    failed_gate = [exp.gate_row("gpu", True, False, False, upstream="host", field="cuda")]
    clean_gate = [exp.gate_row("gpu", True, True, True, upstream="host", field="cuda")]
    canary = {"terminal_state": "response", "latency_s": 0.2}
    complete = [
        {
            "terminal_state": "response",
            "parse_status": "failed" if index == 0 else "valid",
            "truncated": False,
            "unknown": False,
        }
        for index in range(576)
    ]

    blocked = exp.classify_terminal(failed_gate, False, None, [], False, 0.5)
    loaded = exp.classify_terminal(clean_gate, True, None, [], False, 3.0)
    bounded = exp.classify_terminal(clean_gate, True, canary, [], True, 12.0)
    partial = exp.classify_terminal(clean_gate, True, canary, complete[:3], True, 65.0)
    null = exp.classify_terminal(clean_gate, True, canary, complete, True, 65.0)
    too_short = exp.classify_terminal(clean_gate, True, canary, complete, True, 12.0)

    assert blocked["status"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert loaded["inference_substrate_class"] == "model_load_no_generation"
    assert bounded["inference_substrate_class"] == "model_bounded_generation"
    assert partial["status"] == "partial"
    assert partial["inference_substrate_class"] == "model_full_generation"
    assert null["status"] == "complete"
    assert null["score"] == 1
    assert null["verdict_class"] == "null"
    assert too_short["status"] == "complete"
    assert too_short["score"] == 0
    assert too_short["verdict_class"] == "disqualified"


def test_scenario_verify_7196_artifact_counts_full_denominators(
    frozen: tuple[dict[str, object], bytes, list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7196-ARTIFACT reports cold and amortized observed costs."""

    _, _, _, schedule = frozen
    rows = _completion_bank(schedule)

    summary = exp.summarize_capture(rows)
    assert summary["logical_receipt_count"] == 576
    assert summary["cold_request_count"] == 481
    assert summary["source_cache_hit_count"] == 95
    assert summary["per_call_type"]["source"]["denominator"] == 192
    assert summary["per_call_type"]["claim"]["denominator"] == 192
    assert summary["per_call_type"]["direct"]["denominator"] == 192
    assert summary["cold_costs"]["latency_s"] == pytest.approx(481 * 0.25)
    assert summary["amortized_costs"]["latency_s_per_public_row"] == pytest.approx(481 * 0.25 / 192)

    rows[0]["parse_status"] = "failed"
    rows[0]["parse_error"] = "mutated"
    rows[1]["truncated"] = True
    rows[2]["unknown"] = True
    rows[3]["request_error"] = "timeout"
    changed = exp.summarize_capture(rows)
    assert changed["per_call_type"]["source"]["invalid_count"] == 1
    assert changed["per_call_type"]["claim"]["truncated_count"] == 1
    assert changed["per_call_type"]["direct"]["unknown_count"] == 1
    assert changed["per_call_type"]["source"]["request_error_count"] == 1

    projected = exp.build_result_rows(rows)
    assert len(projected) == 192
    assert projected[0]["arm"] == "atomic_separate_capture"
    assert projected[0]["metric"] == 1
    assert projected[0]["error"] is not None


def test_scenario_verify_7196_artifact_base_has_all_principled_fields(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7196-ARTIFACT starts schema-complete off the result path."""

    artifact = exp.base_artifact(exp.RUN_DATE, root=REPO)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_principles"])
    assert artifact["MODEL_SPECS"] == exp.MODEL_SPECS
    assert artifact["sample_size_budget"]["planned_rows"] == 192
    assert artifact["sample_size_budget"]["planned_logical_receipts"] == 576
    assert artifact["runner_receipt"]["model_count"] == 1
    assert exp.validate_artifact(artifact, check_source_hashes=False) == ["status_not_terminal"]

    blocked = deepcopy(artifact)
    failure = exp.gate_row(
        "missing_cache", "present", "missing", False, upstream="model", field="path"
    )
    blocked.update(
        {
            "status": "blocked",
            "preconditions_checked": [failure],
            "gate_check_summary": failure,
            "verdict_class": "blocked",
            "honest_verdict": "blocked_missing_cache",
            "duration_s": 0.2,
        }
    )
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(blocked), encoding="utf-8")
    assert exp.validate_artifact(path, check_source_hashes=False) == []

    blocked["gate_check_summary"] = exp.gate_row(
        "different", True, False, False, upstream="model", field="path"
    )
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    assert "blocked_gate_summary_mismatch" in exp.validate_artifact(
        blocked, check_source_hashes=False
    )


def test_scenario_verify_7196_artifact_cold_validation_rejects_each_forgery(
    frozen: tuple[dict[str, object], bytes, list[dict[str, object]], list[dict[str, object]]],
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7196-ARTIFACT recomputes complete capture evidence."""

    _, _, _, schedule = frozen
    rows = _completion_bank(schedule)
    clean_gate = exp.gate_row("ready", True, True, True, upstream="host", field="gpu")
    artifact = exp.base_artifact(exp.RUN_DATE, root=REPO)
    artifact.update(
        {
            "status": "complete",
            "preconditions_checked": [clean_gate],
            "inference_substrate": "live_llm_inference",
            "inference_substrate_class": "model_full_generation",
            "inference_mode": "live_gpu",
            "duration_s": 65.0,
            "rows": exp.build_result_rows(rows),
            "verdict_class": "positive",
            "honest_verdict": "complete_positive_atomic_transport_capture_no_verifier_value_claim",
            "atomic_capture_complete_score": 1,
            "completion_rows": rows,
            "gpu_receipts": {"provenance_ok": True},
            "runner_receipt": {
                "model_count": 1,
                "replica_count": 1,
                "runner": "native_llama.cpp_server",
                "model_loaded": True,
            },
            "raw_manifest": {"schedule": schedule, "path": str(tmp_path / "missing.json")},
            "cost_summary": exp.summarize_capture(rows),
        }
    )
    artifact["source_artifact_hashes"] = exp._source_artifact_hashes(
        REPO, raw_manifest=tmp_path / "missing.json"
    )
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert exp.validate_artifact(artifact) == []
    changed_sources = deepcopy(artifact)
    changed_sources["source_artifact_hashes"]["module"] = "sha256:changed"
    changed_sources["reproducibility_checksum"] = exp.artifact_checksum(changed_sources)
    assert "source_artifact_hashes_mismatch" in exp.validate_artifact(changed_sources)
    changed_schedule = deepcopy(artifact)
    changed_schedule["raw_manifest"]["schedule"][0]["prompt"] = "tampered"
    changed_schedule["reproducibility_checksum"] = exp.artifact_checksum(changed_schedule)
    assert "schedule_replay:frozen_schedule_mismatch" in exp.validate_artifact(changed_schedule)
    assert exp.frozen_schedule_replay_errors(tmp_path, schedule) == [
        "frozen_schedule_unavailable:FileNotFoundError"
    ]

    mutations = {
        "field_principles": ({}, "field_principles_mismatch"),
        "MODEL_SPECS": ([], "model_specs_mandate_mismatch"),
        "run_date": ("20260909", "run_date_mismatch"),
        "execution_venue": ("unknown", "execution_venue_mismatch"),
        "random_seed": (0, "random_seed_mismatch"),
        "verifier_is_oracle": (True, "verifier_is_oracle_mismatch"),
        "rows": ([], "result_rows_mismatch"),
        "cost_summary": ({}, "cost_summary_mismatch"),
        "inference_mode": ("not_run", "complete_inference_mode_mismatch"),
    }
    for field, (replacement, expected) in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = replacement
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed, check_source_hashes=False)

    changed = deepcopy(artifact)
    changed.pop("inference_mode")
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert any(
        error.startswith("required_fields_missing")
        for error in exp.validate_artifact(changed, check_source_hashes=False)
    )
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        changed, check_source_hashes=False
    )
    changed = deepcopy(artifact)
    changed["raw_manifest"]["schedule"] = schedule[:-1]
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "schedule_missing_or_incomplete" in exp.validate_artifact(
        changed, check_source_hashes=False
    )
    changed = deepcopy(artifact)
    changed["runner_receipt"]["runner"] = "DualGPURunner"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "runner_model_count_mismatch" in exp.validate_artifact(
        changed, check_source_hashes=False
    )
    changed = deepcopy(artifact)
    changed["duration_s"] = 2.0
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    errors = exp.validate_artifact(changed, check_source_hashes=False)
    assert "verdict_class_mismatch" in errors
    assert "atomic_capture_complete_score_mismatch" in errors

    assert exp.validate_artifact(tmp_path / "absent.json") == ["artifact_unreadable"]
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert exp.validate_artifact(bad_json) == ["artifact_unreadable"]
    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    assert exp.validate_artifact(not_object) == ["artifact_unreadable"]
    assert exp.validate_artifact(123) == ["artifact_unreadable"]


def test_scenario_verify_7196_artifact_blocked_shape_defects() -> None:
    """SCENARIO-VERIFY-7196-ARTIFACT rejects dishonest blocked summaries."""

    artifact = exp.base_artifact(exp.RUN_DATE, root=REPO)
    artifact.update(
        {
            "status": "blocked",
            "preconditions_checked": [],
            "inference_substrate_class": "model_full_generation",
            "atomic_capture_complete_score": 1,
            "verdict_class": "positive",
        }
    )
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    errors = exp.validate_artifact(artifact, check_source_hashes=False)
    assert "blocked_failed_gate_missing" in errors
    assert "blocked_substrate_class_mismatch" in errors
    assert "blocked_score_mismatch" in errors
    assert "blocked_verdict_mismatch" in errors
