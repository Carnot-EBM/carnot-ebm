"""Tests for REQ-VERIFY-7167 and SCENARIO-VERIFY-7167-*."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7167_v632_claim_evidence_trace_capture as exp


ROOT = Path(__file__).resolve().parents[2]


def _fixture() -> dict:
    return json.loads((ROOT / exp.FIXTURE_PATH).read_text(encoding="utf-8"))


def _valid_output(schedule_row: dict) -> str:
    evidence = schedule_row["model_input"]["evidence_text"]
    return json.dumps(
        {
            "claim_entities": [{"name": "claim entity"}],
            "claim_facts": [{"subject": "claim entity", "relation": "has", "object": "fact"}],
            "evidence_entities": [{"name": "evidence entity"}],
            "evidence_facts": [
                {"subject": "evidence entity", "relation": "has", "object": "fact"}
            ],
            "cited_source_span": {"start": 0, "end": len(evidence), "text": evidence},
            "missing_fields": [],
            "direct_decision": "supported",
            "rationale": "The cited evidence states the extracted fact.",
        },
        sort_keys=True,
    )


def _resource_receipt() -> dict:
    return {
        "model": {
            "repository": exp.QWEN_MODEL_ID,
            "filename": exp.QWEN_FILENAME,
            "quantization": exp.QUANTIZATION,
            "revision": "a" * 40,
            "bytes": 16_000_000_000,
            "sha256": "sha256:" + "b" * 64,
            "runner_version": "llama.cpp build 9999",
            "embedded_tokenizer": True,
            "embedded_chat_template": True,
        },
        "process": {
            "pid": 321,
            "port": 18_321,
            "lease_id": "lease:test",
            "gpu_uuid": "GPU-test",
            "owned_by_task": True,
        },
        "cuda": {
            "cuda_placement_confirmed": True,
            "cuda_layers_offloaded": 49,
            "task_owned_vram_mb": 16_500,
        },
        "load_time_s": 65.0,
    }


def _teardown() -> dict:
    return {
        "pid": 321,
        "port": 18_321,
        "lease_id": "lease:test",
        "gpu_uuid": "GPU-test",
        "owned_identity_matched": True,
        "process_released": True,
        "port_released": True,
        "lease_released": True,
        "vram_released": True,
        "unrelated_process_kill_count_delta": 0,
    }


def _complete_artifact() -> dict:
    fixture = _fixture()
    schedule, authority = exp.build_sealed_schedule(fixture)
    resource = _resource_receipt()
    trace_rows = []
    for row, truth in zip(schedule, authority, strict=True):
        response = {
            "raw_output": _valid_output(row),
            "raw_response": {"choices": [{"message": {"content": _valid_output(row)}}]},
            "prompt_tokens": 40,
            "completion_tokens": 60,
            "latency_s": 2.0,
        }
        trace_rows.append(exp.build_trace_row(row, truth, response, resource))
    checkpoints = exp.checkpoint_receipts(trace_rows, schedule_identity=exp.schedule_identity(schedule))
    artifact = exp.finalize_artifact(
        exp.base_artifact(exp.RUN_DATE, root=ROOT),
        preconditions=[exp.gate_row("all_preconditions", True, True, True)],
        schedule=schedule,
        trace_rows=trace_rows,
        resource_receipt=resource,
        gpu_rows=[{"phase": "before"}, {"phase": "during"}, {"phase": "after"}],
        teardown=_teardown(),
        checkpoints=checkpoints,
        duration_s=161.0,
    )
    return artifact


def test_schema_shell_and_model_contract() -> None:
    """REQ-VERIFY-7167: the first artifact is complete and names one model."""

    artifact = exp.base_artifact(exp.RUN_DATE, root=ROOT)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["status"] == "running"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert exp.MODEL_SPECS == [
        {
            "hf_id": "unsloth/Qwen3.8-27B-GGUF",
            "filename": "Qwen3.8-27B-Q4_K_M.gguf",
            "quantization": "Q4_K_M",
            "role": "headline_full_generation",
        }
    ]
    assert exp.model_spec_errors(exp.MODEL_SPECS) == []


def test_sealed_selection_is_fixed_balanced_and_label_isolated() -> None:
    """SCENARIO-VERIFY-7167-SELECTION and -BLINDING."""

    fixture = _fixture()
    first, authority = exp.build_sealed_schedule(fixture)
    second, second_authority = exp.build_sealed_schedule(fixture)
    assert first == second
    assert authority == second_authority
    assert len(first) == len(authority) == 48
    assert len({row["pair_id"] for row in first}) == 12
    assert {sum(row["source_family"] == family for row in first) for family in exp.SOURCE_FAMILIES} == {
        24
    }
    row_by_id = {row["fixture_id"]: row for row in fixture["rows"]}
    condition_counts = {
        condition: sum(row_by_id[row["fixture_id"]]["condition"] == condition for row in first)
        for condition in exp.CONDITIONS
    }
    assert max(condition_counts.values()) - min(condition_counts.values()) <= 1
    assert all(sum(row["pair_id"] == pair for row in first) == 4 for pair in {r["pair_id"] for r in first})
    assert exp.label_isolation_errors(first) == []
    assert all("exact_label" in row and "condition" in row for row in authority)
    assert all("exact_label" not in row and "condition" not in row for row in first)


def test_model_and_prompt_mutations_fail() -> None:
    """SCENARIO-VERIFY-7167-IDENTITY and -BLINDING mutation tests."""

    changed = deepcopy(exp.MODEL_SPECS)
    changed[0]["hf_id"] = "unsloth/Qwen3.5-0.8B-GGUF"
    assert "model_spec_mismatch" in exp.model_spec_errors(changed)
    schedule, _ = exp.build_sealed_schedule(_fixture())
    schedule[0]["prompt"] += "\nAUTHORITY EXACT LABEL: supported"
    assert "row_0:authority_marker_in_prompt" in exp.label_isolation_errors(schedule)


def test_parser_preserves_valid_and_failed_outputs() -> None:
    """SCENARIO-VERIFY-7167-PARSE keeps every invalid output."""

    schedule, _ = exp.build_sealed_schedule(_fixture())
    valid = exp.parse_structured_output(_valid_output(schedule[0]), schedule[0]["model_input"])
    assert valid["parser_state"] == "valid"
    assert valid["structured_fields"]["direct_decision"] == "supported"
    malformed = exp.parse_structured_output("{bad", schedule[0]["model_input"])
    assert malformed == {
        "parser_state": "failed",
        "parser_error": "invalid_json",
        "structured_fields": None,
    }
    missing = json.loads(_valid_output(schedule[0]))
    del missing["claim_facts"]
    assert exp.parse_structured_output(json.dumps(missing), schedule[0]["model_input"])[
        "parser_error"
    ] == "missing_field:claim_facts"
    wrong_span = json.loads(_valid_output(schedule[0]))
    wrong_span["cited_source_span"]["text"] = "wrong"
    assert exp.parse_structured_output(json.dumps(wrong_span), schedule[0]["model_input"])[
        "parser_error"
    ] == "cited_source_span_mismatch"


def test_raw_hashes_and_parser_failures_remain_per_row() -> None:
    """REQ-VERIFY-7167 binds raw bytes and keeps parser failures."""

    schedule, authority = exp.build_sealed_schedule(_fixture())
    response = {
        "raw_output": "{bad",
        "raw_response": {"error": "synthetic"},
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "latency_s": 0.5,
    }
    row = exp.build_trace_row(schedule[0], authority[0], response, _resource_receipt())
    assert row["parser_state"] == "failed"
    assert row["raw_output_sha256"] == exp.sha256_text("{bad")
    assert row["raw_response_sha256"] == exp.sha256_json(response["raw_response"])
    assert row["authority_exact_label"] == authority[0]["exact_label"]
    changed = deepcopy(row)
    changed["raw_output"] += " "
    assert "raw_output_hash_mismatch" in exp.trace_row_errors(changed, schedule[0])


def test_checkpoint_resume_binds_schedule_and_cadence(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7167-RESUME accepts only the same frozen run."""

    schedule, authority = exp.build_sealed_schedule(_fixture())
    resource = _resource_receipt()
    rows = [
        exp.build_trace_row(
            row,
            truth,
            {
                "raw_output": _valid_output(row),
                "raw_response": {},
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "latency_s": 1.0,
            },
            resource,
        )
        for row, truth in zip(schedule[:4], authority[:4], strict=True)
    ]
    identity = exp.run_identity(schedule, manifest_sha256="sha256:" + "c" * 64)
    path = tmp_path / "checkpoint.json"
    exp.write_checkpoint(path, identity, rows)
    assert exp.resume_checkpoint(path, identity) == rows
    changed = deepcopy(identity)
    changed["schedule_sha256"] = "sha256:" + "d" * 64
    with pytest.raises(ValueError, match="checkpoint_identity_mismatch"):
        exp.resume_checkpoint(path, changed)
    with pytest.raises(ValueError, match="checkpoint_row_cadence"):
        exp.write_checkpoint(path, identity, rows[:3])


def test_resource_duration_and_teardown_contracts() -> None:
    """SCENARIO-VERIFY-7167-OWNERSHIP, -DURATION, and -TEARDOWN."""

    resource = _resource_receipt()
    teardown = _teardown()
    assert exp.resource_receipt_errors(resource) == []
    assert exp.teardown_errors(teardown, resource) == []
    assert exp.duration_errors(161.0, resource, [2.0] * 48, "model_full_generation") == []
    changed = deepcopy(resource)
    changed["process"]["owned_by_task"] = False
    assert "task_process_not_owned" in exp.resource_receipt_errors(changed)
    changed = deepcopy(resource)
    changed["cuda"]["cuda_placement_confirmed"] = False
    assert "cuda_placement_not_confirmed" in exp.resource_receipt_errors(changed)
    wrong_teardown = deepcopy(teardown)
    wrong_teardown["pid"] = 999
    assert "teardown_pid_mismatch" in exp.teardown_errors(wrong_teardown, resource)
    assert "duration_below_measured_work" in exp.duration_errors(
        70.0, resource, [2.0] * 48, "model_full_generation"
    )
    assert "complete_capture_wrong_substrate_class" in exp.duration_errors(
        161.0, resource, [2.0] * 48, "blocked_no_run"
    )


def test_manifest_and_authority_sidecar_are_atomic(tmp_path: Path) -> None:
    """REQ-VERIFY-7167 seals raw storage before model output."""

    schedule, authority = exp.build_sealed_schedule(_fixture())
    manifest = exp.initialize_raw_storage(tmp_path, schedule, authority)
    loaded = json.loads((tmp_path / exp.RAW_MANIFEST_NAME).read_text(encoding="utf-8"))
    labels = json.loads((tmp_path / exp.AUTHORITY_SIDECAR_NAME).read_text(encoding="utf-8"))
    assert loaded == manifest
    assert labels["rows"] == authority
    assert manifest["raw_output_rows"] == []
    assert manifest["schedule_sha256"] == exp.schedule_identity(schedule)


def test_blocked_artifact_is_terminal_and_schema_complete() -> None:
    """REQ-VERIFY-7167 keeps a stable external resource block terminal."""

    base = exp.base_artifact(exp.RUN_DATE, root=ROOT)
    failure = exp.gate_row(
        "idle_task_ownable_rtx_3090",
        {"minimum_count": 1},
        {"conflicting_pids": [233772]},
        False,
    )
    artifact = exp.finish_blocked(base, [failure], duration_s=1.25)
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_idle_task_ownable_rtx_3090"
    assert artifact["gate_check_summary"] == failure
    assert artifact["claim_evidence_trace_ready_score"] == 0
    assert exp.validate_artifact(artifact, check_source_hashes=False) == []


def test_complete_artifact_reconstructs_transport_readiness() -> None:
    """SCENARIO-VERIFY-7167-ARTIFACT checks rows instead of a forged score."""

    artifact = _complete_artifact()
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "positive"
    assert artifact["claim_evidence_trace_ready_score"] == 1
    assert artifact["parser_failure_rows"] == []
    assert exp.validate_artifact(artifact, check_source_hashes=False) == []
    changed = deepcopy(artifact)
    changed["claim_evidence_trace_rows"][0]["raw_output"] += " "
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "row_0:raw_output_hash_mismatch" in exp.validate_artifact(
        changed, check_source_hashes=False
    )
    changed = deepcopy(artifact)
    changed["claim_evidence_trace_ready_score"] = 0
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "readiness_score_mismatch" in exp.validate_artifact(changed, check_source_hashes=False)


def test_checksum_source_and_gate_helpers(tmp_path: Path) -> None:
    """REQ-VERIFY-7167 records stable hashes and exact failed gates."""

    path = tmp_path / "value.txt"
    path.write_text("value", encoding="utf-8")
    assert exp.sha256_file(path) == exp.sha256_text("value")
    row = exp.gate_row("x", {"a": 1}, {"a": 2}, False)
    assert row == {
        "check": "x",
        "expected_value": {"a": 1},
        "observed_value": {"a": 2},
        "passed": False,
    }
    artifact = {"reproducibility_checksum": "old", "value": 1}
    first = exp.artifact_checksum(artifact)
    artifact["reproducibility_checksum"] = "new"
    assert exp.artifact_checksum(artifact) == first


def test_schedule_and_parser_defensive_failures() -> None:
    """SCENARIO-VERIFY-7167-SELECTION and -PARSE cover malformed inputs."""

    fixture = _fixture()
    schedule, _ = exp.build_sealed_schedule(fixture)
    incomplete = deepcopy(fixture)
    incomplete["rows"] = [
        row for row in incomplete["rows"] if row.get("source_family") != exp.SOURCE_FAMILIES[0]
    ]
    with pytest.raises(ValueError, match="insufficient_complete_pairs"):
        exp.build_sealed_schedule(incomplete)

    invalid_input = deepcopy(schedule)
    invalid_input[0]["model_input"] = None
    assert "row_0:model_input_invalid" in exp.label_isolation_errors(invalid_input)
    leaked_input = deepcopy(schedule)
    leaked_input[0]["model_input"]["exact_label"] = "supported"
    assert any("authority_keys_in_model_input" in error for error in exp.label_isolation_errors(leaked_input))

    mutations = []
    mutations.append(schedule[:-1])
    for field, value in (
        ("fixture_id", schedule[1]["fixture_id"]),
        ("pair_id", "wrong-pair"),
        ("row_order", 99),
        ("source_family", "wrong-family"),
        ("input_sha256", "wrong"),
        ("prompt_sha256", "wrong"),
        ("prompt_template_sha256", "wrong"),
        ("response_schema", {}),
        ("response_schema_sha256", "wrong"),
        ("decoding_parameters", {}),
        ("decoding_parameters_sha256", "wrong"),
        ("output_token_budget", 1),
    ):
        changed = deepcopy(schedule)
        changed[0][field] = value
        mutations.append(changed)
    observed = {error for changed in mutations for error in exp.schedule_errors(changed)}
    assert {
        "schedule_row_count_mismatch",
        "schedule_fixture_ids_not_unique",
        "schedule_pair_groups_mismatch",
        "schedule_order_mismatch",
        "schedule_source_balance_mismatch",
        "row_0:input_hash_mismatch",
        "row_0:prompt_hash_mismatch",
        "row_0:prompt_template_hash_mismatch",
        "row_0:response_schema_mismatch",
        "row_0:response_schema_hash_mismatch",
        "row_0:decoding_parameters_mismatch",
        "row_0:decoding_parameters_hash_mismatch",
        "row_0:token_budget_mismatch",
    } <= observed
    fixture_subset = {
        "rows": [
            {"fixture_id": row["fixture_id"], "condition": "supported"} for row in schedule
        ]
    }
    assert "schedule_condition_balance_mismatch" in exp.schedule_errors(schedule, fixture_subset)

    base = json.loads(_valid_output(schedule[0]))
    cases = [
        ("[]", "root_not_object"),
        (json.dumps({**base, "extra": True}), "field_set_mismatch"),
        (json.dumps({**base, "claim_entities": {}}), "list_field_invalid"),
        (json.dumps({**base, "direct_decision": "maybe"}), "direct_decision_invalid"),
        (json.dumps({**base, "rationale": ""}), "rationale_invalid"),
        (json.dumps({**base, "cited_source_span": []}), "cited_source_span_invalid"),
    ]
    assert [
        exp.parse_structured_output(raw, schedule[0]["model_input"])["parser_error"]
        for raw, _error in cases
    ] == [error for _raw, error in cases]


def test_row_checkpoint_manifest_and_resource_defenses(tmp_path: Path) -> None:
    """REQ-VERIFY-7167 covers defensive receipt and resume branches."""

    schedule, authority = exp.build_sealed_schedule(_fixture())
    response = {
        "raw_output": _valid_output(schedule[0]),
        "raw_response": {},
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "latency_s": 1.0,
    }
    row = exp.build_trace_row(schedule[0], authority[0], response, _resource_receipt())
    observed = set()
    for field in (
        "row_order",
        "fixture_id",
        "pair_id",
        "source_family",
        "input_sha256",
        "prompt_sha256",
        "source_text_sha256",
        "claim_text_sha256",
        "evidence_text_sha256",
    ):
        changed = deepcopy(row)
        changed[field] = "wrong"
        observed.update(exp.trace_row_errors(changed, schedule[0]))
    for field, value in (
        ("raw_response_sha256", "wrong"),
        ("parser_state", "wrong"),
        ("parser_error", "wrong"),
        ("structured_fields", None),
        ("authority_exact_label", "wrong"),
        ("generation_state", "wrong"),
        ("latency_s", -1),
    ):
        changed = deepcopy(row)
        changed[field] = value
        observed.update(exp.trace_row_errors(changed, schedule[0]))
    assert "raw_response_hash_mismatch" in observed
    assert "parser_state_mismatch" in observed
    assert "parser_error_mismatch" in observed
    assert "structured_fields_mismatch" in observed
    assert "authority_exact_label_invalid" in observed
    assert "generation_state_invalid" in observed
    assert "latency_invalid" in observed

    bad_checkpoint = tmp_path / "bad_checkpoint.json"
    bad_checkpoint.write_text(json.dumps({"identity": {}, "rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_row_cadence"):
        exp.resume_checkpoint(bad_checkpoint, {})

    raw_dir = tmp_path / "raw"
    manifest = exp.initialize_raw_storage(raw_dir, schedule, authority)
    assert exp.initialize_raw_storage(raw_dir, schedule, authority) == manifest
    changed_schedule = deepcopy(schedule)
    changed_schedule[0]["prompt"] += "changed"
    with pytest.raises(ValueError, match="raw_manifest_identity_mismatch"):
        exp.initialize_raw_storage(raw_dir, changed_schedule, authority)

    assert exp.resource_receipt_errors({}) == ["model_resource_receipt_missing"]
    bad_resource = _resource_receipt()
    bad_resource["model"].update(
        {
            "repository": "wrong",
            "revision": "",
            "bytes": 0,
            "sha256": "",
            "runner_version": "",
            "embedded_tokenizer": False,
        }
    )
    bad_resource["process"].pop("pid")
    bad_resource["cuda"]["task_owned_vram_mb"] = 0
    bad_resource["load_time_s"] = -1
    resource_errors = exp.resource_receipt_errors(bad_resource)
    assert {
        "resource_model_identity_mismatch",
        "resource_model_provenance_incomplete",
        "resource_model_hash_missing",
        "runner_version_missing",
        "embedded_model_metadata_not_confirmed",
        "task_process_identity_incomplete",
        "task_owned_vram_missing",
        "load_time_invalid",
    } <= set(resource_errors)
    bad_teardown = {"unrelated_process_kill_count_delta": 1}
    teardown_errors = exp.teardown_errors(bad_teardown, _resource_receipt())
    assert "teardown_touched_unrelated_process" in teardown_errors
    assert "teardown_owned_identity_matched_missing" in teardown_errors
    assert exp._generation_receipt_errors([], [row]) == ["generation_receipts_mismatch"]


def test_terminal_validator_defensive_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7167-ARTIFACT rejects forged terminal metadata."""

    no_failure = exp.finish_blocked(exp.base_artifact(exp.RUN_DATE, root=ROOT), [], duration_s=1)
    assert no_failure["gate_check_summary"]["check"] == "blocked_without_failed_gate"
    partial = exp.finalize_artifact(
        exp.base_artifact(exp.RUN_DATE, root=ROOT),
        preconditions=[],
        schedule=[],
        trace_rows=[],
        resource_receipt={},
        gpu_rows=[],
        teardown={},
        checkpoints=[],
        duration_s=1,
    )
    assert partial["status"] == "partial"

    assert exp.validate_artifact(object()) == ["artifact_unreadable"]
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("bad", encoding="utf-8")
    assert exp.validate_artifact(unreadable) == ["artifact_unreadable"]
    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    assert exp.validate_artifact(not_object) == ["artifact_unreadable"]

    blocked = exp.finish_blocked(
        exp.base_artifact(exp.RUN_DATE, root=ROOT),
        [exp.gate_row("x", True, False, False)],
        duration_s=1,
    )
    mutations = [
        ("preconditions_checked", []),
        ("gate_check_summary", {}),
        ("inference_substrate_class", "model_full_generation"),
        ("verdict_class", "partial"),
        ("claim_evidence_trace_ready_score", 1),
        ("honest_verdict", "wrong"),
    ]
    observed = set()
    for field, value in mutations:
        changed = deepcopy(blocked)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        observed.update(exp.validate_artifact(changed, check_source_hashes=False))
    assert {
        "blocked_failed_gate_missing",
        "blocked_gate_check_summary_mismatch",
        "blocked_substrate_class_mismatch",
        "blocked_verdict_class_mismatch",
        "blocked_readiness_mismatch",
        "blocked_honest_verdict_mismatch",
    } <= observed

    complete = _complete_artifact()
    top_level_mutations = [
        ("field_principles", {}),
        ("MODEL_SPECS", []),
        ("run_date", "wrong"),
        ("inference_substrate", "wrong"),
        ("random_seed", 0),
        ("verifier_is_oracle", True),
        ("verdict_class", "wrong"),
        ("duration_s", -1),
        ("rows", []),
        ("generation_receipts", []),
        ("parser_failure_rows", [{}]),
        ("checkpoint_receipts", []),
        ("status", "complete"),
    ]
    for field, value in top_level_mutations:
        changed = deepcopy(complete)
        changed[field] = value
        if field != "duration_s":
            changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        exp.validate_artifact(changed, check_source_hashes=False)
    missing = deepcopy(complete)
    del missing["rows"]
    missing["reproducibility_checksum"] = exp.artifact_checksum(missing)
    assert any("required_fields_missing" in error for error in exp.validate_artifact(missing, check_source_hashes=False))
    broken_checksum = deepcopy(complete)
    broken_checksum["reproducibility_checksum"] = "wrong"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        broken_checksum, check_source_hashes=False
    )
    short_rows = deepcopy(complete)
    short_rows["claim_evidence_trace_rows"] = short_rows["claim_evidence_trace_rows"][:-1]
    short_rows["reproducibility_checksum"] = exp.artifact_checksum(short_rows)
    assert "trace_row_count_mismatch" in exp.validate_artifact(
        short_rows, check_source_hashes=False
    )
    wrong_complete_state = deepcopy(complete)
    wrong_complete_state["status"] = "partial"
    wrong_complete_state["reproducibility_checksum"] = exp.artifact_checksum(
        wrong_complete_state
    )
    assert "complete_transport_state_mismatch" in exp.validate_artifact(
        wrong_complete_state, check_source_hashes=False
    )

    assert "source_artifact_hashes_mismatch" in exp.validate_artifact(complete)
    monkeypatch.setattr(exp, "find_repo_root", lambda: tmp_path)
    assert "fixture_source_unreadable" in exp.validate_artifact(complete)
