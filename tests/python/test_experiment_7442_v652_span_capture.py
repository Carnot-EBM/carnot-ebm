"""Tests for the one-shot paired claim-span capture.

Spec refs: REQ-VERIFY-7442 and SCENARIO-VERIFY-7442-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7437_v652_span_protocol as protocol
from carnot import experiment_7442_v652_span_capture as capture


def _protocol_artifact() -> dict[str, object]:
    return json.loads((capture.REPO_ROOT / capture.PROTOCOL_PATH).read_text(encoding="utf-8"))


def _panel_and_schedule() -> tuple[dict[str, object], list[dict[str, object]]]:
    value = _protocol_artifact()
    manifest = value["span_protocol_manifest"]
    assert isinstance(manifest, dict)
    panel = json.loads((capture.REPO_ROOT / str(manifest["panel"]["path"])).read_text())
    sealed = json.loads((capture.REPO_ROOT / str(manifest["schedule"]["path"])).read_text())
    return panel, sealed["rows"]


def _raw_response(reply: str, *, finish_reason: str = "stop", tokens: int = 8) -> dict[str, object]:
    return {
        "raw_request": {"messages": [{"role": "user", "content": "sealed"}]},
        "raw_response": {
            "choices": [{"message": {"content": reply}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 20, "completion_tokens": tokens},
        },
        "raw_reply": reply,
        "attempted": True,
        "terminal_state": "response",
        "finish_reason": finish_reason,
        "prompt_tokens": 20,
        "completion_tokens": tokens,
        "latency_s": 0.2,
        "error": None,
    }


def _valid_reply(row: dict[str, object]) -> str:
    paragraph = str(row["paragraph"])
    end = min(len(paragraph), max(1, paragraph.find(".") + 1))
    claim = paragraph[:end]
    if row["arm"] == "span":
        return json.dumps({"claims": [[0, end]]})
    return json.dumps({"claims": [claim]})


def _terminal_receipts() -> list[dict[str, object]]:
    return [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in (*capture.AFFECTED_CHECK_NAMES, *capture.TERMINAL_CHECK_NAMES)
    ]


def test_protocol_gates_authenticate_exact_ready_unflagged_artifact() -> None:
    """SCENARIO-VERIFY-7442-PRECONDITIONS keeps upstream operands explicit."""

    value = _protocol_artifact()
    gates = capture.protocol_gate_rows(value)
    assert [row["field"] for row in gates] == [
        "experiment_id",
        "span_protocol_ready_score",
        "verdict_class",
        "flagged_adversarial",
    ]
    assert all(row["passed"] is True for row in gates)
    changed = deepcopy(value)
    changed["span_protocol_ready_score"] = 0
    assert capture.protocol_gate_rows(changed)[1]["observed"] == 0
    assert capture.protocol_gate_rows(changed)[1]["passed"] is False


def test_frozen_schedules_use_both_arms_and_one_budget() -> None:
    """SCENARIO-VERIFY-7442-DEVELOPMENT freezes 8 canary and 96 evaluation calls."""

    panel, sealed = _panel_and_schedule()
    development = capture.build_development_schedule(panel["development"])
    evaluation = capture.build_evaluation_schedule(sealed)
    assert len(development) == 8
    assert len(evaluation) == 96
    assert all(row["max_new_tokens"] == 256 for row in [*development, *evaluation])
    assert all(row["temperature"] == 0.0 and row["parser_retry_count"] == 0 for row in evaluation)
    assert all(
        [row["arm"] for row in evaluation if row["pair_id"] == pair_id]
        == [row["arm"] for row in evaluation if row["pair_id"] == pair_id][:1]
        + [
            "verbatim"
            if [row["arm"] for row in evaluation if row["pair_id"] == pair_id][0] == "span"
            else "span"
        ]
        for pair_id in {row["pair_id"] for row in evaluation}
    )
    frozen_hash = evaluation[0]["evaluation_schedule_sha256"]
    assert all(row["evaluation_schedule_sha256"] == frozen_hash for row in evaluation)
    assert frozen_hash == capture.canonical_hash(
        [
            {key: value for key, value in row.items() if key != "evaluation_schedule_sha256"}
            for row in evaluation
        ]
    )


def test_development_gate_requires_three_usable_outputs_in_each_arm() -> None:
    """SCENARIO-VERIFY-7442-DEVELOPMENT does not combine arm yields."""

    panel, _sealed = _panel_and_schedule()
    schedule = capture.build_development_schedule(panel["development"])
    rows = [capture.build_capture_row(row, _raw_response(_valid_reply(row))) for row in schedule]
    gate = capture.reduce_development_gate(rows)
    assert gate["capture_open"] is True
    assert gate["usable_by_arm"] == {"span": 4, "verbatim": 4}
    for row in rows:
        if row["arm"] == "span" and row["case_index"] >= 2:
            row.update(capture.build_capture_row(row, _raw_response("{")))
    closed = capture.reduce_development_gate(rows)
    assert closed["capture_open"] is False
    assert closed["usable_by_arm"]["span"] == 2


def test_raw_row_hashes_precede_parse_and_failures_remain_measured() -> None:
    """SCENARIO-VERIFY-7442-RAW preserves valid, malformed, and truncated replies."""

    panel, _sealed = _panel_and_schedule()
    row = capture.build_development_schedule(panel["development"])[0]
    valid = capture.build_capture_row(row, _raw_response(_valid_reply(row)))
    assert valid["persisted_before_parse"] is True
    assert valid["parse_valid"] is True
    assert valid["parse_status"] == "valid"
    assert valid["parse_errors"] == []
    assert valid["literal_span_reconstruction"] is True
    assert capture.capture_row_errors(valid) == []
    malformed = capture.build_capture_row(row, _raw_response("{"))
    assert malformed["disposition"] == "malformed"
    assert malformed["parse_valid"] is False
    assert malformed["parse_status"] == "invalid"
    assert malformed["parse_errors"]
    truncated = capture.build_capture_row(
        row, _raw_response(_valid_reply(row), finish_reason="length")
    )
    assert truncated["disposition"] == "truncated"
    assert truncated["completed_valid_output"] is False
    changed = deepcopy(valid)
    changed["raw_reply_sha256"] = "sha256:" + "0" * 64
    assert capture.capture_row_errors(changed) == ["raw_reply_hash_mismatch"]


def test_closed_canary_makes_all_evaluation_calls_explicitly_unstarted() -> None:
    """SCENARIO-VERIFY-7442-DEVELOPMENT retains the unopened panel denominator."""

    _panel, sealed = _panel_and_schedule()
    rows = capture.unstarted_evaluation_rows(capture.build_evaluation_schedule(sealed))
    reduced = capture.reduce_evaluation(rows, protocol.parser_control_rows())
    assert len(rows) == 96
    assert reduced["sample_counts"] == {
        "planned": 96,
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "censored": 0,
        "unstarted": 96,
    }
    assert reduced["span_value_score"] == 0
    assert all(row["disposition"] == "unstarted" for row in reduced["extraction_rows"])


def test_paired_reducer_uses_completion_ci_qualifiers_and_token_cost() -> None:
    """SCENARIO-VERIFY-7442-REDUCTION keeps the representation gate paired and narrow."""

    _panel, sealed = _panel_and_schedule()
    schedule = capture.build_evaluation_schedule(sealed)
    rows: list[dict[str, object]] = []
    for row in schedule:
        if row["arm"] == "span":
            response = _raw_response(_valid_reply(row), tokens=4)
        else:
            response = _raw_response("{", tokens=12)
        rows.append(capture.build_capture_row(row, response))
    reduced = capture.reduce_evaluation(rows, protocol.parser_control_rows())
    assert reduced["paired_completion_advantage"]["estimate"] == 1.0
    assert reduced["paired_completion_advantage"]["ci95_low"] > 0.0
    assert reduced["paired_output_token_delta"] < 0
    assert reduced["constructed_qualifier_delta"] == 0.0
    assert reduced["span_value_score"] == 1
    assert all(
        row["semantic_truth"] == "unknown"
        for row in reduced["extraction_rows"]
        if row["condition"] == "ragtruth_unchanged_response"
    )
    changed = deepcopy(rows)
    for row in changed:
        if row["arm"] == "span":
            row["completion_tokens"] = 20
    assert (
        capture.reduce_evaluation(changed, protocol.parser_control_rows())["span_value_score"] == 0
    )


def test_reducer_rejects_shape_and_pair_drift() -> None:
    """REQ-VERIFY-7442 rejects incomplete or unpaired evaluation evidence."""

    _panel, sealed = _panel_and_schedule()
    schedule = capture.build_evaluation_schedule(sealed)
    rows = capture.unstarted_evaluation_rows(schedule)
    with pytest.raises(ValueError, match="evaluation_row_count"):
        capture.reduce_evaluation(rows[:-1], protocol.parser_control_rows())
    changed = deepcopy(rows)
    changed[-1]["pair_id"] = "drift"
    with pytest.raises(ValueError, match="evaluation_pair_shape"):
        capture.reduce_evaluation(changed, protocol.parser_control_rows())


def test_fixture_artifact_replays_and_mutations_fail_closed() -> None:
    """SCENARIO-VERIFY-7442-TERMINAL recomputes rows, counts, receipts, and scores."""

    artifact = capture.build_fixture_artifact()
    assert capture.validate_artifact(artifact, require_terminal=True) == []
    assert capture.independent_reduce_artifact(artifact) == []
    assert artifact["span_capture_complete_score"] == 1
    assert artifact["invocation_counts"]["generation_calls_attempted"] == 104
    for field, error in (
        ("span_capture_complete_score", "span_capture_complete_score_mismatch"),
        ("span_value_score", "span_value_score_mismatch"),
        ("promotion_score", "declaration_mismatch:promotion_score"),
    ):
        changed = deepcopy(artifact)
        changed[field] = 1 - int(changed[field])
        changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
        assert error in capture.validate_artifact(changed, require_terminal=True)
    changed = deepcopy(artifact)
    changed["validation_receipts"] = [
        row for row in changed["validation_receipts"] if row["name"] != "adversarial_verify"
    ]
    changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
    assert "required_validation_receipt_missing:adversarial_verify" in capture.validate_artifact(
        changed, require_terminal=True
    )


def test_blocked_artifact_has_exact_gate_summary_and_zero_model_work() -> None:
    """SCENARIO-VERIFY-7442-PRECONDITIONS emits a specific external block."""

    failed = capture.protocol_gate_rows({})
    artifact = capture.build_blocked_artifact(failed)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["model_invoked"] is False
    assert sum(artifact["invocation_counts"].values()) == 0
    assert artifact["gate_check_summary"]["observed"] is None
    assert capture.validate_artifact(artifact, require_terminal=True) == []


def test_receipts_date_and_schedule_boundaries() -> None:
    """REQ-VERIFY-7442 keeps terminal names and the run date closed."""

    receipts = _terminal_receipts()
    assert capture.required_receipt_errors(receipts) == []
    assert capture.required_receipt_errors(receipts[:-1]) == [
        "required_validation_receipt_missing:verdict_row_consistency_strict"
    ]
    assert capture.date_argument("20260920") == "20260920"
    with pytest.raises(ValueError, match="date must be 20260920"):
        capture.date_argument("20260919")
    with pytest.raises(ValueError, match="development_paragraph_count"):
        capture.build_development_schedule([])
    with pytest.raises(ValueError, match="sealed_schedule_count"):
        capture.build_evaluation_schedule([])


def test_field_contract_and_model_spec_are_plain_and_exact() -> None:
    """REQ-VERIFY-7442 requires the mandated model and ordinary gate scalars."""

    artifact = capture.build_fixture_artifact()
    assert artifact["MODEL_SPECS"] == ["unsloth/Qwen3.8-27B-GGUF"]
    assert artifact["inference_substrate_class"] == "model_bounded_generation"
    assert artifact["execution_venue"] == "host"
    assert set(artifact["field_principles"]) == capture.REQUIRED_FIELDS
    assert all(isinstance(row["passed"], bool) for row in artifact["acceptance_gate_results"])
    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = []
    changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
    assert "declaration_mismatch:MODEL_SPECS" in capture.validate_artifact(
        changed, require_terminal=True
    )


def test_bound_protocol_sidecars_and_source_flags_are_authenticated(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7442-PRECONDITIONS binds exact sidecar bytes and flags."""

    checks, context = capture.collect_preconditions(capture.REPO_ROOT)
    assert all(row["passed"] is True for row in checks)
    assert len(context["development_schedule"]) == 8
    assert len(context["evaluation_schedule"]) == 96
    copied = tmp_path / "protocol.json"
    copied.write_text((capture.REPO_ROOT / capture.PROTOCOL_PATH).read_text())
    copied_value = json.loads(copied.read_text())
    copied_value["flagged_adversarial"] = True
    copied.write_text(json.dumps(copied_value))
    gates = capture.protocol_gate_rows(copied_value)
    assert gates[-1]["passed"] is False


def test_defensive_transport_schedule_and_receipt_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-7442 gives every malformed boundary a stable failure."""

    assert capture._load_object(tmp_path / "missing.json") == {}
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{")
    assert capture._load_object(malformed_path) == {}
    _panel, sealed = _panel_and_schedule()
    changed_schedule = deepcopy(sealed)
    changed_schedule[0]["arm"] = "other"
    with pytest.raises(ValueError, match="sealed_schedule_arm"):
        capture.build_evaluation_schedule(changed_schedule)
    row = capture.build_evaluation_schedule(sealed)[0]
    failed = capture.build_capture_row(
        row,
        {
            "raw_request": {},
            "raw_response": {},
            "raw_reply": "",
            "attempted": True,
            "terminal_state": "request_error",
            "error": "fixture",
        },
    )
    assert failed["disposition"] == "failed"
    cancelled = capture.build_capture_row(
        row,
        {
            "raw_request": {},
            "raw_response": {},
            "raw_reply": "",
            "attempted": True,
            "terminal_state": "cancelled",
        },
    )
    assert cancelled["disposition"] == "cancelled"
    changed = deepcopy(failed)
    changed["persisted_before_parse"] = False
    assert "raw_not_persisted_before_parse" in capture.capture_row_errors(changed)
    assert capture._paired_bootstrap([])["estimate"] is None
    incomplete_controls = [
        row
        for row in protocol.reduce_constructed_pairs(protocol.constructed_qualifier_pairs())
        if not (row["pair_id"] == "qualifier-01-negation" and row["arm"] == "span")
    ]
    assert len(capture._semantic_pairs(incomplete_controls)) == 11
    receipts = _terminal_receipts()
    assert capture.required_receipt_errors([*receipts, receipts[0]]) == [
        f"required_validation_receipt_duplicate:{receipts[0]['name']}"
    ]
    failed_receipts = deepcopy(receipts)
    failed_receipts[0]["passed"] = False
    assert capture.required_receipt_errors(failed_receipts) == [
        f"required_validation_receipt_failed:{receipts[0]['name']}"
    ]


def test_content_addressed_shards_are_immutable(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7442-RAW keeps a hash-bound checkpoint before parsing."""

    receipt = capture.write_content_addressed_shard(tmp_path, "raw", {"value": 1})
    assert capture.write_content_addressed_shard(tmp_path, "raw", {"value": 1}) == receipt
    (tmp_path / str(receipt["path"])).write_text("changed")
    with pytest.raises(ValueError, match="content_address_collision"):
        capture.write_content_addressed_shard(tmp_path, "raw", {"value": 1})


def test_bound_sidecar_reader_rejects_missing_and_invalid_bytes(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7442-PRECONDITIONS refuses sidecar drift before model work."""

    with pytest.raises(ValueError, match="sidecar_hash_mismatch"):
        capture._read_bound_json(tmp_path, {"path": "missing.json", "sha256": "sha256:" + "0" * 64})
    path = tmp_path / "invalid.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="sidecar_invalid"):
        capture._read_bound_json(
            tmp_path, {"path": "invalid.json", "sha256": capture.sha256_file(path)}
        )


def test_runner_receipt_binds_owned_offload_and_transport_hashes() -> None:
    """SCENARIO-VERIFY-7442-RAW binds PID, lease, CUDA, model, and request bytes."""

    identity = {
        "pid": 12,
        "start_time_ticks": 34,
        "command": ["llama-server", "--n-gpu-layers", "all"],
        "owned_by_task": True,
        "cuda_provenance_ok": True,
        "gpu_uuid": "GPU-test",
        "lease_id": "lease-test",
        "server_build": "b9606",
    }
    row = capture.build_fixture_artifact()["development_rows"][0]
    receipt = capture._runner_receipt(
        {
            "runtime_identity": identity,
            "gpu_receipts": {
                "provenance": {
                    "provenance_ok": True,
                    "cuda_log_evidence": True,
                    "task_owned_vram_mb": 16000,
                },
                "lease_release": {"released": True},
                "cleanup": {"leak_free": True},
            },
            "development_rows": [row],
            "rows": [],
        },
        {
            "model_spec": {
                "model_block_count": 65,
                "sha256": "sha256:" + "1" * 64,
                "revision": "test",
                "runtime_flags": {"n_gpu_layers": "all"},
                "decoding": {"max_new_tokens": 256},
            },
            "server_path": "/external/llama-server",
            "server_sha256": "sha256:" + "2" * 64,
        },
    )
    assert receipt["all_layers_offloaded"] is True
    assert receipt["actual_offloaded_layers"] == 65
    assert receipt["runner_pid_start_ticks"] == 34
    assert receipt["lease_released"] is True


def test_artifact_classifier_covers_null_positive_closed_and_disqualified() -> None:
    """SCENARIO-VERIFY-7442-TERMINAL keeps validity separate from benefit."""

    fixture = capture.build_fixture_artifact()
    context = {
        "model_spec": fixture["model_specs"][0],
        "protocol": _protocol_artifact(),
        "protocol_manifest": _protocol_artifact()["span_protocol_manifest"],
        "evaluation_schedule": capture._fixture_schedules()[1],
        "source_hashes": fixture["source_artifact_hashes"],
    }
    current = {
        "development_rows": fixture["development_rows"],
        "development_gate": fixture["development_gate"],
        "rows": fixture["extraction_rows"],
        "current_invocation_events": fixture["current_invocation_events"],
        "current_run_id": fixture["current_run_id"],
        "current_owner_pid": fixture["current_owner_pid"],
        "event_shards": [],
        "response_shards": [],
        "evaluation_shards": [],
    }
    reduced = capture.reduce_evaluation(fixture["extraction_rows"], protocol.parser_control_rows())
    common = {
        "context": context,
        "checks": fixture["preconditions_checked"],
        "capture": current,
        "reduced": reduced,
        "runner": fixture["runner_receipt"],
        "receipts": _terminal_receipts(),
        "affected_ok": True,
        "phase_spans": [],
        "started_at": "2026-09-20T00:00:00Z",
        "duration_s": 20.0,
        "model_duration_s": 12.0,
        "computation_duration_s": 1.0,
        "validation_duration_s": 7.0,
        "cold_start_duration_s": 1.0,
        "require_terminal": True,
        "flagged_adversarial": False,
    }
    disqualified = capture._measured_artifact(**{**common, "affected_ok": False})
    assert disqualified["verdict_class"] == "disqualified"
    producer_failure = capture._measured_artifact(
        **{**common, "producer_runtime_error": "KeyError:parse_status"}
    )
    assert producer_failure["verdict_class"] == "disqualified"
    assert producer_failure["producer_runtime_error"] == "KeyError:parse_status"
    assert producer_failure["span_capture_complete_score"] == 0
    closed_capture = deepcopy(current)
    closed_capture["development_gate"]["capture_open"] = False
    closed = capture._measured_artifact(**{**common, "capture": closed_capture})
    assert closed["honest_verdict"] == "complete_null_span_capture_development_gate_closed"
    positive_reduced = deepcopy(reduced)
    positive_reduced["span_value_score"] = 1
    positive = capture._measured_artifact(**{**common, "reduced": positive_reduced})
    assert positive["verdict_class"] == "circular_positive"


def test_validator_defensive_mutations_are_named() -> None:
    """SCENARIO-VERIFY-7442-TERMINAL rejects every stored summary drift."""

    fixture = capture.build_fixture_artifact()
    mutations: list[tuple[str, object, str]] = [
        ("field_principles", {}, "field_principles_mismatch"),
        ("verdict_class", "other", "verdict_class_invalid"),
        ("model_invoked", False, "model_invoked_mismatch"),
        ("current_owner_pid", "bad", "current_event_reduction_failed:ValueError"),
        ("event_count", 0, "current_receipt_mismatch:event_count"),
        ("development_rows", [], "development_row_count_mismatch"),
        ("development_gate", {}, "development_gate_mismatch"),
        ("rows", [], "rows_alias_mismatch"),
        ("extraction_rows", [], "extraction_row_count_mismatch"),
        ("arm_metrics", {}, "arm_metrics_mismatch"),
        ("raw_capture_manifest", [], "raw_capture_manifest_mismatch"),
    ]
    for field, value, expected in mutations:
        changed = deepcopy(fixture)
        changed[field] = value
        changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
        assert any(
            error.startswith(expected)
            for error in capture.validate_artifact(changed, require_terminal=True)
        ), field
    changed = deepcopy(fixture)
    changed["model_specs"] = []
    changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
    assert "resolved_model_spec_missing" in capture.validate_artifact(
        changed, require_terminal=True
    )
    changed = deepcopy(fixture)
    changed["model_specs"][0]["hf_id"] = "wrong"
    changed["model_specs"][0]["path"] = ""
    changed["model_specs"][0]["decoding"] = {}
    changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
    errors = capture.validate_artifact(changed, require_terminal=True)
    assert "resolved_model_spec_mismatch:hf_id" in errors
    assert "resolved_model_spec_missing:path" in errors
    assert "resolved_model_decoding_mismatch" in errors
    changed = deepcopy(fixture)
    changed["extraction_rows"][-1]["pair_id"] = "drift"
    changed["rows"] = deepcopy(changed["extraction_rows"])
    changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
    assert any(
        error.startswith("independent_reduction_failed:ValueError:evaluation_pair_shape")
        for error in capture.validate_artifact(changed, require_terminal=True)
    )
    changed = deepcopy(fixture)
    changed["sample_size_budget"]["attempted"] = 0
    changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
    assert "sample_size_budget_mismatch" in capture.validate_artifact(
        changed, require_terminal=True
    )
    changed = deepcopy(fixture)
    changed["reproducibility_checksum"] = "wrong"
    assert "reproducibility_checksum_mismatch" in capture.validate_artifact(
        changed, require_terminal=True
    )


def test_blocked_validator_rejects_false_model_and_verdict_claims() -> None:
    """SCENARIO-VERIFY-7442-PRECONDITIONS keeps blocked work at zero."""

    artifact = capture.build_blocked_artifact(capture.protocol_gate_rows({}))
    artifact["model_invoked"] = True
    artifact["rows"] = [{}]
    artifact["span_capture_complete_score"] = 1
    artifact["honest_verdict"] = "complete_wrong"
    artifact["reproducibility_checksum"] = "wrong"
    errors = capture.validate_artifact(artifact, require_terminal=True)
    assert "model_invoked_mismatch" in errors
    assert "blocked_model_or_rows_invalid" in errors
    assert "blocked_verdict_prefix_invalid" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_terminal_raw_response_reconciles_one_unfinished_generation() -> None:
    """SCENARIO-VERIFY-7442-RAW closes only the attempt proven by persisted bytes."""

    events = [
        {
            "scope": "current",
            "transport": "owned_runtime",
            "run_id": "run",
            "owner_pid": 12,
            "call_id": "model-load",
            "operation": "model_load",
            "state": "attempted",
            "monotonic_ns": 1,
        },
        {
            "scope": "current",
            "transport": "owned_runtime",
            "run_id": "run",
            "owner_pid": 12,
            "call_id": "model-load",
            "operation": "model_load",
            "state": "completed",
            "monotonic_ns": 2,
        },
        {
            "scope": "current",
            "transport": "owned_runtime",
            "run_id": "run",
            "owner_pid": 12,
            "call_id": "generation-0",
            "operation": "generation",
            "state": "attempted",
            "monotonic_ns": 3,
        },
    ]
    reconciled = capture.reconcile_terminal_events(
        events,
        [{"terminal_state": "response", "call_id": "development-00-span"}],
        monotonic_ns=4,
    )
    assert reconciled[-1]["state"] == "completed"
    assert reconciled[-1]["reconciled_from_call_id"] == "development-00-span"
    assert capture.canary.reduce_current_events(reconciled, run_id="run", owner_pid=12)[
        "invocation_counts"
    ]["generation_calls_completed"] == 1
    assert capture.reconcile_terminal_events(reconciled, [], monotonic_ns=5) == reconciled
