"""Tests for REQ-REPORT-7113 and its bounded E3 liveness scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7113_v624_arc_generation_liveness as exp
from carnot.agentic.arc_executable_world_model import LocalGGUFProposer


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _models(tmp_path: Path) -> list[dict[str, object]]:
    """Create tiny GGUF-shaped fixtures with exact model identities."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, declared in enumerate(exp.MODEL_SPECS):
        path = tmp_path / f"model-{index}-{declared['preferred_quantization']}.gguf"
        path.write_bytes(b"GGUF" + bytes([index]) * 32)
        rows.append(
            exp.resolved_model_row(declared, path, gpu_index=index, gpu_uuid=f"GPU-{index}")
        )
    return rows


def _telemetry(model: dict[str, object], pid: int) -> list[dict[str, object]]:
    """Return both required in-request GPU samples for one fake server."""

    return [
        {
            "model_repo_id": model["repo_id"],
            "phase": phase,
            "gpu_index": model["gpu_index"],
            "gpu_uuid": model["gpu_uuid"],
            "sample_ok": True,
            "server_pid": pid,
            "server_pid_visible": True,
            "memory_used_mb": 1024,
            "utilization_pct": 50,
            "process_rows": [{"pid": pid, "used_memory_mb": 1024}],
        }
        for phase in ("before_request", "after_request")
    ]


def _request_row(
    tmp_path: Path,
    model: dict[str, object],
    index: int,
    *,
    raw_output: str = '{"action":2,"data":null}',
    generated_tokens: int = 8,
    finish_reason: str = "eos",
    timed_out: bool = False,
) -> dict[str, object]:
    """Build one raw-bound request row without invoking a model."""

    pid = 4100 + index
    prompt = exp.ACTION_PROMPT
    raw_path = tmp_path / f"raw-{index}.json"
    exp.write_raw_trace(
        raw_path,
        {
            "request_id": f"request-{index}",
            "prompt": prompt,
            "raw_output": raw_output,
        },
    )
    return exp.make_request_row(
        request_id=f"request-{index}",
        model=model,
        prompt=prompt,
        raw_output=raw_output,
        prompt_tokens=41,
        generated_tokens=generated_tokens,
        finish_reason=finish_reason,
        timed_out=timed_out,
        model_loaded=True,
        observed_model_path=str(model["resolved_model_path"]),
        server_pid=pid,
        process_identity={
            "pid": pid,
            "pid_start_ticks": 99 + index,
            "executable": "/opt/llama-server",
            "cmdline_hash": f"sha256:{index + 1:064x}",
        },
        gpu_telemetry=_telemetry(model, pid),
        load_started_s=float(index * 20),
        load_finished_s=float(index * 20 + 4),
        request_started_s=float(index * 20 + 5),
        request_finished_s=float(index * 20 + 16),
        raw_trace_path=raw_path,
        lease={
            "lease_id": f"lease-{index}",
            "released": True,
            "unload_observed": True,
            "signals_sent": [],
        },
    )


def _artifact(tmp_path: Path, *, duration_s: float = 42.0) -> dict[str, object]:
    """Build the complete two-request positive fixture."""

    models = _models(tmp_path)
    rows = [_request_row(tmp_path, model, index) for index, model in enumerate(models)]
    return exp.build_artifact(
        run_date="20260907",
        duration_s=duration_s,
        preconditions_checked=[exp.gate_row("all_preconditions", True, True)],
        source_artifact_hashes={"module": "sha256:" + "a" * 64},
        model_specs=models,
        request_rows=rows,
        registry_hash_before="sha256:" + "b" * 64,
        registry_hash_after="sha256:" + "b" * 64,
    )


def _rewrite_raw(tmp_path: Path, row: dict[str, object], raw_output: str) -> None:
    """Update a row and its external raw trace without hiding the changed output."""

    raw_path = Path(str(row["raw_trace_path"]))
    exp.write_raw_trace(
        raw_path,
        {
            "request_id": row["request_id"],
            "prompt": exp.ACTION_PROMPT,
            "raw_output": raw_output,
        },
    )
    row["raw_output_hash"] = exp.sha256_text(raw_output)
    row["raw_trace_hash"] = exp.sha256_file(raw_path)
    parsed = exp.parse_action(raw_output)
    row.update(parsed)


def test_req_report_7113_spec_precedes_implementation() -> None:
    """REQ-REPORT-7113 owns every required field and named scenario."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7113") :]
    for scenario in (
        "PREFLIGHT",
        "MODELS",
        "ACTION",
        "LIVENESS",
        "SUBSTRATE",
        "NONCLAIM",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-7113-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"action":1,"data":null}', {"action": 1, "data": None}),
        (
            'prefix {"action":6,"data":{"x":3,"y":4}} suffix',
            {"action": 6, "data": {"x": 3, "y": 4}},
        ),
    ],
)
def test_scenario_report_7113_action_accepts_only_raw_schema(
    raw: str, expected: dict[str, object]
) -> None:
    """SCENARIO-REPORT-7113-ACTION accepts one exact schema object from raw output."""

    parsed = exp.parse_action(raw)
    assert parsed["parse_status"] == "parsed"
    assert parsed["proposed_action"] == expected
    assert parsed["action_schema_valid"] is True
    assert parsed["action_source"] == "parsed_raw_output"


@pytest.mark.parametrize(
    "raw",
    [
        "move right",
        '{"action":true,"data":null}',
        '{"action":7,"data":null}',
        '{"action":2,"data":{"x":1,"y":2}}',
        '{"action":6,"data":{"x":1}}',
        '{"action":1,"data":null,"fallback":true}',
        '{"action":1,"data":null} and {"action":2,"data":null}',
    ],
)
def test_scenario_report_7113_action_rejects_prose_and_malformed_schema(raw: str) -> None:
    """SCENARIO-REPORT-7113-ACTION leaves no action after any parser rejection."""

    parsed = exp.parse_action(raw)
    assert parsed["parse_status"] == "rejected"
    assert parsed["proposed_action"] is None
    assert parsed["action_schema_valid"] is False
    assert parsed["action_source"] is None


def test_runtime_diagnostics_preserve_prompt_and_generated_token_counts() -> None:
    """REQ-REPORT-7113 records server-measured token counts at the E3 runtime seam."""

    proposer = LocalGGUFProposer()
    proposer._record_completion_diagnostics(
        {
            "content": "{}",
            "stop_type": "eos",
            "truncated": False,
            "timings": {"prompt_n": 17, "predicted_n": 3},
        }
    )
    assert proposer.last_prompt_tokens == 17
    assert proposer.last_generated_tokens == 3
    assert proposer.last_stop_type == "eos"


def test_scenario_report_7113_artifact_positive_replays_all_receipts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7113-ARTIFACT recomputes one valid request per required model."""

    artifact = _artifact(tmp_path)
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["rows"] == artifact["per_model_request_rows"]
    assert artifact["model_repo_ids"] == list(exp.REQUIRED_MODEL_IDS)
    assert artifact["arc_generation_liveness_ready_score"] == 1
    assert artifact["inference_substrate_class"] == "model_bounded_generation"
    assert artifact["verdict_class"] == "positive"
    assert str(artifact["honest_verdict"]).startswith("complete_positive_")
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["offline_reproduced"] is False
    assert artifact["arc_registry_delta"] == 0
    assert exp.validate_artifact(artifact, verify_raw_traces=True) == []


def test_scenario_report_7113_liveness_zero_generations_is_terminal_null(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7113-LIVENESS does not promote loaded processes with zero tokens."""

    models = _models(tmp_path)
    rows = [
        _request_row(tmp_path, model, index, raw_output="", generated_tokens=0, finish_reason="eos")
        for index, model in enumerate(models)
    ]
    artifact = exp.build_artifact(
        run_date="20260907",
        duration_s=8.0,
        preconditions_checked=[exp.gate_row("all_preconditions", True, True)],
        source_artifact_hashes={},
        model_specs=models,
        request_rows=rows,
        registry_hash_before="sha256:" + "c" * 64,
        registry_hash_after="sha256:" + "c" * 64,
    )
    assert artifact["arc_generation_liveness_ready_score"] == 0
    assert artifact["inference_substrate_class"] == "model_load_no_generation"
    assert artifact["verdict_class"] == "null"
    assert str(artifact["honest_verdict"]).startswith("complete_null_")
    assert exp.validate_artifact(artifact, verify_raw_traces=True) == []


def test_scenario_report_7113_liveness_parser_rejection_and_timeout_are_null(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7113-LIVENESS preserves parser rejection and timeout evidence."""

    parser_artifact = _artifact(tmp_path / "parser")
    parser_row = parser_artifact["per_model_request_rows"][0]
    _rewrite_raw(tmp_path / "parser", parser_row, "move right")
    parser_artifact = exp.recompute_artifact(parser_artifact)
    assert parser_artifact["verdict_class"] == "null"
    assert parser_artifact["action_parse_rows"][0]["parse_status"] == "rejected"
    assert parser_artifact["action_parse_rows"][0]["proposed_action"] is None
    assert exp.validate_artifact(parser_artifact, verify_raw_traces=True) == []

    timeout_artifact = _artifact(tmp_path / "timeout")
    timeout_row = timeout_artifact["per_model_request_rows"][1]
    _rewrite_raw(tmp_path / "timeout", timeout_row, "")
    timeout_row.update(
        timed_out=True,
        generated_token_count=0,
        finish_reason="timeout",
        parse_status="rejected",
        parse_reason="request_timeout",
        proposed_action=None,
        action_schema_valid=False,
        action_schema_errors=["request_timeout"],
        action_source=None,
    )
    timeout_artifact = exp.recompute_artifact(timeout_artifact)
    assert timeout_artifact["verdict_class"] == "null"
    assert timeout_artifact["generated_token_rows"][1]["generated_token_count"] == 0
    assert exp.validate_artifact(timeout_artifact, verify_raw_traces=True) == []


@pytest.mark.parametrize("defect", ["model_path", "synthetic_action", "telemetry"])
def test_scenario_report_7113_models_and_telemetry_defects_disqualify(
    tmp_path: Path, defect: str
) -> None:
    """SCENARIO-REPORT-7113-MODELS rejects identity, leakage, and telemetry defects."""

    artifact = _artifact(tmp_path)
    row = artifact["per_model_request_rows"][0]
    if defect == "model_path":
        row["observed_model_path"] = str(tmp_path / "substitute.gguf")
    elif defect == "synthetic_action":
        row.update(
            parse_status="rejected",
            parse_reason="no_json_object",
            proposed_action={"action": 1, "data": None},
            action_schema_valid=False,
            action_source="synthetic_fixture",
        )
    else:
        row["gpu_telemetry"] = row["gpu_telemetry"][:1]
    artifact = exp.recompute_artifact(artifact)
    assert artifact["arc_generation_liveness_ready_score"] == 0
    assert artifact["verdict_class"] == "disqualified"
    assert str(artifact["honest_verdict"]).startswith("complete_disqualified_")
    assert exp.validate_artifact(artifact, verify_raw_traces=True) == []


def test_scenario_report_7113_substrate_rejects_duration_and_class_mismatch(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7113-SUBSTRATE enforces the declared compute floor."""

    too_short = _artifact(tmp_path / "short", duration_s=1.0)
    assert too_short["verdict_class"] == "disqualified"
    assert too_short["arc_generation_liveness_ready_score"] == 0
    assert exp.validate_artifact(too_short, verify_raw_traces=True) == []

    forged = _artifact(tmp_path / "forged")
    forged["inference_substrate_class"] = "model_full_generation"
    forged["reproducibility_checksum"] = exp.payload_checksum(forged)
    assert "inference_substrate_class_mismatch" in exp.validate_artifact(forged)


def test_scenario_report_7113_preflight_builds_complete_block_without_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7113-PREFLIGHT records an exact no-request block."""

    checks = [
        exp.gate_row("exact_cached_model_files", True, False),
        exp.gate_row("healthy_gpus", 2, 2),
    ]
    artifact = exp.build_artifact(
        run_date="20260907",
        duration_s=0.5,
        preconditions_checked=checks,
        source_artifact_hashes={},
        model_specs=[],
        request_rows=[],
        registry_hash_before=exp.sha256_file(tmp_path / "missing"),
        registry_hash_after=exp.sha256_file(tmp_path / "missing"),
    )
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["rows"] == []
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_check": "exact_cached_model_files",
        "expected_value": True,
        "observed_value": False,
        "checks": checks,
    }
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_7113_artifact_rejects_tampering_and_bad_counts(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7113-ARTIFACT catches raw drift and request-count forgery."""

    artifact = _artifact(tmp_path)
    Path(str(artifact["per_model_request_rows"][0]["raw_trace_path"])).write_text(
        json.dumps({"changed": True}), encoding="utf-8"
    )
    assert "raw_trace_hash_mismatch" in exp.validate_artifact(artifact, verify_raw_traces=True)

    count = _artifact(tmp_path / "count")
    count["per_model_request_rows"][0]["request_count"] = 2
    count = exp.recompute_artifact(count)
    assert count["verdict_class"] == "disqualified"
    assert count["arc_generation_liveness_ready_score"] == 0
    assert exp.validate_artifact(count, verify_raw_traces=True) == []


def test_resolved_model_rows_reject_substitution_and_bad_gguf(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7113-MODELS binds repository, path, size, hash, and GGUF header."""

    models = _models(tmp_path)
    assert exp.model_spec_errors(models) == []
    changed = deepcopy(models)
    changed[1]["repo_id"] = "legacy/small-model"
    assert "required_model_order_mismatch" in exp.model_spec_errors(changed)
    Path(str(models[0]["resolved_model_path"])).write_bytes(b"not-gguf")
    assert "gguf_header_invalid" in exp.model_spec_errors(models)


def test_req_report_7113_model_and_parser_error_ledger_is_complete(tmp_path: Path) -> None:
    """REQ-REPORT-7113 names every model-file and parser integrity failure."""

    models = _models(tmp_path)
    missing = deepcopy(models)
    missing[0]["resolved_model_path"] = str(tmp_path / "missing.gguf")
    duplicate = deepcopy(models)
    duplicate[1]["resolved_model_path"] = duplicate[0]["resolved_model_path"]
    duplicate_gpu = deepcopy(models)
    duplicate_gpu[1]["gpu_uuid"] = duplicate_gpu[0]["gpu_uuid"]
    policy = deepcopy(models)
    policy[0]["download_attempted"] = True
    policy[1]["quantization"] = ""
    size_and_hash = deepcopy(models)
    size_and_hash[0]["model_size_bytes"] = -1
    size_and_hash[1]["model_hash"] = "sha256:" + "0" * 64

    assert "model_file_missing" in exp.model_spec_errors(missing)
    assert "resolved_model_paths_not_distinct" in exp.model_spec_errors(duplicate)
    assert "gpu_assignments_not_distinct" in exp.model_spec_errors(duplicate_gpu)
    assert {"model_policy_mismatch", "quantization_missing"} <= set(exp.model_spec_errors(policy))
    assert {"model_size_mismatch", "model_hash_mismatch"} <= set(
        exp.model_spec_errors(size_and_hash)
    )
    assert exp._gguf_header_ok(tmp_path / "missing.gguf") is False
    assert exp._model_sha256(tmp_path / "missing.gguf") is None
    assert exp.action_schema_errors(None) == ["action_not_object"]
    assert exp.parse_action("unfinished {")["parse_reason"] == "no_json_action"


def test_req_report_7113_raw_receipt_error_ledger_is_complete(tmp_path: Path) -> None:
    """REQ-REPORT-7113 binds parser claims to readable external prompt/output bytes."""

    model = _models(tmp_path)[0]
    inside_results = _request_row(tmp_path / "results" / "raw", model, 0)
    assert "raw_trace_inside_results" in exp._raw_binding_errors(inside_results)

    unreadable = _request_row(tmp_path / "unreadable", model, 0)
    unreadable_path = Path(str(unreadable["raw_trace_path"]))
    unreadable_path.write_text("{", encoding="utf-8")
    unreadable["raw_trace_hash"] = exp.sha256_file(unreadable_path)
    assert exp._raw_binding_errors(unreadable) == ["raw_trace_unreadable"]

    changed = _request_row(tmp_path / "changed", model, 0)
    changed_path = Path(str(changed["raw_trace_path"]))
    exp.write_raw_trace(changed_path, {"prompt": "different", "raw_output": "different"})
    changed["raw_trace_hash"] = exp.sha256_file(changed_path)
    errors = exp._raw_binding_errors(changed)
    assert {
        "raw_output_hash_mismatch",
        "prompt_hash_mismatch",
        "raw_parse_receipt_mismatch",
    } <= set(errors)


def test_req_report_7113_request_integrity_error_ledger_is_complete(tmp_path: Path) -> None:
    """REQ-REPORT-7113 makes every request, timing, process, and cleanup defect terminal."""

    artifact = _artifact(tmp_path)
    rows = deepcopy(artifact["per_model_request_rows"])
    models = deepcopy(artifact["MODEL_SPECS"])
    count_errors = exp.request_structure_errors(rows[:1], models)
    assert {"required_request_count_mismatch", "request_model_order_mismatch"} <= set(count_errors)

    first, second = rows
    first.update(
        e3_request_path="bypass",
        model_download_attempted=True,
        prompt_hash="sha256:" + "0" * 64,
        prompt_token_count=True,
        generated_token_count=-1,
        action_schema_valid=False,
        server_pid=None,
        process_identity={},
        load_started_s="bad",
        model_loaded=False,
        lease_released=False,
        unload_observed=False,
        cleanup_signals_sent=["SIGTERM"],
    )
    second["gpu_telemetry"][0]["sample_ok"] = False
    second["load_duration_s"] = -1.0
    errors = set(exp.request_structure_errors(rows, models))
    assert {
        "e3_request_path_mismatch",
        "model_download_policy_mismatch",
        "prompt_hash_mismatch",
        "prompt_token_count_missing",
        "generated_token_count_invalid",
        "parsed_action_receipt_invalid",
        "process_telemetry_missing",
        "gpu_telemetry_missing",
        "request_timing_invalid",
        "model_load_incomplete",
        "gpu_cleanup_incomplete",
        "unrelated_process_signal_detected",
    } <= errors


def test_req_report_7113_validator_rejects_shape_authority_and_checksum_drift(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-7113 validates files, shape, nonclaims, registry binding, and checksum."""

    artifact = _artifact(tmp_path / "valid")
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.validate_artifact(artifact_path, verify_raw_traces=True) == []
    artifact_path.write_text("{", encoding="utf-8")
    assert exp.validate_artifact(artifact_path) == ["artifact_unreadable"]
    assert exp.validate_artifact([]) == ["artifact_not_object"]

    missing = deepcopy(artifact)
    missing.pop("rows")
    missing["extra"] = True
    assert set(exp.validate_artifact(missing)) == {"missing_field:rows", "extra_field:extra"}

    drifted = deepcopy(artifact)
    drifted.update(
        field_principles={},
        inference_substrate="forged",
        execution_venue="remote",
        solve_provenance="official",
        arc_registry_delta=1,
    )
    errors = set(exp.validate_artifact(drifted))
    assert {
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "execution_venue_mismatch",
        "solve_nonclaim_mismatch",
        "authority_boundary_mismatch",
        "reproducibility_checksum_mismatch",
    } <= errors

    registry_drift = exp.build_artifact(
        run_date="20260907",
        duration_s=42.0,
        preconditions_checked=[exp.gate_row("all_preconditions", True, True)],
        source_artifact_hashes={},
        model_specs=artifact["MODEL_SPECS"],
        request_rows=artifact["per_model_request_rows"],
        registry_hash_before="sha256:" + "a" * 64,
        registry_hash_after="sha256:" + "b" * 64,
    )
    assert registry_drift["verdict_class"] == "disqualified"
    assert "arc_registry_hash_mismatch" in registry_drift["gate_check_summary"]["observed_value"]
