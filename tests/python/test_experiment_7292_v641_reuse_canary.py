"""Tests for REQ-VERIFY-7292 and SCENARIO-VERIFY-7292-*.

The tests create model-shaped rows from public fixture bytes. They do not open
the scorer authority, so schedule and cache tests cannot depend on labels.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7291_v641_reuse_fixture as fixture
from carnot import experiment_7292_v641_reuse_canary as mod


ROOT = Path(__file__).resolve().parents[2]


def _public_groups() -> list[dict[str, object]]:
    """Return the two public development groups used by the canary."""

    public, _authority = fixture.build_fixture()
    return mod.select_development_groups(public)


def _completion_rows(schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create complete transport rows without reading private expected labels."""

    rows: list[dict[str, object]] = []
    for sealed in schedule:
        call_type = str(sealed["call_type"])
        if call_type == "direct":
            parsed: dict[str, object] = {"decision": "a"}
            compiled = {"decision": "supported"}
        else:
            document = sealed["document"]
            assert isinstance(document, dict)
            parsed = fixture.extract_completion(document, call_type)
            compiled = fixture.pointer.compile_pointer_completion(document, parsed, call_type)
        rows.append(
            {
                "call_order": sealed["call_order"],
                "call_id": sealed["call_id"],
                "unit_id": sealed["unit_id"],
                "group_id": sealed["group_id"],
                "source_version": sealed["source_version"],
                "arm": sealed["comparison_arm"],
                "call_type": call_type,
                "draw": sealed.get("draw"),
                "parsed_completion": parsed,
                "compiled_completion": compiled,
                "transport_complete": True,
                "parse_valid": True,
                "usable": True,
                "truncated": False,
                "terminal_state": "complete",
                "errors": [],
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "latency_s": 0.2,
                "native_cache_receipt": {
                    "cache_policy": "cache_prompt_true",
                    "evidence_present": True,
                    "cached_tokens": 0,
                    "cache_n": 0,
                    "measured_absence": True,
                },
            }
        )
    return rows


def _native_response(sealed: dict[str, object], completion: dict[str, object]) -> dict[str, object]:
    """Build one authentic byte-preserving response for replay tests."""

    payload, request_bytes = mod.live_runtime._request_payload(sealed)
    body = {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(completion)}}],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 5},
        },
        "timings": {"cache_n": 5},
    }
    response_bytes = json.dumps(body, separators=(",", ":")).encode()
    return {
        "raw_request": payload,
        "raw_request_bytes_b64": base64.b64encode(request_bytes).decode(),
        "raw_response": body,
        "raw_response_bytes_b64": base64.b64encode(response_bytes).decode(),
        "raw_completion": json.dumps(completion),
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "finish_reason": "stop",
        "latency_s": 0.2,
        "error": None,
    }


def test_scenario_verify_7292_schedule_has_exact_revision_split_and_budget() -> None:
    """SCENARIO-VERIFY-7292-SCHEDULE freezes the exact 44 public calls."""

    groups = _public_groups()
    schedule = mod.build_schedule(groups)

    assert len(groups) == 2
    assert all(
        [row["source_version"] for row in group["claims"]] == [1, 1, 2, 2] for group in groups
    )
    assert len(schedule) == 44
    assert {arm: sum(row["comparison_arm"] == arm for row in schedule) for arm in mod.ARMS} == {
        "warm_prefix_direct": 16,
        "fresh_verifier": 16,
        "versioned_reuse_verifier": 12,
    }
    assert mod.schedule_errors(schedule, groups) == []
    assert all(row["output_token_budget"] == 128 for row in schedule)
    assert all(row["decoding_parameters"]["cache_prompt"] is True for row in schedule)
    assert "expected_decision" not in mod.canonical_json(schedule)

    changed = deepcopy(schedule)
    changed[0]["output_token_budget"] = 129
    assert "call_0:output_token_budget" in mod.schedule_errors(changed, groups)

    extra = deepcopy(schedule)
    extra[0]["unexpected"] = True
    assert "call_0:extra_fields" in mod.schedule_errors(extra, groups)
    assert {"call_denominator", "arm_call_denominators"}.issubset(
        mod.schedule_errors(schedule[:-1], groups)
    )
    assert mod.schedule_errors(schedule, []) == [
        "schedule_rebuild:ValueError:canary_group_denominator"
    ]


def test_scenario_verify_7292_invalidation_and_parity_use_live_outputs() -> None:
    """SCENARIO-VERIFY-7292-INVALIDATION rejects stale source constraints."""

    groups = _public_groups()
    schedule = mod.build_schedule(groups)
    rows = _completion_rows(schedule)
    reduced = mod.reduce_canary(groups, schedule, rows)

    assert len(reduced["rows"]) == 24
    assert reduced["replayable_call_count"] == 44
    assert reduced["source_version_invalidations"] == 2
    assert reduced["served_stale_constraints"] == 0
    assert reduced["fresh_reuse_serialization_mismatches"] == 0
    assert all(
        row["prediction"] in {"supported", "contradicted", "unknown"} for row in reduced["rows"]
    )
    assert all(
        row["fresh_prediction"] == row["reuse_prediction"] for row in reduced["canary_control_rows"]
    )
    assert all(row["censored"] is False for row in reduced["rows"])

    broken = deepcopy(rows)
    target = next(
        row
        for row in broken
        if row["arm"] == "versioned_reuse_verifier" and row["call_type"] == "claim"
    )
    target["compiled_completion"] = {"outcome": "unknown", "relations": [], "errors": []}
    changed = mod.reduce_canary(groups, schedule, broken)
    assert changed["fresh_reuse_serialization_mismatches"] == 1
    assert changed["rows"][2]["error"] is not None


def test_scenario_verify_7292_cost_requires_one_native_warm_policy() -> None:
    """SCENARIO-VERIFY-7292-COST accepts measured zero reuse but not unfair policy."""

    schedule = mod.build_schedule(_public_groups())
    rows = _completion_rows(schedule)
    receipt = mod.warm_prefix_receipt(schedule, rows)

    assert receipt["fair_warm_comparator"] is True
    assert receipt["native_cache_evidence_complete"] is True
    assert receipt["total_cached_tokens"] == 0
    assert all(arm["explicit_measured_absence"] for arm in receipt["arms"])

    rows[0]["native_cache_receipt"]["cache_policy"] = "cache_prompt_false"
    unfair = mod.warm_prefix_receipt(schedule, rows)
    assert unfair["fair_warm_comparator"] is False


def test_scenario_verify_7292_e2e_replays_native_bytes_and_preserves_failure() -> None:
    """SCENARIO-VERIFY-7292-E2E rebuilds parser and cache facts from raw bytes."""

    schedule = mod.build_schedule(_public_groups())
    sealed = next(row for row in schedule if row["call_type"] == "source")
    document = sealed["document"]
    assert isinstance(document, dict)
    completion = fixture.extract_completion(document, "source")
    response = _native_response(sealed, completion)
    resource = {
        "server_pid": 10,
        "server_pid_start_ticks": 20,
        "gpu_uuid": "GPU-test",
        "lease_id": "lease-test",
        "cuda_offload_confirmed": True,
    }

    row = mod.build_completion_row(sealed, response, resource)
    replayed, errors = mod.independent_replay([sealed], [row])
    assert errors == [] and replayed == [row]
    assert row["native_cache_receipt"]["cached_tokens"] == 5
    assert row["native_cache_receipt"]["measured_absence"] is False

    damaged = deepcopy(row)
    damaged["raw_response_bytes_b64"] = "not-base64"
    replayed, errors = mod.independent_replay([sealed], [damaged])
    assert replayed == []
    assert errors == ["call_0:raw_response_bytes"]

    replayed, errors = mod.independent_replay([sealed], [])
    assert replayed == []
    assert errors == ["replay_denominator", "call_0:missing_row"]
    with pytest.raises(ValueError, match="bytes"):
        mod._decode_b64(None, "bytes")

    request_mismatch = deepcopy(row)
    request_mismatch["actual_parameters"]["seed"] = -1
    replayed, errors = mod.independent_replay([sealed], [request_mismatch])
    assert replayed
    assert errors == ["call_0:request_bytes", "call_0:replay_mismatch"]


def test_scenario_verify_7292_raw_directory_reconstructs_all_calls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7292-E2E reduces only the retained public call files."""

    groups = _public_groups()
    schedule = mod.build_schedule(groups)
    (tmp_path / "schedule.json").write_text(json.dumps({"schedule": schedule}))
    (tmp_path / "development-manifest.json").write_text(json.dumps({"groups": groups}))
    resource = {
        "server_pid": 10,
        "server_pid_start_ticks": 20,
        "gpu_uuid": "GPU-test",
        "lease_id": "lease-test",
        "cuda_offload_confirmed": True,
    }
    for index, sealed in enumerate(schedule):
        if sealed["call_type"] == "direct":
            completion = {"decision": "a"}
        else:
            completion = fixture.extract_completion(sealed["document"], sealed["call_type"])
        response = _native_response(sealed, completion)
        row = mod.build_completion_row(sealed, response, resource)
        (tmp_path / f"call_{index:02d}.json").write_text(
            json.dumps({"schedule": sealed, "completion": row})
        )

    reduced, errors = mod.independent_replay_from_raw(tmp_path)
    assert errors == []
    assert len(reduced["rows"]) == 24

    (tmp_path / "call_00.json").write_text("{}")
    _reduced, errors = mod.independent_replay_from_raw(tmp_path)
    assert any(error.startswith("call_0:") for error in errors)
    assert mod.independent_replay_from_raw(tmp_path / "missing")[1][0].startswith("manifest:")


def test_scenario_verify_7292_failure_uses_observed_model_boundaries() -> None:
    """SCENARIO-VERIFY-7292-FAILURE derives substrate from boundary counters."""

    no_load = deepcopy(mod.ZERO_INVOCATION_COUNTS)
    load_failed = {**no_load, "model_loads_attempted": 1, "model_loads_failed": 1}
    load_only = {**no_load, "model_loads_attempted": 1, "model_loads_completed": 1}
    generated = {
        **load_only,
        "generation_calls_attempted": 1,
        "generation_calls_completed": 1,
        "usable_answers": 1,
    }

    assert mod.classify_inference(no_load) == {
        "model_invoked": False,
        "inference_substrate": "blocked_before_qualifying_computation",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "not_invoked",
    }
    assert mod.classify_inference(load_failed)["model_invoked"] is True
    assert (
        mod.classify_inference(load_failed)["inference_substrate_class"]
        == "model_load_no_generation"
    )
    assert mod.classify_inference(load_only)["inference_substrate"] == "model_load_no_generation"
    assert (
        mod.classify_inference(generated)["inference_substrate_class"] == "model_bounded_generation"
    )


def test_req_verify_7292_gates_fail_closed_on_parity_and_warmness() -> None:
    """REQ-VERIFY-7292 requires every transport and reuse check for readiness."""

    groups = _public_groups()
    schedule = mod.build_schedule(groups)
    rows = _completion_rows(schedule)
    reduced = mod.reduce_canary(groups, schedule, rows)
    replayed = deepcopy(rows)
    warm = mod.warm_prefix_receipt(schedule, replayed)
    gates = mod.acceptance_gates(
        schedule,
        replayed,
        [],
        reduced,
        warm,
        model_identity={
            "gguf_sha256": "sha256:model",
            "embedded_tokenizer_sha256": "sha256:tokenizer",
        },
        gpu_receipts={"provenance_ok": True},
    )

    assert all(row["passed"] for row in gates)
    assert mod.ready_score(gates) == 1
    failed = deepcopy(gates)
    next(row for row in failed if row["criterion"] == "fresh_reuse_semantic_parity")["passed"] = (
        False
    )
    assert mod.ready_score(failed) == 0


def test_req_verify_7292_authenticates_exact_upstream_and_terminal_block() -> None:
    """REQ-VERIFY-7292 blocks exact missing or changed upstream evidence."""

    checks, context = mod.authenticate_inputs(ROOT)
    assert all(row["passed"] for row in checks)
    assert context["upstream"]["reuse_fixture_ready_score"] == 1
    assert context["development_manifest"]["authority_fields_present"] is False

    checks, _context = mod.authenticate_inputs(ROOT, expected_upstream_sha256="sha256:wrong")
    failure = mod.gate_summary(checks)
    assert failure["failed_check"] == "upstream_artifact_hash"
    artifact = mod.finalize_blocked_artifact(
        mod.base_artifact(mod.RUN_DATE), checks, duration_s=0.1
    )
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["expected_value"] == "sha256:wrong"
    assert mod.validate_artifact(artifact) == []


def test_req_verify_7292_rejects_malformed_public_schedule_inputs(tmp_path: Path) -> None:
    """REQ-VERIFY-7292 fails closed before a malformed public fixture can run."""

    public, _authority = fixture.build_fixture()
    with pytest.raises(ValueError, match="development_group_denominator"):
        mod.select_development_groups({"development_groups": []})
    bad_claims = deepcopy(public)
    bad_claims["development_groups"][0]["claims"] = []
    with pytest.raises(ValueError, match="development_claim_denominator"):
        mod.select_development_groups(bad_claims)
    leaked = deepcopy(public)
    leaked["development_groups"][0]["expected_decision"] = "supported"
    with pytest.raises(ValueError, match="authority_field"):
        mod.select_development_groups(leaked)

    groups = _public_groups()
    with pytest.raises(ValueError, match="source_version_identity"):
        mod._source_for(groups[0], 3)
    with pytest.raises(ValueError, match="canary_group_denominator"):
        mod.build_schedule([])
    wrong_claim_count = deepcopy(groups)
    wrong_claim_count[0]["claims"] = []
    with pytest.raises(ValueError, match="canary_claim_denominator"):
        mod.build_schedule(wrong_claim_count)
    wrong_revision = deepcopy(groups)
    for claim in wrong_revision[0]["claims"]:
        claim["source_version"] = 1
    with pytest.raises(ValueError, match="revision_claim_denominator"):
        mod.build_schedule(wrong_revision)

    checks, context = mod.authenticate_inputs(tmp_path)
    assert context == {}
    assert any(row["passed"] is False for row in checks)
    for relative in (
        mod.UPSTREAM_PATH,
        mod.UPSTREAM_MANIFEST_PATH,
        mod.UPSTREAM_ANALYSIS_PATH,
        mod.EXCLUSION_PATH,
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid serialized input")
    checks, context = mod.authenticate_inputs(
        tmp_path, expected_upstream_sha256=mod.sha256_file(tmp_path / mod.UPSTREAM_PATH)
    )
    assert context == {}
    assert checks[-1]["check"] == "input_parse"
    assert mod._date_argument(mod.RUN_DATE) == mod.RUN_DATE
    with pytest.raises(argparse.ArgumentTypeError):
        mod._date_argument("20260913")


def test_req_verify_7292_preserves_missing_and_unusable_extractions() -> None:
    """REQ-VERIFY-7292 keeps missing and unusable model outcomes as abstentions."""

    groups = _public_groups()
    schedule = mod.build_schedule(groups)
    rows = _completion_rows(schedule)
    missing = deepcopy(rows)
    missing.pop(
        next(
            index
            for index, row in enumerate(missing)
            if row["arm"] == "fresh_verifier" and row["call_type"] == "claim"
        )
    )
    reduced = mod.reduce_canary(groups, schedule, missing)
    fresh = next(row for row in reduced["rows"] if row["arm"] == "fresh_verifier")
    assert fresh["censored"] is True
    assert fresh["abstention"] is True
    assert "missing_call" in fresh["error"]

    unusable = {"compiled_completion": {}, "usable": False}
    decision = mod._decision(groups[0]["source_versions"][0]["document"], unusable, unusable)
    assert decision["errors"] == ["unusable_extraction"]
    assert mod._compiled(None)["errors"] == ["missing_call"]


def _validation_receipts(
    *, full_passed: bool = True, scoped_passed: bool = True
) -> list[dict[str, object]]:
    """Create terminal validation receipts for artifact classification tests."""

    return [
        {
            "name": name,
            "command": name,
            "exit_code": 0 if (full_passed or name != "full_python_suite") and scoped_passed else 1,
            "passed": (full_passed or name != "full_python_suite") and scoped_passed,
            "duration_s": 0.1,
            "log_sha256": "sha256:log",
        }
        for name in mod.REQUIRED_VALIDATION_NAMES
    ]


def _complete_artifact(
    *, full_passed: bool = True, scoped_passed: bool = True, ready: bool = True
) -> dict[str, object]:
    """Build one complete artifact through the production finalizer."""

    groups = _public_groups()
    schedule = mod.build_schedule(groups)
    rows = _completion_rows(schedule)
    reduced = mod.reduce_canary(groups, schedule, rows)
    warm = mod.warm_prefix_receipt(schedule, rows)
    identity = {
        "hf_id": mod.MODEL_ID,
        "gguf_sha256": "sha256:model",
        "embedded_tokenizer_sha256": "sha256:tokenizer",
    }
    gpu = {"provenance_ok": True}
    gates = mod.acceptance_gates(
        schedule, rows, [], reduced, warm, model_identity=identity, gpu_receipts=gpu
    )
    if not ready:
        gates[0]["passed"] = False
    artifact = mod.base_artifact(mod.RUN_DATE)
    artifact["runtime_model_identity"] = identity
    artifact["gpu_receipts"] = gpu
    artifact["invocation_counts"] = {
        **mod.ZERO_INVOCATION_COUNTS,
        "model_loads_attempted": 1,
        "model_loads_completed": 1,
        "generation_calls_attempted": 44,
        "generation_calls_completed": 44,
        "usable_answers": 44,
    }
    artifact["validation_receipts"] = _validation_receipts(
        full_passed=full_passed, scoped_passed=scoped_passed
    )
    return mod.finalize_measured_artifact(
        artifact, rows, rows, [], reduced, warm, gates, duration_s=12.0
    )


def test_req_verify_7292_terminal_classification_and_cold_validation() -> None:
    """REQ-VERIFY-7292 keeps readiness, validation, and verdict classes separate."""

    complete = _complete_artifact()
    assert complete["status"] == "complete"
    assert complete["verdict_class"] == "circular_positive"
    assert complete["reuse_canary_ready_score"] == 1
    assert mod.validate_artifact(complete) == []

    null = _complete_artifact(ready=False)
    assert null["verdict_class"] == "null"
    full_failed = _complete_artifact(full_passed=False)
    assert full_failed["verdict_class"] == "disqualified"
    assert "full_suite_failed" in full_failed["honest_verdict"]
    scoped_failed = _complete_artifact(scoped_passed=False)
    assert scoped_failed["verdict_class"] == "disqualified"
    assert "scoped_validation_failed" in scoped_failed["honest_verdict"]


def test_req_verify_7292_validator_rejects_corrupt_terminal_shapes() -> None:
    """REQ-VERIFY-7292 cold validation rejects altered terminal evidence."""

    assert mod.validate_artifact(None) == ["artifact_mapping"]
    assert mod.validate_artifact({})[0].startswith("missing_required_field:")
    complete = _complete_artifact()

    corruptions = [
        ("invocation_counts", {"bad": -1}),
        ("duration_s", -1),
        ("reproducibility_checksum", "sha256:wrong"),
        ("status", "running"),
        ("reuse_canary_ready_score", 0),
        ("per_call_rows", [*complete["per_call_rows"], {}]),
        ("rows", []),
        ("canary_control_rows", []),
        ("verdict_class", "positive"),
        ("validation_receipts", []),
    ]
    for field, value in corruptions:
        changed = deepcopy(complete)
        changed[field] = value
        if field != "reproducibility_checksum":
            changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert mod.validate_artifact(changed), field

    wrong_class = deepcopy(complete)
    wrong_class["inference_substrate_class"] = "blocked_no_run"
    wrong_class["reproducibility_checksum"] = mod.artifact_checksum(wrong_class)
    assert "inference_classification" in mod.validate_artifact(wrong_class)

    running = mod.base_artifact(mod.RUN_DATE)
    running["reproducibility_checksum"] = mod.artifact_checksum(running)
    assert "status" in mod.validate_artifact(running, require_validations=False)

    blocked = mod.finalize_blocked_artifact(
        mod.base_artifact(mod.RUN_DATE),
        [mod.gate_row("x", 1, 0, False, upstream="u", field="f")],
        duration_s=0.1,
    )
    blocked["honest_verdict"] = "wrong"
    blocked["reproducibility_checksum"] = mod.artifact_checksum(blocked)
    assert "blocked_terminal_state" in mod.validate_artifact(blocked)
