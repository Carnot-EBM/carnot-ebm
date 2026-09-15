"""Tests for the bounded V643 native batch canary.

Spec refs: REQ-VERIFY-7320 and SCENARIO-VERIFY-7320-*.
"""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7306_v642_batch_fixture as fixture
from carnot import experiment_7317_v643_batch_harness as harness
from carnot import experiment_7320_v643_batch_canary as canary
from carnot.reporting import experiment_7303_validation_scope as scoped


ROOT = Path(__file__).resolve().parents[2]


def _panel() -> dict[str, object]:
    """Use the exact deterministic public fixture without exposing its labels."""

    public, _labels = fixture.build_fixture()
    return public


def _qualified_upstream() -> dict[str, object]:
    """Load the current qualified producer used by the dependency tests."""

    return json.loads(
        (ROOT / "results/experiment_7317_v643_batch_harness.json").read_text(encoding="utf-8")
    )


def _response_for(sealed: dict[str, object]) -> dict[str, object]:
    """Build one valid HTTP receipt from the shipped fixture's exact response."""

    request = sealed["fixture_request"]
    response = fixture.CpuFakeTransport().call(request)
    content = canary.model_content_from_fixture_response(sealed, response)
    body = {
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        "usage": {"prompt_tokens": 31, "completion_tokens": 17},
        "timings": {"prompt_ms": 2.0, "predicted_ms": 3.0},
    }
    body_bytes = canary.canonical_json(body).encode("utf-8")
    request_payload = canary.request_payload(sealed)
    request_bytes = canary.canonical_json(request_payload).encode("utf-8")
    return {
        "raw_request": request_payload,
        "raw_request_bytes_b64": base64.b64encode(request_bytes).decode("ascii"),
        "raw_response": body,
        "raw_response_bytes_b64": base64.b64encode(body_bytes).decode("ascii"),
        "raw_completion": content,
        "prompt_tokens": 31,
        "completion_tokens": 17,
        "finish_reason": "stop",
        "latency_s": 0.005,
        "error": None,
    }


def _valid_rows(
    schedule: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Create complete native-shaped rows for cold replay tests."""

    resource = {
        "server_pid": 123,
        "server_pid_start_ticks": 456,
        "gpu_uuid": "GPU-test",
        "lease_id": "lease-test",
        "cuda_offload_confirmed": True,
        "gpu_sample_sha256": "sha256:" + "1" * 64,
    }
    return [canary.build_per_call_row(row, _response_for(row), resource) for row in schedule]


def _passing_validation() -> dict[str, object]:
    """Create complete named receipts without running subprocesses in a unit test."""

    names = (*scoped.REQUIRED_CHECK_NAMES, *canary.TERMINAL_CHECK_NAMES)
    receipts = [
        {
            "name": name,
            "command": f"fixture:{name}",
            "command_argv": [name],
            "scope": "explicit_test_fixture",
            "exit_code": 0,
            "duration_s": 0.001,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "a" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in names
    ]
    return {
        "validation_receipts": receipts,
        "required_checks_passed": True,
        "missing_required_commands": [],
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "repository_health": scoped.build_repository_health(
            [
                {
                    "experiment_id": "exp7307-batch-canary",
                    "observed_at_utc": "2026-09-14T00:00:00Z",
                    "terminal_class": "blocked",
                    "resolved": False,
                    "collection_errors": [],
                }
            ]
        ),
        "validation_entrypoint_receipt": {
            "runner": harness.SCOPED_RUNNER,
            "called": True,
            "test_paths": [canary.TEST_PATH.as_posix()],
            "changed_modules": [canary.MODULE_PATH.as_posix()],
            "static_paths": [canary.WRAPPER_PATH.as_posix()],
            "legacy_launcher_called": False,
            "repository_wide_target_present": False,
        },
    }


def test_scenario_verify_7320_dependency_fails_closed_with_exact_value() -> None:
    """SCENARIO-VERIFY-7320-DEPENDENCY rejects every failed terminal class."""

    upstream = _qualified_upstream()
    assert canary.dependency_gate_rows(upstream)[-1]["passed"] is True
    for terminal in ("blocked", "partial", "disqualified"):
        changed = deepcopy(upstream)
        changed["verdict_class"] = terminal
        checks = canary.dependency_gate_rows(changed)
        summary = canary.gate_check_summary(checks)
        assert summary == {
            "failed_check": "dependency_terminal_class",
            "upstream": harness.EXPERIMENT_ID,
            "field": "verdict_class",
            "expected_value": "not blocked, partial, or disqualified",
            "observed_value": terminal,
        }
    quarantined = deepcopy(upstream)
    quarantined["flagged_adversarial"] = True
    checks = canary.dependency_gate_rows(quarantined)
    assert canary.gate_check_summary(checks)["failed_check"] == "dependency_quarantine"
    missing = canary.dependency_gate_rows(None)
    assert canary.gate_check_summary(missing)["observed_value"] == "missing_artifact"


def test_scenario_verify_7320_schedule_freezes_two_groups_and_equal_budgets() -> None:
    """SCENARIO-VERIFY-7320-SCHEDULE preserves 32 calls and all public identities."""

    public = _panel()
    group_ids = [row["group_id"] for row in public["development_groups"][:2]]
    schedule = canary.build_schedule(public, group_ids)

    assert len(schedule) == 32
    assert sum(row["output_token_budget"] for row in schedule) == 15_360
    assert [row["call_order"] for row in schedule] == list(range(32))
    assert canary.schedule_errors(schedule, public, group_ids) == []
    counts = Counter(row["arm"] for row in schedule)
    assert counts == {
        "serial_versioned_verifier": 20,
        "batched_versioned_verifier": 8,
        "batched_warm_prefix_direct": 4,
    }
    budgets: defaultdict[tuple[str, int, str], int] = defaultdict(int)
    identity_hashes: defaultdict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    for row in schedule:
        key = (row["group_id"], row["source_version"], row["arm"])
        budgets[key] += row["output_token_budget"]
        identity_hashes[(row["group_id"], row["source_version"])].add(
            (row["source_hash"], row["claim_set_sha256"])
        )
        assert "expected_decision" not in canary.canonical_json(row)
        assert row["decoding_parameters"] == canary.DECODING_PARAMETERS
    assert set(budgets.values()) == {1280}
    assert all(len(values) == 1 for values in identity_hashes.values())

    changed = deepcopy(schedule)
    changed[0]["output_token_budget"] += 1
    changed[1]["claim_ids"] = []
    changed[2]["decoding_parameters"] = {"temperature": 1.0}
    errors = canary.schedule_errors(changed, public, group_ids)
    assert {"schedule_mismatch", "allocated_output_tokens"}.issubset(errors)
    assert canary.call_budget_contract()["total_allocated_output_tokens"] == 15_360


def test_scenario_verify_7320_replay_uses_raw_bytes_and_keeps_failures() -> None:
    """SCENARIO-VERIFY-7320-REPLAY replays bytes and retains bad responses."""

    public = _panel()
    group_ids = [row["group_id"] for row in public["development_groups"][:2]]
    schedule = canary.build_schedule(public, group_ids)
    rows = _valid_rows(schedule)
    replay = canary.cold_replay(public, schedule, rows)

    assert replay["receipt"]["call_count"] == 32
    assert replay["receipt"]["transport_complete"] is True
    assert replay["receipt"]["required_field_omission_count"] == 0
    assert replay["receipt"]["truncation_count"] == 0
    assert replay["receipt"]["source_claim_identity_swap_count"] == 0
    assert replay["receipt"]["replay_parser_match"] is True
    assert len(replay["rows"]) == 48
    assert set(row["prediction"] for row in replay["rows"]) == {
        "supported",
        "contradicted",
        "unknown",
    }
    assert any(row["abstention"] for row in replay["rows"])
    assert any(row["prediction"] == "contradicted" for row in replay["rows"])

    bad_rows = deepcopy(rows)
    bad_response = _response_for(schedule[0])
    bad_response["raw_completion"] = "{}"
    bad_response["raw_response"]["choices"][0]["message"]["content"] = "{}"
    bad_response["raw_response"]["choices"][0]["finish_reason"] = "length"
    body = canary.canonical_json(bad_response["raw_response"]).encode("utf-8")
    bad_response["raw_response_bytes_b64"] = base64.b64encode(body).decode("ascii")
    bad_response["finish_reason"] = "length"
    bad_rows[0] = canary.build_per_call_row(schedule[0], bad_response, {})
    failed = canary.cold_replay(public, schedule, bad_rows)
    assert failed["receipt"]["required_field_omission_count"] == 1
    assert failed["receipt"]["truncation_count"] == 1
    assert failed["receipt"]["transport_complete"] is False
    assert failed["rows"][0]["censored"] is True


def test_scenario_verify_7320_model_rows_reject_malformed_and_swapped_ids() -> None:
    """REQ-VERIFY-7320 records malformed JSON and joint identity faults."""

    public = _panel()
    group_ids = [row["group_id"] for row in public["development_groups"][:2]]
    schedule = canary.build_schedule(public, group_ids)
    joint = next(row for row in schedule if row["call_type"] == "claim_batch")
    response = _response_for(joint)
    payload = json.loads(response["raw_completion"])
    payload["items"][0]["claim_id"] = "unknown-claim"
    content = canary.canonical_json(payload)
    response["raw_completion"] = content
    response["raw_response"]["choices"][0]["message"]["content"] = content
    body = canary.canonical_json(response["raw_response"]).encode("utf-8")
    response["raw_response_bytes_b64"] = base64.b64encode(body).decode("ascii")
    swapped = canary.build_per_call_row(joint, response, {})
    assert swapped["source_claim_identity_swap"] is True
    assert "claim_identity_mismatch" in swapped["failure_reasons"]

    malformed_response = _response_for(schedule[0])
    malformed_response["raw_completion"] = "not-json"
    malformed_response["raw_response"]["choices"][0]["message"]["content"] = "not-json"
    body = canary.canonical_json(malformed_response["raw_response"]).encode("utf-8")
    malformed_response["raw_response_bytes_b64"] = base64.b64encode(body).decode("ascii")
    malformed = canary.build_per_call_row(schedule[0], malformed_response, {})
    assert malformed["parse_valid"] is False
    assert malformed["terminal_state"] == "failed"
    assert malformed["normalized_response"] is None


def test_scenario_verify_7320_provenance_and_terminal_class_are_consistent() -> None:
    """SCENARIO-VERIFY-7320-PROVENANCE and -VALIDATION control readiness."""

    assert canary.classify_inference(canary.ZERO_INVOCATION_COUNTS) == {
        "model_invoked": False,
        "inference_substrate": "blocked_before_qualifying_computation",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "not_run",
    }
    load_counts = {**canary.ZERO_INVOCATION_COUNTS, "model_loads_attempted": 1}
    assert canary.classify_inference(load_counts)["inference_substrate_class"] == (
        "model_load_no_generation"
    )
    live_counts = {
        **load_counts,
        "model_loads_completed": 1,
        "generation_calls_attempted": 32,
        "generation_calls_completed": 32,
    }
    assert canary.classify_inference(live_counts) == {
        "model_invoked": True,
        "inference_substrate": "model_bounded_generation",
        "inference_substrate_class": "model_bounded_generation",
        "inference_mode": "live_gpu",
    }

    dependency_failure = harness.gate_row(
        "dependency_terminal_class",
        harness.EXPERIMENT_ID,
        "verdict_class",
        "not blocked, partial, or disqualified",
        "disqualified",
        False,
    )
    blocked = canary.blocked_artifact(
        canary.RUN_DATE,
        [dependency_failure],
        duration_s=0.25,
        timestamps={
            "started_at_utc": "2026-09-15T00:00:00Z",
            "completed_at_utc": "2026-09-15T00:00:01Z",
        },
    )
    assert blocked["batch_canary_ready_score"] == 0
    assert blocked["verdict_class"] == "blocked"
    assert blocked["invocation_counts"] == canary.ZERO_INVOCATION_COUNTS
    assert canary.validate_artifact(blocked, require_validation=False) == []


def test_scenario_verify_7320_complete_artifact_requires_semantics_and_validation() -> None:
    """SCENARIO-VERIFY-7320-SEMANTICS permits readiness only for a complete replay."""

    public = _panel()
    group_ids = [row["group_id"] for row in public["development_groups"][:2]]
    schedule = canary.build_schedule(public, group_ids)
    rows = _valid_rows(schedule)
    replay = canary.cold_replay(public, schedule, rows)
    validation = _passing_validation()
    capture = {
        "model_loaded": True,
        "model_invoked": True,
        "runtime_error": None,
        "gpu_receipts": {"provenance_ok": True},
        "runner_receipt": {"runner": "native_llama.cpp_server"},
    }
    artifact = canary.assemble_artifact(
        canary.RUN_DATE,
        preconditions=[],
        schedule=schedule,
        per_call_rows=rows,
        replay=replay,
        capture=capture,
        validation=validation,
        source_hashes={"upstream": "sha256:" + "b" * 64},
        duration_s=12.0,
        phase_spans=[
            {
                "phase": "live_generation",
                "duration_s": 10.0,
                "units": 32,
                "checkpoint_boundary": "group:dev-batch-g01",
                "pending_operations": [],
            }
        ],
        timestamps={
            "started_at_utc": "2026-09-15T00:00:00Z",
            "completed_at_utc": "2026-09-15T00:00:12Z",
        },
        model_identity={
            "hf_id": canary.MODEL_ID,
            "quantization": canary.QUANTIZATION,
            "gguf_path": "/cache/model.gguf",
            "revision": "revision",
            "gguf_sha256": "sha256:" + "c" * 64,
            "binary_sha256": "sha256:" + "d" * 64,
        },
    )
    assert artifact["batch_canary_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert canary.validate_artifact(artifact) == []

    failed_validation = deepcopy(validation)
    failed_validation["required_checks_passed"] = False
    failed_validation["failed_required_commands"] = ["focused_pytest"]
    disqualified = canary.assemble_artifact(
        canary.RUN_DATE,
        preconditions=[],
        schedule=schedule,
        per_call_rows=rows,
        replay=replay,
        capture=capture,
        validation=failed_validation,
        source_hashes={},
        duration_s=12.0,
        phase_spans=[],
        timestamps=artifact["timestamps"],
        model_identity=artifact["runtime_model_identity"],
    )
    assert disqualified["batch_canary_ready_score"] == 0
    assert disqualified["verdict_class"] == "disqualified"

    semantic_failure = deepcopy(replay)
    semantic_failure["receipt"]["semantic_controls_passed"] = False
    null = canary.assemble_artifact(
        canary.RUN_DATE,
        preconditions=[],
        schedule=schedule,
        per_call_rows=rows,
        replay=semantic_failure,
        capture=capture,
        validation=validation,
        source_hashes={},
        duration_s=12.0,
        phase_spans=[],
        timestamps=artifact["timestamps"],
        model_identity=artifact["runtime_model_identity"],
    )
    assert null["batch_canary_ready_score"] == 0
    assert null["verdict_class"] == "null"


def test_req_verify_7320_defensive_schedule_and_parser_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-7320 fails closed for malformed schedules and native bytes."""

    marker = tmp_path / "marker.txt"
    marker.write_text("bytes", encoding="utf-8")
    assert canary.sha256_file(marker) == canary.sha256_bytes(b"bytes")
    assert canary._utc_now().endswith("+00:00")
    with pytest.raises(ValueError, match="development_group_denominator"):
        canary.build_schedule({}, ["a", "b"])

    public = _panel()
    group_ids = [row["group_id"] for row in public["development_groups"][:2]]
    with pytest.raises(ValueError, match="development_selection"):
        canary.build_schedule(public, [group_ids[0], "missing"])
    with pytest.raises(ValueError, match="not_first_two"):
        canary.build_schedule(public, list(reversed(group_ids)))
    assert canary.schedule_errors([], {}, group_ids)[0].startswith("schedule_rebuild")

    schedule = canary.build_schedule(public, group_ids)
    short = deepcopy(schedule[:-1])
    short[0]["arm"] = "wrong"
    short[1]["claim_set_sha256"] = "changed"
    errors = canary.schedule_errors(short, public, group_ids)
    assert {"call_count", "arm_call_counts"}.issubset(errors)
    assert any(error.startswith("identity_control") for error in errors)
    assert any(error.startswith("arm_version_budget") for error in errors)

    source = schedule[0]
    scalar = canary._parse_model_content(source, "[]")
    assert scalar["parse_valid"] is False
    missing_fields = canary._parse_model_content(
        source, '{"completion":{"outcome":"bad","relations":null}}'
    )
    assert missing_fields["required_field_omissions"] == [
        "completion.outcome",
        "completion.relations",
    ]
    direct = next(row for row in schedule if row["call_type"] == "direct_batch")
    assert canary._parse_model_content(direct, "{}")["required_field_omissions"] == ["items"]
    missing_item = canary._parse_model_content(
        direct,
        canary.canonical_json(
            {"items": [{"claim_id": claim_id} for claim_id in direct["claim_ids"]]}
        ),
    )
    assert missing_item["required_field_omissions"] == ["items.decision"]

    response = _response_for(source)
    response["error"] = "TimeoutError:bounded"
    response["raw_request_bytes_b64"] = base64.b64encode(b"wrong").decode("ascii")
    response["raw_response_bytes_b64"] = base64.b64encode(b"[]").decode("ascii")
    failed = canary.build_per_call_row(source, response, {})
    assert {
        "transport_error:TimeoutError:bounded",
        "response_bytes_invalid",
        "request_bytes_mismatch",
    }.issubset(failed["failure_reasons"])
    response["raw_response_bytes_b64"] = base64.b64encode(b"\xff").decode("ascii")
    assert canary.build_per_call_row(source, response, {})["response_decode_error"].startswith(
        "UnicodeDecodeError"
    )

    rows = _valid_rows(schedule)
    missing = canary.cold_replay(public, schedule, rows[1:])
    assert missing["receipt"]["call_count"] == 31
    assert missing["receipt"]["replay_parser_match"] is False
    duplicate = canary.cold_replay(public, schedule, [*rows, rows[0]])
    assert duplicate["receipt"]["call_count"] == 31


def test_req_verify_7320_terminal_validator_rejects_inconsistent_shapes() -> None:
    """REQ-VERIFY-7320 checks every readiness-bearing terminal field."""

    assert canary.validate_artifact(None) == ["artifact_mapping"]
    assert canary.validate_artifact({})[0].startswith("missing_required_field")
    public = _panel()
    group_ids = [row["group_id"] for row in public["development_groups"][:2]]
    schedule = canary.build_schedule(public, group_ids)
    rows = _valid_rows(schedule)
    replay = canary.cold_replay(public, schedule, rows)
    validation = _passing_validation()
    capture = {
        "model_loaded": True,
        "model_invoked": True,
        "runtime_error": None,
        "gpu_receipts": {"provenance_ok": True},
        "runner_receipt": {"runner": "native_llama.cpp_server"},
    }
    artifact = canary.assemble_artifact(
        canary.RUN_DATE,
        preconditions=[],
        schedule=schedule,
        per_call_rows=rows,
        replay=replay,
        capture=capture,
        validation=validation,
        source_hashes={},
        duration_s=12.0,
        phase_spans=[],
        timestamps={"started_at_utc": "a", "completed_at_utc": "b"},
        model_identity={"hf_id": canary.MODEL_ID, "quantization": canary.QUANTIZATION},
    )

    corruptions = (
        ("schema", "bad"),
        ("run_date", "bad"),
        ("field_principles", {}),
        ("MODEL_SPECS", []),
        ("execution_host", ""),
        ("verifier_is_oracle", False),
        ("verdict_class", "bad"),
        ("invocation_counts", None),
        ("parser_replay_receipt", None),
    )
    observed_errors: set[str] = set()
    for field, value in corruptions:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = canary.artifact_checksum(changed)
        observed_errors.update(canary.validate_artifact(changed))
    assert {
        "schema",
        "run_identity",
        "field_principles",
        "MODEL_SPECS",
        "execution_identity",
        "verifier_is_oracle",
        "verdict_class",
        "invocation_counts",
        "parser_replay_receipt",
    }.issubset(observed_errors)

    wrong_checksum = deepcopy(artifact)
    wrong_checksum["reproducibility_checksum"] = "wrong"
    assert "reproducibility_checksum" in canary.validate_artifact(wrong_checksum)
    wrong_inference = deepcopy(artifact)
    wrong_inference["inference_mode"] = "cpu"
    wrong_inference["reproducibility_checksum"] = canary.artifact_checksum(wrong_inference)
    assert "inference_classification" in canary.validate_artifact(wrong_inference)

    wrong_score = deepcopy(artifact)
    wrong_score["batch_canary_ready_score"] = 0
    wrong_score["reproducibility_checksum"] = canary.artifact_checksum(wrong_score)
    assert {"batch_canary_ready_score", "verdict_class"}.issubset(
        canary.validate_artifact(wrong_score)
    )
    wrong_ready = deepcopy(artifact)
    wrong_ready["verdict_class"] = "null"
    wrong_ready["inference_substrate"] = "wrong"
    wrong_ready["reproducibility_checksum"] = canary.artifact_checksum(wrong_ready)
    assert {"verdict_class", "inference_substrate"}.issubset(canary.validate_artifact(wrong_ready))
    disqualified = deepcopy(artifact)
    disqualified["verdict_class"] = "disqualified"
    disqualified["reproducibility_checksum"] = canary.artifact_checksum(disqualified)
    assert "disqualified_without_validation_failure" in canary.validate_artifact(disqualified)

    blocked = canary.blocked_artifact(
        canary.RUN_DATE,
        [
            canary.gate_row(
                "blocked", "upstream", "field", True, False, False, "External failure blocks."
            )
        ],
        duration_s=0.1,
        timestamps={"started_at_utc": "a", "completed_at_utc": "b"},
    )
    blocked["batch_canary_ready_score"] = 1
    blocked["reproducibility_checksum"] = canary.artifact_checksum(blocked)
    assert "blocked_terminal_state" in canary.validate_artifact(blocked, require_validation=False)

    assert canary._date_argument(canary.RUN_DATE) == canary.RUN_DATE
    with pytest.raises(Exception, match="--date"):
        canary._date_argument("bad")


def test_req_verify_7320_all_measured_nonready_terminal_classes() -> None:
    """REQ-VERIFY-7320 separates runtime blocks from completed null findings."""

    public = _panel()
    group_ids = [row["group_id"] for row in public["development_groups"][:2]]
    schedule = canary.build_schedule(public, group_ids)
    rows = _valid_rows(schedule)
    replay = canary.cold_replay(public, schedule, rows)
    validation = _passing_validation()
    base_capture = {
        "model_loaded": True,
        "model_invoked": True,
        "runtime_error": None,
        "gpu_receipts": {"provenance_ok": True},
        "runner_receipt": {},
    }

    def assembled(
        selected_rows: list[dict[str, object]],
        selected_replay: dict[str, object],
        capture: dict[str, object],
        duration: float = 12.0,
    ) -> dict[str, object]:
        return canary.assemble_artifact(
            canary.RUN_DATE,
            preconditions=[],
            schedule=schedule,
            per_call_rows=selected_rows,
            replay=selected_replay,
            capture=capture,
            validation=validation,
            source_hashes={},
            duration_s=duration,
            phase_spans=[],
            timestamps={"started_at_utc": "a", "completed_at_utc": "b"},
            model_identity={},
        )

    partial_capture = {**base_capture, "runtime_error": "TimeoutError:external"}
    partial_replay = canary.cold_replay(public, schedule, rows[:1])
    assert assembled(rows[:1], partial_replay, partial_capture)["verdict_class"] == "blocked"
    unusable = deepcopy(replay)
    unusable["receipt"]["transport_complete"] = False
    assert assembled(rows, unusable, base_capture)["honest_verdict"].endswith("transport_unusable")
    no_provenance = {**base_capture, "gpu_receipts": {"provenance_ok": False}}
    assert assembled(rows, replay, no_provenance)["honest_verdict"].endswith("provenance_failed")
    assert assembled(rows, replay, base_capture, duration=1.0)["verdict_class"] == "null"
