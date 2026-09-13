"""Tests for REQ-VERIFY-7277 and SCENARIO-VERIFY-7277-*.

The tests use synthetic native response bytes. They never load a model, claim
GPU work, or write into the repository result directories.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7275_v640_semantic_replay as replay
from carnot import experiment_7277_v640_comparator_canary as canary


JsonDict = dict[str, Any]


def _contract() -> JsonDict:
    return replay.build_comparator_contract({"embedded_chat_template_sha256": "sha256:template"})


def _native_response(content: str, finish_reason: str = "stop") -> tuple[JsonDict, bytes]:
    body = {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content, "role": "assistant"},
            }
        ],
        "model": "qwen3.8-27b",
        "timings": {"prompt_ms": 1.0, "predicted_ms": 2.0},
        "usage": {"completion_tokens": 5, "prompt_tokens": 20},
    }
    return body, replay.canonical_json(body).encode("utf-8")


def _completion(
    sealed: JsonDict,
    content: str,
    *,
    finish_reason: str = "stop",
    resource: JsonDict | None = None,
) -> JsonDict:
    payload, request_bytes = replay.build_native_request(
        sealed["prompt"], sealed["grammar"], sealed["seed"], 128
    )
    body, response_bytes = _native_response(content, finish_reason)
    response = {
        "raw_request": payload,
        "raw_request_bytes_b64": base64.b64encode(request_bytes).decode("ascii"),
        "raw_response": body,
        "raw_response_bytes_b64": base64.b64encode(response_bytes).decode("ascii"),
        "raw_completion": content,
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "finish_reason": finish_reason,
        "latency_s": 0.25,
        "error": None,
        "started_at_utc": "2026-09-13T15:00:00Z",
        "completed_at_utc": "2026-09-13T15:00:01Z",
    }
    return canary.build_completion_row(
        sealed,
        response,
        resource
        or {
            "server_pid": 123,
            "server_pid_start_ticks": 456,
            "gpu_uuid": "GPU-test",
            "lease_id": "lease-test",
            "cuda_offload_confirmed": True,
            "gpu_sample_sha256": "sha256:sample",
        },
    )


def _complete_rows(schedule: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for sealed in schedule:
        expected = canary.DEVELOPMENT_AUTHORITY[sealed["unit_id"]]
        content = replay.canonical_json({"decision": expected})
        rows.append(_completion(sealed, content))
    return rows


def _identity() -> JsonDict:
    return {
        "hf_id": canary.MODEL_ID,
        "quantization": canary.QUANTIZATION,
        "gguf_path": "/cache/model.gguf",
        "revision": "revision-123",
        "gguf_bytes": 16_000_000_000,
        "gguf_sha256": "sha256:model",
        "embedded_tokenizer_sha256": "sha256:tokenizer",
        "embedded_chat_template_sha256": "sha256:template",
        "embedded_chat_template_present": True,
        "auto_tokenizer_used": False,
        "runtime": "native_llama.cpp_server",
    }


def _runner() -> JsonDict:
    return {
        "model_count": 1,
        "replica_count": 1,
        "runner": "native_llama.cpp_server",
        "dual_gpu_runner_used": False,
        "command": ["llama-server", "--parallel", "1"],
        "server_props": {"chat_template": "embedded", "n_ctx": 8192},
        "kv_headroom": {"measured": True, "headroom_mb": 4096},
        "server_identity": {"pid": 123, "start_time_ticks": 456},
        "model_revision": "revision-123",
        "model_sha256": "sha256:model",
        "invocation_counts": {
            "model_loads_attempted": 1,
            "model_loads_completed": 1,
            "generation_calls_attempted": 16,
            "generation_calls_completed": 16,
            "usable_answers": 16,
        },
    }


def _gpu_receipt() -> JsonDict:
    return {
        "provenance_ok": True,
        "server_identity": {"pid": 123, "start_time_ticks": 456},
        "generation_overlap_count": 16,
        "generation_overlap_denominator": 16,
        "generation_overlap_ok": True,
        "generation_overlap_samples": [{"task_owned_vram_mb": 19_000}],
        "lease_release": {"released": True},
        "cleanup": {"leak_free": True},
    }


def _passing_receipts() -> list[JsonDict]:
    return [
        {
            "name": name,
            "command": name,
            "exit_code": 0,
            "duration_s": 0.1,
            "log_path": f"validation/{name}.log",
            "log_sha256": f"sha256:{name}",
            "passed": True,
            "timed_out": False,
        }
        for name in canary.REQUIRED_VALIDATION_NAMES
    ]


def _measured_artifact() -> JsonDict:
    contract = _contract()
    schedule = canary.build_schedule(contract)
    completions = _complete_rows(schedule)
    replay_rows, errors = canary.independent_replay(schedule, completions)
    grammar_rows = canary.request_grammar_rows(schedule, completions)
    accuracy = canary.semantic_accuracy_rows(replay_rows)
    artifact = canary.base_artifact(canary.RUN_DATE)
    artifact["preconditions_checked"] = [
        canary.gate_row("upstream", True, True, True, upstream="exp7275", field="ready")
    ]
    artifact["source_artifact_hashes"] = {
        "upstream": {"sha256": "sha256:upstream", "retired": False, "quarantined": False}
    }
    artifact["model_identity_receipt"] = _identity()
    artifact["gpu_receipts"] = _gpu_receipt()
    artifact["runner_receipt"] = _runner()
    artifact["raw_call_manifest"] = {
        "status": "complete",
        "raw_call_count": 16,
        "manifest_sha256": "sha256:manifest",
    }
    artifact["validation_receipts"] = _passing_receipts()
    return canary.finalize_measured_artifact(
        artifact,
        schedule,
        completions,
        replay_rows,
        errors,
        grammar_rows,
        accuracy,
        duration_s=12.0,
    )


def test_schedule_is_sealed_before_inference() -> None:
    """REQ-VERIFY-7277 / SCENARIO-VERIFY-7277-SCHEDULE."""

    prompts = canary.development_prompts()
    schedule = canary.build_schedule(_contract(), prompts)

    assert len(prompts) == 8
    assert {row["expected_decision"] for row in prompts} == {
        "supported",
        "contradicted",
        "unknown",
    }
    assert any(row["boundary"] == "duplicate_mention" for row in prompts)
    assert len(schedule) == 16
    assert canary.schedule_errors(schedule, _contract()) == []
    assert all("expected_decision" not in row for row in schedule)
    assert all(row["output_token_budget"] == 128 for row in schedule)
    for unit_id in {row["unit_id"] for row in schedule}:
        pair = [row for row in schedule if row["unit_id"] == unit_id]
        assert [row["arm"] for row in pair] == ["constrained", "unconstrained"]
        assert pair[0]["prompt"] == pair[1]["prompt"]
        assert pair[0]["grammar"] == replay.DIRECT_GRAMMAR
        assert pair[1]["grammar"] == ""

    with pytest.raises(ValueError, match="two fixed draw seeds"):
        canary.build_schedule({})


def test_schedule_validator_names_each_frozen_contract_drift() -> None:
    """REQ-VERIFY-7277 fails closed on schedule changes before loading."""

    contract = _contract()
    baseline = canary.build_schedule(contract)
    changed = deepcopy(baseline[:-1])
    changed[0]["unit_id"] = "extra_unit"
    changed[7]["prompt"] = "changed"
    changed[2]["arm"] = "wrong_arm"
    for key, value in (
        ("call_order", -1),
        ("grammar", "changed"),
        ("grammar_requested", False),
        ("output_token_budget", 1),
        ("seed", -1),
        ("retry_budget", 1),
        ("development_only", False),
        ("held_out_eligible", True),
        ("expected_decision", "supported"),
    ):
        changed[4][key] = value
    changed[4]["decoding_parameters"] = {"seed": -1, "temperature": 1.0}

    errors = canary.schedule_errors(changed, contract)

    assert "call_denominator" in errors
    assert "unit_denominator" in errors
    assert any(error.endswith(":arm_pair") for error in errors)
    assert any(error.endswith(":prompt_pair") for error in errors)
    assert {error.split(":")[-1] for error in errors} >= {
        "call_order",
        "grammar",
        "grammar_requested",
        "token_budget",
        "seed",
        "decoding_seed",
        "temperature",
        "retry_budget",
        "development_only",
        "held_out",
        "authority_hidden",
    }


def test_replay_joins_exact_native_bytes_and_preserves_failures() -> None:
    """REQ-VERIFY-7277 / SCENARIO-VERIFY-7277-REPLAY."""

    schedule = canary.build_schedule(_contract())
    completions = _complete_rows(schedule)
    bad_index = next(i for i, row in enumerate(schedule) if row["arm"] == "unconstrained")
    completions[bad_index] = _completion(schedule[bad_index], "not json")

    rows, errors = canary.independent_replay(schedule, completions)
    grammar_rows = canary.request_grammar_rows(schedule, completions)

    assert errors == []
    assert len(rows) == 16
    assert sum(row["request_response_joined"] for row in rows) == 16
    assert rows[bad_index]["parse_valid"] is False
    assert rows[bad_index]["error"] == "parser_rejection"
    assert sum(row["grammar_forwarded_exactly"] for row in grammar_rows) == 8
    assert all(
        row["grammar_forwarded_exactly"] for row in grammar_rows if row["arm"] == "constrained"
    )


def test_replay_rejects_request_response_and_parser_drift() -> None:
    """REQ-VERIFY-7277 rejects changes after the native call."""

    schedule = canary.build_schedule(_contract())
    completions = _complete_rows(schedule)
    request_mutation = deepcopy(completions)
    request_mutation[0]["raw_request_bytes_b64"] = base64.b64encode(b"{}").decode("ascii")
    _rows, request_errors = canary.independent_replay(schedule, request_mutation)
    assert "call_0:request_bytes" in request_errors

    response_mutation = deepcopy(completions)
    body, response_bytes = _native_response('{"decision":"unknown"}')
    response_mutation[0]["raw_response"] = body
    response_mutation[0]["raw_response_bytes_b64"] = base64.b64encode(response_bytes).decode(
        "ascii"
    )
    _rows, response_errors = canary.independent_replay(schedule, response_mutation)
    assert "call_0:raw_completion" in response_errors

    malformed_bytes = deepcopy(completions)
    malformed_bytes[0]["raw_response_bytes_b64"] = "not-base64"
    _rows, malformed_errors = canary.independent_replay(schedule, malformed_bytes)
    assert malformed_errors == ["call_0:base64_invalid"]

    with pytest.raises(ValueError, match="base64_type"):
        canary._decode_b64(None)

    missing = completions[:-1]
    _rows, missing_errors = canary.independent_replay(schedule, missing)
    assert "replay_denominator" in missing_errors
    assert "call_15:missing_completion" in missing_errors

    metadata_mutation = deepcopy(completions)
    metadata_mutation[0]["actual_parameters"] = {}
    metadata_mutation[0]["raw_response"] = {}
    metadata_mutation[0]["decision"] = "unknown"
    _rows, metadata_errors = canary.independent_replay(schedule, metadata_mutation)
    assert {"call_0:actual_parameters", "call_0:raw_response", "call_0:parser_reduction"} <= set(
        metadata_errors
    )


def test_completion_row_preserves_transport_failure() -> None:
    """REQ-VERIFY-7277 keeps an unusable native attempt in its denominator."""

    sealed = canary.build_schedule(_contract())[0]
    payload, _request_bytes = replay.build_native_request(
        sealed["prompt"], sealed["grammar"], sealed["seed"], 128
    )
    failed = canary.build_completion_row(
        sealed,
        {
            "raw_request": payload,
            "raw_request_bytes_b64": "not-base64",
            "raw_response": {},
            "raw_response_bytes_b64": "",
            "raw_completion": "",
            "finish_reason": None,
            "latency_s": 1.0,
            "error": "TimeoutError:request",
        },
        {},
    )

    assert failed["transport_complete"] is False
    assert failed["terminal_state"] == "failed"
    assert failed["parser_classification"] == "transport_error"
    assert failed["error"] == "TimeoutError:request"


def test_readiness_uses_fidelity_not_development_accuracy() -> None:
    """REQ-VERIFY-7277 / SCENARIO-VERIFY-7277-GATE."""

    schedule = canary.build_schedule(_contract())
    completions = _complete_rows(schedule)
    rows, errors = canary.independent_replay(schedule, completions)
    grammar_rows = canary.request_grammar_rows(schedule, completions)
    accurate = canary.semantic_accuracy_rows(rows)
    wrong = [dict(row, correct=False) for row in accurate]

    first = canary.readiness_receipt(
        rows,
        errors,
        grammar_rows,
        _identity(),
        _gpu_receipt(),
        _runner(),
        {"status": "complete", "raw_call_count": 16},
    )
    second = canary.readiness_receipt(
        rows,
        errors,
        grammar_rows,
        _identity(),
        _gpu_receipt(),
        _runner(),
        {"status": "complete", "raw_call_count": 16},
    )

    assert sum(row["correct"] for row in accurate) == 16
    assert sum(row["correct"] for row in wrong) == 0
    assert first == second
    assert first["comparator_canary_ready_score"] == 1


def test_preconditions_authenticate_upstream_contract_and_code(tmp_path: Path) -> None:
    """REQ-VERIFY-7277 / SCENARIO-VERIFY-7277-PREFLIGHT."""

    artifact_path = tmp_path / canary.UPSTREAM_ARTIFACT_PATH
    contract_path = tmp_path / canary.UPSTREAM_CONTRACT_PATH
    module_path = tmp_path / replay.MODULE_PATH
    wrapper_path = tmp_path / replay.WRAPPER_PATH
    artifact_path.parent.mkdir(parents=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_bytes(b"module")
    wrapper_path.write_bytes(b"wrapper")
    contract = _contract()
    contract_path.write_text(replay.canonical_json(contract), encoding="utf-8")
    upstream = {
        "status": "complete",
        "semantic_replay_ready_score": 1,
        "comparator_contract_path": {
            "path": canary.UPSTREAM_CONTRACT_PATH.as_posix(),
            "sha256": canary.sha256_file(contract_path),
        },
        "source_artifact_hashes": {
            "module": canary.sha256_file(module_path),
            "entrypoint": canary.sha256_file(wrapper_path),
        },
        "quarantined": False,
        "flagged_adversarial": False,
    }
    artifact_path.write_text(replay.canonical_json(upstream), encoding="utf-8")
    expected = {
        canary.UPSTREAM_ARTIFACT_PATH: canary.sha256_file(artifact_path),
        canary.UPSTREAM_CONTRACT_PATH: canary.sha256_file(contract_path),
    }

    checks, observed_contract = canary.authenticate_inputs(
        tmp_path,
        expected_hashes=expected,
        upstream_validator=lambda _value: [],
        exclusion_manifest={},
    )

    assert observed_contract == contract
    assert all(row["passed"] for row in checks)
    artifact_path.unlink()
    failed, _contract_value = canary.authenticate_inputs(
        tmp_path,
        expected_hashes=expected,
        upstream_validator=lambda _value: [],
        exclusion_manifest={},
    )
    summary = canary.gate_summary(failed)
    assert summary["failed_check"] == "authenticated_input"
    assert summary["upstream"] == canary.UPSTREAM_ARTIFACT_PATH.as_posix()


def test_real_upstream_identity_authenticates_without_model_work() -> None:
    """REQ-VERIFY-7277 authenticates the shipped Exp7275 code and contract."""

    root = Path(__file__).resolve().parents[2]
    checks, contract = canary.authenticate_inputs(root)

    assert all(row["passed"] for row in checks)
    assert contract["contract_sha256"] == canary._contract_checksum(contract)


def test_input_parse_and_retirement_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-7277 keeps malformed and retired prerequisites terminal."""

    artifact_path = tmp_path / canary.UPSTREAM_ARTIFACT_PATH
    contract_path = tmp_path / canary.UPSTREAM_CONTRACT_PATH
    artifact_path.parent.mkdir(parents=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("{bad", encoding="utf-8")
    contract_path.write_text("{}", encoding="utf-8")
    expected = {
        canary.UPSTREAM_ARTIFACT_PATH: canary.sha256_file(artifact_path),
        canary.UPSTREAM_CONTRACT_PATH: canary.sha256_file(contract_path),
    }
    checks, contract = canary.authenticate_inputs(
        tmp_path,
        expected_hashes=expected,
        upstream_validator=lambda _value: [],
        exclusion_manifest={},
    )
    assert contract == {}
    assert canary.gate_summary(checks)["failed_check"] == "input_parse"

    assert canary._manifest_lists_experiment({"experiment_id": 7277}, 7277)
    assert canary._manifest_lists_experiment({"experiment_ids": [7275]}, 7275)
    assert canary._manifest_lists_experiment([{"experiment_id": 7277}], 7277)
    assert not canary._manifest_lists_experiment("7277", 7277)


def test_terminal_artifacts_are_cold_valid_and_fail_closed() -> None:
    """REQ-VERIFY-7277 / SCENARIO-VERIFY-7277-E2E."""

    artifact = _measured_artifact()
    assert artifact["status"] == "complete"
    assert artifact["comparator_canary_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert canary.validate_artifact(artifact) == []

    for field in (
        "MODEL_SPECS",
        "invocation_counts",
        "request_grammar_rows",
        "runner_receipt",
        "raw_call_manifest",
        "validation_receipts",
    ):
        changed = deepcopy(artifact)
        changed[field] = None
        changed["reproducibility_checksum"] = canary.artifact_checksum(changed)
        assert canary.validate_artifact(changed)

    blocked = canary.base_artifact(canary.RUN_DATE)
    checks = [
        canary.gate_row(
            "model_cache",
            True,
            False,
            False,
            upstream="cached_sota_pair",
            field="model_path",
        )
    ]
    canary.finalize_blocked_artifact(blocked, checks, duration_s=0.5)
    assert blocked["status"] == "blocked"
    assert blocked["invocation_counts"] == canary.ZERO_INVOCATION_COUNTS
    assert blocked["gate_check_summary"]["failed_check"] == "model_cache"
    assert canary.validate_artifact(blocked) == []


def test_terminal_validator_exercises_each_fail_closed_boundary() -> None:
    """REQ-VERIFY-7277 rejects terminal field and denominator drift."""

    source = _measured_artifact()

    def errors_for(**changes: Any) -> list[str]:
        value = deepcopy(source)
        value.update(changes)
        value["reproducibility_checksum"] = canary.artifact_checksum(value)
        return canary.validate_artifact(value)

    assert canary.validate_artifact(None) == ["artifact_mapping"]
    missing = deepcopy(source)
    missing.pop("schema")
    assert canary.validate_artifact(missing) == ["missing_required_field:schema"]
    assert "duration_s" in errors_for(duration_s=-1)

    checksum = deepcopy(source)
    checksum["reproducibility_checksum"] = "wrong"
    assert "reproducibility_checksum" in canary.validate_artifact(checksum)
    assert "invocation_counts" in errors_for(invocation_counts={})

    blocked = canary.base_artifact(canary.RUN_DATE)
    blocked["status"] = "blocked"
    blocked["verdict_class"] = "blocked"
    blocked["honest_verdict"] = "blocked_bad"
    blocked["reproducibility_checksum"] = canary.artifact_checksum(blocked)
    assert "blocked_terminal_state" in canary.validate_artifact(blocked)
    assert "status" in errors_for(status="running")
    assert "live_inference_substrate" in errors_for(inference_mode="not_invoked")
    assert "sample_size_budget" in errors_for(sample_size_budget=None)
    budget = deepcopy(source["sample_size_budget"])
    budget["completed_calls"] = 15
    assert "sample_size_budget" in errors_for(sample_size_budget=budget)
    assert "rows" in errors_for(rows=[])
    assert "canary_rows" in errors_for(canary_rows=[])
    assert "request_grammar_rows" in errors_for(request_grammar_rows=[])
    assert "semantic_accuracy_rows" in errors_for(semantic_accuracy_rows=[])
    assert "readiness_receipt" in errors_for(readiness_receipt={})
    assert "comparator_canary_ready_score" in errors_for(comparator_canary_ready_score=0)
    assert "verdict_class" in errors_for(verdict_class="null")
    assert "oracle_positive" in errors_for(verdict_class="positive")
    receipts = deepcopy(source["validation_receipts"])
    receipts[0]["passed"] = False
    assert "validation_receipts" in errors_for(validation_receipts=receipts)
    counts = deepcopy(source["invocation_counts"])
    counts["generation_calls_attempted"] = 15
    assert "invocation_counts" in errors_for(invocation_counts=counts)
    runner = deepcopy(source["runner_receipt"])
    runner["invocation_counts"] = {}
    assert "runner_invocation_counts" in errors_for(runner_receipt=runner)

    not_ready = deepcopy(source)
    not_ready["rows"][0]["parse_valid"] = False
    not_ready["rows"][0]["decision"] = None
    not_ready["readiness_receipt"] = canary.readiness_receipt(
        not_ready["rows"],
        not_ready["replay_discrepancies"],
        not_ready["request_grammar_rows"],
        not_ready["model_identity_receipt"],
        not_ready["gpu_receipts"],
        not_ready["runner_receipt"],
        not_ready["raw_call_manifest"],
    )
    not_ready["comparator_canary_ready_score"] = 0
    not_ready["verdict_class"] = "circular_positive"
    not_ready["reproducibility_checksum"] = canary.artifact_checksum(not_ready)
    assert "verdict_class" in canary.validate_artifact(not_ready)

    load_only = canary.base_artifact(canary.RUN_DATE)
    load_only["runner_receipt"] = {"invocation_counts": dict(canary.ZERO_INVOCATION_COUNTS)}
    load_only["runner_receipt"]["invocation_counts"]["model_loads_completed"] = 1
    canary.finalize_measured_artifact(load_only, [], [], [], [], [], [], duration_s=2.0)
    assert load_only["inference_substrate_class"] == "model_load_no_generation"
    assert load_only["verdict_class"] == "null"


def test_raw_manifest_replays_all_sixteen_calls(tmp_path: Path) -> None:
    """REQ-VERIFY-7277 retains raw bytes for independent E2E replay."""

    schedule = canary.build_schedule(_contract())
    rows = _complete_rows(schedule)
    raw_dir = tmp_path / "raw"
    manifest = canary.write_raw_manifest(raw_dir, schedule, rows, _identity())

    replayed, errors = canary.independent_replay_from_raw(raw_dir)

    assert manifest["raw_call_count"] == 16
    assert len(list(raw_dir.glob("call_*.json"))) == 16
    assert len(replayed) == 16
    assert errors == []
    assert canary.sha256_file(raw_dir / "raw_call_manifest.json") == manifest["manifest_sha256"]

    prompt_receipt = canary.seal_development_inputs(raw_dir, _contract())
    assert prompt_receipt["development_prompt_count"] == 8
    assert prompt_receipt["model_worker_authority_read_count"] == 0
    canary.seal_development_inputs(raw_dir, _contract())
    with pytest.raises(ValueError, match="existing raw evidence differs"):
        canary._write_or_match(raw_dir / "development_prompts.json", {"prompts": []})


def test_raw_replay_reports_missing_schedule_and_call_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-7277 reports the first raw replay boundary failure."""

    rows, errors = canary.independent_replay_from_raw(tmp_path)
    assert rows == []
    assert errors[0].startswith("schedule:FileNotFoundError")

    schedule = canary.build_schedule(_contract())
    completions = _complete_rows(schedule)
    raw_dir = tmp_path / "raw"
    canary.write_raw_manifest(raw_dir, schedule, completions, _identity())
    call_path = raw_dir / "call_00.json"
    value = json.loads(call_path.read_text(encoding="utf-8"))
    value["schedule"] = {}
    call_path.write_text(json.dumps(value), encoding="utf-8")
    (raw_dir / "call_01.json").unlink()

    _rows, errors = canary.independent_replay_from_raw(raw_dir)
    assert "call_0:schedule" in errors
    assert any(error.startswith("call_1:FileNotFoundError") for error in errors)


def test_validation_commands_are_focused_and_entrypoint_is_thin(tmp_path: Path) -> None:
    """REQ-VERIFY-7277 forbids bare or repository-wide validation."""

    commands = canary.validation_commands(tmp_path, tmp_path / "raw")
    rendered = {name: " ".join(command) for name, command in commands}

    assert tuple(rendered) == canary.REQUIRED_VALIDATION_NAMES
    assert "-n 0" in rendered["focused_pytest"]
    assert "-o addopts=" in rendered["focused_pytest"]
    assert canary.TEST_PATH.as_posix() in rendered["focused_pytest"]
    assert "--fail-under=100" in rendered["scoped_coverage_report"]
    assert "scripts/check_spec_coverage.py" in rendered["scoped_spec_coverage"]
    assert "scripts/adversarial_verify.py" in rendered["adversarial_verify"]
    assert "scripts/verdict_row_consistency_lint.py" in rendered["verdict_row_consistency"]
    assert all("tests/python -q" not in value for value in rendered.values())

    wrapper = Path("scripts/experiments/experiment_7277_v640_comparator_canary.py")
    if wrapper.is_file():
        text = wrapper.read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 12
        assert "main" in text

    assert canary._date_argument(canary.RUN_DATE) == canary.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        canary._date_argument("20260912")


@pytest.mark.parametrize("field", ["status", "semantic_replay_ready_score"])
def test_upstream_contract_failures_are_named(tmp_path: Path, field: str) -> None:
    """REQ-VERIFY-7277 names the exact failed upstream field."""

    artifact_path = tmp_path / canary.UPSTREAM_ARTIFACT_PATH
    contract_path = tmp_path / canary.UPSTREAM_CONTRACT_PATH
    module_path = tmp_path / replay.MODULE_PATH
    wrapper_path = tmp_path / replay.WRAPPER_PATH
    for path in (artifact_path, contract_path, module_path, wrapper_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_bytes(b"module")
    wrapper_path.write_bytes(b"wrapper")
    contract = _contract()
    contract_path.write_text(replay.canonical_json(contract), encoding="utf-8")
    upstream = {
        "status": "complete",
        "semantic_replay_ready_score": 1,
        "comparator_contract_path": {
            "path": canary.UPSTREAM_CONTRACT_PATH.as_posix(),
            "sha256": canary.sha256_file(contract_path),
        },
        "source_artifact_hashes": {
            "module": canary.sha256_file(module_path),
            "entrypoint": canary.sha256_file(wrapper_path),
        },
    }
    upstream[field] = "bad" if field == "status" else 0
    artifact_path.write_text(replay.canonical_json(upstream), encoding="utf-8")
    expected = {
        canary.UPSTREAM_ARTIFACT_PATH: canary.sha256_file(artifact_path),
        canary.UPSTREAM_CONTRACT_PATH: canary.sha256_file(contract_path),
    }

    checks, _ = canary.authenticate_inputs(
        tmp_path,
        expected_hashes=expected,
        upstream_validator=lambda _value: [],
        exclusion_manifest={},
    )
    failure = next(row for row in checks if not row["passed"])

    assert failure["field"] == field
    assert failure["upstream"] == "exp7275-semantic-replay"
