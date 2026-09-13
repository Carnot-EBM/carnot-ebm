"""Tests for REQ-VERIFY-7278 and SCENARIO-VERIFY-7278-*.

The tests use deterministic response bytes. They do not load a model or write
to repository result paths.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7278_v640_source_measurement as measurement


JsonDict = dict[str, Any]


def _response(sealed: JsonDict, content: str, *, finish_reason: str = "stop") -> JsonDict:
    payload, request_bytes = measurement.request_payload(sealed)
    body = {
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "timings": {"prompt_ms": 2.0, "predicted_ms": 3.0},
        "usage": {"completion_tokens": 8, "prompt_tokens": 30},
    }
    response_bytes = measurement.canonical_json(body).encode("utf-8")
    return {
        "raw_request": payload,
        "raw_request_bytes_b64": base64.b64encode(request_bytes).decode("ascii"),
        "raw_response": body,
        "raw_response_bytes_b64": base64.b64encode(response_bytes).decode("ascii"),
        "raw_completion": content,
        "prompt_tokens": 30,
        "completion_tokens": 8,
        "finish_reason": finish_reason,
        "latency_s": 0.25,
        "error": None,
    }


def _resource() -> JsonDict:
    return {
        "server_pid": 123,
        "server_pid_start_ticks": 456,
        "gpu_uuid": "GPU-test",
        "lease_id": "lease-test",
        "cuda_offload_confirmed": True,
        "gpu_sample_sha256": "sha256:sample",
    }


def _perfect_capture() -> tuple[JsonDict, JsonDict, list[JsonDict], list[JsonDict]]:
    public, authority = measurement.build_source_fixture()
    schedule = measurement.build_schedule(public["rows"])
    private = {row["unit_id"]: row for row in authority["rows"]}
    completions: list[JsonDict] = []
    for sealed in schedule:
        hidden = private[sealed["unit_id"]]
        if sealed["arm"] == "mention_pointer":
            completion = hidden[f"gold_{sealed['call_type']}_completion"]
        else:
            completion = {"decision": hidden["expected_decision"]}
        completions.append(
            measurement.build_completion_row(
                sealed,
                _response(sealed, measurement.canonical_json(completion)),
                _resource(),
            )
        )
    return public, authority, schedule, completions


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
        for name in measurement.REQUIRED_VALIDATION_NAMES
    ]


def _identity() -> JsonDict:
    return {
        "hf_id": measurement.MODEL_ID,
        "quantization": measurement.QUANTIZATION,
        "gguf_path": "/cache/model.gguf",
        "revision": "revision-test",
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
        "server_props": {"available": True},
        "kv_headroom": {"measured": True, "headroom_mb": 2048},
        "server_identity": {"pid": 123, "start_time_ticks": 456},
        "model_revision": "revision-test",
        "model_sha256": "sha256:model",
        "invocation_counts": {
            "model_loads_attempted": 1,
            "model_loads_completed": 1,
            "generation_calls_attempted": 256,
            "generation_calls_completed": 256,
            "usable_answers": 256,
        },
    }


def test_fresh_fixture_and_schedule_hide_private_authority() -> None:
    """REQ-VERIFY-7278 / SCENARIO-VERIFY-7278-SCHEDULE."""

    public, authority = measurement.build_source_fixture()
    schedule = measurement.build_schedule(public["rows"])
    receipt = measurement.selection_receipt(public, authority, schedule)

    assert len(public["rows"]) == len(authority["rows"]) == 64
    assert authority["condition_counts"] == {
        "consistent_entity_renaming": 16,
        "insufficient_evidence": 16,
        "relation_reversal": 16,
        "supported": 16,
    }
    assert len(schedule) == 256
    assert measurement.schedule_errors(schedule, public["rows"]) == []
    assert receipt["roster_frozen_before_inference"] is True
    assert receipt["authority_fields_in_schedule"] == 0
    assert receipt["exp7265_held_out_overlap"] == []
    assert len(receipt["source_shuffle_permutation"]) == 64
    assert set(receipt["source_shuffle_permutation"]) == {row["unit_id"] for row in public["rows"]}
    assert all(row["output_token_budget"] == 128 for row in schedule)
    assert all(row["request_timeout_s"] == measurement.REQUEST_CAP_S for row in schedule)
    assert all(row["retry_budget"] == 0 for row in schedule)
    assert all("expected_decision" not in row and "condition" not in row for row in schedule)
    assert all("exp7265" not in measurement.canonical_json(row) for row in schedule)
    for unit_id in {row["unit_id"] for row in schedule}:
        calls = [row for row in schedule if row["unit_id"] == unit_id]
        assert {row["arm"] for row in calls} == {
            "mention_pointer",
            "direct_self_consistency",
        }
        assert sorted(row["call_type"] for row in calls) == [
            "claim",
            "direct",
            "direct",
            "source",
        ]

    broken = deepcopy(schedule)
    broken[0]["output_token_budget"] = 127
    assert "call_0:output_token_budget" in measurement.schedule_errors(broken, public["rows"])
    with pytest.raises(ValueError, match="64 public units"):
        measurement.build_schedule(public["rows"][:-1])


def test_independent_replay_and_reduction_keep_all_units() -> None:
    """REQ-VERIFY-7278 / SCENARIO-VERIFY-7278-REDUCE and E2E."""

    public, authority, schedule, completions = _perfect_capture()
    replay_rows, replay_errors = measurement.independent_replay(schedule, completions)
    reduced = measurement.reduce_measurement(
        public["rows"], authority["rows"], schedule, completions, replay_rows
    )

    assert replay_errors == []
    assert len(replay_rows) == 256
    assert len(reduced["rows"]) == 128
    assert len(reduced["source_fidelity_rows"]) == 128
    assert len(reduced["source_intervention_rows"]) == 64
    assert all(row["decision_correct"] for row in reduced["rows"])
    assert all(
        row["source_fidelity"] is True
        for row in reduced["source_fidelity_rows"]
        if row["arm"] == "mention_pointer"
    )
    assert all(
        row["source_fidelity"] is None
        for row in reduced["source_fidelity_rows"]
        if row["arm"] == "direct_self_consistency"
    )
    direct = [row for row in reduced["rows"] if row["arm"] == "direct_self_consistency"]
    assert all(row["one_shot_decision"] == row["prediction"] for row in direct)
    assert all(row["self_consistency_tie"] is False for row in direct)
    shuffled = [
        row
        for row in reduced["source_intervention_rows"]
        if row["intervention"] == "source_shuffle"
    ]
    assert len(shuffled) == 16
    assert all(row["additional_model_calls"] == 0 for row in shuffled)

    corrupted = deepcopy(completions)
    corrupted[0]["raw_request_bytes_b64"] = base64.b64encode(b"changed").decode("ascii")
    _, errors = measurement.independent_replay(schedule, corrupted)
    assert "call_0:request_bytes" in errors


def test_direct_tie_becomes_unknown_and_quality_does_not_lower_completeness() -> None:
    """REQ-VERIFY-7278 / SCENARIO-VERIFY-7278-COMPLETENESS."""

    public, authority, schedule, completions = _perfect_capture()
    direct_indices = [
        index
        for index, row in enumerate(schedule)
        if row["unit_id"] == public["rows"][0]["unit_id"]
        and row["arm"] == "direct_self_consistency"
    ]
    assert len(direct_indices) == 2
    second = direct_indices[1]
    completions[second] = measurement.build_completion_row(
        schedule[second],
        _response(schedule[second], json.dumps({"decision": "unknown"})),
        _resource(),
    )
    replay_rows, errors = measurement.independent_replay(schedule, completions)
    assert errors == []
    reduced = measurement.reduce_measurement(
        public["rows"], authority["rows"], schedule, completions, replay_rows
    )
    direct = next(
        row
        for row in reduced["rows"]
        if row["unit_id"] == public["rows"][0]["unit_id"]
        and row["arm"] == "direct_self_consistency"
    )
    assert direct["self_consistency_tie"] is True
    assert direct["prediction"] == "unknown"
    assert direct["one_shot_decision"] != "unknown"

    receipt = measurement.completeness_receipt(
        schedule,
        completions,
        replay_rows,
        [],
        public["rows"],
        authority["rows"],
    )
    assert receipt["source_capture_complete_score"] == 1
    assert receipt["semantic_correctness_consulted"] is False

    censored = measurement.censored_completion(schedule[-1], "generation_deadline")
    incomplete = [*completions[:-1], censored]
    replayed, replay_errors = measurement.independent_replay(schedule, incomplete)
    censored_receipt = measurement.completeness_receipt(
        schedule,
        incomplete,
        replayed,
        replay_errors,
        public["rows"],
        authority["rows"],
    )
    assert censored_receipt["outcomes_accounted"] == 256
    assert censored_receipt["source_capture_complete_score"] == 1


def test_terminal_artifact_and_blocked_preflight_are_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-7278 / SCENARIO-VERIFY-7278-PREFLIGHT."""

    public, authority, schedule, completions = _perfect_capture()
    replay_rows, replay_errors = measurement.independent_replay(schedule, completions)
    reduced = measurement.reduce_measurement(
        public["rows"], authority["rows"], schedule, completions, replay_rows
    )
    artifact = measurement.base_artifact(measurement.RUN_DATE)
    artifact.update(
        {
            "preconditions_checked": [
                measurement.gate_row(
                    "upstreams", True, True, True, upstream="exp7275+exp7277", field="ready"
                )
            ],
            "source_artifact_hashes": {
                "upstreams": {
                    "sha256": "sha256:upstreams",
                    "retired": False,
                    "quarantined": False,
                }
            },
            "model_identity_receipt": _identity(),
            "gpu_receipts": {"provenance_ok": True},
            "runner_receipt": _runner(),
            "raw_call_manifest": {"status": "complete", "raw_call_count": 256},
            "validation_receipts": _passing_receipts(),
            "public_manifest_path": {
                "path": measurement.PUBLIC_MANIFEST_PATH.as_posix(),
                "sha256": "sha256:public",
            },
            "private_authority_manifest_path": {
                "path": measurement.PRIVATE_AUTHORITY_PATH.as_posix(),
                "sha256": "sha256:private",
            },
            "invocation_counts": deepcopy(_runner()["invocation_counts"]),
        }
    )
    terminal = measurement.finalize_measured_artifact(
        artifact,
        schedule,
        completions,
        replay_rows,
        replay_errors,
        reduced,
        duration_s=120.0,
    )
    assert terminal["status"] == "complete"
    assert terminal["source_capture_complete_score"] == 1
    assert terminal["model_invoked"] is True
    assert terminal["inference_substrate"] == "live_llm_inference"
    assert terminal["inference_substrate_class"] == "model_full_generation"
    assert measurement.validate_artifact(terminal) == []

    failure = measurement.gate_row(
        "authenticated_input",
        "sha256:wanted",
        None,
        False,
        upstream="missing.json",
        field="sha256",
    )
    blocked = measurement.finalize_blocked_artifact(
        measurement.base_artifact(measurement.RUN_DATE), [failure], duration_s=0.2
    )
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["invocation_counts"] == measurement.ZERO_INVOCATION_COUNTS
    assert blocked["gate_check_summary"] == {
        "failed_check": "authenticated_input",
        "upstream": "missing.json",
        "field": "sha256",
        "expected_value": "sha256:wanted",
        "observed_value": None,
    }
    assert measurement.validate_artifact(blocked) == []

    checks, _, _ = measurement.authenticate_inputs(tmp_path)
    assert checks[0]["passed"] is False
    assert checks[0]["observed_value"] is None

    load_only = deepcopy(terminal)
    load_only["invocation_counts"] = {
        **measurement.ZERO_INVOCATION_COUNTS,
        "model_loads_attempted": 1,
    }
    load_only = measurement.finalize_measured_artifact(
        load_only,
        schedule,
        completions,
        replay_rows,
        replay_errors,
        reduced,
        duration_s=2.0,
    )
    assert load_only["inference_substrate_class"] == "model_load_only"

    value_reduced = deepcopy(reduced)
    value_reduced["source_value_score"] = 1
    positive = measurement.finalize_measured_artifact(
        deepcopy(terminal),
        schedule,
        completions,
        replay_rows,
        replay_errors,
        value_reduced,
        duration_s=120.0,
    )
    assert positive["verdict_class"] == "circular_positive"
    assert positive["honest_verdict"].startswith("complete_circular_positive_")

    assert measurement.validate_artifact(None) == ["artifact_mapping"]
    missing = deepcopy(terminal)
    missing.pop("rows")
    assert measurement.validate_artifact(missing) == ["missing_required_field:rows"]

    def errors_after(**changes: Any) -> list[str]:
        changed = deepcopy(terminal)
        changed.update(changes)
        changed["reproducibility_checksum"] = measurement.artifact_checksum(changed)
        return measurement.validate_artifact(changed)

    assert "schema" in errors_after(schema="wrong")
    assert "identity" in errors_after(run_date="20260912")
    assert "field_principles" in errors_after(field_principles={})
    assert "execution_identity" in errors_after(MODEL_SPECS=[])
    assert "verifier_is_oracle" in errors_after(verifier_is_oracle=False)
    assert "duration_s" in errors_after(duration_s=True)
    assert "denominators" in errors_after(rows=[])
    assert "live_execution_contract" in errors_after(inference_mode="not_invoked")
    assert "validation_receipts" in errors_after(validation_receipts=[])
    assert "value_verdict" in errors_after(source_value_score=1)
    assert "null_verdict" in errors_after(verdict_class="disqualified")
    assert "acceptance_gate_results" in errors_after(acceptance_gate_results=[])
    assert "raw_call_manifest" in errors_after(raw_call_manifest={})
    assert "status" in errors_after(status="running")
    bad_checksum = deepcopy(terminal)
    bad_checksum["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum" in measurement.validate_artifact(bad_checksum)
    bad_blocked = deepcopy(blocked)
    bad_blocked["model_invoked"] = True
    bad_blocked["reproducibility_checksum"] = measurement.artifact_checksum(bad_blocked)
    assert "blocked_terminal_state" in measurement.validate_artifact(bad_blocked)


def test_defensive_replay_and_raw_file_receipts(tmp_path: Path) -> None:
    """REQ-VERIFY-7278 preserves corruptions, censoring, and raw-file identity."""

    public, authority, schedule, completions = _perfect_capture()
    probe = tmp_path / "probe.json"
    probe.write_text("evidence", encoding="utf-8")
    assert measurement.sha256_file(probe).startswith("sha256:")
    assert measurement._manifest_lists_experiment({"experiment_ids": [7278]}, 7278)
    assert not measurement._manifest_lists_experiment({}, 7278)
    projected = measurement.feasibility_projection({"rows": [{"latency_s": 1.0}]})
    assert projected["projected_generation_s"] == 256.0
    assert projected["projected_within_cap"] is True
    assert measurement.feasibility_projection({"rows": []})["projected_within_cap"] is False

    duplicated = deepcopy(public["rows"])
    duplicated[-1]["unit_id"] = duplicated[0]["unit_id"]
    with pytest.raises(ValueError, match="unique public units"):
        measurement.build_schedule(duplicated)
    assert measurement.schedule_errors(schedule, public["rows"][:-1])[0].startswith(
        "schedule_rebuild:ValueError"
    )
    shortened = deepcopy(schedule[:-1])
    assert "call_denominator" in measurement.schedule_errors(shortened, public["rows"])
    leaked = deepcopy(schedule)
    leaked[0]["expected_decision"] = "supported"
    assert "call_0:authority_leakage" in measurement.schedule_errors(leaked, public["rows"])

    with pytest.raises(ValueError, match="base64_type"):
        measurement._decode_b64(None)
    with pytest.raises(ValueError, match="base64_invalid"):
        measurement._decode_b64("not base64")

    replayed, errors = measurement.independent_replay(schedule, completions[:-1])
    assert len(replayed) == 255
    assert "replay_denominator" in errors
    assert "call_255:missing_outcome" in errors

    censored = measurement.censored_completion(schedule[0], "deadline")
    invalid_censor = deepcopy(censored)
    invalid_censor["censoring_reason"] = ""
    rows, errors = measurement.independent_replay(schedule, [invalid_censor, *completions[1:]])
    assert rows[0]["censored"] is True
    assert "call_0:censoring_record" in errors

    corrupted = deepcopy(completions)
    corrupted[0]["call_id"] = "wrong"
    corrupted[0]["schedule_row_sha256"] = "wrong"
    corrupted[0]["actual_parameters"] = {}
    corrupted[0]["raw_completion"] = "wrong"
    corrupted[0]["compiled_completion"] = None
    corrupted[0]["parse_valid"] = False
    corrupted[0]["usable"] = False
    corrupted[2]["raw_response"] = {}
    corrupted[2]["decision"] = "unknown"
    corrupted[2]["parser_classification"] = "wrong"
    _, errors = measurement.independent_replay(schedule, corrupted)
    assert {
        "call_0:call_id",
        "call_0:schedule_row",
        "call_0:actual_parameters",
        "call_0:raw_completion",
        "call_0:compiled_completion",
        "call_2:raw_response",
        "call_2:decision",
        "call_2:parser_classification",
    } <= set(errors)
    shape_corrupted = deepcopy(completions)
    shape_corrupted[2]["parse_valid"] = False
    shape_corrupted[2]["usable"] = False
    _, shape_errors = measurement.independent_replay(schedule, shape_corrupted)
    assert {"call_2:parse_valid", "call_2:usable"} <= set(shape_errors)
    invalid_bytes = deepcopy(completions)
    invalid_bytes[0]["raw_response_bytes_b64"] = "!"
    replayed, errors = measurement.independent_replay(schedule, invalid_bytes)
    assert len(replayed) == 255
    assert "call_0:base64_invalid" in errors

    assert measurement._decision_from_pointer(public["rows"][0], {}, {}) == {
        "decision": "unknown",
        "errors": ["missing_or_invalid_extraction"],
    }
    assert measurement._direct_decision([]) == ("unknown", "unknown", False)

    raw_dir = tmp_path / "raw"
    seal = measurement.seal_inputs(raw_dir, public, authority, schedule)
    assert seal["public_manifest_path"]["sha256"].startswith("sha256:")
    measurement.seal_inputs(raw_dir, public, authority, schedule)
    manifest = measurement.write_raw_manifest(raw_dir, schedule, completions, _identity())
    assert manifest["status"] == "complete"
    assert manifest["raw_call_count"] == 256
    measurement.write_raw_manifest(raw_dir, schedule, completions, _identity())
    (raw_dir / "public_manifest.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="existing raw evidence differs"):
        measurement.seal_inputs(raw_dir, public, authority, schedule)


def test_cli_date_and_validation_command_scope() -> None:
    """REQ-VERIFY-7278 keeps validation focused and the execution date fixed."""

    assert measurement._date_argument("20260913") == "20260913"
    with pytest.raises(Exception, match="run date must be 20260913"):
        measurement._date_argument("20260912")
    commands = measurement.validation_commands(Path("/repo"), Path("/repo/raw"))
    names = [name for name, _ in commands]
    assert names == list(measurement.REQUIRED_VALIDATION_NAMES)
    rendered = [" ".join(command) for _, command in commands]
    assert all("pytest tests/python -q" not in command for command in rendered)
    assert any("--fail-under=100" in command for command in rendered)
    assert any("scripts/check_spec_coverage.py" in command for command in rendered)
    assert any("scripts/adversarial_verify.py" in command for command in rendered)
    assert any("scripts/verdict_row_consistency_lint.py" in command for command in rendered)
