"""Focused checks for the corrected V639 mention canary.

Spec refs: REQ-VERIFY-7264 and SCENARIO-VERIFY-7264-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7237_v637_mention_canary as prior
from carnot import experiment_7264_v639_mention_canary as exp


REPO = Path(__file__).resolve().parents[2]
COMPUTE = REPO / "results/experiment_7261_v639_compute_contract.json"
FIXTURE = REPO / "results/experiment_7236_v637_mention_fixture.json"
PUBLIC = REPO / "results/raw/experiment_7236/public_manifest.json"
AUTHORITY = REPO / "results/raw/experiment_7236/authority_manifest.json"
EXCLUSION = REPO / "ops/exclusion_manifest.yaml"


def _selection() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return exp.load_calibration_manifests(PUBLIC, AUTHORITY)


def _response(sealed: dict[str, object], raw: str) -> dict[str, object]:
    payload, request_bytes = exp.request_payload(sealed)
    body = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": raw, "reasoning_content": ""},
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 12},
        "timings": {"prompt_ms": 2.0, "predicted_ms": 3.0},
    }
    response_bytes = exp.canonical_json(body).encode("utf-8")
    return {
        "raw_request": payload,
        "raw_request_bytes_b64": base64.b64encode(request_bytes).decode("ascii"),
        "raw_response": body,
        "raw_response_bytes_b64": base64.b64encode(response_bytes).decode("ascii"),
        "raw_completion": raw,
        "prompt_tokens": 20,
        "completion_tokens": 12,
        "finish_reason": "stop",
        "latency_s": 0.1,
        "started_at_utc": "2026-09-13T00:00:00Z",
        "completed_at_utc": "2026-09-13T00:00:00Z",
        "error": None,
    }


def _completion_rows(
    schedule: list[dict[str, object]],
    public_rows: list[dict[str, object]],
    authority_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    public_by_id = {str(row["unit_id"]): row for row in public_rows}
    authority_by_id = {str(row["unit_id"]): row for row in authority_rows}
    rows = []
    for sealed in schedule:
        unit_id = str(sealed["unit_id"])
        call_type = str(sealed["call_type"])
        document = exp.unit_document(public_by_id[unit_id], call_type)
        authority = authority_by_id[unit_id]["variants"][0]
        completion = exp.render_gold_completion(
            document,
            authority[f"gold_{call_type}_completion"],
            str(sealed["arm"]),
            call_type,
        )
        rows.append(
            exp.build_completion_row(
                sealed,
                _response(sealed, exp.canonical_json(completion)),
                {
                    "server_pid": 123,
                    "server_pid_start_ticks": 456,
                    "gpu_uuid": "GPU-test",
                    "lease_id": "lease:test",
                    "cuda_offload_confirmed": True,
                },
            )
        )
    return rows


def _measured_artifact(*, validation_passed: bool = True) -> dict[str, object]:
    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule, public_rows, authority_rows)
    semantics = exp.score_semantics(schedule, completions, public_rows, authority_rows)
    artifact = exp.base_artifact(exp.RUN_DATE)
    artifact["model_identity_receipt"] = {
        "hf_id": exp.MODEL_ID,
        "quantization": exp.QUANTIZATION,
        "revision": "rev",
        "gguf_sha256": "sha256:model",
        "embedded_chat_template_present": True,
        "identity_errors": [],
    }
    artifact["gpu_receipts"] = {
        "provenance_ok": True,
        "server_identity": {"pid": 123, "start_time_ticks": 456},
        "cleanup": {"leak_free": True},
        "lease_release": {"released": True},
    }
    artifact["runner_receipt"] = {
        "runner": "native_llama.cpp_server",
        "model_loaded": True,
        "non_thinking_enabled": True,
        "cleanup_ok": True,
    }
    receipts = [
        {
            "name": name,
            "command": name,
            "exit_code": 0 if validation_passed else 1,
            "passed": validation_passed,
            "timed_out": False,
            "duration_s": 0.1,
            "log_path": f"results/raw/experiment_7264/validation/{name}.log",
            "log_sha256": "sha256:log",
        }
        for name in exp.REQUIRED_VALIDATION_NAMES
    ]
    return exp.finalize_measured_artifact(
        artifact,
        schedule,
        completions,
        semantics,
        duration_s=12.0,
        validation_receipts=receipts,
    )


def test_req_verify_7264_spec_and_required_schema() -> None:
    """REQ-VERIFY-7264 keeps ordinary identity fields and all field principles."""

    spec = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-VERIFY-7264" in spec
    assert exp.MODEL_SPECS == [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    artifact = exp.base_artifact(exp.RUN_DATE)
    assert artifact["schema"] == "carnot.exp7264.v639_mention_canary.v1"
    assert artifact["experiment_id"] == "exp7264-mention-canary"
    assert artifact["milestone"] == "2026.09.639"
    assert artifact["status"] == "running"
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate"] == "blocked_no_run"


def test_scenario_verify_7264_schedule_reuses_original_scientific_gate() -> None:
    """SCENARIO-VERIFY-7264-SCHEDULE preserves the fixed public work and seeds."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    assert schedule == prior.build_schedule(public_rows, authority_rows)
    assert len(schedule) == 48
    assert {row["arm"] for row in schedule} == set(exp.ARMS)
    assert {row["call_type"] for row in schedule} == {"source", "claim"}
    assert all(
        row["output_token_budget"] == exp.TOKEN_BUDGETS[row["call_type"]] for row in schedule
    )
    assert exp.schedule_errors(schedule, public_rows, authority_rows) == []
    assert all(not (set(row) & exp.AUTHORITY_ONLY_FIELDS) for row in schedule)


def test_scenario_verify_7264_preflight_authenticates_both_upstreams() -> None:
    """SCENARIO-VERIFY-7264-PREFLIGHT binds Exp7261 and exact Exp7236 bytes."""

    checks = exp.upstream_gate_rows(
        json.loads(COMPUTE.read_text(encoding="utf-8")),
        COMPUTE.read_bytes(),
        json.loads(FIXTURE.read_text(encoding="utf-8")),
        FIXTURE.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        exp.load_yaml(EXCLUSION),
    )
    assert checks and all(row["passed"] is True for row in checks)
    assert {row["check"] for row in checks} >= {
        "compute_contract_exact_bytes",
        "compute_contract_ready",
        "compute_contract_quarantine",
        "exact_upstream_bytes",
        "manifest_authentication",
    }

    missing_ready = json.loads(COMPUTE.read_text(encoding="utf-8"))
    missing_ready["compute_contract_ready_score"] = 0
    failed = exp.upstream_gate_rows(
        missing_ready,
        COMPUTE.read_bytes(),
        json.loads(FIXTURE.read_text(encoding="utf-8")),
        FIXTURE.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        {},
    )
    row = next(item for item in failed if item["check"] == "compute_contract_ready")
    assert row["passed"] is False
    assert row["field"] == "compute_contract_ready_score"


def test_scenario_verify_7264_gate_separates_syntax_usability_and_semantics() -> None:
    """SCENARIO-VERIFY-7264-GATE applies the unchanged pointer thresholds."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule, public_rows, authority_rows)
    semantics = exp.score_semantics(schedule, completions, public_rows, authority_rows)
    receipt = exp.readiness_receipt(schedule, completions, semantics, provenance_errors=[])
    assert receipt["pointer_complete_parse_units"] == 8
    assert receipt["pointer_usable_units"] == 8
    assert receipt["pointer_semantic_correct_units"] == 8
    assert receipt["negative_control_false_accepts"] == 0
    assert receipt["mention_canary_ready_score"] == 1

    changed = deepcopy(completions)
    pointer = next(row for row in changed if row["arm"] == "mention_pointer")
    pointer["usable"] = False
    changed_receipt = exp.readiness_receipt(
        schedule,
        changed,
        semantics,
        provenance_errors=["server_pid"],
    )
    assert changed_receipt["pointer_complete_parse_units"] == 8
    assert changed_receipt["pointer_usable_units"] == 7
    assert changed_receipt["mention_canary_ready_score"] == 0


def test_req_verify_7264_frozen_heldout_settings_bind_exp7265() -> None:
    """REQ-VERIFY-7264 freezes the later arm budgets before held-out scoring."""

    settings = exp.frozen_heldout_settings()
    assert settings["arms"]["mention_pointer"]["call_pattern"] == [384, 128]
    assert settings["arms"]["explicit_schema_offset_control"]["call_pattern"] == [384, 128]
    assert settings["arms"]["direct_judge"]["call_pattern"] == [512]
    assert settings["retry_malformed"] is False
    assert settings["abstention_rule"] == "unknown_or_unusable_is_not_faithful_evidence"
    assert settings["frozen_before_evaluation"] is True


def test_scenario_verify_7264_artifact_handles_ready_null_and_blocked() -> None:
    """SCENARIO-VERIFY-7264-ARTIFACT keeps each terminal class reproducible."""

    complete = _measured_artifact()
    assert complete["status"] == "complete"
    assert complete["verdict_class"] == "circular_positive"
    assert complete["mention_canary_ready_score"] == 1
    assert complete["inference_substrate"] == "live_llm_inference_local_gguf_sota"
    assert complete["invocation_counts"] == {
        "model_loads_attempted": 1,
        "model_loads_completed": 1,
        "generation_calls_attempted": 48,
        "generation_calls_completed": 48,
        "usable_answers": 48,
    }
    assert exp.validate_artifact(complete) == []

    null = _measured_artifact(validation_passed=False)
    assert null["status"] == "complete"
    assert null["verdict_class"] == "null"
    assert null["mention_canary_ready_score"] == 0
    assert exp.validate_artifact(null) == []

    check = exp.gate_row(
        "compute_contract_ready",
        1,
        "missing_artifact",
        False,
        upstream="exp7261-compute-contract",
        field="compute_contract_ready_score",
    )
    blocked = exp.finalize_blocked_artifact(exp.base_artifact(exp.RUN_DATE), [check], 0.1)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["upstream"] == "exp7261-compute-contract"
    assert blocked["gate_check_summary"]["field"] == "compute_contract_ready_score"
    assert exp.validate_artifact(blocked) == []


def test_req_verify_7264_validator_rejects_drift() -> None:
    """REQ-VERIFY-7264 rejects changed counts, classes, rows, and frozen settings."""

    complete = _measured_artifact()
    mutations = {
        "experiment_id": ("wrong", "identity"),
        "run_date": ("wrong", "run_date"),
        "field_principles": ({}, "field_principles"),
        "MODEL_SPECS": ([], "MODEL_SPECS"),
        "random_seed": (0, "random_seed"),
        "execution_venue": ("wrong", "execution_venue"),
        "model_invoked": (False, "live_inference_provenance"),
        "invocation_counts": ({}, "invocation_counts"),
        "inference_substrate": ("live_llm_inference", "live_inference_provenance"),
        "duration_s": (9.9, "bounded_generation_duration_floor"),
        "rows": ([], "rows"),
        "usable_unit_counts": ({}, "usable_unit_counts"),
        "frozen_heldout_settings": ({}, "frozen_heldout_settings"),
        "validation_receipts": ([], "validation_receipts"),
        "raw_call_manifest": ({}, "raw_call_manifest"),
        "verdict_class": ("positive", "verdict_class"),
        "verifier_is_oracle": (False, "verifier_is_oracle"),
    }
    for field, (replacement, expected_error) in mutations.items():
        changed = deepcopy(complete)
        changed[field] = replacement
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected_error in exp.validate_artifact(changed), field

    assert exp.validate_artifact([]) == ["artifact_mapping"]
    missing = deepcopy(complete)
    missing.pop("schema")
    assert exp.validate_artifact(missing) == ["missing_required_field:schema"]
    bad_checksum = deepcopy(complete)
    bad_checksum["reproducibility_checksum"] = "bad"
    assert "reproducibility_checksum" in exp.validate_artifact(bad_checksum)
    bad_duration = deepcopy(complete)
    bad_duration["duration_s"] = -1
    bad_duration["reproducibility_checksum"] = exp.artifact_checksum(bad_duration)
    assert "duration_s" in exp.validate_artifact(bad_duration)
    running = deepcopy(complete)
    running["status"] = "running"
    running["reproducibility_checksum"] = exp.artifact_checksum(running)
    assert "status" in exp.validate_artifact(running)
    bad_rows = deepcopy(complete)
    bad_rows["rows"] = {}
    bad_rows["reproducibility_checksum"] = exp.artifact_checksum(bad_rows)
    assert "rows" in exp.validate_artifact(bad_rows)
    replay_drift = deepcopy(complete)
    replay_drift["raw_rows"][0]["unexpected"] = True
    replay_drift["reproducibility_checksum"] = exp.artifact_checksum(replay_drift)
    assert "raw_replay" in exp.validate_artifact(replay_drift)

    blocked = exp.finalize_blocked_artifact(
        exp.base_artifact(exp.RUN_DATE),
        [exp.gate_row("x", True, False, False, upstream="x", field="x")],
        0.1,
    )
    blocked["verdict_class"] = "positive"
    blocked["gate_check_summary"] = {}
    blocked["inference_substrate"] = "wrong"
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    assert {"blocked_terminal_state", "gate_check_summary", "blocked_substrate"} <= set(
        exp.validate_artifact(blocked)
    )

    load_only = exp.base_artifact(exp.RUN_DATE)
    load_only["invocation_counts"] = {
        **exp.ZERO_INVOCATION_COUNTS,
        "model_loads_attempted": 1,
        "model_loads_completed": 1,
    }
    load_only = exp.finalize_blocked_artifact(
        load_only,
        [exp.gate_row("generation", 1, 0, False, upstream="runtime", field="calls")],
        2.0,
    )
    assert load_only["inference_substrate"] == "model_load_no_generation"

    attempted = exp.base_artifact(exp.RUN_DATE)
    attempted["invocation_counts"] = {
        **exp.ZERO_INVOCATION_COUNTS,
        "model_loads_attempted": 1,
        "model_loads_completed": 1,
        "generation_calls_attempted": 1,
    }
    attempted = exp.finalize_blocked_artifact(
        attempted,
        [exp.gate_row("generation", 48, 1, False, upstream="runtime", field="calls")],
        10.0,
    )
    assert attempted["inference_substrate"] == "live_llm_inference_local_gguf_sota"
    assert attempted["inference_substrate_class"] == "model_bounded_generation"


def test_req_verify_7264_raw_manifest_replays_exact_bytes(tmp_path: Path) -> None:
    """REQ-VERIFY-7264 preserves request, response, token, and timestamp evidence."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule, public_rows, authority_rows)
    manifest = exp.write_raw_manifest(
        tmp_path,
        schedule,
        completions,
        {"gguf_sha256": "sha256:model"},
    )
    assert manifest["schema"] == "carnot.exp7264.raw_calls.v1"
    assert manifest["raw_row_count"] == 48
    assert all(row["actual_parameters"] for row in manifest["rows"])
    assert all(row["raw_response_bytes_b64"] for row in manifest["rows"])
    assert all(row["request_started_at_utc"] for row in manifest["rows"])
    assert all(row["response_observed_at_utc"] for row in manifest["rows"])
    replayed, errors = exp.independent_replay(schedule, completions)
    assert errors == []
    assert replayed == completions

    malformed = deepcopy(completions)
    malformed[0].pop("raw_response_bytes_b64")
    replayed, errors = exp.independent_replay(schedule, malformed)
    assert replayed == []
    assert errors[0].startswith("replay_error:")
    drift = deepcopy(completions)
    drift[0]["unexpected"] = True
    _, errors = exp.independent_replay(schedule, drift)
    assert "call_0:replay_mismatch" in errors
    _, errors = exp.independent_replay(schedule, completions[:-1])
    assert "replay_denominator" in errors


def test_req_verify_7264_date_and_thin_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-7264 keeps the fixed date and a package-only entrypoint."""

    assert exp._date_argument("20260913") == "20260913"
    with pytest.raises(Exception, match="run date must be 20260913"):
        exp._date_argument("20260912")

    complete = _measured_artifact()
    monkeypatch.setattr(exp, "run_experiment", lambda **_kwargs: complete)
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["bad"])
    assert exp.main(["--date", exp.RUN_DATE]) == 1

    monkeypatch.setattr(exp, "find_repo_root", lambda **_kwargs: REPO)
    monkeypatch.setattr(exp, "independent_replay_from_raw", lambda *_args: [])
    assert exp.main(["--date", exp.RUN_DATE, "--replay-raw", "/tmp/raw"]) == 0
    monkeypatch.setattr(exp, "independent_replay_from_raw", lambda *_args: ["bad"])
    assert exp.main(["--date", exp.RUN_DATE, "--replay-raw", "/tmp/raw"]) == 1

    called: list[object] = []
    monkeypatch.setattr(exp, "main", lambda argv=None: called.append(argv) or 0)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(
            str(REPO / "scripts/experiments/experiment_7264_v639_mention_canary.py"),
            run_name="__main__",
        )
    assert raised.value.code == 0
    assert called == [None]
