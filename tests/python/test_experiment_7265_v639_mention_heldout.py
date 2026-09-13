"""Focused tests for the V639 held-out mention capture.

Spec refs: REQ-VERIFY-7265 and SCENARIO-VERIFY-7265-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7237_v637_mention_canary as extraction
from carnot import experiment_7265_v639_mention_heldout as exp


REPO = Path(__file__).resolve().parents[2]
CANARY = REPO / "results/experiment_7264_v639_mention_canary.json"
FIXTURE = REPO / "results/experiment_7236_v637_mention_fixture.json"
PUBLIC = REPO / "results/raw/experiment_7236/public_manifest.json"
AUTHORITY = REPO / "results/raw/experiment_7236/authority_manifest.json"
EXCLUSION = REPO / "ops/exclusion_manifest.yaml"


def _selection() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return exp.load_held_out_manifests(PUBLIC, AUTHORITY)


def _response(sealed: dict[str, object], raw: str) -> dict[str, object]:
    payload, request_bytes = exp.request_payload(sealed)
    body = {
        "choices": [{"finish_reason": "stop", "message": {"content": raw}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 12},
        "timings": {"prompt_ms": 2.0, "predicted_ms": 3.0},
    }
    response_bytes = exp.canonical_json(body).encode()
    return {
        "raw_request": payload,
        "raw_request_bytes_b64": base64.b64encode(request_bytes).decode(),
        "raw_response": body,
        "raw_response_bytes_b64": base64.b64encode(response_bytes).decode(),
        "raw_completion": raw,
        "prompt_tokens": 20,
        "completion_tokens": 12,
        "finish_reason": "stop",
        "latency_s": 0.1,
        "started_at_utc": "2026-09-13T00:00:00Z",
        "completed_at_utc": "2026-09-13T00:00:01Z",
        "error": None,
    }


def _gold_rows(
    schedule: list[dict[str, object]], authority_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    authority = {str(row["unit_id"]): row for row in authority_rows}
    rows = []
    resource = {
        "server_pid": 123,
        "server_pid_start_ticks": 456,
        "gpu_uuid": "GPU-test",
        "lease_id": "lease:test",
        "cuda_offload_confirmed": True,
    }
    for sealed in schedule:
        private = authority[str(sealed["unit_id"])]["variants"][0]
        if sealed["arm"] == "direct_judge":
            completion = {"decision": private["exact_label"]}
        else:
            call_type = str(sealed["call_type"])
            completion = extraction.render_gold_completion(
                sealed["document"],
                private[f"gold_{call_type}_completion"],
                str(sealed["arm"]),
                call_type,
            )
        rows.append(
            exp.build_completion_row(
                sealed,
                _response(sealed, exp.canonical_json(completion)),
                resource,
            )
        )
    return rows


def _receipts(passed: bool = True) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "command": name,
            "exit_code": 0 if passed else 1,
            "passed": passed,
            "timed_out": False,
            "duration_s": 0.1,
            "log_path": f"results/raw/experiment_7265/validation/{name}.log",
            "log_sha256": "sha256:log",
        }
        for name in exp.REQUIRED_VALIDATION_NAMES
    ]


def test_req_verify_7265_contract_and_schema() -> None:
    """REQ-VERIFY-7265 fixes the current model, denominator, and ordinary fields."""

    spec = (REPO / exp.SPEC_PATH).read_text()
    assert "REQ-VERIFY-7265" in spec
    assert exp.MODEL_SPECS == [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]
    assert (exp.PLANNED_UNITS, exp.PLANNED_CALLS, exp.PLANNED_ROWS) == (64, 320, 192)
    artifact = exp.base_artifact(exp.RUN_DATE)
    assert artifact["schema"] == "carnot.exp7265.v639_mention_heldout.v1"
    assert artifact["experiment_id"] == "exp7265-mention-heldout"
    assert artifact["milestone"] == "2026.09.639"
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    with pytest.raises(ValueError, match="held_out_denominator"):
        exp.build_schedule([], [])
    public, authority = _selection()
    wrong_identity = deepcopy(public)
    wrong_identity[0]["unit_id"] = "changed"
    with pytest.raises(ValueError, match="held_out_identity"):
        exp.build_schedule(wrong_identity, authority)
    with pytest.raises(argparse.ArgumentTypeError):
        exp._date_argument("20260912")
    assert exp._date_argument(exp.RUN_DATE) == exp.RUN_DATE


def test_scenario_verify_7265_schedule_is_blind_and_frozen() -> None:
    """SCENARIO-VERIFY-7265-SCHEDULE seals 320 public calls before grading."""

    public, authority = _selection()
    schedule = exp.build_schedule(public, authority)
    assert len(schedule) == 320
    assert [row["call_order"] for row in schedule] == list(range(320))
    assert schedule[0]["seed"] == exp.RANDOM_SEED
    assert {row["arm"] for row in schedule} == set(exp.ARMS)
    assert sum(row["call_type"] == "direct" for row in schedule) == 64
    assert all(not (set(row) & exp.AUTHORITY_ONLY_FIELDS) for row in schedule)
    assert all(
        exp.live_runtime.fixture.grammar_errors(str(row["grammar"])) == [] for row in schedule
    )
    assert exp.schedule_errors(schedule, public, authority) == []
    changed = deepcopy(schedule)
    changed[0]["seed"] = 0
    assert exp.schedule_errors(changed, public, authority) == ["call_0:seed"]
    assert exp.schedule_errors(schedule[:-1], public, authority) == ["schedule_count"]
    receipt = exp.selection_receipt(public, authority, schedule)
    assert receipt["selected_unit_count"] == 64
    assert receipt["authority_fields_in_model_schedule"] == 0
    assert receipt["condition_counts"] == {
        "joint_support": 16,
        "missing_support": 16,
        "reversed": 16,
        "supported": 16,
    }


def test_scenario_verify_7265_preflight_authenticates_clean_canary() -> None:
    """SCENARIO-VERIFY-7265-PREFLIGHT binds clean Exp7264 and fixture bytes."""

    canary = json.loads(CANARY.read_text())
    checks = exp.upstream_gate_rows(
        canary,
        CANARY.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        FIXTURE.read_bytes(),
        exp.load_yaml(EXCLUSION),
    )
    assert checks and all(row["passed"] for row in checks)
    changed = deepcopy(canary)
    changed["mention_canary_ready_score"] = 0
    failed = exp.upstream_gate_rows(
        changed,
        exp.canonical_json(changed).encode(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        FIXTURE.read_bytes(),
        {},
    )
    assert any(row["check"] == "mention_canary_ready" and not row["passed"] for row in failed)


def test_scenario_verify_7265_private_reduction_and_terminal_artifact() -> None:
    """SCENARIO-VERIFY-7265-REDUCE grades all rows after public generation."""

    public, authority = _selection()
    schedule = exp.build_schedule(public, authority)
    raw = _gold_rows(schedule, authority)
    direct = next(row for row in schedule if row["arm"] == "direct_judge")
    neutral = exp.build_completion_row(
        direct,
        _response(direct, '{"decision":"a"}'),
        {
            "server_pid": 123,
            "server_pid_start_ticks": 456,
            "gpu_uuid": "GPU-test",
            "lease_id": "lease:test",
            "cuda_offload_confirmed": True,
        },
    )
    assert neutral["parsed_completion"] == {"decision": "supported"}
    assert neutral["usable"] is True
    assert exp._normalize_direct_row({"arm": "direct_judge"}) == {"arm": "direct_judge"}
    replayed, replay_errors = exp.independent_replay(schedule, raw)
    assert replay_errors == []
    drifted = deepcopy(raw)
    drifted[0]["unexpected"] = True
    assert exp.independent_replay(schedule, drifted)[1] == ["call_0:replay_mismatch"]
    assert exp.independent_replay(schedule, raw[:-1])[1] == ["replay_denominator"]
    rows = exp.score_semantics(schedule, replayed, public, authority)
    source_rows = exp.source_fidelity_rows(rows, raw)
    assert len(rows) == len(source_rows) == 192
    assert {row["condition"] for row in rows} == {
        "supported",
        "reversed",
        "joint_support",
        "missing_support",
    }
    assert all(row["exact_energy"] in {0, 1} for row in source_rows)
    artifact = exp.base_artifact(exp.RUN_DATE)
    assert "hf_id" in exp._identity_errors({}, {}, [])
    artifact["model_identity_receipt"] = {
        "hf_id": exp.MODEL_ID,
        "quantization": exp.QUANTIZATION,
        "revision": "rev",
        "gguf_sha256": "sha256:model",
        "embedded_chat_template_present": True,
        "identity_errors": [],
    }
    artifact["gpu_receipts"] = {"provenance_ok": True}
    artifact = exp.finalize_measured_artifact(
        artifact,
        schedule,
        raw,
        rows,
        duration_s=12.0,
        provenance_errors=[],
        validation_receipts=_receipts(),
    )
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert artifact["mention_capture_complete_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["invocation_counts"] == {
        "model_loads_attempted": 1,
        "model_loads_completed": 1,
        "generation_calls_attempted": 320,
        "generation_calls_completed": 320,
        "usable_answers": 320,
    }
    assert exp.validate_artifact(artifact) == []
    assert exp.validate_artifact(None) == ["artifact_mapping"]
    assert exp.validate_artifact({}) == ["missing_required_field:schema"]

    running = exp.base_artifact(exp.RUN_DATE)
    running["reproducibility_checksum"] = exp.artifact_checksum(running)
    assert "status" in exp.validate_artifact(running)

    malformed_rows = deepcopy(artifact)
    malformed_rows["rows"] = "not-a-list"
    malformed_rows["reproducibility_checksum"] = exp.artifact_checksum(malformed_rows)
    assert "rows" in exp.validate_artifact(malformed_rows)

    broken = deepcopy(artifact)
    broken.update(
        {
            "schema": "wrong",
            "experiment_id": "wrong",
            "run_date": "wrong",
            "field_principles": {},
            "random_seed": 0,
            "verifier_is_oracle": False,
            "duration_s": 1.0,
            "rows": [],
            "source_fidelity_rows": [],
            "token_cost_rows": [],
            "acceptance_gate_results": [],
            "mention_capture_complete_score": 1,
            "verdict_class": "blocked",
            "invocation_counts": {},
            "model_invoked": False,
            "raw_call_manifest": {},
        }
    )
    broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
    errors = exp.validate_artifact(broken)
    assert {
        "schema",
        "identity",
        "run_contract",
        "field_principles",
        "execution_contract",
        "verifier_is_oracle",
        "rows",
        "acceptance_gate_results",
        "mention_capture_complete_score",
        "verdict_class",
        "invocation_counts",
        "live_inference_provenance",
        "bounded_generation_duration_floor",
        "raw_call_manifest",
    } <= set(errors)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "wrong"
    assert "reproducibility_checksum" in exp.validate_artifact(bad_checksum)

    bad_duration = deepcopy(artifact)
    bad_duration["duration_s"] = -1
    bad_duration["reproducibility_checksum"] = exp.artifact_checksum(bad_duration)
    assert "duration_s" in exp.validate_artifact(bad_duration)


def test_scenario_verify_7265_blocked_and_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7265-E2E keeps blocks terminal and the wrapper thin."""

    failure = exp.gate_row(
        "missing_canary",
        True,
        False,
        False,
        upstream="exp7264-mention-canary",
        field="mention_canary_ready_score",
    )
    blocked = exp.finalize_blocked_artifact(exp.base_artifact(exp.RUN_DATE), [failure], 0.1)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "missing_canary"
    assert exp.validate_artifact(blocked) == []
    invalid_blocked = deepcopy(blocked)
    invalid_blocked["verdict_class"] = "null"
    invalid_blocked["mention_capture_complete_score"] = 1
    invalid_blocked["gate_check_summary"] = exp.gate_summary(None)
    invalid_blocked["reproducibility_checksum"] = exp.artifact_checksum(invalid_blocked)
    assert {"blocked_terminal_state", "gate_check_summary"} <= set(
        exp.validate_artifact(invalid_blocked)
    )
    monkeypatch.setattr(exp, "main", lambda: 0)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(REPO / exp.WRAPPER_PATH), run_name="__main__")
    assert raised.value.code == 0


# Import late so the test can prove the CLI rejects an invalid date.
import argparse  # noqa: E402
