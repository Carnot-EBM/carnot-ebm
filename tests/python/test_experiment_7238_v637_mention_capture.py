"""Focused tests for the V637 held-out mention capture.

Spec refs: REQ-VERIFY-7238 and SCENARIO-VERIFY-7238-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7237_v637_mention_canary as canary
from carnot import experiment_7238_v637_mention_capture as exp


REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "results/experiment_7236_v637_mention_fixture.json"
CANARY = REPO / "results/experiment_7237_v637_mention_canary.json"
PUBLIC = REPO / "results/raw/experiment_7236/public_manifest.json"
AUTHORITY = REPO / "results/raw/experiment_7236/authority_manifest.json"
EXCLUSION = REPO / "ops/exclusion_manifest.yaml"


def _selection() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return exp.load_held_out_manifests(PUBLIC, AUTHORITY)


def _response(
    sealed: dict[str, object],
    raw: str,
    *,
    finish_reason: str | None = "stop",
    error: str | None = None,
) -> dict[str, object]:
    payload, request_bytes = exp._request_payload(sealed)
    body = {
        "choices": [{"finish_reason": finish_reason, "message": {"content": raw}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 12},
        "timings": {"prompt_ms": 2.0, "predicted_ms": 3.0},
    }
    response_bytes = exp.canonical_json(body).encode("utf-8") if error is None else b""
    return {
        "raw_request": payload,
        "raw_request_bytes_b64": base64.b64encode(request_bytes).decode("ascii"),
        "raw_response": body if error is None else {},
        "raw_response_bytes_b64": base64.b64encode(response_bytes).decode("ascii"),
        "raw_completion": raw,
        "prompt_tokens": 20 if error is None else 0,
        "completion_tokens": 12 if error is None else 0,
        "finish_reason": finish_reason if error is None else None,
        "latency_s": 0.1,
        "error": error,
    }


def _gold_rows(
    schedule: list[dict[str, object]],
    public_rows: list[dict[str, object]],
    authority_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    authority_by_id = {str(row["unit_id"]): row for row in authority_rows}
    rows = []
    resource = {
        "server_pid": 123,
        "server_pid_start_ticks": 456,
        "gpu_uuid": "GPU-test",
        "lease_id": "lease:test",
        "cuda_offload_confirmed": True,
    }
    for sealed in schedule:
        authority = authority_by_id[str(sealed["unit_id"])]
        private = authority["variants"][0]
        if sealed["arm"] == "direct_judge":
            completion = {"decision": private["exact_label"]}
        else:
            call_type = str(sealed["call_type"])
            completion = canary.render_gold_completion(
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


def _complete_artifact() -> dict[str, object]:
    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _gold_rows(schedule, public_rows, authority_rows)
    paired = exp.score_semantics(schedule, completions, public_rows, authority_rows)
    artifact = exp.finalize_measured_artifact(
        exp.base_artifact(exp.RUN_DATE),
        schedule,
        completions,
        paired,
        duration_s=61.0,
        provenance_errors=[],
    )
    artifact["model_identity_receipt"] = {"identity_errors": []}
    artifact["gpu_receipts"] = {"provenance_ok": True}
    artifact["runner_receipt"].update(
        {
            "runner": "native_llama.cpp_server",
            "completed_transport": True,
            "server_identity": {"pid": 123, "start_time_ticks": 456},
            "cleanup_ok": True,
        }
    )
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    return artifact


def test_req_verify_7238_contract_and_exact_upstreams() -> None:
    """REQ-VERIFY-7238 fixes the full-generation model, budgets, and upstream bytes."""

    spec = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-VERIFY-7238" in spec
    assert exp.MODEL_SPECS == [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]
    assert exp.TOKEN_BUDGETS == {"source": 384, "claim": 128, "direct": 512}
    assert exp.MODEL_LOAD_CAP_S == 240.0
    assert exp.REQUEST_CAP_S == 90.0
    assert exp.INFERENCE_DEADLINE_S == 3000.0
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    assert exp.unwrap_principle({"principle": "why", "value": 1}) == 1
    ordinary = {"principle": "why", "value": 1, "extra": True}
    assert exp.unwrap_principle(ordinary) is ordinary

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    canary_artifact = json.loads(CANARY.read_text(encoding="utf-8"))
    checks = exp.upstream_gate_rows(
        canary_artifact,
        CANARY.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        FIXTURE.read_bytes(),
        exp.load_yaml(EXCLUSION),
    )
    assert checks
    assert next(row for row in checks if row["check"] == "exact_upstream_bytes")["passed"] is True
    quarantine = next(row for row in checks if row["check"] == "structured_quarantine")
    assert quarantine["passed"] is False
    assert quarantine["observed_value"] is True
    quarantined = deepcopy(canary_artifact)
    quarantined["flagged_adversarial"] = {"principle": "why", "value": True}
    bad = exp.upstream_gate_rows(
        quarantined,
        CANARY.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        FIXTURE.read_bytes(),
        {},
    )
    assert next(row for row in bad if row["check"] == "structured_quarantine")["passed"] is False
    fixture["mention_fixture_ready_score"] = 0
    wrong_fixture = exp.canonical_json(fixture).encode("utf-8")
    failed = exp.upstream_gate_rows(
        canary_artifact,
        CANARY.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        wrong_fixture,
        {},
    )
    assert any(row["passed"] is False for row in failed)


def test_scenario_verify_7238_schedule_freezes_320_blind_calls() -> None:
    """SCENARIO-VERIFY-7238-SCHEDULE fixes balanced units and randomized paired arms."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    assert len(public_rows) == len(authority_rows) == 64
    assert len(schedule) == 320
    assert [row["call_order"] for row in schedule] == list(range(320))
    assert {row["arm"] for row in schedule} == set(exp.ARMS)
    assert all(not (set(row) & exp.AUTHORITY_ONLY_FIELDS) for row in schedule)
    assert all(
        row["output_token_budget"] == exp.TOKEN_BUDGETS[row["call_type"]] for row in schedule
    )
    assert exp.schedule_errors(schedule, public_rows, authority_rows) == []
    by_unit: dict[str, list[dict[str, object]]] = {}
    for row in schedule:
        by_unit.setdefault(str(row["unit_id"]), []).append(row)
    paired_orders = []
    for rows in by_unit.values():
        assert len(rows) == 5
        assert sum(row["arm"] == "direct_judge" for row in rows) == 1
        reps = [str(row["arm"]) for row in rows if row["arm"] != "direct_judge"]
        assert set(reps) == {"mention_pointer", "explicit_schema_offset_control"}
        paired_orders.append(tuple(dict.fromkeys(reps)))
    assert len(set(paired_orders)) == 2
    direct = next(row for row in schedule if row["arm"] == "direct_judge")
    assert direct["call_type"] == "direct"
    assert "SOURCE:" in direct["prompt"] and "CLAIM:" in direct["prompt"]
    assert "mention_id" not in direct["prompt"]
    changed = deepcopy(schedule)
    changed[0]["seed"] = 0
    assert "call_0:seed" in exp.schedule_errors(changed, public_rows, authority_rows)


def test_scenario_verify_7238_baseline_is_direct_and_matched() -> None:
    """SCENARIO-VERIFY-7238-BASELINE uses one public direct decision request."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    direct = [row for row in schedule if row["arm"] == "direct_judge"]
    assert len(direct) == 64
    assert all(row["output_token_budget"] == 512 for row in direct)
    assert all(row["model_input"].keys() == {"source", "claim"} for row in direct)
    assert all(
        "score" not in row["prompt"].lower() and "energy" not in row["prompt"].lower()
        for row in direct
    )
    assert {row["call_type"] for row in direct} == {"direct"}


def test_scenario_verify_7238_preserves_bad_terminal_outcomes() -> None:
    """SCENARIO-VERIFY-7238-PRESERVATION keeps malformed, unknown, truncated, and timeout rows."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    sealed_rows = schedule[:4]
    malformed = exp.build_completion_row(sealed_rows[0], _response(sealed_rows[0], "not-json"), {})
    unknown = exp.build_completion_row(
        sealed_rows[1],
        _response(sealed_rows[1], '{"outcome":"unknown","relations":[]}'),
        {},
    )
    truncated = exp.build_completion_row(
        sealed_rows[2],
        _response(
            sealed_rows[2],
            '{"outcome":"unknown","relations":[]}',
            finish_reason="length",
        ),
        {},
    )
    timeout = exp.build_completion_row(
        sealed_rows[3],
        _response(sealed_rows[3], "", error="TimeoutError:request exceeded 90s"),
        {},
    )
    assert malformed["terminal_state"] == "complete" and malformed["parse_valid"] is False
    assert unknown["terminal_state"] == "complete" and unknown["explicit_unknown"] is True
    assert truncated["terminal_state"] == "complete" and truncated["truncated"] is True
    assert timeout["terminal_state"] == "timeout" and timeout["timeout"] is True
    assert all(row["attempted"] is True for row in (malformed, unknown, truncated, timeout))

    completions = _gold_rows(schedule, public_rows, authority_rows)
    completions[:4] = [malformed, unknown, truncated, timeout]
    paired = exp.score_semantics(schedule, completions, public_rows, authority_rows)
    affected = {str(row["unit_id"]) for row in schedule[:4]}
    assert len(paired) == 192
    assert any(row["unit_id"] in affected and row["metric"] == 0 for row in paired)
    receipt = exp.completeness_receipt(schedule, completions, provenance_errors=[])
    assert receipt["attempted_calls"] == 320
    assert receipt["terminal_outcomes"] == 320
    assert receipt["transport_completed_calls"] == 319
    assert receipt["censored_calls"] == 1
    assert receipt["mention_capture_complete_score"] == 1


def test_req_verify_7238_scoring_keeps_192_unit_arm_rows() -> None:
    """REQ-VERIFY-7238 retains every unit and arm with missing-output penalties."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _gold_rows(schedule, public_rows, authority_rows)
    rows = exp.score_semantics(schedule, completions, public_rows, authority_rows)
    assert len(rows) == 192
    assert {row["arm"] for row in rows} == set(exp.ARMS)
    assert all(row["decision_correct"] is True for row in rows)
    assert all(row["fully_correct"] is True for row in rows)
    assert all(row["metric"] == 1 for row in rows)
    assert len(exp.decoding_cost_rows(completions)) == 320

    missing = completions[1:]
    rescored = exp.score_semantics(schedule, missing, public_rows, authority_rows)
    target = next(
        row
        for row in rescored
        if row["unit_id"] == schedule[0]["unit_id"] and row["arm"] == schedule[0]["arm"]
    )
    assert target["fully_correct"] is False
    assert target["missing_output_penalty"] is True


def test_scenario_verify_7238_resume_accepts_only_matching_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7238-RESUME resumes missing calls without regenerating bad rows."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    rows = _gold_rows(schedule[:3], public_rows, authority_rows)
    rows[0] = exp.build_completion_row(schedule[0], _response(schedule[0], "bad"), {})
    identity = exp.checkpoint_identity(
        schedule,
        exp.sha256_file(PUBLIC),
        exp.sha256_file(AUTHORITY),
        "sha256:model",
    )
    path = tmp_path / "resume.json"
    exp.write_resume_checkpoint(path, identity, rows)
    resumed = exp.resume_checkpoint(path, identity, schedule)
    assert [row["call_id"] for row in resumed] == [row["call_id"] for row in rows]
    assert resumed[0]["parse_valid"] is False
    assert len(exp.missing_schedule_rows(schedule, resumed)) == 317
    changed = deepcopy(identity)
    changed["model_sha256"] = "sha256:other"
    with pytest.raises(ValueError, match="checkpoint_identity"):
        exp.resume_checkpoint(path, changed, schedule)


def test_req_verify_7238_artifact_validates_complete_null_and_blocked() -> None:
    """REQ-VERIFY-7238 cold validation separates complete capture from scientific positivity."""

    complete = _complete_artifact()
    assert exp.validate_artifact(complete) == []
    assert complete["status"] == "complete"
    assert complete["verdict_class"] == "null"
    assert complete["mention_capture_complete_score"] == 1
    assert len(complete["paired_unit_rows"]) == 192
    assert len(complete["decoding_cost_rows"]) == 320
    assert complete["inference_substrate_class"] == "model_full_generation"

    changed = deepcopy(complete)
    changed["decoding_cost_rows"][0]["completion_tokens"] += 1
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "decoding_cost_rows" in exp.validate_artifact(changed)

    check = exp.gate_row("model_cache", True, False, False, upstream="cache", field="path")
    blocked = exp.finalize_blocked_artifact(exp.base_artifact(exp.RUN_DATE), [check], 0.2)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate"] == "blocked_no_run"
    assert exp.validate_artifact(blocked) == []


def test_req_verify_7238_capture_manifest_binds_every_call(tmp_path: Path) -> None:
    """REQ-VERIFY-7238 writes the manifest and binds every raw call to a public unit."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _gold_rows(schedule, public_rows, authority_rows)
    manifest = exp.write_capture_manifest(
        tmp_path,
        schedule,
        completions,
        {"gguf_sha256": "sha256:model"},
    )
    assert manifest["schema"] == "carnot.exp7238.capture.v1"
    assert manifest["scheduled_calls"] == manifest["terminal_outcomes"] == 320
    assert len(manifest["rows"]) == 320
    assert all(
        row["unit_id"] in {str(item["unit_id"]) for item in public_rows} for row in manifest["rows"]
    )
    assert (tmp_path / "manifest.json").is_file()


def test_req_verify_7238_entrypoint_is_thin_and_date_is_fixed() -> None:
    """REQ-VERIFY-7238 keeps the runnable wrapper thin and rejects another date."""

    wrapper = REPO / exp.WRAPPER_PATH
    source = wrapper.read_text(encoding="utf-8")
    assert "experiment_7238_v637_mention_capture" in source
    assert "main" in runpy.run_path(str(wrapper), run_name="not_main")
    with pytest.raises(SystemExit):
        exp.main(["--date", "20260911"])


def test_req_verify_7238_defensive_inputs_and_resume_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-7238 rejects malformed manifests, rows, bytes, and resume drift."""

    public_value = json.loads(PUBLIC.read_text(encoding="utf-8"))
    authority_value = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    bad_public = tmp_path / "public.json"
    bad_authority = tmp_path / "authority.json"
    bad_public.write_text(json.dumps({"schema": "bad"}), encoding="utf-8")
    bad_authority.write_text(json.dumps({"schema": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="public_manifest_schema"):
        exp.load_held_out_manifests(bad_public, AUTHORITY)
    with pytest.raises(ValueError, match="authority_manifest_schema"):
        exp.load_held_out_manifests(PUBLIC, bad_authority)
    public_value["rows"] = public_value["rows"][:-1]
    bad_public.write_text(json.dumps(public_value), encoding="utf-8")
    with pytest.raises(ValueError, match="held_out_denominator"):
        exp.load_held_out_manifests(bad_public, AUTHORITY)
    public_value = json.loads(PUBLIC.read_text(encoding="utf-8"))
    held = [row for row in public_value["rows"] if row["split"] == "held_out"]
    held[-1]["unit_id"] = held[0]["unit_id"]
    public_value["rows"] = [
        *[row for row in public_value["rows"] if row["split"] != "held_out"],
        *held,
    ]
    bad_public.write_text(json.dumps(public_value), encoding="utf-8")
    with pytest.raises(ValueError, match="held_out_identity"):
        exp.load_held_out_manifests(bad_public, AUTHORITY)
    authority_value["rows"][-1]["condition_key"] = "supported"
    bad_authority.write_text(json.dumps(authority_value), encoding="utf-8")
    with pytest.raises(ValueError, match="held_out_balance"):
        exp.load_held_out_manifests(PUBLIC, bad_authority)

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    with pytest.raises(ValueError, match="held_out_denominator"):
        exp.build_schedule(public_rows[:-1], authority_rows)
    wrong_split = deepcopy(public_rows)
    wrong_split[0]["split"] = "calibration"
    with pytest.raises(ValueError, match="held_out_split"):
        exp.build_schedule(wrong_split, authority_rows)
    duplicate = deepcopy(public_rows)
    duplicate[-1]["unit_id"] = duplicate[0]["unit_id"]
    with pytest.raises(ValueError, match="held_out_identity"):
        exp.build_schedule(duplicate, authority_rows)
    assert exp.schedule_errors([], public_rows[:-1], authority_rows)[0].startswith(
        "schedule_rebuild:ValueError"
    )
    assert "schedule_count" in exp.schedule_errors(schedule[:-1], public_rows, authority_rows)
    extra = deepcopy(schedule)
    extra[0]["extra"] = True
    assert "call_0:extra_fields" in exp.schedule_errors(extra, public_rows, authority_rows)
    receipt = exp.selection_receipt(public_rows, authority_rows, schedule)
    assert receipt["selected_unit_count"] == 64
    assert receipt["authority_fields_in_model_schedule"] == 0

    canary_artifact = json.loads(CANARY.read_text(encoding="utf-8"))
    malformed_checks = exp.upstream_gate_rows(
        canary_artifact,
        CANARY.read_bytes(),
        b"{",
        b"{",
        b"{",
        {},
    )
    assert any(row["passed"] is False for row in malformed_checks)

    direct = next(row for row in schedule if row["arm"] == "direct_judge")
    malformed = exp.build_completion_row(direct, _response(direct, "bad"), {})
    assert malformed["parse_valid"] is False
    broken_bytes = _response(direct, '{"decision":"supported"}')
    broken_bytes["raw_response_bytes_b64"] = base64.b64encode(b"{").decode("ascii")
    assert exp.build_completion_row(direct, broken_bytes, {})["response_bytes_match"] is False
    broken_request = _response(direct, '{"decision":"supported"}', finish_reason="length")
    broken_request["raw_request_bytes_b64"] = ""
    broken_request["raw_request"] = {**broken_request["raw_request"], "seed": 0}
    broken = exp.build_completion_row(direct, broken_request, {})
    assert {"request_bytes_mismatch", "seed_join_mismatch", "truncated"} <= set(broken["errors"])
    transport = exp.build_completion_row(direct, _response(direct, "", error="OSError:closed"), {})
    assert transport["terminal_state"] == "transport_error"
    assert any(error.startswith("transport_error:") for error in transport["errors"])

    gold = _gold_rows(schedule[:3], public_rows, authority_rows)
    replayed = exp.replay_completion_rows(schedule, gold)
    assert replayed == gold
    identity = exp.checkpoint_identity(
        schedule,
        exp.sha256_file(PUBLIC),
        exp.sha256_file(AUTHORITY),
        "sha256:model",
    )
    assert exp.resume_checkpoint(tmp_path / "absent.json", identity, schedule) == []
    resume = tmp_path / "resume.json"
    exp.write_resume_checkpoint(resume, identity, gold)
    value = json.loads(resume.read_text(encoding="utf-8"))
    value["row_count"] = 0
    resume.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_rows"):
        exp.resume_checkpoint(resume, identity, schedule)
    exp.write_resume_checkpoint(resume, identity, [gold[0], gold[0]])
    with pytest.raises(ValueError, match="checkpoint_call_id"):
        exp.resume_checkpoint(resume, identity, schedule)
    bad_hash = deepcopy(gold[0])
    bad_hash["row_sha256"] = "sha256:bad"
    exp.write_resume_checkpoint(resume, identity, [bad_hash])
    with pytest.raises(ValueError, match="checkpoint_row_hash"):
        exp.resume_checkpoint(resume, identity, schedule)
    bad_join = deepcopy(gold[0])
    bad_join["unit_id"] = "changed"
    bad_join["row_sha256"] = exp._row_hash(bad_join)
    exp.write_resume_checkpoint(resume, identity, [bad_join])
    with pytest.raises(ValueError, match="checkpoint_schedule_join:unit_id"):
        exp.resume_checkpoint(resume, identity, schedule)


def test_req_verify_7238_cold_validation_helpers_and_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-VERIFY-7238 covers timing, identity, token, and artifact failure checks."""

    canary_artifact = json.loads(CANARY.read_text(encoding="utf-8"))
    projection = exp.feasibility_projection(canary_artifact)
    assert projection["measured_calls"] == 48
    assert projection["projected_feasible"] is True
    assert exp.feasibility_projection({})["projected_feasible"] is False

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    rows = _gold_rows(schedule, public_rows, authority_rows)
    valid_identity = {
        "hf_id": exp.QWEN_MODEL_ID,
        "quantization": exp.QUANTIZATION,
        "revision": "revision",
        "gguf_sha256": "sha256:model",
        "embedded_chat_template_present": True,
    }
    assert exp._identity_errors(valid_identity, {"provenance_ok": True}, rows) == []
    bad_resource = deepcopy(rows[:1])
    bad_resource[0].update(
        {
            "server_pid": None,
            "seed_join_valid": False,
            "request_bytes_match": False,
            "cuda_offload_confirmed": False,
        }
    )
    identity_errors = exp._identity_errors({}, {}, bad_resource)
    assert set(identity_errors) == {
        "hf_id",
        "quantization",
        "gguf_revision_or_hash",
        "embedded_chat_template",
        "actual_cuda_execution",
        "per_call_cuda_process_identity",
        "request_seed_join",
    }
    token_receipt = exp._measure_token_budgets(schedule, lambda value: list(value))
    assert token_receipt["all_forms_fit"] is True
    assert token_receipt["direct"]["form_count"] == 64

    complete = _complete_artifact()
    assert exp.validate_artifact([]) == ["artifact_mapping"]
    assert exp.validate_artifact({})[0].startswith("missing_required_field:")

    def errors_after_change(key: str, value: object) -> list[str]:
        changed = deepcopy(complete)
        changed[key] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        return exp.validate_artifact(changed)

    assert "field_principles" in errors_after_change("field_principles", {})
    assert "run_date" in errors_after_change("run_date", "20260911")
    assert "MODEL_SPECS" in errors_after_change("MODEL_SPECS", [])
    assert "random_seed" in errors_after_change("random_seed", 0)
    assert "execution_identity" in errors_after_change("execution_venue", "gpu")
    assert "verifier_is_oracle" in errors_after_change("verifier_is_oracle", False)
    assert "capture_manifest_path" in errors_after_change("capture_manifest_path", "bad")
    assert "frozen_capture_settings" in errors_after_change("frozen_capture_settings", {})
    assert "duration_s" in errors_after_change("duration_s", -1)
    assert "reproducibility_checksum" in exp.validate_artifact(
        {**complete, "reproducibility_checksum": "bad"}
    )
    assert "status" in errors_after_change("status", "running")
    assert "rows" in errors_after_change("schedule", {})
    assert "completeness_receipt" in errors_after_change("completeness_receipt", {})
    assert "acceptance_gate_results" in errors_after_change("acceptance_gate_results", [])
    assert "mention_capture_complete_score" in errors_after_change(
        "mention_capture_complete_score", 0
    )
    assert "paired_unit_rows" in errors_after_change("rows", [])
    assert "sample_size_budget" in errors_after_change("sample_size_budget", {})
    assert "verdict_class" in errors_after_change("verdict_class", "positive")
    assert "live_inference_provenance" in errors_after_change("gpu_receipts", {})
    assert "full_generation_duration_floor" in errors_after_change("duration_s", 59.0)
    assert "raw_request_manifest" in errors_after_change("raw_request_manifest", {})
    no_identity = deepcopy(complete)
    no_identity["model_identity_receipt"] = []
    no_identity["reproducibility_checksum"] = exp.artifact_checksum(no_identity)
    assert "completeness_receipt" in exp.validate_artifact(no_identity)

    check = exp.gate_row("cache", True, False, False, upstream="cache", field="path")
    blocked = exp.finalize_blocked_artifact(exp.base_artifact(exp.RUN_DATE), [check], 0.1)
    for field, value, expected in (
        ("verdict_class", "null", "blocked_terminal_state"),
        ("gate_check_summary", {}, "gate_check_summary"),
        ("inference_substrate", "live_llm_inference", "blocked_substrate"),
    ):
        changed = deepcopy(blocked)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed)

    assert exp._date_argument(exp.RUN_DATE) == exp.RUN_DATE
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda **_kwargs: {"honest_verdict": "complete", "mention_capture_complete_score": 1},
    )
    monkeypatch.setattr(exp, "validate_artifact", lambda _value: [])
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert "terminal verdict=complete" in capsys.readouterr().out
    monkeypatch.setattr(exp, "validate_artifact", lambda _value: ["bad"])
    assert exp.main(["--date", exp.RUN_DATE]) == 1
    assert "invalid artifact" in capsys.readouterr().out

    receipts = [
        {
            "command": "pytest focused",
            "exit_code": 0,
            "classification": "passed",
            "summary": "11 passed",
        }
    ]
    monkeypatch.undo()
    attached = exp.attach_validation_receipts(complete, receipts)
    assert attached["validation_command_rows"] == receipts
    assert exp.validate_artifact(attached) == []
    with pytest.raises(ValueError, match="validation_receipt_source_artifact"):
        exp.attach_validation_receipts({**complete, "status": "bad"}, receipts)
    with pytest.raises(ValueError, match="validation_receipt_schema"):
        exp.attach_validation_receipts(complete, [{"command": "missing fields"}])
