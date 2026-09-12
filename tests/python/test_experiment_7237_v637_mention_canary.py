"""Focused tests for the V637 public-mention canary.

Spec refs: REQ-VERIFY-7237 and SCENARIO-VERIFY-7237-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7236_v637_mention_fixture as fixture
from carnot import experiment_7237_v637_mention_canary as exp


REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "results/experiment_7236_v637_mention_fixture.json"
PUBLIC = REPO / "results/raw/experiment_7236/public_manifest.json"
AUTHORITY = REPO / "results/raw/experiment_7236/authority_manifest.json"
EXCLUSION = REPO / "ops/exclusion_manifest.yaml"


def _selection() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return exp.load_calibration_manifests(PUBLIC, AUTHORITY)


def _response(sealed: dict[str, object], raw: str) -> dict[str, object]:
    payload, request_bytes = exp._request_payload(sealed)
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
        pointer = authority[f"gold_{call_type}_completion"]
        completion = exp.render_gold_completion(
            document,
            pointer,
            str(sealed["arm"]),
            call_type,
        )
        raw = exp.canonical_json(completion)
        rows.append(
            exp.build_completion_row(
                sealed,
                _response(sealed, raw),
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


def test_req_verify_7237_spec_identity_and_principles() -> None:
    """REQ-VERIFY-7237 fixes the current model and required artifact fields."""

    spec = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-VERIFY-7237" in spec
    assert exp.MODEL_SPECS == [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]
    assert exp.MODEL_LOAD_CAP_S == 240.0
    assert exp.REQUEST_CAP_S == 90.0
    assert exp.INFERENCE_DEADLINE_S == 1800.0
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    assert exp.unwrap_principle({"principle": "why", "value": 1}) == 1
    ordinary = {"principle": "why", "value": 1, "extra": True}
    assert exp.unwrap_principle(ordinary) is ordinary


def test_scenario_verify_7237_schedule_freezes_blind_public_calls() -> None:
    """SCENARIO-VERIFY-7237-SCHEDULE fixes eight by three by two public calls."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    assert len(public_rows) == len(authority_rows) == 8
    assert len(schedule) == 48
    assert {row["arm"] for row in schedule} == set(exp.ARMS)
    assert {row["call_type"] for row in schedule} == {"source", "claim"}
    assert [row["call_order"] for row in schedule] == list(range(48))
    assert all(
        row["output_token_budget"] == exp.TOKEN_BUDGETS[row["call_type"]] for row in schedule
    )
    assert all(row["decoding_parameters"]["seed"] == row["seed"] for row in schedule)
    assert all(not (set(row) & exp.AUTHORITY_ONLY_FIELDS) for row in schedule)
    assert exp.schedule_errors(schedule, public_rows, authority_rows) == []
    pointer = next(row for row in schedule if row["arm"] == "mention_pointer")
    original = next(row for row in schedule if row["arm"] == "original_offset")
    assert "mention_id" in pointer["prompt"]
    assert "mention_id" not in original["prompt"]

    changed = deepcopy(schedule)
    changed[0]["seed"] = 0
    assert "call_0:seed" in exp.schedule_errors(changed, public_rows, authority_rows)
    assert "schedule_count" in exp.schedule_errors(schedule[:-1], public_rows, authority_rows)


def test_scenario_verify_7237_schedule_rejects_manifest_drift() -> None:
    """SCENARIO-VERIFY-7237-SCHEDULE rejects private or incomplete calibration inputs."""

    public_rows, authority_rows = _selection()
    with pytest.raises(ValueError, match="calibration_denominator"):
        exp.build_schedule(public_rows[:-1], authority_rows)
    duplicate = deepcopy(public_rows)
    duplicate[-1]["unit_id"] = duplicate[0]["unit_id"]
    with pytest.raises(ValueError, match="calibration_identity"):
        exp.build_schedule(duplicate, authority_rows)
    wrong_split = deepcopy(public_rows)
    wrong_split[0]["split"] = "held_out"
    with pytest.raises(ValueError, match="calibration_split"):
        exp.build_schedule(wrong_split, authority_rows)
    malformed = deepcopy(public_rows)
    malformed[0]["variants"] = []
    with pytest.raises(ValueError, match="public_original_variant"):
        exp.build_schedule(malformed, authority_rows)


def test_req_verify_7237_authenticates_upstream_and_rejects_quarantine() -> None:
    """REQ-VERIFY-7237 binds exact Exp7236 bytes before model invocation."""

    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    checks = exp.upstream_gate_rows(
        upstream,
        UPSTREAM.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        b"",
        exp.load_yaml(EXCLUSION),
    )
    assert checks and all(row["passed"] is True for row in checks)
    assert {row["check"] for row in checks} >= {
        "exact_upstream_bytes",
        "structured_quarantine",
        "exclusion_manifest",
        "producer_gate_fields",
        "upstream_authentication",
        "manifest_authentication",
    }

    quarantined = deepcopy(upstream)
    quarantined["quarantined"] = {"principle": "why", "value": True}
    failed = exp.upstream_gate_rows(
        quarantined,
        UPSTREAM.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        b"",
        {},
    )
    assert next(row for row in failed if row["check"] == "structured_quarantine")["passed"] is False
    tampered = exp.upstream_gate_rows(
        upstream,
        UPSTREAM.read_bytes() + b"\n",
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        b"",
        {},
    )
    assert (
        next(row for row in tampered if row["check"] == "exact_upstream_bytes")["passed"] is False
    )


def test_scenario_verify_7237_usability_separates_syntax_from_semantics() -> None:
    """SCENARIO-VERIFY-7237-USABILITY keeps a valid unknown unusable for fidelity."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    rows = _completion_rows(schedule, public_rows, authority_rows)
    assert len(rows) == 48
    assert all(row["transport_complete"] is True for row in rows)
    assert all(row["parse_valid"] is True for row in rows)
    assert all(row["usable"] is True for row in rows)
    assert all(
        row["request_bytes_match"] is True and row["seed_join_valid"] is True for row in rows
    )

    pointer_call = next(row for row in schedule if row["arm"] == "mention_pointer")
    unknown = exp.build_completion_row(
        pointer_call,
        _response(pointer_call, '{"outcome":"unknown","relations":[]}'),
        {"server_pid": 123, "server_pid_start_ticks": 456, "gpu_uuid": "GPU-test"},
    )
    assert unknown["transport_complete"] is True
    assert unknown["parse_valid"] is True
    assert unknown["explicit_unknown"] is True
    assert unknown["usable"] is False

    malformed = exp.build_completion_row(
        pointer_call,
        _response(pointer_call, "not json"),
        {"server_pid": 123, "server_pid_start_ticks": 456, "gpu_uuid": "GPU-test"},
    )
    assert malformed["parse_valid"] is False
    assert malformed["usable"] is False


def test_scenario_verify_7237_usability_accepts_exact_noncanonical_server_bytes() -> None:
    """SCENARIO-VERIFY-7237-USABILITY parses exact server bytes without rewriting them."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    sealed = next(row for row in schedule if row["arm"] == "mention_pointer")
    authority = authority_rows[0]["variants"][0]
    gold = exp.render_gold_completion(
        sealed["document"],
        authority["gold_source_completion"],
        "mention_pointer",
        "source",
    )
    response = _response(sealed, exp.canonical_json(gold))
    response["raw_response_bytes_b64"] = base64.b64encode(
        json.dumps(response["raw_response"], indent=2).encode("utf-8")
    ).decode("ascii")
    row = exp.build_completion_row(
        sealed,
        response,
        {"server_pid": 123, "server_pid_start_ticks": 456, "gpu_uuid": "GPU-test"},
    )
    assert row["response_bytes_match"] is True
    assert row["transport_complete"] is True
    assert row["usable"] is True

    replayed = exp.replay_completion_rows(schedule, [row])
    assert len(replayed) == 1
    assert replayed[0]["raw_response_bytes_b64"] == row["raw_response_bytes_b64"]
    assert replayed[0]["transport_complete"] is True

    invalid = deepcopy(response)
    invalid["raw_response_bytes_b64"] = base64.b64encode(b"{").decode("ascii")
    rejected = exp.build_completion_row(
        sealed,
        invalid,
        {"server_pid": 123, "server_pid_start_ticks": 456, "gpu_uuid": "GPU-test"},
    )
    assert rejected["response_bytes_match"] is False
    assert rejected["transport_complete"] is False


def test_req_verify_7237_scoring_reports_each_fidelity_dimension() -> None:
    """REQ-VERIFY-7237 scores representation, relations, decisions, and abstention."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule, public_rows, authority_rows)
    semantics = exp.score_semantics(schedule, completions, public_rows, authority_rows)
    assert len(semantics) == 24
    assert {row["arm"] for row in semantics} == set(exp.ARMS)
    assert all(row["offset_valid"] is True for row in semantics)
    assert all(row["source_fidelity"] is True for row in semantics)
    assert all(row["claim_fidelity"] is True for row in semantics)
    assert all(row["decision_correct"] is True for row in semantics)
    assert all(row["fully_correct"] is True for row in semantics)
    assert all(row["metric"] == 1 and row["error"] is None for row in semantics)
    pointer = [row for row in semantics if row["arm"] == "mention_pointer"]
    assert all(row["mention_resolution"] is True for row in pointer)

    changed = deepcopy(completions)
    target = next(row for row in changed if row["arm"] == "mention_pointer")
    target["parsed_completion"] = {"outcome": "unknown", "relations": []}
    target["compiled_completion"] = {
        "outcome": "unknown",
        "relations": [],
        "errors": ["explicit_unknown"],
    }
    target["usable"] = False
    altered = exp.score_semantics(schedule, changed, public_rows, authority_rows)
    unit = next(
        row
        for row in altered
        if row["unit_id"] == target["unit_id"] and row["arm"] == "mention_pointer"
    )
    assert unit["source_fidelity"] is False
    assert unit["fully_correct"] is False
    assert unit["abstention"] is True


def test_req_verify_7237_readiness_uses_pointer_only_thresholds() -> None:
    """REQ-VERIFY-7237 applies seven usable pairs and six semantic decisions."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule, public_rows, authority_rows)
    semantics = exp.score_semantics(schedule, completions, public_rows, authority_rows)
    ready = exp.readiness_receipt(schedule, completions, semantics, provenance_errors=[])
    assert ready["pointer_usable_units"] == 8
    assert ready["pointer_semantic_correct_units"] == 8
    assert ready["negative_control_false_accepts"] == 0
    assert ready["mention_canary_ready_score"] == 1

    one_bad = deepcopy(completions)
    next(row for row in one_bad if row["arm"] == "mention_pointer")["usable"] = False
    assert (
        exp.readiness_receipt(schedule, one_bad, semantics, provenance_errors=[])[
            "mention_canary_ready_score"
        ]
        == 1
    )
    two_bad = deepcopy(one_bad)
    pointer_sources = [
        row for row in two_bad if row["arm"] == "mention_pointer" and row["call_type"] == "source"
    ]
    pointer_sources[1]["usable"] = False
    assert (
        exp.readiness_receipt(schedule, two_bad, semantics, provenance_errors=[])[
            "mention_canary_ready_score"
        ]
        == 0
    )

    six = deepcopy(semantics)
    pointer_semantics = [row for row in six if row["arm"] == "mention_pointer"]
    pointer_semantics[0]["fully_correct"] = False
    pointer_semantics[1]["fully_correct"] = False
    assert (
        exp.readiness_receipt(schedule, completions, six, provenance_errors=[])[
            "mention_canary_ready_score"
        ]
        == 1
    )
    pointer_semantics[2]["fully_correct"] = False
    assert (
        exp.readiness_receipt(schedule, completions, six, provenance_errors=[])[
            "mention_canary_ready_score"
        ]
        == 0
    )
    assert (
        exp.readiness_receipt(schedule, completions, semantics, provenance_errors=["pid"])[
            "mention_canary_ready_score"
        ]
        == 0
    )


def _complete_artifact() -> dict[str, object]:
    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule, public_rows, authority_rows)
    semantics = exp.score_semantics(schedule, completions, public_rows, authority_rows)
    artifact = exp.finalize_measured_artifact(
        exp.base_artifact(exp.RUN_DATE),
        schedule,
        completions,
        semantics,
        duration_s=12.0,
        provenance_errors=[],
    )
    artifact["runner_receipt"] = {
        "runner": "native_llama.cpp_server",
        "completed_transport": True,
        "server_identity": {"pid": 123, "start_time_ticks": 456},
        "cleanup_ok": True,
    }
    artifact["model_identity_receipt"] = {"identity_errors": []}
    artifact["gpu_receipts"] = {"provenance_ok": True}
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    return artifact


def test_req_verify_7237_artifact_validates_ready_null_and_blocked() -> None:
    """REQ-VERIFY-7237 cold validation preserves every terminal class."""

    complete = _complete_artifact()
    assert exp.validate_artifact(complete) == []
    assert complete["status"] == "complete"
    assert complete["verdict_class"] == "circular_positive"
    assert complete["mention_canary_ready_score"] == 1
    assert complete["transport_completed_calls"] == 48
    assert complete["usable_calls"] == 48

    null = deepcopy(complete)
    for row in null["rows"]:
        if row["arm"] == "mention_pointer":
            row["fully_correct"] = False
    null = exp.finalize_measured_artifact(
        exp.base_artifact(exp.RUN_DATE),
        null["schedule"],
        null["raw_rows"],
        null["rows"],
        duration_s=12.0,
        provenance_errors=[],
    )
    null["runner_receipt"] = complete["runner_receipt"]
    null["model_identity_receipt"] = {"identity_errors": []}
    null["gpu_receipts"] = {"provenance_ok": True}
    null["reproducibility_checksum"] = exp.artifact_checksum(null)
    assert null["verdict_class"] == "null"
    assert exp.validate_artifact(null) == []

    check = exp.gate_row("model_cache", True, False, False, upstream="cache", field="path")
    blocked = exp.finalize_blocked_artifact(exp.base_artifact(exp.RUN_DATE), [check], 0.2)
    assert blocked["status"] == "blocked"
    assert blocked["inference_substrate"] == "blocked_no_run"
    assert exp.validate_artifact(blocked) == []

    changed = deepcopy(complete)
    changed["transport_completed_calls"] = 47
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "transport_completed_calls" in exp.validate_artifact(changed)
    assert exp.validate_artifact([]) == ["artifact_mapping"]


def test_req_verify_7237_raw_manifest_binds_requests_and_responses(tmp_path: Path) -> None:
    """REQ-VERIFY-7237 seals exact request bytes and all 48 response hashes."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule, public_rows, authority_rows)
    manifest = exp.write_raw_manifest(
        tmp_path,
        schedule,
        completions,
        {"gguf_sha256": "sha256:model"},
    )
    assert manifest["schema"] == "carnot.exp7237.raw_requests.v1"
    assert manifest["raw_row_count"] == 48
    assert manifest["authority_path_opened_by_model_worker"] is False
    assert all(row["request_bytes_sha256"].startswith("sha256:") for row in manifest["rows"])
    assert (tmp_path / "raw_request_manifest.json").is_file()


def test_req_verify_7237_defensive_parsers_and_receipts(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-7237 names malformed manifests, grammars, bytes, and identities."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    receipt = exp.selection_receipt(public_rows, authority_rows, schedule)
    assert receipt["selected_unit_count"] == 8
    assert receipt["authority_fields_in_model_schedule"] == 0

    public_value = json.loads(PUBLIC.read_text(encoding="utf-8"))
    authority_value = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    bad_public = tmp_path / "bad-public.json"
    bad_authority = tmp_path / "bad-authority.json"
    bad_public.write_text(json.dumps({"schema": "bad"}), encoding="utf-8")
    bad_authority.write_text(json.dumps({"schema": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="public_manifest_schema"):
        exp.load_calibration_manifests(bad_public, AUTHORITY)
    with pytest.raises(ValueError, match="authority_manifest_schema"):
        exp.load_calibration_manifests(PUBLIC, bad_authority)
    public_value["rows"] = [
        row
        for index, row in enumerate(public_value["rows"])
        if not (index == 0 and row["split"] == "calibration")
    ]
    bad_public.write_text(json.dumps(public_value), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration_denominator"):
        exp.load_calibration_manifests(bad_public, AUTHORITY)
    authority_value["rows"][1]["unit_id"] = authority_value["rows"][0]["unit_id"]
    bad_authority.write_text(json.dumps(authority_value), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration_identity"):
        exp.load_calibration_manifests(PUBLIC, bad_authority)

    malformed_document = deepcopy(public_rows[0])
    malformed_document["variants"][0]["source"] = "bad"
    with pytest.raises(ValueError, match="public_document"):
        exp.unit_document(malformed_document, "source")
    document = exp.unit_document(public_rows[0], "source")
    with pytest.raises(ValueError, match="pointer_grammar_input"):
        exp._pointer_grammar(document, "bad")
    no_mentions = deepcopy(document)
    no_mentions["mentions"] = []
    with pytest.raises(ValueError, match="pointer_grammar_mentions"):
        exp._pointer_grammar(no_mentions, "source")
    with pytest.raises(ValueError, match="representation_arm"):
        exp.compile_grammar("bad", document, "source")
    with pytest.raises(ValueError, match="representation_arm"):
        exp._prompt("bad", document, "source")
    assert exp.schedule_errors([], public_rows[:-1], authority_rows)[0].startswith(
        "schedule_rebuild:ValueError"
    )
    extra = deepcopy(schedule)
    extra[0]["private"] = True
    assert "call_0:extra_fields" in exp.schedule_errors(extra, public_rows, authority_rows)

    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    bad_checks = exp.upstream_gate_rows(
        upstream,
        UPSTREAM.read_bytes(),
        b"bad-json",
        b"bad-json",
        b"",
        {},
    )
    assert (
        next(row for row in bad_checks if row["check"] == "manifest_authentication")["passed"]
        is False
    )
    assert exp.render_gold_completion(
        document,
        {"outcome": "unknown", "relations": []},
        "original_offset",
        "source",
    ) == {"outcome": "unknown", "relations": []}

    assert exp._completion_shape_valid([], "mention_pointer", "source") is False
    assert (
        exp._completion_shape_valid(
            {"outcome": "bad", "relations": []}, "mention_pointer", "source"
        )
        is False
    )
    assert (
        exp._completion_shape_valid(
            {"outcome": "known", "relations": []}, "mention_pointer", "source"
        )
        is False
    )
    assert exp._decoded_bytes(None) is None
    assert exp._decoded_bytes("not-base64") is None

    pointer = next(row for row in schedule if row["arm"] == "mention_pointer")
    authority = authority_rows[0]["variants"][0]
    gold = exp.render_gold_completion(
        pointer["document"], authority["gold_source_completion"], "mention_pointer", "source"
    )
    broken_response = _response(pointer, exp.canonical_json(gold))
    broken_response.update(
        {
            "error": "TimeoutError:bounded",
            "raw_request_bytes_b64": "bad",
            "raw_response_bytes_b64": "bad",
            "finish_reason": "length",
        }
    )
    broken_response["raw_request"]["seed"] = 0
    broken = exp.build_completion_row(
        pointer,
        broken_response,
        {"server_pid": 1, "server_pid_start_ticks": 2, "gpu_uuid": "GPU-test"},
    )
    assert broken["terminal_state"] == "transport_error"
    assert set(broken["errors"]) >= {
        "transport_error:TimeoutError:bounded",
        "request_bytes_mismatch",
        "response_bytes_mismatch",
        "seed_join_mismatch",
        "truncated",
    }

    token_receipt = exp.measure_token_budgets(schedule, lambda value: range(len(value) // 4 + 1))
    assert token_receipt["all_forms_fit"] is True
    assert token_receipt["source"]["form_count"] == 24
    identity = {
        "hf_id": exp.QWEN_MODEL_ID,
        "quantization": exp.QUANTIZATION,
        "revision": "rev",
        "gguf_sha256": "sha256:model",
        "embedded_chat_template_present": True,
    }
    completions = _completion_rows(schedule, public_rows, authority_rows)
    gpu = {
        "provenance_ok": True,
        "server_identity": {"pid": 123, "start_time_ticks": 456},
    }
    assert exp._identity_errors(identity, gpu, completions) == []
    assert set(exp._identity_errors({}, {}, broken and [broken])) == {
        "hf_id",
        "quantization",
        "gguf_revision_or_hash",
        "embedded_chat_template",
        "actual_cuda_execution",
        "native_pid_identity",
        "request_seed_join",
    }


def test_req_verify_7237_missing_calls_and_validator_drift() -> None:
    """REQ-VERIFY-7237 retains missing calls and rejects each terminal drift class."""

    public_rows, authority_rows = _selection()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule, public_rows, authority_rows)
    semantics = exp.score_semantics(schedule, completions[2:], public_rows, authority_rows)
    first = [row for row in semantics if row["unit_id"] == public_rows[0]["unit_id"]]
    missing_arm = next(row for row in first if row["arm"] == "original_offset")
    assert missing_arm["metric"] == 0
    assert missing_arm["abstention"] is True
    assert "source_fidelity" in missing_arm["error"]
    assert "claim_fidelity" in missing_arm["error"]
    assert "decision" in missing_arm["error"]

    full_semantics = exp.score_semantics(schedule, completions, public_rows, authority_rows)
    reversed_row = next(
        row
        for row in full_semantics
        if row["arm"] == "mention_pointer" and row["condition"] == "reversed"
    )
    reversed_row["false_accept"] = True
    receipt = exp.readiness_receipt(schedule, completions, full_semantics, provenance_errors=[])
    assert receipt["negative_control_false_accepts"] == 1
    assert receipt["mention_canary_ready_score"] == 0

    complete = _complete_artifact()
    missing = deepcopy(complete)
    missing.pop("rows")
    assert exp.validate_artifact(missing) == ["missing_required_field:rows"]
    mutations = {
        "field_principles": ({}, "field_principles"),
        "run_date": ("bad", "run_date"),
        "MODEL_SPECS": ([], "MODEL_SPECS"),
        "random_seed": (0, "random_seed"),
        "execution_venue": ("bad", "execution_identity"),
        "verifier_is_oracle": (False, "verifier_is_oracle"),
        "duration_s": (-1, "duration_s"),
        "frozen_capture_settings": ({}, "frozen_capture_settings"),
        "readiness_receipt": ({}, "readiness_receipt"),
        "acceptance_gate_results": ([], "acceptance_gate_results"),
        "mention_canary_ready_score": (0, "mention_canary_ready_score"),
        "usable_calls": (47, "usable_calls"),
        "sample_size_budget": ({}, "sample_size_budget"),
        "verdict_class": ("positive", "verdict_class"),
        "model_invoked": (False, "live_inference_provenance"),
        "per_unit_semantics": ([], "per_unit_semantics"),
        "raw_request_manifest": ({}, "raw_request_manifest"),
    }
    for field, (replacement, expected_error) in mutations.items():
        changed = deepcopy(complete)
        changed[field] = replacement
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected_error in exp.validate_artifact(changed), field
    bad_checksum = deepcopy(complete)
    bad_checksum["reproducibility_checksum"] = "bad"
    assert "reproducibility_checksum" in exp.validate_artifact(bad_checksum)
    short = deepcopy(complete)
    short["duration_s"] = 9.0
    short["reproducibility_checksum"] = exp.artifact_checksum(short)
    assert "bounded_generation_duration_floor" in exp.validate_artifact(short)
    bad_rows = deepcopy(complete)
    bad_rows["rows"] = {}
    bad_rows["reproducibility_checksum"] = exp.artifact_checksum(bad_rows)
    assert "rows" in exp.validate_artifact(bad_rows)
    bad_identity = deepcopy(complete)
    bad_identity["model_identity_receipt"] = []
    bad_identity["reproducibility_checksum"] = exp.artifact_checksum(bad_identity)
    assert "live_inference_provenance" in exp.validate_artifact(bad_identity)
    running = deepcopy(complete)
    running["status"] = "running"
    running["reproducibility_checksum"] = exp.artifact_checksum(running)
    assert "status" in exp.validate_artifact(running)

    blocked = exp.finalize_blocked_artifact(
        exp.base_artifact(exp.RUN_DATE),
        [exp.gate_row("x", 1, 0, False, upstream="x", field="x")],
        0.1,
    )
    blocked["verdict_class"] = "positive"
    blocked["gate_check_summary"] = {}
    blocked["inference_substrate"] = "bad"
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    assert {"blocked_terminal_state", "gate_check_summary", "blocked_substrate"} <= set(
        exp.validate_artifact(blocked)
    )


def test_req_verify_7237_date_entrypoint_and_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-7237 keeps one thin entrypoint and fixed execution date."""

    assert exp._date_argument("20260912") == "20260912"
    with pytest.raises(Exception, match="run date must be 20260912"):
        exp._date_argument("20260911")

    complete = _complete_artifact()
    monkeypatch.setattr(exp, "run_experiment", lambda **kwargs: complete)
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["bad"])
    assert exp.main(["--date", exp.RUN_DATE]) == 1

    called: list[object] = []
    monkeypatch.setattr(exp, "main", lambda argv=None: called.append(argv) or 0)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(
            str(REPO / "scripts/experiments/experiment_7237_v637_mention_canary.py"),
            run_name="__main__",
        )
    assert raised.value.code == 0
    assert called == [None]
