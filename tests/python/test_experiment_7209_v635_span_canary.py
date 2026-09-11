"""Tests for REQ-VERIFY-7209 and SCENARIO-VERIFY-7209-*.

The tests use sealed fixture bytes and temporary output paths. They do not
load a model, acquire a GPU, or write the checked-in result.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7208_v635_span_fixture as fixture
from carnot import experiment_7209_v635_span_canary as exp


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
UPSTREAM = REPO / "results/experiment_7208_v635_span_fixture.json"
PUBLIC = REPO / "results/fixtures/experiment_7208/public.jsonl"
AUTHORITY = REPO / "results/fixtures/experiment_7208/authority.jsonl"
MANIFEST = REPO / "results/fixtures/experiment_7208/manifest.json"
EXCLUSION = REPO / "ops/exclusion_manifest.yaml"


@pytest.fixture(scope="module")
def sealed() -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Load the authenticated fixture once for all pure canary tests."""

    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    public_rows, authority_rows = exp.load_canary_split(PUBLIC, AUTHORITY)
    return upstream, public_rows, authority_rows


def _response(raw: str, *, finish_reason: str = "stop", reasoning: str = "") -> dict[str, object]:
    """Make one native response receipt without claiming a live request."""

    body = {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": raw, "reasoning_content": reasoning or None},
            }
        ],
        "usage": {"prompt_tokens": 25, "completion_tokens": 12},
    }
    encoded = json.dumps(body, sort_keys=True).encode("utf-8")
    return {
        "raw_request": {"messages": [], "max_tokens": 384},
        "raw_request_bytes_b64": base64.b64encode(b"{}").decode("ascii"),
        "raw_response": body,
        "raw_response_bytes_b64": base64.b64encode(encoded).decode("ascii"),
        "raw_completion": raw,
        "prompt_tokens": 25,
        "completion_tokens": 12,
        "finish_reason": finish_reason,
        "latency_s": 0.2,
        "error": None,
    }


def _resource() -> dict[str, object]:
    """Bind synthetic rows to a stable fictional task owner."""

    return {
        "server_pid": 101,
        "server_pid_start_ticks": 202,
        "gpu_uuid": "GPU-test",
        "lease_id": "lease:test",
        "cuda_offload_confirmed": True,
        "gpu_sample_sha256": "sha256:sample",
    }


def _completion_rows(schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build a complete bank from the fixture's exact public extractor."""

    rows = []
    for sealed_call in schedule:
        text = str(sealed_call["input_text"]).encode("utf-8")
        completion = fixture.extract_public_completion(text, str(sealed_call["call_type"]))
        raw = fixture.canonical_json(completion)
        rows.append(exp.build_completion_row(sealed_call, _response(raw), _resource()))
    return rows


def test_req_verify_7209_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7209 declares each required scenario and artifact field."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7209") :]
    for scenario in (
        "SCHEDULE",
        "BUDGET",
        "PREFLIGHT",
        "CAPTURE",
        "REASONING",
        "EXECUTION",
        "READINESS",
        "ARTIFACT",
    ):
        assert f"SCENARIO-VERIFY-7209-{scenario}" in section
    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_verify_7209_schedule_is_fixed_bounded_and_blind(
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-SCHEDULE makes exactly 32 isolated calls."""

    _, public_rows, authority_rows = sealed
    schedule = exp.build_schedule(public_rows, authority_rows)

    assert len(public_rows) == len(authority_rows) == 8
    assert len(schedule) == 32
    assert [row["call_order"] for row in schedule] == list(range(32))
    assert {row["arm"] for row in schedule} == {"grammar_only", "reference"}
    assert {row["call_type"] for row in schedule} == {"source", "claim"}
    assert len({row["unit_id"] for row in schedule}) == 8
    assert all(row["seed"] == 7_209_001 for row in schedule)
    assert all(row["cold_request"] is True for row in schedule)
    assert {row["output_token_budget"] for row in schedule if row["call_type"] == "source"} == {384}
    assert {row["output_token_budget"] for row in schedule if row["call_type"] == "claim"} == {128}
    serialized = json.dumps(schedule, sort_keys=True)
    for forbidden in (
        "expected_decision",
        "relation_family",
        '"split"',
        '"variant"',
        "development",
        '"test"',
    ):
        assert forbidden not in serialized
    assert all(set(row["model_input"]) == {f"{row['call_type']}_text"} for row in schedule)
    assert exp.schedule_errors(schedule, public_rows, authority_rows) == []


def test_scenario_verify_7209_schedule_rejects_contract_drift(
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-SCHEDULE checks all frozen request properties."""

    _, public_rows, authority_rows = sealed
    schedule = exp.build_schedule(public_rows, authority_rows)
    mutations = {
        "call_order": 9,
        "arm": "changed",
        "call_type": "changed",
        "input_text": "changed",
        "input_sha256": "sha256:changed",
        "prompt": "changed",
        "prompt_sha256": "sha256:changed",
        "grammar": "changed",
        "grammar_sha256": "sha256:changed",
        "output_token_budget": 1,
        "decoding_parameters": {},
        "cold_request": False,
    }
    for field, value in mutations.items():
        changed = deepcopy(schedule)
        changed[0][field] = value
        assert exp.schedule_errors(changed, public_rows, authority_rows), field

    assert "schedule_count" in exp.schedule_errors(schedule[:-1], public_rows, authority_rows)
    extra = deepcopy(schedule)
    extra[0]["private_label"] = "forbidden"
    assert "call_0:extra_fields" in exp.schedule_errors(extra, public_rows, authority_rows)
    assert exp.schedule_errors([], public_rows[:-1], authority_rows)[0].startswith(
        "schedule_rebuild:ValueError"
    )


def test_scenario_verify_7209_schedule_rejects_bad_fixture_inputs(
    tmp_path: Path,
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-SCHEDULE rejects wrong splits, IDs, and JSONL values."""

    _, public_rows, authority_rows = sealed
    with pytest.raises(ValueError, match="canary_supported_denominator"):
        exp.build_schedule(public_rows[:-1], authority_rows)
    wrong_split = deepcopy(authority_rows)
    wrong_split[0]["split"] = "development"
    with pytest.raises(ValueError, match="non_canary_authority"):
        exp.build_schedule(public_rows, wrong_split)
    wrong_id = deepcopy(public_rows)
    wrong_id[0]["unit_id"] = "changed"
    with pytest.raises(ValueError, match="public_authority_identity"):
        exp.build_schedule(wrong_id, authority_rows)

    public_path = tmp_path / "public.jsonl"
    authority_path = tmp_path / "authority.jsonl"
    public_path.write_text("[]\n", encoding="utf-8")
    authority_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_row_not_object"):
        exp.load_canary_split(public_path, authority_path)
    public_path.write_text("{}\n", encoding="utf-8")
    authority_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canary_supported_denominator"):
        exp.load_canary_split(public_path, authority_path)
    public_path.write_text('{"unit_id":"x"}\n', encoding="utf-8")
    authority_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canary_supported_denominator"):
        exp.load_canary_split(public_path, authority_path)


def test_scenario_verify_7209_budget_uses_actual_tokenizer_and_headroom(
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-BUDGET measures each unique serialized form."""

    _, public_rows, authority_rows = sealed
    schedule = exp.build_schedule(public_rows, authority_rows)

    receipt = exp.measure_token_budgets(schedule, lambda value: range((len(value) + 3) // 4))

    assert receipt["measurement_status"] == "measured_embedded_gguf_tokenizer"
    assert receipt["source"]["measured_form_count"] == 8
    assert receipt["claim"]["measured_form_count"] == 8
    assert receipt["source"]["maximum_serialized_tokens"] > 0
    assert receipt["claim"]["maximum_serialized_tokens"] > 0
    assert receipt["all_forms_fit_with_20_percent_headroom"] is True
    assert receipt["source"]["budget"] == 384
    assert receipt["claim"]["budget"] == 128

    too_large = exp.measure_token_budgets(schedule, lambda value: range(len(value) * 20))
    assert too_large["all_forms_fit_with_20_percent_headroom"] is False
    assert too_large["budget_increased_after_failure"] is False


def test_scenario_verify_7209_preflight_authenticates_upstream_independently() -> None:
    """SCENARIO-VERIFY-7209-PREFLIGHT checks fields, bytes, and quarantine separately."""

    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    checks = exp.upstream_gate_rows(
        upstream,
        UPSTREAM.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        MANIFEST.read_bytes(),
        exp.load_yaml(EXCLUSION),
    )

    assert checks
    assert {row["check"] for row in checks} >= {
        "exact_upstream_bytes",
        "structured_quarantine",
        "exclusion_manifest",
        "producer_gate_fields",
        "upstream_authentication",
        "sidecar_authentication",
        "known_failed_value_preserved",
    }
    by_check = {row["check"]: row for row in checks}
    assert by_check["exact_upstream_bytes"]["passed"] is True
    assert by_check["structured_quarantine"]["passed"] is False
    assert by_check["upstream_authentication"]["passed"] is False
    assert by_check["sidecar_authentication"]["passed"] is True
    assert by_check["known_failed_value_preserved"]["passed"] is True

    quarantined = deepcopy(upstream)
    quarantined["quarantined"] = {"principle": "independent", "value": True}
    failed = exp.upstream_gate_rows(
        quarantined,
        UPSTREAM.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        MANIFEST.read_bytes(),
        exp.load_yaml(EXCLUSION),
    )
    assert next(row for row in failed if row["check"] == "structured_quarantine")["passed"] is False


def test_scenario_verify_7209_preflight_unwraps_only_real_principle_values() -> None:
    """SCENARIO-VERIFY-7209-PREFLIGHT never unwraps an arbitrary dictionary."""

    assert exp.unwrap_principle({"principle": "why", "value": 1}) == 1
    arbitrary = {"value": 1, "extra": True}
    assert exp.unwrap_principle(arbitrary) is arbitrary
    assert exp.is_quarantined({"quarantined": arbitrary}) is False
    assert exp.is_quarantined({"fabricated": {"principle": "why", "value": True}}) is True


def test_scenario_verify_7209_preflight_rejects_invalid_manifest_bytes() -> None:
    """SCENARIO-VERIFY-7209-PREFLIGHT keeps malformed manifest authentication false."""

    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    checks = exp.upstream_gate_rows(
        upstream,
        UPSTREAM.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        b"not-json",
        {},
    )

    assert (
        next(row for row in checks if row["check"] == "sidecar_authentication")["passed"] is False
    )


def test_scenario_verify_7209_capture_retains_raw_and_compiled_evidence(
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-CAPTURE keeps transport and exact references together."""

    _, public_rows, authority_rows = sealed
    call = exp.build_schedule(public_rows, authority_rows)[0]
    completion = fixture.extract_public_completion(
        str(call["input_text"]).encode("utf-8"), str(call["call_type"])
    )
    row = exp.build_completion_row(
        call,
        _response(fixture.canonical_json(completion)),
        _resource(),
    )

    assert row["terminal_state"] == "complete"
    assert row["parse_valid"] is True
    assert row["truncated"] is False
    assert row["exact_reference_valid"] is True
    assert row["compile_errors"] == []
    assert row["reasoning_disabled_observed"] is True
    assert row["request_payload"] == {"messages": [], "max_tokens": 384}
    assert row["raw_completion"]
    assert row["grammar"] == call["grammar"]
    assert row["grammar_sha256"] == call["grammar_sha256"]
    assert row["cuda_offload"]["cuda_offload_confirmed"] is True


@pytest.mark.parametrize(
    ("raw", "finish_reason", "reasoning", "expected_failure"),
    (
        ("not-json", "stop", "", "json_parse_error"),
        ('{"outcome":"unknown","relations":[]}', "length", "", "truncated"),
        ('{"outcome":"unknown","relations":[]}', "stop", "hidden thought", "reasoning_present"),
        ("<think>x</think>{}", "stop", "", "reasoning_present"),
    ),
)
def test_scenario_verify_7209_reasoning_and_parse_fail_closed(
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
    raw: str,
    finish_reason: str,
    reasoning: str,
    expected_failure: str,
) -> None:
    """SCENARIO-VERIFY-7209-REASONING inspects response evidence, not server flags."""

    _, public_rows, authority_rows = sealed
    call = exp.build_schedule(public_rows, authority_rows)[0]
    row = exp.build_completion_row(
        call,
        _response(raw, finish_reason=finish_reason, reasoning=reasoning),
        _resource(),
    )

    assert expected_failure in row["failure_reasons"]
    assert row["terminal_state"] != "complete"


def test_scenario_verify_7209_capture_rejects_transport_shape_and_reference_errors(
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-CAPTURE retains transport, object, and compiler failures."""

    _, public_rows, authority_rows = sealed
    call = exp.build_schedule(public_rows, authority_rows)[0]
    transport = _response("{}")
    transport["error"] = "TimeoutError:bounded"
    transport_row = exp.build_completion_row(call, transport, _resource())
    assert transport_row["failure_reasons"][0].startswith("transport_error:")

    list_row = exp.build_completion_row(call, _response("[]"), _resource())
    assert list_row["parse_valid"] is False
    assert list_row["parse_error"].startswith("ValueError:")

    invalid = fixture.extract_public_completion(
        str(call["input_text"]).encode("utf-8"), str(call["call_type"])
    )
    invalid["relations"][0]["subject_start"] = 999
    compiled_row = exp.build_completion_row(
        call, _response(fixture.canonical_json(invalid)), _resource()
    )
    assert "compile:span_out_of_range" in compiled_row["failure_reasons"]


def test_scenario_verify_7209_execution_reaches_shipped_executor(
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-EXECUTION scores source and claim references separately."""

    _, public_rows, authority_rows = sealed
    schedule = exp.build_schedule(public_rows, authority_rows)
    completion_rows = _completion_rows(schedule)
    comparisons = exp.score_completion_pairs(schedule, completion_rows, authority_rows)

    assert len(comparisons) == 16
    assert all(row["metric"] == 1 for row in comparisons)
    assert all(row["prediction"] == "supported" for row in comparisons)
    assert all(row["source_relation_agreement"] is True for row in comparisons)
    assert all(row["claim_relation_agreement"] is True for row in comparisons)
    assert all(row["executor_invoked"] is True for row in comparisons)
    assert {row["arm"] for row in comparisons} == {"grammar_only", "reference"}


def test_scenario_verify_7209_execution_preserves_wrong_relation_failure(
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-EXECUTION does not repair a parse-valid wrong predicate."""

    _, public_rows, authority_rows = sealed
    schedule = exp.build_schedule(public_rows, authority_rows)
    completion_rows = _completion_rows(schedule)
    target = next(
        row for row in completion_rows if row["arm"] == "reference" and row["call_type"] == "claim"
    )
    target["parsed_completion"]["relations"][0]["polarity"] = "negative"
    target["compiled_completion"]["relations"][0]["polarity"] = "negative"
    comparisons = exp.score_completion_pairs(schedule, completion_rows, authority_rows)

    affected = next(
        row
        for row in comparisons
        if row["unit_id"] == target["unit_id"] and row["arm"] == "reference"
    )
    assert affected["claim_relation_agreement"] is False
    assert affected["metric"] == 0
    assert affected["error"] == "canary_relation_or_decision_disagreement"


def test_scenario_verify_7209_executor_rejects_unusable_counts_and_sentences(
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-EXECUTION preserves invalid compiled pair reasons."""

    _, public_rows, authority_rows = sealed
    schedule = exp.build_schedule(public_rows, authority_rows)
    calls = _completion_rows(schedule)
    source = next(row for row in calls if row["call_type"] == "source")["compiled_completion"]
    claim = next(row for row in calls if row["call_type"] == "claim")["compiled_completion"]
    source_text = str(schedule[0]["input_text"])

    assert exp._execute_compiled_pair(source_text, {"outcome": "unknown"}, claim)["errors"] == [
        "unusable_completion"
    ]
    assert exp._execute_compiled_pair(source_text, {"outcome": "known", "relations": []}, claim)[
        "errors"
    ] == ["relation_count"]
    bad_sentence = deepcopy(source)
    bad_sentence["relations"][0]["sentence_index"] = 99
    assert exp._execute_compiled_pair(source_text, bad_sentence, claim)["errors"] == [
        "sentence_index"
    ]
    assert exp._sentence_ranges(b"A precedes B.   C precedes D.") == [(0, 13), (16, 29)]


def test_scenario_verify_7209_readiness_requires_all_16_and_seven_of_eight(
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-READINESS applies both fixed denominators."""

    _, public_rows, authority_rows = sealed
    schedule = exp.build_schedule(public_rows, authority_rows)
    calls = _completion_rows(schedule)
    comparisons = exp.score_completion_pairs(schedule, calls, authority_rows)
    ready = exp.readiness_receipt(calls, comparisons)

    assert ready["reference_call_denominator"] == 16
    assert ready["complete_parse_valid_exact_reference_calls"] == 16
    assert ready["combined_interpretation_denominator"] == 8
    assert ready["correct_combined_interpretations"] == 8
    assert ready["span_canary_ready_score"] == 1

    incomplete = deepcopy(calls[:-1])
    assert exp.readiness_receipt(incomplete, comparisons)["span_canary_ready_score"] == 0
    six_correct = deepcopy(comparisons)
    reference_rows = [row for row in six_correct if row["arm"] == "reference"]
    reference_rows[0]["metric"] = 0
    reference_rows[1]["metric"] = 0
    assert exp.readiness_receipt(calls, six_correct)["span_canary_ready_score"] == 0


def test_scenario_verify_7209_artifact_validates_block_and_complete_shape(
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-ARTIFACT rejects changed gates and readiness."""

    _, public_rows, authority_rows = sealed
    schedule = exp.build_schedule(public_rows, authority_rows)
    calls = _completion_rows(schedule)
    comparisons = exp.score_completion_pairs(schedule, calls, authority_rows)
    token_receipt = exp.measure_token_budgets(schedule, lambda value: range((len(value) + 3) // 4))
    complete = exp.finalize_measured_artifact(
        exp.base_artifact(exp.RUN_DATE),
        calls,
        comparisons,
        token_receipt,
        duration_s=12.0,
        live_evidence=True,
    )

    assert exp.validate_artifact(complete) == []
    assert complete["status"] == "complete"
    assert complete["verdict_class"] == "circular_positive"
    assert complete["span_canary_ready_score"] == 1
    changed = deepcopy(complete)
    changed["sample_size_budget"]["completed_calls"] = 31
    assert "sample_size_budget" in exp.validate_artifact(changed)

    check = exp.gate_row("idle_gpu", "one", "none", False, upstream="gpu", field="count")
    blocked = exp.finalize_blocked_artifact(
        exp.base_artifact(exp.RUN_DATE), [check], duration_s=0.5
    )
    assert exp.validate_artifact(blocked) == []
    assert blocked["status"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "idle_gpu"

    no_generation = exp.finalize_measured_artifact(
        exp.base_artifact(exp.RUN_DATE),
        [],
        [],
        token_receipt,
        duration_s=2.0,
        live_evidence=False,
    )
    assert exp.validate_artifact(no_generation) == []
    assert no_generation["inference_substrate_class"] == "model_load_no_generation"


def test_scenario_verify_7209_artifact_validator_rejects_each_terminal_drift(
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-ARTIFACT names malformed and inconsistent terminal fields."""

    _, public_rows, authority_rows = sealed
    schedule = exp.build_schedule(public_rows, authority_rows)
    calls = _completion_rows(schedule)
    comparisons = exp.score_completion_pairs(schedule, calls, authority_rows)
    token_receipt = exp.measure_token_budgets(schedule, lambda value: list(value))
    complete = exp.finalize_measured_artifact(
        exp.base_artifact(exp.RUN_DATE),
        calls,
        comparisons,
        token_receipt,
        duration_s=12.0,
        live_evidence=True,
    )

    assert exp.validate_artifact([]) == ["artifact_mapping"]
    missing = deepcopy(complete)
    missing.pop("rows")
    assert exp.validate_artifact(missing) == ["missing_required_field:rows"]
    mutations = {
        "field_principles": {},
        "run_date": "wrong",
        "MODEL_SPECS": [],
        "random_seed": 0,
        "execution_venue": "wrong",
        "verifier_is_oracle": False,
        "duration_s": -1,
        "readiness_receipt": {},
        "span_canary_ready_score": 0,
    }
    expected = {
        "field_principles": "field_principles",
        "run_date": "run_date",
        "MODEL_SPECS": "MODEL_SPECS",
        "random_seed": "random_seed",
        "execution_venue": "execution_identity",
        "verifier_is_oracle": "verifier_is_oracle",
        "duration_s": "duration_s",
        "readiness_receipt": "readiness_receipt",
        "span_canary_ready_score": "span_canary_ready_score",
    }
    for field, value in mutations.items():
        changed = deepcopy(complete)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected[field] in exp.validate_artifact(changed)

    non_mapping_budget = deepcopy(complete)
    non_mapping_budget["sample_size_budget"] = []
    non_mapping_budget["reproducibility_checksum"] = exp.artifact_checksum(non_mapping_budget)
    assert "sample_size_budget" in exp.validate_artifact(non_mapping_budget)
    running = deepcopy(complete)
    running["status"] = "running"
    running["reproducibility_checksum"] = exp.artifact_checksum(running)
    assert "status" in exp.validate_artifact(running)
    bad_token = deepcopy(complete)
    bad_token["token_budget_receipt"] = {}
    bad_token["reproducibility_checksum"] = exp.artifact_checksum(bad_token)
    assert "token_budget_receipt" in exp.validate_artifact(bad_token)
    wrong_ready_verdict = deepcopy(complete)
    wrong_ready_verdict["verdict_class"] = "positive"
    wrong_ready_verdict["reproducibility_checksum"] = exp.artifact_checksum(wrong_ready_verdict)
    assert "verdict_class" in exp.validate_artifact(wrong_ready_verdict)
    bad_live = deepcopy(complete)
    bad_live["gpu_receipts"]["provenance_ok"] = False
    bad_live["reproducibility_checksum"] = exp.artifact_checksum(bad_live)
    assert "live_inference_provenance" in exp.validate_artifact(bad_live)

    null = exp.finalize_measured_artifact(
        exp.base_artifact(exp.RUN_DATE), [], [], token_receipt, duration_s=2.0, live_evidence=False
    )
    null["verdict_class"] = "blocked"
    null["reproducibility_checksum"] = exp.artifact_checksum(null)
    assert "verdict_class" in exp.validate_artifact(null)
    blocked = exp.finalize_blocked_artifact(
        exp.base_artifact(exp.RUN_DATE),
        [exp.gate_row("x", 1, 0, False, upstream="x", field="x")],
        duration_s=0.1,
    )
    blocked["span_canary_ready_score"] = 1
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    assert "blocked_terminal_state" in exp.validate_artifact(blocked)


def test_scenario_verify_7209_artifact_raw_manifest_binds_frozen_contract(
    tmp_path: Path,
    sealed: tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7209-ARTIFACT seals every raw row and frozen setting."""

    _, public_rows, authority_rows = sealed
    schedule = exp.build_schedule(public_rows, authority_rows)
    calls = _completion_rows(schedule)
    manifest = exp.write_raw_manifest(tmp_path, schedule, calls, {"sha256": "sha256:model"})

    assert manifest["schema"] == "carnot.exp7209.raw_manifest.v1"
    assert manifest["raw_row_count"] == 32
    assert manifest["development_or_test_label_read_count"] == 0
    assert manifest["schedule_sha256"] == exp.sha256_json(schedule)
    assert len(manifest["raw_rows"]) == 32
    assert (tmp_path / "raw_manifest.json").is_file()


def test_experiment_7209_entrypoint_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-7209 keeps one executable entrypoint with the fixed date."""

    called: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> int:
        called.append(argv)
        return 0

    monkeypatch.setattr(exp, "main", fake_main)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(
            str(REPO / "scripts/experiments/experiment_7209_v635_span_canary.py"),
            run_name="__main__",
        )

    assert raised.value.code == 0
    assert called == [None]


def test_req_verify_7209_date_parser_is_fixed() -> None:
    """REQ-VERIFY-7209 accepts only the declared execution date."""

    assert exp._date_argument("20260911") == "20260911"
    with pytest.raises(Exception, match="run date must be 20260911"):
        exp._date_argument("20260910")
