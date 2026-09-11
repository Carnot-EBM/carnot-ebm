"""Focused tests for the authenticated V636 span canary.

Spec refs: REQ-VERIFY-7223 and SCENARIO-VERIFY-7223-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7208_v635_span_fixture as fixture
from carnot import experiment_7223_v636_span_canary as exp


REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "results/experiment_7222_v636_span_fixture.json"
PUBLIC = REPO / "results/raw/experiment_7222/public.jsonl"
AUTHORITY = REPO / "results/raw/experiment_7222/authority.jsonl"
MANIFEST = REPO / "results/raw/experiment_7222/manifest.json"
EXCLUSION = REPO / "ops/exclusion_manifest.yaml"


def _selected() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return exp.load_calibration_selection(PUBLIC, AUTHORITY)


def _response(raw: str) -> dict[str, object]:
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": raw, "reasoning_content": ""},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        "timings": {"prompt_ms": 2.0, "predicted_ms": 3.0},
    }
    request = {"messages": [], "max_tokens": 384}
    return {
        "raw_request": request,
        "raw_request_bytes_b64": base64.b64encode(
            exp.canonical_json(request).encode("utf-8")
        ).decode("ascii"),
        "raw_response": response,
        "raw_response_bytes_b64": base64.b64encode(
            exp.canonical_json(response).encode("utf-8")
        ).decode("ascii"),
        "raw_completion": raw,
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "finish_reason": "stop",
        "latency_s": 0.1,
        "error": None,
    }


def _completion_rows(schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for call in schedule:
        completion = fixture.extract_public_completion(
            str(call["input_text"]).encode("utf-8"), str(call["call_type"])
        )
        rows.append(
            exp.build_completion_row(
                call,
                _response(fixture.canonical_json(completion)),
                {
                    "server_pid": 123,
                    "gpu_uuid": "GPU-test",
                    "lease_id": "lease:test",
                    "cuda_offload_confirmed": True,
                },
            )
        )
    return rows


def test_req_verify_7223_spec_and_fixed_identity() -> None:
    """REQ-VERIFY-7223 fixes the model, substrate class, and principle contract."""

    spec = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-VERIFY-7223" in spec
    assert exp.MODEL_SPECS == [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]
    assert exp.REQUEST_CAP_S == 90.0
    assert exp.INFERENCE_DEADLINE_S == 1500.0
    assert max(exp.TOKEN_BUDGETS.values()) <= 512
    assert exp.FIELD_PRINCIPLES["field_principles"].startswith("Annotate actual values")
    assert exp.unwrap_principle({"principle": "why", "value": 1}) == 1
    arbitrary = {"value": 1, "extra": True}
    assert exp.unwrap_principle(arbitrary) is arbitrary


def test_scenario_verify_7223_selection_is_balanced_frozen_and_blind() -> None:
    """SCENARIO-VERIFY-7223-SELECTION freezes eight calibration-only public units."""

    public_rows, authority_rows = _selected()
    schedule = exp.build_schedule(public_rows, authority_rows)
    receipt = exp.selection_receipt(public_rows, authority_rows, schedule)

    assert len(public_rows) == len(authority_rows) == 8
    assert {row["split"] for row in authority_rows} == {"canary"}
    assert {row["relation_family"] for row in authority_rows} == {
        "precedes",
        "starts before",
        "ends before",
        "occurs before",
    }
    assert {
        variant: sum(row["variant"] == variant for row in authority_rows)
        for variant in exp.REQUIRED_VARIANTS
    } == {
        "supported": 2,
        "reversal": 2,
        "joint_support": 2,
        "support_removed": 2,
    }
    assert len(schedule) == 16
    assert all(row["arm"] == "syntax_only_capture" for row in schedule)
    assert all(set(row["model_input"]) in ({"source_text"}, {"claim_text"}) for row in schedule)
    assert all(not (set(row) & exp.AUTHORITY_ONLY_FIELDS) for row in schedule)
    assert receipt["selection_frozen_before_inference"] is True
    assert receipt["held_out_rows_read_for_selection"] == 0
    assert receipt["model_outcomes_read_for_selection"] == 0
    assert exp.schedule_errors(schedule, public_rows, authority_rows) == []

    changed = deepcopy(schedule)
    changed[0]["output_token_budget"] = 513
    assert "call_0:output_token_budget" in exp.schedule_errors(changed, public_rows, authority_rows)


def test_scenario_verify_7223_preflight_authenticates_exp7222_and_rejects_tampering() -> None:
    """SCENARIO-VERIFY-7223-PREFLIGHT authenticates every Exp7222 input separately."""

    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    checks = exp.upstream_gate_rows(
        upstream,
        UPSTREAM.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        MANIFEST.read_bytes(),
        exp.load_yaml(EXCLUSION),
    )
    assert checks and all(row["passed"] is True for row in checks)
    assert {row["check"] for row in checks} >= {
        "exact_upstream_bytes",
        "structured_quarantine",
        "exclusion_manifest",
        "producer_gate_fields",
        "upstream_authentication",
        "sidecar_authentication",
        "historical_exp7209_interpretation",
    }

    quarantined = deepcopy(upstream)
    quarantined["quarantined"] = {"principle": "why", "value": True}
    failed = exp.upstream_gate_rows(
        quarantined,
        UPSTREAM.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        MANIFEST.read_bytes(),
        {},
    )
    assert next(row for row in failed if row["check"] == "structured_quarantine")["passed"] is False

    changed = exp.upstream_gate_rows(
        upstream,
        UPSTREAM.read_bytes() + b"\n",
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        MANIFEST.read_bytes(),
        {},
    )
    assert next(row for row in changed if row["check"] == "exact_upstream_bytes")["passed"] is False


def test_scenario_verify_7223_scoring_uses_same_raw_calls_for_both_offline_arms() -> None:
    """SCENARIO-VERIFY-7223-SCORING compares two validators on one capture."""

    public_rows, authority_rows = _selected()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule)
    rows = exp.score_offline_arms(schedule, completions, authority_rows)

    assert len(rows) == 16
    assert {row["arm"] for row in rows} == {"syntax_only", "reference_type_semantics"}
    assert all(row["metric"] == 1 and row["error"] is None for row in rows)
    assert {row["expected_prediction"] for row in rows} >= {
        "supported",
        "contradicted",
        "unknown",
    }
    for unit_id in {str(row["unit_id"]) for row in rows}:
        pair = [row for row in rows if row["unit_id"] == unit_id]
        assert pair[0]["raw_call_hashes"] == pair[1]["raw_call_hashes"]
    semantic = [row for row in rows if row["arm"] == "reference_type_semantics"]
    assert all(row["source_relation_agreement"] is True for row in semantic)
    assert all(row["claim_relation_agreement"] is True for row in semantic)
    assert all(row["executor_invoked"] is True for row in semantic)

    claim = next(row for row in completions if row["call_type"] == "claim")
    claim["compiled_completion"]["relations"][0]["polarity"] = "negative"
    changed = exp.score_offline_arms(schedule, completions, authority_rows)
    target = next(
        row
        for row in changed
        if row["unit_id"] == claim["unit_id"] and row["arm"] == "reference_type_semantics"
    )
    assert target["metric"] == 0
    assert target["claim_relation_agreement"] is False


def test_scenario_verify_7223_readiness_uses_exact_thresholds_and_complete_score() -> None:
    """SCENARIO-VERIFY-7223-READINESS applies the frozen seven and six thresholds."""

    public_rows, authority_rows = _selected()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule)
    comparisons = exp.score_offline_arms(schedule, completions, authority_rows)

    ready = exp.readiness_receipt(
        schedule,
        completions,
        comparisons,
        authority_leakage_count=0,
        model_identity_errors=[],
    )
    assert ready["parse_complete_units"] == 8
    assert ready["semantic_correct_units"] == 8
    assert ready["span_canary_ready_score"] == 1
    assert ready["span_canary_complete_score"] == 1

    one_bad = deepcopy(completions)
    one_bad[0]["parse_valid"] = False
    assert (
        exp.readiness_receipt(
            schedule, one_bad, comparisons, authority_leakage_count=0, model_identity_errors=[]
        )["span_canary_ready_score"]
        == 1
    )
    two_bad = deepcopy(one_bad)
    two_bad[2]["parse_valid"] = False
    assert (
        exp.readiness_receipt(
            schedule, two_bad, comparisons, authority_leakage_count=0, model_identity_errors=[]
        )["span_canary_ready_score"]
        == 0
    )

    six_semantic = deepcopy(comparisons)
    semantic = [row for row in six_semantic if row["arm"] == "reference_type_semantics"]
    semantic[0]["metric"] = semantic[1]["metric"] = 0
    assert (
        exp.readiness_receipt(
            schedule,
            completions,
            six_semantic,
            authority_leakage_count=0,
            model_identity_errors=[],
        )["span_canary_ready_score"]
        == 1
    )
    semantic[2]["metric"] = 0
    assert (
        exp.readiness_receipt(
            schedule,
            completions,
            six_semantic,
            authority_leakage_count=0,
            model_identity_errors=[],
        )["span_canary_ready_score"]
        == 0
    )
    assert (
        exp.readiness_receipt(
            schedule,
            completions[:-1],
            comparisons,
            authority_leakage_count=0,
            model_identity_errors=[],
        )["span_canary_complete_score"]
        == 0
    )
    assert (
        exp.readiness_receipt(
            schedule,
            completions,
            comparisons,
            authority_leakage_count=1,
            model_identity_errors=[],
        )["span_canary_ready_score"]
        == 0
    )
    assert (
        exp.readiness_receipt(
            schedule,
            completions,
            comparisons,
            authority_leakage_count=0,
            model_identity_errors=["hash"],
        )["span_canary_ready_score"]
        == 0
    )


def test_scenario_verify_7223_artifact_complete_null_block_and_tampering() -> None:
    """SCENARIO-VERIFY-7223-ARTIFACT validates each terminal verdict class."""

    public_rows, authority_rows = _selected()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule)
    comparisons = exp.score_offline_arms(schedule, completions, authority_rows)
    token_receipt = exp.measure_token_budgets(schedule, lambda value: range((len(value) + 3) // 4))
    complete = exp.finalize_measured_artifact(
        exp.base_artifact(exp.RUN_DATE),
        schedule,
        completions,
        comparisons,
        token_receipt,
        duration_s=12.0,
        live_evidence=True,
        authority_leakage_count=0,
        model_identity_errors=[],
    )
    complete["gpu_receipts"] = {"provenance_ok": True}
    complete["model_identity_receipt"] = {"identity_errors": []}
    complete["runner_receipt"].update({"replica_count": 1, "cleanup_ok": True})
    complete["reproducibility_checksum"] = exp.artifact_checksum(complete)

    assert exp.validate_artifact(complete) == []
    assert complete["status"] == "complete"
    assert complete["verdict_class"] == "circular_positive"
    assert complete["span_canary_ready_score"] == 1
    assert complete["span_canary_complete_score"] == 1
    assert complete["sample_size_budget"]["completed_calls"] == 16
    assert complete["sample_size_budget"]["usable_calls"] == 16
    assert complete["token_budget_receipt"]["generated_token_total"] == 128
    assert complete["token_budget_receipt"]["bounded_stop_counts"] == {"stop": 16}
    assert complete["token_budget_receipt"]["all_observed_calls_within_budget"] is True

    failed_rows = deepcopy(comparisons)
    for row in failed_rows:
        if row["arm"] == "reference_type_semantics":
            row["metric"] = 0
    null = exp.finalize_measured_artifact(
        exp.base_artifact(exp.RUN_DATE),
        schedule,
        completions,
        failed_rows,
        token_receipt,
        duration_s=12.0,
        live_evidence=True,
        authority_leakage_count=0,
        model_identity_errors=[],
    )
    null["gpu_receipts"] = {"provenance_ok": True}
    null["model_identity_receipt"] = {"identity_errors": []}
    null["reproducibility_checksum"] = exp.artifact_checksum(null)
    assert null["verdict_class"] == "null"
    assert exp.validate_artifact(null) == []

    check = exp.gate_row("model_cache", True, False, False, upstream="cache", field="path")
    blocked = exp.finalize_blocked_artifact(
        exp.base_artifact(exp.RUN_DATE), [check], duration_s=0.2
    )
    assert exp.validate_artifact(blocked) == []
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["failed_check"] == "model_cache"

    changed = deepcopy(complete)
    changed["sample_size_budget"]["completed_rows"] = 15
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "sample_size_budget" in exp.validate_artifact(changed)
    changed = deepcopy(complete)
    changed["span_canary_ready_score"] = 0
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "span_canary_ready_score" in exp.validate_artifact(changed)
    assert exp.validate_artifact([]) == ["artifact_mapping"]


def test_scenario_verify_7223_artifact_raw_manifest_binds_selection_and_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7223-ARTIFACT seals the one frozen scheduled capture."""

    public_rows, authority_rows = _selected()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule)
    manifest = exp.write_raw_manifest(
        tmp_path,
        schedule,
        completions,
        exp.selection_receipt(public_rows, authority_rows, schedule),
        {"gguf_sha256": "sha256:model"},
    )

    assert manifest["schema"] == "carnot.exp7223.raw_manifest.v1"
    assert manifest["raw_row_count"] == 16
    assert manifest["authority_path_opened_by_model_worker"] is False
    assert manifest["held_out_outcome_count"] == 0
    assert manifest["schedule_sha256"] == exp.sha256_json(schedule)
    assert len(manifest["raw_rows"]) == 16
    assert (tmp_path / "raw_manifest.json").is_file()


def test_scenario_verify_7223_selection_and_schedule_reject_malformed_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-7223-SELECTION fails closed on every selection boundary."""

    public_all = [json.loads(line) for line in PUBLIC.read_text(encoding="utf-8").splitlines()]
    authority_all = [
        json.loads(line) for line in AUTHORITY.read_text(encoding="utf-8").splitlines()
    ]
    valid_public, valid_authority = exp.load_calibration_selection(PUBLIC, AUTHORITY)

    def install(
        public_rows: list[dict[str, object]], authority_rows: list[dict[str, object]]
    ) -> None:
        values = iter((public_rows, authority_rows))
        monkeypatch.setattr(exp, "_read_jsonl", lambda _path: deepcopy(next(values)))

    install(public_all[:-1], authority_all)
    with pytest.raises(ValueError, match="exp7222_panel_denominator"):
        exp.load_calibration_selection(PUBLIC, AUTHORITY)

    duplicate_public = deepcopy(public_all)
    duplicate_public[-1]["unit_id"] = duplicate_public[0]["unit_id"]
    install(duplicate_public, authority_all)
    with pytest.raises(ValueError, match="exp7222_public_authority_identity"):
        exp.load_calibration_selection(PUBLIC, AUTHORITY)

    three_bases = deepcopy(authority_all)
    target = next(
        row
        for row in three_bases
        if row["split"] == "canary" and row["relation_family"] == "precedes"
    )
    target["base_id"] = "third-base"
    install(public_all, three_bases)
    with pytest.raises(ValueError, match="canary_family_base_denominator"):
        exp.load_calibration_selection(PUBLIC, AUTHORITY)

    missing_cell = deepcopy(authority_all)
    target = next(
        row
        for row in missing_cell
        if row["split"] == "canary"
        and row["relation_family"] == "precedes"
        and row["variant"] == "supported"
    )
    target["variant"] = "not-supported"
    install(public_all, missing_cell)
    with pytest.raises(ValueError, match="calibration_selection_cell"):
        exp.load_calibration_selection(PUBLIC, AUTHORITY)

    private_public = deepcopy(public_all)
    selected_id = next(
        row["unit_id"]
        for row in authority_all
        if row["split"] == "canary"
        and row["relation_family"] == "precedes"
        and row["variant"] == "supported"
    )
    next(row for row in private_public if row["unit_id"] == selected_id)["split"] = "canary"
    install(private_public, authority_all)
    with pytest.raises(ValueError, match="public_view_contains_private_fields"):
        exp.load_calibration_selection(PUBLIC, AUTHORITY)

    public_rows, authority_rows = valid_public, valid_authority
    with pytest.raises(ValueError, match="calibration_selection_denominator"):
        exp.build_schedule(public_rows[:-1], authority_rows)
    duplicate = deepcopy(public_rows)
    duplicate[-1]["unit_id"] = duplicate[0]["unit_id"]
    with pytest.raises(ValueError, match="calibration_public_authority_identity"):
        exp.build_schedule(duplicate, authority_rows)
    non_calibration = deepcopy(authority_rows)
    non_calibration[0]["split"] = "test"
    with pytest.raises(ValueError, match="non_calibration_authority"):
        exp.build_schedule(public_rows, non_calibration)
    assert exp.schedule_errors([], public_rows[:-1], authority_rows)[0].startswith(
        "schedule_rebuild:ValueError"
    )
    schedule = exp.build_schedule(public_rows, authority_rows)
    assert "schedule_count" in exp.schedule_errors(schedule[:-1], public_rows, authority_rows)
    extra = deepcopy(schedule)
    extra[0]["private"] = True
    assert "call_0:extra_fields" in exp.schedule_errors(extra, public_rows, authority_rows)


def test_scenario_verify_7223_preflight_handles_malformed_manifests_and_missing_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7223-PREFLIGHT keeps parse and missing-history failures visible."""

    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    upstream["source_artifact_hashes"] = []
    monkeypatch.setattr(exp, "HISTORICAL_EXP7209_PATH", tmp_path / "missing.json")
    checks = exp.upstream_gate_rows(
        upstream,
        UPSTREAM.read_bytes(),
        PUBLIC.read_bytes(),
        AUTHORITY.read_bytes(),
        b"not-json",
        {"retired": ["experiment_7222_v636_span_fixture"]},
    )
    by_name = {row["check"]: row for row in checks}
    assert by_name["exclusion_manifest"]["passed"] is False
    assert by_name["sidecar_authentication"]["passed"] is False
    assert by_name["historical_exp7209_interpretation"]["passed"] is False


def test_scenario_verify_7223_scoring_preserves_missing_call_failures() -> None:
    """SCENARIO-VERIFY-7223-SCORING retains missing transport and executor evidence."""

    public_rows, authority_rows = _selected()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule)[2:]
    rows = exp.score_offline_arms(schedule, completions, authority_rows)
    first = [row for row in rows if row["unit_id"] == authority_rows[0]["unit_id"]]
    assert all(row["metric"] == 0 for row in first)
    semantic = next(row for row in first if row["arm"] == "reference_type_semantics")
    assert semantic["executor_errors"] == ["missing_call"]
    assert semantic["abstention"] is True


def test_scenario_verify_7223_validator_names_all_terminal_drift() -> None:
    """SCENARIO-VERIFY-7223-ARTIFACT names each malformed terminal contract."""

    public_rows, authority_rows = _selected()
    schedule = exp.build_schedule(public_rows, authority_rows)
    completions = _completion_rows(schedule)
    comparisons = exp.score_offline_arms(schedule, completions, authority_rows)
    token_receipt = exp.measure_token_budgets(schedule, lambda value: list(value))
    complete = exp.finalize_measured_artifact(
        exp.base_artifact(exp.RUN_DATE),
        schedule,
        completions,
        comparisons,
        token_receipt,
        duration_s=12.0,
        live_evidence=True,
        authority_leakage_count=0,
        model_identity_errors=[],
    )
    complete["gpu_receipts"] = {"provenance_ok": True}
    complete["model_identity_receipt"] = {"identity_errors": []}
    complete["reproducibility_checksum"] = exp.artifact_checksum(complete)

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
        "frozen_decoding_contract": {},
        "reproducibility_checksum": "wrong",
        "canary_rows": {},
        "sample_size_budget": [],
        "status": "running",
        "schedule": {},
        "model_identity_receipt": [],
        "selection_receipt": [],
        "readiness_receipt": {},
        "span_canary_complete_score": 0,
        "verdict_class": "positive",
        "token_budget_receipt": {},
        "model_invoked": False,
    }
    expected = {
        "field_principles": "field_principles",
        "run_date": "run_date",
        "MODEL_SPECS": "MODEL_SPECS",
        "random_seed": "random_seed",
        "execution_venue": "execution_identity",
        "verifier_is_oracle": "verifier_is_oracle",
        "duration_s": "duration_s",
        "frozen_decoding_contract": "frozen_decoding_contract",
        "reproducibility_checksum": "reproducibility_checksum",
        "canary_rows": "rows",
        "sample_size_budget": "sample_size_budget",
        "status": "status",
        "schedule": "schedule",
        "model_identity_receipt": "model_identity_receipt",
        "selection_receipt": "readiness_receipt",
        "readiness_receipt": "readiness_receipt",
        "span_canary_complete_score": "span_canary_complete_score",
        "verdict_class": "verdict_class",
        "token_budget_receipt": "token_budget_receipt",
        "model_invoked": "live_inference_provenance",
    }
    for field, value in mutations.items():
        changed = deepcopy(complete)
        changed[field] = value
        if field != "reproducibility_checksum":
            changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected[field] in exp.validate_artifact(changed), field

    short = deepcopy(complete)
    short["duration_s"] = 9.0
    short["reproducibility_checksum"] = exp.artifact_checksum(short)
    assert "bounded_generation_duration_floor" in exp.validate_artifact(short)
    identity_error = deepcopy(complete)
    identity_error["model_identity_receipt"]["identity_errors"] = ["hash"]
    identity_error["reproducibility_checksum"] = exp.artifact_checksum(identity_error)
    assert "model_identity_receipt" in exp.validate_artifact(identity_error)

    malformed_block = exp.finalize_blocked_artifact(
        exp.base_artifact(exp.RUN_DATE),
        [exp.gate_row("x", 1, 0, False, upstream="x", field="x")],
        duration_s=0.1,
    )
    malformed_block["span_canary_ready_score"] = 1
    malformed_block["inference_substrate_class"] = "wrong"
    malformed_block["reproducibility_checksum"] = exp.artifact_checksum(malformed_block)
    assert {"blocked_terminal_state", "blocked_substrate"} <= set(
        exp.validate_artifact(malformed_block)
    )


def _mock_live_context() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    public_rows, authority_rows = _selected()
    schedule = exp.build_schedule(public_rows, authority_rows)
    context: dict[str, object] = {
        "schedule": schedule,
        "selection_receipt": exp.selection_receipt(public_rows, authority_rows, schedule),
        "model_path": "/tmp/model.gguf",
        "server_path": "/tmp/llama-server",
        "tokenizer_loader": object(),
        "model_identity": {
            "hf_id": exp.QWEN_MODEL_ID,
            "quantization": exp.QUANTIZATION,
            "revision": "revision",
            "gguf_sha256": "sha256:model",
            "embedded_chat_template_present": True,
        },
    }
    return [], public_rows, authority_rows, context


class _TokenizerOwner:
    def close(self) -> None:
        self.closed = True


def test_req_verify_7223_preflight_adapter_identity_and_source_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-7223 reuses resource checks and reports exact identity faults."""

    returned = _mock_live_context()
    monkeypatch.setattr(exp.capture, "_collect_preflight", lambda *args, **kwargs: returned)
    checks, public_rows, authority_rows, context = exp._collect_preflight(
        REPO, exp.RUN_DATE, tmp_path / "result", tmp_path, tmp_path
    )
    assert checks == [] and len(public_rows) == len(authority_rows) == 8
    assert context["selection_receipt"]["selected_unit_count"] == 8

    no_schedule = ([], [], [], {})
    monkeypatch.setattr(exp.capture, "_collect_preflight", lambda *args, **kwargs: no_schedule)
    assert exp._collect_preflight(REPO, exp.RUN_DATE, tmp_path / "r", tmp_path, tmp_path) == (
        [],
        [],
        [],
        {},
    )

    valid_identity = returned[3]["model_identity"]
    assert exp._identity_errors(valid_identity, {"provenance_ok": True}) == []
    assert exp._identity_errors({}, {}) == [
        "hf_id",
        "quantization",
        "gguf_revision_or_hash",
        "embedded_chat_template",
        "actual_cuda_execution",
    ]
    hashes = exp._source_hashes(REPO)
    assert hashes["module"].startswith("sha256:")
    assert hashes["upstream_artifact"] == exp.PINNED_UPSTREAM_SHA256
    assert exp._source_hashes(tmp_path)["module"] == "missing"
    exp._progress(99, "test", observed=True)
    assert '"observed":true' in capsys.readouterr().out


def _install_run_mocks(
    monkeypatch: pytest.MonkeyPatch,
    preflight: tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
        dict[str, object],
    ],
    *,
    tokenizer: object | None,
    token_fit: bool = True,
    live_result: dict[str, object] | None = None,
) -> None:
    monkeypatch.setattr(exp, "_collect_preflight", lambda *args: deepcopy(preflight))
    monkeypatch.setattr(exp, "_source_hashes", lambda _root: {"module": "sha256:test"})
    monkeypatch.setattr(
        exp.capture,
        "_load_embedded_tokenizer",
        lambda *args: (
            _TokenizerOwner() if tokenizer is not None else None,
            tokenizer,
            {"embedded_tokenizer_available": tokenizer is not None},
        ),
    )
    monkeypatch.setattr(
        exp,
        "measure_token_budgets",
        lambda *args: {
            "measurement_status": "measured_embedded_gguf_tokenizer",
            "all_forms_fit_with_20_percent_headroom": token_fit,
        },
    )
    if live_result is not None:
        monkeypatch.setattr(exp.capture, "_live_capture", lambda *args: deepcopy(live_result))


def test_req_verify_7223_run_experiment_covers_external_block_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-7223 writes terminal blocks for each unchanged external prerequisite."""

    check = exp.gate_row("missing", True, False, False, upstream="x", field="path")
    _install_run_mocks(monkeypatch, ([check], [], [], {}), tokenizer=None)
    blocked = exp.run_experiment(REPO, output_root=tmp_path / "preflight")
    assert blocked["honest_verdict"] == "blocked_exp7223_missing"

    preflight = _mock_live_context()
    _install_run_mocks(monkeypatch, preflight, tokenizer=None)
    no_tokenizer = exp.run_experiment(REPO, output_root=tmp_path / "tokenizer")
    assert no_tokenizer["gate_check_summary"]["failed_check"] == "embedded_tokenizer_load"

    _install_run_mocks(monkeypatch, preflight, tokenizer=lambda value: [1], token_fit=False)
    no_fit = exp.run_experiment(REPO, output_root=tmp_path / "fit")
    assert no_fit["gate_check_summary"]["failed_check"] == "representation_size_fit"


def test_req_verify_7223_run_experiment_covers_live_success_and_runtime_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-7223 publishes one measured canary or one observed runtime block."""

    preflight = _mock_live_context()
    schedule = preflight[3]["schedule"]
    completions = _completion_rows(schedule)
    success = {
        "rows": completions,
        "runtime_error": None,
        "model_invoked": True,
        "gpu_receipts": {
            "provenance_ok": True,
            "cleanup": {"leak_free": True},
            "server_identity": {"pid": 123, "start_time_ticks": 456},
        },
        "runner_receipt": {"runner": "native_llama.cpp_server", "replica_count": 1},
    }
    _install_run_mocks(
        monkeypatch,
        preflight,
        tokenizer=lambda value: [1],
        live_result=success,
    )
    ticks = iter(range(0, 200, 2))
    monkeypatch.setattr(exp.time, "monotonic", lambda: float(next(ticks)))
    complete = exp.run_experiment(REPO, output_root=tmp_path / "success")
    assert complete["status"] == "complete"
    assert complete["span_canary_ready_score"] == 1
    assert complete["runner_receipt"]["cleanup_ok"] is True
    assert complete["runner_receipt"]["pid_identity"] == {
        "pid": 123,
        "start_time_ticks": 456,
    }
    assert complete["runner_receipt"]["decoding_parameters"]["seed"] == exp.RANDOM_SEED

    runtime_failure = deepcopy(success)
    runtime_failure["rows"] = []
    runtime_failure["runtime_error"] = "TimeoutError:bounded"
    runtime_failure["model_invoked"] = False
    runtime_failure["gpu_receipts"] = {
        "provenance_ok": False,
        "cleanup": {"leak_free": False},
    }
    _install_run_mocks(
        monkeypatch,
        preflight,
        tokenizer=lambda value: [1],
        live_result=runtime_failure,
    )
    ticks = iter(range(0, 200, 2))
    monkeypatch.setattr(exp.time, "monotonic", lambda: float(next(ticks)))
    blocked = exp.run_experiment(REPO, output_root=tmp_path / "runtime")
    assert blocked["status"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == (
        "live_runtime_completion_and_cuda_provenance"
    )
    assert blocked["runner_receipt"]["cleanup_ok"] is False


def test_req_verify_7223_terminal_and_main_error_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-7223 refuses an invalid terminal and returns both CLI result codes."""

    invalid = exp.base_artifact(exp.RUN_DATE)
    invalid["status"] = "complete"
    with pytest.raises(ValueError, match="invalid Exp7223 artifact"):
        exp._terminal(invalid, tmp_path / "bad.json", tmp_path / "bad-checkpoint.json", 0.0)

    valid = exp.finalize_blocked_artifact(
        exp.base_artifact(exp.RUN_DATE),
        [exp.gate_row("x", 1, 0, False, upstream="x", field="x")],
        duration_s=0.1,
    )
    monkeypatch.setattr(exp, "run_experiment", lambda **kwargs: valid)
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["bad"])
    assert exp.main(["--date", exp.RUN_DATE]) == 1


def test_req_verify_7223_date_and_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-7223 keeps one thin executable with the fixed date."""

    assert exp._date_argument("20260911") == "20260911"
    with pytest.raises(Exception, match="run date must be 20260911"):
        exp._date_argument("20260910")

    called: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> int:
        called.append(argv)
        return 0

    monkeypatch.setattr(exp, "main", fake_main)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(
            str(REPO / "scripts/experiments/experiment_7223_v636_span_canary.py"),
            run_name="__main__",
        )
    assert raised.value.code == 0
    assert called == [None]
