"""Tests for REQ-REPORT-7443 and SCENARIO-REPORT-7443-*.

These fixtures keep transport bytes visible. They do not call the capture
producer parser or use corpus labels as extraction truth.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7443_v652_span_audit as audit


def _row(
    *,
    arm: str = "span",
    paragraph: str = "Café 😀 did not exceed 5 mg before 2027.",
    reply: str | None = None,
    finish_reason: str | None = "stop",
    attempted: bool = True,
    terminal_state: str = "response",
    error: str | None = None,
    call_id: str = "development-00-span",
    phase: str = "development",
    unit_id: str = "development-00",
) -> dict:
    if reply is None:
        claims: object = [[0, len(paragraph)]] if arm == "span" else [paragraph]
        reply = json.dumps({"claims": claims}, ensure_ascii=False)
    request = {"max_tokens": 256} if attempted else {}
    response = (
        {
            "choices": [{"finish_reason": finish_reason, "message": {"content": reply}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            "timings": {"predicted_n": 7},
        }
        if terminal_state == "response"
        else {}
    )
    return {
        "arm": arm,
        "attempted": attempted,
        "call_id": call_id,
        "capture_phase": phase,
        "condition": "sealed_development_paragraph",
        "error": error,
        "finish_reason": finish_reason if attempted else None,
        "max_new_tokens": 256,
        "paragraph": paragraph,
        "raw_reply": reply if attempted else "",
        "raw_request": request,
        "raw_request_sha256": audit.canonical_hash(request),
        "raw_response": response,
        "raw_response_sha256": audit.canonical_hash(response),
        "raw_reply_sha256": audit.canonical_hash(reply if attempted else ""),
        "terminal_state": terminal_state if attempted else "unstarted",
        "unit_id": unit_id,
    }


def _unstarted(index: int, arm: str, *, phase: str = "evaluation") -> dict:
    prefix = "evaluation" if phase == "evaluation" else "development"
    row = _row(
        arm=arm,
        attempted=False,
        terminal_state="unstarted",
        finish_reason=None,
        call_id=f"{prefix}-{index:02d}-{arm}",
        phase=phase,
        unit_id=f"{prefix}-{index:02d}",
    )
    row["condition"] = "ragtruth_unchanged_response"
    return row


def _valid_integrity_fixture() -> dict:
    development = []
    for index in range(4):
        for arm in audit.ARMS:
            development.append(
                _row(
                    arm=arm,
                    call_id=f"development-{index:02d}-{arm}",
                    unit_id=f"development-{index:02d}",
                    reply=(
                        json.dumps({"claims": [[0, 4]]})
                        if arm == "span"
                        else json.dumps({"claims": ["Café"]}, ensure_ascii=False)
                    ),
                )
            )
    evaluation = [
        _unstarted(index, arm) for index in range(audit.EVALUATION_UNITS) for arm in audit.ARMS
    ]
    return {
        "development_rows": development,
        "evaluation_rows": evaluation,
        "canary_open": True,
        "producer_event_receipt_class": "historical_producer_model_events",
        "producer_events": [
            {
                "call_id": "generation-0",
                "operation": "generation",
                "scope": "current",
                "state": "attempted",
            },
            {
                "call_id": "generation-0",
                "operation": "generation",
                "scope": "current",
                "state": "completed",
            },
        ],
    }


def test_independent_decode_reconstructs_unicode_and_transport_counts() -> None:
    """SCENARIO-REPORT-7443-RAW reconstructs Unicode from raw response bytes."""

    row = _row()
    reduced = audit.decode_raw_outcome(row)
    assert reduced["disposition"] == "complete"
    assert reduced["selected_propositions"] == [row["paragraph"]]
    assert reduced["literal_span_reconstruction"] is True
    assert reduced["request_token_ceiling"] == 256
    assert reduced["server_completion_tokens"] == 7
    assert reduced["server_predicted_tokens"] == 7
    assert reduced["semantic_quality"] is None
    assert reduced["errors"] == []


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (_row(reply='{"claims": []}'), "empty"),
        (_row(reply='{"claims":', finish_reason="length"), "truncated"),
        (_row(reply="not-json"), "malformed"),
        (_row(reply="", finish_reason="stop"), "empty"),
        (_row(terminal_state="failed", error="transport"), "failed"),
        (_row(terminal_state="cancelled", error="cancelled"), "cancelled"),
        (_unstarted(0, "span"), "unstarted"),
    ],
)
def test_dispositions_stay_distinct(row: dict, expected: str) -> None:
    """SCENARIO-REPORT-7443-RAW does not coerce unknown outcomes to zero."""

    reduced = audit.decode_raw_outcome(row)
    assert reduced["disposition"] == expected
    if expected in {"failed", "cancelled", "unstarted"}:
        assert reduced["output_tokens"] is None
        assert reduced["semantic_quality"] is None


def test_verbatim_reconstruction_rejects_ambiguous_and_nonliteral_text() -> None:
    """REQ-REPORT-7443 independently resolves exact verbatim substrings."""

    paragraph = "It repeated. It repeated."
    ambiguous = _row(
        arm="verbatim",
        paragraph=paragraph,
        reply='{"claims":["It repeated."]}',
    )
    assert "verbatim_ambiguous" in audit.decode_raw_outcome(ambiguous)["errors"]
    changed = _row(
        arm="verbatim",
        paragraph=paragraph,
        reply='{"claims":["It repeats."]}',
    )
    assert "verbatim_not_literal" in audit.decode_raw_outcome(changed)["errors"]


def test_span_offsets_and_finish_reason_are_authenticated() -> None:
    """SCENARIO-REPORT-7443-MUTATIONS rejects offsets and finish drift."""

    bad_offset = _row(reply='{"claims":[[0,999]]}')
    assert "span_bounds" in audit.decode_raw_outcome(bad_offset)["errors"]
    bad_finish = _row()
    bad_finish["finish_reason"] = "length"
    assert "finish_reason_mismatch" in audit.decode_raw_outcome(bad_finish)["errors"]
    bad_hash = _row()
    bad_hash["raw_response_sha256"] = "sha256:changed"
    assert "raw_response_hash_mismatch" in audit.decode_raw_outcome(bad_hash)["errors"]


def test_capture_accounting_reports_one_empty_and_unknown_effect() -> None:
    """SCENARIO-REPORT-7443-PAIRS leaves unopened evaluation effects unknown."""

    development = [_row(reply='{"claims":[]}')]
    development.extend(
        _unstarted(index, arm, phase="development")
        for index, arm in [
            (0, "verbatim"),
            (1, "span"),
            (1, "verbatim"),
            (2, "span"),
            (2, "verbatim"),
            (3, "span"),
            (3, "verbatim"),
        ]
    )
    evaluation = [
        _unstarted(index, arm) for index in range(audit.EVALUATION_UNITS) for arm in audit.ARMS
    ]
    reduced = audit.reduce_capture(development, evaluation, seed=7, draws=20)
    assert reduced["raw_disposition_counts"] == {
        "complete": 0,
        "truncated": 0,
        "malformed": 0,
        "empty": 1,
        "failed": 0,
        "cancelled": 0,
        "unstarted": 103,
    }
    assert reduced["evaluation_coverage"] == 0.0
    assert reduced["paired_completion_effect"]["estimate"] is None
    assert reduced["paired_token_cost_effect"]["estimate"] is None
    assert all(row["semantic_quality"] is None for row in reduced["rows"])


def test_paired_intervals_use_all_units_and_keep_cohorts_separate() -> None:
    """SCENARIO-REPORT-7443-PAIRS computes only observed paired effects."""

    rows = []
    for index in range(audit.EVALUATION_UNITS):
        for arm in audit.ARMS:
            row = audit.decode_raw_outcome(
                _row(
                    arm=arm,
                    call_id=f"evaluation-{index:02d}-{arm}",
                    phase="evaluation",
                    unit_id=f"evaluation-{index:02d}",
                )
            )
            row["output_tokens"] = 5 if arm == "span" else 7
            rows.append(row)
    intervals = audit.paired_effects(rows, seed=11, draws=30)
    assert intervals["paired_completion_effect"]["pairs"] == audit.EVALUATION_UNITS
    assert intervals["paired_completion_effect"]["estimate"] == 0.0
    assert intervals["paired_token_cost_effect"]["estimate"] == -2.0
    assert intervals["cohort"] == "sealed_evaluation_panel"


def test_constructed_qualifiers_are_exact_and_not_real_semantic_gold() -> None:
    """SCENARIO-REPORT-7443-SEMANTICS audits only constructed exact markers."""

    controls = []
    for pair_id, (family, (proposition, marker)) in enumerate(
        audit.CONSTRUCTED_QUALIFIERS.items(), 1
    ):
        for arm in audit.ARMS:
            controls.append(
                {
                    "pair_id": f"pair-{pair_id:02d}",
                    "family": family,
                    "arm": arm,
                    "claims": [proposition],
                    "claim_spans": [[0, len(proposition)]],
                    "scope": "constructed_exact_check",
                }
            )
    rows, errors = audit.audit_constructed_controls(controls)
    assert errors == []
    assert len(rows) == len(audit.CONSTRUCTED_QUALIFIERS)
    assert all(row["authority"] == "constructed_exact_string" for row in rows)
    assert all(row["required_marker_retained"] is True for row in rows)
    assert {row["family"] for row in rows} >= {
        "negation",
        "time",
        "unit",
        "comparison",
        "condition",
    }

    corrupted = deepcopy(controls)
    corrupted[0]["claims"] = ["The trial met its endpoint."]
    _rows, errors = audit.audit_constructed_controls(corrupted)
    assert "constructed_marker_missing:pair-01:span" in errors


def test_capture_integrity_rejects_canary_counts_and_event_scope() -> None:
    """SCENARIO-REPORT-7443-MUTATIONS rejects gate, count, and event-tag drift."""

    fixture = _valid_integrity_fixture()
    assert audit.capture_integrity_errors(fixture) == []

    closed = deepcopy(fixture)
    closed["canary_open"] = False
    assert "canary_open_state_mismatch" in audit.capture_integrity_errors(closed)

    missing = deepcopy(fixture)
    missing["evaluation_rows"].pop()
    assert "evaluation_per_arm_count_mismatch" in audit.capture_integrity_errors(missing)

    retagged = deepcopy(fixture)
    retagged["producer_event_receipt_class"] = "current_audit_model_events"
    assert "producer_event_class_invalid" in audit.capture_integrity_errors(retagged)

    historical = deepcopy(fixture)
    historical["producer_events"][0]["scope"] = "historical"
    assert "producer_original_scope_changed" in audit.capture_integrity_errors(historical)


def test_all_registered_mutation_controls_are_rejected() -> None:
    """SCENARIO-REPORT-7443-MUTATIONS plants each required corruption."""

    rows = audit.run_mutation_controls()
    assert {row["attack"] for row in rows} == set(audit.REQUIRED_MUTATIONS)
    assert all(row["passed"] is True and row["observed_errors"] for row in rows)


def test_fixture_artifact_has_zero_current_calls_and_validates() -> None:
    """SCENARIO-REPORT-7443-ARTIFACT preserves producer calls as history."""

    artifact = audit.build_artifact_for_test()
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == audit.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["audited_cohort"] == "development_only"
    assert artifact["extraction_audit_complete_score"] == 1
    assert artifact["promotion_score"] == 0
    assert audit.validate_artifact(artifact, verify_source_bytes=False) == []

    changed = deepcopy(artifact)
    changed["model_invoked"] = True
    assert "current_model_declaration_invalid" in audit.validate_artifact(
        changed, verify_source_bytes=False
    )
    changed = deepcopy(artifact)
    changed["rows"][0]["disposition"] = "complete"
    assert "independent_rows_mismatch" in audit.validate_artifact(
        changed, verify_source_bytes=False
    )


def test_blocked_artifact_names_missing_producer(tmp_path: Path) -> None:
    """REQ-REPORT-7443 names an absent Exp7442 path without fabrication."""

    checks, _hashes, sources = audit.collect_preconditions(tmp_path)
    producer = next(row for row in checks if row["check"] == "exp7442_artifact_bytes")
    assert producer["observed"] is None
    artifact = audit.build_blocked_artifact(checks, sources)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["path"] == audit.EXP7442_PATH.as_posix()


def test_validation_plan_is_exact_and_private(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7443-ARTIFACT freezes the affected command scope."""

    commands = audit.build_validation_commands(audit.REPO_ROOT, tmp_path)
    assert audit.validate_command_plan(audit.REPO_ROOT, audit.V652_MANIFEST, commands) == []
    assert {command.name for command in commands} == set(
        audit.validation_scope.REQUIRED_CHECK_NAMES
    )
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert "-n" in focused.argv and "--no-cov" in focused.argv
    assert "tests/python" not in focused.argv


def test_main_cold_modes_and_date_guard(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """SCENARIO-REPORT-7443-ARTIFACT exposes bounded fresh-process modes."""

    artifact = audit.build_artifact_for_test()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert audit.main(["--cold-replay", str(path), "--skip-source-bytes"]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == []
    assert audit.main(["--independent-reduce", str(path), "--skip-source-bytes"]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == []
    with pytest.raises(SystemExit, match="--date must be"):
        audit.run_experiment(audit.REPO_ROOT, "20260919", output_path=tmp_path / "x.json")


def test_real_sources_authenticate_and_replay_without_producer_reducers(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7443-RAW authenticates the shipped sidecar set."""

    checks, sources, evidence = audit._audit_sources(audit.REPO_ROOT)
    assert all(row["passed"] is True for row in checks)
    assert evidence["audit_errors"] == []
    assert evidence["capture"]["raw_disposition_counts"] == {
        "complete": 0,
        "truncated": 0,
        "malformed": 0,
        "empty": 1,
        "failed": 0,
        "cancelled": 0,
        "unstarted": 103,
    }
    assert len(evidence["constructed_pair_audit_rows"]) == 12
    assert len(evidence["archived_model_event_sidecars"]) == 6
    assert sources[audit.EXP7442_PATH.as_posix()]["original_verdict_class"] == "disqualified"
    assert sources[audit.EXP7442_PATH.as_posix()]["original_flagged_adversarial"] is True

    receipts = [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
        }
        for name in audit.validation_scope.REQUIRED_CHECK_NAMES
    ]
    current = audit._current_receipt(0, 1_000_000_000, [], audit._receipt_sidecars(evidence))
    artifact = audit._build_artifact(
        evidence,
        checks=checks,
        sources=sources,
        validation_receipts=receipts,
        current_receipt=current,
        started_at_utc="2026-09-20T00:00:00+00:00",
        completed_at_utc="2026-09-20T00:00:01+00:00",
        candidate=True,
    )
    path = tmp_path / "real-candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert audit.validate_artifact(artifact) == []
    assert audit.independent_replay(path) == []

    changed = deepcopy(artifact)
    changed["raw_disposition_counts"] = {}
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    changed_path = tmp_path / "changed-candidate.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    assert "independent_reduction_mismatch:raw_disposition_counts" in audit.independent_replay(
        changed_path
    )


def test_decoder_defensive_shapes_fail_closed() -> None:
    """SCENARIO-REPORT-7443-MUTATIONS rejects malformed transport shapes."""

    row = _row()
    row["raw_response"] = []
    row["raw_response_sha256"] = audit.canonical_hash({})
    assert "raw_reply_response_mismatch" in audit.decode_raw_outcome(row)["errors"]

    row = _row(reply='{"wrong":[]}')
    assert "claims_schema" in audit.decode_raw_outcome(row)["errors"]
    row = _row(reply='{"claims":[[true,4]]}')
    assert "span_shape" in audit.decode_raw_outcome(row)["errors"]
    row = _row(arm="verbatim", reply='{"claims":[4]}')
    assert "verbatim_shape" in audit.decode_raw_outcome(row)["errors"]
    row = _row(arm="verbatim", paragraph="Café", reply='{"claims":["Café","Café"]}')
    assert "verbatim_duplicate" in audit.decode_raw_outcome(row)["errors"]
    row = _row(arm="unknown")
    assert "arm_invalid" in audit.decode_raw_outcome(row)["errors"]

    row = _row()
    row["raw_request"]["max_tokens"] = 8
    row["raw_request_sha256"] = audit.canonical_hash(row["raw_request"])
    assert "request_token_ceiling_mismatch" in audit.decode_raw_outcome(row)["errors"]
    row = _row()
    row["raw_response"]["timings"]["predicted_n"] = 6
    row["raw_response_sha256"] = audit.canonical_hash(row["raw_response"])
    assert "server_token_receipt_mismatch" in audit.decode_raw_outcome(row)["errors"]


def test_constructed_control_shape_errors_are_named() -> None:
    """SCENARIO-REPORT-7443-SEMANTICS fails closed on control drift."""

    controls = audit._fixture_controls()
    controls.append({"scope": "unrelated"})
    missing = [
        row
        for row in controls
        if not (row.get("pair_id") == "qualifier-01-negation" and row.get("arm") == "verbatim")
    ]
    _rows, errors = audit.audit_constructed_controls(missing)
    assert "constructed_arm_count:qualifier-01-negation" in errors
    assert "constructed_pair_count_mismatch" in errors

    unknown = audit._fixture_controls()
    for row in unknown:
        if row["pair_id"] == "qualifier-01-negation":
            row["family"] = "unknown"
    _rows, errors = audit.audit_constructed_controls(unknown)
    assert "constructed_family_unknown:qualifier-01-negation" in errors


def test_cold_validation_rejects_each_top_level_contract_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7443-ARTIFACT validates every ordinary boundary."""

    base = audit.build_artifact_for_test()
    mutations = {
        "required_field_missing:schema": lambda value: value.pop("schema"),
        "artifact_identity_invalid": lambda value: value.update(schema="wrong"),
        "artifact_schedule_invalid": lambda value: value.update(run_date="20260919"),
        "current_invocation_counts_invalid": lambda value: value.update(invocation_counts={}),
        "inference_substrate_class_invalid": lambda value: value.update(
            inference_substrate_class="gpu"
        ),
        "execution_venue_invalid": lambda value: value.update(execution_venue="external"),
        "promotion_score_invalid": lambda value: value.update(promotion_score=1),
        "disposition_counts_mismatch": lambda value: value.update(raw_disposition_counts={}),
        "development_only_coverage_invalid": lambda value: value.update(evaluation_coverage=0.5),
        "development_effect_not_unknown": lambda value: value["paired_completion_effect"].update(
            estimate=0.0
        ),
        "mutation_controls_invalid": lambda value: value.update(audit_mutation_rows=[]),
        "semantic_scope_invalid": lambda value: value.update(semantic_scope={}),
        "audit_complete_score_invalid": lambda value: value.update(
            extraction_audit_complete_score=2
        ),
        "reproducibility_checksum_mismatch": lambda value: value.update(
            reproducibility_checksum="sha256:bad"
        ),
    }
    for expected, mutate in mutations.items():
        changed = deepcopy(base)
        mutate(changed)
        assert expected in audit.validate_artifact(changed, verify_source_bytes=False)

    unreadable = tmp_path / "bad.json"
    unreadable.write_text("not-json", encoding="utf-8")
    assert audit.cold_replay(unreadable) == ["candidate_artifact_unreadable"]
    nonobject = tmp_path / "list.json"
    nonobject.write_text("[]", encoding="utf-8")
    assert audit.cold_replay(nonobject) == ["candidate_artifact_unreadable"]


def test_command_and_progress_helpers_cover_bounded_runtime_edges(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7443-ARTIFACT keeps runtime helpers bounded and explicit."""

    terminal = audit._terminal_commands(tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal][-2:] == [
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    started = audit.time.monotonic()
    audit._progress(started, "test", "heartbeat", units=1)
    assert "phase=test event=heartbeat" in capsys.readouterr().out
    span = audit._span("test", started, started, 1)
    assert span["completed_units"] == 1

    monkeypatch.setattr(audit, "validate_command_plan", lambda *_args: ["bad"])
    with pytest.raises(ValueError, match="validation_plan_invalid"):
        audit.build_validation_commands(audit.REPO_ROOT, tmp_path / "bad-plan")
    with pytest.raises(SystemExit, match="--date is required"):
        audit.main([])


def test_pairing_and_integrity_cover_incomplete_observations() -> None:
    """SCENARIO-REPORT-7443-PAIRS excludes incomplete pair and token evidence."""

    span = audit.decode_raw_outcome(_row(arm="span", unit_id="u1"))
    lone = audit.paired_effects([span], seed=1, draws=2)
    assert lone["paired_completion_effect"]["pairs"] == 0
    verbatim = audit.decode_raw_outcome(_row(arm="verbatim", unit_id="u1"))
    verbatim["attempted"] = False
    unmatched = audit.paired_effects([span, verbatim], seed=1, draws=2)
    assert unmatched["paired_completion_effect"]["pairs"] == 0
    verbatim["attempted"] = True
    verbatim["output_tokens"] = None
    no_tokens = audit.paired_effects([span, verbatim], seed=1, draws=2)
    assert no_tokens["paired_completion_effect"]["pairs"] == 1
    assert no_tokens["paired_token_cost_effect"]["pairs"] == 0

    fixture = _valid_integrity_fixture()
    fixture["development_rows"].pop()
    assert "development_per_arm_count_mismatch" in audit.capture_integrity_errors(fixture)


def test_sidecar_loader_rejects_hash_and_json_defects(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7443-RAW fails closed on sidecar byte or JSON drift."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    sources: dict = {}
    rows, errors = audit._load_sidecar_rows(
        tmp_path,
        [
            {"path": "missing.json", "sha256": "sha256:missing"},
            {"path": "bad.json", "sha256": audit.sha256_file(bad_json)},
        ],
        base=Path("."),
        receipt_class="fixture",
        sources=sources,
    )
    assert rows == []
    assert any(error.startswith("sidecar_hash_mismatch") for error in errors)
    assert any(error.startswith("sidecar_json_invalid") for error in errors)


def test_source_audit_names_missing_and_mismatched_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7443-RAW names every external provenance defect."""

    monkeypatch.setattr(audit, "collect_preconditions", lambda _root: ([], {}, {}))
    _checks, _sources, absent = audit._audit_sources(tmp_path)
    assert absent["audit_errors"] == ["exp7442_missing"]

    monkeypatch.setattr(
        audit,
        "collect_preconditions",
        lambda _root: ([], {}, {"exp7437": {}, "exp7442": {}}),
    )
    # An empty object is unavailable by contract.
    assert audit._audit_sources(tmp_path)[2]["available"] is False

    native = _row()
    native_path = tmp_path / "native.json"
    native_path.write_text(json.dumps(native), encoding="utf-8")
    response = deepcopy(native)
    response["call_id"] = "changed"
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    control_path = tmp_path / "controls.json"
    control_path.write_text(json.dumps({"payload": []}), encoding="utf-8")
    capture = {
        "source_artifact_hashes": {
            "current_native_call": {
                "path": "native.json",
                "sha256": audit.sha256_file(native_path),
            },
            "current_response_shards": [
                {"path": "response.json", "sha256": audit.sha256_file(response_path)}
            ],
            "current_event_shards": [],
            "current_evaluation_shards": [],
        },
        "development_rows": [{**native, "raw_reply_sha256": "sha256:changed"}],
        "extraction_rows": [{}],
        "development_gate": {"capture_open": False},
        "runner_receipt": {"decoding": {"max_new_tokens": 8}},
        "status": "failed",
        "verdict_class": "disqualified",
        "flagged_adversarial": True,
    }
    protocol = {
        "receipt_sidecars": [
            {
                "path": "controls.json",
                "sha256": audit.sha256_file(control_path),
                "scope": "simulated_transport_events",
            }
        ]
    }
    monkeypatch.setattr(
        audit,
        "collect_preconditions",
        lambda _root: ([], {}, {"exp7437": protocol, "exp7442": capture}),
    )
    monkeypatch.setattr(audit, "EXP7442_RAW", Path("."))
    errors = audit._audit_sources(tmp_path)[2]["audit_errors"]
    assert "native_response_sidecar_mismatch" in errors
    assert "native_development_mismatch:raw_reply_sha256" in errors
    assert "evaluation_sidecar_count_mismatch" in errors
    assert "runner_request_token_ceiling_mismatch" in errors

    capture["source_artifact_hashes"]["current_native_call"]["sha256"] = "sha256:bad"
    protocol["receipt_sidecars"][0]["sha256"] = "sha256:bad"
    errors = audit._audit_sources(tmp_path)[2]["audit_errors"]
    assert "native_call_hash_mismatch" in errors
    assert "constructed_control_hash_mismatch" in errors

    capture["source_artifact_hashes"].pop("current_native_call")
    capture["source_artifact_hashes"]["current_response_shards"] = []
    protocol["receipt_sidecars"] = []
    errors = audit._audit_sources(tmp_path)[2]["audit_errors"]
    assert "native_call_reference_missing" in errors
    assert "response_sidecar_count_mismatch" in errors
    assert "constructed_control_reference_missing" in errors

    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(audit.build_artifact_for_test()), encoding="utf-8")
    monkeypatch.setattr(
        audit,
        "_audit_sources",
        lambda _root: ([], {}, {"available": False, "audit_errors": ["missing"]}),
    )
    assert audit.independent_replay(fixture_path, root=tmp_path) == []


def test_artifact_builder_and_source_validator_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7443-ARTIFACT covers invalid audit and source branches."""

    base = audit.build_artifact_for_test()
    current = {
        key: deepcopy(base[key])
        for key in (
            "MODEL_SPECS",
            "model_invoked",
            "invocation_counts",
            "current_invocation_events",
            "current_run_id",
            "current_owner_pid",
            "event_count",
            "event_sha256",
            "inference_substrate",
            "inference_substrate_details",
            "inference_substrate_class",
            "execution_venue",
            "started_monotonic_ns",
            "ended_monotonic_ns",
            "duration_s",
            "phase_spans",
            "receipt_sidecars",
            "small_ebm_training",
        )
    }
    receipts = base["validation_receipts"]
    invalid = audit._build_artifact(
        {"available": True, "audit_errors": ["planted"], "producer": {}},
        checks=[],
        sources={},
        validation_receipts=receipts,
        current_receipt=current,
        started_at_utc=base["started_at_utc"],
        completed_at_utc=base["completed_at_utc"],
        candidate=True,
        fixture=True,
    )
    assert invalid["verdict_class"] == "disqualified"
    fallback = audit._build_artifact(
        {"available": True, "audit_errors": [], "producer": {}},
        checks=[],
        sources={},
        validation_receipts=receipts,
        current_receipt=current,
        started_at_utc=base["started_at_utc"],
        completed_at_utc=base["completed_at_utc"],
        candidate=True,
        fixture=True,
    )
    assert fallback["gate_check_summary"]["check"] == "sealed_evaluation_observed"

    bad_sources = deepcopy(base)
    bad_sources["source_artifact_hashes"] = {
        "shape": "wrong",
        "hash": {"path": "missing", "sha256": "sha256:missing"},
    }
    bad_sources["reproducibility_checksum"] = audit.reproducibility_checksum(bad_sources)
    errors = audit.validate_artifact(bad_sources, root=tmp_path)
    assert "source_reference_invalid:shape" in errors
    assert "source_hash_mismatch:hash" in errors

    nonfixture = deepcopy(base)
    nonfixture["fixture_artifact"] = False
    nonfixture["validation_receipts"] = []
    nonfixture["reproducibility_checksum"] = audit.reproducibility_checksum(nonfixture)
    assert "required_validation_failed" in audit.validate_artifact(
        nonfixture, verify_source_bytes=False
    )

    sealed = deepcopy(base)
    sealed["audited_cohort"] = "sealed_evaluation_panel"
    sealed["reproducibility_checksum"] = audit.reproducibility_checksum(sealed)
    assert "development_only_coverage_invalid" not in audit.validate_artifact(
        sealed, verify_source_bytes=False
    )
