"""Tests for Exp 1495 CPU-only interwhen monitor replay.

Spec: REQ-VERIFY-1495, SCENARIO-VERIFY-1495.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.eval import cctu_executable_constraint_microbenchmark as cctu
from carnot.eval import constrainprompt_validator_compiler_audit as compiler
from carnot.eval import interwhen_monitor_prototype as exp


def test_req_verify_1495_missing_gated_inputs_blocks_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1495: missing upstream readiness writes a gated artifact."""

    artifact = exp.run_experiment(
        output_path=tmp_path / "experiment_1495.json",
        event_manifest_path=tmp_path / "events.jsonl",
        certificate_artifact_path=tmp_path / "missing_1493.json",
        certificate_manifest_path=tmp_path / "missing_1493.jsonl",
        validator_artifact_path=tmp_path / "missing_1494.json",
        validator_manifest_path=tmp_path / "missing_1494.jsonl",
    )

    assert artifact["status"] == "blocked"
    assert artifact["gated_inputs_present"] is False
    assert artifact["monitor_intervention_ready"] is False
    assert artifact["monitor_events_emitted"] == 0
    assert "missing_certificate_artifact" in artifact["blockers"]
    assert "missing_validator_artifact" in artifact["blockers"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1495_replay_states_use_synthetic_token_offsets() -> None:
    """REQ-VERIFY-1495: certificate and validator rows become pollable states."""

    cert_rows = [_certificate_row(valid=True), _certificate_row(valid=False)]
    validator_rows = {cert_rows[0]["case_id"]: _validator_row()}

    states = exp.build_replay_states(
        cert_rows,
        validator_rows,
        polling_interval_tokens=32,
    )

    assert len(states) == 2
    assert [state["token_offset"] for state in states] == [32, 64]
    assert states[0]["recorded_error"] is False
    assert states[1]["recorded_error"] is True
    assert states[0]["compiled_validator_available"] is True
    assert states[0]["candidate_output"].startswith("{")


def test_scenario_verify_1495_monitor_event_schema_marks_interrupts_only_on_errors() -> None:
    """SCENARIO-VERIFY-1495: each poll event reports checks and interrupt status."""

    states = exp.build_replay_states(
        [_certificate_row(valid=True), _certificate_row(valid=False)],
        {"cctu-1486-arith-001": _validator_row()},
        polling_interval_tokens=32,
    )
    events = [exp.monitor_event_for_state(state) for state in states]

    assert set(exp.REQUIRED_EVENT_FIELDS) <= set(events[0])
    assert events[0]["error_detected"] is False
    assert events[0]["interruption_triggered"] is False
    assert events[0]["false_interruption"] is False
    assert events[1]["error_detected"] is True
    assert events[1]["interruption_triggered"] is True
    assert events[1]["false_interruption"] is False
    assert "compiled_validator_rejected" in events[1]["error_reasons"]


def test_req_verify_1495_gate_and_replay_edge_cases_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-1495: not-ready gates and missing validators fail closed."""

    certificate_artifact = tmp_path / "experiment_1493.json"
    validator_artifact = tmp_path / "experiment_1494.json"
    _write_json(certificate_artifact, {"status": "in_progress", "trigger_certificate_ready": False})
    _write_json(validator_artifact, {"status": "complete", "validator_compiler_ready": False})

    blockers = exp.gated_input_blockers(
        certificate_artifact_path=certificate_artifact,
        certificate_manifest_path=tmp_path / "missing_certificates.jsonl",
        validator_artifact_path=validator_artifact,
        validator_manifest_path=tmp_path / "missing_validators.jsonl",
    )

    assert "certificate_gate_not_ready" in blockers
    assert "validator_compiler_gate_not_ready" in blockers
    assert "missing_certificate_manifest" in blockers
    assert "missing_validator_manifest" in blockers

    row = _certificate_row(valid=True)
    row["certificate_json"] = None
    row["model_output"] = "not json"
    row["parser_result"] = {"parsed": False, "parse_error": "no_json_object_after_trigger"}
    row["false_accept_status"] = True
    row["verifier_result"]["false_accept"] = True

    state = exp.build_replay_states([row], {}, polling_interval_tokens=16)[0]
    event = exp.monitor_event_for_state(state)

    assert state["candidate_output"] == "not json"
    assert state["compiled_validator_available"] is False
    assert event["error_detected"] is True
    assert event["interruption_triggered"] is True
    assert "parser_failed" in event["error_reasons"]
    assert "compiled_validator_missing" in event["error_reasons"]
    assert "verifier_false_accept" in event["error_reasons"]


def test_scenario_verify_1495_runner_writes_ready_artifact_and_event_manifest(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1495: runner writes one event per poll and gates readiness."""

    certificate_artifact = tmp_path / "experiment_1493.json"
    certificate_manifest = tmp_path / "certificates_1493.jsonl"
    validator_artifact = tmp_path / "experiment_1494.json"
    validator_manifest = tmp_path / "validators_1494.jsonl"
    output_path = tmp_path / "experiment_1495.json"
    event_manifest = tmp_path / "events_1495.jsonl"

    _write_json(
        certificate_artifact,
        {
            "status": "complete",
            "trigger_certificate_ready": True,
            "certificate_manifest_path": str(certificate_manifest),
        },
    )
    _write_jsonl(certificate_manifest, [_certificate_row(valid=True), _certificate_row(valid=False)])
    _write_json(
        validator_artifact,
        {
            "status": "complete",
            "validator_compiler_ready": True,
            "validator_manifest_path": str(validator_manifest),
        },
    )
    _write_jsonl(validator_manifest, [_validator_row()])

    artifact = exp.run_experiment(
        output_path=output_path,
        event_manifest_path=event_manifest,
        certificate_artifact_path=certificate_artifact,
        certificate_manifest_path=certificate_manifest,
        validator_artifact_path=validator_artifact,
        validator_manifest_path=validator_manifest,
        polling_interval_tokens=32,
        tests_run=["focused pytest"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    event_rows = [
        json.loads(line) for line in event_manifest.read_text(encoding="utf-8").splitlines()
    ]

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["gated_inputs_present"] is True
    assert artifact["monitor_intervention_ready"] is True
    assert artifact["traces_replayed"] == 2
    assert artifact["polling_interval_tokens"] == 32
    assert artifact["monitor_events_emitted"] == 2
    assert artifact["errors_detected"] == 1
    assert artifact["interruptions_triggered"] == 1
    assert artifact["false_interruptions"] == 0
    assert artifact["verifier_false_accept_rate"] == 0.0
    assert artifact["monitor_event_manifest_path"] == str(event_manifest)
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(event_rows) == 2
    assert [row["monitor_action"] for row in event_rows] == ["continue", "interrupt"]


def _certificate_row(*, valid: bool) -> dict[str, Any]:
    case = cctu.build_benchmark_cases()[0]
    certificate = json.loads(cctu.compliant_transcript_for_case(case))
    if not valid:
        certificate["final_answer"] = "not 45"
        certificate["verifier"] = {"accept": True}
    validation = cctu.validate_transcript(case, json.dumps(certificate, sort_keys=True))
    return {
        "case_id": case.case_id,
        "family": case.family,
        "lane": "trigger_certificate",
        "trigger_token_present": True,
        "free_form_reasoning_text": "12 + 18 + 15 = 45",
        "certificate_json": certificate,
        "parser_result": {
            "parsed": True,
            "parse_error": None,
            "trigger_token_present": True,
        },
        "validator_result": validation["validator_result"],
        "verifier_result": validation["verifier_result"],
        "deterministic_validation_passed": bool(validation["verifier_result"]["accepted"]),
        "false_accept_status": bool(validation["verifier_result"]["false_accept"]),
    }


def _validator_row() -> dict[str, Any]:
    prompt = next(
        prompt
        for prompt in compiler.build_prompt_set()
        if prompt.prompt_id == "cctu-1486-arith-001"
    )
    compiled = compiler.compile_prompt(prompt)
    return {
        "prompt_id": prompt.prompt_id,
        "validator_compiled": True,
        "manual_review_required": False,
        "compiled_validator": compiled.dsl,
        "known_good_passed": True,
        "known_bad_rejected": True,
        "false_accept": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
