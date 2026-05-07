"""Tests for Exp 1509 executable monitor runtime adapter.

Spec: REQ-VERIFY-1509, SCENARIO-VERIFY-1509.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import executable_monitor_runtime_adapter as exp


def test_req_verify_1509_missing_1507_or_1508_gate_blocks_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1509: absent verifier and certificate gates block runtime readiness."""

    output_path = tmp_path / "experiment_1509.json"
    manifest_path = tmp_path / "runtime_events.jsonl"

    artifact = exp.run_experiment(
        output_path=output_path,
        monitor_event_manifest_path=tmp_path / "missing_monitor.jsonl",
        safe_prefix_manifest_path=tmp_path / "missing_safe_prefix.jsonl",
        verifier_manifest_path=tmp_path / "missing_verifier.jsonl",
        certificate_manifest_path=tmp_path / "missing_certificate.jsonl",
        exp1507_artifact_path=tmp_path / "missing_1507.json",
        exp1508_artifact_path=tmp_path / "missing_1508.json",
        output_manifest_path=manifest_path,
    )

    assert artifact["status"] == "blocked"
    assert artifact["monitor_runtime_ready"] is False
    assert artifact["gated_inputs_present"] is False
    assert artifact["events_loaded"] == 0
    assert artifact["events_normalized"] == 0
    assert f"missing_exp1507_artifact:{tmp_path / 'missing_1507.json'}" in artifact["blockers"]
    assert f"missing_exp1508_artifact:{tmp_path / 'missing_1508.json'}" in artifact["blockers"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert manifest_path.read_text(encoding="utf-8") == ""

    not_ready_1507 = tmp_path / "not_ready_1507.json"
    not_ready_1508 = tmp_path / "not_ready_1508.json"
    manifests = [tmp_path / f"source_{index}.jsonl" for index in range(4)]
    _write_json(not_ready_1507, {"status": "complete", "verifier_induction_ready": False})
    _write_json(not_ready_1508, {"status": "complete", "certificate_decoder_ready": False})
    for manifest in manifests:
        _write_jsonl(manifest, [])

    assert exp.gated_input_blockers(
        monitor_event_manifest_path=manifests[0],
        safe_prefix_manifest_path=manifests[1],
        verifier_manifest_path=manifests[2],
        certificate_manifest_path=manifests[3],
        exp1507_artifact_path=not_ready_1507,
        exp1508_artifact_path=not_ready_1508,
    ) == [
        f"exp1507_not_ready:{not_ready_1507}",
        f"exp1508_not_ready:{not_ready_1508}",
    ]


def test_req_verify_1509_normalized_schema_validation_requires_provenance() -> None:
    """REQ-VERIFY-1509: normalized events carry schema and source provenance."""

    events = exp.normalize_source_rows(
        monitor_rows=[_monitor_event("m-1", "case-a", 64)],
        safe_prefix_rows=[],
        verifier_rows=[],
        certificate_rows=[],
        source_paths={
            "monitor": Path("results/interwhen_monitor_events_1495.jsonl"),
            "safe_prefix": Path("results/safe_prefix_continuations_1496.jsonl"),
            "verifier": Path("results/safe_dsl_verifier_induction_1507.jsonl"),
            "certificate": Path("results/trigger_grammar_certificates_1508.jsonl"),
        },
    )

    assert len(events) == 1
    assert exp.validate_normalized_event(events[0]) == []
    assert events[0]["event_schema_version"] == exp.EVENT_SCHEMA_VERSION
    assert events[0]["source_experiment"] == "1495"
    assert events[0]["source_line"] == 1
    assert events[0]["case_id"] == "case-a"

    missing = dict(events[0])
    del missing["source_path"]
    assert "missing:source_path" in exp.validate_normalized_event(missing)

    invalid = dict(events[0])
    invalid["event_schema_version"] = "wrong"
    assert exp.validate_normalized_event(invalid) == ["invalid:event_schema_version"]


def test_scenario_verify_1509_safe_prefix_links_only_recorded_monitor_matches() -> None:
    """SCENARIO-VERIFY-1509: safe-prefix rows link only to recorded monitor events."""

    events = exp.normalize_source_rows(
        monitor_rows=[
            _monitor_event("m-1", "case-a", 64),
            _monitor_event("m-2", "case-b", 128),
            _monitor_event("m-3", "case-d", 256),
        ],
        safe_prefix_rows=[
            _safe_prefix_row("case-a", selected_event_id="m-1", last_safe_token_offset=0),
            _safe_prefix_row(
                "case-b",
                selected_event_id="",
                selected_event_token_offset=128,
            ),
            _safe_prefix_row("case-d", selected_event_id="", last_safe_token_offset=0),
            _safe_prefix_row("case-c", selected_event_id="not-recorded", last_safe_token_offset=64),
        ],
        verifier_rows=[
            {
                "row_type": "selected_set_summary",
                "candidate_names": ["compiled_validator_sanity"],
                "verifier_false_accept_rate": 0.0,
            }
        ],
        certificate_rows=[
            _certificate_row("case-a", decoder_mode="schema_only", passed=False),
            _certificate_row("case-a", decoder_mode="trigger_grammar", passed=True),
        ],
        source_paths={
            "monitor": Path("results/interwhen_monitor_events_1495.jsonl"),
            "safe_prefix": Path("results/safe_prefix_continuations_1496.jsonl"),
            "verifier": Path("results/safe_dsl_verifier_induction_1507.jsonl"),
            "certificate": Path("results/trigger_grammar_certificates_1508.jsonl"),
        },
    )
    replayed = list(exp.replay_events(events))

    assert [event["replay_index"] for event in replayed] == list(range(1, 11))
    assert [event["source_kind"] for event in replayed] == [
        "monitor",
        "monitor",
        "monitor",
        "safe_prefix",
        "safe_prefix",
        "safe_prefix",
        "safe_prefix",
        "verifier",
        "certificate",
        "certificate",
    ]
    linked = [event for event in replayed if event["source_kind"] == "safe_prefix"]
    assert linked[0]["link_status"] == "linked"
    assert linked[0]["linked_monitor_event_id"] == "m-1"
    assert linked[1]["linked_monitor_event_id"] == "m-2"
    assert linked[2]["linked_monitor_event_id"] == "m-3"
    assert linked[3]["link_status"] == "unmatched"
    assert linked[3]["linked_monitor_event_id"] is None


def test_scenario_verify_1509_runner_writes_manifest_and_ready_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1509: runner writes one normalized event per source row."""

    monitor_manifest = tmp_path / "monitor.jsonl"
    safe_prefix_manifest = tmp_path / "safe_prefix.jsonl"
    verifier_manifest = tmp_path / "verifier.jsonl"
    certificate_manifest = tmp_path / "certificates.jsonl"
    output_path = tmp_path / "experiment_1509.json"
    runtime_manifest = tmp_path / "runtime_events.jsonl"
    exp1507_artifact = tmp_path / "experiment_1507.json"
    exp1508_artifact = tmp_path / "experiment_1508.json"

    _write_json(
        exp1507_artifact,
        {
            "status": "complete",
            "verifier_induction_ready": True,
            "verifier_false_accept_rate": 0.0,
        },
    )
    _write_json(
        exp1508_artifact,
        {
            "status": "complete",
            "certificate_decoder_ready": True,
            "verifier_false_accept_rate": 0.0,
        },
    )
    _write_jsonl(monitor_manifest, [_monitor_event("m-1", "case-a", 64)])
    _write_jsonl(safe_prefix_manifest, [_safe_prefix_row("case-a", selected_event_id="m-1")])
    _write_jsonl(
        verifier_manifest,
        [
            {
                "row_type": "selected_set_summary",
                "candidate_names": ["compiled_validator_sanity"],
                "verifier_false_accept_rate": 0.0,
            }
        ],
    )
    _write_jsonl(certificate_manifest, [_certificate_row("case-a")])

    artifact = exp.run_experiment(
        output_path=output_path,
        monitor_event_manifest_path=monitor_manifest,
        safe_prefix_manifest_path=safe_prefix_manifest,
        verifier_manifest_path=verifier_manifest,
        certificate_manifest_path=certificate_manifest,
        exp1507_artifact_path=exp1507_artifact,
        exp1508_artifact_path=exp1508_artifact,
        output_manifest_path=runtime_manifest,
        tests_run=["focused pytest"],
    )
    manifest_rows = [
        json.loads(line) for line in runtime_manifest.read_text(encoding="utf-8").splitlines()
    ]

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["monitor_runtime_ready"] is True
    assert artifact["gated_inputs_present"] is True
    assert artifact["events_loaded"] == 4
    assert artifact["events_normalized"] == 4
    assert artifact["event_schema_version"] == exp.EVENT_SCHEMA_VERSION
    assert artifact["verifier_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["safe_prefix_events_linked"] == 1
    assert artifact["monitor_event_manifest_path"] == str(runtime_manifest)
    assert artifact["adapter_tests_run"] == ["focused pytest"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(manifest_rows) == 4
    assert manifest_rows[1]["linked_monitor_event_id"] == "m-1"


def _monitor_event(event_id: str, case_id: str, token_offset: int) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "trace_id": f"{case_id}:trigger_certificate:1",
        "case_id": case_id,
        "lane": "trigger_certificate",
        "family": "arithmetic",
        "poll_index": token_offset // 64,
        "token_offset": token_offset,
        "polling_interval_tokens": 64,
        "error_detected": token_offset == 64,
        "interruption_triggered": token_offset == 64,
        "false_interruption": False,
        "monitor_action": "interrupt" if token_offset == 64 else "continue",
        "verifier_false_accept": False,
    }


def _safe_prefix_row(
    case_id: str,
    *,
    selected_event_id: str,
    last_safe_token_offset: int = 0,
    selected_event_token_offset: int | None = None,
) -> dict[str, Any]:
    row = {
        "case_id": case_id,
        "family": "arithmetic",
        "mode": "safe_prefix_continuation",
        "selected_event_id": selected_event_id,
        "last_safe_token_offset": last_safe_token_offset,
        "final_validator_passed": True,
        "verifier_false_accept": False,
        "model_hf_id": "local/test-model",
        "generation_source": "test",
    }
    if selected_event_token_offset is not None:
        row["selected_event_token_offset"] = selected_event_token_offset
    return row


def _certificate_row(
    case_id: str,
    *,
    decoder_mode: str = "trigger_grammar",
    passed: bool = True,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "family": "arithmetic",
        "decoder_mode": decoder_mode,
        "grammar_backend": "llama_cpp_gbnf_exact_certificate_v1",
        "deterministic_validation_passed": passed,
        "parser_result": {"parsed": passed},
        "verifier_result": {"accepted": passed, "base_valid": passed, "false_accept": False},
        "false_accept_status": False,
        "model_hf_id": "local/test-model",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
