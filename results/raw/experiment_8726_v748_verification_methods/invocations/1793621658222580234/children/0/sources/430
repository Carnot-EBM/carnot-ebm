"""Exp 1495 CPU-only interwhen-style monitor replay.

Spec: REQ-VERIFY-1495, SCENARIO-VERIFY-1495.

The prototype replays already-recorded CCTU certificate rows instead of
generating fresh model outputs.  Each row becomes a synthetic polling state:
the monitor extracts the certificate-shaped state available at that offset,
runs the deterministic Exp 1494 safe-DSL validator when available, and
interrupts only when those checks expose an error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.eval import constrainprompt_validator_compiler_audit as compiler

JsonDict = dict[str, Any]

RUN_DATE = "20260507"
POLLING_INTERVAL_TOKENS = 64
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1495_interwhen_monitor_prototype.json")
DEFAULT_EVENT_MANIFEST_PATH = Path("results/interwhen_monitor_events_1495.jsonl")
DEFAULT_CERTIFICATE_ARTIFACT_PATH = Path(
    "results/experiment_1493_trigger_token_certificate_export_v1.json"
)
DEFAULT_CERTIFICATE_MANIFEST_PATH = Path("results/cctu_trigger_certificates_1493.jsonl")
DEFAULT_VALIDATOR_ARTIFACT_PATH = Path(
    "results/experiment_1494_constrainprompt_validator_compiler_audit.json"
)
DEFAULT_VALIDATOR_MANIFEST_PATH = Path("results/constrainprompt_validator_manifest_1494.jsonl")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "monitor_intervention_ready",
    "gated_inputs_present",
    "traces_replayed",
    "polling_interval_tokens",
    "monitor_events_emitted",
    "errors_detected",
    "interruptions_triggered",
    "false_interruptions",
    "verifier_false_accept_rate",
    "monitor_event_manifest_path",
    "blockers",
    "honest_verdict",
)

REQUIRED_EVENT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "event_id",
    "trace_id",
    "case_id",
    "lane",
    "family",
    "poll_index",
    "token_offset",
    "polling_interval_tokens",
    "trigger_token_present",
    "certificate_present",
    "parser_passed",
    "certificate_validation_passed",
    "compiled_validator_available",
    "compiled_validator_accepted",
    "verifier_accepted",
    "verifier_false_accept",
    "error_detected",
    "interruption_triggered",
    "false_interruption",
    "monitor_action",
    "error_reasons",
)


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    event_manifest_path: Path | str = DEFAULT_EVENT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable bootstrap artifact required before loading gates."""

    payload: JsonDict = {
        "status": "in_progress",
        "run_date": run_date,
        "monitor_intervention_ready": False,
        "gated_inputs_present": False,
        "traces_replayed": 0,
        "polling_interval_tokens": 0,
        "monitor_events_emitted": 0,
        "errors_detected": 0,
        "interruptions_triggered": 0,
        "false_interruptions": 0,
        "verifier_false_accept_rate": 0.0,
        "monitor_event_manifest_path": _display_path(event_manifest_path),
        "blockers": [],
        "honest_verdict": "in_progress: bootstrap artifact written before gate loading",
    }
    _write_json(Path(output_path), payload)
    return payload


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    event_manifest_path: Path | str = DEFAULT_EVENT_MANIFEST_PATH,
    certificate_artifact_path: Path | str = DEFAULT_CERTIFICATE_ARTIFACT_PATH,
    certificate_manifest_path: Path | str = DEFAULT_CERTIFICATE_MANIFEST_PATH,
    validator_artifact_path: Path | str = DEFAULT_VALIDATOR_ARTIFACT_PATH,
    validator_manifest_path: Path | str = DEFAULT_VALIDATOR_MANIFEST_PATH,
    polling_interval_tokens: int = POLLING_INTERVAL_TOKENS,
    run_date: str = RUN_DATE,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Replay certificate states, emit monitor events, and write the artifact."""

    output = Path(output_path)
    event_manifest = Path(event_manifest_path)
    write_in_progress_artifact(output, event_manifest_path=event_manifest, run_date=run_date)

    blockers = gated_input_blockers(
        certificate_artifact_path=certificate_artifact_path,
        certificate_manifest_path=certificate_manifest_path,
        validator_artifact_path=validator_artifact_path,
        validator_manifest_path=validator_manifest_path,
    )
    if blockers:
        _write_jsonl(event_manifest, [])
        artifact = _terminal_artifact(
            run_date=run_date,
            status="blocked",
            ready=False,
            gated_inputs_present=False,
            traces_replayed=0,
            polling_interval_tokens=polling_interval_tokens,
            events=[],
            event_manifest_path=event_manifest,
            blockers=blockers,
            tests_run=tests_run,
        )
        _write_json(output, artifact)
        return artifact

    certificate_rows = _load_jsonl(Path(certificate_manifest_path))
    validator_rows = load_validator_rows(Path(validator_manifest_path))
    states = build_replay_states(
        certificate_rows,
        validator_rows,
        polling_interval_tokens=polling_interval_tokens,
    )
    events = [monitor_event_for_state(state) for state in states]
    _write_jsonl(event_manifest, events)

    metrics = aggregate_monitor_metrics(events)
    ready = bool(
        event_manifest.exists()
        and metrics["errors_detected"] > 0
        and metrics["false_interruptions"] == 0
        and metrics["verifier_false_accept_rate"] == 0.0
    )
    artifact = _terminal_artifact(
        run_date=run_date,
        status="complete",
        ready=ready,
        gated_inputs_present=True,
        traces_replayed=len(states),
        polling_interval_tokens=polling_interval_tokens,
        events=events,
        event_manifest_path=event_manifest,
        blockers=[],
        tests_run=tests_run,
    )
    _write_json(output, artifact)
    return artifact


def gated_input_blockers(
    *,
    certificate_artifact_path: Path | str = DEFAULT_CERTIFICATE_ARTIFACT_PATH,
    certificate_manifest_path: Path | str = DEFAULT_CERTIFICATE_MANIFEST_PATH,
    validator_artifact_path: Path | str = DEFAULT_VALIDATOR_ARTIFACT_PATH,
    validator_manifest_path: Path | str = DEFAULT_VALIDATOR_MANIFEST_PATH,
) -> list[str]:
    """Return concrete blockers for missing or not-ready upstream gates."""

    blockers: list[str] = []
    cert_artifact = _load_json_if_exists(Path(certificate_artifact_path))
    validator_artifact = _load_json_if_exists(Path(validator_artifact_path))

    if cert_artifact is None:
        blockers.append("missing_certificate_artifact")
    elif cert_artifact.get("status") != "complete" or cert_artifact.get(
        "trigger_certificate_ready"
    ) is not True:
        blockers.append("certificate_gate_not_ready")

    if validator_artifact is None:
        blockers.append("missing_validator_artifact")
    elif validator_artifact.get("status") != "complete" or validator_artifact.get(
        "validator_compiler_ready"
    ) is not True:
        blockers.append("validator_compiler_gate_not_ready")

    if cert_artifact is not None and not Path(certificate_manifest_path).exists():
        blockers.append("missing_certificate_manifest")
    if validator_artifact is not None and not Path(validator_manifest_path).exists():
        blockers.append("missing_validator_manifest")
    return blockers


def load_validator_rows(path: Path | str) -> dict[str, JsonDict]:
    """Load compiled validator rows keyed by CCTU prompt/case ID."""

    rows = _load_jsonl(Path(path))
    return {
        str(row["prompt_id"]): row
        for row in rows
        if row.get("prompt_id") and row.get("validator_compiled") is True
    }


def build_replay_states(
    certificate_rows: list[JsonDict],
    validator_rows: dict[str, JsonDict],
    *,
    polling_interval_tokens: int = POLLING_INTERVAL_TOKENS,
) -> list[JsonDict]:
    """Convert recorded certificate rows into deterministic polling states."""

    states: list[JsonDict] = []
    for index, row in enumerate(certificate_rows, start=1):
        case_id = str(row.get("case_id") or f"unknown-{index}")
        validator_row = validator_rows.get(case_id)
        candidate_output = _candidate_output(row)
        compiled_result = _run_compiled_validator(validator_row, candidate_output)
        parser_result = row.get("parser_result") if isinstance(row.get("parser_result"), dict) else {}
        verifier_result = (
            row.get("verifier_result") if isinstance(row.get("verifier_result"), dict) else {}
        )
        parser_passed = bool(
            parser_result.get("parsed") is True and isinstance(row.get("certificate_json"), dict)
        )
        certificate_validation_passed = bool(
            row.get("deterministic_validation_passed")
            or verifier_result.get("accepted") is True
        )
        compiled_available = validator_row is not None
        compiled_accepted = bool(compiled_result.get("accepted"))
        verifier_accepted = bool(verifier_result.get("accepted"))
        verifier_false_accept = bool(
            row.get("false_accept_status") or verifier_result.get("false_accept")
        )
        recorded_error = not (
            parser_passed
            and certificate_validation_passed
            and compiled_accepted
            and verifier_accepted
            and not verifier_false_accept
        )
        states.append(
            {
                "trace_id": f"{case_id}:{row.get('lane', 'unknown')}:{index}",
                "case_id": case_id,
                "lane": str(row.get("lane") or "unknown"),
                "family": str(row.get("family") or "unknown"),
                "poll_index": index,
                "token_offset": index * polling_interval_tokens,
                "polling_interval_tokens": polling_interval_tokens,
                "trigger_token_present": bool(row.get("trigger_token_present")),
                "certificate_present": isinstance(row.get("certificate_json"), dict),
                "parser_passed": parser_passed,
                "certificate_validation_passed": certificate_validation_passed,
                "compiled_validator_available": compiled_available,
                "compiled_validator_accepted": compiled_accepted,
                "compiled_validator_result": compiled_result,
                "verifier_accepted": verifier_accepted,
                "verifier_false_accept": verifier_false_accept,
                "recorded_error": recorded_error,
                "candidate_output": candidate_output,
                "reasoning_token_count": _token_count(str(row.get("free_form_reasoning_text") or "")),
            }
        )
    return states


def monitor_event_for_state(state: JsonDict) -> JsonDict:
    """Emit the explicit monitor decision for one synthetic poll."""

    error_detected = bool(state["recorded_error"])
    interrupt = error_detected
    event = {
        key: state[key]
        for key in (
            "trace_id",
            "case_id",
            "lane",
            "family",
            "poll_index",
            "token_offset",
            "polling_interval_tokens",
            "trigger_token_present",
            "certificate_present",
            "parser_passed",
            "certificate_validation_passed",
            "compiled_validator_available",
            "compiled_validator_accepted",
            "verifier_accepted",
            "verifier_false_accept",
        )
    }
    event.update(
        {
            "schema_version": 1,
            "event_id": f"interwhen-1495-{int(state['poll_index']):04d}",
            "reasoning_token_count": int(state["reasoning_token_count"]),
            "recorded_error": error_detected,
            "error_detected": error_detected,
            "interruption_triggered": interrupt,
            "false_interruption": bool(interrupt and not error_detected),
            "monitor_action": "interrupt" if interrupt else "continue",
            "error_reasons": _error_reasons(state),
        }
    )
    return event


def aggregate_monitor_metrics(events: list[JsonDict]) -> JsonDict:
    """Compute intervention metrics from emitted monitor events."""

    invalid_events = [event for event in events if event.get("recorded_error") is True]
    false_accepts = sum(bool(event.get("verifier_false_accept")) for event in invalid_events)
    return {
        "monitor_events_emitted": len(events),
        "errors_detected": sum(bool(event["error_detected"]) for event in events),
        "interruptions_triggered": sum(bool(event["interruption_triggered"]) for event in events),
        "false_interruptions": sum(bool(event["false_interruption"]) for event in events),
        "verifier_false_accept_rate": (
            round(false_accepts / len(invalid_events), 6) if invalid_events else 0.0
        ),
    }


def _run_compiled_validator(
    validator_row: JsonDict | None,
    candidate_output: str,
) -> JsonDict:
    if not validator_row:
        return {"accepted": False, "reason": "compiled_validator_missing"}
    compiled = compiler.CompiledValidator(
        prompt_id=str(validator_row.get("prompt_id") or ""),
        compiled=True,
        dsl=dict(validator_row.get("compiled_validator") or {}),
        manual_review_required=bool(validator_row.get("manual_review_required")),
    )
    result = compiler.evaluate_compiled_validator(compiled, candidate_output)
    return dict(result)


def _candidate_output(row: JsonDict) -> str:
    certificate = row.get("certificate_json")
    if isinstance(certificate, dict):
        return json.dumps(certificate, sort_keys=True)
    return str(row.get("model_output") or "")


def _error_reasons(state: JsonDict) -> list[str]:
    reasons: list[str] = []
    if not state["parser_passed"]:
        reasons.append("parser_failed")
    if not state["certificate_validation_passed"]:
        reasons.append("certificate_validator_failed")
    if not state["compiled_validator_available"]:
        reasons.append("compiled_validator_missing")
    elif not state["compiled_validator_accepted"]:
        reasons.append("compiled_validator_rejected")
    if not state["verifier_accepted"]:
        reasons.append("verifier_rejected")
    if state["verifier_false_accept"]:
        reasons.append("verifier_false_accept")
    return reasons


def _terminal_artifact(
    *,
    run_date: str,
    status: str,
    ready: bool,
    gated_inputs_present: bool,
    traces_replayed: int,
    polling_interval_tokens: int,
    events: list[JsonDict],
    event_manifest_path: Path,
    blockers: list[str],
    tests_run: list[str] | None,
) -> JsonDict:
    metrics = aggregate_monitor_metrics(events)
    return {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "monitor_intervention_ready": ready,
        "gated_inputs_present": gated_inputs_present,
        "traces_replayed": traces_replayed,
        "polling_interval_tokens": polling_interval_tokens,
        "monitor_events_emitted": metrics["monitor_events_emitted"],
        "errors_detected": metrics["errors_detected"],
        "interruptions_triggered": metrics["interruptions_triggered"],
        "false_interruptions": metrics["false_interruptions"],
        "verifier_false_accept_rate": metrics["verifier_false_accept_rate"],
        "monitor_event_manifest_path": _display_path(event_manifest_path),
        "blockers": list(blockers),
        "tests_run": list(tests_run or []),
        "honest_verdict": (
            "complete: CPU-only interwhen monitor replay ready on recorded CCTU errors"
            if ready
            else "complete: CPU-only interwhen monitor replay blocked or not intervention-ready"
        ),
    }


def _load_json_if_exists(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _display_path(path: Path | str) -> str:
    return str(path)


def _token_count(text: str) -> int:
    return len(text.split())
