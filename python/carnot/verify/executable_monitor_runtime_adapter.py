"""Executable monitor runtime adapter for Exp 1509.

Spec: REQ-VERIFY-1509, SCENARIO-VERIFY-1509.

The adapter is intentionally CPU-only: it reads already-recorded monitor,
safe-prefix, verifier-induction, and certificate-decoder rows, converts them
into a small replay schema, and refuses to invent runtime links when the
source artifacts did not record enough provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

RUN_DATE = "20260507"
EVENT_SCHEMA_VERSION = "monitor-runtime-event/v1"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1509_executable_monitor_runtime_adapter.json")
DEFAULT_OUTPUT_MANIFEST_PATH = Path("results/executable_monitor_events_1509.jsonl")
DEFAULT_MONITOR_EVENT_MANIFEST_PATH = Path("results/interwhen_monitor_events_1495.jsonl")
DEFAULT_SAFE_PREFIX_MANIFEST_PATH = Path("results/safe_prefix_continuations_1496.jsonl")
DEFAULT_VERIFIER_MANIFEST_PATH = Path("results/safe_dsl_verifier_induction_1507.jsonl")
DEFAULT_CERTIFICATE_MANIFEST_PATH = Path("results/trigger_grammar_certificates_1508.jsonl")
DEFAULT_EXP1507_ARTIFACT_PATH = Path(
    "results/experiment_1507_autopyverifier_safe_dsl_induction_pack.json"
)
DEFAULT_EXP1508_ARTIFACT_PATH = Path(
    "results/experiment_1508_trigger_grammar_certificate_decoder_audit.json"
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "monitor_runtime_ready",
    "gated_inputs_present",
    "events_loaded",
    "events_normalized",
    "event_schema_version",
    "verifier_false_accept_rate",
    "safe_prefix_events_linked",
    "monitor_event_manifest_path",
    "adapter_tests_run",
    "blockers",
    "honest_verdict",
)

REQUIRED_EVENT_FIELDS: tuple[str, ...] = (
    "event_schema_version",
    "event_id",
    "replay_index",
    "source_experiment",
    "source_kind",
    "source_path",
    "source_line",
    "source_row_id",
    "source_event_id",
    "event_kind",
    "case_id",
    "family",
    "token_offset",
    "validation_status",
    "verifier_false_accept",
    "linked_monitor_event_id",
    "link_status",
    "provenance",
)


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    output_manifest_path: Path | str = DEFAULT_OUTPUT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the bootstrap artifact before any gated source rows are loaded."""

    payload = _terminal_artifact(
        status="in_progress",
        ready=False,
        gated_inputs_present=False,
        events_loaded=0,
        events=[],
        output_manifest_path=Path(output_manifest_path),
        verifier_false_accept_rate=None,
        blockers=[],
        tests_run=[],
        run_date=run_date,
        honest_verdict="complete: in-progress Exp 1509 bootstrap artifact",
    )
    _write_json(Path(output_path), payload)
    return payload


def gated_input_blockers(
    *,
    monitor_event_manifest_path: Path | str = DEFAULT_MONITOR_EVENT_MANIFEST_PATH,
    safe_prefix_manifest_path: Path | str = DEFAULT_SAFE_PREFIX_MANIFEST_PATH,
    verifier_manifest_path: Path | str = DEFAULT_VERIFIER_MANIFEST_PATH,
    certificate_manifest_path: Path | str = DEFAULT_CERTIFICATE_MANIFEST_PATH,
    exp1507_artifact_path: Path | str = DEFAULT_EXP1507_ARTIFACT_PATH,
    exp1508_artifact_path: Path | str = DEFAULT_EXP1508_ARTIFACT_PATH,
) -> list[str]:
    """Return concrete blockers for missing source manifests or not-ready gates."""

    blockers: list[str] = []
    exp1507_path = Path(exp1507_artifact_path)
    exp1508_path = Path(exp1508_artifact_path)
    exp1507 = _load_json_if_exists(exp1507_path)
    exp1508 = _load_json_if_exists(exp1508_path)

    if exp1507 is None:
        blockers.append(f"missing_exp1507_artifact:{exp1507_path}")
    elif exp1507.get("status") != "complete" or exp1507.get("verifier_induction_ready") is not True:
        blockers.append(f"exp1507_not_ready:{exp1507_path}")

    if exp1508 is None:
        blockers.append(f"missing_exp1508_artifact:{exp1508_path}")
    elif (
        exp1508.get("status") != "complete" or exp1508.get("certificate_decoder_ready") is not True
    ):
        blockers.append(f"exp1508_not_ready:{exp1508_path}")

    for label, manifest_path in (
        ("monitor_event", monitor_event_manifest_path),
        ("safe_prefix", safe_prefix_manifest_path),
        ("verifier", verifier_manifest_path),
        ("certificate", certificate_manifest_path),
    ):
        path = Path(manifest_path)
        if not path.exists():
            blockers.append(f"missing_{label}_manifest:{path}")
    return blockers


def normalize_source_rows(
    *,
    monitor_rows: list[JsonDict],
    safe_prefix_rows: list[JsonDict],
    verifier_rows: list[JsonDict],
    certificate_rows: list[JsonDict],
    source_paths: dict[str, Path],
) -> list[JsonDict]:
    """Convert source manifests into deterministic runtime events."""

    monitor_lookup = _build_monitor_lookup(monitor_rows)
    events: list[JsonDict] = []
    for line_number, row in enumerate(monitor_rows, start=1):
        events.append(_normalize_monitor_row(row, source_paths["monitor"], line_number))
    for line_number, row in enumerate(safe_prefix_rows, start=1):
        events.append(
            _normalize_safe_prefix_row(
                row,
                source_paths["safe_prefix"],
                line_number,
                monitor_lookup,
            )
        )
    for line_number, row in enumerate(verifier_rows, start=1):
        events.append(_normalize_verifier_row(row, source_paths["verifier"], line_number))
    for line_number, row in enumerate(certificate_rows, start=1):
        events.append(_normalize_certificate_row(row, source_paths["certificate"], line_number))

    ordered = sorted(events, key=_event_sort_key)
    for replay_index, event in enumerate(ordered, start=1):
        event["replay_index"] = replay_index
        event["event_id"] = f"runtime-1509-{replay_index:06d}"
    return ordered


def validate_normalized_event(event: JsonDict) -> list[str]:
    """Return schema validation errors for one normalized event."""

    errors = [f"missing:{field}" for field in REQUIRED_EVENT_FIELDS if field not in event]
    if not errors and event["event_schema_version"] != EVENT_SCHEMA_VERSION:
        errors.append("invalid:event_schema_version")
    return errors


def replay_events(events: list[JsonDict]) -> list[JsonDict]:
    """Return events in deterministic replay order after schema validation."""

    return sorted(events, key=lambda event: int(event.get("replay_index") or 0))


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    monitor_event_manifest_path: Path | str = DEFAULT_MONITOR_EVENT_MANIFEST_PATH,
    safe_prefix_manifest_path: Path | str = DEFAULT_SAFE_PREFIX_MANIFEST_PATH,
    verifier_manifest_path: Path | str = DEFAULT_VERIFIER_MANIFEST_PATH,
    certificate_manifest_path: Path | str = DEFAULT_CERTIFICATE_MANIFEST_PATH,
    exp1507_artifact_path: Path | str = DEFAULT_EXP1507_ARTIFACT_PATH,
    exp1508_artifact_path: Path | str = DEFAULT_EXP1508_ARTIFACT_PATH,
    output_manifest_path: Path | str = DEFAULT_OUTPUT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Load, normalize, validate, and replay all recorded monitor-runtime rows."""

    output = Path(output_path)
    output_manifest = Path(output_manifest_path)
    write_in_progress_artifact(output, output_manifest_path=output_manifest, run_date=run_date)

    blockers = gated_input_blockers(
        monitor_event_manifest_path=monitor_event_manifest_path,
        safe_prefix_manifest_path=safe_prefix_manifest_path,
        verifier_manifest_path=verifier_manifest_path,
        certificate_manifest_path=certificate_manifest_path,
        exp1507_artifact_path=exp1507_artifact_path,
        exp1508_artifact_path=exp1508_artifact_path,
    )
    exp1507_artifact = _load_json_if_exists(Path(exp1507_artifact_path))
    exp1508_artifact = _load_json_if_exists(Path(exp1508_artifact_path))
    false_accept_rate = _reported_false_accept_rate(exp1507_artifact, exp1508_artifact)
    if blockers:
        _write_jsonl(output_manifest, [])
        artifact = _terminal_artifact(
            status="blocked",
            ready=False,
            gated_inputs_present=False,
            events_loaded=0,
            events=[],
            output_manifest_path=output_manifest,
            verifier_false_accept_rate=false_accept_rate,
            blockers=blockers,
            tests_run=tests_run or [],
            run_date=run_date,
            honest_verdict="complete: gated before executable monitor runtime readiness",
        )
        _write_json(output, artifact)
        return artifact

    source_paths = {
        "monitor": Path(monitor_event_manifest_path),
        "safe_prefix": Path(safe_prefix_manifest_path),
        "verifier": Path(verifier_manifest_path),
        "certificate": Path(certificate_manifest_path),
    }
    monitor_rows = _read_jsonl(source_paths["monitor"])
    safe_prefix_rows = _read_jsonl(source_paths["safe_prefix"])
    verifier_rows = _read_jsonl(source_paths["verifier"])
    certificate_rows = _read_jsonl(source_paths["certificate"])
    events_loaded = (
        len(monitor_rows) + len(safe_prefix_rows) + len(verifier_rows) + len(certificate_rows)
    )
    events = normalize_source_rows(
        monitor_rows=monitor_rows,
        safe_prefix_rows=safe_prefix_rows,
        verifier_rows=verifier_rows,
        certificate_rows=certificate_rows,
        source_paths=source_paths,
    )
    schema_errors = [
        f"{event.get('event_id', 'unknown')}:{','.join(errors)}"
        for event in events
        if (errors := validate_normalized_event(event))
    ]
    blockers = [*schema_errors]
    replayed = replay_events(events)
    _write_jsonl(output_manifest, replayed)

    ready = bool(
        output_manifest.exists()
        and events_loaded == len(replayed)
        and not blockers
        and false_accept_rate is not None
    )
    artifact = _terminal_artifact(
        status="complete" if ready else "blocked",
        ready=ready,
        gated_inputs_present=True,
        events_loaded=events_loaded,
        events=replayed,
        output_manifest_path=output_manifest,
        verifier_false_accept_rate=false_accept_rate,
        blockers=blockers,
        tests_run=tests_run or [],
        run_date=run_date,
        honest_verdict=(
            "complete: executable monitor runtime adapter normalized replay manifest"
            if ready
            else "complete: blocked before executable monitor runtime readiness"
        ),
    )
    _write_json(output, artifact)
    return artifact


def _normalize_monitor_row(row: JsonDict, source_path: Path, line_number: int) -> JsonDict:
    case_id = str(row.get("case_id") or "")
    source_event_id = str(row.get("event_id") or f"monitor:{line_number}")
    return _base_event(
        source_experiment="1495",
        source_kind="monitor",
        source_path=source_path,
        source_line=line_number,
        source_row_id=source_event_id,
        source_event_id=source_event_id,
        event_kind="monitor_decision",
        case_id=case_id,
        family=str(row.get("family") or "unknown"),
        token_offset=_optional_int(row.get("token_offset")),
        validation_status="fail" if row.get("error_detected") else "pass",
        verifier_false_accept=bool(row.get("verifier_false_accept")),
        linked_monitor_event_id=None,
        link_status="not_applicable",
        provenance={
            "lane": row.get("lane"),
            "trace_id": row.get("trace_id"),
            "poll_index": row.get("poll_index"),
            "monitor_action": row.get("monitor_action"),
        },
    )


def _normalize_safe_prefix_row(
    row: JsonDict,
    source_path: Path,
    line_number: int,
    monitor_lookup: JsonDict,
) -> JsonDict:
    case_id = str(row.get("case_id") or "")
    link = _safe_prefix_monitor_link(row, monitor_lookup)
    return _base_event(
        source_experiment="1496",
        source_kind="safe_prefix",
        source_path=source_path,
        source_line=line_number,
        source_row_id=f"safe_prefix:{case_id}:{row.get('mode', 'unknown')}:{line_number}",
        source_event_id=str(row.get("selected_event_id") or ""),
        event_kind="safe_prefix_continuation",
        case_id=case_id,
        family=str(row.get("family") or "unknown"),
        token_offset=_optional_int(row.get("last_safe_token_offset")),
        validation_status="pass" if row.get("final_validator_passed") else "fail",
        verifier_false_accept=bool(
            row.get("verifier_false_accept") or row.get("false_accept_status")
        ),
        linked_monitor_event_id=link["event_id"],
        link_status=link["status"],
        provenance={
            "mode": row.get("mode"),
            "model_hf_id": row.get("model_hf_id"),
            "generation_source": row.get("generation_source"),
            "selection_reason": row.get("selection_reason"),
        },
    )


def _normalize_verifier_row(row: JsonDict, source_path: Path, line_number: int) -> JsonDict:
    row_type = str(row.get("row_type") or "candidate")
    candidate_name = str(row.get("candidate_name") or row_type)
    score = row.get("score") if isinstance(row.get("score"), dict) else {}
    false_accept_rate = row.get("verifier_false_accept_rate", score.get("false_accept_rate"))
    return _base_event(
        source_experiment="1507",
        source_kind="verifier",
        source_path=source_path,
        source_line=line_number,
        source_row_id=f"verifier:{candidate_name}:{line_number}",
        source_event_id=candidate_name,
        event_kind="verifier_induction",
        case_id=str(row.get("case_id") or ""),
        family=str(row.get("family") or "verifier"),
        token_offset=None,
        validation_status="pass" if _as_float(false_accept_rate) == 0.0 else "fail",
        verifier_false_accept=_as_float(false_accept_rate) > 0.0
        or int(score.get("false_accept_count") or 0) > 0,
        linked_monitor_event_id=None,
        link_status="not_applicable",
        provenance={
            "row_type": row_type,
            "candidate_names": row.get("candidate_names"),
            "candidate_name": row.get("candidate_name"),
            "model_hf_id": row.get("model_hf_id"),
        },
    )


def _normalize_certificate_row(row: JsonDict, source_path: Path, line_number: int) -> JsonDict:
    case_id = str(row.get("case_id") or "")
    verifier_result = (
        row.get("verifier_result") if isinstance(row.get("verifier_result"), dict) else {}
    )
    return _base_event(
        source_experiment="1508",
        source_kind="certificate",
        source_path=source_path,
        source_line=line_number,
        source_row_id=f"certificate:{case_id}:{row.get('decoder_mode', 'unknown')}:{line_number}",
        source_event_id=str(row.get("decoder_mode") or ""),
        event_kind="certificate_decoder",
        case_id=case_id,
        family=str(row.get("family") or "unknown"),
        token_offset=None,
        validation_status="pass" if row.get("deterministic_validation_passed") else "fail",
        verifier_false_accept=bool(
            row.get("false_accept_status") or verifier_result.get("false_accept")
        ),
        linked_monitor_event_id=None,
        link_status="not_applicable",
        provenance={
            "decoder_mode": row.get("decoder_mode"),
            "grammar_backend": row.get("grammar_backend"),
            "grammar_enforced": row.get("grammar_enforced"),
            "model_hf_id": row.get("model_hf_id"),
        },
    )


def _base_event(
    *,
    source_experiment: str,
    source_kind: str,
    source_path: Path,
    source_line: int,
    source_row_id: str,
    source_event_id: str,
    event_kind: str,
    case_id: str,
    family: str,
    token_offset: int | None,
    validation_status: str,
    verifier_false_accept: bool,
    linked_monitor_event_id: str | None,
    link_status: str,
    provenance: JsonDict,
) -> JsonDict:
    return {
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "event_id": "",
        "replay_index": 0,
        "source_experiment": source_experiment,
        "source_kind": source_kind,
        "source_path": _display_path(source_path),
        "source_line": int(source_line),
        "source_row_id": source_row_id,
        "source_event_id": source_event_id,
        "event_kind": event_kind,
        "case_id": case_id,
        "family": family,
        "token_offset": token_offset,
        "validation_status": validation_status,
        "verifier_false_accept": bool(verifier_false_accept),
        "linked_monitor_event_id": linked_monitor_event_id,
        "link_status": link_status,
        "provenance": provenance,
    }


def _build_monitor_lookup(monitor_rows: list[JsonDict]) -> JsonDict:
    by_event_id: dict[str, JsonDict] = {}
    by_case: dict[str, list[JsonDict]] = {}
    for row in monitor_rows:
        event_id = str(row.get("event_id") or "")
        case_id = str(row.get("case_id") or "")
        if event_id:
            by_event_id[event_id] = row
        if case_id:
            by_case.setdefault(case_id, []).append(row)
    return {"by_event_id": by_event_id, "by_case": by_case}


def _safe_prefix_monitor_link(row: JsonDict, monitor_lookup: JsonDict) -> JsonDict:
    selected_event_id = str(row.get("selected_event_id") or "")
    by_event_id = monitor_lookup["by_event_id"]
    if selected_event_id and selected_event_id in by_event_id:
        return {"event_id": selected_event_id, "status": "linked"}

    case_id = str(row.get("case_id") or "")
    case_rows = list(monitor_lookup["by_case"].get(case_id) or [])
    selected_offset = _optional_int(row.get("selected_event_token_offset"))
    last_safe_offset = _optional_int(row.get("last_safe_token_offset"))
    for monitor_row in case_rows:
        monitor_offset = _optional_int(monitor_row.get("token_offset"))
        polling_interval = _optional_int(monitor_row.get("polling_interval_tokens"))
        offset_matches = selected_offset is not None and monitor_offset == selected_offset
        prefix_matches = (
            last_safe_offset is not None
            and monitor_offset is not None
            and polling_interval is not None
            and monitor_offset - polling_interval == last_safe_offset
        )
        if offset_matches or prefix_matches:
            return {"event_id": monitor_row.get("event_id"), "status": "linked"}

    if len(case_rows) == 1:
        return {"event_id": case_rows[0].get("event_id"), "status": "linked"}
    return {"event_id": None, "status": "unmatched"}


def _event_sort_key(event: JsonDict) -> tuple[int, int, str, int]:
    source_order = {"monitor": 0, "safe_prefix": 1, "verifier": 2, "certificate": 3}
    return (
        source_order[str(event["source_kind"])],
        int(event["source_line"]),
        str(event["case_id"]),
        int(event["token_offset"] or 0),
    )


def _reported_false_accept_rate(
    exp1507_artifact: JsonDict | None,
    exp1508_artifact: JsonDict | None,
) -> float | None:
    rates = [
        _as_float((exp1507_artifact or {}).get("verifier_false_accept_rate")),
        _as_float((exp1508_artifact or {}).get("verifier_false_accept_rate")),
    ]
    reported = [rate for rate in rates if rate is not None]
    return max(reported) if reported else None


def _terminal_artifact(
    *,
    status: str,
    ready: bool,
    gated_inputs_present: bool,
    events_loaded: int,
    events: list[JsonDict],
    output_manifest_path: Path,
    verifier_false_accept_rate: float | None,
    blockers: list[str],
    tests_run: list[str],
    run_date: str,
    honest_verdict: str,
) -> JsonDict:
    return {
        "status": status,
        "run_date": run_date,
        "monitor_runtime_ready": bool(ready),
        "gated_inputs_present": bool(gated_inputs_present),
        "events_loaded": int(events_loaded),
        "events_normalized": len(events),
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "verifier_false_accept_rate": verifier_false_accept_rate,
        "safe_prefix_events_linked": sum(
            event["source_kind"] == "safe_prefix" and event["link_status"] == "linked"
            for event in events
        ),
        "monitor_event_manifest_path": _display_path(output_manifest_path),
        "adapter_tests_run": list(tests_run),
        "blockers": list(blockers),
        "honest_verdict": honest_verdict,
    }


def _read_jsonl(path: Path) -> list[JsonDict]:
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


def _load_json_if_exists(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path | str) -> str:
    return str(path)


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None
