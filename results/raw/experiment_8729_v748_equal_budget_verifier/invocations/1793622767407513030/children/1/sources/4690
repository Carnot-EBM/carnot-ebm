"""Build the Exp 3136 .291 false-accept root-cause autopsy artifact.

Spec refs: REQ-REPORT-3136, SCENARIO-REPORT-3136.

This module is accounting, not another verifier run. It preserves the failed
live verifier rows from Exp 3124, recomputes the false-accept rate from those
rows, and joins the bounded prefix and fragment-monitor evidence that explains
what the next verifier contract must guard. That separation matters because a
rerun without row-level failure evidence would only hide the failure mode that
blocked the .291 repair gate.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.291"
SCHEMA = "carnot.false_accept_root_cause_autopsy.v291.v1"
ARTIFACT = "experiment_3136_false_accept_root_cause_autopsy_v1"
OUTPUT_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3136_false_accept_root_cause_autopsy_v1.py"

EXP3124_REL_PATH = Path(
    "results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json"
)
EXP3125_REL_PATH = Path(
    "results/experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.json"
)
EXP3126_REL_PATH = Path(
    "results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"
)
EXP3133_REL_PATH = Path("results/experiment_3133_cross_corpus_matrix_v25.json")
EXP3134_REL_PATH = Path("results/experiment_3134_capstone_v291.json")
EXP3099_ROWS_REL_PATH = Path("results/local_sota_confidence_abstention_panel_3099/rows.jsonl")

SOURCE_PATHS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_repo_workflow", Path("CLAUDE.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("exp3124_live_verifier_rows", EXP3124_REL_PATH, True),
    ("exp3125_prefix_bound_evidence", EXP3125_REL_PATH, False),
    ("exp3126_fragment_monitor_ledger", EXP3126_REL_PATH, False),
    ("exp3133_matrix_v25_summary", EXP3133_REL_PATH, False),
    ("exp3134_capstone_v291_summary", EXP3134_REL_PATH, False),
    ("exp3099_prior_panel_rows", EXP3099_ROWS_REL_PATH, False),
)
REQUIRED_FIELDS = {
    "false_accept_autopsy_v1_ready",
    "source_false_accept_rate",
    "false_accept_row_ids",
    "false_accept_mechanism_counts",
    "extraction_failure_count",
    "prompt_ambiguity_count",
    "exact_label_mismatch_count",
    "contradiction_miss_count",
    "regression_row_set",
    "recommended_contract_changes",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
TOKEN_FAMILY_BY_LABEL = {
    "VALID": "validity_token",
    "INVALID": "validity_token",
    "SAT": "sat_token",
    "UNSAT": "sat_token",
    "REPAIRABLE": "repairability_token",
    "UNREPAIRABLE": "repairability_token",
}
ACCEPT_LABELS = {"VALID", "SAT"}
REJECT_LABELS = {"INVALID", "UNSAT", "REPAIRABLE", "UNREPAIRABLE"}


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object while keeping missing or malformed evidence visible."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_rows(path: Path) -> list[JsonDict]:
    """Read optional JSONL rows, ignoring malformed non-object lines."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[JsonDict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def sha256_file(path: Path) -> str | None:
    """Return a checksum so row-level conclusions trace to exact source bytes."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3136: synthesize a row-level false-accept autopsy."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3124 = read_json_object(root_path / EXP3124_REL_PATH)
    exp3125 = read_json_object(root_path / EXP3125_REL_PATH)
    exp3126 = read_json_object(root_path / EXP3126_REL_PATH)
    exp3133 = read_json_object(root_path / EXP3133_REL_PATH)
    exp3134 = read_json_object(root_path / EXP3134_REL_PATH)
    prior_rows = read_jsonl_rows(root_path / EXP3099_ROWS_REL_PATH)
    source_artifacts = _source_artifacts(root_path)

    live_rows = _mapping_rows(exp3124.get("live_rows"))
    monitor_events = _mapping_rows(exp3126.get("monitor_events"))
    monitor_by_fixture = _monitor_events_by_fixture(monitor_events)
    prior_by_fixture = _prior_rows_by_fixture(prior_rows)
    recomputed_rate = _false_accept_rate(live_rows)
    source_rate = _float(exp3124.get("false_accept_rate"), recomputed_rate)
    verifier_rows = [
        _verifier_row(row, monitor_by_fixture, prior_by_fixture)
        for row in sorted(live_rows, key=_row_id)
    ]
    false_accept_rows = [
        _false_accept_autopsy_row(row, exp3125, monitor_by_fixture, prior_by_fixture)
        for row in sorted(_false_accept_rows(live_rows), key=_row_id)
    ]
    false_accept_row_ids = [row["row_id"] for row in false_accept_rows]
    mechanism_counts = dict(Counter(row["primary_mechanism"] for row in false_accept_rows))
    blocked_reasons = _blocked_reasons(
        live_rows=live_rows,
        source_rate=source_rate,
        recomputed_rate=recomputed_rate,
        false_accept_rows=false_accept_rows,
        false_accept_row_ids=false_accept_row_ids,
    )
    ready = not blocked_reasons

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "false_accept_autopsy_v1_ready": ready,
        "source_false_accept_rate": source_rate,
        "recomputed_false_accept_rate": recomputed_rate,
        "source_false_accept_count": len(false_accept_rows),
        "source_live_row_count": len(live_rows),
        "source_reject_denominator": len(
            [row for row in live_rows if _expected_action(row) == "reject"]
        ),
        "false_accept_row_ids": false_accept_row_ids,
        "false_accept_mechanism_counts": mechanism_counts,
        "extraction_failure_count": mechanism_counts.get("answer-extraction failure", 0),
        "prompt_ambiguity_count": mechanism_counts.get("prompt ambiguity", 0),
        "exact_label_mismatch_count": mechanism_counts.get("exact-label mismatch", 0),
        "contradiction_miss_count": _contradiction_miss_count(false_accept_rows),
        "regression_row_set": list(false_accept_row_ids),
        "verifier_rows": verifier_rows,
        "false_accept_rows": false_accept_rows,
        "prefix_bound_summary": _prefix_bound_summary(exp3125),
        "fragment_monitor_summary": _fragment_monitor_summary(exp3126),
        "matrix_capstone_context": _matrix_capstone_context(exp3133, exp3134),
        "recommended_contract_changes": _recommended_contract_changes(false_accept_rows),
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row["sha256"] for row in source_artifacts},
        "inference_substrate": _inference_substrate(exp3124),
        "blocked_reasons": blocked_reasons,
        "scripts_research_conductor_modified": False,
        "ops_status_updated": False,
        "ops_changelog_updated": False,
        "traceability_updated": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_live_repair_rerun": True,
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    _validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3136 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_artifacts(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for role, rel_path, required in SOURCE_PATHS:
        path = root / rel_path
        source_type = "jsonl" if rel_path.suffix == ".jsonl" else rel_path.suffix.lstrip(".")
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type or "file",
                "present": path.is_file(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _mapping_rows(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _prior_rows_by_fixture(rows: list[JsonDict]) -> dict[str, JsonDict]:
    return {str(row.get("source_fixture_id") or ""): row for row in rows}


def _monitor_events_by_fixture(events: list[JsonDict]) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = {}
    for event in events:
        grouped.setdefault(str(event.get("fixture_id") or ""), []).append(event)
    return grouped


def _false_accept_rows(rows: list[JsonDict]) -> list[JsonDict]:
    return [
        row
        for row in rows
        if _expected_action(row) == "reject" and str(row.get("live_decision") or "") == "accept"
    ]


def _false_accept_rate(rows: list[JsonDict]) -> float:
    reject_rows = [row for row in rows if _expected_action(row) == "reject"]
    if not reject_rows:
        return 0.0
    return round(len(_false_accept_rows(rows)) / len(reject_rows), 6)


def _verifier_row(
    row: Mapping[str, Any],
    monitor_by_fixture: Mapping[str, list[JsonDict]],
    prior_by_fixture: Mapping[str, JsonDict],
) -> JsonDict:
    row_id = _row_id(row)
    return {
        "row_id": row_id,
        "exact_label": _exact_label(row),
        "expected_action": _expected_action(row),
        "live_model_verdict": str(
            row.get("raw_output") or row.get("extracted_answer") or ""
        ).strip(),
        "live_decision": str(row.get("live_decision") or ""),
        "extracted_answer": _optional_str(row.get("extracted_answer")),
        "difficulty_buckets": list(row.get("difficulty_bucket_labels") or []),
        "fixture_family": str(row.get("fixture_family") or row.get("task_family") or ""),
        "prompt_hash": _optional_str(row.get("prompt_hash")),
        "source_prompt_payload_sha256": _optional_str(row.get("source_prompt_payload_sha256")),
        "answer_extraction_format": str(
            row.get("answer_extraction_format") or _token_family(_exact_label(row)) or ""
        ),
        "label_source": _optional_str(row.get("label_source")),
        "failure_mechanism_from_exp3124": str(row.get("failure_mechanism") or ""),
        "monitor_events": _compact_monitor_events(monitor_by_fixture.get(row_id, [])),
        "prior_panel_row": prior_by_fixture.get(row_id, {}),
    }


def _false_accept_autopsy_row(
    row: Mapping[str, Any],
    exp3125: Mapping[str, Any],
    monitor_by_fixture: Mapping[str, list[JsonDict]],
    prior_by_fixture: Mapping[str, JsonDict],
) -> JsonDict:
    base = _verifier_row(row, monitor_by_fixture, prior_by_fixture)
    monitor = _monitor_comparison(row, monitor_by_fixture.get(base["row_id"], []))
    prefix = _prefix_bound_comparison(row, exp3125)
    mechanism = _classify_false_accept(row, monitor)
    base.update(
        {
            "primary_mechanism": mechanism,
            "mechanism_evidence": _mechanism_evidence(row, monitor, prefix),
            "prefix_bound_comparison": prefix,
            "monitor_comparison": monitor,
            "must_be_in_rerun_regression": True,
        }
    )
    return base


def _classify_false_accept(row: Mapping[str, Any], monitor: Mapping[str, Any]) -> str:
    exact = _exact_label(row)
    extracted = _optional_str(row.get("extracted_answer"))
    if extracted is None:
        return "answer-extraction failure"
    monitor_exact = _optional_str(monitor.get("exact_label"))
    if monitor_exact is not None and monitor_exact != exact:
        return "exact-label mismatch"
    if _prompt_payload_ambiguous(row):
        return "prompt ambiguity"
    if _token_family(exact) == "sat_token" and _token_family(extracted) == "validity_token":
        return "SAT/validity-token confusion"
    if monitor.get("failure_mechanism") == "data_prior_mismatch":
        return "model prior/data mismatch"
    if _has_failing_fragment(row):
        return "premise/step grounding failure"
    if (
        monitor.get("failure_mechanism") == "contradiction"
        or row.get("failure_mechanism") == "contradiction"
    ):
        return "contradiction miss"
    return "unknown"


def _mechanism_evidence(
    row: Mapping[str, Any],
    monitor: Mapping[str, Any],
    prefix: Mapping[str, Any],
) -> JsonDict:
    return {
        "exact_label": _exact_label(row),
        "extracted_answer": _optional_str(row.get("extracted_answer")),
        "expected_token_family": _token_family(_exact_label(row)),
        "extracted_token_family": _token_family(_optional_str(row.get("extracted_answer"))),
        "monitor_failure_mechanism": monitor.get("failure_mechanism"),
        "monitor_ledger_action": monitor.get("ledger_action"),
        "prefix_exact_label_covered": prefix.get("exact_label_covered"),
    }


def _prompt_payload_ambiguous(row: Mapping[str, Any]) -> bool:
    payload = row.get("prompt_payload")
    if not isinstance(payload, Mapping) or not payload:
        return True
    return "response_schema" not in payload and "required_fields" not in payload


def _has_failing_fragment(row: Mapping[str, Any]) -> bool:
    fragments = row.get("fragment_checks")
    if not isinstance(fragments, list):
        return False
    return any(isinstance(item, Mapping) and item.get("status") == "fail" for item in fragments)


def _prefix_bound_comparison(row: Mapping[str, Any], exp3125: Mapping[str, Any]) -> JsonDict:
    fixture_details = _mapping_rows(exp3125.get("fixture_details"))
    fixture_ids = {str(item.get("fixture_id") or "") for item in fixture_details}
    labels = {str(item.get("expected_answer") or "").upper() for item in fixture_details}
    exact = _exact_label(row)
    return {
        "prefix_artifact_ready": exp3125.get("prefix_closed_bound_pilot_ready") is True,
        "direct_fixture_match": _row_id(row) in fixture_ids,
        "exact_label_covered": exact in labels,
        "covered_fixture_labels": sorted(label for label in labels if label),
        "bound_width": _float(exp3125.get("bound_width"), 0.0),
        "limitation": "bounded_fixture_conditioned_prefix_frontier_not_live_llm_correctness",
    }


def _monitor_comparison(row: Mapping[str, Any], events: list[JsonDict]) -> JsonDict:
    ledger_event = _first_event(events, "constraint_ledger")
    exact_event = _first_event(events, "exact_test_z3_result")
    candidate_event = _first_event(events, "candidate_final_answer")
    drift_event = _first_event(events, "drift_classification")
    candidate_payload = _payload(candidate_event)
    drift_payload = _payload(drift_event)
    exact_payload = _payload(exact_event)
    ledger_payload = _payload(ledger_event)
    return {
        "monitor_event_count": len(events),
        "event_indices": [event.get("event_index") for event in events],
        "ledger_action": ledger_payload.get("ledger_action"),
        "ledger_source": ledger_payload.get("ledger_source"),
        "exact_label": exact_payload.get("exact_label") or _exact_label(row),
        "expected_action": exact_payload.get("expected_action") or _expected_action(row),
        "candidate_extracted_answer": candidate_payload.get("extracted_answer"),
        "candidate_live_decision": candidate_payload.get("live_decision"),
        "final_answer_consistent_with_exact": candidate_payload.get(
            "final_answer_consistent_with_exact"
        ),
        "final_answer_consistent_with_ledger": candidate_payload.get(
            "final_answer_consistent_with_ledger"
        ),
        "failure_mechanism": drift_payload.get("failure_mechanism"),
        "is_monitor_violation": drift_payload.get("is_monitor_violation") is True,
    }


def _first_event(events: list[JsonDict], event_type: str) -> JsonDict:
    for event in events:
        if event.get("event_type") == event_type:
            return event
    return {}


def _payload(event: Mapping[str, Any]) -> JsonDict:
    payload = event.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _compact_monitor_events(events: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "event_type": event.get("event_type"),
            "event_index": event.get("event_index"),
            "payload": _payload(event),
        }
        for event in events
    ]


def _contradiction_miss_count(false_accept_rows: list[JsonDict]) -> int:
    return sum(
        row.get("monitor_comparison", {}).get("failure_mechanism") == "contradiction"
        or row.get("failure_mechanism_from_exp3124") == "contradiction"
        for row in false_accept_rows
    )


def _blocked_reasons(
    *,
    live_rows: list[JsonDict],
    source_rate: float,
    recomputed_rate: float,
    false_accept_rows: list[JsonDict],
    false_accept_row_ids: list[str],
) -> list[str]:
    reasons: list[str] = []
    if not live_rows:
        reasons.append("exp3124_live_rows_missing")
    if abs(source_rate - recomputed_rate) > 1e-6:
        reasons.append("false_accept_rate_mismatch")
    if any(not row.get("primary_mechanism") for row in false_accept_rows):
        reasons.append("unclassified_false_accept_rows")
    if set(false_accept_row_ids) != {
        row["row_id"] for row in false_accept_rows if row.get("must_be_in_rerun_regression")
    }:
        reasons.append("regression_row_set_missing_false_accept")
    return reasons


def _prefix_bound_summary(exp3125: Mapping[str, Any]) -> JsonDict:
    return {
        "prefix_closed_bound_pilot_ready": exp3125.get("prefix_closed_bound_pilot_ready") is True,
        "fixture_count": _int(exp3125.get("fixture_count")),
        "explored_prefix_count": _int(exp3125.get("explored_prefix_count")),
        "accepted_prefix_count": _int(exp3125.get("accepted_prefix_count")),
        "lower_bound": _float(exp3125.get("lower_bound"), 0.0),
        "upper_bound": _float(exp3125.get("upper_bound"), 0.0),
        "bound_width": _float(exp3125.get("bound_width"), 0.0),
        "limitations": list(exp3125.get("limitations") or []),
    }


def _fragment_monitor_summary(exp3126: Mapping[str, Any]) -> JsonDict:
    return {
        "fragment_time_monitor_v1_ready": exp3126.get("fragment_time_monitor_v1_ready") is True,
        "monitor_violation_count": _int(exp3126.get("monitor_violation_count")),
        "contradiction_count": _int(exp3126.get("contradiction_count")),
        "satisfiable_drift_count": _int(exp3126.get("satisfiable_drift_count")),
        "ledger_consistency_rate": _float(exp3126.get("ledger_consistency_rate"), 0.0),
    }


def _matrix_capstone_context(exp3133: Mapping[str, Any], exp3134: Mapping[str, Any]) -> JsonDict:
    verifier_summary = exp3133.get("verifier_repair_summary")
    summary = dict(verifier_summary) if isinstance(verifier_summary, Mapping) else {}
    return {
        "matrix_v25_ready": exp3133.get("matrix_v25_ready") is True,
        "matrix_false_accept_rate": _float(summary.get("false_accept_rate"), 0.0),
        "matrix_repair_gate_state": summary.get("repair_gate_state"),
        "capstone_ready": exp3134.get("capstone_ready") is True,
        "capstone_next_top_gap": exp3134.get("next_top_gap"),
        "capstone_verifier_claim_status": exp3134.get("verifier_claim_status"),
    }


def _recommended_contract_changes(false_accept_rows: list[JsonDict]) -> list[str]:
    rows = ", ".join(row["row_id"] for row in false_accept_rows) or "none"
    return [
        f"Add row-level regression fixtures for known .291 false accepts: {rows}.",
        "Require token-family constraints: SAT/UNSAT prompts must reject VALID/INVALID outputs and validity prompts must reject SAT/UNSAT outputs.",
        "Require every accept on an exact-reject row to carry ledger-backed contradiction evidence before headline verifier or repair gates can open.",
        "Require monitor replay fields in reruns: ledger_action, exact_label, extracted_answer, live_decision, and final_answer_consistent_with_ledger.",
    ]


def _inference_substrate(exp3124: Mapping[str, Any]) -> JsonDict:
    upstream = exp3124.get("inference_substrate")
    live_calls = 0
    if isinstance(upstream, Mapping):
        live_calls = _int(upstream.get("live_model_calls"))
    if live_calls == 0:
        live_calls = _int(exp3124.get("live_call_count"))
    return {
        "kind": "aggregation_from_checked_in_verifier_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "local_repo_only": True,
        "no_live_llm_inference": True,
        "fresh_live_model_calls": 0,
        "upstream_live_model_calls_reused": live_calls,
        "source": EXP3124_REL_PATH.as_posix(),
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("false_accept_autopsy_v1_ready") is True:
        return (
            "complete: false_accept_autopsy_v1_ready=true; "
            f"source_false_accept_rate={artifact.get('source_false_accept_rate')}; "
            f"false_accept_row_ids={artifact.get('false_accept_row_ids')}"
        )
    return "blocked_false_accept_autopsy_missing_row_level_evidence"


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3136 artifact missing required fields: {missing}")


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id") or row.get("fixture_id") or "")


def _exact_label(row: Mapping[str, Any]) -> str:
    return str(row.get("exact_label") or row.get("expected_answer") or "").upper()


def _expected_action(row: Mapping[str, Any]) -> str:
    action = str(row.get("expected_action") or "")
    if action:
        return action
    exact = _exact_label(row)
    if exact in ACCEPT_LABELS:
        return "accept"
    if exact in REJECT_LABELS:
        return "reject"
    return "abstain"


def _token_family(label: str | None) -> str | None:
    return TOKEN_FAMILY_BY_LABEL.get(str(label or "").upper())


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float(value: Any, default: float) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return round(float(default), 6)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
