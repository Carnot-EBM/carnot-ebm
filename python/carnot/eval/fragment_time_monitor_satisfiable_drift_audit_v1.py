"""Exp 3126 fragment-time monitor and satisfiable-drift audit.

Spec refs: REQ-VERIFY-3126, SCENARIO-VERIFY-3126.

This module turns existing exact fixture, fragment, and verifier-panel
artifacts into replayable monitor events. The monitor is deliberately
artifact-only: it checks whether already-returned answers agree with the
maintained fragment/constraint ledger before any downstream repair step is
allowed to claim progress.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1"
SCHEMA = "carnot.fragment_time_monitor_satisfiable_drift_audit.v1"
OUTPUT_REL_PATH = Path(
    "results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"
)
SCRIPT_REL_PATH = (
    REPO_ROOT / "scripts" / ("experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.py")
)

EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
EXP3114_REL_PATH = Path(
    "results/experiment_3114_fragment_level_code_constraint_verification_pilot_v1.json"
)
EXP3124_REL_PATH = Path(
    "results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json"
)
MANIFEST_REL_PATH = Path("results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl")

EVENT_TYPES = (
    "partial_trace_state",
    "constraint_ledger",
    "exact_test_z3_result",
    "candidate_final_answer",
    "drift_classification",
)
FAILURE_MECHANISMS = (
    "no_failure",
    "contradiction",
    "satisfiable_drift",
    "extraction_format_failure",
    "data_prior_mismatch",
    "unknown",
    "not_observed",
)
VIOLATION_MECHANISMS = {
    "contradiction",
    "satisfiable_drift",
    "extraction_format_failure",
    "data_prior_mismatch",
    "unknown",
}
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_FIELDS = {
    "fragment_time_monitor_v1_ready",
    "monitored_fixture_count",
    "monitor_event_schema",
    "monitor_violation_count",
    "satisfiable_drift_count",
    "contradiction_count",
    "ledger_consistency_rate",
    "failure_mechanism_counts",
    "downstream_repair_constraints",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3126_fragment_time_monitor_satisfiable_drift_audit.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/eval -m pytest -o addopts='' tests/python/test_experiment_3126_fragment_time_monitor_satisfiable_drift_audit.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/fragment_time_monitor_satisfiable_drift_audit_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), False),
    ("exp3097_exact_protocol", EXP3097_REL_PATH, True),
    ("exp3097_stratified_manifest", MANIFEST_REL_PATH, True),
    ("exp3114_fragment_level_monitor_source", EXP3114_REL_PATH, True),
    ("exp3124_live_verifier_trace_source", EXP3124_REL_PATH, True),
    (
        "exp3126_module",
        Path("python/carnot/eval/fragment_time_monitor_satisfiable_drift_audit_v1.py"),
        False,
    ),
    (
        "exp3126_script",
        Path("scripts/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.py"),
        False,
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one local JSON object, returning empty evidence on parse failure."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_rows(path: Path) -> list[JsonDict]:
    """Read JSONL object rows from a local manifest."""

    rows: list[JsonDict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
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


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3126: build the terminal monitor artifact from checked-in traces."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3097 = read_json_object(root_path / EXP3097_REL_PATH)
    exp3114 = read_json_object(root_path / EXP3114_REL_PATH)
    exp3124 = read_json_object(root_path / EXP3124_REL_PATH)
    manifest_rel_path = Path(str(exp3097.get("stratified_eval_manifest_path") or MANIFEST_REL_PATH))
    manifest_rows = read_jsonl_rows(root_path / manifest_rel_path)
    fixture_rows = monitored_fixture_rows(manifest_rows, exp3124)
    fragment_checks = merge_fragment_checks(exp3114, exp3124)
    events = build_monitor_events(fixture_rows, fragment_checks, exp3124)
    replay = replay_monitor_events(events)
    source_rows = source_artifacts(root_path, manifest_rel_path)
    self_checks = self_check_monitor(events, replay)
    ready = bool(
        fixture_rows
        and set(EVENT_TYPES) <= set(monitor_event_schema()["event_types"])
        and all(row["exists"] for row in source_rows if row["required"])
        and self_checks["ledger_replay_passed"]
        and self_checks["monitor_determinism_passed"]
        and self_checks["final_answer_consistency_checked"] == replay["observed_final_answer_count"]
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fragment_time_monitor_v1_ready": ready,
        "monitored_fixture_count": replay["monitored_fixture_count"],
        "monitor_event_schema": monitor_event_schema(),
        "monitor_event_count": len(events),
        "monitor_events": events,
        "monitor_violation_count": replay["monitor_violation_count"],
        "satisfiable_drift_count": replay["satisfiable_drift_count"],
        "contradiction_count": replay["contradiction_count"],
        "ledger_consistency_rate": replay["ledger_consistency_rate"],
        "failure_mechanism_counts": replay["failure_mechanism_counts"],
        "ledger_replay_summary": replay,
        "self_checks": self_checks,
        "event_stream_hash": stable_hash(events),
        "downstream_repair_constraints": downstream_repair_constraints(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_rows,
        "inference_substrate": inference_substrate(exp3124),
        "duration_s": duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the Exp 3126 artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    validate_artifact(artifact)
    write_json(out_path, artifact)
    return out_path


def monitored_fixture_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    exp3124: Mapping[str, Any],
) -> list[JsonDict]:
    """Join exact manifest rows with panel metadata and keep deterministic order."""

    panel_by_id = {
        str(row.get("fixture_id") or ""): dict(row)
        for row in exp3124.get("panel_fixture_metadata", [])
        if isinstance(row, Mapping)
    }
    rows: list[JsonDict] = []
    if manifest_rows:
        for manifest in manifest_rows:
            fixture_id = str(manifest.get("source_fixture_id") or manifest.get("fixture_id") or "")
            panel = panel_by_id.get(fixture_id, {})
            exact_label = str(
                panel.get("exact_label") or manifest.get("expected_answer") or ""
            ).upper()
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "exact_label": exact_label,
                    "expected_action": str(
                        panel.get("expected_action")
                        or manifest.get("verifier_target", {}).get("expected_action")
                        or expected_action_from_answer(exact_label)
                    ),
                    "solver_label": manifest.get("solver_label") or panel.get("solver_label"),
                    "label_source": manifest.get("label_source") or panel.get("label_source"),
                    "task_family": manifest.get("task_family") or panel.get("task_family"),
                    "perturbation_type": manifest.get("perturbation_type")
                    or panel.get("perturbation_type"),
                    "source_prompt_payload_sha256": manifest.get("source_prompt_payload_sha256"),
                    "prompt_payload": manifest.get("leakage_safe_prompt_payload")
                    or panel.get("prompt_payload")
                    or {},
                }
            )
    else:
        for fixture_id, panel in panel_by_id.items():
            exact_label = str(panel.get("exact_label") or "").upper()
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "exact_label": exact_label,
                    "expected_action": str(
                        panel.get("expected_action") or expected_action_from_answer(exact_label)
                    ),
                    "solver_label": panel.get("solver_label"),
                    "label_source": panel.get("label_source"),
                    "task_family": panel.get("task_family"),
                    "perturbation_type": panel.get("perturbation_type"),
                    "source_prompt_payload_sha256": panel.get("source_prompt_payload_sha256"),
                    "prompt_payload": panel.get("prompt_payload") or {},
                }
            )
    return sorted((row for row in rows if row["fixture_id"]), key=lambda row: row["fixture_id"])


def merge_fragment_checks(
    exp3114: Mapping[str, Any],
    exp3124: Mapping[str, Any],
) -> dict[str, list[JsonDict]]:
    """Merge fragment checks from Exp 3114 and panel rows, deduplicating IDs."""

    fragments_by_fixture: dict[str, list[JsonDict]] = {}
    seen: set[str] = set()
    sources: list[Mapping[str, Any]] = [
        row for row in exp3114.get("fragment_checks", []) if isinstance(row, Mapping)
    ]
    for panel_key in ("panel_fixture_metadata", "live_rows"):
        for row in exp3124.get(panel_key, []):
            if not isinstance(row, Mapping):
                continue
            sources.extend(
                fragment
                for fragment in row.get("fragment_checks", [])
                if isinstance(fragment, Mapping)
            )
    for fragment in sources:
        fixture_id = str(fragment.get("fixture_id") or "")
        fragment_id = str(fragment.get("fragment_id") or "")
        if not fixture_id or not fragment_id or fragment_id in seen:
            continue
        seen.add(fragment_id)
        fragments_by_fixture.setdefault(fixture_id, []).append(dict(fragment))
    for fixture_id in fragments_by_fixture:
        fragments_by_fixture[fixture_id].sort(key=lambda row: str(row.get("fragment_id") or ""))
    return fragments_by_fixture


def build_monitor_events(
    fixture_rows: Sequence[Mapping[str, Any]],
    fragments_by_fixture: Mapping[str, Sequence[Mapping[str, Any]]],
    exp3124: Mapping[str, Any],
) -> list[JsonDict]:
    """Emit five replayable monitor events per fixture."""

    live_by_id = {
        str(row.get("fixture_id") or ""): dict(row)
        for row in exp3124.get("live_rows", [])
        if isinstance(row, Mapping)
    }
    events: list[JsonDict] = []
    event_index = 0
    for row in fixture_rows:
        fixture_id = str(row["fixture_id"])
        fragments = [dict(fragment) for fragment in fragments_by_fixture.get(fixture_id, [])]
        live_row = live_by_id.get(fixture_id)
        ledger_action = ledger_action_for_fixture(row, fragments)
        ledger_event = constraint_ledger_event(event_index + 1, row, fragments, ledger_action)
        exact_event = exact_result_event(event_index + 2, row)
        candidate_event = candidate_final_answer_event(
            event_index + 3,
            row,
            live_row,
            ledger_action,
        )
        classification = classify_failure(
            str(row.get("expected_action") or "abstain"),
            str(candidate_event["payload"]["live_decision"]),
            candidate_event["payload"].get("extracted_answer"),
            ledger_action,
        )
        events.extend(
            [
                partial_trace_state_event(event_index, row, fragments),
                ledger_event,
                exact_event,
                candidate_event,
                drift_classification_event(event_index + 4, row, classification, candidate_event),
            ]
        )
        event_index += len(EVENT_TYPES)
    return events


def partial_trace_state_event(
    event_index: int,
    row: Mapping[str, Any],
    fragments: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Create the partial-state event for one fixture."""

    status_counts = Counter(str(fragment.get("status") or "unknown") for fragment in fragments)
    return base_event(
        "partial_trace_state",
        event_index,
        row,
        {
            "fragment_count": len(fragments),
            "fragment_status_counts": dict(sorted(status_counts.items())),
            "partial_state": "fragment_trace_present" if fragments else "no_fragment_trace",
            "fragment_ids": [str(fragment.get("fragment_id") or "") for fragment in fragments],
        },
    )


def constraint_ledger_event(
    event_index: int,
    row: Mapping[str, Any],
    fragments: Sequence[Mapping[str, Any]],
    ledger_action: str,
) -> JsonDict:
    """Create the maintained constraint-ledger event for one fixture."""

    constraints = []
    for fragment in fragments:
        constraints.append(
            {
                "constraint_id": str(fragment.get("fragment_id") or ""),
                "status": str(fragment.get("status") or "unknown"),
                "failing_constraint": str(fragment.get("failing_constraint") or ""),
                "expected_direction": str(fragment.get("expected_direction") or ""),
                "solver_evidence": fragment.get("solver_evidence") or {},
            }
        )
    source = "fragment_checks" if constraints else "exact_label_fallback"
    return base_event(
        "constraint_ledger",
        event_index,
        row,
        {
            "ledger_action": ledger_action,
            "ledger_source": source,
            "constraint_count": len(constraints),
            "constraints": constraints,
            "ledger_hash": stable_hash(
                {"ledger_action": ledger_action, "constraints": constraints}
            ),
        },
    )


def exact_result_event(event_index: int, row: Mapping[str, Any]) -> JsonDict:
    """Create the exact test/Z3 result event for one fixture."""

    exact_label = str(row.get("exact_label") or "").upper()
    return base_event(
        "exact_test_z3_result",
        event_index,
        row,
        {
            "exact_label": exact_label,
            "expected_action": row.get("expected_action")
            or expected_action_from_answer(exact_label),
            "solver_label": row.get("solver_label"),
            "label_source": row.get("label_source"),
            "answer_extraction_format": extraction_format_for_answer(exact_label),
            "exact_authority_available": bool(exact_label and row.get("label_source")),
        },
    )


def candidate_final_answer_event(
    event_index: int,
    row: Mapping[str, Any],
    live_row: Mapping[str, Any] | None,
    ledger_action: str,
) -> JsonDict:
    """Create the candidate final-answer event for one fixture."""

    expected_action = str(row.get("expected_action") or "abstain")
    extracted = live_row.get("extracted_answer") if live_row is not None else None
    live_decision = str(live_row.get("live_decision") if live_row is not None else "missing")
    has_returned_answer = extracted is not None and live_decision in {"accept", "reject"}
    return base_event(
        "candidate_final_answer",
        event_index,
        row,
        {
            "observed": live_row is not None,
            "raw_output_hash": live_row.get("raw_output_hash") if live_row is not None else None,
            "prompt_hash": live_row.get("prompt_hash") if live_row is not None else None,
            "extracted_answer": extracted,
            "live_decision": live_decision,
            "expected_action": expected_action,
            "ledger_action": ledger_action,
            "final_answer_consistent_with_exact": live_decision == expected_action
            if live_decision in {"accept", "reject", "abstain"}
            else False,
            "final_answer_consistent_with_ledger": live_decision == ledger_action
            if has_returned_answer and ledger_action in {"accept", "reject"}
            else None,
            "has_returned_answer": has_returned_answer,
        },
    )


def drift_classification_event(
    event_index: int,
    row: Mapping[str, Any],
    classification: str,
    candidate_event: Mapping[str, Any],
) -> JsonDict:
    """Create the failure-mechanism event for one fixture."""

    return base_event(
        "drift_classification",
        event_index,
        row,
        {
            "failure_mechanism": classification,
            "is_monitor_violation": classification in VIOLATION_MECHANISMS,
            "exact_label": row.get("exact_label"),
            "expected_action": row.get("expected_action"),
            "live_decision": candidate_event["payload"]["live_decision"],
            "ledger_action": candidate_event["payload"]["ledger_action"],
        },
    )


def base_event(
    event_type: str,
    event_index: int,
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> JsonDict:
    """Return the common monitor event envelope."""

    return {
        "event_type": event_type,
        "event_version": "v1",
        "event_index": event_index,
        "fixture_id": str(row.get("fixture_id") or ""),
        "source_prompt_payload_sha256": row.get("source_prompt_payload_sha256"),
        "payload": dict(payload),
    }


def ledger_action_for_fixture(
    row: Mapping[str, Any], fragments: Sequence[Mapping[str, Any]]
) -> str:
    """Derive the maintained ledger action from fragments or exact labels."""

    if fragments:
        statuses = {str(fragment.get("status") or "unknown") for fragment in fragments}
        if "unknown" in statuses:
            return "abstain"
        if "fail" in statuses:
            return "reject"
        return "accept"
    exact_label = str(row.get("exact_label") or "").upper()
    return str(row.get("expected_action") or expected_action_from_answer(exact_label))


def classify_failure(
    expected_action: str,
    live_decision: str,
    extracted_answer: str | None,
    ledger_action: str,
) -> str:
    """Classify candidate behavior into the Exp 3126 failure taxonomy."""

    expected = str(expected_action or "abstain")
    live = str(live_decision or "missing")
    ledger = str(ledger_action or "abstain")
    if live == "missing":
        return "not_observed"
    if extracted_answer is None:
        return "extraction_format_failure"
    if ledger in {"accept", "reject"} and expected in {"accept", "reject"} and ledger != expected:
        return "data_prior_mismatch"
    if live == expected:
        return "no_failure"
    if live == "accept" and expected == "reject":
        return "contradiction"
    if live == "reject" and expected == "accept":
        return "satisfiable_drift"
    return "unknown"


def replay_monitor_events(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Replay events and derive public counts without trusting artifact fields."""

    fixture_ids = sorted({str(event.get("fixture_id") or "") for event in events})
    failure_counts = {mechanism: 0 for mechanism in FAILURE_MECHANISMS}
    final_answer_denominator = 0
    ledger_consistent = 0
    for event in events:
        if event.get("event_type") == "candidate_final_answer":
            payload = event.get("payload") or {}
            if (
                payload.get("has_returned_answer") is True
                and payload.get("final_answer_consistent_with_ledger") is not None
            ):
                final_answer_denominator += 1
                ledger_consistent += int(payload.get("final_answer_consistent_with_ledger") is True)
        if event.get("event_type") == "drift_classification":
            mechanism = str((event.get("payload") or {}).get("failure_mechanism") or "unknown")
            if mechanism not in failure_counts:
                mechanism = "unknown"
            failure_counts[mechanism] += 1
    return {
        "monitored_fixture_count": len(fixture_ids),
        "monitor_event_count": len(events),
        "observed_final_answer_count": final_answer_denominator,
        "ledger_consistent_final_answer_count": ledger_consistent,
        "ledger_consistency_rate": rate(ledger_consistent, final_answer_denominator),
        "failure_mechanism_counts": failure_counts,
        "monitor_violation_count": sum(
            failure_counts[mechanism] for mechanism in VIOLATION_MECHANISMS
        ),
        "satisfiable_drift_count": failure_counts["satisfiable_drift"],
        "contradiction_count": failure_counts["contradiction"],
    }


def self_check_monitor(events: Sequence[Mapping[str, Any]], replay: Mapping[str, Any]) -> JsonDict:
    """Run deterministic self-checks over the event stream."""

    replayed_again = replay_monitor_events(events)
    event_hash = stable_hash(events)
    round_trip_hash = stable_hash(json.loads(json.dumps(events, sort_keys=True)))
    return {
        "ledger_replay_passed": replayed_again == dict(replay),
        "final_answer_consistency_checked": int(replay["observed_final_answer_count"]),
        "monitor_determinism_passed": event_hash == round_trip_hash,
        "monitor_determinism_hash": event_hash,
    }


def monitor_event_schema() -> JsonDict:
    """Return the replay contract for monitor event rows."""

    return {
        "schema": "carnot.fragment_time_monitor.event.v1",
        "required_event_fields": [
            "event_type",
            "event_version",
            "event_index",
            "fixture_id",
            "payload",
        ],
        "event_types": {
            "partial_trace_state": {
                "payload_fields": ["fragment_count", "fragment_status_counts", "partial_state"],
                "purpose": "capture partial fragment state before final repair claims",
            },
            "constraint_ledger": {
                "payload_fields": ["ledger_action", "ledger_source", "constraints", "ledger_hash"],
                "purpose": "replay maintained constraints and expected action",
            },
            "exact_test_z3_result": {
                "payload_fields": [
                    "exact_label",
                    "expected_action",
                    "solver_label",
                    "label_source",
                ],
                "purpose": "preserve exact test/Z3 authority",
            },
            "candidate_final_answer": {
                "payload_fields": [
                    "observed",
                    "extracted_answer",
                    "live_decision",
                    "final_answer_consistent_with_ledger",
                ],
                "purpose": "check returned answers against exact labels and ledger state",
            },
            "drift_classification": {
                "payload_fields": ["failure_mechanism", "is_monitor_violation"],
                "purpose": "separate contradiction from satisfiable drift and format/data failures",
            },
        },
    }


def downstream_repair_constraints() -> JsonDict:
    """Return the exact contract downstream repair must honor."""

    return {
        "repair_requires_monitor_evidence": True,
        "required_ready_field": "fragment_time_monitor_v1_ready",
        "required_event_types": list(EVENT_TYPES),
        "must_replay_before_repair": True,
        "must_not_treat_satisfiable_drift_as_unsat_contradiction": True,
        "must_check_final_answer_consistency": True,
        "must_preserve_ledger_hash": True,
        "allowed_failure_mechanisms": list(FAILURE_MECHANISMS),
        "repair_prompt_inputs": [
            "fixture_id",
            "constraint_ledger.payload.constraints",
            "exact_test_z3_result.payload.exact_label",
            "candidate_final_answer.payload.extracted_answer",
            "drift_classification.payload.failure_mechanism",
        ],
    }


def expected_action_from_answer(answer: str | None) -> str:
    """Map exact answer labels onto accept/reject/abstain actions."""

    normalized = str(answer or "").upper()
    if normalized in {"VALID", "SAT"}:
        return "accept"
    if normalized in {"INVALID", "UNSAT", "REPAIRABLE", "UNREPAIRABLE"}:
        return "reject"
    return "abstain"


def extraction_format_for_answer(answer: str | None) -> str:
    """Classify exact labels by answer-token family."""

    normalized = str(answer or "").upper()
    if normalized in {"VALID", "INVALID"}:
        return "validity_token"
    if normalized in {"SAT", "UNSAT"}:
        return "sat_token"
    if normalized in {"REPAIRABLE", "UNREPAIRABLE"}:
        return "repairability_token"
    return "unknown_token"


def source_artifacts(root: Path, manifest_rel_path: Path) -> list[JsonDict]:
    """List source files with checksums so the monitor can be traced."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_ARTIFACTS:
        actual_rel = manifest_rel_path if rel_path == MANIFEST_REL_PATH else rel_path
        path = root / actual_rel
        rows.append(
            {
                "id": source_id,
                "path": actual_rel.as_posix(),
                "required": required,
                "exists": path.is_file(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def inference_substrate(exp3124: Mapping[str, Any]) -> JsonDict:
    """Describe that Exp 3126 reuses traces rather than invoking models."""

    upstream = exp3124.get("inference_substrate")
    upstream_live_calls = 0
    if isinstance(upstream, Mapping):
        upstream_live_calls = int(
            upstream.get("live_model_calls") or upstream.get("live_call_count") or 0
        )
    return {
        "kind": "artifact_only_fragment_time_monitor",
        "fresh_live_inference_calls": 0,
        "executes_models": False,
        "uses_checked_in_artifacts_only": True,
        "upstream_live_trace_source": EXP3124_REL_PATH.as_posix(),
        "upstream_live_model_calls_reused": upstream_live_calls,
        "exact_fixture_authority_source": EXP3097_REL_PATH.as_posix(),
        "fragment_trace_source": EXP3114_REL_PATH.as_posix(),
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return conductor-compatible terminal verdict wording."""

    if artifact.get("fragment_time_monitor_v1_ready") is True:
        return (
            "complete: fragment_time_monitor_v1_ready=true; "
            f"monitored_fixture_count={artifact.get('monitored_fixture_count')}; "
            f"monitor_violation_count={artifact.get('monitor_violation_count')}; "
            f"ledger_consistency_rate={artifact.get('ledger_consistency_rate')}"
        )
    return "blocked_fragment_time_monitor_missing_required_evidence"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3126 artifact violates its replay contract."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or substrate.get("fresh_live_inference_calls") != 0:
        raise ValueError("fresh_live_inference_calls must remain zero")
    rate_value = float(artifact.get("ledger_consistency_rate", math.nan))
    if not math.isfinite(rate_value) or not 0.0 <= rate_value <= 1.0:
        raise ValueError("ledger_consistency_rate must be finite and within [0, 1]")
    events = artifact.get("monitor_events")
    if not isinstance(events, list):
        raise ValueError("monitor_events must be present for replay")
    replay = replay_monitor_events(events)
    for field in (
        "monitored_fixture_count",
        "monitor_event_count",
        "monitor_violation_count",
        "satisfiable_drift_count",
        "contradiction_count",
        "ledger_consistency_rate",
        "failure_mechanism_counts",
    ):
        if artifact.get(field) != replay.get(field):
            raise ValueError(f"{field} does not match replay")
    if artifact.get("event_stream_hash") != stable_hash(events):
        raise ValueError("event_stream_hash does not match monitor_events")
    source_rows = artifact.get("source_artifacts")
    if not isinstance(source_rows, list) or any(
        row.get("required") and not row.get("exists")
        for row in source_rows
        if isinstance(row, Mapping)
    ):
        raise ValueError("required source_artifacts must exist")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("fragment_time_monitor_v1_ready") is True and not any(
        verdict.startswith(prefix) for prefix in SUCCESS_PREFIXES
    ):
        raise ValueError("ready artifacts require an honest_verdict success prefix")


def stable_hash(payload: Any) -> str:
    """Hash JSON-compatible payloads with stable key ordering."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return a file checksum when the local source exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rate(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded safe ratio."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def duration(started_s: float, now_s: float | None) -> float:
    """Return a nonnegative elapsed duration."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
