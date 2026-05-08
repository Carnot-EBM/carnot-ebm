"""Exp 1513 FR-11 policy rollback replay audit.

This module is the promotion gate after Exp 1512's query-time policy cache.
Exp 1512 may propose bounded policy updates, but those updates are not trusted
until this deterministic replay audit proves that keeping them does not add
false accepts and that every kept update is backed by reachable validator
evidence.  The replay is intentionally artifact-only: it never calls an LLM,
never mutates model weights, and writes one rollback decision per proposed
policy update.

Spec: REQ-LEARN-1513, SCENARIO-LEARN-1514, SCENARIO-LEARN-1515.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

OUTPUT_FILE = "experiment_1513_fr11_policy_rollback_replay_audit.json"
MANIFEST_FILE = "fr11_policy_rollback_replay_1513.jsonl"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_ROLLBACK_MANIFEST_PATH = DEFAULT_RESULTS_DIR / MANIFEST_FILE
DEFAULT_POLICY_CACHE_ARTIFACT_PATH = (
    DEFAULT_RESULTS_DIR / "experiment_1512_fr11_verifier_feedback_policy_cache_v11.json"
)
DEFAULT_POLICY_CACHE_MANIFEST_PATH = DEFAULT_RESULTS_DIR / "fr11_policy_cache_events_1512.jsonl"
DEFAULT_DAILY_EVAL_MANIFEST_PATH = (
    DEFAULT_RESULTS_DIR / "fr11_trace2skill_daily_eval_manifest_1497.jsonl"
)
DEFAULT_MONITOR_EVENTS_PATH = DEFAULT_RESULTS_DIR / "executable_monitor_events_1509.jsonl"

RUN_DATE = "20260508"
SCHEMA = "fr11_policy_rollback_replay_audit_v1"
ROW_SCHEMA = "fr11_policy_rollback_replay_row_v1"
POLICY_ROW_SCHEMA = "fr11_policy_cache_event_v1"
DAILY_ROW_SCHEMA = "fr11_trace2skill_daily_eval_row_v1"
MONITOR_EVENT_SCHEMA = "monitor-runtime-event/v1"

PASSED_VERDICT = "complete: fr11_policy_rollback_replay_audit_passed"
BLOCKED_VERDICT = "complete: fr11_policy_rollback_replay_audit_blocked"
GATED_VERDICT = "complete: fr11_policy_rollback_replay_gate_blocked"

TERMINAL_VERDICT_PREFIXES: tuple[str, ...] = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "rollback_audit_passed",
    "gated_inputs_present",
    "policy_updates_replayed",
    "counterfactual_sessions",
    "accepted_policy_updates",
    "rolled_back_policy_updates",
    "soundness_mistakes",
    "false_accept_delta",
    "utility_delta",
    "rollback_manifest_path",
    "blockers",
    "honest_verdict",
)

ROLLBACK_REASONS = {
    "false_accept_increase",
    "stale_or_unreachable_evidence",
    "missing_deterministic_validator_support",
    "soundness_mistake",
    "exp1512_quarantined",
}

JsonDict = dict[str, Any]


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _display_path(path: Path | str, *, project_root: Path | str = REPO_ROOT) -> str:
    target = Path(path)
    try:
        return target.relative_to(Path(project_root)).as_posix()
    except ValueError:
        return target.name


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    destination.write_text(
        json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return serializable


def _write_jsonl(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), sort_keys=True, ensure_ascii=True) for row in rows]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _load_json(path: Path | str) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON artifact must be an object: {path}")  # pragma: no cover
    return payload


def _load_jsonl(path: Path | str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise AssertionError(f"JSONL row must be an object: {path}")  # pragma: no cover
        rows.append(payload)
    return rows


def _source_exists(path: str, *, project_root: Path | str) -> bool:
    if not path:
        return False
    source = Path(path)
    if source.is_absolute():
        return source.exists()
    return (Path(project_root) / source).exists()


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    manifest_path: Path | str = DEFAULT_ROLLBACK_MANIFEST_PATH,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-1513-1/8: create the audit artifact before source loading."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1513", "SCENARIO-LEARN-1514", "SCENARIO-LEARN-1515"],
        "run_date": run_date,
        "started_at": _timestamp(),
        "status": "in_progress",
        "rollback_audit_passed": False,
        "gated_inputs_present": False,
        "policy_updates_replayed": 0,
        "counterfactual_sessions": 0,
        "accepted_policy_updates": 0,
        "rolled_back_policy_updates": 0,
        "soundness_mistakes": 0,
        "false_accept_delta": 0,
        "utility_delta": 0,
        "rollback_manifest_path": _display_path(manifest_path, project_root=project_root),
        "blockers": ["audit_in_progress"],
        "honest_verdict": "complete: audit in progress",
    }
    return _write_json(output_path, artifact)


def _daily_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if case_id:
            index[f"daily_eval:{case_id}"] = row
    return index


def _monitor_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        event_id = row.get("event_id")
        if event_id:
            index[str(event_id)] = row
        source_event_id = row.get("source_event_id")
        if source_event_id:
            index.setdefault(str(source_event_id), row)
    return index


def _resolver_observed(row: Mapping[str, Any], name: str) -> bool:
    checks = row.get("expected_resolver_checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        return False
    for check in checks:
        if isinstance(check, Mapping) and check.get("name") == name:
            return bool(check.get("observed"))
    return False


def _daily_source_evidence(
    row: Mapping[str, Any] | None,
    *,
    project_root: Path | str,
) -> tuple[bool, bool, bool]:
    if row is None:
        return False, True, False
    schema_ok = row.get("schema") == DAILY_ROW_SCHEMA and bool(row.get("spec"))
    source_artifacts = row.get("source_artifacts")
    if not isinstance(source_artifacts, Sequence) or isinstance(source_artifacts, (str, bytes)):
        source_artifacts = []
    artifacts_reachable = bool(source_artifacts) and all(
        _source_exists(str(path), project_root=project_root) for path in source_artifacts
    )
    resolver_reachable = _resolver_observed(row, "source_artifact_present")
    deterministic = all(
        _resolver_observed(row, name)
        for name in (
            "paired_replay_case",
            "verifier_signal_present",
            "zero_soundness_policy_allowed",
        )
    )
    reachable = bool(artifacts_reachable and resolver_reachable)
    stale = not schema_ok
    return reachable, stale, deterministic


def _monitor_source_evidence(
    row: Mapping[str, Any] | None,
    *,
    project_root: Path | str,
) -> tuple[bool, bool, bool]:
    if row is None:
        return False, True, False
    schema_ok = row.get("event_schema_version") == MONITOR_EVENT_SCHEMA
    reachable = _source_exists(str(row.get("source_path") or ""), project_root=project_root)
    deterministic = schema_ok and row.get("validation_status") in {"pass", "fail"}
    stale = not schema_ok
    return reachable, stale, deterministic


def _bool_from_mapping(payload: Any, key: str) -> bool:
    if isinstance(payload, Mapping):
        return bool(payload.get(key))
    return False


def _daily_counterfactual(row: Mapping[str, Any] | None) -> tuple[int, int, int, int]:
    if row is None:
        return 0, 0, 0, 0
    baseline = row.get("baseline_outcome")
    proposed = row.get("memory_assisted_outcome")
    baseline_false_accepts = int(_bool_from_mapping(baseline, "soundness_mistake"))
    proposed_false_accepts = int(_bool_from_mapping(proposed, "soundness_mistake"))
    baseline_utility = int(_bool_from_mapping(baseline, "task_success"))
    proposed_utility = int(_bool_from_mapping(proposed, "task_success"))
    return baseline_false_accepts, proposed_false_accepts, baseline_utility, proposed_utility


def _monitor_policy_utility(policy_action: str, event: Mapping[str, Any] | None) -> int:
    if event is None or event.get("verifier_false_accept") is True:
        return 0
    status = event.get("validation_status")
    if policy_action == "verifier_escalation" and status == "fail":
        return 1
    if policy_action == "continuation_preference" and status == "pass":
        return 1
    if policy_action == "routing_prefer_deterministic_validator" and status == "pass":
        return 1
    return 0


def _monitor_counterfactual(
    policy_action: str,
    event: Mapping[str, Any] | None,
) -> tuple[int, int, int, int]:
    proposed_false_accepts = int(bool(event and event.get("verifier_false_accept") is True))
    return 0, proposed_false_accepts, 0, _monitor_policy_utility(policy_action, event)


def _source_for_policy_row(
    row: Mapping[str, Any],
    *,
    daily_rows: Mapping[str, Mapping[str, Any]],
    monitor_events: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any] | None]:
    source_event_id = str(row.get("source_event_id") or "")
    source_kind = str(row.get("source_kind") or "")
    if source_event_id.startswith("daily_eval:") or source_kind == "daily_eval":
        return "daily_eval", daily_rows.get(source_event_id)
    return "monitor", monitor_events.get(source_event_id)


def _policy_row_stale(row: Mapping[str, Any]) -> bool:
    return row.get("schema") != POLICY_ROW_SCHEMA or not row.get("spec")


def _rollback_reasons(
    *,
    row: Mapping[str, Any],
    false_accept_delta: int,
    soundness_mistakes: int,
    source_reachable: bool,
    source_stale: bool,
    deterministic_supported: bool,
) -> list[str]:
    reasons: set[str] = set()
    rejection_reasons = row.get("rejection_reasons")
    if not isinstance(rejection_reasons, Sequence) or isinstance(rejection_reasons, (str, bytes)):
        rejection_reasons = []
    if false_accept_delta > 0:
        reasons.add("false_accept_increase")
    if soundness_mistakes > 0:
        reasons.add("soundness_mistake")
    if not source_reachable or source_stale or _policy_row_stale(row):
        reasons.add("stale_or_unreachable_evidence")
    if not deterministic_supported:
        reasons.add("missing_deterministic_validator_support")
    if row.get("quarantined") or rejection_reasons:
        reasons.add("exp1512_quarantined")
    if "verifier_false_accept" in rejection_reasons:
        reasons.add("false_accept_increase")
    if "soundness_mistake" in rejection_reasons:
        reasons.add("soundness_mistake")
    if "missing_deterministic_validation" in rejection_reasons:
        reasons.add("missing_deterministic_validator_support")
    if {"stale_provenance", "unreachable_source_artifact"} & set(rejection_reasons):
        reasons.add("stale_or_unreachable_evidence")
    return sorted(reasons)


def _replay_row(
    policy_row: Mapping[str, Any],
    *,
    replay_index: int,
    daily_rows: Mapping[str, Mapping[str, Any]],
    monitor_events: Mapping[str, Mapping[str, Any]],
    project_root: Path | str,
    run_date: str,
) -> JsonDict:
    source_type, source = _source_for_policy_row(
        policy_row,
        daily_rows=daily_rows,
        monitor_events=monitor_events,
    )
    if source_type == "daily_eval":
        reachable, stale, source_deterministic = _daily_source_evidence(
            source,
            project_root=project_root,
        )
        baseline_false, proposed_false, baseline_utility, proposed_utility = _daily_counterfactual(
            source
        )
    else:
        reachable, stale, source_deterministic = _monitor_source_evidence(
            source,
            project_root=project_root,
        )
        baseline_false, proposed_false, baseline_utility, proposed_utility = (
            _monitor_counterfactual(str(policy_row.get("policy_action") or ""), source)
        )
    false_accept_delta = proposed_false - baseline_false
    utility_delta = proposed_utility - baseline_utility
    deterministic_supported = bool(
        policy_row.get("deterministic_validation_observed")
        and source_deterministic
        and policy_row.get("deterministic_validation_required", True)
    )
    soundness_mistakes = proposed_false
    reasons = _rollback_reasons(
        row=policy_row,
        false_accept_delta=false_accept_delta,
        soundness_mistakes=soundness_mistakes,
        source_reachable=reachable,
        source_stale=stale,
        deterministic_supported=deterministic_supported,
    )
    decision = "rollback" if reasons else "keep"
    return {
        "schema": ROW_SCHEMA,
        "spec": ["REQ-LEARN-1513", "SCENARIO-LEARN-1514", "SCENARIO-LEARN-1515"],
        "run_date": run_date,
        "replay_index": replay_index,
        "source_event_id": str(policy_row.get("source_event_id") or ""),
        "source_kind": str(policy_row.get("source_kind") or source_type),
        "source_case_id": str(policy_row.get("source_case_id") or ""),
        "skill_id": str(policy_row.get("skill_id") or ""),
        "policy_action": str(policy_row.get("policy_action") or ""),
        "baseline_policy_action": "baseline_no_policy_update",
        "proposed_policy_action": str(policy_row.get("policy_action") or ""),
        "baseline_false_accepts": baseline_false,
        "proposed_false_accepts": proposed_false,
        "false_accept_delta": false_accept_delta,
        "baseline_utility": baseline_utility,
        "proposed_utility": proposed_utility,
        "utility_delta": utility_delta,
        "soundness_mistakes": soundness_mistakes,
        "source_evidence_reachable": reachable,
        "source_evidence_stale": stale or _policy_row_stale(policy_row),
        "deterministic_validator_supported": deterministic_supported,
        "exp1512_quarantined": bool(policy_row.get("quarantined")),
        "rollback_reasons": reasons,
        "decision": decision,
        "policy_update_replayed": True,
    }


def build_replay_rows(
    policy_rows: Sequence[Mapping[str, Any]],
    *,
    daily_eval_rows: Sequence[Mapping[str, Any]] = (),
    monitor_events: Sequence[Mapping[str, Any]] = (),
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> list[JsonDict]:
    """REQ-LEARN-1513-3/4/5/6: build deterministic update replay rows."""

    daily_by_id = _daily_index(daily_eval_rows)
    monitor_by_id = _monitor_index(monitor_events)
    return [
        _replay_row(
            policy_row,
            replay_index=index,
            daily_rows=daily_by_id,
            monitor_events=monitor_by_id,
            project_root=project_root,
            run_date=run_date,
        )
        for index, policy_row in enumerate(policy_rows, start=1)
    ]


def _artifact_blockers(
    *,
    rows: Sequence[Mapping[str, Any]],
    gated_inputs_present: bool,
    manifest_exists: bool,
    source_blockers: Sequence[str],
) -> list[str]:
    blockers = list(source_blockers)
    if not gated_inputs_present:
        blockers.append("gated_inputs_missing")
    if gated_inputs_present and not rows:
        blockers.append("no_policy_updates_replayed")
    if gated_inputs_present and not manifest_exists:
        blockers.append("rollback_manifest_not_written")
    return sorted(set(blockers))


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    manifest_path: Path | str,
    manifest_exists: bool,
    gated_inputs_present: bool,
    source_blockers: Sequence[str],
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-LEARN-1513-7/8: summarize keep/rollback decisions after replay."""

    accepted = [row for row in rows if row.get("decision") == "keep"]
    rolled_back = [row for row in rows if row.get("decision") == "rollback"]
    soundness_mistakes = sum(int(row.get("soundness_mistakes", 0)) for row in accepted)
    false_accept_delta = sum(int(row.get("false_accept_delta", 0)) for row in accepted)
    utility_delta = sum(int(row.get("utility_delta", 0)) for row in accepted)
    blockers = _artifact_blockers(
        rows=rows,
        gated_inputs_present=gated_inputs_present,
        manifest_exists=manifest_exists,
        source_blockers=source_blockers,
    )
    rollback_audit_passed = bool(
        gated_inputs_present and rows and soundness_mistakes == 0 and false_accept_delta <= 0
    )
    if blockers:
        rollback_audit_passed = False
    artifact: JsonDict = {
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1513", "SCENARIO-LEARN-1514", "SCENARIO-LEARN-1515"],
        "run_date": run_date,
        "finished_at": _timestamp(),
        "status": "complete" if rollback_audit_passed else "blocked",
        "rollback_audit_passed": rollback_audit_passed,
        "gated_inputs_present": gated_inputs_present,
        "policy_updates_replayed": len(rows),
        "counterfactual_sessions": len(rows),
        "accepted_policy_updates": len(accepted),
        "rolled_back_policy_updates": len(rolled_back),
        "soundness_mistakes": soundness_mistakes,
        "false_accept_delta": false_accept_delta,
        "utility_delta": utility_delta,
        "rollback_manifest_path": _display_path(manifest_path, project_root=project_root),
        "blockers": blockers,
        "honest_verdict": (
            PASSED_VERDICT
            if rollback_audit_passed
            else GATED_VERDICT
            if not gated_inputs_present
            else BLOCKED_VERDICT
        ),
        "tests_run": list(tests_run or []),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    manifest_path: Path | str | None = None,
) -> None:
    """REQ-LEARN-1513-7/8: enforce the terminal rollback artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")  # pragma: no cover
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")  # pragma: no cover
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use an allowed terminal prefix")
    if artifact["status"] == "in_progress":
        return
    replayed = int(artifact["policy_updates_replayed"])
    accepted = int(artifact["accepted_policy_updates"])
    rolled_back = int(artifact["rolled_back_policy_updates"])
    if accepted > replayed:
        raise AssertionError("accepted updates cannot exceed replayed updates")
    if rolled_back > replayed:
        raise AssertionError(
            "rolled back updates cannot exceed replayed updates"
        )  # pragma: no cover
    if accepted + rolled_back != replayed:
        raise AssertionError("accepted plus rolled back updates must equal replayed")
    if artifact["rollback_audit_passed"]:
        if artifact["gated_inputs_present"] is not True:
            raise AssertionError("passed audit requires gated inputs")  # pragma: no cover
        if artifact["blockers"]:
            raise AssertionError("passed audit cannot have blockers")  # pragma: no cover
        if int(artifact["soundness_mistakes"]) != 0:
            raise AssertionError("passed audit requires zero soundness mistakes")
        if int(artifact["false_accept_delta"]) > 0:
            raise AssertionError("passed audit cannot increase false accepts")  # pragma: no cover
        if manifest_path is not None and not Path(manifest_path).exists():
            raise AssertionError("passed audit requires a rollback manifest")  # pragma: no cover


def _load_optional_jsonl(path: Path | str) -> list[JsonDict]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    return _load_jsonl(candidate)


def _gate_inputs(
    *,
    policy_cache_artifact_path: Path | str,
    policy_cache_manifest_path: Path | str,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    artifact_path = Path(policy_cache_artifact_path)
    manifest_path = Path(policy_cache_manifest_path)
    if not artifact_path.exists():
        blockers.append("missing_exp1512_policy_cache_artifact")
    else:
        try:
            artifact = _load_json(artifact_path)
        except (json.JSONDecodeError, OSError, AssertionError):
            blockers.append("malformed_exp1512_policy_cache_artifact")
        else:
            if artifact.get("policy_cache_ready") is not True:
                blockers.append("exp1512_policy_cache_not_ready")
    if not manifest_path.exists():
        blockers.append("missing_exp1512_policy_cache_manifest")
    return not blockers, sorted(set(blockers))


def run(
    *,
    policy_cache_artifact_path: Path | str = DEFAULT_POLICY_CACHE_ARTIFACT_PATH,
    policy_cache_manifest_path: Path | str = DEFAULT_POLICY_CACHE_MANIFEST_PATH,
    daily_eval_manifest_path: Path | str = DEFAULT_DAILY_EVAL_MANIFEST_PATH,
    monitor_events_path: Path | str = DEFAULT_MONITOR_EVENTS_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    rollback_manifest_path: Path | str = DEFAULT_ROLLBACK_MANIFEST_PATH,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run Exp 1513 and write the rollback replay manifest plus artifact."""

    write_in_progress_artifact(
        output_path,
        manifest_path=rollback_manifest_path,
        project_root=project_root,
        run_date=run_date,
    )
    gated_inputs_present, gate_blockers = _gate_inputs(
        policy_cache_artifact_path=policy_cache_artifact_path,
        policy_cache_manifest_path=policy_cache_manifest_path,
    )
    if not gated_inputs_present:
        _write_jsonl(rollback_manifest_path, [])
        artifact = build_artifact(
            rows=[],
            manifest_path=rollback_manifest_path,
            manifest_exists=Path(rollback_manifest_path).exists(),
            gated_inputs_present=False,
            source_blockers=gate_blockers,
            project_root=project_root,
            run_date=run_date,
            tests_run=tests_run,
        )
        _write_json(output_path, artifact)
        return artifact

    policy_rows = _load_jsonl(policy_cache_manifest_path)
    daily_rows = _load_optional_jsonl(daily_eval_manifest_path)
    monitor_events = _load_optional_jsonl(monitor_events_path)
    rows = build_replay_rows(
        policy_rows,
        daily_eval_rows=daily_rows,
        monitor_events=monitor_events,
        project_root=project_root,
        run_date=run_date,
    )
    _write_jsonl(rollback_manifest_path, rows)
    artifact = build_artifact(
        rows=rows,
        manifest_path=rollback_manifest_path,
        manifest_exists=Path(rollback_manifest_path).exists(),
        gated_inputs_present=True,
        source_blockers=[],
        project_root=project_root,
        run_date=run_date,
        tests_run=tests_run,
    )
    _write_json(output_path, artifact)
    return artifact
