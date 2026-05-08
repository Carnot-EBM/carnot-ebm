"""Exp 1512 FR-11 verifier-feedback query-time policy cache.

This module turns already-recorded verifier and monitor outcomes into a bounded
policy-cache manifest.  The cache is intentionally weaker than a promotion
system: it can bias query-time retrieval, routing, continuation, and verifier
escalation decisions, but it never mutates model weights and never promotes a
skill without a later rollback audit.

Spec: REQ-LEARN-1512, SCENARIO-LEARN-1512, SCENARIO-LEARN-1513.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

OUTPUT_FILE = "experiment_1512_fr11_verifier_feedback_policy_cache_v11.json"
MANIFEST_FILE = "fr11_policy_cache_events_1512.jsonl"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_POLICY_CACHE_MANIFEST_PATH = DEFAULT_RESULTS_DIR / MANIFEST_FILE
DEFAULT_DAILY_EVAL_MANIFEST_PATH = (
    DEFAULT_RESULTS_DIR / "fr11_trace2skill_daily_eval_manifest_1497.jsonl"
)
DEFAULT_MONITOR_EVENTS_PATH = DEFAULT_RESULTS_DIR / "executable_monitor_events_1509.jsonl"

RUN_DATE = "20260508"
SCHEMA = "fr11_verifier_feedback_policy_cache_v11"
EVENT_SCHEMA = "fr11_policy_cache_event_v1"
MONITOR_EVENT_SCHEMA = "monitor-runtime-event/v1"
CONTINUOUS_SELF_LEARNING_TASK = True
NO_MODEL_WEIGHT_MUTATION = True
PROMOTION_REQUIRES_ROLLBACK_AUDIT = True

READY_VERDICT = "complete: fr11_v11_verifier_feedback_policy_cache_ready"
BLOCKED_VERDICT = "complete: fr11_v11_verifier_feedback_policy_cache_blocked"

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
    "continuous_self_learning_task",
    "policy_cache_ready",
    "no_model_weight_mutation",
    "source_events_loaded",
    "policy_updates_proposed",
    "policy_updates_accepted",
    "policy_update_rules",
    "soundness_mistakes",
    "verifier_false_accept_rate",
    "policy_cache_manifest_path",
    "promotion_requires_rollback_audit",
    "blockers",
    "honest_verdict",
)

POLICY_UPDATE_RULES: tuple[dict[str, str], ...] = (
    {
        "rule_id": "daily_promote_retrieval_boost",
        "action": "retrieval_boost",
        "description": "Promoted, reachable, zero-soundness daily-eval rows may boost retrieval.",
    },
    {
        "rule_id": "daily_retain_retrieval_demote",
        "action": "retrieval_demote",
        "description": "Non-improving retained skills are demoted from active retrieval boosts.",
    },
    {
        "rule_id": "monitor_failure_verifier_escalation",
        "action": "verifier_escalation",
        "description": "Failing monitor or validator events escalate deterministic verification.",
    },
    {
        "rule_id": "safe_prefix_pass_continuation_preference",
        "action": "continuation_preference",
        "description": "Validated safe-prefix continuations may be preferred at query time.",
    },
    {
        "rule_id": "deterministic_validator_routing_preference",
        "action": "routing_prefer_deterministic_validator",
        "description": "Passing verifier or certificate events may route toward deterministic validators.",
    },
    {
        "rule_id": "baseline_routing_preference",
        "action": "routing_prefer_baseline",
        "description": "Benign pass events with no stronger signal keep baseline routing.",
    },
    {
        "rule_id": "unsafe_or_unreachable_skill_quarantine",
        "action": "skill_quarantine",
        "description": "False accepts, unreachable artifacts, stale provenance, or missing validation quarantine the update.",
    },
)

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
    source = Path(path)
    if source.is_absolute():
        return source.exists()
    return (Path(project_root) / source).exists()


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    manifest_path: Path | str = DEFAULT_POLICY_CACHE_MANIFEST_PATH,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-1512-1/7: write the bootstrap artifact before source loading."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1512", "SCENARIO-LEARN-1512", "SCENARIO-LEARN-1513"],
        "run_date": run_date,
        "started_at": _timestamp(),
        "status": "in_progress",
        "continuous_self_learning_task": False,
        "policy_cache_ready": False,
        "no_model_weight_mutation": NO_MODEL_WEIGHT_MUTATION,
        "source_events_loaded": 0,
        "policy_updates_proposed": 0,
        "policy_updates_accepted": 0,
        "policy_update_rules": [dict(rule) for rule in POLICY_UPDATE_RULES],
        "soundness_mistakes": 0,
        "verifier_false_accept_rate": 0.0,
        "policy_cache_manifest_path": _display_path(manifest_path, project_root=project_root),
        "promotion_requires_rollback_audit": PROMOTION_REQUIRES_ROLLBACK_AUDIT,
        "blockers": ["implementation_pending"],
        "honest_verdict": "in_progress",
    }
    return _write_json(output_path, artifact)


def _resolver_observed(row: Mapping[str, Any], name: str) -> bool | None:
    checks = row.get("expected_resolver_checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        return None
    for check in checks:
        if isinstance(check, Mapping) and check.get("name") == name:
            return bool(check.get("observed"))
    return None


def _daily_rejection_reasons(row: Mapping[str, Any], *, project_root: Path | str) -> list[str]:
    reasons: list[str] = []
    if row.get("schema") != "fr11_trace2skill_daily_eval_row_v1" or not row.get("spec"):
        reasons.append("stale_provenance")
    source_artifacts = row.get("source_artifacts")
    if not isinstance(source_artifacts, Sequence) or isinstance(source_artifacts, (str, bytes)):
        source_artifacts = []
    missing_sources = [path for path in source_artifacts if not _source_exists(str(path), project_root=project_root)]
    if missing_sources or _resolver_observed(row, "source_artifact_present") is False:
        reasons.append("unreachable_source_artifact")
    if (
        _resolver_observed(row, "paired_replay_case") is not True
        or _resolver_observed(row, "verifier_signal_present") is not True
        or _resolver_observed(row, "zero_soundness_policy_allowed") is not True
    ):
        reasons.append("missing_deterministic_validation")
    memory_outcome = row.get("memory_assisted_outcome")
    if isinstance(memory_outcome, Mapping) and memory_outcome.get("soundness_mistake"):
        reasons.append("soundness_mistake")
    return sorted(set(reasons))


def _daily_action(row: Mapping[str, Any], rejection_reasons: Sequence[str]) -> str:
    if rejection_reasons or row.get("rotted") or row.get("decision") == "retire":
        return "skill_quarantine"
    if row.get("decision") == "promote":
        return "retrieval_boost"
    if row.get("decision") == "retain":
        return "retrieval_demote"
    return "routing_prefer_baseline"


def _daily_policy_row(
    row: Mapping[str, Any],
    *,
    source_index: int,
    run_date: str,
    project_root: Path | str,
) -> JsonDict:
    reasons = _daily_rejection_reasons(row, project_root=project_root)
    action = _daily_action(row, reasons)
    case_id = str(row.get("case_id") or "unknown-case")
    return _base_policy_row(
        source_event_id=f"daily_eval:{case_id}",
        source_kind="daily_eval",
        source_index=source_index,
        source_case_id=case_id,
        skill_id=str(row.get("skill_id") or ""),
        run_date=run_date,
        policy_action=action,
        rejection_reasons=reasons,
        deterministic_validation_observed="missing_deterministic_validation" not in reasons,
    )


def _monitor_rejection_reasons(event: Mapping[str, Any], *, project_root: Path | str) -> list[str]:
    reasons: list[str] = []
    if event.get("event_schema_version") != MONITOR_EVENT_SCHEMA or not isinstance(
        event.get("provenance"),
        Mapping,
    ):
        reasons.append("stale_provenance")
    source_path = str(event.get("source_path") or "")
    if not source_path or not _source_exists(source_path, project_root=project_root):
        reasons.append("unreachable_source_artifact")
    if event.get("validation_status") not in {"pass", "fail"}:
        reasons.append("missing_deterministic_validation")
    if event.get("verifier_false_accept") is True:
        reasons.append("verifier_false_accept")
    return sorted(set(reasons))


def _monitor_action(event: Mapping[str, Any], rejection_reasons: Sequence[str]) -> str:
    if rejection_reasons:
        return "skill_quarantine"
    provenance = event.get("provenance") if isinstance(event.get("provenance"), Mapping) else {}
    if (
        event.get("event_kind") == "safe_prefix_continuation"
        and event.get("validation_status") == "pass"
        and provenance.get("mode") == "safe_prefix_continuation"
    ):
        return "continuation_preference"
    if event.get("validation_status") == "fail" or provenance.get("monitor_action") == "interrupt":
        return "verifier_escalation"
    if event.get("event_kind") in {"certificate_decoder", "verifier_induction"}:
        return "routing_prefer_deterministic_validator"
    return "routing_prefer_baseline"


def _monitor_policy_row(
    event: Mapping[str, Any],
    *,
    source_index: int,
    run_date: str,
    project_root: Path | str,
) -> JsonDict:
    reasons = _monitor_rejection_reasons(event, project_root=project_root)
    action = _monitor_action(event, reasons)
    return _base_policy_row(
        source_event_id=str(event.get("event_id") or f"monitor:{source_index}"),
        source_kind=str(event.get("source_kind") or "monitor"),
        source_index=source_index,
        source_case_id=str(event.get("case_id") or ""),
        skill_id="",
        run_date=run_date,
        policy_action=action,
        rejection_reasons=reasons,
        deterministic_validation_observed="missing_deterministic_validation" not in reasons,
    )


def _base_policy_row(
    *,
    source_event_id: str,
    source_kind: str,
    source_index: int,
    source_case_id: str,
    skill_id: str,
    run_date: str,
    policy_action: str,
    rejection_reasons: Sequence[str],
    deterministic_validation_observed: bool,
) -> JsonDict:
    quarantine = policy_action == "skill_quarantine"
    return {
        "schema": EVENT_SCHEMA,
        "spec": ["REQ-LEARN-1512", "SCENARIO-LEARN-1512", "SCENARIO-LEARN-1513"],
        "run_date": run_date,
        "source_event_id": source_event_id,
        "source_kind": source_kind,
        "source_index": source_index,
        "source_case_id": source_case_id,
        "skill_id": skill_id,
        "policy_scope": "query_time_only",
        "policy_action": policy_action,
        "policy_update_proposed": True,
        "policy_update_accepted": True,
        "rejection_reasons": list(rejection_reasons),
        "quarantined": quarantine,
        "deterministic_validation_required": True,
        "deterministic_validation_observed": deterministic_validation_observed,
        "model_weight_mutation": False,
        "promotes_skill": False,
        "promotion_deferred_until_rollback_audit": policy_action == "retrieval_boost",
    }


def build_policy_cache_rows(
    daily_eval_rows: Sequence[Mapping[str, Any]],
    monitor_events: Sequence[Mapping[str, Any]],
    *,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> list[JsonDict]:
    """REQ-LEARN-1512-3/4/5: replay source events into bounded policy rows."""

    rows: list[JsonDict] = []
    for index, row in enumerate(daily_eval_rows, start=1):
        rows.append(
            _daily_policy_row(
                row,
                source_index=index,
                run_date=run_date,
                project_root=project_root,
            )
        )
    offset = len(rows)
    for index, event in enumerate(monitor_events, start=1):
        rows.append(
            _monitor_policy_row(
                event,
                source_index=offset + index,
                run_date=run_date,
                project_root=project_root,
            )
        )
    for replay_index, row in enumerate(rows, start=1):
        row["replay_index"] = replay_index
    return rows


def _blockers(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_blockers: Sequence[str],
    manifest_exists: bool,
    soundness_mistakes: int,
) -> list[str]:
    blockers = list(source_blockers)
    if not rows:
        blockers.append("no_source_events_loaded")
    if not manifest_exists:
        blockers.append("policy_cache_manifest_not_written")
    if soundness_mistakes:
        blockers.append("soundness_mistakes_present")
    return sorted(set(blockers))


def _soundness_mistakes(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if "verifier_false_accept" in row["rejection_reasons"]
        or "soundness_mistake" in row["rejection_reasons"]
    )


def _false_accept_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    false_accepts = sum(1 for row in rows if "verifier_false_accept" in row["rejection_reasons"])
    return false_accepts / len(rows)


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    manifest_path: Path | str,
    manifest_exists: bool,
    source_blockers: Sequence[str],
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-LEARN-1512-6/7: summarize policy-cache readiness without promotion."""

    soundness_mistakes = _soundness_mistakes(rows)
    blockers = _blockers(
        rows=rows,
        source_blockers=source_blockers,
        manifest_exists=manifest_exists,
        soundness_mistakes=soundness_mistakes,
    )
    policy_cache_ready = bool(
        manifest_exists
        and NO_MODEL_WEIGHT_MUTATION
        and soundness_mistakes == 0
        and PROMOTION_REQUIRES_ROLLBACK_AUDIT
        and not blockers
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1512", "SCENARIO-LEARN-1512", "SCENARIO-LEARN-1513"],
        "run_date": run_date,
        "finished_at": _timestamp(),
        "status": "complete" if policy_cache_ready else "blocked",
        "continuous_self_learning_task": CONTINUOUS_SELF_LEARNING_TASK,
        "policy_cache_ready": policy_cache_ready,
        "no_model_weight_mutation": NO_MODEL_WEIGHT_MUTATION,
        "source_events_loaded": len(rows),
        "policy_updates_proposed": sum(1 for row in rows if row["policy_update_proposed"]),
        "policy_updates_accepted": sum(1 for row in rows if row["policy_update_accepted"]),
        "policy_update_rules": [dict(rule) for rule in POLICY_UPDATE_RULES],
        "soundness_mistakes": soundness_mistakes,
        "verifier_false_accept_rate": round(_false_accept_rate(rows), 6),
        "policy_cache_manifest_path": _display_path(manifest_path, project_root=project_root),
        "promotion_requires_rollback_audit": PROMOTION_REQUIRES_ROLLBACK_AUDIT,
        "blockers": blockers,
        "honest_verdict": READY_VERDICT if policy_cache_ready else BLOCKED_VERDICT,
        "tests_run": list(tests_run or []),
    }
    validate_artifact(artifact, manifest_path=manifest_path)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    manifest_path: Path | str | None = None,
) -> None:
    """REQ-LEARN-1512-6/7: enforce the terminal artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")  # pragma: no cover
    if artifact["status"] == "in_progress":
        return
    if artifact["status"] not in {"complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")  # pragma: no cover
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use an allowed terminal prefix")  # pragma: no cover
    if int(artifact["policy_updates_accepted"]) > int(artifact["policy_updates_proposed"]):
        raise AssertionError("accepted updates cannot exceed proposed updates")  # pragma: no cover
    if int(artifact["policy_updates_proposed"]) > int(artifact["source_events_loaded"]):
        raise AssertionError("proposed updates cannot exceed loaded source events")  # pragma: no cover
    false_accept_rate = float(artifact["verifier_false_accept_rate"])
    if not 0.0 <= false_accept_rate <= 1.0:
        raise AssertionError("verifier_false_accept_rate must be a probability")  # pragma: no cover
    if artifact["policy_cache_ready"]:
        candidate_path = Path(manifest_path or artifact["policy_cache_manifest_path"])
        if not candidate_path.exists():
            raise AssertionError("policy_cache_ready requires a manifest file")  # pragma: no cover
        if artifact["no_model_weight_mutation"] is not True:
            raise AssertionError("policy cache cannot mutate model weights")  # pragma: no cover
        if int(artifact["soundness_mistakes"]) != 0:
            raise AssertionError("policy cache readiness requires zero soundness")  # pragma: no cover
        if artifact["promotion_requires_rollback_audit"] is not True:
            raise AssertionError("promotion must require rollback audit")  # pragma: no cover
        if artifact["blockers"]:
            raise AssertionError("policy_cache_ready cannot have blockers")  # pragma: no cover


def run(
    *,
    daily_eval_manifest_path: Path | str = DEFAULT_DAILY_EVAL_MANIFEST_PATH,
    monitor_events_path: Path | str = DEFAULT_MONITOR_EVENTS_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    policy_cache_manifest_path: Path | str = DEFAULT_POLICY_CACHE_MANIFEST_PATH,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run Exp 1512 and write the query-time policy cache manifest and artifact."""

    write_in_progress_artifact(
        output_path,
        manifest_path=policy_cache_manifest_path,
        project_root=project_root,
        run_date=run_date,
    )
    source_blockers: list[str] = []
    daily_path = Path(daily_eval_manifest_path)
    monitor_path = Path(monitor_events_path)
    if daily_path.exists():
        daily_rows = _load_jsonl(daily_path)
    else:
        daily_rows = []
        source_blockers.append("missing_exp1497_daily_eval_manifest")
    if monitor_path.exists():
        monitor_events = _load_jsonl(monitor_path)
    else:
        monitor_events = []
        source_blockers.append("missing_exp1509_monitor_events")

    rows = build_policy_cache_rows(
        daily_rows,
        monitor_events,
        project_root=project_root,
        run_date=run_date,
    )
    _write_jsonl(policy_cache_manifest_path, rows)
    artifact = build_artifact(
        rows=rows,
        manifest_path=policy_cache_manifest_path,
        manifest_exists=Path(policy_cache_manifest_path).exists(),
        source_blockers=source_blockers,
        project_root=project_root,
        run_date=run_date,
        tests_run=tests_run,
    )
    _write_json(output_path, artifact)
    return artifact
