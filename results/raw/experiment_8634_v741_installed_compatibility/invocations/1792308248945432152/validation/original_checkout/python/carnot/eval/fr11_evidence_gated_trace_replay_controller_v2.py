"""Exp 3215 FR-11 evidence-gated trace replay controller labels.

Spec refs: REQ-LEARN-3215, SCENARIO-LEARN-3215,
SCENARIO-LEARN-3215-BLOCKED.

This module extends the Exp 3200 trace-memory controller with utility labels
that are grounded in checked-in verifier outcomes.  The labels are controller
policy metadata only: they can explain whether a prior route was useful, but
they do not train, fine-tune, mutate, or claim improvement to any model
weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
MILESTONE = "2026.05.297"
SCHEMA_VERSION = "1.0"
EXPERIMENT_ID = "experiment_3215_fr11_evidence_gated_trace_replay_controller_v2"
SCHEMA = "carnot.fr11.evidence_gated_trace_replay_controller.v2"
OUTPUT_REL_PATH = Path(
    "results/experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.json"
)
EXP3200_REL_PATH = Path("results/experiment_3200_fr11_verify_trace_memory_controller_v1.json")
EXP3201_REL_PATH = Path("results/experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_REL_PATH = Path("python/carnot/eval/fr11_evidence_gated_trace_replay_controller_v2.py")
TEST_REL_PATH = Path(
    "tests/python/test_experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.py"
)

REUSED_TRACE_FIELDS = (
    "trace_id",
    "row_id",
    "replay_role",
    "verification_query",
    "consistency_judgment",
    "answer_abstain_decision",
    "exact_label",
    "routing_outcome",
)
REQUIRED_ARTIFACT_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "continuous_self_learning_task",
    "prior_trace_memory_artifact",
    "trace_count",
    "evidence_backed_trace_count",
    "replay_utility_label_count",
    "redundant_check_suppression_count",
    "heldout_row_count",
    "drift_row_count",
    "routing_improvement_count",
    "negative_control_regression_count",
    "rollback_event_count",
    "model_weight_update_claimed",
    "promotion_allowed",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' "
    "tests/python/test_experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.py -q",
    ".venv/bin/coverage run -m pytest -o addopts='' "
    "tests/python/test_experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.py -q",
    ".venv/bin/coverage report "
    "--include='python/carnot/eval/fr11_evidence_gated_trace_replay_controller_v2.py' "
    "--fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_no_hidden_weight_update_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3200_trace_memory_controller", EXP3200_REL_PATH, True),
    ("exp3201_nonforgetting_sidecar_audit", EXP3201_REL_PATH, True),
    ("exp3215_module", MODULE_REL_PATH, False),
    ("exp3215_tests", TEST_REL_PATH, False),
)
MUTATION_FLAGS = (
    "executes_live_model_inference",
    "model_weight_learning",
    "model_weight_training",
    "model_weight_mutation",
    "base_model_weights_updated",
    "kan_model_weight_training",
    "hidden_state_mutation_claimed",
)
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "passed:", "passed_")


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object while treating malformed evidence as unavailable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the checked-in trace controller and sidecar audit artifacts."""

    root_path = Path(root)
    return {
        "exp3200": read_json_object(root_path / EXP3200_REL_PATH),
        "exp3201": read_json_object(root_path / EXP3201_REL_PATH),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build a schema-complete Exp 3215 artifact from checked-in evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    blocker = source_blocker(sources)
    if blocker:
        artifact = blocked_artifact(root_path, blocker, sources, start, now_s, tests_run)
        validate_artifact(artifact)
        return artifact

    exp3200 = sources["exp3200"]
    exp3201 = sources["exp3201"]
    records = rows_from_trace_payload(exp3200)
    labels = label_replay_candidates(records)
    negative_count = negative_control_regression_count(exp3200, exp3201)
    rollback_count = rollback_event_count(exp3201, labels)
    routing_count = routing_improvement_count(labels)
    heldout_count = role_count(records, "heldout")
    drift_count = role_count(records, "drift")
    suppression_count = sum(int(bool(label["redundant_check_suppressed"])) for label in labels)
    blockers = promotion_blockers(
        trace_count=len(records),
        evidence_count=len(labels),
        label_count=len(labels),
        heldout_count=heldout_count,
        drift_count=drift_count,
        routing_count=routing_count,
        negative_count=negative_count,
        rollback_count=rollback_count,
        model_weight_update_claimed=False,
        conductor_file_modified=False,
        active_roadmap_modified=False,
    )
    promotion_allowed = not blockers
    artifact = {
        "artifact": EXPERIMENT_ID,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "continuous_self_learning_task": True,
        "prior_trace_memory_artifact": EXP3200_REL_PATH.as_posix(),
        "source_artifacts": source_artifacts(root_path),
        "label_schema": label_schema(),
        "trace_count": len(records),
        "evidence_backed_trace_count": len(labels),
        "replay_utility_label_count": len(labels),
        "redundant_check_suppression_count": suppression_count,
        "heldout_row_count": heldout_count,
        "drift_row_count": drift_count,
        "routing_improvement_count": routing_count,
        "negative_control_regression_count": negative_count,
        "rollback_event_count": rollback_count,
        "model_weight_update_claimed": False,
        "promotion_allowed": promotion_allowed,
        "promotion_blockers": blockers,
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "replay_utility_labels": labels,
        "evaluation_summary": evaluation_summary(labels, records),
        "inference_substrate": inference_substrate(),
        "source_preconditions": precondition_checks(sources),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, now_s),
        "honest_verdict": honest_verdict(
            promotion_allowed,
            len(labels),
            routing_count,
            negative_count,
            rollback_count,
        ),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    root: Path,
    blocker: str,
    sources: Mapping[str, Any],
    started_s: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    """Return a terminal artifact when source evidence is unavailable or unsafe."""

    return {
        "artifact": EXPERIMENT_ID,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "continuous_self_learning_task": True,
        "prior_trace_memory_artifact": EXP3200_REL_PATH.as_posix(),
        "source_artifacts": source_artifacts(root),
        "label_schema": label_schema(),
        "trace_count": 0,
        "evidence_backed_trace_count": 0,
        "replay_utility_label_count": 0,
        "redundant_check_suppression_count": 0,
        "heldout_row_count": 0,
        "drift_row_count": 0,
        "routing_improvement_count": 0,
        "negative_control_regression_count": 0,
        "rollback_event_count": 0,
        "model_weight_update_claimed": False,
        "promotion_allowed": False,
        "promotion_blockers": [blocker],
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "replay_utility_labels": [],
        "evaluation_summary": {},
        "inference_substrate": inference_substrate(mode="blocked_precondition_check"),
        "source_preconditions": precondition_checks(sources) | {"blocked_reason": blocker},
        "blocked_reason": blocker,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started_s, now_s),
        "honest_verdict": f"complete: blocked evidence-gated trace replay controller; {blocker}",
    }


def source_blocker(sources: Mapping[str, Any]) -> str:
    """REQ-LEARN-3215-2/5: fail closed on missing or unsafe source artifacts."""

    for key in ("exp3200", "exp3201"):
        payload = sources.get(key, {})
        if not isinstance(payload, Mapping) or not is_terminal(payload):
            return f"{key}_missing_or_not_terminal"
        if payload.get("model_weight_update_performed") is True:
            return f"{key}_model_weight_update_claimed"
        if source_claims_live_or_mutation(payload):
            return f"{key}_live_inference_or_weight_update_claimed"
    return ""


def is_terminal(payload: Mapping[str, Any]) -> bool:
    """Return whether a source artifact has a terminal verdict string."""

    return str(payload.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES)


def source_claims_live_or_mutation(payload: Mapping[str, Any]) -> bool:
    """Return whether a source artifact claims live inference or model mutation."""

    substrate = payload.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        return True
    return int(substrate.get("fresh_live_inference_calls") or 0) != 0 or any(
        substrate.get(flag) is True for flag in MUTATION_FLAGS
    )


def detected_model_weight_update(sources: Mapping[str, Any]) -> bool:
    """Report whether any loaded source already claimed model-weight mutation."""

    for payload in sources.values():
        if not isinstance(payload, Mapping):
            continue
        substrate = payload.get("inference_substrate", {})
        if payload.get("model_weight_update_performed") is True:
            return True
        if isinstance(substrate, Mapping) and any(
            substrate.get(flag) is True for flag in MUTATION_FLAGS if "weight" in flag
        ):
            return True
    return False


def rows_from_trace_payload(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Return trace dictionaries from an Exp 3200 artifact."""

    rows = payload.get("trace_records", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def label_replay_candidates(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """REQ-LEARN-3215-2/3: label only verifier-backed replay candidates."""

    labels: list[JsonDict] = []
    for record in records:
        label = utility_label_for_trace(record)
        if label is not None:
            labels.append(label)
    return labels


def utility_label_for_trace(record: Mapping[str, Any]) -> JsonDict | None:
    """Return one reward-weighted controller utility label for exact evidence."""

    if not trace_is_evidence_backed(record):
        return None
    outcome = exact_verifier_outcome(record)
    rollback_status = rollback_status_for(record, outcome)
    utility, reward = route_utility_for(record, outcome, rollback_status)
    label = {field: record.get(field) for field in REUSED_TRACE_FIELDS}
    label.update(
        {
            "exact_action": exact_action_for_label(
                record.get("exact_label"), record.get("expected_action")
            ),
            "exact_verifier_outcome": outcome,
            "prior_route_utility": utility,
            "reward_weight": reward,
            "redundant_check_suppressed": bool(record.get("redundant_check_suppressed")),
            "rollback_or_retraction_status": rollback_status,
            "controller_utility_label_only": True,
            "model_weight_update_claimed": False,
        }
    )
    return label


def trace_is_evidence_backed(record: Mapping[str, Any]) -> bool:
    """Return whether a trace has exact verifier evidence instead of a plan."""

    return all(str(record.get(field) or "").strip() for field in REUSED_TRACE_FIELDS)


def exact_verifier_outcome(record: Mapping[str, Any]) -> str:
    """Map exact labels and answer/abstain behavior to verifier outcomes."""

    action = exact_action_for_label(record.get("exact_label"), record.get("expected_action"))
    decision = normalize_token(record.get("answer_abstain_decision"))
    consistent = normalize_token(record.get("consistency_judgment")) == "consistent"
    changed = bool(record.get("observed_action_changed"))
    outcome = "exact_replay_failed"
    if consistent and not changed and action == "accept" and decision == "answer":
        outcome = "exact_accept_answered"
    if consistent and not changed and action == "reject" and decision == "abstain":
        outcome = "exact_reject_abstained"
    return outcome


def rollback_status_for(record: Mapping[str, Any], outcome: str) -> str:
    """Return compact rollback or retraction status for a trace label."""

    explicit = normalize_token(record.get("rollback_or_retraction_status"))
    if explicit not in {"unknown", "none"}:
        return explicit
    status = "rollback_required" if outcome == "exact_replay_failed" else "none"
    return (
        "rollback_or_retraction"
        if record.get("rollback_triggered") or record.get("retracted")
        else status
    )


def route_utility_for(
    record: Mapping[str, Any],
    outcome: str,
    rollback_status: str,
) -> tuple[str, float]:
    """Return controller-only utility and reward weight for a trace."""

    if rollback_status != "none":
        return ("block_rollback_or_retraction", -1.0)
    if bool(record.get("redundant_check_suppressed")):
        return ("suppress_redundant_recheck", 1.0)
    if outcome == "exact_accept_answered":
        return ("verified_answer_for_exact_accept", 0.75)
    if outcome == "exact_reject_abstained":
        return ("safe_abstain_for_exact_reject", 0.75)
    return ("block_failed_exact_replay", -1.0)


def exact_action_for_label(label: Any, expected_action: Any) -> str:
    """Map exact labels to the action the controller route must preserve."""

    text = str(label or "").strip().upper()
    if text in {"VALID", "EXACT_ACCEPT", "ACCEPT"}:
        return "accept"
    if text in {"INVALID", "EXACT_REJECT", "REJECT"}:
        return "reject"
    return normalize_token(expected_action)


def label_schema() -> JsonDict:
    """Return the replay-utility label schema for artifact consumers."""

    return {
        "schema_id": "carnot.fr11.evidence_gated_trace_replay_label.v2",
        "schema_version": SCHEMA_VERSION,
        "reused_trace_fields": list(REUSED_TRACE_FIELDS),
        "new_label_fields": [
            "exact_action",
            "exact_verifier_outcome",
            "prior_route_utility",
            "reward_weight",
            "redundant_check_suppressed",
            "rollback_or_retraction_status",
            "controller_utility_label_only",
            "model_weight_update_claimed",
        ],
        "evidence_over_plans_rule": (
            "labels require checked-in verifier outcomes and are never emitted "
            "for planned-only trajectories"
        ),
        "not_authority_for": [
            "model-weight updates",
            "foundation-model improvement claims",
            "accepting answers without exact verifier evidence",
        ],
    }


def evaluation_summary(
    labels: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Expose held-out, drift, negative-control, and rollback evaluation counts."""

    return {
        "heldout_label_count": role_count(labels, "heldout"),
        "drift_label_count": role_count(labels, "drift"),
        "negative_control_label_count": role_count(labels, "negative_control"),
        "planned_or_unverifiable_trace_count": max(0, len(records) - len(labels)),
        "positive_controller_utility_count": sum(
            1 for label in labels if float(label.get("reward_weight") or 0.0) > 0.0
        ),
        "negative_controller_utility_count": sum(
            1 for label in labels if float(label.get("reward_weight") or 0.0) < 0.0
        ),
    }


def role_count(rows: Sequence[Mapping[str, Any]], role: str) -> int:
    """Count rows by replay role."""

    return sum(1 for row in rows if row.get("replay_role") == role)


def routing_improvement_count(labels: Sequence[Mapping[str, Any]]) -> int:
    """REQ-LEARN-3215-4: count positive held-out and drift route utility."""

    return sum(
        1
        for label in labels
        if label.get("replay_role") in {"heldout", "drift"}
        and float(label.get("reward_weight") or 0.0) > 0.0
        and label.get("rollback_or_retraction_status") == "none"
    )


def negative_control_regression_count(
    exp3200: Mapping[str, Any], exp3201: Mapping[str, Any]
) -> int:
    """Return the strict visible negative-control regression count."""

    return max(
        int(exp3200.get("negative_control_regression_count") or 0),
        int(exp3201.get("negative_control_regression_count") or 0),
    )


def rollback_event_count(
    exp3201: Mapping[str, Any],
    labels: Sequence[Mapping[str, Any]],
) -> int:
    """Count source rollback reasons or label-level rollback/retraction events."""

    reasons = exp3201.get("rollback_reasons", [])
    reason_count = (
        len(reasons) if isinstance(reasons, Sequence) and not isinstance(reasons, str) else 0
    )
    source_count = reason_count or int(bool(exp3201.get("rollback_triggered")))
    label_count = sum(1 for label in labels if label.get("rollback_or_retraction_status") != "none")
    return max(source_count, label_count)


def promotion_blockers(
    *,
    trace_count: int,
    evidence_count: int,
    label_count: int,
    heldout_count: int,
    drift_count: int,
    routing_count: int,
    negative_count: int,
    rollback_count: int,
    model_weight_update_claimed: bool,
    conductor_file_modified: bool,
    active_roadmap_modified: bool,
) -> list[str]:
    """REQ-LEARN-3215-5: list exact promotion blockers."""

    blockers = [
        "empty_trace_memory" if trace_count <= 0 else "",
        "missing_heldout_replay" if heldout_count <= 0 else "",
        "missing_drift_replay" if drift_count <= 0 else "",
        "missing_evidence_backed_labels"
        if evidence_count != trace_count or label_count != trace_count
        else "",
        "missing_routing_improvement" if routing_count <= 0 else "",
        "negative_control_regression" if negative_count > 0 else "",
        "rollback_event" if rollback_count > 0 else "",
        "model_weight_update_claimed" if model_weight_update_claimed else "",
        "conductor_file_modified" if conductor_file_modified else "",
        "active_roadmap_modified" if active_roadmap_modified else "",
    ]
    return [blocker for blocker in blockers if blocker]


def precondition_checks(sources: Mapping[str, Any]) -> JsonDict:
    """Expose source readiness and no-mutation checks in the artifact."""

    exp3200 = sources.get("exp3200", {})
    exp3201 = sources.get("exp3201", {})
    return {
        "exp3200_present": isinstance(exp3200, Mapping) and bool(exp3200),
        "exp3200_terminal": isinstance(exp3200, Mapping) and is_terminal(exp3200),
        "exp3201_present": isinstance(exp3201, Mapping) and bool(exp3201),
        "exp3201_terminal": isinstance(exp3201, Mapping) and is_terminal(exp3201),
        "source_model_weight_update_detected": detected_model_weight_update(sources),
        "source_live_or_mutation_detected": any(
            source_claims_live_or_mutation(payload)
            for payload in sources.values()
            if isinstance(payload, Mapping) and payload
        ),
    }


def inference_substrate(mode: str = "controller_memory_evidence_gated_trace_replay") -> JsonDict:
    """Declare controller replay only, with no model-weight mutation."""

    return {
        "mode": mode,
        "controller_memory_replay_only": True,
        "trace_memory_policy_only": True,
        "evidence_gated_utility_labels_only": True,
        "uses_checked_in_artifacts_only": True,
        "executes_live_model_inference": False,
        "fresh_live_inference_calls": 0,
        "model_weight_learning": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
        "hidden_state_mutation_claimed": False,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """List source files and checksums for artifact lineage."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_ARTIFACTS:
        path = root / rel_path
        exists = path.is_file()
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "required": required,
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    return rows


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when Exp 3215 overclaims learning or skips promotion gates."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    false_claim_checks = [
        ("model_weight_update_claimed", artifact.get("model_weight_update_claimed") is not False),
        ("conductor_file_modified", artifact.get("conductor_file_modified") is not False),
        ("active_roadmap_modified", artifact.get("active_roadmap_modified") is not False),
    ]
    for field, failed in false_claim_checks:
        if failed:
            raise ValueError(f"{field} must remain false")
    if artifact.get("promotion_allowed") is True:
        errors = [
            "evidence-backed labels"
            if artifact.get("evidence_backed_trace_count") != artifact.get("trace_count")
            or artifact.get("replay_utility_label_count") != artifact.get("trace_count")
            else "",
            "heldout/drift routing denominators"
            if int(artifact.get("heldout_row_count") or 0) <= 0
            or int(artifact.get("drift_row_count") or 0) <= 0
            or int(artifact.get("routing_improvement_count") or 0) <= 0
            else "",
            "negative-control regression"
            if int(artifact.get("negative_control_regression_count") or 0) != 0
            else "",
            "rollback event" if int(artifact.get("rollback_event_count") or 0) != 0 else "",
        ]
        promotion_errors = [error for error in errors if error]
        if promotion_errors:
            raise ValueError("; ".join(promotion_errors))
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write deterministic Exp 3215 JSON."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def honest_verdict(
    promotion_allowed: bool,
    label_count: int,
    routing_improvement: int,
    negative_regressions: int,
    rollback_events: int,
) -> str:
    """Return a truthful terminal verdict for the controller-label artifact."""

    return (
        "complete: fr11 evidence-gated trace replay controller v2 materialized; "
        f"promotion_allowed={str(promotion_allowed).lower()}; "
        f"replay_utility_label_count={label_count}; "
        f"routing_improvement_count={routing_improvement}; "
        f"negative_control_regression_count={negative_regressions}; "
        f"rollback_event_count={rollback_events}; "
        "model_weight_update_claimed=false; "
        "controller_utility_labels_only=true"
    )


def normalize_token(value: Any) -> str:
    """Normalize small routing and status tokens used by trace rows."""

    text = str(value or "").strip().lower().replace(" ", "_")
    return text or "unknown"


def duration(started_s: float, now_s: float | None) -> float:
    """Return stable elapsed seconds for artifact provenance."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 checksum when the source file exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON output for deterministic artifact diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
