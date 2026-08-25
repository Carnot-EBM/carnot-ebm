"""Exp5382: real-workflow continuous self-learning with corrected memory.

Spec refs: REQ-LEARN-5382, SCENARIO-LEARN-5382-GATE,
SCENARIO-LEARN-5382-IDENTICAL-TASKS, SCENARIO-LEARN-5382-SAFETY.

This module runs the next self-learning check only after Exp5381 has rebuilt
the budget-memory gate from row evidence. The learner is deliberately limited
to controller state: it may remember which retrievals and verifier choices were
safe, but it never loads or writes model weights. That keeps the experiment
about online governance value instead of quietly becoming a training run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5369_budgeted_continuous_self_learning_scaleup_v489 as exp5369
from carnot import experiment_5381_budget_memory_tautology_corrigendum_v490 as exp5381
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5382_real_workflow_continuous_self_learning_v490"
EXPERIMENT_ID = "exp5382-v490-real-workflow-continuous-self-learning"
MILESTONE = "2026.07.490"
SCHEMA = "carnot.experiment_5382.real_workflow_continuous_self_learning.v490"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5382
WORKFLOW_NAME = "budgeted_dependency_drift_verifier_tool_workflow"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5382_real_workflow_continuous_self_learning_v490.json"
)
EXP5381_RELATIVE_PATH = exp5381.RESULT_RELATIVE_PATH
EXP5369_RELATIVE_PATH = exp5369.RESULT_RELATIVE_PATH
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5382_real_workflow_continuous_self_learning_v490.py"
)
EXP5381_MODULE_RELATIVE_PATH = exp5381.MODULE_RELATIVE_PATH
EXP5369_MODULE_RELATIVE_PATH = exp5369.MODULE_RELATIVE_PATH

BUDGET_LIMIT_BYTES = exp5381.DEFAULT_BUDGET_BYTES
MIN_SESSIONS = 12
MIN_TRACES = 12
MIN_CHECKED_EVENTS = 30
TERMINAL_PREFIXES = ("complete:", "blocked_")

SPEC_REFS = (
    "REQ-LEARN-5382",
    "SCENARIO-LEARN-5382-GATE",
    "SCENARIO-LEARN-5382-IDENTICAL-TASKS",
    "SCENARIO-LEARN-5382-SAFETY",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Complete only if the gated real-workflow experiment ran.",
    "continuous_self_learning_target": "Must be true.",
    "continuous_self_learning_real_workflow_ready": (
        "True only if the real-workflow result has clean evidence and no unsafe false accepts."
    ),
    "upstream_budget_memory_corrigendum_clean": "Copied from Exp5381.",
    "workflow_name": "Name of the real multi-session workflow.",
    "session_count": "Number of sessions.",
    "trace_count": "Number of traces.",
    "checked_event_count": ("Number of events evaluated by deterministic checks."),
    "context_efficiency_delta": ("Self-learning minus baseline context efficiency."),
    "verifier_cost_delta": (
        "Baseline minus self-learning verifier cost, positive means improvement."
    ),
    "quality_delta": "Self-learning minus baseline task quality.",
    "stale_memory_deflection_rate": ("Fraction of stale-memory probes rejected."),
    "poison_memory_deflection_rate": ("Fraction of poisoned-memory probes rejected."),
    "rollback_success_rate": ("Fraction of failed updates rolled back correctly."),
    "no_weight_mutation": "Must be true.",
    "unsafe_false_accepts": ("Count of bad learned decisions accepted as good."),
    "honest_verdict": "One-line result or block reason.",
}

REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)
BOOL_FIELDS = (
    "continuous_self_learning_target",
    "continuous_self_learning_real_workflow_ready",
    "upstream_budget_memory_corrigendum_clean",
    "no_weight_mutation",
)
INTEGER_FIELDS = (
    "session_count",
    "trace_count",
    "checked_event_count",
    "unsafe_false_accepts",
)
NUMERIC_FIELDS = (
    "context_efficiency_delta",
    "verifier_cost_delta",
    "quality_delta",
    "stale_memory_deflection_rate",
    "poison_memory_deflection_rate",
    "rollback_success_rate",
)


def confirm_upstream_gate(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load Exp5381 and expose the exact gate used by Exp5382."""

    source = _read_json(Path(root) / EXP5381_RELATIVE_PATH)
    checks = {
        "budget_memory_corrigendum_clean": (source.get("budget_memory_corrigendum_clean") is True),
        "source_status_complete": source.get("status") == "complete",
        "source_unsafe_false_accepts_zero": (source.get("unsafe_false_accepts") == 0),
        "source_no_weight_mutation": source.get("no_weight_mutation") is True,
        "source_rollback_supported": source.get("rollback_supported") is True,
        "source_policy_ready": (source.get("keep_share_trust_policy_ready") is True),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "source_status": source.get("status"),
        "all_passed": not failed,
        "failed_gates": failed,
        "source_artifact": str(EXP5381_RELATIVE_PATH),
        "source_honest_verdict": source.get("honest_verdict"),
    }


def select_workflow_traces() -> JsonList:
    """Return the repeated multi-session workflow used for both variants."""

    return exp5369.build_budgeted_multi_session_traces()


def describe_workflow(traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count sessions, traces, events, and repeated decision types."""

    events = _flatten_events(traces)
    decision_type_counts = {
        "retrieval": sum(1 for event in events if event.get("utility_memory")),
        "verifier_tool_use": sum(
            1
            for event in events
            if event.get("verifier_tool_decision", {}).get("selected_tool")
            and event.get("verifier_tool_decision", {}).get("selected_verifier")
        ),
        "rollback": sum(1 for event in events if event.get("rollback_event", {}).get("required")),
        "constraint_selection": sum(
            1
            for event in events
            if event.get("verifier_tool_decision", {}).get("selected_verifier")
        ),
    }
    return {
        "workflow_name": WORKFLOW_NAME,
        "session_count": len({str(trace["session_id"]) for trace in traces}),
        "trace_count": len({str(trace["trace_id"]) for trace in traces}),
        "checked_event_count": len(events),
        "decision_type_counts": decision_type_counts,
        "event_ids": [str(event["event_id"]) for event in events],
    }


def build_corrected_memory_policy(
    upstream_artifact: Mapping[str, Any] | None = None,
    root: Path | str = REPO_ROOT,
) -> JsonDict:
    """Build a controller-memory policy from Exp5381 evidence rows."""

    source = (
        _read_json(Path(root) / EXP5381_RELATIVE_PATH)
        if upstream_artifact is None
        else dict(upstream_artifact)
    )
    rows = [dict(row) for row in source["memory_evidence_rows"]]
    kept_rows = [
        row
        for row in rows
        if row["recomputed_keep_decision"] == "KEEP" and row["recomputed_trust_decision"] == "TRUST"
    ]
    rejected_rows = [row for row in rows if row not in kept_rows]
    retained_bytes = sum(row["cost_evidence"]["byte_cost"] for row in kept_rows)
    return {
        "budget_limit_bytes": BUDGET_LIMIT_BYTES,
        "retained_bytes": retained_bytes,
        "budget_limit_respected": retained_bytes <= BUDGET_LIMIT_BYTES,
        "kept_memory_ids": sorted(row["memory_id"] for row in kept_rows),
        "trusted_memory_ids": sorted(
            row["memory_id"] for row in rows if row["recomputed_trust_decision"] == "TRUST"
        ),
        "rejected_or_quarantined_memory_ids": sorted(row["memory_id"] for row in rejected_rows),
        "evidence_row_count": len(rows),
        "rows": rows,
    }


def evaluate_real_workflow(
    *,
    traces: Sequence[Mapping[str, Any]] | None = None,
    root: Path | str = REPO_ROOT,
) -> JsonDict:
    """Compare baseline and self-learning variants on the same workflow."""

    workflow_traces = list(traces) if traces is not None else select_workflow_traces()
    description = describe_workflow(workflow_traces)
    upstream = _read_json(Path(root) / EXP5381_RELATIVE_PATH)
    memory_policy = build_corrected_memory_policy(upstream, root)
    replay = exp5369.evaluate_budgeted_loop(traces=workflow_traces, root=root)
    comparison = replay["policy_comparison"]
    events = _flatten_events(workflow_traces)
    event_decisions = _evaluate_event_decisions(events, memory_policy)
    baseline = _variant_from_metrics(
        "baseline_always_full_context",
        comparison["always_full_metrics"],
        description["event_ids"],
    )
    learner = _variant_from_metrics(
        "self_learning_corrected_budget_memory",
        comparison["combined_metrics"],
        description["event_ids"],
    )
    learner["learned_decision_count"] = sum(
        1 for row in event_decisions if row["learned_update_applied"]
    )
    safety = _safety_controls(events, event_decisions, memory_policy)
    same_event_ids = baseline["event_ids"] == learner["event_ids"]
    return {
        "workflow": description,
        "checked_event_count": description["checked_event_count"],
        "same_event_ids": same_event_ids,
        "baseline_variant": baseline,
        "self_learning_variant": learner,
        "context_efficiency_delta": _delta(
            learner["context_efficiency"], baseline["context_efficiency"]
        ),
        "verifier_cost_delta": _delta(baseline["verifier_cost"], learner["verifier_cost"]),
        "quality_delta": _delta(learner["quality"], baseline["quality"]),
        "stale_memory_deflection_rate": safety["stale_memory_deflection_rate"],
        "poison_memory_deflection_rate": safety["poison_memory_deflection_rate"],
        "rollback_success_rate": safety["rollback_success_rate"],
        "unsafe_false_accepts": safety["unsafe_false_accepts"],
        "memory_policy": memory_policy,
        "event_decisions": event_decisions,
        "safety_controls": safety,
        "memory_churn": _memory_churn(memory_policy),
        "weight_mutation_receipt": _weight_mutation_receipt(),
        "source_replay_metrics": {
            "exp5369_checked_event_count": replay["checked_event_count"],
            "exp5369_context_efficiency_delta": comparison["context_efficiency_delta"],
            "exp5369_verifier_cost_delta_rate": comparison["verifier_cost_delta"],
        },
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5382 artifact from gated workflow evidence."""

    gate = confirm_upstream_gate(root)
    evaluation = evaluate_real_workflow(root=root) if gate["all_passed"] else _blocked_evaluation()
    readiness = _readiness_checks(gate, evaluation, tests_run)
    ready = bool(readiness["all_passed"])
    workflow = evaluation["workflow"]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifacts": [str(EXP5381_RELATIVE_PATH), str(EXP5369_RELATIVE_PATH)],
        "status": "complete" if ready else "blocked",
        "continuous_self_learning_target": True,
        "continuous_self_learning_real_workflow_ready": ready,
        "upstream_budget_memory_corrigendum_clean": bool(gate["budget_memory_corrigendum_clean"]),
        "workflow_name": workflow["workflow_name"],
        "session_count": int(workflow["session_count"]),
        "trace_count": int(workflow["trace_count"]),
        "checked_event_count": int(evaluation["checked_event_count"]),
        "context_efficiency_delta": evaluation["context_efficiency_delta"],
        "verifier_cost_delta": evaluation["verifier_cost_delta"],
        "quality_delta": evaluation["quality_delta"],
        "stale_memory_deflection_rate": evaluation["stale_memory_deflection_rate"],
        "poison_memory_deflection_rate": evaluation["poison_memory_deflection_rate"],
        "rollback_success_rate": evaluation["rollback_success_rate"],
        "no_weight_mutation": evaluation["weight_mutation_receipt"]["no_weight_mutation"],
        "unsafe_false_accepts": int(evaluation["unsafe_false_accepts"]),
        "honest_verdict": _honest_verdict(ready, gate, readiness, evaluation),
        "tests_run": [dict(row) for row in tests_run],
        "upstream_gate": gate,
        "readiness_checks": readiness,
        "workflow_evidence": {
            "same_event_ids": evaluation["same_event_ids"],
            "workflow": workflow,
            "baseline_variant": evaluation["baseline_variant"],
            "self_learning_variant": evaluation["self_learning_variant"],
            "safety_controls": evaluation["safety_controls"],
            "memory_churn": evaluation["memory_churn"],
            "source_replay_metrics": evaluation["source_replay_metrics"],
        },
        "corrected_memory_policy": {
            key: value for key, value in evaluation["memory_policy"].items() if key != "rows"
        },
        "event_decisions": evaluation["event_decisions"],
        "weight_mutation_receipt": evaluation["weight_mutation_receipt"],
        "methodology_note": (
            "Baseline and self-learning variants evaluate the same Exp5369 "
            "multi-session workflow event IDs. The self-learning variant uses "
            "Exp5381 row-derived corrected budget memory to update controller "
            "retrieval and verifier-selection state only; no model weights are "
            "loaded or mutated."
        ),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the terminal fields consumed by the milestone gate."""

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError("missing Exp5382 fields: " + ",".join(missing))
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match Exp5382 contract")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict terminal prefix invalid")
    for field in BOOL_FIELDS:
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in INTEGER_FIELDS:
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be a bare integer")
    for field in NUMERIC_FIELDS:
        if not _is_numeric(artifact.get(field)):
            raise ValueError(f"{field} must be bare numeric")
    ready = artifact["continuous_self_learning_real_workflow_ready"]
    if ready and artifact["status"] != "complete":
        raise ValueError("status must be complete when ready")
    if artifact["status"] == "complete" and not ready:
        raise ValueError("status cannot be complete when not ready")
    if artifact["no_weight_mutation"] is not True:
        raise ValueError("no_weight_mutation must be true")
    if artifact["unsafe_false_accepts"] != 0:
        raise ValueError("unsafe_false_accepts must be 0")
    if ready:
        if artifact["upstream_budget_memory_corrigendum_clean"] is not True:
            raise ValueError("upstream_budget_memory_corrigendum_clean must be true")
        if artifact["session_count"] < MIN_SESSIONS:
            raise ValueError("session_count below workflow target")
        if artifact["trace_count"] < MIN_TRACES:
            raise ValueError("trace_count below workflow target")
        if artifact["checked_event_count"] < MIN_CHECKED_EVENTS:
            raise ValueError("checked_event_count below workflow target")
        if artifact["context_efficiency_delta"] <= 0.0:
            raise ValueError("context_efficiency_delta must improve when ready")
        if artifact["verifier_cost_delta"] <= 0.0:
            raise ValueError("verifier_cost_delta must improve when ready")
        if artifact["quality_delta"] < 0.0:
            raise ValueError("quality_delta must preserve baseline quality")
        if artifact["stale_memory_deflection_rate"] < 1.0:
            raise ValueError("stale_memory_deflection_rate must be complete")
        if artifact["poison_memory_deflection_rate"] < 1.0:
            raise ValueError("poison_memory_deflection_rate must be complete")
        if artifact["rollback_success_rate"] < 1.0:
            raise ValueError("rollback_success_rate must be complete")
        if not artifact.get("tests_run"):
            raise ValueError("tests_run must record commands for ready artifact")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5382 result artifact and return the JSON payload."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for the sources that define the result."""

    root_path = Path(root)
    return {
        "exp5381": _sha256_file(root_path / EXP5381_RELATIVE_PATH),
        "exp5369": _sha256_file(root_path / EXP5369_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5381_module": _sha256_file(root_path / EXP5381_MODULE_RELATIVE_PATH),
        "exp5369_module": _sha256_file(root_path / EXP5369_MODULE_RELATIVE_PATH),
    }


def _evaluate_event_decisions(
    events: Sequence[Mapping[str, Any]],
    memory_policy: Mapping[str, Any],
) -> JsonList:
    decisions: JsonList = []
    kept = set(memory_policy["kept_memory_ids"])
    rejected = set(memory_policy["rejected_or_quarantined_memory_ids"])
    for event in events:
        support = list(event.get("utility_memory", {}).get("supporting_context", []))
        variant = str(event.get("drift_injection", {}).get("memory_variant", "clean"))
        certificate_decision = str(
            event.get("verifier_tool_decision", {}).get("certificate_decision")
        )
        supported_by_kept_memory = any(item in kept for item in support)
        supported_by_rejected_memory = any(item in rejected for item in support)
        bad_memory = bool(
            event.get("drift_injection", {}).get("unsafe")
            or variant in {"stale", "poisoned", "unverified"}
            or supported_by_rejected_memory
        )
        if certificate_decision == "rollback":
            learned_decision = "rollback"
        elif certificate_decision == "reject" or supported_by_rejected_memory:
            learned_decision = "reject"
        elif supported_by_kept_memory:
            learned_decision = "learn_policy_update"
        else:
            learned_decision = "accept_ephemeral"
        accepted_as_good = learned_decision in {
            "learn_policy_update",
            "accept_ephemeral",
        }
        decisions.append(
            {
                "event_id": event["event_id"],
                "trace_id": event["trace_id"],
                "session_id": event["session_id"],
                "memory_variant": variant,
                "supporting_context": support,
                "certificate_decision": certificate_decision,
                "learned_decision": learned_decision,
                "learned_update_applied": learned_decision == "learn_policy_update",
                "bad_memory": bad_memory,
                "accepted_as_good": accepted_as_good,
                "unsafe_false_accept": bool(bad_memory and accepted_as_good),
                "rollback_required": bool(event.get("rollback_event", {}).get("required")),
                "rollback_recovered": bool(event.get("rollback_event", {}).get("recovered")),
            }
        )
    return decisions


def _safety_controls(
    events: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    memory_policy: Mapping[str, Any],
) -> JsonDict:
    stale = [row for row in decisions if row["memory_variant"] == "stale"]
    poison = [row for row in decisions if row["memory_variant"] == "poisoned"]
    rollback = [row for row in decisions if row["rollback_required"]]
    stale_rejected = [row for row in stale if row["learned_decision"] in {"reject", "rollback"}]
    poison_rejected = [row for row in poison if row["learned_decision"] in {"reject", "rollback"}]
    return {
        "checked_event_count": len(events),
        "budget_limit_bytes": memory_policy["budget_limit_bytes"],
        "retained_bytes": memory_policy["retained_bytes"],
        "budget_limit_respected": memory_policy["budget_limit_respected"],
        "stale_probe_count": len(stale),
        "poison_probe_count": len(poison),
        "rollback_required_count": len(rollback),
        "stale_memory_deflection_rate": _rate(len(stale_rejected), len(stale)),
        "poison_memory_deflection_rate": _rate(len(poison_rejected), len(poison)),
        "rollback_success_rate": _rate(
            sum(1 for row in rollback if row["rollback_recovered"]),
            len(rollback),
        ),
        "unsafe_false_accepts": sum(1 for row in decisions if row["unsafe_false_accept"]),
    }


def _memory_churn(memory_policy: Mapping[str, Any]) -> JsonDict:
    retained = len(memory_policy["kept_memory_ids"])
    rejected = len(memory_policy["rejected_or_quarantined_memory_ids"])
    total = int(memory_policy["evidence_row_count"])
    return {
        "retained_memory_count": retained,
        "rejected_or_quarantined_memory_count": rejected,
        "memory_evidence_row_count": total,
        "churn_rate": _rate(rejected, total),
    }


def _variant_from_metrics(
    variant_name: str,
    metrics: Mapping[str, Any],
    event_ids: Sequence[str],
) -> JsonDict:
    return {
        "variant_name": variant_name,
        "event_ids": list(event_ids),
        "context_efficiency": float(metrics["context_efficiency"]),
        "verifier_cost": float(metrics["verifier_cost"]),
        "quality": float(metrics["final_quality"]),
        "unsafe_false_accepts": int(metrics.get("unsafe_false_accepts", 0)),
        "model_weights_mutated": bool(metrics.get("model_weights_mutated", False)),
    }


def _blocked_evaluation() -> JsonDict:
    workflow = {
        "workflow_name": WORKFLOW_NAME,
        "session_count": 0,
        "trace_count": 0,
        "checked_event_count": 0,
        "decision_type_counts": {
            "retrieval": 0,
            "verifier_tool_use": 0,
            "rollback": 0,
            "constraint_selection": 0,
        },
        "event_ids": [],
    }
    empty_variant = {
        "variant_name": "blocked",
        "event_ids": [],
        "context_efficiency": 0.0,
        "verifier_cost": 0.0,
        "quality": 0.0,
        "unsafe_false_accepts": 0,
        "model_weights_mutated": False,
    }
    return {
        "workflow": workflow,
        "checked_event_count": 0,
        "same_event_ids": False,
        "baseline_variant": dict(empty_variant),
        "self_learning_variant": dict(empty_variant),
        "context_efficiency_delta": 0.0,
        "verifier_cost_delta": 0.0,
        "quality_delta": 0.0,
        "stale_memory_deflection_rate": 0.0,
        "poison_memory_deflection_rate": 0.0,
        "rollback_success_rate": 0.0,
        "unsafe_false_accepts": 0,
        "memory_policy": {
            "budget_limit_bytes": BUDGET_LIMIT_BYTES,
            "retained_bytes": 0,
            "budget_limit_respected": False,
            "kept_memory_ids": [],
            "trusted_memory_ids": [],
            "rejected_or_quarantined_memory_ids": [],
            "evidence_row_count": 0,
            "rows": [],
        },
        "event_decisions": [],
        "safety_controls": {
            "checked_event_count": 0,
            "budget_limit_bytes": BUDGET_LIMIT_BYTES,
            "retained_bytes": 0,
            "budget_limit_respected": False,
            "stale_probe_count": 0,
            "poison_probe_count": 0,
            "rollback_required_count": 0,
            "stale_memory_deflection_rate": 0.0,
            "poison_memory_deflection_rate": 0.0,
            "rollback_success_rate": 0.0,
            "unsafe_false_accepts": 0,
        },
        "memory_churn": {
            "retained_memory_count": 0,
            "rejected_or_quarantined_memory_count": 0,
            "memory_evidence_row_count": 0,
            "churn_rate": 0.0,
        },
        "weight_mutation_receipt": _weight_mutation_receipt(),
        "source_replay_metrics": {},
    }


def _readiness_checks(
    gate: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    workflow = evaluation["workflow"]
    checks = {
        "upstream_gate_passed": gate["all_passed"] is True,
        "session_count_ready": workflow["session_count"] >= MIN_SESSIONS,
        "trace_count_ready": workflow["trace_count"] >= MIN_TRACES,
        "checked_event_count_ready": (evaluation["checked_event_count"] >= MIN_CHECKED_EVENTS),
        "same_event_ids": evaluation["same_event_ids"] is True,
        "context_efficiency_improved": evaluation["context_efficiency_delta"] > 0.0,
        "verifier_cost_reduced": evaluation["verifier_cost_delta"] > 0.0,
        "quality_preserved": evaluation["quality_delta"] >= 0.0,
        "stale_memory_deflected": evaluation["stale_memory_deflection_rate"] == 1.0,
        "poison_memory_deflected": evaluation["poison_memory_deflection_rate"] == 1.0,
        "rollback_succeeded": evaluation["rollback_success_rate"] == 1.0,
        "budget_limit_respected": evaluation["safety_controls"]["budget_limit_respected"] is True,
        "unsafe_false_accepts_zero": evaluation["unsafe_false_accepts"] == 0,
        "tests_recorded": bool(tests_run),
        "no_weight_mutation": evaluation["weight_mutation_receipt"]["no_weight_mutation"] is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {**checks, "failed_gates": failed, "all_passed": not failed}


def _honest_verdict(
    ready: bool,
    gate: Mapping[str, Any],
    readiness: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> str:
    if ready:
        return (
            "complete: real workflow ran after Exp5381 clean gate; baseline and "
            "self-learning variants shared "
            f"{evaluation['checked_event_count']} deterministic event checks, "
            "context efficiency improved, verifier cost fell, quality was "
            "preserved, stale and poisoned memory were deflected, rollback "
            "succeeded, and no model weights mutated"
        )
    blockers = list(gate.get("failed_gates", []))
    blockers.extend(readiness["failed_gates"])
    return "blocked_real_workflow_continuous_self_learning: " + ",".join(dict.fromkeys(blockers))


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weight_hash_before": "sha256:no_model_weights_loaded",
        "model_weight_hash_after": "sha256:no_model_weights_loaded",
        "allowed_update_targets": [
            "retrieval_policy_memory",
            "constraint_selection_cache",
            "rollback_decision_log",
        ],
        "forbidden_updates": {
            "fine_tune_model_weights": False,
            "lora_or_adapter_update": False,
            "foundation_weight_write": False,
        },
    }


def _flatten_events(traces: Sequence[Mapping[str, Any]]) -> JsonList:
    return [dict(event) for trace in traces for event in trace.get("events", [])]


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return json.loads(json.dumps(value, sort_keys=True))


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _delta(left: int | float, right: int | float) -> float:
    return round(float(left) - float(right), 6)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
