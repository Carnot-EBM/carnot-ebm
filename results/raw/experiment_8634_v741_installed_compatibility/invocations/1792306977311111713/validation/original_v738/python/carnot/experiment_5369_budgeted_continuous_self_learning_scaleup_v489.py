"""Exp5369: budgeted continuous self-learning scale-up.

Spec refs: REQ-LEARN-5369, SCENARIO-LEARN-5369-GATE,
SCENARIO-LEARN-5369-SCALE, SCENARIO-LEARN-5369-SAFETY-COST.

This module is a deterministic replay fixture for the milestone self-learning
slot. It gates on Exp5368, expands the dependency/drift replay to 12 sessions,
applies budget-curated memory decisions, compares always-full and no-memory
baselines, and records that no model weights are loaded or mutated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5357_dependency_drift_self_learning_scaleup_v488 as exp5357
from carnot import experiment_5368_budget_curated_memory_governance_v489 as exp5368
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5369_budgeted_continuous_self_learning_scaleup_v489"
EXPERIMENT_ID = 5369
MILESTONE = "v489"
SCHEMA = "carnot.experiment_5369.budgeted_continuous_self_learning_scaleup.v489"
RUN_DATE = "2026-07-07"
RANDOM_SEED = 5369
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5369_budgeted_continuous_self_learning_scaleup_v489.json"
)
EXP5357_RELATIVE_PATH = exp5357.RESULT_RELATIVE_PATH
EXP5368_RELATIVE_PATH = exp5368.RESULT_RELATIVE_PATH
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5369_budgeted_continuous_self_learning_scaleup_v489.py"
)
EXP5357_MODULE_RELATIVE_PATH = exp5357.MODULE_RELATIVE_PATH
EXP5368_MODULE_RELATIVE_PATH = exp5368.MODULE_RELATIVE_PATH

INFERENCE_SUBSTRATE = "deterministic_budgeted_continuous_self_learning_scaleup"
SPEC_REFS = (
    "REQ-LEARN-5369",
    "SCENARIO-LEARN-5369-GATE",
    "SCENARIO-LEARN-5369-SCALE",
    "SCENARIO-LEARN-5369-SAFETY-COST",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")

MIN_MULTI_SESSION_TRACES = 12
MIN_CHECKED_EVENTS = 30

REQUIRED_FIELD_PRINCIPLES = {
    "experiment_id": "Stable id ties the artifact to this roadmap task.",
    "milestone": ("Keeps the scale-up tied to the `.489` budget-curated memory milestone."),
    "status": "Complete only if the gated scale-up runs.",
    "honest_verdict": "One-line ready or blocked verdict with a terminal prefix.",
    "inference_substrate": (
        "Expected value is deterministic_budgeted_continuous_self_learning_scaleup."
    ),
    "budget_curated_memory_ready": "Copied from the Exp5368 gate source.",
    "continuous_self_learning_target": (
        "Bare boolean must be true to mark this as the required milestone self-learning slot."
    ),
    "continuous_self_learning_budget_scaleup_ready": (
        "Bare boolean true only when dependency, drift, rollback, budget, and "
        "quality metrics are all present."
    ),
    "multi_session_trace_count": "Number of traces evaluated.",
    "checked_event_count": "Number of events checked by verifier or fixture.",
    "context_efficiency_delta": "Context reduction versus always-full baseline.",
    "verifier_cost_delta": (
        "Verifier-cost reduction versus always-full or always-verify baseline."
    ),
    "quality_delta_vs_always_full": ("Task-quality delta versus always-full baseline."),
    "dependency_attribution_rate": (
        "Fraction of selected memories with traceable dependency edges."
    ),
    "drift_detection_rate": "Fraction of induced drift cases detected.",
    "rollback_recovery_rate": ("Fraction of bad-memory cases recovered after rollback."),
    "stale_memory_deflection_rate": ("Fraction of stale memories rejected or quarantined."),
    "poison_memory_deflection_rate": ("Fraction of poisoned memories rejected or quarantined."),
    "retained_bytes_delta": "Bytes saved versus uncurated memory.",
    "unsafe_false_accepts": ("Count of harmful, stale, or poisoned memories accepted as useful."),
    "no_weight_mutation": "Must be true.",
    "tests_run": "Lists deterministic replay, coverage, and pytest commands.",
}
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "tests_run",
)
BARE_BOOL_FIELDS = (
    "budget_curated_memory_ready",
    "continuous_self_learning_target",
    "continuous_self_learning_budget_scaleup_ready",
    "no_weight_mutation",
)
BARE_INTEGER_FIELDS = (
    "multi_session_trace_count",
    "checked_event_count",
    "retained_bytes_delta",
    "unsafe_false_accepts",
)
BARE_NUMERIC_FIELDS = (
    "context_efficiency_delta",
    "verifier_cost_delta",
    "quality_delta_vs_always_full",
    "dependency_attribution_rate",
    "drift_detection_rate",
    "rollback_recovery_rate",
    "stale_memory_deflection_rate",
    "poison_memory_deflection_rate",
)


def confirm_source_gate(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load Exp5368 and expose the exact gate used by this scale-up."""

    source = _read_json(Path(root) / EXP5368_RELATIVE_PATH)
    exp5368_status = _wrapped_value(source.get("status"))
    checks = {
        "budget_curated_memory_ready": (source.get("budget_curated_memory_ready") is True),
        "exp5368_status_ready": exp5368_status == "budget_curated_memory_ready",
        "source_unsafe_false_accepts_zero": source.get("unsafe_false_accepts") == 0,
        "source_no_weight_mutation": source.get("no_weight_mutation") is True,
        "source_stale_deflection_ready": (source.get("stale_memory_deflection_rate") == 1.0),
        "source_poison_deflection_ready": (source.get("poison_memory_deflection_rate") == 1.0),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "exp5368_status": exp5368_status,
        "all_passed": not failed,
        "failed_gates": failed,
        "source_artifacts": [str(EXP5368_RELATIVE_PATH)],
        "exp5368_honest_verdict": _wrapped_value(source.get("honest_verdict")),
    }


def build_budgeted_multi_session_traces() -> JsonList:
    """Build 12 deterministic traces from dependency, drift, and rollback lanes."""

    trace_builders = (
        _clean_trace,
        _stale_rollback_trace,
        _poison_rollback_trace,
        _dependency_fault_trace,
    )
    traces: JsonList = []
    for index in range(1, MIN_MULTI_SESSION_TRACES + 1):
        builder = trace_builders[(index - 1) % len(trace_builders)]
        traces.append(builder(index))
    return traces


def evaluate_budgeted_loop(
    *,
    traces: Sequence[Mapping[str, Any]] | None = None,
    root: Path | str = REPO_ROOT,
) -> JsonDict:
    """Apply provenance, drift, rollback, budget curation, and baselines."""

    trace_rows = list(traces) if traces is not None else build_budgeted_multi_session_traces()
    hash_chain = exp5357.validate_hash_chains(trace_rows)
    provenance = exp5357.evaluate_trace_provenance(trace_rows)
    policy = exp5357.evaluate_policy_comparison(trace_rows)
    budget_curation = evaluate_budget_curation()
    no_memory = evaluate_no_memory_baseline(trace_rows)
    combined = policy["policy_metrics"][exp5357.COMBINED_POLICY]
    always = policy["policy_metrics"][exp5357.ALWAYS_FULL_POLICY]
    event_ids = [row["event_id"] for row in policy["policy_rows"][exp5357.COMBINED_POLICY]]
    same_event_ids = bool(policy["same_event_ids"] and event_ids == no_memory["event_ids"])
    quality_delta = _delta(combined["final_quality"], always["final_quality"])
    context_delta = _rate(
        always["active_token_count"] - combined["active_token_count"],
        always["active_token_count"],
    )
    verifier_delta = _rate(
        always["verifier_cost"] - combined["verifier_cost"],
        always["verifier_cost"],
    )
    unsafe_false_accepts = int(
        combined.get("unsafe_false_accepts", 0) + budget_curation.get("unsafe_false_accepts", 0)
    )
    return {
        "multi_session_trace_count": len(trace_rows),
        "checked_event_count": int(hash_chain["checked_event_count"]),
        "hash_chain": hash_chain,
        "trace_provenance": provenance,
        "budget_curation": budget_curation,
        "unsafe_false_accepts": unsafe_false_accepts,
        "policy_comparison": {
            "same_event_ids": same_event_ids,
            "baselines_compared": {
                "always_full_context": exp5357.ALWAYS_FULL_POLICY in policy["policy_metrics"],
                "no_memory": no_memory["event_count"] == len(event_ids),
            },
            "quality_delta_vs_always_full": quality_delta,
            "context_efficiency_delta": context_delta,
            "verifier_cost_delta": verifier_delta,
            "combined_metrics": combined,
            "always_full_metrics": always,
            "no_memory_metrics": no_memory,
            "all_policy_metrics": policy["policy_metrics"],
        },
        "joint_controls_applied": {
            "dependency_provenance": provenance["dependency_attribution_rate"] > 0.0,
            "memory_drift_detection": provenance["drift_detection_rate"] > 0.0,
            "rollback": provenance["rollback_recovery_rate"] == 1.0,
            "budget_curation": budget_curation["retained_bytes_delta"] > 0,
        },
        "source_artifacts_consulted": [
            str(EXP5357_RELATIVE_PATH),
            str(EXP5368_RELATIVE_PATH),
        ],
        "root": str(root),
    }


def evaluate_budget_curation() -> JsonDict:
    """Run Exp5368 curation and add the uncurated-byte savings metric."""

    curation = exp5368.curate_memory_items(exp5368.build_memory_items())
    uncurated_bytes = sum(int(row["byte_cost"]) for row in curation["decision_rows"])
    retained_bytes_delta = uncurated_bytes - int(curation["retained_bytes"])
    return {
        **curation,
        "uncurated_bytes": uncurated_bytes,
        "retained_bytes_delta": retained_bytes_delta,
    }


def evaluate_no_memory_baseline(
    traces: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Score the same event IDs when no cross-session memory is available."""

    events = _flatten_events(traces)
    trace_count = len({event["trace_id"] for event in events})
    failed_trace_ids = {
        str(event["trace_id"]) for event in events if _no_memory_misses_required_context(event)
    }
    return {
        "policy": "no_memory",
        "event_count": len(events),
        "event_ids": [str(event["event_id"]) for event in events],
        "active_token_count": 0,
        "verifier_cost": 0,
        "unsafe_false_accepts": 0,
        "failed_trace_count": len(failed_trace_ids),
        "final_quality": _rate(trace_count - len(failed_trace_ids), trace_count),
        "model_weights_mutated": False,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5369 result artifact from deterministic evidence."""

    source_gate = confirm_source_gate(root=root)
    if source_gate["all_passed"]:
        traces = build_budgeted_multi_session_traces()
        replay = evaluate_budgeted_loop(traces=traces, root=root)
    else:
        traces = []
        replay = _blocked_replay()
    readiness_gate = _readiness_gate(source_gate, replay, tests_run)
    complete = bool(readiness_gate["all_passed"])
    status = "complete" if complete else "blocked_budgeted_continuous_scaleup_gate"
    comparison = replay["policy_comparison"]
    provenance = replay["trace_provenance"]
    budget = replay["budget_curation"]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [str(EXP5368_RELATIVE_PATH), str(EXP5357_RELATIVE_PATH)],
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(complete, source_gate, replay, readiness_gate),
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "budget_curated_memory_ready": bool(source_gate["budget_curated_memory_ready"]),
        "continuous_self_learning_target": True,
        "continuous_self_learning_budget_scaleup_ready": complete,
        "multi_session_trace_count": int(replay["multi_session_trace_count"]),
        "checked_event_count": int(replay["checked_event_count"]),
        "context_efficiency_delta": comparison["context_efficiency_delta"],
        "verifier_cost_delta": comparison["verifier_cost_delta"],
        "quality_delta_vs_always_full": comparison["quality_delta_vs_always_full"],
        "dependency_attribution_rate": provenance["dependency_attribution_rate"],
        "drift_detection_rate": provenance["drift_detection_rate"],
        "rollback_recovery_rate": provenance["rollback_recovery_rate"],
        "stale_memory_deflection_rate": budget["stale_memory_deflection_rate"],
        "poison_memory_deflection_rate": budget["poison_memory_deflection_rate"],
        "retained_bytes_delta": int(budget["retained_bytes_delta"]),
        "unsafe_false_accepts": int(replay["unsafe_false_accepts"]),
        "no_weight_mutation": bool(source_gate["source_no_weight_mutation"]),
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "source_gate": source_gate,
        "readiness_gate": readiness_gate,
        "multi_session_traces": traces,
        "hash_chain": replay["hash_chain"],
        "trace_provenance": provenance,
        "budget_curation": budget,
        "policy_comparison": comparison,
        "joint_controls_applied": replay["joint_controls_applied"],
        "fixture_scale_limitation": (
            "none: deterministic expansion produced "
            f"{replay['multi_session_trace_count']} traces and "
            f"{replay['checked_event_count']} checked events"
        ),
        "weight_mutation_receipt": _weight_mutation_receipt(),
        "methodology_note": (
            "Exp5369 is deterministic fixture evidence. It copies the Exp5368 "
            "budget gate, expands Exp5357-style dependency/drift traces to the "
            "milestone scale target, applies Exp5368 curation rows, compares "
            "always-full and no-memory baselines, and performs no model-weight "
            "mutation."
        ),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the terminal fields consumed by the milestone reconciler."""

    for field in WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if (
            not isinstance(wrapped, Mapping)
            or "value" not in wrapped
            or wrapped.get("principle") != REQUIRED_FIELD_PRINCIPLES[field]
        ):
            raise ValueError(f"{field} must be principle-wrapped")
    if not str(artifact["honest_verdict"]["value"]).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict terminal prefix invalid")
    if artifact["inference_substrate"]["value"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    for field in BARE_BOOL_FIELDS:
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in BARE_INTEGER_FIELDS:
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be a bare integer")
    for field in BARE_NUMERIC_FIELDS:
        if not _is_numeric(artifact.get(field)):
            raise ValueError(f"{field} must be bare numeric")
    if artifact.get("quality_delta_vs_always_full", 0.0) < 0.0:
        raise ValueError("quality_delta_vs_always_full must be nonnegative")
    if artifact.get("unsafe_false_accepts") != 0:
        raise ValueError("unsafe_false_accepts must be 0")
    if artifact.get("no_weight_mutation") is not True:
        raise ValueError("no_weight_mutation must be true")
    ready = artifact["continuous_self_learning_budget_scaleup_ready"]
    status_value = artifact["status"]["value"]
    if ready and status_value != "complete":
        raise ValueError("status must be complete when ready")
    if status_value == "complete" and not ready:
        raise ValueError("status cannot be complete when not ready")
    if ready:
        if not artifact["tests_run"]["value"]:
            raise ValueError("tests_run must record commands for ready scale-up")
        if artifact["budget_curated_memory_ready"] is not True:
            raise ValueError("budget_curated_memory_ready must be true when ready")
        if artifact["multi_session_trace_count"] < MIN_MULTI_SESSION_TRACES:
            raise ValueError("multi_session_trace_count below scale target")
        if artifact["checked_event_count"] < MIN_CHECKED_EVENTS:
            raise ValueError("checked_event_count below scale target")
        if artifact["context_efficiency_delta"] <= 0.0:
            raise ValueError("context_efficiency_delta must improve when ready")
        if artifact["verifier_cost_delta"] <= 0.0:
            raise ValueError("verifier_cost_delta must improve when ready")
        if artifact["retained_bytes_delta"] <= 0:
            raise ValueError("retained_bytes_delta must be positive when ready")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5369 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for the source artifacts, spec, and modules."""

    root_path = Path(root)
    return {
        "exp5357": _sha256_file(root_path / EXP5357_RELATIVE_PATH),
        "exp5368": _sha256_file(root_path / EXP5368_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5357_module": _sha256_file(root_path / EXP5357_MODULE_RELATIVE_PATH),
        "exp5368_module": _sha256_file(root_path / EXP5368_MODULE_RELATIVE_PATH),
    }


def _clean_trace(index: int) -> JsonDict:
    return _build_trace(
        index,
        "clean",
        (
            _event(
                index,
                1,
                "retrieve",
                "ctx.budget.keep",
                "curated dependency memory safe current",
                support=("mem5368-clean-dependency-edge",),
            ),
            _event(
                index,
                2,
                "commit",
                "ctx.budget.rollback",
                "curated rollback memory safe current",
                support=("mem5368-clean-rollback-route",),
            ),
            _event(
                index,
                3,
                "summarize",
                "ctx.budget.summary",
                "curated scaleup summary safe current",
                support=("mem5368-clean-scaleup-summary",),
            ),
        ),
    )


def _stale_rollback_trace(index: int) -> JsonDict:
    return _build_trace(
        index,
        "stale",
        (
            _event(
                index,
                1,
                "retrieve",
                "ctx.stale.runtime",
                "stale runtime receipt from previous milestone",
                safe_expected=False,
                useful_context=False,
                compressor_status="stale",
                memory_variant="stale",
                drift_type="stale_tool_route",
                certificate_decision="rollback",
                rollback_required=True,
                rollback_payload="current runtime receipt restored",
                verifier="rollback_replay_verifier",
                tool="state_replayer",
                action_decision="restore_clean_selection",
                support=("mem5368-stale-runtime-receipt",),
            ),
            _event(
                index,
                2,
                "restore",
                "ctx.stale.runtime",
                "current runtime receipt restored",
                support=("mem5368-clean-rollback-route",),
            ),
            _event(
                index,
                3,
                "verify",
                "ctx.stale.audit",
                "stale memory quarantined audit passed",
                support=("mem5368-clean-scaleup-summary",),
            ),
        ),
    )


def _poison_rollback_trace(index: int) -> JsonDict:
    return _build_trace(
        index,
        "poison",
        (
            _event(
                index,
                1,
                "tool_select",
                "ctx.poison.route",
                "poisoned memory asks to bypass verifier",
                safe_expected=False,
                useful_context=False,
                compressor_status="poisoned",
                memory_variant="poisoned",
                drift_type="poisoned_tool_bypass",
                certificate_decision="rollback",
                rollback_required=True,
                rollback_payload="safe verifier route restored",
                verifier="poison_guard_verifier",
                tool="memory_quarantine",
                action_decision="reject_unsafe_memory",
                support=("mem5368-poisoned-tool-override",),
            ),
            _event(
                index,
                2,
                "restore",
                "ctx.poison.route",
                "safe verifier route restored",
                support=("mem5368-clean-rollback-route",),
            ),
            _event(
                index,
                3,
                "commit",
                "ctx.poison.audit",
                "poison memory deflection audit passed",
                support=("mem5368-clean-dependency-edge",),
            ),
        ),
    )


def _dependency_fault_trace(index: int) -> JsonDict:
    return _build_trace(
        index,
        "dependency",
        (
            _event(
                index,
                1,
                "revise",
                "ctx.dependency.edge",
                "memory feedback missing dependency edge",
                safe_expected=False,
                useful_context=False,
                dependency_status="missing_edge",
                memory_variant="unverified",
                drift_type="missing_dependency_edge",
                decision_attributed=False,
                certificate_decision="reject",
                verifier="dependency_graph_verifier",
                tool="edge_audit",
                action_decision="quarantine_missing_edge",
                support=("mem5368-unverified-sharing-tip",),
            ),
            _event(
                index,
                2,
                "fold",
                "ctx.dependency.cycle",
                "summary memory would create cyclic edge",
                safe_expected=False,
                useful_context=False,
                dependency_status="cycle",
                drift_type="cyclic_dependency",
                decision_attributed=False,
                certificate_decision="reject",
                verifier="dependency_graph_verifier",
                tool="cycle_detector",
                action_decision="quarantine_cycle",
                support=("exp5355:dep-cyclic-dependency",),
            ),
            _event(
                index,
                3,
                "revise",
                "ctx.dependency.style",
                "benign concise style preference",
                memory_variant="biased",
                drift_type="benign_style_bias",
                drift_detected=False,
                verifier="utility_memory_verifier",
                tool="context_cache",
                action_decision="accept_style_preference",
                support=("operator:benign-style-note",),
            ),
        ),
    )


def _build_trace(
    index: int,
    trace_kind: str,
    raw_events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    raw_trace = {
        "trace_id": f"trace-5369-{index:02d}-{trace_kind}",
        "session_id": f"session-5369-{index:02d}",
        "events": list(raw_events),
    }
    return exp5357._build_trace(raw_trace)


def _event(
    trace_index: int,
    event_index: int,
    event_action: str,
    object_id: str,
    payload: str,
    **kwargs: Any,
) -> JsonDict:
    event_id = f"t{trace_index:02d}-e{event_index}-{event_action}"
    return exp5357._event_spec(
        event_id,
        event_action,
        object_id,
        event_index,
        payload,
        drift_detected=kwargs.pop("drift_detected", True)
        if kwargs.get("drift_type", "none") != "none"
        else kwargs.pop("drift_detected", False),
        **kwargs,
    )


def _blocked_replay() -> JsonDict:
    budget = {
        "decision_rows": [],
        "retained_bytes_delta": 0,
        "stale_memory_deflection_rate": 0.0,
        "poison_memory_deflection_rate": 0.0,
        "unsafe_false_accepts": 0,
    }
    comparison = {
        "same_event_ids": False,
        "baselines_compared": {"always_full_context": False, "no_memory": False},
        "quality_delta_vs_always_full": 0.0,
        "context_efficiency_delta": 0.0,
        "verifier_cost_delta": 0.0,
        "combined_metrics": {},
        "always_full_metrics": {},
        "no_memory_metrics": {},
        "all_policy_metrics": {},
    }
    return {
        "multi_session_trace_count": 0,
        "checked_event_count": 0,
        "hash_chain": {"valid": False, "checked_event_count": 0, "failures": []},
        "trace_provenance": {
            "event_count": 0,
            "dependency_attribution_rate": 0.0,
            "drift_detection_rate": 0.0,
            "rollback_recovery_rate": 0.0,
        },
        "budget_curation": budget,
        "unsafe_false_accepts": 0,
        "policy_comparison": comparison,
        "joint_controls_applied": {
            "dependency_provenance": False,
            "memory_drift_detection": False,
            "rollback": False,
            "budget_curation": False,
        },
    }


def _readiness_gate(
    source_gate: Mapping[str, Any],
    replay: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    comparison = replay["policy_comparison"]
    provenance = replay["trace_provenance"]
    budget = replay["budget_curation"]
    checks = {
        "source_gate_passed": source_gate["all_passed"] is True,
        "trace_count_ready": (replay["multi_session_trace_count"] >= MIN_MULTI_SESSION_TRACES),
        "checked_event_count_ready": (replay["checked_event_count"] >= MIN_CHECKED_EVENTS),
        "dependency_metric_present": provenance["dependency_attribution_rate"] > 0.0,
        "drift_metric_present": provenance["drift_detection_rate"] > 0.0,
        "rollback_metric_present": provenance["rollback_recovery_rate"] == 1.0,
        "budget_metric_present": budget["retained_bytes_delta"] > 0,
        "quality_preserved": comparison["quality_delta_vs_always_full"] >= 0.0,
        "context_reduced": comparison["context_efficiency_delta"] > 0.0,
        "verifier_cost_reduced": comparison["verifier_cost_delta"] > 0.0,
        "baselines_compared": all(comparison["baselines_compared"].values()),
        "same_event_ids": comparison["same_event_ids"] is True,
        "stale_deflected": budget["stale_memory_deflection_rate"] > 0.0,
        "poison_deflected": budget["poison_memory_deflection_rate"] > 0.0,
        "unsafe_false_accepts_zero": replay["unsafe_false_accepts"] == 0,
        "tests_recorded": bool(tests_run),
        "no_weight_mutation": source_gate["source_no_weight_mutation"] is True,
    }
    return {
        **checks,
        "failed_gates": [name for name, passed in checks.items() if not passed],
        "all_passed": all(checks.values()),
    }


def _honest_verdict(
    complete: bool,
    source_gate: Mapping[str, Any],
    replay: Mapping[str, Any],
    readiness_gate: Mapping[str, Any],
) -> str:
    if complete:
        return (
            "complete: budgeted continuous self-learning scale-up ran after "
            "Exp5368 gate, evaluated "
            f"{replay['multi_session_trace_count']} traces and "
            f"{replay['checked_event_count']} checked events, preserved "
            "always-full quality, reduced context and verifier cost, "
            "deflected stale and poisoned memory, recovered rollback, and "
            "preserved no model weight mutation"
        )
    blockers = list(source_gate.get("failed_gates", []))
    blockers.extend(readiness_gate["failed_gates"])
    return "blocked_budgeted_continuous_self_learning_not_ready: " + ",".join(
        dict.fromkeys(blockers)
    )


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weight_hash_before": "sha256:no_model_weights_loaded",
        "model_weight_hash_after": "sha256:no_model_weights_loaded",
        "state_mutation_targets": [
            "deterministic_budgeted_trace_rows",
            "deterministic_budget_curation_rows",
            "deterministic_policy_baseline_rows",
        ],
        "forbidden_updates": {
            "fine_tune_model_weights": False,
            "lora_or_adapter_update": False,
            "foundation_weight_write": False,
        },
    }


def _flatten_events(traces: Sequence[Mapping[str, Any]]) -> JsonList:
    return [dict(event) for trace in traces for event in trace.get("events", [])]


def _no_memory_misses_required_context(event: Mapping[str, Any]) -> bool:
    return bool(
        event["safe_expected"]
        and event["useful_context"]
        and event["certificate_gate"]["decision"] == "accept"
    )


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": REQUIRED_FIELD_PRINCIPLES[field]}


def _wrapped_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


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


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 6)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
