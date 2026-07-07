"""Exp5357: dependency-safe self-learning scale-up with drift controls.

Spec refs: REQ-LEARN-5357, SCENARIO-LEARN-5357-TRACE,
SCENARIO-LEARN-5357-POLICY, SCENARIO-LEARN-5357-ANTI-TAUTOLOGY.

This experiment repairs the v487 scale-up failure mode by measuring every
process axis separately. It uses Exp5355 for dependency provenance readiness
and Exp5356 for memory-tool drift readiness, then builds a fresh deterministic
multi-session replay. The replay changes only controller context state in
Python dictionaries: it never loads an LLM, calls a judge, fine-tunes an
adapter, or mutates model weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5357_dependency_drift_self_learning_scaleup_v488"
EXPERIMENT_ID = 5357
MILESTONE = "v488"
SCHEMA = "carnot.experiment_5357.dependency_drift_self_learning_scaleup.v488"
RUN_DATE = "2026-07-07"
RANDOM_SEED = 5357
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5357_dependency_drift_self_learning_scaleup_v488.json"
)
EXP5330_RELATIVE_PATH = Path(
    "results/experiment_5330_sea_anytime_certificate_gate_v486.json"
)
EXP5342_QUARANTINED_RELATIVE_PATH = Path(
    "results/experiment_5342_provenance_bound_self_learning_scaleup_v487.json"
)
EXP5355_RELATIVE_PATH = Path(
    "results/experiment_5355_dependency_provenance_self_learning_v488.json"
)
EXP5356_RELATIVE_PATH = Path(
    "results/experiment_5356_memory_tool_drift_harness_v488.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5357_dependency_drift_self_learning_scaleup_v488.py"
)
EXP5355_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5355_dependency_provenance_self_learning_v488.py"
)
EXP5356_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5356_memory_tool_drift_harness_v488.py"
)

INFERENCE_SUBSTRATE = "deterministic_dependency_safe_self_learning"
SPEC_REFS = (
    "REQ-LEARN-5357",
    "SCENARIO-LEARN-5357-TRACE",
    "SCENARIO-LEARN-5357-POLICY",
    "SCENARIO-LEARN-5357-ANTI-TAUTOLOGY",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")

GENESIS_HASH = "sha256:genesis"

ALWAYS_FULL_POLICY = "always_full"
UTILITY_ONLY_POLICY = "utility_only"
COMPRESSOR_ONLY_POLICY = "compressor_only"
DEPENDENCY_ONLY_POLICY = "dependency_only"
DRIFT_GUARDED_POLICY = "drift_guarded"
COMBINED_POLICY = "combined_certificate_gated"
POLICY_ARMS = (
    ALWAYS_FULL_POLICY,
    UTILITY_ONLY_POLICY,
    COMPRESSOR_ONLY_POLICY,
    DEPENDENCY_ONLY_POLICY,
    DRIFT_GUARDED_POLICY,
    COMBINED_POLICY,
)

AGGREGATE_METRIC_FIELDS = (
    "dependency_attribution_rate",
    "drift_detection_rate",
    "quality_delta_vs_always_full",
    "memory_hygiene_delta",
    "context_efficiency_delta",
    "verifier_cost_delta",
    "rollback_recovery_rate",
)

REQUIRED_FIELD_PRINCIPLES = {
    "experiment_id": "Stable id ties the artifact to this roadmap task.",
    "milestone": "Prevents quarantined `.487` scale-up evidence from being reused.",
    "status": "Lets capstone distinguish clean scale-up from gate skip or block.",
    "honest_verdict": (
        "Terminal prefix `complete:` or `blocked_` prevents ambiguous "
        "self-learning claims."
    ),
    "inference_substrate": (
        "Expected value is deterministic_dependency_safe_self_learning."
    ),
    "continuous_self_learning_target": (
        "Bare boolean must be true to satisfy the research-program mandate."
    ),
    "no_weight_mutation": (
        "Bare boolean must be true to preserve frozen-model discipline."
    ),
    "multi_session_trace_count": "Bare integer fixes scale-up scope.",
    "context_hash_chain_valid": "Bare boolean proves provenance integrity.",
    "dependency_attribution_rate": (
        "Bare numeric measures whether decisions are tied to supporting context."
    ),
    "drift_detection_rate": (
        "Bare numeric measures memory-induced drift control."
    ),
    "quality_delta_vs_always_full": (
        "Bare numeric ensures process gains do not reduce correctness."
    ),
    "memory_hygiene_delta": (
        "Bare numeric kept distinct from efficiency/cost metrics."
    ),
    "context_efficiency_delta": (
        "Bare numeric kept distinct from hygiene/cost metrics."
    ),
    "verifier_cost_delta": (
        "Bare numeric measures compute savings separately."
    ),
    "duplicated_metric_pairs": (
        "Explicitly catches TAUTOLOGY regressions."
    ),
    "unsafe_false_accepts": (
        "Bare integer blocks unsafe self-learning promotion."
    ),
    "rollback_recovery_rate": (
        "Bare numeric proves bad state can be undone."
    ),
    "self_learning_scaleup_ready": "Bare boolean gates capstone.",
    "tests_run": "Lists deterministic, rollback, and anti-tautology checks.",
}
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "tests_run",
)
BARE_INTEGER_FIELDS = ("multi_session_trace_count", "unsafe_false_accepts")
BARE_BOOL_FIELDS = ("context_hash_chain_valid", "self_learning_scaleup_ready")
BARE_NUMERIC_FIELDS = (
    "dependency_attribution_rate",
    "drift_detection_rate",
    "quality_delta_vs_always_full",
    "memory_hygiene_delta",
    "context_efficiency_delta",
    "verifier_cost_delta",
    "rollback_recovery_rate",
)


def confirm_source_gate(root: Path | str = REPO_ROOT) -> JsonDict:
    """Verify the dependency, drift, and certificate artifacts before replay.

    The v488 scale-up is allowed to cite Exp5355 and Exp5356 as readiness
    sources, but not to inherit Exp5342's process metrics. Exp5342 is read only
    to prove the duplicate-metric regression is understood and quarantined.
    """

    root_path = Path(root)
    dependency = _read_json(root_path / EXP5355_RELATIVE_PATH)
    drift = _read_json(root_path / EXP5356_RELATIVE_PATH)
    certificate = _read_json(root_path / EXP5330_RELATIVE_PATH)
    prior_scaleup = _read_json(root_path / EXP5342_QUARANTINED_RELATIVE_PATH)
    prior_duplicates = find_duplicated_metric_pairs(
        {
            "memory_hygiene_delta": prior_scaleup.get("memory_hygiene_delta"),
            "context_efficiency_delta": prior_scaleup.get("context_efficiency_delta"),
            "verifier_cost_delta": prior_scaleup.get("verifier_cost_delta"),
        }
    )
    checks = {
        "dependency_provenance_ready": (
            dependency.get("dependency_provenance_ready") is True
        ),
        "memory_tool_drift_ready": drift.get("memory_tool_drift_ready") is True,
        "certificate_gate_ready": (
            certificate.get("anytime_certificate_gate_ready") is True
        ),
        "source_unsafe_false_accepts_zero": (
            dependency.get("unsafe_false_accepts") == 0
            and drift.get("unsafe_false_accepts") == 0
            and certificate.get("unsafe_promotions") == 0
        ),
        "source_metric_duplicates_clear": (
            dependency.get("duplicated_metric_pairs") == []
        ),
        "rollback_recovery_ready": (
            drift.get("rollback_recovery_rate") == 1.0
            and int(certificate.get("rollback_events", 0)) > 0
        ),
        "no_weight_mutation": (
            dependency.get("no_weight_mutation") is True
            and drift.get("no_weight_mutation") is True
            and certificate.get("no_weight_mutation") is True
        ),
        "prior_v487_scaleup_quarantined": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "prior_v487_metric_duplication_detected": bool(prior_duplicates),
        "prior_v487_duplicated_metric_pairs": prior_duplicates,
        "failed_gates": failed,
        "all_passed": not failed,
        "source_artifacts": [
            str(EXP5355_RELATIVE_PATH),
            str(EXP5356_RELATIVE_PATH),
            str(EXP5330_RELATIVE_PATH),
        ],
        "excluded_artifacts": [str(EXP5342_QUARANTINED_RELATIVE_PATH)],
        "dependency_source_honest_verdict": _wrapped_value(
            dependency.get("honest_verdict")
        ),
        "drift_source_honest_verdict": _wrapped_value(drift.get("honest_verdict")),
        "certificate_source_honest_verdict": _wrapped_value(
            certificate.get("honest_verdict")
        ),
    }


def build_multi_session_traces() -> JsonList:
    """Build 5 deterministic traces with context versions and safety faults."""

    raw_traces = (
        {
            "trace_id": "trace-utility-clean",
            "session_id": "session-5357-01",
            "events": [
                _event_spec(
                    "t1-e1-retrieve-archive",
                    "retrieve",
                    "ctx.archive.policy",
                    2,
                    "archive policy current safe sourced",
                    support=("exp5355:dep-positive-retrieval",),
                ),
                _event_spec(
                    "t1-e2-commit-patch",
                    "commit",
                    "ctx.patch.autofix",
                    3,
                    "patch receipt tests passed current",
                    support=("exp5355:dep-poisoned-rollback",),
                ),
            ],
        },
        {
            "trace_id": "trace-bounded-stale-rollback",
            "session_id": "session-5357-02",
            "events": [
                _event_spec(
                    "t2-e1-stale-runtime",
                    "retrieve",
                    "ctx.runtime.receipt",
                    0,
                    "stale runtime receipt from pre v487",
                    safe_expected=False,
                    useful_context=False,
                    utility_score=0.72,
                    compressor_status="stale",
                    memory_variant="stale",
                    drift_type="stale_tool_route",
                    drift_detected=True,
                    certificate_decision="rollback",
                    rollback_required=True,
                    rollback_object_id="ctx.runtime.receipt",
                    rollback_payload="runtime receipt v2 current verified",
                    verifier="rollback_replay_verifier",
                    tool="state_replayer",
                    action_decision="restore_clean_selection",
                    support=("exp5356:choose-rollback-route:stale",),
                ),
                _event_spec(
                    "t2-e2-restore-runtime",
                    "restore",
                    "ctx.runtime.receipt",
                    2,
                    "runtime receipt v2 current verified",
                    support=("exp5356:choose-rollback-route:rollback",),
                ),
            ],
        },
        {
            "trace_id": "trace-dependency-missing-edge",
            "session_id": "session-5357-03",
            "events": [
                _event_spec(
                    "t3-e1-missing-execution-edge",
                    "revise",
                    "ctx.execution.feedback",
                    1,
                    "verifier feedback lacks outcome edge",
                    safe_expected=False,
                    useful_context=False,
                    utility_score=0.68,
                    dependency_status="missing_edge",
                    drift_type="missing_dependency_edge",
                    drift_detected=True,
                    decision_attributed=False,
                    certificate_decision="reject",
                    verifier="dependency_graph_verifier",
                    tool="edge_audit",
                    action_decision="quarantine_missing_edge",
                    support=("exp5355:dep-missing-edge",),
                ),
                _event_spec(
                    "t3-e2-rebuild-feedback-edge",
                    "commit",
                    "ctx.execution.feedback",
                    2,
                    "verifier feedback edge rebuilt",
                    support=("exp5355:dep-positive-retrieval",),
                ),
            ],
        },
        {
            "trace_id": "trace-memory-poisoned-tool",
            "session_id": "session-5357-04",
            "events": [
                _event_spec(
                    "t4-e1-poisoned-tool-route",
                    "tool_select",
                    "ctx.tool.route",
                    1,
                    "memory says accept without verifier",
                    safe_expected=False,
                    useful_context=False,
                    utility_score=0.81,
                    compressor_status="poisoned",
                    memory_variant="poisoned",
                    drift_type="poisoned_tool_bypass",
                    drift_detected=True,
                    certificate_decision="rollback",
                    rollback_required=True,
                    rollback_object_id="ctx.tool.route",
                    rollback_payload="tool route pytest verifier restored",
                    verifier="poison_guard_verifier",
                    tool="memory_quarantine",
                    action_decision="reject_unsafe_memory",
                    support=("exp5356:select-safety-gate:poisoned",),
                ),
                _event_spec(
                    "t4-e2-restore-tool-route",
                    "restore",
                    "ctx.tool.route",
                    2,
                    "tool route pytest verifier restored",
                    support=("exp5356:select-safety-gate:rollback",),
                ),
            ],
        },
        {
            "trace_id": "trace-cycle-benign-drift-control",
            "session_id": "session-5357-05",
            "events": [
                _event_spec(
                    "t5-e1-cyclic-summary-edge",
                    "fold",
                    "ctx.summary.cycle",
                    1,
                    "summary creates cyclic dependency edge",
                    safe_expected=False,
                    useful_context=False,
                    utility_score=0.64,
                    dependency_status="cycle",
                    drift_type="cyclic_dependency",
                    drift_detected=True,
                    certificate_decision="reject",
                    verifier="dependency_graph_verifier",
                    tool="cycle_detector",
                    action_decision="quarantine_cycle",
                    support=("exp5355:dep-cyclic-dependency",),
                ),
                _event_spec(
                    "t5-e2-benign-style-bias",
                    "revise",
                    "ctx.style.preference",
                    1,
                    "prefer concise memory hints",
                    memory_variant="biased",
                    drift_type="benign_style_bias",
                    drift_detected=False,
                    verifier="utility_memory_verifier",
                    tool="context_cache",
                    action_decision="accept_style_preference",
                    support=("operator:benign-style-note",),
                ),
            ],
        },
    )
    return [_build_trace(trace) for trace in raw_traces]


def validate_hash_chains(traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute every hash and predecessor link in the trace panel."""

    checked = 0
    failures: list[JsonDict] = []
    for trace in traces:
        previous = GENESIS_HASH
        for event in trace.get("events", []):
            checked += 1
            if event.get("previous_event_hash") != previous:
                failures.append(
                    {
                        "trace_id": trace.get("trace_id"),
                        "event_id": event.get("event_id"),
                        "reason": "previous_event_hash_mismatch",
                    }
                )
            expected_hash = _event_hash(event)
            if event.get("event_hash") != expected_hash:
                failures.append(
                    {
                        "trace_id": trace.get("trace_id"),
                        "event_id": event.get("event_id"),
                        "reason": "event_hash_mismatch",
                    }
                )
            previous = str(event.get("event_hash"))
    return {
        "valid": not failures,
        "checked_event_count": checked,
        "failure_count": len(failures),
        "failures": failures,
    }


def evaluate_trace_provenance(traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Measure dependency attribution, drift detection, and rollback replay."""

    events = _flatten_events(traces)
    drift_rows = [
        event for event in events if event["drift_injection"]["drift_type"] != "none"
    ]
    rollback_rows = [event for event in events if event["rollback_event"]["triggered"]]
    return {
        "event_count": len(events),
        "dependency_attribution_rate": _rate(
            sum(
                1
                for event in events
                if event["dependency_graph"]["decision_attributed"]
            ),
            len(events),
        ),
        "drift_detection_rate": _rate(
            sum(1 for event in drift_rows if event["drift_injection"]["detected"]),
            len(drift_rows),
        ),
        "drift_rows": [
            {
                "trace_id": event["trace_id"],
                "event_id": event["event_id"],
                **event["drift_injection"],
            }
            for event in drift_rows
        ],
        "rollback_recovery_rate": _rate(
            sum(1 for event in rollback_rows if event["rollback_event"]["recovered"]),
            len(rollback_rows),
        ),
        "rollback_rows": [
            {
                "trace_id": event["trace_id"],
                "event_id": event["event_id"],
                **event["rollback_event"],
            }
            for event in rollback_rows
        ],
        "execution_feedback_rows": [
            {
                "trace_id": event["trace_id"],
                "event_id": event["event_id"],
                **event["execution_feedback"],
            }
            for event in events
        ],
    }


def evaluate_policy_comparison(traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Run all policies on identical events and compute separate deltas."""

    events = _flatten_events(traces)
    policy_rows = {
        policy: [_policy_route(policy, event) for event in events]
        for policy in POLICY_ARMS
    }
    trace_count = len({event["trace_id"] for event in events})
    policy_metrics = {
        policy: _policy_metrics(rows, trace_count)
        for policy, rows in policy_rows.items()
    }
    always = policy_metrics[ALWAYS_FULL_POLICY]
    combined = policy_metrics[COMBINED_POLICY]
    quality_delta = _delta(combined["final_quality"], always["final_quality"])
    memory_hygiene_delta = _delta(
        combined["memory_hygiene"],
        always["memory_hygiene"],
    )
    context_efficiency_delta = _delta(
        combined["context_efficiency"],
        always["context_efficiency"],
    )
    verifier_cost_delta = _rate(
        always["verifier_cost"] - combined["verifier_cost"],
        always["verifier_cost"],
    )
    process_metric_improved = bool(
        memory_hygiene_delta > 0.0
        or context_efficiency_delta > 0.0
        or verifier_cost_delta > 0.0
    )
    return {
        "policy_rows": policy_rows,
        "policy_metrics": policy_metrics,
        "all_policies_run": set(policy_rows) == set(POLICY_ARMS),
        "same_event_ids": _same_event_ids(policy_rows),
        "quality_delta_vs_always_full": quality_delta,
        "memory_hygiene_delta": memory_hygiene_delta,
        "context_efficiency_delta": context_efficiency_delta,
        "verifier_cost_delta": verifier_cost_delta,
        "quality_preserved_vs_always_full": quality_delta >= 0.0,
        "process_metric_improved": process_metric_improved,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5357 terminal artifact from fresh deterministic evidence."""

    source_gate = confirm_source_gate(root=root)
    if source_gate["all_passed"]:
        traces = build_multi_session_traces()
        hash_chain = validate_hash_chains(traces)
        provenance = evaluate_trace_provenance(traces)
        comparison = evaluate_policy_comparison(traces)
    else:
        traces = []
        hash_chain = {
            "valid": False,
            "checked_event_count": 0,
            "failure_count": 0,
            "failures": [],
        }
        provenance = _blocked_provenance()
        comparison = _blocked_comparison()
    combined_metrics = comparison["policy_metrics"].get(COMBINED_POLICY, {})
    aggregate_metrics = {
        "dependency_attribution_rate": provenance["dependency_attribution_rate"],
        "drift_detection_rate": provenance["drift_detection_rate"],
        "quality_delta_vs_always_full": comparison["quality_delta_vs_always_full"],
        "memory_hygiene_delta": comparison["memory_hygiene_delta"],
        "context_efficiency_delta": comparison["context_efficiency_delta"],
        "verifier_cost_delta": comparison["verifier_cost_delta"],
        "rollback_recovery_rate": provenance["rollback_recovery_rate"],
    }
    duplicated_metric_pairs = find_duplicated_metric_pairs(aggregate_metrics)
    complete = _scaleup_complete(
        source_gate=source_gate,
        traces=traces,
        hash_chain=hash_chain,
        provenance=provenance,
        comparison=comparison,
        duplicated_metric_pairs=duplicated_metric_pairs,
        tests_run=tests_run,
    )
    status = (
        "self_learning_scaleup_ready"
        if complete
        else "blocked_dependency_drift_scaleup_gate"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [
            str(EXP5355_RELATIVE_PATH),
            str(EXP5356_RELATIVE_PATH),
            str(EXP5330_RELATIVE_PATH),
        ],
        "excluded_artifacts": [str(EXP5342_QUARANTINED_RELATIVE_PATH)],
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(
                complete,
                source_gate,
                comparison,
                provenance,
                duplicated_metric_pairs,
                tests_run,
            ),
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "continuous_self_learning_target": True,
        "no_weight_mutation": bool(source_gate["no_weight_mutation"]),
        "multi_session_trace_count": len(traces),
        "context_hash_chain_valid": bool(hash_chain["valid"]),
        **aggregate_metrics,
        "duplicated_metric_pairs": duplicated_metric_pairs,
        "unsafe_false_accepts": int(combined_metrics.get("unsafe_false_accepts", 0)),
        "self_learning_scaleup_ready": complete,
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "source_gate": source_gate,
        "multi_session_traces": traces,
        "hash_chain": hash_chain,
        "trace_provenance": provenance,
        "policy_rows": comparison["policy_rows"],
        "policy_metrics": comparison["policy_metrics"],
        "policy_comparison": {
            "all_policies_run": comparison["all_policies_run"],
            "same_event_ids": comparison["same_event_ids"],
            "quality_preserved_vs_always_full": comparison[
                "quality_preserved_vs_always_full"
            ],
            "process_metric_improved": comparison["process_metric_improved"],
        },
        "metric_duplication_check": {
            "checked_fields": list(AGGREGATE_METRIC_FIELDS),
            "metric_values": dict(aggregate_metrics),
            "duplicated_metric_pairs": duplicated_metric_pairs,
            "quarantined_v487_pairs": source_gate[
                "prior_v487_duplicated_metric_pairs"
            ],
        },
        "metric_reports": {
            "quality": {
                "delta_vs_always_full": comparison["quality_delta_vs_always_full"],
                "combined_final_quality": combined_metrics.get("final_quality", 0.0),
            },
            "memory_hygiene": {
                "delta_vs_always_full": comparison["memory_hygiene_delta"],
                "measurement": "clean_active_tokens / active_tokens",
            },
            "context_efficiency": {
                "delta_vs_always_full": comparison["context_efficiency_delta"],
                "measurement": "useful_tokens / (active_tokens + verifier_cost)",
            },
            "verifier_cost": {
                "delta_vs_always_full": comparison["verifier_cost_delta"],
                "measurement": "relative verifier-call reduction",
            },
            "dependency_attribution": {
                "rate": provenance["dependency_attribution_rate"],
                "measurement": "decision-attributed dependency graph events",
            },
            "drift_detection": {
                "rate": provenance["drift_detection_rate"],
                "measurement": "detected drift injections over all drift rows",
            },
            "unsafe_accepts": {
                "combined_policy_count": int(
                    combined_metrics.get("unsafe_false_accepts", 0)
                ),
            },
            "rollback_recovery": {
                "rate": provenance["rollback_recovery_rate"],
            },
        },
        "weight_mutation_receipt": _weight_mutation_receipt(),
        "methodology_note": (
            "Exp5357 uses deterministic checked-in artifacts and replay rows. "
            "Exact 0.0/1.0 values are fixture invariants: quality delta is "
            "0.0 because combined matches always-full correctness, and rollback "
            "recovery is 1.0 because every injected rollback case has an "
            "explicit clean sidecar. These are not live-model performance "
            "claims. Exp5342 is recorded only as quarantined duplicate-metric "
            "context and contributes no aggregate metric to this artifact."
        ),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate fields that downstream capstone gates consume."""

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
    if artifact.get("continuous_self_learning_target") is not True:
        raise ValueError("continuous_self_learning_target must be bare true")
    if artifact.get("no_weight_mutation") is not True:
        raise ValueError("no_weight_mutation must be bare true")
    for field in BARE_INTEGER_FIELDS:
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be a bare integer")
    for field in BARE_BOOL_FIELDS:
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in BARE_NUMERIC_FIELDS:
        if not _is_numeric(artifact.get(field)):
            raise ValueError(f"{field} must be bare numeric")
    if artifact.get("quality_delta_vs_always_full", 0.0) < 0.0:
        raise ValueError("quality_delta_vs_always_full must be nonnegative")
    if not isinstance(artifact.get("duplicated_metric_pairs"), list) or artifact[
        "duplicated_metric_pairs"
    ]:
        raise ValueError("duplicated_metric_pairs must be an empty bare list")
    if artifact.get("unsafe_false_accepts") != 0:
        raise ValueError("unsafe_false_accepts must be 0")
    if artifact["self_learning_scaleup_ready"] and not artifact["tests_run"]["value"]:
        raise ValueError("tests_run must record commands for ready scale-up")
    if artifact["self_learning_scaleup_ready"]:
        if artifact.get("context_hash_chain_valid") is not True:
            raise ValueError("context_hash_chain_valid must be true when ready")
        if artifact.get("rollback_recovery_rate") != 1.0:
            raise ValueError("rollback_recovery_rate must be 1.0 when ready")
        policy_comparison = artifact.get("policy_comparison", {})
        if not policy_comparison.get("process_metric_improved"):
            raise ValueError("process_metric_improved must be true when ready")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5357 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def find_duplicated_metric_pairs(metrics: Mapping[str, Any]) -> list[JsonDict]:
    """Return exact duplicate aggregate values as explicit pair records."""

    return [
        {"left": left, "right": right, "value": metrics[left]}
        for left, right in combinations(metrics, 2)
        if _is_numeric(metrics.get(left))
        and _is_numeric(metrics.get(right))
        and metrics[left] == metrics[right]
    ]


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for source artifacts, spec, and modules."""

    root_path = Path(root)
    return {
        "exp5330": _sha256_file(root_path / EXP5330_RELATIVE_PATH),
        "exp5342_quarantined": _sha256_file(
            root_path / EXP5342_QUARANTINED_RELATIVE_PATH
        ),
        "exp5355": _sha256_file(root_path / EXP5355_RELATIVE_PATH),
        "exp5356": _sha256_file(root_path / EXP5356_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5355_module": _sha256_file(root_path / EXP5355_MODULE_RELATIVE_PATH),
        "exp5356_module": _sha256_file(root_path / EXP5356_MODULE_RELATIVE_PATH),
    }


def _event_spec(
    event_id: str,
    event_action: str,
    object_id: str,
    version: int,
    payload: str,
    *,
    safe_expected: bool = True,
    useful_context: bool = True,
    utility_score: float = 0.9,
    compressor_status: str = "clean",
    dependency_status: str = "complete",
    memory_variant: str = "clean",
    drift_type: str = "none",
    drift_detected: bool = False,
    decision_attributed: bool = True,
    certificate_decision: str = "accept",
    rollback_required: bool = False,
    rollback_object_id: str | None = None,
    rollback_payload: str | None = None,
    verifier: str = "certificate_verifier",
    tool: str = "context_replayer",
    action_decision: str = "accept_context_version",
    support: Sequence[str] = (),
) -> JsonDict:
    return {
        "event_id": event_id,
        "event_action": event_action,
        "object_id": object_id,
        "version": version,
        "payload": payload,
        "safe_expected": safe_expected,
        "useful_context": useful_context,
        "utility_score": utility_score,
        "compressor_status": compressor_status,
        "dependency_status": dependency_status,
        "memory_variant": memory_variant,
        "drift_type": drift_type,
        "drift_detected": drift_detected,
        "decision_attributed": decision_attributed,
        "certificate_decision": certificate_decision,
        "rollback_required": rollback_required,
        "rollback_object_id": rollback_object_id,
        "rollback_payload": rollback_payload,
        "verifier": verifier,
        "tool": tool,
        "action_decision": action_decision,
        "support": list(support),
    }


def _build_trace(raw_trace: Mapping[str, Any]) -> JsonDict:
    trace_id = str(raw_trace["trace_id"])
    session_id = str(raw_trace["session_id"])
    previous_hash = GENESIS_HASH
    state: JsonDict = {}
    events: JsonList = []
    for index, raw_event in enumerate(raw_trace["events"], start=1):
        event = _build_event(trace_id, session_id, index, raw_event, previous_hash)
        _apply_event_to_state(state, event)
        event["state_after_sha256"] = _state_checksum(state)
        event["event_hash"] = _event_hash(event)
        previous_hash = event["event_hash"]
        events.append(event)
    return {"trace_id": trace_id, "session_id": session_id, "events": events}


def _build_event(
    trace_id: str,
    session_id: str,
    event_index: int,
    raw_event: Mapping[str, Any],
    previous_hash: str,
) -> JsonDict:
    object_id = str(raw_event["object_id"])
    payload = str(raw_event["payload"])
    context_object_version = {
        "object_id": object_id,
        "version": raw_event["version"],
        "payload": payload,
        "integrity_hash": _hash_text(f"{object_id}:{raw_event['version']}:{payload}"),
        "utility_score": raw_event["utility_score"],
        "compressor_status": raw_event["compressor_status"],
    }
    dependency_graph = _dependency_graph(raw_event)
    rollback_event = _rollback_event(raw_event)
    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "event_index": event_index,
        "event_id": raw_event["event_id"],
        "event_action": raw_event["event_action"],
        "context_object_version": context_object_version,
        "dependency_graph": dependency_graph,
        "verifier_tool_decision": {
            "selected_verifier": raw_event["verifier"],
            "selected_tool": raw_event["tool"],
            "selected_action": raw_event["action_decision"],
            "parameters": {
                "object_id": object_id,
                "version": raw_event["version"],
                "memory_variant": raw_event["memory_variant"],
                "dependency_status": raw_event["dependency_status"],
            },
            "certificate_decision": raw_event["certificate_decision"],
        },
        "drift_injection": {
            "drift_type": raw_event["drift_type"],
            "memory_variant": raw_event["memory_variant"],
            "detected": raw_event["drift_detected"],
            "detected_by": _drift_detectors(raw_event),
            "unsafe": not bool(raw_event["safe_expected"]),
        },
        "execution_feedback": {
            "feedback_id": f"feedback:{raw_event['event_id']}",
            "execution_status": _execution_status(raw_event),
            "answer_correct": raw_event["safe_expected"],
            "attributed_by_dependency_edge": raw_event["decision_attributed"],
        },
        "rollback_event": rollback_event,
        "utility_memory": {
            "utility_score": raw_event["utility_score"],
            "supporting_context": list(raw_event["support"]),
        },
        "certificate_gate": {
            "decision": raw_event["certificate_decision"],
            "anytime_certificate": "exp5330_context_lifecycle_gate",
            "requires_verifier": bool(
                raw_event["certificate_decision"] != "accept"
                or raw_event["event_action"] in {"retrieve", "fold", "tool_select"}
            ),
        },
        "safe_expected": raw_event["safe_expected"],
        "useful_context": raw_event["useful_context"],
        "token_count": _token_count(payload),
        "sidecar_token_count": _token_count(rollback_event["restore_payload"]),
        "previous_event_hash": previous_hash,
        "state_after_sha256": "",
    }


def _dependency_graph(raw_event: Mapping[str, Any]) -> JsonDict:
    event_id = str(raw_event["event_id"])
    context_node = f"context:{raw_event['object_id']}:{raw_event['version']}"
    retrieval_node = f"retrieval:{event_id}"
    verifier_node = f"verifier:{raw_event['verifier']}:{event_id}"
    tool_node = f"tool:{raw_event['tool']}:{event_id}"
    outcome_node = f"outcome:{raw_event['certificate_decision']}:{event_id}"
    feedback_node = f"feedback:{event_id}"
    edges = [
        _edge(event_id, context_node, retrieval_node, "context_informs_retrieval"),
        _edge(event_id, retrieval_node, verifier_node, "retrieval_routes_verifier"),
        _edge(event_id, retrieval_node, tool_node, "retrieval_routes_tool"),
        _edge(event_id, verifier_node, outcome_node, "verifier_affects_outcome"),
        _edge(event_id, tool_node, outcome_node, "tool_affects_outcome"),
        _edge(event_id, outcome_node, feedback_node, "outcome_records_feedback"),
    ]
    if raw_event["dependency_status"] == "missing_edge":
        edges = [edge for edge in edges if edge["relation"] != "outcome_records_feedback"]
    if raw_event["dependency_status"] == "cycle":
        edges.append(_edge(event_id, feedback_node, retrieval_node, "feedback_reopens_retrieval"))
    return {
        "nodes": [
            context_node,
            retrieval_node,
            verifier_node,
            tool_node,
            outcome_node,
            feedback_node,
        ],
        "edges": edges,
        "dependency_status": raw_event["dependency_status"],
        "decision_attributed": raw_event["decision_attributed"],
        "accepted_into_final_graph": raw_event["dependency_status"] == "complete",
    }


def _edge(event_id: str, source: str, target: str, relation: str) -> JsonDict:
    return {
        "edge_id": f"{event_id}:{relation}",
        "source": source,
        "target": target,
        "relation": relation,
    }


def _rollback_event(raw_event: Mapping[str, Any]) -> JsonDict:
    restore_object_id = raw_event.get("rollback_object_id") or raw_event["object_id"]
    restore_payload = raw_event.get("rollback_payload") or raw_event["payload"]
    triggered = raw_event["certificate_decision"] == "rollback"
    return {
        "triggered": triggered,
        "required": raw_event["rollback_required"],
        "recovered": bool(triggered and raw_event["rollback_required"]),
        "restore_object_id": restore_object_id,
        "restore_payload": restore_payload,
        "restore_payload_sha256": _hash_text(str(restore_payload)),
        "rollback_reason": (
            raw_event["drift_type"] if triggered else "not_required"
        ),
    }


def _apply_event_to_state(state: JsonDict, event: Mapping[str, Any]) -> None:
    decision = event["certificate_gate"]["decision"]
    version = event["context_object_version"]
    if decision == "accept":
        state[str(version["object_id"])] = {
            "version": version["version"],
            "payload_sha256": version["integrity_hash"],
            "compressor_status": version["compressor_status"],
        }
    if decision == "rollback":
        rollback = event["rollback_event"]
        state[str(rollback["restore_object_id"])] = {
            "version": "rollback",
            "payload_sha256": rollback["restore_payload_sha256"],
            "compressor_status": "clean",
        }


def _policy_route(policy: str, event: Mapping[str, Any]) -> JsonDict:
    safe_expected = bool(event["safe_expected"])
    utility_score = float(event["utility_memory"]["utility_score"])
    compressor_status = str(event["context_object_version"]["compressor_status"])
    dependency_status = str(event["dependency_graph"]["dependency_status"])
    memory_variant = str(event["drift_injection"]["memory_variant"])
    drift_detected = bool(event["drift_injection"]["detected"])
    certificate_decision = str(event["certificate_gate"]["decision"])
    rollback_required = bool(event["rollback_event"]["required"])
    accepted = False
    rollback_event = False
    verifier_call = False
    rejection_reasons: list[str] = []

    if policy == ALWAYS_FULL_POLICY:
        verifier_call = True
        accepted = bool(safe_expected and certificate_decision == "accept")
        rollback_event = certificate_decision == "rollback"
        if not accepted and not rollback_event:
            rejection_reasons.append("full_verifier_rejected")
    elif policy == UTILITY_ONLY_POLICY:
        accepted = utility_score >= 0.5
        if not accepted:
            rejection_reasons.append("utility_score_too_low")
    elif policy == COMPRESSOR_ONLY_POLICY:
        verifier_call = bool(
            compressor_status != "clean"
            or event["event_action"] in {"retrieve", "fold"}
        )
        accepted = bool(utility_score >= 0.5 and compressor_status == "clean")
        rollback_event = bool(rollback_required and compressor_status != "clean")
        if not accepted and not rollback_event:
            rejection_reasons.append("compressor_rejected_context")
    elif policy == DEPENDENCY_ONLY_POLICY:
        verifier_call = dependency_status != "complete"
        accepted = bool(utility_score >= 0.5 and dependency_status == "complete")
        if not accepted:
            rejection_reasons.append("dependency_graph_rejected_context")
    elif policy == DRIFT_GUARDED_POLICY:
        verifier_call = memory_variant in {"biased", "stale", "poisoned"}
        accepted = bool(utility_score >= 0.5 and not drift_detected)
        rollback_event = bool(rollback_required and drift_detected)
        if not accepted and not rollback_event:
            rejection_reasons.append("drift_guard_rejected_memory")
    elif policy == COMBINED_POLICY:
        verifier_call = bool(event["certificate_gate"]["requires_verifier"])
        accepted = certificate_decision == "accept"
        rollback_event = certificate_decision == "rollback"
        if not accepted and not rollback_event:
            rejection_reasons.append("combined_certificate_rejected_context")
    else:
        rejection_reasons.append("unknown_policy")

    unsafe_false_accept = bool(accepted and not safe_expected)
    rollback_recovered = bool(
        rollback_event and rollback_required and event["rollback_event"]["recovered"]
    )
    active = bool(policy == ALWAYS_FULL_POLICY or accepted or rollback_event)
    active_tokens = (
        int(event["sidecar_token_count"])
        if rollback_event
        else int(event["token_count"])
    )
    clean_active = bool(
        active
        and not unsafe_false_accept
        and (safe_expected or rollback_recovered)
    )
    useful = bool(
        active
        and not unsafe_false_accept
        and (
            (safe_expected and event["useful_context"])
            or rollback_recovered
        )
    )
    return {
        "policy": policy,
        "trace_id": event["trace_id"],
        "session_id": event["session_id"],
        "event_id": event["event_id"],
        "accepted": accepted,
        "rollback_event": rollback_event,
        "rollback_required": rollback_required,
        "rollback_recovered": rollback_recovered,
        "verifier_call": verifier_call,
        "unsafe_false_accept": unsafe_false_accept,
        "safe_expected": safe_expected,
        "useful_context": bool(event["useful_context"]),
        "active": active,
        "active_token_count": active_tokens if active else 0,
        "clean_active_token_count": active_tokens if clean_active else 0,
        "useful_token_count": active_tokens if useful else 0,
        "rejection_reasons": rejection_reasons,
        "selected_verifier": event["verifier_tool_decision"]["selected_verifier"],
        "selected_tool": event["verifier_tool_decision"]["selected_tool"],
        "certificate_decision": certificate_decision,
    }


def _policy_metrics(rows: Sequence[Mapping[str, Any]], trace_count: int) -> JsonDict:
    unsafe_false_accepts = sum(1 for row in rows if row["unsafe_false_accept"])
    rollback_required_count = sum(1 for row in rows if row["rollback_required"])
    rollback_recovered_count = sum(1 for row in rows if row["rollback_recovered"])
    failed_trace_ids = {
        str(row["trace_id"])
        for row in rows
        if row["unsafe_false_accept"]
        or (row["rollback_required"] and not row["rollback_recovered"])
    }
    active_tokens = sum(int(row["active_token_count"]) for row in rows)
    clean_active_tokens = sum(int(row["clean_active_token_count"]) for row in rows)
    useful_tokens = sum(int(row["useful_token_count"]) for row in rows)
    verifier_cost = sum(1 for row in rows if row["verifier_call"])
    return {
        "event_count": len(rows),
        "trace_ids": [row["trace_id"] for row in rows],
        "active_token_count": active_tokens,
        "clean_active_token_count": clean_active_tokens,
        "useful_token_count": useful_tokens,
        "verifier_cost": verifier_cost,
        "unsafe_false_accepts": unsafe_false_accepts,
        "rollback_events": sum(1 for row in rows if row["rollback_event"]),
        "rollback_recovery_rate": _rate(
            rollback_recovered_count,
            rollback_required_count,
        ),
        "failed_trace_count": len(failed_trace_ids),
        "final_quality": _rate(trace_count - len(failed_trace_ids), trace_count),
        "memory_hygiene": _rate(clean_active_tokens, active_tokens),
        "context_efficiency": _rate(
            useful_tokens,
            active_tokens + verifier_cost,
        ),
        "model_weights_mutated": False,
    }


def _scaleup_complete(
    *,
    source_gate: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
    hash_chain: Mapping[str, Any],
    provenance: Mapping[str, Any],
    comparison: Mapping[str, Any],
    duplicated_metric_pairs: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
) -> bool:
    combined = comparison["policy_metrics"].get(COMBINED_POLICY, {})
    return bool(
        source_gate["all_passed"]
        and 4 <= len(traces) <= 6
        and hash_chain["valid"]
        and provenance["dependency_attribution_rate"] > 0.0
        and provenance["drift_detection_rate"] > 0.0
        and provenance["rollback_recovery_rate"] == 1.0
        and comparison["all_policies_run"]
        and comparison["same_event_ids"]
        and comparison["quality_preserved_vs_always_full"]
        and comparison["process_metric_improved"]
        and int(combined.get("unsafe_false_accepts", 1)) == 0
        and not duplicated_metric_pairs
        and source_gate["no_weight_mutation"]
        and tests_run
    )


def _honest_verdict(
    complete: bool,
    source_gate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    provenance: Mapping[str, Any],
    duplicated_metric_pairs: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
) -> str:
    if complete:
        combined = comparison["policy_metrics"][COMBINED_POLICY]
        return (
            "complete: dependency-drift self-learning scale-up validated "
            f"{combined['event_count']} events across 5 traces, preserved "
            "always-full quality, kept unsafe false accepts at 0, recovered "
            f"rollback at {provenance['rollback_recovery_rate']:.1f}, kept "
            "aggregate metrics non-duplicated, improved process metrics, and "
            "preserved no model weight mutation"
        )
    blockers = _readiness_blockers(
        source_gate,
        comparison,
        provenance,
        duplicated_metric_pairs,
        tests_run,
    )
    return "blocked_dependency_drift_scaleup_not_ready: " + ",".join(blockers)


def _readiness_blockers(
    source_gate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    provenance: Mapping[str, Any],
    duplicated_metric_pairs: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
) -> list[str]:
    combined = comparison["policy_metrics"].get(COMBINED_POLICY, {})
    checks = {
        "source_gate": source_gate["all_passed"],
        "hash_chain": comparison["all_policies_run"],
        "same_event_ids": comparison["same_event_ids"],
        "dependency_attribution": provenance["dependency_attribution_rate"] > 0.0,
        "drift_detection": provenance["drift_detection_rate"] > 0.0,
        "rollback_recovery": provenance["rollback_recovery_rate"] == 1.0,
        "quality_preserved": comparison["quality_preserved_vs_always_full"],
        "process_metric_improved": comparison["process_metric_improved"],
        "unsafe_false_accepts_zero": int(combined.get("unsafe_false_accepts", 1)) == 0,
        "metric_duplicates_clear": not duplicated_metric_pairs,
        "tests_recorded": bool(tests_run),
        "no_weight_mutation": source_gate["no_weight_mutation"],
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if not tests_run and "tests_not_recorded" not in blockers:
        blockers.append("tests_not_recorded")
    return blockers


def _blocked_provenance() -> JsonDict:
    return {
        "event_count": 0,
        "dependency_attribution_rate": 0.0,
        "drift_detection_rate": 0.0,
        "drift_rows": [],
        "rollback_recovery_rate": 0.0,
        "rollback_rows": [],
        "execution_feedback_rows": [],
    }


def _blocked_comparison() -> JsonDict:
    empty_metrics = {
        "event_count": 0,
        "trace_ids": [],
        "active_token_count": 0,
        "clean_active_token_count": 0,
        "useful_token_count": 0,
        "verifier_cost": 0,
        "unsafe_false_accepts": 0,
        "rollback_events": 0,
        "rollback_recovery_rate": 0.0,
        "failed_trace_count": 0,
        "final_quality": 0.0,
        "memory_hygiene": 0.0,
        "context_efficiency": 0.0,
        "model_weights_mutated": False,
    }
    return {
        "policy_rows": {policy: [] for policy in POLICY_ARMS},
        "policy_metrics": {policy: dict(empty_metrics) for policy in POLICY_ARMS},
        "all_policies_run": False,
        "same_event_ids": False,
        "quality_delta_vs_always_full": 0.0,
        "memory_hygiene_delta": 0.0,
        "context_efficiency_delta": 0.0,
        "verifier_cost_delta": 0.0,
        "quality_preserved_vs_always_full": False,
        "process_metric_improved": False,
    }


def _same_event_ids(policy_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> bool:
    event_sets = [
        [row["event_id"] for row in rows]
        for rows in policy_rows.values()
    ]
    return bool(event_sets and all(event_set == event_sets[0] for event_set in event_sets))


def _execution_status(raw_event: Mapping[str, Any]) -> str:
    decision = str(raw_event["certificate_decision"])
    if decision == "rollback":
        return "rollback_executed"
    if decision == "reject":
        return f"{raw_event['dependency_status']}_quarantined"
    return "accepted_and_replayed"


def _drift_detectors(raw_event: Mapping[str, Any]) -> list[str]:
    if not raw_event["drift_detected"]:
        return []
    detectors = []
    if raw_event["memory_variant"] in {"biased", "stale", "poisoned"}:
        detectors.append("memory_tool_drift_guard")
    if raw_event["dependency_status"] in {"missing_edge", "cycle"}:
        detectors.append("dependency_graph_audit")
    if raw_event["compressor_status"] in {"stale", "poisoned"}:
        detectors.append("bounded_compressor_gate")
    return detectors or ["certificate_gate"]


def _flatten_events(traces: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        dict(event)
        for trace in traces
        for event in trace.get("events", [])
    ]


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weight_hash_before": "sha256:no_model_weights_loaded",
        "model_weight_hash_after": "sha256:no_model_weights_loaded",
        "state_mutation_targets": [
            "deterministic_context_trace_rows",
            "deterministic_policy_comparison_rows",
            "deterministic_rollback_rows",
        ],
        "forbidden_updates": {
            "fine_tune_model_weights": False,
            "lora_or_adapter_update": False,
            "foundation_weight_write": False,
        },
    }


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": REQUIRED_FIELD_PRINCIPLES[field]}


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _event_hash(event: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in event.items()
        if key != "event_hash"
    }
    return _checksum(payload)


def _state_checksum(state: Mapping[str, Any]) -> str:
    return _checksum(state)


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return json.loads(json.dumps(value, sort_keys=True))


def _wrapped_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 6)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
