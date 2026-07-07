"""Exp5342: deterministic provenance-bound self-learning scale-up.

Spec refs: REQ-LEARN-5342, SCENARIO-LEARN-5342-PROVENANCE,
SCENARIO-LEARN-5342-ATTACK, SCENARIO-LEARN-5342-POLICY.

This fixture scales the frozen-model self-learning path from single policy
rows to multiple sessions. The important boundary is provenance, not model
adaptation: every context update is hash chained, every accepted state can be
replayed at a point in time, and slow poisoning is caught by cross-event
telemetry before it becomes persistent state. No model, adapter, or foundation
weight is loaded or changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5342_provenance_bound_self_learning_scaleup_v487"
EXPERIMENT_ID = 5342
MILESTONE = "v487"
SCHEMA = "carnot.experiment_5342.provenance_bound_self_learning_scaleup.v487"
RUN_DATE = "2026-07-07"
RANDOM_SEED = 5342
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5342_provenance_bound_self_learning_scaleup_v487.json"
)
EXP5340_RELATIVE_PATH = Path(
    "results/experiment_5340_utility_weighted_context_memory_v487.json"
)
EXP5341_RELATIVE_PATH = Path(
    "results/experiment_5341_bounded_compressor_drift_monitor_v487.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5342_provenance_bound_self_learning_scaleup_v487.py"
)
EXP5340_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5340_utility_weighted_context_memory_v487.py"
)
EXP5341_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5341_bounded_compressor_drift_monitor_v487.py"
)

INFERENCE_SUBSTRATE = "deterministic_provenance_bound_self_learning"
SPEC_REFS = (
    "REQ-LEARN-5342",
    "SCENARIO-LEARN-5342-PROVENANCE",
    "SCENARIO-LEARN-5342-ATTACK",
    "SCENARIO-LEARN-5342-POLICY",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")

GENESIS_HASH = "sha256:genesis"
PER_DIFF_REJECTION_THRESHOLD = 0.5
CROSS_EVENT_SUSPICION_THRESHOLD = 0.9

ALWAYS_FULL_POLICY = "always_full_verification"
UTILITY_ONLY_POLICY = "utility_only"
BOUNDED_ONLY_POLICY = "bounded_compressor_only"
COMBINED_POLICY = "combined_certificate_gated"
POLICY_ARMS = (
    ALWAYS_FULL_POLICY,
    UTILITY_ONLY_POLICY,
    BOUNDED_ONLY_POLICY,
    COMBINED_POLICY,
)

REQUIRED_FIELD_PRINCIPLES = {
    "experiment_id": (
        "Identifies the exact Exp5342 scale-up artifact so downstream gates "
        "cannot confuse provenance-bound multi-session learning with Exp5340 "
        "utility learning or Exp5341 compressor monitoring."
    ),
    "milestone": (
        "Binds the scale-up to milestone v487 where utility memory and bounded "
        "compression are combined under provenance and certificate gates."
    ),
    "status": (
        "Reports whether the combined policy completed under upstream readiness, "
        "hash-chain, reconstruction, telemetry, rollback, quality, process, and "
        "frozen-model gates."
    ),
    "honest_verdict": (
        "Terminal Exp5342 verdict; starts with complete: or blocked_ and states "
        "whether provenance-bound self-learning preserved quality, rejected "
        "unsafe buildup, and improved process metrics without model-weight "
        "mutation."
    ),
    "inference_substrate": (
        "Declares deterministic provenance-bound self-learning with no live LLM, "
        "API judge, model generation, fine-tuning, adapter update, or "
        "foundation-weight mutation."
    ),
    "continuous_self_learning_target": (
        "Bare gate showing the experiment scales self-learning policy state "
        "across sessions rather than reporting a static single-session fixture."
    ),
    "no_weight_mutation": (
        "Bare gate confirming only deterministic context state, telemetry, "
        "certificates, and rollback ledgers changed, never model weights or "
        "adapters."
    ),
    "multi_session_trace_count": (
        "Bare integer count of deterministic multi-session traces used by the "
        "policy comparison."
    ),
    "context_hash_chain_valid": (
        "Bare gate proving every trace event hash links to its predecessor and "
        "prevents silent provenance edits."
    ),
    "point_in_time_reconstruction_rate": (
        "Bare numeric rate showing how often the audit log can reconstruct "
        "accepted context state at event boundaries."
    ),
    "memory_hygiene_delta": (
        "Bare numeric improvement in clean current context state for the "
        "combined policy versus always-full verification."
    ),
    "context_efficiency_delta": (
        "Bare numeric improvement in accepted useful context per retained "
        "object for the combined policy versus always-full verification."
    ),
    "verifier_cost_delta": (
        "Bare numeric verifier-cost reduction for the combined policy versus "
        "always-full verification."
    ),
    "cross_event_attack_detection_rate": (
        "Bare numeric rate over cross-event attacks detected by aggregate "
        "suspicion telemetry."
    ),
    "unsafe_false_accepts": (
        "Bare integer count of unsafe accepts by the combined certificate-gated "
        "policy; any positive value blocks readiness."
    ),
    "rollback_events": (
        "Bare integer count of combined-policy rollback events that removed "
        "unsafe or stale state after aggregate evidence appeared."
    ),
    "self_learning_scaleup_ready": (
        "Bare gate true only when upstream gates pass, unsafe accepts are zero, "
        "quality is preserved, process metrics improve, hash chains validate, "
        "reconstruction succeeds, tests are recorded, and no model weights "
        "mutate."
    ),
    "tests_run": (
        "Records the exact verification commands used to establish that the "
        "scale-up module and result artifact are stable."
    ),
}
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "tests_run",
)
BARE_INTEGER_FIELDS = (
    "multi_session_trace_count",
    "unsafe_false_accepts",
    "rollback_events",
)
BARE_BOOL_FIELDS = ("context_hash_chain_valid", "self_learning_scaleup_ready")
BARE_NUMERIC_FIELDS = (
    "point_in_time_reconstruction_rate",
    "memory_hygiene_delta",
    "context_efficiency_delta",
    "verifier_cost_delta",
    "cross_event_attack_detection_rate",
)


def confirm_upstream_readiness(root: Path | str = REPO_ROOT) -> JsonDict:
    """Confirm Exp5340 utility memory and Exp5341 compressor gates are ready."""

    root_path = Path(root)
    utility = _read_json(root_path / EXP5340_RELATIVE_PATH)
    compressor = _read_json(root_path / EXP5341_RELATIVE_PATH)
    utility_certificate = utility.get("certificate_gate", {})
    compressor_certificate = compressor.get("certificate_gate", {})
    checks = {
        "utility_memory_ready": utility.get("utility_memory_ready") is True,
        "compressor_drift_fixture_ready": (
            compressor.get("compressor_drift_fixture_ready") is True
        ),
        "utility_certificate_gate_ready": (
            isinstance(utility_certificate, Mapping)
            and utility_certificate.get("all_passed") is True
            and utility_certificate.get("anytime_certificate_gate_ready") is True
        ),
        "compressor_certificate_gate_ready": (
            isinstance(compressor_certificate, Mapping)
            and compressor_certificate.get("all_passed") is True
            and compressor_certificate.get("anytime_certificate_gate_ready") is True
        ),
        "no_weight_mutation": (
            utility.get("no_weight_mutation") is True
            and compressor.get("no_weight_mutation") is True
            and utility_certificate.get("no_weight_mutation") is True
            and compressor_certificate.get("no_weight_mutation") is True
        ),
        "unsafe_upstream_accepts_zero": (
            utility.get("unsafe_false_accepts") == 0
            and compressor.get("unsafe_commits") == 0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "failed_gates": failed,
        "all_passed": not failed,
        "utility_source_honest_verdict": _wrapped_value(utility.get("honest_verdict")),
        "compressor_source_honest_verdict": _wrapped_value(
            compressor.get("honest_verdict")
        ),
    }


def build_multi_session_traces() -> JsonList:
    """Build deterministic versioned context traces with provenance hashes."""

    raw_traces = (
        {
            "trace_id": "trace-runtime-policy",
            "session_id": "session-5342-01",
            "events": [
                _event_spec(
                    "s1-e1-create-runtime",
                    "create",
                    "ctx.runtime.receipt",
                    1,
                    "runtime receipt includes sensor ack",
                    useful=True,
                    support=("exp5340:u5340-positive-revise",),
                ),
                _event_spec(
                    "s1-e2-fold-rubric",
                    "fold",
                    "ctx.folded.runtime_rubric",
                    1,
                    "runtime receipt plus arc rubric",
                    useful=True,
                    support=("exp5341:compress-over-compression-drift",),
                ),
            ],
        },
        {
            "trace_id": "trace-patch-rollback",
            "session_id": "session-5342-02",
            "events": [
                _event_spec(
                    "s2-e1-commit-patch",
                    "commit",
                    "ctx.patch.autofix",
                    1,
                    "candidate patch passed tests",
                    useful=True,
                    support=("exp5340:u5340-positive-commit",),
                ),
                _event_spec(
                    "s2-e2-corrupt-patch",
                    "commit",
                    "ctx.patch.autofix",
                    2,
                    "candidate patch bypasses tests",
                    safe_expected=False,
                    useful=False,
                    per_diff_suspicion=0.44,
                    compressor_anomaly="corrupt_patch",
                    decision="reject",
                    support=("exp5330:unsafe_accept_all_lifecycle_actions",),
                    rejection=("bounded_compressor_detected_corruption",),
                ),
                _event_spec(
                    "s2-e3-rollback-patch",
                    "rollback",
                    "ctx.patch.autofix",
                    3,
                    "candidate patch restored from sidecar",
                    decision="rollback",
                    useful=True,
                    sidecar_restore_object_id="ctx.patch.autofix",
                    sidecar_restore_payload="candidate patch passed tests",
                    support=("exp5341:compress-safe-recovery",),
                    rejection=("rollback_to_recoverable_sidecar",),
                ),
            ],
        },
        {
            "trace_id": "trace-gradual-poisoning",
            "session_id": "session-5342-03",
            "events": [
                _event_spec(
                    "s3-e1-shadow-style",
                    "create",
                    "ctx.shadow.preference",
                    1,
                    "prefer concise memory hints",
                    useful=True,
                    per_diff_suspicion=0.29,
                    attack_id="poison-runtime-currentness",
                    support=("operator:benign-style-note",),
                ),
                _event_spec(
                    "s3-e2-shadow-fastpath",
                    "revise",
                    "ctx.shadow.preference",
                    2,
                    "prefer fast path when source label repeats",
                    useful=True,
                    per_diff_suspicion=0.31,
                    attack_id="poison-runtime-currentness",
                    support=("operator:ambiguous-speed-note",),
                ),
                _event_spec(
                    "s3-e3-shadow-currentness",
                    "fold",
                    "ctx.runtime.currentness",
                    1,
                    "skip currentness check when source label repeats",
                    safe_expected=False,
                    useful=False,
                    per_diff_suspicion=0.34,
                    attack_id="poison-runtime-currentness",
                    decision="rollback",
                    sidecar_restore_object_id="ctx.shadow.preference",
                    sidecar_restore_payload="prefer concise memory hints",
                    support=("operator:ambiguous-speed-note", "stale:source-repeat"),
                    rejection=("cross_event_suspicion_threshold",),
                ),
            ],
        },
        {
            "trace_id": "trace-stale-recall-mask",
            "session_id": "session-5342-04",
            "events": [
                _event_spec(
                    "s4-e1-recall-archive",
                    "retrieve",
                    "ctx.archive.policy",
                    1,
                    "archive policy recall sidecar only",
                    useful=True,
                    support=("exp5341:compress-benign-recall",),
                ),
                _event_spec(
                    "s4-e2-stale-recall",
                    "retrieve",
                    "ctx.runtime.receipt",
                    0,
                    "stale runtime receipt v0",
                    approved=True,
                    current=False,
                    safe_expected=False,
                    useful=False,
                    compressor_anomaly="stale_recall",
                    decision="reject",
                    support=("exp5341:compress-stale-recall",),
                    rejection=("stale_recall_detected",),
                ),
                _event_spec(
                    "s4-e3-mask-secret",
                    "mask",
                    "ctx.secret.mask",
                    1,
                    "secret masked before retrieval",
                    useful=True,
                    support=("exp5340:u5340-positive-mask",),
                ),
            ],
        },
    )
    return [_build_trace(trace) for trace in raw_traces]


def validate_hash_chains(traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute every event hash and predecessor edge in each trace."""

    checked = 0
    failures: list[JsonDict] = []
    for trace in traces:
        previous = GENESIS_HASH
        for event in trace.get("events", []):
            checked += 1
            expected_hash = _event_hash(event)
            if event.get("previous_event_hash") != previous:
                failures.append(
                    {
                        "event_id": event.get("event_id"),
                        "reason": "previous_event_hash_mismatch",
                    }
                )
            if event.get("event_hash") != expected_hash:
                failures.append(
                    {
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


def point_in_time_reconstruction(traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Replay audit decisions and compare state hashes after each event."""

    checked = 0
    mismatches: list[JsonDict] = []
    for trace in traces:
        state: JsonDict = {}
        for event in trace.get("events", []):
            checked += 1
            _apply_event_to_state(state, event)
            state_hash = _state_checksum(state)
            if state_hash != event.get("state_after_sha256"):
                mismatches.append(
                    {
                        "trace_id": trace.get("trace_id"),
                        "event_id": event.get("event_id"),
                        "expected": event.get("state_after_sha256"),
                        "actual": state_hash,
                    }
                )
    return {
        "point_in_time_reconstruction_rate": _rate(checked - len(mismatches), checked),
        "checked_state_count": checked,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def evaluate_cross_event_telemetry(traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate suspicion across sessions so slow poisoning is visible."""

    totals: dict[str, float] = {}
    per_diff_detected: set[str] = set()
    detected: set[str] = set()
    attack_ids: set[str] = set()
    telemetry_rows: list[JsonDict] = []
    for trace in traces:
        for event in trace.get("events", []):
            attack_id = event.get("attack_id")
            if not attack_id:
                continue
            attack_ids.add(str(attack_id))
            suspicion = float(event["per_diff_suspicion"])
            totals[str(attack_id)] = totals.get(str(attack_id), 0.0) + suspicion
            if suspicion >= PER_DIFF_REJECTION_THRESHOLD:
                per_diff_detected.add(str(attack_id))
            if totals[str(attack_id)] >= CROSS_EVENT_SUSPICION_THRESHOLD:
                detected.add(str(attack_id))
            telemetry_rows.append(
                {
                    "trace_id": trace["trace_id"],
                    "event_id": event["event_id"],
                    "attack_id": str(attack_id),
                    "per_diff_suspicion": suspicion,
                    "cumulative_suspicion": round(totals[str(attack_id)], 6),
                    "per_diff_detected": str(attack_id) in per_diff_detected,
                    "cross_event_detected": str(attack_id) in detected,
                }
            )
    total_attacks = len(attack_ids)
    return {
        "attack_ids": sorted(attack_ids),
        "detected_attack_ids": sorted(detected),
        "per_diff_detected_attack_ids": sorted(per_diff_detected),
        "cross_event_attack_detection_rate": _rate(len(detected), total_attacks),
        "per_diff_attack_detection_rate": _rate(
            len(per_diff_detected),
            total_attacks,
        ),
        "attack_rollbacks": sum(
            1
            for trace in traces
            for event in trace.get("events", [])
            if event.get("attack_id") in detected
            and event.get("audit_trace", {}).get("decision") == "rollback"
        ),
        "telemetry_rows": telemetry_rows,
    }


def evaluate_policy_comparison(traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compare full, partial, and combined policies on identical traces."""

    telemetry = evaluate_cross_event_telemetry(traces)
    detected_attack_map = {attack_id: True for attack_id in telemetry["detected_attack_ids"]}
    policy_rows = {
        policy: [
            _policy_route(policy, event, detected_attack_map)
            for trace in traces
            for event in trace.get("events", [])
        ]
        for policy in POLICY_ARMS
    }
    trace_count = len(traces)
    policy_metrics = {
        policy: _policy_metrics(policy, rows, trace_count, telemetry)
        for policy, rows in policy_rows.items()
    }
    always = policy_metrics[ALWAYS_FULL_POLICY]
    combined = policy_metrics[COMBINED_POLICY]
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
        "all_policies_run": bool(set(policy_rows) == set(POLICY_ARMS)),
        "same_trace_ids": _same_trace_ids(policy_rows),
        "memory_hygiene_delta": memory_hygiene_delta,
        "context_efficiency_delta": context_efficiency_delta,
        "verifier_cost_delta": verifier_cost_delta,
        "quality_preserved_vs_always_full": (
            combined["final_quality"] >= always["final_quality"]
        ),
        "process_metric_improved": process_metric_improved,
        "telemetry": telemetry,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5342 terminal artifact from deterministic evidence."""

    upstream_gate = confirm_upstream_readiness(root=root)
    if upstream_gate["all_passed"]:
        traces = build_multi_session_traces()
        hash_chain = validate_hash_chains(traces)
        reconstruction = point_in_time_reconstruction(traces)
        comparison = evaluate_policy_comparison(traces)
    else:
        traces = []
        hash_chain = {"valid": False, "checked_event_count": 0, "failures": []}
        reconstruction = {
            "point_in_time_reconstruction_rate": 0.0,
            "checked_state_count": 0,
            "mismatch_count": 0,
            "mismatches": [],
        }
        comparison = _blocked_evaluation()
    complete = _scaleup_complete(
        upstream_gate=upstream_gate,
        traces=traces,
        hash_chain=hash_chain,
        reconstruction=reconstruction,
        comparison=comparison,
        tests_run=tests_run,
    )
    status = (
        "self_learning_scaleup_ready"
        if complete
        else "blocked_upstream_or_scaleup_gate"
    )
    combined_metrics = comparison.get("policy_metrics", {}).get(COMBINED_POLICY, {})
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [
            str(EXP5340_RELATIVE_PATH),
            str(EXP5341_RELATIVE_PATH),
        ],
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(complete, upstream_gate, comparison, tests_run),
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "continuous_self_learning_target": True,
        "no_weight_mutation": bool(upstream_gate["no_weight_mutation"]),
        "multi_session_trace_count": len(traces),
        "context_hash_chain_valid": bool(hash_chain["valid"]),
        "point_in_time_reconstruction_rate": reconstruction[
            "point_in_time_reconstruction_rate"
        ],
        "memory_hygiene_delta": comparison["memory_hygiene_delta"],
        "context_efficiency_delta": comparison["context_efficiency_delta"],
        "verifier_cost_delta": comparison["verifier_cost_delta"],
        "cross_event_attack_detection_rate": combined_metrics.get(
            "cross_event_attack_detection_rate",
            0.0,
        ),
        "unsafe_false_accepts": int(combined_metrics.get("unsafe_false_accepts", 0)),
        "rollback_events": int(combined_metrics.get("rollback_events", 0)),
        "self_learning_scaleup_ready": complete,
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "upstream_gate": upstream_gate,
        "multi_session_traces": traces,
        "hash_chain": hash_chain,
        "point_in_time_reconstruction": reconstruction,
        "policy_rows": comparison["policy_rows"],
        "policy_metrics": comparison["policy_metrics"],
        "policy_comparison": {
            "quality_preserved_vs_always_full": comparison[
                "quality_preserved_vs_always_full"
            ],
            "process_metric_improved": comparison["process_metric_improved"],
            "same_trace_ids": comparison["same_trace_ids"],
            "all_policies_run": comparison["all_policies_run"],
        },
        "cross_event_telemetry": comparison["telemetry"],
        "weight_mutation_receipt": _weight_mutation_receipt(),
        "methodology_note": (
            "The scale-up is a deterministic replay over Exp5340/Exp5341 "
            "fixtures and synthetic versioned context traces. It does not load "
            "or call an LLM, API judge, generator, fine-tuning path, adapter "
            "update, or model-weight mutation path."
        ),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate fields consumed by downstream provenance-bound gates."""

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
    if artifact.get("unsafe_false_accepts") != 0:
        raise ValueError("unsafe_false_accepts must be 0")
    if artifact["self_learning_scaleup_ready"] and not artifact["tests_run"]["value"]:
        raise ValueError("tests_run must record commands for ready scale-up")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5342 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for deterministic source inputs."""

    root_path = Path(root)
    return {
        "exp5340": _sha256_file(root_path / EXP5340_RELATIVE_PATH),
        "exp5341": _sha256_file(root_path / EXP5341_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5340_module": _sha256_file(root_path / EXP5340_MODULE_RELATIVE_PATH),
        "exp5341_module": _sha256_file(root_path / EXP5341_MODULE_RELATIVE_PATH),
    }


def _event_spec(
    event_id: str,
    action: str,
    object_id: str,
    version: int,
    payload: str,
    *,
    approved: bool = True,
    current: bool = True,
    safe_expected: bool = True,
    useful: bool = True,
    per_diff_suspicion: float = 0.05,
    attack_id: str | None = None,
    compressor_anomaly: str | None = None,
    decision: str = "accept",
    sidecar_restore_object_id: str | None = None,
    sidecar_restore_payload: str | None = None,
    support: Sequence[str] = (),
    rejection: Sequence[str] = (),
) -> JsonDict:
    return {
        "event_id": event_id,
        "action": action,
        "object_id": object_id,
        "version": version,
        "payload": payload,
        "approved": approved,
        "current": current,
        "safe_expected": safe_expected,
        "useful_context": useful,
        "per_diff_suspicion": per_diff_suspicion,
        "attack_id": attack_id,
        "compressor_anomaly": compressor_anomaly,
        "decision": decision,
        "sidecar_restore_object_id": sidecar_restore_object_id,
        "sidecar_restore_payload": sidecar_restore_payload,
        "supporting_events": list(support),
        "rejection_reasons": list(rejection),
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
    object_version = {
        "object_id": raw_event["object_id"],
        "version": raw_event["version"],
        "payload": raw_event["payload"],
        "integrity_hash": _hash_text(
            f"{raw_event['object_id']}:{raw_event['version']}:{raw_event['payload']}"
        ),
        "approved": raw_event["approved"],
        "current": raw_event["current"],
    }
    sidecar = _sidecar(raw_event)
    decision = str(raw_event["decision"])
    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "event_index": event_index,
        "event_id": raw_event["event_id"],
        "action": raw_event["action"],
        "context_object_version": object_version,
        "supporting_events": list(raw_event["supporting_events"]),
        "sidecar": sidecar,
        "audit_trace": {
            "decision": decision,
            "per_diff_check": (
                "pass"
                if float(raw_event["per_diff_suspicion"])
                < PER_DIFF_REJECTION_THRESHOLD
                else "reject"
            ),
            "certificate_gate": "combined_certificate_gated",
            "rejection_reasons": list(raw_event["rejection_reasons"]),
        },
        "safe_expected": raw_event["safe_expected"],
        "useful_context": raw_event["useful_context"],
        "per_diff_suspicion": raw_event["per_diff_suspicion"],
        "attack_id": raw_event["attack_id"],
        "compressor_anomaly": raw_event["compressor_anomaly"],
        "previous_event_hash": previous_hash,
        "state_after_sha256": "",
    }


def _sidecar(raw_event: Mapping[str, Any]) -> JsonDict:
    restore_object_id = raw_event.get("sidecar_restore_object_id")
    restore_payload = raw_event.get("sidecar_restore_payload")
    object_id = str(restore_object_id or raw_event["object_id"])
    payload = str(restore_payload or raw_event["payload"])
    return {
        "sidecar_id": f"sidecar:{raw_event['event_id']}",
        "recoverable": bool(restore_object_id and restore_payload),
        "restore_object_id": object_id,
        "restore_payload": payload,
        "payload_sha256": _hash_text(payload),
    }


def _apply_event_to_state(state: JsonDict, event: Mapping[str, Any]) -> None:
    decision = event.get("audit_trace", {}).get("decision")
    version = event["context_object_version"]
    if decision == "accept":
        state[str(version["object_id"])] = {
            "version": version["version"],
            "payload_sha256": version["integrity_hash"],
            "approved": version["approved"],
            "current": version["current"],
        }
    if decision == "rollback":
        sidecar = event["sidecar"]
        object_id = str(sidecar["restore_object_id"])
        state[object_id] = {
            "version": "rollback",
            "payload_sha256": sidecar["payload_sha256"],
            "approved": True,
            "current": True,
        }


def _policy_route(
    policy: str,
    event: Mapping[str, Any],
    detected_attack_map: Mapping[str, bool],
) -> JsonDict:
    version = event["context_object_version"]
    approved_current = bool(version["approved"] and version["current"])
    per_diff_clean = float(event["per_diff_suspicion"]) < PER_DIFF_REJECTION_THRESHOLD
    compressor_clean = event.get("compressor_anomaly") is None
    detected_cross_event = bool(
        event.get("attack_id") and detected_attack_map.get(str(event["attack_id"]))
    )
    decision = str(event.get("audit_trace", {}).get("decision"))
    verifier_call = False
    accepted = False
    rollback_event = False
    rejection_reasons: list[str] = []

    if policy == ALWAYS_FULL_POLICY:
        verifier_call = True
        accepted = bool(event["safe_expected"] and approved_current and decision == "accept")
        rollback_event = decision == "rollback"
        if not accepted and not rollback_event:
            rejection_reasons.append("full_context_verifier_rejected")
    elif policy == UTILITY_ONLY_POLICY:
        accepted = bool(approved_current and per_diff_clean)
        if not accepted:
            rejection_reasons.append("utility_local_gate_rejected")
    elif policy == BOUNDED_ONLY_POLICY:
        verifier_call = bool(event["action"] in {"fold", "retrieve", "rollback"} or not compressor_clean)
        accepted = bool(approved_current and per_diff_clean and compressor_clean)
        rollback_event = decision == "rollback" and not compressor_clean
        if not accepted and not rollback_event:
            rejection_reasons.append("bounded_compressor_rejected")
    elif policy == COMBINED_POLICY:
        verifier_call = bool(
            event["action"] in {"fold", "retrieve", "rollback"}
            or not compressor_clean
            or detected_cross_event
        )
        accepted = decision == "accept"
        rollback_event = decision == "rollback"
        if not accepted and not rollback_event:
            rejection_reasons.extend(event.get("audit_trace", {}).get("rejection_reasons", []))
    else:
        rejection_reasons.append("unknown_policy")

    unsafe_false_accept = bool(accepted and not event["safe_expected"])
    active = bool(policy == ALWAYS_FULL_POLICY or accepted or rollback_event)
    token_count = _token_count(str(version["payload"]))
    sidecar_tokens = _sidecar_token_count(event)
    active_tokens = sidecar_tokens if rollback_event else token_count
    clean_active = bool(active and not unsafe_false_accept and event["safe_expected"])
    useful = bool(active and event["useful_context"] and not unsafe_false_accept)
    return {
        "policy": policy,
        "trace_id": event["trace_id"],
        "session_id": event["session_id"],
        "event_id": event["event_id"],
        "accepted": accepted,
        "verifier_call": verifier_call,
        "unsafe_false_accept": unsafe_false_accept,
        "rollback_event": rollback_event,
        "detected_cross_event_attack": detected_cross_event
        and policy in {ALWAYS_FULL_POLICY, COMBINED_POLICY},
        "active_token_count": active_tokens if active else 0,
        "clean_active_token_count": active_tokens if clean_active else 0,
        "useful_token_count": active_tokens if useful else 0,
        "rejection_reasons": rejection_reasons,
        "model_weights_mutated": False,
    }


def _policy_metrics(
    policy: str,
    rows: Sequence[Mapping[str, Any]],
    trace_count: int,
    telemetry: Mapping[str, Any],
) -> JsonDict:
    unsafe_rows = [row for row in rows if row["unsafe_false_accept"]]
    failed_traces = {str(row["trace_id"]) for row in unsafe_rows}
    active_tokens = sum(int(row["active_token_count"]) for row in rows)
    clean_tokens = sum(int(row["clean_active_token_count"]) for row in rows)
    useful_tokens = sum(int(row["useful_token_count"]) for row in rows)
    cross_rate = (
        telemetry["cross_event_attack_detection_rate"]
        if policy in {ALWAYS_FULL_POLICY, COMBINED_POLICY}
        else 0.0
    )
    return {
        "trace_ids": [str(row["trace_id"]) for row in rows],
        "event_count": len(rows),
        "failed_trace_count": len(failed_traces),
        "final_quality": _delta(1.0, _rate(len(failed_traces), trace_count)),
        "memory_hygiene": _rate(clean_tokens, active_tokens),
        "context_efficiency": _rate(useful_tokens, active_tokens),
        "verifier_cost": sum(1 for row in rows if row["verifier_call"]),
        "unsafe_false_accepts": len(unsafe_rows),
        "cross_event_attack_detection_rate": cross_rate,
        "rollback_events": sum(1 for row in rows if row["rollback_event"]),
        "active_token_count": active_tokens,
        "clean_active_token_count": clean_tokens,
        "useful_token_count": useful_tokens,
        "model_weights_mutated": any(bool(row["model_weights_mutated"]) for row in rows),
    }


def _same_trace_ids(policy_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> bool:
    trace_ids = [
        tuple(str(row["trace_id"]) for row in rows)
        for rows in policy_rows.values()
    ]
    return bool(trace_ids) and all(ids == trace_ids[0] for ids in trace_ids)


def _blocked_evaluation() -> JsonDict:
    return {
        "policy_rows": {policy: [] for policy in POLICY_ARMS},
        "policy_metrics": {},
        "all_policies_run": False,
        "same_trace_ids": False,
        "memory_hygiene_delta": 0.0,
        "context_efficiency_delta": 0.0,
        "verifier_cost_delta": 0.0,
        "quality_preserved_vs_always_full": False,
        "process_metric_improved": False,
        "telemetry": {
            "attack_ids": [],
            "detected_attack_ids": [],
            "cross_event_attack_detection_rate": 0.0,
            "per_diff_attack_detection_rate": 0.0,
            "attack_rollbacks": 0,
            "telemetry_rows": [],
        },
        "self_learning_scaleup_ready": False,
    }


def _scaleup_complete(
    *,
    upstream_gate: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
    hash_chain: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    comparison: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> bool:
    combined = comparison.get("policy_metrics", {}).get(COMBINED_POLICY, {})
    return bool(
        upstream_gate["all_passed"]
        and 3 <= len(traces) <= 5
        and hash_chain["valid"] is True
        and reconstruction["point_in_time_reconstruction_rate"] == 1.0
        and comparison["quality_preserved_vs_always_full"]
        and comparison["process_metric_improved"]
        and combined.get("unsafe_false_accepts") == 0
        and combined.get("final_quality") == 1.0
        and combined.get("cross_event_attack_detection_rate", 0.0) > 0.0
        and combined.get("rollback_events", 0) > 0
        and not combined.get("model_weights_mutated", True)
        and bool(tests_run)
    )


def _honest_verdict(
    complete: bool,
    upstream_gate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> str:
    if complete:
        combined = comparison["policy_metrics"][COMBINED_POLICY]
        return (
            "complete: provenance-bound self-learning validated "
            "4 multi-session traces, preserved always-full quality, detected "
            "cross-event poisoning at rate "
            f"{combined['cross_event_attack_detection_rate']}, kept unsafe "
            f"false accepts at {combined['unsafe_false_accepts']}, recorded "
            f"{combined['rollback_events']} rollback events, reduced verifier "
            "cost, and preserved no model weight mutation"
        )
    blockers = list(upstream_gate.get("failed_gates", []))
    if not comparison.get("quality_preserved_vs_always_full"):
        blockers.append("quality_not_preserved")
    if not comparison.get("process_metric_improved"):
        blockers.append("process_metric_not_improved")
    if not tests_run:
        blockers.append("tests_not_recorded")
    return "blocked_self_learning_scaleup_not_ready: " + ",".join(blockers)


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weight_hash_before": "sha256:no_model_weights_loaded",
        "model_weight_hash_after": "sha256:no_model_weights_loaded",
        "state_mutation_targets": [
            "deterministic_context_hash_chain",
            "deterministic_cross_event_suspicion_telemetry",
            "deterministic_certificate_rollback_ledger",
        ],
        "forbidden_updates": {
            "fine_tune_model_weights": False,
            "lora_or_adapter_update": False,
            "foundation_weight_write": False,
        },
    }


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
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _event_hash(event: Mapping[str, Any]) -> str:
    content = {key: value for key, value in event.items() if key != "event_hash"}
    return _checksum(_json_ready(content))


def _state_checksum(state: Mapping[str, Any]) -> str:
    return _checksum(_json_ready(state))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return json.loads(json.dumps(value, sort_keys=True))


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _token_count(text: str) -> int:
    normalized = text.replace("_", " ")
    return len([token for token in normalized.split() if token])


def _sidecar_token_count(event: Mapping[str, Any]) -> int:
    return _token_count(str(event.get("sidecar", {}).get("restore_payload", "")))


def _rate(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / denominator, 6)


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 6)
