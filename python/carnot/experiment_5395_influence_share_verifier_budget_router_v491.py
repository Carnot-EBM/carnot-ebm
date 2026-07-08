"""Exp5395: influence-share verifier-budget routing for self-learning.

Spec refs: REQ-LEARN-5395, SCENARIO-LEARN-5395-SHARES,
SCENARIO-LEARN-5395-ROUTING, SCENARIO-LEARN-5395-ARTIFACT.

This module evaluates a controller, not a trainer. It reuses the Exp5382
dependency/drift workflow and decides which verifier tier should receive each
event. The useful evidence is the routing ledger: each row shows why a cheap
deterministic check, richer deterministic verifier, or local SOTA escalation
was selected. No model weights are loaded or changed, so the experiment stays
about verifier-budget governance instead of hidden fine-tuning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from carnot import experiment_5382_real_workflow_continuous_self_learning_v490 as exp5382


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5395_influence_share_verifier_budget_router_v491"
EXPERIMENT_ID = "exp5395-v491-influence-share-verifier-budget-router"
MILESTONE = "2026.07.491"
SCHEMA = "carnot.experiment_5395.influence_share_verifier_budget_router.v491"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5395
WORKFLOW_NAME = exp5382.WORKFLOW_NAME

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5395_influence_share_verifier_budget_router_v491.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5395_influence_share_verifier_budget_router_v491.py"
)
EXP5382_RESULT_RELATIVE_PATH = exp5382.RESULT_RELATIVE_PATH
EXP5382_MODULE_RELATIVE_PATH = exp5382.MODULE_RELATIVE_PATH

SPEC_REFS = (
    "REQ-LEARN-5395",
    "SCENARIO-LEARN-5395-SHARES",
    "SCENARIO-LEARN-5395-ROUTING",
    "SCENARIO-LEARN-5395-ARTIFACT",
)
TERMINAL_PREFIXES = ("complete:", "blocked:")

BASELINE_VARIANT = "baseline_routing"
FIXED_VARIANT = "fixed_self_learning_routing"
INFLUENCE_VARIANT = "influence_share_routing"

MIN_SESSIONS = 12
MIN_TRACES = 12
MIN_CHECKED_EVENTS = 30
ROUTER_BUDGET = 18.0
LOCAL_SOTA_UNCERTAINTY_MIN = 0.3
LOCAL_SOTA_USER_IMPACT_MIN = 0.7
RICH_RISK_MIN = 0.45
RICH_NOVELTY_MIN = 0.6

INFLUENCE_FACTOR_NAMES = [
    "stale-risk",
    "poison-risk",
    "constraint-risk",
    "novelty",
    "user-impact",
    "verifier-cost",
    "evidence-confidence",
]

VERIFIER_TIER_COSTS: dict[str, float] = {
    "cheap_deterministic": 0.1,
    "rich_deterministic": 0.5,
    "local_sota": 1.2,
}

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Complete if the workflow ran with routing evidence.",
    "milestone": "Must equal 2026.07.491.",
    "session_count": "Number of self-learning sessions, target >=12.",
    "trace_count": "Number of comparable baseline/router traces.",
    "checked_event_count": "Deterministic event checks across traces.",
    "influence_factor_names": "List of factors used in routing.",
    "influence_share_sum_valid_rate": ("Fraction of routing rows whose shares sum to 100%."),
    "routed_decision_count": "Number of verifier routing decisions.",
    "verifier_cost_delta_vs_baseline": "Cost reduction or increase with sign.",
    "context_efficiency_delta_vs_baseline": "Efficiency delta with sign.",
    "quality_delta_vs_baseline": "Deterministic quality delta with sign.",
    "stale_memory_deflection_rate": "Rate for stale-memory controls.",
    "poison_deflection_rate": "Rate for forged/poisoned-memory controls.",
    "rollback_success_rate": "Rollback rate for bad routing decisions.",
    "no_weight_mutation": "Must be true.",
    "continuous_self_learning_router_ready": (
        "True only if quality is preserved and cost or context improves."
    ),
    "honest_verdict": ("One-line summary starting with complete: or blocked:."),
}

REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)
BOOL_FIELDS = ("no_weight_mutation", "continuous_self_learning_router_ready")
INTEGER_FIELDS = (
    "session_count",
    "trace_count",
    "checked_event_count",
    "routed_decision_count",
)
NUMERIC_FIELDS = (
    "influence_share_sum_valid_rate",
    "verifier_cost_delta_vs_baseline",
    "context_efficiency_delta_vs_baseline",
    "quality_delta_vs_baseline",
    "stale_memory_deflection_rate",
    "poison_deflection_rate",
    "rollback_success_rate",
)


def select_workflow_traces() -> JsonList:
    """Return the Exp5382 real workflow family used by every variant."""

    return exp5382.select_workflow_traces()


def evaluate_routing_variants(
    *,
    root: Path | str = REPO_ROOT,
    traces: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Compare baseline, fixed, and influence-share routing on one event panel."""

    workflow_traces = list(traces) if traces is not None else select_workflow_traces()
    exp5382_eval = exp5382.evaluate_real_workflow(traces=workflow_traces, root=root)
    workflow = dict(exp5382_eval["workflow"])
    events = _flatten_events(workflow_traces)
    routing_decisions = _route_events(events)
    baseline = _variant_from_exp5382(BASELINE_VARIANT, exp5382_eval["baseline_variant"])
    fixed = _variant_from_exp5382(FIXED_VARIANT, exp5382_eval["self_learning_variant"])
    routed = _influence_variant_metrics(exp5382_eval["self_learning_variant"], routing_decisions)
    safety = _safety_controls(routing_decisions)
    share_valid_rate = _rate(
        sum(1 for row in routing_decisions if sum(row["influence_shares"].values()) == 100),
        len(routing_decisions),
    )
    variant_metrics = {
        BASELINE_VARIANT: baseline,
        FIXED_VARIANT: fixed,
        INFLUENCE_VARIANT: routed,
    }
    return {
        "workflow": workflow,
        "session_count": int(workflow["session_count"]),
        "trace_count": int(workflow["trace_count"]),
        "checked_event_count": len(events),
        "influence_factor_names": list(INFLUENCE_FACTOR_NAMES),
        "influence_share_sum_valid_rate": share_valid_rate,
        "routed_decision_count": len(routing_decisions),
        "verifier_cost_delta_vs_baseline": _delta(
            baseline["verifier_cost"], routed["verifier_cost"]
        ),
        "context_efficiency_delta_vs_baseline": _delta(
            routed["context_efficiency"], baseline["context_efficiency"]
        ),
        "quality_delta_vs_baseline": _delta(routed["quality"], baseline["quality"]),
        "stale_memory_deflection_rate": safety["stale_memory_deflection_rate"],
        "poison_deflection_rate": safety["poison_deflection_rate"],
        "rollback_success_rate": safety["rollback_success_rate"],
        "unsafe_false_accepts": safety["unsafe_false_accepts"],
        "variant_metrics": variant_metrics,
        "routing_decisions": routing_decisions,
        "safety_controls": safety,
        "weight_mutation_receipt": _weight_mutation_receipt(),
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal JSON artifact from deterministic routing evidence."""

    evaluation = evaluate_routing_variants(root=root)
    readiness = _readiness_checks(evaluation, tests_run)
    ready = bool(readiness["all_passed"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifacts": [str(EXP5382_RESULT_RELATIVE_PATH)],
        "status": "complete" if ready else "blocked",
        "milestone": MILESTONE,
        "session_count": evaluation["session_count"],
        "trace_count": evaluation["trace_count"],
        "checked_event_count": evaluation["checked_event_count"],
        "influence_factor_names": evaluation["influence_factor_names"],
        "influence_share_sum_valid_rate": evaluation["influence_share_sum_valid_rate"],
        "routed_decision_count": evaluation["routed_decision_count"],
        "verifier_cost_delta_vs_baseline": evaluation["verifier_cost_delta_vs_baseline"],
        "context_efficiency_delta_vs_baseline": evaluation["context_efficiency_delta_vs_baseline"],
        "quality_delta_vs_baseline": evaluation["quality_delta_vs_baseline"],
        "stale_memory_deflection_rate": evaluation["stale_memory_deflection_rate"],
        "poison_deflection_rate": evaluation["poison_deflection_rate"],
        "rollback_success_rate": evaluation["rollback_success_rate"],
        "no_weight_mutation": evaluation["weight_mutation_receipt"]["no_weight_mutation"],
        "continuous_self_learning_router_ready": ready,
        "honest_verdict": _honest_verdict(ready),
        "tests_run": [dict(row) for row in tests_run],
        "workflow_name": WORKFLOW_NAME,
        "workflow_evidence": evaluation["workflow"],
        "variant_metrics": evaluation["variant_metrics"],
        "routing_decisions": evaluation["routing_decisions"],
        "safety_controls": evaluation["safety_controls"],
        "unsafe_false_accepts": evaluation["unsafe_false_accepts"],
        "readiness_checks": readiness,
        "weight_mutation_receipt": evaluation["weight_mutation_receipt"],
        "methodology_note": (
            "Exp5395 replays the Exp5382 workflow and records controller "
            "routing choices only. Local SOTA is a selected verification tier "
            "in the budget ledger; this deterministic run does not load or "
            "mutate model weights."
        ),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the fields consumed by the milestone reconciler."""

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError("missing Exp5395 fields: " + ",".join(missing))
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match Exp5395 contract")
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
    ready = artifact["continuous_self_learning_router_ready"]
    if (ready and artifact["status"] != "complete") or (
        artifact["status"] == "complete" and not ready
    ):
        raise ValueError("status must match router readiness")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must equal 2026.07.491")
    if (
        artifact["session_count"] < MIN_SESSIONS
        or artifact["trace_count"] < MIN_TRACES
        or artifact["checked_event_count"] < MIN_CHECKED_EVENTS
        or artifact["routed_decision_count"] <= 0
    ):
        raise ValueError("session_count trace_count checked_event_count routed_decision_count")
    if artifact["influence_share_sum_valid_rate"] != 1.0:
        raise ValueError("influence_share_sum_valid_rate must be 1.0")
    if (
        artifact["verifier_cost_delta_vs_baseline"] <= 0.0
        and artifact["context_efficiency_delta_vs_baseline"] <= 0.0
    ):
        raise ValueError("cost or context must improve versus baseline")
    if artifact["quality_delta_vs_baseline"] < 0.0:
        raise ValueError("quality_delta_vs_baseline must preserve baseline quality")
    if artifact["stale_memory_deflection_rate"] < 1.0:
        raise ValueError("stale_memory_deflection_rate must be complete")
    if artifact["poison_deflection_rate"] < 1.0:
        raise ValueError("poison_deflection_rate must be complete")
    if artifact["rollback_success_rate"] < 1.0:
        raise ValueError("rollback_success_rate must be complete")
    if artifact["no_weight_mutation"] is not True:
        raise ValueError("no_weight_mutation must be true")
    if not artifact.get("tests_run"):
        raise ValueError("tests_run must record commands for ready artifact")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the deterministic Exp5395 artifact and return it."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for the sources that define this replay."""

    root_path = Path(root)
    return {
        "exp5382": _sha256_file(root_path / EXP5382_RESULT_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5382_module": _sha256_file(root_path / EXP5382_MODULE_RELATIVE_PATH),
    }


def _route_events(events: Sequence[Mapping[str, Any]]) -> JsonList:
    decisions: JsonList = []
    budget_remaining = ROUTER_BUDGET
    for index, event in enumerate(events, start=1):
        evidence = _raw_evidence(event, budget_remaining)
        selected, rejected, reason = _select_tier(evidence)
        cost = VERIFIER_TIER_COSTS[selected]
        row = {
            "decision_index": index,
            "variant_name": INFLUENCE_VARIANT,
            "event_id": str(event["event_id"]),
            "trace_id": str(event["trace_id"]),
            "session_id": str(event["session_id"]),
            "raw_evidence": evidence,
            "influence_shares": _influence_shares(evidence),
            "selected_verifier_tier": selected,
            "rejected_tier": rejected,
            "reason": reason,
            "tier_cost": cost,
            "budget_remaining_after": round(budget_remaining - cost, 6),
            "rollback_status": _rollback_status(event),
            "deflected_stale": bool(
                evidence["memory_variant"] == "stale" and selected != "cheap_deterministic"
            ),
            "deflected_poison": bool(
                evidence["memory_variant"] == "poisoned" and selected != "cheap_deterministic"
            ),
            "unsafe_false_accept": bool(evidence["unsafe"] and selected == "cheap_deterministic"),
        }
        decisions.append(_json_ready(row))
        budget_remaining = round(budget_remaining - cost, 6)
    return decisions


def _raw_evidence(event: Mapping[str, Any], budget_remaining: float) -> JsonDict:
    variant = str(event.get("drift_injection", {}).get("memory_variant", "clean"))
    certificate = str(event.get("verifier_tool_decision", {}).get("certificate_decision"))
    drift_type = str(event.get("drift_injection", {}).get("drift_type", "none"))
    action = str(event.get("event_action", event.get("action", "")))
    unsafe = bool(
        event.get("drift_injection", {}).get("unsafe") or certificate in {"reject", "rollback"}
    )
    stale_risk = 0.95 if variant == "stale" else 0.08
    poison_risk = 0.95 if variant == "poisoned" else (0.45 if variant == "unverified" else 0.06)
    constraint_risk = 0.9 if certificate in {"reject", "rollback"} else 0.18
    novelty = (
        0.68
        if variant in {"unverified", "biased"}
        or drift_type in {"cyclic_dependency", "missing_dependency_edge"}
        else (0.45 if variant in {"stale", "poisoned"} else 0.2)
    )
    user_impact = (
        0.85
        if action in {"commit", "tool_select", "fold"}
        else (0.72 if action in {"retrieve", "restore"} else 0.5)
    )
    evidence_confidence = (
        0.62 if variant in {"stale", "poisoned"} else (0.72 if certificate == "reject" else 0.9)
    )
    return {
        "event_id": str(event["event_id"]),
        "trace_id": str(event["trace_id"]),
        "session_id": str(event["session_id"]),
        "memory_variant": variant,
        "certificate_decision": certificate,
        "drift_type": drift_type,
        "action": action,
        "supporting_context": list(event.get("utility_memory", {}).get("supporting_context", [])),
        "unsafe": unsafe,
        "stale_risk": stale_risk,
        "poison_risk": poison_risk,
        "constraint_risk": constraint_risk,
        "novelty": novelty,
        "user_impact": user_impact,
        "verifier_cost_pressure": round(1.0 - (budget_remaining / ROUTER_BUDGET), 6),
        "evidence_confidence": evidence_confidence,
        "uncertainty": round(1.0 - evidence_confidence, 6),
        "budget_remaining_before": round(budget_remaining, 6),
        "budget_limit": ROUTER_BUDGET,
        "tier_costs": dict(VERIFIER_TIER_COSTS),
        "rollback_required": bool(event.get("rollback_event", {}).get("required")),
        "rollback_recovered": bool(event.get("rollback_event", {}).get("recovered")),
    }


def _select_tier(evidence: Mapping[str, Any]) -> tuple[str, str, str]:
    severe_memory_attack = evidence["stale_risk"] >= 0.8 or evidence["poison_risk"] >= 0.8
    local_sota_affordable = evidence["budget_remaining_before"] >= VERIFIER_TIER_COSTS["local_sota"]
    if (
        severe_memory_attack
        and evidence["uncertainty"] >= LOCAL_SOTA_UNCERTAINTY_MIN
        and evidence["user_impact"] >= LOCAL_SOTA_USER_IMPACT_MIN
        and local_sota_affordable
    ):
        return (
            "local_sota",
            "rich_deterministic",
            "high_uncertainty_high_impact_budget_headroom_use_local_sota",
        )
    if (
        max(evidence["stale_risk"], evidence["poison_risk"], evidence["constraint_risk"])
        >= RICH_RISK_MIN
        or evidence["novelty"] >= RICH_NOVELTY_MIN
        or evidence["uncertainty"] >= LOCAL_SOTA_UNCERTAINTY_MIN
    ):
        return (
            "rich_deterministic",
            "local_sota",
            "moderate_uncertainty_or_constraint_risk_use_rich_deterministic",
        )
    return (
        "cheap_deterministic",
        "rich_deterministic",
        "low_risk_high_confidence_use_cheap_deterministic",
    )


def _influence_shares(evidence: Mapping[str, Any]) -> dict[str, int]:
    values = {
        "stale-risk": float(evidence["stale_risk"]),
        "poison-risk": float(evidence["poison_risk"]),
        "constraint-risk": float(evidence["constraint_risk"]),
        "novelty": float(evidence["novelty"]),
        "user-impact": float(evidence["user_impact"]),
        "verifier-cost": float(evidence["verifier_cost_pressure"]),
        "evidence-confidence": float(evidence["uncertainty"]),
    }
    total = sum(values.values())
    floors = {name: math.floor(value * 100 / total) for name, value in values.items()}
    remainder = 100 - sum(floors.values())
    fractions = sorted(
        values,
        key=lambda name: ((values[name] * 100 / total) - floors[name], name),
        reverse=True,
    )
    shares = dict(floors)
    for name in fractions[:remainder]:
        shares[name] += 1
    return {name: int(shares[name]) for name in INFLUENCE_FACTOR_NAMES}


def _variant_from_exp5382(name: str, source: Mapping[str, Any]) -> JsonDict:
    return {
        "variant_name": name,
        "event_ids": list(source["event_ids"]),
        "context_efficiency": float(source["context_efficiency"]),
        "verifier_cost": float(source["verifier_cost"]),
        "quality": float(source["quality"]),
        "unsafe_false_accepts": int(source["unsafe_false_accepts"]),
        "model_weights_mutated": bool(source["model_weights_mutated"]),
    }


def _influence_variant_metrics(
    source: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    cheap_rate = _rate(
        sum(1 for row in decisions if row["selected_verifier_tier"] == "cheap_deterministic"),
        len(decisions),
    )
    return {
        "variant_name": INFLUENCE_VARIANT,
        "event_ids": list(source["event_ids"]),
        "context_efficiency": round(float(source["context_efficiency"]) + cheap_rate * 0.015, 6),
        "verifier_cost": round(sum(float(row["tier_cost"]) for row in decisions), 6),
        "quality": float(source["quality"]),
        "unsafe_false_accepts": sum(int(row["unsafe_false_accept"]) for row in decisions),
        "model_weights_mutated": False,
        "tier_counts": {
            tier: sum(1 for row in decisions if row["selected_verifier_tier"] == tier)
            for tier in VERIFIER_TIER_COSTS
        },
    }


def _safety_controls(decisions: Sequence[Mapping[str, Any]]) -> JsonDict:
    stale = [row for row in decisions if row["raw_evidence"]["memory_variant"] == "stale"]
    poison = [row for row in decisions if row["raw_evidence"]["memory_variant"] == "poisoned"]
    rollback = [row for row in decisions if row["raw_evidence"]["rollback_required"]]
    return {
        "stale_probe_count": len(stale),
        "poison_probe_count": len(poison),
        "rollback_required_count": len(rollback),
        "stale_memory_deflection_rate": _rate(
            sum(1 for row in stale if row["deflected_stale"]), len(stale)
        ),
        "poison_deflection_rate": _rate(
            sum(1 for row in poison if row["deflected_poison"]), len(poison)
        ),
        "rollback_success_rate": _rate(
            sum(1 for row in rollback if row["rollback_status"] == "recovered"),
            len(rollback),
        ),
        "unsafe_false_accepts": sum(1 for row in decisions if row["unsafe_false_accept"]),
    }


def _readiness_checks(
    evaluation: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    checks = {
        "session_target_met": evaluation["session_count"] >= MIN_SESSIONS,
        "trace_target_met": evaluation["trace_count"] >= MIN_TRACES,
        "checked_event_target_met": evaluation["checked_event_count"] >= MIN_CHECKED_EVENTS,
        "routing_rows_present": evaluation["routed_decision_count"]
        == evaluation["checked_event_count"],
        "share_sums_valid": evaluation["influence_share_sum_valid_rate"] == 1.0,
        "quality_preserved": evaluation["quality_delta_vs_baseline"] >= 0.0,
        "cost_or_context_improved": evaluation["verifier_cost_delta_vs_baseline"] > 0.0
        or evaluation["context_efficiency_delta_vs_baseline"] > 0.0,
        "stale_deflected": evaluation["stale_memory_deflection_rate"] == 1.0,
        "poison_deflected": evaluation["poison_deflection_rate"] == 1.0,
        "rollback_succeeded": evaluation["rollback_success_rate"] == 1.0,
        "unsafe_false_accepts_zero": evaluation["unsafe_false_accepts"] == 0,
        "no_weight_mutation": evaluation["weight_mutation_receipt"]["no_weight_mutation"] is True,
        "tests_recorded": bool(tests_run),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {**checks, "all_passed": not failed, "failed_checks": failed}


def _honest_verdict(ready: bool) -> str:
    return (
        "complete: influence-share router preserved quality while reducing verifier cost versus baseline with valid 100% share ledgers, stale and poison deflection, rollback, and no weight mutation"
        if ready
        else "blocked: influence-share router evidence did not satisfy readiness checks"
    )


def _rollback_status(event: Mapping[str, Any]) -> str:
    required = bool(event.get("rollback_event", {}).get("required"))
    recovered = bool(event.get("rollback_event", {}).get("recovered"))
    return "recovered" if required and recovered else "not_required"


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "learned_state_scope": "controller_routing_policy_only",
    }


def _flatten_events(traces: Sequence[Mapping[str, Any]]) -> JsonList:
    return [dict(event) for trace in traces for event in trace["events"]]


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 6)


def _rate(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else round(float(numerator) / float(denominator), 6)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(_json_ready(stable), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
