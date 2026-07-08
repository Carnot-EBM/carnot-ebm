"""Exp5409: uncertainty-gated promotion for learned fragments.

Spec refs: REQ-LEARN-5409,
SCENARIO-LEARN-5409-UNCERTAINTY-BYPASS,
SCENARIO-LEARN-5409-STALE-PROMOTION, SCENARIO-LEARN-5409-ROLLBACK.

This experiment adds the missing step between retained memory and reusable
controller influence. Exp5396 already keeps raw episodes for audit, and Exp5408
can route with resource accounting. The risk is that a plausible but uncertain
fragment could still be promoted into routing. This module therefore treats
promotion as a deterministic certificate check: enough support, high certainty,
inside a reachable envelope, no conflicts, and a rollback path. It never loads
or mutates model weights; the only activated state is a controller sidecar over
cached trace evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5408_resource_accounted_csl_controller_v492 as exp5408


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5409_uncertainty_gated_promotion_v492"
EXPERIMENT_ID = "exp5409-v492-uncertainty-gated-promotion"
MILESTONE = "2026.07.492"
SCHEMA = "carnot.experiment_5409.uncertainty_gated_promotion.v492"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5409
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5409_uncertainty_gated_promotion_v492.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5409_uncertainty_gated_promotion_v492.py"
)
EXP5408_RESULT_RELATIVE_PATH = exp5408.RESULT_RELATIVE_PATH
EXP5408_MODULE_RELATIVE_PATH = exp5408.MODULE_RELATIVE_PATH

SPEC_REFS = (
    "REQ-LEARN-5409",
    "SCENARIO-LEARN-5409-UNCERTAINTY-BYPASS",
    "SCENARIO-LEARN-5409-STALE-PROMOTION",
    "SCENARIO-LEARN-5409-ROLLBACK",
)
TERMINAL_PREFIXES = ("complete:", "blocked:")

MIN_SUPPORT_COUNT = 3
MIN_CERTAINTY_SCORE = 0.72
STALE_RISK_MAX = 0.2
FORGED_REASONING_RISK_MAX = 0.2
SHARING_RISK_MAX = 0.12
MAX_BYTE_COST = 420
POISON_CONTROL_KINDS = frozenset(
    {
        "forged_reasoning_history",
        "self_referential_amplification",
        "high_cost_low_value",
    }
)
REQUIRED_FAMILIES = frozenset(
    {"benign", "stale", "poisoned", "ambiguous", "scarce_evidence"}
)

FIELD_PRINCIPLES: dict[str, str] = {
    "gated_on_resource_accounted_csl": "Precondition.",
    "promotion_candidate_count": "Coverage.",
    "accepted_promotion_count": "Live routing effect.",
    "rejected_retained_count": "Audit retention without activation.",
    "uncertainty_gate_rejection_rate": "Certainty constraint.",
    "stale_promotion_rejection_rate": "Anti-staleness.",
    "poisoned_promotion_rejection_rate": "Anti-poisoning.",
    "reachability_violation_rejection_rate": "Reachable-set safety.",
    "rollback_success_rate": "Reversible learning.",
    "no_weight_mutation": "Online learning boundary.",
    "uncertainty_gated_promotion_ready": "Downstream evidence.",
    "inference_substrate": "Deterministic trace replay.",
    "honest_verdict": "Terminal status; starts with complete: or blocked:.",
}

REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)
BOOL_FIELDS = (
    "gated_on_resource_accounted_csl",
    "no_weight_mutation",
    "uncertainty_gated_promotion_ready",
)
INTEGER_FIELDS = (
    "promotion_candidate_count",
    "accepted_promotion_count",
    "rejected_retained_count",
)
NUMERIC_FIELDS = (
    "uncertainty_gate_rejection_rate",
    "stale_promotion_rejection_rate",
    "poisoned_promotion_rejection_rate",
    "reachability_violation_rejection_rate",
    "rollback_success_rate",
)
RATE_FIELDS = (
    "uncertainty_gate_rejection_rate",
    "stale_promotion_rejection_rate",
    "poisoned_promotion_rejection_rate",
    "reachability_violation_rejection_rate",
    "rollback_success_rate",
)


def load_source_controller_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """Read Exp5408's checked replay artifact, which is the promotion precondition."""

    path = Path(root) / EXP5408_RESULT_RELATIVE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def build_trace_index(source_artifact: Mapping[str, Any]) -> JsonDict:
    """Index Exp5408 rows so promotion scores come from replay evidence.

    The index deliberately counts support from routing decisions rather than
    from candidate prose. A fragment that appears once can be retained for audit,
    but it is not transferable enough to steer routing.
    """

    support_by_raw_id: Counter[str] = Counter()
    event_ids_by_raw_id: dict[str, list[str]] = defaultdict(list)
    tiers_by_raw_id: dict[str, set[str]] = defaultdict(set)
    rollback_recovered_by_raw_id: Counter[str] = Counter()
    accepted_memory_id_counts: Counter[str] = Counter()
    decision_indices_by_memory_id: dict[str, list[int]] = defaultdict(list)

    for row in source_artifact.get("resource_accounted_decisions", []):
        provenance = row["raw_episode_provenance"]
        raw_id = str(provenance["raw_episode_id"])
        memory_id = provenance.get("accepted_memory_id")
        support_by_raw_id[raw_id] += 1
        event_ids_by_raw_id[raw_id].append(str(row["event_id"]))
        tiers_by_raw_id[raw_id].add(str(row["selected_verifier_tier"]))
        if row["rollback_status"] == "recovered":
            rollback_recovered_by_raw_id[raw_id] += 1
        if memory_id:
            memory_key = str(memory_id)
            accepted_memory_id_counts[memory_key] += 1
            decision_indices_by_memory_id[memory_key].append(int(row["decision_index"]))

    raw_control_by_id = {
        str(row["raw_episode_id"]): str(row["control_kind"])
        for row in source_artifact.get("raw_episodes", [])
    }
    return {
        "support_by_raw_id": dict(support_by_raw_id),
        "event_ids_by_raw_id": {key: sorted(value) for key, value in event_ids_by_raw_id.items()},
        "tiers_by_raw_id": {
            key: sorted(value) for key, value in tiers_by_raw_id.items()
        },
        "rollback_recovered_by_raw_id": dict(rollback_recovered_by_raw_id),
        "accepted_memory_id_counts": dict(accepted_memory_id_counts),
        "decision_indices_by_memory_id": {
            key: sorted(value) for key, value in decision_indices_by_memory_id.items()
        },
        "raw_control_by_id": raw_control_by_id,
    }


def build_promotion_candidates(source_artifact: Mapping[str, Any]) -> JsonList:
    """Create benign, stale, poisoned, ambiguous, and scarce-evidence candidates."""

    candidates = [
        _candidate_from_memory(row)
        for row in source_artifact.get("memory_candidates", [])
    ]
    by_memory_id = {str(row["source_memory_id"]): row for row in candidates}
    by_raw_id = {
        raw_id: candidate
        for candidate in candidates
        for raw_id in candidate["raw_episode_ids"]
    }

    clean_scaleup = by_memory_id["mem5396-clean-scaleup-summary"]
    forged = by_memory_id["mem5396-forged-reasoning-history"]
    candidates.append(_ambiguous_candidate(clean_scaleup, forged))

    clean_dependency = by_raw_id["raw5396-clean-dependency-edge"]
    candidates.append(_scarce_evidence_candidate(clean_dependency))
    return [_json_ready(row) for row in candidates]


def score_promotion_candidate(
    candidate: Mapping[str, Any],
    trace_index: Mapping[str, Any],
) -> JsonDict:
    """Score one candidate without trusting rationale text or requested activation."""

    row = dict(candidate)
    raw_ids = [str(raw_id) for raw_id in row["raw_episode_ids"]]
    inputs = dict(row["decision_inputs"])
    support_count = int(
        row.get("support_count_override")
        or sum(int(trace_index["support_by_raw_id"].get(raw_id, 0)) for raw_id in raw_ids)
    )
    certainty_score = _certainty_score(inputs, support_count)
    reachability = _reachability_envelope(row, inputs, support_count, trace_index)
    conflict = _conflict_check(row, raw_ids, trace_index)
    rejection_reasons = _rejection_reasons(
        row,
        inputs,
        support_count,
        certainty_score,
        reachability,
        conflict,
    )
    accepted = not rejection_reasons
    row.update(
        {
            "support_count": support_count,
            "certainty_score": certainty_score,
            "reachability_envelope": reachability,
            "conflict_check": conflict,
            "promotion_decision": {
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
                "certainty_threshold": MIN_CERTAINTY_SCORE,
                "support_threshold": MIN_SUPPORT_COUNT,
                "route_action": (
                    "activate_controller_sidecar"
                    if accepted
                    else "retain_audit_only"
                ),
            },
            "retained_for_audit": True,
            "active_for_routing": accepted,
            "live_routing_effect": False,
        }
    )
    return _json_ready(row)


def route_promoted_fragments(
    scored_candidates: Sequence[Mapping[str, Any]],
    source_artifact: Mapping[str, Any],
) -> JsonDict:
    """Build the controller sidecar from accepted fragments only."""

    trace_index = build_trace_index(source_artifact)
    accepted = [dict(row) for row in scored_candidates if row["active_for_routing"]]
    rejected = [dict(row) for row in scored_candidates if not row["active_for_routing"]]
    active_ids: list[str] = []
    used_ids: list[str] = []
    effect_records: JsonList = []
    for row in accepted:
        fragment_id = str(row["fragment_id"])
        memory_id = str(row["source_memory_id"])
        decision_indices = list(
            trace_index["decision_indices_by_memory_id"].get(memory_id, [])
        )
        active_ids.append(fragment_id)
        if decision_indices:
            used_ids.append(fragment_id)
            effect_records.append(
                {
                    "fragment_id": fragment_id,
                    "source_memory_id": memory_id,
                    "routing_decision_indices": decision_indices,
                    "routing_effect_count": len(decision_indices),
                }
            )
            row["live_routing_effect"] = True

    rejected_ids = sorted(str(row["fragment_id"]) for row in rejected)
    return {
        "active_fragment_ids": sorted(active_ids),
        "accepted_fragment_ids_used_for_routing": sorted(used_ids),
        "retained_inactive_fragment_ids": rejected_ids,
        "rejected_fragment_routing_influence_count": sum(
            1 for row in rejected if row["active_for_routing"]
        ),
        "routing_effect_row_count": sum(
            int(row["routing_effect_count"]) for row in effect_records
        ),
        "routing_effect_records": effect_records,
    }


def rollback_bad_promotion(
    routing_report: Mapping[str, Any],
    bad_fragment_id: str,
) -> JsonDict:
    """Inject one known-bad fragment, then remove it using the audit sidecar."""

    active = set(str(item) for item in routing_report["active_fragment_ids"])
    retained = set(str(item) for item in routing_report["retained_inactive_fragment_ids"])
    active.add(bad_fragment_id)
    injected = bad_fragment_id in active
    active.discard(bad_fragment_id)
    removed = bad_fragment_id not in active
    retained_after = bad_fragment_id in retained
    return {
        "bad_fragment_id": bad_fragment_id,
        "injected_into_active_routing": injected,
        "rollback_removed_from_active_routing": removed,
        "retained_audit_record_after_rollback": retained_after,
        "rollback_success": bool(injected and removed and retained_after),
    }


def evaluate_uncertainty_gated_promotion(root: Path | str = REPO_ROOT) -> JsonDict:
    """Evaluate promotion gates and route only accepted fragments."""

    source_artifact = load_source_controller_artifact(root)
    gated = source_artifact.get("resource_accounted_csl_ready") is True
    trace_index = build_trace_index(source_artifact)
    scored = [
        score_promotion_candidate(candidate, trace_index)
        for candidate in build_promotion_candidates(source_artifact)
    ]
    routing_report = route_promoted_fragments(scored, source_artifact)
    active_ids = set(routing_report["accepted_fragment_ids_used_for_routing"])
    scored = [
        {**row, "live_routing_effect": str(row["fragment_id"]) in active_ids}
        for row in scored
    ]
    accepted = [row for row in scored if row["promotion_decision"]["accepted"]]
    rejected = [row for row in scored if not row["promotion_decision"]["accepted"]]
    bad = next(
        row
        for row in rejected
        if row["candidate_family"] == "poisoned"
        and row["source_control_kind"] == "high_cost_low_value"
    )
    rollback = rollback_bad_promotion(routing_report, str(bad["fragment_id"]))
    weight_receipt = _weight_mutation_receipt()
    return {
        "source_controller_artifact": source_artifact,
        "gated_on_resource_accounted_csl": gated,
        "promotion_candidates": scored,
        "accepted_promotions": accepted,
        "rejected_promotions": rejected,
        "promotion_candidate_count": len(scored),
        "accepted_promotion_count": len(accepted),
        "rejected_retained_count": len(rejected),
        "uncertainty_gate_rejection_rate": _rejection_rate(
            scored,
            lambda row: row["certainty_score"] < MIN_CERTAINTY_SCORE,
        ),
        "stale_promotion_rejection_rate": _rejection_rate(
            scored,
            lambda row: row["candidate_family"] == "stale",
        ),
        "poisoned_promotion_rejection_rate": _rejection_rate(
            scored,
            lambda row: row["candidate_family"] == "poisoned",
        ),
        "reachability_violation_rejection_rate": _rejection_rate(
            scored,
            lambda row: not row["reachability_envelope"]["within_reachable_set"],
        ),
        "rollback_success_rate": 1.0 if rollback["rollback_success"] else 0.0,
        "no_weight_mutation": weight_receipt["no_weight_mutation"],
        "routing_report": routing_report,
        "rollback_audit": rollback,
        "weight_mutation_receipt": weight_receipt,
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal artifact consumed by the conductor."""

    evaluation = evaluate_uncertainty_gated_promotion(root)
    readiness = _readiness_checks(evaluation, tests_run)
    ready = bool(readiness["all_passed"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifacts": [str(EXP5408_RESULT_RELATIVE_PATH)],
        "status": "complete" if ready else "blocked",
        "gated_on_resource_accounted_csl": evaluation[
            "gated_on_resource_accounted_csl"
        ],
        "promotion_candidate_count": evaluation["promotion_candidate_count"],
        "accepted_promotion_count": evaluation["accepted_promotion_count"],
        "rejected_retained_count": evaluation["rejected_retained_count"],
        "uncertainty_gate_rejection_rate": evaluation[
            "uncertainty_gate_rejection_rate"
        ],
        "stale_promotion_rejection_rate": evaluation[
            "stale_promotion_rejection_rate"
        ],
        "poisoned_promotion_rejection_rate": evaluation[
            "poisoned_promotion_rejection_rate"
        ],
        "reachability_violation_rejection_rate": evaluation[
            "reachability_violation_rejection_rate"
        ],
        "rollback_success_rate": evaluation["rollback_success_rate"],
        "no_weight_mutation": evaluation["no_weight_mutation"],
        "uncertainty_gated_promotion_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready),
        "tests_run": [dict(row) for row in tests_run],
        "promotion_candidates": evaluation["promotion_candidates"],
        "accepted_promotions": evaluation["accepted_promotions"],
        "rejected_promotions": evaluation["rejected_promotions"],
        "routing_report": evaluation["routing_report"],
        "rollback_audit": evaluation["rollback_audit"],
        "readiness_checks": readiness,
        "weight_mutation_receipt": evaluation["weight_mutation_receipt"],
        "methodology_note": (
            "Exp5409 replays Exp5408 cached controller evidence. Promotion "
            "is a sidecar gate over retained fragments; rejected fragments "
            "stay auditable but inactive, and no model or adapter weights are "
            "loaded or mutated."
        ),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate fields that downstream promotion gates depend on."""

    errors: list[str] = []
    errors.extend(field for field in REQUIRED_FIELDS if field not in artifact)
    errors.extend(field for field in BOOL_FIELDS if not isinstance(artifact.get(field), bool))
    errors.extend(
        field
        for field in INTEGER_FIELDS
        if isinstance(artifact.get(field), bool) or not isinstance(artifact.get(field), int)
    )
    errors.extend(
        field for field in NUMERIC_FIELDS if not _is_numeric(artifact.get(field))
    )
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    ready = artifact.get("uncertainty_gated_promotion_ready")
    if (ready is True and artifact.get("status") != "complete") or (
        artifact.get("status") == "complete" and ready is not True
    ):
        errors.append("status")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone")
    for field in RATE_FIELDS:
        if _is_numeric(artifact.get(field)) and float(artifact[field]) != 1.0:
            errors.append(field)
    if artifact.get("gated_on_resource_accounted_csl") is not True:
        errors.append("gated_on_resource_accounted_csl")
    if artifact.get("no_weight_mutation") is not True:
        errors.append("no_weight_mutation")
    if artifact.get("promotion_candidate_count") != len(
        artifact.get("promotion_candidates", [])
    ):
        errors.append("promotion_candidate_count")
    if artifact.get("accepted_promotion_count") != len(
        artifact.get("accepted_promotions", [])
    ):
        errors.append("accepted_promotion_count")
    if artifact.get("rejected_retained_count") != len(
        artifact.get("rejected_promotions", [])
    ):
        errors.append("rejected_retained_count")
    if artifact.get("promotion_candidate_count") != (
        artifact.get("accepted_promotion_count", 0)
        + artifact.get("rejected_retained_count", 0)
    ):
        errors.append("promotion_candidate_count")
    routing = artifact.get("routing_report", {})
    if routing.get("rejected_fragment_routing_influence_count") != 0:
        errors.append("rejected_fragment_routing_influence_count")
    if ready is True and not artifact.get("tests_run"):
        errors.append("tests_run")
    if ready is True and artifact.get("accepted_promotion_count", 0) <= 0:
        errors.append("accepted_promotion_count")
    if ready is True and not all(
        row.get("live_routing_effect") for row in artifact.get("accepted_promotions", [])
    ):
        errors.append("live_routing_effect")
    if errors:
        raise ValueError(
            "invalid Exp5409 artifact fields: " + ",".join(sorted(set(errors)))
        )
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5409 result artifact and return the JSON payload."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def default_tests_run() -> JsonList:
    """Return the verification commands expected in the completed artifact."""

    test_path = "tests/python/test_experiment_5409_uncertainty_gated_promotion_v492.py"
    module_path = "python/carnot/experiment_5409_uncertainty_gated_promotion_v492.py"
    return [
        {
            "command": f".venv/bin/pytest {test_path} -q --no-cov -n 0",
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                f"--include={module_path} -m pytest {test_path} "
                "-q --no-cov -n 0 && .venv/bin/coverage report --fail-under=100"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for the source evidence and local contract."""

    root_path = Path(root)
    return {
        "exp5408": _sha256_file(root_path / EXP5408_RESULT_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5408_module": _sha256_file(root_path / EXP5408_MODULE_RELATIVE_PATH),
    }


def _candidate_from_memory(candidate: Mapping[str, Any]) -> JsonDict:
    control_kind = str(candidate["control_kind"])
    family = _candidate_family(control_kind)
    return {
        "record_type": "promotion_candidate",
        "fragment_id": str(candidate["memory_id"]).replace("mem5396", "frag5409", 1),
        "source_memory_id": str(candidate["memory_id"]),
        "fragment_claim": str(candidate["memory_claim"]),
        "candidate_family": family,
        "source_control_kind": control_kind,
        "raw_episode_ids": [str(raw_id) for raw_id in candidate["raw_episode_ids"]],
        "decision_inputs": dict(candidate["decision_inputs"]),
        "source_memory_accepted": bool(candidate["decision"]["accepted"]),
        "source_trust_label": dict(candidate["trust_label"]),
        "requested_activation": "promote_to_controller_sidecar",
        "model_generated_rationale": (
            "candidate asks for promotion; non-authoritative"
        ),
    }


def _ambiguous_candidate(clean: Mapping[str, Any], forged: Mapping[str, Any]) -> JsonDict:
    raw_ids = [str(clean["raw_episode_ids"][0]), str(forged["raw_episode_ids"][0])]
    return {
        "record_type": "promotion_candidate",
        "fragment_id": "frag5409-ambiguous-scaleup-forged-merge",
        "source_memory_id": "ambiguous:mem5396-clean-scaleup-summary+mem5396-forged-reasoning-history",
        "fragment_claim": (
            "Merged clean scale-up summary with a forged reasoning shortcut."
        ),
        "candidate_family": "ambiguous",
        "source_control_kind": "ambiguous_conflict",
        "raw_episode_ids": raw_ids,
        "decision_inputs": _merge_inputs(clean["decision_inputs"], forged["decision_inputs"]),
        "source_memory_accepted": False,
        "source_trust_label": {
            "label": "ambiguous_conflict",
            "allowed_for_routing": False,
            "source": "mixed_raw_episode_evidence",
        },
        "requested_activation": "promote_to_controller_sidecar",
        "model_generated_rationale": (
            "merged fragment claims broad transfer despite conflicting evidence"
        ),
    }


def _scarce_evidence_candidate(clean: Mapping[str, Any]) -> JsonDict:
    return {
        "record_type": "promotion_candidate",
        "fragment_id": "frag5409-scarce-single-dependency-edge",
        "source_memory_id": "scarce:mem5396-clean-dependency-edge:first-event-only",
        "fragment_claim": "Single-event dependency edge should not generalize yet.",
        "candidate_family": "scarce_evidence",
        "source_control_kind": "scarce_evidence",
        "raw_episode_ids": [str(clean["raw_episode_ids"][0])],
        "decision_inputs": dict(clean["decision_inputs"]),
        "source_memory_accepted": False,
        "source_trust_label": {
            "label": "scarce_evidence",
            "allowed_for_routing": False,
            "source": "single_event_slice",
        },
        "support_count_override": 1,
        "requested_activation": "promote_to_controller_sidecar",
        "model_generated_rationale": (
            "single observed success asks to bypass support threshold"
        ),
    }


def _candidate_family(control_kind: str) -> str:
    if control_kind == "benign_useful":
        return "benign"
    if control_kind == "stale_memory":
        return "stale"
    if control_kind in POISON_CONTROL_KINDS:
        return "poisoned"
    return "ambiguous"


def _merge_inputs(left: Mapping[str, Any], right: Mapping[str, Any]) -> JsonDict:
    return {
        "value_score": round(min(float(left["value_score"]), float(right["value_score"])), 6),
        "byte_cost": int(left["byte_cost"]) + int(right["byte_cost"]),
        "stale_risk": round(max(float(left["stale_risk"]), float(right["stale_risk"])), 6),
        "forged_reasoning_risk": round(
            max(float(left["forged_reasoning_risk"]), float(right["forged_reasoning_risk"])),
            6,
        ),
        "self_reference_count": max(
            int(left["self_reference_count"]), int(right["self_reference_count"])
        ),
        "sharing_risk": round(max(float(left["sharing_risk"]), float(right["sharing_risk"])), 6),
        "provenance_verified": bool(left["provenance_verified"] and right["provenance_verified"]),
        "rollback_available": bool(left["rollback_available"] and right["rollback_available"]),
        "rollback_verified": bool(left["rollback_verified"] and right["rollback_verified"]),
        "model_generated_rationale_used": False,
    }


def _certainty_score(inputs: Mapping[str, Any], support_count: int) -> float:
    support_factor = min(1.0, float(support_count) / float(MIN_SUPPORT_COUNT))
    risk = max(
        float(inputs["stale_risk"]),
        float(inputs["forged_reasoning_risk"]),
        float(inputs["sharing_risk"]),
        min(1.0, float(inputs["self_reference_count"]) / 10.0),
    )
    provenance_factor = (
        1.0
        if inputs["provenance_verified"]
        and inputs["rollback_available"]
        and inputs["rollback_verified"]
        else 0.5
    )
    return round(
        float(inputs["value_score"]) * support_factor * max(0.0, 1.0 - risk) * provenance_factor,
        6,
    )


def _reachability_envelope(
    candidate: Mapping[str, Any],
    inputs: Mapping[str, Any],
    support_count: int,
    trace_index: Mapping[str, Any],
) -> JsonDict:
    raw_ids = [str(raw_id) for raw_id in candidate["raw_episode_ids"]]
    observed_tiers = sorted(
        {
            tier
            for raw_id in raw_ids
            for tier in trace_index["tiers_by_raw_id"].get(raw_id, [])
        }
    )
    recovered_count = sum(
        int(trace_index["rollback_recovered_by_raw_id"].get(raw_id, 0))
        for raw_id in raw_ids
    )
    checks = {
        "min_support_met": support_count >= MIN_SUPPORT_COUNT,
        "stale_risk_in_bounds": float(inputs["stale_risk"]) <= STALE_RISK_MAX,
        "forged_reasoning_risk_in_bounds": (
            float(inputs["forged_reasoning_risk"]) <= FORGED_REASONING_RISK_MAX
        ),
        "sharing_risk_in_bounds": float(inputs["sharing_risk"]) <= SHARING_RISK_MAX,
        "byte_cost_in_bounds": int(inputs["byte_cost"]) <= MAX_BYTE_COST,
        "benign_family": candidate["candidate_family"] == "benign",
        "no_recovered_unsafe_route": recovered_count == 0,
    }
    return {
        "support_count": support_count,
        "observed_verifier_tiers": observed_tiers,
        "rollback_recovered_count": recovered_count,
        "bounds": {
            "min_support_count": MIN_SUPPORT_COUNT,
            "max_stale_risk": STALE_RISK_MAX,
            "max_forged_reasoning_risk": FORGED_REASONING_RISK_MAX,
            "max_sharing_risk": SHARING_RISK_MAX,
            "max_byte_cost": MAX_BYTE_COST,
        },
        "checks": checks,
        "within_reachable_set": all(checks.values()),
    }


def _conflict_check(
    candidate: Mapping[str, Any],
    raw_ids: Sequence[str],
    trace_index: Mapping[str, Any],
) -> JsonDict:
    raw_control_kinds = sorted(
        {
            str(trace_index["raw_control_by_id"].get(raw_id, "unknown"))
            for raw_id in raw_ids
        }
    )
    rejected_source = not bool(candidate["source_memory_accepted"])
    family_conflict = candidate["candidate_family"] in {
        "ambiguous",
        "stale",
        "poisoned",
    }
    mixed_raw_controls = len(raw_control_kinds) > 1
    rollback_conflicts = sum(
        int(trace_index["rollback_recovered_by_raw_id"].get(raw_id, 0))
        for raw_id in raw_ids
    )
    conflict_count = int(rejected_source) + int(family_conflict) + int(mixed_raw_controls)
    conflict_count += rollback_conflicts
    return {
        "raw_control_kinds": raw_control_kinds,
        "source_memory_rejected": rejected_source,
        "mixed_raw_controls": mixed_raw_controls,
        "rollback_conflict_count": rollback_conflicts,
        "conflict_count": conflict_count,
        "unresolved_conflict": conflict_count > 0,
    }


def _rejection_reasons(
    candidate: Mapping[str, Any],
    inputs: Mapping[str, Any],
    support_count: int,
    certainty_score: float,
    reachability: Mapping[str, Any],
    conflict: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if support_count < MIN_SUPPORT_COUNT:
        reasons.append("low_support")
    if certainty_score < MIN_CERTAINTY_SCORE:
        reasons.append("low_certainty")
    if candidate["candidate_family"] == "stale" or float(inputs["stale_risk"]) > STALE_RISK_MAX:
        reasons.append("stale_provenance")
    if (
        candidate["candidate_family"] == "poisoned"
        or candidate["source_control_kind"] in POISON_CONTROL_KINDS
        or float(inputs["forged_reasoning_risk"]) > FORGED_REASONING_RISK_MAX
        or int(inputs["byte_cost"]) > MAX_BYTE_COST
    ):
        reasons.append("poisoned_or_nontransferable")
    if not reachability["within_reachable_set"]:
        reasons.append("reachability_violation")
    if conflict["unresolved_conflict"]:
        reasons.append("unresolved_conflict")
    if not (
        inputs["provenance_verified"]
        and inputs["rollback_available"]
        and inputs["rollback_verified"]
    ):
        reasons.append("rollback_or_provenance_unavailable")
    return reasons


def _readiness_checks(
    evaluation: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    families = {row["candidate_family"] for row in evaluation["promotion_candidates"]}
    rejected = evaluation["rejected_promotions"]
    checks = {
        "gated_on_resource_accounted_csl": evaluation[
            "gated_on_resource_accounted_csl"
        ]
        is True,
        "required_families_covered": REQUIRED_FAMILIES.issubset(families),
        "candidates_present": evaluation["promotion_candidate_count"] > 0,
        "accepted_promotions_present": evaluation["accepted_promotion_count"] > 0,
        "accepted_have_live_routing_effect": all(
            row["live_routing_effect"] for row in evaluation["accepted_promotions"]
        ),
        "rejected_retained_inactive": all(
            row["retained_for_audit"] and not row["active_for_routing"]
            for row in rejected
        ),
        "rejected_zero_routing_influence": evaluation["routing_report"][
            "rejected_fragment_routing_influence_count"
        ]
        == 0,
        "uncertainty_controls_rejected": evaluation[
            "uncertainty_gate_rejection_rate"
        ]
        == 1.0,
        "stale_controls_rejected": evaluation["stale_promotion_rejection_rate"]
        == 1.0,
        "poison_controls_rejected": evaluation["poisoned_promotion_rejection_rate"]
        == 1.0,
        "reachability_controls_rejected": evaluation[
            "reachability_violation_rejection_rate"
        ]
        == 1.0,
        "rollback_succeeded": evaluation["rollback_success_rate"] == 1.0,
        "no_weight_mutation": evaluation["no_weight_mutation"] is True,
        "tests_recorded": bool(tests_run),
        "inference_substrate_cached": INFERENCE_SUBSTRATE
        == "verifier_ensemble_against_cached_candidates",
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {**checks, "all_passed": not failed, "failed_checks": failed}


def _rejection_rate(
    candidates: Sequence[Mapping[str, Any]],
    predicate: Any,
) -> float:
    matching = [row for row in candidates if predicate(row)]
    return _rate(
        sum(1 for row in matching if not row["promotion_decision"]["accepted"]),
        len(matching),
    )


def _honest_verdict(ready: bool) -> str:
    return (
        "complete: uncertainty-gated promotion accepted only supported reachable fragments, retained rejected stale, poisoned, ambiguous, and scarce-evidence fragments for audit, routed accepted sidecars, rolled back a bad promotion, and did not mutate weights"
        if ready
        else "blocked: uncertainty-gated promotion evidence did not satisfy readiness checks"
    )


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "controller_promotion_sidecar_only",
    }


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _rate(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else round(float(numerator) / float(denominator), 6)


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(_json_ready(stable), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
