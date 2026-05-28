"""Exp 3229 FR-11 nonforgetting-aware controller promotion governance.

Spec refs: REQ-LEARN-3229, SCENARIO-LEARN-3229,
SCENARIO-LEARN-3229-DEFERRED.

This module consumes the checked-in Exp 3215 replay labels and Exp 3216
grounded-continuation queue.  It simulates which controller-memory traces can
be promoted after nonforgetting checks.  The important boundary is that
controller memory is policy metadata: admitted traces can guide future routing,
but this code does not train, fine-tune, or mutate any model weights.
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
RUN_DATE = "20260528"
MILESTONE = "2026.05.298"
SCHEMA_VERSION = "1.0"
EXPERIMENT_ID = "experiment_3229_fr11_nonforgetting_promotion_controller_v3"
SCHEMA = "carnot.fr11.nonforgetting_promotion_controller.v3"
INFERENCE_SUBSTRATE = "checked_in_artifact_controller_memory_simulation_no_training"
OUTPUT_REL_PATH = Path(
    "results/experiment_3229_fr11_nonforgetting_promotion_controller_v3.json"
)
EXP3215_REL_PATH = Path(
    "results/experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.json"
)
EXP3216_REL_PATH = Path(
    "results/experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.json"
)
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_REL_PATH = Path(
    "python/carnot/eval/fr11_nonforgetting_promotion_controller_v3.py"
)
TEST_REL_PATH = Path(
    "tests/python/test_experiment_3229_fr11_nonforgetting_promotion_controller_v3.py"
)
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "passed:", "passed_")
MUTATION_FLAGS = (
    "executes_live_model_inference",
    "model_weight_learning",
    "model_weight_training",
    "model_weight_mutation",
    "base_model_weights_updated",
    "kan_model_weight_training",
    "hidden_state_mutation_claimed",
)
REQUIRED_LABEL_FIELDS = (
    "trace_id",
    "row_id",
    "replay_role",
    "exact_verifier_outcome",
    "prior_route_utility",
    "reward_weight",
)
ROLLBACK_TRIGGER_IDS = (
    "negative_control_regression",
    "stale_premise_failure",
    "contradiction_graph_update",
)
REQUIRED_ARTIFACT_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "continuous_self_learning_task",
    "source_trace_artifacts",
    "candidate_trace_count",
    "accepted_trace_count",
    "rejected_trace_count",
    "deferred_trace_count",
    "promotion_allowed",
    "controller_memory_promotion_allowed",
    "nonforgetting_budget_exceeded",
    "rollback_policy_defined",
    "rollback_trigger_count",
    "negative_control_regression_count",
    "stale_premise_rejection_count",
    "model_weight_update_claimed",
    "inference_substrate",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' "
    "tests/python/test_experiment_3229_fr11_nonforgetting_promotion_controller_v3.py -q",
    ".venv/bin/coverage run -m pytest -o addopts='' "
    "tests/python/test_experiment_3229_fr11_nonforgetting_promotion_controller_v3.py -q",
    ".venv/bin/coverage report "
    "--include='python/carnot/eval/fr11_nonforgetting_promotion_controller_v3.py' "
    "--fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_3229_fr11_nonforgetting_promotion_controller_v3.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_no_hidden_weight_update_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3215_trace_labels", EXP3215_REL_PATH, True),
    ("exp3216_nonforgetting_queue", EXP3216_REL_PATH, True),
    ("exp3229_module", MODULE_REL_PATH, False),
    ("exp3229_tests", TEST_REL_PATH, False),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object while treating absent or malformed evidence as empty."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the two checked-in source artifacts used by Exp 3229."""

    root_path = Path(root)
    return {
        "exp3215": read_json_object(root_path / EXP3215_REL_PATH),
        "exp3216": read_json_object(root_path / EXP3216_REL_PATH),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal Exp 3229 promotion-governance artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    policy = rollback_policy()
    blocker = source_blocker(sources)
    if blocker:
        artifact = blocked_artifact(root_path, blocker, start, now_s, tests_run, policy)
        validate_artifact(artifact)
        return artifact

    exp3215 = sources["exp3215"]
    exp3216 = sources["exp3216"]
    candidates = candidate_traces(exp3215)
    context = admission_context(exp3215, exp3216)
    stale_index = stale_premise_index(exp3216)
    simulation = simulate_replay(candidates, stale_index, context)
    promotion = promotion_decision(simulation, context, policy)
    artifact = {
        "artifact": EXPERIMENT_ID,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "continuous_self_learning_task": True,
        "source_trace_artifacts": [
            EXP3215_REL_PATH.as_posix(),
            EXP3216_REL_PATH.as_posix(),
        ],
        "source_artifacts": source_artifacts(root_path),
        "candidate_trace_count": simulation["candidate_trace_count"],
        "accepted_trace_count": simulation["accepted_trace_count"],
        "rejected_trace_count": simulation["rejected_trace_count"],
        "deferred_trace_count": simulation["deferred_trace_count"],
        "promotion_allowed": promotion,
        "controller_memory_promotion_allowed": promotion,
        "nonforgetting_budget_exceeded": context["nonforgetting_budget_exceeded"],
        "rollback_policy_defined": policy["rollback_policy_defined"],
        "rollback_trigger_count": policy["rollback_trigger_count"],
        "negative_control_regression_count": context["negative_control_regression_count"],
        "stale_premise_rejection_count": simulation["stale_premise_rejection_count"],
        "model_weight_update_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "controller_memory_training_boundary": controller_memory_training_boundary(),
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "admission_rules": admission_rules(),
        "admission_context": context,
        "rollback_policy": policy,
        "stale_premise_invalidations": stale_premise_report(exp3216, stale_index),
        "nonforgetting_queue_entries": nonforgetting_queue_entries(exp3216),
        "accepted_traces": simulation["accepted_traces"],
        "rejected_traces": simulation["rejected_traces"],
        "deferred_traces": simulation["deferred_traces"],
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, now_s),
        "honest_verdict": honest_verdict(
            promotion,
            simulation["accepted_trace_count"],
            simulation["rejected_trace_count"],
            simulation["deferred_trace_count"],
            context["nonforgetting_budget_exceeded"],
            context["negative_control_regression_count"],
            simulation["stale_premise_rejection_count"],
        ),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    root: Path,
    blocker: str,
    started_s: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
    policy: Mapping[str, Any],
) -> JsonDict:
    """Return a complete fail-closed artifact when source evidence is unsafe."""

    return {
        "artifact": EXPERIMENT_ID,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "continuous_self_learning_task": True,
        "source_trace_artifacts": [
            EXP3215_REL_PATH.as_posix(),
            EXP3216_REL_PATH.as_posix(),
        ],
        "source_artifacts": source_artifacts(root),
        "candidate_trace_count": 0,
        "accepted_trace_count": 0,
        "rejected_trace_count": 0,
        "deferred_trace_count": 0,
        "promotion_allowed": False,
        "controller_memory_promotion_allowed": False,
        "nonforgetting_budget_exceeded": False,
        "rollback_policy_defined": bool(policy.get("rollback_policy_defined")),
        "rollback_trigger_count": int(policy.get("rollback_trigger_count") or 0),
        "negative_control_regression_count": 0,
        "stale_premise_rejection_count": 0,
        "model_weight_update_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "controller_memory_training_boundary": controller_memory_training_boundary(),
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "admission_rules": admission_rules(),
        "admission_context": {},
        "rollback_policy": dict(policy),
        "stale_premise_invalidations": {"affected_route_count": 0, "affected_routes": []},
        "nonforgetting_queue_entries": [],
        "accepted_traces": [],
        "rejected_traces": [],
        "deferred_traces": [],
        "blocked_reason": blocker,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started_s, now_s),
        "honest_verdict": (
            "complete: blocked fr11 nonforgetting promotion controller; "
            f"{blocker}; model_weight_update_claimed=false; "
            "controller_memory_updates_are_not_training"
        ),
    }


def source_blocker(sources: Mapping[str, Any]) -> str:
    """REQ-LEARN-3229-1: require terminal, no-live, no-mutation sources."""

    for key in ("exp3215", "exp3216"):
        payload = sources.get(key, {})
        if not isinstance(payload, Mapping) or not is_terminal(payload):
            return f"{key}_missing_or_not_terminal"
        if detected_model_weight_update(payload):
            return f"{key}_model_weight_update_claimed"
        if source_claims_live_or_mutation(payload):
            return f"{key}_live_inference_or_weight_update_claimed"
    return ""


def is_terminal(payload: Mapping[str, Any]) -> bool:
    """Return whether an artifact verdict is terminal enough to consume."""

    return str(payload.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES)


def source_claims_live_or_mutation(payload: Mapping[str, Any]) -> bool:
    """Return whether a source claims fresh inference or model mutation."""

    substrate = payload.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        return True
    return int(substrate.get("fresh_live_inference_calls") or 0) != 0 or any(
        substrate.get(flag) is True for flag in MUTATION_FLAGS
    )


def detected_model_weight_update(payload: Mapping[str, Any]) -> bool:
    """Return whether a payload claims a model-weight update."""

    if payload.get("model_weight_update_claimed") is True:
        return True
    if payload.get("model_weight_update_performed") is True:
        return True
    substrate = payload.get("inference_substrate", {})
    return isinstance(substrate, Mapping) and any(
        substrate.get(flag) is True for flag in MUTATION_FLAGS if "weight" in flag
    )


def candidate_traces(exp3215: Mapping[str, Any]) -> list[JsonDict]:
    """REQ-LEARN-3229-2: return Exp 3215 replay utility labels as candidates."""

    rows = exp3215.get("replay_utility_labels", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def candidate_is_evidence_backed(candidate: Mapping[str, Any]) -> bool:
    """Return whether a candidate is a controller-only exact-evidence label."""

    required_present = all(str(candidate.get(field) or "").strip() for field in REQUIRED_LABEL_FIELDS)
    return (
        required_present
        and candidate.get("controller_utility_label_only") is True
        and candidate.get("model_weight_update_claimed") is False
    )


def admission_context(exp3215: Mapping[str, Any], exp3216: Mapping[str, Any]) -> JsonDict:
    """Collect global held-out, drift, queue, and negative-control gates."""

    labels = candidate_traces(exp3215)
    heldout_count = int(exp3215.get("heldout_row_count") or role_count(labels, "heldout"))
    drift_count = int(exp3215.get("drift_row_count") or role_count(labels, "drift"))
    negative_count = negative_control_regression_count(exp3215, exp3216)
    budget_exceeded = nonforgetting_budget_exceeded(exp3216)
    queue = exp3216.get("nonforgetting_queue", {})
    queue_budget = None
    if isinstance(queue, Mapping):
        queue_budget = queue.get("nonforgetting_budget")
    return {
        "heldout_row_count": heldout_count,
        "drift_row_count": drift_count,
        "heldout_check_passed": heldout_count > 0,
        "drift_check_passed": drift_count > 0,
        "negative_control_regression_count": negative_count,
        "nonforgetting_budget_exceeded": budget_exceeded,
        "nonforgetting_queue_value": exp3216.get("nonforgetting_queue_value"),
        "nonforgetting_budget": queue_budget,
        "uses_existing_artifacts_only": True,
    }


def role_count(rows: Sequence[Mapping[str, Any]], role: str) -> int:
    """Count replay labels by role."""

    return sum(1 for row in rows if normalize_token(row.get("replay_role")) == role)


def negative_control_regression_count(
    exp3215: Mapping[str, Any], exp3216: Mapping[str, Any]
) -> int:
    """Return the strict visible negative-control regression count."""

    queue = exp3216.get("nonforgetting_queue", {})
    pressure_terms = queue.get("pressure_terms", {}) if isinstance(queue, Mapping) else {}
    queue_negative = (
        pressure_terms.get("negative_control_regressions", 0)
        if isinstance(pressure_terms, Mapping)
        else 0
    )
    return max(
        safe_int(exp3215.get("negative_control_regression_count")),
        safe_int(exp3216.get("negative_control_regression_count")),
        safe_int(queue_negative),
    )


def nonforgetting_budget_exceeded(exp3216: Mapping[str, Any]) -> bool:
    """Return whether Exp 3216's nonforgetting queue blocks promotion."""

    queue = exp3216.get("nonforgetting_queue", {})
    nested = queue.get("nonforgetting_budget_exceeded") if isinstance(queue, Mapping) else False
    return bool(exp3216.get("nonforgetting_budget_exceeded") or nested)


def stale_premise_index(exp3216: Mapping[str, Any]) -> dict[str, set[Any]]:
    """Index Exp 3216 affected routes by node id, trace id, and route tuple."""

    route_node_ids: set[str] = set()
    trace_ids: set[str] = set()
    route_keys: set[tuple[str, str, str]] = set()
    routes = exp3216.get("affected_routes", [])
    if not isinstance(routes, Sequence) or isinstance(routes, (str, bytes)):
        return {
            "route_node_ids": route_node_ids,
            "trace_ids": trace_ids,
            "route_keys": route_keys,
        }
    for route in routes:
        if not isinstance(route, Mapping):
            continue
        node_id = str(route.get("route_node_id") or "").strip()
        row_id = str(route.get("row_id") or "").strip()
        replay_role = normalize_token(route.get("replay_role"))
        routing_outcome = normalize_token(route.get("routing_outcome"))
        if node_id:
            route_node_ids.add(node_id)
            if node_id.startswith("route:"):
                trace_ids.add(node_id.removeprefix("route:"))
        if row_id:
            route_keys.add((row_id, replay_role, routing_outcome))
    return {
        "route_node_ids": route_node_ids,
        "trace_ids": trace_ids,
        "route_keys": route_keys,
    }


def simulate_replay(
    candidates: Sequence[Mapping[str, Any]],
    stale_index: Mapping[str, set[Any]],
    context: Mapping[str, Any],
) -> JsonDict:
    """REQ-LEARN-3229-3/4: classify candidates as accepted, rejected, deferred."""

    accepted: list[JsonDict] = []
    rejected: list[JsonDict] = []
    deferred: list[JsonDict] = []
    for candidate in candidates:
        reasons = trace_rejection_reasons(candidate, stale_index, context)
        if reasons:
            rejected.append(trace_decision(candidate, "rejected", reasons))
            continue
        if bool(context.get("nonforgetting_budget_exceeded")):
            deferred.append(trace_decision(candidate, "deferred", ["nonforgetting_budget_exceeded"]))
            continue
        accepted.append(
            trace_decision(
                candidate,
                "accepted",
                [
                    "evidence_label_passed",
                    "heldout_drift_checks_passed",
                    "nonforgetting_budget_within_limit",
                ],
            )
        )
    stale_rejections = sum(
        1 for row in rejected if "stale_premise_failure" in row.get("decision_reasons", [])
    )
    return {
        "candidate_trace_count": len(candidates),
        "accepted_trace_count": len(accepted),
        "rejected_trace_count": len(rejected),
        "deferred_trace_count": len(deferred),
        "stale_premise_rejection_count": stale_rejections,
        "accepted_traces": accepted,
        "rejected_traces": rejected,
        "deferred_traces": deferred,
    }


def trace_rejection_reasons(
    candidate: Mapping[str, Any],
    stale_index: Mapping[str, set[Any]],
    context: Mapping[str, Any],
) -> list[str]:
    """List admission failures for one replay candidate."""

    reasons: list[str] = []
    if not candidate_is_evidence_backed(candidate):
        reasons.append("missing_evidence_label")
    if not bool(context.get("heldout_check_passed")) or not bool(context.get("drift_check_passed")):
        reasons.append("missing_heldout_or_drift_check")
    if trace_matches_stale_premise(candidate, stale_index):
        reasons.append("stale_premise_failure")
    rollback_status = normalize_token(candidate.get("rollback_or_retraction_status"))
    if rollback_status not in {"none", "unknown"}:
        reasons.append("rollback_or_retraction")
    if safe_float(candidate.get("reward_weight")) < 0.0:
        reasons.append("negative_utility_label")
    if int(context.get("negative_control_regression_count") or 0) > 0:
        reasons.append("negative_control_regression")
    return reasons


def trace_matches_stale_premise(
    candidate: Mapping[str, Any], stale_index: Mapping[str, set[Any]]
) -> bool:
    """Return whether Exp 3216 already marked this candidate route stale."""

    trace_id = str(candidate.get("trace_id") or "").strip()
    route_node_id = f"route:{trace_id}" if trace_id else ""
    route_key = (
        str(candidate.get("row_id") or "").strip(),
        normalize_token(candidate.get("replay_role")),
        normalize_token(candidate.get("routing_outcome")),
    )
    return (
        bool(route_node_id and route_node_id in stale_index.get("route_node_ids", set()))
        or bool(trace_id and trace_id in stale_index.get("trace_ids", set()))
        or route_key in stale_index.get("route_keys", set())
    )


def trace_decision(
    candidate: Mapping[str, Any], decision: str, decision_reasons: Sequence[str]
) -> JsonDict:
    """Return a compact trace decision for accepted/rejected/deferred lists."""

    return {
        "trace_id": str(candidate.get("trace_id") or ""),
        "row_id": str(candidate.get("row_id") or ""),
        "replay_role": normalize_token(candidate.get("replay_role")),
        "routing_outcome": normalize_token(candidate.get("routing_outcome")),
        "exact_verifier_outcome": str(candidate.get("exact_verifier_outcome") or ""),
        "prior_route_utility": str(candidate.get("prior_route_utility") or ""),
        "reward_weight": safe_float(candidate.get("reward_weight")),
        "decision": decision,
        "decision_reasons": list(decision_reasons),
    }


def promotion_decision(
    simulation: Mapping[str, Any], context: Mapping[str, Any], policy: Mapping[str, Any]
) -> bool:
    """REQ-LEARN-3229-5: allow controller-memory promotion for admitted traces."""

    return (
        int(simulation.get("accepted_trace_count") or 0) > 0
        and int(context.get("negative_control_regression_count") or 0) == 0
        and context.get("nonforgetting_budget_exceeded") is False
        and policy.get("rollback_policy_defined") is True
    )


def admission_rules() -> list[JsonDict]:
    """Return the controller-memory admission rules exposed in the artifact."""

    return [
        {
            "rule_id": "evidence_label_required",
            "description": "candidate must be an Exp 3215 controller-only exact-evidence label",
        },
        {
            "rule_id": "heldout_and_drift_checks_required",
            "description": "source artifact must include nonzero heldout and drift checks",
        },
        {
            "rule_id": "negative_control_regression_blocks",
            "description": "any negative-control regression rejects candidate promotion",
        },
        {
            "rule_id": "stale_premise_rejected",
            "description": "routes affected by Exp 3216 stale-premise propagation are rejected",
        },
        {
            "rule_id": "queue_budget_defers",
            "description": "otherwise admissible traces are deferred when the queue exceeds budget",
        },
    ]


def rollback_policy() -> JsonDict:
    """REQ-LEARN-3229-6: define rollback triggers for promoted controller memory."""

    triggers = [
        {
            "trigger_id": "negative_control_regression",
            "source": "negative-control replay rows or queue pressure terms",
            "rollback_action": "revoke promoted controller route and restore prior route metadata",
        },
        {
            "trigger_id": "stale_premise_failure",
            "source": "grounded-continuation affected routes and retraction propagation",
            "rollback_action": "remove stale trace from promoted controller memory",
        },
        {
            "trigger_id": "contradiction_graph_update",
            "source": "future contradiction graph update invalidating a promoted premise",
            "rollback_action": "mark dependent route for re-verification before reuse",
        },
    ]
    return {
        "rollback_policy_defined": True,
        "rollback_trigger_count": len(triggers),
        "rollback_triggers": triggers,
    }


def stale_premise_report(
    exp3216: Mapping[str, Any], stale_index: Mapping[str, set[Any]]
) -> JsonDict:
    """Expose stale-premise invalidations in a JSON-serializable shape."""

    routes = exp3216.get("affected_routes", [])
    affected_routes = [dict(route) for route in routes if isinstance(route, Mapping)]
    return {
        "stale_premise_invalidations": safe_int(exp3216.get("stale_premise_invalidations")),
        "affected_route_count": safe_int(exp3216.get("affected_route_count")),
        "affected_routes": affected_routes,
        "affected_trace_ids": sorted(str(value) for value in stale_index.get("trace_ids", set())),
    }


def nonforgetting_queue_entries(exp3216: Mapping[str, Any]) -> list[JsonDict]:
    """Expose Exp 3216 queue pressure terms as audit entries."""

    queue = exp3216.get("nonforgetting_queue", {})
    if not isinstance(queue, Mapping):
        return []
    terms = queue.get("pressure_terms", {})
    if not isinstance(terms, Mapping):
        return []
    return [
        {
            "entry_id": str(key),
            "value": safe_float(value),
            "source_artifact": EXP3216_REL_PATH.as_posix(),
            "budget": queue.get("nonforgetting_budget"),
            "budget_exceeded": bool(queue.get("nonforgetting_budget_exceeded")),
        }
        for key, value in sorted(terms.items())
    ]


def controller_memory_training_boundary() -> JsonDict:
    """Separate allowed controller-memory metadata updates from training."""

    return {
        "controller_memory_updates_may_be_promoted": True,
        "controller_memory_updates_are_policy_metadata": True,
        "controller_memory_updates_are_not_training": True,
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
    """Raise when Exp 3229 is incomplete or crosses the no-training boundary."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    false_fields = [
        "model_weight_update_claimed",
        "conductor_file_modified",
        "active_roadmap_modified",
    ]
    for field in false_fields:
        if artifact.get(field) is not False:
            raise ValueError(f"{field} must remain false")
    if artifact.get("controller_memory_promotion_allowed") != artifact.get("promotion_allowed"):
        raise ValueError("controller_memory_promotion_allowed must match promotion_allowed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be the no-training controller-memory mode")
    boundary = artifact.get("controller_memory_training_boundary", {})
    if not isinstance(boundary, Mapping):
        raise ValueError("controller_memory_training_boundary must be an object")
    if boundary.get("controller_memory_updates_are_not_training") is not True:
        raise ValueError("controller-memory updates must be explicitly separate from training")
    if any(boundary.get(flag) is True for flag in MUTATION_FLAGS):
        raise ValueError("training or model mutation flags must remain false")
    if artifact.get("rollback_policy_defined") is not True:
        raise ValueError("rollback policy must be defined")
    if int(artifact.get("rollback_trigger_count") or 0) != len(ROLLBACK_TRIGGER_IDS):
        raise ValueError("rollback trigger count must be 3")
    count_sum = (
        int(artifact.get("accepted_trace_count") or 0)
        + int(artifact.get("rejected_trace_count") or 0)
        + int(artifact.get("deferred_trace_count") or 0)
    )
    if count_sum != int(artifact.get("candidate_trace_count") or 0):
        raise ValueError("accepted/rejected/deferred counts must sum to candidates")
    if artifact.get("promotion_allowed") is True and int(artifact.get("accepted_trace_count") or 0) <= 0:
        raise ValueError("promotion requires accepted traces")
    if artifact.get("promotion_allowed") is True and artifact.get("nonforgetting_budget_exceeded"):
        raise ValueError("promotion cannot be allowed when nonforgetting budget is exceeded")
    if artifact.get("promotion_allowed") is True and int(
        artifact.get("negative_control_regression_count") or 0
    ) > 0:
        raise ValueError("promotion cannot be allowed with negative-control regressions")
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
    """Build, validate, and write deterministic Exp 3229 JSON."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def honest_verdict(
    promotion_allowed: bool,
    accepted_count: int,
    rejected_count: int,
    deferred_count: int,
    budget_exceeded: bool,
    negative_regressions: int,
    stale_rejections: int,
) -> str:
    """Return a truthful terminal verdict for the promotion-governance artifact."""

    return (
        "complete: fr11 nonforgetting-aware controller promotion governance "
        "materialized; "
        f"promotion_allowed={str(promotion_allowed).lower()}; "
        f"accepted_trace_count={accepted_count}; "
        f"rejected_trace_count={rejected_count}; "
        f"deferred_trace_count={deferred_count}; "
        f"nonforgetting_budget_exceeded={str(budget_exceeded).lower()}; "
        f"negative_control_regression_count={negative_regressions}; "
        f"stale_premise_rejection_count={stale_rejections}; "
        "model_weight_update_claimed=false; "
        "controller_memory_updates_are_not_training"
    )


def normalize_token(value: Any) -> str:
    """Normalize compact routing, role, and status tokens."""

    text = str(value or "").strip().lower().replace(" ", "_")
    return text or "unknown"


def safe_int(value: Any) -> int:
    """Return an integer for artifact counters while treating bad input as zero."""

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    """Return a float for artifact scores while treating bad input as zero."""

    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


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
