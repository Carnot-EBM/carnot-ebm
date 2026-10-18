"""Exp 3098 MaxSAT-style accept/reject/abstain routing policy.

Spec refs: REQ-VERIFY-3098, SCENARIO-VERIFY-3098.

The policy is intentionally offline. It does not solve a live MaxSAT instance
or call an LLM; instead it publishes the hard clauses, weighted soft clauses,
and deterministic reference evaluator that later `.289` live SOTA experiments
must consume. That makes the claim boundary explicit before expensive verifier
or repair reruns start.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3098_maxsat_abstention_routing_policy_v1"
SCHEMA = "carnot.maxsat_abstention_routing_policy_artifact.v1"
POLICY_SCHEMA = "carnot.maxsat_abstention_routing_policy.v1"
OUTPUT_REL_PATH = Path("results/experiment_3098_maxsat_abstention_routing_policy_v1.json")
POLICY_REL_PATH = Path("results/maxsat_abstention_routing_policy_3098/policy.json")
EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
EXP3085_REL_PATH = Path("results/experiment_3085_icalm_task_abstention_sota_panel_v2.json")
EXP3087_REL_PATH = Path("results/experiment_3087_gated_local_sota_verifier_calibration_v3.json")
EXP3094_REL_PATH = Path("results/experiment_3094_capstone_v288.json")
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = (
    "maxsat_policy_ready",
    "routing_policy_path",
    "hard_constraints",
    "soft_constraints",
    "objective_terms",
    "fallback_evaluator",
    "downstream_usage",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
REQUIRED_HARD_TOPICS = {
    "exact_label_disagreement",
    "model_cache_availability",
    "formal_feedback_lift",
    "syntax_schema_validity",
    "repair_intent_preservation",
    "no_tiny_panel_disqualification",
}
REQUIRED_SOFT_TOPICS = {
    "accept_exact_consistent",
    "reject_exact_inconsistent",
    "abstain_on_uncertainty",
    "prefer_formal_feedback_lift",
    "preserve_repair_intent",
    "minimize_false_accept",
    "minimize_false_reject",
    "minimize_unnecessary_abstention",
}
SOURCE_REL_PATHS: tuple[tuple[str, Path, str], ...] = (
    ("codex", Path("CODEX.md"), "repo spec-first workflow"),
    ("claude", Path("CLAUDE.md"), "artifact-authenticity and tiny-panel discipline"),
    ("research_references", Path("research-references.md"), "local literature context"),
    ("exp3097_protocol", EXP3097_REL_PATH, ".289 exact-fixture protocol authority"),
    ("exp3085_abstention", EXP3085_REL_PATH, ".288 tiny abstention-panel failure"),
    ("exp3087_calibration_gate", EXP3087_REL_PATH, ".288 calibration gate failure"),
    ("exp3094_capstone", EXP3094_REL_PATH, ".288 capstone and .289 blocker summary"),
)
OPENREVIEW_SOURCE = {
    "id": "openreview_qmr9vbwrab",
    "url": "https://openreview.net/forum?id=Qmr9VbwRaB",
    "role": "weighted MaxSAT/MaxSMT routing design context",
    "published": "2026-03-05",
    "last_modified": "2026-04-25",
}


@dataclass(frozen=True)
class PolicyConfig:
    """Runtime paths for the offline policy artifact."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    routing_policy_path: Path | None = None
    started_s: float | None = None
    clock: ClockFn = time.perf_counter

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def policy_path(self) -> Path:
        return self.routing_policy_path or self.repo_root / POLICY_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_s is None else float(self.started_s)


def write_artifact(config: PolicyConfig | None = None) -> JsonDict:
    """Build and write the terminal Exp 3098 artifact."""

    active = config or PolicyConfig()
    started_s = active.start_time()
    artifact = build_artifact(active, started_s=started_s)
    write_json(active.artifact_path(), artifact)
    return artifact


def build_artifact(
    config: PolicyConfig | None = None, *, started_s: float | None = None
) -> JsonDict:
    """Return the authority artifact and write the referenced policy JSON."""

    active = config or PolicyConfig()
    started = active.start_time() if started_s is None else started_s
    policy = build_policy_document()
    policy_path = active.policy_path()
    write_json(policy_path, policy)
    sources = source_artifacts(active.repo_root)
    exp3097 = safe_load_json(active.repo_root / EXP3097_REL_PATH)
    ready = policy_ready(
        policy=policy,
        policy_path=policy_path,
        source_rows=sources,
        exp3097=exp3097,
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "maxsat_policy_ready": ready,
        "routing_policy_path": relative_path(active.repo_root, policy_path),
        "routing_policy_sha256": sha256_file(policy_path),
        "hard_constraints": policy["hard_constraints"],
        "soft_constraints": policy["soft_constraints"],
        "objective_terms": policy["objective_terms"],
        "fallback_evaluator": policy["fallback_evaluator"],
        "downstream_usage": downstream_usage(relative_path(active.repo_root, policy_path)),
        "source_artifacts": sources,
        "inference_substrate": inference_substrate(),
        "duration_s": active.clock() - started,
        "honest_verdict": honest_verdict(ready),
    }
    validate_artifact(artifact)
    return artifact


def build_policy_document() -> JsonDict:
    """Build the machine-readable routing policy consumed by later experiments."""

    return {
        "schema": POLICY_SCHEMA,
        "version": "v1",
        "run_date": RUN_DATE,
        "actions": ["accept", "reject", "abstain"],
        "hard_constraints": hard_constraints(),
        "soft_constraints": soft_constraints(),
        "objective_terms": objective_terms(),
        "fallback_evaluator": fallback_evaluator(),
        "design_reference": OPENREVIEW_SOURCE,
    }


def hard_constraints() -> list[JsonDict]:
    """Return non-negotiable safety clauses for routing decisions."""

    return [
        {
            "id": "HC_EXACT_LABEL_AGREEMENT",
            "topic": "exact_label_disagreement",
            "description": (
                "Accept is infeasible when the candidate disagrees with exact labels; "
                "reject is infeasible for an exact-consistent safe candidate."
            ),
            "blocks_actions": ["accept", "reject"],
        },
        {
            "id": "HC_MODEL_CACHE_AVAILABLE",
            "topic": "model_cache_availability",
            "description": (
                "Headline live accept/reject routes require a mandated local GGUF cache; "
                "missing cache leaves only abstain/blocked evidence."
            ),
            "blocks_actions": ["accept", "reject"],
        },
        {
            "id": "HC_FORMAL_FEEDBACK_NONNEGATIVE",
            "topic": "formal_feedback_lift",
            "description": (
                "Repair promotion cannot accept when measured formal-feedback lift is negative."
            ),
            "blocks_actions": ["accept"],
        },
        {
            "id": "HC_SYNTAX_SCHEMA_VALID",
            "topic": "syntax_schema_validity",
            "description": "Accept requires both syntax validity and declared schema validity.",
            "blocks_actions": ["accept"],
        },
        {
            "id": "HC_REPAIR_INTENT_PRESERVED",
            "topic": "repair_intent_preservation",
            "description": "Accepting a repair candidate requires exact intent preservation.",
            "blocks_actions": ["accept"],
        },
        {
            "id": "HC_NO_TINY_PANEL_HEADLINE",
            "topic": "no_tiny_panel_disqualification",
            "description": (
                "Headline accept/reject decisions are infeasible below the exact-count floor."
            ),
            "blocks_actions": ["accept", "reject"],
        },
    ]


def soft_constraints() -> list[JsonDict]:
    """Return weighted preferences used after hard clauses filter actions."""

    return [
        {
            "id": "SC_ACCEPT_EXACT",
            "topic": "accept_exact_consistent",
            "weight": 100,
            "reward_action": "accept",
            "when": "exact_label_match=true and expected_action=accept",
        },
        {
            "id": "SC_REJECT_INEXACT",
            "topic": "reject_exact_inconsistent",
            "weight": 100,
            "reward_action": "reject",
            "when": "exact_label_match=false or expected_action=reject",
        },
        {
            "id": "SC_ABSTAIN_UNCERTAIN",
            "topic": "abstain_on_uncertainty",
            "weight": 70,
            "reward_action": "abstain",
            "when": "confidence < abstention_confidence_floor",
        },
        {
            "id": "SC_FORMAL_LIFT",
            "topic": "prefer_formal_feedback_lift",
            "weight": 40,
            "reward_action": "accept",
            "when": "formal_feedback_delta > 0",
        },
        {
            "id": "SC_INTENT",
            "topic": "preserve_repair_intent",
            "weight": 35,
            "reward_action": "accept",
            "when": "repair_candidate=false or repair_intent_preserved=true",
        },
        {
            "id": "SC_FALSE_ACCEPT",
            "topic": "minimize_false_accept",
            "weight": 120,
            "reward_action": "reject_or_abstain",
            "when": "candidate is exact-inconsistent or expected_action=reject",
        },
        {
            "id": "SC_FALSE_REJECT",
            "topic": "minimize_false_reject",
            "weight": 60,
            "reward_action": "accept_or_abstain",
            "when": "candidate is exact-consistent, safe, and expected_action=accept",
        },
        {
            "id": "SC_AVOID_ABSTAIN",
            "topic": "minimize_unnecessary_abstention",
            "weight": 20,
            "reward_action": "accept_or_reject",
            "when": "confidence >= abstention_confidence_floor and a hard-safe action exists",
        },
    ]


def objective_terms() -> JsonDict:
    """Declare the optimization target in a solver-independent form."""

    return {
        "maximize": "sum of satisfied weighted soft constraints after hard filtering",
        "hard_constraint_penalty": "infinite",
        "abstention_confidence_floor": 0.70,
        "tie_break_order": ["abstain", "reject", "accept"],
        "weights": {row["topic"]: row["weight"] for row in soft_constraints()},
    }


def fallback_evaluator() -> JsonDict:
    """Describe the deterministic evaluator used when no MaxSAT package exists."""

    return {
        "kind": "deterministic_reference_evaluator",
        "used_when": "no local MaxSAT or MaxSMT solver package is installed",
        "enumerates_actions": ["accept", "reject", "abstain"],
        "filters_hard_constraint_violations": True,
        "scores_remaining_actions": "declared soft weights only",
        "tie_break_order": ["abstain", "reject", "accept"],
        "fail_closed_default": "abstain",
        "silent_solver_fallback_allowed": False,
    }


def evaluate_route(case: Mapping[str, Any], policy: Mapping[str, Any] | None = None) -> JsonDict:
    """Evaluate one case with the deterministic fallback policy."""

    active = dict(policy or build_policy_document())
    actions = list(active["actions"])
    feasible = [action for action in actions if not hard_violations(action, case)]
    if not feasible:
        feasible = ["abstain"]
    scores = {action: soft_score(action, case, active) for action in feasible}
    best_score = max(scores.values())
    tied = [
        action
        for action in active["fallback_evaluator"]["tie_break_order"]
        if action in feasible and scores[action] == best_score
    ]
    decision = tied[0] if tied else "abstain"
    return {
        "decision": decision,
        "used_solver": "deterministic_reference_evaluator",
        "hard_feasible_actions": feasible,
        "blocked_actions": [action for action in actions if action not in feasible],
        "scores": scores,
        "score_breakdown": {action: soft_score_breakdown(action, case, active) for action in feasible},
        "hard_violations": {action: hard_violations(action, case) for action in actions},
    }


def hard_violations(action: str, case: Mapping[str, Any]) -> list[str]:
    """Return hard-constraint IDs violated by an attempted route action."""

    violations: list[str] = []
    if action in {"accept", "reject"}:
        if bool(case.get("headline_claim", True)) and not bool(
            case.get("model_cache_available", False)
        ):
            violations.append("HC_MODEL_CACHE_AVAILABLE")
        if bool(case.get("headline_claim", True)) and int(
            case.get("exact_ground_truth_count") or 0
        ) < int(case.get("minimum_live_eval_count") or 48):
            violations.append("HC_NO_TINY_PANEL_HEADLINE")
    if action == "accept":
        if str(case.get("expected_action", "")) == "reject" or not bool(
            case.get("exact_label_match", False)
        ):
            violations.append("HC_EXACT_LABEL_AGREEMENT")
        if not bool(case.get("syntax_valid", False)) or not bool(case.get("schema_valid", False)):
            violations.append("HC_SYNTAX_SCHEMA_VALID")
        if bool(case.get("repair_candidate", False)) and not bool(
            case.get("repair_intent_preserved", False)
        ):
            violations.append("HC_REPAIR_INTENT_PRESERVED")
        if bool(case.get("repair_promotion", False)) and float(
            case.get("formal_feedback_delta") or 0.0
        ) < 0.0:
            violations.append("HC_FORMAL_FEEDBACK_NONNEGATIVE")
    if action == "reject" and exact_consistent_safe_accept(case):
        violations.append("HC_EXACT_LABEL_AGREEMENT")
    return violations


def soft_score(action: str, case: Mapping[str, Any], policy: Mapping[str, Any]) -> int:
    """Return the summed soft-clause score for one feasible action."""

    return sum(item["weight"] for item in soft_score_breakdown(action, case, policy))


def soft_score_breakdown(
    action: str, case: Mapping[str, Any], policy: Mapping[str, Any]
) -> list[JsonDict]:
    """List the satisfied soft clauses for auditability."""

    weights = policy["objective_terms"]["weights"]
    confidence = float(case.get("confidence") or 0.0)
    threshold = float(policy["objective_terms"]["abstention_confidence_floor"])
    exact_match = bool(case.get("exact_label_match", False))
    expected_accept = str(case.get("expected_action", "")) == "accept"
    expected_reject = str(case.get("expected_action", "")) == "reject"
    exact_inconsistent = (not exact_match) or expected_reject
    breakdown: list[JsonDict] = []
    add = lambda topic: breakdown.append({"topic": topic, "weight": int(weights[topic])})
    if action == "accept" and exact_match and expected_accept:
        add("accept_exact_consistent")
    if action == "reject" and exact_inconsistent:
        add("reject_exact_inconsistent")
    if action == "abstain" and confidence < threshold:
        add("abstain_on_uncertainty")
    if action == "accept" and float(case.get("formal_feedback_delta") or 0.0) > 0.0:
        add("prefer_formal_feedback_lift")
    if action == "accept" and (
        not bool(case.get("repair_candidate", False))
        or bool(case.get("repair_intent_preserved", False))
    ):
        add("preserve_repair_intent")
    if action in {"reject", "abstain"} and exact_inconsistent:
        add("minimize_false_accept")
    if action in {"accept", "abstain"} and exact_consistent_safe_accept(case):
        add("minimize_false_reject")
    if action in {"accept", "reject"} and confidence >= threshold:
        add("minimize_unnecessary_abstention")
    return breakdown


def exact_consistent_safe_accept(case: Mapping[str, Any]) -> bool:
    """Whether rejecting would be a false reject under exact and safety evidence."""

    return (
        str(case.get("expected_action", "")) == "accept"
        and bool(case.get("exact_label_match", False))
        and bool(case.get("syntax_valid", False))
        and bool(case.get("schema_valid", False))
        and (
            not bool(case.get("repair_candidate", False))
            or bool(case.get("repair_intent_preserved", False))
        )
    )


def downstream_usage(routing_policy_path: str) -> JsonDict:
    """Declare how downstream `.289` tasks must consume the same policy."""

    return {
        "exp3099": {
            "task": "local SOTA confidence abstention panel v3",
            "must_load_policy_from": routing_policy_path,
            "required_field": "maxsat_policy_used",
            "required_behavior": (
                "route every accept/reject/abstain row through the policy and report policy decisions"
            ),
        },
        "exp3101": {
            "task": "local SOTA verifier calibration v4",
            "must_load_policy_from": routing_policy_path,
            "required_metric": "verifier_gain_delta_with_maxsat",
            "required_behavior": (
                "compare raw verifier gain against MaxSAT-routed gain using the same policy"
            ),
        },
        "exp3102": {
            "task": "gated structured repair micro-panel v3",
            "must_load_policy_from": routing_policy_path,
            "gate": "exp3101.verifier_gain_delta_with_maxsat > 0.0",
            "required_behavior": (
                "run repair only after positive MaxSAT-routed verifier gain and recheck syntax, schema, exact semantics, and intent preservation"
            ),
        },
    }


def policy_ready(
    *,
    policy: Mapping[str, Any],
    policy_path: Path,
    source_rows: list[JsonDict],
    exp3097: Mapping[str, Any],
) -> bool:
    """Check the readiness predicate without inferring missing evidence."""

    local_sources_exist = all(bool(row.get("exists")) for row in source_rows if row.get("path"))
    return (
        policy_path.is_file()
        and local_sources_exist
        and exp3097.get("eval_protocol_ready") is True
        and REQUIRED_HARD_TOPICS <= {row["topic"] for row in policy["hard_constraints"]}
        and REQUIRED_SOFT_TOPICS <= {row["topic"] for row in policy["soft_constraints"]}
        and policy["fallback_evaluator"].get("fail_closed_default") == "abstain"
        and policy["fallback_evaluator"].get("silent_solver_fallback_allowed") is False
        and set(downstream_usage(relative_path(REPO_ROOT, POLICY_REL_PATH))) == {
            "exp3099",
            "exp3101",
            "exp3102",
        }
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the artifact overstates policy readiness."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(artifact.get("routing_policy_path") or ""):
        raise ValueError("routing policy path must be non-empty")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or substrate.get("no_live_llm_inference") is not True:
        raise ValueError("inference_substrate must declare no live model inference")
    if substrate.get("executes_models") is not False:
        raise ValueError("policy design must not execute models")
    if REQUIRED_HARD_TOPICS - {row["topic"] for row in artifact.get("hard_constraints", [])}:
        raise ValueError("hard constraints do not cover all required topics")
    if REQUIRED_SOFT_TOPICS - {row["topic"] for row in artifact.get("soft_constraints", [])}:
        raise ValueError("soft constraints do not cover all required topics")
    fallback = artifact.get("fallback_evaluator")
    if not isinstance(fallback, Mapping) or fallback.get("fail_closed_default") != "abstain":
        raise ValueError("fallback evaluator must fail closed to abstain")
    if set(artifact.get("downstream_usage", {})) != {"exp3099", "exp3101", "exp3102"}:
        raise ValueError("downstream usage must cover exp3099, exp3101, and exp3102")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("maxsat_policy_ready") is True and not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("ready policy honest_verdict must start with a success prefix")
    if artifact.get("maxsat_policy_ready") is not True and not verdict.startswith(
        "blocked_maxsat_policy_precondition_failed"
    ):
        raise ValueError("blocked verdict must disclose MaxSAT policy precondition failure")


def source_artifacts(repo_root: Path) -> list[JsonDict]:
    """Return local source checksums plus the primary external design reference."""

    rows: list[JsonDict] = []
    for source_id, rel_path, role in SOURCE_REL_PATHS:
        path = repo_root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "role": role,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    rows.append(dict(OPENREVIEW_SOURCE))
    return rows


def inference_substrate() -> JsonDict:
    """Declare the policy-design substrate so downstream claims stay bounded."""

    return {
        "kind": "offline_policy_design_and_reference_evaluator",
        "executes_models": False,
        "live_llm_calls": 0,
        "no_live_llm_inference": True,
        "uses_checked_in_artifacts_only": True,
        "maxsat_solver_required": False,
        "fallback_evaluator_available": True,
    }


def honest_verdict(ready: bool) -> str:
    """Map readiness to the terminal verdict vocabulary expected by the conductor."""

    if ready:
        return "complete: maxsat_policy_ready=true; deterministic_fallback=available"
    return "blocked_maxsat_policy_precondition_failed: required source artifacts or exp3097 readiness missing"


def safe_load_json(path: Path) -> JsonDict:
    """Load a JSON object, returning an empty object for absent or malformed files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON so checksums are reproducible across runs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(root: Path, path: Path) -> str:
    """Return a stable repo-relative path when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()

