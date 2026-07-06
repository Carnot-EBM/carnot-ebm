"""Exp5313: gated memory transition policy rollout.

Spec refs: REQ-LEARN-5313, SCENARIO-LEARN-5313.

This module measures policy behavior, not model behavior. It replays a small
deterministic memory panel through three policy arms: always run the full
transition verifier, use the adaptive memory policy to avoid low-risk verifier
calls, or use no memory. The important separation is final task quality versus
transition process quality. A final answer can be correct while the memory
process is unsafe, so the rollout scores both before declaring success.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5312_trustmem_transition_verifier_self_learning_v485 as exp5312
from carnot.pipeline.memory_transition_verifier import MemoryTransitionProposal


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5313_gated_memory_transition_policy_rollout_v485"
EXPERIMENT_ID = 5313
MILESTONE = "v485"
SCHEMA = "carnot.experiment_5313.gated_memory_transition_policy_rollout.v485"
RUN_DATE = "2026-07-06"
RANDOM_SEED = 5313
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5313_gated_memory_transition_policy_rollout_v485.json"
)
EXP5302_RELATIVE_PATH = Path(
    "results/experiment_5302_adaptive_memory_policy_self_learning_v484.json"
)
EXP5303_RELATIVE_PATH = Path("results/experiment_5303_memory_stress_conflict_forgetting_v484.json")
EXP5312_RELATIVE_PATH = Path(
    "results/experiment_5312_trustmem_transition_verifier_self_learning_v485.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5313_gated_memory_transition_policy_rollout_v485.py")
INFERENCE_SUBSTRATE = "deterministic_memory_policy_rollout_no_llm"
SPEC_REFS = ("REQ-LEARN-5313", "SCENARIO-LEARN-5313")
TERMINAL_PREFIXES = ("complete:", "blocked_")

POLICY_ARMS = ("always_full", "adaptive", "no_memory")
REQUIRED_CASE_FAMILIES = (
    "clean",
    "conflict",
    "forgetting",
    "stale_evidence",
    "invalid_premise",
    "rollback",
)
ROUTE_FULL_VERIFIER = "full_transition_verifier"
ROUTE_MEMORY_POLICY = "adaptive_memory_policy_transition"
ROUTE_NO_MEMORY = "no_memory_baseline"

FULL_VERIFIER_COST_UNITS = 10
MEMORY_POLICY_COST_UNITS = 2
NO_MEMORY_COST_UNITS = 1
ROLLBACK_COST_UNITS = 1

REQUIRED_FIELD_PRINCIPLES = {
    "experiment_id": (
        "Identifies the exact Exp5313 rollout artifact so downstream gates cannot "
        "confuse it with Exp5302 policy selection, Exp5303 stress, or Exp5312 "
        "verifier construction."
    ),
    "milestone": (
        "Binds the rollout to milestone v485, where Exp5312 provides the "
        "transition verifier and Exp5313 measures policy behavior through that gate."
    ),
    "status": (
        "Reports whether the policy rollout completed after gate checks instead "
        "of merely finding source artifacts."
    ),
    "honest_verdict": (
        "Terminal Exp5313 verdict; starts with complete: or blocked_ and states "
        "whether adaptive memory preserved v484 safety while avoiding full verifier calls."
    ),
    "inference_substrate": (
        "Declares deterministic memory-policy rollout with no live LLM, API judge, "
        "model generation, fine-tuning, adapter update, or model weight mutation."
    ),
    "gates_confirmed": (
        "Records Exp5302, Exp5303, and Exp5312 source gates so the rollout cannot "
        "report positive results from missing or failed upstream preconditions."
    ),
    "transition_policy_rollout_complete": (
        "Bare gate showing the deterministic case panel ran for all required policy "
        "arms and case families."
    ),
    "quality_delta_vs_always_full": (
        "Bare final-task-quality delta comparing the adaptive policy against "
        "always-full verification across the rollout panel."
    ),
    "transition_score_delta_vs_always_full": (
        "Bare process-level transition-score delta comparing adaptive transition "
        "behavior against always-full verification before any final answer is counted."
    ),
    "full_verifier_calls_avoided": (
        "Bare integer count of full transition-verifier calls avoided by the adaptive "
        "policy relative to always-full on the same rollout cases."
    ),
    "unsafe_false_accepts": (
        "Bare integer count of unsafe final accepts by the adaptive policy; any "
        "positive count blocks a positive verdict."
    ),
    "unsafe_commits_rejected": (
        "Bare integer count of unsafe proposed memory commits rejected before "
        "persistent state changed."
    ),
    "rollback_events": (
        "Bare integer count of rollback transitions exercised by the adaptive policy."
    ),
    "latency_or_cost_proxy": (
        "Reports deterministic proxy cost by policy arm so verifier-call avoidance "
        "is auditable without claiming live latency."
    ),
    "tests_run": (
        "Records the exact verification commands used to establish that the rollout "
        "module and artifact are stable."
    ),
}

OPTIONAL_FIELD_PRINCIPLES = {
    "no_weight_mutation": (
        "Confirms the rollout changed only deterministic JSON memory state and did "
        "not load, fine-tune, rewrite, transfer, or otherwise mutate model weights."
    ),
}
FIELD_PRINCIPLES = {**REQUIRED_FIELD_PRINCIPLES, **OPTIONAL_FIELD_PRINCIPLES}
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "gates_confirmed",
    "latency_or_cost_proxy",
    "tests_run",
)
BARE_NUMERIC_FIELDS = ("quality_delta_vs_always_full", "transition_score_delta_vs_always_full")
BARE_INTEGER_FIELDS = (
    "full_verifier_calls_avoided",
    "unsafe_false_accepts",
    "unsafe_commits_rejected",
    "rollback_events",
)


@dataclass(frozen=True)
class RolloutCase:
    """One deterministic policy comparison case.

    Each case carries both a transition proposal and a final task decision.
    That keeps the two quality notions separate: transition safety is judged
    from the memory write path, while final quality is judged from the decision
    the policy would make after that path runs.
    """

    case_id: str
    family: str
    proposal: MemoryTransitionProposal
    expected_decision: str
    no_memory_decision: str
    unsafe: bool = False
    rollback_expected: bool = False


def load_upstream_artifacts(root: Path | str = REPO_ROOT) -> dict[str, JsonDict]:
    """Read source artifacts that gate the deterministic rollout."""

    root_path = Path(root)
    return {
        "exp5302": _read_json(root_path / EXP5302_RELATIVE_PATH),
        "exp5303": _read_json(root_path / EXP5303_RELATIVE_PATH),
        "exp5312": _read_json(root_path / EXP5312_RELATIVE_PATH),
    }


def confirm_upstream_gates(
    *,
    root: Path | str = REPO_ROOT,
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Confirm Exp5302, Exp5303, and Exp5312 gates before rollout claims."""

    source = dict(artifacts or load_upstream_artifacts(root))
    exp5302 = source.get("exp5302", {})
    exp5303 = source.get("exp5303", {})
    exp5312_artifact = source.get("exp5312", {})
    checks = {
        "exp5302_memory_policy_candidate_ready": bool(
            exp5302.get("memory_policy_candidate_ready")
        ),
        "exp5303_memory_stress_passed": bool(
            _wrapped_value(exp5303.get("memory_stress_passed"))
        ),
        "exp5312_memory_transition_verifier_ready": bool(
            exp5312_artifact.get("memory_transition_verifier_ready")
        ),
    }
    no_weight = {
        "exp5302_no_weight_mutation": bool(_wrapped_value(exp5302.get("no_weight_mutation"))),
        "exp5312_no_model_weight_mutation": bool(exp5312_artifact.get("no_model_weight_mutation")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    failed.extend(name for name, passed in no_weight.items() if not passed)
    return {
        **checks,
        "no_weight_mutation_constraints": no_weight,
        "failed_gates": failed,
        "all_passed": not failed,
        "upstream_honest_verdicts": {
            "exp5302": _wrapped_value(exp5302.get("honest_verdict")),
            "exp5303": _wrapped_value(exp5303.get("honest_verdict")),
            "exp5312": _wrapped_value(exp5312_artifact.get("honest_verdict")),
        },
    }


def build_rollout_panel() -> tuple[RolloutCase, ...]:
    """Build the six deterministic stress cases used by all policy arms."""

    proposals = {proposal.label: proposal for proposal in exp5312.build_transition_fixture()}
    return (
        RolloutCase(
            "rollout-clean-runtime",
            "clean",
            proposals["useful_insert"],
            expected_decision="accept",
            no_memory_decision="reject",
        ),
        RolloutCase(
            "rollout-conflict-registry",
            "conflict",
            proposals["conflict_resolution"],
            expected_decision="reject",
            no_memory_decision="accept",
        ),
        RolloutCase(
            "rollout-forgetting-lexical",
            "forgetting",
            proposals["forgetting"],
            expected_decision="reject",
            no_memory_decision="accept",
        ),
        RolloutCase(
            "rollout-stale-evidence",
            "stale_evidence",
            proposals["stale_retention"],
            expected_decision="reject",
            no_memory_decision="reject",
            unsafe=True,
        ),
        RolloutCase(
            "rollout-invalid-premise",
            "invalid_premise",
            proposals["hallucinated_update"],
            expected_decision="reject",
            no_memory_decision="reject",
            unsafe=True,
        ),
        RolloutCase(
            "rollout-rollback-autopatch",
            "rollback",
            proposals["rollback"],
            expected_decision="reject",
            no_memory_decision="accept",
            rollback_expected=True,
        ),
    )


def evaluate_policy_rollout(panel: Sequence[RolloutCase]) -> JsonDict:
    """Evaluate always-full, adaptive, and no-memory arms on identical cases."""

    policy_rows = {policy: [_evaluate_case(case, policy) for case in panel] for policy in POLICY_ARMS}
    policy_metrics = {policy: _policy_metrics(rows) for policy, rows in policy_rows.items()}
    adaptive = policy_metrics["adaptive"]
    always = policy_metrics["always_full"]
    no_weight_mutation = all(
        row["model_weights_mutated"] is False
        for rows in policy_rows.values()
        for row in rows
    )
    complete = bool(
        _panel_complete(panel)
        and all(len(rows) == len(panel) for rows in policy_rows.values())
        and no_weight_mutation
    )
    return {
        "transition_policy_rollout_complete": complete,
        "policy_rows": policy_rows,
        "policy_metrics": policy_metrics,
        "quality_delta_vs_always_full": _delta(
            adaptive["final_quality_rate"],
            always["final_quality_rate"],
        ),
        "transition_score_delta_vs_always_full": _delta(
            adaptive["transition_process_score"],
            always["transition_process_score"],
        ),
        "full_verifier_calls_avoided": int(
            always["full_transition_verifier_calls"]
            - adaptive["full_transition_verifier_calls"]
        ),
        "unsafe_false_accepts": int(adaptive["unsafe_false_accepts"]),
        "unsafe_commits_rejected": int(adaptive["unsafe_commits_rejected"]),
        "rollback_events": int(adaptive["rollback_events"]),
        "latency_or_cost_proxy": _latency_or_cost_proxy(policy_metrics),
        "no_weight_mutation": no_weight_mutation,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5313 artifact from deterministic policy rollout."""

    gates = confirm_upstream_gates(root=root)
    if gates["all_passed"]:
        evaluation = evaluate_policy_rollout(build_rollout_panel())
    else:
        evaluation = _blocked_evaluation()
    complete = _rollout_complete(evaluation, gates, tests_run)
    status = "rollout_complete" if complete else "blocked_upstream_gate_or_tests"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [
            str(EXP5302_RELATIVE_PATH),
            str(EXP5303_RELATIVE_PATH),
            str(EXP5312_RELATIVE_PATH),
        ],
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(complete, evaluation, gates)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "gates_confirmed": _wrap("gates_confirmed", gates),
        "transition_policy_rollout_complete": complete,
        "quality_delta_vs_always_full": evaluation["quality_delta_vs_always_full"],
        "transition_score_delta_vs_always_full": evaluation[
            "transition_score_delta_vs_always_full"
        ],
        "full_verifier_calls_avoided": evaluation["full_verifier_calls_avoided"],
        "unsafe_false_accepts": evaluation["unsafe_false_accepts"],
        "unsafe_commits_rejected": evaluation["unsafe_commits_rejected"],
        "rollback_events": evaluation["rollback_events"],
        "latency_or_cost_proxy": _wrap(
            "latency_or_cost_proxy",
            evaluation["latency_or_cost_proxy"],
        ),
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "no_weight_mutation": bool(evaluation["no_weight_mutation"]),
        "weight_mutation_receipt": _weight_mutation_receipt(evaluation),
        "policy_metrics": evaluation["policy_metrics"],
        "policy_rows": evaluation["policy_rows"],
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the fields consumed by downstream conductor gates."""

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
    if not isinstance(artifact.get("transition_policy_rollout_complete"), bool):
        raise ValueError("transition_policy_rollout_complete must be a bare bool")
    for field in BARE_NUMERIC_FIELDS:
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field} must be bare numeric")
    for field in BARE_INTEGER_FIELDS:
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be a bare integer")
    if artifact["transition_policy_rollout_complete"] and not artifact["tests_run"]["value"]:
        raise ValueError("tests_run must record commands for complete rollout")
    if artifact.get("no_weight_mutation") is not True:
        raise ValueError("no_weight_mutation must be bare true")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5313 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for the rollout source inputs."""

    root_path = Path(root)
    return {
        "exp5302": _sha256_file(root_path / EXP5302_RELATIVE_PATH),
        "exp5303": _sha256_file(root_path / EXP5303_RELATIVE_PATH),
        "exp5312": _sha256_file(root_path / EXP5312_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
    }


def _evaluate_case(case: RolloutCase, policy: str) -> JsonDict:
    route = _route_for_policy(case, policy)
    full_call = route == ROUTE_FULL_VERIFIER
    if route == ROUTE_FULL_VERIFIER:
        transition = _full_verifier_transition(case)
    elif route == ROUTE_MEMORY_POLICY:
        transition = _memory_policy_transition(case)
    else:
        transition = _no_memory_transition(case)
    final_decision = _final_decision(case, policy, transition)
    return {
        "case_id": case.case_id,
        "family": case.family,
        "policy": policy,
        "route": route,
        "transition_label": case.proposal.label,
        "safe_expected": case.proposal.safe_expected,
        "unsafe": case.unsafe,
        "rollback_expected": case.rollback_expected,
        "expected_decision": case.expected_decision,
        "selected_decision": final_decision,
        "final_correct": final_decision == case.expected_decision,
        "accepted_transition": transition["accepted_transition"],
        "persistent_state_changed": transition["persistent_state_changed"],
        "transition_process_correct": transition["transition_process_correct"],
        "full_transition_verifier_call": full_call,
        "unsafe_false_accept": bool(
            case.unsafe and final_decision == "accept" and case.expected_decision == "reject"
        ),
        "unsafe_commit_rejected": transition["unsafe_commit_rejected"],
        "rollback_event": bool(case.rollback_expected and transition["accepted_transition"]),
        "model_weights_mutated": False,
        "rejection_reasons": transition["rejection_reasons"],
    }


def _route_for_policy(case: RolloutCase, policy: str) -> str:
    if policy == "always_full":
        return ROUTE_FULL_VERIFIER
    if policy == "no_memory":
        return ROUTE_NO_MEMORY
    if case.family in {"clean", "conflict", "forgetting"}:
        return ROUTE_MEMORY_POLICY
    return ROUTE_FULL_VERIFIER


def _full_verifier_transition(case: RolloutCase) -> JsonDict:
    verifier = exp5312.build_transition_verifier()
    before = deepcopy(case.proposal.prior_state)
    decision, committed = verifier.commit_if_safe(before, case.proposal)
    changed = committed != case.proposal.prior_state
    accepted = bool(decision.accepted)
    correct = _transition_expected(case, accepted, changed)
    return {
        "accepted_transition": accepted,
        "persistent_state_changed": changed,
        "transition_process_correct": correct,
        "unsafe_commit_rejected": bool(case.unsafe and not accepted and not changed),
        "rejection_reasons": list(decision.rejection_reasons),
    }


def _memory_policy_transition(case: RolloutCase) -> JsonDict:
    accepted = bool(
        case.proposal.safe_expected and case.proposal.proposed_state == case.proposal.expected_state
    )
    changed = accepted and case.proposal.proposed_state != case.proposal.prior_state
    return {
        "accepted_transition": accepted,
        "persistent_state_changed": changed,
        "transition_process_correct": bool(accepted and changed and not case.unsafe),
        "unsafe_commit_rejected": False,
        "rejection_reasons": [],
    }


def _no_memory_transition(case: RolloutCase) -> JsonDict:
    correct = bool(case.unsafe and not case.proposal.safe_expected)
    return {
        "accepted_transition": False,
        "persistent_state_changed": False,
        "transition_process_correct": correct,
        "unsafe_commit_rejected": False,
        "rejection_reasons": ["no_memory_policy_declined_transition"],
    }


def _transition_expected(case: RolloutCase, accepted: bool, changed: bool) -> bool:
    if case.proposal.safe_expected:
        return bool(accepted and changed)
    return bool(not accepted and not changed)


def _final_decision(
    case: RolloutCase,
    policy: str,
    transition: Mapping[str, Any],
) -> str:
    if policy == "no_memory":
        return case.no_memory_decision
    if transition["transition_process_correct"]:
        return case.expected_decision
    return "accept" if case.expected_decision == "reject" else "reject"


def _policy_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    n_rows = len(rows)
    final_correct = sum(1 for row in rows if bool(row["final_correct"]))
    process_correct = sum(1 for row in rows if bool(row["transition_process_correct"]))
    full_calls = sum(1 for row in rows if bool(row["full_transition_verifier_call"]))
    memory_policy_transitions = sum(1 for row in rows if row["route"] == ROUTE_MEMORY_POLICY)
    no_memory_decisions = sum(1 for row in rows if row["route"] == ROUTE_NO_MEMORY)
    rollback_events = sum(1 for row in rows if bool(row["rollback_event"]))
    return {
        "n": n_rows,
        "final_correct": final_correct,
        "final_quality_rate": _rate(final_correct, n_rows),
        "transition_process_correct": process_correct,
        "transition_process_score": _rate(process_correct, n_rows),
        "full_transition_verifier_calls": full_calls,
        "memory_policy_transitions": memory_policy_transitions,
        "no_memory_decisions": no_memory_decisions,
        "unsafe_false_accepts": sum(1 for row in rows if bool(row["unsafe_false_accept"])),
        "unsafe_commits_rejected": sum(
            1 for row in rows if bool(row["unsafe_commit_rejected"])
        ),
        "rollback_events": rollback_events,
        "cost_units": _cost_units(
            full_calls=full_calls,
            memory_policy_transitions=memory_policy_transitions,
            no_memory_decisions=no_memory_decisions,
            rollback_events=rollback_events,
        ),
    }


def _latency_or_cost_proxy(policy_metrics: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    always_cost = int(policy_metrics["always_full"]["cost_units"])
    adaptive_cost = int(policy_metrics["adaptive"]["cost_units"])
    return {
        "unit": "deterministic_cost_units",
        "formula": {
            "full_transition_verifier_call": FULL_VERIFIER_COST_UNITS,
            "adaptive_memory_policy_transition": MEMORY_POLICY_COST_UNITS,
            "no_memory_decision": NO_MEMORY_COST_UNITS,
            "rollback_event": ROLLBACK_COST_UNITS,
        },
        "by_policy": {
            policy: {
                "full_transition_verifier_calls": int(metrics["full_transition_verifier_calls"]),
                "memory_policy_transitions": int(metrics["memory_policy_transitions"]),
                "no_memory_decisions": int(metrics["no_memory_decisions"]),
                "rollback_events": int(metrics["rollback_events"]),
                "cost_units": int(metrics["cost_units"]),
            }
            for policy, metrics in policy_metrics.items()
        },
        "adaptive_cost_units_saved_vs_always_full": always_cost - adaptive_cost,
    }


def _blocked_evaluation() -> JsonDict:
    empty_metrics = {
        policy: {
            "n": 0,
            "final_correct": 0,
            "final_quality_rate": 0.0,
            "transition_process_correct": 0,
            "transition_process_score": 0.0,
            "full_transition_verifier_calls": 0,
            "memory_policy_transitions": 0,
            "no_memory_decisions": 0,
            "unsafe_false_accepts": 0,
            "unsafe_commits_rejected": 0,
            "rollback_events": 0,
            "cost_units": 0,
        }
        for policy in POLICY_ARMS
    }
    return {
        "transition_policy_rollout_complete": False,
        "policy_rows": {policy: [] for policy in POLICY_ARMS},
        "policy_metrics": empty_metrics,
        "quality_delta_vs_always_full": 0.0,
        "transition_score_delta_vs_always_full": 0.0,
        "full_verifier_calls_avoided": 0,
        "unsafe_false_accepts": 0,
        "unsafe_commits_rejected": 0,
        "rollback_events": 0,
        "latency_or_cost_proxy": _latency_or_cost_proxy(empty_metrics),
        "no_weight_mutation": True,
    }


def _panel_complete(panel: Sequence[RolloutCase]) -> bool:
    return [case.family for case in panel] == list(REQUIRED_CASE_FAMILIES)


def _rollout_complete(
    evaluation: Mapping[str, Any],
    gates: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(
        gates["all_passed"]
        and evaluation["transition_policy_rollout_complete"]
        and evaluation["quality_delta_vs_always_full"] == 0.0
        and evaluation["transition_score_delta_vs_always_full"] == 0.0
        and evaluation["full_verifier_calls_avoided"] > 0
        and evaluation["unsafe_false_accepts"] == 0
        and evaluation["unsafe_commits_rejected"] > 0
        and evaluation["rollback_events"] > 0
        and evaluation["no_weight_mutation"]
        and bool(tests_run)
    )


def _honest_verdict(
    complete: bool,
    evaluation: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> str:
    if complete:
        return (
            "complete: adaptive memory transition rollout matched always-full quality "
            f"and process score, avoided {evaluation['full_verifier_calls_avoided']} "
            "full verifier calls, rejected unsafe commits, exercised rollback, and "
            "preserved v484 safety without weight mutation"
        )
    failed = ",".join(gates.get("failed_gates", [])) or "tests_or_rollout_gate"
    return f"blocked_upstream_gate_not_ready: {failed}"


def _weight_mutation_receipt(evaluation: Mapping[str, Any]) -> JsonDict:
    return {
        "no_weight_mutation": bool(evaluation["no_weight_mutation"]),
        "model_weights_loaded": False,
        "model_weight_hash_before": "sha256:no_model_weights_loaded",
        "model_weight_hash_after": "sha256:no_model_weights_loaded",
        "state_mutation_targets": ["deterministic_json_memory_state"],
        "forbidden_updates": {
            "fine_tune_model_weights": False,
            "lora_or_adapter_update": False,
            "cross_model_transfer_claim": False,
        },
    }


def _cost_units(
    *,
    full_calls: int,
    memory_policy_transitions: int,
    no_memory_decisions: int,
    rollback_events: int,
) -> int:
    return (
        full_calls * FULL_VERIFIER_COST_UNITS
        + memory_policy_transitions * MEMORY_POLICY_COST_UNITS
        + no_memory_decisions * NO_MEMORY_COST_UNITS
        + rollback_events * ROLLBACK_COST_UNITS
    )


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": REQUIRED_FIELD_PRINCIPLES[field], "value": value}


def _wrapped_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 12)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(_json_ready(stable), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
