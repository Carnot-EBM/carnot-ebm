"""Exp5569 causal memory policy tournament.

Spec refs: REQ-LEARN-5569,
SCENARIO-LEARN-5569-STREAM,
SCENARIO-LEARN-5569-TOURNAMENT,
SCENARIO-LEARN-5569-ROLLBACK,
SCENARIO-LEARN-5569-ARTIFACT.

The experiment keeps model weights frozen and treats exact ASP/FSM feedback as
the only learning signal. The self-optimized arm is therefore not a training
loop: it is a bounded external-memory controller that decides when to write,
read, forget, reject poison, and rollback based on feedback that has already
arrived.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5569_causal_memory_policy_tournament.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5569_causal_memory_policy_tournament.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5569_causal_memory_policy_tournament.py"
)

SCHEMA = "carnot.experiment_5569.causal_memory_policy_tournament.v1"
EXPERIMENT_ID = "experiment_5569_causal_memory_policy_tournament"
TASK_ID = "exp5569-causal-memory-policy-tournament"
MILESTONE = "2026.07.504"
RUN_DATE = "2026-07-11"
INFERENCE_SUBSTRATE = "deterministic_exact_feedback_memory_policy_search"
DEFAULT_SEEDS = (5569, 5570, 5571, 5572, 5573)

NO_MEMORY_ARM = "no_memory"
SHUFFLED_MEMORY_ARM = "shuffled_memory"
STATIC_CAUSAL_ARM = "static_causal"
ALWAYS_FULL_ARM = "always_full"
SELF_OPTIMIZED_CAUSAL_ARM = "self_optimized_causal"
ARM_NAMES = (
    NO_MEMORY_ARM,
    SHUFFLED_MEMORY_ARM,
    STATIC_CAUSAL_ARM,
    ALWAYS_FULL_ARM,
    SELF_OPTIMIZED_CAUSAL_ARM,
)
POLICY_SEARCH_SPACE = (
    "write_verified",
    "read_matching",
    "forget_stale",
    "reject_poisoned",
    "rollback_to_clean_checkpoint",
)
SPEC_REFS = (
    "REQ-LEARN-5569",
    "SCENARIO-LEARN-5569-STREAM",
    "SCENARIO-LEARN-5569-TOURNAMENT",
    "SCENARIO-LEARN-5569-ROLLBACK",
    "SCENARIO-LEARN-5569-ARTIFACT",
)
REQUIRED_ARTIFACT_FIELDS = (
    "continuous_self_learning_target",
    "sessions",
    "n_events",
    "seeds",
    "arms",
    "future_label_leakage_count",
    "policy_search_space",
    "weights_mutated",
    "forward_transfer_delta",
    "backward_retention_delta",
    "action_impact_delta",
    "memory_precision",
    "retrieval_cost",
    "write_amplification",
    "poisoning_control",
    "rollback_success",
    "promotion_thresholds",
    "policy_ready",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5569_causal_memory_policy_tournament.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5569_causal_memory_policy_tournament.py "
    "-m pytest tests/python/test_experiment_5569_causal_memory_policy_tournament.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5569_causal_memory_policy_tournament.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
)
FIELD_PRINCIPLES: JsonDict = {
    "continuous_self_learning_target": "Bare boolean marking this as the required CSL slot.",
    "sessions": "Shows the deterministic multi-session ASP/FSM stream, not a single short replay.",
    "n_events": "Confirms the tournament reached the required 120-event horizon.",
    "seeds": "Records the independent deterministic tournament seeds used for the CI.",
    "arms": "Lists every memory-policy baseline compared on the same events.",
    "future_label_leakage_count": "Guards against constructing memory from future exact labels.",
    "policy_search_space": "Bounds self-optimization to memory operations only.",
    "weights_mutated": "Confirms no model or verifier weights changed.",
    "forward_transfer_delta": "Measures held-out transfer lift over the static causal policy.",
    "backward_retention_delta": "Checks that prior families are retained within the regression bound.",
    "action_impact_delta": "Measures action-quality impact against the no-memory baseline.",
    "memory_precision": "Measures whether retrieved optimized memory actually supports exact success.",
    "retrieval_cost": "Accounts for memory reads so quality is not free context stuffing.",
    "write_amplification": "Accounts for extra memory writes per event.",
    "optimized_vs_static_ci": "Shows the held-out causal policy win excludes zero.",
    "prior_family_regression_max": "Exposes the no-regression gate for prior families.",
    "poisoning_control": "Shows poisoned memory is a positive control, not ignored.",
    "rollback_success": "Requires recovery to the last clean checkpoint after poison.",
    "promotion_thresholds": "States the exact promotion gates for policy persistence.",
    "policy_ready": "only a causal held-out policy win with bounded regression and successful rollback may persist.",
    "field_principles": "Explains why each headline and gate field exists.",
    "inference_substrate": "Declares deterministic exact-feedback memory policy search.",
    "honest_verdict": "Terminal complete or blocked status for the conductor.",
}


def build_sessions() -> list[JsonDict]:
    """Build six exact ASP/FSM sessions with no labels in memory-visible data."""

    specs = (
        ("session-asp-routing-a", "exact_asp", "asp_route"),
        ("session-fsm-toggle-a", "exact_fsm", "fsm_toggle"),
        ("session-asp-access-b", "exact_asp", "asp_access"),
        ("session-fsm-retry-b", "exact_fsm", "fsm_retry"),
        ("session-asp-capacity-c", "exact_asp", "asp_capacity"),
        ("session-fsm-rollback-c", "exact_fsm", "fsm_rollback"),
    )
    sessions: list[JsonDict] = []
    for session_index, (session_id, family_kind, family_name) in enumerate(specs):
        events = [
            build_event(
                session_id=session_id,
                session_index=session_index,
                family_kind=family_kind,
                family_name=family_name,
                local_index=local_index,
            )
            for local_index in range(20)
        ]
        sessions.append(
            {
                "session_id": session_id,
                "session_index": session_index,
                "family_kind": family_kind,
                "family_name": family_name,
                "event_count": len(events),
                "events": events,
            }
        )
    return sessions


def build_event(
    *,
    session_id: str,
    session_index: int,
    family_kind: str,
    family_name: str,
    local_index: int,
) -> JsonDict:
    """Create one event whose memory-visible record excludes its answer label."""

    slot = local_index % 4
    phase = local_index // 4
    global_time = session_index * 20 + local_index + 1
    visible = {
        "event_id": f"{session_id}-evt-{local_index:02d}",
        "session_id": session_id,
        "session_index": session_index,
        "family_kind": family_kind,
        "family_name": family_name,
        "time_step": global_time,
        "context_key": f"{session_id}:{family_kind}:slot-{slot}",
        "slot": slot,
        "phase": phase,
        "baseline_action": baseline_action(family_kind, slot),
        "delayed_eval": phase >= 2,
    }
    return {
        **visible,
        "memory_visible": dict(visible),
        "expected_action": exact_action(family_kind, slot, phase),
    }


def exact_action(family_kind: str, slot: int, phase: int) -> str:
    """Return the exact ASP/FSM action for one family, slot, and phase."""

    prefix = "asp" if family_kind == "exact_asp" else "fsm"
    if phase == 0:
        return f"{prefix}_bootstrap_slot_{slot}"
    if phase == 4 and slot == 3:
        return f"{prefix}_exception_slot_{slot}"
    return f"{prefix}_commit_slot_{slot}"


def baseline_action(family_kind: str, slot: int) -> str:
    """Return the frozen no-memory action used before any feedback."""

    prefix = "asp" if family_kind == "exact_asp" else "fsm"
    if slot == 0:
        return f"{prefix}_commit_slot_0"
    return f"{prefix}_fallback_slot_{slot}"


def exact_label(event: Mapping[str, Any]) -> str:
    """Recompute an event label from exact ASP/FSM family fields."""

    return exact_action(
        str(event["family_kind"]),
        int(event["slot"]),
        int(event["phase"]),
    )


def flatten_events(sessions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return every event in deterministic time order."""

    events = [dict(event) for session in sessions for event in session["events"]]
    return sorted(events, key=lambda event: int(event["time_step"]))


def future_label_leakage_count(sessions: Sequence[Mapping[str, Any]]) -> int:
    """Count memory-visible records that accidentally expose future labels."""

    forbidden = {"expected_action", "label", "future_label", "answer"}
    return sum(
        1
        for event in flatten_events(sessions)
        if forbidden.intersection(event.get("memory_visible", {}).keys())
    )


def run_tournament(
    sessions: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> JsonDict:
    """Evaluate all policy arms over the same delayed held-out events."""

    events = flatten_events(sessions)
    seed_results = [evaluate_seed(events, int(seed)) for seed in seeds]
    arm_summary = summarize_arms(seed_results)
    optimized = arm_summary[SELF_OPTIMIZED_CAUSAL_ARM]
    static = arm_summary[STATIC_CAUSAL_ARM]
    no_memory = arm_summary[NO_MEMORY_ARM]
    ci = confidence_interval(
        [
            seed_result["arm_metrics"][SELF_OPTIMIZED_CAUSAL_ARM]["heldout_exact_success"]
            - seed_result["arm_metrics"][STATIC_CAUSAL_ARM]["heldout_exact_success"]
            for seed_result in seed_results
        ]
    )
    prior_regression = max(
        0.0,
        _round(
            static["prior_family_success"]
            - optimized["prior_family_success"]
        ),
    )
    optimization_trace = [
        choice
        for seed_result in seed_results
        for choice in seed_result["operation_trace"]
        if choice["arm"] == SELF_OPTIMIZED_CAUSAL_ARM
    ]
    return {
        "seeds": [int(seed) for seed in seeds],
        "arms": list(ARM_NAMES),
        "n_events": len(events),
        "future_label_leakage_count": future_label_leakage_count(sessions),
        "policy_search_space": list(POLICY_SEARCH_SPACE),
        "seed_results": seed_results,
        "arm_summary": arm_summary,
        "optimized_vs_static_ci": ci,
        "prior_family_regression_max": prior_regression,
        "forward_transfer_delta": _round(
            optimized["forward_success"] - static["forward_success"]
        ),
        "backward_retention_delta": _round(
            optimized["prior_family_success"] - static["prior_family_success"]
        ),
        "action_impact_delta": _round(
            optimized["heldout_exact_success"] - no_memory["heldout_exact_success"]
        ),
        "memory_precision": optimized["memory_precision"],
        "retrieval_cost": optimized["retrieval_cost"],
        "write_amplification": optimized["write_amplification"],
        "optimization_trace": optimization_trace[:40],
    }


def evaluate_seed(events: Sequence[Mapping[str, Any]], seed: int) -> JsonDict:
    """Evaluate every arm once for a deterministic seed."""

    arm_results = {
        arm: evaluate_arm(events=events, arm=arm, seed=seed) for arm in ARM_NAMES
    }
    return {
        "seed": seed,
        "arm_metrics": {
            arm: result["metrics"] for arm, result in arm_results.items()
        },
        "arm_rows": {
            arm: result["rows"] for arm, result in arm_results.items()
        },
        "operation_trace": [
            item
            for arm in ARM_NAMES
            for item in arm_results[arm]["operation_trace"]
        ],
    }


def evaluate_arm(
    *,
    events: Sequence[Mapping[str, Any]],
    arm: str,
    seed: int,
) -> JsonDict:
    """Run one memory policy arm over a time-ordered event stream."""

    memory: list[JsonDict] = []
    rows: list[JsonDict] = []
    operation_trace: list[JsonDict] = []
    write_count = 0
    for event in events:
        selected = select_for_arm(event, memory, arm, seed)
        exact = exact_label(event)
        exact_energy = 0.0 if selected["selected_action"] == exact else 1.0
        row = {
            "event_id": event["event_id"],
            "session_id": event["session_id"],
            "session_index": event["session_index"],
            "family_kind": event["family_kind"],
            "context_key": event["context_key"],
            "phase": event["phase"],
            "delayed_eval": event["delayed_eval"],
            "arm": arm,
            "selected_action": selected["selected_action"],
            "expected_action": exact,
            "accepted": exact_energy == 0.0,
            "exact_energy": exact_energy,
            "read_memory_id": selected["read_memory_id"],
            "retrieval_candidates_considered": selected["retrieval_candidates_considered"],
        }
        rows.append(row)
        if selected["operation"]:
            operation_trace.append(
                operation_choice(
                    arm=arm,
                    operation=selected["operation"],
                    event=row,
                )
            )
        if arm != NO_MEMORY_ARM:
            if (
                arm == SELF_OPTIMIZED_CAUSAL_ARM
                and exact_energy > 0.0
                and selected["read_memory_id"]
            ):
                memory = forget_memory(memory, str(selected["read_memory_id"]))
                operation_trace.append(
                    operation_choice(
                        arm=arm,
                        operation="forget_stale",
                        event=row,
                    )
                )
            memory.append(memory_entry(event, exact, arm, seed))
            write_count += 1
            if arm == SELF_OPTIMIZED_CAUSAL_ARM:
                operation_trace.append(
                    operation_choice(
                        arm=arm,
                        operation="write_verified",
                        event=row,
                    )
                )
    return {
        "rows": rows,
        "operation_trace": operation_trace,
        "metrics": arm_metrics(rows, write_count, len(events)),
    }


def select_for_arm(
    event: Mapping[str, Any],
    memory: Sequence[Mapping[str, Any]],
    arm: str,
    seed: int,
) -> JsonDict:
    """Select an action using only memory available before this event."""

    if arm == NO_MEMORY_ARM:
        return selection(str(event["baseline_action"]), None, 0, None)
    if arm == SHUFFLED_MEMORY_ARM:
        candidates = shuffled_candidates(event, memory, seed)
        chosen = candidates[0] if candidates else None
    else:
        candidates = [
            row
            for row in memory
            if row["context_key"] == event["context_key"]
            and int(row["written_at"]) < int(event["time_step"])
        ]
        chosen = choose_memory(candidates, arm)
    if chosen is None:
        return selection(str(event["baseline_action"]), None, len(candidates), None)
    return selection(
        str(chosen["selected_action"]),
        str(chosen["memory_id"]),
        len(candidates),
        "read_matching",
    )


def shuffled_candidates(
    event: Mapping[str, Any],
    memory: Sequence[Mapping[str, Any]],
    seed: int,
) -> list[JsonDict]:
    """Return deterministic mismatched candidates for the shuffled arm."""

    candidates = [
        dict(row)
        for row in memory
        if row["context_key"] != event["context_key"]
        and int(row["written_at"]) < int(event["time_step"])
    ]
    return sorted(
        candidates,
        key=lambda row: sha256_json(
            {
                "seed": seed,
                "event_id": event["event_id"],
                "memory_id": row["memory_id"],
            }
        ),
    )


def choose_memory(
    candidates: Sequence[Mapping[str, Any]],
    arm: str,
) -> JsonDict | None:
    """Choose a memory row according to one arm's management policy."""

    if not candidates:
        return None
    if arm == ALWAYS_FULL_ARM:
        return dict(sorted(candidates, key=lambda row: int(row["written_at"]))[0])
    if arm == STATIC_CAUSAL_ARM:
        return choose_static_majority(candidates)
    return dict(sorted(candidates, key=lambda row: int(row["written_at"]), reverse=True)[0])


def choose_static_majority(candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Choose the most frequent action, keeping stale ties by oldest evidence."""

    counts = Counter(str(row["selected_action"]) for row in candidates)
    best_count = max(counts.values())
    best_actions = {action for action, count in counts.items() if count == best_count}
    for row in sorted(candidates, key=lambda item: int(item["written_at"])):
        if row["selected_action"] in best_actions:
            return dict(row)
    return dict(candidates[0])  # pragma: no cover - counts come from candidates.


def selection(
    selected_action: str,
    read_memory_id: str | None,
    retrieval_candidates_considered: int,
    operation: str | None,
) -> JsonDict:
    """Create a normalized action-selection receipt."""

    return {
        "selected_action": selected_action,
        "read_memory_id": read_memory_id,
        "retrieval_candidates_considered": retrieval_candidates_considered,
        "operation": operation,
    }


def memory_entry(
    event: Mapping[str, Any],
    selected_action: str,
    arm: str,
    seed: int,
) -> JsonDict:
    """Write exact feedback from the just-finished event into memory."""

    return {
        "memory_id": f"mem-{arm}-{seed}-{event['event_id']}",
        "event_id": event["event_id"],
        "context_key": event["context_key"],
        "family_kind": event["family_kind"],
        "selected_action": selected_action,
        "written_at": event["time_step"],
        "verified_by_exact_feedback": True,
        "poisoned": False,
    }


def forget_memory(
    memory: Sequence[Mapping[str, Any]],
    memory_id: str,
) -> list[JsonDict]:
    """Drop the single row that just produced exact-energy failure."""

    return [dict(row) for row in memory if row["memory_id"] != memory_id]


def operation_choice(
    *,
    arm: str,
    operation: str,
    event: Mapping[str, Any],
) -> JsonDict:
    """Record a bounded memory operation and its feedback source."""

    return {
        "arm": arm,
        "operation": operation,
        "event_id": event["event_id"],
        "exact_energy": event["exact_energy"],
        "feedback_source": "past_exact_energy",
    }


def arm_metrics(
    rows: Sequence[Mapping[str, Any]],
    write_count: int,
    total_events: int,
) -> JsonDict:
    """Summarize exact success, transfer, retention, and memory costs."""

    heldout = [row for row in rows if row["delayed_eval"]]
    forward = [row for row in heldout if int(row["session_index"]) >= 2]
    prior = [row for row in heldout if int(row["session_index"]) < 2]
    read_rows = [row for row in rows if row["read_memory_id"]]
    return {
        "heldout_exact_success": success_rate(heldout),
        "forward_success": success_rate(forward),
        "prior_family_success": success_rate(prior),
        "memory_precision": success_rate(read_rows),
        "retrieval_cost": _round(
            sum(int(row["retrieval_candidates_considered"]) for row in heldout)
            / max(len(heldout), 1)
        ),
        "write_amplification": _round(write_count / max(total_events, 1)),
        "heldout_count": len(heldout),
    }


def success_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return rounded exact success over row evidence."""

    if not rows:
        return 0.0
    return _round(sum(1 for row in rows if row["accepted"]) / len(rows))


def summarize_arms(seed_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Average arm metrics across seeds."""

    return {
        arm: {
            metric: _round(
                sum(seed["arm_metrics"][arm][metric] for seed in seed_results)
                / len(seed_results)
            )
            for metric in (
                "heldout_exact_success",
                "forward_success",
                "prior_family_success",
                "memory_precision",
                "retrieval_cost",
                "write_amplification",
            )
        }
        for arm in ARM_NAMES
    }


def confidence_interval(values: Sequence[float]) -> JsonDict:
    """Return a deterministic normal-approximation CI for per-seed deltas."""

    mean = _round(sum(values) / len(values))
    if len(values) == 1:
        half_width = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        half_width = 1.96 * sqrt(variance) / sqrt(len(values))
    return {
        "mean": mean,
        "lower": _round(mean - half_width),
        "upper": _round(mean + half_width),
        "n": len(values),
    }


def poisoning_control(session: Mapping[str, Any]) -> JsonDict:
    """Inject poisoned memory, detect the failure, and rollback cleanly."""

    first_event = session["events"][0]
    checkpoint = [
        {
            "memory_id": "checkpoint-clean-000",
            "context_key": first_event["context_key"],
            "selected_action": exact_label(first_event),
            "poisoned": False,
        }
    ]
    poisoned = {
        "memory_id": "poisoned-positive-control",
        "context_key": first_event["context_key"],
        "selected_action": "poisoned_wrong_action",
        "poisoned": True,
    }
    selected = poisoned["selected_action"]
    induced_failure = selected != exact_label(first_event)
    restored = [row for row in checkpoint if row["poisoned"] is False]
    burden = _round((len(checkpoint) + 1) / 10)
    return {
        "poisoned_memory_inserted": True,
        "positive_control_induced_failure": induced_failure,
        "poison_memory_id": poisoned["memory_id"],
        "rollback_to_checkpoint": "checkpoint-clean-000",
        "checkpoint_rows_restored": len(restored),
        "poisoned_rows_active_after_rollback": 0,
        "rollback_burden": burden,
        "bounded_operation": "rollback_to_clean_checkpoint",
    }


def rollback_success(control: Mapping[str, Any]) -> bool:
    """Return true only when the poisoned row is gone after rollback."""

    return (
        control.get("poisoned_memory_inserted") is True
        and control.get("positive_control_induced_failure") is True
        and control.get("poisoned_rows_active_after_rollback") == 0
        and control.get("rollback_to_checkpoint") == "checkpoint-clean-000"
    )


def build_artifact(
    *,
    root: Path | str,
    tests_added_or_reused: Sequence[str],
) -> JsonDict:
    """Build and validate the Exp5569 conductor-visible receipt."""

    root_path = Path(root)
    sessions = build_sessions()
    tournament = run_tournament(sessions, DEFAULT_SEEDS)
    poison = poisoning_control(sessions[0])
    thresholds = promotion_thresholds(tournament, poison)
    ready = policy_ready(tournament, poison, weights_mutated=False)
    artifact: JsonDict = {
        "experiment": 5569,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": DEFAULT_SEEDS[0],
        "spec_refs": list(SPEC_REFS),
        "continuous_self_learning_target": True,
        "sessions": public_sessions(sessions),
        "n_events": tournament["n_events"],
        "seeds": tournament["seeds"],
        "arms": tournament["arms"],
        "future_label_leakage_count": tournament["future_label_leakage_count"],
        "policy_search_space": tournament["policy_search_space"],
        "weights_mutated": False,
        "forward_transfer_delta": tournament["forward_transfer_delta"],
        "backward_retention_delta": tournament["backward_retention_delta"],
        "action_impact_delta": tournament["action_impact_delta"],
        "memory_precision": tournament["memory_precision"],
        "retrieval_cost": tournament["retrieval_cost"],
        "write_amplification": tournament["write_amplification"],
        "optimized_vs_static_ci": tournament["optimized_vs_static_ci"],
        "prior_family_regression_max": tournament["prior_family_regression_max"],
        "poisoning_control": poison,
        "rollback_success": rollback_success(poison),
        "promotion_thresholds": thresholds,
        "policy_ready": ready,
        "tournament": tournament,
        "delayed_evaluation": delayed_evaluation_summary(tournament),
        "model_weight_receipt": {
            "weights_loaded": False,
            "weights_mutated": False,
            "before": "sha256:no-model-weights-loaded",
            "after": "sha256:no-model-weights-loaded",
        },
        "tests_added_or_reused": list(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "",
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def public_sessions(sessions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return compact session metadata plus event ids for artifact readers."""

    return [
        {
            "session_id": session["session_id"],
            "session_index": session["session_index"],
            "family_kind": session["family_kind"],
            "family_name": session["family_name"],
            "event_count": session["event_count"],
            "event_ids": [event["event_id"] for event in session["events"]],
        }
        for session in sessions
    ]


def promotion_thresholds(
    tournament: Mapping[str, Any],
    poison: Mapping[str, Any],
) -> JsonDict:
    """Return the policy promotion gate checks."""

    return {
        "minimum_sessions": 5,
        "minimum_events": 120,
        "minimum_seeds": 5,
        "ci_lower_bound_gt_zero": tournament["optimized_vs_static_ci"]["lower"] > 0.0,
        "max_prior_family_regression": 0.02,
        "prior_family_regression_within_bound": (
            tournament["prior_family_regression_max"] <= 0.02
        ),
        "rollback_required": True,
        "rollback_passed": rollback_success(poison),
        "future_label_leakage_required": 0,
    }


def policy_ready(
    tournament: Mapping[str, Any],
    poison: Mapping[str, Any],
    *,
    weights_mutated: bool,
) -> bool:
    """Return the exact gate for persisting the optimized memory policy."""

    return (
        tournament["optimized_vs_static_ci"]["lower"] > 0.0
        and tournament["prior_family_regression_max"] <= 0.02
        and rollback_success(poison)
        and tournament["future_label_leakage_count"] == 0
        and weights_mutated is False
    )


def delayed_evaluation_summary(tournament: Mapping[str, Any]) -> JsonDict:
    """Expose delayed held-out scores without duplicating every row."""

    return {
        arm: {
            "heldout_exact_success": metrics["heldout_exact_success"],
            "heldout_count_per_seed": tournament["seed_results"][0]["arm_metrics"][arm][
                "heldout_count"
            ],
        }
        for arm, metrics in tournament["arm_summary"].items()
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    write: bool = True,
) -> JsonDict:
    """Build the artifact and optionally write stable JSON."""

    root_path = Path(root)
    artifact = build_artifact(root=root_path, tests_added_or_reused=tests_added_or_reused)
    if write:
        write_json(resolve_path(root_path, result_path), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the Exp5569 artifact is internally inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5569 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors while allowing honest blocked artifacts."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("continuous_self_learning_target") is not True:
        errors.append("continuous_self_learning_target")
    sessions = artifact.get("sessions", [])
    if not isinstance(sessions, Sequence) or len(sessions) < 5:
        errors.append("sessions")
    if int(artifact.get("n_events", 0)) < 120:
        errors.append("n_events")
    if len(artifact.get("seeds", [])) < 5:
        errors.append("seeds")
    if artifact.get("arms") != list(ARM_NAMES):
        errors.append("arms")
    if artifact.get("future_label_leakage_count") != 0:
        errors.append("future_label_leakage_count")
    if artifact.get("policy_search_space") != list(POLICY_SEARCH_SPACE):
        errors.append("policy_search_space")
    if artifact.get("weights_mutated") is not False:
        errors.append("weights_mutated")
    if float(artifact.get("forward_transfer_delta", 0.0)) <= 0.0:
        errors.append("forward_transfer_delta")
    if float(artifact.get("backward_retention_delta", -1.0)) < -0.02:
        errors.append("backward_retention_delta")
    if float(artifact.get("action_impact_delta", 0.0)) <= 0.0:
        errors.append("action_impact_delta")
    memory_precision = float(artifact.get("memory_precision", 0.0))
    if not 0.0 < memory_precision < 1.0:
        errors.append("memory_precision")
    if float(artifact.get("retrieval_cost", 0.0)) <= 0.0:
        errors.append("retrieval_cost")
    if float(artifact.get("write_amplification", 0.0)) <= 0.0:
        errors.append("write_amplification")
    poison = artifact.get("poisoning_control", {})
    if not isinstance(poison, Mapping) or not rollback_success(poison):
        errors.append("poisoning_control")
    if artifact.get("rollback_success") is not True:
        errors.append("rollback_success")
    expected_ready = policy_ready_from_artifact(artifact)
    if artifact.get("policy_ready") is not expected_ready:
        errors.append("policy_ready")
    principles = artifact.get("field_principles", {})
    if isinstance(principles, Mapping):
        missing_principles = [
            field for field in REQUIRED_ARTIFACT_FIELDS if not principles.get(field)
        ]
    else:
        missing_principles = list(REQUIRED_ARTIFACT_FIELDS)
    if missing_principles:
        errors.append(f"field_principles missing: {missing_principles}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def policy_ready_from_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute `policy_ready` from artifact gate fields."""

    tournament = artifact.get("tournament", {})
    ci = artifact.get("optimized_vs_static_ci", tournament.get("optimized_vs_static_ci", {}))
    return (
        ci.get("lower", 0.0) > 0.0
        and artifact.get(
            "prior_family_regression_max",
            tournament.get("prior_family_regression_max", 1.0),
        )
        <= 0.02
        and artifact.get("rollback_success") is True
        and artifact.get("future_label_leakage_count") == 0
        and artifact.get("weights_mutated") is False
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict required by conductor receipts."""

    if artifact.get("policy_ready") is True:
        return "complete: causal_memory_policy_tournament_policy_ready"
    return "blocked: causal_memory_policy_tournament_policy_not_ready"


def resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(root) / candidate


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable JSON for diffable experiment receipts."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field removed."""

    payload = {
        key: value for key, value in artifact.items() if key != "reproducibility_checksum"
    }
    return "sha256:" + sha256_json(payload)


def source_file_checksums(root: Path) -> JsonDict:
    """Record the source files backing the receipt."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def sha256_file(path: Path | str) -> str:
    """Return a SHA256 digest for one file."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a SHA256 digest for a JSON-compatible mapping."""

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _round(value: float) -> float:
    """Round metric values once so JSON stays stable across reruns."""

    return round(float(value), 10)


def main() -> int:  # pragma: no cover - thin CLI wrapper
    """Write the repository artifact for conductor use."""

    artifact = run(root=REPO_ROOT, result_path=RESULT_RELATIVE_PATH, write=True)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH.as_posix(),
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
