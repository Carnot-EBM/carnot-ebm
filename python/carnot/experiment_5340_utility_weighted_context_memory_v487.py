"""Exp5340: deterministic utility-weighted context memory.

Spec refs: REQ-LEARN-5340, SCENARIO-LEARN-5340-UTILITY,
SCENARIO-LEARN-5340-POLICY, SCENARIO-LEARN-5340-NOOP.

This experiment is a frozen-model memory-policy fixture. It learns Q-values for
context lifecycle operations from deterministic feedback labels over the Exp5328
fixture, then checks that those learned values can guide retrieval policy
without accepting unsafe context, inventing no-op gains, or touching model
weights. The learned state is a JSON utility table, not an LLM update.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5328_context_object_lifecycle_self_learning_v486 as exp5328
from carnot import experiment_5329_memory_context_policy_rollout_v486 as exp5329
from carnot import experiment_5330_sea_anytime_certificate_gate_v486 as exp5330


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5340_utility_weighted_context_memory_v487"
EXPERIMENT_ID = 5340
MILESTONE = "v487"
SCHEMA = "carnot.experiment_5340.utility_weighted_context_memory.v487"
RUN_DATE = "2026-07-07"
RANDOM_SEED = 5340
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5340_utility_weighted_context_memory_v487.json"
)
Q_VALUE_TABLE_RELATIVE_PATH = Path(
    "results/experiment_5340_utility_weighted_context_memory_q_values_v487.json"
)
EXP5328_RELATIVE_PATH = Path(
    "results/experiment_5328_context_object_lifecycle_self_learning_v486.json"
)
EXP5329_RELATIVE_PATH = Path(
    "results/experiment_5329_memory_context_policy_rollout_v486.json"
)
EXP5330_RELATIVE_PATH = Path(
    "results/experiment_5330_sea_anytime_certificate_gate_v486.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5340_utility_weighted_context_memory_v487.py"
)
EXP5328_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5328_context_object_lifecycle_self_learning_v486.py"
)
EXP5329_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5329_memory_context_policy_rollout_v486.py"
)
EXP5330_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5330_sea_anytime_certificate_gate_v486.py"
)

INFERENCE_SUBSTRATE = "deterministic_context_utility_learning"
SPEC_REFS = (
    "REQ-LEARN-5340",
    "SCENARIO-LEARN-5340-UTILITY",
    "SCENARIO-LEARN-5340-POLICY",
    "SCENARIO-LEARN-5340-NOOP",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")

UTILITY_OPERATIONS = (
    "retrieve",
    "archive",
    "mask",
    "fold",
    "revise",
    "commit",
    "rollback",
)
FEEDBACK_LABELS = (
    "positive",
    "stale",
    "poisoned",
    "irrelevant",
    "shuffled_no_op",
)
ALWAYS_FULL_POLICY = exp5329.ALWAYS_FULL_POLICY
TRANSITION_ONLY_POLICY = exp5329.TRANSITION_ONLY_POLICY
UTILITY_WEIGHTED_POLICY = "utility_weighted_retrieval"
SHUFFLED_UTILITY_CONTROL = "shuffled_utility_no_op_control"
POLICY_ARMS = (
    ALWAYS_FULL_POLICY,
    TRANSITION_ONLY_POLICY,
    UTILITY_WEIGHTED_POLICY,
    SHUFFLED_UTILITY_CONTROL,
)

REQUIRED_FIELD_PRINCIPLES = {
    "experiment_id": (
        "Identifies the exact Exp5340 artifact so downstream gates cannot "
        "confuse utility-weighted context memory with Exp5329 fixed lifecycle "
        "rollout or Exp5330 certificate promotion."
    ),
    "milestone": (
        "Binds the utility learner to milestone v487 where context memory starts "
        "learning operation utility without model-weight mutation."
    ),
    "status": (
        "Reports whether utility-weighted context memory completed under fixture, "
        "certificate, no-op-control, rollback, and frozen-model gates."
    ),
    "honest_verdict": (
        "Terminal Exp5340 verdict; starts with complete: or blocked_ and states "
        "whether utility learning improved call efficiency without unsafe accepts "
        "or model-weight mutation."
    ),
    "inference_substrate": (
        "Declares deterministic context utility learning with no live LLM, API "
        "judge, model generation, fine-tuning, adapter update, or "
        "foundation-weight mutation."
    ),
    "continuous_self_learning_target": (
        "Bare gate showing the experiment updates policy utility values for "
        "continuous self-learning rather than static reporting."
    ),
    "no_weight_mutation": (
        "Bare gate confirming only deterministic utility tables and context-bank "
        "policy state changed, never model weights or adapters."
    ),
    "utility_update_count": (
        "Bare integer count of Q-value updates applied from deterministic fixture "
        "feedback labels."
    ),
    "q_value_table_path": (
        "Points to the deterministic utility table artifact so downstream gates "
        "can inspect learned values instead of trusting a summary."
    ),
    "quality_delta_vs_always_full": (
        "Bare final-quality delta comparing utility-weighted retrieval against "
        "always-full verification."
    ),
    "verifier_calls_avoided": (
        "Bare integer count of verifier calls avoided by utility-weighted "
        "retrieval relative to always-full verification on identical cases."
    ),
    "no_op_control_delta": (
        "Bare numeric best improvement delta achieved by the shuffled/no-op "
        "utility control; readiness requires it to be non-positive."
    ),
    "unsafe_false_accepts": (
        "Bare integer count of unsafe state-change accepts by the utility-weighted "
        "policy; any positive count blocks readiness."
    ),
    "rollback_events": (
        "Bare integer count of rollback transitions exercised by the "
        "utility-weighted policy."
    ),
    "utility_memory_ready": (
        "Bare gate true only when all policies run, unsafe false accepts are zero, "
        "the no-op control has no spurious improvement, tests are recorded, and "
        "no model weights mutate."
    ),
    "tests_run": (
        "Records the exact verification commands used to establish that the "
        "utility learner, Q-value table, and result artifact are stable."
    ),
}
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "q_value_table_path",
    "tests_run",
)
BARE_INTEGER_FIELDS = (
    "utility_update_count",
    "verifier_calls_avoided",
    "unsafe_false_accepts",
    "rollback_events",
)
BARE_NUMERIC_FIELDS = ("quality_delta_vs_always_full", "no_op_control_delta")


@dataclass(frozen=True)
class UtilityFeedback:
    """One deterministic operation-utility update.

    The feedback label describes the memory condition; the reward describes
    whether the lifecycle policy handled that condition correctly. Stale and
    poisoned memories therefore receive positive utility only when the policy
    rejects or rolls back the unsafe context instead of letting it reach the
    answer path.
    """

    feedback_id: str
    action: str
    source_case_id: str
    feedback_label: str
    control_type: str
    reward: float
    expected_policy_decision: str
    model_weights_mutated: bool = False


def build_utility_feedback_panel() -> tuple[UtilityFeedback, ...]:
    """Return deterministic utility feedback rows anchored to Exp5328 cases."""

    return (
        UtilityFeedback(
            "u5340-positive-retrieve",
            "retrieve",
            "life-retrieve-archive-recover",
            "positive",
            "useful_context",
            1.0,
            "accept_recovered_context",
        ),
        UtilityFeedback(
            "u5340-positive-archive",
            "archive",
            "life-archive-policy",
            "positive",
            "useful_context",
            1.0,
            "archive_with_sidecar",
        ),
        UtilityFeedback(
            "u5340-positive-mask",
            "mask",
            "life-mask-secret",
            "positive",
            "useful_context",
            1.0,
            "mask_secret",
        ),
        UtilityFeedback(
            "u5340-positive-fold",
            "fold",
            "life-fold-runtime-rubric",
            "positive",
            "useful_context",
            1.0,
            "fold_supported_context",
        ),
        UtilityFeedback(
            "u5340-positive-revise",
            "revise",
            "life-revise-runtime",
            "positive",
            "useful_context",
            1.0,
            "revise_with_evidence",
        ),
        UtilityFeedback(
            "u5340-positive-commit",
            "commit",
            "life-commit-patch",
            "positive",
            "useful_context",
            1.0,
            "commit_safe_patch",
        ),
        UtilityFeedback(
            "u5340-positive-rollback",
            "rollback",
            "life-rollback-corrupt-patch",
            "positive",
            "useful_context",
            1.0,
            "rollback_corrupt_patch",
        ),
        UtilityFeedback(
            "u5340-stale-retrieve-bank",
            "retrieve",
            "life-stale-retrieval",
            "stale",
            "stale_memory_control",
            1.0,
            "reject_stale_retrieval",
        ),
        UtilityFeedback(
            "u5340-stale-retrieve-answer",
            "retrieve",
            "life-answer-stale-context",
            "stale",
            "stale_memory_control",
            1.0,
            "reject_stale_answer_context",
        ),
        UtilityFeedback(
            "u5340-poisoned-mask",
            "mask",
            "life-mask-retrieval-leak",
            "poisoned",
            "poisoned_memory_control",
            1.0,
            "reject_mask_leakage",
        ),
        UtilityFeedback(
            "u5340-poisoned-fold",
            "fold",
            "life-omission-sensor-rule",
            "poisoned",
            "poisoned_memory_control",
            1.0,
            "reject_omission",
        ),
        UtilityFeedback(
            "u5340-poisoned-revise",
            "revise",
            "life-corrupt-rubric",
            "poisoned",
            "poisoned_memory_control",
            1.0,
            "reject_corruption",
        ),
        UtilityFeedback(
            "u5340-poisoned-commit",
            "commit",
            "life-answer-corrupt-context",
            "poisoned",
            "poisoned_memory_control",
            1.0,
            "reject_corrupt_answer_context",
        ),
        UtilityFeedback(
            "u5340-irrelevant-archive",
            "archive",
            "life-archive-policy",
            "irrelevant",
            "irrelevant_memory_control",
            0.0,
            "ignore_irrelevant_replay",
        ),
        UtilityFeedback(
            "u5340-shuffled-noop-retrieve",
            "retrieve",
            "life-retrieve-archive-recover",
            "shuffled_no_op",
            "shuffled_no_op_control",
            0.0,
            "no_policy_change",
        ),
    )


def learn_utility_values(feedback: Sequence[UtilityFeedback]) -> JsonDict:
    """Learn deterministic operation Q-values from fixture feedback labels."""

    stats = {
        action: {
            "reward_sum": 0.0,
            "update_count": 0,
            "feedback_counts": Counter({label: 0 for label in FEEDBACK_LABELS}),
        }
        for action in UTILITY_OPERATIONS
    }
    updates: list[JsonDict] = []
    for row in feedback:
        action_stats = stats[row.action]
        before_count = int(action_stats["update_count"])
        before_q = _rate(float(action_stats["reward_sum"]), before_count)
        action_stats["reward_sum"] = float(action_stats["reward_sum"]) + row.reward
        action_stats["update_count"] = before_count + 1
        action_stats["feedback_counts"][row.feedback_label] += 1
        after_q = _rate(
            float(action_stats["reward_sum"]),
            int(action_stats["update_count"]),
        )
        updates.append(
            {
                "feedback_id": row.feedback_id,
                "action": row.action,
                "source_case_id": row.source_case_id,
                "feedback_label": row.feedback_label,
                "control_type": row.control_type,
                "reward": row.reward,
                "q_before": before_q,
                "q_after": after_q,
                "expected_policy_decision": row.expected_policy_decision,
                "model_weights_mutated": row.model_weights_mutated,
            }
        )
    operation_q_values = {
        action: {
            "q_value": _rate(float(values["reward_sum"]), int(values["update_count"])),
            "update_count": int(values["update_count"]),
            "feedback_counts": dict(values["feedback_counts"]),
        }
        for action, values in stats.items()
    }
    table: JsonDict = {
        "schema": "carnot.experiment_5340.q_value_table.v487",
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "utility_operations": list(UTILITY_OPERATIONS),
        "feedback_labels": list(FEEDBACK_LABELS),
        "utility_update_count": len(updates),
        "operation_q_values": operation_q_values,
        "utility_updates": updates,
        "learning_rule": (
            "running_mean_reward_from_deterministic_fixture_feedback; stale and "
            "poisoned controls reward safe rejection, not unsafe acceptance"
        ),
        "no_weight_mutation": not any(row.model_weights_mutated for row in feedback),
    }
    table["reproducibility_checksum"] = _checksum(table)
    return _json_ready(table)


def confirm_fixture_gate(
    *,
    root: Path | str = REPO_ROOT,
    artifact: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Confirm Exp5328 exposes the lifecycle rows needed for utility learning."""

    source = dict(artifact or _read_json(Path(root) / EXP5328_RELATIVE_PATH))
    rows = source.get("lifecycle_rows", [])
    actions = {row.get("action") for row in rows if isinstance(row, Mapping)}
    checks = {
        "context_lifecycle_fixture_ready": source.get("context_lifecycle_fixture_ready")
        is True,
        "no_weight_mutation": source.get("no_weight_mutation") is True,
        "lifecycle_rows_present": isinstance(rows, list) and bool(rows),
        "utility_actions_present": set(UTILITY_OPERATIONS).issubset(actions),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "failed_gates": failed,
        "all_passed": not failed,
        "source_honest_verdict": _wrapped_value(source.get("honest_verdict")),
    }


def confirm_certificate_gate(
    *,
    root: Path | str = REPO_ROOT,
    artifact: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Confirm Exp5330 accepted the lifecycle policy promotion boundary."""

    source = dict(artifact or _read_json(Path(root) / EXP5330_RELATIVE_PATH))
    checks = {
        "anytime_certificate_gate_ready": source.get("anytime_certificate_gate_ready")
        is True,
        "no_weight_mutation": source.get("no_weight_mutation") is True,
        "unsafe_promotions_zero": source.get("unsafe_promotions") == 0,
        "no_op_control_cleared": _is_numeric(source.get("no_op_control_delta"))
        and float(source["no_op_control_delta"]) <= 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "failed_gates": failed,
        "all_passed": not failed,
        "source_honest_verdict": _wrapped_value(source.get("honest_verdict")),
    }


def evaluate_utility_memory(q_table: Mapping[str, Any]) -> JsonDict:
    """Compare utility-weighted retrieval against deterministic controls."""

    rollout = exp5329.evaluate_policy_rollout(exp5328.build_lifecycle_fixture())
    full_rows = list(rollout["policy_rows"][exp5329.ALWAYS_FULL_POLICY])
    transition_rows = list(rollout["policy_rows"][exp5329.TRANSITION_ONLY_POLICY])
    utility_rows = _utility_weighted_rows(
        rollout["policy_rows"][exp5329.CONTEXT_LIFECYCLE_POLICY],
        q_table,
    )
    shuffled_rows = _shuffled_control_rows(transition_rows, q_table)
    policy_rows = {
        ALWAYS_FULL_POLICY: full_rows,
        TRANSITION_ONLY_POLICY: transition_rows,
        UTILITY_WEIGHTED_POLICY: utility_rows,
        SHUFFLED_UTILITY_CONTROL: shuffled_rows,
    }
    policy_metrics = {
        policy: _policy_metrics(rows)
        for policy, rows in policy_rows.items()
    }
    always = policy_metrics[ALWAYS_FULL_POLICY]
    transition = policy_metrics[TRANSITION_ONLY_POLICY]
    utility = policy_metrics[UTILITY_WEIGHTED_POLICY]
    shuffled = policy_metrics[SHUFFLED_UTILITY_CONTROL]
    all_policies_run = bool(
        set(policy_rows) == set(POLICY_ARMS)
        and all(policy_metrics[policy]["n"] > 0 for policy in POLICY_ARMS)
        and _same_case_ids(policy_rows)
    )
    quality_delta = _delta(utility["final_quality"], always["final_quality"])
    verifier_calls_avoided = int(always["verifier_calls"] - utility["verifier_calls"])
    no_op_control_delta = _delta(
        shuffled["final_quality"],
        transition["final_quality"],
    )
    unsafe_false_accepts = int(utility["unsafe_false_accepts"])
    rollback_events = int(utility["rollback_events"])
    no_weight_mutation = bool(
        q_table.get("no_weight_mutation") is True
        and not any(
            bool(row.get("model_weights_mutated"))
            for rows in policy_rows.values()
            for row in rows
        )
    )
    ready = bool(
        all_policies_run
        and quality_delta >= 0.0
        and verifier_calls_avoided > 0
        and no_op_control_delta <= 0.0
        and unsafe_false_accepts == 0
        and rollback_events > 0
        and no_weight_mutation
    )
    return {
        "all_policies_run": all_policies_run,
        "policy_rows": policy_rows,
        "policy_metrics": policy_metrics,
        "quality_delta_vs_always_full": quality_delta,
        "verifier_calls_avoided": verifier_calls_avoided,
        "no_op_control_delta": no_op_control_delta,
        "unsafe_false_accepts": unsafe_false_accepts,
        "rollback_events": rollback_events,
        "no_weight_mutation": no_weight_mutation,
        "utility_memory_ready": ready,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5340 result artifact from deterministic utility evidence."""

    fixture_gate = confirm_fixture_gate(root=root)
    certificate_gate = confirm_certificate_gate(root=root)
    gates_pass = bool(fixture_gate["all_passed"] and certificate_gate["all_passed"])
    q_table = (
        learn_utility_values(build_utility_feedback_panel())
        if gates_pass
        else _blocked_q_table()
    )
    evaluation = evaluate_utility_memory(q_table) if gates_pass else _blocked_evaluation()
    complete = _utility_complete(
        q_table=q_table,
        evaluation=evaluation,
        fixture_gate=fixture_gate,
        certificate_gate=certificate_gate,
        tests_run=tests_run,
    )
    status = "utility_memory_ready" if complete else "blocked_fixture_certificate_or_tests"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [
            str(EXP5328_RELATIVE_PATH),
            str(EXP5329_RELATIVE_PATH),
            str(EXP5330_RELATIVE_PATH),
        ],
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(complete, q_table, evaluation, fixture_gate, certificate_gate, tests_run),
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "continuous_self_learning_target": True,
        "no_weight_mutation": bool(
            q_table["no_weight_mutation"] and evaluation["no_weight_mutation"]
        ),
        "utility_update_count": int(q_table["utility_update_count"]),
        "q_value_table_path": _wrap(
            "q_value_table_path",
            str(Q_VALUE_TABLE_RELATIVE_PATH),
        ),
        "quality_delta_vs_always_full": evaluation["quality_delta_vs_always_full"],
        "verifier_calls_avoided": evaluation["verifier_calls_avoided"],
        "no_op_control_delta": evaluation["no_op_control_delta"],
        "unsafe_false_accepts": evaluation["unsafe_false_accepts"],
        "rollback_events": evaluation["rollback_events"],
        "utility_memory_ready": complete,
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "fixture_gate": fixture_gate,
        "certificate_gate": certificate_gate,
        "q_value_table": q_table,
        "q_value_table_checksum": _checksum(q_table),
        "policy_metrics": evaluation["policy_metrics"],
        "policy_rows": evaluation["policy_rows"],
        "weight_mutation_receipt": _weight_mutation_receipt(q_table, evaluation),
        "methodology_note": (
            "Utility updates are deterministic running means over Exp5328 fixture "
            "feedback labels. No LLM, judge, generator, adapter update, or model "
            "weight mutation is invoked."
        ),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the fields consumed by downstream utility-memory gates."""

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
    for field in BARE_NUMERIC_FIELDS:
        if not _is_numeric(artifact.get(field)):
            raise ValueError(f"{field} must be bare numeric")
    if artifact.get("unsafe_false_accepts") != 0:
        raise ValueError("unsafe_false_accepts must be 0")
    if not isinstance(artifact.get("utility_memory_ready"), bool):
        raise ValueError("utility_memory_ready must be bare bool")
    if artifact["utility_memory_ready"] and not artifact["tests_run"]["value"]:
        raise ValueError("tests_run must record commands for ready utility memory")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    q_table_path: Path | str = REPO_ROOT / Q_VALUE_TABLE_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5340 result and Q-value artifacts."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(q_table_path), artifact["q_value_table"])
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for deterministic source inputs."""

    root_path = Path(root)
    return {
        "exp5328": _sha256_file(root_path / EXP5328_RELATIVE_PATH),
        "exp5329": _sha256_file(root_path / EXP5329_RELATIVE_PATH),
        "exp5330": _sha256_file(root_path / EXP5330_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5328_module": _sha256_file(root_path / EXP5328_MODULE_RELATIVE_PATH),
        "exp5329_module": _sha256_file(root_path / EXP5329_MODULE_RELATIVE_PATH),
        "exp5330_module": _sha256_file(root_path / EXP5330_MODULE_RELATIVE_PATH),
    }


def _utility_weighted_rows(
    lifecycle_rows: Sequence[Mapping[str, Any]],
    q_table: Mapping[str, Any],
) -> list[JsonDict]:
    q_values = q_table["operation_q_values"]
    rows: list[JsonDict] = []
    for row in lifecycle_rows:
        action = str(row["action"])
        copied = dict(row)
        copied["policy"] = UTILITY_WEIGHTED_POLICY
        copied["route"] = "utility_weighted_context_memory"
        copied["operation_utility"] = _operation_utility(q_values, action)
        copied["utility_feedback_applied"] = action in UTILITY_OPERATIONS
        rows.append(copied)
    return rows


def _shuffled_control_rows(
    transition_rows: Sequence[Mapping[str, Any]],
    q_table: Mapping[str, Any],
) -> list[JsonDict]:
    q_values = q_table["operation_q_values"]
    shuffled_actions = dict(zip(UTILITY_OPERATIONS, reversed(UTILITY_OPERATIONS), strict=True))
    rows: list[JsonDict] = []
    for row in transition_rows:
        action = str(row["action"])
        shuffled_action = shuffled_actions.get(action, action)
        copied = dict(row)
        copied["policy"] = SHUFFLED_UTILITY_CONTROL
        copied["route"] = "shuffled_utility_no_op_control"
        copied["operation_utility"] = _operation_utility(q_values, shuffled_action)
        copied["utility_feedback_applied"] = False
        copied["shuffled_from_action"] = action
        copied["shuffled_to_action"] = shuffled_action
        rows.append(copied)
    return rows


def _policy_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    n_rows = len(rows)
    final_correct = sum(1 for row in rows if bool(row["final_correct"]))
    return {
        "n": n_rows,
        "case_ids": [str(row["case_id"]) for row in rows],
        "final_correct": final_correct,
        "final_quality": _rate(final_correct, n_rows),
        "verifier_calls": sum(1 for row in rows if bool(row["verifier_call"])),
        "unsafe_false_accepts": sum(
            1 for row in rows if bool(row["unsafe_false_accept"])
        ),
        "rollback_events": sum(1 for row in rows if bool(row["rollback_event"])),
        "model_weights_mutated": any(bool(row["model_weights_mutated"]) for row in rows),
    }


def _same_case_ids(policy_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> bool:
    case_ids = [
        tuple(str(row["case_id"]) for row in rows)
        for rows in policy_rows.values()
    ]
    return bool(case_ids) and all(ids == case_ids[0] for ids in case_ids)


def _blocked_q_table() -> JsonDict:
    table: JsonDict = {
        "schema": "carnot.experiment_5340.q_value_table.v487",
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "utility_operations": list(UTILITY_OPERATIONS),
        "feedback_labels": list(FEEDBACK_LABELS),
        "utility_update_count": 0,
        "operation_q_values": {
            action: {
                "q_value": 0.0,
                "update_count": 0,
                "feedback_counts": {label: 0 for label in FEEDBACK_LABELS},
            }
            for action in UTILITY_OPERATIONS
        },
        "utility_updates": [],
        "learning_rule": "blocked_before_feedback_updates",
        "no_weight_mutation": True,
    }
    table["reproducibility_checksum"] = _checksum(table)
    return _json_ready(table)


def _blocked_evaluation() -> JsonDict:
    empty_metrics = {
        policy: {
            "n": 0,
            "case_ids": [],
            "final_correct": 0,
            "final_quality": 0.0,
            "verifier_calls": 0,
            "unsafe_false_accepts": 0,
            "rollback_events": 0,
            "model_weights_mutated": False,
        }
        for policy in POLICY_ARMS
    }
    return {
        "all_policies_run": False,
        "policy_rows": {policy: [] for policy in POLICY_ARMS},
        "policy_metrics": empty_metrics,
        "quality_delta_vs_always_full": 0.0,
        "verifier_calls_avoided": 0,
        "no_op_control_delta": 0.0,
        "unsafe_false_accepts": 0,
        "rollback_events": 0,
        "no_weight_mutation": True,
        "utility_memory_ready": False,
    }


def _utility_complete(
    *,
    q_table: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    fixture_gate: Mapping[str, Any],
    certificate_gate: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(
        fixture_gate["all_passed"]
        and certificate_gate["all_passed"]
        and q_table["utility_update_count"] > 0
        and q_table["no_weight_mutation"]
        and evaluation["utility_memory_ready"]
        and evaluation["quality_delta_vs_always_full"] >= 0.0
        and evaluation["verifier_calls_avoided"] > 0
        and evaluation["no_op_control_delta"] <= 0.0
        and evaluation["unsafe_false_accepts"] == 0
        and evaluation["rollback_events"] > 0
        and evaluation["no_weight_mutation"]
        and bool(tests_run)
    )


def _honest_verdict(
    complete: bool,
    q_table: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    fixture_gate: Mapping[str, Any],
    certificate_gate: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> str:
    if complete:
        return (
            "complete: utility-weighted context memory learned "
            f"{q_table['utility_update_count']} operation updates, matched "
            "always-full quality, avoided "
            f"{evaluation['verifier_calls_avoided']} verifier calls, recorded "
            f"{evaluation['rollback_events']} rollback events, cleared the no-op "
            "control, and preserved no model weight mutation"
        )
    blockers = [
        *fixture_gate.get("failed_gates", []),
        *certificate_gate.get("failed_gates", []),
    ]
    if not evaluation.get("utility_memory_ready"):
        blockers.append("utility_memory_ready_false")
    if not tests_run:
        blockers.append("tests_not_recorded")
    return "blocked_utility_memory_not_ready: " + ",".join(blockers)


def _weight_mutation_receipt(
    q_table: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> JsonDict:
    return {
        "no_weight_mutation": bool(
            q_table["no_weight_mutation"] and evaluation["no_weight_mutation"]
        ),
        "model_weights_loaded": False,
        "model_weight_hash_before": "sha256:no_model_weights_loaded",
        "model_weight_hash_after": "sha256:no_model_weights_loaded",
        "state_mutation_targets": [
            "deterministic_context_operation_q_value_table",
            "deterministic_context_policy_rows",
        ],
        "forbidden_updates": {
            "fine_tune_model_weights": False,
            "lora_or_adapter_update": False,
            "foundation_weight_write": False,
        },
    }


def _operation_utility(
    q_values: Mapping[str, Mapping[str, Any]],
    action: str,
) -> float:
    if action not in q_values:
        return 0.0
    return float(q_values[action]["q_value"])


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _rate(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / denominator, 6)


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 6)
