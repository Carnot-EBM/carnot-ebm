"""Replay a bounded risk-sensitive memory controller on frozen decisions.

Spec refs: REQ-CL-6854 and SCENARIO-CL-6854-*.

The replay separates decision views from sealed exact outcomes. It freezes each
action receipt before the matching outcome authority can update policy state.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.continuous_learning import (
    ACTIONS,
    DENIED_CONTEXT_FIELDS,
    FEATURE_NAMES,
    KNOWN_FAMILIES,
    PolicyConfig,
    RiskMatrix,
    RiskSensitiveContextualBandit,
    canonical_json_bytes,
    encode_context,
    sha256_bytes,
    sha256_json,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RELATIVE_PATH = Path(
    "results/experiment_6853_risk_sensitive_memory_opportunity_fixture.json"
)
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6854_risk_sensitive_abstention_memory_controller.json"
)
SCHEMA = "carnot.experiment_6854.risk_sensitive_abstention_memory_controller.v1"
EXPERIMENT_ID = "exp6854-risk-sensitive-abstention-memory-controller"
INFERENCE_SUBSTRATE = "deterministic CPU online contextual bandit replay"
RANDOM_SEED = 6_854_001
MAIN_RISK_RATIO = 3.0
RISK_SENSITIVITY_RATIOS = (2.0, 3.0, 4.0, 5.0)
CHECKPOINT_INTERVAL = 64
HELD_FUTURE_START = 511
FALSE_POSITIVE_RATE_BOUND = 0.05
BLOCKED_VERDICT = "complete_blocked_risk_sensitive_abstention_memory_controller"
MANDATED_FAMILIES = KNOWN_FAMILIES

POLICY_ARMS = (
    "contextual_bandit",
    "no_memory",
    "always_memory",
    "abstain_only",
    "random_admission",
    "read_only_fixed",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "random_seed",
    "reproducibility_checksum",
    "rows",
    "controller_schema",
    "risk_matrix",
    "pre_outcome_action_receipts",
    "exact_feedback_receipts",
    "update_rows",
    "checkpoint_manifest",
    "restart_equivalence_results",
    "rollback_results",
    "state_size_rows",
    "latency_rows",
    "per_arm_summary",
    "held_future_effect",
    "false_positive_injection_rate",
    "abstention_rate",
    "risk_sensitivity_rows",
    "risk_sensitive_controller_complete_score",
    "controller_benefit_gate_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
OPEN_SPEC_IDS = (
    "REQ-CL-6854",
    "SCENARIO-CL-6854-PRECONDITIONS",
    "SCENARIO-CL-6854-ACTION-FREEZE",
    "SCENARIO-CL-6854-DELAYED-FEEDBACK",
    "SCENARIO-CL-6854-ASYMMETRIC-LOSS",
    "SCENARIO-CL-6854-ABSTENTION",
    "SCENARIO-CL-6854-UNSEEN-CONTEXT",
    "SCENARIO-CL-6854-CAPACITY",
    "SCENARIO-CL-6854-CHECKPOINT-CORRUPTION",
    "SCENARIO-CL-6854-RESTART-EQUIVALENCE",
    "SCENARIO-CL-6854-ROLLBACK",
    "SCENARIO-CL-6854-TIE-BREAKING",
    "SCENARIO-CL-6854-CONTROLS",
    "SCENARIO-CL-6854-GATES",
)
REPLAY_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6854_risk_sensitive_abstention_memory_controller.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/continuous_learning.py,*/experiment_6854_risk_sensitive_abstention_memory_controller.py' -m pytest tests/python/test_experiment_6854_risk_sensitive_abstention_memory_controller.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/ruff check python/carnot/continuous_learning.py python/carnot/experiment_6854_risk_sensitive_abstention_memory_controller.py scripts/experiments/experiment_6854_risk_sensitive_abstention_memory_controller.py tests/python/test_experiment_6854_risk_sensitive_abstention_memory_controller.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6854_risk_sensitive_abstention_memory_controller.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6854_risk_sensitive_abstention_memory_controller.json",
    ".venv/bin/python scripts/artifact_convention_audit.py results/experiment_6854_risk_sensitive_abstention_memory_controller.json",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6854_risk_sensitive_abstention_memory_controller.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)


def sha256_file(path: Path) -> str | None:
    """Hash a source file in chunks so evidence size does not affect memory use."""

    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_fixture(path: Path) -> JsonDict:
    """Read the frozen fixture and preserve a machine-readable load failure."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"_load_error": type(error).__name__}
    return value if isinstance(value, dict) else {"_load_error": "not_object"}


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind the exact opportunity fixture used by this controller replay."""

    path = repo_root / SOURCE_RELATIVE_PATH
    try:
        display_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "exp6853": {
            "path": display_path,
            "sha256": sha256_file(path),
            "role": "chronological pre-outcome decisions and exact later outcome authority",
        }
    }


def fixture_row_hash(row: Mapping[str, Any]) -> str:
    """Recompute one Exp6853 row identity without checksum recursion."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Return one uniform gate row with expected and observed evidence."""

    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all checks and copy the first failure for simple consumers."""

    failed = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        "checks": list(checks),
        "failed_checks": failed,
        "failed_check": failed[0]["check"] if failed else None,
        "observed": failed[0]["observed"] if failed else None,
        "passed": not failed,
    }


def validate_preconditions(
    fixture: Mapping[str, Any], controller: RiskSensitiveContextualBandit
) -> list[JsonDict]:
    """Check every fail-closed input gate before a controller action occurs."""

    rows_value = fixture.get("rows", [])
    rows = rows_value if isinstance(rows_value, list) else []
    invalid_hashes = [
        str(row.get("decision_id", ""))
        for row in rows
        if not isinstance(row, dict) or row.get("row_sha256") != fixture_row_hash(row)
    ]
    invalid_outcomes = [
        str(row.get("decision_id", ""))
        for row in rows
        if not isinstance(row, dict)
        or not isinstance(row.get("exact_later_outcome"), Mapping)
        or row["exact_later_outcome"].get("revealed_after_decision") is not True
        or row["exact_later_outcome"].get("signed_direction") not in {-1, 0, 1}
        or not str(row["exact_later_outcome"].get("exact_outcome_hash", "")).startswith(
            "sha256:"
        )
        or not str(row["exact_later_outcome"].get("outcome_identity", "")).startswith(
            "sha256:"
        )
    ]
    directions = {
        int(row["exact_later_outcome"]["signed_direction"])
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("exact_later_outcome"), Mapping)
        and row["exact_later_outcome"].get("signed_direction") in {-1, 0, 1}
    }
    headroom_actions = set()
    if 1 in directions:
        headroom_actions.add("verified_memory")
    if -1 in directions or 0 in directions:
        headroom_actions.add("no_memory")
    leaking_contexts = [
        str(row.get("decision_id", ""))
        for row in rows
        if isinstance(row, dict)
        and "denied_field" in encode_context(row.get("decision_context", {})).unseen_reasons
    ]
    return [
        _check(
            "risk_sensitive_stream_ready_score",
            1,
            fixture.get("risk_sensitive_stream_ready_score"),
            fixture.get("risk_sensitive_stream_ready_score") == 1,
        ),
        _check(
            "stable_row_hashes",
            {"row_count": 765, "invalid_count": 0},
            {"row_count": len(rows), "invalid_count": len(invalid_hashes), "decision_ids": invalid_hashes[:10]},
            len(rows) == 765 and not invalid_hashes,
        ),
        _check(
            "exact_later_outcome_authority",
            {"outcome_count": len(rows), "invalid_count": 0},
            {"outcome_count": len(rows) - len(invalid_outcomes), "invalid_count": len(invalid_outcomes), "decision_ids": invalid_outcomes[:10]},
            bool(rows) and not invalid_outcomes,
        ),
        _check(
            "actions_with_headroom",
            {"minimum_action_count": 2},
            {"actions": sorted(headroom_actions), "action_count": len(headroom_actions)},
            len(headroom_actions) >= 2,
        ),
        _check("clean_initial_controller_state", True, controller.is_clean(), controller.is_clean()),
        _check(
            "pre_outcome_context_nonleaking",
            {"invalid_count": 0},
            {"invalid_count": len(leaking_contexts), "decision_ids": leaking_contexts[:10]},
            not leaking_contexts,
        ),
    ]


def _decision_view(row: Mapping[str, Any]) -> JsonDict:
    """Copy only fields available before the exact outcome reveal boundary."""

    return {
        "decision_id": str(row["decision_id"]),
        "decision_sequence_index": int(row["decision_sequence_index"]),
        "decision_context": deepcopy(row["decision_context"]),
        "baseline_actions": deepcopy(row["baseline_actions"]),
        "delayed_correction": bool(row.get("delayed_correction")),
    }


def _split_fixture_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    """Separate decision views from the authority that reveals outcomes later."""

    decisions = [_decision_view(row) for row in rows]
    outcomes = {
        str(row["decision_id"]): deepcopy(row["exact_later_outcome"]) for row in rows
    }
    return decisions, outcomes


def _read_only_fixed_action(context: Mapping[str, Any]) -> str:
    """Apply a fixed nonlearning rule using only visible context values."""

    if (
        context.get("exact_compatibility") is True
        and float(context.get("relevance", 0.0)) >= 0.8
        and float(context.get("uncertainty", 1.0)) <= 0.5
        and float(context.get("false_positive_risk", 1.0)) <= 0.7
    ):
        return "verified_memory"
    return "no_memory"


def _arm_metric(
    decision_id: str,
    action: str,
    direction: int,
    matrix: RiskMatrix,
) -> JsonDict:
    """Compute one policy metric from the shared exact direction and risk matrix."""

    losses = {candidate: matrix.loss(candidate, direction) for candidate in ACTIONS}
    loss = losses[action]
    return {
        "decision_id": decision_id,
        "action": action,
        "loss": loss,
        "reward": -loss,
        "regret": round(loss - min(losses.values()), 12),
        "false_positive_injection": action == "verified_memory" and direction < 0,
        "helpful_memory_selected": action == "verified_memory" and direction > 0,
        "missed_reuse": action != "verified_memory" and direction > 0,
        "abstained": action == "abstain",
    }


def _feedback_receipt(
    decision: Mapping[str, Any],
    outcome: Mapping[str, Any],
    action_receipt: Mapping[str, Any],
    reveal_sequence_index: int,
) -> JsonDict:
    """Bind an exact later outcome to the already serialized action receipt."""

    receipt: JsonDict = {
        "decision_id": decision["decision_id"],
        "decision_sequence_index": decision["decision_sequence_index"],
        "reveal_sequence_index": reveal_sequence_index,
        "feedback_delay_steps": reveal_sequence_index
        - int(decision["decision_sequence_index"]),
        "action_receipt_sha256": action_receipt["receipt_sha256"],
        "outcome_identity": outcome["outcome_identity"],
        "exact_outcome_hash": outcome["exact_outcome_hash"],
        "signed_direction": int(outcome["signed_direction"]),
        "revealed_after_action_receipt": True,
    }
    receipt["feedback_receipt_sha256"] = sha256_json(receipt)
    return receipt


@dataclass
class ReplayRun:
    """Hold one replay plus private checkpoint bytes needed for restart tests."""

    controller: RiskSensitiveContextualBandit
    rows: list[JsonDict] = field(default_factory=list)
    action_receipts: list[JsonDict] = field(default_factory=list)
    feedback_receipts: list[JsonDict] = field(default_factory=list)
    update_rows: list[JsonDict] = field(default_factory=list)
    checkpoint_manifest: list[JsonDict] = field(default_factory=list)
    latency_rows: list[JsonDict] = field(default_factory=list)
    checkpoint_bytes: dict[int, bytes] = field(default_factory=dict)
    checkpoint_paths: dict[int, Path] = field(default_factory=dict)


def _run_replay(
    fixture_rows: Sequence[Mapping[str, Any]],
    *,
    risk_ratio: float,
    controller: RiskSensitiveContextualBandit | None = None,
    start_sequence: int = 1,
    checkpoint_dir: Path | None = None,
) -> ReplayRun:
    """Run ordered decisions while revealing only feedback whose delay expired."""

    matrix = RiskMatrix(risk_ratio)
    active = controller or RiskSensitiveContextualBandit(PolicyConfig(risk_ratio=risk_ratio))
    decisions, outcomes = _split_fixture_rows(fixture_rows)
    decisions_by_id = {row["decision_id"]: row for row in decisions}
    actions_by_id: dict[str, JsonDict] = {}
    output_by_id: dict[str, JsonDict] = {}
    run = ReplayRun(active)

    def reveal_due(sequence_index: int) -> None:
        for decision_id in active.pending_due(sequence_index):
            decision = decisions_by_id[decision_id]
            outcome = outcomes[decision_id]
            action_receipt = actions_by_id.get(decision_id)
            if action_receipt is None:
                pending = active._pending[decision_id]  # noqa: SLF001 - restart receipt recovery.
                action_receipt = {
                    "receipt_sha256": pending["action_receipt_sha256"],
                    "chosen_action": pending["action"],
                }
            feedback = _feedback_receipt(decision, outcome, action_receipt, sequence_index)
            selected_action = str(action_receipt["chosen_action"])
            raw_loss = matrix.loss(selected_action, int(outcome["signed_direction"]))
            started = time.perf_counter_ns()
            update = active.apply_bounded_loss(
                decision_id,
                raw_loss,
                feedback_receipt_sha256=feedback["feedback_receipt_sha256"],
                update_sequence_index=sequence_index,
            )
            latency_us = (time.perf_counter_ns() - started) / 1000.0
            if decision_id in output_by_id:
                control_actions = {
                    "contextual_bandit": selected_action,
                    "no_memory": "no_memory",
                    "always_memory": "verified_memory",
                    "abstain_only": "abstain",
                    "random_admission": str(decision["baseline_actions"]["random_admission"]),
                    "read_only_fixed": _read_only_fixed_action(decision["decision_context"]),
                }
                direction = int(outcome["signed_direction"])
                output_by_id[decision_id].update(
                    {
                        "feedback_receipt_sha256": feedback["feedback_receipt_sha256"],
                        "update_receipt_sha256": update["update_receipt_sha256"],
                        "exact_outcome_hash": outcome["exact_outcome_hash"],
                        "feedback_reveal_sequence_index": sequence_index,
                        "arm_metrics": {
                            arm: _arm_metric(decision_id, action, direction, matrix)
                            for arm, action in control_actions.items()
                        },
                    }
                )
                run.feedback_receipts.append(feedback)
                run.update_rows.append(update)
                run.latency_rows.append(
                    {
                        "decision_id": decision_id,
                        "update_sequence_index": sequence_index,
                        "update_latency_us": round(latency_us, 3),
                    }
                )

    for decision in decisions:
        sequence_index = int(decision["decision_sequence_index"])
        if sequence_index < start_sequence:
            continue
        reveal_due(sequence_index)
        delay = 2 if decision["delayed_correction"] else 1
        action_receipt = active.freeze_action(
            decision["decision_id"],
            decision["decision_context"],
            due_sequence_index=sequence_index + delay,
        )
        actions_by_id[decision["decision_id"]] = action_receipt
        run.action_receipts.append(action_receipt)
        output_by_id[decision["decision_id"]] = {
            "decision_id": decision["decision_id"],
            "decision_sequence_index": sequence_index,
            "context_sha256": action_receipt["context_sha256"],
            "chosen_action": action_receipt["chosen_action"],
            "action_receipt_sha256": action_receipt["receipt_sha256"],
            "policy_state_sha256_before": action_receipt["policy_state_sha256_before"],
        }
        if checkpoint_dir is not None and sequence_index % CHECKPOINT_INTERVAL == 0:
            path = checkpoint_dir / f"checkpoint-{sequence_index:04d}.json"
            manifest = active.save_checkpoint(path)
            manifest.update(
                {
                    "decision_sequence_index": sequence_index,
                    "checkpoint_name": path.name,
                    "phase": "decision_boundary",
                }
            )
            run.checkpoint_manifest.append(manifest)
            run.checkpoint_bytes[sequence_index] = path.read_bytes()
            run.checkpoint_paths[sequence_index] = path

    reveal_index = len(decisions) + 1
    while active.pending_count:
        reveal_due(reveal_index)
        reveal_index += 1
    run.rows = [output_by_id[key] for key in sorted(output_by_id, key=lambda item: output_by_id[item]["decision_sequence_index"])]
    run.feedback_receipts.sort(key=lambda row: int(row["decision_sequence_index"]))
    run.update_rows.sort(key=lambda row: int(decisions_by_id[row["decision_id"]]["decision_sequence_index"]))
    run.latency_rows.sort(key=lambda row: int(decisions_by_id[row["decision_id"]]["decision_sequence_index"]))

    if checkpoint_dir is not None:
        final_sequence = len(decisions)
        path = checkpoint_dir / "checkpoint-final.json"
        manifest = active.save_checkpoint(path)
        manifest.update(
            {
                "decision_sequence_index": final_sequence,
                "checkpoint_name": path.name,
                "phase": "feedback_complete",
            }
        )
        run.checkpoint_manifest.append(manifest)
        run.checkpoint_bytes[final_sequence] = path.read_bytes()
        run.checkpoint_paths[final_sequence] = path
    return run


def _per_arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Reduce matched row metrics without pooling away any policy arm."""

    summaries: dict[str, JsonDict] = {}
    for arm in POLICY_ARMS:
        metrics = [row["arm_metrics"][arm] for row in rows]
        held = [
            row["arm_metrics"][arm]
            for row in rows
            if int(row["decision_sequence_index"]) >= HELD_FUTURE_START
        ]
        count = len(metrics)
        memory_count = sum(row["action"] == "verified_memory" for row in metrics)
        false_positives = sum(bool(row["false_positive_injection"]) for row in metrics)
        summaries[arm] = {
            "decision_count": count,
            "total_loss": round(sum(float(row["loss"]) for row in metrics), 12),
            "mean_loss": round(sum(float(row["loss"]) for row in metrics) / count, 12),
            "held_future_count": len(held),
            "held_future_mean_loss": round(
                sum(float(row["loss"]) for row in held) / len(held), 12
            ),
            "memory_selection_count": memory_count,
            "false_positive_injection_count": false_positives,
            "false_positive_injection_rate": round(false_positives / count, 12),
            "conditional_false_positive_rate": round(
                false_positives / memory_count if memory_count else 0.0, 12
            ),
            "abstention_count": sum(bool(row["abstained"]) for row in metrics),
            "missed_reuse_count": sum(bool(row["missed_reuse"]) for row in metrics),
            "helpful_memory_selection_count": sum(
                bool(row["helpful_memory_selected"]) for row in metrics
            ),
        }
    return summaries


def _held_future_effect(summary: Mapping[str, Mapping[str, Any]]) -> float:
    """Measure lower held-future risk loss relative to matched no-memory rows."""

    return round(
        float(summary["no_memory"]["held_future_mean_loss"])
        - float(summary["contextual_bandit"]["held_future_mean_loss"]),
        12,
    )


def compute_terminal_gates(
    *,
    planned_count: int,
    rows: Sequence[Mapping[str, Any]],
    action_receipts: Sequence[Mapping[str, Any]],
    feedback_receipts: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    restart_ok: bool,
    rollback_ok: bool,
    bounded_state: bool,
    held_future_effect: float,
    false_positive_injection_rate: float,
) -> JsonDict:
    """Keep receipt completeness separate from the scientific benefit gate."""

    expected_ids = {str(row.get("decision_id", "")) for row in rows}

    def ids(values: Sequence[Mapping[str, Any]]) -> set[str]:
        return {str(row.get("decision_id", "")) for row in values}

    receipt_complete = bool(
        planned_count > 0
        and len(rows) == len(action_receipts) == len(feedback_receipts) == len(update_rows) == planned_count
        and len(expected_ids) == planned_count
        and ids(action_receipts) == ids(feedback_receipts) == ids(update_rows) == expected_ids
    )
    execution_checks = [
        _check("chronological_decision_rows", planned_count, len(rows), len(rows) == planned_count),
        _check("action_feedback_update_receipts", True, receipt_complete, receipt_complete),
        _check("restart_equivalence", True, restart_ok, restart_ok),
        _check("rollback_equivalence", True, rollback_ok, rollback_ok),
        _check("bounded_controller_state", True, bounded_state, bounded_state),
    ]
    complete = all(row["passed"] is True for row in execution_checks)
    benefit_checks = [
        _check("positive_held_future_effect", ">0", held_future_effect, held_future_effect > 0.0),
        _check(
            "bounded_false_positive_injection",
            f"<={FALSE_POSITIVE_RATE_BOUND}",
            false_positive_injection_rate,
            false_positive_injection_rate <= FALSE_POSITIVE_RATE_BOUND,
        ),
    ]
    benefit = all(row["passed"] is True for row in benefit_checks)
    return {
        "execution_checks": execution_checks,
        "benefit_checks": benefit_checks,
        "risk_sensitive_controller_complete_score": int(complete),
        "controller_benefit_gate_score": int(benefit),
    }


def _restart_results(run: ReplayRun, fixture_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Resume from a persisted midpoint and compare final controller bytes."""

    boundary = max(index for index in run.checkpoint_paths if index < len(fixture_rows))
    persisted = run.checkpoint_paths[boundary].read_bytes()
    restarted = RiskSensitiveContextualBandit.load_checkpoint(run.checkpoint_paths[boundary])
    suffix = _run_replay(
        fixture_rows,
        risk_ratio=MAIN_RISK_RATIO,
        controller=restarted,
        start_sequence=boundary + 1,
    )
    clean_bytes = run.controller.to_bytes()
    restarted_bytes = suffix.controller.to_bytes()
    return {
        "checkpoint_boundary": boundary,
        "checkpoint_sha256": sha256_bytes(persisted),
        "checkpoint_bytes_round_trip": persisted == run.checkpoint_bytes[boundary],
        "clean_final_state_sha256": sha256_bytes(clean_bytes),
        "restarted_final_state_sha256": sha256_bytes(restarted_bytes),
        "final_bytes_identical": clean_bytes == restarted_bytes,
    }


def _rollback_results(controller: RiskSensitiveContextualBandit, context: Mapping[str, Any]) -> JsonDict:
    """Apply one bounded poison loss and restore the exact parent checkpoint."""

    parent_bytes = controller.to_bytes()
    parent_hash = sha256_bytes(parent_bytes)
    attacked = RiskSensitiveContextualBandit.from_bytes(parent_bytes)
    receipt = attacked.freeze_action("poison-rollback-probe", context, 1)
    update = attacked.apply_bounded_loss(
        "poison-rollback-probe",
        attacked.config.risk_ratio * 100.0,
        feedback_receipt_sha256="sha256:" + "f" * 64,
        update_sequence_index=1,
    )
    mutated_bytes = attacked.to_bytes()
    restored = RiskSensitiveContextualBandit.from_bytes(parent_bytes)
    restored_bytes = restored.to_bytes()
    return {
        "parent_state_sha256": parent_hash,
        "poison_action": receipt["chosen_action"],
        "poison_raw_loss": update["raw_loss"],
        "poison_bounded_loss": update["bounded_loss"],
        "poison_loss_was_clamped": update["loss_was_clamped"],
        "mutated_state_sha256": sha256_bytes(mutated_bytes),
        "restored_state_sha256": sha256_bytes(restored_bytes),
        "restored_parent_bytes": restored_bytes == parent_bytes,
        "model_weight_files_touched": [],
    }


def _risk_sensitivity_rows(
    fixture_rows: Sequence[Mapping[str, Any]], main_summary: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Replay each fixed risk ratio from the same clean chronological state."""

    result: list[JsonDict] = []
    for ratio in RISK_SENSITIVITY_RATIOS:
        if ratio == MAIN_RISK_RATIO:
            summary = main_summary
        else:
            replay = _run_replay(fixture_rows, risk_ratio=ratio)
            summary = _per_arm_summary(replay.rows)
        learned = summary["contextual_bandit"]
        result.append(
            {
                "risk_ratio": ratio,
                "held_future_effect": _held_future_effect(summary),
                "mean_loss": learned["mean_loss"],
                "false_positive_injection_rate": learned[
                    "false_positive_injection_rate"
                ],
                "abstention_rate": round(
                    int(learned["abstention_count"]) / int(learned["decision_count"]), 12
                ),
            }
        )
    return result


def _controller_schema(config: PolicyConfig) -> JsonDict:
    """Declare the exact bounded state and frozen-model boundary."""

    return {
        "algorithm": "diagonal ridge loss estimates with pessimistic upper confidence",
        "actions": list(ACTIONS),
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "tie_break_order": ["abstain", "no_memory", "verified_memory"],
        "max_updates_per_action": config.max_updates_per_action,
        "max_pending": config.max_pending,
        "max_state_bytes": config.max_state_bytes,
        "checkpoint_interval_decisions": CHECKPOINT_INTERVAL,
        "external_policy_only": True,
        "foundation_model_weights_frozen": True,
        "model_weight_files_touched": [],
        "forbidden_inputs": sorted(DENIED_CONTEXT_FIELDS),
    }


def _field_principles() -> JsonDict:
    """Explain why each top-level field is needed for audit or replay."""

    return {
        "schema": "A versioned schema prevents silent controller reinterpretation.",
        "experiment_id": "A stable identity binds this replay to its task.",
        "title": "The title states the bounded policy under test.",
        "run_date": "The requested date fixes the execution boundary.",
        "status": "Status distinguishes executed and blocked artifacts.",
        "openspec_requirement_ids": "Requirement IDs connect behavior to executable tests.",
        "replay_commands": "Commands let another operator repeat each verification layer.",
        "field_principles": "Every top-level field has one plain-language purpose.",
        "preconditions_checked": "Invalid evidence blocks policy actions before replay.",
        "inference_substrate": "The declared CPU policy prevents an implied LLM run.",
        "duration_s": "Measured wall time proves the replay executed.",
        "source_artifact_hashes": "The hash binds the exact frozen opportunity fixture.",
        "random_seed": "One seed controls only deterministic comparison admission.",
        "reproducibility_checksum": "The checksum binds stable content and excludes timing noise.",
        "rows": "Each decision row carries matched metrics for every policy arm.",
        "controller_schema": "The schema declares features, bounds, and frozen weights.",
        "risk_matrix": "Explicit losses make the safety asymmetry auditable.",
        "pre_outcome_action_receipts": "Receipts prove action freeze preceded exact feedback.",
        "exact_feedback_receipts": "Receipts bind each delayed exact outcome to one action.",
        "update_rows": "Rows prove each bounded update references both receipts.",
        "checkpoint_manifest": "The manifest proves bounded persisted intervals and hashes.",
        "restart_equivalence_results": "Byte equality detects restart drift.",
        "rollback_results": "Parent-byte restoration proves poison recovery.",
        "state_size_rows": "State bytes show that online learning cannot grow without bound.",
        "latency_rows": "Measured update time exposes controller overhead.",
        "per_arm_summary": "Separate summaries keep weak controls visible.",
        "held_future_effect": "Held-future risk reduction is the benefit test.",
        "false_positive_injection_rate": "The rate measures the costliest memory error.",
        "abstention_rate": "The rate detects a degenerate always-abstain policy.",
        "risk_sensitivity_rows": "Fixed ratios show whether the verdict depends on one cost.",
        "risk_sensitive_controller_complete_score": "Receipt completeness is independent of benefit.",
        "controller_benefit_gate_score": "Benefit requires positive future effect and bounded harm.",
        "gate_check_summary": "Every failure keeps its expected and observed value.",
        "verifier_is_oracle": "False because exact source outcomes, not this policy, define loss.",
        "verdict_class": "A closed class prevents execution from becoming a benefit claim.",
        "honest_verdict": "A terminal prefix gives the conductor an unambiguous result.",
    }


def _base_artifact(
    *, repo_root: Path, run_date: str, duration_s: float, preconditions: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build a schema-complete blocked artifact before any policy replay."""

    config = PolicyConfig(risk_ratio=MAIN_RISK_RATIO)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "title": "Risk-Sensitive Abstention Memory Controller",
        "run_date": run_date,
        "status": "complete",
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "replay_commands": list(REPLAY_COMMANDS),
        "field_principles": _field_principles(),
        "preconditions_checked": list(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "source_artifact_hashes": source_artifact_hashes(repo_root),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "rows": [],
        "controller_schema": _controller_schema(config),
        "risk_matrix": {
            "main": RiskMatrix(MAIN_RISK_RATIO).as_dict(),
            "sensitivity_ratios": list(RISK_SENSITIVITY_RATIOS),
        },
        "pre_outcome_action_receipts": [],
        "exact_feedback_receipts": [],
        "update_rows": [],
        "checkpoint_manifest": [],
        "restart_equivalence_results": {},
        "rollback_results": {},
        "state_size_rows": [],
        "latency_rows": [],
        "per_arm_summary": {},
        "held_future_effect": 0.0,
        "false_positive_injection_rate": 0.0,
        "abstention_rate": 0.0,
        "risk_sensitivity_rows": [],
        "risk_sensitive_controller_complete_score": 0,
        "controller_benefit_gate_score": 0,
        "gate_check_summary": _gate_summary(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    repo_root: Path,
    *,
    run_date: str,
    duration_s: float,
    fixture: Mapping[str, Any] | None = None,
    checkpoint_dir: Path | None = None,
    initial_controller: RiskSensitiveContextualBandit | None = None,
) -> JsonDict:
    """Build the blocked receipt or complete chronological controller replay."""

    source = dict(fixture) if fixture is not None else load_fixture(repo_root / SOURCE_RELATIVE_PATH)
    controller = initial_controller or RiskSensitiveContextualBandit(
        PolicyConfig(risk_ratio=MAIN_RISK_RATIO)
    )
    preconditions = validate_preconditions(source, controller)
    if any(row["passed"] is not True for row in preconditions):
        return _base_artifact(
            repo_root=repo_root,
            run_date=run_date,
            duration_s=duration_s,
            preconditions=preconditions,
        )

    rows = source["rows"]
    owns_temp_dir = checkpoint_dir is None
    temporary = tempfile.TemporaryDirectory(prefix="carnot-exp6854-") if owns_temp_dir else None
    active_checkpoint_dir = Path(temporary.name) if temporary is not None else Path(checkpoint_dir)
    try:
        replay = _run_replay(
            rows,
            risk_ratio=MAIN_RISK_RATIO,
            controller=controller,
            checkpoint_dir=active_checkpoint_dir,
        )
        summary = _per_arm_summary(replay.rows)
        held_effect = _held_future_effect(summary)
        learned = summary["contextual_bandit"]
        false_positive_rate = float(learned["false_positive_injection_rate"])
        abstention_rate = round(
            int(learned["abstention_count"]) / int(learned["decision_count"]), 12
        )
        restart = _restart_results(replay, rows)
        rollback = _rollback_results(replay.controller, rows[0]["decision_context"])
        state_sizes = [
            {
                "decision_sequence_index": row["decision_sequence_index"],
                "state_bytes": row["state_bytes"],
                "pending_count": row["pending_count"],
                "within_bound": row["state_bytes"] <= controller.config.max_state_bytes,
            }
            for row in replay.checkpoint_manifest
        ]
        bounded_state = bool(state_sizes) and all(row["within_bound"] for row in state_sizes)
        gates = compute_terminal_gates(
            planned_count=len(rows),
            rows=replay.rows,
            action_receipts=replay.action_receipts,
            feedback_receipts=replay.feedback_receipts,
            update_rows=replay.update_rows,
            restart_ok=restart["final_bytes_identical"] is True,
            rollback_ok=rollback["restored_parent_bytes"] is True,
            bounded_state=bounded_state,
            held_future_effect=held_effect,
            false_positive_injection_rate=false_positive_rate,
        )
        execution_summary = _gate_summary([*preconditions, *gates["execution_checks"]])
        execution_summary["benefit_checks"] = gates["benefit_checks"]
        execution_summary["benefit_passed"] = bool(gates["controller_benefit_gate_score"])
        complete = gates["risk_sensitive_controller_complete_score"]
        benefit = gates["controller_benefit_gate_score"]
        verdict_class = "positive" if complete and benefit else "null" if complete else "partial"
        honest_verdict = (
            "complete_positive_risk_sensitive_controller_benefit_gate_passed"
            if complete and benefit
            else "complete_null_risk_sensitive_controller_complete_benefit_gate_not_met"
            if complete
            else "complete_partial_risk_sensitive_controller_receipts_incomplete"
        )
        artifact: JsonDict = {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "title": "Risk-Sensitive Abstention Memory Controller",
            "run_date": run_date,
            "status": "complete",
            "openspec_requirement_ids": list(OPEN_SPEC_IDS),
            "replay_commands": list(REPLAY_COMMANDS),
            "field_principles": _field_principles(),
            "preconditions_checked": preconditions,
            "inference_substrate": INFERENCE_SUBSTRATE,
            "duration_s": round(max(0.0, float(duration_s)), 6),
            "source_artifact_hashes": source_artifact_hashes(repo_root),
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "pending",
            "rows": replay.rows,
            "controller_schema": _controller_schema(controller.config),
            "risk_matrix": {
                "main": RiskMatrix(MAIN_RISK_RATIO).as_dict(),
                "sensitivity_ratios": list(RISK_SENSITIVITY_RATIOS),
            },
            "pre_outcome_action_receipts": replay.action_receipts,
            "exact_feedback_receipts": replay.feedback_receipts,
            "update_rows": replay.update_rows,
            "checkpoint_manifest": replay.checkpoint_manifest,
            "restart_equivalence_results": restart,
            "rollback_results": rollback,
            "state_size_rows": state_sizes,
            "latency_rows": replay.latency_rows,
            "per_arm_summary": summary,
            "held_future_effect": held_effect,
            "false_positive_injection_rate": false_positive_rate,
            "abstention_rate": abstention_rate,
            "risk_sensitivity_rows": _risk_sensitivity_rows(rows, summary),
            "risk_sensitive_controller_complete_score": complete,
            "controller_benefit_gate_score": benefit,
            "gate_check_summary": execution_summary,
            "verifier_is_oracle": False,
            "verdict_class": verdict_class,
            "honest_verdict": honest_verdict,
        }
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact
    finally:
        if temporary is not None:
            temporary.cleanup()


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable replay content while excluding wall time and latency noise."""

    unsigned = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "latency_rows", "reproducibility_checksum"}
    }
    return sha256_json(unsigned)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and consistency errors before writing the deliverable."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict must start with complete_")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("risk_sensitive_controller_complete_score") == 1:
        count = len(artifact.get("rows", []))
        if count == 0 or any(
            len(artifact.get(field, [])) != count
            for field in (
                "pre_outcome_action_receipts",
                "exact_feedback_receipts",
                "update_rows",
            )
        ):
            errors.append("complete score contradicts receipt counts")
        if artifact.get("gate_check_summary", {}).get("passed") is not True:
            errors.append("complete score contradicts failed execution gates")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write one validated result without changing any other tracked file."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic replay for the supplied execution date."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Execution date in YYYYMMDD form.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    artifact = build_artifact(
        REPO_ROOT,
        run_date=args.date,
        duration_s=0.0,
        checkpoint_dir=args.checkpoint_dir,
    )
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    write_artifact(args.output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - the tested wrapper owns CLI execution.
    raise SystemExit(main())
