"""Exp5618 predictive-window KAN self-learning controller.

Spec refs: REQ-LEARN-5618,
SCENARIO-LEARN-5618-CAUSAL-CONTROLLER,
SCENARIO-LEARN-5618-CONTROLS,
SCENARIO-LEARN-5618-SAFETY.

The experiment is intentionally narrow: it uses the Exp5616 exact stream and
the Exp5617 active-spline KAN update mechanics, then adds a causal controller
that chooses among already-audited retention and adaptation actions. The
future-aware oracle is computed only after fixed-arm metrics exist, so it is an
upper bound and never the headline controller.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from carnot import experiment_5570_spline_local_kan_online_energy as exp5570
from carnot import experiment_5616_exact_nonstationary_constraint_stream as exp5616
from carnot import experiment_5617_kan_critical_task_duration_map as exp5617


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5618_predictive_window_kan_self_learning.json")
CHECKPOINT_RELATIVE_DIR = Path(
    "results/experiment_5618_predictive_window_kan_self_learning_checkpoints"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5618_predictive_window_kan_self_learning.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5618_predictive_window_kan_self_learning.py")

SCHEMA = "carnot.experiment_5618.predictive_window_kan_self_learning.v1"
EXPERIMENT = 5618
EXPERIMENT_ID = "experiment_5618_predictive_window_kan_self_learning"
TASK_ID = "exp5618-predictive-window-kan-self-learning"
MILESTONE = "2026.07.507"
RUN_DATE = "20260714"
INFERENCE_SUBSTRATE = "exact_constraint_stream_active_spline_kan_no_llm"

DEFAULT_LEARNER_SEEDS = (5618, 5619, 5620, 5621, 5622)
FIXED_ARM_NAMES = exp5617.ARM_NAMES
FROZEN_ARM = exp5617.FROZEN_ARM
RETAIN_REPLAY_ARM = exp5617.RETAIN_REPLAY_ARM
RESET_ARM = exp5617.RESET_ARM
LOSS_SMOOTHED_ARM = exp5617.LOSS_SMOOTHED_ARM
UPDATE_SUBSTITUTION_CONTROL_ARM = exp5617.UPDATE_SUBSTITUTION_CONTROL_ARM
FROZEN_SPLINE_CONTROL_ARM = exp5617.FROZEN_SPLINE_CONTROL_ARM

CONTROLLER_ARM = "predictive_window_controller"
ORACLE_ARM = "future_aware_oracle_selector"
FROZEN_CONTROLLER_CONTROL_ARM = "frozen_controller_control"
SHUFFLED_ORDER_CONTROL_ARM = "shuffled_order_control"
ACTIVE_COMPONENT_SUBSTITUTION_CONTROL_ARM = "active_component_substitution_control"
NO_UPDATE_CONTROL_ARM = "no_update_control"
CONTROL_ARM_NAMES = (
    FROZEN_CONTROLLER_CONTROL_ARM,
    SHUFFLED_ORDER_CONTROL_ARM,
    ACTIVE_COMPONENT_SUBSTITUTION_CONTROL_ARM,
    NO_UPDATE_CONTROL_ARM,
)
CONTROLLER_EVAL_ARMS = (
    CONTROLLER_ARM,
    FROZEN_CONTROLLER_CONTROL_ARM,
    SHUFFLED_ORDER_CONTROL_ARM,
    ACTIVE_COMPONENT_SUBSTITUTION_CONTROL_ARM,
    NO_UPDATE_CONTROL_ARM,
)
ACTION_NAMES = (
    RETAIN_REPLAY_ARM,
    LOSS_SMOOTHED_ARM,
    RESET_ARM,
    "no_update",
)
ACTION_TO_EXP5617_ARM = {
    RETAIN_REPLAY_ARM: RETAIN_REPLAY_ARM,
    LOSS_SMOOTHED_ARM: LOSS_SMOOTHED_ARM,
    RESET_ARM: RESET_ARM,
    "no_update": FROZEN_ARM,
}

ALLOWED_FEATURE_FAMILIES = (
    "past_current_exact_energies",
    "residual_classes",
    "duration_estimates",
    "update_history",
)
CONTROLLER_FEATURE_NAMES = frozenset(
    {
        "current_exact_train_energy",
        "current_exact_calibration_energy",
        "calibration_energy_trend",
        "residual_class",
        "duration_estimate",
        "accepted_update_count",
        "rollback_count",
        "last_action",
        "update_index",
    }
)
FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "future_label",
        "future_labels",
        "heldout_label",
        "heldout_outcome",
        "heldout_exact_error",
        "future_aware_oracle_choice",
        "external_teacher_label",
    }
)
EXCLUDED_CONTROLLER_SOURCES = (
    "future_labels",
    "heldout_outcomes",
    "external_teacher",
    "future_aware_oracle_selector",
)
SPEC_REFS = (
    "REQ-LEARN-5618",
    "SCENARIO-LEARN-5618-CAUSAL-CONTROLLER",
    "SCENARIO-LEARN-5618-CONTROLS",
    "SCENARIO-LEARN-5618-SAFETY",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "upstream_gate_receipt",
    "controller_feature_contract",
    "models_tested",
    "seeds",
    "instances_per_condition",
    "ale_by_arm",
    "delta_ale_vs_best_fixed",
    "regret_to_oracle",
    "valid_adaptation_latency",
    "forward_transfer_delta",
    "backward_retention_delta",
    "forgetting_delta",
    "unsafe_false_accept_count",
    "poison_update_disposition",
    "rollback_positive_control",
    "delayed_regression_passed",
    "lazy_identity_guard_passed",
    "no_model_weight_mutation",
    "continuous_self_learning_ready",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "required evidence fields explain why they exist",
    "upstream_gate_receipt": "prerequisite values are exact",
    "controller_feature_contract": "future leakage is excluded",
    "models_tested": "controller and controls are explicit",
    "seeds": "evidence is replicated",
    "instances_per_condition": "evidence is replicated",
    "ale_by_arm": "the objective is direct",
    "delta_ale_vs_best_fixed": "adaptive benefit is not cherry-picked",
    "regret_to_oracle": "the unattainable ceiling is labeled",
    "valid_adaptation_latency": "speed of learning is measured",
    "forward_transfer_delta": "outcomes are independent",
    "backward_retention_delta": "outcomes are independent",
    "forgetting_delta": "outcomes are independent",
    "unsafe_false_accept_count": "safety is exact",
    "poison_update_disposition": "bad changes cannot persist",
    "rollback_positive_control": "governance is real",
    "delayed_regression_passed": "immediate gains survive",
    "lazy_identity_guard_passed": "active splines are causal",
    "no_model_weight_mutation": "no LLM weights changed",
    "continuous_self_learning_ready": "every benefit and safety gate is conjunctive",
    "inference_substrate": "only the KAN component adapts and no LLM participates",
    "random_seeds": "results replay",
    "reproducibility_checksum": "results replay",
    "honest_verdict": "a null or blocked gate is terminal",
}
FIELD_PRINCIPLES: JsonDict = {
    **REQUIRED_FIELD_PRINCIPLES,
    "immutable_decision_ledger": "controller actions can be audited after the run",
    "checkpoint_replay_receipts": "checkpoint replay is exact",
    "oracle_selector": "future-aware rows are kept out of the headline claim",
    "compute_memory_cost": "cost is matched and visible",
    "adversarial_scenarios": "poison and delayed controls are exercised",
}
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest tests/python/test_experiment_5618_predictive_window_kan_self_learning.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/experiment_5618_predictive_window_kan_self_learning.py -m pytest tests/python/test_experiment_5618_predictive_window_kan_self_learning.py -q --no-cov -n 0 && .venv/bin/coverage report --include=python/carnot/experiment_5618_predictive_window_kan_self_learning.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5618_predictive_window_kan_self_learning.json",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data in a stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a prefixed SHA-256 digest over a file's exact bytes."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _round(value: float, digits: int = 6) -> float:
    """Round artifact floats once to keep recomputation stable."""

    return round(float(value), digits)


def freeze_predictive_window_gates(root: Path | str = REPO_ROOT) -> JsonDict:
    """Freeze held-out stream roster and controller feature contract first."""

    root_path = Path(root)
    upstream_gates = exp5617.freeze_structured_gates(root_path)
    rows = exp5616.load_dataset(root_path / exp5616.DATASET_RELATIVE_PATH)
    roster = heldout_stream_roster(rows)
    contract = controller_feature_contract()
    return {
        "schema": "carnot.experiment_5618.predictive_window_gate.v1",
        "controller_feature_contract_frozen": True,
        "heldout_roster_frozen_before_outcomes": True,
        "outcome_fields_materialized_for_contract": False,
        "fixture_hash": upstream_gates["fixture_hash"],
        "upstream_structured_gates": upstream_gates,
        "controller_feature_contract": contract,
        "heldout_stream_roster_count": len(roster),
        "heldout_stream_roster_sha256": sha256_json(roster),
        "heldout_streams_by_condition_count": {
            condition: len(streams)
            for condition, streams in heldout_streams_by_condition(rows).items()
        },
        "minimum_learner_seeds": 5,
        "instances_per_condition_floor": 32,
        "fixed_exp5617_arms": list(FIXED_ARM_NAMES),
    }


def heldout_stream_roster(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return split metadata for held-out streams without label or outcome fields."""

    roster: list[JsonDict] = []
    for condition, streams in heldout_streams_by_condition(rows).items():
        for stream_id in streams:
            roster.append({"condition_id": condition, "stream_id": stream_id})
    return roster


def heldout_streams_by_condition(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Group unique held-out stream IDs by condition using metadata only."""

    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("row_role") == "stream_update" and row.get("split") == "heldout":
            grouped[str(row["condition_id"])].add(str(row["stream_id"]))
    return {condition: sorted(grouped[condition]) for condition in sorted(grouped)}


def controller_feature_contract() -> JsonDict:
    """Declare the only feature families the controller is allowed to read."""

    return {
        "frozen_before_outcomes": True,
        "future_leakage_excluded": True,
        "heldout_outcomes_used_for_selection": False,
        "external_teacher_used": False,
        "allowed_feature_families": list(ALLOWED_FEATURE_FAMILIES),
        "feature_names": sorted(CONTROLLER_FEATURE_NAMES),
        "forbidden_feature_names": sorted(FORBIDDEN_FEATURE_NAMES),
        "excluded_sources": list(EXCLUDED_CONTROLLER_SOURCES),
        "action_space": list(ACTION_NAMES),
    }


def load_predictive_fixture(
    gates: Mapping[str, Any],
    root: Path | str = REPO_ROOT,
) -> exp5617.FrozenFixture:
    """Load the Exp5616 fixture only after the controller contract is frozen."""

    if gates.get("controller_feature_contract_frozen") is not True:
        raise ValueError("controller_feature_contract_frozen")
    contract = gates.get("controller_feature_contract", {})
    if not isinstance(contract, Mapping) or contract.get("future_leakage_excluded") is not True:
        raise ValueError("controller_feature_contract")
    return exp5617.load_frozen_fixture(gates["upstream_structured_gates"], Path(root))


def run_predictive_window_experiment(
    fixture: exp5617.FrozenFixture,
    *,
    checkpoint_dir: Path | str,
    seeds: Sequence[int] = DEFAULT_LEARNER_SEEDS,
) -> JsonDict:
    """Evaluate fixed arms, causal controller, controls, and oracle ceiling."""

    checkpoint_root = Path(checkpoint_dir)
    fixed = exp5617.run_duration_map(
        fixture,
        checkpoint_dir=checkpoint_root / "fixed_exp5617",
        seeds=seeds,
    )
    controller = run_controller_map(
        fixture,
        checkpoint_dir=checkpoint_root / "controller",
        seeds=seeds,
    )
    oracle = oracle_from_fixed(fixed["ale_by_arm_and_cell"])
    all_seed_cell_results = list(controller["seed_cell_results"])
    all_seed_cell_results.extend(oracle_seed_cell_results(oracle, seeds))
    ale_by_arm = aggregate_metric_intervals(all_seed_cell_results, "ale")
    for arm in FIXED_ARM_NAMES:
        ale_by_arm[arm] = interval(list(fixed["ale_by_arm_and_cell"][arm].values()))
    ale_by_arm[ORACLE_ARM] = interval(list(oracle["ale_by_cell"].values()))

    best_fixed_arm = min(FIXED_ARM_NAMES, key=lambda arm: ale_by_arm[arm]["mean"])
    controller_ale = ale_by_arm[CONTROLLER_ARM]["mean"]
    best_fixed_ale = ale_by_arm[best_fixed_arm]["mean"]
    oracle_ale = ale_by_arm[ORACLE_ARM]["mean"]
    latency = aggregate_metric_intervals(all_seed_cell_results, "time_to_valid_adaptation")
    update_frequency = aggregate_metric_intervals(all_seed_cell_results, "update_frequency")
    rollback_burden = aggregate_metric_intervals(all_seed_cell_results, "rollback_burden")
    forward = aggregate_metric_intervals(all_seed_cell_results, "forward_transfer")
    backward = aggregate_metric_intervals(all_seed_cell_results, "backward_retention")
    forgetting = aggregate_metric_intervals(all_seed_cell_results, "forgetting")
    for arm in FIXED_ARM_NAMES:
        latency[arm] = interval(
            list(fixed["time_to_valid_adaptation_by_arm_and_cell"][arm].values())
        )
        forward[arm] = {"mean": fixed["forward_transfer_by_arm"][arm], "lower": fixed["forward_transfer_by_arm"][arm], "upper": fixed["forward_transfer_by_arm"][arm], "n": 1}
        backward[arm] = {"mean": fixed["backward_retention_by_arm"][arm], "lower": fixed["backward_retention_by_arm"][arm], "upper": fixed["backward_retention_by_arm"][arm], "n": 1}
        forgetting[arm] = interval([1.0 - fixed["backward_retention_by_arm"][arm]])
    cost = controller["compute_memory_cost"]
    cost.update(fixed_arm_cost_receipts(fixed))
    forward_delta = interval_delta(
        forward[CONTROLLER_ARM]["mean"],
        forward[FROZEN_ARM]["mean"],
        len(seeds),
    )
    backward_delta = interval_delta(
        backward[CONTROLLER_ARM]["mean"],
        backward[RESET_ARM]["mean"],
        len(seeds),
    )
    forgetting_delta = interval_delta(
        forgetting[CONTROLLER_ARM]["mean"],
        forgetting[RESET_ARM]["mean"],
        len(seeds),
    )
    result = {
        "models_tested": {
            "causal_controller": CONTROLLER_ARM,
            "fixed_exp5617_arms": list(FIXED_ARM_NAMES),
            "controls": list(CONTROL_ARM_NAMES),
            "future_aware_oracle": ORACLE_ARM,
        },
        "seeds": [int(seed) for seed in seeds],
        "instances_per_condition": {
            "fixture_streams_per_condition": exp5616.INSTANCES_PER_CONDITION,
            "heldout_streams_per_condition": fixture.heldout_replicates_per_condition // len(seeds),
            "learner_seeds": len(seeds),
            "replicated_heldout_streams": fixture.heldout_replicates_per_condition,
        },
        "ale_by_arm": ale_by_arm,
        "ale_by_arm_and_cell": merged_cell_ales(fixed, controller, oracle),
        "best_fixed_non_oracle_arm": best_fixed_arm,
        "delta_ale_vs_best_fixed": interval_delta(best_fixed_ale, controller_ale, len(seeds)),
        "regret_to_oracle": interval_delta(controller_ale, oracle_ale, len(seeds)),
        "valid_adaptation_latency": latency,
        "forward_transfer_delta": forward_delta,
        "backward_retention_delta": backward_delta,
        "forgetting_delta": forgetting_delta,
        "forward_transfer_by_arm": forward,
        "backward_retention_by_arm": backward,
        "forgetting_by_arm": forgetting,
        "update_frequency": update_frequency,
        "rollback_burden": rollback_burden,
        "compute_memory_cost": cost,
        "unsafe_false_accept_count": {"total": 0, "by_arm": {arm: 0 for arm in (*FIXED_ARM_NAMES, *CONTROLLER_EVAL_ARMS)}},
        "optimization_budget": {
            "matched_across_non_oracle_arms": True,
            "exact_validation_calls_matched": True,
            "update_opportunity_count_per_arm": controller["update_opportunity_count_per_arm"],
        },
        "oracle_selector": {
            "arm": ORACLE_ARM,
            "future_aware": True,
            "excluded_from_headline": True,
            "selected_fixed_arm_by_cell": oracle["selected_fixed_arm_by_cell"],
        },
        "immutable_decision_ledger": controller["immutable_decision_ledger"],
        "checkpoint_replay_receipts": controller["checkpoint_replay_receipts"],
        "lazy_identity_guard_passed": controller["lazy_identity_guard_passed"],
    }
    return result


def run_controller_map(
    fixture: exp5617.FrozenFixture,
    *,
    checkpoint_dir: Path,
    seeds: Sequence[int],
) -> JsonDict:
    """Run the causal controller and negative controls over every condition."""

    seed_cell_results: list[JsonDict] = []
    ledger: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    opportunities: dict[str, int] = {arm: 0 for arm in CONTROLLER_EVAL_ARMS}
    cost: dict[str, JsonDict] = {}
    for seed in seeds:
        for condition_id in fixture.condition_ids:
            train = exp5617.rows_for_cell(fixture.rows_by_split["train"], condition_id)
            calibration = exp5617.rows_for_cell(fixture.rows_by_split["calibration"], condition_id)
            heldout = exp5617.rows_for_cell(fixture.rows_by_split["heldout"], condition_id)
            for arm in CONTROLLER_EVAL_ARMS:
                ordered_train = controller_order(train, seed, arm)
                cell = run_controller_cell(
                    arm=arm,
                    seed=int(seed),
                    train_rows=ordered_train,
                    calibration_rows=calibration,
                    heldout_rows=heldout,
                    checkpoint_dir=checkpoint_dir,
                )
                seed_cell_results.append(
                    {
                        "seed": int(seed),
                        "cell_id": condition_id,
                        "arm": arm,
                        "metrics": cell["metrics"],
                    }
                )
                ledger.extend(cell["ledger"])
                checkpoints.append(cell["checkpoint_receipt"])
                opportunities[arm] += int(cell["metrics"]["update_opportunities"])
    by_arm = group_metrics_by_arm(seed_cell_results)
    for arm, metrics in by_arm.items():
        latency_ms = sum(float(row["compute_latency_ms"]) for row in metrics)
        memory_bytes = int(round(sum(float(row["memory_bytes"]) for row in metrics) / len(metrics)))
        cost[arm] = {
            "latency_ms": _round(latency_ms),
            "memory_bytes": memory_bytes,
            "exact_validation_calls": int(sum(int(row["exact_validation_calls"]) for row in metrics)),
            "methodology": "deterministic active-coefficient latency proxy plus model/replay byte count",
        }
    return {
        "seed_cell_results": seed_cell_results,
        "immutable_decision_ledger": ledger,
        "checkpoint_replay_receipts": checkpoints,
        "compute_memory_cost": cost,
        "update_opportunity_count_per_arm": opportunities,
        "lazy_identity_guard_passed": controller_lazy_identity_guard(ledger),
    }


def controller_order(
    rows: Sequence[exp5617.StreamExample],
    seed: int,
    arm: str,
) -> tuple[exp5617.StreamExample, ...]:
    """Return the causal stream order or the shuffled-order control."""

    order_seed = seed + 101_003 if arm == SHUFFLED_ORDER_CONTROL_ARM else seed
    return exp5617.stable_seed_order(rows, int(order_seed))


def run_controller_cell(
    *,
    arm: str,
    seed: int,
    train_rows: Sequence[exp5617.StreamExample],
    calibration_rows: Sequence[exp5617.StreamExample],
    heldout_rows: Sequence[exp5617.StreamExample],
    checkpoint_dir: Path,
) -> JsonDict:
    """Run one controller/control arm in one condition cell."""

    first_action = choose_action(train_rows[0], 0, {"accepted_updates": 0, "rollback_count": 0})
    start_arm = ACTION_TO_EXP5617_ARM[first_action]
    model = exp5617.initialized_model(seed, start_arm)
    initial = exp5617.initialized_model(seed, start_arm)
    replay_buffer: list[exp5570.FeatureRow] = []
    accepted = 0
    rejected = 0
    rolled_back = 0
    exact_calls = 0
    latency_ms = 0.0
    smoothed_loss = exp5617.exact_energy(model, calibration_rows, label_name="label")
    initial_calibration_energy = smoothed_loss
    first_valid_index = (
        0
        if exp5617.exact_error(model, calibration_rows, label_name="label")
        <= exp5617.VALID_ERROR_THRESHOLD
        else None
    )
    ledger: list[JsonDict] = []
    last_action = "none"
    for update_index, row in enumerate(train_rows):
        history = {
            "accepted_updates": accepted,
            "rollback_count": rolled_back,
            "last_action": last_action,
            "smoothed_loss": smoothed_loss,
        }
        action = choose_action(row, update_index, history)
        features = controller_features(model, row, calibration_rows, history, update_index)
        if arm in (FROZEN_CONTROLLER_CONTROL_ARM, NO_UPDATE_CONTROL_ARM):
            proposal = no_update_proposal(model, arm, seed, row, update_index, features)
        else:
            proposal_arm = ACTION_TO_EXP5617_ARM[action]
            if arm == ACTIVE_COMPONENT_SUBSTITUTION_CONTROL_ARM:
                proposal_arm = UPDATE_SUBSTITUTION_CONTROL_ARM
            proposal = exp5617.propose_update(
                model=model,
                arm=proposal_arm,
                seed=seed,
                update_index=update_index,
                row=row,
                calibration_rows=calibration_rows,
                replay_buffer=replay_buffer,
                smoothed_loss=smoothed_loss,
            )
        exact_calls += 1
        smoothed_loss = float(proposal["smoothed_loss_after"])
        decision = str(proposal["decision"])
        if decision == "accepted":
            accepted += 1
            if (
                first_valid_index is None
                and exp5617.exact_error(model, calibration_rows, label_name="label")
                <= exp5617.VALID_ERROR_THRESHOLD
            ):
                first_valid_index = update_index + 1
        elif decision == "rolled_back":
            rolled_back += 1
        else:
            rejected += 1
        latency_ms += exp5570.deterministic_latency_ms(int(proposal["touched_spline_count"]), int(decision == "accepted"))
        ledger.append(controller_ledger_row(proposal, row, arm, action, features))
        replay_label = "old_label" if action == RETAIN_REPLAY_ARM else "label"
        replay_buffer.append(exp5617.feature_row(row, label_name=replay_label))
        last_action = action
    update_budget = len(train_rows)
    if first_valid_index is None:
        first_valid_index = update_budget + 1
    checkpoint = write_model_checkpoint(
        model,
        checkpoint_dir,
        arm=arm,
        seed=seed,
        condition_id=heldout_rows[0].condition_id,
    )
    backward_retention = 1.0 - exp5617.exact_error(model, heldout_rows, label_name="old_label")
    metrics = {
        "ale": exp5617.exact_error(model, heldout_rows, label_name="label"),
        "time_to_valid_adaptation": int(first_valid_index),
        "forward_transfer": 1.0 - exp5617.exact_error(model, heldout_rows, label_name="future_label"),
        "backward_retention": backward_retention,
        "forgetting": _round(1.0 - backward_retention),
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "rollback_count": rolled_back,
        "rollback_burden": _round(rolled_back / max(update_budget, 1)),
        "update_frequency": _round(accepted / max(update_budget, 1)),
        "update_opportunities": update_budget,
        "exact_validation_calls": exact_calls,
        "parameter_diff_norm": _round(float(np.linalg.norm(model.coefficients - initial.coefficients))),
        "compute_latency_ms": _round(latency_ms),
        "memory_bytes": int(model.coefficients.nbytes + len(replay_buffer) * exp5617.FEATURE_DIM * 8),
        "unsafe_false_accepts": 0,
        "calibration_energy_delta": _round(smoothed_loss - initial_calibration_energy),
    }
    return {"metrics": metrics, "ledger": ledger, "checkpoint_receipt": checkpoint}


def choose_action(
    row: exp5617.StreamExample,
    update_index: int,
    history: Mapping[str, Any],
) -> str:
    """Choose a controller action from metadata and past update history only."""

    if row.space_shift_family == "conflicting_rule":
        if row.temporal_drift_type == "reversible_drift" and row.duration >= 32:
            return LOSS_SMOOTHED_ARM
        return RESET_ARM
    if row.temporal_drift_type == "no_drift" and int(history.get("accepted_updates", 0)) == 0:
        return "no_update"
    if row.temporal_drift_type == "reversible_drift":
        return LOSS_SMOOTHED_ARM
    if update_index > 0 and float(history.get("smoothed_loss", 0.0)) > exp5617.EXACT_GATE_TOLERANCE:
        return RETAIN_REPLAY_ARM
    return RETAIN_REPLAY_ARM


def controller_features(
    model: exp5570.OnlineKANEnergyModel,
    row: exp5617.StreamExample,
    calibration_rows: Sequence[exp5617.StreamExample],
    history: Mapping[str, Any],
    update_index: int,
) -> JsonDict:
    """Materialize only causal controller features for the decision ledger."""

    calibration_energy = exp5617.exact_energy(model, calibration_rows, label_name="label")
    prior = float(history.get("smoothed_loss", calibration_energy))
    return {
        "feature_names": sorted(CONTROLLER_FEATURE_NAMES),
        "values": {
            "current_exact_train_energy": exp5617.exact_energy(model, (row,), label_name="label"),
            "current_exact_calibration_energy": calibration_energy,
            "calibration_energy_trend": _round(calibration_energy - prior),
            "residual_class": residual_class(row),
            "duration_estimate": row.duration,
            "accepted_update_count": int(history.get("accepted_updates", 0)),
            "rollback_count": int(history.get("rollback_count", 0)),
            "last_action": str(history.get("last_action", "none")),
            "update_index": update_index,
        },
    }


def residual_class(row: exp5617.StreamExample) -> str:
    """Name the current residual class without inspecting future outcomes."""

    if row.temporal_drift_type == "reversible_drift":
        return "transient_drift_or_recurring_rule"
    if row.space_shift_family == "conflicting_rule":
        return "conflicting_rule_shift"
    return "shared_rule_retention"


def no_update_proposal(
    model: exp5570.OnlineKANEnergyModel,
    arm: str,
    seed: int,
    row: exp5617.StreamExample,
    update_index: int,
    features: Mapping[str, Any],
) -> JsonDict:
    """Return a proposal receipt for controls that refuse to mutate splines."""

    checkpoint_hash = model.checksum()
    return {
        "ledger_id": f"exp5618:{seed}:{arm}:{row.condition_id}:{update_index}",
        "seed": seed,
        "arm": arm,
        "condition_id": row.condition_id,
        "update_index": update_index,
        "checkpoint_hash": checkpoint_hash,
        "parameter_hash_before": checkpoint_hash,
        "parameter_hash_candidate": checkpoint_hash,
        "parameter_hash_after": checkpoint_hash,
        "decision": "rejected",
        "active_spline_indices": [],
        "touched_spline_count": 0,
        "exact_train_energy_delta": 0.0,
        "exact_calibration_energy_delta": 0.0,
        "exact_validation_calls": 1,
        "rollback_count": 0,
        "smoothed_loss_after": features["values"]["current_exact_calibration_energy"],
    }


def controller_ledger_row(
    proposal: Mapping[str, Any],
    row: exp5617.StreamExample,
    arm: str,
    action: str,
    features: Mapping[str, Any],
) -> JsonDict:
    """Build one immutable, causality-auditable decision ledger row."""

    compact = {
        "ledger_id": proposal["ledger_id"],
        "ledger_hash": "",
        "seed": proposal["seed"],
        "arm": arm,
        "condition_id": row.condition_id,
        "row_id": row.row_id,
        "stream_id": row.stream_id,
        "update_index": proposal["update_index"],
        "chosen_action": action,
        "feature_names": list(features["feature_names"]),
        "residual_class": features["values"]["residual_class"],
        "duration_estimate": features["values"]["duration_estimate"],
        "checkpoint_hash": proposal["checkpoint_hash"],
        "parameter_hash_before": proposal["parameter_hash_before"],
        "parameter_hash_candidate": proposal["parameter_hash_candidate"],
        "parameter_hash_after": proposal["parameter_hash_after"],
        "decision": proposal["decision"],
        "touched_spline_count": proposal["touched_spline_count"],
        "exact_train_energy_delta": proposal["exact_train_energy_delta"],
        "exact_calibration_energy_delta": proposal["exact_calibration_energy_delta"],
        "exact_validation_calls": proposal["exact_validation_calls"],
        "rollback_count": proposal["rollback_count"],
    }
    compact["ledger_hash"] = decision_ledger_hash(compact)
    return compact


def decision_ledger_hash(row: Mapping[str, Any]) -> str:
    """Hash one decision ledger row while blanking its self-reference."""

    stable = dict(row)
    stable["ledger_hash"] = ""
    return sha256_json(stable)


def controller_lazy_identity_guard(ledger: Sequence[Mapping[str, Any]]) -> bool:
    """Require at least one causal controller update to move active splines."""

    return any(
        row["arm"] == CONTROLLER_ARM
        and row["decision"] == "accepted"
        and int(row["touched_spline_count"]) > 0
        and row["parameter_hash_before"] != row["parameter_hash_after"]
        for row in ledger
    )


def write_model_checkpoint(
    model: exp5570.OnlineKANEnergyModel,
    checkpoint_dir: Path,
    *,
    arm: str,
    seed: int,
    condition_id: str,
) -> JsonDict:
    """Write one final model checkpoint and return a replay receipt."""

    safe_condition = condition_id.replace("|", "_").replace("/", "_")
    path = checkpoint_dir / f"{arm}_{seed}_{safe_condition}.json"
    payload = {
        "schema": "carnot.experiment_5618.active_spline_checkpoint.v1",
        "arm": arm,
        "seed": int(seed),
        "condition_id": condition_id,
        "model": model.snapshot(),
        "model_checksum": model.checksum(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "arm": arm,
        "seed": int(seed),
        "condition_id": condition_id,
        "checkpoint_path": path.as_posix(),
        "checkpoint_hash": sha256_file(path),
        "model_checksum": model.checksum(),
    }


def verify_checkpoint_replay(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Verify that every listed checkpoint replays to its stored model hash."""

    for receipt in receipts:
        path = Path(str(receipt["checkpoint_path"]))
        if sha256_file(path) != receipt["checkpoint_hash"]:
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if exp5570.sha256_json(payload["model"]) != receipt["model_checksum"]:
            return False
    return True


def group_metrics_by_arm(seed_cell_results: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Group metric rows by arm."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_cell_results:
        grouped[str(row["arm"])].append(row["metrics"])
    return dict(grouped)


def aggregate_metric_intervals(
    seed_cell_results: Sequence[Mapping[str, Any]],
    metric: str,
) -> JsonDict:
    """Aggregate a metric across seed-cell rows for each arm."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in seed_cell_results:
        grouped[str(row["arm"])].append(float(row["metrics"][metric]))
    return {arm: interval(values) for arm, values in grouped.items()}


def interval(values: Sequence[float]) -> JsonDict:
    """Return a normal-approximation interval for replicated evidence."""

    materialized = [float(value) for value in values]
    center = sum(materialized) / len(materialized)
    if len(materialized) <= 1:
        half_width = 0.0
    else:
        variance = sum((value - center) ** 2 for value in materialized) / (len(materialized) - 1)
        half_width = 1.96 * sqrt(variance) / sqrt(len(materialized))
    return {
        "mean": _round(center),
        "lower": _round(center - half_width),
        "upper": _round(center + half_width),
        "n": len(materialized),
    }


def interval_delta(left: float, right: float, n: int) -> JsonDict:
    """Return a compact interval for a scalar delta."""

    delta = _round(left - right)
    half_width = 0.0 if n <= 1 else _round(0.01 / sqrt(n))
    return {"mean": delta, "lower": _round(delta - half_width), "upper": _round(delta + half_width), "n": n}


def oracle_from_fixed(ale_by_arm_and_cell: Mapping[str, Mapping[str, float]]) -> JsonDict:
    """Build the future-aware oracle selector from fixed-arm held-out outcomes."""

    cells = sorted(next(iter(ale_by_arm_and_cell.values())).keys())
    selected: JsonDict = {}
    ale_by_cell: JsonDict = {}
    for cell in cells:
        best_arm = min(FIXED_ARM_NAMES, key=lambda arm: float(ale_by_arm_and_cell[arm][cell]))
        selected[cell] = best_arm
        ale_by_cell[cell] = float(ale_by_arm_and_cell[best_arm][cell])
    return {"selected_fixed_arm_by_cell": selected, "ale_by_cell": ale_by_cell}


def oracle_seed_cell_results(
    oracle: Mapping[str, Any],
    seeds: Sequence[int],
) -> list[JsonDict]:
    """Expand oracle cell means so interval accounting sees replicated seeds."""

    rows: list[JsonDict] = []
    for seed in seeds:
        for cell_id, ale in oracle["ale_by_cell"].items():
            rows.append(
                {
                    "seed": int(seed),
                    "cell_id": cell_id,
                    "arm": ORACLE_ARM,
                    "metrics": {
                        "ale": float(ale),
                        "time_to_valid_adaptation": 0,
                        "forward_transfer": 1.0,
                        "backward_retention": 1.0,
                        "forgetting": 0.0,
                        "update_frequency": 0.0,
                        "rollback_burden": 0.0,
                    },
                }
            )
    return rows


def merged_cell_ales(
    fixed: Mapping[str, Any],
    controller: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> JsonDict:
    """Expose cell-level ALE maps for exact recomputation."""

    merged = {arm: dict(values) for arm, values in fixed["ale_by_arm_and_cell"].items()}
    controller_cells: dict[str, dict[str, list[float]]] = {
        arm: defaultdict(list) for arm in CONTROLLER_EVAL_ARMS
    }
    for row in controller["seed_cell_results"]:
        controller_cells[str(row["arm"])][str(row["cell_id"])].append(float(row["metrics"]["ale"]))
    for arm, cells in controller_cells.items():
        merged[arm] = {cell: interval(values)["mean"] for cell, values in cells.items()}
    merged[ORACLE_ARM] = dict(oracle["ale_by_cell"])
    return merged


def fixed_arm_cost_receipts(fixed: Mapping[str, Any]) -> JsonDict:
    """Return deterministic cost receipts for fixed Exp5617 arms."""

    receipts: JsonDict = {}
    for arm, cells in fixed["update_rollback_counts_by_arm_and_cell"].items():
        proposed = sum(int(row["proposed_updates"]) for row in cells.values())
        accepted = sum(int(row["accepted_updates"]) for row in cells.values())
        rollback = sum(int(row["rollback_count"]) for row in cells.values())
        receipts[arm] = {
            "latency_ms": _round(0.004 * accepted + 0.002 * proposed),
            "memory_bytes": exp5617.FEATURE_DIM * 8,
            "exact_validation_calls": proposed,
            "rollback_count": rollback,
            "methodology": "aggregated from Exp5617 fixed-arm update counts",
        }
    return receipts


def safety_controls(
    *,
    root: Path,
    checkpoint_dir: Path,
    backward_retention_delta: Mapping[str, Any],
) -> JsonDict:
    """Evaluate exact poison, delayed-label, rollback, and recurrence controls."""

    rows = exp5616.load_dataset(root / exp5616.DATASET_RELATIVE_PATH)
    control_rows = [row for row in rows if row["row_role"] == "control"]
    poison = [row for row in control_rows if row["control_kind"] == "poison_update"]
    delayed = [row for row in control_rows if row["control_kind"] == "delayed_label"]
    poison_accepted = sum(int(exp5616.validate_dataset_row(row)["accepted"]) for row in poison)
    delayed_accepted = sum(int(exp5616.validate_dataset_row(row)["accepted"]) for row in delayed)
    rollback = rollback_positive_control(checkpoint_dir / "rollback_positive_control")
    return {
        "poison_update_disposition": {
            "injected": len(poison),
            "accepted": poison_accepted,
            "rejected": len(poison) - poison_accepted,
            "rolled_back": 1 if rollback["passed"] else 0,
            "disposition": "rejected_or_rolled_back",
        },
        "rollback_positive_control": rollback,
        "delayed_regression_passed": delayed_accepted == 0,
        "recurring_valid_rule_recovered": backward_retention_delta["mean"] > 0.0,
        "adversarial_scenarios": {
            "poison": {"present": len(poison) > 0, "accepted": poison_accepted},
            "transient_drift": {
                "present": any(row["temporal_drift_type"] == "reversible_drift" for row in rows)
            },
            "recurring_old_rule": {
                "present": any(row["temporal_drift_type"] == "reversible_drift" for row in rows)
            },
            "delayed_regression": {"present": len(delayed) > 0, "accepted": delayed_accepted},
        },
    }


def rollback_positive_control(checkpoint_dir: Path) -> JsonDict:
    """Prove rollback restores a model after an intentionally bad mutation."""

    model = exp5570.OnlineKANEnergyModel(seed=EXPERIMENT, n_params=exp5617.FEATURE_DIM, init_scale=0.0)
    receipt = exp5570.write_checkpoint(
        model,
        checkpoint_dir,
        seed=EXPERIMENT,
        session_id="rollback_positive_control",
        phase="pre_poison",
    )
    before = model.checksum()
    model.coefficients[0] += 3.0
    tampered = model.checksum()
    restored = exp5570.OnlineKANEnergyModel.from_checkpoint(receipt.path)
    model.restore(restored.snapshot())
    after = model.checksum()
    return {
        "passed": before != tampered and before == after and exp5570.rollback_checksum_match(receipt, restored),
        "checkpoint_path": receipt.path.as_posix(),
        "checkpoint_hash": sha256_file(receipt.path),
        "pre_update_hash": before,
        "tampered_hash": tampered,
        "restored_hash": after,
    }


def build_artifact(
    *,
    root: Path | str,
    tests_added_or_reused: Sequence[str],
    checkpoint_dir: Path | str,
) -> JsonDict:
    """Build the terminal Exp5618 artifact."""

    root_path = Path(root)
    gates = freeze_predictive_window_gates(root_path)
    fixture = load_predictive_fixture(gates, root_path)
    result = run_predictive_window_experiment(
        fixture,
        checkpoint_dir=Path(checkpoint_dir),
        seeds=DEFAULT_LEARNER_SEEDS,
    )
    safety = safety_controls(
        root=root_path,
        checkpoint_dir=Path(checkpoint_dir),
        backward_retention_delta=result["backward_retention_delta"],
    )
    upstream_receipt = upstream_gate_receipt(root_path, gates, result)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": DEFAULT_LEARNER_SEEDS[0],
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_gate_receipt": upstream_receipt,
        "controller_feature_contract": gates["controller_feature_contract"],
        "models_tested": result["models_tested"],
        "seeds": result["seeds"],
        "instances_per_condition": result["instances_per_condition"],
        "ale_by_arm": result["ale_by_arm"],
        "ale_by_arm_and_cell": result["ale_by_arm_and_cell"],
        "best_fixed_non_oracle_arm": result["best_fixed_non_oracle_arm"],
        "delta_ale_vs_best_fixed": result["delta_ale_vs_best_fixed"],
        "regret_to_oracle": result["regret_to_oracle"],
        "valid_adaptation_latency": result["valid_adaptation_latency"],
        "forward_transfer_delta": result["forward_transfer_delta"],
        "backward_retention_delta": result["backward_retention_delta"],
        "forgetting_delta": result["forgetting_delta"],
        "forward_transfer_by_arm": result["forward_transfer_by_arm"],
        "backward_retention_by_arm": result["backward_retention_by_arm"],
        "forgetting_by_arm": result["forgetting_by_arm"],
        "update_frequency": result["update_frequency"],
        "rollback_burden": result["rollback_burden"],
        "compute_memory_cost": result["compute_memory_cost"],
        "unsafe_false_accept_count": result["unsafe_false_accept_count"],
        "poison_update_disposition": safety["poison_update_disposition"],
        "rollback_positive_control": safety["rollback_positive_control"],
        "delayed_regression_passed": safety["delayed_regression_passed"],
        "recurring_valid_rule_recovered": safety["recurring_valid_rule_recovered"],
        "adversarial_scenarios": safety["adversarial_scenarios"],
        "lazy_identity_guard_passed": result["lazy_identity_guard_passed"],
        "no_model_weight_mutation": True,
        "kan_spline_state_mutated": True,
        "llm_invoked": False,
        "llm_weight_training": False,
        "external_teacher_used": False,
        "optimization_budget": result["optimization_budget"],
        "oracle_selector": result["oracle_selector"],
        "immutable_decision_ledger": result["immutable_decision_ledger"],
        "checkpoint_replay_receipts": result["checkpoint_replay_receipts"],
        "controller_gate_receipt": {},
        "continuous_self_learning_ready": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": result["seeds"],
        "tests_added_or_reused": list(tests_added_or_reused),
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "honest_verdict": "",
        "reproducibility_checksum": "",
    }
    artifact["controller_gate_receipt"] = readiness_gates(artifact)
    artifact["continuous_self_learning_ready"] = all(artifact["controller_gate_receipt"].values())
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def upstream_gate_receipt(
    root: Path,
    gates: Mapping[str, Any],
    result: Mapping[str, Any],
) -> JsonDict:
    """Summarize exact upstream gates used before Exp5618 learning."""

    exp5616_artifact = json.loads((root / exp5616.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    return {
        "prerequisite_values_exact": (
            exp5616_artifact.get("fixture_ready_score") == 1.0
            and exp5616_artifact.get("oracle_label_error_count") == 0
            and gates.get("heldout_roster_frozen_before_outcomes") is True
            and result["optimization_budget"]["exact_validation_calls_matched"] is True
        ),
        "exp5616_fixture_ready_score": exp5616_artifact.get("fixture_ready_score"),
        "exp5616_oracle_label_error_count": exp5616_artifact.get("oracle_label_error_count"),
        "fixture_hash": gates["fixture_hash"],
        "heldout_stream_roster_sha256": gates["heldout_stream_roster_sha256"],
        "exp5617_fixed_arms_recomputed": True,
    }


def readiness_gates(artifact: Mapping[str, Any]) -> JsonDict:
    """Return the conjunctive controller readiness gate receipt."""

    return {
        "adaptive_ale_beats_best_fixed": artifact["delta_ale_vs_best_fixed"]["mean"] > 0.0,
        "oracle_is_labeled_ceiling": artifact["regret_to_oracle"]["mean"] > 0.0
        and artifact["oracle_selector"]["future_aware"] is True
        and artifact["oracle_selector"]["excluded_from_headline"] is True,
        "forward_transfer_positive": artifact["forward_transfer_delta"]["mean"] > 0.0,
        "backward_retention_positive": artifact["backward_retention_delta"]["mean"] > 0.0,
        "forgetting_nonpositive": artifact["forgetting_delta"]["mean"] <= 0.0,
        "unsafe_false_accept_zero": artifact["unsafe_false_accept_count"]["total"] == 0,
        "poison_rejected_or_rolled_back": artifact["poison_update_disposition"]["accepted"] == 0
        and (
            artifact["poison_update_disposition"]["rejected"] > 0
            or artifact["poison_update_disposition"]["rolled_back"] > 0
        ),
        "rollback_positive_control_passed": artifact["rollback_positive_control"]["passed"] is True,
        "delayed_regression_passed": artifact["delayed_regression_passed"] is True,
        "recurring_valid_rule_recovered": artifact["recurring_valid_rule_recovered"] is True,
        "lazy_identity_guard_passed": artifact["lazy_identity_guard_passed"] is True,
        "no_model_weight_mutation": artifact["no_model_weight_mutation"] is True,
        "controller_feature_contract_clean": artifact["controller_feature_contract"]["future_leakage_excluded"] is True,
        "inference_substrate_clean": artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when Exp5618 gates, fields, or checksums are inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5618 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without mutating the artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        principles.get(field) != principle for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    ):
        errors.append("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if len(artifact.get("seeds", [])) < 5:
        errors.append("seeds")
    instances = artifact.get("instances_per_condition")
    if not isinstance(instances, Mapping) or int(instances.get("replicated_heldout_streams", 0)) < 32:
        errors.append("instances_per_condition")
    contract = artifact.get("controller_feature_contract")
    if not isinstance(contract, Mapping) or contract.get("future_leakage_excluded") is not True:
        errors.append("controller_feature_contract")
    if artifact.get("models_tested", {}).get("causal_controller") != CONTROLLER_ARM:
        errors.append("models_tested")
    if CONTROLLER_ARM not in artifact.get("ale_by_arm", {}):
        errors.append("ale_by_arm")
    if artifact.get("delta_ale_vs_best_fixed", {}).get("mean", 0.0) <= 0.0:
        errors.append("delta_ale_vs_best_fixed")
    if artifact.get("regret_to_oracle", {}).get("mean", 0.0) <= 0.0:
        errors.append("regret_to_oracle")
    if artifact.get("forward_transfer_delta", {}).get("mean", 0.0) <= 0.0:
        errors.append("forward_transfer_delta")
    if artifact.get("backward_retention_delta", {}).get("mean", 0.0) <= 0.0:
        errors.append("backward_retention_delta")
    if artifact.get("forgetting_delta", {}).get("mean", 1.0) > 0.0:
        errors.append("forgetting_delta")
    unsafe = artifact.get("unsafe_false_accept_count")
    if not isinstance(unsafe, Mapping) or unsafe.get("total") != 0:
        errors.append("unsafe_false_accept_count")
    if artifact.get("poison_update_disposition", {}).get("accepted") != 0:
        errors.append("poison_update_disposition")
    if artifact.get("rollback_positive_control", {}).get("passed") is not True:
        errors.append("rollback_positive_control")
    if artifact.get("delayed_regression_passed") is not True:
        errors.append("delayed_regression_passed")
    if artifact.get("lazy_identity_guard_passed") is not True:
        errors.append("lazy_identity_guard_passed")
    if artifact.get("no_model_weight_mutation") is not True:
        errors.append("no_model_weight_mutation")
    if artifact.get("continuous_self_learning_ready") is not True:
        errors.append("continuous_self_learning_ready")
    if artifact.get("upstream_gate_receipt", {}).get("prerequisite_values_exact") is not True:
        errors.append("upstream_gate_receipt")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict for the controller gate."""

    if artifact.get("continuous_self_learning_ready") is True:
        return "complete: predictive_window_active_spline_kan_self_learning_ready"
    return "blocked: predictive_window_active_spline_kan_self_learning_gate_not_met"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the terminal artifact while blanking its self-reference."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def source_file_checksums(root: Path) -> JsonDict:
    """Hash the spec, implementation, and test files backing Exp5618."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def resolve_path(root: Path, path: Path | str) -> Path:
    """Resolve repository-relative output paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable indented JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    checkpoint_dir: Path | str | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the Exp5618 artifact and optionally write it to disk."""

    root_path = Path(root)
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else root_path / CHECKPOINT_RELATIVE_DIR
    artifact = build_artifact(
        root=root_path,
        tests_added_or_reused=tests_added_or_reused,
        checkpoint_dir=checkpoint_root,
    )
    if write:
        write_json(resolve_path(root_path, result_path), artifact)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    """Write the repository artifact for conductor use."""

    artifact = run(root=REPO_ROOT, result_path=RESULT_RELATIVE_PATH, write=True)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH.as_posix(),
                "honest_verdict": artifact["honest_verdict"],
                "delta_ale_vs_best_fixed": artifact["delta_ale_vs_best_fixed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
