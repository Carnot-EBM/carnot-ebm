"""Exp5617 active-spline KAN critical task duration map.

Spec refs: REQ-KAN-5617, SCENARIO-KAN-5617.

The experiment measures a narrow KAN-only adaptation boundary on the exact
Exp5616 nonstationary fixture. The target is the exact current-rule state label,
because Exp5616 stream updates are intentionally all exact-valid update records.
Corrupted update rows stay in the safety gate: they prove exact validation is not
outsourced to the KAN and that unsafe update controls receive no acceptance
credit. Mutable arms reuse the Exp5570 active-spline updater; no LLM, PACE,
causal memory, model-weight training, or external teacher participates.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from carnot import experiment_5570_spline_local_kan_online_energy as exp5570
from carnot import experiment_5616_exact_nonstationary_constraint_stream as exp5616


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5617_kan_critical_task_duration_map.json")
CHECKPOINT_RELATIVE_DIR = Path("results/experiment_5617_kan_critical_task_duration_map_checkpoints")
FIXTURE_RESULT_RELATIVE_PATH = exp5616.RESULT_RELATIVE_PATH
FIXTURE_DATASET_RELATIVE_PATH = exp5616.DATASET_RELATIVE_PATH
SPEC_RELATIVE_PATH = Path("openspec/capabilities/kan/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5617_kan_critical_task_duration_map.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5617_kan_critical_task_duration_map.py")

SCHEMA = "carnot.experiment_5617.kan_critical_task_duration_map.v1"
EXPERIMENT = 5617
EXPERIMENT_ID = "experiment_5617_kan_critical_task_duration_map"
TASK_ID = "exp5617-kan-critical-task-duration-map"
MILESTONE = "2026.07.507"
RUN_DATE = "20260714"
INFERENCE_SUBSTRATE = "exact_constraint_stream_active_spline_kan_no_llm"

FEATURE_DIM = exp5570.FEATURE_DIM
DEFAULT_LEARNER_SEEDS = (5617, 5618, 5619, 5620, 5621)
SPACE_SHIFT_FAMILIES = exp5616.SPACE_SHIFT_FAMILIES
TEMPORAL_DRIFT_TYPES = exp5616.TEMPORAL_DRIFT_TYPES
DURATION_CELLS = exp5616.TASK_DURATIONS
SPLIT_NAMES = exp5616.SPLITS
LEARNING_RATE = exp5570.LEARNING_RATE
EXACT_GATE_TOLERANCE = 0.03
VALID_ERROR_THRESHOLD = 0.28
REPLAY_BUDGET = 4

FROZEN_ARM = "frozen_no_update"
RETAIN_REPLAY_ARM = "retain_exact_replay"
RESET_ARM = "reset_adapt"
LOSS_SMOOTHED_ARM = "loss_smoothed_adaptation"
UPDATE_SUBSTITUTION_CONTROL_ARM = "update_substitution_control"
FROZEN_SPLINE_CONTROL_ARM = "frozen_spline_control"
ARM_NAMES = (
    FROZEN_ARM,
    RETAIN_REPLAY_ARM,
    RESET_ARM,
    LOSS_SMOOTHED_ARM,
    UPDATE_SUBSTITUTION_CONTROL_ARM,
    FROZEN_SPLINE_CONTROL_ARM,
)
MUTABLE_ARMS = (
    RETAIN_REPLAY_ARM,
    RESET_ARM,
    LOSS_SMOOTHED_ARM,
    UPDATE_SUBSTITUTION_CONTROL_ARM,
    FROZEN_SPLINE_CONTROL_ARM,
)
DISAGGREGATED_METRICS = (
    "average_lifelong_error",
    "inherited_instability",
    "transient_relearning_error",
    "backward_retention",
    "forward_transfer",
    "time_to_valid_adaptation",
    "update_rollback_counts",
    "unsafe_false_accepts",
)
SPEC_REFS = ("REQ-KAN-5617", "SCENARIO-KAN-5617")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "fixture_hash",
    "models_tested",
    "duration_cells",
    "seeds",
    "instances_per_condition",
    "ale_by_arm_and_cell",
    "instability_by_arm_and_cell",
    "transient_error_by_arm_and_cell",
    "backward_retention_by_arm",
    "forward_transfer_by_arm",
    "unsafe_false_accept_count",
    "empirical_switch_durations",
    "critical_duration_fit_r2",
    "nondegenerate_switch_cases",
    "lazy_identity_guard_passed",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "required evidence fields explain why they exist",
    "fixture_hash": "upstream exact dataset identity is fixed before learning",
    "models_tested": "every KAN and control arm is visible",
    "duration_cells": "the independent duration variable is explicit",
    "seeds": "learner replication is explicit",
    "instances_per_condition": "held-out evidence floor is checkable",
    "ale_by_arm_and_cell": "lifelong error is disaggregated",
    "instability_by_arm_and_cell": "inherited bias is measured",
    "transient_error_by_arm_and_cell": "relearning cost is measured",
    "backward_retention_by_arm": "prior-rule retention is independent",
    "forward_transfer_by_arm": "future-rule transfer is independent",
    "unsafe_false_accept_count": "exact safety is non-negotiable",
    "empirical_switch_durations": "crossings are data-backed",
    "critical_duration_fit_r2": "downstream gating receives a numeric fit",
    "nondegenerate_switch_cases": "one accidental crossing is insufficient",
    "lazy_identity_guard_passed": "the KAN update must be active",
    "inference_substrate": "only the active-spline KAN adapts and no LLM participates",
    "random_seeds": "the run is repeatable",
    "reproducibility_checksum": "the run is repeatable",
    "honest_verdict": "a bounded null is terminal",
}
FIELD_PRINCIPLES: JsonDict = {
    **REQUIRED_FIELD_PRINCIPLES,
    "structured_gates": "preregistered gates prevent outcome-driven selection",
    "immutable_update_ledger": "every proposed update is attributable",
    "checkpoint_receipts": "accepted model states can be replayed",
    "control_credit_guard": "bypassed learners receive no adaptation credit",
}
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest tests/python/test_experiment_5617_kan_critical_task_duration_map.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/experiment_5617_kan_critical_task_duration_map.py -m pytest tests/python/test_experiment_5617_kan_critical_task_duration_map.py -q --no-cov -n 0 && .venv/bin/coverage report --include=python/carnot/experiment_5617_kan_critical_task_duration_map.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5617_kan_critical_task_duration_map.json",
)


@dataclass(frozen=True)
class StreamExample:
    """One exact state-label row visible to the active-spline KAN."""

    row_id: str
    stream_id: str
    condition_id: str
    space_shift_family: str
    temporal_drift_type: str
    conflict_class: str
    duration: int
    instance_index: int
    split: str
    step_index: int
    label: int
    old_label: int
    future_label: int
    features: np.ndarray


@dataclass(frozen=True)
class FrozenFixture:
    """Exp5616 rows after the structured gates have been frozen."""

    fixture_hash: str
    stream_count: int
    heldout_replicates_per_condition: int
    rows_by_split: dict[str, tuple[StreamExample, ...]]
    control_error_count: int
    condition_ids: tuple[str, ...]


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data in one stable order for hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for stable JSON-compatible payloads."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a prefixed SHA-256 digest over a file's exact bytes."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _round(value: float, digits: int = 6) -> float:
    """Round once at artifact boundaries so recomputation is stable."""

    return round(float(value), digits)


def cell_key(space_shift_family: str, temporal_drift_type: str, duration: int) -> str:
    """Build the stable metric cell identifier."""

    return f"{space_shift_family}|{temporal_drift_type}|d{int(duration)}"


def conflict_class_for(space_shift_family: str) -> str:
    """Name whether the shifted rule shares or conflicts with old knowledge."""

    return "conflict" if space_shift_family == "conflicting_rule" else "shared"


def freeze_structured_gates(root: Path | str = REPO_ROOT) -> JsonDict:
    """Freeze splits, cells, metrics, and promotion rules before row outcomes load."""

    root_path = Path(root)
    fixture_artifact = json.loads((root_path / FIXTURE_RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    fixture_hash = "sha256:" + str(fixture_artifact["dataset_sha256"])
    return {
        "structured_gates_frozen": True,
        "fixture_rows_loaded": False,
        "fixture_result_path": FIXTURE_RESULT_RELATIVE_PATH.as_posix(),
        "fixture_dataset_path": FIXTURE_DATASET_RELATIVE_PATH.as_posix(),
        "fixture_hash": fixture_hash,
        "split_names": list(SPLIT_NAMES),
        "duration_cells": list(DURATION_CELLS),
        "space_shift_families": list(SPACE_SHIFT_FAMILIES),
        "temporal_drift_types": list(TEMPORAL_DRIFT_TYPES),
        "models_tested": list(ARM_NAMES),
        "metrics_frozen": list(DISAGGREGATED_METRICS),
        "minimum_learner_seeds": 5,
        "instances_per_condition_floor": 32,
        "promotion_rules": {
            "selection_label_source": "calibration_only",
            "future_heldout_labels_for_selection": False,
            "minimum_nondegenerate_switch_cases": 2,
            "exact_gate_tolerance": EXACT_GATE_TOLERANCE,
            "valid_error_threshold": VALID_ERROR_THRESHOLD,
        },
    }


def load_frozen_fixture(gates: Mapping[str, Any], root: Path | str = REPO_ROOT) -> FrozenFixture:
    """Load Exp5616 rows only after the structured gate packet is frozen."""

    if gates.get("structured_gates_frozen") is not True:
        raise ValueError("structured_gates_frozen")
    root_path = Path(root)
    artifact = json.loads((root_path / FIXTURE_RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    exp5616.validate_artifact(artifact, repo_root=root_path)
    rows = exp5616.load_dataset(root_path / FIXTURE_DATASET_RELATIVE_PATH)
    examples = [example_from_row(row) for row in rows if row["row_role"] == "stream_update"]
    rows_by_split = {
        split: tuple(example for example in examples if example.split == split)
        for split in SPLIT_NAMES
    }
    stream_count = len({example.stream_id for example in examples})
    condition_ids = tuple(
        cell_key(space, drift, duration)
        for space in SPACE_SHIFT_FAMILIES
        for drift in TEMPORAL_DRIFT_TYPES
        for duration in DURATION_CELLS
    )
    heldout_per_condition = min(
        len({row.stream_id for row in rows_by_split["heldout"] if row.condition_id == condition_id})
        for condition_id in condition_ids
    )
    control_error_count = sum(
        int(exp5616.validate_dataset_row(row)["accepted"] != bool(row["accepted_by_exact_validator"]))
        for row in rows
        if row["row_role"] == "control"
    )
    return FrozenFixture(
        fixture_hash=str(gates["fixture_hash"]),
        stream_count=stream_count,
        heldout_replicates_per_condition=heldout_per_condition * len(DEFAULT_LEARNER_SEEDS),
        rows_by_split=rows_by_split,
        control_error_count=control_error_count,
        condition_ids=condition_ids,
    )


def example_from_row(row: Mapping[str, Any]) -> StreamExample:
    """Convert one Exp5616 stream row into the fixed KAN feature schema."""

    return StreamExample(
        row_id=str(row["row_id"]),
        stream_id=str(row["stream_id"]),
        condition_id=str(row["condition_id"]),
        space_shift_family=str(row["space_shift_family"]),
        temporal_drift_type=str(row["temporal_drift_type"]),
        conflict_class=conflict_class_for(str(row["space_shift_family"])),
        duration=int(row["duration"]),
        instance_index=int(row["instance_index"]),
        split=str(row["split"]),
        step_index=int(row["step_index"]),
        label=1 if row["state_labels"]["current_rule"] else -1,
        old_label=1 if row["state_labels"]["old_rule"] else -1,
        future_label=1 if row["state_labels"]["future_rule"] else -1,
        features=feature_vector(row),
    )


def feature_vector(row: Mapping[str, Any]) -> np.ndarray:
    """Derive KAN features from state and rule structure, not learned labels."""

    state = row["state"]
    rules = row["rules"]
    score = float(state["variables"]["score"])
    old_rule = rules["old_rule"]
    current_rule = rules["current_rule"]
    future_rule = rules["future_rule"]
    duration = int(row["duration"])
    step_index = int(row["step_index"])
    features = np.zeros(FEATURE_DIM, dtype=np.float64)
    old_margin = rule_margin(score, old_rule)
    current_margin = rule_margin(score, current_rule)
    future_margin = rule_margin(score, future_rule)
    duration_scaled = duration / max(DURATION_CELLS)
    features[0] = 1.0
    features[1] = old_margin
    features[2] = -old_margin
    features[3] = current_margin
    features[4] = -current_margin
    features[5] = 1.0 if current_rule["comparator"] == ">=" else 0.0
    features[6] = 1.0 if current_rule["comparator"] == "<=" else 0.0
    features[7] = 1.0 if row["space_shift_family"] == "shared_rule" else 0.0
    features[8] = 1.0 if row["space_shift_family"] == "conflicting_rule" else 0.0
    features[9] = 1.0 if row["temporal_drift_type"] == "no_drift" else 0.0
    features[10] = 1.0 if row["temporal_drift_type"] == "reversible_drift" else 0.0
    features[11] = 1.0 if row["temporal_drift_type"] == "persistent_drift" else 0.0
    features[12] = duration_scaled
    features[13] = step_index / max(duration - 1, 1)
    features[14] = (float(current_rule["threshold"]) - float(old_rule["threshold"])) / 100.0
    features[15] = abs(features[14])
    features[16] = current_margin * duration_scaled
    features[17] = -current_margin * duration_scaled
    features[18] = future_margin
    features[19] = 1.0 if row["state"]["domain_space"] == "shifted_domain" else 0.0
    return np.clip(features, -1.0, 1.0)


def rule_margin(score: float, rule: Mapping[str, Any]) -> float:
    """Return a signed margin that is positive exactly when the rule is true."""

    threshold = float(rule["threshold"])
    if rule["comparator"] == ">=":
        return _round((score - threshold) / 100.0)
    return _round((threshold - score) / 100.0)


def run_duration_map(
    fixture: FrozenFixture,
    *,
    checkpoint_dir: Path | str,
    seeds: Sequence[int] = DEFAULT_LEARNER_SEEDS,
) -> JsonDict:
    """Run every KAN/control arm over each exact duration condition."""

    checkpoint_root = Path(checkpoint_dir)
    seed_cell_results: list[JsonDict] = []
    ledger: list[JsonDict] = []
    checkpoint_receipts: list[JsonDict] = []
    for seed in seeds:
        for condition_id in fixture.condition_ids:
            train = rows_for_cell(fixture.rows_by_split["train"], condition_id)
            calibration = rows_for_cell(fixture.rows_by_split["calibration"], condition_id)
            heldout = rows_for_cell(fixture.rows_by_split["heldout"], condition_id)
            ordered_train = stable_seed_order(train, int(seed))
            for arm in ARM_NAMES:
                arm_result = run_arm_cell(
                    arm=arm,
                    seed=int(seed),
                    train_rows=ordered_train,
                    calibration_rows=calibration,
                    heldout_rows=heldout,
                    checkpoint_dir=checkpoint_root,
                )
                seed_cell_results.append(
                    {
                        "seed": int(seed),
                        "cell_id": condition_id,
                        "arm": arm,
                        "metrics": arm_result["metrics"],
                    }
                )
                ledger.extend(arm_result["ledger"])
                checkpoint_receipts.append(arm_result["checkpoint_receipt"])
    summaries = summarize_duration_results(seed_cell_results)
    switches = estimate_switches(summaries["ale_by_arm_and_cell"])
    lazy_guard = lazy_identity_guard(ledger)
    unsafe = {"total": fixture.control_error_count, "by_arm_and_cell": summaries["unsafe_false_accept_by_arm_and_cell"]}
    result: JsonDict = {
        "models_tested": list(ARM_NAMES),
        "duration_cells": list(DURATION_CELLS),
        "seeds": [int(seed) for seed in seeds],
        "instances_per_condition": {
            "fixture_streams_per_condition": exp5616.INSTANCES_PER_CONDITION,
            "heldout_streams_per_condition": fixture.heldout_replicates_per_condition // len(seeds),
            "learner_seeds": len(seeds),
            "replicated_heldout_streams": fixture.heldout_replicates_per_condition,
        },
        "ale_by_arm_and_cell": summaries["ale_by_arm_and_cell"],
        "instability_by_arm_and_cell": summaries["instability_by_arm_and_cell"],
        "transient_error_by_arm_and_cell": summaries["transient_error_by_arm_and_cell"],
        "backward_retention_by_arm": summaries["backward_retention_by_arm"],
        "forward_transfer_by_arm": summaries["forward_transfer_by_arm"],
        "time_to_valid_adaptation_by_arm_and_cell": summaries["time_to_valid_adaptation_by_arm_and_cell"],
        "update_rollback_counts_by_arm_and_cell": summaries["update_rollback_counts_by_arm_and_cell"],
        "unsafe_false_accept_count": unsafe,
        "empirical_switch_durations": switches["empirical_switch_durations"],
        "critical_duration_fit_r2": switches["critical_duration_fit_r2"],
        "nondegenerate_switch_cases": switches["nondegenerate_switch_cases"],
        "critical_task_duration": switches["critical_task_duration"],
        "lazy_identity_guard_passed": lazy_guard,
        "control_credit_guard": control_credit_guard(summaries["ale_by_arm_and_cell"]),
        "optimization_budget": optimization_budget_summary(ledger),
        "immutable_update_ledger": ledger,
        "checkpoint_receipts": checkpoint_receipts,
        "selection_protocol": {
            "selection_label_source": "calibration_only",
            "future_heldout_labels_for_selection": False,
        },
    }
    return result


def rows_for_cell(rows: Sequence[StreamExample], condition_id: str) -> tuple[StreamExample, ...]:
    """Return one preregistered representative row per stream instance.

    Exp5616 stores one row per task step. Exp5617 measures the duration cell as
    the independent variable, so each stream contributes its final in-duration
    state to train, calibration, or held-out evaluation. This keeps the 32
    independent stream instances as the evidence unit instead of overweighting
    longer durations merely because they have more rows.
    """

    representatives: dict[str, StreamExample] = {}
    for row in rows:
        if row.condition_id != condition_id:
            continue
        current = representatives.get(row.stream_id)
        if current is None or row.step_index > current.step_index:
            representatives[row.stream_id] = row
    return tuple(sorted(representatives.values(), key=lambda row: (row.instance_index, row.row_id)))


def stable_seed_order(rows: Sequence[StreamExample], seed: int) -> tuple[StreamExample, ...]:
    """Use one deterministic row order per learner seed and share it across arms."""

    keyed = sorted(
        rows,
        key=lambda row: sha256_json([seed, row.stream_id, row.step_index, row.row_id]),
    )
    return tuple(keyed)


def run_arm_cell(
    *,
    arm: str,
    seed: int,
    train_rows: Sequence[StreamExample],
    calibration_rows: Sequence[StreamExample],
    heldout_rows: Sequence[StreamExample],
    checkpoint_dir: Path,
) -> JsonDict:
    """Run one arm on one condition cell with matched update opportunities."""

    model = initialized_model(seed, arm)
    initial_model = initialized_model(seed, arm)
    replay_buffer: list[exp5570.FeatureRow] = []
    accepted = 0
    rejected = 0
    rolled_back = 0
    validation_calls = 0
    first_valid_index = 0 if exact_error(model, calibration_rows, label_name="label") <= VALID_ERROR_THRESHOLD else None
    transient_error = exact_error(model, calibration_rows, label_name="label")
    smoothed_loss = exact_energy(model, calibration_rows, label_name="label")
    ledger: list[JsonDict] = []
    for update_index, row in enumerate(train_rows):
        proposal = propose_update(
            model=model,
            arm=arm,
            seed=seed,
            update_index=update_index,
            row=row,
            calibration_rows=calibration_rows,
            replay_buffer=replay_buffer,
            smoothed_loss=smoothed_loss,
        )
        validation_calls += 1
        smoothed_loss = proposal["smoothed_loss_after"]
        if proposal["decision"] == "accepted":
            accepted += 1
            if first_valid_index is None and exact_error(model, calibration_rows, label_name="label") <= VALID_ERROR_THRESHOLD:
                first_valid_index = update_index + 1
        elif proposal["decision"] == "rolled_back":
            rolled_back += 1
        else:
            rejected += 1
        ledger.append(compact_ledger_row(proposal))
        replay_buffer.append(feature_row(row, label_name="old_label" if arm == RETAIN_REPLAY_ARM else "label"))
    update_budget = len(train_rows)
    if first_valid_index is None:
        first_valid_index = update_budget + 1
    checkpoint_receipt = write_checkpoint_receipt(
        model,
        checkpoint_dir,
        arm=arm,
        seed=seed,
        condition_id=heldout_rows[0].condition_id,
    )
    metrics = {
        "ale": exact_error(model, heldout_rows, label_name="label"),
        "instability": inherited_instability(model, heldout_rows),
        "transient_error": transient_error,
        "backward_retention": 1.0 - exact_error(model, heldout_rows, label_name="old_label"),
        "forward_transfer": 1.0 - exact_error(model, heldout_rows, label_name="future_label"),
        "time_to_valid_adaptation": int(first_valid_index),
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "rollback_count": rolled_back,
        "proposed_updates": update_budget if arm in MUTABLE_ARMS else 0,
        "exact_validation_calls": validation_calls,
        "parameter_diff_norm": _round(float(np.linalg.norm(model.coefficients - initial_model.coefficients))),
        "unsafe_false_accepts": 0,
    }
    return {"metrics": metrics, "ledger": ledger, "checkpoint_receipt": checkpoint_receipt}


def initialized_model(seed: int, arm: str) -> exp5570.OnlineKANEnergyModel:
    """Create the arm's starting KAN, with retained arms warm-started on old rules."""

    model = exp5570.OnlineKANEnergyModel(seed=seed, n_params=FEATURE_DIM, init_scale=0.0)
    if arm != RESET_ARM:
        model.coefficients[1] = 7.0
        model.coefficients[2] = -0.5
    return model


def propose_update(
    *,
    model: exp5570.OnlineKANEnergyModel,
    arm: str,
    seed: int,
    update_index: int,
    row: StreamExample,
    calibration_rows: Sequence[StreamExample],
    replay_buffer: Sequence[exp5570.FeatureRow],
    smoothed_loss: float,
) -> JsonDict:
    """Apply the arm-specific proposal and exact calibration gate."""

    before_snapshot = model.snapshot()
    before_hash = model.checksum()
    candidate = exp5570.OnlineKANEnergyModel(seed=seed, n_params=FEATURE_DIM, init_scale=0.0)
    candidate.restore(before_snapshot)
    train_rows = proposal_rows_for_arm(arm, row, replay_buffer)
    train_pre = exact_energy(model, (row,), label_name="label")
    cal_pre = exact_energy(model, calibration_rows, label_name="label")
    receipt = exp5570.apply_online_update(
        candidate,
        train_rows,
        learning_rate=LEARNING_RATE,
        arm=exp5570.ACTIVE_ARM if arm != FROZEN_SPLINE_CONTROL_ARM else FROZEN_SPLINE_CONTROL_ARM,
    )
    train_post = exact_energy(candidate, (row,), label_name="label")
    cal_post = exact_energy(candidate, calibration_rows, label_name="label")
    candidate_smooth = _round(0.55 * cal_post + 0.45 * smoothed_loss)
    decision = update_decision(
        arm=arm,
        receipt=receipt,
        train_pre=train_pre,
        train_post=train_post,
        cal_pre=cal_pre,
        cal_post=cal_post,
        smoothed_loss=smoothed_loss,
        candidate_smooth=candidate_smooth,
    )
    if decision == "accepted":
        model.restore(candidate.snapshot())
    after_hash = model.checksum()
    return {
        "ledger_id": f"exp5617:{seed}:{arm}:{row.condition_id}:{update_index}",
        "ledger_hash": "",
        "seed": seed,
        "arm": arm,
        "condition_id": row.condition_id,
        "update_index": update_index,
        "checkpoint_hash": before_hash,
        "parameter_hash_before": before_hash,
        "parameter_hash_candidate": candidate.checksum(),
        "parameter_hash_after": after_hash,
        "decision": decision,
        "active_spline_indices": receipt.touched_indices,
        "touched_spline_count": len(receipt.touched_indices),
        "exact_train_energy_delta": _round(train_post - train_pre),
        "exact_calibration_energy_delta": _round(cal_post - cal_pre),
        "exact_validation_calls": 1,
        "rollback_count": 1 if decision == "rolled_back" else 0,
        "smoothed_loss_after": candidate_smooth if arm == LOSS_SMOOTHED_ARM else smoothed_loss,
    }


def proposal_rows_for_arm(
    arm: str,
    row: StreamExample,
    replay_buffer: Sequence[exp5570.FeatureRow],
) -> tuple[exp5570.FeatureRow, ...]:
    """Return the exact-feedback rows supplied to the shared active updater."""

    current = feature_row(row, label_name="label")
    if arm == RETAIN_REPLAY_ARM:
        replay = tuple(replay_buffer[-REPLAY_BUDGET:])
        return (current, *replay)
    if arm == UPDATE_SUBSTITUTION_CONTROL_ARM:
        return (exp5570.FeatureRow(
            row_id=f"substitution:{row.row_id}",
            family=row.space_shift_family,
            partition=row.split,
            session_id=row.condition_id,
            label=row.label,
            accepted_by_exact_validator=row.label == 1,
            features=np.zeros(FEATURE_DIM, dtype=np.float64),
        ),)
    return (current,)


def update_decision(
    *,
    arm: str,
    receipt: exp5570.UpdateReceipt,
    train_pre: float,
    train_post: float,
    cal_pre: float,
    cal_post: float,
    smoothed_loss: float,
    candidate_smooth: float,
) -> str:
    """Apply exact gates and control-arm rules to one proposed update."""

    if arm == FROZEN_ARM:
        return "rejected"
    if arm in (UPDATE_SUBSTITUTION_CONTROL_ARM, FROZEN_SPLINE_CONTROL_ARM):
        return "rejected"
    if not receipt.touched_indices:
        return "rolled_back"
    train_ok = train_post <= train_pre
    cal_ok = cal_post <= cal_pre + EXACT_GATE_TOLERANCE
    smooth_ok = candidate_smooth <= smoothed_loss + EXACT_GATE_TOLERANCE
    if arm == LOSS_SMOOTHED_ARM:
        return "accepted" if train_ok and smooth_ok else "rejected"
    return "accepted" if train_ok and cal_ok else "rejected"


def compact_ledger_row(row: Mapping[str, Any]) -> JsonDict:
    """Keep the per-update ledger immutable but compact enough for artifacts."""

    compact = {
        "ledger_id": row["ledger_id"],
        "ledger_hash": "",
        "seed": row["seed"],
        "arm": row["arm"],
        "condition_id": row["condition_id"],
        "update_index": row["update_index"],
        "checkpoint_hash": row["checkpoint_hash"],
        "parameter_hash_before": row["parameter_hash_before"],
        "parameter_hash_candidate": row["parameter_hash_candidate"],
        "parameter_hash_after": row["parameter_hash_after"],
        "decision": row["decision"],
        "touched_spline_count": row["touched_spline_count"],
        "exact_train_energy_delta": row["exact_train_energy_delta"],
        "exact_calibration_energy_delta": row["exact_calibration_energy_delta"],
        "exact_validation_calls": row["exact_validation_calls"],
        "rollback_count": row["rollback_count"],
    }
    compact["ledger_hash"] = ledger_hash(compact)
    return compact


def feature_row(row: StreamExample, *, label_name: str) -> exp5570.FeatureRow:
    """Convert a stream example to the FeatureRow shape required by Exp5570."""

    label = int(getattr(row, label_name))
    return exp5570.FeatureRow(
        row_id=row.row_id,
        family=row.space_shift_family,
        partition=row.split,
        session_id=row.condition_id,
        label=label,
        accepted_by_exact_validator=label == 1,
        features=row.features.copy(),
    )


def exact_energy(
    model: exp5570.OnlineKANEnergyModel,
    rows: Sequence[StreamExample],
    *,
    label_name: str,
) -> float:
    """Return mean hinge energy against one exact label family."""

    if not rows:
        return 0.0
    losses = [
        max(0.0, 1.0 - int(getattr(row, label_name)) * model.score(feature_row(row, label_name=label_name)))
        for row in rows
    ]
    return _round(sum(losses) / len(losses))


def exact_error(
    model: exp5570.OnlineKANEnergyModel,
    rows: Sequence[StreamExample],
    *,
    label_name: str,
) -> float:
    """Return exact classification error for current, old, or future labels."""

    if not rows:
        return 0.0
    misses = 0
    for row in rows:
        label = int(getattr(row, label_name))
        misses += int(model.predict_label(feature_row(row, label_name=label_name)) != label)
    return _round(misses / len(rows))


def inherited_instability(
    model: exp5570.OnlineKANEnergyModel,
    rows: Sequence[StreamExample],
) -> float:
    """Measure cases where the model follows old labels when old/current conflict."""

    conflicting = [row for row in rows if row.old_label != row.label]
    if not conflicting:
        return 0.0
    inherited = 0
    for row in conflicting:
        inherited += int(model.predict_label(feature_row(row, label_name="label")) == row.old_label)
    return _round(inherited / len(conflicting))


def write_checkpoint_receipt(
    model: exp5570.OnlineKANEnergyModel,
    checkpoint_dir: Path,
    *,
    arm: str,
    seed: int,
    condition_id: str,
) -> JsonDict:
    """Write one compact final checkpoint per arm/seed/cell."""

    safe_condition = condition_id.replace("|", "_").replace("/", "_")
    path = checkpoint_dir / f"{arm}_{seed}_{safe_condition}.json"
    payload = {"arm": arm, "seed": seed, "condition_id": condition_id, "model": model.snapshot()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "arm": arm,
        "seed": seed,
        "condition_id": condition_id,
        "checkpoint_path": path.as_posix(),
        "checkpoint_hash": sha256_file(path),
    }


def summarize_duration_results(seed_cell_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate seed-level cell metrics into artifact-visible maps."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_cell_results:
        grouped[(str(row["arm"]), str(row["cell_id"]))].append(row["metrics"])
        by_arm[str(row["arm"])].append(row["metrics"])
    ale: JsonDict = {arm: {} for arm in ARM_NAMES}
    instability: JsonDict = {arm: {} for arm in ARM_NAMES}
    transient: JsonDict = {arm: {} for arm in ARM_NAMES}
    time_to_valid: JsonDict = {arm: {} for arm in ARM_NAMES}
    counts: JsonDict = {arm: {} for arm in ARM_NAMES}
    unsafe: JsonDict = {arm: {} for arm in ARM_NAMES}
    for (arm, condition_id), metrics in grouped.items():
        ale[arm][condition_id] = mean_metric(metrics, "ale")
        instability[arm][condition_id] = mean_metric(metrics, "instability")
        transient[arm][condition_id] = mean_metric(metrics, "transient_error")
        time_to_valid[arm][condition_id] = _round(mean_metric(metrics, "time_to_valid_adaptation"))
        counts[arm][condition_id] = {
            "accepted_updates": int(round(mean_metric(metrics, "accepted_updates"))),
            "rejected_updates": int(round(mean_metric(metrics, "rejected_updates"))),
            "rollback_count": int(round(mean_metric(metrics, "rollback_count"))),
            "proposed_updates": int(round(mean_metric(metrics, "proposed_updates"))),
        }
        unsafe[arm][condition_id] = int(round(mean_metric(metrics, "unsafe_false_accepts")))
    return {
        "ale_by_arm_and_cell": ale,
        "instability_by_arm_and_cell": instability,
        "transient_error_by_arm_and_cell": transient,
        "time_to_valid_adaptation_by_arm_and_cell": time_to_valid,
        "update_rollback_counts_by_arm_and_cell": counts,
        "unsafe_false_accept_by_arm_and_cell": unsafe,
        "backward_retention_by_arm": {
            arm: mean_metric(metrics, "backward_retention") for arm, metrics in by_arm.items()
        },
        "forward_transfer_by_arm": {
            arm: mean_metric(metrics, "forward_transfer") for arm, metrics in by_arm.items()
        },
    }


def mean_metric(metrics: Sequence[Mapping[str, Any]], key: str) -> float:
    """Return a rounded metric mean from seed-level rows."""

    return _round(sum(float(row[key]) for row in metrics) / len(metrics))


def estimate_switches(ale_by_arm_and_cell: Mapping[str, Mapping[str, float]]) -> JsonDict:
    """Estimate retain-vs-reset crossings without using future held-out labels."""

    switch_cases: list[JsonDict] = []
    for space in SPACE_SHIFT_FAMILIES:
        for drift in TEMPORAL_DRIFT_TYPES:
            points = [
                (
                    duration,
                    float(ale_by_arm_and_cell[RETAIN_REPLAY_ARM][cell_key(space, drift, duration)])
                    - float(ale_by_arm_and_cell[RESET_ARM][cell_key(space, drift, duration)]),
                )
                for duration in DURATION_CELLS
            ]
            for (left_duration, left_delta), (right_duration, right_delta) in zip(points, points[1:]):
                if left_delta <= 0.0 < right_delta or left_delta < 0.0 <= right_delta:
                    span = right_delta - left_delta
                    switch_duration = left_duration if span == 0 else left_duration + (0.0 - left_delta) * (right_duration - left_duration) / span
                    switch_cases.append(
                        {
                            "space_shift_family": space,
                            "temporal_drift_type": drift,
                            "conflict_class": conflict_class_for(space),
                            "left_duration": left_duration,
                            "right_duration": right_duration,
                            "switch_duration": _round(switch_duration),
                            "left_delta_retain_minus_reset": _round(left_delta),
                            "right_delta_retain_minus_reset": _round(right_delta),
                            "ci95": confidence_interval([left_delta, right_delta]),
                        }
                    )
    fit_r2 = switch_fit_r2(switch_cases)
    critical = None
    if len(switch_cases) >= 2:
        critical = nearest_duration(mean([case["switch_duration"] for case in switch_cases]))
    return {
        "empirical_switch_durations": switch_cases,
        "critical_duration_fit_r2": fit_r2,
        "nondegenerate_switch_cases": switch_cases,
        "critical_task_duration": critical,
    }


def confidence_interval(values: Sequence[float]) -> JsonDict:
    """Return a simple normal approximation interval for displayed deltas."""

    center = mean(values)
    if len(values) <= 1:
        half_width = 0.0
    else:
        variance = sum((float(value) - center) ** 2 for value in values) / (len(values) - 1)
        half_width = 1.96 * sqrt(variance) / sqrt(len(values))
    return {"mean": center, "lower": _round(center - half_width), "upper": _round(center + half_width), "n": len(values)}


def switch_fit_r2(switch_cases: Sequence[Mapping[str, Any]]) -> float:
    """Score how tightly switch durations cluster for downstream gating."""

    if len(switch_cases) < 2:
        return 0.0
    values = [float(case["switch_duration"]) for case in switch_cases]
    center = mean(values)
    sst = sum((value - center) ** 2 for value in values)
    if sst == 0.0:
        return 1.0
    nearest = float(nearest_duration(center))
    sse = sum((value - nearest) ** 2 for value in values)
    return _round(max(0.0, min(1.0, 1.0 - sse / sst)))


def mean(values: Sequence[float] | Any) -> float:
    """Return a rounded mean for a materialized sequence or generator."""

    materialized = [float(value) for value in values]
    if not materialized:
        return 0.0
    return _round(sum(materialized) / len(materialized))


def nearest_duration(value: float) -> int:
    """Map a real-valued crossing estimate to the nearest preregistered cell."""

    return min(DURATION_CELLS, key=lambda duration: abs(duration - value))


def lazy_identity_guard(ledger: Sequence[Mapping[str, Any]]) -> bool:
    """Require at least one real retained KAN update with touched spline movement."""

    return any(
        row["arm"] == RETAIN_REPLAY_ARM
        and row["decision"] == "accepted"
        and int(row["touched_spline_count"]) > 0
        and row["parameter_hash_before"] != row["parameter_hash_after"]
        for row in ledger
    )


def control_credit_guard(ale_by_arm_and_cell: Mapping[str, Mapping[str, float]]) -> JsonDict:
    """Report zero adaptation credit for bypassed and frozen-spline controls."""

    frozen = ale_by_arm_and_cell[FROZEN_ARM]
    return {
        "update_substitution_control_credit": adaptation_credit(frozen, ale_by_arm_and_cell[UPDATE_SUBSTITUTION_CONTROL_ARM]),
        "frozen_spline_control_credit": adaptation_credit(frozen, ale_by_arm_and_cell[FROZEN_SPLINE_CONTROL_ARM]),
    }


def adaptation_credit(frozen: Mapping[str, float], control: Mapping[str, float]) -> float:
    """Return mean frozen-minus-control ALE improvement for a control arm."""

    return max(0.0, mean(float(frozen[key]) - float(control[key]) for key in frozen))


def optimization_budget_summary(ledger: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check optimization budgets and exact-validation calls across mutable arms."""

    by_arm_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in ledger:
        by_arm_cell[(str(row["arm"]), str(row["condition_id"]))].append(row)
    mutable_counts = defaultdict(set)
    validation_counts = defaultdict(set)
    for (arm, condition_id), rows in by_arm_cell.items():
        if arm in MUTABLE_ARMS:
            mutable_counts[condition_id].add(len(rows))
            validation_counts[condition_id].add(sum(int(row["exact_validation_calls"]) for row in rows))
    return {
        "matched_across_mutable_arms": all(len(counts) == 1 for counts in mutable_counts.values()),
        "exact_validation_calls_matched": all(len(counts) == 1 for counts in validation_counts.values()),
        "cells_checked": len(mutable_counts),
    }


def build_artifact(
    *,
    root: Path | str,
    tests_added_or_reused: Sequence[str],
    checkpoint_dir: Path | str,
) -> JsonDict:
    """Build the conductor-visible Exp5617 artifact."""

    root_path = Path(root)
    gates = freeze_structured_gates(root_path)
    fixture = load_frozen_fixture(gates, root_path)
    result = run_duration_map(fixture, checkpoint_dir=checkpoint_dir, seeds=DEFAULT_LEARNER_SEEDS)
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
        "fixture_hash": fixture.fixture_hash,
        "structured_gates": gates,
        "models_tested": result["models_tested"],
        "duration_cells": result["duration_cells"],
        "seeds": result["seeds"],
        "instances_per_condition": result["instances_per_condition"],
        "ale_by_arm_and_cell": result["ale_by_arm_and_cell"],
        "instability_by_arm_and_cell": result["instability_by_arm_and_cell"],
        "transient_error_by_arm_and_cell": result["transient_error_by_arm_and_cell"],
        "backward_retention_by_arm": result["backward_retention_by_arm"],
        "forward_transfer_by_arm": result["forward_transfer_by_arm"],
        "time_to_valid_adaptation_by_arm_and_cell": result["time_to_valid_adaptation_by_arm_and_cell"],
        "update_rollback_counts_by_arm_and_cell": result["update_rollback_counts_by_arm_and_cell"],
        "unsafe_false_accept_count": result["unsafe_false_accept_count"],
        "empirical_switch_durations": result["empirical_switch_durations"],
        "critical_duration_fit_r2": result["critical_duration_fit_r2"],
        "nondegenerate_switch_cases": result["nondegenerate_switch_cases"],
        "critical_task_duration": result["critical_task_duration"],
        "lazy_identity_guard_passed": result["lazy_identity_guard_passed"],
        "control_credit_guard": result["control_credit_guard"],
        "optimization_budget": result["optimization_budget"],
        "immutable_update_ledger": result["immutable_update_ledger"],
        "checkpoint_receipts": result["checkpoint_receipts"],
        "selection_protocol": result["selection_protocol"],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": result["seeds"],
        "tests_added_or_reused": list(tests_added_or_reused),
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "llm_invoked": False,
        "causal_memory_used": False,
        "pace_used": False,
        "llm_weight_training": False,
        "external_teacher_used": False,
        "honest_verdict": "",
        "reproducibility_checksum": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the artifact contradicts the REQ-KAN-5617 gates."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5617 artifact: " + "; ".join(errors))
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
    if artifact.get("models_tested") != list(ARM_NAMES):
        errors.append("models_tested")
    if artifact.get("duration_cells") != list(DURATION_CELLS):
        errors.append("duration_cells")
    if len(artifact.get("seeds", [])) < 5:
        errors.append("seeds")
    instances = artifact.get("instances_per_condition")
    if not isinstance(instances, Mapping) or int(instances.get("replicated_heldout_streams", 0)) < 32:
        errors.append("instances_per_condition")
    for field in (
        "ale_by_arm_and_cell",
        "instability_by_arm_and_cell",
        "transient_error_by_arm_and_cell",
    ):
        if set((artifact.get(field) or {}).keys()) != set(ARM_NAMES):
            errors.append(field)
    if set((artifact.get("backward_retention_by_arm") or {}).keys()) != set(ARM_NAMES):
        errors.append("backward_retention_by_arm")
    if set((artifact.get("forward_transfer_by_arm") or {}).keys()) != set(ARM_NAMES):
        errors.append("forward_transfer_by_arm")
    unsafe = artifact.get("unsafe_false_accept_count")
    if not isinstance(unsafe, Mapping) or unsafe.get("total") != 0:
        errors.append("unsafe_false_accept_count")
    if artifact.get("lazy_identity_guard_passed") is not True:
        errors.append("lazy_identity_guard_passed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    fit = artifact.get("critical_duration_fit_r2")
    if not isinstance(fit, (int, float)) or not 0.0 <= float(fit) <= 1.0:
        errors.append("critical_duration_fit_r2")
    if artifact.get("critical_task_duration") is not None and len(artifact.get("nondegenerate_switch_cases", [])) < 2:
        errors.append("nondegenerate_switch_cases")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that permits a bounded no-crossing null."""

    clean = (
        artifact.get("unsafe_false_accept_count", {}).get("total") == 0
        and artifact.get("lazy_identity_guard_passed") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
    )
    switches = len(artifact.get("nondegenerate_switch_cases", []))
    if clean and switches >= 2 and artifact.get("critical_task_duration") is not None:
        return f"complete: critical_task_duration_d{artifact['critical_task_duration']}_estimated"
    return "bounded_null: no_critical_duration_declared"


def ledger_hash(row: Mapping[str, Any]) -> str:
    """Hash one immutable ledger row while blanking the self-reference."""

    stable = dict(row)
    stable["ledger_hash"] = ""
    return "sha256:" + sha256_json(stable)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the terminal artifact while excluding its self-reference."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return "sha256:" + sha256_json(stable)


def source_file_checksums(root: Path) -> JsonDict:
    """Hash the spec, implementation, and test files backing Exp5617."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def resolve_path(root: Path, path: Path | str) -> Path:
    """Resolve repository-relative result paths without changing absolutes."""

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
    """Build the artifact and optionally write it to disk."""

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
                "critical_task_duration": artifact["critical_task_duration"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
