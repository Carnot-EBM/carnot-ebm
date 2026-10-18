"""Exp5616 exact nonstationary constraint-stream fixture.

Spec refs: REQ-BENCH-5616, SCENARIO-BENCH-5616-SCHEMA,
SCENARIO-BENCH-5616-CONTROLS.

This module builds a benchmark fixture, not a learner. The important boundary is
that every state and every update is labeled by a local exact verifier before
any policy or KAN is allowed to consume the stream. That keeps later retention,
adaptation, and leakage measurements from being circular model-label audits.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5616_exact_nonstationary_constraint_stream.json")
DATASET_RELATIVE_PATH = Path(
    "data/research/experiment_5616_exact_nonstationary_constraint_stream.jsonl"
)

ARTIFACT_SCHEMA = "carnot.experiment_5616.exact_nonstationary_constraint_stream.v1"
ROW_SCHEMA_VERSION = "carnot.experiment_5616.exact_nonstationary_constraint_stream.row.v1"
EXPERIMENT = 5616
EXPERIMENT_ID = "exp5616-exact-nonstationary-constraint-stream"
MILESTONE = "2026.07.507"
RUN_DATE = "20260714"
RANDOM_SEED = 5616
INFERENCE_SUBSTRATE = "deterministic_verifier"
SPEC_REFS = (
    "REQ-BENCH-5616",
    "SCENARIO-BENCH-5616-SCHEMA",
    "SCENARIO-BENCH-5616-CONTROLS",
)

SPACE_SHIFT_FAMILIES = ("shared_rule", "conflicting_rule")
TEMPORAL_DRIFT_TYPES = ("no_drift", "reversible_drift", "persistent_drift")
TASK_DURATIONS = (1, 2, 4, 8, 16, 32)
INSTANCES_PER_CONDITION = 32
SPLITS = ("train", "calibration", "heldout")
CORRUPTION_KINDS = ("wrong_predicate", "wrong_binding", "delayed_label", "poison_update")
CONTROL_KINDS = ("known_valid", *CORRUPTION_KINDS)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Every required evidence field states why it exists.",
    "dataset_path": "the fixture is durable",
    "schema_version": "rows have a stable contract",
    "space_shift_families": "domain changes are explicit",
    "temporal_drift_types": "time changes are explicit",
    "task_durations": "the duration axis is preregistered",
    "instances_per_condition": "denominators meet the evidence floor",
    "split_receipts": "leakage is excluded",
    "exact_oracle_label_count": "authority is deterministic",
    "oracle_label_error_count": "corrupted and valid controls are exact",
    "corruption_controls": "unsafe cases are represented",
    "fixture_ready_score": "only complete fixture gates can unlock learners",
    "inference_substrate": "no LLM participated",
    "random_seeds": "generation replays exactly",
    "reproducibility_checksum": "generation replays exactly",
    "honest_verdict": "a deficient fixture blocks learners",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

REQUIRED_ROW_FIELDS = (
    "schema_version",
    "row_id",
    "stream_id",
    "condition_id",
    "space_shift_family",
    "temporal_drift_type",
    "duration",
    "instance_index",
    "seed",
    "split",
    "step_index",
    "row_role",
    "control_kind",
    "state",
    "update",
    "rules",
    "state_labels",
    "update_labels",
    "expected_validator_decision",
    "accepted_by_exact_validator",
    "exact_validator_errors",
    "row_sha256",
)

ROW_ROLE_ORDER = {
    "stream_update": 0,
    "control": 1,
}
CONTROL_ORDER = {kind: index for index, kind in enumerate(("none", *CONTROL_KINDS))}


def canonical_json(value: Any) -> str:
    """Serialize JSON in one stable form so byte hashes are reproducible."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for JSON-compatible data."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest for already serialized bytes."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest over the exact bytes on disk."""

    return sha256_bytes(path.read_bytes())


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def row_content_hash(row: Mapping[str, Any]) -> str:
    """Hash one dataset row while blanking its self-referential row hash."""

    stable = deepcopy(dict(row))
    stable["row_sha256"] = ""
    return sha256_json(stable)


def expected_seed_list() -> list[int]:
    """Return the deterministic seed assigned to each independent stream."""

    return [
        stream_seed(space_index, drift_index, duration_index, instance_index)
        for space_index, _space in enumerate(SPACE_SHIFT_FAMILIES)
        for drift_index, _drift in enumerate(TEMPORAL_DRIFT_TYPES)
        for duration_index, _duration in enumerate(TASK_DURATIONS)
        for instance_index in range(INSTANCES_PER_CONDITION)
    ]


def build_dataset_rows() -> list[JsonDict]:
    """Build all stream and control rows in stable replay order."""

    rows: list[JsonDict] = []
    for space_index, space_shift_family in enumerate(SPACE_SHIFT_FAMILIES):
        for drift_index, temporal_drift_type in enumerate(TEMPORAL_DRIFT_TYPES):
            for duration_index, duration in enumerate(TASK_DURATIONS):
                for instance_index in range(INSTANCES_PER_CONDITION):
                    seed = stream_seed(space_index, drift_index, duration_index, instance_index)
                    for step_index in range(duration):
                        rows.append(
                            build_row(
                                space_shift_family=space_shift_family,
                                temporal_drift_type=temporal_drift_type,
                                duration=duration,
                                instance_index=instance_index,
                                seed=seed,
                                step_index=step_index,
                                row_role="stream_update",
                                control_kind="none",
                            )
                        )
                    valid_control = build_row(
                        space_shift_family=space_shift_family,
                        temporal_drift_type=temporal_drift_type,
                        duration=duration,
                        instance_index=instance_index,
                        seed=seed,
                        step_index=duration,
                        row_role="control",
                        control_kind="known_valid",
                    )
                    rows.append(valid_control)
                    for control_kind in CORRUPTION_KINDS:
                        rows.append(corrupt_control_row(valid_control, control_kind))
    return sorted(rows, key=row_sort_key)


def build_row(
    *,
    space_shift_family: str,
    temporal_drift_type: str,
    duration: int,
    instance_index: int,
    seed: int,
    step_index: int,
    row_role: str,
    control_kind: str,
) -> JsonDict:
    """Build one exact-labeled state/update row."""

    _require(space_shift_family in SPACE_SHIFT_FAMILIES, "space_shift_family")
    _require(temporal_drift_type in TEMPORAL_DRIFT_TYPES, "temporal_drift_type")
    _require(duration in TASK_DURATIONS, "duration")
    _require(0 <= instance_index < INSTANCES_PER_CONDITION, "instance_index")
    _require(row_role in ROW_ROLE_ORDER, "row_role")
    _require(control_kind in CONTROL_ORDER, "control_kind")

    condition_id = condition_key(space_shift_family, temporal_drift_type, duration)
    stream_id = f"exp5616_{condition_id}_i{instance_index:02d}_s{seed}"
    row_id = f"{stream_id}_t{step_index:03d}_{control_kind}"
    rules = rules_for(space_shift_family, temporal_drift_type, duration, seed, step_index)
    state = state_for(
        stream_id=stream_id,
        row_id=row_id,
        space_shift_family=space_shift_family,
        temporal_drift_type=temporal_drift_type,
        duration=duration,
        instance_index=instance_index,
        seed=seed,
        step_index=step_index,
    )
    state_labels = exact_state_labels(state, rules)
    update = {
        "update_id": f"{row_id}_update",
        "entity_id": state["entity_id"],
        "predicate_id": rules["current_rule"]["predicate_id"],
        "variable_name": rules["current_rule"]["variable_name"],
        "observed_label": state_labels["current_rule"],
        "label_step_index": step_index,
        "poison_update": False,
    }
    row: JsonDict = {
        "schema_version": ROW_SCHEMA_VERSION,
        "row_id": row_id,
        "stream_id": stream_id,
        "condition_id": condition_id,
        "space_shift_family": space_shift_family,
        "temporal_drift_type": temporal_drift_type,
        "duration": duration,
        "instance_index": instance_index,
        "seed": seed,
        "split": split_for_instance(instance_index),
        "step_index": step_index,
        "row_role": row_role,
        "control_kind": control_kind,
        "state": state,
        "update": update,
        "rules": rules,
        "state_labels": state_labels,
        "update_labels": exact_update_labels(state, update, rules, step_index)[0],
        "expected_validator_decision": "accepted",
        "accepted_by_exact_validator": False,
        "exact_validator_errors": [],
        "row_sha256": "",
    }
    return stamp_row(row)


def corrupt_control_row(valid_control: Mapping[str, Any], control_kind: str) -> JsonDict:
    """Return one controlled invalid update derived from a known-valid control."""

    _require(control_kind in CORRUPTION_KINDS, "control_kind")
    row = deepcopy(dict(valid_control))
    stream_id = str(row["stream_id"])
    row["control_kind"] = control_kind
    row["row_id"] = f"{stream_id}_t{int(row['step_index']):03d}_{control_kind}"
    row["state"]["state_id"] = f"{row['row_id']}_state"
    row["update"]["update_id"] = f"{row['row_id']}_update"
    if control_kind == "wrong_predicate":
        row["update"]["predicate_id"] = "wrong_predicate"
    elif control_kind == "wrong_binding":
        row["update"]["entity_id"] = f"{row['state']['entity_id']}_wrong"
    elif control_kind == "delayed_label":
        row["update"]["label_step_index"] = int(row["step_index"]) - 1
    elif control_kind == "poison_update":
        row["update"]["poison_update"] = True
    row["expected_validator_decision"] = "rejected"
    row["accepted_by_exact_validator"] = False
    row["exact_validator_errors"] = []
    row["row_sha256"] = ""
    labels, _errors = exact_update_labels(
        row["state"], row["update"], row["rules"], int(row["step_index"])
    )
    row["update_labels"] = labels
    return stamp_row(row)


def stamp_row(row: Mapping[str, Any]) -> JsonDict:
    """Attach exact validator output and the final deterministic row hash."""

    stamped = deepcopy(dict(row))
    validation = validate_dataset_row(stamped)
    stamped["accepted_by_exact_validator"] = validation["accepted"]
    stamped["exact_validator_errors"] = validation["errors"]
    stamped["row_sha256"] = row_content_hash(stamped)
    return stamped


def state_for(
    *,
    stream_id: str,
    row_id: str,
    space_shift_family: str,
    temporal_drift_type: str,
    duration: int,
    instance_index: int,
    seed: int,
    step_index: int,
) -> JsonDict:
    """Construct a bounded state whose score is deterministic from stream metadata."""

    space_index = SPACE_SHIFT_FAMILIES.index(space_shift_family)
    drift_index = TEMPORAL_DRIFT_TYPES.index(temporal_drift_type)
    score = (
        seed * 37
        + step_index * 19
        + duration * 11
        + instance_index * 5
        + space_index * 7
        + drift_index * 13
    ) % 101
    return {
        "state_id": f"{row_id}_state",
        "stream_id": stream_id,
        "entity_id": f"entity_{seed}_{step_index:03d}",
        "domain_space": "source_domain" if space_shift_family == "shared_rule" else "shifted_domain",
        "variables": {"score": score},
    }


def rules_for(
    space_shift_family: str,
    temporal_drift_type: str,
    duration: int,
    seed: int,
    step_index: int,
) -> JsonDict:
    """Return old, current, and future rule objects for one row."""

    old_rule = baseline_old_rule(seed)
    shifted_rule = shifted_rule_for(space_shift_family, seed, drifted=False)
    drifted_rule = shifted_rule_for(space_shift_family, seed, drifted=True)
    if temporal_drift_type == "no_drift":
        current_rule = shifted_rule
        future_rule = shifted_rule
    elif temporal_drift_type == "persistent_drift":
        current_rule = drifted_rule
        future_rule = drifted_rule
    else:
        current_rule = drifted_rule if step_index < duration else shifted_rule
        future_rule = shifted_rule
    return {
        "old_rule": deepcopy(old_rule),
        "current_rule": deepcopy(current_rule),
        "future_rule": deepcopy(future_rule),
    }


def baseline_old_rule(seed: int) -> JsonDict:
    """Return the retained source-domain rule used for old-rule labels."""

    threshold = base_threshold(seed)
    return rule(
        rule_id=f"old_score_at_least_{threshold}",
        family="source_domain_baseline",
        predicate_id="score_at_least",
        comparator=">=",
        threshold=threshold,
        variable_name="score",
    )


def shifted_rule_for(space_shift_family: str, seed: int, *, drifted: bool) -> JsonDict:
    """Return the domain-space rule, optionally with temporal threshold drift."""

    threshold = base_threshold(seed)
    drift = drift_delta(seed) if drifted else 0
    if space_shift_family == "shared_rule":
        return rule(
            rule_id=f"shared_score_at_least_{threshold + drift}",
            family="shared_rule_space_shift",
            predicate_id="score_at_least",
            comparator=">=",
            threshold=min(100, threshold + drift),
            variable_name="score",
        )
    return rule(
        rule_id=f"conflicting_score_at_most_{threshold - drift}",
        family="conflicting_rule_space_shift",
        predicate_id="score_at_most",
        comparator="<=",
        threshold=max(0, threshold - drift),
        variable_name="score",
    )


def rule(
    *,
    rule_id: str,
    family: str,
    predicate_id: str,
    comparator: str,
    threshold: int,
    variable_name: str,
) -> JsonDict:
    """Build one schema-shaped exact predicate rule."""

    return {
        "rule_id": rule_id,
        "family": family,
        "predicate_id": predicate_id,
        "comparator": comparator,
        "threshold": int(threshold),
        "variable_name": variable_name,
    }


def exact_state_labels(state: Mapping[str, Any], rules: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    """Evaluate old, current, and future rules on one state."""

    return {
        label_name: evaluate_rule(state, rule_row)
        for label_name, rule_row in rules.items()
    }


def exact_update_labels(
    state: Mapping[str, Any],
    update: Mapping[str, Any],
    rules: Mapping[str, Mapping[str, Any]],
    step_index: int,
) -> tuple[dict[str, bool], dict[str, list[str]]]:
    """Evaluate whether one update is valid under old, current, and future rules."""

    labels: dict[str, bool] = {}
    errors: dict[str, list[str]] = {}
    for label_name, rule_row in rules.items():
        rule_errors = update_errors(state, update, rule_row, step_index)
        labels[label_name] = not rule_errors
        errors[label_name] = rule_errors
    return labels, errors


def evaluate_rule(state: Mapping[str, Any], rule_row: Mapping[str, Any]) -> bool:
    """Evaluate a bounded integer predicate exactly."""

    variables = state.get("variables", {})
    score = int(variables[str(rule_row["variable_name"])])
    threshold = int(rule_row["threshold"])
    comparator = str(rule_row["comparator"])
    if comparator == ">=":
        return score >= threshold
    if comparator == "<=":
        return score <= threshold
    raise ValueError(f"unsupported_comparator:{comparator}")


def update_errors(
    state: Mapping[str, Any],
    update: Mapping[str, Any],
    rule_row: Mapping[str, Any],
    step_index: int,
) -> list[str]:
    """Return exact reasons an update cannot be accepted for a rule."""

    errors: list[str] = []
    if update.get("predicate_id") != rule_row.get("predicate_id"):
        errors.append("wrong_predicate")
    if update.get("entity_id") != state.get("entity_id"):
        errors.append("wrong_binding")
    if update.get("variable_name") != rule_row.get("variable_name"):
        errors.append("wrong_binding")
    if int(update.get("label_step_index", -999999)) != int(step_index):
        errors.append("delayed_label")
    if update.get("poison_update") is True:
        errors.append("poison_update")
    if bool(update.get("observed_label")) != evaluate_rule(state, rule_row):
        errors.append("label_mismatch")
    return sorted(set(errors))


def validate_dataset_row(row: Mapping[str, Any]) -> JsonDict:
    """Recompute exact labels and update validity for one dataset row."""

    errors = [f"missing:{field}" for field in REQUIRED_ROW_FIELDS if field not in row]
    if errors:
        return {"accepted": False, "errors": errors}
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        errors.append("schema_version")
    if row.get("space_shift_family") not in SPACE_SHIFT_FAMILIES:
        errors.append("space_shift_family")
    if row.get("temporal_drift_type") not in TEMPORAL_DRIFT_TYPES:
        errors.append("temporal_drift_type")
    if row.get("duration") not in TASK_DURATIONS:
        errors.append("duration")
    if row.get("split") not in SPLITS:
        errors.append("split")
    expected_state_labels = exact_state_labels(row["state"], row["rules"])
    if row.get("state_labels") != expected_state_labels:
        errors.append("state_labels")
    expected_update_labels, update_error_map = exact_update_labels(
        row["state"], row["update"], row["rules"], int(row["step_index"])
    )
    if row.get("update_labels") != expected_update_labels:
        errors.append("update_labels")
    if expected_update_labels.get("current_rule") is not True:
        errors.extend(update_error_map.get("current_rule", ["current_update_rejected"]))
    accepted_without_receipt_check = not errors
    if row.get("row_sha256") and row.get("accepted_by_exact_validator") != accepted_without_receipt_check:
        errors.append("accepted_by_exact_validator")
    expected_decision = "accepted" if accepted_without_receipt_check else "rejected"
    if row.get("expected_validator_decision") != expected_decision:
        errors.append("expected_validator_decision")
    return {"accepted": not errors, "errors": sorted(set(errors))}


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate dataset counts, split receipts, controls, and oracle receipts."""

    validations = [validate_dataset_row(row) for row in rows]
    oracle_label_error_count = sum(
        int(validation["accepted"] != (row.get("expected_validator_decision") == "accepted"))
        for row, validation in zip(rows, validations, strict=True)
    )
    stream_ids = ordered_unique(str(row["stream_id"]) for row in rows)
    random_seeds = ordered_unique_int(int(row["seed"]) for row in rows if row["row_role"] == "stream_update")
    stream_condition_counts = {
        condition: len(
            {
                str(row["stream_id"])
                for row in rows
                if row["condition_id"] == condition and row["row_role"] == "stream_update"
            }
        )
        for condition in condition_keys()
    }
    family_duration_counts = {
        f"{space}|d{duration}": len(
            {
                str(row["stream_id"])
                for row in rows
                if row["space_shift_family"] == space
                and int(row["duration"]) == duration
                and row["row_role"] == "stream_update"
            }
        )
        for space in SPACE_SHIFT_FAMILIES
        for duration in TASK_DURATIONS
    }
    split_receipts = build_split_receipts(rows)
    corruption_controls = summarize_corruption_controls(rows, validations)
    schema_validation_passed = all(row.get("schema_version") == ROW_SCHEMA_VERSION for row in rows)
    counts_passed = (
        stream_condition_counts
        == {condition: INSTANCES_PER_CONDITION for condition in condition_keys()}
    )
    split_validation_passed = (
        split_receipts["stream_id_overlap_count"] == 0
        and split_receipts["state_id_overlap_count"] == 0
        and split_receipts["update_id_overlap_count"] == 0
        and all(
            counts == {"calibration": 8, "heldout": 8, "train": 16}
            for counts in split_receipts["per_condition_streams"].values()
        )
    )
    oracle_validation_passed = oracle_label_error_count == 0
    replay_validation_passed = rows == sorted(rows, key=row_sort_key)
    readiness_gates = {
        "schema_validation_passed": schema_validation_passed,
        "count_validation_passed": counts_passed,
        "split_validation_passed": split_validation_passed,
        "oracle_validation_passed": oracle_validation_passed,
        "replay_validation_passed": replay_validation_passed,
    }
    fixture_ready_score = 1.0 if all(readiness_gates.values()) else 0.0
    return {
        "dataset_row_count": len(rows),
        "stream_count": len(stream_ids),
        "stream_ordering": stream_ids,
        "random_seeds": random_seeds,
        "stream_condition_counts": stream_condition_counts,
        "family_duration_counts": family_duration_counts,
        "split_receipts": split_receipts,
        "exact_oracle_label_count": len(rows) * 6,
        "oracle_label_error_count": oracle_label_error_count,
        "corruption_controls": corruption_controls,
        "readiness_gates": readiness_gates,
        "fixture_ready_score": fixture_ready_score,
    }


def summarize_corruption_controls(
    rows: Sequence[Mapping[str, Any]],
    validations: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize valid and injected-control dispositions."""

    summary: JsonDict = {
        "known_valid": {"injected": 0, "accepted": 0, "rejected": 0},
    }
    for kind in CORRUPTION_KINDS:
        summary[kind] = {"injected": 0, "accepted": 0, "rejected": 0}
    for row, validation in zip(rows, validations, strict=True):
        kind = str(row.get("control_kind"))
        if kind == "none":
            continue
        accepted = validation["accepted"] is True
        summary[kind]["injected"] += 1
        summary[kind]["accepted"] += int(accepted)
        summary[kind]["rejected"] += int(not accepted)
    return summary


def build_split_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build leakage receipts for train, calibration, and held-out splits."""

    streams_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    states_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    updates_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    per_condition: dict[str, dict[str, set[str]]] = {
        condition: {split: set() for split in SPLITS} for condition in condition_keys()
    }
    for row in rows:
        split = str(row["split"])
        streams_by_split[split].add(str(row["stream_id"]))
        states_by_split[split].add(str(row["state"]["state_id"]))
        updates_by_split[split].add(str(row["update"]["update_id"]))
        if row["row_role"] == "stream_update":
            per_condition[str(row["condition_id"])][split].add(str(row["stream_id"]))
    return {
        "split_names": list(SPLITS),
        "streams_per_split": {
            "calibration": len(streams_by_split["calibration"]),
            "heldout": len(streams_by_split["heldout"]),
            "train": len(streams_by_split["train"]),
        },
        "per_condition_streams": {
            condition: {
                "calibration": len(counts["calibration"]),
                "heldout": len(counts["heldout"]),
                "train": len(counts["train"]),
            }
            for condition, counts in per_condition.items()
        },
        "stream_id_overlap_count": overlap_count(streams_by_split.values()),
        "state_id_overlap_count": overlap_count(states_by_split.values()),
        "update_id_overlap_count": overlap_count(updates_by_split.values()),
    }


def build_artifact(
    *,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal artifact and compute all replay gates in memory."""

    rows = build_dataset_rows()
    summary = summarize_rows(rows)
    dataset_digest = sha256_bytes(dataset_bytes(rows))
    content_hashes = {
        "dataset_sha256": dataset_digest,
        "stream_order_sha256": sha256_json(summary["stream_ordering"]),
        "random_seeds_sha256": sha256_json(summary["random_seeds"]),
        "condition_counts_sha256": sha256_json(summary["stream_condition_counts"]),
    }
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA,
        "schema_version": ROW_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "duration_s": 0.0,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "dataset_path": DATASET_RELATIVE_PATH.as_posix(),
        "dataset_sha256": dataset_digest,
        "space_shift_families": list(SPACE_SHIFT_FAMILIES),
        "temporal_drift_types": list(TEMPORAL_DRIFT_TYPES),
        "task_durations": list(TASK_DURATIONS),
        "instances_per_condition": INSTANCES_PER_CONDITION,
        "content_hashes": content_hashes,
        "replay_loader": {
            "module": "carnot.experiment_5616_exact_nonstationary_constraint_stream",
            "function": "replay_dataset",
            "dataset_path_argument": DATASET_RELATIVE_PATH.as_posix(),
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "llm_invoked": False,
        "policy_fit": False,
        "model_specs": None,
        "tests_run": [dict(row) for row in tests_run],
        "honest_verdict": honest_verdict(float(summary["fixture_ready_score"])),
        "reproducibility_checksum": "",
    }
    artifact.update(summary)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    *,
    repo_root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the dataset JSONL and terminal Exp5616 result artifact."""

    rows = build_dataset_rows()
    artifact = build_artifact(tests_run=tests_run)
    dataset_path = repo_root / DATASET_RELATIVE_PATH
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_bytes(dataset_bytes(rows))
    result_path = repo_root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    validate_artifact(artifact, repo_root=repo_root)
    return artifact


def validate_artifact(artifact: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> None:
    """Validate the terminal artifact and fail closed on overclaim or replay drift."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(artifact.get("dataset_path") == DATASET_RELATIVE_PATH.as_posix(), "dataset_path")
    _require(artifact.get("schema_version") == ROW_SCHEMA_VERSION, "schema_version")
    _require(artifact.get("space_shift_families") == list(SPACE_SHIFT_FAMILIES), "space_shift_families")
    _require(artifact.get("temporal_drift_types") == list(TEMPORAL_DRIFT_TYPES), "temporal_drift_types")
    _require(artifact.get("task_durations") == list(TASK_DURATIONS), "task_durations")
    _require(artifact.get("instances_per_condition") == INSTANCES_PER_CONDITION, "instances_per_condition")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("llm_invoked") is False, "llm_invoked")
    _require(artifact.get("policy_fit") is False, "policy_fit")
    _require(artifact.get("model_specs") is None, "model_specs")
    _require(str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")), "honest_verdict")
    _require(artifact.get("random_seeds") == expected_seed_list(), "random_seeds")
    _require(int(artifact.get("stream_count", -1)) == len(expected_seed_list()), "stream_count")
    _require(int(artifact.get("exact_oracle_label_count", -1)) == int(artifact.get("dataset_row_count", -2)) * 6, "exact_oracle_label_count")
    _require(artifact.get("content_hashes", {}).get("stream_order_sha256") == sha256_json(artifact.get("stream_ordering", [])), "content_hashes")
    _require(artifact.get("content_hashes", {}).get("random_seeds_sha256") == sha256_json(artifact.get("random_seeds", [])), "content_hashes")
    _require(float(artifact.get("fixture_ready_score", 0.0)) == expected_fixture_ready_score(artifact), "fixture_ready_score")
    if float(artifact.get("fixture_ready_score", 0.0)) == 1.0:
        _require(str(artifact.get("honest_verdict", "")).startswith("complete:"), "honest_verdict")
    dataset_path = repo_root / DATASET_RELATIVE_PATH
    _require(dataset_path.exists(), "dataset_path")
    dataset_digest = sha256_file(dataset_path)
    _require(artifact.get("dataset_sha256") == dataset_digest, "dataset_sha256")
    _require(artifact.get("content_hashes", {}).get("dataset_sha256") == dataset_digest, "dataset_sha256")
    replay = replay_dataset(dataset_path)
    _require(replay["row_count"] == artifact.get("dataset_row_count"), "replay_row_count")
    _require(replay["dataset_sha256"] == artifact.get("dataset_sha256"), "dataset_sha256")
    _require(replay["family_duration_counts"] == artifact.get("family_duration_counts"), "family_duration_counts")
    _require(replay["oracle_label_error_count"] == artifact.get("oracle_label_error_count"), "oracle_label_error_count")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def expected_fixture_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the only readiness score allowed by artifact gate fields."""

    gates = artifact.get("readiness_gates", {})
    return 1.0 if gates and all(gates.values()) and artifact.get("oracle_label_error_count") == 0 else 0.0


def replay_dataset(dataset_path: Path | str) -> JsonDict:
    """Load the durable dataset and recompute replay checks from row bytes."""

    path = Path(dataset_path)
    rows = load_dataset(path)
    summary = summarize_rows(rows)
    return {
        "dataset_path": path.as_posix(),
        "dataset_sha256": sha256_file(path),
        "row_count": len(rows),
        "rows_in_stable_order": rows == sorted(rows, key=row_sort_key),
        "family_duration_counts": summary["family_duration_counts"],
        "stream_condition_counts": summary["stream_condition_counts"],
        "split_receipts": summary["split_receipts"],
        "oracle_label_error_count": summary["oracle_label_error_count"],
        "fixture_ready_score": summary["fixture_ready_score"],
    }


def load_dataset(dataset_path: Path | str) -> list[JsonDict]:
    """Load a JSONL dataset, ignoring blank lines but not malformed JSON."""

    rows: list[JsonDict] = []
    for line in Path(dataset_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            decoded = json.loads(line)
            _require(isinstance(decoded, dict), "dataset_row_object")
            rows.append(decoded)
    return rows


def dataset_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize rows as stable JSONL bytes in replay order."""

    ordered = sorted(rows, key=row_sort_key)
    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in ordered)
    return (text + "\n").encode("utf-8")


def row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int, int, int]:
    """Return the preregistered stable row ordering key."""

    return (
        SPACE_SHIFT_FAMILIES.index(str(row["space_shift_family"])),
        TEMPORAL_DRIFT_TYPES.index(str(row["temporal_drift_type"])),
        TASK_DURATIONS.index(int(row["duration"])),
        int(row["instance_index"]),
        int(row["step_index"]),
        ROW_ROLE_ORDER[str(row["row_role"])],
        CONTROL_ORDER[str(row["control_kind"])],
    )


def condition_keys() -> list[str]:
    """Return every crossed condition key in generation order."""

    return [
        condition_key(space, drift, duration)
        for space in SPACE_SHIFT_FAMILIES
        for drift in TEMPORAL_DRIFT_TYPES
        for duration in TASK_DURATIONS
    ]


def condition_key(space_shift_family: str, temporal_drift_type: str, duration: int) -> str:
    """Build a stable condition identifier."""

    return f"{space_shift_family}|{temporal_drift_type}|d{duration}"


def stream_seed(
    space_index: int,
    drift_index: int,
    duration_index: int,
    instance_index: int,
) -> int:
    """Map a condition and instance index to a deterministic independent seed."""

    return 5_616_000 + space_index * 100_000 + drift_index * 10_000 + duration_index * 1_000 + instance_index


def base_threshold(seed: int) -> int:
    """Return a bounded baseline threshold away from the extremes."""

    return 40 + (seed % 21)


def drift_delta(seed: int) -> int:
    """Return a deterministic predicate-drift magnitude."""

    return 5 + (seed % 7)


def split_for_instance(instance_index: int) -> str:
    """Assign stream instances to disjoint train/calibration/held-out splits."""

    if instance_index < 16:
        return "train"
    if instance_index < 24:
        return "calibration"
    return "heldout"


def overlap_count(groups: Sequence[set[str]]) -> int:
    """Count IDs that appear in more than one split group."""

    membership: Counter[str] = Counter()
    for group in groups:
        membership.update(group)
    return sum(1 for count in membership.values() if count > 1)


def ordered_unique(values: Sequence[str] | Any) -> list[str]:
    """Return first-seen unique strings without disturbing generation order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def ordered_unique_int(values: Sequence[int] | Any) -> list[int]:
    """Return first-seen unique integers without disturbing generation order."""

    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def honest_verdict(fixture_ready_score: float) -> str:
    """Return a terminal verdict that cannot imply learner benefit."""

    if fixture_ready_score == 1.0:
        return "complete: exact nonstationary constraint-stream fixture ready; no learner fit"
    return "blocked: exact nonstationary constraint-stream fixture deficient; learners blocked"


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise ValueError(field)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "dataset_path": artifact["dataset_path"],
                "fixture_ready_score": artifact["fixture_ready_score"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
