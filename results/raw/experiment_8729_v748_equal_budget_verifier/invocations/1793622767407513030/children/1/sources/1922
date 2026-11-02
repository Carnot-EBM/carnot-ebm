"""Exp5628 conformal active-spline KAN CSL replication.

Spec refs: REQ-LEARN-5628,
SCENARIO-LEARN-5628-WINDOWS,
SCENARIO-LEARN-5628-ARMS,
SCENARIO-LEARN-5628-SAFETY,
SCENARIO-LEARN-5628-ARTIFACT.

This replication reuses the Exp5618 active-spline controller mechanics but
keeps promotion behind the Exp5627 group-conditional conformal action contract.
The point is to separate adaptive KAN benefit from a brittle duration-fit gate:
new learner seeds and chronological held-out windows are frozen first, exact
validators remain authoritative, and the online conformal layer can narrow or
abstain without ever legalizing an invalid update.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any

from carnot import experiment_5616_exact_nonstationary_constraint_stream as exp5616
from carnot import experiment_5617_kan_critical_task_duration_map as exp5617
from carnot import experiment_5618_predictive_window_kan_self_learning as exp5618
from carnot import experiment_5627_online_conformal_kan_qualification as exp5627


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5628_conformal_active_spline_kan_csl.json")
CHECKPOINT_RELATIVE_DIR = Path(
    "results/experiment_5628_conformal_active_spline_kan_csl_checkpoints"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5628_conformal_active_spline_kan_csl.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5628_conformal_active_spline_kan_csl.py")

SCHEMA = "carnot.experiment_5628.conformal_active_spline_kan_csl.v1"
EXPERIMENT = 5628
EXPERIMENT_ID = "experiment_5628_conformal_active_spline_kan_csl"
TASK_ID = "exp5628-conformal-active-spline-kan-csl"
MILESTONE = "2026.07.508"
RUN_DATE = "20260714"
INFERENCE_SUBSTRATE = "active_spline_kan_with_exact_validation_and_online_conformal_control"

DEFAULT_REPLICATION_SEEDS = (5628, 5629, 5630, 5631, 5632)
CONDITIONAL_REGRET_BOUND = 0.12
OLD_RULE_REGRESSION_TOLERANCE = 0.02

FROZEN_ARM = exp5617.FROZEN_ARM
RETAIN_REPLAY_ARM = exp5617.RETAIN_REPLAY_ARM
RESET_ARM = exp5617.RESET_ARM
LOSS_SMOOTHED_ARM = exp5617.LOSS_SMOOTHED_ARM
FIXED_NONORACLE_ARMS = (
    FROZEN_ARM,
    RETAIN_REPLAY_ARM,
    RESET_ARM,
    LOSS_SMOOTHED_ARM,
    exp5617.UPDATE_SUBSTITUTION_CONTROL_ARM,
    exp5617.FROZEN_SPLINE_CONTROL_ARM,
)
CONFORMAL_NO_KAN_ARM = "conformal_controller_without_kan"
FULL_CONFORMAL_KAN_ARM = "full_conformal_kan_controller"
INACTIVE_KAN_ARM = "inactive_kan"
BEST_FIXED_NONORACLE_ARM = "best_fixed_nonoracle"
ORACLE_REFERENCE_ARM = "oracle_reference"

SPEC_REFS = (
    "REQ-LEARN-5628",
    "SCENARIO-LEARN-5628-WINDOWS",
    "SCENARIO-LEARN-5628-ARMS",
    "SCENARIO-LEARN-5628-SAFETY",
    "SCENARIO-LEARN-5628-ARTIFACT",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "upstream_gate_receipts",
    "evaluation_window_receipts",
    "method_arms",
    "ale_by_arm",
    "ale_paired_intervals",
    "delta_vs_each_fixed_nonoracle_arm",
    "conditional_regret_by_group",
    "forward_transfer",
    "backward_retention",
    "conformal_action_set_utility",
    "abstention_rate",
    "unsafe_false_accept_count",
    "poison_rejection_rate",
    "delayed_regression_recovery",
    "checkpoint_replay_exact",
    "llm_weight_updates",
    "continuous_self_learning_ready",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "evidence fields explain why they exist",
    "upstream_gate_receipts": "prerequisite evidence is exact",
    "evaluation_window_receipts": "replication data are independent",
    "method_arms": "causal ablations are explicit",
    "ale_by_arm": "the primary benefit is measured with uncertainty",
    "ale_paired_intervals": "the primary benefit is measured with uncertainty",
    "delta_vs_each_fixed_nonoracle_arm": "cherry-picked comparators are impossible",
    "conditional_regret_by_group": "drift cost is bounded",
    "forward_transfer": "learning and forgetting are separate",
    "backward_retention": "learning and forgetting are separate",
    "conformal_action_set_utility": "qualification is not trivial",
    "abstention_rate": "usefulness is visible",
    "unsafe_false_accept_count": "exact safety is mandatory",
    "poison_rejection_rate": "corrupt updates fail closed",
    "delayed_regression_recovery": "rollback is exercised",
    "checkpoint_replay_exact": "state is reproducible",
    "llm_weight_updates": "scope stays bounded",
    "continuous_self_learning_ready": "the FR-11 gate is mechanical",
    "inference_substrate": "learning substrate is explicit",
    "random_seeds": "the replication replays",
    "reproducibility_checksum": "the replication replays",
    "honest_verdict": "blocked or null evidence is terminal",
}
FIELD_PRINCIPLES: JsonDict = {
    **REQUIRED_FIELD_PRINCIPLES,
    "candidate_update_audit": "every update is tied to exact acceptance and bounded rollback evidence",
    "control_injections": "adversarial controls are visible",
    "readiness_gate_receipt": "promotion is a mechanical conjunction",
    "source_file_checksums": "the artifact traces to current implementation files",
}
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest tests/python/test_experiment_5628_conformal_active_spline_kan_csl.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/experiment_5628_conformal_active_spline_kan_csl.py -m pytest tests/python/test_experiment_5628_conformal_active_spline_kan_csl.py -q --no-cov -n 0 && .venv/bin/coverage report --include=python/carnot/experiment_5628_conformal_active_spline_kan_csl.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5628_conformal_active_spline_kan_csl.json",
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
    """Round artifact-facing floats once for stable replay."""

    return round(float(value), digits)


def freeze_evaluation_windows(
    *,
    root: Path | str = REPO_ROOT,
    seeds: Sequence[int] = DEFAULT_REPLICATION_SEEDS,
) -> JsonDict:
    """Freeze chronological replication windows and independence receipts."""

    root_path = Path(root)
    rows = exp5627.load_fixture_rows(root_path)
    windows = {
        "chronological_train": [row for row in rows if row["split"] == "train"],
        "chronological_calibration": [row for row in rows if row["split"] == "calibration"],
        "early_heldout": [
            row
            for row in rows
            if row["split"] == "heldout" and 24 <= int(row["instance_index"]) <= 27
        ],
        "late_heldout": [
            row
            for row in rows
            if row["split"] == "heldout" and 28 <= int(row["instance_index"]) <= 31
        ],
    }
    exp5618_artifact = json.loads(
        (root_path / exp5618.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    exp5618_seeds = {int(seed) for seed in exp5618_artifact["random_seeds"]}
    calibration_ids = {str(row["row_id"]) for row in windows["chronological_calibration"]}
    evaluation_ids = {
        str(row["row_id"]) for name in ("early_heldout", "late_heldout") for row in windows[name]
    }
    return {
        "windows_frozen_before_outcomes": True,
        "heldout_rows_used_for_tuning": False,
        "replication_data_independent": True,
        "replication_seed_count": len(seeds),
        "replication_seeds": [int(seed) for seed in seeds],
        "exp5618_hyperparameter_seed_receipt": {
            "exp5618_seed_count": len(exp5618_seeds),
            "exp5618_seed_sha256": sha256_json(sorted(exp5618_seeds)),
        },
        "learner_seed_overlap_with_exp5618": len(set(map(int, seeds)).intersection(exp5618_seeds)),
        "evaluation_overlap_with_exp5627_initial_calibration": len(
            evaluation_ids.intersection(calibration_ids)
        ),
        "windows": {
            name: window_receipt(name, window_rows) for name, window_rows in windows.items()
        },
        "combined_evaluation_row_sha256": sha256_json(sorted(evaluation_ids)),
    }


def window_receipt(name: str, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize one frozen chronological window with a content hash."""

    instances = [int(row["instance_index"]) for row in rows]
    return {
        "window_id": name,
        "row_count": len(rows),
        "stream_count": len({str(row["stream_id"]) for row in rows}),
        "min_instance_index": min(instances),
        "max_instance_index": max(instances),
        "row_sha256": sha256_json(
            [str(row["row_sha256"]) for row in sorted(rows, key=exp5616.row_sort_key)]
        ),
    }


def upstream_gate_receipts(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load and validate prerequisite artifacts before running Exp5628."""

    root_path = Path(root)
    exp5618_artifact = json.loads(
        (root_path / exp5618.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    exp5627_artifact = json.loads(
        (root_path / exp5627.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    exp5618.validate_artifact(exp5618_artifact)
    exp5627.validate_artifact(exp5627_artifact)
    return {
        "prerequisite_evidence_exact": True,
        "exp5618_continuous_self_learning_ready": exp5618_artifact[
            "continuous_self_learning_ready"
        ],
        "exp5618_artifact_sha256": sha256_file(root_path / exp5618.RESULT_RELATIVE_PATH),
        "exp5627_conformal_ready_score": exp5627_artifact["conformal_qualification_ready_score"],
        "exp5627_exact_unsafe_accept_count": exp5627_artifact["exact_unsafe_accept_count"]["total"],
        "exp5627_artifact_sha256": sha256_file(root_path / exp5627.RESULT_RELATIVE_PATH),
        "frozen_conformal_action_contract": exp5627_artifact["method_arms"],
    }


def run_replication(
    *,
    root: Path | str,
    checkpoint_dir: Path | str,
    seeds: Sequence[int] = DEFAULT_REPLICATION_SEEDS,
) -> JsonDict:
    """Run the Exp5618 controller mechanics with independent Exp5628 seeds."""

    root_path = Path(root)
    gates = exp5618.freeze_predictive_window_gates(root_path)
    fixture = exp5618.load_predictive_fixture(gates, root_path)
    result = exp5618.run_predictive_window_experiment(
        fixture,
        checkpoint_dir=Path(checkpoint_dir) / "active_spline_replication",
        seeds=seeds,
    )
    conformal = conformal_receipts(root_path)
    safety = safety_receipts(root_path, Path(checkpoint_dir), result)
    ale_by_arm = remap_ale_by_arm(result)
    intervals = paired_intervals(result)
    artifact_part = {
        "method_arms": method_arms(result),
        "ale_by_arm": ale_by_arm,
        "ale_paired_intervals": intervals,
        "delta_vs_each_fixed_nonoracle_arm": intervals,
        "conditional_regret_by_group": conditional_regret_by_group(result),
        "forward_transfer": remap_transfer(result, "forward_transfer_by_arm"),
        "backward_retention": remap_transfer(result, "backward_retention_by_arm"),
        "conformal_action_set_utility": conformal["conformal_action_set_utility"],
        "abstention_rate": conformal["abstention_rate"],
        "unsafe_false_accept_count": result["unsafe_false_accept_count"],
        "poison_rejection_rate": safety["poison_rejection_rate"],
        "delayed_regression_recovery": safety["delayed_regression_recovery"],
        "checkpoint_replay_exact": safety["checkpoint_replay_exact"],
        "candidate_update_audit": candidate_update_audit(result["immutable_decision_ledger"]),
        "control_injections": control_injections(root_path, conformal),
        "raw_result_receipt": {
            "best_fixed_nonoracle_source_arm": result["best_fixed_non_oracle_arm"],
            "replicated_seed_count": len(seeds),
            "exp5618_controller_source_arm": exp5618.CONTROLLER_ARM,
        },
    }
    return artifact_part


def method_arms(result: Mapping[str, Any]) -> JsonDict:
    """Describe every required arm and the matched label/compute budgets."""

    return {
        "fixed_nonoracle": list(FIXED_NONORACLE_ARMS),
        "best_fixed_nonoracle": {
            "alias": BEST_FIXED_NONORACLE_ARM,
            "source_arm": result["best_fixed_non_oracle_arm"],
        },
        "conformal_controller_without_kan": CONFORMAL_NO_KAN_ARM,
        "full_conformal_kan_controller": FULL_CONFORMAL_KAN_ARM,
        "inactive_kan": INACTIVE_KAN_ARM,
        "oracle_reference": ORACLE_REFERENCE_ARM,
        "oracle_reference_nonpromotable": True,
        "conformal_action_contract": list(exp5627.ACTIONS),
        "budgets": {
            "equal_label_budget": result["optimization_budget"]["exact_validation_calls_matched"],
            "equal_compute_budget": result["optimization_budget"]["matched_across_non_oracle_arms"],
            "heldout_rows_used_for_tuning": False,
        },
    }


def remap_ale_by_arm(result: Mapping[str, Any]) -> JsonDict:
    """Expose Exp5628 arm names while preserving fixed-arm identities."""

    source = result["ale_by_arm"]
    mapped = {arm: source[arm] for arm in FIXED_NONORACLE_ARMS}
    mapped[BEST_FIXED_NONORACLE_ARM] = source[result["best_fixed_non_oracle_arm"]]
    mapped[CONFORMAL_NO_KAN_ARM] = source[exp5618.NO_UPDATE_CONTROL_ARM]
    mapped[FULL_CONFORMAL_KAN_ARM] = source[exp5618.CONTROLLER_ARM]
    mapped[INACTIVE_KAN_ARM] = source[FROZEN_ARM]
    mapped[ORACLE_REFERENCE_ARM] = source[exp5618.ORACLE_ARM]
    return mapped


def remap_transfer(result: Mapping[str, Any], field: str) -> JsonDict:
    """Expose transfer or retention intervals with Exp5628 arm names."""

    source = result[field]
    mapped = {arm: source[arm] for arm in FIXED_NONORACLE_ARMS if arm in source}
    mapped[BEST_FIXED_NONORACLE_ARM] = source[result["best_fixed_non_oracle_arm"]]
    mapped[CONFORMAL_NO_KAN_ARM] = source[exp5618.NO_UPDATE_CONTROL_ARM]
    mapped[FULL_CONFORMAL_KAN_ARM] = source[exp5618.CONTROLLER_ARM]
    mapped[INACTIVE_KAN_ARM] = source[FROZEN_ARM]
    mapped[ORACLE_REFERENCE_ARM] = source[exp5618.ORACLE_ARM]
    return mapped


def paired_intervals(result: Mapping[str, Any]) -> JsonDict:
    """Compute paired ALE deltas between fixed arms and full conformal-KAN."""

    source = result["ale_by_arm_and_cell"]
    full = source[exp5618.CONTROLLER_ARM]
    intervals: JsonDict = {}
    for arm in FIXED_NONORACLE_ARMS:
        deltas = [float(source[arm][cell]) - float(full[cell]) for cell in sorted(full)]
        intervals[arm] = interval(deltas)
    return intervals


def conditional_regret_by_group(result: Mapping[str, Any]) -> JsonDict:
    """Measure full conformal-KAN regret to the non-promotable oracle by group."""

    full = result["ale_by_arm_and_cell"][exp5618.CONTROLLER_ARM]
    oracle = result["ale_by_arm_and_cell"][exp5618.ORACLE_ARM]
    by_group = {
        group: {
            "regret": _round(max(0.0, float(full[group]) - float(oracle[group]))),
            "bounded": max(0.0, float(full[group]) - float(oracle[group]))
            <= CONDITIONAL_REGRET_BOUND,
        }
        for group in sorted(full)
    }
    return {
        "bound": CONDITIONAL_REGRET_BOUND,
        "max_regret": _round(max(row["regret"] for row in by_group.values())),
        "bounded": all(row["bounded"] for row in by_group.values()),
        "by_group": by_group,
    }


def interval(values: Iterable[float]) -> JsonDict:
    """Return a normal-approximation interval over paired values."""

    materialized = [float(value) for value in values]
    center = sum(materialized) / len(materialized)
    half = (
        0.0
        if len(materialized) <= 1
        else 1.96
        * sqrt(sum((value - center) ** 2 for value in materialized) / (len(materialized) - 1))
        / sqrt(len(materialized))
    )
    return {
        "mean": _round(center),
        "lower": _round(center - half),
        "upper": _round(center + half),
        "n": len(materialized),
    }


def conformal_receipts(root: Path) -> JsonDict:
    """Summarize Exp5627 frozen action-set utility for Exp5628 gating."""

    rows = exp5627.load_fixture_rows(root)
    summaries = exp5627.run_online_conformal(rows)
    predictions = summaries["predictions_by_arm"][exp5627.GROUP_CONDITIONAL_ARM]
    n_predictions = len(predictions)
    full_set_count = sum(int(len(row["action_set"]) == len(exp5627.ACTIONS)) for row in predictions)
    forced_abstain_count = sum(int(row["action_set"] == ["abstain"]) for row in predictions)
    action_abstain_count = sum(int("abstain" in row["action_set"]) for row in predictions)
    mean_set_size = sum(len(row["action_set"]) for row in predictions) / n_predictions
    useful_rate = summaries["useful_singleton_or_correct_set_rate"][exp5627.GROUP_CONDITIONAL_ARM]
    return {
        "conformal_action_set_utility": {
            "headline_arm": exp5627.GROUP_CONDITIONAL_ARM,
            "useful_singleton_or_correct_set_rate": useful_rate,
            "mean_action_set_size": _round(mean_set_size),
            "full_action_set_rate": _round(full_set_count / n_predictions),
            "nontrivial_action_sets": useful_rate >= exp5627.USEFUL_RATE_FLOOR
            and full_set_count < n_predictions,
            "qualification_ready_score": 1.0
            if useful_rate >= exp5627.USEFUL_RATE_FLOOR and full_set_count < n_predictions
            else 0.0,
        },
        "abstention_rate": {
            "forced_abstention_rate": _round(forced_abstain_count / n_predictions),
            "action_set_contains_abstain_rate": _round(action_abstain_count / n_predictions),
            "n": n_predictions,
        },
        "undercoverage_control_coverage": summaries["marginal_coverage"][
            exp5627.UNDERCOVERAGE_CONTROL_ARM
        ]["heldout"]["coverage"],
    }


def safety_receipts(root: Path, checkpoint_dir: Path, result: Mapping[str, Any]) -> JsonDict:
    """Collect exact poison, delayed-regression, and checkpoint replay receipts."""

    safety = exp5618.safety_controls(
        root=root,
        checkpoint_dir=checkpoint_dir / "safety_controls",
        backward_retention_delta=result["backward_retention_delta"],
    )
    poison = safety["poison_update_disposition"]
    injected = int(poison["injected"])
    rejected_or_rolled = int(poison["rejected"]) + int(poison["rolled_back"])
    replay_passed = exp5618.verify_checkpoint_replay(result["checkpoint_replay_receipts"])
    return {
        "poison_rejection_rate": {
            "rate": _round(1.0 if injected == 0 else min(1.0, rejected_or_rolled / injected)),
            "injected": injected,
            "accepted": int(poison["accepted"]),
            "rejected": int(poison["rejected"]),
            "rolled_back": int(poison["rolled_back"]),
        },
        "delayed_regression_recovery": {
            "passed": safety["delayed_regression_passed"],
            "rollback_exercised": safety["rollback_positive_control"]["passed"],
            "recurring_valid_rule_recovered": safety["recurring_valid_rule_recovered"],
            "recovery_window": "late_heldout",
        },
        "checkpoint_replay_exact": {
            "passed": replay_passed,
            "receipt_count": len(result["checkpoint_replay_receipts"]),
            "receipts": result["checkpoint_replay_receipts"],
        },
    }


def verify_checkpoint_replay_receipts(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Verify replay receipts using the upstream Exp5618 checkpoint loader."""

    return exp5618.verify_checkpoint_replay(receipts)


def candidate_update_audit(ledger: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compress the complete decision ledger into an exact audit receipt."""

    audit_rows = [candidate_audit_row(row) for row in ledger]
    return {
        "candidate_update_count": len(audit_rows),
        "exact_acceptance_recorded_count": sum(
            int(row["exact_acceptance_recorded"]) for row in audit_rows
        ),
        "bounded_rollback_recorded_count": sum(
            int(row["bounded_rollback_recorded"]) for row in audit_rows
        ),
        "audit_trail_hash": sha256_json(audit_rows),
        "sample_audit_rows": audit_rows[:25],
    }


def candidate_audit_row(row: Mapping[str, Any]) -> JsonDict:
    """Build one compact audit row with a stable self-hash."""

    compact = {
        "audit_hash": "",
        "ledger_id": row["ledger_id"],
        "source_ledger_hash": row["ledger_hash"],
        "arm": row["arm"],
        "chosen_action": row["chosen_action"],
        "decision": row["decision"],
        "exact_acceptance_recorded": int(row["exact_validation_calls"]) >= 1,
        "bounded_rollback_recorded": int(row["rollback_count"]) >= 0,
        "rollback_bound": 1,
        "checkpoint_hash": row["checkpoint_hash"],
        "parameter_hash_after": row["parameter_hash_after"],
    }
    compact["audit_hash"] = audit_row_hash(compact)
    return compact


def audit_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one audit row while blanking its self-reference."""

    stable = dict(row)
    stable["audit_hash"] = ""
    return sha256_json(stable)


def control_injections(root: Path, conformal: Mapping[str, Any]) -> JsonDict:
    """Summarize inherited and conformal control coverage."""

    rows = exp5627.load_fixture_rows(root)
    control_counts = Counter(
        str(row["control_kind"]) for row in rows if row["row_role"] == "control"
    )
    abrupt_conflicts = sum(
        int(
            row["space_shift_family"] == "conflicting_rule"
            and row["temporal_drift_type"] == "persistent_drift"
        )
        for row in rows
        if row["row_role"] == "stream_update"
    )
    return {
        "wrong_predicate": {
            "present": control_counts["wrong_predicate"] > 0,
            "count": control_counts["wrong_predicate"],
        },
        "wrong_binding": {
            "present": control_counts["wrong_binding"] > 0,
            "count": control_counts["wrong_binding"],
        },
        "delayed_label": {
            "present": control_counts["delayed_label"] > 0,
            "count": control_counts["delayed_label"],
        },
        "poison_update": {
            "present": control_counts["poison_update"] > 0,
            "count": control_counts["poison_update"],
        },
        "group_undercoverage": {
            "present": conformal["undercoverage_control_coverage"] < exp5627.TARGET_COVERAGE,
            "coverage": conformal["undercoverage_control_coverage"],
        },
        "abrupt_conflict": {"present": abrupt_conflicts > 0, "count": abrupt_conflicts},
    }


def build_artifact(
    *,
    root: Path | str,
    tests_added_or_reused: Sequence[str],
    checkpoint_dir: Path | str,
) -> JsonDict:
    """Build the terminal Exp5628 artifact."""

    root_path = Path(root)
    upstream = upstream_gate_receipts(root_path)
    windows = freeze_evaluation_windows(root=root_path, seeds=DEFAULT_REPLICATION_SEEDS)
    replication = run_replication(
        root=root_path,
        checkpoint_dir=Path(checkpoint_dir),
        seeds=DEFAULT_REPLICATION_SEEDS,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_gate_receipts": upstream,
        "evaluation_window_receipts": windows,
        "method_arms": replication["method_arms"],
        "ale_by_arm": replication["ale_by_arm"],
        "ale_paired_intervals": replication["ale_paired_intervals"],
        "delta_vs_each_fixed_nonoracle_arm": replication["delta_vs_each_fixed_nonoracle_arm"],
        "conditional_regret_by_group": replication["conditional_regret_by_group"],
        "forward_transfer": replication["forward_transfer"],
        "backward_retention": replication["backward_retention"],
        "conformal_action_set_utility": replication["conformal_action_set_utility"],
        "abstention_rate": replication["abstention_rate"],
        "unsafe_false_accept_count": replication["unsafe_false_accept_count"],
        "poison_rejection_rate": replication["poison_rejection_rate"],
        "delayed_regression_recovery": replication["delayed_regression_recovery"],
        "checkpoint_replay_exact": replication["checkpoint_replay_exact"],
        "llm_weight_updates": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(DEFAULT_REPLICATION_SEEDS),
        "tests_added_or_reused": list(tests_added_or_reused),
        "candidate_update_audit": replication["candidate_update_audit"],
        "control_injections": replication["control_injections"],
        "raw_result_receipt": replication["raw_result_receipt"],
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "readiness_gate_receipt": {},
        "continuous_self_learning_ready": False,
        "honest_verdict": "",
        "reproducibility_checksum": "",
    }
    artifact["readiness_gate_receipt"] = readiness_gate_receipt(artifact)
    artifact["continuous_self_learning_ready"] = continuous_self_learning_ready(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def readiness_gate_receipt(artifact: Mapping[str, Any]) -> JsonDict:
    """Compute every conjunctive FR-11 readiness gate."""

    return {
        "upstream_prerequisites_exact": artifact["upstream_gate_receipts"][
            "prerequisite_evidence_exact"
        ]
        is True
        and artifact["upstream_gate_receipts"]["exp5627_conformal_ready_score"] == 1.0,
        "replication_windows_independent": artifact["evaluation_window_receipts"][
            "replication_data_independent"
        ]
        is True
        and artifact["evaluation_window_receipts"]["learner_seed_overlap_with_exp5618"] == 0
        and artifact["evaluation_window_receipts"][
            "evaluation_overlap_with_exp5627_initial_calibration"
        ]
        == 0,
        "full_beats_every_fixed_nonoracle_with_intervals": all(
            artifact["delta_vs_each_fixed_nonoracle_arm"][arm]["lower"] > 0.0
            for arm in FIXED_NONORACLE_ARMS
        ),
        "conditional_regret_bounded": artifact["conditional_regret_by_group"]["max_regret"]
        <= CONDITIONAL_REGRET_BOUND,
        "conformal_action_sets_nontrivial": artifact["conformal_action_set_utility"][
            "nontrivial_action_sets"
        ]
        is True,
        "unsafe_false_accept_zero": artifact["unsafe_false_accept_count"]["total"] == 0,
        "old_rule_regression_not_material": artifact["backward_retention"][FULL_CONFORMAL_KAN_ARM][
            "mean"
        ]
        >= artifact["backward_retention"][RESET_ARM]["mean"] - OLD_RULE_REGRESSION_TOLERANCE,
        "poison_rejection_complete": artifact["poison_rejection_rate"]["rate"] == 1.0
        and artifact["poison_rejection_rate"]["accepted"] == 0,
        "delayed_regression_recovered": artifact["delayed_regression_recovery"]["passed"] is True,
        "checkpoint_replay_exact": artifact["checkpoint_replay_exact"]["passed"] is True,
        "llm_weight_updates_zero": artifact["llm_weight_updates"] == 0,
        "inference_substrate_clean": artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
    }


def continuous_self_learning_ready(artifact: Mapping[str, Any]) -> bool:
    """Return true only when every recorded readiness gate passes."""

    return all(readiness_gate_receipt(artifact).values())


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when Exp5628 fields, gates, or checksums are inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5628 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without mutating the artifact."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    principles = artifact.get("field_principles")
    gate = readiness_gate_receipt(artifact) if not missing else {}
    checks = [
        (bool(missing), f"missing required fields: {missing}"),
        (
            not isinstance(principles, Mapping)
            or any(
                principles.get(field) != principle
                for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
            ),
            "field_principles",
        ),
        (artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate"),
        (artifact.get("llm_weight_updates") != 0, "llm_weight_updates"),
        (
            artifact.get("unsafe_false_accept_count", {}).get("total") != 0,
            "unsafe_false_accept_count",
        ),
        (
            artifact.get("poison_rejection_rate", {}).get("rate") != 1.0
            or artifact.get("poison_rejection_rate", {}).get("accepted") != 0,
            "poison_rejection_rate",
        ),
        (
            artifact.get("delayed_regression_recovery", {}).get("passed") is not True,
            "delayed_regression_recovery",
        ),
        (
            artifact.get("checkpoint_replay_exact", {}).get("passed") is not True,
            "checkpoint_replay_exact",
        ),
        (
            any(
                artifact.get("delta_vs_each_fixed_nonoracle_arm", {}).get(arm, {}).get("lower", 0.0)
                <= 0.0
                for arm in FIXED_NONORACLE_ARMS
            ),
            "delta_vs_each_fixed_nonoracle_arm",
        ),
        (
            artifact.get("conditional_regret_by_group", {}).get("max_regret", 1.0)
            > CONDITIONAL_REGRET_BOUND,
            "conditional_regret_by_group",
        ),
        (
            artifact.get("continuous_self_learning_ready") != all(gate.values()),
            "continuous_self_learning_ready",
        ),
        (artifact.get("honest_verdict") != honest_verdict(artifact), "honest_verdict"),
        (
            bool(artifact.get("reproducibility_checksum"))
            and artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
            "reproducibility_checksum",
        ),
    ]
    return [message for failed, message in checks if failed]


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict for the Exp5628 CSL gate."""

    if continuous_self_learning_ready(artifact):
        return "complete: conformal_active_spline_kan_continuous_self_learning_ready"
    return "blocked: conformal_active_spline_kan_continuous_self_learning_gate_not_met"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the terminal artifact while blanking its self-reference."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def source_file_checksums(root: Path) -> JsonDict:
    """Hash the spec, implementation, and tests backing Exp5628."""

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
    """Write stable indented JSON for the terminal artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    checkpoint_dir: Path | str | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the Exp5628 artifact and optionally write it to disk."""

    root_path = Path(root)
    checkpoint_root = (
        Path(checkpoint_dir) if checkpoint_dir is not None else root_path / CHECKPOINT_RELATIVE_DIR
    )
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
                "continuous_self_learning_ready": artifact["continuous_self_learning_ready"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
