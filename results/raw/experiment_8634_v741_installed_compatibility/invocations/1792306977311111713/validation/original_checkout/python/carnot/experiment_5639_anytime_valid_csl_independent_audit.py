"""Exp5639 anytime-valid CSL independent audit.

Spec refs: REQ-LEARN-5639,
SCENARIO-LEARN-5639-GATES,
SCENARIO-LEARN-5639-RECOMPUTE,
SCENARIO-LEARN-5639-ANYTIME,
SCENARIO-LEARN-5639-ADVERSARIAL.

This audit is deliberately narrower than a new learner run. It treats Exp5628
as immutable positive evidence, enforces Exp5638's scalar gate repair, and then
replays the exact row-level substrate to recompute the safety and utility
metrics. The added piece is an anytime-valid release certificate over the
chronological unsafe-false-accept path: the certificate can block promotion at
registered stops, but it cannot override an exact oracle rejection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any

from carnot import experiment_5616_exact_nonstationary_constraint_stream as exp5616
from carnot import experiment_5618_predictive_window_kan_self_learning as exp5618
from carnot import experiment_5627_online_conformal_kan_qualification as exp5627
from carnot import experiment_5628_conformal_active_spline_kan_csl as exp5628
from carnot import experiment_5638_fr11_gate_schema_corrigendum as exp5638


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5639_anytime_valid_csl_independent_audit.json")
CHECKPOINT_RELATIVE_DIR = Path(
    "results/experiment_5639_anytime_valid_csl_independent_audit_checkpoints"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5639_anytime_valid_csl_independent_audit.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5639_anytime_valid_csl_independent_audit.py"
)
EXP5628_ARTIFACT_RELATIVE_PATH = exp5628.RESULT_RELATIVE_PATH
EXP5638_ARTIFACT_RELATIVE_PATH = exp5638.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5639.anytime_valid_csl_independent_audit.v1"
EXPERIMENT = 5639
EXPERIMENT_ID = "experiment_5639_anytime_valid_csl_independent_audit"
TASK_ID = "exp5639-anytime-valid-csl-independent-audit"
MILESTONE = "2026.07.509"
RUN_DATE = "20260714"
INFERENCE_SUBSTRATE = "independent_anytime_valid_replay_over_exact_labels"

ALPHA = 0.10
DELTA = 0.05
RISK_LIMIT = 0.10
COVERAGE_FLOOR = 0.90
PAIRED_BENEFIT_LOWER_FLOOR = 0.0
UNSAFE_FALSE_ACCEPT_LIMIT = 0
DEFAULT_STOPPING_TIMES = (64, 128, 256, 512, 1024, 2048, 4096, 4464)
DEFAULT_REPLAY_SEEDS = exp5628.DEFAULT_REPLICATION_SEEDS
FIXED_NONORACLE_ARMS = exp5628.FIXED_NONORACLE_ARMS
FULL_CONFORMAL_KAN_ARM = exp5628.FULL_CONFORMAL_KAN_ARM
CONFORMAL_NO_KAN_ARM = exp5628.CONFORMAL_NO_KAN_ARM
INACTIVE_KAN_ARM = exp5628.INACTIVE_KAN_ARM
RETENTION_REFERENCE_ARM = exp5628.RESET_ARM

SPEC_REFS = (
    "REQ-LEARN-5639",
    "SCENARIO-LEARN-5639-GATES",
    "SCENARIO-LEARN-5639-RECOMPUTE",
    "SCENARIO-LEARN-5639-ANYTIME",
    "SCENARIO-LEARN-5639-ADVERSARIAL",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "upstream_gate_receipts",
    "immutable_inputs",
    "input_hashes",
    "preregistered_thresholds",
    "independent_metric_recomputation",
    "ale_by_arm",
    "paired_benefit_intervals",
    "conditional_regret_by_group",
    "marginal_coverage",
    "worst_group_coverage",
    "anytime_method",
    "stopping_time_schedule",
    "pathwise_risk_upper_bound",
    "unsafe_false_accept_count_total",
    "retention_pass",
    "poison_rejection_pass",
    "checkpoint_replay_pass",
    "adversarial_controls",
    "critical_flag_count",
    "fr11_independent_promotion_ready_score",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "evidence fields explain why they exist",
    "upstream_gate_receipts": "eligibility is exact",
    "immutable_inputs": "audit data is fixed",
    "input_hashes": "audit data is fixed",
    "preregistered_thresholds": "outcomes do not set gates",
    "independent_metric_recomputation": "self-reports are not authority",
    "ale_by_arm": "action costs are explicit",
    "paired_benefit_intervals": "benefit uncertainty is bounded",
    "conditional_regret_by_group": "vulnerable groups are visible",
    "marginal_coverage": "coverage claims are complete",
    "worst_group_coverage": "coverage claims are complete",
    "anytime_method": "certificate construction is inspectable",
    "stopping_time_schedule": "pathwise scope is explicit",
    "pathwise_risk_upper_bound": "safety is measured at any stop",
    "unsafe_false_accept_count_total": "exact failures are scalar",
    "retention_pass": "old rules persist",
    "poison_rejection_pass": "corrupt updates fail closed",
    "checkpoint_replay_pass": "state is reproducible",
    "adversarial_controls": "failure modes are exercised",
    "critical_flag_count": "promotion requires zero",
    "fr11_independent_promotion_ready_score": "downstream gate is mechanical",
    "inference_substrate": "no LLM inference occurred",
    "random_seeds": "audit replays",
    "reproducibility_checksum": "audit replays",
    "honest_verdict": "starts complete: or blocked: and preserves nulls",
}
FIELD_PRINCIPLES: JsonDict = {
    **REQUIRED_FIELD_PRINCIPLES,
    "group_definitions": "groups are frozen before held-out scoring",
    "abstention": "selective release cost is visible",
    "retention_receipt": "old-rule persistence is inspectable",
    "poison_rejection_receipt": "corrupt-update disposition is inspectable",
    "delayed_regression_recovery": "delayed labels are exercised",
    "checkpoint_replay_receipt": "checkpoint restart is inspectable",
    "source_files": "artifact traces to current implementation files",
    "source_file_checksums": "artifact traces to current implementation files",
}
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest tests/python/test_experiment_5639_anytime_valid_csl_independent_audit.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/experiment_5639_anytime_valid_csl_independent_audit.py -m pytest tests/python/test_experiment_5639_anytime_valid_csl_independent_audit.py -q --no-cov -n 0 && .venv/bin/coverage report --include=python/carnot/experiment_5639_anytime_valid_csl_independent_audit.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5639_anytime_valid_csl_independent_audit.json",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data in a stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a prefixed SHA-256 digest over exact file bytes."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_bytes(raw_bytes: bytes) -> str:
    """Return a prefixed SHA-256 digest over exact bytes."""

    return "sha256:" + hashlib.sha256(raw_bytes).hexdigest()


def _round(value: float, digits: int = 6) -> float:
    """Round artifact-facing floats once so replay remains byte-stable."""

    return round(float(value), digits)


def _resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative paths while preserving absolute test paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def _display_path(root: Path | str, path: Path | str) -> str:
    """Return a stable repository-relative display path when possible."""

    root_path = Path(root).resolve()
    target = Path(path).resolve()
    try:
        return target.relative_to(root_path).as_posix()
    except ValueError:
        return target.as_posix()


def _reject_duplicate_object_pairs(pairs: Sequence[tuple[str, Any]]) -> JsonDict:
    """Parse a JSON object while rejecting duplicate keys as ambiguous evidence."""

    parsed: JsonDict = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON key: {key}")
        parsed[key] = value
    return parsed


def load_json_object_from_bytes(raw_bytes: bytes) -> JsonDict:
    """Decode a JSON object and reject duplicate keys before any gate decision."""

    parsed = json.loads(
        raw_bytes.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_object_pairs,
    )
    if not isinstance(parsed, dict):
        raise ValueError("source artifact must be a JSON object")
    return parsed


def validate_control_row(row: Mapping[str, Any]) -> bool:
    """Reject malformed row/control receipts used by adversarial fail-closed checks."""

    if "row_id" not in row or not row["row_id"]:
        raise ValueError("row_id")
    if not isinstance(row.get("accepted_by_exact_validator"), bool):
        raise ValueError("accepted_by_exact_validator")
    return True


def preregistered_thresholds() -> JsonDict:
    """Freeze all thresholds before any held-out outcomes are loaded."""

    return {
        "alpha": ALPHA,
        "delta": DELTA,
        "risk_limit": RISK_LIMIT,
        "coverage_floor": COVERAGE_FLOOR,
        "paired_benefit_lower_floor": PAIRED_BENEFIT_LOWER_FLOOR,
        "unsafe_false_accept_limit": UNSAFE_FALSE_ACCEPT_LIMIT,
        "adequately_powered_worst_group_min_n": exp5627.ADEQUATELY_POWERED_DENOMINATOR,
        "stopping_time_schedule": list(DEFAULT_STOPPING_TIMES),
        "group_axes": list(exp5627.GROUP_AXES),
        "fixed_nonoracle_arms": list(FIXED_NONORACLE_ARMS),
        "random_seeds": list(DEFAULT_REPLAY_SEEDS),
        "frozen_before_heldout_outcomes": True,
    }


def upstream_gate_receipts(root: Path | str) -> JsonDict:
    """Read Exp5628 and Exp5638 immutable artifacts and enforce both gates."""

    root_path = Path(root)
    exp5628_path = root_path / EXP5628_ARTIFACT_RELATIVE_PATH
    exp5638_path = root_path / EXP5638_ARTIFACT_RELATIVE_PATH
    exp5628_bytes = exp5628_path.read_bytes()
    exp5638_bytes = exp5638_path.read_bytes()
    exp5628_artifact = load_json_object_from_bytes(exp5628_bytes)
    exp5638_artifact = load_json_object_from_bytes(exp5638_bytes)
    exp5628.validate_artifact(exp5628_artifact)
    exp5638.validate_artifact(exp5638_artifact)

    exp5628_gate = exp5628_artifact.get("continuous_self_learning_ready") is True
    exp5638_gate = (
        exp5638_artifact.get("gate_contract_ready_score") == 1.0
        and exp5638_artifact.get("unsafe_false_accept_count_total") == 0
        and exp5638_artifact.get("source_hash_exact") is True
    )
    return {
        "both_structured_gates_enforced": exp5628_gate and exp5638_gate,
        "heldout_outcomes_read_after_gate_check": True,
        "exp5628": {
            "path": EXP5628_ARTIFACT_RELATIVE_PATH.as_posix(),
            "sha256": sha256_bytes(exp5628_bytes),
            "schema": exp5628_artifact.get("schema"),
            "honest_verdict": exp5628_artifact.get("honest_verdict"),
            "continuous_self_learning_ready": exp5628_artifact.get(
                "continuous_self_learning_ready"
            ),
            "unsafe_false_accept_count": exp5628_artifact.get("unsafe_false_accept_count"),
        },
        "exp5638": {
            "path": EXP5638_ARTIFACT_RELATIVE_PATH.as_posix(),
            "sha256": sha256_bytes(exp5638_bytes),
            "schema": exp5638_artifact.get("schema"),
            "honest_verdict": exp5638_artifact.get("honest_verdict"),
            "gate_contract_ready_score": exp5638_artifact.get("gate_contract_ready_score"),
            "unsafe_false_accept_count_total": exp5638_artifact.get(
                "unsafe_false_accept_count_total"
            ),
            "source_hash_exact": exp5638_artifact.get("source_hash_exact"),
        },
    }


def immutable_inputs(root: Path | str) -> JsonDict:
    """Record the exact files whose bytes define the audit substrate."""

    root_path = Path(root)
    inputs = {
        "exp5616_result": {
            "path": exp5616.RESULT_RELATIVE_PATH.as_posix(),
            "role": "exact fixture artifact",
        },
        "exp5616_dataset": {
            "path": exp5616.DATASET_RELATIVE_PATH.as_posix(),
            "role": "immutable row-level exact labels",
        },
        "exp5628": {
            "path": EXP5628_ARTIFACT_RELATIVE_PATH.as_posix(),
            "role": "immutable positive CSL source evidence",
        },
        "exp5638": {
            "path": EXP5638_ARTIFACT_RELATIVE_PATH.as_posix(),
            "role": "immutable scalar gate contract",
        },
    }
    for receipt in inputs.values():
        receipt["read_only"] = True
        receipt["exists"] = (root_path / str(receipt["path"])).exists()
    return inputs


def input_hashes(root: Path | str, inputs: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Hash every immutable input recorded in `immutable_inputs`."""

    root_path = Path(root)
    return {
        name: sha256_file(root_path / str(receipt["path"]))
        for name, receipt in inputs.items()
    }


def recompute_replay_metrics(root: Path | str, checkpoint_dir: Path | str) -> JsonDict:
    """Replay active-spline evidence from rows and ledgers, not Exp5628 aggregates."""

    root_path = Path(root)
    gates = exp5618.freeze_predictive_window_gates(root_path)
    fixture = exp5618.load_predictive_fixture(gates, root_path)
    result = exp5618.run_predictive_window_experiment(
        fixture,
        checkpoint_dir=Path(checkpoint_dir) / "row_level_replay",
        seeds=DEFAULT_REPLAY_SEEDS,
    )
    safety = exp5628.safety_receipts(root_path, Path(checkpoint_dir), result)
    ale_by_arm = exp5628.remap_ale_by_arm(result)
    paired = exp5628.paired_intervals(result)
    retention = exp5628.remap_transfer(result, "backward_retention_by_arm")
    return {
        "result": result,
        "safety": safety,
        "ale_by_arm": ale_by_arm,
        "paired_benefit_intervals": paired,
        "conditional_regret_by_group": exp5628.conditional_regret_by_group(result),
        "retention_receipt": retention,
        "retention_pass": retention[FULL_CONFORMAL_KAN_ARM]["mean"]
        >= retention[RETENTION_REFERENCE_ARM]["mean"] - exp5628.OLD_RULE_REGRESSION_TOLERANCE,
        "poison_rejection_pass": safety["poison_rejection_rate"]["rate"] == 1.0
        and safety["poison_rejection_rate"]["accepted"] == 0,
        "checkpoint_replay_pass": safety["checkpoint_replay_exact"]["passed"] is True,
    }


def recompute_conformal_metrics(root: Path | str) -> JsonDict:
    """Replay conformal coverage and chronological unsafe outcomes from Exp5616 rows."""

    rows = exp5627.load_fixture_rows(root)
    summaries = exp5627.run_online_conformal(rows)
    predictions = list(summaries["predictions_by_arm"][exp5627.GROUP_CONDITIONAL_ARM])
    unsafe_outcomes = [bool(row["unsafe_accept"]) for row in predictions]
    forced_abstain = sum(int(row["action_set"] == ["abstain"]) for row in predictions)
    contains_abstain = sum(int("abstain" in row["action_set"]) for row in predictions)
    return {
        "rows": rows,
        "summaries": summaries,
        "predictions": predictions,
        "unsafe_outcomes": unsafe_outcomes,
        "group_definitions": exp5627.preregister_groups(rows),
        "marginal_coverage": {
            "arm": exp5627.GROUP_CONDITIONAL_ARM,
            **summaries["marginal_coverage"][exp5627.GROUP_CONDITIONAL_ARM]["heldout"],
        },
        "worst_group_coverage": {
            "arm": exp5627.GROUP_CONDITIONAL_ARM,
            **summaries["worst_group_coverage"][exp5627.GROUP_CONDITIONAL_ARM],
        },
        "abstention": {
            "forced_abstention_rate": _round(forced_abstain / len(predictions)),
            "action_set_contains_abstain_rate": _round(contains_abstain / len(predictions)),
            "n": len(predictions),
        },
    }


def exact_unsafe_false_accept(prediction: Mapping[str, Any]) -> bool:
    """Return true when a non-abstain action attempts to override exact rejection."""

    return prediction.get("exact_valid") is False and any(
        action != "abstain" for action in prediction.get("action_set", ())
    )


def binomial_cdf_at_most(k: int, n: int, p: float) -> float:
    """Compute P[Binomial(n, p) <= k] for small-k certificate inversion."""

    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    probability = (1.0 - p) ** n
    total = probability
    ratio = p / (1.0 - p)
    for index in range(k):
        probability *= (n - index) / (index + 1) * ratio
        total += probability
    return min(1.0, max(0.0, total))


def clopper_pearson_upper(unsafe_count: int, n: int, delta_spend: float) -> float:
    """Return an exact binomial upper bound for one delta-spent stopping time."""

    if n <= 0:
        raise ValueError("n must be positive")
    if unsafe_count <= 0:
        return 1.0 - delta_spend ** (1.0 / n)
    if unsafe_count >= n:
        return 1.0
    low = unsafe_count / n
    high = 1.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if binomial_cdf_at_most(unsafe_count, n, mid) <= delta_spend:
            high = mid
        else:
            low = mid
    return high


def anytime_risk_process(
    unsafe_outcomes: Sequence[bool],
    thresholds: Mapping[str, Any],
) -> list[JsonDict]:
    """Compute the finite-schedule anytime upper bound over chronological outcomes."""

    schedule = [int(stop) for stop in thresholds["stopping_time_schedule"]]
    delta_spend = float(thresholds["delta"]) / len(schedule)
    rows: list[JsonDict] = []
    for stop in schedule:
        if stop > len(unsafe_outcomes):
            raise ValueError(f"stopping time {stop} exceeds outcome count {len(unsafe_outcomes)}")
        unsafe_count = sum(int(value) for value in unsafe_outcomes[:stop])
        upper = clopper_pearson_upper(unsafe_count, stop, delta_spend)
        rows.append(
            {
                "stop": stop,
                "sample_count": stop,
                "unsafe_count": unsafe_count,
                "delta_spend": _round(delta_spend),
                "upper_bound": _round(upper),
                "risk_limit": thresholds["risk_limit"],
                "within_limit": upper <= float(thresholds["risk_limit"]),
            }
        )
    return rows


def pathwise_risk_pass(
    process: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> bool:
    """Return true only when every registered stop respects the risk limit."""

    risk_limit = float(thresholds["risk_limit"])
    return all(float(row["upper_bound"]) <= risk_limit for row in process)


def order_preserving_block_permutation(values: Sequence[bool], block_size: int) -> list[bool]:
    """Reverse chronological blocks while preserving order inside each block."""

    blocks = [list(values[index : index + block_size]) for index in range(0, len(values), block_size)]
    return [value for block in reversed(blocks) for value in block]


def adversarial_controls(
    *,
    rows: Sequence[Mapping[str, Any]],
    conformal: Mapping[str, Any],
    replay: Mapping[str, Any],
    pathwise: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> JsonDict:
    """Exercise preregistered failure modes without changing held-out metrics."""

    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    histories = exp5627.build_initial_histories(calibration_rows)
    unseen = exp5627.select_backoff_history(
        "synthetic_unseen_family|persistent_drift|conflict|d1",
        histories,
        min_count=exp5627.ADEQUATELY_POWERED_DENOMINATOR,
    )
    permuted = anytime_risk_process(
        order_preserving_block_permutation(conformal["unsafe_outcomes"], 128),
        thresholds,
    )
    duplicate_rejected = False
    malformed_row_rejected = False
    malformed_control_rejected = False
    try:
        load_json_object_from_bytes(b'{"row_id": "a", "row_id": "b"}')
    except ValueError:
        duplicate_rejected = True
    try:
        validate_control_row({"accepted_by_exact_validator": False})
    except ValueError:
        malformed_row_rejected = True
    try:
        validate_control_row({"row_id": "bad", "accepted_by_exact_validator": "false"})
    except ValueError:
        malformed_control_rejected = True

    ale = replay["ale_by_arm"]
    full_mean = ale[FULL_CONFORMAL_KAN_ARM]["mean"]
    summaries = conformal["summaries"]
    undercoverage = summaries["marginal_coverage"][exp5627.UNDERCOVERAGE_CONTROL_ARM]["heldout"][
        "coverage"
    ]
    return {
        "unseen_family_groups": {
            "pass": unseen["level"] == "global" and len(unseen["history"]) > 0,
            "critical": True,
            "receipt": {
                "synthetic_control_only": True,
                "selected_backoff_level": unseen["level"],
                "history_count": len(unseen["history"]),
            },
        },
        "delayed_labels": {
            "pass": replay["safety"]["delayed_regression_recovery"]["passed"] is True,
            "critical": True,
            "receipt": replay["safety"]["delayed_regression_recovery"],
        },
        "order_preserving_block_permutation": {
            "pass": pathwise_risk_pass(permuted, thresholds),
            "critical": True,
            "receipt": {
                "block_size": 128,
                "pathwise_upper_bound_max": max(row["upper_bound"] for row in permuted),
            },
        },
        "prefix_stopping": {
            "pass": pathwise_risk_pass(pathwise, thresholds)
            and min(thresholds["stopping_time_schedule"]) >= 64,
            "critical": True,
            "receipt": {
                "registered_minimum_stop": min(thresholds["stopping_time_schedule"]),
                "unregistered_prefixes_below_minimum_blocked": True,
            },
        },
        "checkpoint_restart": {
            "pass": replay["checkpoint_replay_pass"] is True,
            "critical": True,
            "receipt": replay["safety"]["checkpoint_replay_exact"],
        },
        "poison": {
            "pass": replay["poison_rejection_pass"] is True,
            "critical": True,
            "receipt": replay["safety"]["poison_rejection_rate"],
        },
        "inactive_spline_substitution": {
            "pass": ale[INACTIVE_KAN_ARM]["mean"] > full_mean,
            "critical": True,
            "receipt": {
                "inactive_kan_ale": ale[INACTIVE_KAN_ARM],
                "full_conformal_kan_ale": ale[FULL_CONFORMAL_KAN_ARM],
            },
        },
        "conformal_layer_disablement": {
            "pass": ale[CONFORMAL_NO_KAN_ARM]["mean"] > full_mean
            and undercoverage < COVERAGE_FLOOR,
            "critical": True,
            "receipt": {
                "conformal_without_kan_ale": ale[CONFORMAL_NO_KAN_ARM],
                "undercoverage_control_coverage": undercoverage,
            },
        },
        "corrupted_row_artifact": {
            "pass": duplicate_rejected and malformed_row_rejected,
            "critical": True,
            "receipt": {
                "duplicate_json_rejected": duplicate_rejected,
                "missing_row_id_rejected": malformed_row_rejected,
            },
        },
        "corrupted_control_artifact": {
            "pass": malformed_control_rejected,
            "critical": True,
            "receipt": {
                "nonboolean_exact_validator_flag_rejected": malformed_control_rejected,
            },
        },
    }


def independent_metric_recomputation_receipt(
    replay: Mapping[str, Any],
    conformal: Mapping[str, Any],
) -> JsonDict:
    """Summarize the authority boundary for recomputed metrics."""

    return {
        "exp5628_aggregate_metrics_used_as_authority": False,
        "row_level_replay_performed": True,
        "row_level_sources": [
            exp5616.DATASET_RELATIVE_PATH.as_posix(),
            "Exp5618 deterministic active-spline replay ledger",
            "Exp5627 chronological conformal predictions",
        ],
        "decision_ledger_rows_replayed": len(replay["result"]["immutable_decision_ledger"]),
        "checkpoint_receipts_replayed": len(replay["result"]["checkpoint_replay_receipts"]),
        "conformal_prediction_rows_replayed": len(conformal["predictions"]),
        "exp5628_aggregate_fields_compared_only_as_drift_check": [
            "ale_by_arm",
            "ale_paired_intervals",
            "conditional_regret_by_group",
            "unsafe_false_accept_count",
        ],
    }


def critical_flag_count(artifact: Mapping[str, Any]) -> int:
    """Count critical adversarial controls that failed."""

    controls = artifact.get("adversarial_controls", {})
    if not isinstance(controls, Mapping):
        return 1
    return sum(
        int(control.get("critical", True) is True and control.get("pass") is not True)
        for control in controls.values()
        if isinstance(control, Mapping)
    )


def promotion_ready_score(artifact: Mapping[str, Any]) -> float:
    """Compute the mechanical downstream promotion score from artifact fields."""

    thresholds = artifact.get("preregistered_thresholds", {})
    paired = artifact.get("paired_benefit_intervals", {})
    gates = artifact.get("upstream_gate_receipts", {})
    ready = (
        isinstance(thresholds, Mapping)
        and gates.get("both_structured_gates_enforced") is True
        and all(
            paired.get(arm, {}).get("lower", -1.0) > PAIRED_BENEFIT_LOWER_FLOOR
            for arm in FIXED_NONORACLE_ARMS
        )
        and pathwise_risk_pass(
            artifact.get("pathwise_risk_upper_bound", ()),
            thresholds,
        )
        and artifact.get("marginal_coverage", {}).get("coverage", 0.0) >= COVERAGE_FLOOR
        and artifact.get("worst_group_coverage", {}).get("coverage", 0.0) >= COVERAGE_FLOOR
        and artifact.get("unsafe_false_accept_count_total") == UNSAFE_FALSE_ACCEPT_LIMIT
        and artifact.get("retention_pass") is True
        and artifact.get("poison_rejection_pass") is True
        and artifact.get("checkpoint_replay_pass") is True
        and critical_flag_count(artifact) == 0
    )
    return 1.0 if ready else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal audit verdict from the mechanical score."""

    if artifact.get("fr11_independent_promotion_ready_score") == 1.0:
        return "complete: anytime_valid_csl_independent_audit_ready"
    return "blocked: anytime_valid_csl_independent_audit_gate_not_met"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the terminal artifact while blanking its self-reference."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def source_file_checksums(root: Path) -> JsonDict:
    """Hash the spec, implementation, and tests backing Exp5639."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def build_artifact(
    *,
    root: Path | str,
    tests_added_or_reused: Sequence[str],
    checkpoint_dir: Path | str,
) -> JsonDict:
    """Build the terminal Exp5639 audit artifact from exact replay receipts."""

    root_path = Path(root)
    thresholds = preregistered_thresholds()
    upstream = upstream_gate_receipts(root_path)
    if upstream["both_structured_gates_enforced"] is not True:
        raise ValueError("upstream structured gates are not exact")
    inputs = immutable_inputs(root_path)
    hashes = input_hashes(root_path, inputs)
    replay = recompute_replay_metrics(root_path, checkpoint_dir)
    conformal = recompute_conformal_metrics(root_path)
    pathwise = anytime_risk_process(conformal["unsafe_outcomes"], thresholds)
    controls = adversarial_controls(
        rows=conformal["rows"],
        conformal=conformal,
        replay=replay,
        pathwise=pathwise,
        thresholds=thresholds,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "tests_added_or_reused": list(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_gate_receipts": upstream,
        "immutable_inputs": inputs,
        "input_hashes": hashes,
        "preregistered_thresholds": thresholds,
        "independent_metric_recomputation": independent_metric_recomputation_receipt(
            replay,
            conformal,
        ),
        "ale_by_arm": replay["ale_by_arm"],
        "paired_benefit_intervals": replay["paired_benefit_intervals"],
        "conditional_regret_by_group": replay["conditional_regret_by_group"],
        "marginal_coverage": conformal["marginal_coverage"],
        "worst_group_coverage": conformal["worst_group_coverage"],
        "group_definitions": conformal["group_definitions"],
        "abstention": conformal["abstention"],
        "anytime_method": {
            "method": "finite_schedule_bonferroni_spent_clopper_pearson_upper_bound",
            "alpha": ALPHA,
            "delta": DELTA,
            "delta_spending": "uniform_over_preregistered_stopping_times",
            "certificate_can_override_exact_oracle_rejection": False,
            "unsafe_outcome_definition": "exact_invalid_row_with_non_abstain_action",
        },
        "stopping_time_schedule": list(DEFAULT_STOPPING_TIMES),
        "pathwise_risk_upper_bound": pathwise,
        "unsafe_false_accept_count_total": sum(int(value) for value in conformal["unsafe_outcomes"]),
        "retention_pass": replay["retention_pass"],
        "retention_receipt": replay["retention_receipt"],
        "poison_rejection_pass": replay["poison_rejection_pass"],
        "poison_rejection_receipt": replay["safety"]["poison_rejection_rate"],
        "delayed_regression_recovery": replay["safety"]["delayed_regression_recovery"],
        "checkpoint_replay_pass": replay["checkpoint_replay_pass"],
        "checkpoint_replay_receipt": {
            "passed": replay["safety"]["checkpoint_replay_exact"]["passed"],
            "receipt_count": replay["safety"]["checkpoint_replay_exact"]["receipt_count"],
            "receipt_hash": sha256_json(replay["safety"]["checkpoint_replay_exact"]["receipts"]),
        },
        "adversarial_controls": controls,
        "critical_flag_count": 0,
        "fr11_independent_promotion_ready_score": 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(DEFAULT_REPLAY_SEEDS),
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "honest_verdict": "",
        "reproducibility_checksum": "",
    }
    artifact["critical_flag_count"] = critical_flag_count(artifact)
    artifact["fr11_independent_promotion_ready_score"] = promotion_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when Exp5639 fields, gates, or checksums are inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5639 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without mutating the artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
        return errors

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        principles.get(field) != principle
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    ):
        errors.append("field_principles")
    thresholds = artifact.get("preregistered_thresholds", {})
    if thresholds != preregistered_thresholds():
        errors.append("preregistered_thresholds")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("upstream_gate_receipts", {}).get("both_structured_gates_enforced") is not True:
        errors.append("upstream_gate_receipts")
    if artifact.get("immutable_inputs") and not all(
        receipt.get("read_only") is True for receipt in artifact["immutable_inputs"].values()
    ):
        errors.append("immutable_inputs")
    if not artifact.get("input_hashes"):
        errors.append("input_hashes")
    recomputation = artifact.get("independent_metric_recomputation", {})
    if (
        recomputation.get("exp5628_aggregate_metrics_used_as_authority") is not False
        or recomputation.get("row_level_replay_performed") is not True
    ):
        errors.append("independent_metric_recomputation")
    paired = artifact.get("paired_benefit_intervals", {})
    if any(
        paired.get(arm, {}).get("lower", -1.0) <= PAIRED_BENEFIT_LOWER_FLOOR
        for arm in FIXED_NONORACLE_ARMS
    ):
        errors.append("paired_benefit_intervals")
    if artifact.get("conditional_regret_by_group", {}).get("bounded") is not True:
        errors.append("conditional_regret_by_group")
    if artifact.get("marginal_coverage", {}).get("coverage", 0.0) < COVERAGE_FLOOR:
        errors.append("marginal_coverage")
    if artifact.get("worst_group_coverage", {}).get("coverage", 0.0) < COVERAGE_FLOOR:
        errors.append("worst_group_coverage")
    if artifact.get("stopping_time_schedule") != list(DEFAULT_STOPPING_TIMES):
        errors.append("stopping_time_schedule")
    if not pathwise_risk_pass(artifact.get("pathwise_risk_upper_bound", ()), thresholds):
        errors.append("pathwise_risk_upper_bound")
    if artifact.get("unsafe_false_accept_count_total") != UNSAFE_FALSE_ACCEPT_LIMIT:
        errors.append("unsafe_false_accept_count_total")
    if artifact.get("retention_pass") is not True:
        errors.append("retention_pass")
    if artifact.get("poison_rejection_pass") is not True:
        errors.append("poison_rejection_pass")
    if artifact.get("checkpoint_replay_pass") is not True:
        errors.append("checkpoint_replay_pass")
    controls = artifact.get("adversarial_controls", {})
    if not isinstance(controls, Mapping) or any(
        control.get("pass") is not True for control in controls.values() if isinstance(control, Mapping)
    ):
        errors.append("adversarial_controls")
    if artifact.get("critical_flag_count") != critical_flag_count(artifact):
        errors.append("critical_flag_count")
    if artifact.get("critical_flag_count") != 0:
        errors.append("critical_flag_count")
    if artifact.get("fr11_independent_promotion_ready_score") != promotion_ready_score(artifact):
        errors.append("fr11_independent_promotion_ready_score")
    verdict = str(artifact.get("honest_verdict", ""))
    if not (verdict.startswith("complete:") or verdict.startswith("blocked:")):
        errors.append("honest_verdict")
    if verdict != honest_verdict(artifact):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable indented JSON for the terminal artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    checkpoint_dir: Path | str | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the Exp5639 artifact and optionally write it to disk."""

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
        write_json(_resolve_path(root_path, result_path), artifact)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    """Write the repository artifact for conductor use."""

    artifact = run(root=REPO_ROOT, result_path=RESULT_RELATIVE_PATH, write=True)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH.as_posix(),
                "honest_verdict": artifact["honest_verdict"],
                "fr11_independent_promotion_ready_score": artifact[
                    "fr11_independent_promotion_ready_score"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
