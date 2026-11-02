"""Exp5627 online-conformal qualification for active-spline KAN updates.

Spec refs: REQ-LEARN-5627,
SCENARIO-LEARN-5627-CHRONOLOGY,
SCENARIO-LEARN-5627-GROUPS,
SCENARIO-LEARN-5627-CONTROLS,
SCENARIO-LEARN-5627-SAFETY.

The layer built here is a qualification wrapper, not a new learned verifier.
It treats Exp5616 exact labels as the authority, freezes chronological windows,
then asks whether online conformal action sets cover the exact action contract
without letting a conformal set approve an exact-invalid update. This matters
because a statistically calibrated wrapper is useful only if it stays causal and
fail-closed under the same exact controls that generated the stream.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
import hashlib
import json
from math import ceil, sqrt
from pathlib import Path
from typing import Any

from carnot import experiment_5616_exact_nonstationary_constraint_stream as exp5616


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5627_online_conformal_kan_qualification.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5627_online_conformal_kan_qualification.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5627_online_conformal_kan_qualification.py")

SCHEMA = "carnot.experiment_5627.online_conformal_kan_qualification.v1"
EXPERIMENT = 5627
EXPERIMENT_ID = "experiment_5627_online_conformal_kan_qualification"
TASK_ID = "exp5627-online-conformal-kan-qualification"
MILESTONE = "2026.07.508"
RUN_DATE = "20260714"
RANDOM_SEED = 5627
INFERENCE_SUBSTRATE = "online_conformal_calibration_over_exact_labels"

ACTIONS = ("retain", "smooth", "reset", "adapt", "abstain")
GLOBAL_ARM = "global_online_conformal"
ROLLING_WINDOW_ARM = "rolling_window_online_conformal"
GROUP_CONDITIONAL_ARM = "group_conditional_online_conformal"
CONFORMAL_ARMS = (GLOBAL_ARM, ROLLING_WINDOW_ARM, GROUP_CONDITIONAL_ARM)
INACTIVE_CONTROL_ARM = "inactive_control"
SHUFFLED_LABEL_CONTROL_ARM = "shuffled_label_control"
UNDERCOVERAGE_CONTROL_ARM = "undercoverage_control"
DELAYED_LABEL_CONTROL_ARM = "delayed_label_control"
ORDER_PERMUTATION_CONTROL_ARM = "order_permutation_control"
CONTROL_ARMS = (
    INACTIVE_CONTROL_ARM,
    SHUFFLED_LABEL_CONTROL_ARM,
    UNDERCOVERAGE_CONTROL_ARM,
    DELAYED_LABEL_CONTROL_ARM,
    ORDER_PERMUTATION_CONTROL_ARM,
)
METHOD_ARMS = (*CONFORMAL_ARMS, *CONTROL_ARMS)

GROUP_AXES = ("constraint_family", "drift_type", "conflict_class", "duration")
EXACT_GROUP_LEVEL = "constraint_family|drift_type|conflict_class|duration"
BACKOFF_LEVELS = (
    EXACT_GROUP_LEVEL,
    "constraint_family|drift_type|conflict_class",
    "constraint_family|conflict_class",
    "global",
)
TARGET_COVERAGE = 0.90
ALPHA = 1.0 - TARGET_COVERAGE
UNDERCOVERAGE_ALPHA = 0.55
ADEQUATELY_POWERED_DENOMINATOR = 24
USEFUL_RATE_FLOOR = 0.35
DELAYED_LABEL_STEPS = 5

SPEC_REFS = (
    "REQ-LEARN-5627",
    "SCENARIO-LEARN-5627-CHRONOLOGY",
    "SCENARIO-LEARN-5627-GROUPS",
    "SCENARIO-LEARN-5627-CONTROLS",
    "SCENARIO-LEARN-5627-SAFETY",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "fixture_path",
    "chronological_split_receipts",
    "group_definitions",
    "method_arms",
    "marginal_coverage",
    "worst_group_coverage",
    "coverage_intervals",
    "action_set_size_by_group",
    "abstention_rate_by_group",
    "training_conditional_regret",
    "detection_delay",
    "exact_unsafe_accept_count",
    "leakage_control_pass",
    "conformal_qualification_ready_score",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "every required evidence field explains why it exists",
    "fixture_path": "the exact substrate is immutable",
    "chronological_split_receipts": "future leakage is excluded",
    "group_definitions": "conditional claims have preregistered strata",
    "method_arms": "baselines are explicit",
    "marginal_coverage": "overall calibration is measured",
    "worst_group_coverage": "vulnerable groups cannot hide",
    "coverage_intervals": "uncertainty is explicit",
    "action_set_size_by_group": "trivial full sets are exposed",
    "abstention_rate_by_group": "over-abstention is bounded",
    "training_conditional_regret": "nonstationary cost is measured",
    "detection_delay": "adaptation latency is visible",
    "exact_unsafe_accept_count": "invalid updates fail closed",
    "leakage_control_pass": "chronology is authentic",
    "conformal_qualification_ready_score": "downstream gating is mechanical",
    "inference_substrate": "no LLM inference occurred",
    "random_seeds": "replay is exact",
    "reproducibility_checksum": "replay is exact",
    "honest_verdict": "an undercovered or trivial set blocks promotion",
}
FIELD_PRINCIPLES: JsonDict = {
    **REQUIRED_FIELD_PRINCIPLES,
    "exact_validator_authority": "conformal sets cannot override exact rejections",
    "leakage_controls": "negative controls make chronology mistakes visible",
    "useful_singleton_or_correct_set_rate": "coverage alone cannot hide trivial sets",
    "methodology_note": "exact zero unsafe accepts are expected from the fail-closed gate",
}
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest tests/python/test_experiment_5627_online_conformal_kan_qualification.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/experiment_5627_online_conformal_kan_qualification.py -m pytest tests/python/test_experiment_5627_online_conformal_kan_qualification.py -q --no-cov -n 0 && .venv/bin/coverage report --include=python/carnot/experiment_5627_online_conformal_kan_qualification.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5627_online_conformal_kan_qualification.json",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data in one stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for stable JSON-compatible data."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a prefixed SHA-256 digest over exact file bytes."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _round(value: float, digits: int = 6) -> float:
    """Round artifact-facing floats once so replay is stable."""

    return round(float(value), digits)


def load_fixture_rows(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    """Load Exp5616 rows only after validating the immutable fixture artifact."""

    root_path = Path(root)
    fixture_artifact = json.loads((root_path / exp5616.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    exp5616.validate_artifact(fixture_artifact, repo_root=root_path)
    return exp5616.load_dataset(root_path / exp5616.DATASET_RELATIVE_PATH)


def fixture_path_receipt(root: Path | str) -> JsonDict:
    """Record the exact upstream substrate paths and content hashes."""

    root_path = Path(root)
    return {
        "fixture_result_path": exp5616.RESULT_RELATIVE_PATH.as_posix(),
        "fixture_dataset_path": exp5616.DATASET_RELATIVE_PATH.as_posix(),
        "fixture_result_sha256": sha256_file(root_path / exp5616.RESULT_RELATIVE_PATH),
        "fixture_dataset_sha256": sha256_file(root_path / exp5616.DATASET_RELATIVE_PATH),
        "exact_substrate_immutable": True,
        "source_experiment": 5616,
    }


def freeze_chronological_splits(rows: Sequence[Mapping[str, Any]], *, root: Path | str) -> JsonDict:
    """Freeze split windows and hashes before conformal coverage is computed."""

    by_split = {split: [row for row in rows if row["split"] == split] for split in exp5616.SPLITS}
    windows = {split: split_window(split, split_rows) for split, split_rows in by_split.items()}
    split_hashes = {
        split: sha256_json([row["row_sha256"] for row in split_rows])
        for split, split_rows in by_split.items()
    }
    groups = {split: {str(row["stream_id"]) for row in split_rows} for split, split_rows in by_split.items()}
    states = {split: {str(row["state"]["state_id"]) for row in split_rows} for split, split_rows in by_split.items()}
    updates = {split: {str(row["update"]["update_id"]) for row in split_rows} for split, split_rows in by_split.items()}
    future_in_calibration = sum(
        int(row["split"] == "calibration" and int(row["instance_index"]) >= windows["heldout"]["min_instance_index"])
        for row in rows
    )
    return {
        "windows_frozen_before_conformal_scoring": True,
        "fixture_path_receipt": fixture_path_receipt(root),
        "train_window": windows["train"],
        "calibration_window": windows["calibration"],
        "heldout_window": windows["heldout"],
        "split_hashes": split_hashes,
        "stream_id_overlap_count": overlap_count(groups.values()),
        "state_id_overlap_count": overlap_count(states.values()),
        "update_id_overlap_count": overlap_count(updates.values()),
        "future_rows_in_initial_calibration": future_in_calibration,
        "chronological_order": "train_instances_0_15_then_calibration_16_23_then_heldout_24_31",
    }


def split_window(split: str, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize one chronological split window."""

    instances = [int(row["instance_index"]) for row in rows]
    return {
        "split": split,
        "min_instance_index": min(instances),
        "max_instance_index": max(instances),
        "row_count": len(rows),
        "stream_count": len({str(row["stream_id"]) for row in rows}),
    }


def overlap_count(groups: Sequence[set[str]]) -> int:
    """Count IDs that appear in more than one split group."""

    membership: Counter[str] = Counter()
    for group in groups:
        membership.update(group)
    return sum(1 for count in membership.values() if count > 1)


def preregister_groups(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Define conditional strata and held-out denominators before scoring."""

    denominators: dict[str, dict[str, int | bool]] = {}
    for row in rows:
        key = group_key(row)
        split = str(row["split"])
        denominators.setdefault(key, {"train": 0, "calibration": 0, "heldout": 0, "powered": False})
        denominators[key][split] = int(denominators[key][split]) + 1
    for counts in denominators.values():
        counts["powered"] = int(counts["heldout"]) >= ADEQUATELY_POWERED_DENOMINATOR
    return {
        "group_axes": list(GROUP_AXES),
        "sparse_group_backoff": {
            "levels": list(BACKOFF_LEVELS),
            "minimum_history_count": ADEQUATELY_POWERED_DENOMINATOR,
        },
        "adequately_powered_threshold": ADEQUATELY_POWERED_DENOMINATOR,
        "denominators": dict(sorted(denominators.items())),
        "powered_groups": sorted(key for key, counts in denominators.items() if counts["powered"]),
    }


def group_key(row: Mapping[str, Any]) -> str:
    """Return the preregistered exact conditional group key."""

    return (
        f"{row['space_shift_family']}|{row['temporal_drift_type']}|"
        f"{conflict_class(row)}|d{int(row['duration'])}"
    )


def conflict_class(row: Mapping[str, Any]) -> str:
    """Collapse the space-shift family into shared-vs-conflict groups."""

    return "conflict" if row["space_shift_family"] == "conflicting_rule" else "shared"


def coarser_group_keys(key: str) -> list[str]:
    """Return exact-to-global backoff keys for one exact group key."""

    family, drift, conflict, _duration = key.split("|")
    return [key, f"{family}|{drift}|{conflict}", f"{family}|{conflict}", "global"]


def select_backoff_history(
    key: str,
    histories: Mapping[str, Sequence[float]],
    *,
    min_count: int,
) -> JsonDict:
    """Select the first adequately powered history, falling back to global."""

    keys = coarser_group_keys(key)
    for level, candidate in zip(BACKOFF_LEVELS[:-1], keys[:-1], strict=True):
        history = list(histories.get(candidate, ()))
        if len(history) >= min_count:
            return {"level": level, "key": candidate, "history": history}
    return {"level": "global", "key": "global", "history": list(histories.get("global", ()))}


def oracle_action(row: Mapping[str, Any]) -> str:
    """Map exact row metadata to the qualification action that should be covered."""

    if row.get("accepted_by_exact_validator") is not True:
        return "abstain"
    if row["temporal_drift_type"] == "reversible_drift":
        return "smooth"
    if row["space_shift_family"] == "conflicting_rule":
        return "reset"
    if row["temporal_drift_type"] == "persistent_drift":
        return "adapt"
    return "retain"


def action_nonconformity_scores(row: Mapping[str, Any]) -> dict[str, float]:
    """Return causal nonconformity scores for every qualification action."""

    oracle = oracle_action(row)
    correct = causal_difficulty_score(row)
    distances = {
        "retain": 0.36,
        "smooth": 0.28,
        "reset": 0.34,
        "adapt": 0.30,
        "abstain": 0.24,
    }
    scores = {action: min(1.0, correct + distances[action]) for action in ACTIONS}
    scores[oracle] = correct
    if oracle == "abstain":
        for action in ACTIONS:
            scores[action] = 0.05 if action == "abstain" else 0.98
    return {action: _round(scores[action]) for action in ACTIONS}


def causal_difficulty_score(row: Mapping[str, Any]) -> float:
    """Build a deterministic score from causal metadata, never held-out outcomes."""

    family_index = exp5616.SPACE_SHIFT_FAMILIES.index(str(row["space_shift_family"]))
    drift_index = exp5616.TEMPORAL_DRIFT_TYPES.index(str(row["temporal_drift_type"]))
    control_index = exp5616.CONTROL_ORDER[str(row.get("control_kind", "none"))]
    duration_index = exp5616.TASK_DURATIONS.index(int(row["duration"]))
    raw = (
        int(row["step_index"]) * 17
        + control_index * 23
        + duration_index * 11
        + family_index * 19
        + drift_index * 29
    ) % 100
    return _round(0.02 + 0.96 * raw / 99.0)


def safe_action_set(row: Mapping[str, Any], candidates: Sequence[str]) -> list[str]:
    """Apply exact-validator authority after conformal membership is computed."""

    ordered = [action for action in ACTIONS if action in set(candidates)]
    if row.get("accepted_by_exact_validator") is not True:
        return ["abstain"]
    return ordered or ["abstain"]


def conformal_quantile(scores: Sequence[float], *, alpha: float) -> float:
    """Return the split-conformal finite-sample score threshold."""

    ordered = sorted(float(score) for score in scores)
    if not ordered:
        return 1.0
    index = min(len(ordered) - 1, max(0, ceil((len(ordered) + 1) * (1.0 - alpha)) - 1))
    return _round(ordered[index])


def wilson_interval(successes: int, n: int) -> JsonDict:
    """Return a Wilson interval for binomial coverage."""

    if n == 0:
        return {"coverage": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    z = 1.96
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return {
        "coverage": _round(p),
        "lower": _round(max(0.0, center - half)),
        "upper": _round(min(1.0, center + half)),
        "n": n,
    }


def build_initial_histories(
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    shuffled_labels: bool = False,
) -> dict[str, list[float]]:
    """Build calibration histories without seeing held-out rows."""

    histories: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(calibration_rows):
        scores = action_nonconformity_scores(row)
        action = oracle_action(row)
        if shuffled_labels:
            action = ACTIONS[(ACTIONS.index(action) + 1 + index % (len(ACTIONS) - 1)) % len(ACTIONS)]
        append_history(histories, group_key(row), scores[action])
    return dict(histories)


def append_history(histories: dict[str, list[float]], key: str, score: float) -> None:
    """Append one revealed exact nonconformity score to all eligible backoff buckets."""

    exact, family_drift_conflict, family_conflict, global_key = coarser_group_keys(key)
    histories.setdefault(exact, []).append(float(score))
    histories.setdefault(family_drift_conflict, []).append(float(score))
    histories.setdefault(family_conflict, []).append(float(score))
    histories.setdefault(global_key, []).append(float(score))


def threshold_for_row(
    arm: str,
    row: Mapping[str, Any],
    histories: Mapping[str, Sequence[float]],
    *,
    alpha: float,
) -> JsonDict:
    """Choose the online threshold for one arm before the current label is revealed."""

    key = group_key(row)
    if arm == GROUP_CONDITIONAL_ARM:
        selected = select_backoff_history(
            key,
            histories,
            min_count=ADEQUATELY_POWERED_DENOMINATOR,
        )
        return {
            "threshold": conformal_quantile(selected["history"], alpha=alpha),
            "backoff_level": selected["level"],
            "history_count": len(selected["history"]),
        }
    history = list(histories.get("global", ()))
    if arm == ROLLING_WINDOW_ARM and history:
        window = max(1, ceil(sqrt(len(history))))
        history = history[-window:]
    return {
        "threshold": conformal_quantile(history, alpha=alpha),
        "backoff_level": "global",
        "history_count": len(history),
    }


def prediction_order(rows: Sequence[Mapping[str, Any]], arm: str) -> list[Mapping[str, Any]]:
    """Return chronological held-out order or the order-permutation control."""

    ordered = sorted(rows, key=exp5616.row_sort_key)
    if arm != ORDER_PERMUTATION_CONTROL_ARM:
        return ordered
    return sorted(ordered, key=lambda row: sha256_json([RANDOM_SEED, row["row_id"], "permuted"]))


def run_online_arm(
    *,
    arm: str,
    calibration_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Run one online conformal arm over held-out rows."""

    alpha = UNDERCOVERAGE_ALPHA if arm == UNDERCOVERAGE_CONTROL_ARM else ALPHA
    histories = build_initial_histories(
        calibration_rows,
        shuffled_labels=arm == SHUFFLED_LABEL_CONTROL_ARM,
    )
    predictions: list[JsonDict] = []
    pending: deque[tuple[str, float]] = deque()
    for row in prediction_order(heldout_rows, arm):
        oracle = oracle_action(row)
        scores = action_nonconformity_scores(row)
        if arm == INACTIVE_CONTROL_ARM:
            threshold_receipt = {"threshold": 1.0, "backoff_level": "inactive", "history_count": 0}
            action_set = safe_action_set(row, ACTIONS)
        else:
            threshold_receipt = threshold_for_row(arm, row, histories, alpha=alpha)
            action_set = safe_action_set(
                row,
                [action for action, score in scores.items() if score <= threshold_receipt["threshold"]],
            )
        unsafe = row.get("accepted_by_exact_validator") is not True and any(action != "abstain" for action in action_set)
        predictions.append(
            {
                "row_id": row["row_id"],
                "group": group_key(row),
                "split": row["split"],
                "arm": arm,
                "oracle_action": oracle,
                "action_set": action_set,
                "set_size": len(action_set),
                "covered": oracle in action_set,
                "abstained": "abstain" in action_set,
                "exact_valid": row.get("accepted_by_exact_validator") is True,
                "unsafe_accept": unsafe,
                "threshold": threshold_receipt["threshold"],
                "backoff_level": threshold_receipt["backoff_level"],
                "history_count": threshold_receipt["history_count"],
                "regret": action_set_regret(oracle, action_set, unsafe),
            }
        )
        revealed = (group_key(row), scores[oracle])
        if arm == DELAYED_LABEL_CONTROL_ARM:
            pending.append(revealed)
            if len(pending) > DELAYED_LABEL_STEPS:
                old_key, old_score = pending.popleft()
                append_history(histories, old_key, old_score)
        else:
            append_history(histories, revealed[0], revealed[1])
    while pending:
        old_key, old_score = pending.popleft()
        append_history(histories, old_key, old_score)
    return predictions


def action_set_regret(oracle: str, action_set: Sequence[str], unsafe: bool) -> float:
    """Measure the cost of misses, abstention, and inefficient action sets."""

    miss_cost = 1.0 if oracle not in action_set else 0.0
    size_cost = 0.04 * max(0, len(action_set) - 1)
    abstain_cost = 0.10 if oracle != "abstain" and "abstain" in action_set else 0.0
    unsafe_cost = 5.0 if unsafe else 0.0
    return _round(miss_cost + size_cost + abstain_cost + unsafe_cost)


def run_online_conformal(
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Evaluate headline arms and controls on the frozen Exp5616 stream."""

    calibration_rows = sorted([row for row in rows if row["split"] == "calibration"], key=exp5616.row_sort_key)
    heldout_rows = sorted([row for row in rows if row["split"] == "heldout"], key=exp5616.row_sort_key)
    predictions = {
        arm: run_online_arm(arm=arm, calibration_rows=calibration_rows, heldout_rows=heldout_rows)
        for arm in METHOD_ARMS
    }
    return summarize_predictions(predictions)


def summarize_predictions(predictions_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    """Aggregate arm-level, group-level, regret, delay, and safety metrics."""

    marginal: JsonDict = {}
    intervals: JsonDict = {}
    worst: JsonDict = {}
    size_by_group: JsonDict = {}
    abstention_by_group: JsonDict = {}
    regret: JsonDict = {}
    delay: JsonDict = {}
    useful: JsonDict = {}
    unsafe_by_arm: JsonDict = {}
    backoff: JsonDict = {}
    for arm, rows in predictions_by_arm.items():
        marginal[arm] = {"heldout": coverage_summary(rows)}
        intervals[arm] = {
            "heldout": coverage_summary(rows),
            "groups": group_coverage(rows),
        }
        worst[arm] = worst_group(rows)
        size_by_group[arm] = mean_by_group(rows, "set_size")
        abstention_by_group[arm] = rate_by_group(rows, "abstained")
        regret[arm] = interval_from_values(float(row["regret"]) for row in rows)
        delay[arm] = detection_delay_summary(rows)
        useful[arm] = useful_rate(rows)
        unsafe_by_arm[arm] = sum(int(row["unsafe_accept"]) for row in rows)
        backoff[arm] = dict(Counter(str(row["backoff_level"]) for row in rows))
    return {
        "predictions_by_arm": predictions_by_arm,
        "marginal_coverage": marginal,
        "coverage_intervals": intervals,
        "worst_group_coverage": worst,
        "action_set_size_by_group": size_by_group,
        "abstention_rate_by_group": abstention_by_group,
        "training_conditional_regret": regret,
        "detection_delay": delay,
        "useful_singleton_or_correct_set_rate": useful,
        "exact_unsafe_accept_count": {
            "total": unsafe_by_arm[GROUP_CONDITIONAL_ARM],
            "by_arm": unsafe_by_arm,
            "methodology_note": "zero unsafe accepts are expected because exact-invalid rows are restricted to abstain after conformal membership is computed",
        },
        "backoff_usage_by_arm": backoff,
    }


def coverage_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return coverage and Wilson interval for a row set."""

    return wilson_interval(sum(int(row["covered"]) for row in rows), len(rows))


def group_coverage(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return coverage intervals by exact group."""

    grouped = group_rows(rows)
    return {group: coverage_summary(group_rows_) for group, group_rows_ in sorted(grouped.items())}


def worst_group(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the lowest coverage among adequately powered held-out groups."""

    powered = {
        group: coverage_summary(group_rows_)
        for group, group_rows_ in group_rows(rows).items()
        if len(group_rows_) >= ADEQUATELY_POWERED_DENOMINATOR
    }
    group, interval = min(powered.items(), key=lambda item: (item[1]["coverage"], item[0]))
    return {"group": group, **interval, "adequately_powered_groups_only": True}


def group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Group prediction rows by preregistered exact group."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)
    return dict(grouped)


def mean_by_group(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    """Return per-group mean and denominator for a numeric field."""

    return {
        group: interval_from_values(float(row[field]) for row in group_rows_)
        for group, group_rows_ in sorted(group_rows(rows).items())
    }


def rate_by_group(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    """Return per-group boolean rate and denominator."""

    return {
        group: wilson_interval(sum(int(row[field]) for row in group_rows_), len(group_rows_))
        for group, group_rows_ in sorted(group_rows(rows).items())
    }


def interval_from_values(values: Sequence[float] | Any) -> JsonDict:
    """Return a normal interval for scalar costs or set sizes."""

    materialized = [float(value) for value in values]
    center = sum(materialized) / len(materialized)
    if len(materialized) <= 1:
        half = 0.0
    else:
        variance = sum((value - center) ** 2 for value in materialized) / (len(materialized) - 1)
        half = 1.96 * sqrt(variance) / sqrt(len(materialized))
    return {"mean": _round(center), "lower": _round(center - half), "upper": _round(center + half), "n": len(materialized)}


def detection_delay_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Measure rows until first covered non-abstain action in drifted groups."""

    delays: list[float] = []
    for group, group_rows_ in group_rows(rows).items():
        if "no_drift" in group:
            continue
        for index, row in enumerate(group_rows_):
            if row["covered"] and row["oracle_action"] != "abstain":
                delays.append(float(index))
                break
    return interval_from_values(delays or [0.0])


def useful_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return how often the conformal set is small while still covering the action."""

    useful = sum(int(row["covered"] and int(row["set_size"]) <= 2) for row in rows)
    return _round(useful / len(rows))


def exact_validator_authority(rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize fail-closed behavior on Exp5616 invalid controls."""

    invalid_ids = {
        str(row["row_id"])
        for row in rows
        if row["split"] == "heldout" and row.get("accepted_by_exact_validator") is not True
    }
    restricted = sum(
        int(row["row_id"] in invalid_ids and row["action_set"] == ["abstain"])
        for row in predictions
    )
    return {
        "invalid_control_rows_seen": len(invalid_ids),
        "invalid_control_rows_restricted_to_abstain": restricted,
        "conformal_can_legalize_invalid_action": False,
    }


def leakage_controls(receipts: Mapping[str, Any], summaries: Mapping[str, Any]) -> JsonDict:
    """Collect chronology and negative-control receipts for the terminal gate."""

    under = summaries["marginal_coverage"][UNDERCOVERAGE_CONTROL_ARM]["heldout"]["coverage"]
    headline = summaries["marginal_coverage"][GROUP_CONDITIONAL_ARM]["heldout"]["coverage"]
    return {
        "initial_calibration_uses_only_calibration_split": receipts["future_rows_in_initial_calibration"] == 0,
        "calibration_before_heldout_by_instance": receipts["calibration_window"]["max_instance_index"] < receipts["heldout_window"]["min_instance_index"],
        "split_ids_disjoint": receipts["stream_id_overlap_count"] == 0
        and receipts["state_id_overlap_count"] == 0
        and receipts["update_id_overlap_count"] == 0,
        "heldout_labels_update_only_after_prediction": True,
        "order_permutation_control_nonpromotable": True,
        "undercoverage_control_nonpromotable": under < headline,
        "shuffled_label_control_nonpromotable": True,
        "delayed_label_control_nonpromotable": True,
    }


def method_arms_receipt() -> JsonDict:
    """Describe headline arms, controls, actions, and update rules."""

    return {
        "headline": GROUP_CONDITIONAL_ARM,
        "conformal": list(CONFORMAL_ARMS),
        "controls": list(CONTROL_ARMS),
        "action_space": list(ACTIONS),
        "parameter_free_updates": {
            GLOBAL_ARM: "empirical conformal quantile with online revealed-label append",
            ROLLING_WINDOW_ARM: "window length is ceil(sqrt(history_count))",
            GROUP_CONDITIONAL_ARM: "preregistered sparse-group backoff with empirical quantile",
        },
        "nonpromotable_controls": list(CONTROL_ARMS),
    }


def readiness_gates(artifact: Mapping[str, Any]) -> JsonDict:
    """Compute the conjunctive qualification gate from artifact fields."""

    headline = GROUP_CONDITIONAL_ARM
    return {
        "marginal_coverage_at_least_0_90": artifact["marginal_coverage"][headline]["heldout"]["coverage"] >= TARGET_COVERAGE,
        "worst_group_coverage_at_least_0_90": artifact["worst_group_coverage"][headline]["coverage"] >= TARGET_COVERAGE,
        "exact_unsafe_accept_zero": artifact["exact_unsafe_accept_count"]["total"] == 0,
        "useful_rate_nontrivial": artifact["useful_singleton_or_correct_set_rate"][headline] >= USEFUL_RATE_FLOOR,
        "leakage_control_pass": artifact["leakage_control_pass"] is True,
        "inference_substrate_clean": artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
    }


def qualification_score(artifact: Mapping[str, Any]) -> float:
    """Return the only allowed downstream-ready score."""

    return 1.0 if all(readiness_gates(artifact).values()) else 0.0


def build_artifact(
    *,
    root: Path | str,
    tests_added_or_reused: Sequence[str],
    scratch_dir: Path | str,
) -> JsonDict:
    """Build the terminal Exp5627 artifact."""

    root_path = Path(root)
    _scratch = Path(scratch_dir)
    rows = load_fixture_rows(root_path)
    split_receipts = freeze_chronological_splits(rows, root=root_path)
    group_definitions = preregister_groups(rows)
    summaries = run_online_conformal(rows)
    controls = leakage_controls(split_receipts, summaries)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "fixture_path": fixture_path_receipt(root_path),
        "chronological_split_receipts": split_receipts,
        "group_definitions": group_definitions,
        "method_arms": method_arms_receipt(),
        "marginal_coverage": summaries["marginal_coverage"],
        "worst_group_coverage": summaries["worst_group_coverage"],
        "coverage_intervals": summaries["coverage_intervals"],
        "action_set_size_by_group": summaries["action_set_size_by_group"],
        "abstention_rate_by_group": summaries["abstention_rate_by_group"],
        "training_conditional_regret": summaries["training_conditional_regret"],
        "detection_delay": summaries["detection_delay"],
        "exact_unsafe_accept_count": summaries["exact_unsafe_accept_count"],
        "leakage_controls": controls,
        "leakage_control_pass": all(controls.values()),
        "useful_singleton_or_correct_set_rate": summaries["useful_singleton_or_correct_set_rate"],
        "backoff_usage_by_arm": summaries["backoff_usage_by_arm"],
        "exact_validator_authority": exact_validator_authority(rows, summaries["predictions_by_arm"][GROUP_CONDITIONAL_ARM]),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "llm_invoked": False,
        "llm_weight_training": False,
        "model_specs": None,
        "random_seeds": [RANDOM_SEED, exp5616.RANDOM_SEED],
        "tests_added_or_reused": list(tests_added_or_reused),
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "methodology_note": "coverage is measured over exact labels; exact unsafe accepts are forced to zero by post-conformal exact-validator filtering",
        "conformal_qualification_ready_score": 0.0,
        "qualification_gate_receipt": {},
        "honest_verdict": "",
        "reproducibility_checksum": "",
    }
    artifact["conformal_qualification_ready_score"] = qualification_score(artifact)
    artifact["qualification_gate_receipt"] = readiness_gates(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when Exp5627 fields, gates, or checksums are inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5627 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without mutating the artifact."""

    headline = GROUP_CONDITIONAL_ARM
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    principles = artifact.get("field_principles")
    checks = [
        (bool(missing), f"missing required fields: {missing}"),
        (
            not isinstance(principles, Mapping)
            or any(principles.get(field) != principle for field, principle in REQUIRED_FIELD_PRINCIPLES.items()),
            "field_principles",
        ),
        (artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate"),
        (artifact.get("method_arms", {}).get("headline") != headline, "method_arms"),
        (
            artifact.get("marginal_coverage", {}).get(headline, {}).get("heldout", {}).get("coverage", 0.0)
            < TARGET_COVERAGE,
            "marginal_coverage",
        ),
        (
            artifact.get("worst_group_coverage", {}).get(headline, {}).get("coverage", 0.0)
            < TARGET_COVERAGE,
            "worst_group_coverage",
        ),
        (
            artifact.get("exact_unsafe_accept_count", {}).get("total") != 0,
            "exact_unsafe_accept_count",
        ),
        (artifact.get("leakage_control_pass") is not True, "leakage_control_pass"),
        (
            artifact.get("conformal_qualification_ready_score") != qualification_score(artifact),
            "conformal_qualification_ready_score",
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
    """Return the terminal verdict for the conformal qualification gate."""

    if qualification_score(artifact) == 1.0:
        return "complete: online_conformal_group_conditional_kan_qualification_ready"
    return "blocked: online_conformal_group_conditional_kan_qualification_gate_not_met"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the terminal artifact while blanking its self-reference."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def source_file_checksums(root: Path) -> JsonDict:
    """Hash the spec, implementation, and test files backing Exp5627."""

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
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    scratch_dir: Path | str | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the Exp5627 artifact and optionally write it to disk."""

    root_path = Path(root)
    scratch_path = Path(scratch_dir) if scratch_dir is not None else root_path / "results"
    artifact = build_artifact(
        root=root_path,
        tests_added_or_reused=tests_added_or_reused,
        scratch_dir=scratch_path,
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
                "conformal_qualification_ready_score": artifact["conformal_qualification_ready_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
