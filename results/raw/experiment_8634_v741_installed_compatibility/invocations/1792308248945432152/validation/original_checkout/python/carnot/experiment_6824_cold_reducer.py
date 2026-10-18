"""Reduce cold arbiter rows without using the producer's reducer.

Every headline starts from a complete paired row roster. The reducer refuses
missing or duplicate identities before it computes a mean or interval. This
prevents aggregate agreement from hiding lost experimental units.
"""

from __future__ import annotations

from collections import Counter
from math import sqrt
from random import Random
from statistics import NormalDist
from typing import Any, Mapping, Sequence


REDUCER_VERSION = "carnot.exp6824.row_owned_reducer.v1"
MODULE_PATH = "python/carnot/experiment_6824_cold_reducer.py"
SELECTIVE_ARM = "selective_priority"
FLAT_ARM = "flat_reject_retry"
ARMS = (SELECTIVE_ARM, FLAT_ARM)
BUDGET_FIELDS = (
    "candidate_count",
    "cpu_allowance_us",
    "exact_check_count",
    "outcome_check_count",
    "retry_cap",
    "work_units",
)
HEADLINE_FIELDS = (
    "accepted_progress_by_arm",
    "certificate_completeness_by_arm",
    "false_intervention_rate_by_arm",
    "hard_violation_rate_by_arm",
    "harmful_selections_by_arm",
    "legal_support_by_arm",
    "paired_progress_delta",
    "paired_retry_delta",
    "retry_cost_by_arm",
    "safe_action_identity_by_arm",
    "acceptance_gate_positive",
)


class RowRosterError(ValueError):
    """Report missing, repeated, or unpaired cold row identities."""


def _cold_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Exclude explicit audit cases from scientific aggregation."""

    return [row for row in rows if row.get("row_type") == "cold_replay"]


def validate_roster(
    rows: Sequence[Mapping[str, Any]], expected_row_ids: Sequence[str]
) -> dict[str, Any]:
    """Require each frozen comparison identity exactly once."""

    observed = [str(row.get("row_id")) for row in _cold_rows(rows)]
    counts = Counter(observed)
    duplicates = sorted(row_id for row_id, count in counts.items() if count > 1)
    expected = set(expected_row_ids)
    missing = sorted(expected.difference(observed))
    extra = sorted(set(observed).difference(expected))
    if duplicates:
        raise RowRosterError(f"duplicate row identities: {duplicates[:2]}")
    if missing:
        raise RowRosterError(f"missing row identities: {missing[:2]}")
    if extra:
        raise RowRosterError(f"extra row identities: {extra[:2]}")
    return {
        "duplicate_identities": [],
        "expected_identity_count": len(expected_row_ids),
        "extra_identities": [],
        "missing_identities": [],
        "observed_identity_count": len(observed),
        "passed": len(observed) == len(expected_row_ids),
    }


def join_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Join each source unit to exactly one row from each frozen arm."""

    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in _cold_rows(rows):
        pair_id = str(row.get("pair_id"))
        arm = str(row.get("arm"))
        if arm not in ARMS or arm in pairs.setdefault(pair_id, {}):
            raise RowRosterError(f"paired arm roster is invalid for {pair_id}")
        pairs[pair_id][arm] = row
    if any(set(pair) != set(ARMS) for pair in pairs.values()):
        raise RowRosterError("paired arm roster is incomplete")
    return pairs


def recompute_budgets(
    rows: Sequence[Mapping[str, Any]], *, expected_pair_count: int
) -> dict[str, Any]:
    """Check equal planned and observed work for every paired unit."""

    try:
        pairs = join_pairs(rows)
    except RowRosterError as exc:
        return {
            "expected_pair_count": expected_pair_count,
            "mismatched_pair_ids": [],
            "observed_pair_count": 0,
            "passed": False,
            "reason": str(exc),
        }
    mismatched = [
        pair_id
        for pair_id, arms in pairs.items()
        if any(
            arms[SELECTIVE_ARM].get(field) != arms[FLAT_ARM].get(field) for field in BUDGET_FIELDS
        )
    ]
    return {
        "expected_pair_count": expected_pair_count,
        "matched_fields": list(BUDGET_FIELDS),
        "mismatched_pair_ids": sorted(mismatched),
        "observed_pair_count": len(pairs),
        "passed": len(pairs) == expected_pair_count and not mismatched,
    }


def wilson_upper(successes: int, total: int, alpha: float) -> float | None:
    """Compute the two-sided Wilson interval's upper endpoint."""

    if total == 0:
        return None
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    radius = z * sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return (center + radius) / denominator


def paired_interval(
    values: Sequence[float], *, seed: int, resamples: int, alpha: float = 0.05
) -> dict[str, Any]:
    """Return a deterministic paired bootstrap interval over row deltas."""

    if not values:
        return {
            "estimate": None,
            "lower_bound": None,
            "pair_count": 0,
            "resamples": resamples,
            "seed": seed,
            "upper_bound": None,
        }
    rng = Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(resamples)
    )
    lower_index = int((alpha / 2.0) * resamples)
    upper_index = min(resamples - 1, int((1.0 - alpha / 2.0) * resamples))
    return {
        "estimate": sum(values) / count,
        "lower_bound": samples[lower_index],
        "pair_count": count,
        "resamples": resamples,
        "seed": seed,
        "upper_bound": samples[upper_index],
    }


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    """Reduce one Boolean row field to exact numerator accounting."""

    numerator = sum(bool(row.get(field)) for row in rows)
    denominator = len(rows)
    return {
        "denominator": denominator,
        "numerator": numerator,
        "rate": numerator / denominator if denominator else None,
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    """Reduce one numeric row field without dropping zero values."""

    return {
        "mean": sum(float(row[field]) for row in rows) / len(rows) if rows else None,
        "unit_count": len(rows),
    }


def recompute_aggregates(
    rows: Sequence[Mapping[str, Any]],
    *,
    held_scenario_ids: Sequence[str],
    constants: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild every held producer headline from paired cold rows."""

    held = [
        row for row in _cold_rows(rows) if str(row.get("scenario_id")) in set(held_scenario_ids)
    ]
    pairs = join_pairs(held)
    by_arm = {arm: [pair[arm] for pair in pairs.values()] for arm in ARMS}
    progress = {arm: _mean(by_arm[arm], "accepted_progress") for arm in ARMS}
    retries = {arm: _mean(by_arm[arm], "retry_count") for arm in ARMS}
    hard = {arm: _rate(by_arm[arm], "accepted_hard_violation") for arm in ARMS}
    harmful = {arm: _rate(by_arm[arm], "harmful_selection") for arm in ARMS}
    certificates = {arm: _rate(by_arm[arm], "certificate_complete") for arm in ARMS}
    legal: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = by_arm[arm]
        by_model = {
            model: _rate([row for row in arm_rows if row["model_id"] == model], "legality")
            for model in sorted({str(row["model_id"]) for row in arm_rows})
        }
        legal[arm] = {**_rate(arm_rows, "legality"), "by_model": by_model}
    false_intervention: dict[str, Any] = {}
    safe_identity: dict[str, Any] = {}
    alpha = float(constants["false_intervention_alpha"])
    for arm in ARMS:
        safe_rows = [row for row in by_arm[arm] if row.get("base_already_valid")]
        false_rate = _rate(safe_rows, "false_intervention")
        false_intervention[arm] = {
            "alpha": alpha,
            **false_rate,
            "upper_bound": wilson_upper(
                int(false_rate["numerator"]), int(false_rate["denominator"]), alpha
            ),
        }
        safe_identity[arm] = _rate(safe_rows, "safe_action_identity")
    progress_values = [
        float(pair[SELECTIVE_ARM]["accepted_progress"]) - float(pair[FLAT_ARM]["accepted_progress"])
        for pair in pairs.values()
    ]
    retry_values = [
        float(pair[FLAT_ARM]["retry_count"]) - float(pair[SELECTIVE_ARM]["retry_count"])
        for pair in pairs.values()
    ]
    interval_seed = int(constants["interval_seed"])
    resamples = int(constants["paired_interval_resamples"])
    progress_delta = {
        "direction": "selective_minus_flat",
        **paired_interval(progress_values, seed=interval_seed, resamples=resamples),
    }
    retry_delta = {
        "direction": "flat_minus_selective",
        **paired_interval(retry_values, seed=interval_seed + 1, resamples=resamples),
    }
    conditions = {
        "false_intervention_within_limit": false_intervention[SELECTIVE_ARM]["upper_bound"]
        <= float(constants["false_intervention_upper_limit"]),
        "no_family_support_loss": all(
            legal[SELECTIVE_ARM]["by_model"][model]["rate"]
            >= legal[FLAT_ARM]["by_model"][model]["rate"]
            for model in legal[SELECTIVE_ARM]["by_model"]
        ),
        "no_harmful_selection_increase": harmful[SELECTIVE_ARM]["numerator"]
        <= harmful[FLAT_ARM]["numerator"],
        "positive_paired_lower_bound": progress_delta["lower_bound"] > 0
        or retry_delta["lower_bound"] > 0,
        "zero_accepted_hard_violations": hard[SELECTIVE_ARM]["numerator"] == 0,
    }
    return {
        "accepted_progress_by_arm": progress,
        "certificate_completeness_by_arm": certificates,
        "false_intervention_rate_by_arm": false_intervention,
        "hard_violation_rate_by_arm": hard,
        "harmful_selections_by_arm": harmful,
        "legal_support_by_arm": legal,
        "paired_progress_delta": progress_delta,
        "paired_retry_delta": retry_delta,
        "retry_cost_by_arm": retries,
        "safe_action_identity_by_arm": safe_identity,
        "acceptance_gate_positive": {"conditions": conditions, "passed": all(conditions.values())},
    }


def _walk(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten headline leaves so every producer claim gets one comparison."""

    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_walk(child, path))
    return flattened


def compare_headlines(
    producer: Mapping[str, Any], cold: Mapping[str, Any], *, tolerance: float
) -> dict[str, Any]:
    """Report numerical differences and exact non-numeric disagreements."""

    expected = _walk({field: producer.get(field) for field in HEADLINE_FIELDS})
    observed = _walk({field: cold.get(field) for field in HEADLINE_FIELDS})
    comparisons: list[dict[str, Any]] = []
    for field in sorted(expected):
        producer_value = expected[field]
        cold_value = observed.get(field)
        numeric = (
            isinstance(producer_value, (int, float))
            and not isinstance(producer_value, bool)
            and isinstance(cold_value, (int, float))
            and not isinstance(cold_value, bool)
        )
        difference = float(cold_value) - float(producer_value) if numeric else None
        within = abs(difference) <= tolerance if numeric else cold_value == producer_value
        comparisons.append(
            {
                "cold_value": cold_value,
                "field": field,
                "numerical_difference": difference,
                "producer_value": producer_value,
                "within_tolerance": within,
            }
        )
    return {
        "all_within_tolerance": all(row["within_tolerance"] for row in comparisons),
        "comparisons": comparisons,
        "frozen_tolerance": tolerance,
    }


def run_row_fault_audits(
    rows: Sequence[Mapping[str, Any]], expected_row_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Prove deletion and duplication stop reduction before a headline exists."""

    cold = _cold_rows(rows)
    cases = (
        ("row_deletion", cold[1:]),
        ("duplicate_row", [*cold, dict(cold[0])]),
    )
    results: list[dict[str, Any]] = []
    for audit_case, attacked in cases:
        try:
            validate_roster(attacked, expected_row_ids)
            detected = False
            reason = "fault was not detected"
        except RowRosterError as exc:
            detected = True
            reason = str(exc)
        results.append(
            {
                "aggregate_recompute_stopped": detected,
                "audit_case": audit_case,
                "detected": detected,
                "reason": reason,
                "row_type": "row_fault_audit",
            }
        )
    return results
