"""Independent V668 arithmetic over authenticated raw operands.

REQ-REPORT-7664; SCENARIO-REPORT-7664-REDUCTION. This module never calls a
producer reducer and never interprets a partial atom as whole-answer truth.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import log
import random
from typing import Any


def _cost(action: str, label: int) -> float:
    """Rebuild the registered typed decision loss from the released label."""
    if action == "escalate":
        return 0.2
    if action == "accept":
        return float(label)
    if action == "reject":
        return float(1 - label)
    raise ValueError(f"unknown action: {action}")


def _percentile(values: list[float], fraction: float) -> float:
    """Use an ordered empirical percentile with no producer statistic."""
    position = (len(values) - 1) * fraction
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def paired_interval(differences: Sequence[float], seed: int, block_size: int = 1) -> dict[str, Any]:
    """Bootstrap groups or fixed chronological blocks, never arm rows."""
    blocks = [list(differences[i : i + block_size]) for i in range(0, len(differences), block_size)]
    if not blocks:
        raise ValueError("empty paired contrast")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(2000):
        selected = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        values = [value for block in selected for value in block]
        draws.append(sum(values) / len(values))
    draws.sort()
    return {
        "estimate": sum(differences) / len(differences),
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "effective_groups": len(differences),
        "effective_blocks": len(blocks),
        "block_size": block_size,
        "draws": len(draws),
        "seed": seed,
    }


def _paired_rows(rows: Sequence[Mapping[str, Any]], *, chronological: bool) -> dict[str, Any]:
    """Verify every saved row metric and aggregate one observation per group and arm."""
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        unit = str(row["unit_id"])
        arm = str(row["arm"])
        if arm in grouped[unit]:
            raise ValueError("duplicate group arm")
        probability = float(row["probability"])
        label = int(row["label"])
        if not 0 <= probability <= 1 or label not in (0, 1):
            raise ValueError("invalid probability or label")
        brier = (probability - label) ** 2
        cost = _cost(str(row["typed_action"]), label)
        if (
            abs(brier - float(row["brier"])) > 1e-10
            or abs(cost - float(row["decision_cost"])) > 1e-10
        ):
            raise ValueError("saved arm metric differs from raw operands")
        if "clipped_log_loss" in row:
            clipped = min(max(probability, 1e-6), 1 - 1e-6)
            loss = -label * log(clipped) - (1 - label) * log(1 - clipped)
            if abs(loss - float(row["clipped_log_loss"])) > 1e-6:
                raise ValueError("saved log loss differs from raw operands")
        if chronological:
            origin = int(row["origin_ordinal"])
            release = int(row["label_release_ordinal"])
            linked = row["label_from"] == unit and row["released_label"] == label
            missing = row.get("feedback_status") == "missing" and row["label_from"] is None
            permuted = arm == "permuted" and row["label_from"] == "eligible_past_update_history"
            if release <= origin or not (linked or missing or permuted):
                raise ValueError("future label or wrong release identity")
        grouped[unit][arm] = {
            "brier": brier,
            "decision_cost": cost,
            "probability": probability,
            "label": label,
            "censored": bool(row["censored"]),
            "excluded": bool(row["excluded"]),
            "unknown_claims": int(row.get("raw_metrics", {}).get("unknown_claims", 0)),
            "origin_ordinal": row.get("origin_ordinal", 0),
        }
    arms = set(next(iter(grouped.values()))) if grouped else set()
    if any(set(group) != arms for group in grouped.values()):
        raise ValueError("missing paired arm or group")
    order = sorted(grouped, key=lambda unit: grouped[unit][next(iter(arms))]["origin_ordinal"])
    metrics = (
        {
            arm: {
                metric: sum(grouped[unit][arm][metric] for unit in order) / len(order)
                for metric in ("brier", "decision_cost")
            }
            for arm in sorted(arms)
        }
        if order
        else {}
    )
    return {
        "effective_groups": len(order),
        "arms": metrics,
        "censored_groups": sum(
            any(row["censored"] for row in group.values()) for group in grouped.values()
        ),
        "excluded_groups": sum(
            any(row["excluded"] for row in group.values()) for group in grouped.values()
        ),
        "unknown_claims": sum(
            group[next(iter(arms))]["unknown_claims"] for group in grouped.values()
        )
        if arms
        else 0,
        "groups": grouped,
        "order": order,
    }


def reduce_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild coverage, probability, utility, causal order, and retention."""
    source_rows = inputs["features"]
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    source_units: set[str] = set()
    for row in source_rows:
        if row["arm"] != "original_source":
            continue
        unit = str(row["unit_id"])
        if unit in source_units or row["source_sha256"] != row["original_source_sha256"]:
            raise ValueError("source hash or duplicate original group")
        if row.get("whole_answer_certified") is True:
            raise ValueError("partial atom claimed fixture truth")
        if int(row["unknown_claims"]) < 0 or int(row["checked_structural_propositions"]) < 0:
            raise ValueError("unknown or checked claims removed from denominator")
        source_units.add(unit)
        role = str(row["role"])
        coverage[role]["independent_groups"] += 1
        coverage[role]["checked_propositions"] += int(row["checked_structural_propositions"])
        coverage[role]["unknown_claims"] += int(row["unknown_claims"])
        coverage[role]["covered_groups"] += int(row["checked_structural_propositions"] > 0)
        coverage[role]["censored_groups"] += int(row["censored"])
        coverage[role]["excluded_groups"] += int(row["excluded"])
        coverage[role]["previously_exposed_groups"] += int(row["historically_exposed"])
    coverage_total = {
        key: sum(role[key] for role in coverage.values())
        for key in (
            "independent_groups",
            "checked_propositions",
            "unknown_claims",
            "covered_groups",
            "censored_groups",
            "excluded_groups",
            "previously_exposed_groups",
        )
    }
    evaluation = _paired_rows(inputs["evaluation"], chronological=False)
    delayed = _paired_rows(inputs["delayed"], chronological=False)
    continuous = _paired_rows(inputs["continuous"], chronological=True)
    for event in inputs.get("delayed_events", []):
        if (
            int(event["release_ordinal"]) <= int(event["origin_ordinal"])
            or event["acknowledgment"]["state_hash"] != event["next_state_hash"]
            or not event["acknowledgment"]["durable"]
            or event["event_id"] not in delayed["groups"]
        ):
            raise ValueError("delayed update order or state acknowledgment failed")
    for event in inputs["feedback"]:
        linked = event["label_from"] == event["unit_id"]
        missing = event["label_from"] is None and event["released_label"] is None
        permuted = (
            event["arm"] == "permuted" and event["label_from"] == "eligible_past_update_history"
        )
        acknowledgment = event.get("acknowledgment") or {}
        if (
            int(event["label_release_ordinal"]) <= int(event["origin_ordinal"])
            or not (linked or missing or permuted)
            or (
                linked
                and (not acknowledgment.get("acknowledged") or not acknowledgment.get("durable"))
            )
        ):
            raise ValueError("feedback chronology or durability failed")
    admitted: dict[str, set[str]] = defaultdict(set)
    for event in inputs["admissions"]:
        arm = str(event["arm"])
        ids = [str(value) for value in event["admission_ids"]]
        if len(ids) != len(set(ids)) or any(unit in admitted[arm] for unit in ids):
            raise ValueError("one-use admission group reused")
        admitted[arm].update(ids)
    retention: dict[str, dict[str, Any]] = {}
    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in inputs["retention"]:
        label = int(row["label"])
        for prefix in ("final", "frozen"):
            if (
                abs(
                    (float(row[f"{prefix}_probability"]) - label) ** 2
                    - float(row[f"{prefix}_brier"])
                )
                > 1e-10
            ):
                raise ValueError("retention metric differs from raw operands")
        by_role[str(row["role"])].append(row)
    for role, rows in by_role.items():
        if len({row["unit_id"] for row in rows}) != len(rows):
            raise ValueError("duplicate retention group")
        differences = [float(row["frozen_brier"]) - float(row["final_brier"]) for row in rows]
        retention[role] = {
            "groups": len(rows),
            "brier_improvement": sum(differences) / len(rows),
            "censored_groups": sum(bool(row["censored"]) for row in rows),
            "interval": paired_interval(differences, 7664),
        }
    contrasts: dict[str, Any] = {}
    for name, reduced, treatment, block in (
        ("evaluation", evaluation, "atom", 1),
        ("delayed", delayed, "source", 8),
        ("continuous", continuous, "source", 8),
    ):
        contrasts[name] = {}
        if treatment not in reduced["arms"]:
            continue
        for control in reduced["arms"]:
            if control == treatment:
                continue
            contrasts[name][control] = {}
            for metric in ("brier", "decision_cost"):
                differences = [
                    reduced["groups"][unit][control][metric]
                    - reduced["groups"][unit][treatment][metric]
                    for unit in reduced["order"]
                ]
                contrasts[name][control][metric] = paired_interval(differences, 7664, block)
    return {
        "coverage": coverage_total,
        "coverage_by_role": {k: dict(v) for k, v in coverage.items()},
        "evaluation": {k: v for k, v in evaluation.items() if k not in ("groups", "order")},
        "delayed": {k: v for k, v in delayed.items() if k not in ("groups", "order")},
        "continuous": {k: v for k, v in continuous.items() if k not in ("groups", "order")},
        "retention": retention,
        "contrasts": contrasts,
        "feedback_events": len(inputs["feedback"]),
        "admitted_unique_by_arm": {arm: len(ids) for arm, ids in admitted.items()},
    }


def mutate(inputs: dict[str, Any], mutation: str) -> None:
    """Apply one private corruption; every mutation must close the audit."""
    if mutation == "source_hash":
        inputs["features"][0]["source_sha256"] = "sha256:swapped"
    elif mutation == "future_label":
        inputs["continuous"][0]["label_release_ordinal"] = 0
    elif mutation == "admission_reuse":
        inputs["admissions"].append(dict(inputs["admissions"][0]))
    elif mutation == "arm_metric":
        inputs["continuous"][0]["brier"] = 0.9
    elif mutation == "unknown_removed":
        inputs["features"][0]["unknown_claims"] = -1
    elif mutation == "fixture_truth":
        inputs["features"][0]["whole_answer_certified"] = True
    else:
        raise ValueError("unknown mutation")
