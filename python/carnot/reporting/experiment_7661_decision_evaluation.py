"""Frozen source-atom decision evaluation for REQ-REPORT-7661."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from carnot.reporting.experiment_7660_atom_energy import action, score

ARMS = ("identity", "scalar", "cheap_atom", "atom", "source_erased", "source_deranged")
CONTROLS = ("identity", "scalar", "cheap_atom", "source_erased")


def decision_cost(choice: str, label: int) -> float:
    """Apply the frozen asymmetric cost to one answer decision."""

    return 5.0 * label if choice == "accept" else 1.0 - label if choice == "reject" else 0.2


def build_rows(
    features: list[dict],
    labels: list[dict],
    bundle: dict,
    roster: list[str],
    derangement: dict[str, str],
) -> list[dict]:
    """Join isolated labels after the caller authenticates all input bytes."""

    if len(roster) != len(set(roster)) or len(labels) != len(roster):
        raise ValueError("role_roster")
    by_label = {r["component_hash"]: r for r in labels}
    by_feature = {(r["unit_id"], r["arm"]): r for r in features}
    if set(by_label) != set(roster) or len(by_feature) != 3 * len(roster):
        raise ValueError("role_roster")
    if set(by_feature) != {
        (u, arm)
        for u in roster
        for arm in ("original_source", "evidence_erasure", "within_role_derangement")
    }:
        raise ValueError("feature_arms")
    if set(derangement) != set(roster) or set(derangement.values()) != set(roster):
        raise ValueError("derangement")
    thresholds = tuple(bundle["thresholds"])
    if len(thresholds) != 2 or not 0 <= thresholds[0] <= thresholds[1] <= 1:
        raise ValueError("thresholds")
    result = []
    for unit in roster:
        label_row = by_label[unit]
        if (
            label_row["role"] != "evaluation"
            or label_row["learning_partition"] != "evaluation_only"
            or label_row["training_allowed"] is not False
            or label_row["evaluator_only"] is not True
            or label_row["label"] not in (0, 1)
        ):
            raise ValueError("label_custody")
        y = label_row["label"]
        baseline = float(label_row["raw_probability"])
        if not math.isfinite(baseline) or not 0 <= baseline <= 1:
            raise ValueError("baseline_probability")
        original = by_feature[(unit, "original_source")]
        erased = by_feature[(unit, "evidence_erasure")]
        shuffled = by_feature[(unit, "within_role_derangement")]
        if derangement[unit] == unit or shuffled["source_group_id"] != derangement[unit]:
            raise ValueError("derangement")
        for feature in (original, erased, shuffled):
            if feature["role"] != "evaluation" or feature["partition"] != "evaluation_only":
                raise ValueError("feature_custody")
        for arm in ARMS:
            feature = (
                shuffled
                if arm == "source_deranged"
                else erased
                if arm == "source_erased"
                else original
            )
            head_name = bundle["selected"] if arm in ("atom", "source_deranged") else arm
            probability = score(feature, baseline, bundle["heads"][head_name])
            choice = action(probability, thresholds)
            clipped = min(1 - 1e-6, max(1e-6, probability))
            result.append(
                {
                    "unit_id": unit,
                    "arm": arm,
                    "label": y,
                    "probability": probability,
                    "baseline_probability": baseline,
                    "typed_action": choice,
                    "brier": (probability - y) ** 2,
                    "clipped_log_loss": -y * math.log(clipped) - (1 - y) * math.log(1 - clipped),
                    "decision_cost": decision_cost(choice, y),
                    "raw_metrics": {
                        "error_probability": probability,
                        "source_propositions_checked": feature["checked_structural_propositions"],
                    },
                    "counts": {
                        "independent_group": 1,
                        "paired_view": 1,
                        "unknown_claims": feature["unknown_claims"],
                    },
                    "excluded": feature["excluded"],
                    "exclusions": [],
                    "censored": feature["censored"],
                    "proposition_soundness": feature["checked_structural_propositions"],
                    "whole_answer_certified": feature["whole_answer_certified"],
                    "provenance": {
                        "source_sha256": feature["source_sha256"],
                        "label_sidecar": "Exp7602 evaluation isolated evaluator",
                    },
                }
            )
    return result


def _auc(probabilities: np.ndarray, labels: np.ndarray) -> float | None:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    return float(
        np.mean(
            (positives[:, None] > negatives[None, :])
            + 0.5 * (positives[:, None] == negatives[None, :])
        )
    )


def reduce_rows(rows: list[dict], *, seed: int = 7661, draws: int = 10000) -> dict[str, Any]:
    """Recompute every arm and fixed paired source-group comparison."""

    units = list(dict.fromkeys(r["unit_id"] for r in rows if r["arm"] == "identity"))
    if not units or len(rows) != len(units) * len(ARMS):
        raise ValueError("row_roster")
    paired = {(r["unit_id"], r["arm"]): r for r in rows}
    if len(paired) != len(rows) or set(paired) != {(u, a) for u in units for a in ARMS}:
        raise ValueError("row_roster")
    labels = np.array([paired[(u, "identity")]["label"] for u in units], dtype=int)
    metric: dict[str, dict[str, Any]] = {}
    losses: dict[str, np.ndarray] = {}
    costs: dict[str, np.ndarray] = {}
    for arm in ARMS:
        current = [paired[(u, arm)] for u in units]
        p = np.array([r["probability"] for r in current], dtype=float)
        choices = [r["typed_action"] for r in current]
        if any(r["label"] != int(y) for r, y in zip(current, labels, strict=True)):
            raise ValueError("paired_label")
        if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("probability")
        brier = (p - labels) ** 2
        clipped = np.clip(p, 1e-6, 1 - 1e-6)
        log_loss = -labels * np.log(clipped) - (1 - labels) * np.log(1 - clipped)
        cost = np.array([decision_cost(c, int(y)) for c, y in zip(choices, labels, strict=True)])
        if any(
            abs(r["brier"] - b) > 1e-12
            or abs(r["clipped_log_loss"] - ll) > 1e-12
            or abs(r["decision_cost"] - co) > 1e-12
            for r, b, ll, co in zip(current, brier, log_loss, cost, strict=True)
        ):
            raise ValueError("row_loss")
        escalation = sum(c == "escalate" for c in choices)
        accepted_errors = int(
            sum(c == "accept" and y == 1 for c, y in zip(choices, labels, strict=True))
        )
        rejected_correct = int(
            sum(c == "reject" and y == 0 for c, y in zip(choices, labels, strict=True))
        )
        metric[arm] = {
            "brier": float(np.mean(brier)),
            "clipped_log_loss": float(np.mean(log_loss)),
            "auroc": _auc(p, labels),
            "typed_decision_cost": float(np.mean(cost)),
            "accepted_error_rate": accepted_errors / len(units),
            "rejected_correct_rate": rejected_correct / len(units),
            "escalation_fraction": escalation / len(units),
            "non_escalation_coverage": 1 - escalation / len(units),
            "counts": {
                "independent_groups": len(units),
                "accepted_errors": accepted_errors,
                "rejected_correct": rejected_correct,
                "escalated": escalation,
                "unknown_claims": sum(r["counts"]["unknown_claims"] for r in current),
                "censored": sum(r["censored"] for r in current),
                "excluded": sum(r["excluded"] for r in current),
            },
        }
        losses[arm], costs[arm] = brier, cost
    indices = np.random.default_rng(seed).integers(0, len(units), size=(draws, len(units)))

    def interval(difference: np.ndarray) -> dict[str, Any]:
        replicates = np.mean(difference[indices], axis=1)
        return {
            "estimate": float(np.mean(difference)),
            "ci95": [float(v) for v in np.quantile(replicates, [0.025, 0.975])],
            "independent_groups": len(units),
            "bootstrap_seed": seed,
            "draws": draws,
        }

    comparisons = {arm: interval(losses[arm] - losses["atom"]) for arm in CONTROLS}
    best = min(CONTROLS, key=lambda arm: metric[arm]["brier"])
    comparisons["best_prespecified_control"] = interval(losses[best] - losses["atom"])
    comparisons["source_deranged"] = interval(losses["source_deranged"] - losses["atom"])
    utility = interval(costs[best] - costs["atom"])
    probability_benefit = all(
        comparisons[a]["ci95"][0] > 0.01 for a in (*CONTROLS, "best_prespecified_control")
    )
    utility_benefit = (
        utility["ci95"][0] > 0.01 and metric["atom"]["non_escalation_coverage"] >= 0.20
    )
    source_dependence = comparisons["source_deranged"]["ci95"][0] > 0
    return {
        "metrics": metric,
        "confidence_intervals": {"brier": comparisons, "utility": utility},
        "best_prespecified_control": best,
        "probability_benefit": probability_benefit,
        "utility_benefit": utility_benefit,
        "source_dependence": source_dependence,
        "sample_size_budget": {
            "intended": 40,
            "observed": len(units),
            "eligible": len(units),
            "excluded": 0,
            "censored": metric["atom"]["counts"]["censored"],
            "prior_exposure": "all groups exposed; exploratory only",
            "claim_limit": "no fresh confirmatory claim",
        },
    }
