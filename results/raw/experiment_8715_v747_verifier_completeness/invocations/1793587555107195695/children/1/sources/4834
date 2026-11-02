"""Score frozen V671 decisions and paired family contrasts (REQ-ENERGY-7704)."""

from __future__ import annotations

from typing import Any

import numpy as np

from carnot.reporting.typed_decision_energy import action, decision_cost, probabilities

FAMILIES = ("fit_prior", "atom_only", "matched_logistic", "matched_mlp", "typed_gibbs")
VIEWS = ("original_source", "evidence_erasure", "within_role_derangement")


def score_evaluation(
    features: list[dict], labels: list[dict], heads: dict, policy: dict, roster: list[str]
) -> list[dict[str, Any]]:
    """Join authenticated labels once and retain every frozen head and evidence view."""
    if len(roster) != len(set(roster)) or len(labels) != len(roster):
        raise ValueError("label_roster")
    by_label = {row["family_id"]: row for row in labels}
    if set(by_label) != set(roster) or any(
        row["role"] != "evaluation" or row["label"] not in (0, 1) for row in labels
    ):
        raise ValueError("label_roster")
    by_feature = {(row["unit_id"], row["arm"]): row for row in features}
    if len(by_feature) != len(features) or set(by_feature) != {
        (unit, view) for unit in roster for view in VIEWS
    }:
        raise ValueError("feature_roster")
    if set(heads["best_by_family"]) != set(FAMILIES):
        raise ValueError("head_families")
    thresholds = tuple(policy["thresholds"])
    if len(thresholds) != 2 or not 0 <= thresholds[0] <= thresholds[1] <= 1:
        raise ValueError("frozen_thresholds")
    if policy["costs"] != {"correct": 0, "wrong": 1, "escalate": 0.2}:
        raise ValueError("frozen_costs")
    output: list[dict[str, Any]] = []
    for unit in roster:
        label = by_label[unit]["label"]
        for family in FAMILIES:
            key = heads["best_by_family"][family]["key"]
            head = heads["heads"][key]
            for view in VIEWS:
                feature = by_feature[(unit, view)]
                if feature["role"] != "evaluation" or feature["excluded"]:
                    raise ValueError("feature_role_or_exclusion")
                vector = np.asarray(feature["vector"], dtype=float)
                if head.get("feature_indices") is not None:
                    vector = vector[head["feature_indices"]]
                p = probabilities(vector, head)[1]
                chosen = action(p, thresholds)
                output.append(
                    {
                        "unit_id": unit,
                        "arm": f"{family}:{view}",
                        "head_key": key,
                        "label": label,
                        "probability_error": p,
                        "probability_correct": 1 - p,
                        "typed_action": chosen,
                        "brier": (p - label) ** 2,
                        "decision_cost": decision_cost(chosen, label),
                        "checked_count": feature["checked_count"],
                        "proposition_count": feature["proposition_count"],
                        "evidence_coverage": feature["checked_count"]
                        / max(1, feature["proposition_count"]),
                        "raw_metrics": {
                            "error_probability": p,
                            "brier": (p - label) ** 2,
                            "decision_cost": decision_cost(chosen, label),
                        },
                        "counts": {"independent_family": 1, "paired_view": 1},
                        "excluded": False,
                        "exclusions": list(feature["exclusions"]),
                        "censored": feature["censored"],
                        "provenance": {
                            "feature": feature["provenance"],
                            "label": by_label[unit]["provenance"],
                            "source_group_id": feature["source_group_id"],
                        },
                    }
                )
    return output


def paired_reduce(
    rows: list[dict], roster: list[str], control: str, *, seed: int, draws: int = 10_000
) -> dict[str, Any]:
    """Reduce one family per bootstrap block with all escalation charges."""
    if control not in {"fit_prior", "atom_only", "matched_logistic", "matched_mlp"}:
        raise ValueError("control_invalid")
    if len(roster) != len(set(roster)) or draws != 10_000:
        raise ValueError("bootstrap_registration")
    by_arm = {(row["unit_id"], row["arm"]): row for row in rows}
    if len(by_arm) != len(rows) or set(by_arm) != {
        (unit, f"{family}:{view}") for unit in roster for family in FAMILIES for view in VIEWS
    }:
        raise ValueError("row_roster")

    def paired(a: str, b: str, field: str) -> dict[str, Any]:
        differences = np.asarray(
            [by_arm[(unit, b)][field] - by_arm[(unit, a)][field] for unit in roster],
            dtype=float,
        )
        if not len(differences) or not np.isfinite(differences).all():
            raise ValueError("paired_values")
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(roster), size=(draws, len(roster)))
        sample_means = differences[indices].mean(axis=1)
        return {
            "estimate": float(np.mean(differences)),
            "lower": float(np.quantile(sample_means, 0.025)),
            "upper": float(np.quantile(sample_means, 0.975)),
            "draws": draws,
            "seed": seed,
            "independent_blocks": len(roster),
            "direction": "control_minus_candidate",
        }

    original = "typed_gibbs:original_source"
    baseline = f"{control}:original_source"
    brier = paired(original, baseline, "brier")
    cost = paired(original, baseline, "decision_cost")
    non_escalation = sum(
        by_arm[(unit, original)]["typed_action"] != "escalate" for unit in roster
    ) / max(1, len(roster))
    benefit = brier["lower"] > 0.01 and cost["lower"] > 0.01 and non_escalation >= 0.20
    means = {
        arm: {
            "brier": float(np.mean([by_arm[(unit, arm)]["brier"] for unit in roster])),
            "decision_cost": float(
                np.mean([by_arm[(unit, arm)]["decision_cost"] for unit in roster])
            ),
            "evidence_coverage": float(
                np.mean([by_arm[(unit, arm)]["evidence_coverage"] for unit in roster])
            ),
        }
        for arm in sorted({row["arm"] for row in rows})
    }
    return {
        "paired_brier_reduction_ci": brier,
        "paired_cost_reduction_ci": cost,
        "non_escalation_coverage": non_escalation,
        "registered_decision_benefit_score": int(benefit and len(roster) == 40),
        "decision_measurement_complete_score": int(len(roster) == 40),
        "effective_blocks": len(roster),
        "arm_means": means,
        "energy_form_contrast": {
            family: {
                field: paired(original, f"{family}:original_source", field)
                for field in ("brier", "decision_cost")
            }
            for family in ("matched_mlp", "matched_logistic")
        },
        "feature_information_contrast": {
            family: {
                field: paired(original, f"{family}:original_source", field)
                for field in ("brier", "decision_cost")
            }
            for family in ("atom_only", "fit_prior")
        },
        "evidence_interventions": {
            view: {
                field: paired(f"typed_gibbs:{view}", original, field)
                for field in ("brier", "decision_cost")
            }
            for view in VIEWS[1:]
        },
    }
