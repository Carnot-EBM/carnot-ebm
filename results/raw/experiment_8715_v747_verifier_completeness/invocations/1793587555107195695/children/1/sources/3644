"""DSP-style feasibility-channel diagnostics for continuous repair.

Spec: REQ-KONA-030, SCENARIO-KONA-030
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FeasibilityChannelCase:
    """Before/after repair row for DSP-style feasibility-channel diagnostics.

    **Researcher summary:**
        Each row asks whether a proposed next repair step was useful. The
        channel observes only the before-state violation pressure, then the
        label is derived from whether the after-state actually reduced hard
        violation energy or count.

    Spec: REQ-KONA-030, SCENARIO-KONA-030
    """

    case_id: str
    cohort: str
    before_violation_energy: float
    before_violation_count: int
    after_violation_energy: float
    after_violation_count: int
    distortion_delta: float


def _feasibility_pressure(
    violation_energy: float,
    violation_count: float,
    energy_scale: float,
    count_scale: float,
) -> float:
    if violation_energy <= 0.0 and violation_count <= 0.0:
        return 0.0
    energy_term = violation_energy / max(energy_scale, 1e-12)
    count_term = violation_count / max(count_scale, 1e-12)
    pressure = 1.0 - np.exp(-(energy_term + count_term))
    return float(np.clip(pressure, 0.0, 1.0))


def _binary_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return 0.5

    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif abs(positive - negative) <= 1e-12:
                wins += 0.5
    return float(wins / (len(positives) * len(negatives)))


def evaluate_feasibility_channels(
    cases: Sequence[FeasibilityChannelCase],
    *,
    threshold: float = 0.5,
    help_energy_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Evaluate local/global feasibility channels as repair-step predictors.

    **Researcher summary:**
        ``phi_local`` is a bounded pressure signal for the current state. It is
        high when this specific latent still violates hard constraints.
        ``Phi_global`` is the same pressure measured at the repair-cohort level.
        Their geometric mean predicts whether another repair step should be
        attempted. Labels are computed after the fact from hard violation
        energy/count reduction, so a step that only adds distortion is treated
        as an unhelpful continue.

    Args:
        cases: Before/after candidate repair transitions.
        threshold: Combined-channel score at or above this value predicts
            ``continue repair``.
        help_energy_tolerance: Minimum hard violation-energy drop that counts
            as useful when hard violation count does not change.

    Returns:
        JSON-serialisable aggregate metrics and per-case channel rows.

    Raises:
        ValueError: If no cases are supplied or numeric fields are impossible.

    Spec: REQ-KONA-030, SCENARIO-KONA-030
    """
    case_list = list(cases)
    if not case_list:
        raise ValueError("at least one feasibility channel case is required")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if help_energy_tolerance < 0.0:
        raise ValueError("help_energy_tolerance must be non-negative")

    cohorts: dict[str, list[FeasibilityChannelCase]] = {}
    for case in case_list:
        numeric_values = [
            case.before_violation_energy,
            float(case.before_violation_count),
            case.after_violation_energy,
            float(case.after_violation_count),
            case.distortion_delta,
        ]
        if any(value < 0.0 for value in numeric_values):
            raise ValueError("violation and distortion values must be non-negative")
        cohorts.setdefault(case.cohort or "default", []).append(case)

    cohort_stats: dict[str, dict[str, float]] = {}
    for cohort, cohort_cases in cohorts.items():
        positive_energies = [
            case.before_violation_energy
            for case in cohort_cases
            if case.before_violation_energy > help_energy_tolerance
        ]
        positive_counts = [
            float(case.before_violation_count)
            for case in cohort_cases
            if case.before_violation_count > 0
        ]
        energy_scale = float(np.mean(positive_energies)) if positive_energies else 1.0
        count_scale = float(np.mean(positive_counts)) if positive_counts else 1.0
        global_energy = float(
            np.mean([case.before_violation_energy for case in cohort_cases])
        )
        global_count = float(
            np.mean([case.before_violation_count for case in cohort_cases])
        )
        Phi_global = _feasibility_pressure(
            global_energy,
            global_count,
            energy_scale,
            count_scale,
        )
        cohort_stats[cohort] = {
            "energy_scale": energy_scale,
            "count_scale": count_scale,
            "Phi_global": Phi_global,
        }

    per_case: list[dict[str, Any]] = []
    scores: list[float] = []
    labels: list[bool] = []
    predictions: list[bool] = []
    wrong_distortions: list[float] = []

    for case in case_list:
        stats = cohort_stats[case.cohort or "default"]
        phi_local = _feasibility_pressure(
            case.before_violation_energy,
            float(case.before_violation_count),
            stats["energy_scale"],
            stats["count_scale"],
        )
        Phi_global = stats["Phi_global"]
        channel_score = float(np.sqrt(phi_local * Phi_global))
        predicted_continue = channel_score >= threshold
        energy_drop = case.before_violation_energy - case.after_violation_energy
        repair_helped = bool(
            case.after_violation_count < case.before_violation_count
            or energy_drop > help_energy_tolerance
        )
        wrong_prediction = predicted_continue != repair_helped
        if wrong_prediction:
            wrong_distortions.append(case.distortion_delta)

        scores.append(channel_score)
        labels.append(repair_helped)
        predictions.append(predicted_continue)
        per_case.append(
            {
                "case_id": case.case_id,
                "cohort": case.cohort,
                "phi_local": phi_local,
                "Phi_global": Phi_global,
                "channel_score": channel_score,
                "predicted_continue": bool(predicted_continue),
                "repair_helped": bool(repair_helped),
                "wrong_prediction": bool(wrong_prediction),
                "before_violation_energy": case.before_violation_energy,
                "before_violation_count": case.before_violation_count,
                "after_violation_energy": case.after_violation_energy,
                "after_violation_count": case.after_violation_count,
                "distortion_delta": case.distortion_delta,
            }
        )

    positives = sum(labels)
    negatives = len(labels) - positives
    false_continue = sum(
        1
        for prediction, label in zip(predictions, labels, strict=True)
        if prediction and not label
    )
    false_stop = sum(
        1
        for prediction, label in zip(predictions, labels, strict=True)
        if not prediction and label
    )
    accuracy = float(
        np.mean(
            [
                prediction == label
                for prediction, label in zip(predictions, labels, strict=True)
            ]
        )
    )
    auc = _binary_auc(scores, labels)
    false_continue_rate = float(false_continue / negatives) if negatives else 0.0
    false_stop_rate = float(false_stop / positives) if positives else 0.0
    distortion_when_wrong = (
        float(np.mean(wrong_distortions)) if wrong_distortions else 0.0
    )

    return {
        "n_cases": len(case_list),
        "n_positive_helpful_repairs": int(positives),
        "n_negative_unhelpful_repairs": int(negatives),
        "n_predicted_continue": int(sum(predictions)),
        "n_predicted_stop": int(len(predictions) - sum(predictions)),
        "phi_local": float(np.mean([row["phi_local"] for row in per_case])),
        "Phi_global": float(np.mean([row["Phi_global"] for row in per_case])),
        "feasibility_channel_auc": auc,
        "repair_help_prediction_accuracy": accuracy,
        "false_continue_rate": false_continue_rate,
        "false_stop_rate": false_stop_rate,
        "distortion_when_wrong": distortion_when_wrong,
        "feasibility_channel_predictive": bool(auc >= 0.60 and accuracy >= 0.60),
        "per_case": per_case,
    }
