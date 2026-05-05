"""Transparent learned stop policy for HardNet++/DSP replay rows.

Spec: REQ-KONA-032, SCENARIO-KONA-032
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

REQUIRED_REPLAY_FIELDS = (
    "case_id",
    "cohort",
    "before_violation_energy",
    "before_violation_count",
    "channel_score",
    "repair_helped",
)
_SEED_RE = re.compile(r"_seed(\d+)")


@dataclass(frozen=True)
class StopPolicyExample:
    """Pre-decision features and verifier-backed label for one replay row.

    Spec: REQ-KONA-032, SCENARIO-KONA-032
    """

    case_id: str
    cohort: str
    repair_family: str
    before_violation_energy: float
    before_violation_count: int
    channel_score: float
    should_continue: bool
    label_source: str = "repair_helped"
    split_bucket: int = -1


@dataclass(frozen=True)
class StopPolicySplit:
    """Deterministic train/held-out split for replay generalization checks.

    Spec: REQ-KONA-032, SCENARIO-KONA-032
    """

    train: tuple[StopPolicyExample, ...]
    held_out: tuple[StopPolicyExample, ...]
    holdout_modulus: int
    holdout_remainder: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "split_rule": "case_id seed modulo holdout_modulus",
            "holdout_modulus": self.holdout_modulus,
            "holdout_remainder": self.holdout_remainder,
            "train_count": len(self.train),
            "held_out_count": len(self.held_out),
            "label_source": "repair_helped",
            "held_out_case_ids": [example.case_id for example in self.held_out],
        }


@dataclass(frozen=True)
class TransparentStopPolicy:
    """Small learned policy using repair-family rates with a DSP fallback.

    **Researcher summary:**
        The policy first applies the hard-feasibility stop guard. For remaining
        rows, it learns whether each transparent repair family usually helps on
        the training split. Unknown families fall back to the learned DSP channel
        threshold. The only label is the replay `repair_helped` validator
        outcome.

    Spec: REQ-KONA-032, SCENARIO-KONA-032
    """

    family_continue_rates: dict[str, float]
    channel_threshold: float
    help_energy_tolerance: float = 1e-4
    family_continue_threshold: float = 0.5
    label_source: str = "repair_helped"

    def predict(self, example: StopPolicyExample) -> bool:
        hard_feasible = (
            example.before_violation_count == 0
            and example.before_violation_energy <= self.help_energy_tolerance
        )
        if hard_feasible:
            return False
        family_rate = self.family_continue_rates.get(example.repair_family)
        if family_rate is not None:
            return family_rate >= self.family_continue_threshold
        return example.channel_score >= self.channel_threshold

    def to_metadata(self) -> dict[str, Any]:
        return {
            "policy_type": "transparent_family_rate_with_dsp_threshold_fallback",
            "features": [
                "before_violation_energy",
                "before_violation_count",
                "channel_score",
                "repair_family",
            ],
            "family_continue_rates": dict(sorted(self.family_continue_rates.items())),
            "channel_threshold": self.channel_threshold,
            "family_continue_threshold": self.family_continue_threshold,
            "help_energy_tolerance": self.help_energy_tolerance,
            "label_source": self.label_source,
        }


def _case_seed(case_id: str) -> int:
    match = _SEED_RE.search(case_id)
    if match:
        return int(match.group(1))
    return sum(ord(char) for char in case_id)  # pragma: no cover - defensive fallback


def _repair_family(cohort: str) -> str:
    if cohort.endswith("_local_linear"):
        return "local_linear"
    if "_to_hardnetpp" in cohort:
        return "hardnetpp"
    if "_to_fsnet" in cohort:
        return "fsnet"
    if "_to_adaptive" in cohort:
        return "adaptive"
    return "other"


def _normalise_replay_row(row: Mapping[str, Any]) -> StopPolicyExample:
    missing = [field for field in REQUIRED_REPLAY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"missing replay row fields: {', '.join(missing)}")
    if not isinstance(row["repair_helped"], bool):
        raise ValueError("repair_helped must be a verifier-backed boolean label")

    before_violation_energy = float(row["before_violation_energy"])
    before_violation_count = float(row["before_violation_count"])
    channel_score = float(row["channel_score"])
    numeric_values = [
        before_violation_energy,
        before_violation_count,
        channel_score,
    ]
    if any(not math.isfinite(value) for value in numeric_values):
        raise ValueError("replay row numeric values must be finite")
    if any(value < 0.0 for value in numeric_values):
        raise ValueError("replay row numeric values must be non-negative")

    cohort = str(row["cohort"])
    return StopPolicyExample(
        case_id=str(row["case_id"]),
        cohort=cohort,
        repair_family=_repair_family(cohort),
        before_violation_energy=before_violation_energy,
        before_violation_count=int(before_violation_count),
        channel_score=channel_score,
        should_continue=row["repair_helped"],
    )


def build_stop_policy_examples(
    replay_rows: Sequence[Mapping[str, Any]],
) -> list[StopPolicyExample]:
    """Build examples whose labels are backed by replay validator outcomes.

    Spec: REQ-KONA-032, SCENARIO-KONA-032
    """
    rows = list(replay_rows)
    if not rows:
        raise ValueError("at least one replay row is required")
    return [_normalise_replay_row(row) for row in rows]


def split_stop_policy_examples(
    examples: Sequence[StopPolicyExample],
    *,
    holdout_modulus: int = 5,
    holdout_remainder: int = 0,
) -> StopPolicySplit:
    """Split examples by deterministic seed buckets.

    Spec: REQ-KONA-032, SCENARIO-KONA-032
    """
    if holdout_modulus <= 1:
        raise ValueError("holdout_modulus must be greater than 1")

    train: list[StopPolicyExample] = []
    held_out: list[StopPolicyExample] = []
    for example in examples:
        bucket = _case_seed(example.case_id) % holdout_modulus
        bucketed = replace(example, split_bucket=bucket)
        if bucket == holdout_remainder:
            held_out.append(bucketed)
        else:
            train.append(bucketed)

    if not train or not held_out:
        raise ValueError("split must contain both training and held-out examples")
    return StopPolicySplit(
        train=tuple(train),
        held_out=tuple(held_out),
        holdout_modulus=holdout_modulus,
        holdout_remainder=holdout_remainder,
    )


def fit_transparent_stop_policy(
    training_examples: Sequence[StopPolicyExample],
    *,
    help_energy_tolerance: float = 1e-4,
) -> TransparentStopPolicy:
    """Fit a transparent family-rate stop/continue policy.

    Spec: REQ-KONA-032, SCENARIO-KONA-032
    """
    train = list(training_examples)
    if not train:
        raise ValueError("at least one training example is required")

    family_counts: dict[str, list[int]] = {}
    for example in train:
        positives, total = family_counts.setdefault(example.repair_family, [0, 0])
        family_counts[example.repair_family] = [
            positives + int(example.should_continue),
            total + 1,
        ]

    positive_scores = [example.channel_score for example in train if example.should_continue]
    channel_threshold = min(positive_scores or [1.0])
    return TransparentStopPolicy(
        family_continue_rates={
            family: positives / total
            for family, (positives, total) in sorted(family_counts.items())
        },
        channel_threshold=channel_threshold,
        help_energy_tolerance=help_energy_tolerance,
    )


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
    return wins / (len(positives) * len(negatives))


def _precision_recall(
    predicted_positive: Sequence[bool],
    actual_positive: Sequence[bool],
) -> tuple[float, float]:
    true_positive = sum(
        prediction and actual
        for prediction, actual in zip(predicted_positive, actual_positive, strict=True)
    )
    predicted_count = sum(predicted_positive)
    actual_count = sum(actual_positive)
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / actual_count if actual_count else 0.0
    return precision, recall


def _continue_recall(
    predicted_continue: Sequence[bool],
    actual_continue: Sequence[bool],
) -> float:
    _precision, recall = _precision_recall(predicted_continue, actual_continue)
    return recall


def conservative_replay_continue_predictions(
    examples: Sequence[StopPolicyExample],
    *,
    threshold: float = 0.5,
    help_energy_tolerance: float = 1e-4,
) -> tuple[bool, ...]:
    """Apply the Exp 1305 conservative replay policy to normalized examples.

    Spec: REQ-KONA-032, SCENARIO-KONA-032
    """
    return tuple(
        (
            (
                example.before_violation_count > 0
                or example.before_violation_energy > help_energy_tolerance
            )
            and example.repair_family != "local_linear"
            and example.channel_score >= threshold
        )
        for example in examples
    )


def evaluate_learned_stop_policy(
    policy: TransparentStopPolicy,
    held_out_examples: Sequence[StopPolicyExample],
    *,
    baseline_continue_predictions: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Evaluate learned stop/continue decisions on held-out replay examples.

    Spec: REQ-KONA-032, SCENARIO-KONA-032
    """
    held_out = list(held_out_examples)
    if not held_out:
        raise ValueError("at least one held-out example is required")

    learned_continue = tuple(policy.predict(example) for example in held_out)
    if baseline_continue_predictions is None:
        baseline_continue = conservative_replay_continue_predictions(held_out)
    else:
        baseline_continue = tuple(baseline_continue_predictions)
    if len(baseline_continue) != len(held_out):
        raise ValueError("baseline predictions must match held-out examples")

    actual_continue = tuple(example.should_continue for example in held_out)
    learned_stop = tuple(not prediction for prediction in learned_continue)
    actual_stop = tuple(not label for label in actual_continue)
    baseline_stop = tuple(not prediction for prediction in baseline_continue)
    stop_precision, stop_recall = _precision_recall(learned_stop, actual_stop)
    baseline_stop_precision, baseline_stop_recall = _precision_recall(
        baseline_stop,
        actual_stop,
    )

    hardnetpp_learned = [
        prediction
        for example, prediction in zip(held_out, learned_continue, strict=True)
        if example.repair_family == "hardnetpp"
    ]
    hardnetpp_baseline = [
        prediction
        for example, prediction in zip(held_out, baseline_continue, strict=True)
        if example.repair_family == "hardnetpp"
    ]
    hardnetpp_actual = [
        example.should_continue
        for example in held_out
        if example.repair_family == "hardnetpp"
    ]

    return {
        "n_held_out": len(held_out),
        "learned_continue_predictions": int(sum(learned_continue)),
        "learned_stop_predictions": int(sum(learned_stop)),
        "true_stop_predictions": int(
            sum(prediction and actual for prediction, actual in zip(learned_stop, actual_stop, strict=True))
        ),
        "false_stop_predictions": int(
            sum(prediction and not actual for prediction, actual in zip(learned_stop, actual_stop, strict=True))
        ),
        "stop_policy_precision": stop_precision,
        "stop_policy_recall": stop_recall,
        "dsp_feasibility_auc": _binary_auc(
            [example.channel_score for example in held_out],
            actual_continue,
        ),
        "hardnetpp_delta_over_replay_policy": _continue_recall(
            hardnetpp_learned,
            hardnetpp_actual,
        )
        - _continue_recall(
            hardnetpp_baseline,
            hardnetpp_actual,
        ),
        "replay_policy": {
            "continue_predictions": int(sum(baseline_continue)),
            "stop_predictions": int(sum(baseline_stop)),
            "stop_precision": baseline_stop_precision,
            "stop_recall": baseline_stop_recall,
        },
    }
