"""REQ-REPORT-8529: compare strong controls with whole-question uncertainty.

The shipped summaries keep their fixed decision thresholds and missing bounds.
Only the registered contrast family and question reduction change here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.verify import v730_information_metrics as shipped
from carnot.verify.v733_identity_brier_fit import ARMS as ARMS, SEEDS as SEEDS

Json = dict[str, Any]
RAW, LEGACY, SCALAR, FULL, PERMUTED = ARMS
CONTROLS = [RAW, SCALAR, PERMUTED]
RANDOM_SEED, DRAWS = 7318502, 10000
CONFIDENCE = 0.983333333333
CONFIG = dict(
    shipped.CONFIG,
    bootstrap_seed=RANDOM_SEED,
    confidence_level=CONFIDENCE,
    contrasts=CONTROLS,
    family_size=3,
)


def contrast(rows: list[Json], target: str) -> Json:
    """Each resampled unit contains the same observed siblings and all seed slots."""
    groups: dict[str, list[Json]] = defaultdict(list)
    for row in rows:
        groups[row["question_hash"]].append(row)
    paired: list[Json] = []
    omitted: list[Json] = []
    for question, group in sorted(groups.items()):
        arms = {a: [r for r in group if r["arm"] == a and r.get(target) in (0, 1)] for a in ARMS}
        keys = [{(r["unit_id"], r["seed"]) for r in arms[a]} for a in ARMS]
        if (
            not keys[0]
            or any(k != keys[0] for k in keys)
            or any(r["probability"] is None for a in ARMS for r in arms[a])
        ):
            omitted.append(
                dict(question_hash=question, reason="missing_paired_probability_or_observed_target")
            )
            continue
        scores = {a: shipped.shipped.summary(rs, target) for a, rs in arms.items()}
        paired.append(
            dict(
                question_hash=question,
                sibling_seed_count=len(keys[0]),
                arm_losses={
                    a: dict(brier=s["brier"], typed_cost=s["decision_loss"])
                    for a, s in scores.items()
                },
                gains={a: scores[a]["brier"] - scores[FULL]["brier"] for a in CONTROLS},
                loss_deltas={
                    a: scores[FULL]["decision_loss"] - scores[a]["decision_loss"] for a in CONTROLS
                },
            )
        )
    gains: Json = dict.fromkeys(CONTROLS)
    intervals: Json = dict.fromkeys(CONTROLS)
    deltas: Json = dict.fromkeys(CONTROLS)
    if paired:
        vectors = np.asarray([[r["gains"][a] for a in CONTROLS] for r in paired])
        rng, draws = np.random.default_rng(RANDOM_SEED), []
        for done in range(0, DRAWS, 1000):
            print(
                f"[exp8529] phase=bootstrap_{target} completed={done} pending={DRAWS - done}",
                flush=True,
            )
            draws.append(
                vectors[rng.integers(0, len(paired), size=(1000, len(paired)))].mean(axis=1)
            )
        sampled = np.concatenate(draws)
        tail = (1 - CONFIDENCE) / 2
        for i, arm in enumerate(CONTROLS):
            gains[arm] = float(vectors[:, i].mean())
            intervals[arm] = np.quantile(sampled[:, i], [tail, 1 - tail]).tolist()
            deltas[arm] = float(np.mean([r["loss_deltas"][arm] for r in paired]))
        print(f"[exp8529] phase=bootstrap_{target}_end completed={DRAWS} pending=0", flush=True)
    return dict(
        contrasts=gains,
        familywise_intervals=intervals,
        decision_loss_deltas=deltas,
        paired_question_rows=paired,
        omitted_questions=omitted,
        independent_questions=len(paired),
        intended_questions=len(groups),
        draws=DRAWS,
        seed=RANDOM_SEED,
        confidence_level=CONFIDENCE,
        family_size=3,
        familywise_confidence_level=0.95,
        conditional_reason="conditional_on_observed_labels_and_complete_paired_questions"
        if paired
        else "zero_complete_paired_questions",
    )


def evaluate(predictions: list[Json], targets: list[Json]) -> Json:
    """Reuse fixed reporting and require every registered strong control to pass."""
    with patch.multiple(
        shipped, ARMS=ARMS, SCALAR=SCALAR, FULL=FULL, PERMUTED=PERMUTED, contrast=contrast
    ):
        value = dict(shipped.evaluate(predictions, targets))
    rows = value["rows"]
    degeneracy = {}
    full = {(r["unit_id"], r["seed"]): r["probability"] for r in rows if r["arm"] == FULL}
    for arm in CONTROLS:
        differences = [
            abs(r["probability"] - full[(r["unit_id"], r["seed"])])
            for r in rows
            if r["arm"] == arm
            and r["probability"] is not None
            and full.get((r["unit_id"], r["seed"])) is not None
        ]
        degeneracy[arm] = dict(
            identical_predictions=bool(differences) and max(differences) == 0,
            paired_count=len(differences),
            maximum_absolute_prediction_difference=max(differences) if differences else None,
        )
    value["negative_control_degeneracy"] = degeneracy
    value["class_support"] = {
        name: {
            str(y): dict(
                answers=len({r["unit_id"] for r in rows if r[field] == y}),
                questions=len({r["question_hash"] for r in rows if r[field] == y}),
            )
            for y in [0, 1]
        }
        for name, field in [("A1", "y"), ("A2", "y_sensitivity")]
    }
    value["exploratory_benefit_score"] = int(
        bool(
            all(v and v[0] > 0 for v in value["familywise_intervals"].values())
            and all(
                value["decision_loss_deltas"][a] is not None
                and value["decision_loss_deltas"][a] <= 0
                for a in [RAW, SCALAR]
            )
            and not any(d["identical_predictions"] for d in degeneracy.values())
            and value["bootstrap"]["independent_questions"]
            == value["bootstrap"]["intended_questions"]
        )
    )
    return value
