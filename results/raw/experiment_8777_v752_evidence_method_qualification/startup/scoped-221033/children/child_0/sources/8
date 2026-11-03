"""REQ-REPORT-8768: retain all decisions and pair evaluation errors by world."""

from __future__ import annotations

from collections import defaultdict
import random
import time
from typing import Any

from carnot.reporting import evidence_set_methods_8768 as m

Json = dict[str, Any]


def protocol() -> Json:
    """One frozen analysis definition prevents outcome-dependent threshold changes."""
    return dict(m.method()["statistical_protocol"])


def measure(inputs: list[Json]) -> list[Json]:
    """Receive only input projections; oracle labels are unavailable to this call."""
    rows = []
    m.progress("before_decisions", 0, len(inputs) * 3)
    for index, unit in enumerate(inputs):
        for arm in m.ARMS:
            started = time.monotonic_ns()
            features = m.decide(unit["input"]) if arm == "evidence_set" else {}
            decision, calls = (
                (str(features["decision"]), 1) if features else m.compare(arm, unit["input"])
            )
            ended = time.monotonic_ns()
            rows.append(
                dict(
                    unit_id=unit["unit_id"],
                    arm=arm,
                    decision=decision,
                    features=features,
                    evaluator_invocations=calls,
                    started_monotonic_ns=started,
                    ended_monotonic_ns=ended,
                    wall_time_s=(ended - started) / 1e9,
                )
            )
        if (index + 1) % 120 == 0:
            m.progress("decisions", (index + 1) * 3, (len(inputs) - index - 1) * 3)
    m.progress("after_decisions", len(rows), 0)
    return rows


def audit(inputs: list[Json], decisions: list[Json]) -> bool:
    """Recompute from input records so a repaired summary hash cannot invent decisions."""
    if len(decisions) != len(inputs) * 3:
        return False
    for index, row in enumerate(decisions):
        unit, arm = inputs[index // 3], m.ARMS[index % 3]
        decision, calls = m.compare(arm, unit["input"])
        features = m.decide(unit["input"]) if arm == "evidence_set" else {}
        if (
            row["unit_id"] != unit["unit_id"]
            or row["arm"] != arm
            or row["decision"] != decision
            or row["evaluator_invocations"] != calls
            or not 1 <= calls <= 5
            or row["features"] != features
            or row["wall_time_s"] != (row["ended_monotonic_ns"] - row["started_monotonic_ns"]) / 1e9
            or row["wall_time_s"] < 0
        ):
            return False
    return True


def score(inputs: list[Json], decisions: list[Json], labels: list[Json]) -> list[Json]:
    """Open labels only after decision bytes are sealed; incorrect answers remain complete rows."""
    by_id = {r["unit_id"]: r["decision"] for r in labels}
    metadata = {p["unit_id"]: p for p in inputs}
    return [
        dict(
            r,
            **{k: metadata[r["unit_id"]][k] for k in ["world_id", "split", "condition", "view"]},
            expected=by_id[r["unit_id"]],
            raw_numerator=int(r["decision"] != by_id[r["unit_id"]]),
            raw_denominator=1,
            absolute_metric=int(r["decision"] == by_id[r["unit_id"]]),
            status="completed",
            failed=False,
            censored=False,
            excluded=False,
            missingness=None,
        )
        for r in decisions
    ]


def summarize(rows: list[Json]) -> Json:
    """Macro class errors keep the majority insufficient class from hiding false acceptance."""
    evaluation = [r for r in rows if r["split"] == "evaluation"]
    groups: dict[tuple[str, str], list[Json]] = defaultdict(list)
    for row in evaluation:
        groups[(row["world_id"], row["arm"])].append(row)
    worlds = sorted({r["world_id"] for r in evaluation})
    world_rows = []
    for world in worlds:
        errors = {}
        for arm in m.ARMS:
            local = groups[(world, arm)]
            errors[arm] = (
                sum(
                    sum(r["raw_numerator"] for r in local if r["expected"] == label)
                    / sum(r["expected"] == label for r in local)
                    for label in m.Decision
                )
                / 3
            )
        world_rows.append(dict(world_id=world, macro_errors=errors))
    intervals = []
    for comparator in m.ARMS[:2]:
        deltas = [
            r["macro_errors"][comparator] - r["macro_errors"]["evidence_set"] for r in world_rows
        ]
        rng = random.Random(m.SEED)
        m.progress("before_bootstrap_" + comparator, 0, 10000)
        samples = sorted(
            sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(10000)
        )
        intervals.append(
            dict(
                comparator=comparator,
                estimate=sum(deltas) / len(deltas),
                lower_simultaneous_97_5=samples[62],
                upper_simultaneous_97_5=samples[9937],
                draws=10000,
                independent_clusters=len(deltas),
            )
        )
        m.progress("after_bootstrap_" + comparator, 10000, 0)
    arm_rows = []
    for arm in m.ARMS:
        local = [r for r in evaluation if r["arm"] == arm]
        arm_rows.append(
            dict(
                arm=arm,
                denominator=len(local),
                macro_error=sum(r["macro_errors"][arm] for r in world_rows) / len(world_rows),
                false_acceptances=sum(
                    r["decision"] == "support" and r["expected"] != "support" for r in local
                ),
                valid_alternative_rejections=sum(
                    r["expected"] == "support"
                    and r["decision"] != "support"
                    and r["view"] == "renamed_ordered"
                    for r in local
                ),
                unknown_to_refute=sum(
                    r["expected"] == "insufficient" and r["decision"] == "refute" for r in local
                ),
                evaluator_invocations=sum(r["evaluator_invocations"] for r in local),
                wall_time_s=sum(r["wall_time_s"] for r in local),
            )
        )
    by_key = {(r["world_id"], r["condition"], r["view"], r["arm"]): r for r in rows}
    pairs = []
    for row in rows:
        if row["view"] == "original":
            other = by_key[(row["world_id"], row["condition"], "renamed_ordered", row["arm"])]
            pairs.append(
                dict(
                    world_id=row["world_id"],
                    arm=row["arm"],
                    condition=row["condition"],
                    label_preserved=row["expected"] == other["expected"],
                    decision_preserved=row["decision"] == other["decision"],
                    correct_both=row["absolute_metric"] == other["absolute_metric"] == 1,
                )
            )
    correct = [r for r in rows if r["arm"] == "evidence_set"]
    invariant = all(
        p["label_preserved"] and p["decision_preserved"] and p["correct_both"]
        for p in pairs
        if p["arm"] == "evidence_set"
    )
    controls = [
        dict(
            control=name,
            decision=constant,
            denominator=len(correct),
            failures=sum(r["expected"] != constant for r in correct),
            evaluator_invocations=len(correct),
        )
        for name, constant in [("all_accept", "support"), ("all_reject", "refute")]
    ]
    control_passed = (
        all(r["absolute_metric"] for r in correct)
        and invariant
        and all(r["failures"] > 0 for r in controls)
    )
    truth_controls = [
        dict(
            condition=condition,
            denominator=len(local),
            oracle_decisions=sorted({r["expected"] for r in local}),
            correct=sum(r["absolute_metric"] for r in local),
            evaluation_denominator=sum(r["split"] == "evaluation" for r in local),
        )
        for condition in sorted({r["condition"] for r in correct})
        for local in [[r for r in correct if r["condition"] == condition]]
    ]
    promotion = (
        all(i["lower_simultaneous_97_5"] > 0 for i in intervals)
        and invariant
        and all(
            arm_rows[2]["false_acceptances"] <= x["false_acceptances"]
            and arm_rows[2]["valid_alternative_rejections"] <= x["valid_alternative_rejections"]
            for x in arm_rows[:2]
        )
    )
    return dict(
        independent_count=len(worlds),
        paired_world_rows=world_rows,
        paired_intervals=intervals,
        arm_summary_rows=arm_rows,
        surface_pair_rows=pairs,
        negative_control_rows=controls,
        truth_control_rows=truth_controls,
        truth_controls_passed=bool(control_passed),
        correct_control_invariance=invariant,
        scientific_promotion=bool(promotion),
        statistical_protocol=protocol(),
    )
