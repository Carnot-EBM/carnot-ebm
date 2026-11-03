"""REQ-VERIFY-8755: pair worlds before reduction and keep representation errors visible."""

from __future__ import annotations

import random
from typing import Any

from carnot.reporting import verifier_family_surface_8755 as m

Json = dict[str, Any]


def summarize(rows: list[Json]) -> Json:
    """Repeated probes, variants and surfaces do not multiply thirty exposed worlds."""
    evaluation = [r for r in rows if r["partition"] == "evaluation"]
    canonical = [r for r in evaluation if r["surface"] == "canonical"]
    worlds = sorted({r["world_id"] for r in canonical})
    family_rows, paired = [], []
    intervals: Json = {}
    for surface in ["canonical", "transformed"]:
        for variant in m.numerical.VARIANTS:
            for arm in ["baseline", "targeted"]:
                local = [
                    r
                    for r in evaluation
                    if r["surface"] == surface and r["variant"] == variant and r["arm"] == arm
                ]
                detected = sum(not r["passed"] for r in local)
                family_rows.append(
                    dict(
                        surface=surface,
                        defect_family=variant,
                        cohort="new"
                        if variant in m.numerical.NEW
                        else "correct"
                        if variant == "correct"
                        else "old",
                        arm=arm,
                        detected=detected,
                        denominator=len(local),
                        detection_rate=detected / len(local) if local else None,
                        false_rejections=detected if variant == "correct" else None,
                    )
                )
    for cohort, variants in [
        ("old", ("reject_all", "accept_all", "reversed_arguments")),
        ("new", m.numerical.NEW),
    ]:
        deltas = []
        for world in worlds:
            local = [r for r in canonical if r["world_id"] == world and r["variant"] in variants]
            counts = {
                arm: sum(not r["passed"] for r in local if r["arm"] == arm)
                for arm in ["baseline", "targeted"]
            }
            delta = (counts["targeted"] - counts["baseline"]) / len(variants)
            deltas.append(delta)
            paired.append(
                dict(
                    world_id=world,
                    cohort=cohort,
                    defective_denominator=len(variants),
                    **counts,
                    detection_delta=delta,
                )
            )
        rng = random.Random(m.SEED)
        m.progress("before_bootstrap_" + cohort, 0, 10000 if deltas else 0)
        samples = (
            sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(10000))
            if deltas
            else []
        )
        intervals[cohort] = dict(
            method="paired percentile world-cluster bootstrap; original canonical evaluation only",
            draws=len(samples),
            independent_clusters=len(deltas),
            defective_units_per_arm=len(deltas) * len(variants),
            estimate=sum(deltas) / len(deltas) if deltas else None,
            lower95=samples[249] if samples else None,
            upper95=samples[9749] if samples else None,
            random_seed=m.SEED,
        )
        m.progress("after_bootstrap_" + cohort, len(samples), 0)
    index = {(r["world_id"], r["variant"], r["arm"], r["surface"]): r for r in rows}
    surfaces = []
    for row in rows:
        if row["surface"] == "canonical":
            changed = index[(row["world_id"], row["variant"], row["arm"], "transformed")]
            disagreements = sum(
                a["actual"] != b["actual"]
                for a, b in zip(row["probes"], changed["probes"], strict=True)
            )
            surfaces.append(
                dict(
                    world_id=row["world_id"],
                    partition=row["partition"],
                    defect_family=row["variant"],
                    arm=row["arm"],
                    decision_disagreements=disagreements,
                    row_decision_changed=row["passed"] != changed["passed"],
                    detection_delta=int(not changed["passed"]) - int(not row["passed"]),
                    error_delta=(5 - changed["raw_numerator"]) - (5 - row["raw_numerator"]),
                    truth_discrepancies=sum(
                        a["truth_check"] != b["truth_check"]
                        for a, b in zip(row["probes"], changed["probes"], strict=True)
                    ),
                    independent_cluster=row["world_id"],
                )
            )
    surface_families = [
        dict(
            defect_family=v,
            **{
                k: sum(r[k] for r in surfaces if r["defect_family"] == v)
                for k in [
                    "decision_disagreements",
                    "row_decision_changed",
                    "detection_delta",
                    "error_delta",
                    "truth_discrepancies",
                ]
            },
        )
        for v in m.numerical.VARIANTS
    ]
    correct = [r for r in canonical if r["variant"] == "correct"]
    extra = sum(not r["passed"] for r in correct if r["arm"] == "targeted") - sum(
        not r["passed"] for r in correct if r["arm"] == "baseline"
    )
    valid = [
        dict(world_id=r["world_id"], surface=r["surface"], arm=r["arm"], variant=r["variant"], **p)
        for r in evaluation
        for p in r["probes"]
        if p["transform"] in ["equivalent", "entity_bijection"]
    ]
    gates = dict(
        new_family_gain=bool(worlds and intervals["new"]["lower95"] > 0),
        no_extra_correct_rejection=bool(worlds and extra <= 0),
        truth_invariance=bool(surfaces and not any(r["truth_discrepancies"] for r in surfaces)),
        correct_control_invariance=bool(
            surfaces
            and not any(
                r["decision_disagreements"] for r in surfaces if r["defect_family"] == "correct"
            )
        ),
        reject_all_fails=bool(
            valid
            and all(p["actual"] != p["expected"] for p in valid if p["variant"] == "reject_all")
        ),
        independent_generalization=False,
        oracle_dependent_benefit_verdict="circular_positive",
    )
    return dict(
        family_detection_rows=family_rows,
        valid_alternative_rows=valid,
        surface_pair_rows=surfaces,
        surface_family_rows=surface_families,
        paired_world_rows=paired,
        paired_world_intervals=intervals,
        correct_control_extra_false_rejections=extra,
        scientific_promotion_gates=gates,
        transfer_gain=bool(gates["new_family_gain"] and gates["no_extra_correct_rejection"]),
    )
