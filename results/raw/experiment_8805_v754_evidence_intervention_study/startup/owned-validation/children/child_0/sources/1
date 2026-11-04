"""REQ-REPORT-8805: reduce whole-world paired errors under frozen safety gates."""

from collections import defaultdict
import random
from typing import Any

from carnot.reporting import evidence_intervention_study_8805 as m

Json = dict[str, Any]


def summarize(rows: list[Json], controls: list[Json]) -> Json:
    """Keep every scheduled row in its denominator and every sibling in its world cluster."""
    groups: dict[tuple[str, str], list[Json]] = defaultdict(list)
    for row in rows + controls:
        groups[(row["cluster_id"], row["arm"])].append(row)
    worlds = sorted({r["cluster_id"] for r in rows})
    pairs, defects, world_metrics = [], [], {}
    for world in worlds:
        for arm in m.ARMS:
            local = groups[(world, arm)]
            scheduled = [r for r in local if r["scope"] == "scheduled"]
            lookup = {(r["condition"], r["ordering"], r["token_bin"]): r for r in scheduled}
            local_pairs = []
            for r in scheduled:
                base = lookup[(r["condition"], "forward", 1024)]
                if r is not base:
                    local_pairs.append(
                        dict(
                            condition=r["condition"],
                            left=base["unit_id"],
                            right=r["unit_id"],
                            invariant=base["decision"] == r["decision"],
                            label_preserved=base["expected"] == r["expected"],
                        )
                    )
                if r["condition"] in {"irrelevant_padding", "reordered_evidence"}:
                    base = lookup[("intact_evidence", r["ordering"], r["token_bin"])]
                    local_pairs.append(
                        dict(
                            condition=r["condition"],
                            left=base["unit_id"],
                            right=r["unit_id"],
                            invariant=base["decision"] == r["decision"],
                            label_preserved=base["expected"] == r["expected"],
                        )
                    )
            pairs += [dict(p, cluster_id=world, arm=arm) for p in local_pairs]
            unsupported = [r for r in scheduled if r["expected"] != "support"]
            unknown = [r for r in scheduled if r["expected"] == "insufficient"]
            alternative = [r for r in local if r["scope"] == "valid_alternative"]
            world_metrics[(world, arm)] = [
                sum(r["decision"] == "support" for r in unsupported) / len(unsupported),
                sum(r["decision"] != "support" for r in alternative) / len(alternative),
                sum(p["invariant"] for p in local_pairs) / len(local_pairs),
                sum(r["decision"] == "insufficient" for r in unknown) / len(unknown),
            ]
    for arm in m.ARMS:
        for condition in (*m.up.CONDITIONS, "alternative_proof"):
            local = [r for r in rows + controls if r["arm"] == arm and r["condition"] == condition]
            relevant_pairs = [p for p in pairs if p["arm"] == arm and p["condition"] == condition]
            unknown = sum(r["expected"] == "insufficient" for r in local)
            unsupported = sum(r["expected"] != "support" for r in local)
            alternatives = sum(r["scope"] == "valid_alternative" for r in local)
            violations = sum(not p["invariant"] for p in relevant_pairs)
            defects.append(
                dict(
                    arm=arm,
                    defect=condition,
                    denominator=len(local),
                    false_accepts=sum(
                        r["decision"] == "support" and r["expected"] != "support" for r in local
                    ),
                    false_accept_denominator=unsupported,
                    insufficient_evidence_recalled=sum(
                        r["expected"] == r["decision"] == "insufficient" for r in local
                    ),
                    insufficient_evidence_denominator=unknown,
                    insufficient_evidence_recall=sum(
                        r["expected"] == r["decision"] == "insufficient" for r in local
                    )
                    / unknown
                    if unknown
                    else None,
                    unknown_to_false=sum(
                        r["expected"] == "insufficient" and r["decision"] == "refute" for r in local
                    ),
                    valid_alternative_false_rejects=sum(
                        r["scope"] == "valid_alternative" and r["decision"] != "support"
                        for r in local
                    ),
                    valid_alternative_denominator=alternatives,
                    nuisance_invariance_violations=violations,
                    nuisance_pair_denominator=len(relevant_pairs),
                    abstention_count=sum(r["decision"] == "insufficient" for r in local),
                    abstention_coverage=sum(r["decision"] == "insufficient" for r in local)
                    / len(local)
                    if local
                    else None,
                    latency_s=sum(r["wall_time_s"] for r in local),
                    mean_latency_s=sum(r["wall_time_s"] for r in local) / len(local)
                    if local
                    else None,
                    primitive_operations=sum(r["primitive_operations"] for r in local),
                    primitive_budget_per_unit=m.up.BUDGET,
                    exhausted_count=sum(r["budget_exhausted"] for r in local),
                )
            )
    deltas, intervals, gates = [], [], []
    protocol = m.protocol()
    for arm in m.ARMS[1:]:
        vectors = []
        for world in worlds:
            local, baseline = world_metrics[(world, arm)], world_metrics[(world, m.ARMS[0])]
            vector = [local[0] - baseline[0], local[1] - baseline[1], local[2], local[3]]
            vectors.append(vector)
            deltas.append(
                dict(
                    cluster_id=world,
                    arm=arm,
                    comparator=m.ARMS[0],
                    values=dict(zip(protocol["families"], vector, strict=True)),
                )
            )
        if not vectors:
            continue
        rng = random.Random(m.SEED)
        draws: list[list[float]] = [[], [], [], []]
        m.progress("before_bootstrap_" + arm, 0, 10000)
        for draw in range(10000):
            sample = rng.choices(vectors, k=len(vectors))
            for family in range(4):
                draws[family].append(sum(v[family] for v in sample) / len(sample))
            if (draw + 1) % 2500 == 0:
                m.progress("bootstrap_" + arm, draw + 1, 10000 - draw - 1)
        arm_intervals = []
        for family, name in enumerate(protocol["families"]):
            ordered = sorted(draws[family])
            interval = dict(
                arm=arm,
                family=name,
                comparator=m.ARMS[0],
                estimate=sum(v[family] for v in vectors) / len(vectors),
                lower=ordered[62],
                upper=ordered[9937],
                confidence=0.9875,
                draws=10000,
                independent_clusters=len(worlds),
                cluster_unit="world",
            )
            intervals.append(interval)
            arm_intervals.append(interval)
        checks = dict(
            false_accept_upper=arm_intervals[0]["upper"] <= -0.05,
            false_reject_upper=arm_intervals[1]["upper"] <= 0.02,
            nuisance_invariance=arm_intervals[2]["estimate"] >= 0.95,
            no_higher_paid_check_budget=all(
                r["primitive_limit"] == m.up.BUDGET for r in rows if r["arm"] in {arm, m.ARMS[0]}
            ),
        )
        gates.append(dict(arm=arm, checks=checks, passed=all(checks.values())))
    return dict(
        per_defect_rows=defects,
        paired_deltas=deltas,
        simultaneous_intervals=intervals,
        nuisance_pair_rows=pairs,
        benefit_gates=gates,
        method_benefit=any(g["passed"] for g in gates),
        statistical_protocol=protocol,
        headroom_controls=[
            dict(
                control="per_chunk_false_accept_headroom",
                arm=m.ARMS[0],
                false_accepts=sum(
                    r["decision"] == "support" and r["expected"] != "support"
                    for r in rows
                    if r["arm"] == m.ARMS[0]
                ),
                denominator=sum(r["expected"] != "support" for r in rows if r["arm"] == m.ARMS[0]),
                constructed_evidence_only=True,
            ),
            *[
                dict(
                    control="constant_" + decision,
                    denominator=len(rows) // 3,
                    errors=sum(r["expected"] != decision for r in rows if r["arm"] == m.ARMS[0]),
                )
                for decision in m.up.LABELS
            ],
        ],
    )
