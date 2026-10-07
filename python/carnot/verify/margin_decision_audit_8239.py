"""REQ-VERIFY-8239: original costs distinguish energy from shared margin weighting.

All arms share source draws. Missing decisions keep escalation cost in the
primary denominator; complete-case probability metrics report their own support.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import restricted_decision_rule_8210 as qualified
from carnot.verify import margin_prediction_seal_8238 as seal

Json = dict[str, Any]
from carnot.reporting import decision_margin_methods_8234 as methods

PROTOCOL = methods.PROTOCOL_VALUE
H1 = PROTOCOL["H1"]
BASELINE = "original_frozen_v707_radial"
ARM_KEYS = [(a, None) for a in [*seal.ARMS, "energy_global"]] + [
    (BASELINE, None),
    ("always_escalate", None),
]


def statistics(rows: list[Json], comparator: Json) -> Json:
    """The frozen primary supplies H1; matched simple heads constrain harm."""
    groups: dict[str, dict[tuple[str, Any], Json]] = {}
    for row in rows:
        group = groups.setdefault(row["source_cluster_id"], {})
        key = (row["arm"], row["seed"])
        if key in group:
            raise ValueError("duplicate_source_arm")
        group[key] = row
    if any(set(g) != set(ARM_KEYS) for g in groups.values()):
        raise ValueError("unpaired_source")
    paired = list(groups.values())
    treatment, primary = ("energy_margin", None), (comparator["arm"], None)
    complete = [g for g in paired if g[treatment]["complete_pair"] and g[primary]["complete_pair"]]
    support = Counter(g[treatment]["y"] for g in complete)

    def contrast(sources: list[Any]) -> Json:
        """Use the qualified10000-draw implementation with this protocol's frozen seed."""
        gains = [g[primary]["numerator"] - g[treatment]["numerator"] for g in sources]
        with patch.dict(qualified.historical.old.CONFIG, seed=H1["seed"]):
            result = qualified.historical.old.interval(gains)
        return dict(
            result,
            denominator=len(sources),
            effective_independent_sources=len(complete),
            fixed_missing_slots=len(paired) - len(complete),
            confidence=H1["confidence"],
            treatment=treatment[0],
            comparator=primary[0],
        )

    bootstrap = contrast(paired)
    comparisons = []
    for arm, seed in ARM_KEYS:
        rs = [g[arm, seed] for g in paired]
        ps = [
            g
            for g in paired
            if g[treatment]["brier"] is not None and g[arm, seed]["brier"] is not None
        ]
        brier = math.fsum(g[treatment]["brier"] - g[arm, seed]["brier"] for g in ps)
        costs = math.fsum(g[treatment]["numerator"] - g[arm, seed]["numerator"] for g in paired)
        comparisons.append(
            dict(
                arm=arm,
                seed=seed,
                all_slot_cost=dict(
                    numerator=math.fsum(r["numerator"] for r in rs), denominator=len(rs)
                ),
                complete_case_cost=dict(
                    numerator=math.fsum(r["numerator"] for r in rs if r["complete_pair"]),
                    denominator=sum(r["complete_pair"] for r in rs),
                ),
                complete_case_brier=dict(
                    numerator=math.fsum(r["brier"] for r in rs if r["brier"] is not None),
                    denominator=sum(r["brier"] is not None for r in rs),
                ),
                treatment_cost_increase=dict(
                    numerator=costs,
                    denominator=len(paired),
                    mean=costs / len(paired) if paired else None,
                ),
                treatment_brier_increase=dict(
                    numerator=brier, denominator=len(ps), mean=brier / len(ps) if ps else None
                ),
                extra_false_accepts=sum(
                    g[treatment]["false_accept"] and not g[arm, seed]["false_accept"]
                    for g in paired
                ),
                false_accept_count=sum(r["false_accept"] for r in rs),
            )
        )
    lookup = {r["arm"]: r for r in comparisons if r["seed"] is None}
    operands = dict(
        all_slots=dict(observed=len(paired), op="==", expected=H1["intended"]),
        complete_sources=dict(observed=len(complete), op=">=", expected=H1["minimum_complete"]),
        per_class=dict(
            observed=min(support[0], support[1]), op=">=", expected=H1["minimum_per_class"]
        ),
        valid_draws=dict(
            observed=bootstrap["valid_draws"], op=">=", expected=H1["minimum_valid_draws"]
        ),
        lower_gain=dict(
            observed=bootstrap["lower_one_sided_975"], op=">", expected=H1["lower_cost_gain_gt"]
        ),
        improved_sources=dict(
            observed=sum(g[treatment]["numerator"] < g[primary]["numerator"] for g in paired),
            op=">=",
            expected=H1["improved_sources_min"],
        ),
    )
    for label, arm in [
        ("primary", primary[0]),
        ("original", BASELINE),
        ("additive_margin", "additive_margin"),
        ("logistic_margin", "logistic_margin"),
    ]:
        operands[label + "_brier_increase"] = dict(
            observed=lookup[arm]["treatment_brier_increase"]["mean"],
            op="<=",
            expected=H1["brier_degradation_max"],
        )
        operands[label + "_extra_false_accepts"] = dict(
            observed=lookup[arm]["extra_false_accepts"],
            op="<=",
            expected=H1["extra_false_accepts_max"],
        )
    for arm in [BASELINE, "additive_margin", "logistic_margin"]:
        operands[arm + "_cost_increase"] = dict(
            observed=lookup[arm]["treatment_cost_increase"]["mean"],
            op="<=",
            expected=H1["other_control_cost_increase_max"],
        )
    operators = {
        "==": lambda a, b: a == b,
        ">=": lambda a, b: a >= b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
    }
    for operand in operands.values():
        operand["passed"] = operand["observed"] is not None and operators[operand["op"]](
            operand["observed"], operand["expected"]
        )
    passed = all(o["passed"] for o in operands.values())
    shared = []
    for family in ["energy", "additive", "logistic"]:
        gain = math.fsum(
            g[family + "_uniform", None]["numerator"] - g[family + "_margin", None]["numerator"]
            for g in paired
        )
        shared.append(
            dict(
                family=family,
                numerator=gain,
                denominator=len(paired),
                mean_gain=gain / len(paired) if paired else None,
            )
        )
    return dict(
        H1=dict(
            protocol=H1,
            passed=passed,
            operands=operands,
            failed_conditions=[k for k, o in operands.items() if not o["passed"]],
            complete_class_support={str(y): support[y] for y in (0, 1)},
        ),
        per_source_deltas=[
            dict(
                source_cluster_id=g[treatment]["source_cluster_id"],
                unit_id=g[treatment]["unit_id"],
                slot=g[treatment]["slot"],
                status=g[treatment]["status"],
                y=g[treatment]["y"],
                complete_pair=g in complete,
                treatment_cost=g[treatment]["numerator"],
                comparator_cost=g[primary]["numerator"],
                numerator=g[primary]["numerator"] - g[treatment]["numerator"],
                denominator=1,
                missing_evidence=g[treatment]["missing_evidence"],
                arm_deltas={
                    a: g[a, seed]["numerator"] - g[treatment]["numerator"] for a, seed in ARM_KEYS
                },
            )
            for g in paired
        ],
        calibration_and_cost_comparisons=comparisons,
        energy_specific_advantage=[lookup[a] for a in ["additive_margin", "logistic_margin"]],
        shared_weighting_effect=shared,
        bootstrap_diagnostics=bootstrap,
        complete_case_bootstrap=contrast(complete),
        comparator_sha256=canonical_hash(comparator),
        h1_development_signal_score=int(passed),
        **describe(paired, treatment, primary, passed),
    )


def reduce(data: Json) -> Json:
    """Derive evaluator targets from original bytes, never imported producer costs."""
    if not 0 < data["clock"]["predictions_sealed_ns"] < data["clock"]["labels_opened_ns"]:
        raise ValueError("prediction_label_access_order")
    if data["protocol"] != PROTOCOL:
        raise ValueError("frozen_protocol")
    predicted = seal.reduce(data["sealed"])
    if predicted["prediction_rows"] != data["predictions"]:
        raise ValueError("sealed_prediction_drift")
    slots = {r["unit_id"]: r for r in data["slots"]}
    originals = {str(r["id"]): r for r in data["original_response_records"]}
    if (
        len(slots) != 128
        or len(data["slots"]) != 128
        or len(originals) != 128
        or len(data["original_response_records"]) != 128
    ):
        raise ValueError("original_label_roster")
    targets = {}
    for feature in data["sealed"]["public"]["features"]:
        slot = slots[feature["unit_id"]]
        original = originals[slot["response_id"]]
        if (
            str(original["source_id"]) != slot["source_id"]
            or feature["source_cluster_id"] != slot["source_cluster_id"]
        ):
            raise ValueError("original_source_identity")
        if any(
            feature[k + "_sha256"] != canonical_hash(slot[k + "_bytes"])
            for k in ["answer", "source"]
        ):
            raise ValueError("original_byte_identity")
        target, _ = qualified.historical.old.base.human.target(
            original, bytes.fromhex(slot["answer_bytes"])
        )
        if target["y"] not in (0, 1) and feature["status"] == "completed":
            raise ValueError("original_annotation_not_qualified")
        targets[feature["unit_id"]] = target["y"]
    rows = []
    for prediction in data["predictions"]:
        y = targets[prediction["unit_id"]]
        action, p = prediction["action"], prediction["p"]
        cost = 0.5 if action == "escalate" else 5 * y if action == "accept" else 1 - y
        rows.append(
            dict(
                prediction,
                seed=None,
                y=y,
                metric="typed_decision_cost",
                numerator=cost,
                denominator=1,
                complete_pair=prediction["status"] == "completed" and y in (0, 1),
                brier=(p - y) ** 2 if p is not None and y in (0, 1) else None,
                false_accept=int(action == "accept" and y == 1),
            )
        )
    return dict(
        rows=rows,
        **{
            k: predicted[k]
            for k in [
                "intended_count",
                "completed_count",
                "independent_count",
                "failed_count",
                "censored_count",
                "excluded_count",
            ]
        },
        **statistics(rows, data["comparator"]),
    )


def describe(
    paired: list[Any], treatment: tuple[str, Any], primary: tuple[str, Any], passed: bool
) -> Json:
    """Action changes show whether probability movement creates useful decisions.

    Simple heads share the weighting objective. Their contrasts prevent shared
    improvements from being described as an advantage of energy alone.
    """
    switches, matrices, advantage = [], [], []
    for arm, seed in ARM_KEYS:
        matrix = Counter((g[arm, seed]["action"], g[treatment]["action"]) for g in paired)
        matrices.append(
            dict(
                arm=arm,
                cells=[
                    dict(from_action=a, to_action=b, count=matrix[a, b])
                    for a in ["accept", "reject", "escalate"]
                    for b in ["accept", "reject", "escalate"]
                ],
            )
        )
        for g in paired:
            switches.append(
                dict(
                    source_cluster_id=g[treatment]["source_cluster_id"],
                    slot=g[treatment]["slot"],
                    arm=arm,
                    from_action=g[arm, seed]["action"],
                    to_action=g[treatment]["action"],
                    switched=g[arm, seed]["action"] != g[treatment]["action"],
                    p_delta=None
                    if g[arm, seed]["p"] is None or g[treatment]["p"] is None
                    else g[treatment]["p"] - g[arm, seed]["p"],
                    cost_gain=g[arm, seed]["numerator"] - g[treatment]["numerator"],
                    missing_evidence=g[treatment]["missing_evidence"],
                )
            )
        if arm in ["additive_margin", "logistic_margin"]:
            gains = [g[arm, seed]["numerator"] - g[treatment]["numerator"] for g in paired]
            with patch.dict(qualified.historical.old.CONFIG, seed=H1["seed"]):
                interval = qualified.historical.old.interval(gains)
            advantage.append(
                dict(arm=arm, bootstrap=interval, improved_sources=sum(v > 0 for v in gains))
            )
    strata = []
    for name, lower, upper in [
        ("near_tie", 0, 0.01),
        ("small", 0.01, 0.05),
        ("medium", 0.05, 0.1),
        ("large", 0.1, math.inf),
        ("missing", None, None),
    ]:
        selected = [
            g
            for g in paired
            if (
                g[treatment]["public_margin"] is None
                if lower is None
                else g[treatment]["public_margin"] is not None
                and lower <= g[treatment]["public_margin"] < upper
            )
        ]
        strata.append(
            dict(
                stratum=name,
                denominator=len(selected),
                complete_count=sum(g[treatment]["complete_pair"] for g in selected),
                numerator=math.fsum(
                    g[primary]["numerator"] - g[treatment]["numerator"] for g in selected
                ),
                switched_count=sum(
                    g[primary]["action"] != g[treatment]["action"] for g in selected
                ),
            )
        )
    return dict(
        action_switch_rows=switches,
        action_switch_matrices=matrices,
        margin_strata=strata,
        equally_weighted_energy_contrasts=advantage,
        energy_specific_advantage_score=int(
            passed
            and all(
                a["bootstrap"]["lower_one_sided_975"] is not None
                and a["bootstrap"]["lower_one_sided_975"] > H1["lower_cost_gain_gt"]
                and a["improved_sources"] >= H1["improved_sources_min"]
                for a in advantage
            )
        ),
    )
