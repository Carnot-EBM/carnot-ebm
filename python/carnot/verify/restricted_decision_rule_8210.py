"""REQ-VERIFY-8210: independent human costs separate policy and energy gains.

The same source appears in several arms, so paired source draws preserve their
dependence. Missing slots retain their escalation cost in the primary mean.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import restricted_sealed_evaluation_8209 as sealed
from carnot.verify import selective_decision_audit_8197 as historical

Json = dict[str, Any]
ARMS = sealed.n.ARMS
COSTS = dict(correct=0, escalate=0.5, false_accept=5, false_reject=1)
H1 = dict(
    alpha=0.025,
    baseline_cost_degradation_max=0.02,
    brier_degradation_max=0.01,
    comparison="energy versus tune-selected additive/logistic with identical mask",
    confidence=0.975,
    draws=10000,
    extra_false_accepts_max=0,
    improved_sources_min=5,
    intended=128,
    lower_cost_gain_gt=0.02,
    minimum_complete=96,
    minimum_per_class=12,
    minimum_valid_draws=9500,
    missing_slots="all-slot costs include escalation; support gate uses complete pairs",
    resampling="paired source-cluster bootstrap with original missing masks",
    seed=7098207,
    unit="original_source_cluster",
)


def contrast(groups: list[Json], comparator: str) -> Json:
    """The fixed bootstrap counts sources once and never resamples arms apart."""
    gains = [g[comparator]["numerator"] - g["energy"]["numerator"] for g in groups]
    with patch.dict(historical.old.CONFIG, seed=H1["seed"]):
        result = historical.old.interval(gains)
    return dict(result, comparator=comparator, treatment="energy", denominator=len(groups))


def statistics(rows: list[Json], control: str) -> Json:
    """Every frozen gate remains visible even when a stronger gate already fails."""
    groups: Json = {}
    for row in rows:
        group = groups.setdefault(row["source_cluster_id"], {})
        if row["arm"] in group:
            raise ValueError("duplicate_source_arm")
        group[row["arm"]] = row
    if any(set(g) != set(ARMS) for g in groups.values()):
        raise ValueError("unpaired_source")
    paired = list(groups.values())
    complete = [g for g in paired if all(g[a]["complete_pair"] for a in ARMS)]
    support = Counter(g["energy"]["y"] for g in complete)
    increment, policy = contrast(paired, control), contrast(paired, "original_frozen_v707_radial")
    improved = sum(g["energy"]["numerator"] < g[control]["numerator"] for g in paired)
    extra = sum(g["energy"]["false_accept"] and not g[control]["false_accept"] for g in paired)
    original_extra = sum(
        g["energy"]["false_accept"] and not g["original_frozen_v707_radial"]["false_accept"]
        for g in paired
    )
    brier = (
        math.fsum(g["energy"]["brier"] - g[control]["brier"] for g in complete) / len(complete)
        if complete
        else None
    )
    degradation = -policy["mean_gain"] if paired else None
    operands = dict(
        all_slots=dict(observed=len(paired), op="==", expected=128),
        complete_sources=dict(observed=len(complete), op=">=", expected=96),
        per_class=dict(observed=min(support[0], support[1]), op=">=", expected=12),
        valid_draws=dict(observed=increment["valid_draws"], op=">=", expected=9500),
        lower_gain=dict(observed=increment["lower_one_sided_975"], op=">", expected=0.02),
        improved_sources=dict(observed=improved, op=">=", expected=5),
        extra_false_accepts=dict(observed=extra, op="<=", expected=0),
        original_extra_false_accepts=dict(observed=original_extra, op="<=", expected=0),
        brier_increase=dict(observed=brier, op="<=", expected=0.01),
        original_baseline_cost_increase=dict(observed=degradation, op="<=", expected=0.02),
        nontrivial_actions=dict(
            observed=sum(g["energy"]["action"] != "escalate" for g in paired), op=">", expected=0
        ),
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
    metrics, risks = [], []
    for arm in ARMS:
        rs = [g[arm] for g in paired]
        accepts = [r for r in rs if r["action"] == "accept"]
        labeled = [r for r in accepts if r["y"] in (0, 1)]
        bad = [r for r in rs if r["y"] == 1]
        cost = math.fsum(r["numerator"] for r in rs)
        metrics.append(
            dict(
                arm=arm,
                all_slot_cost=dict(numerator=cost, denominator=128, mean=cost / 128),
                complete_pair_cost=dict(
                    numerator=math.fsum(g[arm]["numerator"] for g in complete),
                    denominator=len(complete),
                ),
                brier=dict(
                    numerator=math.fsum(
                        g[arm]["brier"] for g in complete if g[arm]["brier"] is not None
                    ),
                    denominator=len(complete) if arm != "always_escalate" else 0,
                ),
            )
        )
        risks.append(
            dict(
                arm=arm,
                acceptance_coverage=historical.binomial(len(accepts), len(rs)),
                false_accept_count=sum(r["false_accept"] for r in rs),
                conditional_accepted_error=historical.binomial(
                    sum(r["y"] == 1 for r in labeled), len(labeled)
                ),
                false_accept_given_unsupported=historical.binomial(
                    sum(r["false_accept"] for r in bad), len(bad)
                ),
                population_safety_claim=False,
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
        policy_gain=dict(policy, primary=False),
        energy_increment_gain=dict(increment, primary=True),
        complete_pair_contrasts=dict(
            energy_increment=contrast(complete, control),
            policy=contrast(complete, "original_frozen_v707_radial"),
        ),
        all_slot_metrics=metrics,
        acceptance_risk_rows=risks,
        h1_development_signal_score=int(passed),
        relative_count_principle="Acceptance subset bounds relative false-accept counts for all targets; it is not population safety or a semantic verifier claim.",
    )


def reduce(data: Json) -> Json:
    """Rebuild sealed scores, then join original annotations without producer costs."""
    if not 0 < data["clock"]["predictions_sealed_ns"] < data["clock"]["labels_opened_ns"]:
        raise ValueError("prediction_label_access_order")
    if data["costs"] != COSTS or data["H1"] != H1:
        raise ValueError("frozen_protocol")
    predictions = sealed.n.reduce(data["sealed"])["prediction_rows"]
    historical.old.base.audit.equal(predictions, data["predictions"])
    slots = {r["unit_id"]: r for r in data["slots"]}
    originals = {str(r["id"]): r for r in data["original_response_records"]}
    if (
        len(slots) != 128
        or len(data["slots"]) != 128
        or len(originals) != 128
        or len(data["original_response_records"]) != 128
    ):
        raise ValueError("original_label_roster")
    targets: Json = {}
    for feature in data["sealed"]["features"]:
        slot = slots[feature["unit_id"]]
        original = originals[slot["response_id"]]
        if (
            str(original["source_id"]) != slot["source_id"]
            or feature["source_cluster_id"] != slot["source_cluster_id"]
        ):
            raise ValueError("original_source_identity")
        if any(
            feature[k + "_sha256"] != canonical_hash(slot[k + "_bytes"])
            for k in ("answer", "source")
        ):
            raise ValueError("original_byte_identity")
        target, _ = historical.old.base.human.target(original, bytes.fromhex(slot["answer_bytes"]))
        if target["y"] not in (0, 1) and feature["status"] == "completed":
            raise ValueError("original_annotation_not_qualified")
        targets[feature["unit_id"]] = target
    rows = []
    for prediction in predictions:
        y = targets[prediction["unit_id"]]["y"]
        action, p = prediction["action"], prediction["p"]
        cost = (
            COSTS["escalate"]
            if action == "escalate"
            else COSTS["false_accept"] * y
            if action == "accept"
            else COSTS["false_reject"] * (1 - y)
        )
        rows.append(
            dict(
                prediction,
                y=y,
                metric="typed_decision_cost",
                numerator=cost,
                denominator=1,
                complete_pair=prediction["status"] == "completed" and y in (0, 1),
                brier=(p - y) ** 2 if p is not None and y in (0, 1) else None,
                false_accept=int(action == "accept" and y == 1),
            )
        )
    counts = Counter(r["status"] for r in data["sealed"]["roster"])
    return dict(
        rows=rows,
        intended_count=128,
        completed_count=counts["completed"],
        independent_count=counts["completed"],
        failed_count=counts["failed"],
        censored_count=counts["censored"],
        excluded_count=counts["excluded"],
        equivalent_logistic_parity=dict(
            passed=all(
                g["action"] == h["action"]
                and g["numerator"] == h["numerator"]
                and (g["p"] is None or abs(g["p"] - h["p"]) <= 1e-10)
                for g, h in zip(
                    [r for r in rows if r["arm"] == "energy"],
                    [r for r in rows if r["arm"] == "equivalent_logistic_identity"],
                    strict=True,
                )
            )
        ),
        **statistics(rows, data["sealed"]["frozen"]["selected_simple_control"]["selected"]),
    )
