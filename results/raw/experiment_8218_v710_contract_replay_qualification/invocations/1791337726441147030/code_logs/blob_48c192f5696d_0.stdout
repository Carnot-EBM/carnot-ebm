"""REQ-VERIFY-8209: apply frozen heads without giving scoring code targets.

Original missingness fixes the denominator. The permission mask limits actions
before minimization, so a cached probability cannot create a new acceptance.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import restricted_action_rule_8207 as rule

Json = dict[str, Any]
ARMS = [
    "energy",
    "additive",
    "logistic",
    "equivalent_logistic_identity",
    "original_frozen_v707_radial",
    "always_escalate",
]
PUBLIC = {
    "unit_id",
    "source_cluster_id",
    "slot",
    "x",
    "status",
    "exclusion_reason",
    "source_sha256",
    "answer_sha256",
    "feature_sha256",
    "arm",
    "condition",
    "metric",
    "numerator",
    "denominator",
    "role",
}


def reject_labels(value: Any) -> None:
    """Nested target aliases must fail before a public view can strip them away."""
    if isinstance(value, dict):
        if any(
            k in {"y", "target", "oracle_y", "human_target"} or "label" in k or "target" in k
            for k in value
        ):
            raise ValueError("evaluator_label")
        for item in value.values():
            reject_labels(item)
    elif isinstance(value, list):
        for item in value:
            reject_labels(item)


def roster_check(roster: list[Json], roles: Json) -> None:
    """Check identity and missingness before any feature vector is inspected."""
    reject_labels(roster)
    identities = [(r["unit_id"], r["source_cluster_id"]) for r in roster]
    if (
        len(roster) != 128
        or len(set(identities)) != 128
        or set(identities) != {(r["unit_id"], r["source_cluster_id"]) for r in roles["reserved"]}
    ):
        raise ValueError("original_source_join")
    if sorted(r["slot"] for r in roster) != list(range(1, 129)):
        raise ValueError("original_slot_join")
    if any(
        r["status"] not in {"completed", "failed", "excluded", "censored"}
        or r["numerator"] != int(r["status"] == "completed")
        for r in roster
    ):
        raise ValueError("original_missingness")


def subset_probe() -> list[Json]:
    """Exercise probability boundaries because missing inputs still need permission."""
    rows = []
    for baseline in ("accept", "reject", "escalate"):
        for p in (None, 0.0, 1e-300, 0.1, 0.1000000001, 0.5, 0.9999999999, 1.0):
            action = rule.action(p, baseline)
            passed = (action != "accept" or baseline == "accept") and (
                p is not None or action == "escalate"
            )
            rows.append(dict(baseline_action=baseline, p_bad=p, action=action, passed=passed))
    if not all(r["passed"] for r in rows):
        raise ValueError("acceptance_subset_violation")
    return rows


def reduce(data: Json) -> Json:
    """Recompute probabilities and raw energy gauges using only frozen bytes."""
    reject_labels(data)
    frozen, roster, features = data["frozen"], data["roster"], data["features"]
    if canonical_hash(frozen) != data["frozen_content_sha256"]:
        raise ValueError("modified_head")
    roster_check(roster, frozen["roles"])
    if len(features) != 128 or any(set(r) - PUBLIC for r in features):
        raise ValueError("feature_schema")
    joined = {r["slot"]: r for r in features}
    if len(joined) != 128 or set(joined) != set(range(1, 129)):
        raise ValueError("feature_slot_join")
    comparator = {r["unit_id"]: r for r in data["comparator"]}
    if len(comparator) != 128 or len(data["comparator"]) != 128:
        raise ValueError("unmatched_comparator")
    rows, predictions, parity, violations = [], [], [], []
    for index, identity in enumerate(sorted(roster, key=lambda r: r["slot"])):
        source = joined[identity["slot"]]
        if any(
            source[k] != identity[k]
            for k in ("unit_id", "source_cluster_id", "status", "exclusion_reason")
        ):
            raise ValueError("source_or_missingness_join")
        old = comparator[identity["unit_id"]]
        if old["source_cluster_id"] != identity["source_cluster_id"]:
            raise ValueError("comparator_source")
        complete = identity["status"] == "completed"
        x = source["x"] if complete else None
        if complete and (x is None or len(x) != 16 or not all(map(math.isfinite, x))):
            raise ValueError("feature_dimensions")
        query = dict(
            unit_id=identity["unit_id"],
            source_cluster_id=identity["source_cluster_id"],
            x=x,
            historical_x=x[:12] if x is not None else None,
        )
        fp, baseline_z = None, None
        if complete:
            baseline = frozen["baseline"]
            phi = rule.base.design("radial16", np.asarray([x[:12]]), baseline["geometry"])[0]
            offset, slope = baseline["calibration"]
            baseline_z = offset + slope * math.fsum(
                float(a) * float(b) for a, b in zip(phi, baseline["weights"], strict=True)
            )
            fp = float(expit(baseline_z))
        baseline_action = rule.base.action(fp)
        if (
            old["action"] != baseline_action
            or (old["p"] is None) != (fp is None)
            or (fp is not None and abs(old["p"] - fp) > 1e-10)
        ):
            raise ValueError("authentic_baseline")
        produced = []
        for head in frozen["heads"]:
            row = rule.predict(head, query, frozen["baseline"])
            z = (
                math.fsum(
                    float(a) * float(b)
                    for a, b in zip(
                        rule.design(head["arm"], np.asarray([x]), head["geometry"])[0],
                        head["weights"],
                        strict=True,
                    )
                )
                if complete
                else None
            )
            row.update(
                energies=[0.0, -z] if z is not None else [None, None],
                temperature=head["temperature"],
                head_sha256=canonical_hash(head),
            )
            produced.append(row)
            if head["arm"] == "energy":
                p = float(expit(z / head["temperature"])) if z is not None else None
                equivalent = dict(
                    row,
                    arm="equivalent_logistic_identity",
                    p=p,
                    action=rule.action(p, baseline_action),
                )
                if (p is not None and abs(p - row["p"]) > 1e-10) or equivalent["action"] != row[
                    "action"
                ]:
                    raise ValueError("energy_logistic_parity")
                produced.append(equivalent)
                parity.append(dict(slot=identity["slot"], passed=True))
        common = dict(
            unit_id=identity["unit_id"],
            source_cluster_id=identity["source_cluster_id"],
            baseline_action=baseline_action,
            allowed_actions=[
                "reject",
                "escalate",
                *(["accept"] if baseline_action == "accept" else []),
            ],
            temperature=1.0,
        )
        produced.extend(
            [
                dict(
                    common,
                    arm="original_frozen_v707_radial",
                    p=fp,
                    action=baseline_action,
                    energies=[0.0, -baseline_z] if baseline_z is not None else [None, None],
                    head_sha256=canonical_hash(frozen["baseline"]),
                ),
                dict(
                    common,
                    arm="always_escalate",
                    p=None,
                    action="escalate",
                    energies=[None, None],
                    head_sha256=canonical_hash("always_escalate"),
                ),
            ]
        )
        for row in produced:
            if row["action"] == "accept" and baseline_action != "accept":
                violations.append(dict(slot=identity["slot"], arm=row["arm"]))
            row.update(
                slot=identity["slot"],
                source_sha256=source["source_sha256"],
                answer_sha256=source["answer_sha256"],
                feature_sha256=source["feature_sha256"],
                status=identity["status"],
                exclusion_reason=identity["exclusion_reason"],
                p_bad=row["p"],
                chosen_action=row["action"],
                metric="sealed_probability",
                numerator=row["p"],
                denominator=int(row["p"] is not None),
                primary_simple_control=row["arm"] == frozen["selected_simple_control"]["selected"],
            )
            predictions.append(dict(row, prediction_sha256=canonical_hash(row)))
        rows.append(
            dict(
                identity,
                arm="all_six_frozen_arms",
                condition="original_reserved_slots",
                metric="complete_paired_source",
                denominator=1,
            )
        )
        if (index + 1) % 16 == 0:
            print(f"[exp8209] phase=score completed={index + 1} pending={127 - index}", flush=True)
    if violations:
        raise ValueError("acceptance_subset_violation")
    counts = Counter(r["status"] for r in rows)
    return dict(
        rows=rows,
        prediction_rows=predictions,
        intended_count=128,
        completed_count=counts["completed"],
        independent_count=counts["completed"],
        failed_count=counts["failed"],
        excluded_count=counts["excluded"],
        censored_count=counts["censored"],
        acceptance_subset_violations=violations,
        equivalent_logistic_parity=dict(passed=True, rows=parity),
    )
