"""REQ-REPORT-7952-V689: preserve scientific units while reducing actual rows.

Repeated masks and seeds provide sensitivity checks. They cannot increase the
number of original source groups or turn descriptive contrasts into benefit.
"""

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from carnot.reporting import v688_capstone as qualified


def reduce_primitives(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reuse qualified scoring and audit masks, restart custody and paired units."""
    masks: Counter[str] = Counter()
    identities: dict[tuple[str, ...], Any] = {}
    paired: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    complete_spans = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed_primitive_row")
        mask = row.get("known_mask")
        if mask is not None:
            if not isinstance(mask, list) or any(
                type(v) is not int or v not in (-1, 0, 1) for v in mask
            ):
                raise ValueError("invalid_mask")
            masks[",".join(map(str, mask))] += 1
        if "restart_identity" in row:
            key = tuple(str(row.get(k)) for k in ("family_id", "arm", "seed", "role"))
            if key in identities and identities[key] != row["restart_identity"]:
                raise ValueError("restart_identity_changed")
            identities[key] = row["restart_identity"]
        if "started_monotonic_ns" in row and "ended_monotonic_ns" in row:
            if row["ended_monotonic_ns"] < row["started_monotonic_ns"]:
                raise ValueError("negative_service_span")
            complete_spans += int(row.get("service_span_complete") is True)
        if row.get("role") == "evaluation" and row.get("status") == "completed":
            if type(row.get("probability")) in (int, float) and row.get("label") in (0, 1):
                family = str(row.get("family_id", row.get("family", "unknown")))
                paired[family][str(row.get("arm", "default"))].append(
                    (row["probability"] - row["label"]) ** 2
                )
    value = qualified.reduce_primitives(rows)
    value["false_accepts_by_arm"] = value["unsupported_false_accepts_by_arm"]
    value["abstention_by_arm"] = {
        arm: sum(r.get("action") in {"abstain", "escalate"} for r in items)
        / max(1, sum("action" in r for r in items))
        for arm in {str(r.get("arm", "default")) for r in rows}
        if (items := [r for r in rows if str(r.get("arm", "default")) == arm])
    }
    arms = sorted({arm for items in paired.values() for arm in items})
    comparisons = {}
    source_groups = {
        str(r.get("family_id", r.get("family", "unknown"))): str(
            r.get("source_group", r.get("source_cluster_id", r.get("family_id", "unknown")))
        )
        for r in rows
    }
    for left, right in combinations(arms, 2):
        groups: dict[str, list[float]] = defaultdict(list)
        for family, arm_values in paired.items():
            if left in arm_values and right in arm_values:
                groups[source_groups[family]].append(
                    sum(arm_values[left]) / len(arm_values[left])
                    - sum(arm_values[right]) / len(arm_values[right])
                )
        means = [sum(items) / len(items) for items in groups.values()]
        comparisons[left + "__" + right] = dict(
            independent_source_groups=len(means),
            brier_left_minus_right=sum(means) / len(means) if means else None,
            descriptive_only=True,
        )
    value.update(
        mask_denominators=dict(masks),
        restart_identities=len(identities),
        paired_comparisons=comparisons,
        multiplicity=dict(comparisons=len(comparisons), adjusted_significance_claimed=False),
        complete_service_spans=complete_spans,
        complete_service_measurement_ready=bool(complete_spans),
        scientifically_independent_holdout_count=0,
    )
    return value
