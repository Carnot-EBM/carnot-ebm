"""REQ-VERIFY-8238: frozen heads produce decisions without evaluator targets.

All source slots retain their original permissions and missing evidence. A seal
protects this invocation, while prior development exposure remains unchanged.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import margin_energy_training_8237 as fitted
from carnot.verify import restricted_sealed_rule_8209 as original
from carnot.verify import utility_kernel_8221 as kernel

Json = dict[str, Any]
ARMS = [
    "energy_uniform",
    "energy_margin",
    "additive_uniform",
    "additive_margin",
    "logistic_uniform",
    "logistic_margin",
]


def losses(p: float | None, baseline: str) -> Json:
    """Only historically allowed actions participate in expected loss minimization."""
    costs = dict(reject=None if p is None else 1 - p, escalate=0.5)
    if baseline == "accept":
        costs["accept"] = None if p is None else 5 * p
    return costs


def reduce(data: Json) -> Json:
    """Authenticate the public view before applying the unchanged fitted scoring API."""
    original.reject_labels(data)
    if [canonical_hash(h) for h in data["heads"]] != data["head_hashes"] or canonical_hash(
        data["global_head"]
    ) != data["global_head_hash"]:
        raise ValueError("head_hash")
    if (
        canonical_hash(data["roles"]) != data["roles_sha256"]
        or data["roles"] != data["public"]["frozen"]["roles"]
    ):
        raise ValueError("role_hash")
    heads = {h["arm"]: h for h in data["heads"]}
    if list(heads) != ARMS or len(data["heads"]) != 6:
        raise ValueError("arm_manifest")
    comparator = data["comparator"]
    if comparator["arm"] not in [*ARMS, "energy_global"] or comparator["sha256"] != canonical_hash(
        data["global_head"] if comparator["arm"] == "energy_global" else heads[comparator["arm"]]
    ):
        raise ValueError("comparator_hash")
    original.roster_check(data["public"]["roster"], data["roles"])
    if [r["slot"] for r in data["public"]["roster"]] != list(range(1, 129)) or [
        r["slot"] for r in data["public"]["features"]
    ] != list(range(1, 129)):
        raise ValueError("slot_order")
    native = data["native"]
    if len(native) != 128 or [(r["unit_id"], r["source_cluster_id"]) for r in native] != [
        (r["unit_id"], r["source_cluster_id"]) for r in data["public"]["roster"]
    ]:
        raise ValueError("native_source_join")
    base = original.reduce(data["public"])
    lookup = {(r["slot"], r["arm"]): r for r in base["prediction_rows"]}
    predictions, rows, permissions = [], [], []
    for index, (identity, source, cached) in enumerate(
        zip(base["rows"], data["public"]["features"], native, strict=True)
    ):
        slot = identity["slot"]
        baseline = lookup[slot, "original_frozen_v707_radial"]
        p0 = cached["p0"]
        query = dict(x=source["x"], p0=p0, baseline_action=baseline["action"])
        produced = [(h["arm"], fitted.score(h, query), canonical_hash(h)) for h in data["heads"]]
        missing = source["x"] is None or p0 is None or identity["status"] != "completed"
        reason = identity["exclusion_reason"] or (
            "missing_native_probability"
            if p0 is None
            else "missing_features"
            if source["x"] is None
            else None
        )
        old = lookup[slot, "energy"]
        global_p = (
            None
            if missing
            else kernel.predict(
                data["global_head"]["correction"], dict(old, baseline_p=baseline["p"])
            )
        )
        produced += [
            (
                "energy_global",
                dict(p_bad=global_p, action=kernel.rule.action(global_p, baseline["action"])),
                data["global_head_hash"],
            ),
            (
                "original_frozen_v707_radial",
                dict(
                    p_bad=None if missing else baseline["p"],
                    action="escalate" if missing else baseline["action"],
                ),
                baseline["head_sha256"],
            ),
            (
                "always_escalate",
                dict(p_bad=None, action="escalate"),
                canonical_hash("always_escalate"),
            ),
        ]
        native_losses = losses(p0, baseline["action"])
        ordered = sorted(v for v in native_losses.values() if v is not None)
        margin = None if p0 is None else ordered[1] - ordered[0]
        permission = dict(
            slot=slot,
            unit_id=identity["unit_id"],
            source_cluster_id=identity["source_cluster_id"],
            baseline_action=baseline["action"],
            accept_allowed=baseline["action"] == "accept",
        )
        permissions.append(permission)
        for arm, prediction, digest in produced:
            p = prediction["p_bad"]
            good, bad = kernel.energies(p) if p is not None else (None, None)
            normalized = kernel.rule.probability(good, bad, 1) if p is not None else None
            if p is not None and abs(p - normalized) > 1e-10:
                raise ValueError("energy_parity")
            action = prediction["action"]
            if action == "accept" and not permission["accept_allowed"]:
                raise ValueError("acceptance_subset_violation")
            record = dict(
                permission,
                arm=arm,
                condition="original_reserved_slot",
                p=p,
                p_bad=p,
                p0=p0,
                public_margin=margin,
                native_feasible_expected_losses=native_losses,
                feasible_expected_losses=losses(p, baseline["action"]),
                chosen_action=action,
                action=action,
                energy_good=good,
                energy_bad=bad,
                normalized_energy_probability=normalized,
                head_sha256=digest,
                primary_comparator=arm == comparator["arm"],
                baseline_mask=permission["accept_allowed"],
                original_status=identity["status"],
                status=identity["status"]
                if not missing or identity["status"] != "completed"
                else "excluded",
                missing_reason=reason,
                exclusion_reason=reason,
                missing_evidence=missing,
                source_sha256=source["source_sha256"],
                answer_sha256=source["answer_sha256"],
                feature_sha256=source["feature_sha256"],
                metric="sealed_probability",
                numerator=p,
                denominator=int(p is not None),
            )
            predictions.append(dict(record, prediction_sha256=canonical_hash(record)))
        status = (
            identity["status"] if not missing or identity["status"] != "completed" else "excluded"
        )
        rows.append(
            dict(
                identity,
                arm="all_frozen_margin_arms",
                status=status,
                original_status=identity["status"],
                exclusion_reason=reason,
                numerator=int(status == "completed"),
                denominator=1,
            )
        )
        if (index + 1) % 16 == 0:
            print(f"[exp8238] phase=score completed={index + 1} pending={127 - index}", flush=True)
    counts = Counter(r["status"] for r in rows)
    return dict(
        rows=rows,
        prediction_rows=predictions,
        permission_mask=permissions,
        permission_mask_sha256=canonical_hash(permissions),
        intended_count=128,
        independent_count=counts["completed"],
        **{k + "_count": counts[k] for k in ["completed", "failed", "censored", "excluded"]},
    )
