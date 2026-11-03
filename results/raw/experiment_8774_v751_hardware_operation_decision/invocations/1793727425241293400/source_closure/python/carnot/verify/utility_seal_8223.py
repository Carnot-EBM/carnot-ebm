"""REQ-VERIFY-8223: saved corrections can change decisions without opening targets.

The original public rows define both available evidence and accept permission.
Repeated sources remain development data even when their predictions are sealed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import restricted_sealed_rule_8209 as original
from carnot.verify import utility_fit_8222 as fitted
from carnot.verify import utility_kernel_8221 as kernel

Json = dict[str, Any]


def reduce(data: Json) -> Json:
    """Recompute every base probability before applying the saved probability tree.

    Target aliases fail before any input view is constructed. State hashes bind
    corrections and base heads, so a correction cannot silently change later.
    """
    original.reject_labels(data)
    parameters, public = data["parameters"], data["public"]
    if canonical_hash(parameters) != data["parameters_sha256"]:
        raise ValueError("modified_head")
    if (
        parameters["base_heads"] != public["frozen"]["heads"]
        or parameters["baseline"] != public["frozen"]["baseline"]
    ):
        raise ValueError("modified_head")
    if parameters["role_hashes"] != fitted.role_hashes(dict(roles=public["frozen"]["roles"])):
        raise ValueError("role_hash")
    expected = {a + ":" + (str(s) if s is not None else "none") for a, s in fitted.SPECS}
    if set(parameters["models"]) != expected:
        raise ValueError("arm_manifest")
    base = original.reduce(public)
    lookup = {(r["slot"], r["arm"]): r for r in base["prediction_rows"]}
    predictions, permissions, max_error = [], [], 0.0
    for identity in base["rows"]:
        slot = identity["slot"]
        baseline = lookup[slot, "original_frozen_v707_radial"]
        permissions.append(
            dict(
                slot=slot,
                unit_id=identity["unit_id"],
                source_cluster_id=identity["source_cluster_id"],
                baseline_action=baseline["action"],
                allowed_actions=baseline["allowed_actions"],
            )
        )
        produced = []
        for arm, seed in fitted.SPECS:
            key = arm + ":" + (str(seed) if seed is not None else "none")
            head_arm = arm.split("_")[0]
            model = parameters["models"][key]
            source = lookup[slot, head_arm]
            public_row = dict(source, baseline_p=baseline["p"])
            p = kernel.predict(model, public_row)
            good, bad = kernel.energies(p) if p is not None else (None, None)
            energy_p = kernel.rule.probability(good, bad, 1) if p is not None else None
            error = abs(p - energy_p) if p is not None else 0.0
            if error > 1e-10:
                raise ValueError("energy_parity")
            max_error = max(max_error, error)
            action = kernel.rule.action(p, baseline["action"])
            state = dict(
                base_head=next(h for h in parameters["base_heads"] if h["arm"] == head_arm),
                model=model,
                baseline=parameters["baseline"],
            )
            produced.append(
                dict(
                    source,
                    arm=arm,
                    seed=seed,
                    p=p,
                    p_bad=p,
                    action=action,
                    chosen_action=action,
                    energy_good=good,
                    energy_bad=bad,
                    normalized_energy_probability=energy_p,
                    state_sha256=canonical_hash(state),
                    primary_comparator=arm == data["comparator"]["arm"] and seed is None,
                    mandatory_group_control=arm in ["additive_group", "logistic_group"],
                )
            )
        produced.extend(
            dict(
                lookup[slot, arm],
                seed=None,
                state_sha256=lookup[slot, arm]["head_sha256"],
                primary_comparator=False,
                mandatory_group_control=False,
            )
            for arm in ["original_frozen_v707_radial", "always_escalate"]
        )
        for row in produced:
            row.pop("prediction_sha256", None)
            row.update(
                metric="sealed_probability",
                numerator=row["p"],
                denominator=int(row["p"] is not None),
                condition="original_reserved_slot",
                missing_evidence=identity["status"] != "completed",
            )
            row["prediction_sha256"] = canonical_hash(row)
            predictions.append(row)
        if slot % 16 == 0:
            print(
                f"[exp8223] phase=corrected_score completed={slot} pending={128 - slot}", flush=True
            )
    rows = [
        dict(r, arm="all_corrected_arms", condition="original_reserved_slot") for r in base["rows"]
    ]
    return dict(
        rows=rows,
        prediction_rows=predictions,
        permission_mask=permissions,
        permission_mask_sha256=canonical_hash(permissions),
        energy_parity_max_error=max_error,
        **{
            k: base[k]
            for k in [
                "intended_count",
                "completed_count",
                "failed_count",
                "excluded_count",
                "censored_count",
                "independent_count",
            ]
        },
    )
