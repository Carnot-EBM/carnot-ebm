"""REQ-VERIFY-8203: preserve set decisions while bounding only owned CPU work.

The radial basis stays in float64 on this CPU. Lower precision applies to
multiply and accumulation only, so these rows cannot establish a fabric kernel.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from carnot.reporting.hardware_service_8190 import REOPEN
from carnot.verify import selective_rule_8194 as selective

Json = dict[str, Any]
CONFIG: Json = dict(
    seed=7088203,
    precisions=["float64", "float32", "fixed16"],
    fixed_max=32767,
    probability_roundoff_allowance=1e-12,
    required_arithmetic_fraction_for_100x=0.99,
)


def action(labels: list[int]) -> str:
    """Ambiguous and empty sets escalate because neither supports a decision."""
    return "accept" if labels == [0] else "reject" if labels == [1] else "escalate"


def precision(head: Json, samples: list[Json], training: list[Json]) -> tuple[list[Json], Json]:
    """Freeze scales before measurement; intervals guard exact reference ties.

    Operand error plus an accumulation bound limits the logit error. Sigmoid is
    1/(4T) Lipschitz. A host transcendental allowance is explicit, not a claim
    about every library or hardware implementation.
    """
    dimensions = head["dimensions"]
    train = np.asarray([r["x"][:dimensions] for r in training], dtype=np.float64)
    lower, upper = train.min(axis=0), train.max(axis=0)
    weights = np.asarray(head["weights"], dtype=np.float64)
    ws = max(float(np.max(np.abs(weights))), 1e-30) / CONFIG["fixed_max"]
    ps = 1.0 / CONFIG["fixed_max"]
    wi = np.rint(weights / ws).astype(np.int16)
    domain = dict(
        arm=head["arm"],
        lower=lower.tolist(),
        upper=upper.tolist(),
        basis_bounds=[0, 1],
        weight_scale=ws,
        basis_scale=ps,
        scale_source="training bounds and frozen weights",
        quantiles=head["quantiles"],
        scope="shared float64 radial basis; lower-precision multiply/accumulate",
        unsupported_operations=["native radial exponential", "spline lookup"],
        host_exp_sigmoid_absolute_allowance=CONFIG["probability_roundoff_allowance"],
    )
    rows = []
    for index, source in enumerate(samples):
        phi = selective.basis(head, source["x"]).astype(np.float64)
        z = math.fsum(float(a) * float(b) for a, b in zip(phi, weights, strict=True))
        p = selective.logit_probability([0, z], head["temperature"])
        reference_set = selective.prediction_set(p, head["quantiles"])
        outside = bool(
            np.any(np.asarray(source["x"][:dimensions]) < lower)
            or np.any(np.asarray(source["x"][:dimensions]) > upper)
        )
        for kind in CONFIG["precisions"]:
            began = time.perf_counter_ns()
            if kind == "float64":
                approximate, error = p, 0.0
            else:
                if kind == "float32":
                    a, b = phi.astype(np.float32), weights.astype(np.float32)
                    qz = float(np.sum(a * b, dtype=np.float32))
                    unit = float(np.finfo(np.float32).eps)
                    gamma = (2 * len(phi) * unit) / (1 - 2 * len(phi) * unit)
                    rounding = gamma * float(np.sum(np.abs(a.astype(float) * b.astype(float))))
                else:
                    ai = np.rint(phi / ps).astype(np.int16)
                    qz = int(np.sum(ai.astype(np.int64) * wi.astype(np.int64))) * ps * ws
                    a, b = ai.astype(float) * ps, wi.astype(float) * ws
                    rounding = 1e-12 * (1 + abs(qz))
                operand = float(
                    np.sum(np.abs(phi - a) * np.abs(weights) + np.abs(a) * np.abs(weights - b))
                )
                approximate = selective.logit_probability([0, qz], head["temperature"])
                error = (operand + rounding) / (4 * head["temperature"]) + 1e-12
            approximate_set = selective.prediction_set(approximate, head["quantiles"])
            uncertain = any(
                not q["infinity"]
                and abs(score - q["threshold"])
                <= error + abs(float(np.float32(q["threshold"])) - q["threshold"])
                for score, q in zip([approximate, 1 - approximate], head["quantiles"], strict=True)
            )
            fallback = kind != "float64" and (outside or uncertain)
            fallback_ns = 0
            if fallback:
                start = time.perf_counter_ns()
                final_p = selective.probability(head, source["x"], scalar=True)
                final_set = selective.prediction_set(final_p, head["quantiles"])
                fallback_ns = time.perf_counter_ns() - start
            else:
                final_set = approximate_set
            elapsed = time.perf_counter_ns() - began
            rows.append(
                dict(
                    unit_id=source["unit_id"],
                    source_cluster_id=source["source_cluster_id"],
                    arm=head["arm"],
                    condition="exposed_frozen_head",
                    precision=kind,
                    metric="final_decision_mismatch",
                    decision_mismatch_rate=int(final_set != reference_set),
                    numerator=int(final_set != reference_set),
                    denominator=1,
                    status="completed",
                    exclusion_reason=None,
                    probability_reference=p,
                    probability_approximate=approximate,
                    probability_error=abs(p - approximate),
                    probability_interval_radius=error,
                    reference_set=reference_set,
                    approximate_set=approximate_set,
                    raw_membership_mismatch=int(approximate_set != reference_set),
                    final_membership_mismatch=int(final_set != reference_set),
                    raw_typed_mismatch=int(action(approximate_set) != action(reference_set)),
                    final_typed_mismatch=int(action(final_set) != action(reference_set)),
                    fallback=fallback,
                    outside_domain=outside,
                    uncertain_membership=uncertain,
                    fallback_ns=fallback_ns,
                    elapsed_ns=elapsed,
                    operation_counts=dict(
                        scope="per standalone predictor; reference radial basis is shared between precision variants",
                        distance_coordinate_terms=dimensions * (len(phi) - 1),
                        gaussian_exponentials=len(phi) - 1,
                        multiplies=len(phi),
                        additions=len(phi) - 1,
                        membership_comparisons=2,
                        sigmoid=1,
                        float64_fallbacks=int(fallback),
                    ),
                )
            )
        if index % 32 == 0 or index + 1 == len(samples):
            print(
                f"[exp8203] precision completed={index + 1} pending={len(samples) - index - 1}",
                flush=True,
            )
    return rows, domain


def costs(source: list[Json]) -> tuple[list[Json], list[Json]]:
    """Mixed timers remove an envelope optimistically; pure arithmetic is separate."""
    rows, bounds = [], []
    for r in source:
        if r["status"] != "completed":
            continue
        total, kept = r["numerator"], r["denominator"]
        retained = r.get("retained_components", {})
        arithmetic = r.get("measured_arithmetic_fraction")
        storage = sum(retained.get(k, 0) for k in ("cache_write_ns", "decision_write_ack_ns"))
        acquisition = retained.get("generation_ns", 0)
        rows.append(
            dict(
                r,
                arithmetic_ns=total * arithmetic if arithmetic is not None else None,
                acquisition_ns=acquisition,
                durable_storage_ns=storage,
                orchestration_ns=sum(retained.values()) - storage - acquisition,
                transfer_readout_ns=None,
                mixed_arithmetic_transfer_readout_ns=total - kept,
                scope="historical research request; no deployment demand",
            )
        )
    for condition in sorted({r["condition"] for r in rows}):
        chosen = [r for r in rows if r["condition"] == condition]
        total, kept = sum(r["numerator"] for r in chosen), sum(r["denominator"] for r in chosen)
        exact = all(r["arithmetic_ns"] is not None for r in chosen)
        fraction = sum(r["arithmetic_ns"] for r in chosen) / total if exact else None
        bounds.append(
            dict(
                condition=condition,
                total_ns=total,
                retained_ns=kept,
                qualified_count=len(chosen),
                optimistic_ceiling=total / kept,
                removable_arithmetic_fraction=fraction,
                arithmetic_only_ceiling=1 / (1 - fraction) if exact else None,
                required_arithmetic_fraction_for_100x=0.99,
                supports_100x=exact and fraction >= 0.99,
                timer_scope="mixed timer is optimistic only",
            )
        )
    return rows, bounds


def reduce(data: Json) -> Json:
    """Independent branches retain evidence even when another prerequisite fails."""
    precision_rows, domains, excluded_rows = [], [], []
    for head in data["heads"]:
        print(f"[exp8203] before_benchmark head={head['arm']} completed=0 pending=1", flush=True)
        measured, domain = precision(
            head,
            head.get("bounded_samples", data["samples"]),
            head.get("bounded_training", data["training"]),
        )
        precision_rows.extend(measured)
        domains.append(domain)
        for source in head.get("excluded_samples", data.get("excluded_samples", [])):
            for kind in CONFIG["precisions"]:
                excluded_rows.append(
                    dict(
                        unit_id=source["unit_id"],
                        source_cluster_id=source["source_cluster_id"],
                        arm=head["arm"],
                        condition="missing_source_evidence",
                        precision=kind,
                        metric="final_decision_mismatch",
                        numerator=None,
                        denominator=1,
                        status="excluded",
                        exclusion_reason=source["exclusion_reason"],
                    )
                )
        print(f"[exp8203] after_benchmark head={head['arm']} completed=1 pending=0", flush=True)
    workload, bounds = costs(data["history"].get("workload_rows", []))
    failures = [c for c in data["checks"] if not c["passed"]]
    missing = [name for name, ready in data["branches"].items() if not ready]
    mismatch = any(
        r["final_membership_mismatch"] or r["final_typed_mismatch"] for r in precision_rows
    )
    check = failures[0]["check"] if failures else missing[0] + "_exists" if missing else ""
    kind = "blocked" if check else "null"
    if mismatch:
        kind, check = "disqualified", "software_parity"
    rows = precision_rows + workload + excluded_rows
    count = len(rows)
    fallback = [
        dict(
            unit_id=r["unit_id"],
            source_cluster_id=r["source_cluster_id"],
            arm=r["arm"],
            precision=r["precision"],
            fallback=r["fallback"],
            measured_incremental_fallback_ns=r["fallback_ns"],
        )
        for r in precision_rows
        if r["precision"] != "float64"
    ]
    return dict(
        honest_verdict="complete_" + kind + "_" + (check or "hardware_decision_boundary"),
        verdict_class=kind,
        verifier_is_oracle=bool(data.get("fixture")),
        hardware_boundary_ready_score=int(not check and not mismatch and not data.get("fixture")),
        branch_readiness=data["branches"],
        precision_rows=precision_rows,
        interval_bound_domain=domains,
        fallback_cost_rows=fallback,
        workload_rows=workload,
        amdahl_bounds=bounds,
        board_rows=data["boards"],
        rows=rows,
        gate_check_summary=data["checks"],
        intended_count=count,
        eligible_count=count - len(excluded_rows),
        completed_count=count - len(excluded_rows),
        independent_count=0,
        excluded_count=len(excluded_rows),
        censored_count=0,
        failed_count=0,
        sample_size_budget=dict(
            unit="original exposed source clusters; dependent precision arms",
            intended=count,
            available_source_count=len({r["source_cluster_id"] for r in rows}),
        ),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        current_device_execution_count=0,
        reopen_conditions=REOPEN,
        trained_head_specs=data["trained_head_specs"],
        cited_upstream_artifacts=data["cited"],
        acceptance_gates=dict(CONFIG, zero_final_decision_mismatches=True),
        hardware_spending_decision="defer; no purchase authorized; no100x complete-workload evidence",
        hardware_decisions=[
            dict(tier="CPU counter/update", decision="retain CPU integer counters and updates"),
            dict(
                tier="durable radial memory",
                decision="retain CPU exponentials and durable storage; learning prerequisite must qualify",
            ),
            dict(
                tier="future batched predictor",
                decision="defer until observed demand amortizes acquisition and transfer",
            ),
            dict(
                tier="deferred structure change",
                decision="no hardware integration or flashing scheduled",
            ),
        ],
        required_redesign="Reduce original acquisition and durable commit costs; kernel acceleration alone cannot deliver100x.",
        measured_repeat_frequency=data.get("research_reuse_frequency"),
        deployment_demand_observed=False,
        deployment_claim=False,
        current_model_calls=0,
        probability_error_summary=[
            dict(
                precision=k,
                maximum_error=max(
                    (r["probability_error"] for r in precision_rows if r["precision"] == k),
                    default=0,
                ),
                final_mismatches=sum(
                    r["final_typed_mismatch"] for r in precision_rows if r["precision"] == k
                ),
                fallback_count=sum(r["fallback"] for r in precision_rows if r["precision"] == k),
            )
            for k in CONFIG["precisions"]
        ],
    )
