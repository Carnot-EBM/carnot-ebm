"""REQ-VERIFY-8134: software ceilings retain unknown memory and boundary costs."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from carnot.reporting import hardware_batch_8121 as batch
from carnot.reporting import radial_hardware_8108 as prior
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
CONFIG: Json = dict(
    seed=7038134, centers=list(range(16, 29)), bits=[8, 12, 16], dimensions=9, pending_capacity=32
)
REOPEN = {
    "KV260": "New k<=5 fabric-compatible workload and authenticated SSH fabric transcript with transport and complete transaction timing.",
    "PolarFire": "Fabric capability and hash-verified dispatch with full timing; existing Linux CPU dispatch grants no fabric credit.",
    "GateMate": "Operator-authored dated cable, port, power or wiring change newer than Exp6525; valid GM1Ax IDCODE instead of 0xffffffff before flash or smoke.",
    "NPU": "Measured arithmetic bottleneck, supported radial/exp operators, guarded decision parity and transfer-inclusive native EP timing.",
    "TSU": "Accessible SDK and hardware, compatible sampler workload and measured acquisition, readout, transfer, fallback and persistence.",
}


def operations() -> list[Json]:
    """Count dense work and packed bytes; byte traffic has no measured bus price.

    Pending records store nine float64 features plus a target and sequence.
    Keys, object overhead and durable encoding are additional unknown bytes.
    """
    rows = []
    for centers in CONFIG["centers"]:
        for bits in CONFIG["bits"]:
            for size in [1, 32]:
                row = batch.traffic(size, centers, bits, size, "software_count_fixture")
                row.update(
                    coefficient_read_touches=size * (centers + 1),
                    coefficient_update_touches=centers + 1,
                    pending_event_capacity=32,
                    pending_event_numeric_bytes=32 * (9 * 8 + 8 + 8),
                    pending_event_total_bytes=None,
                    pending_event_omissions="keys, object overhead and durable serialization unknown",
                    transfer_latency_ns=None,
                    source_cluster_id="software_count_fixture",
                )
                rows.append(row)
    return rows


def reduce(data: Json) -> Json:
    """Separate authentic board history, fixture precision and workload cost gates."""
    boards = deepcopy(data["boards"])
    checks = deepcopy(data["checks"])
    for board in boards:
        observed = [board.get("processor_class"), board.get("k_max"), board.get("blocker")]
        expected = list(prior.CONTRACTS[board["board"]])
        valid = bool(board.get("custody_valid") and observed == expected)
        board.update(
            custody_valid=valid,
            current_hardware_execution=False,
            status="completed" if valid else "blocked",
            current_reachability="not_probed",
        )
        checks.append(
            dict(
                check="board_boundary",
                upstream=board["board"],
                path=board.get("source_path"),
                hash=board.get("source_hash"),
                artifact_field="recorded_workload_boundary",
                op="==",
                expected=expected,
                observed=observed,
                passed=valid,
            )
        )
    checks.append(
        dict(
            check="complete_service_ready_score",
            upstream="exp8132",
            path="results/experiment_8132_v703_service_cost.json",
            hash=None,
            artifact_field="complete_service_ready_score",
            op="==",
            expected=1,
            observed=int(data["complete_service"]),
            passed=bool(data["complete_service"]),
        )
    )
    op = operations()
    quant = []
    for index, system in enumerate(data["systems"]):
        for bits in CONFIG["bits"]:
            row = prior.numerical(system, bits)
            row["source_cluster_id"] = "exposed_numerical_fixture"
            quant.append(row)
        if (index + 1) % 16 == 0 or index + 1 == len(data["systems"]):
            print(
                f"[exp8134] precision completed={index + 1} pending={len(data['systems']) - index - 1}",
                flush=True,
            )
    costs = cost_rows(data)
    bounds = []
    groups = sorted({(r["arm"], r["condition"], r["workload_assumption"]) for r in costs})
    for arm, condition, scope in groups:
        selected = [
            r
            for r in costs
            if (r["arm"], r["condition"], r["workload_assumption"]) == (arm, condition, scope)
        ]
        usable = all(r["status"] == "completed" for r in selected)
        total = sum(r["numerator"] for r in selected) if usable else None
        kept = sum(r["denominator"] for r in selected) if usable else None
        ceiling = total / kept if usable else None
        bounds.append(
            dict(
                arm=arm,
                condition=condition,
                workload_assumption=scope,
                units=len(selected),
                total_ns=total,
                retained_ns=kept,
                outer_ceiling=ceiling,
                exact_arithmetic_only_ceiling=None,
                target_100x="ruled_out_even_by_outer_ceiling"
                if usable and ceiling < 100
                else "unknown",
                scope="optimistic score-envelope removal; arithmetic-only speedup cannot exceed this outer bound",
            )
        )
    precision_ok = all(r["action_disagreements"] == 0 for r in quant)
    ready = int(
        data["host_qualified"] and precision_ok and any(r["status"] == "completed" for r in costs)
    )
    verdict = (
        "blocked"
        if not data["complete_service"] or not ready
        else "circular_positive"
        if data["fixture"]
        else "null"
    )
    reason = (
        "host_component_cost"
        if not ready
        else "complete_service_ready_score"
        if verdict == "blocked"
        else "hardware_service_boundary"
    )
    rows = costs + op + quant
    return dict(
        honest_verdict="complete_"
        + ("blocked_" if verdict == "blocked" else verdict + "_")
        + reason,
        verdict_class=verdict,
        verifier_is_oracle=bool(data["fixture"]),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        hardware_boundary_ready_score=int(
            len(boards) == 3 and all(b["custody_valid"] for b in boards)
        ),
        measured_workload_bound_ready_score=ready,
        board_rows=boards,
        workload_operation_rows=op,
        quantization_rows=quant,
        fallback_cost_rows=[
            dict(
                unit_id=r["unit_id"],
                bits=r["bits"],
                requests=r["points"],
                fallback_requests=sum(r["fallback_flags"]),
                measured_incremental_fallback_ns=None,
                incremental_fallback_cost_status="unknown; retain float64 cost in future complete transactions",
            )
            for r in quant
        ],
        amdahl_bounds=bounds,
        rows=rows,
        gate_check_summary=checks,
        reopen_conditions=REOPEN,
        intended_count=len(rows),
        eligible_count=sum(r["status"] == "completed" for r in rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        failed_count=0,
        censored_count=0,
        independent_count=0,
        sample_size_budget=dict(
            unit="historical transaction repetitions and exposed software fixtures",
            intended=len(rows),
            independent_sources=0,
        ),
        acceptance_gates=dict(
            board_custody_independent=True,
            complete_service=bool(data["complete_service"]),
            host_component_rows=bool(ready),
            exact_decision_parity=precision_ok,
            required_arithmetic_fraction_for_100x=0.99,
        ),
        trained_head_specs=[
            dict(
                kind="historical supplied radial head",
                centers_range=[16, 28],
                dimensions=9,
                trained_currently=False,
            )
        ],
        cited_upstream_artifacts=data["cited"],
        purchase_recommendation="defer: no measured complete learning workload bottleneck or qualified device capability justifies a purchase",
        purchase_executed=False,
        cost_accounting_motivation="Z1T readout/transfer omission and FPGA decomposition motivate inclusion of omitted costs; no literature speed factors imported",
    )


def cost_rows(data: Json) -> list[Json]:
    """Use the entire timed scoring envelope only as a generous outer bound.

    Its pure arithmetic share is not separated from head reads and crossings.
    Removing the whole envelope overstates acceleration, so a ceiling below
    100 still rules out 100x; a larger ceiling proves no useful speedup.
    """
    rows: list[Json] = []
    models = {(r["unit_id"], r["arm"]): r for r in data["modeled"]}
    for pair in data["pairs"]:
        for arm in pair["arms"]:
            total, envelope, residual = (
                arm.get("full_latency_ns"),
                arm.get("arithmetic_ns"),
                arm.get("residual_ns"),
            )
            components = arm.get("components", {})
            valid = bool(
                data["host_qualified"]
                and pair["status"] == "completed"
                and components
                and all(
                    isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                    for v in components.values()
                )
                and isinstance(total, (int, float))
                and isinstance(envelope, (int, float))
                and isinstance(residual, (int, float))
                and 0 < envelope < total
                and residual >= 0
                and total == sum(components.values()) + residual
                and components.get("arithmetic_and_boundary_ns") == envelope
            )
            modeled = models.get((pair["unit_id"], arm["arm"]), {})
            for scope in [
                "cached_host_transaction",
                "modeled_full_service_reuse",
                "modeled_full_service_no_reuse",
            ]:
                acquisition = 0.0 if scope == "cached_host_transaction" else None
                if scope != "cached_host_transaction" and modeled.get("matched"):
                    request = modeled.get(
                        "historical_acquisition_s"
                        if scope.endswith("_reuse") and not scope.endswith("no_reuse")
                        else "no_reuse_acquisition_s"
                    )
                    load = modeled.get("historical_load_s")
                    if (
                        isinstance(request, (int, float))
                        and isinstance(load, (int, float))
                        and request >= 0
                        and load >= 0
                    ):
                        acquisition = (request + load) * 1e9
                usable = valid and acquisition is not None
                full = total + acquisition if usable else None
                kept = full - envelope if usable else None
                ceiling = full / kept if usable else None
                rows.append(
                    dict(
                        unit_id=pair["unit_id"] + ":" + scope,
                        source_cluster_id=canonical_hash(arm.get("source_cluster_ids", [])),
                        arm=arm["arm"],
                        condition=pair["condition"],
                        metric="arithmetic_envelope_outer_ceiling",
                        numerator=full,
                        denominator=kept,
                        status="completed" if usable else "excluded",
                        exclusion_reason=None if usable else "unknown_or_inconsistent_cost",
                        workload_assumption=scope,
                        outer_ceiling=ceiling,
                        exact_arithmetic_only_ceiling=None,
                        measured_scoring_envelope_ns=envelope,
                        retained_non_scoring_ns=total - envelope if valid else None,
                        acquisition_ns=acquisition,
                        retained_components=components,
                        retained_residual_ns=residual,
                        unknown_in_envelope="memory reads and native crossings; not declared free",
                        target_100x="ruled_out_even_by_outer_ceiling"
                        if usable and ceiling < 100
                        else "not_established",
                        directly_measured_complete_service=False,
                    )
                )
    rows.append(
        dict(
            unit_id="direct_full_service",
            source_cluster_id="exp8132",
            arm="unavailable",
            condition="natural_learning",
            metric="arithmetic_only_ceiling",
            numerator=None,
            denominator=None,
            status="excluded",
            exclusion_reason="complete_service_and_natural_update_cost_unknown",
            workload_assumption="directly_measured_full_service",
            outer_ceiling=None,
            exact_arithmetic_only_ceiling=None,
            target_100x="unknown",
        )
    )
    return rows
