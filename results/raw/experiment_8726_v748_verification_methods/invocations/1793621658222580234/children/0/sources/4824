"""REQ-VERIFY-8190: hardware ceilings apply to the measured reuse service only."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from carnot.reporting import hardware_workload_8162 as old

Json = dict[str, Any]
CONFIG: Json = dict(old.CONFIG, seed=7078190, startup_amortization_requests=96)
CONDITIONS = ("cold_miss", "exact_repeat", "forced_refresh", "changed_source")
COMPONENTS = (
    "generation_ns",
    "key_hash_ns",
    "cache_read_ns",
    "cache_write_ns",
    "native_crossing_including_arithmetic_ns",
    "decision_write_ack_ns",
    "queue_ns",
    "load_ns",
    "parse_ns",
    "features_ns",
    "decision_serialization_ns",
    "other_host_and_ack_ns",
)
REOPEN = dict(
    KV260="Useful measured k<=5 workload, authenticated SSH/device fabric receipt and transfer-inclusive timing.",
    PolarFire="Real authenticated fabric dispatch with compatible operators and complete timing.",
    GateMate="Dated physical/JTAG change resolving0xffffffff, then authenticated dispatch and complete timing.",
    NPU="Authenticated access, supported radial/exp operators, guarded parity and transfer-inclusive timing.",
    TSU="Authenticated hardware/SDK access, supported operators and local acquisition/storage/transfer/readout timing.",
)


def reduce(data: Json) -> Json:
    """Retain all host work and keep independently qualified branches usable.

    Native crossing includes array handling and readout. Removing that whole
    timer gives an optimistic ceiling, not a measured pure-arithmetic speedup.
    Historical fabric custody cannot qualify a different Gaussian workload.
    """
    checks = deepcopy(data["checks"])
    checks.append(
        old.gate("exp8188", data["source"], "cached_service_ready_score", 1, data["service_score"])
    )
    rows, pairs = [], []
    for condition in CONDITIONS:
        selected = [r for r in data["requests"] if r["request"]["condition"] == condition]
        if not selected:
            selected = [
                dict(
                    request=dict(
                        unit_id=condition + ":missing",
                        source_cluster_id="exp8188",
                        condition=condition,
                        arm=condition,
                        status="excluded",
                    ),
                    components={},
                    qualified=False,
                )
            ]
        for item in selected:
            r, c = item["request"], item["components"]
            total, envelope = r.get("numerator"), c.get("native_crossing_including_arithmetic_ns")
            valid = bool(
                data["service_qualified"]
                and item["qualified"]
                and r["status"] == "completed"
                and old.finite(total)
                and all(old.finite(c.get(k)) for k in COMPONENTS)
                and sum(c[k] for k in COMPONENTS) == total
                and c.get("complete_latency_ns", total) == total
                and envelope < total
            )
            checks.append(
                old.gate(
                    "exp8188",
                    data["source"],
                    r["unit_id"] + ".component_cost_consistency",
                    True,
                    valid,
                )
            )
            exact = c.get("arithmetic_ns")
            exact_valid = valid and old.finite(exact) and exact <= envelope
            rows.append(
                dict(
                    unit_id=r["unit_id"],
                    source_cluster_id=r["source_cluster_id"],
                    arm=condition,
                    condition=condition,
                    metric="arithmetic_acceleration_ceiling",
                    numerator=total if valid else None,
                    denominator=total - envelope if valid else None,
                    status="completed" if valid else "excluded",
                    exclusion_reason=None if valid else "missing_or_unqualified_component_cost",
                    upstream="exp8188",
                    outer_ceiling=total / (total - envelope) if valid else None,
                    measured_arithmetic_fraction=exact / total if exact_valid else None,
                    exact_arithmetic_only_ceiling=1 / (1 - exact / total) if exact_valid else None,
                    scoring_envelope_fraction=envelope / total if valid else None,
                    retained_components={
                        k: c.get(k)
                        for k in COMPONENTS
                        if k != "native_crossing_including_arithmetic_ns"
                    },
                    retained_unknown_components=["transfer and readout inside native crossing"],
                    startup_amortized_ns=data["startup_ns"]
                    / CONFIG["startup_amortization_requests"],
                    cold_inclusive_outer_ceiling=(total + data["startup_ns"] / 96)
                    / (total - envelope + data["startup_ns"] / 96)
                    if valid
                    else None,
                    current_hardware_execution=False,
                    fixedpoint_arithmetic_executed=False,
                )
            )
            if valid:
                pairs.append(
                    dict(
                        unit_id=r["unit_id"],
                        status="completed",
                        condition=condition,
                        arms=[dict(requests=[r], components=dict(arithmetic_ns=envelope))],
                    )
                )
    boards = deepcopy(data["boards"])
    for board in boards:
        name = board["board"]
        valid = bool(
            board["custody_valid"]
            and (board.get("processor_class"), board.get("k_max"), board.get("blocker"))
            == old.prior.CONTRACTS[name]
        )
        board.update(
            custody_valid=valid,
            status="historical_qualified" if valid else "blocked_receipt",
            current_hardware_execution=False,
            current_reachability="not_probed",
            terminal_condition=REOPEN[name],
            terminal_condition_met=False,
            reopen_condition=REOPEN[name],
        )
        checks.append(
            old.gate(
                name,
                dict(path=board.get("source_path"), hash=board.get("source_hash")),
                "board_custody_" + name,
                True,
                valid,
            )
        )
    hardware_ready = int(len(boards) == 3 and all(b["custody_valid"] for b in boards))
    for name in ("NPU", "TSU"):
        boards.append(
            dict(
                board=name,
                status="blocked_authenticated_access",
                custody_valid=False,
                current_hardware_execution=False,
                current_reachability="not_probed",
                terminal_condition=REOPEN[name],
                terminal_condition_met=False,
                reopen_condition=REOPEN[name],
            )
        )
    quant, fallback = old.precision(
        dict(data, branches=dict(exp8159=dict(qualified=True, pairs=pairs)))
    )
    centers = len(data["state"].get("centers", []))
    for r in quant:
        r.update(
            packed_head_storage_bytes=math.ceil((10 * centers + 1) * r["bits"] / 8),
            float64_head_storage_bytes=8 * (10 * centers + 1),
            distance_coordinate_terms=9 * centers,
            gaussian_exponentials=centers,
            quantized_kernel_measured=False,
        )
    bounds = []
    for condition in CONDITIONS:
        chosen = [r for r in rows if r["condition"] == condition]
        usable = [r for r in chosen if r["status"] == "completed"]
        total = sum(r["numerator"] for r in usable)
        kept = sum(r["denominator"] for r in usable)
        exact = bool(usable) and all(r["measured_arithmetic_fraction"] is not None for r in usable)
        fraction = (
            sum(r["numerator"] * r["measured_arithmetic_fraction"] for r in usable) / total
            if exact
            else None
        )
        bounds.append(
            dict(
                condition=condition,
                arm=condition,
                upstream="exp8188",
                qualified_count=len(usable),
                excluded_count=len(chosen) - len(usable),
                total_ns=total if usable else None,
                retained_ns=kept if usable else None,
                outer_ceiling=total / kept if usable else None,
                measured_arithmetic_fraction=fraction,
                exact_arithmetic_only_ceiling=1 / (1 - fraction) if exact else None,
                required_arithmetic_fraction_for_100x=0.99,
                bound_scope="optimistic mixed native crossing; pure arithmetic unavailable unless separately measured",
            )
        )
    allrows = rows + quant + fallback
    completed = sum(r["status"] == "completed" for r in allrows)
    failed = [c for c in checks if not c["passed"]]
    ready = int(bool(pairs) and all(r["action_disagreements"] == 0 for r in quant))
    kind = "blocked" if failed or not ready else "circular_positive" if data["fixture"] else "null"
    return dict(
        honest_verdict="complete_blocked_" + (failed[0]["check"] if failed else "component_cost")
        if kind == "blocked"
        else "complete_" + kind + "_hardware_service_boundary",
        verdict_class=kind,
        verifier_is_oracle=data["fixture"],
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        hardware_boundary_ready_score=hardware_ready,
        measured_workload_bound_ready_score=ready,
        board_rows=boards,
        workload_rows=rows,
        quantization_rows=quant,
        fallback_cost_rows=fallback,
        amdahl_bounds=bounds,
        rows=allrows,
        gate_check_summary=checks,
        reopen_conditions=REOPEN,
        intended_count=len(allrows),
        eligible_count=completed,
        completed_count=completed,
        excluded_count=len(allrows) - completed,
        censored_count=0,
        failed_count=0,
        independent_count=0
        if data["fixture"]
        else len({r["source_cluster_id"] for r in rows if r["status"] == "completed"}),
        sample_size_budget=dict(
            unit="original source clusters; branches are dependent repeats", intended=len(allrows)
        ),
        trained_head_specs=data["trained_head_specs"],
        cited_upstream_artifacts=data["cited"],
        acceptance_gates=dict(
            required_arithmetic_fraction_for_100x=0.99, pure_arithmetic_required=True
        ),
        historical_complete_request_context=data["historical_context"],
        historical_composition=dict(
            experiment_id=8173,
            qualified=False,
            included_in_estimand=False,
            reason="unavailable historical composition remains outside this reuse workload",
        ),
        measured_repeat_frequency=None,
        cache_hit_weighting="hypothetical only; no deployment traffic",
        startup_costs=dict(
            charged_once=True, load_and_warmups_ns=data["startup_ns"], amortization_requests=96
        ),
        hardware_integration_executed=False,
        purchase_executed=False,
        flashing_executed=False,
        measured_bottleneck="Acquisition on misses/invalidation and durable storage/acknowledgement on cache hits.",
        hardware_spending_decision="defer: no measured whole-workload hardware benefit",
        external_context="Extropic projections do not replace local TSU timing.",
    )
