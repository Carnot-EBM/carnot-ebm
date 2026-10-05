"""REQ-VERIFY-8162: keep durable batches and composed requests in separate sums."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from carnot.reporting import hardware_workload_8148 as old
from carnot.reporting import hardware_service_8134 as previous
from carnot.reporting import radial_hardware_8108 as prior
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.hardware_workload_inputs_8162 import FIELDS

Json = dict[str, Any]
CONFIG: Json = dict(
    seed=7058162,
    bits=[8, 12, 16],
    dimensions=9,
    target_speedup=100,
    precision="quantized storage with float64 arithmetic; fixed-point operators unimplemented",
)


def finite(value: Any) -> bool:
    """Reject missing costs rather than treating missing measurements as zero."""
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def gate(branch: str, source: Json, field: str, expected: Any, observed: Any) -> Json:
    """Keep exact operands so a terminal block identifies the original failed field."""
    return dict(
        check=field,
        upstream=branch,
        path=source.get("path"),
        hash=source.get("hash"),
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


def costs(data: Json, checks: list[Json]) -> list[Json]:
    """Remove only qualified arithmetic and keep acquisition and durable host work.

    The upstream scoring clock includes array preparation and readout. Its whole
    duration therefore supplies an upper bound on removable arithmetic; it does
    not measure pure arithmetic. Exact arithmetic remains unknown unless a
    separate pure-arithmetic operand exists. Overlapping request waits are kept
    in request latency and are never added to elapsed batch time a second time.
    """
    rows = []
    centers = len(data["state"].get("centers", []))
    for branch, source in data["branches"].items():
        for pair in source["pairs"]:
            for arm in pair["arms"]:
                components = arm["components"]
                total = arm["duration_ns"]
                envelope = components.get("arithmetic_ns")
                valid = bool(
                    source["qualified"]
                    and pair.get("status", "completed") == "completed"
                    and finite(total)
                    and finite(envelope)
                    and all(finite(v) for v in components.values())
                    and sum(components.values()) == total
                    and envelope < total
                )
                operands = dict(duration_ns=total, **components)
                operands.setdefault("arithmetic_ns", envelope)
                for key, value in operands.items():
                    if not finite(value):
                        checks.append(
                            gate(
                                branch,
                                source,
                                pair["unit_id"] + "." + arm["arm"] + "." + key,
                                "finite_nonnegative",
                                value,
                            )
                        )
                requests = arm["requests"]
                count = len(requests)
                units = (
                    requests
                    if branch == "exp8160"
                    else [
                        dict(
                            source_cluster_id=canonical_hash(
                                sorted({r["source_cluster_id"] for r in requests})
                            )
                        )
                    ]
                )
                for index, unit in enumerate(units):
                    retained = {k: v for k, v in components.items() if k != "arithmetic_ns"}
                    current = total
                    removable = envelope
                    exact = arm.get("arithmetic_only_ns")
                    usable = valid and count > 0
                    if branch == "exp8160":
                        capture = next(
                            (
                                r
                                for r in source.get("captures", [])
                                if r["source_cluster_id"] == unit["source_cluster_id"]
                            ),
                            {},
                        )
                        acquisition = capture.get("acquisition_ns")
                        startup = source.get("startup_amortized_ns")
                        overhead = arm.get("request_overhead_ns")
                        if overhead is None:
                            clocks = [r.get("response_ns") for r in requests] + [
                                r.get("enqueue_ns") for r in requests
                            ]
                            overhead = (
                                max(
                                    0,
                                    total
                                    - (
                                        max(r["response_ns"] for r in requests)
                                        - min(r["enqueue_ns"] for r in requests)
                                    ),
                                )
                                / count
                                if all(finite(v) for v in clocks)
                                else None
                            )
                        usable = bool(
                            usable
                            and finite(acquisition)
                            and finite(startup)
                            and finite(overhead)
                            and finite(unit.get("latency_ns"))
                        )
                        retained = dict(
                            acquisition_ns=acquisition,
                            startup_amortized_ns=startup,
                            queue_wait_ns=unit.get("queue_ns"),
                            host_latency_ns=unit.get("latency_ns"),
                            transaction_overhead_ns=overhead,
                        )
                        current = (
                            acquisition + startup + overhead + unit["latency_ns"]
                            if usable
                            else None
                        )
                        removable = envelope / count if valid and count else None
                        exact = exact / count if finite(exact) and count else None
                        usable = bool(usable and current > removable)
                        checks.append(
                            gate(
                                branch,
                                source,
                                pair["unit_id"] + ".acquisition_ns_known",
                                True,
                                finite(acquisition),
                            )
                        )
                    exact_valid = usable and finite(exact) and exact <= removable
                    if not usable:
                        checks.append(
                            gate(
                                branch,
                                source,
                                pair["unit_id"] + ".component_cost_consistency",
                                True,
                                False,
                            )
                        )
                    row = dict(
                        unit_id=branch + ":" + pair["unit_id"] + f":{index}",
                        source_cluster_id=unit["source_cluster_id"],
                        source_cluster_ids=sorted({r["source_cluster_id"] for r in requests})
                        if branch == "exp8159"
                        else [unit["source_cluster_id"]],
                        arm=arm["arm"],
                        condition=pair["condition"],
                        metric="arithmetic_acceleration_ceiling",
                        numerator=current if usable else None,
                        denominator=current - removable if usable else None,
                        status="completed" if usable else "excluded",
                        exclusion_reason=None
                        if usable
                        else "missing_or_inconsistent_component_cost",
                        upstream=branch,
                        comparison_scope="durable_host_batch"
                        if branch == "exp8159"
                        else "component_composed_request",
                        measured_arithmetic_fraction=exact / current if exact_valid else None,
                        scoring_envelope_fraction=removable / current if usable else None,
                        exact_arithmetic_only_ceiling=current / (current - exact)
                        if exact_valid
                        else None,
                        outer_ceiling=current / (current - removable) if usable else None,
                        retained_components=retained,
                        queue_wait_sum_ns=sum(r.get("queue_ns", 0) for r in requests),
                        centers_touched=centers,
                        points=count if branch == "exp8159" else 1,
                        distance_evaluations=centers * (count if branch == "exp8159" else 1),
                        distance_coordinate_terms=9
                        * centers
                        * (count if branch == "exp8159" else 1),
                        numeric_operand_bytes=8 * (9 * count + 10 * centers + 1),
                        total_bytes_moved=None,
                        durable_state_bytes=arm.get("durable_state_bytes"),
                        commit_count=arm.get("commit_count"),
                        retained_unknown_components=[
                            "transfer",
                            "memory traffic",
                            "readout inside scoring timer",
                        ],
                        current_hardware_execution=False,
                        fixedpoint_arithmetic_executed=False,
                    )
                    rows.append(row)
        if not source["pairs"]:
            rows.append(
                dict(
                    unit_id=branch + ":missing",
                    source_cluster_id=branch,
                    source_cluster_ids=[],
                    arm="unavailable",
                    condition="natural",
                    metric="arithmetic_acceleration_ceiling",
                    numerator=None,
                    denominator=None,
                    status="excluded",
                    exclusion_reason=FIELDS[int(branch[3:])] + "!=1_or_missing_rows",
                    upstream=branch,
                    outer_ceiling=None,
                    exact_arithmetic_only_ceiling=None,
                )
            )
    return rows


def precision(data: Json) -> tuple[list[Json], list[Json]]:
    """Reuse qualified storage bounds on original natural decision margins.

    Each source is evaluated once per precision. Repeated timings cannot create
    independent evidence. The float64 shadow cost is a measured scoring envelope;
    incremental fallback overhead is unknown because upstream did not time it.
    """
    units: Json = {}
    for branch, source in data["branches"].items():
        if source["qualified"]:
            for pair in source["pairs"]:
                for arm in pair["arms"]:
                    for request in arm["requests"]:
                        identity = canonical_hash([request["source_cluster_id"], request["values"]])
                        units[identity] = dict(
                            request=request,
                            envelope=arm["components"].get("arithmetic_ns"),
                            points=len(arm["requests"]),
                            upstream=branch,
                        )
    rows, fallback = [], []
    print(f"[exp8162] precision_before completed=0 pending={len(units)}", flush=True)
    for index, (identity, unit) in enumerate(units.items()):
        request = unit["request"]
        system = dict(seed=CONFIG["seed"], state=data["state"], x=[request["values"]])
        for bits in [*CONFIG["bits"], 64]:
            row = old.natural_precision(system, 16 if bits == 64 else bits)
            if bits == 64:
                row.update(
                    bits=64,
                    arm="float64_fallback",
                    quantized_probabilities=row["reference_probabilities"],
                    issued_probabilities=row["reference_probabilities"],
                    fallback_flags=[False],
                    numerator=0.0,
                    maximum_probability_error=0.0,
                    probability_error_bound=0.0,
                    unguarded_action_disagreements=0,
                    action_disagreements=0,
                )
            row.update(
                unit_id=identity + f":b{bits}",
                source_cluster_id=request["source_cluster_id"],
                source_id="private_fixture" if data["fixture"] else "authenticated_natural_margin",
                natural_margin=min(abs(row["reference_probabilities"][0] - t) for t in [0.1, 0.5]),
                software_bound_only=True,
                arithmetic_class="float64",
                fixedpoint_arithmetic_executed=False,
            )
            rows.append(row)
            fallback.append(
                dict(
                    unit_id=row["unit_id"],
                    source_cluster_id=row["source_cluster_id"],
                    arm=row["arm"],
                    condition=row["condition"],
                    metric="fallback_request_fraction",
                    numerator=int(row["fallback_flags"][0]),
                    denominator=1,
                    status="completed",
                    exclusion_reason=None,
                    bits=bits,
                    requests=1,
                    fallback_requests=int(row["fallback_flags"][0]),
                    float64_scoring_envelope_ns_per_request=unit["envelope"] / unit["points"]
                    if finite(unit["envelope"])
                    else None,
                    measured_incremental_fallback_ns=None,
                    scope="imported float64 readout envelope; incremental fallback unknown",
                )
            )
        if (index + 1) % 16 == 0 or index + 1 == len(units):
            print(
                f"[exp8162] precision completed={index + 1} pending={len(units) - index - 1}",
                flush=True,
            )
    print(f"[exp8162] precision_after completed={len(units)} pending=0", flush=True)
    return rows, fallback


def reduce(data: Json) -> Json:
    """Reduce custody, costs and numerical parity through independent gates.

    A valid software bound does not meet a board's execution obligation. The
    arithmetic ceiling is optimistic when its timer mixes readout and conversion;
    required retained-work reduction is therefore a lower bound on redesign.
    """
    checks = deepcopy(data["checks"])
    for eid, field in FIELDS.items():
        source = data["branches"][f"exp{eid}"]
        checks.append(gate(f"exp{eid}", source, field, 1, source["score"]))
        checks.append(gate(f"exp{eid}", source, field + ".qualified", True, source["qualified"]))
    boards = deepcopy(data["boards"])
    for board in boards:
        observed = [board.get("processor_class"), board.get("k_max"), board.get("blocker")]
        valid = board["custody_valid"] and observed == list(prior.CONTRACTS[board["board"]])
        board.update(
            custody_valid=bool(valid),
            current_hardware_execution=False,
            current_reachability="not_probed",
        )
        checks.append(
            gate(
                board["board"],
                dict(path=board.get("source_path"), hash=board.get("source_hash")),
                "board_custody_" + board["board"],
                True,
                bool(valid),
            )
        )
        checks.append(
            gate(
                board["board"],
                dict(path=board.get("source_path"), hash=board.get("source_hash")),
                "board_terminal_condition_" + board["board"],
                True,
                board.get("terminal_criterion_met", False),
            )
        )
    rows = costs(data, checks)
    quant, fallback = precision(data)
    bounds = []
    for branch, arm in sorted({(r["upstream"], r["arm"]) for r in rows}):
        selected = [r for r in rows if r["upstream"] == branch and r["arm"] == arm]
        valid = all(r["status"] == "completed" for r in selected)
        total = sum(r["numerator"] for r in selected) if valid else None
        kept = sum(r["denominator"] for r in selected) if valid else None
        bounds.append(
            dict(
                upstream=branch,
                arm=arm,
                units=len(selected),
                total_ns=total,
                retained_ns=kept,
                outer_ceiling=total / kept if valid else None,
                exact_arithmetic_only_ceiling=None,
                required_arithmetic_fraction_for_100x=0.99,
                maximum_retained_ns_for_100x=total / 100 if valid else None,
                required_retained_work_reduction_ns=max(0, kept - total / 100) if valid else None,
                required_retained_work_reduction_fraction=max(0, 1 - total / (100 * kept))
                if valid
                else None,
                whole_workload_redesign_required=bool(valid and total / kept < 100),
                scope="software upper bound on removable scoring envelope; exact arithmetic unknown",
            )
        )
    parity = all(r["action_disagreements"] == 0 for r in quant)
    failed = [c for c in checks if not c["passed"]]
    ready = int(parity and any(r["status"] == "completed" for r in rows))
    kind = "blocked" if failed or not ready else "circular_positive" if data["fixture"] else "null"
    allrows = rows + quant + fallback
    completed = sum(r["status"] == "completed" for r in allrows)
    return dict(
        honest_verdict="complete_blocked_" + (failed[0]["check"] if failed else "component_cost")
        if kind == "blocked"
        else "complete_" + kind + "_hardware_workload_boundary",
        verdict_class=kind,
        verifier_is_oracle=data["fixture"],
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        hardware_boundary_ready_score=int(
            len(boards) == 3 and all(b["custody_valid"] for b in boards)
        ),
        measured_workload_bound_ready_score=ready,
        board_rows=boards,
        workload_rows=rows,
        quantization_rows=[] if data["fixture"] else quant,
        fixture_quantization_rows=quant if data["fixture"] else [],
        fallback_cost_rows=fallback,
        amdahl_bounds=bounds,
        rows=allrows,
        gate_check_summary=checks,
        reopen_conditions=previous.REOPEN,
        intended_count=len(allrows),
        eligible_count=completed,
        completed_count=completed,
        excluded_count=len(allrows) - completed,
        censored_count=0,
        failed_count=0,
        independent_count=len(
            {s for r in rows if r["status"] == "completed" for s in r["source_cluster_ids"]}
        )
        if not data["fixture"]
        else 0,
        sample_size_budget=dict(
            unit="original source clusters; batch repetitions are dependent", intended=len(allrows)
        ),
        trained_head_specs=data["trained_head_specs"],
        cited_upstream_artifacts=data["cited"],
        acceptance_gates=dict(
            exact_decision_parity=parity,
            required_arithmetic_fraction_for_100x=0.99,
            host_batch=data["branches"]["exp8159"]["qualified"],
            composition=data["branches"]["exp8160"]["qualified"],
        ),
        hardware_integration_executed=False,
        purchase_executed=False,
        flashing_executed=False,
        cost_accounting_references=[
            dict(
                url="https://extropic.ai/writing/z1t",
                role="retain transfer and final dense readout",
            ),
            dict(
                url="https://arxiv.org/abs/2602.15985v2",
                role="retain decomposition, boundary transport and orchestration",
            ),
        ],
        published_speedups_imported=False,
    )
