"""REQ-VERIFY-8148: retain unknown host costs when bounding arithmetic benefit."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from carnot.reporting import hardware_service_8134 as previous
from carnot.reporting import radial_hardware_8108 as prior
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
CONFIG: Json = dict(seed=7048148, bits=[8, 12, 16], dimensions=9, pending_capacity=32)
FIELDS = dict(exp8145="natural_service_ready_score", exp8146="complete_service_ready_score")


def natural_precision(system: Json, bits: int) -> Json:
    """Keep the frozen generator offset while quantizing only small-head storage.

    The existing error envelope uses the maximum sigmoid slope, so adding the
    same fixed offset to both logits preserves its conservative bound.
    """
    row = prior.numerical(system, bits)
    np = prior.np
    state = system["state"]
    x = prior.kernel.matrix(system["x"])
    g = state["geometry"]
    z = (x - np.asarray(g["mean"])) / np.asarray(g["std"])
    centers = np.asarray([c["x"] for c in state["centers"]])
    theta = np.asarray(state["coefficients"])
    qc = np.rint((np.clip(centers, -4, 4) + 4) / row["center_step"]) * row["center_step"] - 4
    qt = (
        np.rint((np.clip(theta, -8, 8) + 8) / row["coefficient_step"]) * row["coefficient_step"] - 8
    )
    phi = np.exp(-np.sum((z[:, None, :] - qc[None, :, :]) ** 2, axis=2) / (2 * g["sigma"] ** 2))
    reference = prior.expit(x[:, 0] + prior.kernel.design(state, x) @ theta)
    quantized = prior.expit(x[:, 0] + qt[0] + phi @ qt[1:])
    flags = (
        (
            np.minimum(abs(quantized - 0.1), abs(quantized - 0.5))
            <= row["probability_error_bound"] + 1e-12
        )
        | np.any(abs(z) > 4, axis=1)
        | bool(row["clipping_count"])
    )
    issued = np.where(flags, reference, quantized)
    actions = [prior.kernel.action(float(p)) for p in issued]
    canonical = [prior.kernel.action(float(p)) for p in reference]
    error = float(np.max(abs(reference - quantized)))
    row.update(
        reference_probabilities=reference.tolist(),
        quantized_probabilities=quantized.tolist(),
        issued_probabilities=issued.tolist(),
        fallback_flags=flags.tolist(),
        reference_actions=canonical,
        issued_actions=actions,
        numerator=error,
        maximum_probability_error=error,
        unguarded_action_disagreements=sum(
            prior.kernel.action(float(p)) != a for p, a in zip(quantized, canonical, strict=True)
        ),
        action_disagreements=sum(a != b for a, b in zip(actions, canonical, strict=True)),
        frozen_offset_policy="retain unquantized historical input first-coordinate base logit",
    )
    return row


def cost_rows(data: Json) -> list[Json]:
    """Keep all non-arithmetic work; mixed scoring timers give outer bounds only.

    Packed storage counts describe numeric operands, not measured bus traffic.
    Candidate fitting and intermediate update scans need separate instrumentation.
    """
    rows = []
    for branch, source in data["branches"].items():
        for pair in source["pairs"]:
            for arm in pair["arms"]:
                total = arm.get("full_latency_ns")
                components = arm.get("components", {})
                envelope = components.get("arithmetic_and_boundary_ns")
                residual = arm.get("residual_ns")
                numbers = [total, envelope, residual, *components.values()]
                valid = bool(
                    source["qualified"]
                    and pair["status"] == "completed"
                    and components
                    and all(
                        type(v) in {int, float} and math.isfinite(v) and v >= 0 for v in numbers
                    )
                )
                valid = bool(
                    valid and 0 <= envelope < total and total == sum(components.values()) + residual
                )
                arithmetic = arm.get("arithmetic_only_ns")
                exact = bool(
                    valid
                    and type(arithmetic) in {int, float}
                    and math.isfinite(arithmetic)
                    and 0 <= arithmetic <= envelope
                )
                removable = arithmetic if exact else envelope
                state = arm.get("durable_state", {})
                centers = len(state.get("centers", []))
                points = len(arm.get("values", []))
                clusters = arm.get("source_cluster_ids", [])
                rows.append(
                    dict(
                        unit_id=branch + ":" + pair["unit_id"],
                        source_cluster_id=canonical_hash(clusters),
                        source_cluster_ids=clusters,
                        arm=arm["arm"],
                        condition=pair["condition"],
                        metric="arithmetic_acceleration_ceiling",
                        numerator=total if valid else None,
                        denominator=total - removable if valid else None,
                        status="completed" if valid else "excluded",
                        exclusion_reason=None
                        if valid
                        else "missing_or_inconsistent_component_cost",
                        upstream=branch,
                        exact_arithmetic_only_ceiling=total / (total - arithmetic)
                        if exact
                        else None,
                        measured_arithmetic_fraction=arithmetic / total if exact else None,
                        scoring_envelope_fraction=envelope / total if valid else None,
                        outer_ceiling=total / (total - removable) if valid else None,
                        retained_components={
                            k: v for k, v in components.items() if k != "arithmetic_and_boundary_ns"
                        },
                        retained_residual_ns=residual,
                        unknown_in_envelope="memory traffic, conversion and call boundaries",
                        acquisition_scope="cached historical host inputs"
                        if branch == "exp8145"
                        else "complete natural service",
                        current_hardware_execution=False,
                        centers_touched=centers,
                        points=points,
                        distance_evaluations=centers * points,
                        distance_coordinate_terms=centers * points * 9,
                        counted_operation_scope="final decision evaluation only; intermediate candidate fit and scans unknown",
                        numeric_operand_bytes=8 * (9 * points + 9 * centers + centers + 1),
                        total_bytes_moved=None,
                        pending_feedback_capacity=32,
                        pending_feedback_numeric_capacity_bytes=32 * (9 * 8 + 8 + 8),
                        pending_feedback_actual_bytes=None,
                        pending_storage_scope="capacity lower bound; keys, queue occupancy and encoding unknown",
                    )
                )
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
                    exclusion_reason=FIELDS[branch] + "!=1_or_missing_rows",
                    upstream=branch,
                    outer_ceiling=None,
                    exact_arithmetic_only_ceiling=None,
                )
            )
    return rows


def reduce(data: Json) -> Json:
    """Board custody and each workload branch qualify without borrowing readiness."""
    checks = deepcopy(data["checks"])
    boards = deepcopy(data["boards"])
    for board in boards:
        observed = [board.get("processor_class"), board.get("k_max"), board.get("blocker")]
        expected = list(prior.CONTRACTS[board["board"]])
        valid = bool(board.get("custody_valid") and observed == expected)
        board.update(
            custody_valid=valid,
            current_hardware_execution=False,
            current_reachability="not_probed",
            status="completed" if valid else "excluded",
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
    for branch, field in FIELDS.items():
        source = data["branches"][branch]
        checks.append(
            dict(
                check=field,
                upstream=branch,
                path=source.get("path"),
                hash=source.get("hash"),
                artifact_field=field,
                op="==",
                expected=1,
                observed=source["score"],
                passed=source["score"] == 1,
            )
        )
    rows = cost_rows(data)
    for branch, source in data["branches"].items():
        for pair in source["pairs"]:
            for arm in pair["arms"]:
                operands = dict(
                    full_latency_ns=arm.get("full_latency_ns"),
                    residual_ns=arm.get("residual_ns"),
                    **{"components." + k: v for k, v in arm.get("components", {}).items()},
                )
                operands["components.arithmetic_and_boundary_ns"] = arm.get("components", {}).get(
                    "arithmetic_and_boundary_ns"
                )
                for field, observed in operands.items():
                    if not (
                        type(observed) in {int, float} and math.isfinite(observed) and observed >= 0
                    ):
                        checks.append(
                            dict(
                                check="component_cost",
                                upstream=branch,
                                path=source.get("path"),
                                hash=source.get("hash"),
                                artifact_field="pairs["
                                + pair["unit_id"]
                                + "]."
                                + arm["arm"]
                                + "."
                                + field,
                                op="finite_nonnegative",
                                expected=True,
                                observed=observed,
                                passed=False,
                            )
                        )
    systems: Json = {}
    for source in data["branches"].values():
        if source["qualified"]:
            for pair in source["pairs"]:
                for arm in pair["arms"]:
                    if arm.get("values") and arm.get("durable_state"):
                        system = dict(
                            seed=CONFIG["seed"], state=arm["durable_state"], x=arm["values"]
                        )
                        systems[canonical_hash(system)] = system
    quant, fixtures = [], []
    print(f"[exp8148] before_precision_benchmark completed=0 pending={len(systems)}", flush=True)
    for index, (identity, system) in enumerate(systems.items()):
        for bits in CONFIG["bits"]:
            row = natural_precision(system, bits)
            row.update(
                unit_id=identity + f":b{bits}",
                source_cluster_id=identity,
                source_id="authenticated_natural_decision",
                exposure_scope="exposed natural inputs",
            )
            quant.append(row)
        if (index + 1) % 16 == 0 or index + 1 == len(systems):
            print(
                f"[exp8148] precision completed={index + 1} pending={len(systems) - index - 1}",
                flush=True,
            )
    if not systems:
        fixtures = [prior.numerical(s, b) for s in data["systems"] for b in CONFIG["bits"]]
    print(f"[exp8148] after_precision_benchmark completed={len(systems)} pending=0", flush=True)
    bounds = []
    for branch in FIELDS:
        for arm in sorted({r["arm"] for r in rows if r["upstream"] == branch}):
            selected = [r for r in rows if r["upstream"] == branch and r["arm"] == arm]
            usable = all(r["status"] == "completed" for r in selected)
            total = sum(r["numerator"] for r in selected) if usable else None
            kept = sum(r["denominator"] for r in selected) if usable else None
            ceiling = total / kept if usable else None
            bounds.append(
                dict(
                    upstream=branch,
                    arm=arm,
                    total_ns=total,
                    retained_ns=kept,
                    outer_ceiling=ceiling,
                    exact_arithmetic_only_ceiling=None,
                    units=len(selected),
                    required_arithmetic_fraction_for_100x=0.99,
                    target_100x="whole_workload_redesign_before_board_port"
                    if usable and ceiling < 100
                    else "unknown",
                    scope="optimistic mixed scoring envelope ceiling; arithmetic-only fraction unknown",
                )
            )
    precision = all(r["action_disagreements"] == 0 for r in quant + fixtures)
    ready = int(precision and any(r["status"] == "completed" for r in rows))
    failed = [c for c in checks if not c["passed"]]
    blocked = bool(failed or not ready)
    verdict = "blocked" if blocked else "circular_positive" if data["fixture"] else "null"
    allrows = rows + quant + fixtures
    completed = sum(r["status"] == "completed" for r in allrows)
    return dict(
        honest_verdict="complete_blocked_" + (failed[0]["check"] if failed else "component_cost")
        if blocked
        else "complete_" + verdict + "_hardware_workload_boundary",
        verdict_class=verdict,
        verifier_is_oracle=bool(data["fixture"]),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        hardware_boundary_ready_score=int(
            len(boards) == 3 and all(b["custody_valid"] for b in boards)
        ),
        measured_workload_bound_ready_score=ready,
        board_rows=boards,
        natural_workload_rows=rows,
        quantization_rows=quant,
        fixture_quantization_rows=fixtures,
        rows=allrows,
        fallback_cost_rows=[
            dict(
                unit_id=r["unit_id"],
                bits=r["bits"],
                requests=r["points"],
                fallback_requests=sum(r["fallback_flags"]),
                measured_incremental_fallback_ns=None,
                incremental_fallback_cost_status="unknown; retain authoritative float64 CPU cost",
            )
            for r in quant + fixtures
        ],
        amdahl_bounds=bounds,
        gate_check_summary=checks,
        reopen_conditions=previous.REOPEN,
        intended_count=len(allrows),
        eligible_count=completed,
        completed_count=completed,
        excluded_count=len(allrows) - completed,
        failed_count=0,
        censored_count=0,
        independent_count=len(
            {s for r in rows if r["status"] == "completed" for s in r["source_cluster_ids"]}
        )
        if not data["fixture"]
        else 0,
        sample_size_budget=dict(
            unit="original natural source clusters; repetitions are dependent",
            intended=len(allrows),
        ),
        acceptance_gates=dict(
            exact_decision_parity=precision,
            required_arithmetic_fraction_for_100x=0.99,
            natural_host=data["branches"]["exp8145"]["qualified"],
            complete_service=data["branches"]["exp8146"]["qualified"],
        ),
        trained_head_specs=data.get(
            "trained_head_specs",
            [
                dict(
                    kind="historical Gaussian residual head",
                    centers_range=[16, 28],
                    dimensions=9,
                    trained_currently=False,
                )
            ],
        ),
        cited_upstream_artifacts=data["cited"],
        purchase_executed=False,
        flashing_executed=False,
        hardware_integration_executed=False,
        cost_accounting_motivation="Historical Z1T estimates omit final dense compute, readout and transfer; higher-order FPGA decomposition needs projection, transport and host work. These are cost cautions; no published speed factor is imported.",
    )
