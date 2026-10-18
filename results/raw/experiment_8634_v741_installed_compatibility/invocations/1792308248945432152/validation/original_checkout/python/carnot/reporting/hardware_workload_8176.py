"""REQ-VERIFY-8176: bound each workload without equating mixed timers to arithmetic."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from carnot.reporting import hardware_workload_8162 as old

Json = dict[str, Any]
FIELDS = {
    8159: "host_batch_ready_score",
    8173: "composition_replay_ready_score",
    8174: "complete_service_ready_score",
}
CONFIG: Json = dict(old.CONFIG, seed=7068176, pure_arithmetic_required_for_exact_fraction=True)


def reduce(data: Json) -> Json:
    """Keep independent evidence usable when a different branch is unavailable.

    The inherited reducers already retain host work and guard natural margins.
    Here each upstream has its own denominator. A complete request includes the
    measured queue after acquisition, so overlapping timers are never added.
    Board terminal obligations do not become software acceleration results.
    """
    checks = deepcopy(data["checks"])
    rows: list[Json] = []
    for eid, field in FIELDS.items():
        branch = f"exp{eid}"
        source = data["branches"][branch]
        checks.append(old.gate(branch, source, field, 1, source["score"]))
        alias = "exp8160" if eid == 8173 else "exp8159"
        begin = len(checks)
        branch_rows = old.costs(dict(data, branches={alias: source}), checks)
        for check in checks[begin:]:
            check["upstream"] = branch
        for row in branch_rows:
            row.update(
                upstream=branch,
                unit_id=row["unit_id"].replace(alias, branch),
                comparison_scope={
                    8159: "durable_host_batch",
                    8173: "component_composed_request",
                    8174: "independent_complete_request",
                }[eid],
            )
            if eid == 8174:
                row["startup_amortized_ns"] = data["startup_costs"].get("cold_load_ns", 0) / data[
                    "startup_costs"
                ].get("amortization_requests", 48)
                row["startup_total_s"] = data["startup_costs"].get("startup_total_s")
                row["cold_inclusive_outer_ceiling"] = (
                    (
                        (row["numerator"] + row["startup_amortized_ns"])
                        / (row["denominator"] + row["startup_amortized_ns"])
                    )
                    if row["status"] == "completed"
                    else None
                )
            rows.append(row)
    boards = deepcopy(data["boards"])
    for board in boards:
        valid = board["custody_valid"] and [
            board.get("processor_class"),
            board.get("k_max"),
            board.get("blocker"),
        ] == list(old.prior.CONTRACTS[board["board"]])
        board.update(
            custody_valid=bool(valid),
            current_hardware_execution=False,
            current_reachability="not_probed",
        )
        source = dict(path=board.get("source_path"), hash=board.get("source_hash"))
        checks.append(
            old.gate(board["board"], source, "board_custody_" + board["board"], True, bool(valid))
        )
        checks.append(
            old.gate(
                board["board"],
                source,
                "board_terminal_condition_" + board["board"],
                True,
                board.get("terminal_criterion_met", False),
            )
        )
    quant, fallback = old.precision(data)
    centers = len(data["state"].get("centers", []))
    for row in quant:
        row["packed_head_storage_bytes"] = math.ceil((10 * centers + 1) * row["bits"] / 8)
        row["float64_head_storage_bytes"] = 8 * (10 * centers + 1)
        row["distance_coordinate_terms"] = 9 * centers
        row["gaussian_exponentials"] = centers
    bounds = []
    for branch, arm in sorted({(r["upstream"], r["arm"]) for r in rows}):
        chosen = [r for r in rows if r["upstream"] == branch and r["arm"] == arm]
        valid = all(r["status"] == "completed" for r in chosen)
        total = sum(r["numerator"] for r in chosen) if valid else None
        kept = sum(r["denominator"] for r in chosen) if valid else None
        exact = valid and all(r["measured_arithmetic_fraction"] is not None for r in chosen)
        fraction = (
            sum(r["numerator"] * r["measured_arithmetic_fraction"] for r in chosen) / total
            if exact
            else None
        )
        bounds.append(
            dict(
                upstream=branch,
                arm=arm,
                units=len(chosen),
                total_ns=total,
                retained_ns=kept,
                outer_ceiling=total / kept if valid else None,
                measured_arithmetic_fraction=fraction,
                exact_arithmetic_only_ceiling=1 / (1 - fraction) if exact else None,
                required_arithmetic_fraction_for_100x=0.99,
                required_retained_work_reduction_fraction=max(0, 1 - total / (100 * kept))
                if valid
                else None,
                whole_workload_redesign_required=bool(valid and total / kept < 100),
                scope="optimistic mixed scoring envelope; exact arithmetic only when independently measured",
            )
        )
    allrows = rows + quant + fallback
    completed = sum(r["status"] == "completed" for r in allrows)
    failed = [c for c in checks if not c["passed"]]
    ready = int(
        all(r["action_disagreements"] == 0 for r in quant)
        and any(r["status"] == "completed" for r in rows)
    )
    kind = "blocked" if failed or not ready else "circular_positive" if data["fixture"] else "null"
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
        reopen_conditions=dict(
            old.previous.REOPEN,
            NPU="Authenticated access, supported radial/exp operators, guarded decision parity and transfer-inclusive complete timing.",
            TSU="Authenticated hardware and SDK access, compatible supported operators and measured acquisition, transfer, readout, fallback and persistence.",
        ),
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
            unit="original source clusters; repeated batches are dependent", intended=len(allrows)
        ),
        trained_head_specs=data["trained_head_specs"],
        cited_upstream_artifacts=data["cited"],
        acceptance_gates=dict(
            required_arithmetic_fraction_for_100x=0.99, pure_arithmetic_required=True
        ),
        hardware_integration_executed=False,
        purchase_executed=False,
        flashing_executed=False,
        published_speedups_imported=False,
        measured_bottleneck="Retained acquisition and queue dominate complete requests; durable storage dominates host batches.",
        falsifiable_whole_workload_redesign="Reduce acquisition and queue via independently equivalent cached judgments and a bounded admission window; group durable acknowledgements without weakening recovery. Remeasure complete source-matched latency including transfer/readout and require retained work <=1% of the original for a100x claim.",
        external_context="Extropic October research-agent execution/rubric rewards and Z1T chip projections provide context only; no local TSU execution or speedup.",
    )
