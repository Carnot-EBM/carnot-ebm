"""REQ-REPORT-8735: costs from different parents cannot share an accelerator fraction."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from carnot.reporting.hardware_operation_boundary_8707 import bound

Json = dict[str, Any]
BOARDS = ("KV260", "PolarFire", "GateMate")
BOUNDARIES = ("causal_learning", "constructed_resident_update", "installed_inference_only")


def cost_rows(branches: list[Json], support: list[Json]) -> Json:
    """Reduce only qualified complete spans; unsupported work remains visible host cost."""
    costs, missing, mapped = [], [], list(support)
    groups: dict[tuple[str, str, str], list[Json]] = defaultdict(list)
    for branch, boundary in zip(branches, BOUNDARIES, strict=True):
        if not branch["qualified"]:
            missing.append(
                dict(
                    producer=branch["task_id"],
                    boundary=boundary,
                    duration_s=None,
                    checks=branch["checks"],
                )
            )
            continue
        value = branch["value"]
        spans = value.get("component_span_rows", value.get("complete_cost_rows", []))
        for index, row in enumerate(spans):
            total = row.get("parent_duration_s")
            cost = dict(
                row,
                cost_id=f"{branch['task_id']}:{index}",
                producer=branch["task_id"],
                boundary=boundary,
                clock_domain=row.get("parent", "complete_boundary"),
                parent_duration_s=total,
                primitive_reference=value.get("primitive_reference"),
            )
            costs.append(cost)
            groups[
                branch["task_id"], str(row.get("process_id", index)), cost["clock_domain"]
            ].append(cost)
            for board in BOARDS:
                capability = next(
                    (
                        s
                        for s in support
                        if s["board"] == board and s["operation"] == row["operation"]
                    ),
                    {},
                )
                mapped.append(
                    dict(
                        board=board,
                        operation=row["operation"],
                        cost_id=cost["cost_id"],
                        producer=branch["task_id"],
                        boundary=boundary,
                        clock_domain=cost["clock_domain"],
                        support=capability.get(
                            "support", "missing-evidence" if board == "GateMate" else "unsupported"
                        ),
                        physical_execution_this_invocation=False,
                        reason="GateMate0xffffffff: dated physical change, IDCODE0x20000001 then ordered n16 smoke required"
                        if board == "GateMate"
                        else "Linux CPU dispatch is not FPGA fabric"
                        if board == "PolarFire"
                        else "Only authenticated quadratic Ising k_max<=5 is supported; no spline kernel",
                        measured_duration_s=row.get("duration_s"),
                        parent_duration_s=total,
                    )
                )
    ceilings = []
    for (producer, process, domain), rows in groups.items():
        totals = {r["parent_duration_s"] for r in rows}
        total = next(iter(totals)) if len(totals) == 1 else None
        for board in BOARDS:
            compatible = [
                r
                for r in rows
                if any(
                    s["board"] == board
                    and s["operation"] == r["operation"]
                    and s["support"] == "supported"
                    for s in support
                )
            ]
            timing = total is not None and all(r.get("duration_s") is not None for r in rows)
            if timing and (
                any(r["duration_s"] < 0 for r in rows) or sum(r["duration_s"] for r in rows) > total
            ):
                raise ValueError("non_disjoint_parent_costs")
            compatible_s = (
                sum(r["duration_s"] for r in compatible) if timing and compatible else None
            )
            ceiling = bound(total, compatible_s, bool(compatible))
            if ceiling["compatible_fraction"] is not None:
                ceiling["missing_reason"] = None
            ceilings.append(
                dict(
                    board=board,
                    producer=producer,
                    process_id=process,
                    clock_domain=domain,
                    boundary=rows[0]["boundary"],
                    parent_duration_s=total,
                    compatible_duration_s=compatible_s,
                    includes_measured_host_and_transfer=True,
                    minimum_additional_compatible_fraction_10x=max(
                        0, 0.9 - ceiling["compatible_fraction"]
                    )
                    if ceiling["compatible_fraction"] is not None
                    else None,
                    minimum_additional_compatible_fraction_100x=max(
                        0, 0.99 - ceiling["compatible_fraction"]
                    )
                    if ceiling["compatible_fraction"] is not None
                    else None,
                    **ceiling,
                )
            )
    return dict(
        complete_cost_rows=costs,
        missing_cost_rows=missing,
        operation_support_rows=mapped,
        conditional_acceleration_bounds=ceilings
        or [dict(board=b, **bound(None, None, False)) for b in BOARDS],
        limiting_host_costs=costs,
        unknown_costs=[r["boundary"] for r in missing]
        + [
            "physical host/device transfer",
            "FPGA service clocks",
            "compatible spline fabric implementation",
        ],
        cost_scope_note="Keep child_serve and parent_process clocks separate. Inference spans do not supply constructed-update or real causal-learning denominators. Missing costs and compatibility remain unknown.",
    )
