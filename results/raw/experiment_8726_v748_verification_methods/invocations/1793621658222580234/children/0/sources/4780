"""REQ-REPORT-8679: retain original acceptance failures and measured operation scopes."""

from __future__ import annotations

import json
from typing import Any

from carnot.reporting import hardware_operation_evidence_8665 as old

Json = dict[str, Any]


def acceptance(value: Any) -> Json:
    """Only producer forms seen in original evidence are safe to interpret as gates.

    NFR-01 tests benefit rather than measurement completeness. Its failure stays
    visible without throwing away the complete contemporaneous cost measurement.
    """
    if (
        isinstance(value, dict)
        and value
        and all(isinstance(k, str) and bool(k) and type(v) is bool for k, v in value.items())
    ):
        shape, gates = "boolean_map", dict(value)
    elif isinstance(value, list) and value:
        shape, gates = "named_list", {}
        for item in value:
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "passed"}
                or not isinstance(item["name"], str)
                or not item["name"]
                or type(item["passed"]) is not bool
                or item["name"] in gates
            ):
                raise ValueError("unknown_or_ambiguous_acceptance_shape")
            gates[item["name"]] = item["passed"]
    else:
        raise ValueError("unknown_or_ambiguous_acceptance_shape")
    failures = [k for k, v in gates.items() if not v]
    return dict(
        shape=shape,
        gates=gates,
        failures=failures,
        qualified=not any(k != "NFR-01" for k in failures)
        and (shape == "boolean_map" or gates.get("qualified_complete_measurement") is True),
    )


def flags(value: Any) -> list[Json]:
    """The original source producer wraps reports; installed costs use a plain list."""
    if isinstance(value, dict):
        value = [f for r in value["reports"] for f in r["flags"]]
    if not isinstance(value, list) or any(
        not isinstance(f, dict) or f.get("severity") not in {"info", "warning", "error", "critical"}
        for f in value
    ):
        raise ValueError("unknown_findings_shape")
    return value


def conditional(cell: Json, implementation: bool = False) -> Json:
    """Removing measured arithmetic gives only an Amdahl ceiling, never achieved speed.

    Existing boards have no compatible spline implementation. Even isolated CPU
    endpoints cannot establish device support or erase transfer/update/store work.
    """
    total = cell["total_s"]
    compatible = cell.get("compatible_s")
    reason = (
        "missing_compatible_operation_implementation"
        if not implementation
        else "missing_isolated_inner_operation_endpoints"
        if compatible is None
        else None
    )
    if total <= 0 or compatible is not None and not 0 <= compatible <= total:
        raise ValueError("invalid_compatible_fraction")
    fraction = compatible / total if compatible is not None and not reason else None
    return dict(
        cell,
        compatible_s=compatible if not reason else None,
        compatible_fraction=fraction,
        unaccelerated_share=1 - fraction if fraction is not None else None,
        conditional_ceiling=1 / (1 - fraction) if fraction is not None and fraction < 1 else None,
        conditional_limit=reason or "conditional_arithmetic_removal_only",
        missing_reason=reason,
        achieved_speedup=None,
        required_costs=[
            "host_device_transfer",
            "sparse_updates",
            "durable_store",
            "feedback_drain",
        ],
        uncertainty="Original repeated clocks are descriptive; no device-clock or compatible-kernel uncertainty measured.",
    )


def source_cells(value: Json, refs: list[Json]) -> list[Json]:
    """Reduce sealed extraction clocks without rerunning the historical extractor.

    Forked worker imports happened before the measured interval. The three
    disjoint segments preserve that boundary; the outer residual includes
    transport and teardown and cannot identify any spline arithmetic.
    """
    authority = value["primitive_reference"]
    work = json.loads(old.checked(authority).read_bytes())
    refs.append(authority)
    for ref in work["code_refs"]:
        old.checked(ref)
        refs.append(ref)
    costs = value["complete_extraction_cost_rows"]
    if costs != work["report"]["complete_extraction_cost_rows"] or len(costs) != 1140:
        raise ValueError("original_source_cost_roster")
    by_key = {
        (r["opaque_id"], i["arm"], i["repeat"]): i["reference"]
        for r in work["response_records"]
        for i in r["references"]
        if i["chunk"] == 32
    }
    cells = []
    for index, row in enumerate(costs):
        ref = by_key[(row["opaque_id"], row["arm"], row["repeat"])]
        original = json.loads(old.checked(ref).read_bytes())["cost"]
        refs.append(ref)
        if any(row[k] != v for k, v in original.items()):
            raise ValueError("original_source_cost_bytes")
        receipt = row["receipt"]
        points = [
            receipt["started_monotonic_ns"],
            row["started_monotonic_ns"],
            row["ended_monotonic_ns"],
            receipt["ended_monotonic_ns"],
        ]
        if (
            any(type(p) is not int for p in points)
            or points != sorted(points)
            or points[0] == points[-1]
            or receipt["passed"] is not True
            or receipt["exit_code"] != 0
            or receipt["timed_out"] is not False
        ):
            raise ValueError("source_disjoint_endpoints_or_child_exit")
        operations = ["postfork_startup", "full_text_features", "transport_teardown_residual"]
        seconds = {op: (b - a) / 1e9 for op, a, b in zip(operations, points, points[1:])}
        cells.append(
            conditional(
                dict(
                    boundary="source_extraction",
                    cohort=f"{row['opaque_id']}-R{row['repeat']}",
                    arm=row["arm"],
                    total_s=(points[-1] - points[0]) / 1e9,
                    seconds=seconds,
                    compatible_s=None,
                    primitive_reference=ref,
                    measured_scope="fork_after_imports; startup imports excluded; residual not spline time",
                    spans=[
                        dict(operation=op, start_ns=a, end_ns=b)
                        for op, a, b in zip(operations, points, points[1:])
                    ],
                )
            )
        )
        if (index + 1) % 190 == 0:
            old.progress("reduce_original_source_costs", index + 1, len(costs) - index - 1)
    return cells


def reduce_spans(value: Json, boundary: str) -> Json:
    """Preserve opaque binding spans rather than requiring invented inner spline clocks."""
    spans = value["spans"]
    if (
        value["boundary"] != boundary
        or value["exit_code"] != 0
        or not spans
        or spans[0]["start_ns"] != value["start_ns"]
        or spans[-1]["end_ns"] != value["end_ns"]
    ):
        raise ValueError("complete_service_endpoints")
    seconds: dict[str, float] = {}
    cursor = value["start_ns"]
    for span in spans:
        start, end, operation = span["start_ns"], span["end_ns"], span["operation"]
        if (
            type(start) is not int
            or type(end) is not int
            or start != cursor
            or end < start
            or not isinstance(operation, str)
            or not operation
        ):
            raise ValueError("disjoint_complete_operation_spans")
        seconds[operation] = seconds.get(operation, 0) + (end - start) / 1e9
        cursor = end
    total = (value["end_ns"] - value["start_ns"]) / 1e9
    if total <= 0:
        raise ValueError("nonpositive_complete_span")
    return dict(
        boundary=boundary,
        cohort=value["cohort"],
        arm=value["arm"],
        total_s=total,
        seconds=seconds,
        compatible_s=seconds.get("deterministic_spline_109"),
        measured_scope="declared complete operation interval",
        spans=spans,
    )


def reproduce_shapes(raw: Any) -> list[Json]:
    """Keep a real TypeError receipt so the historical mismatch is evidence, not narration."""
    from carnot.reporting.current_work_receipt import atomic_json

    fixtures = [
        dict(
            producer="Exp8659 boolean-map form",
            acceptance_gates=dict(owned_checks=True, paired_costs=True),
        ),
        dict(
            producer="Exp8661 named-list form",
            acceptance_gates=[
                dict(name="qualified_complete_measurement", passed=True),
                dict(name="NFR-01", passed=False),
            ],
        ),
    ]
    rows = []
    for index, fixture in enumerate(fixtures):
        error = None
        try:
            any(g["name"] == "qualified_complete_measurement" for g in fixture["acceptance_gates"])
        except TypeError as failure:
            error = str(failure)
        path = raw / f"private-acceptance-shape-{index}.json"
        atomic_json(
            path,
            dict(
                fixture=fixture,
                original_list_assumption_typeerror=error,
                adapted=acceptance(fixture["acceptance_gates"]),
                scope="constructed regression only",
            ),
        )
        path.chmod(0o444)
        rows.append(
            dict(
                primitive_reference=old.reference(path),
                historical_typeerror=error,
                **acceptance(fixture["acceptance_gates"]),
            )
        )
    return rows
