"""REQ-VERIFY-8258: imported costs cannot extend the implemented fabric scope.

Each science operand has its own custody gate. Missing current science leaves
the historical board record readable and earns no learning benefit.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import kv260_decision_boundary_8244 as prior
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.request_trace_inventory_8200 import operand
from carnot.verify.evidence_view_kernel_8249 import mix

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8258_v713_kv260_evidence_cost_boundary"
CLI = "scripts/experiments/" + NAME + ".py"
MODEL_SPECS: list[Json] = []
CONFIG: Json = dict(seed=7138258, formats=["Q8.8", "Q16.16"], current_device_calls=0)
CURRENT = dict(
    capture=(8251, "v713_fit_view_capture"),
    heads=(8252, "v713_intervention_energy_fit"),
    seal=(8253, "v713_reserved_view_seal"),
    counters=(8255, "v713_continuous_constraint_admission"),
)
progress = prior.progress
freeze = prior.freeze


def authenticated(path: Path, raw: Path, data: Json, pin: str | None = None) -> Json:
    """Accept terminal custody only when both primary and sidecar bind these bytes."""
    data["checks"].append(operand("exists", path, True, path.is_file()))
    value: Json = json.loads(freeze(path, raw, data).read_bytes())
    validate_primary(value, path)
    for key, expected in [("required_checks_passed", True), ("flagged_adversarial", False)]:
        data["checks"].append(operand(key, path, expected, value.get(key)))
        if value.get(key) != expected:
            raise ValueError(key)
    if pin is not None and sha256_file(path) != pin:
        raise ValueError("pinned_primary_sha256")
    side = path.parent / "raw" / path.stem / "validators" / (sha256_file(path)[7:] + ".json")
    report = read_bound_sidecar(path, side)
    freeze(side, raw, data)
    if report["report"]["passed"] is not True or report["primary_path"] != str(path.absolute()):
        raise ValueError("terminal_binding")
    for ref in value.get("raw_shard_hashes", []):
        freeze(checked(ref), raw, data)
    return value


def load(root: Path, raw: Path) -> Json:
    """Reuse the qualified V712 reader, then resolve each current operand separately."""
    data = prior.load(root, raw)
    data.update(
        current={}, current_heads=[], current_samples=[], counter_states=[], service_spans=[]
    )
    sources = dict(boundary=(8244, "v712_kv260_decision_boundary"), **CURRENT)
    for name, (eid, suffix) in sources.items():
        progress("authenticate_8258_" + name)
        path = root / "results" / f"experiment_{eid}_{suffix}.json"
        try:
            value = authenticated(path, raw, data)
            if name == "boundary":
                if value["kv260_obligation"]["historical"] != data["board"]:
                    raise ValueError("historical_board_drift")
            else:
                ref = value["boundary_primitives_reference"]
                primitive: Json = json.loads(freeze(checked(ref), raw, data).read_bytes())
                if primitive["schema"] != "carnot.v713.kv260-operands.v1":
                    raise ValueError("current_primitive_schema")
                for field in [
                    "current_heads",
                    "current_samples",
                    "counter_states",
                    "requests",
                    "cold_costs",
                ]:
                    if field in primitive:
                        data[field] = primitive[field]
            data["current"][name] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            data["checks"].append(
                operand(
                    "authenticated_terminal_primitives",
                    path,
                    "qualified bytes and operation primitives",
                    str(error),
                )
            )
            data["current"][name] = False
        data["cited"].append(
            dict(
                experiment_id=eid,
                path=str(path),
                sha256=sha256_file(path) if path.is_file() else None,
                imported_fields=[
                    "terminal binding",
                    "boundary_primitives_reference",
                    "kv260_obligation",
                ],
            )
        )
    service = root / "results/experiment_8242_v712_independent_concurrent_service.json"
    if data["branches"]["service"]:
        data["service_spans"] = json.loads(service.read_bytes())["phase_spans"]
    return data


def operations() -> list[Json]:
    """The quadratic spin kernel supplies none of the current evidence operators."""
    return [
        dict(
            operation=name,
            existing_fabric_supported=False,
            execution="host",
            required_numerical_device_evidence=why,
        )
        for name, why in [
            ("source_segmentation", "UTF-8 sentence and offset parity"),
            ("qwen_original", "autoregressive model runtime"),
            ("qwen_evidence_deleted", "autoregressive model runtime"),
            ("qwen_control_deleted", "autoregressive model runtime"),
            ("scalar_features", "typed evidence delta parity"),
            ("gaussian_head", "distance, exponential, coefficients and normalization"),
            ("additive_head", "tanh, coefficients and normalization"),
            ("count_lookup", "admitted keyed counter lookup parity"),
            ("counter_update", "causal release and sparse state update parity"),
            ("persistence", "durable commit and recovery"),
            ("host_transfer", "measured complete transport"),
            ("device_transfer", "authenticated SSH fabric transfer and dispatch"),
        ]
    ]


def costs(data: Json) -> tuple[list[Json], Json]:
    """Charge each server once and keep overlapping request work out of makespan."""
    qualified, _ = prior.costs(data)
    rows = []
    for workload in sorted({r["condition"] for r in qualified}):
        requests = [r for r in qualified if r["condition"] == workload]
        cold = [c for c in data["cold_costs"] if c["workload"] == workload]
        valid = len(cold) == 1 and all(r["clocks_qualified"] for r in requests)
        valid = valid and all(
            type(cold[0].get(k)) in {float, int} and math.isfinite(cold[0][k]) and cold[0][k] >= 0
            for k in ["startup_s", "shutdown_s"]
        )
        total = sum(r["total_s"] for r in requests) if valid else None
        cold_s = cold[0]["startup_s"] + cold[0]["shutdown_s"] if valid else None
        elapsed = (
            (
                max(r["clocks"]["durability"] for r in requests)
                - min(r["clocks"]["issue"] for r in requests)
            )
            / 1e9
            + cold_s
            if valid
            else None
        )
        spans = [s for s in data["service_spans"] if s["phase"] == workload]
        if spans and valid:
            elapsed = (spans[0]["end_ns"] - spans[0]["start_ns"]) / 1e9
            valid = (
                len(spans) == 1
                and elapsed >= cold_s
                and spans[0]["start_ns"] <= min(r["clocks"]["issue"] for r in requests)
                and spans[0]["end_ns"] >= max(r["clocks"]["durability"] for r in requests)
            )
        work = total + cold_s if valid else None
        components = (
            {
                k: sum(r["observed_spans"][k] for r in requests)
                for k in [
                    "acquisition_s",
                    "scoring_s",
                    "service_s",
                    "queue_s",
                    "issue_fsync_s",
                    "terminal_fsync_s",
                ]
            }
            if valid
            else {}
        )
        original = [r for r in data["requests"] if r["workload"] == workload]
        phase_walls: Json = {}
        for phase, start_key, end_key in [
            ("queue", "queue", "start"),
            ("issue_fsync", "issue", "queue"),
            ("terminal_fsync", "end", "durability"),
        ]:
            pairs = [(r["clocks"].get(start_key), r["clocks"].get(end_key)) for r in original]
            phase_walls[phase] = (
                union_seconds(pairs)
                if valid and all(type(a) is int and type(b) is int and a <= b for a, b in pairs)
                else None
            )
        rows.append(
            dict(
                unit_id=workload,
                source_cluster_id=workload,
                arm="cost",
                condition=workload,
                metric="compatible_work_fraction",
                status="completed" if valid else "excluded",
                exclusion_reason=None if valid else "missing_or_inconsistent_clocks",
                numerator=0 if valid else None,
                denominator=elapsed if valid else None,
                request_work_sum_s=total,
                sequential_latency_s=elapsed if valid and workload.endswith("serial") else None,
                observed_parallel_makespan_s=elapsed
                if valid and workload.endswith("concurrent")
                else None,
                request_latency_work_with_cold_s=work,
                observed_workload_elapsed_s=elapsed if valid else None,
                cold_cost_s=cold_s,
                cold_start_count=len(cold),
                compatible_measured_s=0 if valid else None,
                component_work_s=components,
                component_work_shares={k: v / work for k, v in components.items()},
                phase_wall_s=phase_walls,
                phase_wall_shares={
                    k: v / elapsed if v is not None and valid else None
                    for k, v in phase_walls.items()
                },
                acquisition_scope="Latency-weighted request work shares include overlapping queue waits and are not sequential latency or Amdahl fractions. Phase wall shares use interval unions. Service contains scoring/persistence; shares are not additive.",
                request_count=len(requests),
                failed_request_count=sum(
                    r["upstream_request_status"] != "completed" for r in requests
                ),
            )
        )
    valid = bool(rows) and all(r["status"] == "completed" for r in rows)
    return rows, dict(
        status="available" if valid else "unavailable",
        compatible_fraction=0 if valid else None,
        maximum_gain=1.0 if valid else None,
        formula="1/(1-f)",
        compatible_measured_span_count=0,
        scope="implemented historical quadratic fabric; no current compatible kernel; no measured device gain",
    )


def union_seconds(spans: list[tuple[int, int]]) -> float:
    """Count an overlapping clock interval once when describing observed wall cost."""
    elapsed, end = 0, 0
    for start, stop in sorted(spans):
        elapsed += max(0, stop - max(start, end))
        end = max(end, stop)
    return elapsed / 1e9


def quantize(values: list[float], fractional: int) -> tuple[list[float], bool]:
    """Saturate signed storage explicitly so overflow cannot silently wrap a decision."""
    scale = 2**fractional
    integers = [round(v * scale) for v in values]
    limit = 2 ** (2 * fractional - 1)
    return [max(-limit, min(limit - 1, v)) / scale for v in integers], any(
        not -limit <= v < limit for v in integers
    )


def precision(data: Json) -> list[Json]:
    """Keep CPU bases exact while testing fixed coefficient storage and counter math.

    A synthetic threshold and large count reveal rounding and overflow when no
    current natural state is eligible. They supply no natural learning result.
    """
    heads, samples = data["current_heads"], data["current_samples"]
    natural = bool(heads and samples)
    if not natural:
        heads = [
            dict(
                arm="fixture_logistic",
                basis="logistic",
                weights=[math.log(1 / 9), 0.03] + [0.0] * 14 + [40000.0],
                temperature=1.0,
                geometry=dict(mean=[0.0] * 16, scale=[1.0] * 16),
            )
        ]
        samples = [
            dict(
                unit_id="fixture_threshold",
                source_cluster_id="synthetic",
                x=[0.0] * 16,
                p0=0.1,
                baseline_action="accept",
            )
        ]
    prior.validate_heads(heads)
    rows = []
    progress("benchmark_before_fixed_point_8258", 0, len(heads) * len(samples))
    for head in heads:
        for source in samples:
            query = {k: source[k] for k in ["x", "p0", "baseline_action"]}
            reference = prior.cpu.score(head, query)
            for fractional in [8, 16]:
                rounded = deepcopy(head)
                rounded["weights"], overflow = quantize(head["weights"], fractional)
                approximate = prior.cpu.score(rounded, query)
                radius = None
                if source["x"] is not None:
                    phi = prior.cpu.rule.design(
                        head["basis"], np.asarray([source["x"]]), head["geometry"]
                    )[0]
                    radius = (
                        float(sum(abs(phi) * abs(np.asarray(head["weights"]) - rounded["weights"])))
                        / (4 * head["temperature"])
                        + 1e-12
                    )
                rows.append(
                    numerical_row(
                        source["unit_id"],
                        head["arm"],
                        fractional,
                        reference["p"],
                        approximate["p"],
                        overflow,
                        radius,
                        query["baseline_action"],
                        "exposed_natural_state" if natural else "synthetic_numerical_mechanics",
                    )
                )
        progress("fixed_point_8258_" + head["arm"], len(rows), 0)
    states = data["counter_states"] or [
        dict(unit_id="fixture_counter", n=8, bad=1, p0=0.1),
        dict(unit_id="fixture_overflow", n=40000, bad=20000, p0=0.2),
    ]
    for state in states:
        n, bad, p = state["n"], state["bad"], state["p0"]
        if type(n) is not int or type(bad) is not int or not 0 <= bad <= n or not 0 <= p <= 1:
            raise ValueError("counter_schema")
        reference_p = mix(p, dict(n=n, unsupported=bad)) if n >= 8 else p
        for fractional in [8, 16]:
            values, overflow = quantize([n, bad, p], fractional)
            weight, beta = quantize(
                [values[0] / (values[0] + 16), (values[1] + 1) / (values[0] + 2)], fractional
            )[0]
            approximate_p = (
                quantize([(1 - weight) * values[2] + weight * beta], fractional)[0][0]
                if n >= 8
                else values[2]
            )
            rows.append(
                numerical_row(
                    state["unit_id"],
                    "counter_mix",
                    fractional,
                    reference_p,
                    approximate_p,
                    overflow,
                    1.0 if overflow else 4 / 2**fractional,
                    "accept",
                    "exposed_natural_state"
                    if data["counter_states"]
                    else "synthetic_numerical_mechanics",
                )
            )
    progress("benchmark_after_fixed_point_8258", len(rows), 0)
    return rows


def numerical_row(
    unit: str,
    arm: str,
    fractional: int,
    p: float | None,
    approximate: float | None,
    overflow: bool,
    radius: float | None,
    permission: str,
    scope: str,
) -> Json:
    """Report raw changes before an exact CPU fallback protects threshold decisions."""
    action = prior.cpu.rule.action(p, permission)
    raw_action = prior.cpu.rule.action(approximate, permission)
    missing = p is None or approximate is None
    fallback = (
        overflow
        or missing
        or raw_action != action
        or min(abs(approximate - t) for t in [0.1, 1 / 6, 0.5]) <= radius
    )
    return dict(
        unit_id=unit,
        source_cluster_id=unit,
        arm=arm,
        condition="fixed_point",
        format=f"Q{fractional}.{fractional}",
        status="excluded" if missing else "completed",
        exclusion_reason="missing_features" if missing else None,
        metric="raw_action_changed",
        numerator=None if missing else int(raw_action != action),
        denominator=1,
        overflow=overflow,
        probability_reference=p,
        probability_quantized=approximate,
        probability_error=None if missing else abs(p - approximate),
        probability_error_bound=radius,
        raw_action_changed=int(raw_action != action),
        final_action_changed=0 if fallback else int(raw_action != action),
        cpu_fallback=bool(fallback),
        evidence_scope=scope,
        coefficient_storage_only=arm != "counter_mix",
        nonlinear_arithmetic="float64 CPU",
        current_fpga_measurement=False,
    )


def reduce(data: Json, measured: list[Json]) -> Json:
    """Boundary readiness records custody and mechanics, independently of benefit."""
    phases, bound = costs(data)
    missing = [
        dict(
            unit_id=name + "_obligation",
            source_cluster_id=name,
            arm=name,
            condition="upstream_availability",
            metric="available",
            numerator=None,
            denominator=1,
            status="excluded",
            exclusion_reason="unavailable_current_science_operand",
        )
        for name in ["boundary", *CURRENT]
        if not data["current"].get(name)
    ]
    requests, _ = prior.costs(data)
    for request in requests:
        if request["upstream_request_status"] != "completed":
            request["status"] = "failed"
    rows = measured + requests + phases + missing
    safe = all(
        r["final_action_changed"] == 0
        and (
            r["probability_error"] is None
            or r["probability_error"] <= r["probability_error_bound"] + 1e-12
        )
        for r in measured
    )
    ready = int(
        safe
        and data.get("resources_available", True)
        and data["board"].get("custody_valid") is True
        and data["board"].get("k_max") == 5
    )
    blocker = (
        "resources_and_scratch"
        if not data.get("resources_available", True)
        else missing[0]["arm"]
        if missing
        else "request_clocks"
        if bound["status"] == "unavailable"
        else "hardware_custody"
        if not ready
        else None
    )
    kind = "disqualified" if not safe else "blocked" if blocker else "null"
    verdict = (
        "complete_disqualified_numerical_check"
        if not safe
        else "complete_blocked_" + blocker
        if blocker
        else "complete_null_kv260_evidence_cost_boundary"
    )
    return dict(
        honest_verdict=verdict,
        verdict_class=kind,
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=0,
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        independent_count=0,
        verifier_is_oracle=bool(data.get("fixture")),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        kv260_boundary_ready_score=ready,
        operation_rows=operations(),
        phase_cost_rows=phases,
        fixed_point_rows=measured,
        ideal_whole_request_bound=bound,
        kv260_obligation=dict(
            historical=data["board"],
            future_access_command=["ssh", "kria"],
            current_board_execution=False,
            current_reachability="not_probed",
            useful_compatible_workload="Useful quadratic Ising workload with k<=5, bounded coupling encoding and authenticated SSH fabric transcript; include transfers, CPU parity and full request timing.",
        ),
        current_device_execution_count=0,
        nfr01_met=False,
        nfr01_status="unmet: no matched measured Rust/Python 10x evidence",
        source_cost_scope=dict(
            v713="unavailable unless current primitive gate passes",
            v712="authenticated historical request clocks; not three-view measurements",
            fixture="CPU numerical mechanics only",
        ),
        trained_head_specs=[
            dict(
                arm=head["arm"],
                head_sha256=canonical_hash(head),
                current_fit=False,
                evidence_scope="exposed_development",
            )
            for head in data["current_heads"]
        ],
        gate_check_summary=data["checks"],
        branch_readiness=dict(
            data["branches"], **{"current_" + k: v for k, v in data["current"].items()}
        ),
        acceptance_gates=dict(
            numerical_fallback_preserves_actions=safe,
            independent_learning_benefit=False,
            device_speedup=False,
        ),
    )
