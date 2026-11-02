"""REQ-VERIFY-8301: CPU locality costs do not qualify quadratic fabric work.

Each imported span belongs to one authenticated CPU invocation. Missing live
requests remain separate obligations, even when CPU fixture timing is readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import dependency_admission_execution_8291 as cpu
from carnot.reporting import kv260_evidence_cost_boundary_8287 as legacy
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.request_trace_inventory_8200 import operand

Json = dict[str, Any]
ROOT = legacy.ROOT
NAME = "experiment_8301_v716_kv260_evidence_cost_boundary"
CLI = "scripts/experiments/" + NAME + ".py"
MODEL_SPECS: list[Json] = []
CONFIG: Json = dict(seed=7168301, formats=["Q8.8", "Q16.16"], current_device_calls=0)
PRIMITIVE_SCHEMA = "carnot.v716.kv260-operands.v1"
CURRENT = dict(
    capture=(8293, "v716_fit_view_capture"),
    tune=(8294, "v716_tune_view_capture"),
    heads=(8295, "v716_intervention_energy_fit"),
    seal=(8296, "v716_reserved_view_seal"),
    counters=(8298, "v716_continuous_constraint_admission"),
)
SOURCES = dict(
    boundary=(8244, "v712_kv260_decision_boundary"),
    **CURRENT,
    historical_boundary=(8287, "v715_kv260_evidence_cost_boundary"),
)
freeze = legacy.freeze
authenticated = legacy.authenticated
precision = legacy.precision


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts at phase boundaries for the bounded process supervisor."""
    print(f"[exp8301] phase={phase} completed={completed} pending={pending}", flush=True)


def load(root: Path, raw: Path) -> Json:
    """Resolve CPU evidence independently from the unavailable live branch."""
    with (
        patch.object(legacy, "CURRENT", CURRENT),
        patch.object(legacy, "SOURCES", SOURCES),
        patch.object(legacy, "PRIMITIVE_SCHEMA", PRIMITIVE_SCHEMA),
        patch.object(legacy, "progress", progress),
    ):
        data = legacy.load(root, raw)
    data.update(cpu_ready=False, cpu_cost_rows=[], cpu_binding={}, operand_root=str(root))
    path = root / "results/experiment_8291_v716_dependency_scoped_admission.json"
    progress("authenticate_before_cpu", 0, 1)
    try:
        value = authenticated(path, raw, data)
        legacy.gate(data, path, "soundness_ready_score", 1, value.get("soundness_ready_score"))
        legacy.gate(data, path, "primitive_replay", True, cpu.replay(path))
        frozen = next(r for r in data["references"] if r.get("original_path") == str(path))
        data.update(
            cpu_ready=True,
            cpu_primary_reference=frozen,
            cpu_cost_rows=value["operation_cost_rows"],
            cpu_binding=dict(
                path=str(path),
                sha256=frozen["sha256"],
                task_id=value["task_id"],
                invocation_argv=value["invocation_argv"],
                phase_spans=value.get("phase_spans", []),
            ),
        )
        raw_work = (
            Path(value.get("terminal_validation_sidecar_path", "missing")).parent / "work.json"
        )
        if raw_work.is_file():
            work = json.loads(raw_work.read_bytes())
            freeze(
                checked(dict(path=work["manifest_path"], sha256=work["manifest_sha256"])), raw, data
            )
            for index, run in enumerate(work["runs"]):
                for key in ["journal", "costs", "prefix"]:
                    freeze(checked(run[key]), raw, data)
                if index % 30 == 0:
                    progress("freeze_cpu_primitives", index + 1, len(work["runs"]) - index - 1)
    except (OSError, ValueError, KeyError, TypeError) as error:
        data["cpu_ready"] = False
        data["checks"].append(
            operand("authenticated_cpu_primitives", path, "qualified CPU evidence", str(error))
        )
    data["cited"].append(
        dict(
            experiment_id=8291,
            path=str(path),
            sha256=cpu.sha256_file(path) if path.is_file() else None,
            imported_fields=[
                "operation_cost_rows",
                "soundness_ready_score",
                "raw_shard_hashes",
                "terminal binding",
                "invocation_argv",
            ],
        )
    )
    progress("authenticate_after_cpu", 1, 0)
    return data


def verify_cpu(data: Json) -> None:
    """Rehashing a reduction cannot replace the authenticated parent invocation."""
    if data.get("cpu_ready") and data.get("cpu_primary_reference"):
        path = checked(data["cpu_primary_reference"])
        value = json.loads(path.read_bytes())
        if value["operation_cost_rows"] != data["cpu_cost_rows"] or not cpu.replay(path):
            raise ValueError("cpu_cost_drift")


def scoped_rows(data: Json) -> list[Json]:
    """Keep measured checking spans intact and never invent a separate counter timer."""
    rows = []
    for index, source in enumerate(data.get("cpu_cost_rows", []) if data.get("cpu_ready") else []):
        components = {
            k: source[k] for k in ["prefix_issue_ns", "closure_ns", "replay_ns", "persistence_ns"]
        }
        total = source["total_ns"]
        if (
            any(type(v) is not int or v < 0 for v in [total, *components.values()])
            or total <= 0
            or sum(components.values()) > total
        ):
            raise ValueError("cpu_span")
        components["host_overhead_ns"] = total - sum(components.values())
        rows.append(
            dict(
                source,
                unit_id=f"cpu:{source['graph_id']}:{source['arm']}:{source['repetition']}",
                source_cluster_id=source["graph_id"],
                condition=source["topology"] + ":" + source["arm"],
                status="completed",
                metric="compatible_work_fraction",
                numerator=0,
                denominator=total,
                host_overhead_ns=components["host_overhead_ns"],
                disjoint_component_ns=components,
                phase_shares={k: v / total for k, v in components.items()},
                counter_update_ns=None,
                counter_timing_scope="integer counters within checking/host spans; no isolated upstream timer",
                typed_constraint_check_ns=None,
                replay_and_typed_checks_ns=source["replay_ns"],
                compatible_fraction=0,
                maximum_gain=1,
                current_fpga_measurement=False,
                evidence_scope="current_exp8291_cpu_fixture",
                invocation_binding=data["cpu_binding"],
                independent_source_count=0,
            )
        )
        if index % 60 == 0:
            progress("cpu_cost_reduction", index + 1, len(data["cpu_cost_rows"]) - index - 1)
    return rows


def reduce(data: Json, measured: list[Json]) -> Json:
    """Publish both ready CPU fixture costs and unavailable complete live costs."""
    with patch.object(legacy, "CURRENT", CURRENT):
        value = legacy.reduce(data, measured)
    scoped = scoped_rows(data)
    cpu_ready = int(
        bool(scoped)
        and data.get("resources_available", True)
        and value["verdict_class"] != "disqualified"
    )
    live_ready = int(
        value["ideal_whole_request_bound"]["status"] != "unavailable"
        and all(data["current"].get(k) for k in CURRENT)
        and value["kv260_boundary_ready_score"] == 1
    )
    conditions = []
    for condition in sorted({r["condition"] for r in scoped}):
        selected = [r for r in scoped if r["condition"] == condition]
        total = sum(r["total_ns"] for r in selected)
        conditions.append(
            dict(
                condition=condition,
                numerator=0,
                denominator=total,
                compatible_fraction=0,
                maximum_gain=1,
                measured_row_count=len(selected),
                graph_unit_count=len({r["graph_id"] for r in selected}),
                source_cost_scope="current_exp8291_cpu_fixture",
                current_fpga_measurement=False,
            )
        )
    value.update(
        scoped_cpu_boundary_ready_score=cpu_ready,
        live_request_boundary_ready_score=live_ready,
        scoped_operation_rows=scoped,
        cpu_condition_bounds=conditions,
        cpu_graph_unit_count=len({r["graph_id"] for r in scoped}),
    )
    for row in value["phase_cost_rows"]:
        if row["source_cost_scope"] == "current_v715":
            row["source_cost_scope"] = "current_v716"
    for row in value["rows"]:
        if row.get("source_cost_scope") == "current_v715":
            row["source_cost_scope"] = "current_v716"
    value["source_cost_scope"].update(
        v716="authenticated current live primitives only; absent clocks unavailable",
        v715="Exp8287 blocked_capture; no intervention costs exist to replace V716 evidence",
        cpu="Exp8291 current CPU oracle fixtures, all graphs/arms/repeats retained; no current board execution",
    )
    value["operation_rows"].extend(
        dict(
            operation=name,
            existing_fabric_supported=False,
            execution="host",
            required_numerical_device_evidence="implemented numerical parity and full transaction timing",
        )
        for name in [
            "graph_closure",
            "integer_counters",
            "typed_constraint_checks",
            "host_overhead",
        ]
    )
    if not live_ready:
        path = (
            Path(data.get("operand_root", ROOT))
            / "results/experiment_8293_v716_fit_view_capture.json"
        )
        value["gate_check_summary"].append(
            operand(
                "complete_live_request_spans",
                path,
                "authenticated tokenization, three views, head, durable feedback, host/transfer and shutdown clocks",
                None,
            )
        )
    if not cpu_ready:
        value["rows"].append(
            dict(
                unit_id="cpu_obligation",
                source_cluster_id="exp8291",
                arm="cpu",
                condition="upstream_availability",
                status="excluded",
                metric="available",
                numerator=None,
                denominator=1,
                exclusion_reason="unavailable_cpu_operand",
            )
        )
    if value["verdict_class"] == "null":
        value.update(
            honest_verdict="complete_circular_positive_cpu_numerical_mechanics"
            if cpu_ready
            else "complete_blocked_cpu",
            verdict_class="circular_positive" if cpu_ready else "blocked",
        )
    value["verifier_is_oracle"] = True
    value["rows"] += scoped
    value.update(
        intended_count=len(value["rows"]),
        completed_count=sum(r["status"] == "completed" for r in value["rows"]),
        failed_count=sum(r["status"] == "failed" for r in value["rows"]),
        excluded_count=sum(r["status"] == "excluded" for r in value["rows"]),
    )
    return value
