"""REQ-VERIFY-8287: preserve hardware scope while resolving current evidence costs.

Qualified readers perform the arithmetic. This adapter binds V715 operands
and never replaces missing acquisition clocks with historical measurements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import kv260_evidence_cost_boundary_8273 as legacy
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.request_trace_inventory_8200 import operand

Json = dict[str, Any]
ROOT = legacy.ROOT
NAME = "experiment_8287_v715_kv260_evidence_cost_boundary"
CLI = "scripts/experiments/" + NAME + ".py"
MODEL_SPECS: list[Json] = []
CONFIG: Json = dict(seed=7158287, formats=["Q8.8", "Q16.16"], current_device_calls=0)
PRIMITIVE_SCHEMA = "carnot.v715.kv260-operands.v1"
CURRENT = dict(
    capture=(8279, "v715_fit_view_capture"),
    tune=(8280, "v715_tune_view_capture"),
    heads=(8281, "v715_intervention_energy_fit"),
    seal=(8282, "v715_reserved_view_seal"),
    counters=(8284, "v715_continuous_constraint_admission"),
)
SOURCES = dict(
    boundary=(8244, "v712_kv260_decision_boundary"),
    **CURRENT,
    historical_boundary=(8273, "v714_kv260_evidence_cost_boundary"),
)
prior = legacy.prior
freeze = legacy.freeze


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush finite work counts so supervisors can detect stalled work."""
    print(f"[exp8287] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(data: Json, path: Path, field: str, expected: Any, observed: Any) -> None:
    """Retain the exact failed field rather than only the exception text."""
    data["checks"].append(operand(field, path, expected, observed))
    if observed != expected:
        raise ValueError(field)


def authenticated(path: Path, raw: Path, data: Json) -> Json:
    """Bind terminal reports to the primary path and bytes before reading primitives."""
    gate(data, path, "exists", True, path.is_file())
    value: Json = json.loads(freeze(path, raw, data).read_bytes())
    validate_primary(value, path)
    for field, expected in [("required_checks_passed", True), ("flagged_adversarial", False)]:
        gate(data, path, field, expected, value.get(field))
    digest = sha256_file(path)
    side = path.parent / "raw" / path.stem / "validators" / (digest[7:] + ".json")
    gate(data, side, "exists", True, side.is_file())
    report = json.loads(side.read_bytes())
    gate(data, side, "primary_sha256", digest, report.get("primary_sha256"))
    gate(data, side, "primary_path", str(path.absolute()), report.get("primary_path"))
    gate(data, side, "report.passed", True, report.get("report", {}).get("passed"))
    read_bound_sidecar(path, side)
    freeze(side, raw, data)
    for ref in value.get("raw_shard_hashes", []):
        gate(
            data,
            Path(ref["path"]),
            "sha256",
            ref["sha256"],
            sha256_file(Path(ref["path"])) if Path(ref["path"]).is_file() else None,
        )
        freeze(checked(ref), raw, data)
    return value


def load(root: Path, raw: Path) -> Json:
    """Resolve branches through qualified readers while retaining each invocation."""
    with (
        patch.object(legacy, "CURRENT", CURRENT),
        patch.object(legacy, "SOURCES", SOURCES),
        patch.object(legacy, "PRIMITIVE_SCHEMA", PRIMITIVE_SCHEMA),
        patch.object(legacy, "progress", progress),
        patch.object(legacy.qualified, "authenticated", authenticated),
    ):
        data = legacy.load(root, raw)
    frozen = {r["original_path"]: Path(r["path"]) for r in data["references"]}
    data["request_bindings"] = {}
    for role, binding in data["cost_bindings"].items():
        if role in {"capture", "tune", "seal"}:
            primary = json.loads(frozen[binding["path"]].read_bytes())
            ref = primary["boundary_primitives_reference"]
            primitive = json.loads(frozen[ref["path"]].read_bytes())
            for request in primitive["requests"]:
                data["request_bindings"][request["request_id"]] = dict(binding, role=role)
    return data


def precision(data: Json) -> list[Json]:
    """Keep exact CPU bases and separate fixtures from eligible natural states."""
    with patch.object(legacy.qualified, "progress", progress):
        return legacy.precision(data)


def reduce(data: Json, measured: list[Json]) -> Json:
    """Bind request costs individually and keep current and historical scopes distinct."""
    with patch.object(legacy, "CURRENT", CURRENT):
        value = legacy.reduce(data, measured)
    value["source_cost_scope"] = dict(
        v715="authenticated current primitives only; absent clocks are unavailable",
        v712="historical Exp8242 request spans; no current three-view acquisition",
        v714="Exp8273 f=0 used historical Exp8242 spans because current captures did not exist",
        fixture="CPU numerical mechanics only; no natural benefit or FPGA timing",
    )
    bindings = data.get("request_bindings", {})
    for row in value["phase_cost_rows"]:
        if row["source_cost_scope"] == "current_v714":
            row["source_cost_scope"] = "current_v715"
            row["invocation_binding"] = [
                bindings.get(r["request_id"])
                for r in data["requests"]
                if r["workload"] == row["condition"]
            ]
    for row in value["rows"]:
        if "clocks" in row:
            row.update(
                source_cost_scope="current_v715", invocation_binding=bindings.get(row["unit_id"])
            )
    value["acquisition_accounting"].update(
        imported_canary_rows=sum(bool(r.get("imported_canary")) for r in data["requests"]),
        actual_new_request_counts={
            role: sum(b.get("role") == role for b in bindings.values())
            for role in ["capture", "tune", "seal"]
        },
        imported_cold_load_count=len(data["cold_costs"]),
        current_cold_loads=0,
        current_llm_calls=0,
        tokenizer_only_is_model_inference=False,
    )
    return value
