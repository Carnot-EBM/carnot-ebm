"""REQ-VERIFY-8273: current acquisition cannot inherit historical fabric timing.

The qualified CPU precision and cost readers remain unchanged. This adapter
resolves current operands independently and labels every imported cost row.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import kv260_evidence_cost_boundary_8258 as qualified
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.request_trace_inventory_8200 import operand

Json = dict[str, Any]
ROOT = qualified.ROOT
NAME = "experiment_8273_v714_kv260_evidence_cost_boundary"
CLI = "scripts/experiments/" + NAME + ".py"
MODEL_SPECS: list[Json] = []
CONFIG: Json = dict(seed=7148273, formats=["Q8.8", "Q16.16"], current_device_calls=0)
PRIMITIVE_SCHEMA = "carnot.v714.kv260-operands.v1"
CURRENT = dict(
    capture=(8265, "v714_fit_view_capture"),
    tune=(8266, "v714_tune_view_capture"),
    heads=(8267, "v714_intervention_energy_fit"),
    seal=(8268, "v714_reserved_view_seal"),
    counters=(8270, "v714_continuous_constraint_admission"),
)
SOURCES = dict(
    boundary=(8244, "v712_kv260_decision_boundary"),
    **CURRENT,
    historical_boundary=(8258, "v713_kv260_evidence_cost_boundary"),
)
prior = qualified.prior
freeze = qualified.freeze
precision = qualified.precision


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush each phase boundary and finite-loop count for process supervision."""
    print(f"[exp8273] phase={phase} completed={completed} pending={pending}", flush=True)


def load(root: Path, raw: Path) -> Json:
    """Keep qualified board custody even when current science is unavailable."""
    data = prior.load(root, raw)
    data.update(
        historical_requests=data["requests"],
        historical_cold_costs=data["cold_costs"],
        historical_service_spans=[],
        historical_cost_binding={},
        requests=[],
        cold_costs=[],
        service_spans=[],
        current={},
        current_heads=[],
        current_samples=[],
        counter_states=[],
        cost_bindings={},
    )
    service = root / "results/experiment_8242_v712_independent_concurrent_service.json"
    if data["branches"]["service"]:
        value = json.loads(service.read_bytes())
        data["historical_service_spans"] = value["phase_spans"]
        data["historical_cost_binding"] = dict(
            path=str(service),
            sha256=sha256_file(service),
            task_id=value.get("task_id"),
            invocation=value.get("invocation_clocks"),
        )
    for index, (name, (eid, suffix)) in enumerate(SOURCES.items()):
        progress("authenticate_before_" + name, index, len(SOURCES) - index)
        path = root / "results" / f"experiment_{eid}_{suffix}.json"
        field = "exists"
        observed: Any = path.is_file()
        expected: Any = True
        try:
            data["checks"].append(operand(field, path, expected, observed))
            preview = json.loads(path.read_bytes())
            field, expected, observed = (
                "required_checks_passed",
                True,
                preview.get("required_checks_passed"),
            )
            data["checks"].append(operand(field, path, expected, observed))
            value = qualified.authenticated(path, raw, data)
            if name == "boundary":
                field, expected, observed = (
                    "kv260_obligation.historical",
                    data["board"],
                    value["kv260_obligation"]["historical"],
                )
                if observed != expected:
                    raise ValueError("historical_board_drift")
            elif name in CURRENT:
                field, expected = (
                    "boundary_primitives_reference",
                    "qualified current primitive reference",
                )
                observed = value.get(field)
                primitive = json.loads(freeze(checked(value[field]), raw, data).read_bytes())
                field, expected, observed = (
                    "boundary_primitives_reference.schema",
                    PRIMITIVE_SCHEMA,
                    primitive.get("schema"),
                )
                if observed != expected:
                    raise ValueError("current_primitive_schema")
                if name in {"capture", "tune", "seal"}:
                    data["requests"].extend(primitive["requests"])
                    data["cold_costs"].extend(primitive["cold_costs"])
                    data["service_spans"].extend(primitive["service_spans"])
                if name == "heads":
                    prior.validate_heads(primitive["current_heads"])
                    data["current_heads"], data["current_samples"] = (
                        primitive["current_heads"],
                        primitive["current_samples"],
                    )
                if name == "counters":
                    data["counter_states"] = primitive["counter_states"]
                data["cost_bindings"][name] = dict(
                    path=str(path),
                    sha256=sha256_file(path),
                    task_id=value["task_id"],
                    invocation=value.get("invocation_clocks"),
                )
            else:
                data["imported_exp8258_scope"] = value.get("source_cost_scope")
            data["current"][name] = True
            data["checks"].append(operand(field, path, expected, observed))
        except (OSError, ValueError, KeyError, TypeError) as error:
            data["checks"].append(
                dict(operand(field, path, expected, observed), passed=False, error=str(error))
            )
            data["current"][name] = False
        data["cited"].append(
            dict(
                experiment_id=eid,
                path=str(path),
                sha256=sha256_file(path) if path.is_file() else None,
                imported_fields=[
                    "required_checks_passed",
                    "terminal binding",
                    "boundary_primitives_reference",
                    "kv260_obligation.historical",
                    "source_cost_scope",
                    "invocation_clocks",
                ],
            )
        )
        progress("authenticate_after_" + name, index + 1, len(SOURCES) - index - 1)
    return data


def reduce(data: Json, measured: list[Json]) -> Json:
    """Separate current complete-request bounds from exposed historical work."""
    with patch.object(qualified, "CURRENT", CURRENT):
        value = qualified.reduce(data, measured)
    historical = deepcopy(data)
    for key in ["requests", "cold_costs", "service_spans"]:
        historical[key] = data["historical_" + key]
    history_rows, history_bound = qualified.costs(historical)
    for row in history_rows:
        row.update(
            unit_id="historical_" + row["unit_id"],
            source_cost_scope="historical_v712",
            invocation_binding=data["historical_cost_binding"],
            current_acquisition=False,
        )
    for row in value["phase_cost_rows"]:
        row.update(
            source_cost_scope="current_v714",
            invocation_binding=data.get("cost_bindings", {}),
            current_acquisition=True,
        )
    value["phase_cost_rows"] += history_rows
    value["rows"] += history_rows
    for status, count in [
        ("completed", "completed_count"),
        ("failed", "failed_count"),
        ("excluded", "excluded_count"),
    ]:
        value[count] = sum(r["status"] == status for r in value["rows"])
    value["intended_count"] = len(value["rows"])
    value.update(
        historical_ideal_whole_request_bound=history_bound,
        source_cost_scope=dict(
            v714="authenticated current primitives only; unavailable operands retain named rows",
            v712="authenticated historical Exp8242 clocks, not current three-view costs",
            v713="Exp8258 f=0 used historical Exp8242 spans before current captures existed",
            fixture="CPU numerical mechanics only",
        ),
        acquisition_accounting=dict(
            imported_canary="only with producer invocation binding",
            new_fit_tune_reserved="only with qualified current request primitives",
            current_llm_calls=0,
            current_cold_loads=0,
            audit_only_cached_scoring=len(measured),
            complete_request_components=[
                "tokenization",
                "three_view_requests",
                "head_scoring",
                "durable_feedback",
                "host_transfer",
                "shutdown",
            ],
        ),
    )
    value["operation_rows"].append(
        dict(
            operation="quadratic_ising",
            existing_fabric_supported=True,
            k_max=5,
            execution="historical fabric",
            current_device_execution=False,
        )
    )
    value["kv260_obligation"].update(
        k_max=5,
        supported_operation="historical quadratic Ising only",
        npu_access="no authenticated NPU execution",
        tsu_access="no authenticated TSU hardware access",
        vendor_estimates="literature and vendor projections are not measured speedups",
    )
    return value
