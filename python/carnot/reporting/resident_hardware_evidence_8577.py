"""REQ-REPORT-8577: sealed original operations authorize only their measured boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import nine_input_board_evidence_8538 as old
from carnot.reporting.nine_input_board_evidence_8538 import base as base
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.v736_contract_methods import reference

Json = dict[str, Any]
failure = old.failure
BOARD = "results/experiment_8563_v735_hardware_operation_boundary.json"
BOARD_PIN = "sha256:8afa61147f5868f76af1338c09fae6a72f4965464a19ede36acbefd8fdaacfba"
COST = "results/experiment_8573_v736_resident_complete_cost.json"
BLOCKED_COST = "results/experiment_8573_resident_complete_cost.json"
OPERATIONS = [
    "startup_imports",
    "frame_transport",
    "text_features",
    "scaling",
    "offset",
    "residual_109",
    "normalization_fallback",
    "typed_policy",
    "serialization",
    "shutdown",
]
BOARDS = ["KV260", "PolarFire", "GateMate", "NPU", "TSU"]
SUBSETS = [["scaling", "offset", "residual_109"], ["residual_109"]]


def history(root: Path) -> Json:
    """The V735 pin and its terminal report preserve scope without another device probe."""
    path = root / BOARD
    result: Json = dict(qualified=False, checks=[], refs=[], fields={})
    try:
        if reference(path)["sha256"] != BOARD_PIN:
            raise ValueError("original_V735_pin")
        value = json.loads(path.read_bytes())
        if value["required_checks_passed"] is not True or value["flagged_adversarial"] is not False:
            raise ValueError("original_V735_qualification")
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        side = Path(terminal["publication"]["sidecar_path"])
        if read_bound_sidecar(path, side)["report"]["passed"] is not True:
            raise ValueError("original_terminal_binding")
        refs = [
            value["work_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
        ]
        for ref in refs:
            old.base.checked(ref)
        keys = [
            "kv260_supported_operations",
            "polarfire_graduation",
            "board_reopen_requirements",
            "last_authenticated_idcode",
            "missing_source_hashes",
            "vendor_exclusion_rows",
            "historical_model_provenance",
            "conditional_ceiling_rows",
        ]
        result.update(
            qualified=True,
            fields={k: value[k] for k in keys},
            refs=[reference(path), reference(side), *refs],
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(
            failure(
                path,
                "original_V735_board_authority",
                BOARD_PIN,
                str(error),
                "exp8563-hardware-operation-boundary",
            )
        )
    return result


def ceilings(cost: Json) -> list[Json]:
    """An ideal zero arithmetic cost keeps measured transport and all fixed host work."""
    cells = cost["cells"] if cost["qualified"] else []
    if cells:
        reconcile(cells)
    groups = sorted({(c["arm"], c["session_panels"], c["batch"]) for c in cells})
    groups = groups or [
        (a, s, b)
        for a in ["resident_python", "resident_native"]
        for s in [1, 4, 16]
        for b in [1, 8]
    ]
    rows = []
    for arm, session, batch in groups:
        selected = [
            c for c in cells if (c["arm"], c["session_panels"], c["batch"]) == (arm, session, batch)
        ]
        for subset in SUBSETS:
            total = sum(c["wall_ns"] for c in selected) if selected else None
            eligible = (
                sum(
                    s["end_ns"] - s["start_ns"]
                    for c in selected
                    for s in c["spans"]
                    if s["operation"] in subset
                )
                if selected
                else None
            )
            fixed = total - eligible if total is not None else None
            gain = total / fixed if fixed else None
            rows.append(
                dict(
                    arm=arm,
                    session_panels=session,
                    batch=batch,
                    candidate_operations=subset,
                    repeats=len(selected),
                    independent_count=0,
                    complete_clock_ns=total,
                    offloaded_clock_ns=eligible,
                    fixed_host_clock_ns=fixed,
                    zero_cost_ceiling=gain,
                    measured_acceleration=None,
                    conditional=True,
                    nfr01_boundary=("insufficient_for_NFR01" if gain < 10 else "conditional_only")
                    if gain is not None
                    else "unknown_missing_qualified_cost",
                    unknown_costs=dict(precision=None, transport=None, update=None, sampling=None),
                    missing_reason=None
                    if selected
                    else "Exp8573_complete_rows_and_parity_unqualified",
                )
            )
    return rows


def reconcile(cells: list[Json]) -> None:
    """One session wall contains each disjoint operation once; repeats never add independence."""
    seen = set()
    for cell in cells:
        identity = tuple(cell[k] for k in ["arm", "session_panels", "batch", "repeat"])
        spans = sorted(cell["spans"], key=lambda s: s["start_ns"])
        if (
            identity in seen
            or not spans
            or len(spans) != len(OPERATIONS)
            or set(s["operation"] for s in spans) != set(OPERATIONS)
            or any(type(s[k]) is not int or s[k] < 0 for s in spans for k in ["start_ns", "end_ns"])
            or any(s["end_ns"] < s["start_ns"] for s in spans)
            or spans[0]["start_ns"] != 0
            or spans[-1]["end_ns"] != cell["wall_ns"]
            or cell["wall_ns"] <= 0
            or any(a["end_ns"] != b["start_ns"] for a, b in zip(spans, spans[1:]))
        ):
            raise ValueError("complete_disjoint_operation_clocks")
        seen.add(identity)
    if not cells:
        raise ValueError("complete_disjoint_operation_clocks")


def costs(root: Path) -> Json:
    """A pre-gate diagnostic is retained but cannot supply resident timing operands."""
    path = root / COST
    if not path.is_file() and (root / BLOCKED_COST).is_file():
        path = root / BLOCKED_COST
    result: Json = dict(qualified=False, path=str(path), refs=[], checks=[], cells=[])
    expected: Any
    observed: Any
    field, expected, observed = "resident_complete_cost_ready_score", 1, None
    try:
        value = json.loads(path.read_bytes())
        result["refs"].append(reference(path))
        for field, expected in dict(
            resident_complete_cost_ready_score=1,
            resident_inference_ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
            experiment_id=8573,
            task_id="exp8573-resident-complete-cost",
            run_date="20261016",
        ).items():
            observed = value.get(field)
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(field)
        validate_primary(value, path)
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        side = Path(terminal["publication"]["sidecar_path"])
        if read_bound_sidecar(path, side)["report"]["passed"] is not True:
            raise ValueError("resident_terminal_binding")
        parity_ref = value["resident_parity_reference"]
        parity_path = old.base.checked(parity_ref)
        parity = json.loads(parity_path.read_bytes())
        parity_terminal = json.loads(Path(parity["terminal_validation_sidecar_path"]).read_bytes())
        parity_side = Path(parity_terminal["publication"]["sidecar_path"])
        if (
            parity.get("experiment_id") != 8572
            or parity.get("resident_inference_ready_score") != 1
            or parity.get("required_checks_passed") is not True
            or parity.get("flagged_adversarial") is not False
            or read_bound_sidecar(parity_path, parity_side)["report"]["passed"] is not True
        ):
            raise ValueError("qualified_original_resident_parity")
        field, expected, observed = "complete_disjoint_operation_clocks", True, False
        ref = value["resident_operation_clocks_reference"]
        cells = json.loads(old.base.checked(ref).read_bytes())["cells"]
        reconcile(cells)
        expected_cells = {
            (a, s, b, r)
            for a in ["resident_python", "resident_native"]
            for s in [1, 4, 16]
            for b in [1, 8]
            for r in range(10)
        }
        if {
            (c["arm"], c["session_panels"], c["batch"], c["repeat"]) for c in cells
        } != expected_cells:
            raise ValueError("complete_resident_session_matrix")
        result.update(qualified=True, cells=cells)
        result["refs"] += [reference(side), ref, parity_ref, reference(parity_side)]
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(
            failure(path, field, expected, observed, "exp8573-resident-complete-cost")
        )
        result["missing_reason"] = str(error)
    return result
