"""REQ-REPORT-8589: original producer bytes separate board scope from resident cost.

The board report already authenticates historical device evidence. Its immutable
publication is sufficient for this read-only mapping; no device is contacted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting import resident_complete_cost_8584 as resident
from carnot.reporting.nine_input_board_evidence_8538 import failure, reference
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
BOARD = "results/experiment_8563_v735_hardware_operation_boundary.json"
COST = "results/experiment_8584_v737_resident_complete_cost.json"
QUALIFICATION = "results/experiment_8572_v736_resident_inference_qualification.json"
PINS = {
    QUALIFICATION: "sha256:12adfee3ca0fe07bf305438484eab8362ff573f715c8379715f4c1399da0d3db",
    BOARD: "sha256:8afa61147f5868f76af1338c09fae6a72f4965464a19ede36acbefd8fdaacfba",
    COST: "sha256:356aad75439ecaedbbe102e372dc28c2b92e15c848e0d6099165adb6a53995ed",
}
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
MAPPING = dict(
    zip(
        [
            "immutable_setup_launch_imports",
            "transport_parse_and_parent",
            "public_text_features",
            "scaling",
            "offset",
            "binding_residual",
            "normalization_and_python_fallback",
            "typed_decision",
            "serialization_transfer",
            "shutdown",
        ],
        OPERATIONS,
        strict=True,
    )
)
MAPPING["serialization"] = "serialization"
BOARDS = ["KV260", "PolarFire", "GateMate", "NPU", "TSU"]


def original(root: Path, name: str, field: str) -> Json:
    """Independent byte pins and a bound terminal report prevent a summary from authorizing itself."""
    path = root / name
    result: Json = dict(qualified=False, checks=[], refs=[], value=None)
    expected: Any = PINS[name]
    observed: Any = reference(path)["sha256"] if path.is_file() else None
    check = "original_sha256"
    try:
        if observed != expected:
            raise ValueError(check)
        value = json.loads(path.read_bytes())
        result["value"] = value
        result["refs"].append(reference(path))
        check, expected, observed = "original_primary_identity", True, False
        validate_primary(value, path)
        for check, expected in {
            field: 1,
            "required_checks_passed": True,
            "flagged_adversarial": False,
            "run_date": "20261016",
        }.items():
            observed = value.get(check)
            if type(expected) is not type(observed) or observed != expected:
                raise ValueError(check)
        check, expected, observed = "bound_original_publication", True, False
        terminal = Path(value["terminal_validation_sidecar_path"])
        side = Path(json.loads(terminal.read_bytes())["publication"]["sidecar_path"])
        observed = read_bound_sidecar(path, side)["report"]["passed"]
        if observed is not True:
            raise ValueError(check)
        check, expected, observed = "original_primitive_manifest", True, False
        refs = [
            reference(terminal),
            reference(side),
            value["primitive_reference" if name == QUALIFICATION else "work_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
        ]
        for ref in refs:
            operand = Path(ref["path"])
            check, expected = "original_primitive_bytes", ref["sha256"]
            observed = reference(operand)["sha256"] if operand.is_file() else None
            if observed != expected:
                raise ValueError(check)
        result.update(qualified=True, refs=result["refs"] + refs)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(failure(path, check, expected, observed, Path(name).stem))
        result["checks"][-1]["reason"] = str(error)
    return result


def partition(cells: list[Json]) -> None:
    """Complete disjoint host clocks must sum once before any arithmetic can be removed."""
    expected = {
        (length, batch, repeat) for length in [1, 4, 16] for batch in [1, 8] for repeat in range(10)
    }
    if len(cells) != 60 or {(r["length"], r["batch"], r["repeat"]) for r in cells} != expected:
        raise ValueError("complete_resident_roster")
    for cell in cells:
        seconds = cell["seconds"]
        if (
            set(seconds) != set(OPERATIONS)
            or not all(math.isfinite(x) and x >= 0 for x in seconds.values())
            or not math.isfinite(cell["total_s"])
            or cell["total_s"] <= 0
            or not math.isclose(sum(seconds.values()), cell["total_s"], abs_tol=1e-8, rel_tol=0)
        ):
            raise ValueError("disjoint_complete_cost")


def authenticate(root: Path) -> Json:
    """Both branches run unconditionally, so absent optional resident cost cannot hide board evidence."""
    board = original(root, BOARD, "hardware_boundary_ready_score")
    cost = original(root, COST, "resident_cost_ready_score")
    cost["cells"] = []
    if cost["qualified"]:
        try:
            resident.consume(root / COST)
            value = cost["value"]
            if not (
                len(value["rows"]) == value["completed_count"] == 240
                and value["primary_contrast"]["full_parity"]
                and value["lifecycle_contrast"]["full_parity"]
            ):
                raise ValueError("complete_rows_and_parity")
            work = resident.read(value["work_reference"])
            reduced = resident.reduce(work)
            for row in reduced["paired_session_rows"]:
                if row["arm"] != "resident_python":
                    continue
                seconds = {op: 0.0 for op in OPERATIONS}
                for clock in reduced["operation_cost_rows"]:
                    if clock["unit_id"] == row["unit_id"]:
                        seconds[MAPPING[clock["operation"]]] += clock["duration_s"]
                cost["cells"].append(
                    dict(
                        length=row["length"],
                        batch=row["batch"],
                        repeat=row["repeat"],
                        arm=row["arm"],
                        total_s=row["total_session_wall_s"],
                        seconds=seconds,
                    )
                )
            partition(cost["cells"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            cost.update(qualified=False, cells=[])
            cost["checks"].append(
                failure(
                    root / COST,
                    "complete_disjoint_parity",
                    True,
                    str(error),
                    "exp8584-resident-complete-cost",
                )
            )
    qualification = original(root, QUALIFICATION, "resident_inference_ready_score")
    return dict(board=board, cost=cost, resident_qualification=qualification)


def ceilings(cost: Json) -> list[Json]:
    """An ideal removal keeps all fixed host and measured frame work; unknown device costs add overhead."""
    result = []
    for length in [1, 4, 16]:
        for batch in [1, 8]:
            for subset in [["scaling", "offset", "residual_109"], ["residual_109"]]:
                cells = [r for r in cost["cells"] if r["length"] == length and r["batch"] == batch]
                total = sum(r["total_s"] for r in cells) if cost["qualified"] else None
                removed = (
                    sum(r["seconds"][op] for r in cells for op in subset)
                    if cost["qualified"]
                    else None
                )
                fixed = total - removed if total is not None else None
                gain = total / fixed if fixed else None
                result.append(
                    dict(
                        boundary="resident_session",
                        arm="resident_python",
                        length_panels=length,
                        answers_per_panel=64,
                        batch=batch,
                        paired_repeats=10,
                        independent_count=0,
                        candidate_operations=subset,
                        complete_clock_s=total,
                        offloaded_clock_s=removed,
                        fixed_host_clock_s=fixed,
                        included_frame_transport_s=(
                            sum(
                                r["seconds"]["frame_transport"] + r["seconds"]["serialization"]
                                for r in cells
                            )
                            if cost["qualified"]
                            else None
                        ),
                        zero_cost_ceiling=gain,
                        measured_acceleration=None,
                        conditional=True,
                        nfr01_boundary=(
                            "insufficient_for_NFR01" if gain < 10 else "conditional_only"
                        )
                        if gain is not None
                        else "unknown_missing_qualified_cost",
                        unknown_costs=dict(
                            device_transfer=None, precision=None, update=None, sampling=None
                        ),
                        missing_reason=None if cost["qualified"] else "Exp8584_not_qualified",
                    )
                )
    return result


def matrix(source: Json) -> list[Json]:
    """Mapping a host operation to a board does not install a compatible fabric implementation."""
    rows = []
    for board in BOARDS:
        for op in OPERATIONS:
            rows.append(
                dict(
                    board=board,
                    operation=op,
                    assigned_substrate="host_CPU",
                    input_count=9,
                    residual_coefficient_count=109 if op == "residual_109" else None,
                    device_supported=False,
                    deployment_ready=False,
                    cost_scope="resident_session_by_length_and_batch",
                    qualified_cost_cells=(
                        [
                            dict(
                                length_panels=c["length"],
                                batch=c["batch"],
                                repeat=c["repeat"],
                                duration_s=c["seconds"][op],
                            )
                            for c in source["cost"]["cells"]
                        ]
                        if source["cost"]["qualified"]
                        else []
                    ),
                    device_clock_s=None,
                    transport_clock_s=None,
                    board_cpu_scope="historical_Linux_CPU_dispatch_only"
                    if board == "PolarFire"
                    else None,
                    missing_reason="no_authenticated_compatible_installed_operation",
                )
            )
    return rows
