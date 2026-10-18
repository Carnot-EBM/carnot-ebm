"""REQ-REPORT-8613: pinned original publications authorize historical clocks only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import resident_hardware_evidence_8589 as prior
from carnot.reporting import v739_contract_methods as authority
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.nine_input_board_evidence_8538 import failure, reference
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
BOARD = "results/experiment_8600_v738_resident_hardware_boundary.json"
COST = "results/experiment_8596_v738_resident_complete_cost.json"
LEAN = "results/experiment_8610_v739_lean_resident_cost.json"
LEARNING = "results/experiment_8605_v739_causal_brier_stream.json"
PINS = {
    BOARD: "sha256:9a3e51adf232e530cbb4ad885b43d216b05dbe5cbb6210fc064b3223b16b2270",
    COST: "sha256:5dd00385cb4f330b900f842af1a43b580f06c5948a658c36ce3b01d35b46ac6c",
}
OPERATIONS, BOARDS, partition = prior.OPERATIONS, prior.BOARDS, prior.partition
LEARNING_OPERATIONS = [*OPERATIONS, "gradient_updates", "checkpoint_sync", "feedback_drain"]


def checked(ref: Json) -> Json:
    """Only the sealed original authority may replace a scheduling path that later advanced."""
    path = Path(ref["path"])
    if path == authority.ROOT / authority.ACTIVE:
        original = authority.protocol()["v738_activation_reference"]
        if ref["sha256"] == original["sha256"]:
            path = Path(original["path"])
    if not path.is_file() or sha256_file(path) != ref["sha256"]:
        raise ValueError("original_operand_hash:" + str(path))
    return dict(reference(path))


def sealed(root: Path, name: str, field: str) -> Json:
    """A summary cannot grant itself authority by rehashing its own mutated operands."""
    path = root / name
    result: Json = dict(
        qualified=False,
        value=None,
        refs=[],
        checks=[],
        requested_path=str(path),
        diagnostic_reference=None,
    )
    diagnostic = root / name.replace("_v739", "")
    if name in {LEAN, LEARNING} and not path.is_file() and diagnostic.is_file():
        result["checks"].append(
            failure(path, "required_producer_exists", True, None, Path(name).stem)
        )
        data = json.loads(diagnostic.read_bytes())
        result["diagnostic_reference"] = reference(diagnostic)
        result["refs"].append(reference(diagnostic))
        for gate in data["gates_evaluated"]:
            result["checks"].append(
                failure(
                    Path(gate["artifact_path"]),
                    gate["artifact_field"],
                    gate["expected"],
                    gate["actual"],
                    gate["upstream"],
                )
            )
        path = diagnostic
    expected: Any = PINS.get(name)
    observed: Any = None
    check = "original_sha256"
    try:
        observed = sha256_file(path) if path.is_file() else None
        if expected is not None and observed != expected:
            raise ValueError(check)
        value = json.loads(path.read_bytes())
        result["refs"].append(reference(path))
        for check, expected in {
            field: 1,
            "required_checks_passed": True,
            "flagged_adversarial": False,
        }.items():
            observed = value.get(check)
            if type(expected) is not type(observed) or expected != observed:
                raise ValueError(check)
        check, expected, observed = "original_publication", True, False
        validate_primary(value, path)
        terminal = Path(value["terminal_validation_sidecar_path"])
        side = Path(json.loads(terminal.read_bytes())["publication"]["sidecar_path"])
        observed = read_bound_sidecar(path, side)["report"]["passed"]
        if observed is not True:
            raise ValueError(check)
        refs = [
            value["work_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
        ]
        check, expected, observed = "immutable_original_operands", True, False
        result["refs"] += [checked(r) for r in refs]
        result["refs"] += [reference(terminal), reference(side)]
        result.update(qualified=True, value=value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(failure(path, check, expected, observed, Path(name).stem))
        result["checks"][-1]["reason"] = str(error)
    return result


def authenticate(root: Path) -> Json:
    """Read each producer once; blocked future outputs cannot supply zero-cost operations."""
    board = sealed(root, BOARD, "hardware_boundary_ready_score")
    lean = sealed(root, LEAN, "lean_cost_ready_score")
    learning = sealed(root, LEARNING, "learning_cost_ready_score")
    cost = lean if lean["qualified"] else sealed(root, COST, "resident_cost_ready_score")
    cost["cells"] = []
    if cost["qualified"]:
        try:
            value = cost["value"]
            if not (
                value["primary_contrast"]["full_parity"]
                and value["lifecycle_contrast"]["full_parity"]
            ):
                raise ValueError("complete_parity")
            for row in value["paired_session_rows"]:
                if row["arm"] != "resident_python":
                    continue
                seconds = dict.fromkeys(OPERATIONS, 0.0)
                for clock in value["operation_cost_rows"]:
                    if clock["unit_id"] == row["unit_id"]:
                        seconds[prior.MAPPING[clock["operation"]]] += clock["duration_s"]
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
        except (KeyError, ValueError, TypeError) as error:
            cost.update(qualified=False, cells=[])
            cost["checks"].append(
                failure(root / COST, "complete_disjoint_parity", True, str(error), "exp8596")
            )
    return dict(board=board, cost=cost, lean=lean, learning=learning)


def matrices(source: Json) -> tuple[list[Json], list[Json]]:
    """Host residence is executable; no compatible installed accelerator follows from sparsity."""
    resident = prior.matrix(source)
    for row in resident:
        row["boundary"] = (
            "historical_Exp8596_resident_session"
            if not source["lean"]["qualified"]
            else "qualified_Exp8610_resident_session"
        )
    learning = [
        dict(
            board=board,
            operation=op,
            boundary="delayed_learning",
            assigned_substrate="host_CPU_Rust",
            device_supported=False,
            deployment_ready=False,
            host_clock_s=None,
            device_clock_s=None,
            missing_reason="Exp8605_complete_update_cost_not_qualified",
        )
        for board in BOARDS
        for op in LEARNING_OPERATIONS
    ]
    return resident, learning


def ceilings(cost: Json) -> list[Json]:
    """Ideal arithmetic removal keeps every fixed host clock and charges unknown device work."""
    rows = prior.ceilings(cost)
    for row in rows:
        if row["missing_reason"]:
            row["missing_reason"] = "no_qualified_complete_resident_cost"
    return list(rows)
