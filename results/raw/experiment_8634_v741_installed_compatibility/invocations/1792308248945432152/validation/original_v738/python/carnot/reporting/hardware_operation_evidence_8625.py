"""REQ-REPORT-8625: original byte pins prevent summaries from granting themselves authority."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_operation_evidence_8613 as old
from carnot.reporting import resident_hardware_evidence_8589 as clocks
from carnot.reporting import v740_contract_methods as authority
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.nine_input_board_evidence_8538 import failure, reference
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
BOARD, COST = old.BOARD, old.COST
LEAN = "results/experiment_8622_v740_lean_resident_cost.json"
LEARNING = "results/experiment_8618_v740_causal_brier_stream.json"
METHODS = "results/experiment_8615_v740_contract_methods.json"
PINS = dict(
    old.PINS,
    **{
        LEAN: "sha256:53ad0975f8b227e921bd9164541b846d617c5c338e8216d0bc0761c5dcf7d40a",
        METHODS: "sha256:6f7ea729a61997543357bc926f558510d0dbfdb95cabcd42aecbce936267209e",
    },
)
OPERATIONS = [*clocks.OPERATIONS, "parent_residual"]
BOARDS = clocks.BOARDS
MAPPING = dict(
    clocks.MAPPING,
    startup="startup_imports",
    imports="startup_imports",
    frame_transport="frame_transport",
    parent_residual="parent_residual",
)


def checked(ref: Json) -> Json:
    """Resolve only registered original snapshots when a live roadmap has since advanced."""
    path = Path(ref["path"])
    if str(path) in {
        str(authority.ROOT / authority.DESIGN),
        str(authority.ROOT / authority.ACTIVE),
    }:
        protocol = authority.scientific.protocol()
        for key in ["v738_design_reference", "v738_activation_reference"]:
            original = protocol[key]
            if original["source_path"] == str(path) and original["sha256"] == ref["sha256"]:
                path = Path(original["path"])
    if not path.is_file() or sha256_file(path) != ref["sha256"]:
        raise ValueError("original_operand_hash:" + str(path))
    return dict(reference(path))


def sealed(root: Path, name: str, field: str) -> Json:
    """Pin producer identity, its publication, and its immutable operands without running its controller."""
    path = root / name
    result: Json = dict(qualified=False, value=None, refs=[], checks=[], requested_path=str(path))
    field_name, expected, observed = "original_sha256", PINS.get(name), None
    try:
        observed = sha256_file(path) if path.is_file() else None
        if observed is None or expected is None or observed != expected:
            raise ValueError("unregistered_or_missing_original")
        value = json.loads(path.read_bytes())
        result["refs"].append(reference(path))
        validate_primary(value, path)
        for field_name, expected in {
            field: 1,
            "required_checks_passed": True,
            "flagged_adversarial": False,
        }.items():
            observed = value.get(field_name)
            if type(expected) is not type(observed) or expected != observed:
                raise ValueError(field_name)
        field_name, expected, observed = "bound_original_publication", True, False
        terminal = Path(value["terminal_validation_sidecar_path"])
        envelope = json.loads(terminal.read_bytes())
        side = (
            Path(envelope["publication"]["sidecar_path"]) if "publication" in envelope else terminal
        )
        observed = read_bound_sidecar(path, side)["report"]["passed"]
        if observed is not True:
            raise ValueError(field_name)
        field_name, expected, observed = "immutable_original_operands", True, False
        refs = [
            value.get("work_reference", value.get("primitive_reference")),
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
        ]
        if name == METHODS:
            # Methods authority comes from its registered protocol and primitive.
            # Old validator source versions remain historical validation provenance.
            refs = [
                value["primitive_reference"],
                value["methods_protocol_reference"],
                value["learning_protocol_reference"],
                *value["raw_shard_hashes"],
            ]
        result["refs"] += [checked(r) for r in refs]
        if value.get("target_model"):
            result["refs"].append(checked(value["target_model"]["checkpoint"]))
        result["refs"] += [reference(terminal), reference(side)]
        result.update(qualified=True, value=value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if expected is None:
            field_name, expected = field, 1
        row = failure(path, field_name, expected, observed, Path(name).stem)
        row.update(passed=False, reason=str(error))
        result["checks"].append(row)
    return result


def partition(rows: list[Json]) -> None:
    """The full paired roster must have one nonnegative clock for every host operation."""
    projected = deepcopy(rows)
    for row in projected:
        if set(row["seconds"]) != set(OPERATIONS):
            raise ValueError("operation_roster")
        row["seconds"]["frame_transport"] += row["seconds"].pop("parent_residual")
    clocks.partition(projected)


def cells(cost: Json) -> list[Json]:
    """Read integer primitive spans rather than treating a reported operation sum as a measurement."""
    value = cost["value"]
    lean = value["experiment_id"] == 8622
    rows = []
    for row in value["paired_session_rows"]:
        if row["arm"] != ("lean_python" if lean else "resident_python"):
            continue
        primitive = json.loads(Path(checked(row["primitive_reference"])["path"]).read_bytes())
        seconds = dict.fromkeys(OPERATIONS, 0.0)
        if lean:
            spans = primitive["operations"]
            if (
                primitive["exit_code"] != 0
                or spans[0]["start_ns"] != primitive["start_ns"]
                or spans[-1]["end_ns"] != primitive["end_ns"]
            ):
                raise ValueError("primitive_endpoints_or_exit")
            for index, span in enumerate(spans):
                if (
                    type(span["start_ns"]) is not int
                    or type(span["end_ns"]) is not int
                    or span["end_ns"] < span["start_ns"]
                    or (index and spans[index - 1]["end_ns"] != span["start_ns"])
                ):
                    raise ValueError("primitive_overlap_or_gap")
                seconds[MAPPING[span["operation"]]] += (span["end_ns"] - span["start_ns"]) / 1e9
        else:
            for span in value["operation_cost_rows"]:
                if span["unit_id"] == row["unit_id"]:
                    seconds[MAPPING[span["operation"]]] += span["duration_s"]
        rows.append(
            dict(
                length=row["length"],
                batch=row["batch"],
                repeat=row["repeat"],
                arm=row["arm"],
                total_s=row["total_session_wall_s"],
                seconds=seconds,
            )
        )
    partition(rows)
    return rows


def authenticate(root: Path) -> Json:
    """Every CPU branch runs even when another producer or an optional device is absent."""
    board = sealed(root, BOARD, "hardware_boundary_ready_score")
    lean = sealed(root, LEAN, "lean_cost_ready_score")
    learning = sealed(root, LEARNING, "learning_cost_ready_score")
    diagnostic = root / LEARNING.replace("_v740", "")
    if not learning["qualified"] and diagnostic.is_file():
        learning["refs"].append(reference(diagnostic))
        for gate in json.loads(diagnostic.read_bytes()).get("gates_evaluated", []):
            learning["checks"].append(
                failure(
                    Path(gate["artifact_path"]),
                    gate["artifact_field"],
                    gate["expected"],
                    gate["actual"],
                    gate["upstream"],
                )
            )
    cost = deepcopy(lean) if lean["qualified"] else sealed(root, COST, "resident_cost_ready_score")
    cost["cells"] = []
    if cost["qualified"]:
        try:
            if (
                not cost["value"]["primary_contrast"]["full_parity"]
                or not cost["value"]["lifecycle_contrast"]["full_parity"]
            ):
                raise ValueError("complete_parity")
            cost["cells"] = cells(cost)
        except (OSError, KeyError, ValueError, TypeError) as error:
            cost.update(qualified=False, cells=[])
            cost["checks"].append(
                failure(root / LEAN, "disjoint_complete_cost", True, str(error), "exp8622")
            )
            lean.update(qualified=False, checks=list(cost["checks"]))
            cost = sealed(root, COST, "resident_cost_ready_score")
            cost["cells"] = cells(cost) if cost["qualified"] else []
    return dict(board=board, lean=lean, learning=learning, cost=cost)


def matrices(source: Json) -> tuple[list[Json], list[Json]]:
    """Each operation stays on the host until compatible installed execution is authenticated."""
    boundary = (
        "qualified_Exp8622_lean_session"
        if source["lean"]["qualified"]
        else "historical_Exp8596_resident_session"
    )
    resident = [
        dict(
            board=board,
            operation=op,
            boundary=boundary,
            assigned_substrate="host_CPU_Rust" if op == "residual_109" else "host_CPU",
            device_supported=False,
            deployment_ready=False,
            residual_coefficient_count=109 if op == "residual_109" else None,
            qualified_cost_cells=[
                dict(
                    length_panels=c["length"],
                    batch=c["batch"],
                    repeat=c["repeat"],
                    duration_s=c["seconds"][op],
                )
                for c in source["cost"]["cells"]
            ],
            device_clock_s=None,
            transport_clock_s=None,
            board_cpu_scope="historical_Linux_CPU_dispatch_only" if board == "PolarFire" else None,
            missing_reason="no_authenticated_compatible_installed_operation",
        )
        for board in BOARDS
        for op in OPERATIONS
    ]
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
            missing_reason="Exp8618_complete_update_cost_not_qualified",
        )
        for board in BOARDS
        for op in [*OPERATIONS, "gradient_updates", "checkpoint_sync", "feedback_drain"]
    ]
    return resident, learning


def ceilings(cost: Json) -> list[Json]:
    """Ideal arithmetic removal retains fixed host work and leaves transfer and precision costs unknown."""
    rows = clocks.ceilings(cost)
    for row in rows:
        row.update(
            boundary="qualified_Exp8622_lean_session"
            if cost["qualified"] and cost["value"]["experiment_id"] == 8622
            else "historical_Exp8596_resident_session",
            arm=cost["cells"][0]["arm"] if cost["cells"] else None,
        )
        if not cost["qualified"]:
            row["missing_reason"] = "no_qualified_complete_resident_cost"
    return list(rows)
