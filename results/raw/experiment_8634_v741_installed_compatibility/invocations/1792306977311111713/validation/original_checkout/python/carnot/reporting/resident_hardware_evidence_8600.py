"""REQ-REPORT-8600: new cost authority never repairs rejected historical operands."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot.reporting import resident_complete_cost_8596 as resident
from carnot.reporting import resident_hardware_evidence_8589 as base
from carnot.reporting.nine_input_board_evidence_8538 import failure, reference
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
__all__ = ["base", "resident"]
BOARD, BOARDS, OPERATIONS = base.BOARD, base.BOARDS, base.OPERATIONS
PINS = base.PINS
COST = "results/experiment_8596_v738_resident_complete_cost.json"
COST_PIN = "sha256:5dd00385cb4f330b900f842af1a43b580f06c5948a658c36ce3b01d35b46ac6c"
matrix, partition = base.matrix, base.partition
_original, _authenticate = base.original, base.authenticate


def original(root: Path, name: str, field: str) -> Json:
    """An independent cost pin and exact publication receipt precede primitive reconstruction."""
    if name != COST:
        return dict(_original(root, name, field))
    path = root / name
    result: Json = dict(qualified=False, checks=[], refs=[], value=None)
    expected: Any
    check, expected = "original_sha256", COST_PIN
    observed: Any = reference(path)["sha256"] if path.is_file() else None
    try:
        if observed != expected:
            raise ValueError(check)
        value = json.loads(path.read_bytes())
        validate_primary(value, path)
        for check, expected in {
            field: 1,
            "required_checks_passed": True,
            "flagged_adversarial": False,
        }.items():
            observed = value.get(check)
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(check)
        check, expected, observed = "bound_original_publication", True, False
        terminal = Path(value["terminal_validation_sidecar_path"])
        sidecar = Path(json.loads(terminal.read_bytes())["publication"]["sidecar_path"])
        observed = read_bound_sidecar(path, sidecar)["report"]["passed"]
        if observed is not True:
            raise ValueError(check)
        result.update(
            qualified=True,
            value=value,
            refs=[reference(path), reference(terminal), reference(sidecar)],
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(
            failure(path, check, expected, observed, "exp8596-resident-complete-cost")
        )
        result["checks"][-1]["reason"] = str(error)
    return result


def consume(path: Path) -> Json:
    """Use the producer's own immutable replay under its registered identity, without rerunning measurements."""
    value = json.loads(path.read_bytes())
    with resident.bindings(value["run_date"]):
        return dict(resident.base.consume(path))


def authenticate(root: Path) -> Json:
    """Both CPU branches run even when boards or fresh cost evidence are unavailable."""
    bridge = SimpleNamespace(consume=consume, read=resident.base.read, reduce=resident.base.reduce)
    with patch.multiple(base, COST=COST, original=original, resident=bridge):
        result = dict(_authenticate(root))
    for row in result["cost"]["checks"]:
        row["upstream"] = "exp8596-resident-complete-cost"
    return result


def ceilings(cost: Json) -> list[Json]:
    """The inherited disjoint-clock formula retains host and frame costs exactly once."""
    rows = list(base.ceilings(cost))
    for row in rows:
        if row["missing_reason"]:
            row["missing_reason"] = "Exp8596_not_qualified"
    return rows
