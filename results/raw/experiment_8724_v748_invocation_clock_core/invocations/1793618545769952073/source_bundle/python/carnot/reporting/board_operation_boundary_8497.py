"""REQ-REPORT-8497: complete current authority preserves original board limits.

The shipped reader retains original byte seals. This narrow adapter changes
the current task and optional cost producer without claiming new device work.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8483 as old
from carnot.reporting import v730_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json

Json = dict[str, Any]
ROOT, base = old.ROOT, old.base
NAME, TASK, MILESTONE = (
    "experiment_8497_v730_board_operation_boundary",
    "exp8497-board-operation-boundary",
    "2026.10.730",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_board_operation_boundary_8497.py"
OWNED = [
    "python/carnot/reporting/board_operation_boundary_8497.py",
    "python/carnot/reporting/board_operation_runner_8497.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
COST = "results/experiment_8494_v730_append_service_cost.json"
PRIOR, PRIOR_PIN, OPERATIONS = old.PRIOR, old.PRIOR_PIN, old.OPERATIONS
reference, checksum, aliases = old.reference, old.checksum, old.aliases
_MEASURE, _BUILD, _REPLAY, _COSTS = old.measure, old.build, old.replay, old.costs
operation_mapping, prior = old.operation_mapping, old.prior


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts because an inventory must not look like a silent device probe."""
    print(f"[exp8497] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Restore imported globals so historical consumers retain their original byte authority."""
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        COST=COST,
        authority=authority,
        build=build,
        progress=progress,
    ):
        yield


def costs(root: Path, raw: Path) -> Json:
    """Current append clocks remain optional; fallback clocks keep their original producer."""
    with binding():
        return dict(_COSTS(root, raw))


def measure(root: Path, raw: Path, private: Path | None = None) -> Json:
    """Authenticate the complete V730 task objects before reducing sealed original evidence."""
    with binding():
        work = dict(_MEASURE(root, raw, private))
    work["code_config_hashes"].append(
        reference(ROOT / "python/carnot/reporting/v730_contract_methods.py")
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Administrative recording grants no compatible kernel, durable service or purchase claim."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    value.update(
        experiment_id=8497,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7308497,
        honest_verdict=f"complete_{value['verdict_class']}_v730_board_operation_boundary",
        future_dependencies=[
            dict(
                experiment_id=8494,
                status="optional_current_artifact"
                if work["cost"]["current"]["observed"] is not None
                else "future_dependency_not_observation",
            )
        ],
        current_authority_missing_reason=None
        if work["contract"]["activated"]
        else "independent_V730_design_contract_absent_or_mismatched",
    )
    value["rows"][2]["source_id"] = (
        "Exp8494" if work["cost"]["current"]["qualified"] else "Exp8466_historical_fallback"
    )
    value["field_principles"].update(
        {
            k: "Bind complete V730 authority and original producer hashes; no device or model work."
            for k in value
        }
    )
    value["field_principles"]["append_cost_boundary"] = (
        "Exp8466 clocks retain date20261013; missing Exp8494 readiness differs from measured zero."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild original and current operands so repaired summary hashes cannot grant readiness."""
    with binding():
        return bool(_REPLAY(path))
