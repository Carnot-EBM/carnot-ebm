"""REQ-REPORT-8414: reuse sealed obligations without claiming a changed board.

The shipped reader already checks historical source bytes and physical receipts.
This adapter binds those checks to the complete current design and task.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import gatemate_continuity_8400 as old
from carnot.reporting import kv260_local_cost_boundary_8315 as base
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
ROOT, ACTIVE, DESIGN = old.ROOT, old.ACTIVE, old.DESIGN
NAME, TASK = "experiment_8414_v724_gatemate_obligation", "exp8414-gatemate-obligation"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_gatemate_obligation_8414.py"
NOTE = "docs/research-notes/v724-gatemate-obligation.md"
OWNED = [
    "python/carnot/reporting/gatemate_obligation_8414.py",
    "python/carnot/reporting/gatemate_obligation_runner_8414.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:3bdba1d737be8f70e9fb30507f476c01e7db7ab40c3eef0724326cf79c50c173"
TASKS_PIN = "sha256:82778c8abdbdbf2a69db3c7382e054515098e71056bb91d4521afe47792b2c34"
UPSTREAM = "results/" + old.NAME + ".json"
PINS = dict(
    old.PINS,
    **{
        UPSTREAM: "sha256:11ad83e04d615ba46d9ef61565ae761400519fea3e508a46f9db120c895589dd",
        "openspec/change-proposals/v723-service-and-label-protocol.json": "sha256:0ed6f87ede5046c2113257c9bff4dacfee88415fddb8d7be18506668f6301e90",
    },
)
ORIGINAL_BUILD = old.build
gate, failure, budget = old.gate, old.failure, old.budget


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush documentation boundaries so a bounded receipt reader remains visible."""
    print(f"[exp8414] phase={phase} completed={completed} pending={pending}", flush=True)


def current_design(text: str, *, milestone: str) -> tuple[Any, Any]:
    """Select current design bytes without changing any frozen date contract."""
    table, tasks = parse_design(text, milestone="2026.10.724")
    return table, tasks


@contextmanager
def adapter() -> Iterator[None]:
    """Restore historical exports after each call so old readers keep their identity."""
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        NOTE=NOTE,
        OWNED=OWNED,
        TASK_PIN=TASK_PIN,
        UPSTREAM=UPSTREAM,
        PINS=PINS,
        progress=progress,
        current_design=current_design,
        build=build,
    ):
        yield


def measure(root: Path, raw: Path, supplied: list[Path] | None = None) -> Json:
    """Seal explicit inputs once; no missing receipt triggers source or device searches."""
    with adapter():
        work: Json = old.measure(root, raw, supplied)
    for name in [
        "python/carnot/reporting/gatemate_continuity_8400.py",
        "python/carnot/reporting/gatemate_continuity_runner_8400.py",
    ]:
        base.pin(ROOT / name, raw / "code", work["code_config_hashes"])
    return work


def design_checks(work: Json) -> list[Json]:
    """Check every task because an unchanged local task can hide drift elsewhere."""
    checks = []
    for name in [ACTIVE, DESIGN]:
        observed = None
        ref = next(r["reference"] for r in work["inputs"] if r["name"] == name)
        try:
            data = base.checked(ref).read_bytes() if ref else b""
            tasks = (
                yaml.safe_load(data)["tasks"]
                if name == ACTIVE
                else current_design(data.decode(), milestone="2026.10.724")[1]
            )
            observed = canonical_hash(tasks)
        except (OSError, ValueError, KeyError, IndexError, TypeError, yaml.YAMLError):
            pass
        check = gate(Path(work["root"]) / name, "complete_design_tasks_sha256", TASKS_PIN, observed)
        check["hash"] = ref["sha256"] if ref else None
        checks.append(check)
    return checks


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Continuity credit requires full authority and never permits hardware execution."""
    with adapter():
        value: Json = ORIGINAL_BUILD(work, receipts, raw, output)
    checks = design_checks(work)
    contract = bool(value["current_contract_ready_score"] and all(c["passed"] for c in checks))
    recorded = bool(value["obligation_recorded"] and contract)
    value["rows"][0].update(
        completed=recorded,
        censored=not recorded,
        numerator=int(recorded),
        missing_reason=None
        if recorded
        else "sealed obligation or complete current authority unavailable",
    )
    value.update(
        experiment_id=8414,
        milestone="2026.10.724",
        honest_verdict="complete_" + value["verdict_class"] + "_gatemate_obligation",
        obligation_recorded=recorded,
        obligation_recorded_score=int(recorded and value["required_checks_passed"]),
        current_contract_ready_score=int(contract),
        canonical_tasks_sha256=checks[0]["observed"],
        completed_count=sum(r["completed"] for r in value["rows"]),
        censored_count=sum(r["censored"] for r in value["rows"]),
        random_seed=7248414,
        methodology="Seal Exp8400 and its exact missing-source and physical records; bind all fourteen V724 tasks; inspect only explicit receipts within60 seconds. No model, hardware probe, flash, source reconstruction or benchmark.",
    )
    value["gate_check_summary"].extend(checks)
    value["acceptance_gates"].update(obligation_recorded=recorded, current_task_contract=contract)
    value["field_principles"]["canonical_tasks_sha256"] = (
        "Bind all fourteen task objects; local task agreement cannot hide changed design authority."
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reuse sealed-byte and reduction checks with current full-design authentication."""
    with adapter():
        return bool(old.replay(path))
