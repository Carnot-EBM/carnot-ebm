"""REQ-REPORT-8400: current authority records obligations without repairing history.

The earlier reader already checks source custody and terminal bytes. This
adapter changes invocation identity and date while preserving those checks.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import signal
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_obligation_delta_8386 as old
from carnot.reporting import kv260_local_cost_boundary_8315 as base
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
ROOT, ACTIVE, DESIGN = old.ROOT, old.ACTIVE, old.DESIGN
NAME, TASK = "experiment_8400_v723_gatemate_continuity", "exp8400-gatemate-continuity"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_gatemate_continuity_8400.py"
NOTE = "docs/research-notes/v723-gatemate-obligation.md"
OWNED = [
    "python/carnot/reporting/gatemate_continuity_8400.py",
    "python/carnot/reporting/gatemate_continuity_runner_8400.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:db35abcbe859a09cc79c48bfec7dbd50aa21dee64c528036cdcf97c411ba643d"
UPSTREAM = "results/" + old.NAME + ".json"
PINS = dict(
    old.PINS,
    **{
        UPSTREAM: "sha256:2d6b45fb2f6f4bfea94e310e94dab6a098dc158185823476ad4f90ee918c2644",
        "openspec/change-proposals/v722-direct-service-protocol.json": "sha256:ab877c98112dc1a1497bb9aa1f2c28751e1b78671de743a6e4c036eeeddcc47e",
    },
)
ORIGINAL_BUILD, ORIGINAL_CONSUME = old.build, old.old.consume_receipt
gate, failure = old.gate, old.gate


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts because documentation work still needs visible boundaries."""
    print(f"[exp8400] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def budget(seconds: float) -> Iterator[None]:
    """Interrupt stalled reads while preserving the remaining outer invocation deadline."""

    def expire(signum: int, frame: Any) -> None:
        raise TimeoutError("invocation_deadline")

    start = time.monotonic_ns()
    handler = signal.getsignal(signal.SIGALRM)
    previous = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, previous[0]) if previous[0] else seconds)
    try:
        yield
    finally:
        remaining = (
            max(0.001, previous[0] - (time.monotonic_ns() - start) / 1e9) if previous[0] else 0
        )
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, remaining, previous[1])


def current_design(text: str, *, milestone: str) -> tuple[Any, Any]:
    """Select this invocation's full design without changing the frozen reader."""
    table, tasks = parse_design(text, milestone="2026.10.723")
    return table, tasks


@contextmanager
def adapter() -> Iterator[None]:
    """Limit inherited identity changes to this process and restore all historical exports."""
    with (
        patch.multiple(
            old,
            NAME=NAME,
            TASK=TASK,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            TASK_PIN=TASK_PIN,
            NOTE=NOTE,
            PINS=PINS,
            progress=progress,
            parse_design=current_design,
            scan=scan,
            build=build,
        ),
        patch.object(old.old, "consume_receipt", consume_receipt),
    ):
        yield


def consume_receipt(path: Path, raw: Path, missing: list[Json]) -> list[Json]:
    """Keep original dates and bytes while sharing the existing exact-source checks.

    The historical consumer accepts its own date only. A private normalized
    copy lets it check sources; returned physical evidence retains the real date.
    """
    refs: list[Json] = []
    try:
        value = json.loads(base.pin(path, raw, refs).read_bytes())
        physical = value.get("physical_change")
        if not "20261009" < value["received_date"] <= "20261011":
            raise ValueError("receipt_current_date")
        if physical is not None and (
            not "20261009" < physical["date"] <= "20261011"
            or not set(physical["changed_fields"]) & {"cable", "port", "power"}
        ):
            raise ValueError("physical_current_date_or_change")
        normalized = deepcopy(value)
        normalized["received_date"] = "20261010"
        if physical is not None:
            normalized["physical_change"]["date"] = "20261010"
        copy = raw / (canonical_hash(normalized)[7:] + "-consumer.json")
        atomic_json(copy, normalized)
        rows: list[Json] = ORIGINAL_CONSUME(copy, raw, missing)
        for row in rows:
            row["receipt_reference"] = refs[0]
            if row["disposition"] == "queued_next_hardware_task":
                row["physical_change"] = physical
        return rows
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [dict(disposition="rejected", reason=str(error), references=refs)]


def scan(paths: list[Path], raw: Path) -> list[Json]:
    """Read only explicit receipts; an empty list never triggers a device or source search."""
    with budget(60):
        return scan_rows(paths, raw)


def scan_rows(paths: list[Path], raw: Path) -> list[Json]:
    """Keep rejection rows intact when a supplied receipt fails authentication."""
    start, rows = time.monotonic(), []
    for index, path in enumerate(paths):
        progress("delta_receipt", index, len(paths) - index)
        if time.monotonic() - start > 60:
            raise TimeoutError("delta_scan_deadline")
        refs: list[Json] = []
        try:
            value = json.loads(base.pin(path, raw, refs).read_bytes())
            if (
                value.get("exp8372_primary_sha256") != old.HISTORY_PIN
                or type(value.get("received_wall_ns")) is not int
                or not old.CUTOFF_NS < value["received_wall_ns"] <= time.time_ns()
            ):
                raise ValueError("receipt_frontier")
            imported = consume_receipt(base.checked(refs[0]), raw, old.old.MISSING)
            if not imported or any(r["disposition"] == "rejected" for r in imported):
                raise ValueError("supplied_receipt_contract")
            rows.append(
                dict(
                    path=str(path),
                    authenticated=True,
                    reference=refs[0],
                    imported=imported,
                    reason=None,
                    could_reopen=[
                        "physical_change"
                        if r["disposition"] == "queued_next_hardware_task"
                        else "source_custody:" + r["original_path"]
                        for r in imported
                    ],
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            rows.append(
                dict(
                    path=str(path),
                    authenticated=False,
                    reference=refs[0] if refs else None,
                    imported=[],
                    reason=str(error),
                    could_reopen=[],
                )
            )
    progress("delta_scan_complete", len(rows), 0)
    return rows


def measure(root: Path, raw: Path, supplied: list[Path] | None = None) -> Json:
    """Seal current authority, all imported protocol bytes and the inherited source reader."""
    with adapter():
        work: Json = old.measure(root, raw, supplied)
    for name in [
        "python/carnot/reporting/gatemate_obligation_delta_8386.py",
        "python/carnot/reporting/gatemate_obligation_runner_8386.py",
        "python/carnot/reporting/roadmap_contract.py",
    ]:
        base.pin(ROOT / name, raw / "code", work["code_config_hashes"])
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Recompute continuity from sealed history; no documentation score permits device execution."""
    with adapter():
        value: Json = ORIGINAL_BUILD(work, receipts, raw, output)
    upstream_ref = next(
        (
            r
            for r in value["source_artifact_hashes"]
            if r["original_path"] == str(Path(work["root"]) / UPSTREAM)
        ),
        None,
    )
    upstream: Json = json.loads(base.checked(upstream_ref).read_bytes()) if upstream_ref else {}
    if upstream and (
        upstream.get("exact_missing_hashes") != old.old.MISSING
        or upstream.get("next_evidence_conditions") != old.CONDITIONS
        or upstream.get("original_idcode") != "0xffffffff"
        or upstream.get("required_idcode") != "0x20000001"
        or upstream.get("device_command_count") != 0
    ):
        raise ValueError("upstream_obligation_contract")
    contract = all(
        c["passed"]
        for c in value["gate_check_summary"]
        if c["check"] in {"full_task_sha256", "independent_design.full_task_sha256"}
    )
    recorded = bool(value["obligation_recorded"] and upstream and contract)
    row = value["rows"][0]
    row.update(
        completed=recorded,
        censored=not recorded,
        numerator=int(recorded),
        missing_reason=None if recorded else "sealed obligation or current authority unavailable",
    )
    value.update(
        experiment_id=8400,
        task_id=TASK,
        milestone="2026.10.723",
        run_date="20261011",
        honest_verdict="complete_" + value["verdict_class"] + "_gatemate_continuity",
        obligation_recorded=recorded,
        obligation_recorded_score=int(recorded and value["required_checks_passed"]),
        current_contract_ready_score=int(contract),
        evidence_delta_scan_limit_s=60,
        completed_count=sum(r["completed"] for r in value["rows"]),
        censored_count=sum(r["censored"] for r in value["rows"]),
        random_seed=7238400,
        repository_health=dict(
            scope="global",
            measured_in_current_invocation=False,
            upstream_path=UPSTREAM,
            upstream_sha256=PINS[UPSTREAM],
            receipts=[
                r for r in upstream.get("validation_receipts", []) if r.get("scope") == "global"
            ],
        ),
        methodology="Seal Exp8386 and Exp8372; independently bind the full V723 task; inspect only explicit operator receipts within 60 seconds; preserve two source and four physical prerequisites. No model, source recovery, device probe, flash or benchmark.",
    )
    value["acceptance_gates"].update(obligation_recorded=recorded, current_task_contract=contract)
    for ref in value["cited_upstream_artifacts"]:
        if ref["original_path"] == str(Path(work["root"]) / UPSTREAM):
            ref["imported_fields"] = [
                "exact_missing_hashes",
                "next_evidence_conditions",
                "original_idcode",
                "required_idcode",
                "device_command_count",
                "validation_receipts",
            ]
    value["field_principles"].update(
        repository_health="Retain previously measured global failures separately from scoped current validation.",
        current_contract_ready_score="The full active and independent task must match; this grants no hardware readiness.",
        obligation_recorded_score="Qualified documentation records the unchanged obligation and grants no scientific benefit.",
        evidence_delta_scan_limit_s="Bound operator-supplied receipt inspection to sixty seconds.",
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Use inherited byte, log and reduction checks with this invocation's pinned full task."""
    with adapter():
        return bool(old.replay(path))
