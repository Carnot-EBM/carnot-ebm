"""REQ-REPORT-8441: carry sealed history without manufacturing a board result.

The earlier reader owns exact-source and physical-receipt checks. This narrow
adapter binds those checks to the complete activated V726 task authority.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_obligation_8414 as old
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import assess_authorities

Json = dict[str, Any]
ROOT, ACTIVE, DESIGN, base = old.ROOT, old.ACTIVE, old.DESIGN, old.base
NAME, TASK = "experiment_8441_v726_gatemate_obligation", "exp8441-gatemate-obligation"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_gatemate_obligation_8441.py"
NOTE = "docs/research-notes/v726-gatemate-obligation.md"
OWNED = [
    "python/carnot/reporting/gatemate_obligation_8441.py",
    "python/carnot/reporting/gatemate_obligation_runner_8441.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:258afa845a06190ce70d82ee3d023c34d70a3958fcc801f6ddcce4fa068fb6f5"
TASKS_PIN = "sha256:1c6d4e9588015b1359ce285551a7770b388036638f28e2bb9b5bfdb5482605de"
AUTHORITY_DIGEST = "72f4cca447651e88180fd3196f947615d653b04cc7280ff8f12fd8f5eaedaa88"
UPSTREAM = "results/" + old.NAME + ".json"
PINS = dict(
    old.PINS,
    **{
        UPSTREAM: "sha256:e572d8a1b6c31eaaf632338a6e55228c1696f652a44927afb4a151ba569d495a",
        "results/raw/"
        + old.NAME
        + "/terminal_validation.json": "sha256:818800c37950d2997f55254c9cc48ff2ec6c8acdafde45c9841a480ece85787e",
        "results/raw/"
        + old.NAME
        + "/validators/e572d8a1b6c31eaaf632338a6e55228c1696f652a44927afb4a151ba569d495a.json": "sha256:32c7059bfb2b9fa33c3cebdafb0faf8d4490310e5e9861c33598faaa22a927cb",
    },
)
ORIGINAL_BUILD, ORIGINAL_CHECKS = old.build, old.design_checks
gate, failure, budget = old.gate, old.failure, old.budget


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so bounded receipt work remains visible to its owner."""
    print(f"[exp8441] phase={phase} completed={completed} pending={pending}", flush=True)


def current_design(text: str, *, milestone: str) -> tuple[Any, Any]:
    """Select the present contract while leaving frozen historical dates intact."""
    table, tasks = parse_design(text, milestone="2026.10.726")
    return table, tasks


@contextmanager
def adapter() -> Iterator[None]:
    """Restore historical module exports after each call to preserve old readers."""
    with (
        patch.multiple(
            old,
            NAME=NAME,
            TASK=TASK,
            CLI=CLI,
            TEST=TEST,
            NOTE=NOTE,
            OWNED=OWNED,
            TASK_PIN=TASK_PIN,
            TASKS_PIN=TASKS_PIN,
            UPSTREAM=UPSTREAM,
            PINS=PINS,
            progress=progress,
            current_design=current_design,
            design_checks=design_checks,
            build=build,
        ),
        patch.object(old.old, "consume_receipt", consume_receipt),
    ):
        yield


def consume_receipt(path: Path, raw: Path, missing: list[Json]) -> list[Json]:
    """Extend the explicit receipt date only; exact bytes still use the old consumer.

    A private normalized copy lets the historical date guard check all other
    operands. Returned evidence always retains its actual receipt and date.
    """
    refs: list[Json] = []
    try:
        value = json.loads(base.pin(path, raw, refs).read_bytes())
        physical = value.get("physical_change")
        if not "20261009" < value["received_date"] <= "20261012":
            raise ValueError("receipt_current_date")
        if physical is not None and (
            not "20261009" < physical["date"] <= "20261012"
            or not set(physical["changed_fields"]) & {"cable", "port", "power"}
        ):
            raise ValueError("physical_current_date_or_change")
        normalized = deepcopy(value)
        normalized["received_date"] = "20261010"
        if physical is not None:
            normalized["physical_change"]["date"] = "20261010"
        copy = raw / (canonical_hash(normalized)[7:] + "-consumer.json")
        atomic_json(copy, normalized)
        rows: list[Json] = old.old.ORIGINAL_CONSUME(copy, raw, missing)
        for row in rows:
            row["receipt_reference"] = refs[0]
            if row["disposition"] == "queued_next_hardware_task":
                row["physical_change"] = physical
        return rows
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [dict(disposition="rejected", reason=str(error), references=refs)]


def measure(root: Path, raw: Path, supplied: list[Path] | None = None) -> Json:
    """Seal named dependencies once; no absent receipt triggers a device search."""
    with adapter():
        work: Json = old.measure(root, raw, supplied)
    for name in [
        "python/carnot/reporting/gatemate_obligation_8414.py",
        "python/carnot/reporting/gatemate_obligation_runner_8414.py",
        "python/carnot/reporting/v685_authority_lifecycle.py",
    ]:
        base.pin(ROOT / name, raw / "code", work["code_config_hashes"])
    return work


def design_checks(work: Json) -> list[Json]:
    """Authenticate the visible table and digest as well as every complete task."""
    checks = ORIGINAL_CHECKS(work)
    refs = {r["name"]: r["reference"] for r in work["inputs"]}
    observed = None
    try:
        with TemporaryDirectory(prefix="exp8441-authority-", dir="/var/tmp") as directory:
            private = Path(directory)
            result = assess_authorities(
                base.checked(refs[DESIGN]),
                private / "absent-staged.yaml",
                base.checked(refs[ACTIVE]),
                private / "snapshots",
                milestone="2026.10.726",
                first_id=8429,
                count=14,
            )
            observed = bool(
                result["activated"] and result["canonical_tasks_sha256"] == AUTHORITY_DIGEST
            )
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass
    check = gate(Path(work["root"]) / DESIGN, "visible_table_and_full_authority", True, observed)
    check["hash"] = refs[DESIGN]["sha256"] if refs[DESIGN] else None
    return [*checks, check]


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reduce sealed rows; continuity qualification cannot grant hardware readiness."""
    with adapter():
        value: Json = ORIGINAL_BUILD(work, receipts, raw, output)
    obligations = []
    missing = {r["path"]: r["sha256"] for r in value["exact_missing_hashes"]}
    for row in value["rows"]:
        hashes = [PINS[UPSTREAM]]
        if row["source_id"] in missing:
            hashes.append(missing[row["source_id"]])
        obligations.append(
            dict(
                row,
                change_status="new"
                if row["completed"] and row["source_id"] != "continuity_record"
                else "unchanged",
                source_hashes=hashes,
            )
        )
    value.update(
        experiment_id=8441,
        milestone="2026.10.726",
        run_date="20261012",
        honest_verdict="complete_" + value["verdict_class"] + "_gatemate_obligation",
        random_seed=7268441,
        obligation_rows=obligations,
        gatemate_obligation_recorded_score=value["obligation_recorded_score"],
        missing_source_hashes=value["exact_missing_hashes"],
        physical_change_reference=value["physical_change_receipt"],
        gatemate_bitstream_flashed=False,
        last_authenticated_idcode=value["original_idcode"],
        device_commands_executed=[],
        reopening_condition=value["next_evidence_conditions"],
        methodology="Bind all fourteen activated V726 tasks and the visible design; carry qualified Exp8414 immutable evidence and exact source hashes. Inspect only explicit receipts within60 seconds. Zero current model or device commands. Changed setup queues separate IDCODE, n16 flash and device sample/hash smoke work.",
    )
    value["field_principles"].update(
        {
            "obligation_rows": "One row per original obligation; unchanged/new status and source hashes remain explicit.",
            "gatemate_obligation_recorded_score": "Qualified continuity records the external obligation and grants no hardware readiness.",
            "missing_source_hashes": "Exact original hashes are required; replacement source cannot restore historical custody.",
            "physical_change_reference": "Authentic dated setup evidence queues only a separate physical task.",
            "gatemate_bitstream_flashed": "No flash is authorized or performed in this invocation.",
            "last_authenticated_idcode": "Import the original failure observation; no new IDCODE is measured.",
            "device_commands_executed": "An empty list records zero physical commands.",
            "reopening_condition": "Retain ordered physical change, IDCODE, flash and actual sample/hash requirements.",
        }
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Cold readers rederive primitive rows and reject changed meaning after rehashing."""
    with adapter():
        return bool(old.replay(path))
