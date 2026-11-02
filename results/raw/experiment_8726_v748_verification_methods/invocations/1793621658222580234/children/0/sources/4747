"""REQ-REPORT-8455: retain exact recovery operands without a physical experiment.

The qualified reader checks source and physical receipts. This adapter binds
current authority separately, so missing V727 design cannot rewrite V726 history.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import gatemate_obligation_8441 as old
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import assess_authorities

Json = dict[str, Any]
ROOT, ACTIVE, DESIGN, base = old.ROOT, old.ACTIVE, old.DESIGN, old.base
NAME, TASK = "experiment_8455_v727_gatemate_obligation", "exp8455-gatemate-obligation"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_gatemate_obligation_8455.py"
NOTE = "docs/research-notes/v727-gatemate-obligation.md"
OWNED = [
    "python/carnot/reporting/gatemate_obligation_8455.py",
    "python/carnot/reporting/gatemate_obligation_runner_8455.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:9b2b472df76b3063deee2b8cdcb4460dd7e7f68ea2226defd355eada8948d127"
TASKS_PIN = "sha256:026228ee20a3160ad55ba35fe8286fab6db3a14d62a89b8d3d3d171e1acb1aa8"
AUTHORITY_DIGEST = TASKS_PIN[7:]
UPSTREAM = f"results/{old.NAME}.json"
UPSTREAM_PIN = "sha256:2c9f2e6b573db75a5db9466668b88f30f89a8f6be1a4765dd2a802be076759d1"
PINS = dict(
    old.PINS,
    **{
        UPSTREAM: UPSTREAM_PIN,
        f"results/raw/{old.NAME}/terminal_validation.json": "sha256:722b2df2c563e98084ddd883ec61fa5e44ffd8dbbf0185949381d8a88e03818b",
        f"results/raw/{old.NAME}/validators/{UPSTREAM_PIN[7:]}.json": "sha256:07a08c8b676c209b66258aa28f01f335b3094e8a845ad9fa29c55ea2c3d880fb",
    },
)
EXPORTS = [
    "NAME",
    "TASK",
    "CLI",
    "TEST",
    "NOTE",
    "OWNED",
    "TASK_PIN",
    "TASKS_PIN",
    "AUTHORITY_DIGEST",
    "UPSTREAM",
    "PINS",
    "progress",
    "current_design",
    "design_checks",
    "build",
]
HISTORICAL_EXPORTS = {name: getattr(old, name) for name in EXPORTS}
ORIGINAL_BUILD = old.build
gate, failure, budget = old.gate, old.failure, old.budget


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so bounded receipt work stays visible to its owner."""
    print(f"[exp8455] phase={phase} completed={completed} pending={pending}", flush=True)


def current_design(text: str, *, milestone: str) -> tuple[Any, Any]:
    """Select current full objects without changing the historical date contract."""
    table, tasks = parse_design(text, milestone="2026.10.727")
    return table, tasks


@contextmanager
def adapter() -> Iterator[None]:
    """Restore every imported export so historical readers retain their own identity."""
    with patch.multiple(old, **{name: globals()[name] for name in EXPORTS}):
        yield


def history(work: Json) -> Json:
    """Replay exact original producer bytes under its sealed original authority."""
    ref = next(r["reference"] for r in work["inputs"] if r["name"] == UPSTREAM)
    passed = False
    try:
        with patch.multiple(old, **HISTORICAL_EXPORTS):
            passed = bool(ref and ref["sha256"] == UPSTREAM_PIN and old.replay(base.checked(ref)))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    check = gate(
        Path(work["root"]) / UPSTREAM,
        "original_producer_sealed_replay",
        True,
        passed if ref else None,
    )
    check["hash"] = ref["sha256"] if ref else None
    return dict(passed=passed, check=check, reference=ref)


def measure(root: Path, raw: Path, supplied: list[Path] | None = None) -> Json:
    """Seal only named operands; absent receipts never trigger a history or device search."""
    with budget(3600), adapter():
        work: Json = old.measure(root, raw, supplied)
    work["original_history"] = history(work)
    work["historical_authority_references"] = []
    if work["original_history"]["passed"]:
        producer = json.loads(base.checked(work["original_history"]["reference"]).read_bytes())
        base.pin(
            base.checked(producer["work_reference"]),
            raw / "historical",
            work["historical_authority_references"],
        )
        for ref in producer["source_artifact_hashes"]:
            if Path(ref["original_path"]).relative_to(ROOT).as_posix() in {ACTIVE, DESIGN}:
                base.pin(
                    base.checked(ref), raw / "historical", work["historical_authority_references"]
                )
    for name in [
        "python/carnot/reporting/gatemate_obligation_8441.py",
        "python/carnot/reporting/gatemate_obligation_runner_8441.py",
    ]:
        base.pin(ROOT / name, raw / "code", work["code_config_hashes"])
    return work


def design_checks(work: Json) -> list[Json]:
    """Full-object equality and the visible table each need independent authentication."""
    refs = {r["name"]: r["reference"] for r in work["inputs"]}
    checks = []
    for name in [ACTIVE, DESIGN]:
        observed = None
        try:
            data = base.checked(refs[name]).read_bytes() if refs[name] else b""
            tasks = (
                yaml.safe_load(data)["tasks"]
                if name == ACTIVE
                else current_design(data.decode(), milestone="2026.10.727")[1]
            )
            observed = canonical_hash(tasks)
        except (OSError, ValueError, KeyError, IndexError, TypeError, yaml.YAMLError):
            pass
        check = gate(Path(work["root"]) / name, "complete_design_tasks_sha256", TASKS_PIN, observed)
        check["hash"] = refs[name]["sha256"] if refs[name] else None
        checks.append(check)
    observed = None
    try:
        with TemporaryDirectory(prefix="exp8455-authority-", dir="/var/tmp") as directory:
            private = Path(directory)
            result = assess_authorities(
                base.checked(refs[DESIGN]),
                private / "absent.yaml",
                base.checked(refs[ACTIVE]),
                private / "snapshots",
                milestone="2026.10.727",
                first_id=8443,
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
    """Continuity qualification never turns an absent physical operand into progress."""
    original = history(work)
    if original != work["original_history"]:
        raise ValueError("original_history_reduction")
    with adapter():
        value: Json = ORIGINAL_BUILD(work, receipts, raw, output)
    recorded = bool(original["passed"] and value["obligation_recorded"])
    value["rows"][0].update(
        completed=recorded,
        censored=not recorded,
        numerator=int(recorded),
        missing_reason=None
        if recorded
        else "sealed obligation or complete current authority unavailable",
    )
    for row in value["obligation_rows"]:
        if row["source_id"] == "continuity_record":
            row.update(value["rows"][0])
        key = (
            "physical_change"
            if row["source_id"] == "physical_change"
            else "source_custody:" + row["source_id"]
        )
        row["supplied_delta_references"] = [
            r["reference"]
            for r in work["supplied"]
            if r["authenticated"] and key in r["could_reopen"]
        ]
    value.update(
        experiment_id=8455,
        milestone="2026.10.727",
        run_date="20261012",
        honest_verdict="complete_" + value["verdict_class"] + "_gatemate_obligation",
        random_seed=7278455,
        original_producer_replay_passed=original["passed"],
        historical_authority_references=work["historical_authority_references"],
        obligation_recorded=recorded,
        obligation_recorded_score=int(recorded and value["required_checks_passed"]),
        gatemate_obligation_recorded_score=int(recorded and value["required_checks_passed"]),
        completed_count=sum(r["completed"] for r in value["rows"]),
        censored_count=sum(r["censored"] for r in value["rows"]),
        reopening_condition=[
            dict(id="source_authority", expected=value["missing_source_hashes"], requires=[]),
            *value["next_evidence_conditions"],
        ],
        methodology="Authenticate activated V727 full objects independently; replay pinned Exp8441 under original sealed V726 authority. Carry exact missing source hashes and last IDCODE0xffffffff. Inspect only explicit receipts within60 seconds. Zero current model calls and device commands; source recovery, dated physical change, GM1Ax IDCODE0x20000001, n16 flash and actual sample/hash smoke remain ordered reopening obligations.",
    )
    value["gate_check_summary"].append(original["check"])
    value["source_artifact_hashes"].extend(work["historical_authority_references"])
    value["acceptance_gates"].update(
        original_producer_sealed_replay=original["passed"], obligation_recorded=recorded
    )
    for ref in value["cited_upstream_artifacts"]:
        if ref["original_path"] == str(Path(work["root"]) / UPSTREAM):
            ref["imported_fields"] = [
                "missing_source_hashes",
                "reopening_condition",
                "last_authenticated_idcode",
                "historical_model_provenance",
                "work_reference",
                "source_artifact_hashes",
            ]
    value["field_principles"].update(
        original_producer_replay_passed="Exact producer bytes replay under original sealed authority; current vNEXT never replaces historical operands.",
        historical_authority_references="Preserve original V726 work, active YAML and independent full design bytes in current immutable custody.",
        reopening_condition="Recover exact source authority before dated physical change, required IDCODE, n16 flash and actual sample/hash smoke.",
        obligation_rows="Each original obligation retains unchanged/new status, original hashes and authenticated supplied-delta references.",
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Fresh readers reconstruct primitive meaning even when a tamper is rehashed."""
    with adapter():
        return bool(old.replay(path))
