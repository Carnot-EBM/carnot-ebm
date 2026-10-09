"""REQ-VERIFY-8328: reuse bounded children and the unchanged primary publisher.

Execution checks have their own frozen manifest. They cannot change the science
cutoff or create a live outcome. Private CLI controls receive no readiness credit.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time
from types import FunctionType
from typing import Any, cast

from carnot.reporting import arc_supervisor_artifact_8328 as a
from carnot.reporting import arc_supervisor_frontier_8328 as m
from carnot.reporting import v717_contract_runner as qualified
from carnot.reporting.arc_supervisor_v689_delta import operand
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child, execute
from carnot.reporting.v718_replay_runner import audit

Json = dict[str, Any]
NAME, CLI, OUTPUT, TEST, OWNED = a.NAME, a.CLI, a.OUTPUT, a.TEST, a.OWNED
build, replay = a.build, a.replay


def plan(private: Path) -> list[Json]:
    """The existing coverage framework already measures actual CLI subprocesses."""
    bound = FunctionType(qualified.manifest.__code__, dict(vars(qualified), m=a))
    specs = cast(list[Json], bound(private))
    specs[0]["deadline"] = 240
    pytest = str(m.ROOT / ".venv/bin/pytest")
    for name, files in [
        ("private_E2E017", ["tests/python/test_arc_supervisor_delta_7874.py"]),
        (
            "qualified_consumers",
            [
                "tests/python/test_arc_supervisor_frontier_8243.py",
                "tests/python/test_arc_coverage_frontier_8314.py::test_cell_floor",
                "tests/python/test_v718_contract_replay_8318.py::test_finding_policy",
            ],
        ),
    ]:
        specs.append(
            dict(
                name=name,
                argv=[
                    pytest,
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    *files,
                    "--basetemp=" + str(private / name),
                ],
                expected=0,
                deadline=180,
                scope="owned",
            )
        )
    return specs


def controls(value: Json, raw: Path) -> list[Json]:
    """Cold children reject altered claims even when an attacker updates a hash."""
    rows = []
    for name in ["valid", "negative", "rehashed_tamper"]:
        changed = deepcopy(value)
        if name == "negative":
            changed["solve_claims"] = ["invented"]
        if name == "rehashed_tamper":
            work = m.json_document(Path(changed["work_reference"]["path"]))
            work["delta"]["new_outcome_count"] += 1
            altered = raw / "tampered_measurement.json"
            atomic_json(altered, work)
            changed["work_reference"] = dict(path=str(altered), sha256=sha256_file(altered))
            changed["reproducibility_checksum"] = canonical_hash(work)
        path = raw / (name + ".json")
        atomic_json(path, changed)
        rows.append(
            child(
                "cold_" + name,
                [sys.executable, "-u", str(m.ROOT / CLI), "--cold-replay", str(path)],
                raw / "controls",
                expected=int(name != "valid"),
                deadline=60,
            )
        )
    return rows


def publish(value: Json, work: Json, output: Path, raw: Path) -> None:
    """Clear readiness after failed validation and retain all byte-bound findings."""
    attempts: list[Json] = []

    def validate(candidate: Path) -> Json:
        logs = raw / "terminal" / str(len(attempts))
        cold = child(
            "terminal_cold",
            [sys.executable, "-u", str(m.ROOT / CLI), "--cold-replay", str(candidate)],
            logs,
            deadline=60,
        )
        found = audit(candidate, logs, {})
        rows = child(
            "strict_rows",
            [
                sys.executable,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ],
            logs,
            deadline=60,
        )
        recorded_failure = (
            value["verdict_class"] == "disqualified"
            and found["findings"]
            and value["flagged_adversarial"]
        )
        passed = bool(cold["passed"] and rows["passed"] and (found["passed"] or recorded_failure))
        report = dict(passed=passed, checks=[cold, found["receipt"], rows], adversarial=found)
        attempts.append(report)
        return report

    try:
        publication = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        atomic_json(raw / "rejected_candidate.json", value)
        work["audits"] = [attempts[0]["adversarial"]]
        value = build(work, value["validation_receipts"] + attempts[0]["checks"], raw, output)
        publication = publish_primary(output, value, validate)
    atomic_json(
        output.parent / "raw" / output.stem / "terminal_validation.json",
        dict(
            publication=publication,
            attempts=attempts,
            checks=attempts[-1]["checks"],
            passed=attempts[-1]["passed"],
        ),
    )


def run(output: Path, private: Path, *, fixture: bool = False) -> int:
    """Qualify current code, then inspect external observations once without reruns."""
    start = time.monotonic_ns()
    raw = output.parent / "raw" / output.stem / "invocations" / str(start)
    raw.mkdir(parents=True, exist_ok=True)
    m.progress("execution_preflight")
    specs = [] if fixture else plan(private)
    missing = [
        str(m.ROOT / ".venv/bin" / name)
        for name in ["python", "pytest", "ruff", "mypy", "coverage"]
        if not (m.ROOT / ".venv/bin" / name).is_file()
    ]
    atomic_json(
        raw / "command_manifest.json",
        dict(commands=specs, heartbeat_s=30, MODEL_SPECS=[], sample_size_budget=a.BUDGET),
    )
    receipts = execute(specs, raw / "checks") if not missing else []
    failures = [
        dict(
            operand(Path(p), "executable_available", True, None, None),
            passed=False,
        )
        for p in missing
    ]
    work = m.measure(raw, private, failures)
    work["fixture_claim_scope"] = fixture
    report = private / "coverage.json"
    if report.is_file():
        saved = raw / "coverage.json"
        shutil.copyfile(report, saved)
        work["snapshots"][str(saved)] = sha256_file(saved)
        work["coverage_statement_counts"] = m.json_document(saved).get("files", {})
    database = private / ".coverage"
    if database.is_file():
        saved = raw / "coverage.sqlite"
        shutil.copyfile(database, saved)
        work["snapshots"][str(saved)] = sha256_file(saved)
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    value = build(work, receipts, raw, output)
    receipts += controls(value, raw)
    value = build(work, receipts, raw, output)
    m.progress("publication_before")
    publish(value, work, output, raw)
    m.progress("publication_after")
    published = m.json_document(output)
    return int(published["verdict_class"] == "disqualified")
