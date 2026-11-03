"""REQ-REPORT-8591: freeze current authority without changing scientific operands.

Registration establishes which work may proceed. It cannot establish learning
benefit, and optional resources cannot suppress independent CPU evidence.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v737_contract_methods as previous
from carnot.reporting import v736_contract_methods as base
from carnot.reporting import v736_historical_inputs as readers
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v709_execution import child as child

Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8591_v738_contract_methods",
    "exp8591-contract-methods",
    "2026.10.738",
)
DESIGN, ACTIVE, STAGED = base.DESIGN, base.ACTIVE, base.STAGED
PROTOCOL = "openspec/change-proposals/v738-methods-protocol.json"
PROTOCOL_PIN = "sha256:a0b1532e18a3910a94d90e79d500aadfb67dc4ac77054058c07d804e7dbb21da"
TASKS_PIN = "7912e311c832df51ec3654c819976cb7558b1d26294f83717d5286ae774ca301"
INHERITED = [*base.INHERITED, "resident_session"]
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [f"tests/python/test_v738_contract_{s}_8591.py" for s in ["methods", "runner"]]
TEST = TESTS[0]
OWNED = [
    f"python/carnot/reporting/v738_contract_{s}.py"
    for s in ["methods", "inputs", "report", "runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
reference, read, tasks_digest = base.reference, base.read, base.tasks_digest
checksum, atomic_json, sha256_file = base.checksum, base.atomic_json, base.sha256_file
checked, gate = readers.checked, base.gate


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so the parent can distinguish progress from a stall."""
    if phase in {
        "preconditions_before",
        "authority_before",
        "authority_after",
        "measurement_frozen",
    }:
        completed = min(completed, 11)
        pending = 11 - completed
    print(f"[exp8591] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Independent byte pins prevent a repaired envelope from changing inherited science."""
    p = read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN))
    inherited = read(p["inherited_v737_protocol_reference"])
    checks = {
        k: dict(source="V737", value_equal=p[k] == inherited[k], sha256=canonical_hash(p[k]))
        for k in INHERITED
    }
    refs = [
        p[k]
        for k in [
            "inherited_v737_protocol_reference",
            "v738_activation_reference",
            "v738_design_reference",
            "literature_reference",
            "methods_reference",
            "conductor_reference",
        ]
    ]
    refs += [r["source_reference"] for r in p["literature_method_rows"]]
    if (
        p["inherited_v737_protocol_reference"]["sha256"] != previous.PROTOCOL_PIN
        or not all(r["value_equal"] for r in checks.values())
        or checks != p["inherited_operand_checks"]
        or tasks_digest(p["task_map_v738"]) != TASKS_PIN
        or not all(checked(r) for r in refs)
    ):
        raise ValueError("sealed_scientific_or_registration_operand")
    return dict(p)


def authority(root: Path) -> Json:
    """Use staging or real V738 activation; an older live queue has no current authority."""
    p = protocol()
    refs, failures = [], []
    try:
        text = (root / DESIGN).read_text()
        stage, active = root / STAGED, root / ACTIVE
        chosen = stage if stage.is_file() else active
        planned = yaml.safe_load(chosen.read_bytes())
        actual = (
            yaml.safe_load(active.read_bytes()) if stage.is_file() and active.is_file() else planned
        )
        if actual.get("milestone") != MILESTONE and stage.is_file():
            actual = planned
        result = compare_contract(
            text, planned, actual, milestone=MILESTONE, first_id=8591, count=11
        )
        machine = parse_design(text, milestone=MILESTONE)[1]
        for i, row in enumerate(result["rows"]):
            row["checks"]["full_dictionary"] = (
                i < len(planned["tasks"])
                and planned["tasks"][i] == p["task_map_v738"][i] == machine[i] == actual["tasks"][i]
            )
            row["matched"] = all(row["checks"].values())
        ready = bool(
            result["passed"]
            and all(r["matched"] for r in result["rows"])
            and re.search(
                r"(?:canonical_tasks_sha256:|Canonical task SHA256:)\s*`?" + TASKS_PIN, text
            )
            and "<!-- V738_TASK_CONTRACT_END -->" in text
            and tasks_digest(planned["tasks"]) == TASKS_PIN
        )
        refs = [reference(root / DESIGN), reference(chosen)]
        rows = result["rows"]
        if not ready:
            failures.append(gate(root / DESIGN, "eleven_complete_objects", True, False, TASK))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        ready = False
        failures.append(
            gate(root / DESIGN, "v738_design_and_authority", MILESTONE, str(error), TASK)
        )
        rows = [
            dict(
                unit_id=t["id"],
                order=i + 1,
                matched=False,
                status="unstarted",
                checks={},
                censored=True,
                excluded=False,
            )
            for i, t in enumerate(p["task_map_v738"])
        ]
    for row in rows:
        row.update(
            absolute_metric=int(row["matched"]),
            effective_independent_groups=0,
            missing_reason=None if row["matched"] else "authority_absent_or_mismatched",
        )
    return dict(
        ready=ready,
        rows=rows,
        gate_check_summary=failures,
        references=refs,
        staging_present=(root / STAGED).is_file(),
    )


def historical(p: Json) -> Json:
    """Import only the affected original families, keeping unrelated CPU gates separate."""
    from carnot.reporting.v738_contract_inputs import historical as inputs

    return inputs(p)


def runtime(root: Path, raw: Path | None = None) -> Json:
    """The existing receipt reader owns a 60-second child and never opens a device."""
    return dict(previous.runtime(root, raw))


def measure(root: Path, raw: Path, run_date: str) -> Json:
    """Reuse resource checks, then freeze this invocation and its optional revision controls."""
    from carnot.reporting import v738_contract_inputs as inputs

    snapshot = raw / "authority"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in [DESIGN, ACTIVE, STAGED]:
        source, target = root / name, snapshot / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    with (
        patch.multiple(
            base,
            NAME=NAME,
            TASK=TASK,
            MILESTONE=MILESTONE,
            PROTOCOL=PROTOCOL,
            OWNED=OWNED,
            TESTS=TESTS,
            protocol=protocol,
            authority=authority,
            progress=progress,
        ),
        patch.object(readers, "historical", historical),
        patch.object(readers, "runtime", lambda ignored, logs: runtime(root, logs)),
    ):
        work = dict(base.measure(snapshot, raw))
    work["run_date"] = run_date
    binding = inputs.canary_descriptor(work["protocol"])
    checks = []
    for mode in ["valid", "drift"]:
        candidate = deepcopy(binding)
        if mode == "drift":
            candidate["current_validator_reference"]["sha256"] = "sha256:drift"
        path = raw / ("canary_" + mode + ".json")
        atomic_json(path, candidate)
        checks.append(
            child(
                "canary_" + mode,
                [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--canary-binding",
                    str(path),
                ],
                raw / "canary_logs",
                deadline=60,
                expected=int(mode == "drift"),
                scope="optional",
            )
        )
    binding["private_cli_receipts"] = checks
    binding["ready"] = all(r["passed"] for r in checks) and inputs.accept_canary(
        raw / "canary_valid.json"
    )
    work["canary_revision_binding"] = binding
    work["history"]["canary_inputs_ready_score"] = int(binding["ready"])
    work["end_ns"] = time.monotonic_ns()
    work["duration_s"] = (work["end_ns"] - work["start_ns"]) / 1e9
    work["phase_spans"][0].update(end_monotonic_ns=work["end_ns"], duration_s=work["duration_s"])
    atomic_json(raw / "measurement.json", work)
    progress("eleven_task_measurement_frozen", 11, 0)
    return work
