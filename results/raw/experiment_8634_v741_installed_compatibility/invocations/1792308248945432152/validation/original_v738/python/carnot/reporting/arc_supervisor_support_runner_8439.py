"""REQ-VERIFY-8439: private children validate exact terminal support bytes.

Owned commands keep deadlines, exits and stream hashes. Control mode avoids
recursive test invocation and writes only to private destinations.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import arc_supervisor_support_8439 as e
from carnot.reporting.arc_supervisor_support_evidence_8439 import read
from carnot.reporting import v717_contract_runner as shipped
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute
from carnot.reporting.v726_feature_panel_runner import scan

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Existing coverage and E2E assertions remain unchanged in private scratch."""
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    with patch.object(shipped, "m", e):
        specs = list(shipped.manifest(private))
    next(s for s in specs if s["name"] == "owned_tests")["deadline"] = 600
    next(s for s in specs if s["name"] == "coverage_combine")["argv"].append("--keep")
    specs.insert(
        2,
        dict(
            name="historical_wrapper_assertions",
            argv=[
                sys.executable,
                "-u",
                "-c",
                "from pathlib import Path; from carnot.reporting.arc_panel_qualification_8412 import historical_checks; import sys; sys.exit(historical_checks(Path(sys.argv[1])))",
                str(private / "historical-wrapper"),
            ],
            expected=0,
            deadline=240,
            scope="owned",
        ),
    )
    specs.insert(
        3,
        dict(
            name="private_live_wrapper_receipts",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_arc_decision_telemetry.py",
                "tests/python/test_arc_supervisor_delta_7874.py",
                "tests/python/test_arc_submitted_agent_parity.py",
                "--basetemp=" + str(private / "wrapper"),
            ],
            expected=0,
            deadline=240,
            scope="owned",
        ),
    )
    return specs


def controls(value: Json, raw: Path) -> list[Json]:
    """Cold children reject missing files, errors and rehashed aggregate claims."""
    path = raw / "cold_candidate.json"
    atomic_json(path, value)
    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipts = [
        child(name, [*cli, *args], raw / "controls", expected=expected, deadline=60)
        for name, args, expected in [
            ("cold", ["--cold-replay", str(path)], 0),
            ("missing", ["--cold-replay", str(raw / "absent")], 1),
            ("deliberate_error", ["--deliberate-error"], 7),
        ]
    ]
    for name, field, replacement in [
        ("rehashed_count", "completed_count", 999),
        ("changed_arm", "arm_refinement_authorized", True),
    ]:
        changed = deepcopy(value)
        changed[field] = replacement
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = raw / (name + ".json")
        atomic_json(path, changed)
        receipts.append(
            child(
                name, [*cli, "--cold-replay", str(path)], raw / "controls", expected=1, deadline=60
            )
        )
    changed = deepcopy(value)
    work = read(value["work_reference"])
    if work["rows"]:
        work["rows"][0]["action_count"] += 1
    else:
        work["authority"]["activated"] = not work["authority"]["activated"]
    operand = raw / "rehashed_measurement.json"
    atomic_json(operand, work)
    changed["work_reference"] = e.reference(operand)
    changed["reproducibility_checksum"] = e.checksum(changed)
    path = raw / "rehashed_primitive.json"
    atomic_json(path, changed)
    receipts.append(
        child(
            "rehashed_primitive",
            [*cli, "--cold-replay", str(path)],
            raw / "controls",
            expected=1,
            deadline=60,
        )
    )
    return receipts


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, supplied: Path | None = None
) -> int:
    """All owned qualification precedes atomic publication through shipped consumers."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    work = e.measure(root, raw, private, supplied)
    plan = (
        [
            dict(
                name="private_child",
                argv=[sys.executable, "-u", "-c", "print('actual private control',flush=True)"],
                expected=0,
                deadline=30,
                scope="owned",
            )
        ]
        if control
        else manifest(private)
    )
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            owned=e.OWNED,
            task_cap_s=4800,
            validation_reserve_s=900,
            child_poll_s=30,
            full_repository_suite="not_run",
            MODEL_SPECS=[],
        ),
    )
    e.progress("validation_before", 0, len(plan))
    receipts = execute(plan, raw / "validation")
    if (private / "coverage.json").is_file():
        shards = raw / "coverage"
        shards.mkdir()
        for path in private.glob(".coverage*"):
            shutil.copyfile(path, shards / path.name)
        shutil.copyfile(private / "coverage.json", shards / "coverage.json")
        work["coverage_reference"] = e.reference(shards / "coverage.json")
        work["raw_shard_hashes"].update(
            {str(path): e.reference(path)["sha256"] for path in shards.iterdir()}
        )
    atomic_json(raw / "measurement.json", work)
    e.progress("cold_controls_before")
    receipts.extend(controls(e.build(work, receipts, raw, output), raw))
    candidate = raw / "preterminal.json"
    atomic_json(candidate, e.build(work, receipts, raw, output))
    checks, findings = scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"] += [dict(phase=r["name"], duration_s=r["duration_s"]) for r in receipts]
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)

    def validator(path: Path) -> Json:
        """Final cold reconstruction binds scanner reports to the exact published hash."""
        cold = child(
            "terminal_cold",
            [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
            raw / "terminal_validators",
            deadline=60,
        )
        checks, flags = scan(path, raw / "terminal_validators")
        return dict(
            passed=cold["passed"] and all(c["passed"] for c in checks),
            findings=flags,
            receipts=[cold, *checks],
        )

    e.progress("terminal_validation_before")
    publication = publish_primary(output, value, validator)
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="arc_support_ready_score",
        expected=value["arc_support_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=consumer,
            primary_sha256=publication["primary_sha256"],
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    e.progress("finished", value["completed_count"], value["censored_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Date, deadline and private-output guards bind the task's actual invocation."""
    e.progress("start_no_model_load_LLM_loads_0_generations_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261012"], default="20261012")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8439-owned-", dir="/var/tmp") as directory:
            return run(
                args.root,
                output,
                Path(directory),
                control=args.private_control,
                supplied=args.frontier,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
