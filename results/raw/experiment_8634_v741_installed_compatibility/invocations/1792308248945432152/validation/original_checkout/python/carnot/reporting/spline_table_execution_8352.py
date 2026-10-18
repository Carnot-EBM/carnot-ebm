"""REQ-REPORT-8352: bounded validators own the final atomic publication decision."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import os
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import spline_table_fidelity_8352 as e
from carnot.reporting import v717_contract_runner as commands
from carnot.reporting import v718_replay_runner as findings
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child, execute
from carnot.verify.spline_table_fidelity_8352 import progress

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Reuse scoped statement coverage and consumers without a repository-wide suite."""
    with patch.object(commands, "m", e):
        plan = commands.manifest(private)
    plan[0]["deadline"] = 600
    plan[1]["argv"] += ["tests/python/test_local_update_isolation_8306.py"]
    plan[1]["deadline"] = 300
    return list(plan)


def controls(value: Json, raw: Path) -> list[Json]:
    """Run fresh processes for valid, invalid and repaired-hash semantic mutations."""
    receipts = []
    for name in ["valid", "negative", "rehashed_tamper"]:
        changed = deepcopy(value)
        if name == "negative":
            changed["reproducibility_checksum"] = "invalid"
        if name == "rehashed_tamper":
            changed["table_candidate_score"] = 1 - changed["table_candidate_score"]
            changed.pop("reproducibility_checksum")
            changed["reproducibility_checksum"] = canonical_hash(changed)
        path = raw / (name + ".json")
        atomic_json(path, changed)
        receipts.append(
            child(
                "cold_" + name,
                [
                    str(e.ROOT / ".venv/bin/python"),
                    "-u",
                    str(e.ROOT / e.CLI),
                    "--cold-replay",
                    str(path),
                ],
                raw / "cold",
                expected=int(name != "valid"),
                deadline=120,
                heartbeat=20,
            )
        )
    return receipts


def terminal_checks(candidate: Path, logs: Path) -> Json:
    """Unchanged typed findings reject warnings, unknown severity and unresolved errors."""
    cold = child(
        "terminal_cold",
        [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            str(e.ROOT / e.CLI),
            "--cold-replay",
            str(candidate),
        ],
        logs,
        deadline=120,
        heartbeat=20,
    )
    found = findings.audit(candidate, logs, {})
    rows = child(
        "strict_rows",
        [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            "scripts/verdict_row_consistency_lint.py",
            "--strict",
            str(candidate),
        ],
        logs,
        deadline=60,
        heartbeat=20,
    )
    return dict(
        passed=all(r["passed"] for r in [cold, found["receipt"], rows]),
        checks=[cold, found["receipt"], rows],
        adversarial=found,
    )


def publish(value: Json, output: Path, raw: Path) -> None:
    """Preserve rejected candidates before publishing a checked owned failure."""
    attempts: list[Json] = []

    def validate(candidate: Path) -> Json:
        report = terminal_checks(candidate, raw / "terminal" / str(len(attempts)))
        attempts.append(report)
        return report

    try:
        publication = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        atomic_json(raw / "rejected_candidate.json", value)
        work = json.loads((raw / "measurement.json").read_bytes())
        work["adversarial_findings"] = attempts[0]["adversarial"]["findings"]
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, value["validation_receipts"] + attempts[0]["checks"], raw, output)
        publication = publish_primary(output, value, validate)
    atomic_json(
        output.parent / "raw" / output.stem / "terminal_validation.json",
        dict(
            publication=publication,
            attempts=attempts,
            checks=attempts[-1]["checks"],
            normal_process_completion=True,
        ),
    )
    progress("published", 1, 0)


def main(argv: list[str] | None = None) -> int:
    """Check source authority and private resources before any numerical benchmark."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261009"], default="20261009")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    began = time.monotonic()
    output = args.output.absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    with TemporaryDirectory(prefix="exp8352-owned-") as directory:
        private = Path(directory)
        probe = private / "probe"
        probe.write_bytes(b"private")
        plan = manifest(private)
        atomic_json(
            raw / "execution_manifest.json",
            dict(
                commands=plan,
                owned=e.OWNED,
                scientific_protocol_sha256=e.PINS[e.PROTOCOL],
                private_scratch=str(private),
                heartbeat_s=20,
                no_full_repository_suite=True,
            ),
        )
        work: Json = dict(
            failures=[], inputs={}, configurations=[], primitive_refs=[], phase_spans=[]
        )
        try:
            e.require(private, "private_scratch", "private", probe.read_text())
            e.require(
                private,
                "private_disk_at_least_1GB",
                True,
                shutil.disk_usage(private).free >= 1_000_000_000,
            )
            for binary in dict.fromkeys(row["argv"][0] for row in plan):
                e.require(
                    Path(binary), "executable_available", True, shutil.which(binary) is not None
                )
            progress("before_authentication")
            inputs = e.authenticate(args.root, raw)
            progress("after_authentication", 1, 0)
            progress("preconditions_checked", 1, 0)
            work = e.measure(inputs, raw)
            work["failures"] = []
        except e.OperandError as error:
            work["failures"] = [error.finding]
            progress("external_operand_blocked")
        work["preconditions"] = dict(
            private_scratch=True,
            no_model_resources_required=True,
            external_operands_authenticated=not work["failures"],
        )
        progress("preconditions_checked", 1, 0)
        receipts = execute(plan, raw / "checks") if not work["failures"] else []
        if (private / "coverage.json").is_file():
            work["coverage"] = json.loads((private / "coverage.json").read_bytes())
            atomic_json(raw / "owned_coverage.json", work["coverage"])
        work["manifest_reference"] = e.reference(raw / "execution_manifest.json")
        work["code_refs"] = []
        for path in e.OWNED + [
            e.TEST,
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "python/carnot/reporting/primary_publication.py",
        ]:
            dest = raw / "code" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(e.ROOT / path, dest)
            work["code_refs"].append(e.reference(dest))
        work["duration_s"] = time.monotonic() - began
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, receipts, raw, output)
        receipts.extend(controls(value, raw))
        candidate = raw / "private_candidate.json"
        atomic_json(candidate, e.build(work, receipts, raw, output))
        checked = terminal_checks(candidate, raw / "prepublication")
        receipts.extend(checked["checks"])
        work["adversarial_findings"] = checked["adversarial"]["findings"]
        work["duration_s"] = time.monotonic() - began
        atomic_json(raw / "measurement.json", work)
        final = e.build(work, receipts, raw, output)
        note_root = output.parent.parent if output.parent.name == "results" else output.parent
        e.write_note(final, note_root / "docs/research-notes/v720-table-fidelity.md")
        publish(final, output, raw)
    return 0
