"""REQ-VERIFY-8329: bounded private checks precede atomic publication.

This adapter freezes execution commands separately from the scientific protocol.
The existing supervisor owns child process groups and durable receipt streams.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import kv260_workload_cost_8329 as e
from carnot.reporting import kv260_workload_primitives_8329 as primitives
from carnot.reporting import v718_replay_history as history
from carnot.reporting import v717_contract_runner as qualified
from carnot.reporting import v718_replay_runner as audit_runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.request_trace_inventory_8200 import operand
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Reuse qualified scoped checks with private E2E-018/020 and no full suite."""
    with patch.object(qualified, "m", e):
        plan = qualified.manifest(private)
    plan[0]["deadline"] = 480
    plan[1]["argv"].insert(-1, "tests/python/test_hard_exit_learning_qualification_8206.py")
    plan[1]["argv"].insert(-1, "tests/python/test_kv260_local_cost_boundary_8315.py")
    plan[1]["name"] = "private_E2E018_E2E020_consumers"
    plan[1]["deadline"] = 480
    return list(plan)


def controls(value: Json, raw: Path) -> list[Json]:
    """Fresh processes must reject altered summaries even after an attacker rehashes."""
    receipts = []
    for name in ["valid", "negative", "rehashed_tamper"]:
        changed = deepcopy(value)
        if name == "negative":
            changed["reproducibility_checksum"] = "invalid"
        if name == "rehashed_tamper":
            changed["cpu_cost_ready_score"] = 1 - changed["cpu_cost_ready_score"]
            changed.pop("reproducibility_checksum")
            changed["reproducibility_checksum"] = canonical_hash(changed)
        path = raw / (name + ".json")
        atomic_json(path, changed)
        receipts.append(
            child(
                name,
                [
                    str(e.ROOT / ".venv/bin/python"),
                    "-u",
                    str(e.ROOT / e.CLI),
                    "--cold-replay",
                    str(path),
                ],
                raw / "controls",
                deadline=60,
                expected=0 if name == "valid" else 1,
                heartbeat=20,
            )
        )
    return receipts


def publish(work: Json, output: Path, raw: Path, receipts: list[Json]) -> None:
    """Only checked candidate bytes become visible; failed checks clear readiness."""
    attempts: list[Json] = []

    def validate(candidate: Path) -> Json:
        logs = raw / "terminal" / str(len(attempts))
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
            deadline=60,
            heartbeat=20,
        )
        found = audit_runner.audit(candidate, logs, {})
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
        report = dict(
            passed=all(r["passed"] for r in [cold, found["receipt"], rows]),
            checks=[cold, found["receipt"], rows],
            adversarial=found,
        )
        attempts.append(report)
        return report

    candidate = raw / "audit_candidate.json"
    atomic_json(candidate, e.build(work, raw, receipts))
    found = audit_runner.audit(candidate, raw / "prepublication", {})
    work.update(adversarial_findings=found["findings"], finding_dispositions=found["dispositions"])
    receipts = receipts + [found["receipt"]]
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, raw, receipts)
    try:
        publication = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        atomic_json(raw / "rejected_candidate.json", value)
        receipts += attempts[0]["checks"]
        work["adversarial_findings"] += attempts[0]["adversarial"]["findings"]
        atomic_json(raw / "measurement.json", work)
        publication = publish_primary(output, e.build(work, raw, receipts), validate)
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            attempts=attempts,
            checks=attempts[-1]["checks"],
            normal_process_exit=True,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Separate worker, replay and private controls from fully checked production."""
    e.progress("start_no_model_load", 0, 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261009"], default="20261009")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift")
        return 0 if passed else 1
    if args.worker_output:
        e.measure(args.root, args.worker_output.parent)
        return 0
    output = (
        args.private_output or args.output or e.ROOT / "results" / (e.NAME + ".json")
    ).absolute()
    if (
        args.private_output or args.root.resolve() != e.ROOT.resolve()
    ) and output.resolve().is_relative_to((e.ROOT / "results").resolve()):
        parser.error("private controls require private outputs")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    with TemporaryDirectory(prefix="carnot8329-validation-") as directory:
        private = Path(directory)
        plan = manifest(private)
        worker = [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            str(e.ROOT / e.CLI),
            "--root",
            str(args.root),
            "--worker-output",
            str(raw / "measurement.json"),
        ]
        atomic_json(
            raw / "execution_manifest.json",
            dict(
                commands=plan,
                worker_argv=worker,
                worker_deadline_s=600,
                heartbeat_s=20,
                science_parameters=primitives.SCIENCE,
                finding_consumer_policy=history.POLICY,
            ),
        )
        if args.private_output:
            work = e.measure(args.root, raw)
            receipts = [dict(name="private_control_route", passed=True)]
        else:
            worker_receipt = child("measurement", worker, raw / "logs", deadline=600, heartbeat=20)
            if (raw / "measurement.json").is_file():
                work = json.loads((raw / "measurement.json").read_bytes())
            else:
                work = e.measure(private / "missing", raw)
                work["checks"].append(
                    operand(
                        "measurement_child",
                        Path(worker_receipt["stdout_path"]),
                        "complete",
                        worker_receipt["exit_code"],
                    )
                )
            receipts = [worker_receipt] + execute(plan, raw / "logs")
            if (private / "coverage.json").is_file():
                atomic_json(
                    raw / "owned_coverage.json",
                    json.loads((private / "coverage.json").read_bytes()),
                )
                work["owned_coverage_reference"] = e.reference(raw / "owned_coverage.json")
        work["execution_manifest_reference"] = e.reference(raw / "execution_manifest.json")
        work["invocation_argv"] = list(sys.argv if argv is None else [e.CLI, *argv])
        atomic_json(raw / "measurement.json", work)
        receipts += controls(e.build(work, raw, receipts), raw)
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        publish(work, output, raw, receipts)
    e.progress("published_terminal", 1, 0)
    return 0
