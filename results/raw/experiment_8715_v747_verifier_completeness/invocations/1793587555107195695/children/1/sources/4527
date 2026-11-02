"""REQ-REPORT-8559: bounded controls validate exact bytes before atomic publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any

from carnot.reporting import complete_native_cost_8559 as e
from carnot.reporting import complete_native_validation_8559 as validation
from carnot.reporting import complete_native_worker_8559 as worker
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, progress
from carnot.reporting.v726_feature_panel_runner import scan

Json = dict[str, Any]


def run(root: Path, output: Path, control: bool) -> int:
    """Keep every failed receipt and bind the hardware consumer to terminal primary bytes."""
    start = time.monotonic_ns()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = e.measure(root, raw)
    receipts = [] if control or not work["bundle"] else validation.validate(work, raw)
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preliminary_candidate.json"
    atomic_json(candidate, e.artifact(work, receipts, output))
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("owned_failure", ["--deliberate-error"], 7),
    ]:
        receipts.append(child(name, argv + args, raw / "controls", expected=expected, deadline=120))
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    tamper = raw / "self_consistently_rehashed_tamper.json"
    atomic_json(tamper, changed)
    receipts.append(
        child(
            "rehashed_tamper",
            argv + ["--cold-replay", str(tamper)],
            raw / "controls",
            expected=1,
            deadline=120,
        )
    )
    if work["bundle"] and not control:
        receipts.extend(validation.finish(work, raw))
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "post_validation_candidate.json"
    atomic_json(candidate, e.artifact(work, receipts, output))
    checks, findings = scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    atomic_json(raw / "measurement.json", work)
    value = e.artifact(work, receipts, output)

    def validator(path: Path) -> Json:
        """Final scans and replay see the exact bytes that publication will replace."""
        checked, flags = scan(path, raw / "terminal_validators")
        cold = child(
            "final_byte_cold", argv + ["--cold-replay", str(path)], raw / "controls", deadline=120
        )
        return dict(
            passed=cold["passed"]
            and (
                all(r["passed"] for r in checked)
                or value["verdict_class"] in {"blocked", "disqualified"}
            ),
            receipts=[*checked, cold],
            findings=flags,
            publication_qualification_passed=all(r["passed"] for r in checked),
        )

    publication = publish_primary(output, value, validator)
    selected = reader_receipt(
        e.TASK,
        output.parent,
        field="complete_native_cost_ready_score",
        expected=value["complete_native_cost_ready_score"],
    )
    downstream = child(
        "actual_hardware_cost_consumer",
        argv + ["--consume", str(output)],
        raw / "controls",
        expected=0 if value["complete_native_cost_ready_score"] else 1,
        deadline=120,
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            primary_sha256=publication["primary_sha256"],
            publication=publication,
            selected_identity=selected,
            downstream_consumer=downstream,
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    progress("8559_finished", value["completed_count"], value["failed_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Flush startup before imports in children and keep private outputs outside the checkout."""
    progress("8559_start_no_model_load_MODEL_SPECS_empty_LLM_calls_0")
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "--sealed-tests":
        if "8533" in arguments[1]:
            from carnot.reporting.nine_input_validation_8558 import historical

            return int(historical(arguments[1], arguments[2:]))
        import pytest

        return int(pytest.main(arguments[2:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261016"], default="20261016")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--arm", choices=["python", "native"])
    parser.add_argument("--batch", type=int, choices=[1, 8], default=1)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--consume", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.worker:
        if not args.worker_output or not args.arm:
            parser.error("worker output and arm required")
        return worker.worker(args.worker, args.worker_output, args.arm, args.batch)
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    if args.consume:
        try:
            print(json.dumps(e.consume(args.consume)), flush=True)
            return 0
        except (OSError, ValueError, KeyError):
            return 1
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private control output must be outside repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(args.root, output, args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
