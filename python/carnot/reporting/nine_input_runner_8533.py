"""REQ-REPORT-8533: bounded children validate private candidates before atomic publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any

from carnot.reporting import nine_input_qualification_8533 as e
from carnot.reporting import nine_input_validation_8533 as validation
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v726_feature_panel_runner import scan

Json = dict[str, Any]


def run(root: Path, output: Path, control: bool) -> int:
    """Every actual command keeps a durable log, status and bounded deadline."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = e.measure(root, raw)
    receipts = [] if control or not work["extension"] else validation.validate(work, raw)
    candidate = raw / "private_candidate.json"
    atomic_json(candidate, e.artifact(work, receipts, output))
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("owned_failure", ["--deliberate-error"], 7),
    ]:
        receipts.append(
            child(name, [*argv, *args], raw / "controls", expected=expected, deadline=120)
        )
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["native_execution_count"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    tamper = raw / "self_consistently_rehashed_tamper.json"
    atomic_json(tamper, changed)
    receipts.append(
        child(
            "rehashed_tamper",
            [*argv, "--cold-replay", str(tamper)],
            raw / "controls",
            expected=1,
            deadline=120,
        )
    )
    if work["extension"] and not control:
        receipts.extend(validation.rust_coverage(work, raw))
    atomic_json(candidate, e.artifact(work, receipts, output))
    checks, findings = scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    atomic_json(raw / "measurement.json", work)
    value = e.artifact(work, receipts, output)

    def validator(path: Path) -> Json:
        """Failure artifacts publish with no readiness and retain every external finding."""
        checks, flags = scan(path, raw / "terminal_validators")
        return dict(
            passed=e.replay(path)
            and (
                all(r["passed"] for r in checks)
                or value["verdict_class"] in {"blocked", "disqualified"}
            ),
            publication_qualification_passed=all(r["passed"] for r in checks),
            receipts=checks,
            findings=flags,
        )

    published = publish_primary(output, value, validator)
    reader = reader_receipt(
        e.TASK,
        output.parent,
        field="nine_input_native_ready_score",
        expected=value["nine_input_native_ready_score"],
    )
    consumer = (
        e.consume(output)
        if value["nine_input_native_ready_score"]
        else dict(ready=False, reason=value["honest_verdict"])
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            primary_sha256=published["primary_sha256"],
            publication=published,
            selected_identity=reader,
            downstream_consumer=consumer,
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    e.progress("finished", value["completed_count"], value["failed_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Start visibly; private controls cannot write reader-visible repository outputs."""
    e.progress("start_no_model_load_LLM_calls_0")
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in {"--historical-tests", "--historical-exec"}:
        return validation.historical(arguments[1:], script=arguments[0] == "--historical-exec")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261015"], default="20261015")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--consume", type=Path)
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.consume:
        print(json.dumps(e.consume(args.consume)), flush=True)
        return 0
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private control output must be outside repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(args.root, output, args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
