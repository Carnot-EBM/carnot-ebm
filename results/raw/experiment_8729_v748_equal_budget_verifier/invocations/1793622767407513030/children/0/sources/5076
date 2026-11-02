"""REQ-REPORT-8545: bounded command contracts preserve checked terminal bytes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any

from carnot.reporting import nine_input_qualification_8545 as e
from carnot.reporting import nine_input_validation_8545 as validation
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v726_feature_panel_runner import scan

Json = dict[str, Any]


def request_bundle(bundle: Path, output: Path) -> int:
    """A private CLI must run text extraction through typed JSON, not just a kernel."""
    try:
        refs = json.loads(bundle.read_bytes())
        extension = refs["extension"]
        if e.inputs.reference(Path(extension["path"]))["sha256"] != extension["sha256"]:
            raise ValueError("extension_hash")
        rows = e.requests(
            e.inputs.read(refs["head"]),
            e.inputs.read(refs["tape"]),
            e.legacy.build.load(Path(extension["path"])),
        )
        atomic_json(output, dict(rows=rows))
        return int(not all(r["passed"] for r in rows))
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return 1


def run(root: Path, output: Path, control: bool) -> int:
    """All potentially long children have deadlines, heartbeats and durable stream hashes."""
    start = time.monotonic_ns()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = e.measure(root, raw)
    receipts = []
    if not control and work["extension"]:
        receipts.extend(validation.reproduce(work, raw))
        receipts.extend(validation.validate(work, raw))
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    if work["extension"]:
        bundle = raw / "request_bundle.json"
        atomic_json(
            bundle,
            dict(
                head=work["head_reference"],
                tape=work["request_tape_reference"],
                extension=work["extension"],
            ),
        )
        receipt = child(
            "private_complete_cli",
            [
                *argv,
                "--request-bundle",
                str(bundle),
                "--request-output",
                str(raw / "cli_responses.json"),
            ],
            raw / "controls",
            deadline=120,
        )
        if receipt["passed"]:
            receipt["passed"] = (
                json.loads((raw / "cli_responses.json").read_bytes())["rows"]
                == work["complete_request_rows"]
            )
        receipts.append(receipt)
    candidate = raw / "private_candidate.json"
    atomic_json(candidate, e.artifact(work, receipts, output))
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("owned_failure", ["--deliberate-error"], 7),
    ]:
        receipts.append(child(name, argv + args, raw / "controls", expected=expected, deadline=120))
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["native_execution_count"] += 1
    changed["reproducibility_checksum"] = e.legacy.checksum(changed)
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
    if work["extension"] and not control:
        receipts.extend(validation.finish(work, raw))
    atomic_json(raw / "measurement.json", work)
    atomic_json(candidate, e.artifact(work, receipts, output))
    checks, findings = scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["total_duration_s"] = (time.monotonic_ns() - start) / 1e9
    atomic_json(raw / "measurement.json", work)
    value = e.artifact(work, receipts, output)

    def validator(path: Path) -> Json:
        """An honest failure remains publishable, while readiness requires every validator."""
        checked, flags = scan(path, raw / "terminal_validators")
        return dict(
            passed=e.replay(path)
            and (
                all(r["passed"] for r in checked)
                or value["verdict_class"] in {"blocked", "disqualified"}
            ),
            receipts=checked,
            findings=flags,
            publication_qualification_passed=all(r["passed"] for r in checked),
        )

    publication = publish_primary(output, value, validator)
    selected = reader_receipt(
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
            primary_sha256=publication["primary_sha256"],
            publication=publication,
            selected_identity=selected,
            downstream_consumer=consumer,
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    e.legacy.progress("finished_8545", value["completed_count"], value["failed_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Expose independent request/replay controls without starting nested validation."""
    e.legacy.progress("start_8545_no_model_load_LLM_calls_0")
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in {"--sealed-tests", "--sealed-exec"}:
        return validation.historical(
            arguments[1], arguments[2:], script=arguments[0] == "--sealed-exec"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261015"], default="20261015")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--request-bundle", type=Path)
    parser.add_argument("--request-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--consume", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.request_bundle:
        if not args.request_output:
            parser.error("--request-bundle requires --request-output")
        return request_bundle(args.request_bundle, args.request_output)
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    if args.consume:
        print(json.dumps(e.consume(args.consume)), flush=True)
        return 0
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private control output must be outside repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(args.root, output, args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
