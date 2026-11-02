"""REQ-REPORT-8572: qualify, seal, cold-replay and publish exact checked bytes."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import signal
import sys
import time
from typing import Any

from carnot.reporting import resident_qualification_8572 as e
from carnot.reporting import resident_validation_8572 as validation
from carnot.reporting import resident_worker_8572 as worker
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, progress
from carnot.reporting.v726_feature_panel_runner import scan


def run(root: Path, output: Path, control: bool = False) -> int:
    """Required failures remain terminal disqualifications; absent originals remain blocked."""
    start = time.monotonic_ns()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = e.measure(root, raw)
    work["development_diagnostics"] = [
        e.reference(p)
        for p in sorted((e.ROOT / "results/raw" / e.NAME / "preimplementation").glob("*"))
        if p.is_file()
    ]
    receipts = [] if control or not work["bundle"] else validation.validate(work, raw)
    if work.get("owned_failure"):
        receipts.append(dict(name="owned_measurement", passed=False))
    candidate = raw / "preliminary_candidate.json"
    sealed = e.seal(work, raw, "preliminary")
    atomic_json(candidate, e.artifact(sealed, receipts, output))
    work["snapshot_history"] = [sealed["reference"]]
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("owned_failure", ["--deliberate-error"], 7),
    ]:
        receipts.append(child(name, argv + args, raw / "controls", expected=expected, deadline=120))
    changed = deepcopy(e.read(e.reference(candidate)))
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    tamper = raw / "rehashed_tamper.json"
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
    audit = e.seal(work, raw, "audit")
    atomic_json(candidate, e.artifact(audit, receipts, output))
    work["snapshot_history"].append(audit["reference"])
    checked, findings = scan(candidate, raw / "preterminal_validators")
    receipts.extend(checked)
    work["adversarial_findings"] = findings
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    cost_snapshot = e.seal(work, raw, "cost_consumer")
    atomic_json(candidate, e.artifact(cost_snapshot, receipts, output))
    work["snapshot_history"].append(cost_snapshot["reference"])
    if work["primitive"] and all(r["passed"] for r in receipts):
        work["cost_consumer_receipt"] = e.consume(candidate)
        receipts.append(dict(name="cost_consumer", passed=work["cost_consumer_receipt"]["passed"]))
    terminal = e.seal(work, raw, "terminal")
    value = e.artifact(terminal, receipts, output)

    def validator(path: Path) -> dict[str, Any]:
        """The publisher exposes only bytes read by final scanners and fresh cold replay."""
        checks, flags = scan(path, raw / "terminal_validators")
        cold = child(
            "final_byte_cold", argv + ["--cold-replay", str(path)], raw / "controls", deadline=120
        )
        return dict(
            passed=cold["passed"]
            and (
                all(r["passed"] for r in checks)
                or value["verdict_class"] in {"blocked", "disqualified"}
            ),
            receipts=[*checks, cold],
            findings=flags,
        )

    publication = publish_primary(output, value, validator)
    selected = reader_receipt(
        e.TASK,
        output.parent,
        field="resident_inference_ready_score",
        expected=value["resident_inference_ready_score"],
    )
    consumed = e.consume(output) if value["resident_inference_ready_score"] else None
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            selected_identity=selected,
            actual_cost_consumer=consumed,
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    progress("8572_finished", value["completed_count"], value["failed_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Explicit modes keep private worker/replay commands separate from the experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261016"], default="20261016")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--config-hash", default="")
    parser.add_argument("--arm", choices=["python", "native"], default="python")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.worker:
        return worker.serve(args.worker, args.config_hash, args.arm, args.timeout)
    if args.panel:
        return worker.panel(args.panel, args.config_hash, args.arm, args.worker_output)
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    if args.private_control and args.output.absolute().is_relative_to(e.ROOT):
        parser.error("private output must be outside checkout")
    progress("8572_start")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(args.root, args.output.absolute(), args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
