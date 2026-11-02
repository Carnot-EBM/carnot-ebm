"""REQ-VERIFY-8530: bounded children and unchanged consumers bind final qualification bytes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import v733_update_accounting as e
from carnot.reporting import v733_update_fixture as fixture
from carnot.reporting import v733_update_validation as validation
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v726_feature_panel_runner import scan

Json = dict[str, Any]


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Publish an honest disqualification when any required check or validator fails."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    work = e.measure(root, raw)
    plan = [] if control else validation.manifest(private)
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            task_cap_s=4800,
            measurement_cap_s=3600,
            validation_reserve_s=900,
            heartbeat_s=30,
        ),
    )
    collection = private / "collection_archive.json"
    work["collected_node_ids"] = []
    if collection.exists():
        archive = json.loads(collection.read_bytes())
        work["collected_node_ids"] = archive["node_ids"]
        # Durable collection receipts retain log bytes after private scratch is removed.
        for row in archive["receipts"]:
            for field in ["stdout", "stderr"]:
                target = raw / "collection" / Path(row[field + "_path"]).name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(row[field + "_path"]).read_bytes())
                row[field + "_path"] = str(target)
        atomic_json(raw / "collection_archive.json", archive)
        work["primitive_files"].append(e.reference(raw / "collection_archive.json"))
    receipts = []
    for i, spec in enumerate(plan):
        e.progress("validation", i, len(plan) - i)
        receipt = child(
            spec["name"],
            spec["argv"],
            raw / "validation",
            expected=spec["expected"],
            deadline=spec["deadline"],
        )
        receipt["collected_node_ids"] = spec.get("collected_node_ids", [])
        atomic_json(raw / "validation" / (spec["name"] + ".receipt.json"), receipt)
        receipts.append(receipt)
    if (private / "coverage.json").exists():
        atomic_json(raw / "coverage.json", json.loads((private / "coverage.json").read_bytes()))
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, e.build(work, receipts, raw, output))
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(
            child(name, [*argv, *args], raw / "controls", expected=expected, deadline=120)
        )
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["changed_coordinate_count"] += 1
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
    atomic_json(candidate, e.build(work, receipts, raw, output))
    checks, findings = scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["ended_monotonic_ns"] = time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)

    def validator(path: Path) -> Json:
        """A failed scientific gate can still publish valid failure accounting with no readiness."""
        checks, flags = scan(path, raw / "terminal_validators")
        honest_failure = (
            value["verdict_class"] == "disqualified"
            and value["online_adapter_ready_score"] == 0
            and value["required_checks_passed"] is False
        )
        return dict(
            passed=e.replay(path) and (all(r["passed"] for r in checks) or honest_failure),
            publication_qualification_passed=all(r["passed"] for r in checks),
            findings=flags,
            receipts=checks,
        )

    publication = publish_primary(output, value, validator)
    selected = reader_receipt(
        e.TASK,
        output.parent,
        field="online_adapter_ready_score",
        expected=value["online_adapter_ready_score"],
    )
    consumer = e.stream_consumer(output, Path(publication["sidecar_path"]))
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=consumer,
            selected_identity=selected,
            downstream_rejection_expected=value["online_adapter_ready_score"] == 0,
            primary_sha256=publication["primary_sha256"],
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    e.progress("finished", value["completed_count"], value["failed_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """The date, private path guard and finite clock bind work to this invocation."""
    e.progress("start_no_model_load_LLM_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261015"], default="20261015")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--fixture-worker", type=Path)
    parser.add_argument("--crash", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    if args.fixture_worker:
        try:
            cfg = json.loads(args.configuration.read_bytes())
            fixture.worker(cfg, args.fixture_worker, args.crash)
            return 0
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return 1
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8530-owned-", dir="/var/tmp") as directory:
            return run(args.root, output, Path(directory), control=args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
