"""REQ-REPORT-8465: scoped validation and unchanged consumers precede publication."""

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

from carnot.reporting import journal_equivalence_8465 as r
from carnot.reporting import v726_feature_panel_runner as consumers
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute
from carnot.verify import journal_equivalence_8465 as k
from carnot.verify import python_transaction_cost_8407 as service

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """The shipped plan owns only new statements; real exits save genuine child shards."""
    with patch.object(consumers, "e", r):
        plan = list(consumers.manifest(private))
    config = private / "coverage.ini"
    config.write_text(config.read_text().replace("patch=subprocess", "patch=subprocess,_exit"))
    next(row for row in plan if row["name"] == "owned_tests")["deadline"] = 1200
    return plan


def run(root: Path, output: Path, private: Path, control: bool = False) -> int:
    """Only complete checked evidence can expose readiness; failures remain terminal and honest."""
    raw = output.parent / "raw" / r.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, mode=0o700)
    plan = [] if control else manifest(private)
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            owned=r.OWNED,
            measurement_cap_s=3600,
            validation_reserve_s=900,
            task_cap_s=4800,
            full_repository_suite="not_run",
        ),
    )
    work = r.measure(root, raw, private)
    receipts = execute(plan, raw / "validation")
    if (private / "coverage.json").is_file():
        shards = raw / "coverage_shards"
        shards.mkdir()
        for source in private.glob(".coverage*"):
            shutil.copyfile(source, shards / source.name)
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        report = json.loads((raw / "coverage.json").read_bytes())
        work["coverage_shards"] = [r.reference(p) for p in sorted(shards.iterdir())]
        work["owned_coverage"] = dict(
            report["totals"], reference=r.reference(raw / "coverage.json")
        )
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, r.build(work, receipts, output))
    argv = [sys.executable, "-u", str(r.ROOT / r.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(child(name, argv + args, raw / "controls", expected=expected, deadline=180))
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["journal_ready_score"] = 999
    changed["reproducibility_checksum"] = r.checksum(changed)
    atomic_json(raw / "rehashed_tamper.json", changed)
    receipts.append(
        child(
            "rehashed_tamper",
            argv + ["--cold-replay", str(raw / "rehashed_tamper.json")],
            raw / "controls",
            expected=1,
            deadline=180,
        )
    )
    if work["primitives"]:
        changed_work = deepcopy(work)
        folder = raw / "rehashed_primitive"
        changed_work["raw"] = str(folder)
        primitive = deepcopy(r.read(changed_work["primitives"][0]))
        primitive["final"]["state"]["release_cursor"] += 1
        atomic_json(folder / "primitive.json", primitive)
        changed_work["primitives"][0] = r.reference(folder / "primitive.json")
        changed_work["rows"][0]["primitive_reference"] = changed_work["primitives"][0]
        atomic_json(folder / "measurement.json", changed_work)
        tampered = folder / "candidate.json"
        atomic_json(tampered, r.build(changed_work, receipts, tampered))
        receipts.append(
            child(
                "rehashed_primitive",
                argv + ["--cold-replay", str(tampered)],
                folder / "logs",
                expected=1,
                deadline=180,
            )
        )
    atomic_json(candidate, r.build(work, receipts, output))
    checks, findings = consumers.scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"] += [dict(phase=c["name"], duration_s=c["duration_s"]) for c in receipts]
    atomic_json(raw / "measurement.json", work)
    value = r.build(work, receipts, output)

    def validator(path: Path) -> Json:
        """Unchanged validators retain findings even when publishing an honest disqualification."""
        checks, flags = consumers.scan(path, raw / "terminal_validators")
        honest_failure = (
            value["verdict_class"] == "disqualified" and value["journal_ready_score"] == 0
        )
        return dict(
            passed=r.replay(path) and (all(c["passed"] for c in checks) or honest_failure),
            receipts=checks,
            findings=flags,
        )

    publication = publish_primary(output, value, validator)
    service.s.sync_directory(output.parent)
    reader = reader_receipt(
        r.TASK, output.parent, field="journal_ready_score", expected=value["journal_ready_score"]
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=reader,
            primary_sha256=publication["primary_sha256"],
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    r.progress(
        "published_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Private controls exercise actual CLI branches without recursive test execution."""
    r.progress("start_no_model_load_MODEL_SPECS_empty_current_llm_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261012"], default="20261012")
    parser.add_argument("--root", type=Path, default=r.ROOT)
    parser.add_argument("--output", type=Path, default=r.ROOT / "results" / (r.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--reader", type=Path)
    parser.add_argument("--pin", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--arm", choices=["journal", "snapshot"])
    parser.add_argument("--barrier", choices=[*k.BARRIERS, "none"])
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.worker:
        return r.worker(args.worker, args.store, args.arm, args.barrier)
    if args.reader:
        return r.reader(args.reader, args.store, args.arm, args.pin, args.release)
    if args.cold_replay:
        passed = r.replay(args.cold_replay)
        r.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(r.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="carnot8465-", dir="/var/tmp") as directory:
            return run(args.root, output, Path(directory), args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
