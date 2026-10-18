"""REQ-VERIFY-8492: bounded real checks precede atomic reader-visible terminal bytes."""

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

from carnot.reporting import publication_cost_model_8492 as r
from carnot.reporting import v726_feature_panel_runner as consumers
from carnot.reporting import v729_contract_runner as collected
from carnot.reporting import v729_historical_inputs as history
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Keep exact collections and disjoint E2E-018/021 shards through the shipped planner."""
    with patch.object(collected, "e", r):
        return list(collected.manifest(private))


def run(root: Path, output: Path, private: Path, control: bool = False) -> int:
    """Retain owned failures honestly; only immutable validated candidate bytes can publish."""
    raw = output.parent / "raw" / r.NAME / "invocations" / str(time.time_ns())
    plan = [] if control else manifest(private)
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            owned=r.OWNED,
            task_cap_s=4800,
            measurement_cap_s=3600,
            validation_reserve_s=900,
            full_repository_suite="not_run",
            partition_proof="each collected historical node appears in exactly one shard",
        ),
    )
    work = r.measure(root, raw)
    receipts = execute(plan, raw / "validation")
    if (private / "coverage.json").exists():
        shards = raw / "coverage_shards"
        shards.mkdir()
        for source in private.glob(".coverage*"):
            shutil.copyfile(source, shards / source.name)
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        work["coverage_shards"] = [r.reference(p) for p in sorted(shards.iterdir())]
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, r.build(work, receipts, raw, output))
    argv = [sys.executable, "-u", str(r.ROOT / r.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(child(name, argv + args, raw / "controls", expected=expected, deadline=120))
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["append_protocol_ready_score"] = 999
    changed["reproducibility_checksum"] = r.checksum(changed)
    atomic_json(raw / "rehashed_tamper.json", changed)
    receipts.append(
        child(
            "rehashed_tamper",
            argv + ["--cold-replay", str(raw / "rehashed_tamper.json")],
            raw / "controls",
            expected=1,
            deadline=120,
        )
    )
    atomic_json(candidate, r.build(work, receipts, raw, output))
    checks, findings = consumers.scan(candidate, raw / "preterminal_validators")
    receipts += checks
    work["adversarial_findings"] = findings
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"] += [dict(phase=c["name"], duration_s=c["duration_s"]) for c in receipts]
    atomic_json(raw / "measurement.json", work)
    value = r.build(work, receipts, raw, output)

    def validator(path: Path) -> Json:
        """Unchanged scanners and fresh primitive reconstruction bind terminal bytes."""
        checks, flags = consumers.scan(path, raw / "terminal_validators")
        return dict(
            passed=r.replay(path) and all(c["passed"] for c in checks),
            receipts=checks,
            findings=flags,
        )

    publication = publish_primary(output, value, validator)
    consumer = reader_receipt(
        r.TASK,
        output.parent,
        field="append_protocol_ready_score",
        expected=value["append_protocol_ready_score"],
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
    r.progress(
        "published_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Private controls own disk scratch and cannot write into the checkout."""
    r.progress("start_no_model_load_MODEL_SPECS_empty_current_llm_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261013"], default="20261013")
    parser.add_argument("--root", type=Path, default=r.ROOT)
    parser.add_argument("--output", type=Path, default=r.ROOT / "results" / (r.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--history-request", type=Path)
    parser.add_argument("--history-output", type=Path)
    args = parser.parse_args(argv)
    if args.history_request:
        if not args.history_output:
            parser.error("history output required")
        return int(history.worker(args.history_request, args.history_output))
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = r.replay(args.cold_replay)
        r.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(r.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8492-owned-", dir="/var/tmp") as folder:
            return run(args.root, output, Path(folder), args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
