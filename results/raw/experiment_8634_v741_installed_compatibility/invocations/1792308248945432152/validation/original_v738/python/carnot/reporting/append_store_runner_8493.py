"""REQ-VERIFY-8493: bounded scoped checks and unchanged consumers precede publication."""

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

from carnot.reporting import append_store_qualification_8493 as r
from carnot.reporting import v726_feature_panel_runner as consumers
from carnot.reporting import v729_contract_runner as collected
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute
from carnot.verify import append_store_8493 as k

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Keep genuine child exit coverage and exact disjoint historical E2E-018/021 collections."""
    with patch.object(collected, "e", r):
        plan = list(collected.manifest(private))
    config = private / "coverage.ini"
    config.write_text(config.read_text().replace("patch=subprocess", "patch=subprocess,_exit"))
    return plan


def run(root: Path, output: Path, private: Path, control: bool = False) -> int:
    """Execute frozen checks, retain raw failures and atomically publish checked terminal bytes."""
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
            partition_proof="disjoint nonoverlapping shards equal each original collection",
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
        report = json.loads((raw / "coverage.json").read_bytes())
        work["coverage_shards"] = [r.reference(p) for p in sorted(shards.iterdir())]
        work["owned_coverage"] = dict(
            report["totals"], reference=r.reference(raw / "coverage.json")
        )
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, r.build(work, receipts, raw, output))
    argv = [sys.executable, "-u", str(r.ROOT / r.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(child(name, argv + args, raw / "controls", expected=expected, deadline=180))
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["append_store_ready_score"] = 999
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
    atomic_json(candidate, r.build(work, receipts, raw, output))
    checks, findings = consumers.scan(candidate, raw / "preterminal_validators")
    receipts += checks
    work["adversarial_findings"] = findings
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"] += [dict(phase=c["name"], duration_s=c["duration_s"]) for c in receipts]
    atomic_json(raw / "measurement.json", work)
    value = r.build(work, receipts, raw, output)

    def validator(path: Path) -> Json:
        """Retain unchanged findings and exact primitive replay in the bound terminal sidecar."""
        checks, flags = consumers.scan(path, raw / "terminal_validators")
        return dict(
            passed=r.replay(path) and all(c["passed"] for c in checks),
            receipts=checks,
            findings=flags,
        )

    publication = publish_primary(output, value, validator)
    k.base.source.s.sync_directory(output.parent)
    consumer = reader_receipt(
        r.TASK,
        output.parent,
        field="append_store_ready_score",
        expected=value["append_store_ready_score"],
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
    """Real CLI controls use private disk scratch and cannot redirect output into the checkout."""
    r.progress("start_no_model_load_MODEL_SPECS_empty_current_llm_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261013"], default="20261013")
    parser.add_argument("--root", type=Path, default=r.ROOT)
    parser.add_argument("--output", type=Path, default=r.ROOT / "results" / (r.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--barrier", choices=[*k.BARRIERS, "none"], default="none")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.worker:
        if not args.store:
            parser.error("worker store required")
        return r.worker(args.worker, args.store, args.barrier)
    if args.cold_replay:
        passed = r.replay(args.cold_replay)
        r.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(r.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8493-owned-", dir="/var/tmp") as folder:
            return run(args.root, output, Path(folder), args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
