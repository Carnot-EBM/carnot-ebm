"""REQ-VERIFY-8372: qualify a read-only request through bounded real children.

The frozen manifest measures only this invocation. Historical producer failures
remain terminal evidence and cannot be repaired by these administrative checks.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_missing_evidence_8372 as e
from carnot.reporting import kv260_local_cost_boundary_8315 as base
from carnot.reporting import v717_contract_runner as qualified
from carnot.reporting import v718_replay_runner as terminal
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Use the established scoped coverage, strict typing and private E2E-018 commands."""
    with patch.object(qualified, "m", e):
        plan = list(qualified.manifest(private))
    plan[0]["deadline"] = 600
    return plan


def run(
    root: Path, output: Path, private: Path, supplied: Path | None = None, *, control: bool = False
) -> int:
    """Owned check failures disqualify; absent external bytes remain an honest block."""
    start = time.monotonic_ns()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    plan = manifest(private)
    if control:
        plan = [
            dict(
                name="private_child",
                argv=[sys.executable, "-u", "-c", "print('private child')"],
                expected=0,
                deadline=10,
                scope="owned",
            )
        ]
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            task_cap_s=4800,
            private_scratch=str(private),
            heartbeat_s=30,
            no_model_load=True,
            private_control=control,
        ),
    )
    with patch.object(qualified, "m", e):
        failures = qualified.preflight(plan)
    probe = private / "scratch_probe"
    probe.write_bytes(b"disk-backed-private-scratch")
    work = e.measure(root, raw, supplied)
    work["checks"].extend(failures)
    work["preflight_passed"] = not failures
    work["checks"].append(
        e.gate(
            private, "private_scratch", True, probe.read_bytes() == b"disk-backed-private-scratch"
        )
    )
    work["checks"].append(
        e.gate(
            private,
            "available_disk_at_least_1GiB",
            True,
            shutil.disk_usage(private).free >= 1024**3,
        )
    )
    receipts = execute(plan, raw / "checks") if not failures else []
    if (private / "coverage.json").is_file():
        atomic_json(
            raw / "owned_coverage.json", json.loads((private / "coverage.json").read_bytes())
        )
        work["owned_coverage_reference"] = base.reference(raw / "owned_coverage.json")
    work["execution_manifest_reference"] = base.reference(raw / "execution_manifest.json")
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["started_monotonic_ns"] = start
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)
    for name in ["valid", "negative", "rehashed_tamper"]:
        changed = deepcopy(value)
        if name == "negative":
            changed["reproducibility_checksum"] = "invalid"
        if name == "rehashed_tamper":
            changed["execution_ready_score"] = 1
            changed.pop("reproducibility_checksum")
            changed["reproducibility_checksum"] = canonical_hash(changed)
        path = raw / (name + ".json")
        atomic_json(path, changed)
        receipts.append(
            child(
                "cold_" + name,
                [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
                raw / "cold",
                expected=int(name != "valid"),
                deadline=60,
                heartbeat=20,
            )
        )
    atomic_json(raw / "audit_candidate.json", e.build(work, receipts, raw, output))
    with patch.object(terminal, "e", e):
        audit = terminal.audit(raw / "audit_candidate.json", raw / "audit", {})
        work["adversarial_findings"] = audit["findings"]
        receipts.append(audit["receipt"])
        atomic_json(raw / "measurement.json", work)
        terminal.publish(e.build(work, receipts, raw, output), output, raw)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Private controls exercise the actual CLI while natural outputs use real upstream bytes."""
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261010"], default="20261010")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--supplied-receipt", type=Path)
    parser.add_argument("--private-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_e2e and output.is_relative_to(e.ROOT):
        parser.error("private controls require an output outside the repository")
    with TemporaryDirectory(prefix="exp8372-owned-", dir="/var/tmp") as directory:
        return run(
            args.root, output, Path(directory), args.supplied_receipt, control=args.private_e2e
        )
