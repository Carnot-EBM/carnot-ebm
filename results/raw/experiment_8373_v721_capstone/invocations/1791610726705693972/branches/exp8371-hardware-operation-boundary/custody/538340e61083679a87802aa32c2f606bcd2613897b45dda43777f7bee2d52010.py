"""REQ-VERIFY-8371: reuse bounded children and unchanged atomic validators.

The command manifest covers only added code. Private consumers exercise real
process recovery and authority rejection without reopening board measurements.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import hardware_operation_boundary_8371 as e
from carnot.reporting import kv260_local_cost_boundary_8315 as base
from carnot.reporting import v717_contract_runner as qualified
from carnot.reporting import v718_replay_runner as terminal
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Seal invocation coverage and private E2E controls before reading producer bytes."""
    with patch.object(qualified, "m", e):
        plan = list(qualified.manifest(private))
    plan[0]["deadline"] = 600
    plan[1]["argv"].insert(-1, "tests/python/test_hard_exit_learning_qualification_8206.py")
    plan[1]["argv"].insert(
        -1, "tests/python/test_v721_contract_methods_8360.py::test_private_e2e018"
    )
    plan[1]["argv"].insert(
        -1, "tests/python/test_kv260_workload_cost_8356.py::test_table_operation_semantics"
    )
    plan[1].update(name="private_E2E018_E2E020_consumers", deadline=600)
    plan.append(
        dict(
            name="private_disk_scratch",
            argv=["/usr/bin/df", "-T", str(private)],
            deadline=10,
            expected=0,
            scope="owned",
        )
    )
    return plan


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Real child receipts decide readiness; owned failures remain disqualified."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    plan = manifest(private)
    if control:
        plan = [
            dict(
                name="private_child",
                argv=[sys.executable, "-u", "-c", "print('private child')"],
                deadline=10,
                expected=0,
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
        ),
    )
    with patch.object(qualified, "m", e):
        failures = qualified.preflight(plan)
    disk = private / "scratch_probe"
    disk.write_bytes(b"disk-backed-private-scratch")
    work = e.measure(root, raw)
    work["checks"].extend(failures)
    work["validation_preflight_complete"] = not failures
    work["checks"].append(
        e.gate(
            private,
            "available_disk_at_least_1GiB",
            True,
            shutil.disk_usage(private).free >= 1024**3,
        )
    )
    work["checks"].append(
        e.gate(
            private, "private_scratch", True, disk.read_bytes() == b"disk-backed-private-scratch"
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
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)
    with patch.object(qualified, "m", e):
        receipts += qualified.controls(value, raw)
    atomic_json(raw / "audit_candidate.json", e.build(work, receipts, raw, output))
    with patch.object(terminal, "e", e):
        audit = terminal.audit(raw / "audit_candidate.json", raw / "audit", {})
        work["adversarial_findings"] = audit["findings"]
        receipts.append(audit["receipt"])
        atomic_json(raw / "measurement.json", work)
        terminal.publish(e.build(work, receipts, raw, output), output, raw)
    return 0
