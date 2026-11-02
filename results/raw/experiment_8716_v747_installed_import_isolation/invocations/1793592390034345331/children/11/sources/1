"""REQ-REPORT-8716: qualify process isolation without changing numerical work.

Legacy public exports load training dependencies. Their identity assertions belong
in a short-lived child so pytest's unchanged memory guard measures its own owner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import installed_batch_8702 as previous
from carnot.reporting import v747_contract_methods as contract
from carnot.reporting.current_work_receipt import (
    atomic_json as atomic_json,
    canonical_hash as canonical_hash,
)
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = previous.ROOT
NAME, TASK = "experiment_8716_v747_installed_import_isolation", "exp8716-installed-import-isolation"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = previous.TESTS + ["tests/python/test_installed_import_isolation_8716.py"]
OWNED = [
    "python/carnot/inference/legacy_import_isolation.py",
    "python/carnot/reporting/installed_import_isolation_8716.py",
    "python/carnot/reporting/installed_import_runner_8716.py",
    CLI,
]
LEGACY = "tests/python/test_installed_lean_8633.py"
LEGACY_PIN = "sha256:3e1c5dd8062cfe7c5e22c6b636dc9765b5a82f0107641f04e981790bb2a8375b"
HISTORY = "results/experiment_8702_v746_installed_batch_prototype.json"
HISTORY_PIN = "sha256:879b1c1ebe5aa1996cd34ba4b635b6d30da6dbd4ee58e5fe4cbf8ef68b790e36"
reference, load = previous.reference, previous.load
protocol_added_lines = previous.protocol_added_lines


def inputs(root: Path) -> Json:
    """Use accepted original scientific operands and the current full scheduling contract."""
    with patch.object(previous, "authority", contract.authority):
        value = previous.inputs(root)
    checked = previous.authenticate(
        dict(path=str(root / HISTORY), sha256=HISTORY_PIN),
        dict(
            experiment_id=8702,
            task_id=previous.TASK,
            verdict_class="disqualified",
            required_checks_passed=False,
            flagged_adversarial=True,
        ),
    )
    value["references"] += checked["references"]
    value["gate_check_summary"] += checked["failures"]
    if checked["passed"]:
        try:
            historical = checked["value"]
            side = load(reference(Path(historical["terminal_validation_sidecar_path"])))
            bound = Path(side["publication"]["sidecar_path"])
            report = read_bound_sidecar(root / HISTORY, bound)
            if report["report"]["passed"] is not True:
                raise ValueError("historical_terminal_rejected")
            receipt = next(r for r in historical["validation_receipts"] if r["name"] == "tests-10")
            log = dict(path=receipt["stdout_path"], sha256=receipt["stdout_sha256"])
            if reference(Path(log["path"])) != log:
                raise ValueError("historical_log_changed")
            text = Path(log["path"]).read_text()
            if "5 passed, 1 error" not in text or "Memory leak: +553MB" not in text:
                raise ValueError("historical_diagnostic_changed")
            value["historical_failure"] = dict(
                retained_delta_mb=553,
                producer_flagged=True,
                original_assertions_passed=5,
                log_reference=log,
                primary_reference=checked["references"][0],
            )
            value["references"] += [
                log,
                reference(bound),
                reference(Path(historical["terminal_validation_sidecar_path"])),
            ]
        except (OSError, ValueError, KeyError, StopIteration) as error:
            value["gate_check_summary"].append(
                contract.gate(
                    root / HISTORY, "retained_diagnostic", True, str(error), previous.TASK
                )
            )
    value["passed"] = not value["gate_check_summary"]
    return value


def build(work: Json, output: Path) -> Json:
    """Reuse the published schema while deriving readiness from this invocation's checks."""
    value = previous.build(work, output)
    measured = work["measurement"]
    value.update(
        experiment_id=8716,
        task_id=TASK,
        milestone="2026.11.747",
        run_date="20261102",
        honest_verdict=f"complete_{value['verdict_class']}_v747_installed_import_isolation",
        random_seed=7478716,
        canonical_tasks_sha256=contract.PIN,
        task_contract_rows=work["inputs"]["contract"]["rows"],
        historical_8702_failure=work["inputs"].get("historical_failure"),
        guard_reproduction=measured.get("guard_reproduction"),
        installed_identity_rows=measured.get("installed_identity_rows", []),
        parity_rows=value["rows"],
        original_node_rows=work["validation"].get("original_node_rows", []),
        memory_lifecycle_rows=measured.get("memory_lifecycle_rows", []),
        worker_exit_rows=measured.get("worker_exit_rows", []),
        retained_producer_coverage=work["validation"].get("producer_coverage", {}),
        complete_cost_protocol=dict(
            task_id="exp8717-installed-batch-cost",
            method=work["inputs"].get("method"),
            frozen_installed_workload=measured.get("frozen_installed_workload"),
            roster=[dict(r, status="pending_Exp8717") for r in value["complete_cost_roster"]],
        ),
        methodology="Authenticate original accepted deployment operands and all fourteen V747 task objects. Reproduce the original guarded teardown failure, then run the identical export assertions in a fresh reaped child. Build a frozen wheel and run the original64 numerical cases at sizes1-16 in actual installed Python/native workers outside checkout. Preserve original122 compatibility nodes, batch producer coverage and resource guards. This establishes installation/parity only; speed, learning and physical acceleration remain unmeasured.",
    )
    value["field_principles"].update(
        {k: "Bind exact current evidence and preserve historical scientific limits." for k in value}
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
