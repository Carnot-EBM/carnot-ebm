"""REQ-REPORT-9179-V781: original bytes keep failed and missing branches visible."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
from typing import Any
from unittest.mock import patch

from carnot.reporting import v776_capstone_evidence as history
from carnot.reporting import v781_contract_evidence as contract
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.encoder_head_diagnosis_8972 import reference
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v778_capstone_evidence import edge as edge
from carnot.reporting.v781_capstone_reduction import reduce as reduce

Json = dict[str, Any]
ROOT = contract.ROOT
CONFIG = "openspec/change-proposals/v781-capstone-inputs.json"
PROTOCOL = json.loads((ROOT / CONFIG).read_bytes())
PINS, TASK_PIN, TASKS = PROTOCOL["pins"], PROTOCOL["task_hash"], PROTOCOL["task_ids"]
SUFFIXES = [t.split("-", 1)[1] for t in TASKS]
project = deepcopy


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Strict operand types distinguish absence and false booleans from integer zero."""
    row = dict(contract.check(path, field, expected, observed, name))
    row["upstream"] = TASKS[-1]
    return row


def authenticate(root: Path, raw: Path) -> Json:
    """Copy authenticated producer bytes without repinning historical implementation code."""
    checks, snapshots, values, operands = [], {}, {}, {}
    for name, pin in PINS.items():
        path = root / name
        row = check(
            path, "sha256", pin, reference(path)["sha256"] or "missing_file", "source_bytes"
        )
        checks.append(row)
        if row["passed"]:
            snapshots[name] = retain(path, raw, pin)
    authority = contract.authority(root)
    tasks = authority["tasks"]
    checks.extend(authority["checks"])
    checks.append(check(root, "task_hash", TASK_PIN, canonical_hash(tasks), "task_authority"))
    for name, ref in snapshots.items():
        if name.startswith("results/"):
            value = json.loads(Path(ref["path"]).read_bytes())
            number = int(Path(name).name.split("_")[1])
            identity = TASKS[number - 9166] if 9166 <= number < 9179 else f"prior:{number}"
            scheduler = value.get("blocked_at_layer") == "conductor_pre_gate"
            valid = value.get("experiment" if scheduler else "experiment_id") == number
            checks.append(check(root / name, "producer_identity", True, valid, "producer_identity"))
            values[identity] = dict(value=value, reference=ref, selected_name=name)
    bindings = [
        ref
        for entry in values.values()
        for key, item in entry["value"].items()
        for ref in (
            item
            if key == "current_work_receipts"
            else list(item.values())
            if key == "protocol_references"
            else [item]
            if key.endswith("_reference") and isinstance(item, dict)
            else []
        )
        if ref.get("path") and ref.get("sha256")
    ]
    for index, bound in enumerate(bindings):
        original = Path(bound["path"])
        if str(original) in operands:
            continue
        path = root / original.relative_to(ROOT) if original.is_relative_to(ROOT) else original
        row = check(
            path,
            "sha256",
            bound["sha256"],
            reference(path)["sha256"] or "missing_file",
            "producer_binding",
        )
        checks.append(row)
        if row["passed"]:
            operands[str(original)] = retain(path, raw, bound["sha256"])
        if index % 20 == 0:
            print(
                f"[exp9179] phase=producer_bindings completed={index} pending={len(bindings) - index}",
                flush=True,
            )
    for name in PROTOCOL["primary_paths"][:-1]:
        if name not in PINS:
            checks.append(
                check(
                    root / name,
                    "declared_primary",
                    "missing_file",
                    reference(root / name)["sha256"] or "missing_file",
                    "frozen_primary_absence",
                )
            )
    limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    cpu_limit = resource.getrlimit(resource.RLIMIT_CPU)[0]
    resources: list[Json] = [
        dict(
            resource="private_disk",
            available=shutil.disk_usage(raw).free >= 2 * 1024**3,
            receipt=dict(
                free_bytes=shutil.disk_usage(raw).free, minimum_bytes=2 * 1024**3, path=str(raw)
            ),
        ),
        dict(
            resource="memory",
            available=limit == resource.RLIM_INFINITY or limit >= 256 * 1024**2,
            receipt=dict(
                rlimit_as=limit, meminfo=Path("/proc/meminfo").read_text().splitlines()[:3]
            ),
        ),
        dict(
            resource="cpu",
            available=bool(os.sched_getaffinity(0)),
            receipt=dict(affinity=sorted(os.sched_getaffinity(0)), rlimit_cpu=cpu_limit),
        ),
        dict(
            resource="imports",
            available=all(
                importlib.util.find_spec(n) is not None for n in ["yaml", "coverage", "pytest"]
            ),
            receipt=dict(modules=["yaml", "coverage", "pytest"]),
        ),
    ]
    for row in resources:
        checks.append(check(raw, row["resource"], True, row["available"], "resource_precondition"))
    return dict(
        checks=checks,
        snapshots=snapshots,
        values=dict(sorted(values.items())),
        operands=operands,
        tasks=tasks,
        preconditions_checked=resources,
        historical_code_resolutions=[],
    )


def outcomes(work: Json) -> list[Json]:
    """A scheduler alternative accounts for its slot without becoming a scientific primary."""
    with patch.multiple(history, TASKS=TASKS, SUFFIXES=SUFFIXES, edge=edge):
        rows = history.outcomes(work)
    for i, row in enumerate(rows):
        row.update(experiment_id=9166 + i, primary_path=PROTOCOL["primary_paths"][i])
        row["task"]["deliverable"] = PROTOCOL["primary_paths"][i]
        row["honest_verdict"] = row["honest_verdict"] or row["accounting_honest_verdict"]
        row["primary_sha256"] = (
            row["reference"]["sha256"] if row["primary_status"] == "present" else None
        )
        row["quarantine"] = (
            "preserve_original_diagnostic_and_forward_disqualifications"
            if i in {1, 2}
            else "preserve_disqualified_ARC_controls"
            if i == 12
            else None
        )
        for operand in row["gate_check_summary"]:
            if not operand.get("path"):
                operand["path"] = str(
                    ROOT / PROTOCOL["primary_paths"][TASKS.index(operand["upstream"])]
                )
        row["gate_operands"] = row["gate_check_summary"]
    return list(rows)
