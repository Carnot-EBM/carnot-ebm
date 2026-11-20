"""REQ-REPORT-9207-V783: retain producer authority and explicit scheduler absence."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v776_capstone_evidence as history
from carnot.reporting import v782_capstone_evidence as prior
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.encoder_head_diagnosis_8972 import reference
from carnot.reporting.v749_contract_methods import encode
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v778_capstone_evidence import edge as edge
from carnot.reporting.v783_capstone_reduction import reduce as reduce

Json = dict[str, Any]
ROOT = prior.ROOT
CONFIG = "openspec/change-proposals/v783-capstone-inputs.json"
PROTOCOL = json.loads((ROOT / CONFIG).read_bytes())
PINS, TASK_PIN, TASKS = PROTOCOL["pins"], PROTOCOL["task_hash"], PROTOCOL["task_ids"]
SUFFIXES = [t.split("-", 1)[1] for t in TASKS]
project = deepcopy


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Strict operand types prevent absent evidence and false from becoming measured zero."""
    return dict(prior.check(path, field, expected, observed, name), upstream=TASKS[-1])


def authenticate(root: Path, raw: Path) -> Json:
    """Copy only hash-matching sources and their original bindings before any reduction."""
    checks, snapshots, values, operands = [], {}, {}, {}
    for name, pin in PINS.items():
        path = root / name
        row = check(
            path, "sha256", pin, reference(path)["sha256"] or "missing_file", "source_bytes"
        )
        checks.append(row)
        if row["passed"]:
            snapshots[name] = retain(path, raw, pin)
    tasks = []
    if all(
        n in snapshots
        for n in [PROTOCOL["active"], "openspec/change-proposals/research-roadmap-vNEXT.md"]
    ):
        tasks = yaml.safe_load(Path(snapshots[PROTOCOL["active"]]["path"]).read_bytes())["tasks"]
        text = Path(
            snapshots["openspec/change-proposals/research-roadmap-vNEXT.md"]["path"]
        ).read_text()
        machine = yaml.safe_load(
            text.split("## Full machine-readable contract", 1)[1]
            .split("```yaml", 1)[1]
            .split("```", 1)[0]
        )["tasks"]
        table = [
            [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
            for line in text.split("## Task contract", 1)[1]
            .split("## Full machine-readable", 1)[0]
            .splitlines()
            if line.startswith("| ") and line.split("|")[1].strip().isdigit()
        ]
        matched = len(tasks) == len(machine) == len(table) == 14 and all(
            t in (m, dict(m, prompt=encode(m["prompt"])))
            and table[i] == [str(i + 1), str(t["phase"]), t["id"], t["deliverable"]]
            for i, (t, m) in enumerate(zip(tasks, machine, strict=False))
        )
        checks.append(check(root, "full_contract", True, matched, "full_contract"))
    checks.append(check(root, "task_hash", TASK_PIN, canonical_hash(tasks), "task_authority"))
    for name, ref in snapshots.items():
        if name.startswith("results/"):
            value = json.loads(Path(ref["path"]).read_bytes())
            number = int(Path(name).name.split("_")[1])
            identity = TASKS[number - 9194] if 9194 <= number < 9207 else f"prior:{number}"
            field = (
                "experiment"
                if value.get("blocked_at_layer") == "conductor_pre_gate"
                else "experiment_id"
            )
            checks.append(
                check(
                    root / name,
                    field,
                    number,
                    value.get(field, "missing_field"),
                    "producer_identity",
                )
            )
            values[identity] = dict(value=value, reference=ref, selected_name=name)
    bindings = [
        r
        for entry in values.values()
        for key, item in entry["value"].items()
        for r in (
            item
            if key in {"source_references", "code_config_hashes", "current_work_receipts"}
            else [item]
            if key.endswith("_reference") and isinstance(item, dict)
            else []
        )
        if r.get("path") and r.get("sha256")
    ]
    # Historical code is authenticated at its retained producer path, never repinned to checkout.
    old = values.get("prior:9193", {}).get("value", {})
    if old:
        ref = old["work_reference"]
        path = Path(ref["path"])
        gate = check(
            path,
            "sha256",
            ref["sha256"],
            reference(path)["sha256"] or "missing_file",
            "historical_work",
        )
        checks.append(gate)
        if gate["passed"]:
            historical = json.loads(path.read_bytes())
            bindings.extend(
                [
                    *historical["snapshots"].values(),
                    *historical["operands"].values(),
                    *historical["code"],
                ]
            )
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
        if index % 100 == 0:
            print(
                f"[exp9207] phase=authenticate completed={index} pending={len(bindings) - index}",
                flush=True,
            )
    for name in PROTOCOL["expected_absent"]:
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
            receipt=dict(
                affinity=sorted(os.sched_getaffinity(0)),
                rlimit_cpu=resource.getrlimit(resource.RLIMIT_CPU),
            ),
        ),
        dict(
            resource="imports",
            available=all(
                importlib.util.find_spec(n) is not None
                for n in ["yaml", "pytest", "coverage", "ruff", "mypy"]
            ),
            receipt=dict(modules=["yaml", "pytest", "coverage", "ruff", "mypy"]),
        ),
    ]
    checks.extend(
        check(raw, r["resource"], True, r["available"], "resource_precondition") for r in resources
    )
    return dict(
        checks=checks,
        snapshots=snapshots,
        values=dict(sorted(values.items())),
        operands=operands,
        tasks=tasks,
        preconditions_checked=resources,
        historical_code_resolutions=old.get("historical_code_resolutions", []),
    )


def outcomes(work: Json) -> list[Json]:
    """Scheduler artifacts and log-only skips certify accounting, never scientific execution."""
    with patch.multiple(history, TASKS=TASKS, SUFFIXES=SUFFIXES, edge=edge):
        rows = history.outcomes(work)
    for index, row in enumerate(rows):
        row.update(experiment_id=9194 + index, primary_path=PROTOCOL["primary_paths"][index])
        if row["scheduler_alternative"]:
            scheduler = work["values"][row["task_id"]]["value"]
            for observed in scheduler["gates_evaluated"]:
                row["gate_check_summary"].append(
                    dict(
                        check="scheduler_gate",
                        upstream=observed["upstream"],
                        path=observed["artifact_path"],
                        sha256=observed["artifact_sha256"],
                        artifact_field=observed["artifact_field"],
                        op=observed["op"],
                        expected=observed["expected"],
                        observed=observed["actual"],
                        passed=observed["passed"],
                    )
                )
        elif row["primary_status"] == "missing_file" and row["scheduler_events"]:
            row.update(
                disposition="authenticated_scheduler_log_skip",
                scheduler_honest_verdict="blocked_upstream_retired",
            )
        row["accounting_evidence_reference"] = row["reference"] or work["snapshots"].get(
            "ops/conductor-log.md"
        )
        row["honest_verdict"] = row["honest_verdict"] or row["accounting_honest_verdict"]
        row["gate_operands"] = row["gate_check_summary"]
        for operand in row["gate_operands"]:
            operand["path"] = operand.get("path") or str(
                ROOT / PROTOCOL["primary_paths"][TASKS.index(operand["upstream"])]
            )
    return list(rows)
