"""REQ-REPORT-9165-V780: immutable sources distinguish missing science from closure."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import resource
import shutil
from typing import Any
from unittest.mock import patch

from carnot.reporting import v779_capstone_evidence as prior
from carnot.reporting import v780_contract_evidence as contract
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.encoder_head_diagnosis_8972 import reference
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v778_capstone_evidence import edge as edge
from carnot.reporting.v780_capstone_reduction import reduce as reduce

Json = dict[str, Any]
ROOT = contract.ROOT
CONFIG = "openspec/change-proposals/v780-capstone-inputs.json"
PROTOCOL = json.loads((ROOT / CONFIG).read_bytes())
PINS, TASK_PIN, TASKS = PROTOCOL["pins"], PROTOCOL["task_hash"], PROTOCOL["task_ids"]
SUFFIXES = [t.split("-", 1)[1] for t in TASKS]
project = deepcopy


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Strict types prevent a true boolean from satisfying an integer readiness gate."""
    return dict(contract.check(path, field, expected, observed, name))


def authenticate(root: Path, raw: Path) -> Json:
    """Retain exact historical bytes; current checkout code cannot replace their authority."""
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
    checks.extend(authority["checks"])
    tasks = authority["tasks"]
    checks.append(check(root, "task_hash", TASK_PIN, canonical_hash(tasks), "task_authority"))
    for name, ref in snapshots.items():
        if not name.startswith("results/"):
            continue
        value = json.loads(Path(ref["path"]).read_bytes())
        number = int(Path(name).name.split("_")[1])
        identity = TASKS[number - 9152] if 9152 <= number < 9165 else f"prior:{number}"
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
                f"[exp9165] phase=producer_bindings completed={index} pending={len(bindings) - index}",
                flush=True,
            )
    for index in [0, 1]:
        entry = values.get(TASKS[index], {})
        observed = entry.get("value", {}).get(
            "required_checks_passed", "missing_field" if entry else "missing_file"
        )
        checks.append(
            check(
                root / PROTOCOL["primary_paths"][index],
                "required_checks_passed",
                True,
                observed,
                "upstream_custody_gate",
            )
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
    for field, observed in [
        ("free_disk_at_least_2GiB", shutil.disk_usage(raw).free >= 2 * 1024**3),
        (
            "address_space_at_least_256MiB",
            limit == resource.RLIM_INFINITY or limit >= 256 * 1024**2,
        ),
        ("yaml_import_available", importlib.util.find_spec("yaml") is not None),
    ]:
        checks.append(check(raw, field, True, observed, "resource_precondition"))
    return dict(
        checks=checks,
        snapshots=snapshots,
        values=dict(sorted(values.items())),
        operands=operands,
        tasks=tasks,
        historical_code_resolutions=[],
        declared_gate_preconditions=[
            dict(edge(g, values), downstream=t["id"]) for t in tasks for g in t.get("gated_on", [])
        ],
    )


def outcomes(work: Json) -> list[Json]:
    """A scheduler refusal accounts for its slot without inventing a scientific primary."""
    with patch.multiple(prior, TASKS=TASKS, SUFFIXES=SUFFIXES, PROTOCOL=PROTOCOL, ROOT=ROOT):
        rows = prior.outcomes(work)
    for i, row in enumerate(rows):
        row["primary_sha256"] = (
            row["reference"]["sha256"] if row["primary_status"] == "present" else None
        )
        row.update(
            experiment_id=9152 + i,
            quarantine="preserve_all_original_forward_and_judge_disqualifications"
            if i in {1, 2}
            else None,
        )
    return list(rows)
