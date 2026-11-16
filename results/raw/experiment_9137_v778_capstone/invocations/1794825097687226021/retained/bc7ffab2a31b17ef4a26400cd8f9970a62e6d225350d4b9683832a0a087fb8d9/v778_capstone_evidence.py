"""REQ-REPORT-9137-V778: retain immutable outcomes and undefined science bounds."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import resource
import shutil
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v777_capstone_evidence as prior
from carnot.reporting import v776_capstone_evidence as history
from carnot.reporting import v778_contract as contract
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.encoder_head_diagnosis_8972 import reference
from carnot.reporting.roadmap_contract import parse_design, TABLE_FIELDS
from carnot.reporting.v749_contract_methods import encode
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v764_capstone_sources import retain

Json = dict[str, Any]
ROOT = contract.ROOT
FROZEN = ROOT / "openspec/change-proposals/v778-capstone-inputs.json"
PROTOCOL = json.loads(FROZEN.read_bytes())
PINS, TASK_PIN = PROTOCOL["pins"], PROTOCOL["task_hash"]
TASKS = PROTOCOL["task_ids"]
SUFFIXES = [t.split("-", 1)[1] for t in TASKS]
project = deepcopy


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Reject bool readiness and name absence without turning it into measured zero."""
    row = contract.check(path, field, expected, observed, name)
    row["passed"] = row["passed"] and type(expected) is type(observed)
    return row


def edge(gate: Json, values: Json) -> Json:
    """Preserve the declared operator and the exact observed operand type."""
    row = prior.edge(gate, values)
    row["passed"] = row["passed"] and type(row["observed"]) is type(gate["value"])
    return row


def authenticate(root: Path, raw: Path) -> Json:
    """Authenticate pins and original retained producer bytes before dependent reduction."""
    checks, snapshots, values, operands = [], {}, {}, {}
    for name, pin in PINS.items():
        path = root / name
        row = check(
            path, "sha256", pin, reference(path)["sha256"] or "missing_file", "source_bytes"
        )
        checks.append(row)
        if row["passed"]:
            snapshots[name] = retain(path, raw, pin)
    for name, ref in snapshots.items():
        if not name.startswith("results/"):
            continue
        value = json.loads(Path(ref["path"]).read_bytes())
        number = int(Path(name).name.split("_")[1])
        identity = TASKS[number - 9124] if 9124 <= number < 9137 else f"prior:{number}"
        scheduler = value.get("blocked_at_layer") == "conductor_pre_gate"
        valid = (
            value.get("experiment") == number if scheduler else value.get("experiment_id") == number
        )
        checks.append(check(root / name, "producer_identity", True, valid, "producer_identity"))
        values[identity] = dict(value=value, reference=ref, selected_name=name)
    source = values.get(TASKS[0], {}).get("value", {})
    tasks = source.get("full_task_objects", [])
    bindings = [
        r
        for entry in values.values()
        for key, item in entry["value"].items()
        for r in (
            item
            if key
            in {
                "source_references",
                "raw_shard_hashes",
                "code_config_hashes",
                "current_work_receipts",
            }
            else [item]
            if key.endswith("_reference") and isinstance(item, dict)
            else []
        )
        if r.get("path") and r.get("sha256")
    ]
    for index, bound in enumerate(bindings):
        original = Path(bound["path"])
        if original.suffix in {".gguf", ".so"} or str(original) in operands:
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
                f"[exp9137] phase=authenticated_bindings completed={index} pending={len(bindings) - index}",
                flush=True,
            )
    for index in [0, 2, 8, 12]:
        entry = values.get(TASKS[index], {})
        checks.append(
            check(
                root / PROTOCOL["primary_paths"][index],
                "required_checks_passed",
                True,
                entry.get("value", {}).get(
                    "required_checks_passed", "missing_field" if entry else "missing_file"
                ),
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
    if tasks:
        checks.append(check(root, "task_hash", TASK_PIN, canonical_hash(tasks), "task_authority"))
        queue = yaml.safe_load(Path(snapshots["research-roadmap.yaml"]["path"]).read_bytes())
        table, machine = parse_design(
            Path(snapshots[contract.DESIGN]["path"]).read_text(), milestone="2026.11.778"
        )
        matched = len(tasks) == len(queue["tasks"]) == len(table) == len(machine) == 14 and all(
            q in (t, dict(t, prompt=encode(t["prompt"])))
            and t in (m, dict(m, prompt=encode(m["prompt"])))
            and all(t[k] == visible[k] for k in TABLE_FIELDS)
            for t, q, m, visible in zip(tasks, queue["tasks"], machine, table, strict=False)
        )
        checks.append(check(root, "full_task_objects", True, matched, "active_design_authority"))
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
            dict(edge(g, values), downstream=t["id"]) for t in tasks for g in t["gated_on"]
        ],
    )


def outcomes(work: Json) -> list[Json]:
    """Keep scheduler completion separate from missing canonical scientific primaries."""
    with patch.multiple(history, TASKS=TASKS, SUFFIXES=SUFFIXES):
        rows = history.outcomes(work)
    for i, row in enumerate(rows):
        row.update(experiment_id=9124 + i, primary_path=PROTOCOL["primary_paths"][i])
        row["quarantine"] = "preserve_disqualified_receipts" if i == 1 else None
    return list(rows)


def bounds() -> Json:
    """Retain two static and seven online conjuncts with undefined measurements."""
    static = [(c, "Brier_improvement", "lower>0") for c in ["probability_length", "combined_donor"]]
    online = [
        (c, "future_Brier_improvement", "lower>0")
        for c in [
            "frozen_spline",
            "bias_only_spline",
            "adaptive_linear",
            "donor_label_spline",
            "temperature_only_spline",
        ]
    ]
    online += [
        ("initial_state", "retained_Brier_increase", "upper<=0.005"),
        ("initial_state", "retained_error_increase", "upper<=0"),
    ]
    return {
        key: [
            dict(
                unit_id=f"{key}:{i + 1}",
                qid=None,
                group_id=None,
                sibling=None,
                arm="combined" if key == "two_bound_rows" else "adaptive_spline",
                condition=control,
                seed=777121 if key == "two_bound_rows" else 778135,
                metric=metric,
                criterion=criterion,
                absolute_metric=None,
                value=None,
                numerator=None,
                denominator=0,
                status="unmeasured",
                passed=False,
                censor_reason="missing_qualified_scientific_predictions",
                exclusion_reason=None,
            )
            for i, (control, metric, criterion) in enumerate(items)
        ]
        for key, items in [("two_bound_rows", static), ("seven_bound_rows", online)]
    }


def reduce(work: Json) -> Json:
    """Recompute support from retained rows and leave absent scientific comparisons unmeasured."""
    require(canonical_hash(work["tasks"]) == TASK_PIN, "task_authority")
    source = work["values"][TASKS[0]]["value"]
    methods = work["values"][TASKS[2]]["value"]
    online = work["values"][TASKS[8]]["value"]
    cost = work["values"][TASKS[12]]["value"]
    previous = deepcopy(source["preserved_results"])
    scientific = [r for r in cost["rows"] if r["scope"] == "science"]
    require(not any(r["status"] == "completed" for r in scientific), "unmeasured_cost")
    donors = online["donor_label_change_rows"]
    changed, support = sum(r["numerator"] for r in donors), sum(r["denominator"] for r in donors)
    require(
        support > 0 and changed / support == online["donor_changed_label_fraction"],
        "donor_aggregate",
    )
    result = dict(
        previous,
        **bounds(),
        gauge_control_rows=methods["gauge_control_rows"],
        donor_label_change_rows=donors,
        donor_changed_label_fraction=changed / support,
        hardware_limits=source["hardware_limits"],
        cost_summary=dict(
            rows=scientific,
            scientific_request_counts=dict(
                intended=len(scientific),
                attempted=sum(r["status"] != "unstarted" for r in scientific),
                completed=sum(r["status"] == "completed" for r in scientific),
                failed=sum(r["status"] == "failed" for r in scientific),
                censored=sum(r["status"] == "censored" for r in scientific),
                unstarted=sum(r["status"] == "unstarted" for r in scientific),
            ),
            measured_latency_s=None,
            energy_joules=None,
            amdahl_ceiling=None,
            complete_adaptation_rows=cost["complete_adaptation_rows"],
            hardware_operation_rows=cost["hardware_operation_rows"],
        ),
        continuation_decisions=[
            dict(branch=b, decision="terminal_blocked", next_prerequisite=p)
            for b, p in [
                (
                    "receipts",
                    "All original assertions and real successful orchestration with100 percent owned coverage; preserve Exp9125 disqualification.",
                ),
                (
                    "independent_human_confirmation",
                    "Descriptive controlled signal first; then genuinely independent matched-control human confirmation.",
                ),
                (
                    "embeddings",
                    "Separately qualified original-tolerance repair, including ragged masks, parity and offload.",
                ),
                (
                    "hardware",
                    "Operation-matched measured complete-service bottleneck and parity before any hardware work.",
                ),
                (
                    "static_energy",
                    "Qualified forward receipts, complete shards and both original static bounds plus defined accepted risk.",
                ),
                (
                    "online_energy",
                    "Qualified predictions, actual changed donor labels, all seven original bounds and exact recovery.",
                ),
                (
                    "ARC",
                    "Retain opportunistic evidence; no registered-level re-solving or new solve claim.",
                ),
            ]
        ],
    )
    with patch.multiple(prior, TASKS=TASKS, outcomes=outcomes):
        result["retirement_recommendations"] = prior.retirement(work, previous)
    return result
