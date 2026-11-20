"""REQ-REPORT-9268-V788: retain producer authority, V787 record, and explicit V788 outcomes."""

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

from carnot.reporting import historical_contract_inputs as _digest_cache
from carnot.reporting import v776_capstone_evidence as history
from carnot.reporting import v783_capstone_evidence as prior
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.encoder_head_diagnosis_8972 import reference
from carnot.reporting.v749_contract_methods import encode
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v778_capstone_evidence import edge as edge
from carnot.reporting.v788_capstone_reduction import reduce as reduce

Json = dict[str, Any]
ROOT = prior.ROOT
CONFIG = "openspec/change-proposals/v788-capstone-inputs.json"
PROTOCOL = json.loads((ROOT / CONFIG).read_bytes())
PINS, TASK_PIN, TASKS = PROTOCOL["pins"], PROTOCOL["task_hash"], PROTOCOL["task_ids"]
SUFFIXES = [t.split("-", 1)[1] for t in TASKS]
_DIGESTS = _digest_cache._DIGESTS
_digest = _digest_cache._digest
_PROJECTED_CACHE: dict[str, Json] = {}
IMPORTED = {
    "experiment_id",
    "experiment",
    "task_id",
    "honest_verdict",
    "verdict_class",
    "blocked_at_layer",
    "required_checks_passed",
    "current_work_receipts",
    "protocol_references",
    "protocol_owner_rows",
    "historical_code_resolutions",
    "historical_outcomes",
    "outcome_rows",
    "retirement_recommendations",
    "preserved_v782_accounting",
    "preserved_v783_accounting",
    "preserved_v784_accounting",
    "preserved_v785_accounting",
    "preserved_v785_outcomes",
    "previous_milestone_accounting",
    "current_milestone_accounting",
    "consumer_shard_rows",
    "full_task_objects",
    "censored_count",
    "preserved_v784",
    "preserved_v785",
    "preserved_v786",
    "preserved_v787",
    "rows",
    "target_identifiability_rows",
    "target_identifiability_summary",
    "effective_independent_groups",
    "scientific_effect_measured",
    "estimator_control_rows",
    "spline_control_rows",
    "coverage_control_rows",
    "edge_control_rows",
    "ca_ocrc_control_rows",
    "provenance_summary",
    "candidate_counts",
    "arena_sim_installed",
    "heldout_matches_played",
    "heldout_summary",
    "archive_versions",
    "label_ambiguity_protocol",
    "selective_deferral_protocol",
    "conformal_risk_protocol",
    "validation_obligations",
    "gates_evaluated",
}


def retain(path: Path, raw: Path, pin: str) -> Json:
    """Copy exact authenticated bytes without redundant whole-file memory copies."""
    require(_digest(path) == pin, "source_bytes:" + str(path))
    target = raw / "retained" / pin.split(":")[1] / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    resolved = target.resolve()
    stat = resolved.stat()
    _DIGESTS[(str(resolved), stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)] = pin
    return dict(reference(target), original_path=str(path))


def project(value: Json) -> Json:
    """Keep used operands small; retained complete producer bytes remain the authority."""
    result = {
        k: v
        for k, v in value.items()
        if k in IMPORTED or k.endswith(("_reference", "_ready_score"))
    }
    if isinstance(result.get("corpus_manifest_reference"), dict):
        result["corpus_manifest_reference"] = {
            k: v for k, v in result["corpus_manifest_reference"].items() if k != "ordered_paths"
        }
    gates = value.get("gate_check_summary", [])
    result["gate_check_summary"] = (
        gates
        if isinstance(gates, dict)
        else [r for r in gates if not isinstance(r, dict) or not r["passed"]]
    )
    if value.get("experiment_id") not in {9222, 9241}:
        result.pop("validation_obligations", None)
        result.pop("full_task_objects", None)
    return result


def load_projected(path: Path) -> Json:
    """Cache projected producer dictionaries by exact stat and digest identity."""
    digest = _digest(path)
    if digest not in _PROJECTED_CACHE:
        raw = path.read_bytes()
        i1 = raw.find(b'\n    "ordered_paths": [')
        i2 = raw.find(b'\n    "population_sha256":', i1 + 1) if i1 >= 0 else -1
        i3 = raw.find(b'\n  "scan_rows": [', i2 + 1) if i2 >= 0 else -1
        i4 = raw.find(b'\n  "schema":', i3 + 1) if i3 >= 0 else -1
        if i4 >= 0:
            raw = (
                raw[:i1]
                + b'\n    "ordered_paths": [],'
                + raw[i2:i3]
                + b'\n  "scan_rows": [],'
                + raw[i4:]
            )
        _PROJECTED_CACHE[digest] = project(json.loads(raw))
    return deepcopy(_PROJECTED_CACHE[digest])


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Strict operand types prevent absent evidence and false from becoming measured zero."""
    target = "20261120" if field == "run_date" else expected
    return dict(prior.check(path, field, target, observed, name), upstream=TASKS[-1])


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
            .split("## Planning verification", 1)[0]
            .splitlines()
            if line.startswith("| ") and line.split("|")[1].strip().isdigit()
        ]
        matched = (
            "**Milestone:** 2026.11.788" in text
            and len(tasks) == len(machine) == len(table) == 14
            and all(
                t in (m, dict(m, prompt=encode(m["prompt"])))
                and table[i] == [str(i + 1), str(t["phase"]), t["id"], t["deliverable"]]
                for i, (t, m) in enumerate(zip(tasks, machine, strict=False))
            )
        )
        checks.append(check(root, "full_contract", True, matched, "full_contract"))
    checks.append(check(root, "task_hash", TASK_PIN, canonical_hash(tasks), "task_authority"))
    for name, ref in snapshots.items():
        if name.startswith("results/"):
            value = load_projected(Path(ref["path"]))
            number = int(Path(name).name.split("_")[1])
            identity = TASKS[number - 9255] if 9255 <= number < 9268 else f"prior:{number}"
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
        if not (
            key == "work_reference" and entry["value"].get("experiment_id") in {9241, 9255, 9256}
        )
        for r in (
            item
            if key == "current_work_receipts"
            else list(item.values())
            if key == "protocol_references" and isinstance(item, dict)
            else [item]
            if key.endswith("_reference") and isinstance(item, dict)
            else []
        )
        if isinstance(r, dict) and r.get("path") and r.get("sha256")
    ]
    for owner_key in ["prior:9208", "prior:9222", "prior:9241", TASKS[0]]:
        for owner_row in values.get(owner_key, {}).get("value", {}).get("protocol_owner_rows", []):
            bindings.extend([owner_row["owning_configuration"], owner_row["sidecar"]])
    old = values.get("prior:9193", {}).get("value", {})
    for prior_key, include_operands in [
        ("prior:9193", True),
        ("prior:9207", False),
        ("prior:9221", False),
        ("prior:9240", False),
    ]:
        prior_val = values.get(prior_key, {}).get("value", {})
        if prior_val:
            ref = prior_val["work_reference"]
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
                bindings.extend(historical["snapshots"].values())
                if include_operands:
                    bindings.extend([*historical["operands"].values(), *historical["code"]])
    for index, bound in enumerate(bindings):
        original = Path(bound["path"])
        if str(original) in operands:
            continue
        candidate = root / original.relative_to(ROOT) if original.is_relative_to(ROOT) else original
        path = candidate if candidate.exists() or not root.exists() else original
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
        if index % 50 == 0:
            print(
                f"[exp9268] phase=authenticate completed={index} pending={len(bindings) - index}",
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
    """Scheduler pre-gate skips, upstream-retired log skips and primaries stay distinct."""
    with patch.multiple(history, TASKS=TASKS, SUFFIXES=SUFFIXES, edge=edge):
        rows = history.outcomes(work)
    for index, row in enumerate(rows):
        row.update(experiment_id=9255 + index, primary_path=PROTOCOL["primary_paths"][index])
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


def v787_outcomes(work: Json) -> list[Json]:
    """Reconstruct V787's seven queued tasks and seven omitted slots from retained authority."""
    queue_path = Path(
        work["snapshots"][
            "openspec/change-proposals/research-roadmap-v787-queue-preserved-20261120.yaml"
        ]["path"]
    )
    v787_tasks = yaml.safe_load(queue_path.read_bytes())["tasks"]
    v787_values = {
        "exp9241-contract": work["values"]["prior:9241"],
        "exp9242-corpus-prefilter": work["values"]["prior:9242"],
        "exp9244-conformal-risk-protocol": work["values"]["prior:9244"],
    }
    v787_ids = [*PROTOCOL["v787_task_ids"], "exp9248-capstone"]
    v787_suffixes = [t.split("-", 1)[1] for t in v787_ids]
    v787_work = dict(work, tasks=v787_tasks, values=v787_values)
    with patch.multiple(history, TASKS=v787_ids, SUFFIXES=v787_suffixes, edge=edge):
        rows = history.outcomes(v787_work)
    for index, row in enumerate(rows):
        row.update(
            experiment_id=9241 + index,
            primary_path=PROTOCOL["v787_primary_paths"][index],
        )
        if row["scheduler_alternative"]:
            scheduler = v787_values[row["task_id"]]["value"]
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
        elif row["primary_status"] == "missing_file":
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
                ROOT
                / PROTOCOL["v787_primary_paths"][
                    PROTOCOL["v787_task_ids"].index(operand["upstream"])
                ]
            )
    contract_ref = work["values"]["prior:9241"]["reference"]
    for slot in range(7, 14):
        rows.append(
            dict(
                task_id=f"v787-omitted-slot-{slot + 1}",
                v787_contract_unit_id=f"exp{9241 + slot}",
                slot_index=slot + 1,
                experiment_id=None,
                primary_path=None,
                primary_status="missing_field",
                reference=None,
                scheduler_alternative=None,
                disposition="omitted_from_v787_queue",
                scheduler_events=[],
                honest_verdict="complete_blocked_missing_task_in_active_queue",
                accounting_honest_verdict="complete_blocked_missing_task_in_active_queue",
                scheduler_honest_verdict=None,
                verdict_class="blocked",
                readiness={},
                failed_gate_value="missing_field",
                gate_check_summary=[],
                gate_operands=[],
                qualified=False,
                censor_reason="missing_task_in_active_queue",
                accounting_evidence_reference=contract_ref,
            )
        )
    return list(rows)
