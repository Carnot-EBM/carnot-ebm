"""REQ-REPORT-8551: authenticate terminal evidence without rerunning failed studies."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v733_capstone_evidence as old
from carnot.reporting import v734_contract_methods as methods

__all__ = ["methods"]
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import (
    read_bound_sidecar,
    reader_receipt,
    validate_primary,
)
from carnot.reporting.v734_capstone_samples import samples
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK, MILESTONE = "experiment_8551_v734_capstone", "exp8551-capstone", "2026.10.734"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v734_capstone_8551.py"
OWNED = [
    f"python/carnot/reporting/v734_capstone{s}.py" for s in ["_evidence", "_samples", "_report", ""]
] + [CLI]
MODEL_SPECS: list[Json] = []
PINS = dict(
    zip(
        [8540, 8541, 8542, 8545, 8546, 8547, 8548, 8549, 8550],
        [
            "f0fc7fee59e48f83f025e65682c8ec4f5f5f1bd5170e09689746341e61260a1d",
            "9321fc009824b58b7345df2212ea7d5ebef4cea6df0db7acc0b67359edb89d75",
            "99d05a8b090327b7e7725fadbf542e2a0d043aa6112b879f1c5acd70b6ea7f41",
            "5c4cef207e9972a6ec5d70347918c186dba832f0d45776d2df35167e0548955a",
            "97168fea70b2768e73c65517792204fd2e697ea58b68d36648377d20c3062eee",
            "50f2ce60532c23354b2c480e5cb3ce3e148b13b12f6bd4533fa49e3552e239c9",
            "ba5a834865487d4f75c427638aefe0beecb5316b5e321d03fe94f5b3c01ad942",
            "7cb7b1a21a2eb531dc34e91a97803a6150f5f27c0083d8a33eb6e59a492b7d5c",
            "45edf9a07d66b9e17fe5893b1c3c638096799aa6d9ab91344b176eef2c59c67e",
        ],
        strict=True,
    )
)
PREGATE_PATHS = {
    n: f"results/experiment_{n}_{s}.json"
    for n, s in [
        (8542, "canonical_update_accounting"),
        (8546, "complete_native_cost"),
        (8547, "qwen_evidence_canary"),
        (8549, "arc_capability_panel"),
    ]
}
checksum, reference, freeze, load, slot, memory = (
    old.checksum,
    old.reference,
    old.freeze,
    old.load,
    old.slot,
    old.memory,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase counts so bounded CPU children remain observable."""
    print(
        f"[exp8551] phase={phase} completed={min(completed, 12)} pending={min(pending, 12)}",
        flush=True,
    )


def gate(ref: Json, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Missing values retain exact paths and differ from measured zero."""
    return dict(old.gate(ref, field, expected, observed, upstream))


def worker_process(request: Path, output: Path) -> int:
    """A fresh sample process releases historical episode memory when it exits."""
    if json.loads(request.read_bytes()) != {"task_id": TASK}:
        raise ValueError("sample_request")
    progress("sample_worker_before", 0, 2)
    atomic_json(output, samples())
    progress("sample_worker_after", 2, 0)
    return 0


def sample_child(raw: Path) -> tuple[Json, Json]:
    """Bound one deterministic sample rather than retaining large episode operands in the parent."""
    request, output = raw / "sample_request.json", raw / "sample.json"
    atomic_json(request, dict(task_id=TASK))
    receipt = child(
        "sample_operands",
        [
            sys.executable,
            "-u",
            str(ROOT / CLI),
            "--worker-request",
            str(request),
            "--worker-output",
            str(output),
        ],
        raw / "sample_logs",
        deadline=180,
        heartbeat=30,
    )
    if not receipt["passed"]:
        raise ValueError("sample_child_failed")
    return load(reference(output)), receipt


def outcome(task: Json, primary: Json, raw: Path) -> Json:
    """Read the actual terminal interfaces; valid seals cannot clear producer failures."""
    number = int(task["id"][3:7])
    if primary["sha256"] != "sha256:" + PINS[number]:
        raise ValueError("primary_pin")
    value = load(primary)
    if value.get("blocked_at_layer") == "conductor_pre_gate":
        for g in value["gates_evaluated"]:
            if sha256_file(Path(g["artifact_path"])) != g["artifact_sha256"]:
                raise ValueError("pre_gate_operand_hash")
        with patch.dict(old.base.PINS, PINS):
            result = dict(old._OUTCOME(task, primary, raw))
        return dict(result, original_run_date=value.get("date"), evidence_references=[])
    validate_primary(value, Path(primary["path"]))
    if value["task_id"] != task["id"]:
        raise ValueError("task_identity")
    primitive = value.get("primitive_reference", value.get("work_reference"))
    if sha256_file(Path(primitive["path"])) != primitive["sha256"]:
        raise ValueError("primitive_hash")
    terminal_ref = reference(Path(value["terminal_validation_sidecar_path"]))
    terminal = load(terminal_ref)
    publication = terminal.get("publication", terminal)
    bound_ref = reference(Path(publication["sidecar_path"]))
    bound = read_bound_sidecar(Path(primary["path"]), Path(bound_ref["path"]))
    sealed = (
        publication["primary_sha256"] == primary["sha256"] == bound["primary_sha256"]
        and bound["report"]["passed"] is True
    )
    consumer = reader_receipt(
        task["id"],
        Path(primary["path"]).parent,
        field="required_checks_passed",
        expected=value["required_checks_passed"],
    )
    eligible = bool(
        sealed
        and consumer["passed"]
        and value["required_checks_passed"]
        and not value["flagged_adversarial"]
        and value["verdict_class"] != "disqualified"
    )
    return dict(
        producer_executed=True,
        eligible=eligible,
        selected={
            k: v
            for k, v in value.items()
            if k.endswith("_score")
            or k in {"physical_hardware_ready_score", "conditional_acceleration_ceiling"}
        },
        checks=[]
        if eligible
        else [
            gate(
                primary,
                "upstream_required_checks",
                True,
                dict(
                    sealed=sealed,
                    required_checks_passed=value["required_checks_passed"],
                    flagged_adversarial=value["flagged_adversarial"],
                ),
                task["id"],
            )
        ],
        disposition="qualified_producer" if eligible else "disqualified_producer",
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
        source_gates=value["gate_check_summary"],
        historical_model_provenance=value["historical_model_provenance"],
        upstream_adversarial_findings=value["adversarial_findings"],
        upstream_failed_receipts=[r for r in value["validation_receipts"] if not r["passed"]],
        upstream_flagged_adversarial=value["flagged_adversarial"],
        original_required_checks_passed=value["required_checks_passed"],
        original_run_date=value["run_date"],
        primitive_reference=primitive,
        evidence_references=[primitive, terminal_ref, bound_ref],
        downstream_consumer=consumer,
        branch_replay=dict(
            status="authenticated_terminal_and_primitive_seals", passed=bool(sealed)
        ),
    )


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate independent CPU resources and every scheduling slot before validation."""
    start, before = time.monotonic_ns(), memory()
    progress("preconditions", 0, 12)
    authority = methods.authority(root, raw / "authority")
    tasks = authority["tasks"] or methods.protocol()["task_map_v734"]
    if (
        methods.tasks_digest(tasks) != methods.TASKS_PIN
        or tasks[-1].get("requires")
        or tasks[-1].get("gated_on")
    ):
        raise ValueError("exact_unconditional_twelve_tasks")
    resource = dict(
        resources(private), optional_cuda_required=False, optional_boards_required=False
    )
    checks = list(authority["gate_check_summary"])
    if not (
        resource["disk_backed"]
        and resource["memory_available_bytes"] >= 1500 * 1024**2
        and resource["free_disk_bytes"] >= 1024**3
        and resource["tools_available"]
    ):
        checks.append(gate(dict(path=str(private)), "required_resources", True, resource))
    log = freeze(root / "ops/conductor-log.md", raw / "scheduling")
    lines = Path(log["snapshot_path"]).read_text().splitlines() if log["exists"] else []
    activation = next(
        (i for i, l in enumerate(lines) if "Milestone 2026.10.734 activated" in l), len(lines)
    )
    lines = lines[activation:]
    slots, refs = [], [log]
    for index, task in enumerate(tasks[:-1]):
        progress("branch_before_" + task["id"], index, 11 - index)
        if (time.monotonic_ns() - start) / 1e9 >= 3600:
            raise TimeoutError("measurement_deadline")
        number = int(task["id"][3:7])
        primary = freeze(root / task["deliverable"], raw / "primaries" / str(number))
        refs.append(primary)
        if (
            not primary["exists"]
            and number in PREGATE_PATHS
            and (root / PREGATE_PATHS[number]).is_file()
        ):
            primary = freeze(root / PREGATE_PATHS[number], raw / "pregates" / str(number))
            refs.append(primary)
        summary = outcome(task, primary, raw) if primary["exists"] else None
        cascade = next(
            (
                l
                for l in lines
                if "GATE_BLOCK" in l and l.split("|")[2].strip() == task["title"][:50].strip()
            ),
            None,
        )
        row = slot(task, primary, cascade, summary)
        slots.append(row)
        checks.extend(row["checks"])
        refs.extend(row.get("evidence_references", []))
        progress("branch_after_" + task["id"], index + 1, 10 - index)
    original = methods.protocol()["v734_capstone_reference"]
    predecessor = freeze(Path(original["path"]), raw / "predecessors")
    if predecessor["sha256"] != original["sha256"]:
        raise ValueError("original_v733_pin")
    progress("deterministic_sample_before", 11, 1)
    diagnostic, sample_receipt = sample_child(raw)
    if not diagnostic["learning"]["matches_original"] or not diagnostic["arc"]["matches_original"]:
        checks.append(gate(predecessor, "sample_reconciliation", True, False))
    progress("deterministic_sample_after", 12, 0)
    refs.extend(
        freeze(ROOT / n, raw / "code")
        for n in [
            *OWNED,
            TEST,
            methods.PROTOCOL,
            "python/carnot/reporting/primary_publication.py",
            "scripts/publication_gate.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "ops/exclusion_manifest.yaml",
            "docs/research-notes/v734-capstone.md",
        ]
    )
    after, end = memory(), time.monotonic_ns()
    work = dict(
        root=str(root),
        raw=str(raw),
        tasks=tasks,
        authority=authority,
        contract_digest=methods.TASKS_PIN,
        slots=slots,
        refs=refs,
        predecessors=[predecessor],
        branch_receipts=[sample_receipt],
        checks=checks,
        preconditions=resource,
        samples=diagnostic,
        memory=dict(
            before=before,
            after=after,
            peak_bound_mb=1500,
            growth_bound_mb=500,
            passed=after["peak_rss_mb"] <= 1500
            and after["current_rss_mb"] - before["current_rss_mb"] <= 500,
        ),
        phase_spans=[
            dict(
                name="terminal_evidence_reduction",
                started_monotonic_ns=start,
                ended_monotonic_ns=end,
            )
        ],
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        adversarial_findings=[],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def replay(path: Path) -> bool:
    """Rebuild rows, original seals and full authority so a repaired envelope cannot invent claims."""
    from carnot.reporting.v734_capstone_report import build

    try:
        value = json.loads(path.read_bytes())
        work = load(value["primitive_reference"])
        if (
            value["reproducibility_checksum"] != checksum(value)
            or value["task_count"] != 12
            or methods.tasks_digest(work["tasks"]) != methods.TASKS_PIN
            or work["contract_digest"] != methods.TASKS_PIN
        ):
            return False
        if (
            work["predecessors"][0]["sha256"]
            != methods.protocol()["v734_capstone_reference"]["sha256"]
        ):
            return False
        for ref in [*work["refs"], *work["predecessors"], *work.get("coverage_shards", [])]:
            if (
                ref.get("exists", True)
                and sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]
            ):
                return False
        with TemporaryDirectory(prefix="exp8551-authority-", dir="/var/tmp") as directory:
            cold = Path(directory)
            for role, name in [
                ("design", methods.DESIGN),
                ("active", methods.ACTIVE),
                ("staged", methods.STAGED),
            ]:
                ref = work["authority"]["authority_snapshots"][role]
                if ref["exists"]:
                    if sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]:
                        return False
                    target = cold / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ref["snapshot_path"], target)
            actual = methods.authority(cold, cold / "checked")
            if (
                actual["activated"] != work["authority"]["activated"]
                or actual["tasks"] != work["authority"]["tasks"]
            ):
                return False
        for row, task in zip(work["slots"], work["tasks"][:-1], strict=True):
            summary = (
                outcome(task, row["primary_reference"], Path(work["raw"]))
                if row["primary_reference"]["exists"]
                else None
            )
            if row != slot(task, row["primary_reference"], row["cascade_log_line"], summary):
                return False
        with TemporaryDirectory(prefix="exp8551-sample-", dir="/var/tmp") as directory:
            diagnostic, receipt = sample_child(Path(directory))
        if work["samples"] != diagnostic:
            return False
        for receipt in [*work["branch_receipts"], *value["validation_receipts"]]:
            for stream in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                    return False
        return bool(
            value
            == build(
                work,
                value["validation_receipts"],
                Path(work["raw"]),
                Path(value["publication_output"]),
            )
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
