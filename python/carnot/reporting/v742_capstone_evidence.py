"""REQ-REPORT-8652: keep producer identity separate from scheduling receipts."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v741_capstone_inputs as original
from carnot.reporting import v742_contract_methods as methods
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.v709_execution import child
from scripts.recurring_blocker_ledger import normalise

Json = dict[str, Any]
__all__ = ["methods"]
ROOT: Path = methods.ROOT
NAME, TASK, MILESTONE = "experiment_8652_v742_capstone", "exp8652-capstone", "2026.10.742"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_v742_capstone_8652.py"]
OWNED = [
    f"python/carnot/reporting/v742_capstone{s}.py" for s in ["_evidence", "_samples", "_report", ""]
] + [CLI]
MODEL_SPECS: list[Json] = []
reference, read, checksum, atomic_json = (
    methods.reference,
    methods.read,
    methods.checksum,
    methods.atomic_json,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so CPU work remains visible to the supervisor."""
    print(f"[exp8652] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(path: Path, raw: Path) -> Json:
    """Save exact source bytes; a missing input remains missing rather than zero."""
    ref: Json = dict(path=str(path.absolute()), exists=path.is_file(), sha256=None)
    if path.is_file():
        ref.update(reference(path))
        target = raw / ref["sha256"].split(":")[-1] / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError("immutable_snapshot_changed")
        shutil.copyfile(path, target)
        ref["snapshot_path"] = str(target)
    return ref


def locate(task: Json) -> Path:
    """Only the declared name or the conductor's exact task slug can identify this slot."""
    declared = ROOT / task["deliverable"]
    number, slug = task["id"].split("-", 1)
    fallback = (
        ROOT / "results" / ("experiment_" + number[3:] + "_" + slug.replace("-", "_") + ".json")
    )
    candidates = list((ROOT / "results").glob("experiment_" + number[3:] + "_*.json"))
    if len(candidates) > 1 or any(path not in {declared, fallback} for path in candidates):
        raise ValueError("unexpected_primary_identity")
    return Path(declared if declared.is_file() or not fallback.is_file() else fallback)


def failure_evidence(value: Json) -> list[Json]:
    """Original failed child exits and exception lines distinguish changes in failure cause."""
    result = []
    for receipt in value.get("validation_receipts", []):
        if receipt.get("passed"):
            continue
        logs = [
            reference(Path(receipt[k]))
            for k in ["stdout_path", "stderr_path"]
            if k in receipt and Path(receipt[k]).is_file()
        ]
        lines = [
            line.strip()
            for ref in logs
            for line in Path(ref["path"]).read_text(errors="replace").splitlines()
            if line.startswith(("E ", "FAILED ", "ERROR "))
        ]
        shape = (
            "owned_child_killed"
            if (receipt.get("exit_code") or 0) < 0
            else " ".join(
                sorted(
                    {
                        match
                        for line in lines
                        for match in re.findall(r"\b\w*(?:Error|Exception)\b", line)
                    }
                )
            )
            or receipt["name"]
        )
        result.append(
            dict(
                name=receipt["name"],
                exit_code=receipt.get("exit_code"),
                timed_out=receipt.get("timed_out"),
                argv=receipt.get("argv", receipt.get("command")),
                duration_s=receipt.get("duration_s"),
                logs=logs,
                failure_lines=sorted(set(lines))[:20],
                normalized_underlying_cause=normalise(shape),
            )
        )
    return result


def slots(raw: Path, log: Json) -> list[Json]:
    """Resolve exact producer paths and authenticated conductor fallback dispositions."""
    lines = Path(log["snapshot_path"]).read_text().splitlines()
    rows = []
    with patch.object(original, "methods", methods), patch.object(original, "locate", locate):
        for i, task in enumerate(methods.protocol()["task_map_v742"][:-1]):
            progress("slot_before_" + task["id"], i, 13 - i)
            declared = freeze(ROOT / task["deliverable"], raw / "declared")
            row = original.resolve(task, raw, lines)
            row["declared_reference"] = declared
            row["checks"] = list(row["checks"])
            if row["producer_executed"]:
                value = read(row["primary_reference"])
                row["failed_owned_checks"] = failure_evidence(value)
                row["evidence_references"] += [
                    freeze(Path(ref["path"]), raw / "failed_children")
                    for failure in row["failed_owned_checks"]
                    for ref in failure["logs"]
                ]
                gates = value["acceptance_gates"]
                gate_items = (
                    gates.items()
                    if isinstance(gates, dict)
                    else ((r["name"], r["passed"]) for r in gates)
                )
                row["checks"] += [
                    methods.gate(
                        Path(row["primary_reference"]["path"]),
                        "acceptance_gates." + str(name),
                        True,
                        passed,
                        task["id"],
                    )
                    for name, passed in gate_items
                    if not passed
                ]
                if (
                    value["verdict_class"] == "blocked"
                    and value["required_checks_passed"]
                    and not value["flagged_adversarial"]
                ):
                    row["disposition"] = "qualified_blocked_reconciliation"
                if value["flagged_adversarial"]:
                    row["checks"].append(
                        methods.gate(
                            Path(row["primary_reference"]["path"]),
                            "flagged_adversarial",
                            False,
                            True,
                            task["id"],
                        )
                    )
            if not declared["exists"]:
                row["checks"].append(
                    methods.gate(
                        Path(declared["path"]), "declared_producer_exists", True, None, task["id"]
                    )
                )
            rows.append(row)
            progress("slot_after_" + task["id"], i + 1, 12 - i)
    return rows


def bindings() -> list[Json]:
    """Keep actual scalar values and distinguish absent paths from absent fields."""
    with patch.object(original, "methods", methods), patch.object(original, "locate", locate):
        rows = list(original.bindings(methods.protocol()["task_map_v742"]))
    for task in methods.protocol()["task_map_v742"][:-1]:
        declared = ROOT / task["deliverable"]
        if not declared.is_file():
            rows.append(
                dict(
                    methods.gate(declared, "declared_producer_exists", True, None, task["id"]),
                    downstream=TASK,
                    operand_state="missing_path",
                    edge_kind="declared_producer_path",
                )
            )
    return rows


def sample(raw: Path) -> tuple[Json, Json]:
    """Release numerical allocations after the bounded reduction child exits normally."""
    output = raw / "samples.json"
    receipt = child(
        "sample_operands",
        [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / CLI), "--sample-output", str(output)],
        raw / "logs",
        deadline=180,
    )
    if not receipt["passed"]:
        raise ValueError("sample_child_failed")
    return read(reference(output)), receipt


def measure(raw: Path, run_date: str) -> Json:
    """Optional devices cannot suppress this independent evidence reconciliation."""
    start = time.monotonic_ns()
    progress("preconditions_before", 0, 13)
    private = Path(tempfile.mkdtemp(prefix="exp8652-", dir="/var/tmp"))
    pre = resources(private)
    pre.update(
        optional_cuda_required=False,
        optional_boards_required=False,
        scratch_write_read_passed=(private / "write_probe").read_bytes() == b"private disk scratch",
    )
    failures = []
    if not (
        pre["disk_backed"]
        and pre["tools_available"]
        and pre["scratch_write_read_passed"]
        and pre["free_disk_bytes"] >= 1024**3
        and pre["memory_available_bytes"] >= 1500 * 1024**2
    ):
        failures.append(methods.gate(private, "disk_memory_tools", True, False, TASK))
    protocol, contract = methods.protocol(), methods.authority(ROOT)
    log = freeze(ROOT / "ops/conductor-log.md", raw / "conductor")
    rows = slots(raw / "primaries", log)
    diagnostic, receipt = sample(raw / "samples")
    publication = child(
        "publication_gate",
        [str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"],
        raw / "publication",
        deadline=60,
    )
    refs = [log] + [r[k] for r in rows for k in ["primary_reference", "declared_reference"]]
    refs += [ref for r in rows for ref in r["evidence_references"]]
    refs += [freeze(Path(ref["path"]), raw / "operands") for ref in diagnostic["references"]]
    refs += [
        freeze(ROOT / p, raw / "source_inputs")
        for p in [
            methods.PROTOCOL,
            methods.DESIGN,
            methods.ACTIVE,
            "results/experiment_8639_v741_capstone.json",
            "ops/exclusion_manifest.yaml",
            "scripts/publication_gate.py",
            "scripts/conductor_gates.py",
            "scripts/recurring_blocker_ledger.py",
            "scripts/experiment_template.py",
            "research-references.md",
            "research-studying.md",
            "_bmad/prd.md",
            "ops/verifier_gaps.md",
            "python/carnot/inference/resident_energy.py",
            "python/carnot/verify/source_alignment.py",
            "python/carnot/reporting/arc_reset_8649.py",
            "python/carnot/reporting/arc_reset_transfer_8650.py",
            "python/carnot/reporting/v741_capstone_inputs.py",
            "ops/publication_gate_state.json",
            "docs/technical-report.md",
            "docs/arxiv-paper/main.tex",
        ]
    ]
    end = time.monotonic_ns()
    if end - start >= 3600_000_000_000:
        raise TimeoutError("measurement_deadline")
    work = dict(
        raw=str(raw),
        run_date=run_date,
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(
                phase="authentication_and_reduction",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
        protocol=protocol,
        contract=contract,
        conductor=log,
        slots=rows,
        declared_gates=bindings(),
        preconditions=pre,
        precondition_failures=failures,
        branch_receipts=[receipt],
        publication_receipt=publication,
        publication=json.loads(Path(publication["stdout_path"]).read_bytes()),
        samples=diagnostic,
        source_refs=refs,
        code_refs=[reference(ROOT / p) for p in OWNED + TESTS],
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", 12, 1)
    return work
