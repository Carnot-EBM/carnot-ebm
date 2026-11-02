"""REQ-REPORT-8614: retain original producer and conductor authority.

A scheduling disposition closes a slot. It cannot qualify a measurement whose
original checks failed, or turn an absent producer into a measured zero.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import v739_contract_methods as methods
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v735_capstone_evidence import freeze
from scripts.conductor_gates import _find_artifact_by_task_id

__all__ = ["methods"]

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK, MILESTONE = "experiment_8614_v739_capstone", "exp8614-capstone", "2026.10.739"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_v739_capstone_8614.py"]
OWNED = [
    f"python/carnot/reporting/v739_capstone{s}.py" for s in ["_evidence", "_samples", "_report", ""]
] + [CLI]
MODEL_SPECS: list[Json] = []
reference, read, checksum, atomic_json = (
    methods.reference,
    methods.read,
    methods.checksum,
    methods.atomic_json,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual boundaries so a parent can supervise finite CPU work."""
    print(f"[exp8614] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Missing remains distinct from zero, with the original failing operand named."""
    return dict(methods.gate(path, field, expected, observed, upstream))


def resolve(task: Json, raw: Path, lines: list[str]) -> Json:
    """Both shipped consumers and terminal byte seals control producer eligibility."""
    selected = _find_artifact_by_task_id(task["id"], ROOT / "results")
    ref = freeze(selected or ROOT / task["deliverable"], raw / task["id"])
    cascade = next(
        (
            s
            for s in lines
            if "GATE_BLOCK" in s and s.split("|")[2].strip() == task["title"][:50].strip()
        ),
        None,
    )
    row: Json = dict(
        experiment_id=int(task["id"][3:7]),
        task_id=task["id"],
        unit_id=task["id"],
        primary_reference=ref,
        producer_executed=False,
        eligible=False,
        status="completed",
        censored=False,
        excluded=False,
        absolute_metric=0,
        effective_independent_groups=0,
        checks=[],
        evidence_references=[],
        conductor_receipt=cascade,
    )
    if not ref["exists"]:
        row.update(
            verdict_class="blocked",
            honest_verdict="complete_blocked_missing_primary",
            disposition="missing_primary_conductor_skip" if cascade else "missing_primary",
            missing_reason="producer_primary_absent",
            status="completed" if cascade else "missing",
        )
        row["checks"] = [gate(Path(ref["path"]), "producer_primary_exists", True, None, task["id"])]
        return row
    value = read(ref)
    row["original_honest_verdict"] = value["honest_verdict"]
    if value.get("blocked_at_layer") == "conductor_pre_gate":
        upstream = dict(path=value["failed_evidence_path"], sha256=value["failed_evidence_sha256"])
        read(upstream)
        row.update(
            verdict_class="blocked",
            honest_verdict="complete_blocked_conductor_pre_gate",
            disposition="pre_gate_receipt",
            missing_reason="producer_not_executed",
            evidence_references=[upstream],
        )
        row["checks"] = [
            gate(
                Path(upstream["path"]),
                value["failed_field"],
                value["failed_expected"],
                value["failed_observed"],
                value["failed_upstream"],
            )
        ]
        return row
    primitive = value.get("primitive_reference", value.get("work_reference"))
    read(primitive)
    terminal = reference(Path(value["terminal_validation_sidecar_path"]))
    side = read(terminal)
    publication = side.get("publication", side)
    bound_path = Path(publication.get("sidecar_path", terminal["path"]))
    sealed = False
    try:
        bound = read_bound_sidecar(Path(ref["path"]), bound_path)
        sealed = bound["primary_sha256"] == ref["sha256"] and bound["report"]["passed"] is True
    except (OSError, ValueError, KeyError):
        pass
    consumer = reader_receipt(
        task["id"],
        ROOT / "results",
        field="required_checks_passed",
        expected=value["required_checks_passed"],
    )
    gates = value["acceptance_gates"]
    gates_passed = (
        all(gates.values()) if isinstance(gates, dict) else all(r["passed"] for r in gates)
    )
    eligible = bool(
        sealed
        and consumer["passed"]
        and value["required_checks_passed"]
        and not value["flagged_adversarial"]
        and value["verdict_class"] not in {"blocked", "disqualified"}
        and gates_passed
    )
    row.update(
        producer_executed=True,
        eligible=eligible,
        verdict_class=value["verdict_class"],
        honest_verdict=value["honest_verdict"],
        disposition="qualified_producer"
        if eligible
        else "external_blocked_producer"
        if value["verdict_class"] == "blocked"
        else "disqualified_producer",
        missing_reason=None,
        acceptance_gates=gates,
        downstream_consumer=consumer,
        original_required_checks_passed=value["required_checks_passed"],
        upstream_flagged_adversarial=value["flagged_adversarial"],
        checks=value["gate_check_summary"],
        evidence_references=[primitive, terminal, reference(bound_path)],
        terminal_sealed=sealed,
    )
    if not sealed:
        row["checks"] = row["checks"] + [
            gate(Path(ref["path"]), "original_terminal_seal", True, False, task["id"])
        ]
    return row


def slots(raw: Path, log: Json) -> list[Json]:
    """Retain all twelve predecessor slots, including absent cascade artifacts."""
    lines = Path(log["snapshot_path"]).read_text().splitlines()
    rows = []
    for i, task in enumerate(methods.protocol()["task_map_v739"][:-1]):
        progress("slot_before_" + task["id"], i, 13 - i)
        rows.append(resolve(task, raw, lines))
        progress("slot_after_" + task["id"], i + 1, 12 - i)
    return rows


def sample(raw: Path) -> tuple[Json, Json]:
    """A fresh bounded process releases large historical allocations after reduction."""
    output = raw / "sample.json"
    receipt = child(
        "sample_operands",
        [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / CLI), "--sample-output", str(output)],
        raw / "logs",
        deadline=120,
    )
    if not receipt["passed"]:
        raise ValueError("sample_child_failed")
    return read(reference(output)), receipt


def measure(raw: Path, run_date: str) -> Json:
    """Authenticate required resources while optional hardware remains independent."""
    start = time.monotonic_ns()
    private = Path(tempfile.mkdtemp(prefix="exp8614-", dir="/var/tmp"))
    progress("preconditions_before", 0, 13)
    preconditions = resources(private)
    preconditions.update(
        optional_cuda_required=False,
        optional_boards_required=False,
        scratch_write_read_passed=(private / "write_probe").read_bytes() == b"private disk scratch",
    )
    failures = []
    if not (
        preconditions["disk_backed"]
        and preconditions["tools_available"]
        and preconditions["free_disk_bytes"] >= 1024**3
        and preconditions["memory_available_bytes"] >= 1500 * 1024**2
    ):
        failures.append(gate(private, "disk_memory_tools", True, False))
    contract = methods.authority(ROOT)
    log = freeze(ROOT / "ops/conductor-log.md", raw / "conductor")
    rows = slots(raw / "primaries", log)
    diagnostic, receipt = sample(raw / "samples")
    publication = child(
        "publication_gate",
        [str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"],
        raw / "publication",
        deadline=60,
    )
    end = time.monotonic_ns()
    if end - start >= 3600_000_000_000:
        raise TimeoutError("measurement_deadline")
    refs = (
        [log]
        + [r["primary_reference"] for r in rows]
        + [ref for r in rows for ref in r["evidence_references"]]
    )
    refs += [
        reference(ROOT / p)
        for p in [
            methods.PROTOCOL,
            "results/experiment_8601_v738_capstone.json",
            "ops/exclusion_manifest.yaml",
            "scripts/publication_gate.py",
            "scripts/experiment_template.py",
            "research-complete.yaml",
            "research-references.md",
            "openspec/change-proposals/research-roadmap-v738-preserved-20261017.md",
        ]
    ]
    work = dict(
        root=str(ROOT),
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
        protocol=methods.protocol(),
        contract=contract,
        conductor=log,
        slots=rows,
        preconditions=preconditions,
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
