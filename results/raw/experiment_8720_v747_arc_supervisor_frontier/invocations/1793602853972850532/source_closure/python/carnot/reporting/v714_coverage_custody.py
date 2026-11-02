"""REQ-REPORT-8262: bind current execution while preserving unmeasured science.

The frozen protocol fixes science choices. This module binds current producers
and their gate operands without turning historical missing work into null data.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import v685_authority_lifecycle as authority
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v710_contract_replay import snapshot

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8262_v714_coverage_custody"
TASK = "exp8262-coverage-custody"
MILESTONE = "2026.10.714"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_coverage_custody_8262.py"
OWNED = [
    "python/carnot/reporting/coverage_custody_8262.py",
    "python/carnot/reporting/v714_coverage_custody.py",
    "python/carnot/reporting/v714_coverage_runner.py",
    CLI,
]
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
ACTIVE = "research-roadmap.yaml"
STAGED = "research-roadmap-next.yaml"
HISTORY = "openspec/change-proposals/research-roadmap-v713-preserved-20261007.md"
PROTOCOL = "openspec/change-proposals/v713-evidence-intervention-protocol.json"
EXECUTION = "openspec/change-proposals/v714-evidence-execution-contract.json"
PIN = "sha256:f4c06e17a9f8eb72f1be80b265ab7e3507cbc5f6f3b918df6421fe49dae02018"
MODEL_SPECS: list[Json] = []
REUSED = [
    "python/carnot/reporting/v685_authority_lifecycle.py",
    "python/carnot/reporting/roadmap_contract.py",
    "python/carnot/reporting/v709_execution.py",
    "python/carnot/reporting/v710_contract_replay.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "scripts/experiment_template.py",
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
]


def failure(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Name each operand so missing evidence remains different from a measured zero."""
    return dict(
        upstream=path.stem,
        artifact_path=str(path),
        artifact_hash=sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def authority_work(root: Path, raw: Path) -> Json:
    """Use separate source bytes for table, machine design, staging and activation."""
    refs = [
        snapshot(root / name, raw / "inputs", role)
        for name, role in [
            (DESIGN, "design"),
            (STAGED, "staged"),
            (ACTIVE, "active"),
            (PROTOCOL, "protocol"),
            (HISTORY, "history"),
        ]
    ]
    for ref, fields in zip(
        refs,
        [
            ["tasks", "canonical_tasks_sha256", "visible_task_table"],
            ["milestone", "tasks"],
            ["milestone", "tasks"],
            ["terminal_bindings"],
            ["tasks", "visible_task_table"],
        ],
    ):
        ref["fields_imported"] = fields
    work: Json = dict(
        refs=refs,
        failures=[],
        tasks=[],
        execution_contract={},
        contract=dict(
            activated=False,
            planning_matched=False,
            canonical_tasks_sha256=None,
            contract_rows=[],
            authority_snapshots={},
            gate_check_summary=[],
        ),
    )
    try:
        paths = [
            Path(r.get("snapshot_path", raw / ("absent-" + str(i)))) for i, r in enumerate(refs)
        ]
        contract = authority.assess_authorities(
            *paths[:3], raw / "authority", milestone=MILESTONE, first_id=8262, count=14
        )
        _, tasks = parse_design(paths[0].read_text(), milestone=MILESTONE)
        if authority.tasks_digest(tasks) != contract["canonical_tasks_sha256"]:
            contract["activated"] = False
            contract["gate_check_summary"].append(
                failure(
                    root / DESIGN,
                    "full_machine_digest",
                    contract["canonical_tasks_sha256"],
                    authority.tasks_digest(tasks),
                )
            )
        work.update(contract=contract, tasks=tasks)
        work["failures"].extend(contract["gate_check_summary"])
        if refs[3]["sha256"] != PIN:
            work["failures"].append(
                failure(root / PROTOCOL, "science_protocol_sha256", PIN, refs[3]["sha256"])
            )
        work["execution_contract"] = dict(
            milestone=MILESTONE,
            science_protocol_path=PROTOCOL,
            science_protocol_sha256=PIN,
            canonical_tasks_sha256=contract["canonical_tasks_sha256"],
            producers=[
                dict(
                    task_id=t["id"],
                    producer_path=t["deliverable"],
                    current_gate_fields=t["gated_on"],
                )
                for t in tasks
            ],
            historical_task_numbers="provenance_only",
            science_choices="Unchanged roles, thresholds, costs, seeds and missing-view fallback.",
            producer_consumer_contract=dict(
                producer="Explicit coverage json -o operand from frozen validation argv.",
                durable_report="coverage/coverage.json",
                durable_receipt="coverage/coverage_command_receipt.json",
                consumer="Fresh process recomputes statements from saved code and durable JSON.",
                cleanup="Copy both primitives before scratch deletion; read only durable paths.",
            ),
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        work["failures"].append(failure(root / DESIGN, "authority_readable", True, str(error)))
    return work


def measure(root: Path, raw: Path) -> Json:
    """Authenticate required originals and qualified terminals before administrative work."""
    start, wall = time.monotonic_ns(), time.time_ns()
    work = authority_work(root, raw)
    work["history"] = []

    def bind(path: Path) -> Json:
        ref: Json = dict(snapshot(path, raw / "inputs", "input-" + str(len(work["refs"]))))
        ref["fields_imported"] = []
        work["refs"].append(ref)
        return ref

    try:
        protocol_ref = work["refs"][3]
        protocol = json.loads(Path(protocol_ref["snapshot_path"]).read_bytes())
        for binding in protocol["terminal_bindings"]:
            primary, sidecar, terminal = [
                root / Path(binding[k]).relative_to(ROOT)
                for k in ["primary", "sidecar", "terminal"]
            ]
            for path in [primary, sidecar, terminal]:
                ref = bind(path)
                ref["fields_imported"] = (
                    ["primary_sha256", "report.passed"]
                    if path == sidecar
                    else ["publication.primary_sha256"]
                    if path == terminal
                    else []
                )
                if not ref["exists"]:
                    work["failures"].append(failure(path, "exists", True, None))
            report = read_bound_sidecar(primary, sidecar)
            terminal_value = json.loads(terminal.read_bytes())
            if report["report"]["passed"] is not True or terminal_value["publication"][
                "primary_sha256"
            ] != sha256_file(primary):
                work["failures"].append(failure(sidecar, "qualified_terminal", True, False))
        _, old_tasks = parse_design((root / HISTORY).read_text(), milestone="2026.10.713")
        for task in old_tasks:
            path = root / task["deliverable"]
            if not path.is_file() and task["id"].startswith("exp8250-"):
                path = root / "results/experiment_8250_evidence_view_canary.json"
            ref = bind(path)
            source = json.loads(path.read_bytes()) if path.is_file() else {}
            ref["fields_imported"] = (
                [
                    "honest_verdict",
                    "verdict_class",
                    "terminal_validation_sidecar_path",
                    "intended_count",
                    "completed_count",
                    "failed_count",
                    "censored_count",
                    "excluded_count",
                ]
                if source
                else []
            )
            work["history"].append(
                dict(
                    task_id=task["id"],
                    path=str(path),
                    sha256=ref["sha256"],
                    disposition="unavailable_output"
                    if not source
                    else "conductor_pre_gate"
                    if source.get("honest_verdict", "").startswith("blocked_gate")
                    else "producer_terminal",
                    honest_verdict=source.get("honest_verdict"),
                    verdict_class=source.get("verdict_class"),
                    fields_imported=ref["fields_imported"],
                    source_counts={
                        k: source.get(k)
                        for k in [
                            "intended_count",
                            "completed_count",
                            "failed_count",
                            "censored_count",
                            "excluded_count",
                        ]
                    },
                )
            )
            if source.get("terminal_validation_sidecar_path"):
                terminal = Path(source["terminal_validation_sidecar_path"])
                terminal_ref = bind(terminal)
                terminal_value = json.loads(terminal.read_bytes())
                terminal_ref["fields_imported"] = (
                    ["primary_sha256", "report.passed"]
                    if "primary_sha256" in terminal_value
                    else ["publication.sidecar_path"]
                )
                sidecar = (
                    terminal
                    if "primary_sha256" in terminal_value
                    else Path(terminal_value["publication"]["sidecar_path"])
                )
                if sidecar != terminal:
                    bind(sidecar)["fields_imported"] = ["primary_sha256", "report.passed"]
                if read_bound_sidecar(path, sidecar)["report"]["passed"] is not True:
                    work["failures"].append(
                        failure(sidecar, "historical_terminal_binding", True, False)
                    )
        for name in ["ops/exclusion_manifest.yaml"]:
            ref = bind(root / name)
            if not ref["exists"]:
                work["failures"].append(failure(root / name, "exists", True, None))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        work["failures"].append(
            failure(root / PROTOCOL, "required_input_custody", True, str(error))
        )
    work.update(
        duration_s=(time.monotonic_ns() - start) / 1e9,
        clock=dict(
            started_monotonic_ns=start, ended_monotonic_ns=time.monotonic_ns(), started_wall_ns=wall
        ),
    )
    return work


def reduce(work: Json, receipts: list[Json], coverage_ok: bool, cold_ok: bool) -> Json:
    """Keep independently tested execution readiness separate from science benefit."""
    passed_names = {r["name"] for r in receipts if r["passed"] and r["exit_code"] == 0}
    owned = bool(receipts) and all(r["passed"] for r in receipts) and coverage_ok and cold_ok
    contract_ok = (
        owned and "current_contract_tests" in passed_names and work["contract"]["activated"]
    )
    custody_ok = owned and "coverage_custody_tests" in passed_names
    failed = [*work["failures"], *work.get("owned_failures", [])]
    verdict = (
        "blocked"
        if failed and not receipts
        else "disqualified"
        if not owned
        else "blocked"
        if failed
        else "null"
    )
    suffix = "required_checks" if not owned else "coverage_custody"
    if verdict == "blocked":
        suffix = Path(failed[0].get("artifact_path", failed[0].get("path", "operand"))).stem
    rows = []
    for index in range(14):
        primitive = (
            work["contract"]["contract_rows"][index]
            if index < len(work["contract"]["contract_rows"])
            else {}
        )
        checks = primitive.get("checks", {})
        present = bool(primitive) and primitive["status"] == "completed"
        rows.append(
            dict(
                unit_id=f"exp{8262 + index}",
                source_cluster_id="current_authority",
                arm="current_contract",
                condition="full_task_agreement",
                metric="contract_agreement",
                status="completed" if present else "censored",
                completed=present,
                failed=present and not all(checks.values()),
                censored=not present,
                excluded=False,
                seed=None,
                numerator=sum(checks.values()) if present else None,
                denominator=len(checks),
                independent_source_count=0,
            )
        )
    return dict(
        experiment_id=8262,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261008",
        honest_verdict="complete_" + verdict + "_" + suffix,
        verdict_class=verdict,
        gate_check_summary=failed,
        rows=rows,
        intended_count=14,
        completed_count=sum(r["completed"] for r in rows),
        failed_count=sum(r["failed"] for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        verifier_is_oracle=False,
        exposure_scope="exposed_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=False,
        coverage_custody_ready_score=int(custody_ok),
        current_contract_ready_score=int(contract_ok and not failed),
        acceptance_gates=dict(
            owned_checks=owned,
            coverage_custody=custody_ok,
            current_contract=contract_ok and not failed,
            scientific_benefit=False,
        ),
        canonical_tasks_sha256=work["contract"]["canonical_tasks_sha256"],
        task_contract=work["tasks"],
        historical_dispositions=work["history"],
        staged_readiness=work["contract"]["planning_matched"],
        activated_readiness=work["contract"]["activated"],
        H1=dict(measured_here=False),
        H2=dict(measured_here=False),
    )
