"""REQ-REPORT-8538: complete operation accounting retains blocked physical deployment."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting import nine_input_board_evidence_8538 as evidence
from carnot.reporting import v733_contract_methods as authority
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json

Json = dict[str, Any]
ROOT, base, reference, checksum = (
    evidence.ROOT,
    evidence.base,
    evidence.reference,
    evidence.checksum,
)
COST, OPERATIONS, COST_OPERATIONS, ceiling, costs = (
    evidence.COST,
    evidence.OPERATIONS,
    evidence.COST_OPERATIONS,
    evidence.ceiling,
    evidence.costs,
)
NAME, TASK, MILESTONE = (
    "experiment_8538_v733_nine_input_board_boundary",
    "exp8538-nine-input-board-boundary",
    "2026.10.733",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_nine_input_board_boundary_8538.py"
OWNED = [
    "python/carnot/reporting/nine_input_board_evidence_8538.py",
    "python/carnot/reporting/nine_input_board_boundary_8538.py",
    "python/carnot/reporting/nine_input_board_runner_8538.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts because no board availability probe is part of this work."""
    print(f"[exp8538] phase={phase} completed={completed} pending={pending}", flush=True)


def operands(root: Path) -> Json:
    """Read independent original seals unconditionally, even when optional costs are absent."""
    return dict(history=evidence.history(root), cost=costs(root))


def measure(root: Path, raw: Path, private: Path | None = None) -> Json:
    """Seal current task admission and bounded disk-backed evidence before validation."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    progress("preconditions_and_current_task")
    resources = evidence.resources(private or raw)
    contract = authority.authority(root, raw / "authority")
    progress("supplied_evidence_scan_before", 0, 2)
    scan = time.monotonic_ns()
    observed = operands(root)
    duration = (time.monotonic_ns() - scan) / 1e9
    progress("supplied_evidence_scan_after", 2, 0)
    checks = [
        evidence.failure(raw, "supplied_evidence_scan_within_60s", True, duration <= 60, TASK)
    ]
    checks += [
        evidence.failure(
            raw,
            "disk_backed_writable_scratch_and_tools",
            True,
            bool(
                resources["disk_backed"]
                and resources["tools_available"]
                and resources["memory_available_bytes"] >= 1024**3
                and resources["free_disk_bytes"] >= 1024**3
            ),
            TASK,
        )
    ]
    checks += [
        evidence.failure(
            root / authority.ACTIVE,
            "current_full_task_authority",
            True,
            contract["activated"],
            TASK,
        )
    ]
    work: Json = dict(
        root=str(root.absolute()),
        evidence=observed,
        contract=dict(
            activated=contract["activated"],
            task_sha256=authority.tasks_digest([t for t in contract["tasks"] if t["id"] == TASK]),
            canonical_tasks_sha256=contract["canonical_tasks_sha256"],
        ),
        checks=checks,
        preconditions=resources,
        scan_duration_s=duration,
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=[dict(name="supplied_evidence_scan", duration_s=duration)],
        refs=[*observed["history"]["refs"], *observed["cost"]["refs"]],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "scripts/experiment_template.py",
                "ops/exclusion_manifest.yaml",
                authority.PROTOCOL,
                authority.METHODS,
                "docs/research-notes/v733-nine-input-board-boundary.md",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Mapping readiness and qualified CPU dispatch never create missing fabric operations."""
    history, cost = work["evidence"]["history"], work["evidence"]["cost"]
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    ready = (
        owned
        and history["qualified"]
        and work["contract"]["activated"]
        and all(g["passed"] for g in work["checks"])
    )
    klass = "blocked" if owned else "disqualified"
    fields = history["fields"]
    operation_rows = [
        dict(
            board=board,
            operation=op,
            assigned_substrate="host_CPU",
            device_supported=False,
            deployment_ready=False,
            input_count=9,
            cost_ns=sum(r["end_ns"] - r["start_ns"] for r in cost["spans"] if r["operation"] == op)
            if cost["qualified"]
            else None,
            transport_clock_ns=None,
            device_clock_ns=None,
            missing_reason="no_installed_compatible_operation",
            board_cpu_scope="historical_Linux_CPU_dispatch_only" if board == "PolarFire" else None,
        )
        for board in ["KV260", "PolarFire", "GateMate"]
        for op in OPERATIONS
    ]
    rows = [
        dict(
            source_id=f"{r['board']}:{r['operation']}",
            arm="operation_boundary",
            intended=1,
            completed=True,
            failed=False,
            censored=False,
            excluded=False,
            independent=0,
            numerator=1,
            denominator=1,
            missing_reason=None,
            operation=r["operation"],
            board=r["board"],
        )
        for r in operation_rows
    ]
    gates = [
        *work["checks"],
        *history["checks"],
        *cost["checks"],
        evidence.failure(
            Path(work["root"]) / "hardware/nine_input_kernel.json",
            "deployment_ready",
            True,
            False,
            "installed_board_operations",
        ),
    ]
    value: Json = dict(
        experiment_id=8538,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261015",
        honest_verdict=f"complete_{klass}_v733_nine_input_board_boundary",
        verdict_class=klass,
        gate_check_summary=gates,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, current_llm_invocations=0),
        historical_model_provenance=history.get("models", []),
        rows=rows,
        intended_count=len(rows),
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            operation_mappings=21, historical_readers=2, independent_deployment_samples=0
        ),
        verifier_is_oracle=False,
        exposure_scope="historical_exposed_evidence_engineering_accounting_only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates=dict(
            owned_validation=owned,
            original_readers=history["qualified"],
            current_task_authority=work["contract"]["activated"],
            compatible_installed_operations=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=7338538,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work.get("coverage_shards", []),
        work_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        cited_upstream_artifacts=history["readers"],
        board_boundary_recorded_score=int(ready),
        board_operation_rows=operation_rows,
        nine_input_cost_reference=cost,
        conditional_acceleration_ceiling=cost["conditional_ceiling"],
        unsupported_operations=OPERATIONS,
        kv260_synthesis_succeeded=fields.get("kv260_synthesis_succeeded"),
        polarfire_workload_validated=fields.get("polarfire_workload_validated", False),
        polarfire_graduation=fields.get("polarfire_graduation"),
        kv260_supported_operations=fields.get("kv260_supported_operations", []),
        gatemate_bitstream_flashed=False,
        gatemate_obligation_rows=fields.get("gatemate_obligation_rows", []),
        missing_source_hashes=fields.get("missing_source_hashes", []),
        last_authenticated_idcode=fields.get("last_authenticated_idcode"),
        reopening_condition=fields.get("reopening_condition", []),
        device_commands_executed=[],
        deployment_ready=False,
        NPU_qualified=False,
        TSU_qualified=False,
        newly_owned_coverage=work.get("owned_coverage"),
        evidence_delta_scan_limit_s=60,
        evidence_delta_scan_duration_s=work["scan_duration_s"],
        future_dependencies=[
            dict(
                experiment_id=8534,
                status="qualified_current_evidence"
                if cost["qualified"]
                else "future_dependency_not_observation",
            )
        ],
        acquisition_decision=dict(
            purchase_authorized=False,
            recommended=False,
            measured_dominant_span=None
            if not cost["qualified"]
            else max(cost["spans"], key=lambda r: r["end_ns"] - r["start_ns"])["operation"],
            desired_hardware_can_change_dominant_span=None,
            missing_reason="no_qualified_cost_and_no_compatible_installed_kernel",
        ),
        literature_operation_mapping=[
            dict(
                source="https://arxiv.org/abs/2602.02056",
                operations=["cubic_residual", "sparse_local_spline_update"],
                installed=False,
            ),
            dict(
                source="https://arxiv.org/abs/2607.21077",
                operations=["quadratic_Ising_energy", "memory_pipeline"],
                installed_nine_input_kernel=False,
            ),
            dict(
                source="https://extropic.ai/writing/z1t/",
                operations=["probabilistic_sampling"],
                vendor_context_only=True,
            ),
        ],
        methodology_note="Read original independently sealed readers and primitive hashes without device commands. Map nine-input host operations; optional disjoint clocks cannot install hardware. No imported five-input speed, Rust/Python NFR-01, hidden-game generalization or learning benefit is claimed.",
    )
    value["field_principles"] = {
        k: "Bind exact invocation and original producer hashes; accounting grants no deployed operation or scientific benefit."
        for k in value
    }
    value["field_principles"]["field_principles"] = (
        "Describe every reported field without changing historical authority."
    )
    value["field_principles"]["reproducibility_checksum"] = (
        "Cold replay reconstructs operands independently of this summary hash."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh reductions defeat rehashed summaries and self-consistently changed primitive reports."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            base.checked(ref)
        if operands(Path(work["root"])) != work["evidence"]:
            return False
        rebuilt = build(
            work,
            value["validation_receipts"],
            Path(value["work_reference"]["path"]).parent,
            Path(value["publication_output"]),
        )
        return bool(rebuilt == value)
    except (OSError, ValueError, KeyError, TypeError):
        return False
