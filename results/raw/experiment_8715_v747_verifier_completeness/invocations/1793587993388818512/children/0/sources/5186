"""REQ-REPORT-8577: resident accounting grants no installed hardware or learning credit."""

from __future__ import annotations

import json
from pathlib import Path
import signal
import time
from typing import Any

from carnot.reporting import resident_hardware_evidence_8577 as evidence
from carnot.reporting import v736_contract_methods as authority
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.hardware_operation_boundary_8563 import node_union

Json = dict[str, Any]
ROOT, reference, checksum = authority.ROOT, authority.reference, authority.checksum
NAME, TASK, MILESTONE = (
    "experiment_8577_v736_resident_hardware_boundary",
    "exp8577-resident-hardware-boundary",
    "2026.10.736",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_resident_hardware_boundary_8577.py"
OWNED = [
    f"python/carnot/reporting/resident_hardware_{s}_8577.py"
    for s in ["evidence", "boundary", "runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Visible phase counts let the supervisor detect a stalled evidence read."""
    print(f"[exp8577] phase={phase} completed={completed} pending={pending}", flush=True)


def operands(root: Path) -> Json:
    """The unconditional board authentication has a real60-second wall deadline."""

    def timeout(signum: int, frame: Any) -> None:
        raise TimeoutError("evidence_scan_60_seconds")

    handler = signal.signal(signal.SIGALRM, timeout)
    previous = signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        return dict(
            history=evidence.history(root),
            cost=evidence.costs(root),
            authority=authority.authority(root),
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
        signal.signal(signal.SIGALRM, handler)


def measure(root: Path, raw: Path, private: Path | None = None) -> Json:
    """Seal invocation resources and original operands before any validation children run."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    pre = resources(private or raw)
    progress("authentication_before", 0, 3)
    observed = operands(root)
    progress("authentication_after", 3, 0)
    work = dict(
        root=str(root.absolute()),
        evidence=observed,
        preconditions=pre,
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=[dict(name="authentication", duration_s=(time.monotonic_ns() - start) / 1e9)],
        refs=[
            *observed["history"]["refs"],
            *observed["cost"]["refs"],
            *observed["authority"]["references"],
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "ops/exclusion_manifest.yaml",
                "scripts/experiment_template.py",
                authority.PROTOCOL,
                "docs/research-notes/v736-resident-hardware-boundary.md",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Scope readiness can complete while missing resident measurements keep benefit blocked."""
    history, cost, contract = (work["evidence"][k] for k in ["history", "cost", "authority"])
    union = node_union(raw, receipts)
    owned = bool(receipts) and all(r["passed"] for r in receipts) and union["passed"]
    pre = work["preconditions"]
    resource_ok = (
        pre["disk_backed"]
        and pre["tools_available"]
        and min(pre["free_disk_bytes"], pre["memory_available_bytes"]) >= 1024**3
    )
    ready = owned and history["qualified"] and resource_ok
    fields = history["fields"]
    operation_rows = [
        dict(
            board=b,
            operation=op,
            assigned_substrate="host_CPU",
            device_supported=False,
            device_clock_ns=None,
            transport_clock_ns=None,
            resident_clock_ns=None,
            coefficient_count=109,
            input_count=9,
            board_cpu_scope="historical_Linux_CPU_only" if b == "PolarFire" else None,
            missing_reason="no_qualified_resident_clock_or_compatible_device_execution",
        )
        for b in evidence.BOARDS
        for op in evidence.OPERATIONS
    ]
    klass = "blocked" if owned else "disqualified"
    gates = [
        *history["checks"],
        *cost["checks"],
        *contract["gate_check_summary"],
        evidence.failure(
            Path(work["root"]), "disk_backed_scratch_memory_tools", True, bool(resource_ok), TASK
        ),
        evidence.failure(
            Path(work["root"]) / "hardware/resident_execution.json",
            "compatible_physical_execution",
            True,
            None,
            "installed_hardware",
        ),
    ]
    rows = [
        dict(
            source_id=f"{r['board']}:{r['operation']}",
            arm="operation_mapping",
            completed=True,
            failed=False,
            censored=False,
            excluded=False,
            independent=0,
            board=r["board"],
            operation=r["operation"],
            missing_reason=r["missing_reason"],
        )
        for r in operation_rows
    ]
    value: Json = dict(
        experiment_id=8577,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261016",
        honest_verdict=f"complete_{klass}_v736_resident_hardware_boundary",
        verdict_class=klass,
        gate_check_summary=gates,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, current_llm_invocations=0),
        historical_model_provenance=fields.get("historical_model_provenance", []),
        rows=rows,
        intended_count=len(rows),
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(operation_mappings=50, independent_hardware_samples=0),
        verifier_is_oracle=False,
        exposure_scope="exposed_historical_engineering_evidence",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates=dict(
            owned_validation=owned,
            original_V735_board=history["qualified"],
            current_authority=contract["ready"],
            resident_cost=cost["qualified"],
            physical_execution=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=pre,
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=7368577,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work.get("coverage_shards", []),
        work_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        cited_upstream_artifacts=history["refs"][:2],
        hardware_boundary_ready_score=int(ready),
        board_boundary_recorded_score=int(ready),
        physical_hardware_ready_score=0,
        operation_coverage_rows=operation_rows,
        qualified_cost_reference=cost,
        conditional_ceiling_rows=evidence.ceilings(cost),
        vendor_exclusion_rows=fields.get("vendor_exclusion_rows", []),
        board_reopen_requirements=fields.get("board_reopen_requirements", []),
        kv260_supported_operations=fields.get("kv260_supported_operations", []),
        polarfire_graduation=fields.get("polarfire_graduation"),
        newly_owned_coverage=work.get("owned_coverage"),
        exact_union_receipt=union,
        device_commands_executed=[],
        NPU_qualified=False,
        TSU_qualified=False,
        cold_boundary_reference=dict(
            path=str(Path(work["root"]) / evidence.BOARD),
            sha256=evidence.BOARD_PIN,
            scope="cold64_answer_panel_not_resident_or_durable_learning",
            ceilings=fields.get("conditional_ceiling_rows", []),
        ),
        acquisition_decision=dict(
            recommended=False,
            purchase_authorized=False,
            existing_CPU_residence_removes_startup_bottleneck=None,
            reason="Measure qualified CPU residence before acquisition; cold startup dominates but resident benefit is unmeasured.",
            required_evidence=[
                "qualified_Exp8573_complete_session_clocks_and_parity",
                "compatible_operation_and_precision_transport_update_sampling_costs",
            ],
        ),
        methodology_note="Authenticate original board evidence once without device commands. Separate cold panels, resident sessions and durable learning. No measured acceleration or independent scientific benefit.",
    )
    value.update(
        future_dependencies=[
            dict(
                experiment_id=8573,
                status="qualified_current_evidence"
                if cost["qualified"]
                else "future_dependency_not_observation",
            )
        ],
        literature_operation_mapping=[
            dict(
                source="https://arxiv.org/abs/2607.21077",
                scope="pipelined_quadratic_Ising_and_memory_architecture",
                installed_resident_kernel=False,
                Gibbs_sampling_quality_qualified=False,
                local_efficiency_ratio=None,
            )
        ],
        preserved_history=dict(
            Exp8529="qualified_exposed_source_null",
            Exp8553="HALLMARK_independent_label_rejection",
            Exp8555="disqualified_seven_TAUTOLOGY_and_owned_failures",
            Exp8558="qualified_native_parity",
            Exp8559=dict(ratio=1.010149, CI95=[0.997428, 1.023458]),
            Exp8561="publication_closure_line282_unqualified",
            learning_benefit="unmeasured",
            capability_panel="unmeasured",
            old_quarantines="unchanged",
        ),
    )
    value["field_principles"] = {
        k: "Bind this invocation to original evidence; scope accounting grants no physical or scientific benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(
        hardware_boundary_ready_score="One means qualified operation mapping; resident cost and physical readiness are separate.",
        physical_hardware_ready_score="Zero until a compatible physical operation executes with authenticated parity.",
        conditional_ceiling_rows="Complete wall divided by retained fixed wall; zero arithmetic is theoretical, never measured speed.",
        acquisition_decision="Measure the existing CPU resident boundary before buying hardware.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh reductions require original pins even when an attacker repairs summary hashes."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(evidence.base.checked(value["work_reference"]).read_bytes())
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            evidence.base.checked(ref)
        if operands(Path(work["root"])) != work["evidence"]:
            return False
        return bool(
            build(
                work,
                value["validation_receipts"],
                Path(value["work_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
