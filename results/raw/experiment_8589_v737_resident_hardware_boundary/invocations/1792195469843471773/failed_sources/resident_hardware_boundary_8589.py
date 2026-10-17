"""REQ-REPORT-8589: resident cost mapping cannot create physical or learning readiness."""

from __future__ import annotations

import json
from pathlib import Path
import signal
import time
from typing import Any

from carnot.reporting import resident_hardware_evidence_8589 as evidence
from carnot.reporting import v737_contract_methods as authority
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.hardware_operation_boundary_8563 import node_union
from carnot.reporting.nine_input_board_evidence_8538 import checksum, failure, reference

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8589_v737_resident_hardware_boundary", "exp8589-resident-hardware-boundary"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_resident_hardware_boundary_8589.py"
OWNED = [
    f"python/carnot/reporting/resident_hardware_{s}_8589.py"
    for s in ["evidence", "boundary", "runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts let the parent distinguish a bounded evidence read from a stalled task."""
    print(f"[exp8589] phase={phase} completed={completed} pending={pending}", flush=True)


def scan_timeout(_signum: int, _frame: Any) -> None:
    """Stop the read-only evidence branch at its declared bound rather than inventing missing data."""
    raise TimeoutError("evidence_scan_60s")


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Record resources and full activated authority before independently reading both CPU branches."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    progress("preconditions")
    resource = resources(private)
    contract = authority.authority(root)
    gates = list(contract["gate_check_summary"])
    gates.append(
        failure(
            root / authority.ACTIVE,
            "current_twelve_object_authority",
            True,
            contract["ready"],
            TASK,
        )
    )
    gates.append(
        failure(
            private,
            "disk_backed_memory_tools",
            True,
            bool(
                resource["disk_backed"]
                and resource["tools_available"]
                and resource["memory_available_bytes"] >= 1024**3
                and resource["free_disk_bytes"] >= 1024**3
            ),
            TASK,
        )
    )
    progress("authenticate_supplied_board_and_cost_before", 0, 2)
    began = time.monotonic_ns()
    handler = signal.signal(signal.SIGALRM, scan_timeout)
    timer = signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        source = evidence.authenticate(root)
    except TimeoutError as error:
        source = {
            name: dict(
                qualified=False,
                checks=[failure(root, str(error), True, False, TASK)],
                refs=[],
                value=None,
                cells=[],
            )
            for name in ["board", "cost", "resident_qualification"]
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
        signal.signal(signal.SIGALRM, handler)
    elapsed = (time.monotonic_ns() - began) / 1e9
    progress("authenticate_supplied_board_and_cost_after", 2, 0)
    operand_path = raw / "operands.json"
    atomic_json(operand_path, source)
    work: Json = dict(
        root=str(root),
        evidence=source,
        operand_reference=reference(operand_path),
        checks=gates,
        contract=contract,
        preconditions=dict(
            resource,
            optional_cuda_required=False,
            optional_board_required=False,
            measurement_stop_s=3600,
            validation_reserve_s=900,
            task_cap_s=4800,
        ),
        scan_duration_s=elapsed,
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=[dict(name="evidence_authentication", duration_s=elapsed)],
        refs=[
            *contract["references"],
            *source["board"]["refs"],
            *source["cost"]["refs"],
            *source["resident_qualification"]["refs"],
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "scripts/experiment_template.py",
                "ops/exclusion_manifest.yaml",
                authority.PROTOCOL,
                "docs/research-notes/v737-resident-hardware-boundary.md",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Readiness credits completed accounting; unavailable external costs and kernels remain blocked."""
    source = work["evidence"]
    board, cost = source["board"], source["cost"]
    original = board["value"] or {}
    union = node_union(raw, receipts)
    owned = bool(receipts) and all(r["passed"] for r in receipts) and union["passed"]
    ready = (
        owned
        and board["qualified"]
        and source["resident_qualification"]["qualified"]
        and work["contract"]["ready"]
        and all(g["passed"] for g in work["checks"])
    )
    operations = evidence.matrix(source)
    rows = [
        dict(
            source_id=f"{r['board']}:{r['operation']}",
            arm="operation_mapping",
            intended=1,
            completed=board["qualified"],
            failed=False,
            censored=not board["qualified"],
            excluded=False,
            independent=0,
            numerator=int(board["qualified"]),
            denominator=1,
            missing_reason=None if board["qualified"] else "original_board_evidence_absent",
            board=r["board"],
            operation=r["operation"],
        )
        for r in operations
    ]
    gates = [
        *work["checks"],
        *board["checks"],
        *cost["checks"],
        *source["resident_qualification"]["checks"],
    ]
    for name in evidence.BOARDS:
        gates.append(
            failure(
                Path(work["root"]) / "hardware" / name / "resident_execution.json",
                "authenticated_compatible_resident_execution",
                True,
                None,
                name,
            )
        )
    for receipt in receipts:
        if not receipt["passed"]:
            gates.append(
                failure(
                    Path(receipt.get("stdout_path", str(raw))),
                    "owned_check_" + receipt["name"],
                    True,
                    False,
                    TASK,
                )
            )
    klass = "blocked" if owned else "disqualified"
    value: Json = dict(
        experiment_id=8589,
        task_id=TASK,
        milestone="2026.10.737",
        run_date="20261016",
        honest_verdict=f"complete_{klass}_v737_resident_hardware_boundary",
        verdict_class=klass,
        gate_check_summary=gates,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, current_llm_invocations=0),
        historical_model_provenance=original.get("historical_model_provenance", []),
        rows=rows,
        intended_count=50,
        completed_count=sum(r["completed"] for r in rows),
        failed_count=0,
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            operation_mappings=50, intended_resident_cells=60, independent_device_measurements=0
        ),
        verifier_is_oracle=False,
        exposure_scope="exposed_engineering_accounting_only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates=dict(
            owned_validation=owned,
            original_board=board["qualified"],
            activated_twelve_object_contract=work["contract"]["ready"],
            qualified_resident_cost=cost["qualified"],
            compatible_physical_execution=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=7378589,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[work["operand_reference"], *work.get("coverage_shards", [])],
        work_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        cited_upstream_artifacts=[
            reference(Path(r["path"])) for r in [*board["refs"][:1], *cost["refs"][:1]]
        ],
        hardware_boundary_ready_score=int(ready),
        board_boundary_recorded_score=int(ready),
        physical_hardware_ready_score=0,
        operation_coverage_rows=operations,
        qualified_cost_reference={k: v for k, v in cost.items() if k != "value"},
        conditional_ceiling_rows=evidence.ceilings(cost),
        cold_boundary_reference=dict(
            path=evidence.BOARD,
            sha256=evidence.PINS[evidence.BOARD],
            boundary="one_cold_64_answer_panel",
            scaling_offset_residual_ceiling=1.008746,
            residual_only_ceiling=1.007981,
            resident_acceleration_established=False,
        ),
        vendor_exclusion_rows=original.get("vendor_exclusion_rows", []),
        literature_operation_mapping=original.get("literature_operation_mapping", []),
        board_reopen_requirements=original.get("board_reopen_requirements", []),
        kv260_supported_operations=original.get("kv260_supported_operations", []),
        polarfire_graduation=original.get("polarfire_graduation"),
        acquisition_decision=dict(
            recommended=False,
            purchase_authorized=False,
            cpu_residence_removes_bottleneck="unmeasured_without_qualified_complete_resident_cost",
            residence_is_available=source["resident_qualification"]["qualified"],
            measured_resident_acceleration=None,
            required_before_acquisition=[
                "qualified_complete_resident_cost",
                "dominant_supported_operation",
                "precision_transport_update_sampling_costs",
                "complete_device_parity_and_cost",
            ],
            missing_reason="resident_cost_unqualified_and_no_compatible_installed_kernel",
        ),
        historical_cost_null="Exp8559 cold-cost null remains unchanged",
        preserved_historical_outcomes=[
            "Exp8529 exposed source null",
            "Exp8553 label rejection",
            "Exp8559 cold-cost null",
            "Exp8568 original validator revision rejection",
        ],
        preserved_quarantines=dict(
            never_clear=True,
            prior_board=original.get("preserved_quarantines", []),
            original_v737_protocol=authority.PROTOCOL,
        ),
        future_dependencies=[
            dict(
                experiment_id=8584,
                status="qualified_current_evidence"
                if cost["qualified"]
                else "future_dependency_not_qualified_observation",
            )
        ],
        device_commands_executed=[],
        NPU_qualified=False,
        TSU_qualified=False,
        evidence_delta_scan_limit_s=60,
        evidence_delta_scan_duration_s=work["scan_duration_s"],
        exact_union_receipt=union,
        newly_owned_coverage=work.get("owned_coverage"),
        methodology_note="Read original sealed evidence once with no model or device commands. Ten operation groups map to five boards. Missing resident cost is not zero cost. Ideal removal bounds include host and frame work and cannot establish measured speedup, durable learning or NFR01. CPU residence is qualified as an execution mode; its bottleneck reduction is unqualified until complete cost evidence qualifies.",
    )
    value["field_principles"] = {
        k: "Bind original producer bytes and this invocation; mapping grants no device or learning benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Recompute from independent original pins so repaired report and primitive hashes cannot authorize changes."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        for ref in [
            value["work_reference"],
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                return False
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if (
                    stream + "_path" in receipt
                    and reference(Path(receipt[stream + "_path"]))["sha256"]
                    != receipt[stream + "_sha256"]
                ):
                    return False
        source = evidence.authenticate(Path(work["root"]))
        if (
            source != work["evidence"]
            or authority.authority(Path(work["root"])) != work["contract"]
        ):
            return False
        if json.loads(Path(work["operand_reference"]["path"]).read_bytes()) != source:
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
