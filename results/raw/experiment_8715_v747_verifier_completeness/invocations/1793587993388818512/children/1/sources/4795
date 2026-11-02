"""REQ-REPORT-8613: complete mapping can coexist with blocked physical execution."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import signal
import time
from typing import Any

from carnot.reporting import hardware_operation_evidence_8613 as evidence
from carnot.reporting import v739_contract_methods as authority
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.nine_input_board_evidence_8538 import (
    checksum as checksum,
    failure,
    reference as reference,
)

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = (
    "experiment_8613_v739_hardware_operation_boundary",
    "exp8613-hardware-operation-boundary",
)
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hardware_operation_boundary_8613.py"
OWNED = [
    f"python/carnot/reporting/hardware_operation_{s}_8613.py"
    for s in ["evidence", "boundary", "runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush each boundary so a supervisor sees bounded read-only work progressing."""
    print(f"[exp8613] phase={phase} completed={completed} pending={pending}", flush=True)


def scan_timeout(_signum: int, _frame: Any) -> None:
    """A stale evidence read must fail within the declared sixty-second budget."""
    raise TimeoutError("evidence_authentication_60s")


def measure(root: Path, raw: Path, private: Path, run_date: str) -> Json:
    """Authenticate inputs, original authority and writable disk before CPU-only reduction."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    progress("preconditions_before")
    resource = resources(private)
    contract = authority.authority(root)
    checks = list(contract["gate_check_summary"])
    checks.append(
        failure(
            root / authority.ACTIVE, "thirteen_object_v739_authority", True, contract["ready"], TASK
        )
    )
    checks.append(
        failure(
            private,
            "disk_memory_tools",
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
    progress("board_and_cost_authentication_before", 0, 4)
    handler = signal.signal(signal.SIGALRM, scan_timeout)
    timer = signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        source = evidence.authenticate(root)
    except TimeoutError:
        source = {
            name: dict(
                qualified=False,
                value=None,
                refs=[],
                cells=[],
                checks=[failure(root, "evidence_authentication_60s", True, False, name)],
            )
            for name in ["board", "cost", "lean", "learning"]
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
        signal.signal(signal.SIGALRM, handler)
    end = time.monotonic_ns()
    progress("board_and_cost_authentication_after", 4, 0)
    work = dict(
        root=str(root),
        run_date=run_date,
        evidence=source,
        contract=contract,
        checks=checks,
        refs=[],
        preconditions=resource,
        phase_spans=[
            dict(
                phase="preconditions_and_authentication",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
        duration_s=(end - start) / 1e9,
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "ops/exclusion_manifest.yaml",
                authority.PROTOCOL,
                "docs/research-notes/v739-hardware-operation-boundary.md",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Mapping credits accounting only; missing physical kernels remain externally blocked."""
    source = work["evidence"]
    board = source["board"]["value"] or {}
    cost = source["cost"]
    original = cost["value"] or {}
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and work.get("exact_union_receipt", {}).get("passed", True)
    )
    ready = (
        owned
        and source["board"]["qualified"]
        and cost["qualified"]
        and work["contract"]["ready"]
        and all(r["passed"] for r in work["checks"])
    )
    resident, learning = evidence.matrices(source)
    rows = [
        dict(
            source_id=f"{r['boundary']}:{r['board']}:{r['operation']}",
            arm="resident_inference" if r in resident else "delayed_learning",
            intended=1,
            completed=True,
            failed=False,
            censored=False,
            excluded=False,
            independent=0,
            numerator=1,
            denominator=1,
            metric=None,
            missing_reason=r.get("missing_reason"),
        )
        for r in [*resident, *learning]
    ]
    gates = [*work["checks"], *(g for item in source.values() for g in item["checks"])]
    gates.append(
        failure(
            Path(work["root"]) / "hardware/compatible_operation_execution.json",
            "compatible_physical_execution",
            True,
            None,
            TASK,
        )
    )
    gates += [
        failure(Path(r.get("stdout_path", str(raw))), "owned_" + r["name"], True, False, TASK)
        for r in receipts
        if not r["passed"]
    ]
    klass = "blocked" if owned else "disqualified"
    value: Json = dict(
        experiment_id=8613,
        task_id=TASK,
        milestone="2026.10.739",
        run_date=work["run_date"],
        honest_verdict=f"complete_{klass}_v739_hardware_operation_boundary",
        verdict_class=klass,
        gate_check_summary=gates,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, current_llm_invocations=0),
        historical_model_provenance=board.get("historical_model_provenance", []),
        target_model=original.get("target_model"),
        numerical_inference_this_invocation=False,
        rows=rows,
        intended_count=len(rows),
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            operation_mappings=len(rows),
            historical_resident_cells=len(cost["cells"]),
            independent_device_measurements=0,
        ),
        verifier_is_oracle=False,
        exposure_scope="exposed_engineering_accounting_only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_validation=owned,
            original_board=source["board"]["qualified"],
            qualified_resident_cost=cost["qualified"],
            thirteen_object_authority=work["contract"]["ready"],
            qualified_learning_cost=source["learning"]["qualified"],
            compatible_physical_execution=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7398613,
        source_artifact_hashes=[r for item in source.values() for r in item["refs"]]
        + work["contract"]["references"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[reference(raw / "measurement.json")],
        cited_upstream_artifacts=[evidence.BOARD, evidence.COST, evidence.LEAN, evidence.LEARNING],
        field_principles={},
        hardware_boundary_ready_score=int(ready),
        board_boundary_recorded_score=int(source["board"]["qualified"]),
        physical_hardware_ready_score=0,
        operation_coverage_rows=resident + learning,
        resident_inference_operation_rows=resident,
        delayed_learning_operation_rows=learning,
        conditional_ceiling_rows=evidence.ceilings(cost),
        learning_locality_rows=learning,
        qualified_cost_reference=dict(
            qualified=cost["qualified"],
            reference=reference(ROOT / evidence.COST)
            if cost["qualified"] and not source["lean"]["qualified"]
            else None,
            historical_boundary=not source["lean"]["qualified"],
            boundary="complete resident sessions; length1/4/16 panels, batch1/8, ten repeats on one host",
            current_lean_cost_qualified=source["lean"]["qualified"],
            sparse_update_cost_qualified=source["learning"]["qualified"],
        ),
        board_reopen_requirements=board.get("board_reopen_requirements", []),
        NPU_qualified=False,
        TSU_qualified=False,
        acquisition_decision=dict(
            recommended=False,
            purchase_authorized=False,
            prerequisite="Authenticated compatible installed kernel for a measured dominant operation, binary64/typed-policy parity, complete host/transfer/update/sampling clocks; physical access required first.",
            measured_dominant_operations=["startup_imports", "shutdown", "text_features"]
            if cost["qualified"]
            else [],
            dominant_operation_has_compatible_installed_accelerator=False,
        ),
        device_commands_executed=[],
        work_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        future_dependencies=[
            dict(
                experiment_id=n,
                status="future_dependency_unqualified",
                qualified=source[key]["qualified"],
            )
            for n, key in [(8610, "lean"), (8605, "learning")]
        ],
        nfr01_score=0,
        native_arithmetic_reference=original.get("primary_contrast"),
        lifecycle_savings_reference=original.get("lifecycle_contrast"),
        learning_path="Delayed released feedback drives sparse CPU/Rust local spline coefficient gradients, then checkpoint sync and feedback drain. Local support does not remove text features, transport, durable writes or admission clocks. No qualified Exp8605 timing or learning benefit is available.",
        literature_operation_mapping=[
            dict(
                source="https://arxiv.org/abs/2610.03972",
                operations=["distribution_sampling", "argmin_search"],
                deterministic_spline_evaluation=False,
                local_speedup=None,
            ),
            dict(
                source="https://arxiv.org/abs/2606.25313",
                operations=["distributed_Gibbs_sampling", "boundary_state_exchange"],
                limitation="Boundary communication frequency controls throughput versus sampling accuracy; no deterministic spline or local installed kernel follows.",
                local_speedup=None,
            ),
            dict(
                source="https://extropic.ai/writing/z1t/",
                operations=["sparse_tanh_sampling", "FPGA_digital_fallback"],
                excluded_operations=["dense_final_logits", "inter_device_movement"],
                vendor_projections_only=True,
                local_speedup=None,
            ),
        ],
        methodology_note="Read pinned historical evidence without model loads, device commands or benchmarks. Derive conditional ideal arithmetic removal only from disjoint qualified complete clocks. Charge fixed host work. Transfer, precision and sampling costs remain unknown. Distribution sampling, argmin search and deterministic spline evaluation are different workloads. Ceilings below10x cannot support NFR01 on this boundary.",
    )
    for key in [
        "kv260_supported_operations",
        "polarfire_graduation",
        "cold_boundary_reference",
        "preserved_historical_outcomes",
        "preserved_quarantines",
    ]:
        value[key] = deepcopy(board.get(key))
    value["newly_owned_coverage"] = work.get("owned_coverage")
    value["exact_union_receipt"] = work.get("exact_union_receipt")
    value["preserved_v738_outcomes"] = dict(
        scheduling_slots=11,
        Exp8592="second deadline-clock read line94 uncovered; scientific chain unqualified",
        Exp8593="resident provenance qualified",
        Exp8596="complete costs qualified; native CI95[1.0740525,1.1057774]; startup reuse separate; NFR01 unmet",
        Exp8597="no usable MedExpert release evidence",
        Exp8599="zero resolved ARC outcomes",
    )
    value["literature_references"] = [
        reference(p)
        for p in sorted((ROOT / "results/raw" / NAME / "literature").iterdir())
        if p.is_file()
    ]
    value["learning_implementation_references"] = [
        reference(ROOT / p)
        for p in [
            "python/carnot/verify/sparse_energy_7996.py",
            "crates/carnot-core/src/direct_spline_8379.rs",
        ]
    ]
    value["field_principles"] = {
        key: "Invocation-bound accounting; original pinned producers retain authority. Missing costs remain null; mapping is not device performance."
        for key in value
    }
    value["field_principles"].update(
        hardware_boundary_ready_score="One requires authenticated mapping, qualified cost, current authority and passing owned checks. It makes no acceleration claim.",
        physical_hardware_ready_score="Zero: no compatible physical operation execution has been authenticated.",
        qualified_cost_reference="Original qualified Exp8596 complete session clocks remain historical while Exp8610 is unqualified.",
        conditional_ceiling_rows="Total divided by total minus removable arithmetic, from disjoint clocks only. Fixed host work remains charged; unknown overhead is null.",
        learning_locality_rows="Delayed feedback gradients, checkpoint sync and feedback drain have no qualified Exp8605 costs; null is missing, not zero.",
        native_arithmetic_reference="Preserve native versus Python timing separately from startup reuse.",
        lifecycle_savings_reference="Startup reuse is a lifecycle contrast; it cannot establish native arithmetic speed or NFR01.",
        target_model="Authenticate the original numerical energy head and checkpoint without loading or executing it.",
        rows="Each unit is an operation-to-board mapping, not an independent question or device measurement.",
        field_principles="Explain the interpretation and limits of every declared field.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh original authentication defeats self-consistent edits to cached summaries."""
    try:
        value = json.loads(path.read_bytes())
        ref = value["work_reference"]
        work = json.loads(Path(evidence.checked(ref)["path"]).read_bytes())
        if evidence.authenticate(Path(work["root"])) != work["evidence"]:
            return False
        for code in work["code_config_hashes"]:
            evidence.checked(code)
        if authority.authority(Path(work["root"])) != work["contract"]:
            return False
        return bool(
            build(
                work,
                value["validation_receipts"],
                Path(ref["path"]).parent,
                Path(value["publication_output"]),
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
