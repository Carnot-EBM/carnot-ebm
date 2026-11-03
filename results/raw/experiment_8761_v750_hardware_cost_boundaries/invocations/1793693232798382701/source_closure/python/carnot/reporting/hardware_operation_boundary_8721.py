"""REQ-REPORT-8721: a compiled CPU port does not install a physical board kernel."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_operation_inputs_8721 as inputs
from carnot.reporting.current_work_receipt import canonical_hash, ZERO_INVOCATION_COUNTS
from carnot.reporting.hardware_operation_boundary_8707 import bound

Json = dict[str, Any]
ROOT = inputs.ROOT
NAME, TASK = (
    "experiment_8721_v747_hardware_operation_boundary",
    "exp8721-hardware-operation-boundary",
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/hardware_operation_{p}_8721.py"
    for p in ["inputs", "boundary", "runner"]
] + [CLI]
TESTS = ["tests/python/test_hardware_operation_boundary_8721.py"]
MODEL_SPECS: list[Json] = []
OPERATIONS = [
    "quadratic_Ising_energy_k_max_5",
    "compiled_signed32_24_basis",
    "compiled_signed32_24_residual",
    "compiled_signed32_24_sparse_Brier_update",
    "checked_signed64_accumulator",
    "native_host_coefficient_crossings",
    "dense_109_renormalization",
    "global_decay",
    "host_sigmoid_scaling",
    "complete_feature_processing",
    "host_device_transfer",
    "serialization_fsync",
    "recovery",
    "complete_learning",
    "distribution_sampling",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts keep external supervisors informed during large evidence reads."""
    print(f"[exp8721] phase={phase} completed={completed} pending={pending}", flush=True)


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Preserve every branch while keeping mapping, benefit and physical execution separate."""
    evidence = work["evidence"]
    branches = evidence["branches"]
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    mapping = (
        owned
        and evidence["board_qualified"]
        and not work["resource_failures"]
        and not work["private_control"]
    )
    verdict = "blocked" if owned else "disqualified"
    rows = [
        dict(
            unit_id=b["task_id"],
            arm=b["field"],
            status="completed" if b["qualified"] else "censored",
            value=int(b["qualified"]),
            censored=not b["qualified"],
            excluded=False,
            effective_independent_groups=0,
        )
        for b in branches
    ]
    costs = [
        dict(producer=b["task_id"], boundary="complete_installed_inference", **r)
        for b in branches
        if b["qualified"]
        for r in b["value"].get("complete_cost_rows", [])
    ]
    costs += [
        dict(
            producer=b["task_id"],
            boundary="CPU_native_counts_only",
            duration_s=None,
            missing_reason="producer_measured_parity_and_counts_not_disjoint_durations",
            **r,
        )
        for b in branches
        if b["qualified"]
        for r in b["value"].get("operation_partition_rows", [])
    ]
    costs += [
        dict(
            producer=branches[0]["task_id"],
            boundary="complete_learning",
            duration_s=None,
            missing_reason="no_authenticated_complete_learning_operand",
        )
    ]
    gates = [
        *evidence["failures"],
        *work["resource_failures"],
        *(c for b in branches for c in b["checks"] if not c["passed"]),
    ]
    gates.append(
        dict(
            upstream="target_boards",
            **inputs.reference(ROOT / inputs.PRIOR),
            check="compatible_physical_execution",
            field="physical_hardware_ready_score",
            operator="==",
            expected=1,
            observed=0,
            operand_state="observed_zero",
            passed=False,
        )
    )
    operations = []
    for board in ["KV260", "PolarFire", "GateMate"]:
        for op in OPERATIONS:
            support = (
                "supported"
                if mapping and board == "KV260" and op == OPERATIONS[0]
                else "unsupported"
                if mapping and board != "GateMate"
                else "missing-evidence"
            )
            operations.append(
                dict(
                    board=board,
                    operation=op,
                    support=support,
                    precision="original quadratic implementation"
                    if op == OPERATIONS[0]
                    else "signed32/24; checked signed64",
                    workload="quadratic Ising k_max<=5"
                    if op == OPERATIONS[0]
                    else "nine inputs;109 coefficients; at most37 local terms",
                    physical_execution_this_invocation=False,
                    actual_implementation="retained quadratic kernel"
                    if support == "supported"
                    else "no compatible fabric implementation authenticated",
                    reason="Linux CPU dispatch is not fabric acceleration"
                    if board == "PolarFire"
                    else "retained board evidence; no unchanged probes",
                )
            )
    value: Json = dict(
        experiment_id=8721,
        task_id=TASK,
        milestone=inputs.MILESTONE,
        run_date=work["run_date"],
        invocation_id=Path(work["raw"]).name,
        honest_verdict=f"complete_{verdict}_missing_complete_cost_or_physical_execution",
        verdict_class=verdict,
        gate_check_summary=gates,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        target_model=None,
        historical_model_provenance=evidence["prior"].get("historical_model_provenance", []),
        rows=rows,
        intended_count=3,
        completed_count=sum(b["qualified"] for b in branches),
        failed_count=0,
        censored_count=sum(not b["qualified"] for b in branches),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(independent_questions=0, intended_operand_gates=3),
        verifier_is_oracle=True,
        exposure_scope="authenticated_exposed_CPU_artifacts_no_new_questions",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            owned_validation=owned,
            board_authority=evidence["board_qualified"],
            physical_execution=False,
        ),
        validation_receipts=deepcopy(receipts),
        adversarial_findings=[],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=7478721,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[ref for receipt in receipts for ref in receipt.get("references", [])],
        cited_upstream_artifacts=evidence["refs"],
        branch_gate_values={b["field"]: int(b["qualified"]) for b in branches},
        hardware_mapping_ready_score=int(mapping),
        physical_hardware_ready_score=0,
        board_status_rows=deepcopy(evidence["prior"].get("board_status_rows", [])),
        operation_support_rows=operations,
        complete_cost_rows=costs,
        conditional_acceleration_bounds=[
            dict(board=b, **bound(None, None, False)) for b in ["KV260", "PolarFire", "GateMate"]
        ],
        acquisition_prerequisites=deepcopy(evidence["prior"].get("acquisition_prerequisites", {})),
        preserved_quarantines=deepcopy(evidence["prior"].get("preserved_quarantines", {})),
        preserved_scientific_nulls=deepcopy(
            evidence["prior"].get("preserved_scientific_nulls", {})
        ),
        authority=evidence["authority"],
        source_admission_rows=[{k: v for k, v in b.items() if k != "value"} for b in branches],
        primitive_reference=inputs.reference(Path(work["raw"]) / "measurement.json"),
        validation_reference=inputs.reference(Path(work["raw"]) / "validation.json"),
        publication_output=str(output.absolute()),
        qualification_scope="private_control_only" if work["private_control"] else "production",
        complete_100x_learning_ready_score=0,
        limiting_host_costs=[
            "complete features",
            "native coefficient crossings",
            "sigmoid/scaling",
            "serialization/fsync",
            "recovery",
            "host/device transfer",
        ],
        measurement_to_change_conclusion="Authenticated compatible FPGA image with precision parity and disjoint complete learning clocks, including all host, transfer, durability and recovery costs;100x requires unaccelerated share<=0.01.",
        native_host_call_rows=deepcopy(branches[2]["value"].get("host_call_rows", [])),
        supported_native_domain=deepcopy(branches[2]["value"].get("supported_domain", {})),
        unknown_costs=[
            "current complete learning",
            "compiled operation durations",
            "physical transfer and FPGA service",
        ],
        board_access_policy="Future authorized KV260 work uses SSH/scp/xmutil and hash-bound images; GateMate requires dated physical change, IDCODE0x20000001 then ordered n16 smoke. NPU/TSU unqualified.",
    )
    value["field_principles"] = {
        k: "Bind exact evidence; mapping, numerical portability and measured physical benefit are separate. Absence is not zero."
        for k in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
