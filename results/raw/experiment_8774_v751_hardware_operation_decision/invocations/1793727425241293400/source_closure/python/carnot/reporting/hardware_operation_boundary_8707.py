"""REQ-REPORT-8707: CPU lifecycle evidence does not install an FPGA kernel."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_operation_inputs_8707 as inputs
from carnot.reporting.current_work_receipt import canonical_hash, ZERO_INVOCATION_COUNTS

Json = dict[str, Any]
ROOT = inputs.ROOT
NAME, TASK = (
    "experiment_8707_v746_hardware_operation_boundary",
    "exp8707-hardware-operation-boundary",
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/hardware_operation_{part}_8707.py"
    for part in ["inputs", "boundary", "runner"]
] + [CLI]
TESTS = ["tests/python/test_hardware_operation_boundary_8707.py"]
MODEL_SPECS: list[Json] = []
OPERATIONS = [
    "quadratic_Ising_energy_k_max_5",
    "integer_basis_residual_sparse_Brier_32_24",
    "dense_109_renormalization",
    "global_decay",
    "host_sigmoid_logit_fallback",
    "complete_feature_processing",
    "host_device_transfer",
    "serialization_fsync",
    "restart",
    "complete_learning",
    "distribution_sampling",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts let the parent distinguish real work from a stalled child."""
    print(f"[exp8707] phase={phase} completed={completed} pending={pending}", flush=True)


def bound(total: float | None, compatible: float | None, implemented: bool) -> Json:
    """Only measured compatible work permits a conditional arithmetic-removal ceiling."""
    if total is not None and (
        total <= 0 or compatible is not None and not 0 <= compatible <= total
    ):
        raise ValueError("invalid_compatible_fraction")
    fraction = compatible / total if total and compatible is not None and implemented else None
    return dict(
        compatible_fraction=fraction,
        conditional_ceiling=1 / (1 - fraction) if fraction is not None and fraction < 1 else None,
        achieved_speedup=None,
        missing_reason="no_implemented_compatible_kernel_or_disjoint_complete_cost",
        interpretation="Conditional1/(1-f); neither observed speed nor a100x complete-learning result",
    )


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Reduce sealed evidence without granting readiness to absent or flagged producers."""
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
            unit_id=b["task_id"] + ":" + b["field"],
            arm=b["field"],
            status="completed" if b["qualified"] else "censored",
            metric="qualified_original_evidence",
            value=int(b["qualified"]),
            censored=not b["qualified"],
            excluded=False,
            effective_independent_groups=0,
        )
        for b in branches
    ]
    board_rows = deepcopy(evidence["prior"].get("board_status_rows", []))
    board_rows += [
        dict(
            board=board,
            retained_scope="unqualified",
            physical_execution_this_invocation=False,
            required=[
                "authenticated compatible implementation",
                "local quality and complete cost evidence",
            ],
        )
        for board in ["NPU", "TSU"]
    ]
    operation_rows = []
    for board in ["KV260", "PolarFire", "GateMate", "NPU", "TSU"]:
        for operation in OPERATIONS:
            support = (
                "supported"
                if board == "KV260" and operation == OPERATIONS[0] and mapping
                else "unsupported"
                if board in {"KV260", "PolarFire"} and mapping
                else "missing-evidence"
            )
            operation_rows.append(
                dict(
                    board=board,
                    operation=operation,
                    support=support,
                    precision="original quadratic implementation"
                    if support == "supported"
                    else "signed32/24 with checked signed64 required for spline",
                    workload="nine inputs,109 coefficients"
                    if operation != OPERATIONS[0]
                    else "quadratic Ising k_max<=5",
                    actual_implementation="authenticated original quadratic kernel"
                    if support == "supported"
                    else "no compatible fabric implementation authenticated",
                    physical_execution_this_invocation=False,
                    reason="PolarFire evidence is Linux CPU dispatch only"
                    if board == "PolarFire"
                    else "historical scope; no unchanged device probes",
                )
            )
    gates = [*evidence["failures"], *work["resource_failures"]]
    gates += [c for b in branches for c in b["checks"] if not c["passed"]]
    gates.append(
        dict(
            upstream="target_boards",
            check="compatible_physical_execution",
            **inputs.reference(
                ROOT / "results/experiment_8693_v745_hardware_operation_boundary.json"
            ),
            field="physical_hardware_ready_score",
            operator="==",
            expected=1,
            observed=0,
            operand_state="observed_zero",
            passed=False,
        )
    )
    prior = evidence["prior"]
    quarantines = deepcopy(prior.get("preserved_quarantines", {}))
    quarantines.update(
        Exp8681=dict(
            flagged_adversarial=True,
            verdict_class="disqualified",
            reason="empty final coverage retention; original rejection line113 unproved",
        ),
        Exp8691=dict(
            flagged_adversarial=True,
            verdict_class="disqualified",
            reason="failed owned validation; later source fix does not qualify original",
        ),
    )
    value: Json = dict(
        experiment_id=8707,
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
        historical_model_provenance=evidence["historical_model_provenance"],
        rows=rows,
        intended_count=4,
        completed_count=sum(b["qualified"] for b in branches),
        failed_count=0,
        censored_count=sum(not b["qualified"] for b in branches),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(independent_questions=0, intended_operand_gates=4),
        verifier_is_oracle=True,
        exposure_scope="constructed_upstream_CPU_controls_no_new_questions",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            owned_validation=owned,
            original_board_authority=evidence["board_qualified"],
            complete_learning_cost=branches[0]["qualified"],
            complete_batch_cost=branches[1]["qualified"],
            lifecycle_rows=branches[2]["qualified"],
            numerical_lifecycle=branches[3]["qualified"],
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
        random_seed=7468707,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[ref for r in receipts for ref in r.get("references", [])],
        cited_upstream_artifacts=evidence["refs"],
        branch_gate_values={b["field"]: int(b["qualified"]) for b in branches},
        hardware_mapping_ready_score=int(mapping),
        physical_hardware_ready_score=0,
        board_status_rows=board_rows,
        operation_support_rows=operation_rows,
        complete_cost_rows=deepcopy(
            evidence["lifecycle_cost_rows"] + evidence["historical_cost_rows"]
        ),
        conditional_acceleration_bounds=[
            dict(board=b, **bound(None, None, False))
            for b in ["KV260", "PolarFire", "GateMate", "NPU", "TSU"]
        ],
        acquisition_prerequisites=deepcopy(prior.get("acquisition_prerequisites", {})),
        NPU_qualified=False,
        TSU_qualified=False,
        device_commands_executed=[],
        preserved_quarantines=quarantines,
        preserved_historical_outcomes=[
            "V733 source-information null",
            "V735 cold-cost null",
            "V737 supervisor null",
            "V743 batch-one null",
        ],
        g1_g4_separation="Original G1–G4 publication conditions remain separate; this map grants none.",
        hardware_100x_requirements=dict(
            target_met=False,
            necessary_unaccelerated_fraction_below=0.01,
            finite_kernel_speed_required=True,
            complete_learning_cost=None,
            required=[
                "compatible signed32/24 kernel and input-domain proof",
                "complete features, transfer, nonlinear host work and dense renormalization",
                "serialization/fsync, feedback drain, restart and retention",
                "matched quality and measured100x complete learning",
            ],
        ),
        acquisition_priority_delta=dict(
            changed=False, reason="No new evidence-backed acquisition need"
        ),
        future_hardware_commands=dict(
            tools=["SSH", "scp", "xmutil"],
            prerequisite="authenticated source/build/bitstream and compatible kernel parity",
        ),
        cost_scope_note="CPU nested/reference/worker envelopes are not additive; combined update clocks do not isolate dense renormalization. Transfer and complete causal service remain unknown.",
        authority=evidence["authority"],
        source_admission_rows=[{k: v for k, v in b.items() if k != "value"} for b in branches],
        primitive_reference=inputs.reference(Path(work["raw"]) / "measurement.json"),
        validation_reference=inputs.reference(Path(work["raw"]) / "validation.json"),
        publication_output=str(output.absolute()),
        qualification_scope="private_control_only" if work["private_control"] else "production",
        methodology_note="Read-only reduction of authenticated original board and CPU lifecycle artifacts. No model, numerical benchmark, probe, flash or physical acceleration.",
    )
    value["branch_gate_values"].update(
        hardware_mapping_ready_score=int(mapping), physical_hardware_ready_score=0
    )
    value["field_principles"] = {
        k: "Exact immutable invocation evidence; mapping, physical readiness and benefit are separate. Missing operands remain unknown."
        for k in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
