"""REQ-REPORT-8774: a valid device map does not establish physical acceleration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_operation_boundary_8721 as original
from carnot.reporting import hardware_cost_boundaries_8761 as costs
from carnot.reporting import hardware_operation_inputs_8774 as source
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
e = namespace(original, "operation_decision_8774")
e.NAME, e.TASK = (
    "experiment_8774_v751_hardware_operation_decision",
    "exp8774-hardware-operation-decision",
)
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.TESTS = ["tests/python/test_hardware_operation_decision_8774.py"]
e.OWNED = [
    f"python/carnot/reporting/hardware_operation_{p}_8774.py"
    for p in ["decision", "inputs", "runner"]
] + [e.CLI]
BASE_BUILD = e.build
BOARDS = ("KV260", "PolarFire", "GateMate")
OPERATIONS = [
    "quadratic_Ising_energy_k_max_5",
    "feature_extraction",
    "digest_serialization",
    "exact_spline_energy",
    "sparse_update_arithmetic",
    "nonlinear_host_work",
    "durable_storage",
    "host_device_transfer",
    "higher_order_PUBO",
]
METHOD = dict(
    boundaries=["causal_learning", "optimized_installed_inference", "ordinary_durable_window"],
    fallback="Exp8757 installed inference only",
    compatible_operation=OPERATIONS[0],
    fraction="disjoint leaves within one producer, process, parent and clock",
    overhead="retain measured setup; unknown physical transfer gives unknown service bound",
    model_loads=0,
    device_commands=0,
    prototype_recommended=False,
)


def cost_rows(branches: list[Json], support: list[Json]) -> Json:
    """The existing disjoint reducer admits only the installed quadratic operation."""
    result: Json = costs.cost_rows(branches, support)
    for row in result["conditional_acceleration_bounds"]:
        row.update(
            complete_service_bound=None,
            physical_setup_duration_s=None,
            measured_acceleration=False,
            unknown_reason="Physical setup/transfer and compatible device service are unmeasured.",
        )
    return result


def higher_order_rows() -> list[Json]:
    """The paper's compiled clause updates require operations absent from these interfaces."""
    return [
        dict(
            board=board,
            operation="higher_order_PUBO",
            support="unsupported" if board != "GateMate" else "unknown",
            paper_source="https://arxiv.org/html/2610.40522v1",
            paper_device="Versal Premium VP1802",
            required_operations=[
                "order<=4 clause products",
                "clause-state XOR updates",
                "colored stochastic spin updates",
            ],
            paper_coefficient_bits=24,
            paper_local_field_bits=28,
            paper_control_interface="AXI4",
            installed_interface="SSH/scp/xmutil; UIO quadratic evaluator k_max<=5"
            if board == "KV260"
            else "hash-verified Linux CPU dispatch"
            if board == "PolarFire"
            else "JTAG blocked:0xffffffff",
            paper_native_variables=179,
            paper_comparison_quadratic_variables=1074,
            paper_comparison_additional_variables=1074 - 179,
            paper_comparison_breakdown=dict(
                additional_routing=577 - 128, intermediate=434, auxiliary=12
            ),
            prototype_quadratization_auxiliary_count=None,
            unshared_product_chain_auxiliaries_per_degree={"3": 1, "4": 2},
            encoding_count_scope="Product-chain examples only; no local target graph or shared reduction is specified.",
            installed_coefficient_precision="Original qualified quadratic precision; no 24/28-bit PUBO parity established",
            host_preprocessing=[
                "expand polynomial and penalties",
                "count all auxiliary variables and graph degree",
                "quantize coefficients and prove overflow/energy parity",
                "color clauses and compile connectivity",
            ],
            preprocessing_duration_s=None,
            transfer_setup_duration_s=None,
            paper_latency_scope="External core only; excludes communication/readout. No local speed transfer.",
            prototype_recommended=False,
        )
        for board in BOARDS
    ]


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Execution acceptance can release a complete map while scientific branches remain blocked."""
    value: Json = BASE_BUILD(work, receipts, output)
    evidence = work["evidence"]
    owned = value["required_checks_passed"]
    prepared = deepcopy(evidence["branches"])
    fallback = evidence["inference_fallback"]
    if not prepared[1]["qualified"] and fallback["qualified"]:
        prepared[1] = deepcopy(fallback)
    support = [
        dict(
            board=board,
            operation=op,
            support="supported"
            if evidence["board_qualified"] and board == "KV260" and op == OPERATIONS[0]
            else "unknown"
            if board == "GateMate"
            else "unsupported",
            execution_location="qualified historical fabric"
            if board == "KV260" and op == OPERATIONS[0]
            else "host",
            physical_execution_this_invocation=False,
            interface="SSH/scp/xmutil and UIO"
            if board == "KV260"
            else "Linux CPU"
            if board == "PolarFire"
            else "blocked JTAG",
            reason="Only historical quadratic Ising fabric k_max<=5 is qualified; no inventory-based reachability.",
        )
        for board in BOARDS
        for op in OPERATIONS
    ]
    mapped = cost_rows(prepared, support)
    value.update(mapped)
    diagnostics = [
        dict(
            row,
            producer=b["task_id"],
            boundary="unqualified_optimized_diagnostic",
            eligible_for_bound=False,
            source_reference=b["refs"][0],
        )
        for b in evidence["branches"]
        for row in b.get("observed_cost_rows", [])
    ]
    value["operation_support_rows"] += [
        dict(
            board=board,
            operation=row["operation"],
            support="unknown" if board == "GateMate" else "unsupported",
            execution_location="host",
            producer=row["producer"],
            clock_domain=row["clock_domain"],
            measured_duration_s=row["duration_s"],
            eligible_for_bound=False,
            physical_execution_this_invocation=False,
            reason="Unqualified host diagnostics have no authenticated fabric implementation.",
        )
        for row in diagnostics
        for board in BOARDS
    ]
    mapping = bool(value["hardware_mapping_ready_score"])
    verdict = "disqualified" if not owned else "blocked"
    boards = {r["board"]: r for r in evidence["prior"].get("board_status_rows", [])}
    value.update(
        experiment_id=8774,
        task_id=e.TASK,
        milestone=source.contract.MILESTONE,
        honest_verdict=f"complete_{verdict}_hardware_operation_decision",
        verdict_class=verdict,
        random_seed=7518774,
        method=METHOD,
        canonical_tasks_sha256=source.contract.PIN,
        numerical_training=dict(performed=False),
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0),
        device_commands_executed=[],
        physical_hardware_ready_score=0,
        complete_100x_learning_ready_score=0,
        board_status_rows=[
            dict(
                boards.get(board, {}),
                board=board,
                current_reachability="unknown_not_probed",
                physical_execution_this_invocation=False,
            )
            for board in BOARDS
        ],
        higher_order_interface_rows=higher_order_rows(),
        cost_scope_rows=[*mapped["boundary_cost_rows"], *mapped["missing_cost_rows"], *diagnostics],
        diagnostic_cost_rows=diagnostics,
        inference_fallback_admission={k: v for k, v in fallback.items() if k != "value"},
        imported_cost_diagnostics=[
            dict(
                task_id=b["task_id"],
                reference=b["refs"][0],
                admitted=b["qualified"],
                checks=b["checks"],
            )
            for b in evidence["branches"]
        ],
        acceptance_gates=dict(owned_validation=owned, board_authority=evidence["board_qualified"]),
        scientific_promotion_gates=dict(physical_execution=False, complete_100x_learning=False),
        hardware_mapping_ready_score=int(mapping),
        repository_health_reference=dict(
            reference=work.get("repository_health_reference"),
            proves_complete_full_suite_pass=False,
            acceptance_dependency=False,
        ),
        scoped_validation_scope=dict(
            test_files=e.TESTS, e2e=["E2E-018", "E2E-021"], repository_health_is_dependency=False
        ),
        measurement_to_change_conclusion=[
            dict(
                board=board,
                minimum_missing="Authenticated compatible operation and same-clock host/setup/transfer/device parity measurement via authorized SSH"
                if board == "KV260"
                else "Authenticated fabric implementation; existing Linux CPU execution is insufficient"
                if board == "PolarFire"
                else "Dated cable/port/power change, then IDCODE0x20000001 before authorized n16 smoke",
                further_required="Exact precision and complete relevant costs before acceleration or purchase",
            )
            for board in BOARDS
        ],
        deferred_devices=["NPU", "TSU"],
        vendor_efficiencies="external_not_measured_locally",
        acquisition_prerequisites=dict(purchase_authorized=False, recommended=False),
        cost_scope_note="Installed inference never fills causal learning or ordinary durability. Window remainders retain setup, restart and persistence together. Unknown transfer is not zero overhead. Child and parent clocks are separate.",
        ideal_required_compatible_fraction={"10x": 0.9, "100x": 0.99},
    )
    for row, boundary in zip(value["rows"], METHOD["boundaries"], strict=True):
        row.update(
            boundary=boundary, raw_denominator=1, raw_numerator=row["value"], duration_s=None
        )
    value["gate_check_summary"] += [
        dict(
            upstream=e.TASK,
            path=r.get("stdout_path"),
            sha256=r.get("stdout_sha256"),
            check=r.get("name", "owned_validation"),
            field="passed",
            operator="==",
            expected=True,
            observed=False,
            passed=False,
            operand_state="failed_owned_validation",
        )
        for r in receipts
        if not r["passed"]
    ]
    value.pop("reproducibility_checksum")
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Bind exact invocation and producer bytes. Missing costs are unknown. Historical board access, compatible operations and complete-service benefit are separate claims.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


e.inputs, e.build, e.progress = source.i, build, source.progress

from carnot.reporting import hardware_operation_runner_8774 as r  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Use the common sealed publisher and its private output guard."""
    return int(r.main(argv))
