"""REQ-VERIFY-8121: count memory work without claiming a device measurement."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from carnot.reporting import radial_hardware_8108 as prior
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
CONFIG: Json = dict(
    seed=7028121,
    batches=[1, 8, 32, 128],
    centers=[16, 28],
    dimensions=9,
    bits=[8, 12, 16],
    fallback="worst_case_every_request",
)
COSTS = ("radial_ns", "acquisition_ns", "transfer_ns", "fallback_ns", "persistence_ns", "other_ns")
MAPPING = [
    "scaled radial distance",
    "Gaussian exp/LUT with analytic error contract",
    "center/coefficient versioned memory transfer",
    "sigmoid and typed decision",
    "float64 fallback",
    "feedback update and durable synchronization",
    "measured device transport and complete transaction",
]


def traffic(batch: int, centers: int, bits: int, fallback: int, scope: str) -> Json:
    """Count exact packed bytes and operations under explicit transfer assumptions.

    Storage rounds up the final partially occupied byte. Resident centers need
    only input and result traffic after installation. Streaming reads the entire
    head for every request. Neither assumption is a measured bus transaction.
    """
    if batch < 1 or centers < 1 or bits not in CONFIG["bits"] or not 0 <= fallback <= batch:
        raise ValueError("traffic_domain")
    dimension = CONFIG["dimensions"]
    cb = math.ceil(centers * dimension * bits / 8)
    wb = math.ceil((centers + 1) * bits / 8)
    base = batch * (dimension * 8 + 8)
    return dict(
        unit_id=f"{scope}-b{batch}-m{centers}-q{bits}",
        source_cluster_id=scope,
        arm=f"packed_{bits}",
        condition="analytic_operation_count",
        metric="streaming_transfer_bytes",
        numerator=base + batch * (cb + wb),
        denominator=1,
        status="completed",
        exclusion_reason=None,
        batch_size=batch,
        centers=centers,
        bits=bits,
        center_storage_bytes=cb,
        coefficient_storage_bytes=wb,
        head_storage_bytes=cb + wb,
        float64_head_storage_bytes=(centers * dimension + centers + 1) * 8,
        resident_transfer_bytes=base,
        cold_install_transfer_bytes=base + cb + wb,
        streaming_transfer_bytes=base + batch * (cb + wb),
        actual_device_transfer_bytes=None,
        durable_writes=0,
        durable_write_scope="prediction fixture has no state update; observed updates separate",
        distance_evaluations=batch * centers,
        distance_coordinate_terms=batch * centers * dimension,
        gaussian_evaluations=batch * centers,
        fallback_requests=fallback,
        fallback_distance_evaluations=fallback * centers,
        fallback_float64_head_read_bytes=fallback * (centers * dimension + centers + 1) * 8,
        scope=scope,
        assumptions="float64 inputs/results; ideal bit packing; no framing/cache reuse charge",
    )


def service(rows: list[Json]) -> Json:
    """Remove only radial arithmetic from complete nonoverlapping transactions.

    Acquisition and fallback remain charged even when the radial device is free.
    Missing transfer or persistence is unknown cost, so no ceiling is published.
    """
    seen: set[tuple[str, str]] = set()
    try:
        if not rows:
            raise ValueError("no_qualified_whole_service_rows")
        for row in rows:
            identity = (row["unit_id"], row["arm"])
            if identity in seen or any(type(row.get(k)) is not int or row[k] < 0 for k in COSTS):
                raise ValueError("incomplete_or_duplicate_cost:" + str(identity))
            seen.add(identity)
            if (
                type(row["total_ns"]) is not int
                or row["total_ns"] <= row["radial_ns"]
                or sum(row[k] for k in COSTS) != row["total_ns"]
            ):
                raise ValueError("nonoverlapping_cost_identity:" + str(identity))
        total = sum(r["total_ns"] for r in rows)
        radial = sum(r["radial_ns"] for r in rows)
        bound = total / (total - radial)
        return dict(
            status="available",
            rows=[
                dict(r, conditional_speedup_bound=r["total_ns"] / (r["total_ns"] - r["radial_ns"]))
                for r in rows
            ],
            conditional_speedup_bound=bound,
            arithmetic_fraction=radial / total,
            target_100x="infeasible_under_measured_costs"
            if bound < 100
            else "feasible_only_if_device_costs_fit_remaining_budget",
            target_100x_added_cost_budget_ns=total / 100 - (total - radial),
            assumption="All submitted costs measured; free radial arithmetic adds zero device cost.",
        )
    except (KeyError, TypeError, ValueError) as error:
        return dict(
            status="unavailable",
            rows=[],
            conditional_speedup_bound=None,
            arithmetic_fraction=None,
            target_100x="blocked_unavailable_transaction_costs",
            failed_operand=str(error),
        )


def reduce(data: Json) -> Json:
    """Keep valid board rows and optional scientific blocks independent.

    Readiness means this mapping assessment completed. It never means a device
    ran, the 100x goal was met, or exposed fixtures established generalization.
    """
    boards = deepcopy(data["boards"])
    for board in boards:
        valid = (
            board["custody_valid"]
            and (board.get("processor_class"), board.get("k_max"), board.get("blocker"))
            == prior.CONTRACTS[board["board"]]
        )
        board.update(
            custody_valid=bool(valid),
            status="completed" if valid else "blocked",
            current_hardware_execution=False,
            radial_mapping_implemented=False,
        )
    print(
        f"[exp8121] before_numerical_replay completed=0 pending={len(data['systems'])}", flush=True
    )
    precision: list[Json] = []
    for i, system in enumerate(data["systems"]):
        precision.extend(
            dict(prior.numerical(system, bits), source_cluster_id="exposed_numerical_fixture")
            for bits in CONFIG["bits"]
        )
        if (i + 1) % 16 == 0:
            print(
                f"[exp8121] numerical_replay completed={i + 1} pending={len(data['systems']) - i - 1}",
                flush=True,
            )
    print(
        f"[exp8121] after_numerical_replay completed={len(data['systems'])} pending=0", flush=True
    )
    numerical = prior.summarize(precision)
    counts = [
        traffic(
            b,
            m,
            q,
            b,
            "qualified_batch_operation_fixture"
            if data["batch_available"]
            else "operation_count_fixture",
        )
        for b in data["batches"]
        for m in CONFIG["centers"]
        for q in CONFIG["bits"]
    ]
    costs = service(data["costs"])
    ready = int(data["numerical_available"] and numerical["passed"])
    klass = "circular_positive" if data["fixture"] else "null"
    verdict = "complete_" + klass + "_hardware_batch_boundary"
    if not ready:
        klass = "blocked"
        reason = next(
            (c["check"] for c in data["checks"] if not c["passed"]), "exp8108_numerical_inputs"
        )
        verdict = "complete_blocked_" + reason
    if not numerical["passed"]:
        klass, verdict = "disqualified", "complete_disqualified_numerical_envelope"
    return dict(
        honest_verdict=verdict,
        verdict_class=klass,
        hardware_boundary_ready_score=ready,
        hardware_execution=False,
        board_rows=boards,
        operation_count_rows=counts,
        precision_rows=precision,
        precision_summary=numerical,
        rows=counts,
        conditional_speedup_bound=costs["conditional_speedup_bound"],
        service_cost_subresult=costs,
        gate_check_summary=data["checks"],
        missing_mapping_operations=MAPPING,
        observed_touch_rows=data["updates"],
        batch_input_status="available" if data["batch_available"] else "unavailable",
        online_touch_input_status="available" if data["touches_available"] else "unavailable",
        subinput_rows=[
            dict(
                upstream=f"exp{eid}",
                status="completed" if data[field] else "blocked",
                verdict_class="null" if data[field] else "blocked",
                honest_verdict="complete_null_qualified_subinput"
                if data[field]
                else "complete_blocked_exp" + str(eid) + "_qualification",
                failed_operands=[
                    c
                    for c in data["checks"]
                    if c.get("upstream") == f"exp{eid}" and not c["passed"]
                ],
            )
            for eid, field in [(8119, "batch_available"), (8116, "touches_available")]
        ],
        intended_count=len(counts),
        eligible_count=len(counts),
        completed_count=len(counts),
        independent_count=0,
        excluded_count=0,
        censored_count=0,
        failed_count=0,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        verifier_is_oracle=data["fixture"],
        trained_head_specs=[],
        sample_size_budget=dict(
            operation_count_fixtures=len(counts),
            independent_sources=0,
            repeated_batches_are_not_sources=True,
        ),
        acceptance_gates=dict(
            numerical_inputs=bool(ready),
            board_gates_independent=True,
            optional_whole_service=costs["status"] == "available",
            required_arithmetic_fraction_for_100x=0.99,
        ),
        numerical_bound_scope="Exp8108 analytic Gaussian/sigmoid storage envelope, declared host rounding; float64 decisions. No LUT/fixed-point/global sample guarantee.",
        operation_count_checksum=canonical_hash(counts),
        research_program_100x=dict(
            target=100,
            disposition=costs["target_100x"],
            measured_arithmetic_free_upper_bound=costs["conditional_speedup_bound"],
            necessary_removable_fraction=0.99,
            local_speedup_measured=False,
            vendor_estimates_count_as_local_speedup=False,
        ),
        reopen_conditions=dict(
            KV260="Authenticated radial distance/Gaussian LUT mapping within fabric limits; versioned center transfer and complete transaction timings; existing Ising k_max<=5 unchanged.",
            PolarFire="Measured board-local radial CPU transactions including acquisition, transport, float64 fallback and persistence; Linux dispatch alone establishes no fabric acceleration.",
            GateMate="Operator physical/JTAG change resolving 0xffffffff, then authorized mapped radial workload with custody and full cost receipts; unchanged probes have no value.",
            TSU="Authorized hardware access plus explicit dense radial distance/LUT encoding, sparse connectivity embedding and measured acquisition/transfer/readout/fallback/persistence.",
            GPU="Qualified batched radial kernel and observed online touches with complete transfer and durable transaction measurement.",
            NPU="Authorized supported distance/exp operator mapping and equal-correctness complete transaction benchmark.",
        ),
        source_scan_contrast=dict(
            dense_radial="Every Gaussian center contributes; batch*m*d distance terms and potentially all m+1 coefficient updates. Small values do not prove compact support.",
            sparse_ising="Existing fabric and TSU use sparse pairwise couplings; Gaussian distances and exp/LUT require a separate encoding.",
            spline_locality="B-splines have compact support and sparse active updates; dense Gaussian centers do not inherit that sparsity.",
            references=[
                "https://arxiv.org/abs/2602.02056",
                "https://arxiv.org/abs/2602.15985",
                "https://extropic.ai/writing/z1t",
            ],
            vendor_scope="Z1T estimates omit final dense readout and some movement; energy efficiency estimates cannot establish local whole-service latency speedup.",
        ),
    )
