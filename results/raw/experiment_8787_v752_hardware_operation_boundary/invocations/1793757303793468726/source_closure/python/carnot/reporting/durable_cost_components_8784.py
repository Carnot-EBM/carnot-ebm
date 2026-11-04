"""REQ-REPORT-8784: bounded costs and attribution have independent readiness gates."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from carnot.reporting import durable_producer_identity_8783 as identity
from carnot.reporting import ordinary_durable_cost_8772 as historical
from carnot.reporting import durable_component_clocks_8784 as clocks
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
__all__ = ["historical", "identity", "atomic_json", "canonical_hash"]
ROOT, ARMS = historical.ROOT, historical.ARMS
NAME, TASK = "experiment_8784_v752_durable_cost_components", "exp8784-durable-cost-components"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/durable_component_clocks_8784.py",
    "python/carnot/reporting/durable_cost_components_8784.py",
    "python/carnot/reporting/durable_component_runner_8784.py",
    CLI,
]
TESTS = ["tests/python/test_durable_cost_components_8784.py"]
SEED = 7528784
MODEL_SPECS: list[Json] = []
UPSTREAM = ROOT / "results/experiment_8783_v752_durable_producer_identity.json"
PIN = "sha256:537f1632af0e69620b63d2ca62838a6d9346db9917edc7cc680d2d0ae901924e"
reference, verify, read = historical.reference, historical.verify, historical.read
METHOD = dict(
    historical.METHOD,
    repeats=2,
    process_cells=36,
    measured_updates=1152,
    child_launches=72,
    restart="after_16_updates_every_window",
    maximum_observer_overhead=0.05,
    maximum_reconciliation_error=0.02,
    bootstrap_draws=0,
    bootstrap_seed=SEED,
    descriptive_only=True,
)
measurement = namespace(historical, "components8784")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose phase boundaries and actual counts during disk and subprocess waits."""
    if phase in {"sources_plan_frozen", "window_before", "window_after"}:
        pending = max(0, len(roster()) - completed)
    print(f"[exp8784] phase={phase} completed={completed} pending={pending}", flush=True)


def descriptor(invocation: str) -> Json:
    """A task alias or reused invocation cannot acquire producer authority."""
    return dict(
        experiment_id=8784,
        task_id=TASK,
        milestone="2026.11.752",
        invocation_id=invocation,
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
    )


def roster() -> list[Json]:
    """Randomize arms and observer order once, before seeing any timings."""
    rng = random.Random(SEED)
    rows = []
    for repeat in range(2):
        for offset in METHOD["offsets"]:
            arms = list(ARMS)
            rng.shuffle(arms)
            for order, arm in enumerate(arms):
                conditions = ["plain", "instrumented"]
                rng.shuffle(conditions)
                for condition in conditions:
                    rows.append(
                        dict(
                            unit_id=f"{repeat}:{offset}:{arm}:{condition}",
                            repeat=repeat,
                            offset=offset,
                            arm=arm,
                            order=order,
                            condition=condition,
                            updates=32,
                        )
                    )
    return rows


def failure(
    ref: Json,
    field: str,
    expected: Any,
    observed: Any,
    state: str = "observed_value",
    operator: str = "==",
) -> Json:
    """Keep absent files, absent fields, zero readiness and rejected producers distinct."""
    return dict(
        historical.failure(ref, field, expected, observed, state, operator),
        upstream=identity.TASK,
        check="exact_durable_producer_admission",
    )


def inputs() -> Json:
    """Use exact identity qualification while retaining original lifecycle assertions."""
    ref = dict(path=str(UPSTREAM), sha256=PIN)
    result: Json = dict(refs=[ref], failures=[])
    field = "durable_producer_ready_score"
    try:
        producer = read(ref)
        if field not in producer or producer[field] != 1:
            result["failures"].append(
                failure(
                    ref,
                    field,
                    1,
                    producer.get(field),
                    "missing_field" if field not in producer else "observed_zero",
                )
            )
            return result
        if (
            producer["task_id"] != identity.TASK
            or producer["experiment_id"] != 8783
            or not producer["required_checks_passed"]
            or producer["flagged_adversarial"]
        ):
            raise ValueError("rejected_producer")
        terminal = read(reference(Path(producer["terminal_validation_sidecar_path"])))
        if terminal["publication"]["primary_sha256"] != PIN or not terminal["passed"]:
            raise ValueError("upstream_terminal_binding")
        result = identity.inputs()
        result["refs"] += [ref, reference(Path(producer["terminal_validation_sidecar_path"]))]
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["failures"].append(
            failure(
                ref,
                field,
                1,
                str(error),
                "missing_file" if isinstance(error, FileNotFoundError) else "rejected_producer",
            )
        )
    return result


def projection(latencies: list[int], complete_ns: int) -> Json:
    """Reserve900 seconds for validation and reject an unaffordable complete study."""
    value: Json = historical.projection(latencies, complete_ns)
    value["projected_with_reserve_s"] = (
        (1152 * max(latencies) + 72 * value["measured_process_overhead_ns"]) / 1e9 * 1.25
    )
    value.update(
        durable_updates=1152,
        process_launches=72,
        admitted=value["projected_with_reserve_s"] <= 3000,
    )
    return value


def reduce(rows: list[Json]) -> Json:
    """Six process pairs describe one workload; overhead rejects attribution alone."""
    lookup = {(r["repeat"], r["offset"], r["arm"], r["condition"]): r for r in rows}
    overhead, pairs, components, residual, restarts, processes = [], [], [], [], [], []
    for repeat in range(2):
        for offset in METHOD["offsets"]:
            costs = {}
            for arm in ARMS:
                plain = lookup.get((repeat, offset, arm, "plain"), {})
                observed = lookup.get((repeat, offset, arm, "instrumented"), {})
                complete = all(r.get("status") == "completed" for r in [plain, observed])
                ratio = observed["complete_ns"] / plain["complete_ns"] if complete else None
                overhead.append(
                    dict(
                        repeat=repeat,
                        offset=offset,
                        arm=arm,
                        ratio=ratio,
                        overhead_fraction=ratio - 1 if ratio is not None else None,
                        passed=ratio is not None and ratio <= 1.05,
                    )
                )
                if complete:
                    costs[arm] = plain["complete_ns"]
                for row in [plain, observed]:
                    parent_total = 0
                    for segment, command in zip(
                        row.get("segments", []), row.get("commands", []), strict=False
                    ):
                        reduced = clocks.partition(
                            segment["spans"], segment["begin_ns"], segment["end_ns"]
                        )
                        reduced["costs_ns"].update(
                            process_startup=segment["begin_ns"] - command["started_monotonic_ns"],
                            process_shutdown=command["ended_monotonic_ns"] - segment["end_ns"],
                        )
                        parent_ns = command["ended_monotonic_ns"] - command["started_monotonic_ns"]
                        parent_total += parent_ns
                        processes.append(
                            dict(
                                unit_id=row["unit_id"],
                                start=segment["start"],
                                stop=segment["stop"],
                                duration_ns=parent_ns,
                                complete_amortized_ns=parent_ns / 16,
                                costs_ns=reduced["costs_ns"],
                                unassigned_ns=reduced["unassigned_ns"],
                                world_count=1,
                                workload_count=1,
                                denominator_updates=16,
                            )
                        )
                        components += [
                            dict(
                                unit_id=row["unit_id"],
                                segment=segment["start"],
                                component=k,
                                cost_ns=v,
                                world_count=1,
                                workload_count=1,
                            )
                            for k, v in reduced.get("costs_ns", {}).items()
                        ]
                        residual.append(
                            dict(
                                unit_id=row["unit_id"],
                                segment=segment["start"],
                                cost_ns=reduced.get("unassigned_ns"),
                                reconciliation_error_fraction=reduced.get(
                                    "reconciliation_error_fraction", 1
                                ),
                            )
                        )
                    if row and row.get("complete_ns") is not None:
                        residual.append(
                            dict(
                                unit_id=row["unit_id"],
                                segment="parent_orchestration",
                                cost_ns=row["complete_ns"] - parent_total,
                                reconciliation_error_fraction=abs(
                                    sum(
                                        p["duration_ns"]
                                        for p in processes
                                        if p["unit_id"] == row["unit_id"]
                                    )
                                    - parent_total
                                )
                                / row["complete_ns"],
                            )
                        )
                    if row:
                        restarts.append(
                            dict(
                                unit_id=row["unit_id"],
                                passed=row.get("restart_verified", False),
                                process_lifetimes=len(row.get("commands", [])),
                            )
                        )
            pairs.append(
                dict(
                    repeat=repeat,
                    offset=offset,
                    world_count=1,
                    workload_count=1,
                    python_over_resident_ratio=costs["python"] / costs["resident_native"]
                    if len(costs) == 3
                    else None,
                    json_native_over_resident_ratio=costs["json_native"] / costs["resident_native"]
                    if len(costs) == 3
                    else None,
                )
            )
    component_ready = (
        len(rows) == 36
        and len(processes) == 72
        and all(r["passed"] for r in overhead)
        and bool(residual)
        and all(
            r["cost_ns"] is not None and r["reconciliation_error_fraction"] <= 0.02
            for r in residual
        )
    )
    return dict(
        per_window_rows=rows,
        per_update_rows=[u for r in rows for u in r["update_rows"]],
        observer_overhead_rows=overhead,
        process_pair_rows=pairs,
        component_cost_rows=components,
        unassigned_cost_rows=residual,
        restart_rows=restarts,
        per_process_rows=processes,
        component_ready=component_ready,
    )


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """A complete null is useful cost evidence even when the observer is too expensive."""
    raw = Path(work["raw"])
    reduced = reduce(work["window_rows"])
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    coverage = any(r["name"] == "coverage_report" and r["passed"] for r in receipts)
    complete = sum(r["status"] == "completed" for r in work["window_rows"])
    execution = (
        complete == len(work["roster"])
        and (work["private_control"] or work["roster"] == roster())
        and all(r["passed"] for r in reduced["restart_rows"])
        and all(u["passed"] for u in reduced["per_update_rows"])
        and len(reduced["per_update_rows"]) == 32 * len(work["roster"])
        and all(
            len(r["segments"]) == 2
            and sum(s["durable_commits"] for s in r["segments"]) == 64
            and sum(s["fsync_calls"] for s in r["segments"]) >= 256
            for r in work["window_rows"]
        )
    )
    bad = (
        not owned
        or (not work["private_control"] and not coverage)
        or (not work["failures"] and not execution)
    )
    ready = (
        owned and coverage and execution and not work["failures"] and not work["private_control"]
    )
    verdict = "disqualified" if bad else "blocked" if work["failures"] else "null"
    gates = dict(
        owned_validation=owned,
        owned_statement_coverage=coverage,
        authenticated_sources=not work["failures"],
        complete_costs=execution,
        component_attribution=reduced["component_ready"],
    )
    value = dict(
        descriptor(raw.name),
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_durable_cost_components",
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="deterministic_verifier",
        model_invocation_counts=dict(language_model_calls=0, model_loads=0, generation_calls=0),
        numerical_training_calls=0,
        numerical_training=dict(performed=False, arithmetic_replay_only=True),
        rows=work["window_rows"],
        intended_count=len(work["roster"]),
        completed_count=complete,
        failed_count=sum(r["status"] == "failed" for r in work["window_rows"]),
        censored_count=sum(r["status"] == "censored" for r in work["window_rows"]),
        excluded_count=0,
        independent_count=int(complete > 0),
        world_count=1,
        workload_count=1,
        verifier_is_oracle=True,
        exposure_scope="constructed_ordinary_durable_trajectory",
        independent_generalization_score=0,
        durable_cost_rows_ready_score=int(ready),
        component_cost_ready_score=int(ready and reduced["component_ready"]),
        acceptance_gates=gates,
        required_checks_passed=not bad,
        flagged_adversarial=bad,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=SEED,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["raw_refs"],
        cited_upstream_artifacts=work["source_refs"],
        repository_health_reference=work.get("repository_health_reference", {}),
        scoped_validation_scope=dict(
            owned=OWNED,
            tests=TESTS,
            named_e2es=["E2E-018", "E2E-021"],
            full_suite_pass_claimed=False,
        ),
        prefix_preparation_cost_rows=work["excluded_prefix_cost_rows"],
        ordinary_starting_state_rows=work["starting_state_rows"],
        primitive_reference=reference(raw / "measurement.json"),
        validation_reference=reference(raw / "validation.json"),
        publication_output=str(output),
        private_control=work["private_control"],
        frozen_methods=METHOD,
        resource_projection=work["resource_projection"],
        long_run_claim_allowed=False,
        historical_cost_intervals_admitted=False,
        physical_fpga_execution=False,
        component_boundary_note="boundary_transfer includes native computation, telemetry and binding transit; pure transfer is not separately identifiable. update_arithmetic is exclusive host work. Serialization includes durable envelope hashing and encoding. Unassigned includes I/O, verification, timer bookkeeping and parent orchestration.",
        methodology_note="Three frozen ordinary starting states, two paired process repeats, three randomized arms and plain/instrumented conditions. Every32-update window restarts after16 and verifies acknowledged state after restart. Complete parent time includes startup, import, persistence, verification and shutdown. Prefix arithmetic is separate. Six process pairs describe one constructed workload, not population speedup, natural dense-event frequency, useful learning, whole-service throughput or physical acceleration.",
    )
    value.update({k: v for k, v in reduced.items() if k != "component_ready"})
    for row in value["component_cost_rows"]:
        row["admitted"] = bool(value["component_cost_ready_score"])
    value["descriptive_ratio_ranges"] = {
        key: dict(minimum=min(values), maximum=max(values), mean=sum(values) / len(values))
        for key in ["python_over_resident_ratio", "json_native_over_resident_ratio"]
        if (values := [r[key] for r in reduced["process_pair_rows"] if r[key] is not None])
    }
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            method=METHOD,
            sources=work["source_refs"],
            code=work["code_refs"],
            roster=work["roster"],
        )
    )
    value["field_principles"] = {
        k: "Bind actual task identity and sealed measured operands. Repeats remain one workload; readiness does not claim an advantage."
        for k in value
    }
    return value


measurement.NAME, measurement.TASK, measurement.CLI = NAME, TASK, CLI
measurement.OWNED, measurement.TESTS, measurement.SEED = OWNED, TESTS, SEED
measurement.METHOD, measurement.roster = METHOD, roster
measurement.progress, measurement.projection, measurement.build = progress, projection, build
measurement.inputs = inputs
