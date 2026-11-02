"""REQ-REPORT-8745: a fixed window cannot invent a missing lifecycle boundary."""

from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any

from carnot.reporting import resident_update_cost_8732 as previous
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

__all__ = ["atomic_json", "reference", "verify", "read", "previous"]
Json = dict[str, Any]
ROOT = previous.ROOT
NAME, TASK = "experiment_8745_v749_bounded_durable_cost", "exp8745-bounded-durable-cost"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/bounded_durable_cost_8745.py",
    "python/carnot/reporting/bounded_durable_runner_8745.py",
    CLI,
]
TESTS = ["tests/python/test_bounded_durable_cost_8745.py"]
ARMS = previous.ARMS
SEED = 7498745
MODEL_SPECS: list[Json] = []
reference, verify, read = previous.reference, previous.verify, previous.read
numeric = previous.numerical.numeric
PRODUCERS = [
    ("results/experiment_8731_v748_resident_sparse_prototype.json", previous.PIN),
    (
        "results/experiment_8732_v748_resident_update_cost.json",
        "sha256:063218e46ed7d2ad36a3bf3e4282ffde79a9888b7cdbacfcaf3178132157e6e3",
    ),
]
METHOD: Json = dict(
    offsets=[0, 2032, 4064],
    updates=32,
    repeats=10,
    arms=list(ARMS),
    process_cells=90,
    measured_updates=2880,
    child_launches=120,
    restart_after=2048,
    bootstrap_draws=10000,
    bootstrap_seed=SEED,
    bootstrap_unit="repeat_block_all_arms_and_windows_together",
    primary="equal_window_weighted_resident_over_original_native_complete_throughput",
    benefit_lower95_gt=1.05,
    maximum_p95_regression=0.05,
    reserve_fraction=0.25,
    measurement_cap_s=3000,
    prefix_policy="arithmetic_only_excluded_cost_no_full_service_extrapolation",
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Keep authentication and real storage waits visible to the parent."""
    print(f"[exp8745] phase={phase} completed={completed} pending={pending}", flush=True)


def roster() -> list[Json]:
    """Retain every intended cell even when an external prerequisite blocks it."""
    rng = random.Random(SEED)
    rows: list[Json] = []
    for repeat in range(10):
        for offset in METHOD["offsets"]:
            arms = list(ARMS)
            rng.shuffle(arms)
            rows.extend(
                dict(
                    unit_id=f"{repeat}:{offset}:{arm}",
                    repeat=repeat,
                    offset=offset,
                    arm=arm,
                    order=order,
                    updates=32,
                )
                for order, arm in enumerate(arms)
            )
    return rows


def failure(
    ref: Json,
    field: str,
    expected: Any,
    observed: Any,
    *,
    operator: str = "==",
    state: str = "observed_value",
) -> Json:
    """Missing prerequisites retain their precise byte identity and failed comparison."""
    return dict(
        upstream="exp8731-resident-sparse-prototype",
        **ref,
        check="bounded_original_lifecycle_prerequisite",
        field=field,
        operator=operator,
        expected=expected,
        observed=observed,
        operand_state=state,
        passed=False,
    )


def inputs() -> Json:
    """Authenticate retained V748 execution without requiring its retired live roadmap."""
    result: Json = dict(
        failures=[],
        refs=[],
        historical_probe={},
        trace=[],
        job={},
        trace_reference={},
        extension={},
    )
    operand: Json = {}
    try:
        values = []
        for filename, pin in PRODUCERS:
            operand = dict(path=str(ROOT / filename), sha256=pin)
            value = read(operand)
            result["refs"].append(operand)
            if not value["required_checks_passed"] or value["flagged_adversarial"]:
                raise ValueError("historical_owned_validation")
            terminal_ref = reference(Path(value["terminal_validation_sidecar_path"]))
            terminal = read(terminal_ref)
            bound_ref = reference(Path(terminal["publication"]["sidecar_path"]))
            bound = read(bound_ref)
            if (
                terminal["publication"]["primary_sha256"] != pin
                or bound["primary_sha256"] != pin
                or not bound["report"]["passed"]
            ):
                raise ValueError("historical_terminal_binding")
            result["refs"] += [terminal_ref, bound_ref]
            for ref in [value["primitive_reference"], value["validation_reference"]]:
                verify(ref)
                result["refs"].append(ref)
            checks = read(value["validation_reference"])
            if checks["receipts"] != value["validation_receipts"] or not all(
                r["passed"] for r in checks["receipts"]
            ):
                raise ValueError("historical_validation_binding")
            values.append(value)
        qualified, cost = values
        operand = result["refs"][0]
        if qualified["resident_update_ready_score"] != 1:
            raise ValueError("resident_update_ready_score_not_one")
        work = read(qualified["primitive_reference"])
        if work["private_control"]:
            raise ValueError("private_upstream")
        for ref in qualified["code_config_hashes"]:
            relative = Path(ref["path"]).relative_to(ROOT)
            operand = dict(
                path=str(Path(work["raw"]) / "source_closure" / relative), sha256=ref["sha256"]
            )
            verify(operand)
            result["refs"].append(operand)
            if str(relative).startswith(("python/carnot/verify/", "crates/")):
                verify(ref)
                result["refs"].append(ref)
        summary = work["summaries"][0]
        for key in ["job_reference", "rows_reference", "historical_reference"]:
            operand = summary[key]
            verify(operand)
            result["refs"].append(operand)
        result["job"] = read(summary["job_reference"])
        result["trace_reference"] = summary["rows_reference"]
        result["trace"] = [
            json.loads(line)
            for line in Path(result["trace_reference"]["path"]).read_text().splitlines()
        ]
        if len(result["trace"]) != 4096 or result["job"]["boundary"]:
            raise ValueError("original_4096_trajectory")
        result["extension"] = qualified["loaded_extension_reference"]
        operand = result["extension"]
        verify(operand)
        if not operand["actual_loaded"]:
            raise ValueError("historical_binary_not_loaded")
        result["refs"].append(operand)
        result["historical_probe"] = cost["durability_probe"]
        if (
            cost["honest_verdict"] != "complete_blocked_durability_budget"
            or result["historical_probe"]["observed_updates"] != 4
            or result["historical_probe"]["projected_minimum_measurement_s"] != 71507.5252224
        ):
            raise ValueError("preserved_full_workload_block")
        cost_work = read(cost["primitive_reference"])
        for ref in cost_work["raw_refs"]:
            operand = ref
            verify(ref)
            result["refs"].append(ref)
        original_job = Path(cost_work["raw"]) / "job-0.json"
        operand = reference(original_job)
        if read(operand) != result["job"]:
            raise ValueError("original_durable_job")
        result["refs"].append(operand)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["failures"].append(
            failure(
                operand,
                "qualified_original_operands",
                "authenticated",
                str(error),
                state="missing_file"
                if isinstance(error, FileNotFoundError)
                else "invalid_or_changed_operand",
            )
        )
    return result


def starting_states(admitted: Json, raw: Path) -> tuple[list[Json], list[Json], Json]:
    """Rebuild arithmetic only; verify every old state and acknowledgement before timing."""
    job, trace = admitted["job"], admitted["trace"]
    f = numeric.Fixed(32, 24)
    state = numeric.encode(job["head"], f)
    ack = canonical_hash(dict(job_hash=canonical_hash(job), next_index=0, state=state))
    starts, costs = [], []
    begin = time.perf_counter_ns()
    for index in range(4097):
        if index in METHOD["offsets"]:
            target = raw / f"start-{index}.json"
            atomic_json(
                target,
                dict(
                    state=state,
                    counts=f.counts,
                    operations=f.operations,
                    maximum=f.maximum,
                    ack=ack,
                    next_index=index,
                ),
            )
            starts.append(
                dict(
                    offset=index,
                    state_hash=canonical_hash(state),
                    ack=ack,
                    reference=reference(target),
                    passed=True,
                )
            )
            costs.append(
                dict(
                    offset=index,
                    excluded_arithmetic_prefix_updates=index,
                    prefix_construction_cumulative_ns=time.perf_counter_ns() - begin,
                    durability_omitted=True,
                    whole_service_extrapolation_allowed=False,
                )
            )
        if index == 4096:
            break
        x = job["vectors"][index % len(job["vectors"])]["x"]
        label = job["labels"][index % len(job["labels"])]
        state, receipt = numeric.step(state, x, label, f)
        p, _ = numeric.probability(state, x, f)
        ack = canonical_hash(dict(previous_ack=ack, next_index=index + 1, state=state))
        row = trace[index]
        if (
            row["update_index"] != index + 1
            or row["state_hash"] != canonical_hash(state)
            or row["ack"] != ack
            or row["probability"] != p
            or not row["passed"]
            or row["dense_writes"] != receipt["renormalization_writes"]
            or not row["counters_identical"]
        ):
            raise ValueError("original_parity_trace")
        if (index + 1) % 256 == 0:
            progress("arithmetic_prefix_verified", index + 1, 4096 - index - 1)
    dense = sum(r["dense_writes"] for r in trace[2032:2064])
    gate = failure(
        admitted["trace_reference"],
        "middle_window_dense_renormalization_writes",
        1,
        dense,
        operator=">=",
    )
    gate["passed"] = dense >= 1
    return starts, costs, gate


def projection(latencies: list[int], complete_ns: int) -> Json:
    """Use the slowest actual update and complete launch overhead with the fixed reserve."""
    if len(latencies) != 4 or min(latencies) <= 0 or complete_ns < 0:
        raise ValueError("probe_contract")
    overhead = max(0, complete_ns - sum(latencies))
    seconds = (2880 * max(latencies) + 120 * overhead) / 1e9 * 1.25
    return dict(
        maximum_latency_ns=max(latencies),
        measured_process_overhead_ns=overhead,
        projected_with_reserve_s=seconds,
        reserve_fraction=0.25,
        durable_updates=2880,
        process_launches=120,
        cap_s=3000,
        admitted=seconds <= 3000,
    )


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """A historical missing boundary cannot become zero cost or a bounded benefit."""
    bad = any(not r["passed"] for r in receipts)
    verdict = "disqualified" if bad else "blocked"
    raw = Path(work["raw"])
    rows = [
        dict(
            r,
            status="censored",
            censored=True,
            excluded=False,
            passed=False,
            complete_ns=None,
            observed_updates=0,
        )
        for r in work["roster"]
    ]
    value = dict(
        experiment_id=8745,
        task_id=TASK,
        milestone="2026.11.749",
        run_date=work["run_date"],
        invocation_id=raw.name,
        honest_verdict="complete_" + verdict + "_original_window_prerequisite",
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0, generation_calls=0),
        numerical_training=dict(performed=False, arithmetic_prefix_replay_only=True),
        rows=rows,
        intended_count=90,
        completed_count=0,
        failed_count=0,
        censored_count=90,
        excluded_count=0,
        independent_count=0,
        verifier_is_oracle=True,
        exposure_scope="historical_constructed_parity_trajectory",
        independent_generalization_score=0,
        physical_fpga_execution=False,
        bounded_update_cost_ready_score=0,
        bounded_resident_benefit_score=0,
        starting_state_rows=work["starting_state_rows"],
        window_process_rows=[],
        durable_update_rows=[],
        durability_probe=work["durability_probe"],
        resource_projection=work["resource_projection"],
        restart_rows=[],
        paired_window_intervals={},
        excluded_prefix_cost_rows=work["excluded_prefix_cost_rows"],
        long_run_claim_allowed=False,
        frozen_methods=METHOD,
        historical_full_workload_probe=work["historical_probe"],
        loaded_extension_reference=work["loaded_extension_reference"],
        acceptance_gates=dict(
            owned_validation=not bad,
            authenticated_original_operands=bool(work["starting_state_rows"]),
            actual_middle_renormalization=False,
            complete_windows=False,
            bounded_resident_benefit=False,
            resource_admission=work["resource_projection"].get("admitted", False),
        ),
        required_checks_passed=not bad,
        flagged_adversarial=bad,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=SEED,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["raw_refs"],
        cited_upstream_artifacts=work["source_refs"],
        primitive_reference=reference(raw / "measurement.json"),
        validation_reference=reference(raw / "validation.json"),
        publication_output=str(output),
        methodology_note="Fixed90-cell plan remains unexecuted: authenticated original4096 "
        "trajectory has zero dense writes in updates2033..2064. No boundary is injected. "
        "Four durable probe updates are admission evidence only. Arithmetic prefixes omit "
        "durability and cannot support uninterrupted throughput, useful learning or hardware speed.",
        preserved_historical_limits=[
            "Exp8732 full122880-update design remains blocked at "
            "71507.5252224 seconds minimum projection.",
            "No useful learning, current CUDA repair, new ARC outcome "
            "or physical acceleration is established.",
        ],
    )
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            method=METHOD,
            code=work["code_refs"],
            sources=work["source_refs"],
            roster=work["roster"],
        )
    )
    value["field_principles"] = {
        k: "Bind observed work to exact bytes; missing bounded "
        "measurements remain censored and cannot support benefit."
        for k in value
    }
    return value
