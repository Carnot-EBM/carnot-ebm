"""REQ-REPORT-8758: qualify actual events without inferring costs from ordinals."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import bounded_durable_cost_8745 as previous
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import resident_sparse_8731 as resident

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8758_v750_lifecycle_event_qualification"
TASK = "exp8758-lifecycle-event-qualification"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/lifecycle_event_8758.py",
    "python/carnot/reporting/lifecycle_event_runner_8758.py",
    CLI,
]
TESTS = ["tests/python/test_lifecycle_event_8758.py"]
MODEL_SPECS: list[Json] = []
ARMS = ("python", "json_native", "resident")
reference, verify, read = previous.reference, previous.verify, previous.read
numeric = resident.numeric
METHOD: Json = dict(
    offsets=[0, 2032, 4064],
    updates=32,
    restart_after=2048,
    arms=list(ARMS),
    threshold_offsets=[-2, -1, 0, 1, 2],
    natural_renormalization_required=False,
    timing_performed=False,
    scope="ordinary sparse updates and restart; no renormalization timing",
    durable_cost_successor="exp8759-ordinary-durable-cost",
    repeats=10,
    process_cells=90,
    durable_updates=2880,
    reserve_fraction=0.25,
    measurement_cap_s=3000,
    synthetic_controls_enter_costs=False,
    complete_boundary="setup, exact state import, 2 Store commits/update, restart and shutdown",
    persistence="file fsync, atomic replacement, directory fsync before acknowledgment",
    prefix_policy="report arithmetic preparation separately; no uninterrupted throughput extrapolation",
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real counts let the parent distinguish live arithmetic from a stalled child."""
    print(f"[exp8758] phase={phase} completed={completed} pending={pending}", flush=True)


def scalar(before: int) -> tuple[int, bool]:
    """Compute ties to even independently of the numerical update implementation."""
    unit = 1 << 24
    multiplier = round((1 - 0.002 * 0.01) * unit)
    quotient, remainder = divmod(before * multiplier, unit)
    rounded = quotient + int(2 * remainder > unit or (2 * remainder == unit and quotient % 2))
    return rounded, rounded < max(2, unit // 256)


def inputs() -> Json:
    """Historical qualification supplies operands; the rejected cost producer supplies no gate."""
    value: Json = previous.inputs()
    value["threshold_jobs"] = []
    if value["failures"]:
        return value
    operand = value["refs"][0]
    try:
        producer = read(operand)
        work = read(producer["primitive_reference"])
        for summary in work["summaries"][20:]:
            for key in ["job_reference", "rows_reference"]:
                operand = summary[key]
                verify(operand)
                value["refs"].append(operand)
            value["threshold_jobs"].append(read(summary["job_reference"]))
        value["historical_restart"] = work["restart_rows"][0]
        value["refs"].append(
            reference(ROOT / "results/experiment_8745_v749_bounded_durable_cost.json")
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        value["failures"].append(
            dict(
                upstream="exp8731-resident-sparse-prototype",
                **operand,
                check="retained_threshold_controls",
                field="threshold_jobs",
                operator="==",
                expected="authenticated five controls",
                observed=str(error),
                operand_state="missing_file"
                if isinstance(error, FileNotFoundError)
                else "invalid_or_changed_operand",
                passed=False,
            )
        )
    return value


def reconstruct(admitted: Json, raw: Path) -> Json:
    """Rebuild each original boundary before importing any ordinary window."""
    job, trace = admitted["job"], admitted["trace"]
    f = numeric.Fixed(32, 24)
    state = numeric.encode(job["head"], f)
    ack = canonical_hash(dict(job_hash=canonical_hash(job), next_index=0, state=state))
    starts, events, scalars = [], [], []
    for index in range(4097):
        if index in [0, 2032, 2048, 4064]:
            path = raw / f"start-{index}.json"
            atomic_json(
                path,
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
                    reference=reference(path),
                    passed=True,
                )
            )
        if index == 4096:
            break
        before = state["decay_scale"]
        expected_scale, dense = scalar(before)
        x = job["vectors"][index % len(job["vectors"])]["x"]
        state, receipt = numeric.step(state, x, job["labels"][index % len(job["labels"])], f)
        p, _ = numeric.probability(state, x, f)
        ack = canonical_hash(dict(previous_ack=ack, next_index=index + 1, state=state))
        old = trace[index]
        if not (
            old["update_index"] == index + 1
            and old["state_hash"] == canonical_hash(state)
            and old["ack"] == ack
            and old["probability"] == p
            and old["passed"]
            and old["counters_identical"]
            and old["dense_writes"] == receipt["renormalization_writes"]
            and state["decay_scale"] == (16776880 if dense else expected_scale)
            and receipt["renormalization_writes"] == 109 * dense
        ):
            raise ValueError("original_parity_trace")
        scalars.append(
            dict(
                update_index=index + 1,
                before_decay_scale=before,
                rounded_decay_scale=expected_scale,
                threshold=65536,
                predicate=dense,
                after_decay_scale=state["decay_scale"],
                passed=True,
            )
        )
        events.append(
            dict(
                update_index=index + 1,
                dense_writes=receipt["renormalization_writes"],
                local_writes=receipt["local_writes"],
                ack=ack,
                restart_after_update=index + 1 == 2048,
                synthetic_boundary=False,
                passed=True,
            )
        )
        if (index + 1) % 256 == 0:
            progress("original_arithmetic_replayed", index + 1, 4095 - index)
    return dict(
        ordinary_starting_state_rows=starts, natural_event_rows=events, scalar_decay_rows=scalars
    )


def worker(request: Json, output: Path) -> Json:
    """Use original arithmetic and telemetry, retaining full boundaries outside timing."""
    arm, job, saved = request["arm"], request["job"], request["snapshot"]
    extension = None
    if arm != "python":
        progress("native_load_before")
        extension = resident.load(Path(request["extension"]["path"]))
        if reference(Path(extension.__file__))["sha256"] != request["extension"]["sha256"]:
            raise ValueError("loaded_binary_hash")
        progress("native_load_after")
    f: Any = (
        numeric.Fixed(32, 24)
        if arm == "python"
        else resident.original.NativeFixed(extension)
        if arm == "json_native"
        else resident.ResidentFixed(extension)
    )
    state, ack = deepcopy(saved["state"]), saved["ack"]
    f.counts, f.operations, f.maximum = (
        deepcopy(saved["counts"]),
        deepcopy(saved["operations"]),
        saved["maximum"],
    )
    step = (
        numeric.step
        if arm == "python"
        else resident.original.step
        if arm == "json_native"
        else resident.engine.step
    )
    probability = (
        numeric.probability
        if arm == "python"
        else resident.original.probability
        if arm == "json_native"
        else resident.probability
    )
    serialize = numeric.serialize if arm == "python" else resident.original.serialize
    expected = dict(
        state=deepcopy(state),
        counts=deepcopy(f.counts),
        operations=deepcopy(f.operations),
        maximum=f.maximum,
    )
    observed = deepcopy(expected)
    if arm == "resident":
        f.bind(state)
        parameters, scale, counts, operations, maximum = f.resident.audit(0)
        observed = dict(
            state=dict(state, parameters=parameters, decay_scale=scale),
            counts=counts,
            operations=operations,
            maximum=maximum,
        )
    imported = dict(expected=expected, observed=observed, passed=observed == expected)
    rows = []
    for index in range(request["start"], request["stop"]):
        x = job["vectors"][index % len(job["vectors"])]["x"]
        state, receipt = step(state, x, job["labels"][index % len(job["labels"])], f)
        p, _ = probability(state, x, f)
        ack = canonical_hash(dict(previous_ack=ack, next_index=index + 1, state=state))
        rows.append(
            dict(
                update_index=index + 1,
                state=deepcopy(state),
                counts=deepcopy(f.counts),
                operations=deepcopy(f.operations),
                maximum=f.maximum,
                probability_before=receipt["probability"],
                probability=p,
                ids=receipt["ids"],
                weights=receipt["weights"],
                touched_coefficients=[state["parameters"][i] for i in receipt["ids"]],
                local_writes=receipt["local_writes"],
                dense_writes=receipt["renormalization_writes"],
                serialized_hash=canonical_hash(serialize(state, f).hex()),
                ack=ack,
            )
        )
        if (index + 1) % 16 == 0:
            progress("qualified_updates_" + arm, index + 1, request["stop"] - index - 1)
    result = dict(
        import_check=imported,
        rows=rows,
        checkpoint=dict(
            state=state,
            counts=f.counts,
            operations=f.operations,
            maximum=f.maximum,
            ack=ack,
            next_index=request["stop"],
        ),
    )
    atomic_json(output, result)
    return result


def roster() -> list[Json]:
    """Keep missing branches visible without counting repeated representations as cases."""
    return [
        dict(
            unit_id=f"ordinary:{offset}:{arm}",
            arm=arm,
            offset=offset,
            synthetic_boundary=False,
            scope="restart-only" if offset == 2032 else "ordinary",
        )
        for offset in METHOD["offsets"]
        for arm in ARMS
    ] + [
        dict(
            unit_id=f"threshold:{offset}:{arm}",
            arm=arm,
            threshold_offset=offset,
            synthetic_boundary=True,
            scope="constructed threshold fixture",
        )
        for offset in METHOD["threshold_offsets"]
        for arm in ARMS
    ]


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Execution acceptance and constructed arithmetic evidence cannot imply natural benefit."""
    bad = any(not r["passed"] for r in receipts)
    by_id = {r["unit_id"]: r for r in work["rows"]}
    rows = [by_id.get(r["unit_id"], dict(r, passed=False, status="censored")) for r in roster()]
    ordinary = (
        not work["failures"]
        and len(work["natural_event_rows"]) == 4096
        and all(r["passed"] for r in rows[:9])
        and len(work["restart_rows"]) == 3
        and all(r["passed"] for r in work["restart_rows"])
    )
    threshold = not work["failures"] and all(r["passed"] for r in rows[9:])
    verdict = (
        "disqualified" if bad else "circular_positive" if ordinary and threshold else "blocked"
    )
    raw = Path(work["raw"])
    value: Json = dict(
        experiment_id=8758,
        task_id=TASK,
        milestone="2026.11.750",
        run_date=work["run_date"],
        invocation_id=raw.name,
        honest_verdict="complete_" + verdict + "_lifecycle_event_qualification",
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0, generation_calls=0),
        numerical_training=dict(
            performed=True,
            scope="fixed-point replay of exposed historical labels; no learning benefit evaluated",
        ),
        rows=rows,
        intended_count=24,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=int(bool(work["natural_event_rows"])),
        verifier_is_oracle=True,
        exposure_scope="one retained exposed trajectory and separate constructed threshold fixtures",
        independent_generalization_score=0,
        ordinary_window_ready_score=int(ordinary and not bad),
        threshold_fixture_ready_score=int(threshold and not bad),
        natural_event_rows=work["natural_event_rows"],
        scalar_decay_rows=work["scalar_decay_rows"],
        ordinary_starting_state_rows=work["ordinary_starting_state_rows"],
        restart_rows=work["restart_rows"],
        threshold_control_rows=work["threshold_control_rows"],
        natural_renormalization_observed=any(r["dense_writes"] for r in work["natural_event_rows"]),
        natural_dense_event_count=sum(bool(r["dense_writes"]) for r in work["natural_event_rows"]),
        natural_update_denominator=len(work["natural_event_rows"]),
        synthetic_event_frequency_estimate=None,
        cost_protocol_reference=reference(raw / "cost_protocol.json"),
        acceptance_gates=dict(
            owned_validation=not bad,
            ordinary_restart_qualification=bool(ordinary),
            constructed_threshold_qualification=bool(threshold),
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
        random_seed=7508758,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["raw_refs"]
        + [shard["reference"] for r in receipts for shard in r.get("shards", [])],
        cited_upstream_artifacts=work["source_refs"],
        repository_health_reference=work["repository_health_reference"],
        scoped_validation_scope=dict(
            owned=OWNED,
            tests=TESTS,
            full_suite_pass_claimed=False,
            historical_producers_requalified=False,
        ),
        primitive_reference=reference(raw / "measurement.json"),
        validation_reference=reference(raw / "validation.json"),
        publication_output=str(output),
        private_control=work.get("private_control", False),
        methodology_note="Every original update is replayed without timing. Exact imported ordinary32-update "
        "windows retain restart after2048. Dense writes are measured; constructed threshold fixtures "
        "are separate and cannot estimate natural event frequency or complete durable costs. "
        "No model, performance benefit, hidden generalization or physical hardware claim.",
    )
    value["reproducibility_checksum"] = canonical_hash(
        dict(method=METHOD, code=work["code_refs"], sources=work["source_refs"])
    )
    value["field_principles"] = {
        k: "Bind exact events and parity to one invocation; keep ordinary, restart and synthetic evidence separate."
        for k in value
    }
    return value
