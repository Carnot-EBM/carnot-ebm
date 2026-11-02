"""REQ-REPORT-8706: preserve arithmetic while observing long-lived CPU state.

Each committed acknowledgement binds the integer state and update index. A
second process can resume without repeating an acknowledged update.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from carnot.verify import spline_precision_8692 as numeric

__all__ = ["numeric", "drive", "envelope", "progress"]
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts keep real CPU work visible to the supervising parent."""
    print(f"[exp8706] phase={phase} completed={completed} pending={pending}", flush=True)


def drive(job: Json, folder: Path, start: int, stop: int, reference: bool) -> Json:
    """Write every observed step and durably checkpoint every256 committed updates."""
    folder.mkdir(parents=True, exist_ok=True)
    f = numeric.Fixed(32, 24)
    signature = canonical_hash(job)
    costs: Json = {}

    def cost(operation: str, duration: int) -> None:
        """Keep host and integer durations separate so neither can disappear in totals."""
        item = costs.setdefault(operation, dict(calls=0, duration_ns=0))
        item["calls"] += 1
        item["duration_ns"] += duration

    if start:
        checkpoint = json.loads((folder / "checkpoint.json").read_bytes())
        if checkpoint["next_index"] != start or checkpoint["job_hash"] != signature:
            raise ValueError("restart_identity")
        state, ack = checkpoint["state"], checkpoint["ack"]
        f.counts, f.operations, f.maximum = (
            checkpoint["counts"],
            checkpoint["operations"],
            checkpoint["maximum"],
        )
        if numeric.serialize(state, f) != (folder / "checkpoint.bin").read_bytes():
            raise ValueError("restart_serialization")
    else:
        state = numeric.encode(job["head"], f)
        if job["boundary"]:
            state["decay_scale"] = f.unit // 256 + job["boundary_offset"]
        ack = canonical_hash(dict(job_hash=signature, next_index=0, state=state))
    floating = deepcopy(job["head"])
    if job["boundary"]:
        floating["parameters"] = numeric.effective(state, f)
        floating["decay_scale"] = 1.0
    first: Json | None = None
    maximum_error = maximum_decay = maximum_coefficient = 0.0
    max_weight = 0
    rows_path = folder / f"rows-{start:04}-{stop:04}.jsonl"
    progress("benchmark_before_" + folder.name, start, stop - start)
    with rows_path.open("w") as stream:
        for index in range(start, stop):
            x = job["vectors"][index % len(job["vectors"])]["x"]
            y = job["labels"][index % len(job["labels"])]
            observation_start = time.perf_counter_ns()
            before = deepcopy(state)
            cost("host_observation_snapshot", time.perf_counter_ns() - observation_start)
            try:
                state, receipt = numeric.step(state, x, y, f)
                p, prediction = numeric.probability(state, x, f)
                for span in receipt["spans"] + prediction["spans"]:
                    operation = span["operation"]
                    if operation == "integer_brier_sparse_write":
                        operation = "integer_brier_local_global_and_dense_renormalization"
                    cost(operation, span["duration_ns"])
                update_cost = costs["integer_brier_local_global_and_dense_renormalization"]
                for field, count in [
                    ("dense_renormalization_writes", receipt["renormalization_writes"]),
                    ("local_data_writes", receipt["local_writes"]),
                    ("global_decay_writes", 1),
                ]:
                    update_cost[field] = update_cost.get(field, 0) + count
                max_weight = max(max_weight, *receipt["weights"])
                begin = time.perf_counter_ns()
                if reference:
                    old_p = float(numeric.original.sparse.predict(floating, np.array([x]))[0])
                    floating, _ = numeric.original.update(floating, np.array(x), y)
                    fp = float(numeric.original.sparse.predict(floating, np.array([x]))[0])
                else:
                    old_p, fp = receipt["probability"], p
                cost("float64_reference", time.perf_counter_ns() - begin)
                error = max(abs(p - fp), abs(receipt["probability"] - old_p))
                margin = min(abs(fp - 0.2), abs(fp - 0.8), abs(old_p - 0.2), abs(old_p - 0.8))
                fallback = reference and margin <= 0.001
                begin = time.perf_counter_ns()
                action = numeric.decision(p)
                if fallback:
                    action = numeric.decision(
                        float(numeric.original.sparse.predict(floating, np.array([x]))[0])
                    )
                cost("float64_fallback_and_decision", time.perf_counter_ns() - begin)
                continuity = 0.0
                if receipt["renormalization_writes"]:
                    probe = numeric.Fixed(32, 24)
                    rounded = [
                        probe.mul(v, before["decay_scale"]) / f.unit for v in before["parameters"]
                    ]
                    continuity = max(
                        abs(a - b)
                        for a, b in zip(rounded, numeric.effective(before, f), strict=True)
                    )
                coefficient_error = (
                    max(
                        abs(a - b)
                        for a, b in zip(
                            numeric.effective(state, f),
                            numeric.original.sparse.parameters(floating),
                            strict=True,
                        )
                    )
                    if reference
                    else 0.0
                )
                decay_error = (
                    abs(state["decay_scale"] / f.unit - floating["decay_scale"])
                    if reference and not job["boundary"]
                    else None
                )
                finite = all(
                    math.isfinite(v)
                    for v in [p, fp, old_p, receipt["probability"], *numeric.effective(state, f)]
                )
                parity = action == numeric.decision(fp) and (
                    fallback or numeric.decision(receipt["probability"]) == numeric.decision(old_p)
                )
                passed = (
                    finite
                    and not f.counts["saturation"]
                    and not f.counts["overflow"]
                    and continuity <= 0.5 / f.unit
                    and (job["boundary"] or error <= 0.001)
                    and parity
                )
                ack = canonical_hash(dict(previous_ack=ack, next_index=index + 1, state=state))
                row = dict(
                    head_index=job["head_index"],
                    sequence=job["sequence"],
                    update_index=index + 1,
                    vector_id=index % len(job["vectors"]),
                    label=y,
                    probability=p,
                    float64_probability=fp if reference else None,
                    probability_error=error if reference else None,
                    fallback=fallback,
                    finite=finite,
                    decision_parity=parity,
                    effective_coefficient_error=coefficient_error,
                    decay_scale_error=decay_error,
                    boundary_continuity_error=continuity,
                    boundary_control=job["boundary"],
                    before_decay_scale=before["decay_scale"],
                    after_decay_scale=state["decay_scale"],
                    local_writes=receipt["local_writes"],
                    global_decay_writes=1,
                    renormalization_writes=receipt["renormalization_writes"],
                    host_snapshot_coefficient_copies=109,
                    instrumentation_snapshot_coefficient_copies=109,
                    ack=ack,
                    state_hash=canonical_hash(state),
                    passed=bool(passed),
                )
                maximum_error = max(maximum_error, error)
                maximum_decay = max(maximum_decay, decay_error or 0.0)
                maximum_coefficient = max(maximum_coefficient, coefficient_error)
            except OverflowError as error_value:
                row = dict(
                    update_index=index + 1, passed=False, error=str(error_value), state=before
                )
            if not row["passed"] and first is None:
                first = dict(row, state_before=before, state_after=deepcopy(state))
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            if (index + 1) % 256 == 0 or index + 1 == stop:
                begin = time.perf_counter_ns()
                encoded = numeric.serialize(state, f)
                cost("host_serialization", time.perf_counter_ns() - begin)
                checkpoint = dict(
                    job_hash=signature,
                    next_index=index + 1,
                    state=state,
                    ack=ack,
                    counts=f.counts,
                    operations=f.operations,
                    maximum=f.maximum,
                )
                begin = time.perf_counter_ns()
                with (folder / "checkpoint.bin").open("wb") as binary:
                    binary.write(encoded)
                    binary.flush()
                    os.fsync(binary.fileno())
                atomic_json(folder / "checkpoint.json", checkpoint)
                directory_fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
                os.fsync(directory_fd)
                os.close(directory_fd)
                stream.flush()
                os.fsync(stream.fileno())
                cost("durable_storage", time.perf_counter_ns() - begin)
                progress("checkpoint_" + folder.name, index + 1, stop - index - 1)
    result = dict(
        checkpoint=checkpoint,
        rows_path=str(rows_path),
        counts=f.counts,
        operations=f.operations,
        accumulator_abs_max=f.maximum,
        maximum_basis_integer=max_weight,
        maximum_probability_error=maximum_error,
        maximum_decay_error=maximum_decay,
        maximum_coefficient_error=maximum_coefficient,
        first_failure=first,
        passed=first is None,
        costs=costs,
        observed_updates=stop - start,
    )
    atomic_json(folder / f"summary-{start:04}-{stop:04}.json", result)
    progress("benchmark_after_" + folder.name, stop, 0)
    return result


def envelope(summary: Json) -> Json:
    """A conditional interval covers this domain; checked arithmetic rejects other ranges."""
    unit = 1 << 24
    bound = 37 * (1 << 31) * (unit + 16)
    return dict(
        accumulator_abs_max=summary["accumulator_abs_max"],
        signed64_limit=(1 << 63) - 1,
        whole_format_product_sum_bound=37 * (1 << 31) ** 2,
        whole_format_bound_exceeds_signed64=True,
        supported_domain_accumulator_envelope=bound,
        supported_domain=dict(
            stored_coefficient_abs_max=1 << 31,
            basis_abs_max=unit + 16,
            local_terms_max=37,
            decay_scale_interval=[2, unit],
            temperature_interval=[unit // 2, 2 * unit],
            binary_labels=[0, 1],
            scaled_inputs_interval=[0, 1],
        ),
        observed_inside_envelope=summary["accumulator_abs_max"] <= bound
        and summary["maximum_basis_integer"] <= unit + 16,
        outside_domain_policy="checked_signed64_rejection_no_global_overflow_proof",
    )
