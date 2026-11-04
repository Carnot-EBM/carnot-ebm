"""REQ-VERIFY-8732: match numerical work and durable barriers across all arms."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import resident_sparse_8731 as resident

__all__ = ["resident"]

Json = dict[str, Any]
numeric = resident.numeric
store_module = numeric.original.legacy


def drive(job: Json, folder: Path, arm: str, start: int, stop: int, extension: Any) -> Json:
    """Use the shipped store so resident arithmetic cannot omit persistence cost."""
    folder.mkdir(parents=True, exist_ok=True)
    configuration = dict(capacity=8, job_hash=canonical_hash(job))
    store = store_module.Store(folder / "store", configuration)
    f: Any = (
        numeric.Fixed(32, 24)
        if arm == "python"
        else resident.original.NativeFixed(extension)
        if arm == "json_native"
        else resident.ResidentFixed(extension)
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
    if start:
        saved = store.read()
        if saved["cursor"] != start or saved["configuration_hash"] != canonical_hash(configuration):
            raise ValueError("restart_identity")
        state = saved["head"]
        f.counts, f.operations, f.maximum = saved["counts"], saved["operations"], saved["maximum"]
    else:
        state = numeric.encode(job["head"], f)
        if job["boundary"]:
            state["decay_scale"] = f.unit // 256 + job["boundary_offset"]
    latencies, rows = [], []
    kernel_ns = writes = 0
    resident.progress("cost_benchmark_before_" + arm, start, stop - start)
    for index in range(start, stop):
        begin = time.perf_counter_ns()
        raw = job["vectors"][index % len(job["vectors"])]["x"]
        before, _ = probability(state, raw, f)
        common = dict(
            schema=store_module.SCHEMA,
            configuration_hash=canonical_hash(configuration),
            pending=[],
            cursor=index,
            head=state,
            counts=f.counts,
            operations=f.operations,
            maximum=f.maximum,
        )
        store.commit(dict(common, phase="sealed", prediction=before, version=2 * index))
        label = job["labels"][index % len(job["labels"])]
        arithmetic_start = time.perf_counter_ns()
        state, receipt = step(state, raw, label, f)
        kernel_ns += time.perf_counter_ns() - arithmetic_start
        after, _ = probability(state, raw, f)
        packed = serialize(state, f)
        semantic = dict(
            update_index=index + 1,
            probability_before=before,
            probability_after=after,
            decision_before=numeric.decision(before),
            decision_after=numeric.decision(after),
            state_hash=canonical_hash(state),
            serialized_hash=canonical_hash(packed.hex()),
            renormalization_writes=receipt["renormalization_writes"],
        )
        checkpoint = dict(
            common,
            head=state,
            cursor=index + 1,
            version=2 * index + 1,
            phase="acknowledged",
            counts=f.counts,
            operations=f.operations,
            maximum=f.maximum,
            serialized_hex=packed.hex(),
            acknowledgment=canonical_hash(semantic),
        )
        store.commit(checkpoint)
        elapsed = time.perf_counter_ns() - begin
        latencies.append(elapsed)
        writes += receipt["renormalization_writes"]
        rows.append(dict(semantic, latency_ns=elapsed, status="completed", passed=True))
        if (index + 1) % 256 == 0 or index + 1 == stop:
            resident.progress("cost_committed_" + arm, index + 1, stop - index - 1)
    rows_path = folder / f"rows-{start}-{stop}.json"
    atomic_json(rows_path, dict(rows=rows))
    result = dict(
        checkpoint=checkpoint,
        latencies_ns=latencies,
        kernel_ns=kernel_ns,
        renormalization_writes=writes,
        durable_commits=2 * (stop - start),
        rows_path=str(rows_path),
        updates=stop - start,
    )
    atomic_json(folder / f"summary-{start}-{stop}.json", result)
    resident.progress("cost_benchmark_after_" + arm, stop, 0)
    return result
