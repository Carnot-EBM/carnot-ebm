"""REQ-VERIFY-8407: measure the whole local service while keeping labels delayed."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.direct_state_qualification_8404 import QualifiedStore
from carnot.reporting.historical_consumer_roots_8402 import cost_tape
from carnot.verify import direct_atomic_state_8376 as s
from carnot.verify.independent_online_memory_8116 import LabelVault

Json = dict[str, Any]
SEED = 7248407
CELLS = [
    dict(workload=w, batch=b, predictions_per_update=r)
    for w in ("ordinary", "boundary", "natural")
    for b in (1, 8, 32)
    for r in (1, 8, 64)
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush work counts so long timing and replay phases remain observable."""
    print(f"[exp8407] phase={phase} completed={completed} pending={pending}", flush=True)


def source(bundle: Json, cell: Json, tick: int) -> Json:
    """Reuse cached natural identities; other workloads are constructed numeric controls."""
    slot = (tick - 1) % len(bundle["slots"]) + 1
    row = deepcopy(bundle["slots"][slot - 1])
    if cell["workload"] != "natural":
        v = [0.0, 0.2, 0.3, 0.4, 0.5]
        if cell["workload"] == "boundary":
            v[1:] = [float(np.nextafter(a, 1.0 if tick % 2 else 0.0)) for a in (0.0, 0.2, 0.8, 1.0)]
        row.update(x=v, unit_id=f"{cell['workload']}-{slot}", source_cluster_id=None)
    return dict(row)


def release_target(bundle: Json, row: Json, issued: int, clock: int, sealed: bool) -> int | None:
    """The event clock and durable issue must qualify before any target is decoded."""
    if clock != issued + 8 or not sealed:
        raise ValueError("eight_tick_causality")
    if row["x"] is None:
        return None
    if row["source_cluster_id"] is None:
        return int(row["slot"] % 2)
    vault = LabelVault(Path(bundle["labels"]["path"]), bundle["slots"])
    return vault.release(row["slot"], row["slot"] + 8, sealed=True)["y"]  # type: ignore[no-any-return]


def step(
    bundle: Json,
    cell: Json,
    state: Json,
    event: Json,
    identity: str,
    mark: Callable[[str], None] = lambda name: None,
) -> Json:
    """Adapt designated scheduling while preserving the shipped scoring and update arithmetic."""
    before = canonical_hash(state)
    row: Json = dict(event, request_id=identity, changed=[], probability=None, y=None)
    if event["kind"] == "prediction":
        item = source(bundle, cell, event["tick"])
        row.update(source=item, input_hash=canonical_hash(item))
        x: Any = None if item["x"] is None else np.asarray(item["x"], dtype=np.float64)
        mark("python_boundary_conversion")
        try:
            s.validate(state["head"], x)
            p = None if x is None else float(s.evaluator.direct(state["head"], x[None])[0])
            row.update(
                status="completed",
                probability=p,
                action="escalate" if p is None else s.evaluator.kernel.action(p),
            )
        except ValueError as error:
            row.update(status="failed", action=None, missing_reason=str(error))
        mark("direct_scoring")
        state["issued"][str(event["tick"])] = deepcopy(row)
        if event["designated"]:
            state["pending"].append(event["tick"])
    else:
        issue = state["issued"].get(str(event["issued_tick"]))
        if (
            issue is None
            or event["issued_tick"] not in state["pending"]
            or event["tick"] != event["issued_tick"] + 8
        ):
            raise ValueError("feedback_order")
        item = issue["source"]
        y = (
            None
            if issue["status"] != "completed"
            else release_target(bundle, item, event["issued_tick"], event["tick"], True)
        )
        mark("delayed_release")
        change = s.optimizer.learn(
            state["head"], None if issue["status"] != "completed" else item["x"], y, "online_sparse"
        )
        mark("coefficient_update")
        state["head"]["coefficients"] = change["coefficients"]
        row.update(
            status="nonapplicable" if y is None else "completed",
            input_hash=canonical_hash(dict(issue=issue["input_hash"], y=y)),
            y=y,
            changed=change["changed"],
            missing_reason=change["reason"],
        )
        state["pending"].remove(event["issued_tick"])
        state["applied"][identity] = deepcopy(row)
        state["release_cursor"] += 1
    state["version"] += 1
    row.update(before_state_hash=before, after_state_hash=canonical_hash(state))
    mark("scheduling")
    return row


def read_snapshot(directory: Path) -> Json:
    """Pin a whole immutable version and verify exact bytes before evaluating a request."""
    pointer = json.loads((directory / "current.json").read_bytes())
    name = pointer["file"]
    if Path(name).name != name:
        raise ValueError("snapshot_path")
    envelope = json.loads((directory / name).read_bytes())
    state: Json = envelope["state"]
    if canonical_hash(state) != envelope["sha256"] or envelope["sha256"] != pointer["sha256"]:
        raise ValueError("snapshot_checksum")
    return state


def batches(tape: list[Json], size: int) -> list[list[Json]]:
    """A feedback boundary flushes a remainder so no later prediction moves before a release."""
    result: list[list[Json]] = []
    pending: list[Json] = []
    for event in tape:
        if event["kind"] == "feedback":
            if pending:
                result.append(pending)
                pending = []
            result.append([event])
        else:
            pending.append(event)
            if len(pending) == size:
                result.append(pending)
                pending = []
    if pending:
        result.append(pending)
    return result


def run_repeat(bundle: Json, cell: Json, repeat: int, directory: Path) -> Json:
    """Include scheduling, request conversion and durable tail writes in one wall interval."""
    if cell not in CELLS or repeat not in range(6):
        raise ValueError("fixed_work_units")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    store = QualifiedStore(directory / "store", dict(head=bundle["head"], events=[]))
    state = s.initial(bundle["head"])
    transactions: list[Json] = []
    rows: list[Json] = []
    start = time.monotonic_ns()
    store.commit(state)
    initialized = time.monotonic_ns()
    initial_pointer = directory / "store/current.json"
    initial_snapshot = directory / "store" / json.loads(initial_pointer.read_bytes())["file"]
    initialization_bytes = initial_snapshot.stat().st_size + initial_pointer.stat().st_size
    tape = cost_tape(cell["predictions_per_update"])
    groups = batches(tape, cell["batch"])
    tail = None
    prefix = f"{cell['workload']}-b{cell['batch']}-r{cell['predictions_per_update']}-rep{repeat}"
    for number, group in enumerate(groups):
        if time.monotonic_ns() - start > 600_000_000_000:
            raise TimeoutError("repeat_deadline")
        if tail is None and group[0]["tick"] > 8 * cell["predictions_per_update"]:
            tail = time.monotonic_ns()
        began = time.monotonic_ns()
        stages: list[Json] = []
        last = began

        def mark(name: str) -> None:
            nonlocal last
            now = time.monotonic_ns()
            stages.append(dict(name=name, start_ns=last, end_ns=now))
            last = now

        converted = json.loads(json.dumps(group))
        mark("input_validation")
        state = read_snapshot(directory / "store")
        mark("snapshot_pinning")
        produced: list[Json] = []
        for event in converted:
            produced.append(
                step(bundle, cell, state, event, f"{prefix}-q{len(rows) + len(produced) + 1}", mark)
            )
        store.commit(state, lambda name: mark(name))
        pointer = json.loads((directory / "store/current.json").read_bytes())
        snapshot = directory / "store" / pointer["file"]
        payload = json.dumps(produced, sort_keys=True).encode()
        mark("response")
        end = time.monotonic_ns()
        tx: Json = dict(
            transaction_id=f"{prefix}-tx{number}",
            start_ns=began,
            end_ns=end,
            stages=stages,
            request_ids=[r["request_id"] for r in produced],
            snapshot_path=str(snapshot),
            snapshot_sha256="",
            state_hash=canonical_hash(state),
            bytes=snapshot.stat().st_size
            + (directory / "store/current.json").stat().st_size
            + len(payload),
            fsync_count=4,
        )
        tx["snapshot_sha256"] = sha256_file(snapshot)
        transactions.append(tx)
        for row in produced:
            row.update(transaction_id=tx["transaction_id"], start_ns=began, end_ns=end)
        rows.extend(produced)
        if number % 64 == 0:
            progress("benchmark_transactions", number + 1, len(groups) - number - 1)
    end = time.monotonic_ns()
    return dict(
        cell=cell,
        repeat=repeat,
        repeat_id=prefix,
        initial_state=s.initial(bundle["head"]),
        start_ns=start,
        initialized_ns=initialized,
        tail_start_ns=tail,
        end_ns=end,
        rows=rows,
        transactions=transactions,
        final_state=state,
        initialization_fsync_count=4,
        initialization_bytes=initialization_bytes,
    )


def reduce_repeat(bundle: Json, result: Json) -> Json:
    """Replay frozen work independently and check actual snapshot bytes and nested clocks."""
    cell, repeat = result["cell"], result["repeat"]
    if cell not in CELLS or repeat not in range(6):
        raise ValueError("repeat_configuration")
    start, initialized, tail, end = [
        result[n] for n in ("start_ns", "initialized_ns", "tail_start_ns", "end_ns")
    ]
    if (
        not all(type(v) is int and v > 0 for v in (start, initialized, tail, end))
        or start <= 1_000_000_000
        or not start < initialized <= tail < end
    ):
        raise ValueError("repeat_clocks")
    prefix = f"{cell['workload']}-b{cell['batch']}-r{cell['predictions_per_update']}-rep{repeat}"
    if result["repeat_id"] != prefix:
        raise ValueError("fixed_repeat_identity")
    state = s.initial(bundle["head"])
    if result["initial_state"] != state:
        raise ValueError("initial_state")
    tape = cost_tape(cell["predictions_per_update"])
    if len(tape) != len(result["rows"]):
        raise ValueError("lost_work")
    identities: set[str] = set()
    for event, row in zip(tape, result["rows"], strict=True):
        identity = f"{result['repeat_id']}-q{len(identities) + 1}"
        if row["request_id"] != identity or identity in identities:
            raise ValueError("request_identity")
        identities.add(identity)
        expected = step(bundle, cell, state, event, identity)
        if expected != {key: row[key] for key in expected}:
            raise ValueError("primitive_semantics")
    previous = initialized
    total_bytes, fsyncs = result["initialization_bytes"], result["initialization_fsync_count"]
    costs: dict[str, int] = {}
    flattened = []
    by_id = {r["request_id"]: r for r in result["rows"]}
    expected_groups = batches(tape, cell["batch"])
    if len(expected_groups) != len(result["transactions"]):
        raise ValueError("fixed_batch_count")
    for index, tx in enumerate(result["transactions"]):
        if (
            tx["transaction_id"] != f"{result['repeat_id']}-tx{index}"
            or not previous <= tx["start_ns"] < tx["end_ns"] <= end
        ):
            raise ValueError("transaction_clocks")
        clock = tx["start_ns"]
        for stage in tx["stages"]:
            if not clock == stage["start_ns"] <= stage["end_ns"] <= tx["end_ns"]:
                raise ValueError("stage_nesting")
            clock = stage["end_ns"]
            costs[stage["name"]] = costs.get(stage["name"], 0) + stage["end_ns"] - stage["start_ns"]
        if not {
            "temporary_write",
            "file_fsync",
            "pointer_publication",
            "directory_fsync",
            "acknowledgment",
            "response",
        } <= {x["name"] for x in tx["stages"]}:
            raise ValueError("durable_stages")
        actual_rows = [by_id[n] for n in tx["request_ids"]]
        group = expected_groups[index]
        if [
            dict((key, r[key]) for key in event)
            for r, event in zip(actual_rows, group, strict=True)
        ] != group:
            raise ValueError("fixed_batch_order")
        if any(
            r["start_ns"] != tx["start_ns"]
            or r["end_ns"] != tx["end_ns"]
            or r["transaction_id"] != tx["transaction_id"]
            for r in actual_rows
        ):
            raise ValueError("request_clock_binding")
        path = Path(tx["snapshot_path"])
        envelope = json.loads(path.read_bytes())
        if (
            sha256_file(path) != tx["snapshot_sha256"]
            or canonical_hash(envelope["state"]) != tx["state_hash"]
            or tx["fsync_count"] != 4
            or envelope["sha256"] != tx["state_hash"]
            or tx["state_hash"] != actual_rows[-1]["after_state_hash"]
            or tx["bytes"]
            != path.stat().st_size
            + len(
                json.dumps(
                    dict(file=path.name, sha256=tx["state_hash"]), indent=2, sort_keys=True
                ).encode()
            )
            + 1
            + len(
                json.dumps(
                    [
                        {
                            key: value
                            for key, value in r.items()
                            if key not in ("transaction_id", "start_ns", "end_ns")
                        }
                        for r in actual_rows
                    ],
                    sort_keys=True,
                ).encode()
            )
        ):
            raise ValueError("durable_snapshot")
        flattened.extend(tx["request_ids"])
        previous = tx["end_ns"]
        total_bytes += tx["bytes"]
        fsyncs += tx["fsync_count"]
    if (
        flattened != [r["request_id"] for r in result["rows"]]
        or state != result["final_state"]
        or state["pending"]
    ):
        raise ValueError("tail_conservation")
    predictions = [r for r in result["rows"] if r["kind"] == "prediction"]
    feedback = [r for r in result["rows"] if r["kind"] == "feedback"]
    latencies = [(r["end_ns"] - r["start_ns"]) / 1e9 for r in predictions]
    return dict(
        cell=cell,
        repeat=repeat,
        repeat_id=result["repeat_id"],
        status="completed",
        prediction_attempts=len(predictions),
        completed_decisions=sum(r["status"] == "completed" for r in predictions),
        failed_count=sum(r["status"] == "failed" for r in predictions),
        feedback_attempts=len(feedback),
        released=len(feedback),
        applied=sum(bool(r["changed"]) for r in feedback),
        missing=sum(r["status"] == "nonapplicable" for r in feedback),
        pending=0,
        lost=0,
        wall_seconds=(end - start) / 1e9,
        tail_drain_seconds=(end - tail) / 1e9,
        p50_seconds=float(np.percentile(latencies, 50)),
        p95_seconds=float(np.percentile(latencies, 95)),
        stage_ns=costs,
        serialized_bytes=total_bytes,
        fsync_count=fsyncs,
    )


def block_interval(repeats: list[Json]) -> Json:
    """Resample five whole repeat blocks; requests never become independent experiments."""
    if len(repeats) != 5 or {r["repeat"] for r in repeats} != {1, 2, 3, 4, 5}:
        raise ValueError("five_distinct_repeats")
    rng = np.random.default_rng(SEED)
    choices = rng.integers(0, 5, size=(10000, 5))
    counts = np.array([r["completed_decisions"] for r in repeats])
    seconds = np.array([r["wall_seconds"] for r in repeats])
    values = counts[choices].sum(axis=1) / seconds[choices].sum(axis=1)
    return dict(
        low=float(np.percentile(values, 2.5)),
        high=float(np.percentile(values, 97.5)),
        estimate=float(counts.sum() / seconds.sum()),
        resamples=10000,
        seed=SEED,
    )
