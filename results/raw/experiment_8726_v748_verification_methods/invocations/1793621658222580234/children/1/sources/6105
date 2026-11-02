"""REQ-VERIFY-8466: compare unchanged stores while charging their whole durable work.

Fresh processes clear numerical prefix caches. Cold reconstruction is part of
the complete wall clock and also reported separately for deployment planning.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import journal_equivalence_8465 as stores

Json = dict[str, Any]
CELLS = [dict(workload="natural", batch=b, predictions_per_update=8) for b in (1, 8)]
SEED = 7278451


def progress(phase: str, completed: int = 0, pending: int = 0) -> int:
    """Return printed bytes so process write accounting can subtract progress output."""
    line = f"[exp8466] phase={phase} completed={completed} pending={pending}"
    print(line, flush=True)
    return len((line + "\n").encode())


def io_counts() -> Json:
    """Linux process counters include actual writes, including overwritten pointer bytes."""
    return {
        k: int(v)
        for k, v in (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())
    }


def run_repeat(frozen: Json, directory: Path, arm: str) -> Json:
    """Run one restored arm, retaining every acknowledgment before measuring restart."""
    if arm not in ("snapshot", "journal") or frozen["cell"] not in CELLS:
        raise ValueError("registered_arm_and_cell")
    stores._prefix.cache_clear()
    printed, fsyncs = 0, 0
    sync = os.fsync

    def counted(fd: int) -> None:
        nonlocal fsyncs
        sync(fd)
        fsyncs += 1

    progress("benchmark_before_" + arm)
    io_before = io_counts()
    start = time.monotonic_ns()
    expected = stores.oracle(frozen, len(frozen["transactions"]))
    cold_end = time.monotonic_ns()
    with patch.object(os, "fsync", counted):
        store = (stores.Journal if arm == "journal" else stores.Snapshot)(
            directory / "store", frozen
        )
        store.initialize()
        initialized = time.monotonic_ns()
        requests, acknowledgments = [], []
        for index, tx in enumerate(frozen["transactions"]):
            before = time.monotonic_ns()
            reply = store.apply(tx)
            after = time.monotonic_ns()
            acknowledgments.append(dict(id=tx["id"], response=reply))
            requests.append(
                dict(id=tx["id"], request_count=len(tx["events"]), start_ns=before, end_ns=after)
            )
            if index % 16 == 0:
                printed += progress(
                    "transactions_" + arm, index + 1, len(frozen["transactions"]) - index - 1
                )
        if isinstance(store, stores.Journal):
            store.drain()
        final = store.read()
    end = time.monotonic_ns()
    io_after = io_counts()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if stores.encode(final) != stores.encode(expected):
        raise ValueError("exact_final_state")
    disk = sum(p.stat().st_size for p in store.path.iterdir() if p.is_file())
    stores._prefix.cache_clear()
    progress("restart_before_" + arm)
    restart_start = time.monotonic_ns()
    restarted = store.read()
    restart_end = time.monotonic_ns()
    progress("restart_after_" + arm)
    if stores.encode(restarted) != stores.encode(expected):
        raise ValueError("restart_state")
    progress("benchmark_after_" + arm, len(requests), 0)
    return dict(
        arm=arm,
        cell=frozen["cell"],
        directory=str(directory),
        start_ns=start,
        cold_end_ns=cold_end,
        initialized_ns=initialized,
        end_ns=end,
        restart_start_ns=restart_start,
        restart_end_ns=restart_end,
        requests=requests,
        acknowledgments=acknowledgments,
        final=final,
        restarted=restarted,
        bytes=io_after["wchar"] - io_before["wchar"] - printed,
        kernel_write_bytes=io_after["write_bytes"] - io_before["write_bytes"],
        fsyncs=fsyncs,
        peak_memory_bytes=peak_rss,
        peak_disk_bytes=disk,
    )


def reduce(frozen: Json, row: Json) -> Json:
    """Rebuild every prediction and feedback response from independently sealed operands."""
    if row["arm"] not in ("snapshot", "journal") or stores.encode(row["cell"]) != stores.encode(
        frozen["cell"]
    ):
        raise ValueError("primitive_binding")
    expected = stores.oracle(frozen, len(frozen["transactions"]))
    acknowledgments = [
        dict(id=tx["id"], response=reply)
        for tx, reply in zip(frozen["transactions"], expected["replies"], strict=True)
    ]
    if (
        stores.encode(row["final"]) != stores.encode(expected)
        or stores.encode(row["restarted"]) != stores.encode(expected)
        or stores.encode(row["acknowledgments"]) != stores.encode(acknowledgments)
    ):
        raise ValueError("logical_work")
    n = len(acknowledgments)
    expected_fsyncs = 4 * (n + 1) if row["arm"] == "snapshot" else 9 + 3 * n + 6 * (n // 32 + 1)
    if row["fsyncs"] != expected_fsyncs or row["bytes"] <= 0:
        raise ValueError("durable_io")
    if (
        not row["start_ns"]
        < row["cold_end_ns"]
        < row["initialized_ns"]
        < row["end_ns"]
        < row["restart_start_ns"]
        < row["restart_end_ns"]
    ):
        raise ValueError("complete_clock")
    clock = row["initialized_ns"]
    latencies = []
    for tx, request in zip(frozen["transactions"], row["requests"], strict=True):
        if (
            request["id"] != tx["id"]
            or request["request_count"] != len(tx["events"])
            or not clock <= request["start_ns"] < request["end_ns"] <= row["end_ns"]
        ):
            raise ValueError("request_clock")
        clock = request["end_ns"]
        latencies.extend([(request["end_ns"] - request["start_ns"]) / 1e9] * len(tx["events"]))
    replies = [r for group in expected["replies"] for r in group]
    decisions = [r for r in replies if r["kind"] == "prediction"]
    feedback = [r for r in replies if r["kind"] != "prediction"]
    cold = (row["cold_end_ns"] - row["start_ns"]) / 1e9
    return dict(
        arm=row["arm"],
        wall_s=(row["end_ns"] - row["start_ns"]) / 1e9,
        cold_reconstruction_s=cold,
        cold_amortized_per_decision_s=cold / len(decisions),
        startup_s=(row["initialized_ns"] - row["cold_end_ns"]) / 1e9,
        restart_replay_s=(row["restart_end_ns"] - row["restart_start_ns"]) / 1e9,
        tail_checkpoint_s=(row["end_ns"] - clock) / 1e9,
        decision_attempts=len(decisions),
        completed_decisions=sum(r["status"] == "completed" for r in decisions),
        feedback_attempts=len(feedback),
        applied_updates=sum(r["status"] == "completed" for r in feedback),
        pending=len(expected["state"]["pending"]),
        p50_s=float(np.quantile(latencies, 0.5)),
        p95_s=float(np.quantile(latencies, 0.95)),
        bytes=row["bytes"],
        fsyncs=row["fsyncs"],
        peak_memory_bytes=row["peak_memory_bytes"],
        peak_disk_bytes=row["peak_disk_bytes"],
        logical_hash=canonical_hash(expected),
        acknowledgment_hash=canonical_hash(acknowledgments),
    )


def pair(frozen: Json, snapshot: Json, journal: Json) -> Json:
    """Only equal completed work can contribute one paired repeat to the estimate."""
    a, b = reduce(frozen, snapshot), reduce(frozen, journal)
    if (
        snapshot["arm"] != "snapshot"
        or journal["arm"] != "journal"
        or a["logical_hash"] != b["logical_hash"]
        or a["acknowledgment_hash"] != b["acknowledgment_hash"]
    ):
        raise ValueError("paired_semantics")
    return dict(
        snapshot=a,
        journal=b,
        snapshot_wall_s=a["wall_s"],
        journal_wall_s=b["wall_s"],
        p95_ratio=b["p95_s"] / a["p95_s"],
        accounting_equality=True,
    )


def interval(pairs: list[Json]) -> Json:
    """Resample whole paired repeat indices; requests cannot increase independence."""
    if len(pairs) != 5:
        raise ValueError("five_complete_repeat_blocks")
    a = np.array([p["snapshot_wall_s"] for p in pairs])
    b = np.array([p["journal_wall_s"] for p in pairs])
    indices = np.random.default_rng(SEED).integers(0, 5, (10000, 5))
    samples = a[indices].sum(axis=1) / b[indices].sum(axis=1)
    return dict(
        pooled_speedup=float(a.sum() / b.sum()),
        speedup_ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
    )
