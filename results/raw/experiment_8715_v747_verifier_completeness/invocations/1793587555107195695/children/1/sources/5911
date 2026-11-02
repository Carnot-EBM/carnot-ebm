"""REQ-VERIFY-8494: charge unchanged complete clocks to a different durable store.

The historical runner supplies cold reconstruction and request boundaries.
Both arms use identical syscall observers; their cost stays inside the clock.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import io
import os
from pathlib import Path
import stat
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import append_store_8493 as append
from carnot.verify import journal_service_cost_8466 as legacy

Json = dict[str, Any]
CELLS = legacy.CELLS
SEED = 7298480
base = append.base


def progress(phase: str, completed: int = 0, pending: int = 0) -> int:
    """Return emitted bytes so write counts exclude only progress, never service work."""
    line = f"[exp8494] phase={phase} completed={completed} pending={pending}"
    print(line, flush=True)
    return len((line + "\n").encode())


def run_repeat(frozen: Json, directory: Path, arm: str) -> Json:
    """Keep the shipped complete clock and actual operating system call counters."""
    if arm not in ("snapshot", "append") or frozen["cell"] not in CELLS:
        raise ValueError("registered_arm_and_cell")
    operations: list[Json] = []
    sync, opened, replaced, unlinked = os.fsync, io.open, os.replace, os.unlink
    low_open = os.open
    mkdir = os.mkdir
    peak_disk, printed_extra, completed = 0, 0, 0

    def counted_sync(fd: int) -> None:
        nonlocal peak_disk
        kind = "directory_fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_fsync"
        sync(fd)
        operations.append(dict(operation=kind))
        peak_disk = max(
            peak_disk,
            sum(p.stat().st_size for p in (directory / "store").rglob("*") if p.is_file()),
        )

    def counted_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        creates = (
            isinstance(file, (str, Path))
            and any(c in mode for c in "wax")
            and not Path(file).exists()
        )
        result = opened(file, mode, *args, **kwargs)
        if creates:
            operations.append(dict(operation="create", path=str(file)))
        return result

    def counted_low_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        result = low_open(path, flags, *args, **kwargs)
        if flags & os.O_CREAT:
            operations.append(dict(operation="create_open", path=str(path)))
        return result

    def counted_replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        replaced(src, dst, *args, **kwargs)
        operations.append(dict(operation="replace", path=str(dst)))

    def counted_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        unlinked(path, *args, **kwargs)
        operations.append(dict(operation="unlink", path=str(path)))

    def counted_mkdir(path: Any, *args: Any, **kwargs: Any) -> None:
        mkdir(path, *args, **kwargs)
        operations.append(dict(operation="mkdir", path=str(path)))

    def observed_apply(original: Callable[..., list[Json]]) -> Callable[..., list[Json]]:
        """Emit request counts inside the complete clock; subtract only their written bytes."""

        def apply(store: Any, tx: Json, *args: Any, **kwargs: Any) -> list[Json]:
            nonlocal printed_extra, completed
            total = len(frozen["transactions"])
            printed_extra += progress("request_before_" + tx["id"], completed, total - completed)
            result = original(store, tx, *args, **kwargs)
            completed += 1
            printed_extra += progress("request_after_" + tx["id"], completed, total - completed)
            return result

        return apply

    with (
        patch.object(legacy.stores, "Journal", append.Store),
        patch.object(legacy, "progress", progress),
        patch.object(os, "fsync", counted_sync),
        patch.object(io, "open", counted_open),
        patch.object(os, "open", counted_low_open),
        patch.object(os, "replace", counted_replace),
        patch.object(os, "unlink", counted_unlink),
        patch.object(os, "mkdir", counted_mkdir),
        patch.object(append.Store, "apply", observed_apply(append.Store.apply)),
        patch.object(base.Snapshot, "apply", observed_apply(base.Snapshot.apply)),
    ):
        row = legacy.run_repeat(frozen, directory, "journal" if arm == "append" else arm)
    row["bytes"] -= printed_extra
    # The shipped restart is a separate clock; its counters cannot enter complete-service totals.
    n = row["fsyncs"]
    complete_ops, seen = [], 0
    for op in operations:
        if seen == n:
            break
        complete_ops.append(op)
        seen += op["operation"].endswith("fsync")
    row.update(
        arm=arm,
        operation_rows=complete_ops,
        file_fsyncs=sum(op["operation"] == "file_fsync" for op in complete_ops),
        directory_fsyncs=sum(op["operation"] == "directory_fsync" for op in complete_ops),
        namespace_calls=sum(not op["operation"].endswith("fsync") for op in complete_ops),
        peak_disk_bytes=peak_disk,
    )
    return dict(row)


def reduce(frozen: Json, row: Json) -> Json:
    """Reconstruct responses independently; primitive clocks and sync counts must agree."""
    if row["arm"] not in ("snapshot", "append") or row["cell"] != frozen["cell"]:
        raise ValueError("primitive_binding")
    expected = base.oracle(frozen, len(frozen["transactions"]))
    acknowledgments = [
        dict(id=tx["id"], response=reply)
        for tx, reply in zip(frozen["transactions"], expected["replies"], strict=True)
    ]
    if any(
        base.encode(row[key]) != base.encode(expected) for key in ("final", "restarted")
    ) or base.encode(row["acknowledgments"]) != base.encode(acknowledgments):
        raise ValueError("logical_work")
    n = len(acknowledgments)
    checkpoints = n // 32 + 1
    files = 2 * (n + 1) if row["arm"] == "snapshot" else 7 + 2 * n + 2 * checkpoints
    directories = 2 * (n + 1) if row["arm"] == "snapshot" else 6 + 3 * checkpoints
    if (
        row["file_fsyncs"] != files
        or row["directory_fsyncs"] != directories
        or row["fsyncs"] != files + directories
        or row["bytes"] <= 0
    ):
        raise ValueError("durable_io")
    ops = row["operation_rows"]
    if (
        row["file_fsyncs"] != sum(o["operation"] == "file_fsync" for o in ops)
        or row["directory_fsyncs"] != sum(o["operation"] == "directory_fsync" for o in ops)
        or row["namespace_calls"] != sum(not o["operation"].endswith("fsync") for o in ops)
    ):
        raise ValueError("operation_accounting")
    if (
        not row["start_ns"]
        < row["cold_end_ns"]
        < row["initialized_ns"]
        < row["end_ns"]
        < row["restart_start_ns"]
        < row["restart_end_ns"]
    ):
        raise ValueError("complete_clock")
    clock, latencies = row["initialized_ns"], []
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
    applied = sum(r["status"] == "completed" for r in feedback)
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
        scheduled_feedback=len(feedback),
        applied_updates=applied,
        nonapplicable_feedback=sum(r["status"] == "nonapplicable" for r in feedback),
        lost=sum(r["status"] not in ("completed", "nonapplicable") for r in feedback),
        missing=64 - len(decisions),
        pending=len(expected["state"]["pending"]),
        p50_s=float(np.quantile(latencies, 0.5)),
        p95_s=float(np.quantile(latencies, 0.95)),
        **{
            key: row[key]
            for key in (
                "bytes",
                "kernel_write_bytes",
                "fsyncs",
                "file_fsyncs",
                "directory_fsyncs",
                "namespace_calls",
                "peak_memory_bytes",
                "peak_disk_bytes",
            )
        },
        logical_hash=canonical_hash(expected),
        acknowledgment_hash=canonical_hash(acknowledgments),
    )


def pair(frozen: Json, snapshot: Json, journal: Json) -> Json:
    """The old reducer's journal key is only an internal adapter name for the append arm."""
    a, b = reduce(frozen, snapshot), reduce(frozen, journal)
    if (
        snapshot["arm"] != "snapshot"
        or journal["arm"] != "append"
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
    """Keep the unchanged five-block bootstrap while binding the V730 registered seed."""
    with patch.object(legacy, "SEED", SEED):
        return dict(legacy.interval(deepcopy(pairs)))
