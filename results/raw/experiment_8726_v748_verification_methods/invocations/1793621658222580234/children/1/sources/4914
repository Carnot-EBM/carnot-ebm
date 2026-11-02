"""REQ-REPORT-8329: measure complete CPU work without inventing fabric support.

The existing kernel owns numerical and causal semantics. This adapter times its
actual operations; durable snapshots include file and directory fsync costs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import local_update_isolation_8306 as k

Json = dict[str, Any]
ARMS = ["full", "indexed"]
OPERATIONS = [
    "feature_access",
    "numerical_update",
    "invalidation",
    "pending_serialization",
    "durable_write",
    "recovery",
    "dispatch",
]
SCIENCE = dict(
    warmups=1,
    repetitions=5,
    delay=8,
    online_decay=0,
    online_slope_frozen=True,
    k_max=5,
    model_load=False,
)


def expected(trajectory: Json, arm: str) -> Json:
    """Recompute states without measured clocks so changed checkpoints cannot pass."""
    state = k.initial(trajectory, arm)
    for slot in range(72):
        k.issue(state, slot)
        k.release(trajectory, state, slot, arm)
    return dict(k.semantic(state))


def durable(path: Path, encoded: bytes) -> None:
    """A completed write includes directory durability after the atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".pending")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def transaction(trajectory: Json, arm: str, path: Path) -> Json:
    """Time issue-before-release snapshots and exact recovery as one CPU service.

    Invalidation time is subtracted from its enclosing update span. Operation
    panels are disjoint; loop, adapter and clock overhead stays in complete_ns.
    Dispatch means host decoding and decisions, with no device or transfer claim.
    """
    spans: Json = dict.fromkeys(OPERATIONS, 0)
    started = time.monotonic_ns()
    before = time.monotonic_ns()
    state = k.initial(trajectory, arm)
    spans["feature_access"] += time.monotonic_ns() - before
    invalidate = k.invalidate

    def timed_invalidate(state: Json, changed: list[int], arm: str) -> Json:
        began = time.monotonic_ns()
        result = invalidate(state, changed, arm)
        spans["invalidation"] += time.monotonic_ns() - began
        return dict(result)

    with patch.object(k, "invalidate", timed_invalidate):
        for slot in range(72):
            before = time.monotonic_ns()
            k.issue(state, slot)
            spans["feature_access"] += time.monotonic_ns() - before
            for stage in ["issue", "release"]:
                if stage == "release":
                    prior = spans["invalidation"]
                    before = time.monotonic_ns()
                    k.release(trajectory, state, slot, arm)
                    spans["numerical_update"] += (
                        time.monotonic_ns() - before - (spans["invalidation"] - prior)
                    )
                before = time.monotonic_ns()
                encoded = json.dumps(
                    dict(
                        encoded_state=json.dumps(state, sort_keys=True, separators=(",", ":")),
                        sha256=canonical_hash(state),
                    ),
                    sort_keys=True,
                ).encode()
                spans["pending_serialization"] += time.monotonic_ns() - before
                before = time.monotonic_ns()
                durable(path, encoded)
                spans["durable_write"] += time.monotonic_ns() - before
            if (slot + 1) % 24 == 0:
                print(
                    f"[exp8329] phase=transaction_{arm} completed={slot + 1} pending={71 - slot}",
                    flush=True,
                )
    before = time.monotonic_ns()
    recovered = k.load(trajectory, path)
    spans["recovery"] += time.monotonic_ns() - before
    before = time.monotonic_ns()
    decoded = json.loads(json.loads(encoded)["encoded_state"])
    dispatched = [k.action(row["p"]) for row in decoded["issues"]]
    spans["dispatch"] += time.monotonic_ns() - before
    ended = time.monotonic_ns()
    return dict(
        trajectory=trajectory,
        arm=arm,
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        complete_ns=ended - started,
        operation_ns=spans,
        expected_semantics=k.semantic(state),
        recovered_semantics=k.semantic(recovered),
        dispatch_actions=dispatched,
        checkpoint_sha256=canonical_hash(json.loads(encoded)),
    )


def verify_row(row: Json) -> None:
    """Independent causal replay and exact clock arithmetic reject rehashed claims."""
    wanted = expected(row["trajectory"], row["arm"])
    if row["expected_semantics"] != wanted or row["recovered_semantics"] != wanted:
        raise ValueError("transaction_semantics")
    if row["dispatch_actions"] != [k.action(x["p"]) for x in wanted["issues"]]:
        raise ValueError("dispatch_semantics")
    spans = row["operation_ns"]
    if (
        set(spans) != set(OPERATIONS)
        or any(type(v) is not int or v < 0 for v in spans.values())
        or row["complete_ns"] != row["ended_monotonic_ns"] - row["started_monotonic_ns"]
        or row["complete_ns"] <= 0
        or sum(spans.values()) > row["complete_ns"]
    ):
        raise ValueError("transaction_clock")


def boundary(rows: list[Json]) -> Json:
    """No measured update operation matches the deployed quadratic Ising kernel."""
    return dict(
        operation_rows=[
            dict(
                operation=op,
                assigned_substrate="host_CPU",
                measured_ns=[r["operation_ns"][op] for r in rows],
                kv260_supported=False,
                reason="quadratic_Ising_overlay_has_no_spline_update_or_persistence_kernel",
            )
            for op in OPERATIONS
        ],
        compatible_fraction=None,
        amdahl_upper_bound=None,
        nfr01_met=False,
        accelerator_benefit="unproved_no_compatible_operation",
        k_max=5,
        transfer_cost_ns=None,
        transfer_status="unavailable_no_board_execution",
        denominator="complete CPU transactions; any future transfer and CPU costs must remain",
    )
