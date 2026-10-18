"""REQ-VERIFY-8436: stage sums need a complete wall clock and disjoint intervals.

The inherited clocks already include durable writes and response construction.
Reading them avoids replacing historical costs with a different host benchmark.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

Json = dict[str, Any]
PERSISTENCE = (
    "temporary_write",
    "file_fsync",
    "pointer_publication",
    "directory_fsync",
    "pointer_replace",
    "acknowledgment",
)


def attribute(value: Json) -> Json:
    """Keep initialization and unmeasured gaps in T so arithmetic cannot claim them."""
    cell, repeat = value["cell"], value["repeat"]
    prefix = f"{cell['workload']}-b{cell['batch']}-r{cell['predictions_per_update']}-rep{repeat}"
    start, initialized, tail, end = [
        value[n] for n in ("start_ns", "initialized_ns", "tail_start_ns", "end_ns")
    ]
    if (
        not all(type(n) is int and n > 0 for n in (start, initialized, tail, end))
        or not start < initialized <= tail < end
    ):
        raise ValueError("repeat_clocks")
    if value["repeat_id"] != prefix or repeat not in range(6):
        raise ValueError("repeat_identity")
    stages: Counter[str] = Counter()
    previous, stage_count = initialized, 0
    flattened = []
    for index, tx in enumerate(value["transactions"]):
        if (
            tx["transaction_id"] != f"{prefix}-tx{index}"
            or not previous <= tx["start_ns"] < tx["end_ns"] <= end
        ):
            raise ValueError("transaction_containment_or_overlap")
        clock = tx["start_ns"]
        names = set()
        for span in tx["stages"]:
            left, right = span["start_ns"], span["end_ns"]
            if (
                type(left) is not int
                or type(right) is not int
                or not clock == left <= right <= tx["end_ns"]
            ):
                raise ValueError("stage_containment_or_overlap")
            stages[span["name"]] += right - left
            names.add(span["name"])
            clock, stage_count = right, stage_count + 1
        if not set(PERSISTENCE + ("response",)) <= names:
            raise ValueError("missing_stage")
        previous = tx["end_ns"]
        flattened.extend(tx["request_ids"])
    rows = value["rows"]
    if flattened != [row["request_id"] for row in rows]:
        raise ValueError("request_transaction_binding")
    if rows:
        for index, row in enumerate(rows):
            if row["request_id"] != f"{prefix}-q{index + 1}":
                raise ValueError("request_identity")
        tail_tx = next(
            tx
            for tx in value["transactions"]
            if next(r for r in rows if r["request_id"] == tx["request_ids"][0])["tick"]
            > 8 * cell["predictions_per_update"]
        )
        before = value["transactions"].index(tail_tx)
        if not value["transactions"][before - 1]["end_ns"] <= tail <= tail_tx["start_ns"]:
            raise ValueError("tail_boundary")
    total = end - start
    kernel = stages["direct_scoring"] + stages["coefficient_update"]
    conversion = stages["python_boundary_conversion"]
    persistence = initialized - start + sum(stages[n] for n in PERSISTENCE)
    other = sum(stages.values()) - kernel - conversion - sum(stages[n] for n in PERSISTENCE)
    remainder = total - kernel - conversion - persistence - other
    return dict(
        cell=cell,
        repeat=repeat,
        repeat_id=prefix,
        status="completed",
        complete_wall_s=total / 1e9,
        kernel_s=kernel / 1e9,
        conversion_s=conversion / 1e9,
        persistence_stage_s=persistence / 1e9,
        other_stage_s=other / 1e9,
        unattributed_s=remainder / 1e9,
        initialization_s=(initialized - start) / 1e9,
        tail_drain_s=(end - tail) / 1e9,
        stage_ns=dict(stages),
        interval_checks=dict(
            contained=True,
            nonoverlapping=True,
            stage_count=stage_count,
            transaction_count=len(value["transactions"]),
            tail_not_double_counted=True,
        ),
        work_counts=dict(
            predictions=sum(r["kind"] == "prediction" for r in rows),
            scheduled_feedback=sum(r["kind"] == "feedback" for r in rows),
            applied_updates=sum(bool(r.get("changed")) for r in rows),
            transaction_rows=len(rows),
        ),
    )
