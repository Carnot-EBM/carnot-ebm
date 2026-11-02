"""REQ-VERIFY-8449: observe the shipped durable service without changing writes.

Each operation keeps a nested interval. Exclusive times remove child intervals
once, so Python serialization and storage calls never share the same duration.
The original service reducer still checks actual snapshots and delayed updates.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, ExitStack, nullcontext
import json
import os
from pathlib import Path
import stat
import time
from typing import Any, Callable, Iterator
from unittest.mock import patch

from carnot.verify import python_transaction_cost_8407 as source
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
CELLS = [dict(workload="natural", batch=b, predictions_per_update=8) for b in (1, 8)]
ARMS = ("baseline", "instrumented")
cost_tape = source.cost_tape
REQUIRED_STAGES = (
    "state_copy_validation",
    "serialization_hash",
    "write",
    "file_fsync",
    "rename",
    "directory_fsync",
    "pointer",
    "pointer_serialization",
    "pointer_write",
    "pointer_file_fsync",
    "pointer_publication",
    "remaining_orchestration",
)


class Recorder:
    """Record real operation boundaries and keep parent IDs for independent replay."""

    def __init__(self) -> None:
        self.spans: list[Json] = []
        self.stack: list[int] = []
        self.pointer = False

    def call(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Failures close their observed span before they propagate to the caller."""
        index = len(self.spans)
        span = dict(
            name=name,
            parent=self.stack[-1] if self.stack else None,
            start_ns=time.monotonic_ns(),
            end_ns=None,
        )
        self.spans.append(span)
        self.stack.append(index)
        try:
            return fn(*args, **kwargs)
        finally:
            span["end_ns"] = time.monotonic_ns()
            self.stack.pop()

    @contextmanager
    def observe(self) -> Iterator[None]:
        """Wrap only primitive boundaries; original functions still perform every write."""
        dumps, dump, opened, replaced, fsync = (
            json.dumps,
            json.dump,
            Path.open,
            Path.replace,
            os.fsync,
        )
        hashed, copied, snapshot, atomic = (
            source.s.canonical_hash,
            source.deepcopy,
            source.read_snapshot,
            source.s.atomic_json,
        )

        def encode(*args: Any, **kwargs: Any) -> str:
            return str(
                self.call(
                    "pointer_serialization" if self.pointer else "serialization_hash",
                    dumps,
                    *args,
                    **kwargs,
                )
            )

        def stream_encode(*args: Any, **kwargs: Any) -> None:
            self.call(
                "pointer_serialization" if self.pointer else "serialization_hash",
                dump,
                *args,
                **kwargs,
            )

        def open_file(path: Path, *args: Any, **kwargs: Any) -> Any:
            stream = opened(path, *args, **kwargs)
            mode = args[0] if args else kwargs.get("mode", "r")
            return ObservedFile(stream, self) if "w" in mode else stream

        def rename(path: Path, target: Any) -> Path:
            return Path(
                self.call(
                    "pointer_publication" if self.pointer else "rename", replaced, path, target
                )
            )

        def sync(descriptor: int) -> None:
            name = (
                "directory_fsync"
                if stat.S_ISDIR(os.fstat(descriptor).st_mode)
                else ("pointer_file_fsync" if self.pointer else "file_fsync")
            )
            self.call(name, fsync, descriptor)

        def publish(path: Path, value: Json) -> None:
            self.pointer = True
            try:
                self.call("pointer", atomic, path, value)
            finally:
                self.pointer = False

        with ExitStack() as stack:
            for owner, name, wrapped in [
                (json, "dumps", encode),
                (json, "dump", stream_encode),
                (Path, "open", open_file),
                (Path, "replace", rename),
                (os, "fsync", sync),
                (source.s, "atomic_json", publish),
                (
                    source.s,
                    "canonical_hash",
                    lambda *a, **kw: self.call("serialization_hash", hashed, *a, **kw),
                ),
                (
                    source,
                    "deepcopy",
                    lambda *a, **kw: self.call("state_copy_validation", copied, *a, **kw),
                ),
                (
                    source.s,
                    "deepcopy",
                    lambda *a, **kw: self.call("state_copy_validation", copied, *a, **kw),
                ),
                (
                    source,
                    "read_snapshot",
                    lambda *a, **kw: self.call("state_copy_validation", snapshot, *a, **kw),
                ),
            ]:
                stack.enter_context(patch.object(owner, name, wrapped))
            yield


class ObservedFile:
    """Observe buffered writes and flush separately from the following fsync call."""

    def __init__(self, stream: Any, recorder: Recorder) -> None:
        self.stream, self.recorder = stream, recorder

    def __enter__(self) -> ObservedFile:
        self.stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stream.__exit__(*args)

    def write(self, data: Any) -> int:
        """Byte counts reflect data handed to the real stream, including pointer JSON."""
        name = "pointer_write" if self.recorder.pointer else "write"
        index = len(self.recorder.spans)
        count = int(self.recorder.call(name, self.stream.write, data))
        self.recorder.spans[index]["bytes"] = len(data.encode() if isinstance(data, str) else data)
        return count

    def flush(self) -> None:
        """Buffered filesystem work belongs to write time rather than serialization."""
        self.recorder.call("pointer_write" if self.recorder.pointer else "write", self.stream.flush)

    def fileno(self) -> int:
        return int(self.stream.fileno())


def run_repeat(bundle: Json, cell: Json, repeat: int, directory: Path, arm: str) -> Json:
    """Preserve the original complete service clock inside an earlier outer clock."""
    if cell not in CELLS or repeat not in range(6) or arm not in ARMS:
        raise ValueError("fixed_work_units")
    recorder = Recorder()
    with recorder.observe() if arm == "instrumented" else nullcontext():
        result: Json = recorder.call(
            "remaining_orchestration", source.run_repeat, bundle, cell, repeat, directory
        )
    result.update(
        arm=arm,
        spans=recorder.spans,
        outer_start_ns=recorder.spans[0]["start_ns"],
        outer_end_ns=recorder.spans[0]["end_ns"],
    )
    result["semantic_hash"] = canonical_hash(
        dict(
            final_state=result["final_state"],
            rows=[
                {
                    key: value
                    for key, value in row.items()
                    if key not in ("start_ns", "end_ns", "transaction_id")
                }
                for row in result["rows"]
            ],
        )
    )
    return result


def reduce_repeat(bundle: Json, value: Json) -> Json:
    """Recheck real snapshots, causal rows and the exclusive nested clock partition."""
    reduced = source.reduce_repeat(bundle, value)
    spans = value["spans"]
    left, right = value["outer_start_ns"], value["outer_end_ns"]
    if not left <= value["start_ns"] < value["end_ns"] <= right or spans[0] != dict(
        name="remaining_orchestration", parent=None, start_ns=left, end_ns=right
    ):
        raise ValueError("outer_clock")
    exclusive: Counter[str] = Counter()
    children: dict[int, list[Json]] = {}
    for index, span in enumerate(spans):
        parent = span["parent"]
        if span["name"] not in REQUIRED_STAGES or not (
            type(span["start_ns"]) is int
            and type(span["end_ns"]) is int
            and left <= span["start_ns"] <= span["end_ns"] <= right
        ):
            raise ValueError("stage_nesting")
        if index and (type(parent) is not int or not 0 <= parent < index):
            raise ValueError("stage_parent")
        if index:
            ancestor = spans[parent]
            siblings = children.setdefault(parent, [])
            if not ancestor["start_ns"] <= span["start_ns"] <= span["end_ns"] <= ancestor[
                "end_ns"
            ] or (siblings and siblings[-1]["end_ns"] > span["start_ns"]):
                raise ValueError("stage_overlap")
            siblings.append(span)
    for index, span in enumerate(spans):
        exclusive[span["name"]] += (
            span["end_ns"]
            - span["start_ns"]
            - sum(child["end_ns"] - child["start_ns"] for child in children.get(index, []))
        )
    fsync_count = sum(
        span["name"] in ("file_fsync", "pointer_file_fsync", "directory_fsync") for span in spans
    )
    if value["arm"] == "instrumented" and (
        not set(REQUIRED_STAGES) <= exclusive.keys() or fsync_count != reduced["fsync_count"]
    ):
        raise ValueError("instrumentation_completeness")
    semantic_hash = canonical_hash(
        dict(
            final_state=value["final_state"],
            rows=[
                {
                    key: item
                    for key, item in row.items()
                    if key not in ("start_ns", "end_ns", "transaction_id")
                }
                for row in value["rows"]
            ],
        )
    )
    if semantic_hash != value["semantic_hash"]:
        raise ValueError("semantic_hash")
    return dict(
        cell=value["cell"],
        repeat=value["repeat"],
        arm=value["arm"],
        status="completed",
        complete_wall_s=(right - left) / 1e9,
        service_wall_s=reduced["wall_seconds"],
        startup_s=(value["initialized_ns"] - left) / 1e9,
        tail_drain_s=(right - value["tail_start_ns"]) / 1e9,
        wallclock_gaps_s=(
            (value["start_ns"] - left)
            + (right - value["end_ns"])
            + sum(
                b["start_ns"] - a["end_ns"]
                for a, b in zip(value["transactions"], value["transactions"][1:])
            )
        )
        / 1e9,
        unclassified_residual_s=exclusive["remaining_orchestration"] / 1e9,
        stage_rows=[
            dict(name=name, exclusive_ns=ns, duration_s=ns / 1e9)
            for name, ns in sorted(exclusive.items())
        ],
        fsync_count=reduced["fsync_count"],
        serialized_bytes=reduced["serialized_bytes"],
        observed_write_bytes=sum(span.get("bytes", 0) for span in spans)
        if value["arm"] == "instrumented"
        else None,
        semantic_hash=semantic_hash,
        final_state_hash=canonical_hash(value["final_state"]),
        response_hashes=[
            canonical_hash(
                [
                    {
                        key: item
                        for key, item in row.items()
                        if key not in ("start_ns", "end_ns", "transaction_id")
                    }
                    for row in value["rows"]
                    if row["request_id"] in tx["request_ids"]
                ]
            )
            for tx in value["transactions"]
        ],
        prediction_count=reduced["prediction_attempts"],
        feedback_count=reduced["feedback_attempts"],
    )
