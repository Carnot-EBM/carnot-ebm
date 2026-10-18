"""REQ-VERIFY-8464: isolate durable operations without changing service semantics.

The adapter writes the same snapshot and pointer bytes in the same order as the
qualified store. Extra clocks cost time; paired original runs expose that cost.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.verify import python_transaction_cost_8407 as legacy
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
CELLS = [dict(workload="natural", batch=b, predictions_per_update=8) for b in (1, 8)]
STAGES = (
    "state_copy_validation",
    "serialization_hash",
    "write",
    "file_fsync",
    "rename",
    "directory_fsync",
    "pointer_serialization",
    "pointer_write",
    "pointer_file_fsync",
    "pointer_publication",
    "pointer_directory_fsync",
    "remaining_orchestration",
)


class ProfiledStore:
    """Mirror original durable bytes so only instrumentation changes between arms."""

    def __init__(self, path: Path, frozen: Json, spans: list[Json]):
        self.path, self.spans = path, spans
        path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def commit(self, state: Json, hook: Callable[[str], None] = lambda phase: None) -> None:
        """Keep both file and directory fsyncs before acknowledgment, as originally qualified."""
        start = last = time.monotonic_ns()
        stages: list[Json] = []

        def mark(name: str) -> None:
            nonlocal last
            now = time.monotonic_ns()
            stages.append(dict(name=name, start_ns=last, end_ns=now))
            last = now

        digest = canonical_hash(state)
        encoded = json.dumps(dict(state=state, sha256=digest), sort_keys=True).encode()
        mark("serialization_hash")
        if len(encoded) > legacy.s.MAX_BYTES:
            raise ValueError("state_byte_budget")
        mark("state_copy_validation")
        target = self.path / ("version-" + digest[7:] + ".json")
        temporary = self.path / ".state.tmp"
        with temporary.open("wb") as stream:
            stream.write(encoded)
            mark("write")
            hook("temporary_write")
            mark("remaining_orchestration")
            stream.flush()
            mark("write")
            os.fsync(stream.fileno())
            mark("file_fsync")
            hook("file_fsync")
        mark("remaining_orchestration")
        temporary.replace(target)
        mark("rename")
        legacy.s.sync_directory(self.path)
        mark("directory_fsync")
        pointer = self.path / "current.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        temporary = pointer.with_name(f".{pointer.name}.tmp-{os.getpid()}")
        mark("remaining_orchestration")
        with temporary.open("w", encoding="utf-8") as pointer_stream:
            mark("pointer_write")

            class TimedWriter:
                """Measure each original encoder write so CPU and I/O remain separate."""

                def write(self, chunk: str) -> int:
                    mark("pointer_serialization")
                    written = pointer_stream.write(chunk)
                    mark("pointer_write")
                    return written

            json.dump(
                dict(file=target.name, sha256=digest), TimedWriter(), indent=2, sort_keys=True
            )
            mark("pointer_serialization")
            pointer_stream.write("\n")
            mark("pointer_write")
            pointer_stream.flush()
            mark("pointer_write")
            os.fsync(pointer_stream.fileno())
            mark("pointer_file_fsync")
        mark("remaining_orchestration")
        temporary.replace(pointer)
        mark("pointer_publication")
        hook("pointer_publication")
        mark("remaining_orchestration")
        legacy.s.sync_directory(self.path)
        mark("pointer_directory_fsync")
        hook("directory_fsync")
        hook("pointer_replace")
        hook("acknowledgment")
        mark("remaining_orchestration")
        self.spans.append(
            dict(
                start_ns=start,
                end_ns=last,
                stages=stages,
                state_hash=digest,
                bytes=len(encoded) + pointer.stat().st_size,
                fsync_count=4,
            )
        )


def run_repeat(bundle: Json, cell: Json, repeat: int, directory: Path, instrumented: bool) -> Json:
    """Include constructor startup, initialization and tail in the same complete clock."""
    if cell not in CELLS:
        raise ValueError("registered_cells_only")
    spans: list[Json] = []
    start = time.monotonic_ns()
    if instrumented:
        with patch.object(legacy, "QualifiedStore", lambda p, f: ProfiledStore(p, f, spans)):
            result = dict(legacy.run_repeat(bundle, cell, repeat, directory))
    else:
        result = dict(legacy.run_repeat(bundle, cell, repeat, directory))
    responses = response_hashes(result)
    result.update(
        start_ns=start,
        persistence_spans=spans,
        instrumented=instrumented,
        response_hashes=responses,
        response_hash=canonical_hash(responses),
        state_hash=canonical_hash(result["final_state"]),
    )
    return result


def response_hashes(result: Json) -> list[str]:
    """Hash the actual response encoding before request timing metadata was attached."""
    by_id = {row["request_id"]: row for row in result["rows"]}
    hashes = []
    for tx in result["transactions"]:
        payload = [
            {
                key: value
                for key, value in by_id[identity].items()
                if key not in ("transaction_id", "start_ns", "end_ns")
            }
            for identity in tx["request_ids"]
        ]
        hashes.append(
            "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        )
    return hashes


def reduce(bundle: Json, result: Json) -> Json:
    """Original semantic replay plus exact leaf clocks prevents a hash-only qualification."""
    summary = dict(legacy.reduce_repeat(bundle, result))
    if result["cell"] not in CELLS or (
        result["response_hashes"] != response_hashes(result)
        or result["response_hash"] != canonical_hash(result["response_hashes"])
        or result["state_hash"] != canonical_hash(result["final_state"])
    ):
        raise ValueError("bound_state_and_response")
    costs = dict.fromkeys(STAGES, 0)
    spans = result["persistence_spans"]
    if bool(spans) != result["instrumented"]:
        raise ValueError("instrumentation_binding")
    if spans:
        if len(spans) != len(result["transactions"]) + 1:
            raise ValueError("commit_roster")
        parents = [
            dict(
                start_ns=result["start_ns"],
                end_ns=result["initialized_ns"],
                state_hash=canonical_hash(result["initial_state"]),
            )
        ]
        parents.extend(result["transactions"])
        for span, parent in zip(spans, parents, strict=True):
            if not parent["start_ns"] <= span["start_ns"] < span["end_ns"] <= parent["end_ns"]:
                raise ValueError("commit_nesting")
            clock = span["start_ns"]
            names = []
            for stage in span["stages"]:
                name = stage["name"]
                if name not in STAGES or not clock == stage["start_ns"] <= stage["end_ns"]:
                    raise ValueError("disjoint_stage_nesting")
                clock = stage["end_ns"]
                names.append(name)
                costs[name] += stage["end_ns"] - stage["start_ns"]
            required = [
                "serialization_hash",
                "state_copy_validation",
                "file_fsync",
                "rename",
                "directory_fsync",
                "pointer_file_fsync",
                "pointer_publication",
                "pointer_directory_fsync",
            ]
            if (
                clock != span["end_ns"]
                or span["state_hash"] != parent["state_hash"]
                or span["fsync_count"] != 4
                or any(names.count(n) != 1 for n in required)
                or [n for n in names if n in required] != required
                or not {"write", "pointer_serialization", "pointer_write"} <= set(names)
            ):
                raise ValueError("exact_stage_topology")
            if parent.get("transaction_id"):
                tx_bytes = Path(parent["snapshot_path"]).stat().st_size
                pointer_bytes = (
                    len(
                        json.dumps(
                            dict(
                                file=Path(parent["snapshot_path"]).name, sha256=parent["state_hash"]
                            ),
                            sort_keys=True,
                            indent=2,
                        ).encode()
                    )
                    + 1
                )
                if span["bytes"] != tx_bytes + pointer_bytes:
                    raise ValueError("persistence_byte_count")
    duration = result["end_ns"] - result["start_ns"]
    classified = sum(costs.values())
    costs["remaining_orchestration"] += duration - classified
    return dict(
        summary,
        stage_ns=costs,
        response_hash=result["response_hash"],
        state_hash=result["state_hash"],
        complete_wall_s=duration / 1e9,
        tail_drain_s=summary["tail_drain_seconds"],
        explicit_unclassified_ns=duration - classified,
        persistence_state_copy_operations=0,
        instrumentation_overhead_subtracted=False,
    )


def pair(bundle: Json, baseline: Json, instrumented: Json) -> Json:
    """Two arms are one repeated engineering question; extra clocks never add sources."""
    b, i = reduce(bundle, baseline), reduce(bundle, instrumented)
    if (
        baseline["instrumented"]
        or not instrumented["instrumented"]
        or any(b[key] != i[key] for key in ("cell", "repeat", "state_hash", "response_hash"))
    ):
        raise ValueError("paired_semantics")
    return dict(
        i,
        baseline_wall_s=b["complete_wall_s"],
        instrumented_wall_s=i["complete_wall_s"],
        instrumentation_overhead_s=i["complete_wall_s"] - b["complete_wall_s"],
        instrumentation_overhead_fraction=i["complete_wall_s"] / b["complete_wall_s"] - 1,
    )
